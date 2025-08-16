import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

from src.core.models import Order, OrderSide, OrderType, OrderStatus, OrderBook, MarketData
from src.utils.logger import LoggerMixin


class AlgorithmStatus(Enum):
    """算法状态"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AlgorithmConfig:
    """算法配置基类"""
    max_participation_rate: float = 0.2  # 最大参与率 20%
    min_order_size: float = 0.01  # 最小订单大小
    max_order_size: float = 1000.0  # 最大订单大小
    aggressive_threshold: float = 0.3  # 激进阈值
    passive_threshold: float = 0.1  # 保守阈值
    price_improvement_threshold: float = 0.0001  # 价格改善阈值 0.01%


@dataclass
class SliceOrder:
    """分片订单"""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float]
    order_type: OrderType
    scheduled_time: datetime
    status: OrderStatus = OrderStatus.NEW
    actual_order: Optional[Order] = None
    execution_time: Optional[datetime] = None
    filled_qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class AlgorithmResult:
    """算法执行结果"""
    algorithm_id: str
    parent_order_id: str
    status: AlgorithmStatus
    total_filled: float
    avg_price: float
    total_slices: int
    completed_slices: int
    total_commission: float
    implementation_shortfall: float
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class ExecutionAlgorithm(ABC, LoggerMixin):
    """执行算法基类"""
    
    def __init__(self, algorithm_id: str, config: AlgorithmConfig):
        self.algorithm_id = algorithm_id
        self.config = config
        self.status = AlgorithmStatus.CREATED
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 订单分片
        self.slices: List[SliceOrder] = []
        self.completed_slices: List[SliceOrder] = []
        
        # 执行统计
        self.total_filled = 0.0
        self.total_commission = 0.0
        self.weighted_avg_price = 0.0
        
        # 市场数据
        self.current_market_data: Optional[MarketData] = None
        self.current_order_book: Optional[OrderBook] = None
        
        # 控制参数
        self._stop_requested = False
        self._pause_requested = False
    
    @abstractmethod
    async def initialize(self, parent_order: Order) -> List[SliceOrder]:
        """初始化算法，生成订单分片"""
        pass
    
    @abstractmethod
    async def update_market_data(self, market_data: MarketData, order_book: OrderBook):
        """更新市场数据"""
        pass
    
    @abstractmethod
    async def on_slice_filled(self, slice_order: SliceOrder, fill_qty: float, fill_price: float):
        """分片订单成交回调"""
        pass
    
    @abstractmethod
    async def should_adjust_strategy(self) -> bool:
        """是否应该调整策略"""
        pass
    
    async def start(self, parent_order: Order) -> AlgorithmResult:
        """启动算法"""
        self.log_info(f"Starting algorithm {self.__class__.__name__}", algorithm_id=self.algorithm_id)
        
        self.status = AlgorithmStatus.RUNNING
        self.start_time = datetime.utcnow()
        
        try:
            # 初始化算法
            self.slices = await self.initialize(parent_order)
            
            self.log_info(
                f"Algorithm initialized with {len(self.slices)} slices",
                algorithm_id=self.algorithm_id,
                total_quantity=parent_order.quantity
            )
            
            # 执行算法
            await self._execute()
            
            # 创建结果
            result = AlgorithmResult(
                algorithm_id=self.algorithm_id,
                parent_order_id=parent_order.client_order_id or "",
                status=self.status,
                total_filled=self.total_filled,
                avg_price=self.weighted_avg_price,
                total_slices=len(self.slices),
                completed_slices=len(self.completed_slices),
                total_commission=self.total_commission,
                implementation_shortfall=self._calculate_implementation_shortfall(parent_order),
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            return result
            
        except Exception as e:
            self.status = AlgorithmStatus.ERROR
            self.end_time = datetime.utcnow()
            self.log_error(f"Algorithm execution failed: {e}", algorithm_id=self.algorithm_id)
            raise
    
    async def stop(self):
        """停止算法"""
        self._stop_requested = True
        self.status = AlgorithmStatus.CANCELLED
        self.log_info(f"Algorithm stop requested", algorithm_id=self.algorithm_id)
    
    async def pause(self):
        """暂停算法"""
        self._pause_requested = True
        self.status = AlgorithmStatus.PAUSED
        self.log_info(f"Algorithm paused", algorithm_id=self.algorithm_id)
    
    async def resume(self):
        """恢复算法"""
        self._pause_requested = False
        self.status = AlgorithmStatus.RUNNING
        self.log_info(f"Algorithm resumed", algorithm_id=self.algorithm_id)
    
    async def _execute(self):
        """执行算法主逻辑"""
        while not self._stop_requested and len(self.completed_slices) < len(self.slices):
            if self._pause_requested:
                await asyncio.sleep(1.0)
                continue
            
            # 获取下一个要执行的分片
            next_slice = self._get_next_slice()
            if not next_slice:
                await asyncio.sleep(0.1)
                continue
            
            # 检查是否到了执行时间
            if datetime.utcnow() < next_slice.scheduled_time:
                await asyncio.sleep(0.1)
                continue
            
            # 执行分片订单
            await self._execute_slice(next_slice)
            
            # 检查是否需要调整策略
            if await self.should_adjust_strategy():
                await self._adjust_strategy()
            
            await asyncio.sleep(0.1)  # 避免过于频繁的检查
        
        self.status = AlgorithmStatus.COMPLETED
        self.end_time = datetime.utcnow()
        self.log_info(f"Algorithm execution completed", algorithm_id=self.algorithm_id)
    
    def _get_next_slice(self) -> Optional[SliceOrder]:
        """获取下一个要执行的分片"""
        for slice_order in self.slices:
            if slice_order.status == OrderStatus.NEW:
                return slice_order
        return None
    
    async def _execute_slice(self, slice_order: SliceOrder):
        """执行分片订单"""
        try:
            self.log_debug(
                f"Executing slice order",
                slice_id=slice_order.slice_id,
                quantity=slice_order.quantity,
                price=slice_order.price
            )
            
            # 这里应该调用实际的订单执行接口
            # 为了演示，我们模拟订单执行
            slice_order.status = OrderStatus.FILLED
            slice_order.execution_time = datetime.utcnow()
            slice_order.filled_qty = slice_order.quantity
            slice_order.avg_price = slice_order.price or (self.current_market_data.price if self.current_market_data else 50000.0)
            
            # 更新统计
            await self.on_slice_filled(slice_order, slice_order.filled_qty, slice_order.avg_price)
            self.completed_slices.append(slice_order)
            
            self.log_debug(
                f"Slice order executed successfully",
                slice_id=slice_order.slice_id,
                filled_qty=slice_order.filled_qty,
                avg_price=slice_order.avg_price
            )
            
        except Exception as e:
            slice_order.status = OrderStatus.REJECTED
            self.log_error(f"Failed to execute slice order: {e}", slice_id=slice_order.slice_id)
    
    async def _adjust_strategy(self):
        """调整策略"""
        # 基类默认不做任何调整，子类可以重写
        pass
    
    def _calculate_implementation_shortfall(self, parent_order: Order) -> float:
        """计算实施缺口"""
        if not self.current_market_data or self.total_filled == 0:
            return 0.0
        
        # 实施缺口 = (实际执行价格 - 决策时价格) / 决策时价格
        decision_price = parent_order.price or self.current_market_data.price
        shortfall = (self.weighted_avg_price - decision_price) / decision_price
        
        # 如果是卖单，符号相反
        if parent_order.side == OrderSide.SELL:
            shortfall = -shortfall
        
        return shortfall
    
    def _update_weighted_avg_price(self, fill_qty: float, fill_price: float):
        """更新加权平均价格"""
        total_value = self.weighted_avg_price * self.total_filled + fill_qty * fill_price
        self.total_filled += fill_qty
        self.weighted_avg_price = total_value / self.total_filled if self.total_filled > 0 else 0.0


class TWAPAlgorithm(ExecutionAlgorithm):
    """时间加权平均价格算法"""
    
    @dataclass
    class TWAPConfig(AlgorithmConfig):
        duration_minutes: int = 60  # 执行时长(分钟)
        slice_interval_seconds: int = 60  # 分片间隔(秒)
        randomize_timing: bool = True  # 随机化时间
        timing_randomness_pct: float = 0.1  # 时间随机性 10%
    
    def __init__(self, algorithm_id: str, config: TWAPConfig):
        super().__init__(algorithm_id, config)
        self.twap_config = config
    
    async def initialize(self, parent_order: Order) -> List[SliceOrder]:
        """初始化TWAP算法"""
        slices = []
        
        # 计算分片数量
        total_slices = self.twap_config.duration_minutes * 60 // self.twap_config.slice_interval_seconds
        slice_quantity = parent_order.quantity / total_slices
        
        # 确保分片大小在合理范围内
        slice_quantity = max(slice_quantity, self.config.min_order_size)
        slice_quantity = min(slice_quantity, self.config.max_order_size)
        
        # 重新计算实际分片数量
        actual_slices = math.ceil(parent_order.quantity / slice_quantity)
        
        current_time = datetime.utcnow()
        
        for i in range(actual_slices):
            # 计算分片时间
            base_time = current_time + timedelta(seconds=i * self.twap_config.slice_interval_seconds)
            
            # 添加随机性
            if self.twap_config.randomize_timing:
                random_offset = (i % 2 - 0.5) * 2 * self.twap_config.timing_randomness_pct * self.twap_config.slice_interval_seconds
                base_time += timedelta(seconds=random_offset)
            
            # 计算分片数量
            remaining_qty = parent_order.quantity - i * slice_quantity
            current_slice_qty = min(slice_quantity, remaining_qty)
            
            if current_slice_qty <= 0:
                break
            
            slice_order = SliceOrder(
                slice_id=f"{self.algorithm_id}_slice_{i}",
                parent_order_id=parent_order.client_order_id or "",
                symbol=parent_order.symbol,
                side=parent_order.side,
                quantity=current_slice_qty,
                price=parent_order.price,  # TWAP通常使用限价单
                order_type=OrderType.LIMIT,
                scheduled_time=base_time
            )
            
            slices.append(slice_order)
        
        return slices
    
    async def update_market_data(self, market_data: MarketData, order_book: OrderBook):
        """更新市场数据"""
        self.current_market_data = market_data
        self.current_order_book = order_book
    
    async def on_slice_filled(self, slice_order: SliceOrder, fill_qty: float, fill_price: float):
        """分片订单成交回调"""
        self._update_weighted_avg_price(fill_qty, fill_price)
        self.log_debug(
            f"TWAP slice filled",
            slice_id=slice_order.slice_id,
            fill_qty=fill_qty,
            fill_price=fill_price,
            total_filled=self.total_filled
        )
    
    async def should_adjust_strategy(self) -> bool:
        """TWAP通常不需要调整策略"""
        return False


class VWAPAlgorithm(ExecutionAlgorithm):
    """成交量加权平均价格算法"""
    
    @dataclass
    class VWAPConfig(AlgorithmConfig):
        duration_minutes: int = 60  # 执行时长(分钟)
        historical_volume_periods: int = 20  # 历史成交量周期数
        target_participation_rate: float = 0.1  # 目标参与率 10%
        volume_curve_adjustment: float = 1.0  # 成交量曲线调整因子
    
    def __init__(self, algorithm_id: str, config: VWAPConfig):
        super().__init__(algorithm_id, config)
        self.vwap_config = config
        self.historical_volumes: List[float] = []
        self.cumulative_volume = 0.0
        self.vwap_price = 0.0
    
    async def initialize(self, parent_order: Order) -> List[SliceOrder]:
        """初始化VWAP算法"""
        slices = []
        
        # 获取历史成交量模式（这里简化处理）
        volume_profile = self._get_volume_profile()
        
        current_time = datetime.utcnow()
        cumulative_qty = 0.0
        
        for i, volume_weight in enumerate(volume_profile):
            # 根据成交量权重计算分片大小
            slice_qty = parent_order.quantity * volume_weight
            slice_qty = max(slice_qty, self.config.min_order_size)
            
            if cumulative_qty + slice_qty > parent_order.quantity:
                slice_qty = parent_order.quantity - cumulative_qty
            
            if slice_qty <= 0:
                break
            
            slice_time = current_time + timedelta(minutes=i * (self.vwap_config.duration_minutes / len(volume_profile)))
            
            slice_order = SliceOrder(
                slice_id=f"{self.algorithm_id}_vwap_slice_{i}",
                parent_order_id=parent_order.client_order_id or "",
                symbol=parent_order.symbol,
                side=parent_order.side,
                quantity=slice_qty,
                price=None,  # VWAP通常使用市价单或接近市价的限价单
                order_type=OrderType.MARKET,
                scheduled_time=slice_time
            )
            
            slices.append(slice_order)
            cumulative_qty += slice_qty
            
            if cumulative_qty >= parent_order.quantity:
                break
        
        return slices
    
    def _get_volume_profile(self) -> List[float]:
        """获取成交量分布模式（简化实现）"""
        # 典型的U型成交量分布
        periods = max(10, self.vwap_config.duration_minutes // 5)
        profile = []
        
        for i in range(periods):
            # U型曲线：开盘和收盘成交量大，中间成交量小
            t = i / (periods - 1)
            volume_weight = 0.3 + 0.7 * (t**2 + (1-t)**2)
            profile.append(volume_weight)
        
        # 归一化
        total_weight = sum(profile)
        return [w / total_weight for w in profile]
    
    async def update_market_data(self, market_data: MarketData, order_book: OrderBook):
        """更新市场数据和VWAP计算"""
        self.current_market_data = market_data
        self.current_order_book = order_book
        
        # 更新VWAP计算
        if market_data.volume > 0:
            total_value = self.vwap_price * self.cumulative_volume + market_data.price * market_data.volume
            self.cumulative_volume += market_data.volume
            self.vwap_price = total_value / self.cumulative_volume
    
    async def on_slice_filled(self, slice_order: SliceOrder, fill_qty: float, fill_price: float):
        """分片订单成交回调"""
        self._update_weighted_avg_price(fill_qty, fill_price)
        self.log_debug(
            f"VWAP slice filled",
            slice_id=slice_order.slice_id,
            fill_qty=fill_qty,
            fill_price=fill_price,
            vwap_price=self.vwap_price,
            performance=((fill_price - self.vwap_price) / self.vwap_price if self.vwap_price > 0 else 0)
        )
    
    async def should_adjust_strategy(self) -> bool:
        """根据成交量情况调整策略"""
        if not self.current_market_data:
            return False
        
        # 如果当前成交量显著偏离预期，可能需要调整
        expected_volume = sum(self.historical_volumes) / len(self.historical_volumes) if self.historical_volumes else 0
        if expected_volume > 0:
            volume_ratio = self.current_market_data.volume / expected_volume
            return volume_ratio > 2.0 or volume_ratio < 0.5
        
        return False


class POVAlgorithm(ExecutionAlgorithm):
    """市场参与率算法"""
    
    @dataclass
    class POVConfig(AlgorithmConfig):
        target_participation_rate: float = 0.15  # 目标参与率 15%
        min_participation_rate: float = 0.05  # 最小参与率 5%
        max_participation_rate: float = 0.30  # 最大参与率 30%
        volume_measurement_window: int = 300  # 成交量测量窗口(秒)
        adjustment_frequency: int = 60  # 调整频率(秒)
    
    def __init__(self, algorithm_id: str, config: POVConfig):
        super().__init__(algorithm_id, config)
        self.pov_config = config
        self.recent_volumes: List[Tuple[datetime, float]] = []  # (时间, 成交量)
        self.current_participation_rate = config.target_participation_rate
    
    async def initialize(self, parent_order: Order) -> List[SliceOrder]:
        """初始化POV算法"""
        # POV算法动态生成分片，这里先创建初始分片
        slice_order = SliceOrder(
            slice_id=f"{self.algorithm_id}_pov_slice_0",
            parent_order_id=parent_order.client_order_id or "",
            symbol=parent_order.symbol,
            side=parent_order.side,
            quantity=min(parent_order.quantity * 0.1, self.config.max_order_size),  # 初始10%
            price=parent_order.price,
            order_type=OrderType.LIMIT,
            scheduled_time=datetime.utcnow()
        )
        
        return [slice_order]
    
    async def update_market_data(self, market_data: MarketData, order_book: OrderBook):
        """更新市场数据"""
        self.current_market_data = market_data
        self.current_order_book = order_book
        
        # 记录成交量
        current_time = datetime.utcnow()
        self.recent_volumes.append((current_time, market_data.volume))
        
        # 清理过期数据
        cutoff_time = current_time - timedelta(seconds=self.pov_config.volume_measurement_window)
        self.recent_volumes = [(t, v) for t, v in self.recent_volumes if t > cutoff_time]
    
    async def on_slice_filled(self, slice_order: SliceOrder, fill_qty: float, fill_price: float):
        """分片订单成交回调"""
        self._update_weighted_avg_price(fill_qty, fill_price)
        
        # 动态生成下一个分片
        remaining_qty = sum(slice.quantity for slice in self.slices if slice.status == OrderStatus.NEW)
        if remaining_qty > 0:
            await self._generate_next_slice(slice_order)
    
    async def _generate_next_slice(self, completed_slice: SliceOrder):
        """生成下一个分片"""
        # 计算剩余数量
        total_target = sum(slice.quantity for slice in self.slices)
        remaining_qty = total_target - self.total_filled
        
        if remaining_qty <= 0:
            return
        
        # 根据当前市场成交量计算下一个分片大小
        market_volume_rate = self._calculate_market_volume_rate()
        if market_volume_rate > 0:
            next_slice_qty = market_volume_rate * self.current_participation_rate
            next_slice_qty = min(next_slice_qty, remaining_qty)
            next_slice_qty = max(next_slice_qty, self.config.min_order_size)
            next_slice_qty = min(next_slice_qty, self.config.max_order_size)
            
            next_slice = SliceOrder(
                slice_id=f"{self.algorithm_id}_pov_slice_{len(self.slices)}",
                parent_order_id=completed_slice.parent_order_id,
                symbol=completed_slice.symbol,
                side=completed_slice.side,
                quantity=next_slice_qty,
                price=completed_slice.price,
                order_type=completed_slice.order_type,
                scheduled_time=datetime.utcnow() + timedelta(seconds=5)  # 5秒后执行
            )
            
            self.slices.append(next_slice)
    
    def _calculate_market_volume_rate(self) -> float:
        """计算市场成交量速率"""
        if len(self.recent_volumes) < 2:
            return 0.0
        
        total_volume = sum(v for _, v in self.recent_volumes)
        time_span = (self.recent_volumes[-1][0] - self.recent_volumes[0][0]).total_seconds()
        
        if time_span > 0:
            return total_volume / time_span
        
        return 0.0
    
    async def should_adjust_strategy(self) -> bool:
        """检查是否需要调整参与率"""
        current_time = datetime.utcnow()
        if hasattr(self, '_last_adjustment_time'):
            time_since_adjustment = (current_time - self._last_adjustment_time).total_seconds()
            if time_since_adjustment < self.pov_config.adjustment_frequency:
                return False
        
        self._last_adjustment_time = current_time
        return True
    
    async def _adjust_strategy(self):
        """调整参与率策略"""
        # 根据市场状况调整参与率
        if self.current_order_book:
            spread = self.current_order_book.spread
            mid_price = self.current_order_book.mid_price
            
            if mid_price > 0:
                spread_pct = spread / mid_price
                
                # 如果价差较大，降低参与率
                if spread_pct > 0.001:  # 0.1%
                    self.current_participation_rate = max(
                        self.current_participation_rate * 0.8,
                        self.pov_config.min_participation_rate
                    )
                else:
                    # 价差较小，可以稍微提高参与率
                    self.current_participation_rate = min(
                        self.current_participation_rate * 1.1,
                        self.pov_config.max_participation_rate
                    )
                
                self.log_debug(
                    f"Adjusted participation rate",
                    new_rate=self.current_participation_rate,
                    spread_pct=spread_pct
                )


class ImplementationShortfall(ExecutionAlgorithm):
    """实施缺口算法"""
    
    @dataclass
    class ISConfig(AlgorithmConfig):
        risk_aversion: float = 0.5  # 风险厌恶系数
        volatility_estimate: float = 0.02  # 波动率估计 2%
        temporary_impact_coefficient: float = 0.5  # 临时冲击系数
        permanent_impact_coefficient: float = 0.3  # 永久冲击系数
        max_duration_minutes: int = 120  # 最大执行时长
    
    def __init__(self, algorithm_id: str, config: ISConfig):
        super().__init__(algorithm_id, config)
        self.is_config = config
        self.optimal_rate: Optional[float] = None
        self.decision_price: Optional[float] = None
    
    async def initialize(self, parent_order: Order) -> List[SliceOrder]:
        """初始化IS算法"""
        self.decision_price = parent_order.price or (
            self.current_market_data.price if self.current_market_data else 50000.0
        )
        
        # 计算最优执行率
        self.optimal_rate = self._calculate_optimal_rate(parent_order)
        
        # 生成分片
        slices = []
        total_time = self.is_config.max_duration_minutes * 60  # 转换为秒
        slice_interval = 60  # 每分钟一个分片
        
        current_time = datetime.utcnow()
        remaining_qty = parent_order.quantity
        
        for i in range(0, total_time, slice_interval):
            if remaining_qty <= 0:
                break
            
            # 根据最优执行率计算分片大小
            slice_qty = min(
                remaining_qty * self.optimal_rate * (slice_interval / 60),  # 转换为分钟
                remaining_qty
            )
            slice_qty = max(slice_qty, self.config.min_order_size)
            
            slice_time = current_time + timedelta(seconds=i)
            
            slice_order = SliceOrder(
                slice_id=f"{self.algorithm_id}_is_slice_{i//slice_interval}",
                parent_order_id=parent_order.client_order_id or "",
                symbol=parent_order.symbol,
                side=parent_order.side,
                quantity=slice_qty,
                price=parent_order.price,
                order_type=OrderType.LIMIT,
                scheduled_time=slice_time
            )
            
            slices.append(slice_order)
            remaining_qty -= slice_qty
        
        return slices
    
    def _calculate_optimal_rate(self, parent_order: Order) -> float:
        """计算最优执行率"""
        # 简化的实施缺口优化
        # 最优执行率 = sqrt(volatility^2 / (risk_aversion * temporary_impact))
        
        volatility = self.is_config.volatility_estimate
        risk_aversion = self.is_config.risk_aversion
        temporary_impact = self.is_config.temporary_impact_coefficient
        
        if temporary_impact > 0 and risk_aversion > 0:
            optimal_rate = math.sqrt(volatility**2 / (risk_aversion * temporary_impact))
            # 限制在合理范围内
            return min(max(optimal_rate, 0.01), 1.0)  # 1% - 100%
        
        return 0.1  # 默认10%
    
    async def update_market_data(self, market_data: MarketData, order_book: OrderBook):
        """更新市场数据"""
        self.current_market_data = market_data
        self.current_order_book = order_book
    
    async def on_slice_filled(self, slice_order: SliceOrder, fill_qty: float, fill_price: float):
        """分片订单成交回调"""
        self._update_weighted_avg_price(fill_qty, fill_price)
        
        # 计算当前实施缺口
        if self.decision_price:
            current_shortfall = (fill_price - self.decision_price) / self.decision_price
            if slice_order.side == OrderSide.SELL:
                current_shortfall = -current_shortfall
            
            self.log_debug(
                f"IS slice filled",
                slice_id=slice_order.slice_id,
                fill_price=fill_price,
                decision_price=self.decision_price,
                shortfall=current_shortfall
            )
    
    async def should_adjust_strategy(self) -> bool:
        """IS算法根据市场冲击调整策略"""
        if not self.current_market_data or not self.decision_price:
            return False
        
        # 如果价格偏离决策价格较大，可能需要调整
        price_deviation = abs(self.current_market_data.price - self.decision_price) / self.decision_price
        return price_deviation > 0.01  # 1%阈值
    
    async def _adjust_strategy(self):
        """调整IS策略"""
        if not self.current_market_data or not self.decision_price:
            return
        
        # 根据价格冲击调整执行速度
        price_impact = (self.current_market_data.price - self.decision_price) / self.decision_price
        
        # 如果价格不利变动较大，加速执行
        if abs(price_impact) > 0.005:  # 0.5%
            # 缩短剩余分片的间隔
            remaining_slices = [s for s in self.slices if s.status == OrderStatus.NEW]
            if remaining_slices:
                current_time = datetime.utcnow()
                for i, slice_order in enumerate(remaining_slices):
                    # 将执行时间提前
                    slice_order.scheduled_time = current_time + timedelta(seconds=i * 30)  # 30秒间隔
                
                self.log_info(
                    f"Adjusted IS strategy due to price impact",
                    price_impact=price_impact,
                    remaining_slices=len(remaining_slices)
                )