"""
智能订单路由器

此模块实现了高频交易中的智能订单路由功能，包括：
1. 动态订单分割和路由
2. 执行算法选择  
3. 市场冲击最小化
4. 实时风险控制
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import numpy as np

from src.utils.logger import LoggerMixin
from src.hft.integrated_signal_processor import OrderRequest, OrderType
from src.core.models.trading import MarketData


class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    TWAP = "twap"  # Time Weighted Average Price
    VWAP = "vwap"  # Volume Weighted Average Price  
    IMPLEMENTATION_SHORTFALL = "shortfall"  # 实施缺陷算法
    MARKET_ON_CLOSE = "moc"  # 收盘价交易
    ICEBERG = "iceberg"  # 冰山算法
    SNIPER = "sniper"  # 狙击算法


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    ROUTING = "routing"
    PARTIALLY_FILLED = "partially_filled" 
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class ChildOrder:
    """子订单"""
    parent_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: OrderType
    venue: str
    algorithm: ExecutionAlgorithm
    created_time: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """执行报告"""
    order_id: str
    symbol: str
    total_quantity: float
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    total_cost: float
    market_impact: float
    execution_time_ms: float
    child_orders: List[ChildOrder] = field(default_factory=list)
    slippage: float = 0.0
    commission: float = 0.0
    success_rate: float = 0.0


@dataclass
class VenueInfo:
    """交易场所信息"""
    name: str
    priority: int
    latency_ms: float
    liquidity_score: float  # 0-1
    fee_rate: float
    min_order_size: float
    max_order_size: float
    is_active: bool = True
    recent_fill_rate: float = 0.0
    average_slippage: float = 0.0


class SmartOrderRouter(LoggerMixin):
    """
    智能订单路由器
    
    核心功能：
    1. 订单分割和时间调度
    2. 执行场所选择和优化
    3. 算法执行和风险控制
    4. 实时监控和调整
    """
    
    def __init__(self, 
                 max_child_orders: int = 20,
                 max_order_value: float = 100000.0,
                 default_slice_size: float = 0.1):
        """
        初始化智能订单路由器
        
        Args:
            max_child_orders: 最大子订单数量
            max_order_value: 最大订单价值
            default_slice_size: 默认分片大小比例
        """
        super().__init__()
        
        self.max_child_orders = max_child_orders
        self.max_order_value = max_order_value  
        self.default_slice_size = default_slice_size
        
        # 交易场所配置
        self.venues: Dict[str, VenueInfo] = {}
        self._initialize_default_venues()
        
        # 活跃订单管理
        self.active_orders: Dict[str, OrderRequest] = {}
        self.child_orders: Dict[str, List[ChildOrder]] = {}
        self.execution_reports: Dict[str, ExecutionReport] = {}
        
        # 市场数据缓存
        self.market_data_cache: Dict[str, MarketData] = {}
        self.order_book_cache: Dict[str, Dict] = {}
        
        # 运行状态
        self._running = False
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # 性能统计
        self.routing_stats = {
            "total_orders": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "avg_execution_time_ms": 0.0,
            "avg_market_impact": 0.0,
            "total_slippage": 0.0
        }
        
        # 回调函数
        self.execution_callbacks: List[Callable[[ExecutionReport], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        
    def _initialize_default_venues(self):
        """初始化默认交易场所"""
        self.venues = {
            "binance": VenueInfo(
                name="binance",
                priority=1,
                latency_ms=10.0,
                liquidity_score=0.9,
                fee_rate=0.001,
                min_order_size=0.001,
                max_order_size=1000.0
            ),
            "okex": VenueInfo(
                name="okex", 
                priority=2,
                latency_ms=15.0,
                liquidity_score=0.8,
                fee_rate=0.0015,
                min_order_size=0.001,
                max_order_size=500.0
            ),
            "huobi": VenueInfo(
                name="huobi",
                priority=3,
                latency_ms=20.0,
                liquidity_score=0.7,
                fee_rate=0.002,
                min_order_size=0.001,
                max_order_size=300.0
            )
        }
    
    async def start(self):
        """启动智能订单路由器"""
        if self._running:
            return
            
        self._running = True
        self.log_info("智能订单路由器已启动")
    
    async def stop(self):
        """停止智能订单路由器"""
        self._running = False
        
        # 取消所有执行任务
        for task in self._execution_tasks.values():
            task.cancel()
            
        await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)
        self._execution_tasks.clear()
        
        self.log_info("智能订单路由器已停止")
    
    async def route_order(self, order: OrderRequest, market_data: MarketData) -> Optional[str]:
        """
        路由订单到最优执行路径
        
        Args:
            order: 原始订单请求
            market_data: 市场数据
            
        Returns:
            execution_id: 执行ID，失败时返回None
        """
        start_time = time.perf_counter()
        execution_id = f"{order.symbol}_{int(time.time()*1000)}"
        
        try:
            self.routing_stats["total_orders"] += 1
            
            # 步骤1: 订单验证和风险检查
            if not await self._validate_order(order, market_data):
                self.log_warning(f"订单验证失败: {order.signal_id}")
                return None
            
            # 步骤2: 选择执行算法
            algorithm = self._select_execution_algorithm(order, market_data)
            
            # 步骤3: 订单分片
            child_orders = await self._slice_order(order, market_data, algorithm)
            if not child_orders:
                self.log_error(f"订单分片失败: {order.signal_id}")
                return None
            
            # 步骤4: 选择交易场所
            await self._assign_venues(child_orders, market_data)
            
            # 步骤5: 创建执行报告
            execution_report = ExecutionReport(
                order_id=execution_id,
                symbol=order.symbol,
                total_quantity=order.quantity,
                filled_quantity=0.0,
                remaining_quantity=order.quantity,
                avg_fill_price=0.0,
                total_cost=0.0,
                market_impact=0.0,
                execution_time_ms=0.0,
                child_orders=child_orders
            )
            
            # 步骤6: 存储执行信息
            self.active_orders[execution_id] = order
            self.child_orders[execution_id] = child_orders
            self.execution_reports[execution_id] = execution_report
            self.market_data_cache[order.symbol] = market_data
            
            # 步骤7: 启动异步执行
            execution_task = asyncio.create_task(
                self._execute_order_async(execution_id, algorithm)
            )
            self._execution_tasks[execution_id] = execution_task
            
            execution_time = (time.perf_counter() - start_time) * 1000
            self.routing_stats["successful_routes"] += 1
            
            self.log_info(
                f"订单路由成功: {order.symbol}",
                execution_id=execution_id,
                algorithm=algorithm.name,
                child_orders_count=len(child_orders),
                routing_time_ms=execution_time
            )
            
            return execution_id
            
        except Exception as e:
            self.routing_stats["failed_routes"] += 1
            self.log_error(f"订单路由失败: {e}")
            await self._handle_routing_error(execution_id, e)
            return None
    
    async def _validate_order(self, order: OrderRequest, market_data: MarketData) -> bool:
        """验证订单"""
        # 基本参数检查
        if order.quantity <= 0:
            return False
            
        if order.price and order.price <= 0:
            return False
        
        # 价值限制检查
        estimated_value = order.quantity * (order.price or market_data.close)
        if estimated_value > self.max_order_value:
            self.log_warning(f"订单价值超限: {estimated_value} > {self.max_order_value}")
            return False
        
        # 市场状态检查
        if market_data.volume <= 0:
            self.log_warning(f"市场无成交量: {order.symbol}")
            return False
            
        return True
    
    def _select_execution_algorithm(self, order: OrderRequest, market_data: MarketData) -> ExecutionAlgorithm:
        """选择执行算法"""
        # 根据订单特征和市场状况选择算法
        
        if order.urgency_score > 0.9:
            # 高紧急度使用狙击算法
            return ExecutionAlgorithm.SNIPER
            
        elif order.quantity * (order.price or market_data.close) > 50000:
            # 大额订单使用冰山算法
            return ExecutionAlgorithm.ICEBERG
            
        elif order.order_type == OrderType.MARKET:
            # 市价单使用TWAP
            return ExecutionAlgorithm.TWAP
            
        elif market_data.volume > np.mean([market_data.volume]) * 1.5:
            # 高成交量时使用VWAP
            return ExecutionAlgorithm.VWAP
            
        else:
            # 默认使用实施缺陷算法
            return ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
    
    async def _slice_order(self, order: OrderRequest, market_data: MarketData, algorithm: ExecutionAlgorithm) -> List[ChildOrder]:
        """订单分片"""
        child_orders = []
        
        try:
            # 计算分片参数
            slice_params = self._calculate_slice_parameters(order, market_data, algorithm)
            total_quantity = order.quantity
            remaining_quantity = total_quantity
            slice_count = 0
            
            while remaining_quantity > 0 and slice_count < self.max_child_orders:
                # 计算当前分片大小
                slice_size = self._calculate_slice_size(
                    remaining_quantity, slice_params, slice_count
                )
                
                # 确保最小分片大小
                if slice_size < 0.001:
                    slice_size = remaining_quantity
                    
                slice_size = min(slice_size, remaining_quantity)
                
                # 创建子订单
                child_order = ChildOrder(
                    parent_id=order.signal_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=slice_size,
                    price=order.price,
                    order_type=order.order_type,
                    venue="",  # 稍后分配
                    algorithm=algorithm,
                    metadata={
                        "slice_number": slice_count + 1,
                        "total_slices": min(self.max_child_orders, int(total_quantity / slice_size) + 1),
                        "parent_urgency": order.urgency_score,
                        "parent_confidence": order.confidence
                    }
                )
                
                child_orders.append(child_order)
                remaining_quantity -= slice_size
                slice_count += 1
            
            # 处理剩余数量（如果有）
            if remaining_quantity > 0:
                if child_orders:
                    child_orders[-1].quantity += remaining_quantity
                else:
                    # 如果没有子订单，创建一个包含全部数量的订单
                    child_orders.append(ChildOrder(
                        parent_id=order.signal_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=total_quantity,
                        price=order.price,
                        order_type=order.order_type,
                        venue="",
                        algorithm=algorithm
                    ))
            
        except Exception as e:
            self.log_error(f"订单分片失败: {e}")
            return []
        
        return child_orders
    
    def _calculate_slice_parameters(self, order: OrderRequest, market_data: MarketData, algorithm: ExecutionAlgorithm) -> Dict[str, float]:
        """计算分片参数"""
        params = {}
        
        if algorithm == ExecutionAlgorithm.TWAP:
            # TWAP: 时间均匀分布
            params["time_factor"] = 1.0
            params["volume_factor"] = 0.5
            
        elif algorithm == ExecutionAlgorithm.VWAP:
            # VWAP: 基于历史成交量分布
            params["time_factor"] = 0.3
            params["volume_factor"] = 1.0
            
        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # 冰山: 小批量隐藏大订单
            params["slice_ratio"] = 0.05  # 每次显示5%
            params["min_slices"] = 10
            
        elif algorithm == ExecutionAlgorithm.SNIPER:
            # 狙击: 快速执行
            params["slice_ratio"] = 0.5  # 较大分片
            params["urgency_multiplier"] = 2.0
            
        else:
            # 默认参数
            params["slice_ratio"] = self.default_slice_size
            params["time_factor"] = 0.7
            params["volume_factor"] = 0.7
        
        return params
    
    def _calculate_slice_size(self, remaining_quantity: float, params: Dict[str, float], slice_index: int) -> float:
        """计算分片大小"""
        if "slice_ratio" in params:
            # 固定比例分片
            base_size = remaining_quantity * params["slice_ratio"]
        else:
            # 动态分片
            time_weight = params.get("time_factor", 1.0)
            volume_weight = params.get("volume_factor", 1.0)
            
            # 基于时间和成交量的权重计算
            base_size = remaining_quantity * self.default_slice_size
            
            # 添加一些随机性避免模式识别
            randomness = np.random.uniform(0.8, 1.2)
            base_size *= randomness
        
        return max(0.001, min(base_size, remaining_quantity))
    
    async def _assign_venues(self, child_orders: List[ChildOrder], market_data: MarketData):
        """为子订单分配交易场所"""
        # 获取活跃场所并按优先级排序
        active_venues = [
            venue for venue in self.venues.values() 
            if venue.is_active and venue.min_order_size <= min(co.quantity for co in child_orders)
        ]
        
        if not active_venues:
            self.log_error("没有可用的交易场所")
            return
        
        active_venues.sort(key=lambda v: (v.priority, -v.liquidity_score, v.latency_ms))
        
        # 分配策略：轮询分配以分散风险
        for i, child_order in enumerate(child_orders):
            venue = active_venues[i % len(active_venues)]
            child_order.venue = venue.name
            
            # 根据场所特性调整订单参数
            if venue.max_order_size < child_order.quantity:
                # 如果订单超过场所限制，需要进一步分割
                # 这里简化处理，实际应该重新分片
                child_order.quantity = venue.max_order_size
    
    async def _execute_order_async(self, execution_id: str, algorithm: ExecutionAlgorithm):
        """异步执行订单"""
        start_time = time.perf_counter()
        
        try:
            child_orders = self.child_orders[execution_id]
            execution_report = self.execution_reports[execution_id]
            
            if algorithm == ExecutionAlgorithm.TWAP:
                await self._execute_twap(child_orders, execution_report)
                
            elif algorithm == ExecutionAlgorithm.VWAP:
                await self._execute_vwap(child_orders, execution_report)
                
            elif algorithm == ExecutionAlgorithm.ICEBERG:
                await self._execute_iceberg(child_orders, execution_report)
                
            elif algorithm == ExecutionAlgorithm.SNIPER:
                await self._execute_sniper(child_orders, execution_report)
                
            else:
                await self._execute_default(child_orders, execution_report)
            
            # 更新执行统计
            execution_time = (time.perf_counter() - start_time) * 1000
            execution_report.execution_time_ms = execution_time
            
            # 计算最终指标
            self._calculate_execution_metrics(execution_report)
            
            # 通知执行完成
            await self._notify_execution_complete(execution_report)
            
        except Exception as e:
            self.log_error(f"订单执行失败: {execution_id}, {e}")
            await self._handle_execution_error(execution_id, e)
            
        finally:
            # 清理执行任务
            self._execution_tasks.pop(execution_id, None)
    
    async def _execute_twap(self, child_orders: List[ChildOrder], report: ExecutionReport):
        """执行TWAP算法"""
        total_time = 30.0  # 30秒内完成
        time_per_order = total_time / len(child_orders)
        
        for child_order in child_orders:
            if not self._running:
                break
                
            await self._send_child_order(child_order, report)
            
            # TWAP需要时间间隔
            if child_order != child_orders[-1]:  # 最后一个订单不需要等待
                await asyncio.sleep(time_per_order)
    
    async def _execute_vwap(self, child_orders: List[ChildOrder], report: ExecutionReport):
        """执行VWAP算法"""
        # VWAP算法根据历史成交量模式调整发送时机
        # 这里简化为与TWAP相似的实现
        await self._execute_twap(child_orders, report)
    
    async def _execute_iceberg(self, child_orders: List[ChildOrder], report: ExecutionReport):
        """执行冰山算法"""
        # 冰山算法一次只显示一个小订单
        for child_order in child_orders:
            if not self._running:
                break
                
            await self._send_child_order(child_order, report)
            
            # 等待当前订单部分成交后再发送下一个
            await asyncio.sleep(2.0)  # 简化的等待逻辑
    
    async def _execute_sniper(self, child_orders: List[ChildOrder], report: ExecutionReport):
        """执行狙击算法"""
        # 狙击算法快速发送所有订单
        tasks = []
        for child_order in child_orders:
            task = asyncio.create_task(self._send_child_order(child_order, report))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_default(self, child_orders: List[ChildOrder], report: ExecutionReport):
        """执行默认算法"""
        # 默认顺序执行
        for child_order in child_orders:
            if not self._running:
                break
            await self._send_child_order(child_order, report)
            await asyncio.sleep(0.5)  # 短暂延迟
    
    async def _send_child_order(self, child_order: ChildOrder, report: ExecutionReport):
        """发送子订单"""
        try:
            child_order.status = OrderStatus.ROUTING
            
            # 模拟订单发送和执行（实际应该调用交易所API）
            await asyncio.sleep(0.1)  # 模拟网络延迟
            
            # 模拟执行结果
            fill_success = np.random.random() > 0.1  # 90%成功率
            
            if fill_success:
                child_order.status = OrderStatus.FILLED
                child_order.filled_quantity = child_order.quantity
                child_order.avg_fill_price = child_order.price or report.avg_fill_price or 100.0
                
                # 更新执行报告
                report.filled_quantity += child_order.filled_quantity
                report.remaining_quantity -= child_order.filled_quantity
                
                # 更新平均成交价
                if report.filled_quantity > 0:
                    total_cost = report.avg_fill_price * (report.filled_quantity - child_order.filled_quantity)
                    total_cost += child_order.avg_fill_price * child_order.filled_quantity
                    report.avg_fill_price = total_cost / report.filled_quantity
                    report.total_cost = total_cost
                
                self.log_debug(f"子订单执行成功: {child_order.parent_id}")
            else:
                child_order.status = OrderStatus.REJECTED
                self.log_warning(f"子订单执行失败: {child_order.parent_id}")
                
        except Exception as e:
            child_order.status = OrderStatus.ERROR
            self.log_error(f"发送子订单时出错: {e}")
    
    def _calculate_execution_metrics(self, report: ExecutionReport):
        """计算执行指标"""
        if report.filled_quantity <= 0:
            return
            
        # 计算成功率
        total_children = len(report.child_orders)
        successful_children = sum(1 for co in report.child_orders if co.status == OrderStatus.FILLED)
        report.success_rate = successful_children / total_children if total_children > 0 else 0.0
        
        # 计算市场影响（简化）
        market_data = self.market_data_cache.get(report.symbol)
        if market_data and report.avg_fill_price > 0:
            expected_price = market_data.close
            report.market_impact = abs(report.avg_fill_price - expected_price) / expected_price
            report.slippage = (report.avg_fill_price - expected_price) / expected_price
        
        # 更新全局统计
        if report.success_rate > 0.8:  # 成功执行
            current_count = self.routing_stats["successful_routes"]
            self.routing_stats["avg_execution_time_ms"] = (
                (self.routing_stats["avg_execution_time_ms"] * (current_count - 1) + 
                 report.execution_time_ms) / current_count
            )
            self.routing_stats["avg_market_impact"] = (
                (self.routing_stats["avg_market_impact"] * (current_count - 1) + 
                 report.market_impact) / current_count  
            )
            self.routing_stats["total_slippage"] += abs(report.slippage)
    
    async def _notify_execution_complete(self, report: ExecutionReport):
        """通知执行完成"""
        try:
            for callback in self.execution_callbacks:
                callback(report)
            
            self.log_info(
                f"订单执行完成: {report.symbol}",
                execution_id=report.order_id,
                filled_rate=report.filled_quantity / report.total_quantity,
                avg_price=report.avg_fill_price,
                market_impact=report.market_impact,
                execution_time_ms=report.execution_time_ms
            )
            
        except Exception as e:
            self.log_error(f"执行通知时出错: {e}")
    
    async def _handle_routing_error(self, execution_id: str, error: Exception):
        """处理路由错误"""
        for callback in self.error_callbacks:
            try:
                callback(execution_id, error)
            except Exception as cb_error:
                self.log_error(f"错误回调失败: {cb_error}")
    
    async def _handle_execution_error(self, execution_id: str, error: Exception):
        """处理执行错误"""
        if execution_id in self.execution_reports:
            report = self.execution_reports[execution_id]
            report.market_impact = -1.0  # 标记为失败
        
        await self._handle_routing_error(execution_id, error)
    
    # 公共接口
    def add_execution_callback(self, callback: Callable[[ExecutionReport], None]):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def get_execution_report(self, execution_id: str) -> Optional[ExecutionReport]:
        """获取执行报告"""
        return self.execution_reports.get(execution_id)
    
    def get_active_executions(self) -> List[str]:
        """获取活跃执行列表"""
        return list(self._execution_tasks.keys())
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return dict(self.routing_stats)
    
    def update_venue_config(self, venue_name: str, config: Dict[str, Any]):
        """更新交易场所配置"""
        if venue_name in self.venues:
            venue = self.venues[venue_name]
            for key, value in config.items():
                if hasattr(venue, key):
                    setattr(venue, key, value)
            
            self.log_info(f"场所配置已更新: {venue_name}", **config)
        else:
            self.log_warning(f"未知交易场所: {venue_name}")
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        if execution_id in self._execution_tasks:
            task = self._execution_tasks[execution_id]
            task.cancel()
            
            # 更新子订单状态
            if execution_id in self.child_orders:
                for child_order in self.child_orders[execution_id]:
                    if child_order.status in [OrderStatus.PENDING, OrderStatus.ROUTING]:
                        child_order.status = OrderStatus.CANCELLED
            
            self.log_info(f"执行已取消: {execution_id}")
            return True
        
        return False