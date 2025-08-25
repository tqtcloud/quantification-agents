"""
高频交易信号处理集成系统

此模块实现了整合多维度技术指标引擎和延迟监控的高频交易信号处理系统，
包括信号过滤、订单生成、容错机制和异常处理。
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from collections import deque
import numpy as np

from src.utils.logger import LoggerMixin
from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, DataSourceStatus
from src.hft.signal_processor import LatencySensitiveSignalProcessor, SignalPriority, ActionType
from src.hft.microstructure_analyzer import MicrostructureSignal
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
from src.core.models.trading import MarketData


class FilterResult(Enum):
    """信号过滤结果"""
    ACCEPTED = "accepted"
    REJECTED_CONFIDENCE = "rejected_confidence"
    REJECTED_STRENGTH = "rejected_strength"  
    REJECTED_RISK = "rejected_risk"
    REJECTED_LATENCY = "rejected_latency"
    REJECTED_CORRELATION = "rejected_correlation"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    signal_id: str = ""
    urgency_score: float = 0.5
    confidence: float = 0.0
    risk_level: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def priority(self) -> SignalPriority:
        """根据紧急度和置信度计算优先级"""
        if self.urgency_score > 0.8 and self.confidence > 0.8:
            return SignalPriority.CRITICAL
        elif self.urgency_score > 0.6 and self.confidence > 0.6:
            return SignalPriority.HIGH
        elif self.urgency_score > 0.4:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW


@dataclass  
class ProcessingStats:
    """处理统计信息"""
    total_signals_received: int = 0
    signals_processed: int = 0
    signals_filtered_out: int = 0
    orders_generated: int = 0
    orders_sent: int = 0
    processing_errors: int = 0
    avg_processing_latency_ms: float = 0.0
    max_processing_latency_ms: float = 0.0
    filter_breakdown: Dict[str, int] = field(default_factory=dict)
    
    def update_latency(self, latency_ms: float):
        """更新延迟统计"""
        self.avg_processing_latency_ms = (
            (self.avg_processing_latency_ms * self.signals_processed + latency_ms) 
            / (self.signals_processed + 1)
        )
        self.max_processing_latency_ms = max(self.max_processing_latency_ms, latency_ms)


class IntegratedHFTSignalProcessor(LoggerMixin):
    """
    集成的高频交易信号处理系统
    
    功能：
    1. 整合多维度技术指标引擎输出
    2. 集成延迟监控和容错机制  
    3. 实现智能信号过滤逻辑
    4. 生成具体交易订单请求
    5. 提供完整的异常处理和容错
    """
    
    def __init__(self, 
                 multidimensional_engine: MultiDimensionalIndicatorEngine,
                 latency_monitor: LatencyMonitor,
                 signal_processor: LatencySensitiveSignalProcessor,
                 max_latency_ms: float = 5.0,
                 min_confidence_threshold: float = 0.6,
                 min_signal_strength: float = 0.4):
        """
        初始化集成信号处理器
        
        Args:
            multidimensional_engine: 多维度技术指标引擎
            latency_monitor: 延迟监控器
            signal_processor: 延迟敏感信号处理器  
            max_latency_ms: 最大允许延迟（毫秒）
            min_confidence_threshold: 最小置信度阈值
            min_signal_strength: 最小信号强度阈值
        """
        super().__init__()
        
        self.multidimensional_engine = multidimensional_engine
        self.latency_monitor = latency_monitor
        self.signal_processor = signal_processor
        
        # 过滤配置
        self.max_latency_ms = max_latency_ms
        self.min_confidence_threshold = min_confidence_threshold
        self.min_signal_strength = min_signal_strength
        
        # 风险管理配置
        self.max_position_size = 1.0
        self.max_daily_orders = 1000
        self.correlation_threshold = 0.8
        
        # 运行状态
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.stats = ProcessingStats()
        self.processing_times = deque(maxlen=1000)
        
        # 信号缓存和历史
        self.signal_cache: Dict[str, List[MultiDimensionalSignal]] = {}
        self.order_history: Dict[str, List[OrderRequest]] = {}
        self.active_positions: Dict[str, float] = {}  # symbol -> position_size
        
        # 回调函数
        self.order_callbacks: List[Callable[[OrderRequest], None]] = []
        self.error_callbacks: List[Callable[[Exception, Dict[str, Any]], None]] = []
        
        # 每日订单计数器
        self.daily_order_count = 0
        self.last_reset_date = time.strftime("%Y-%m-%d")
        
    async def start(self):
        """启动集成信号处理器"""
        if self._running:
            return
            
        self._running = True
        
        # 启动依赖组件
        await self.multidimensional_engine.start() if hasattr(self.multidimensional_engine, 'start') else None
        await self.latency_monitor.start()
        await self.signal_processor.start()
        
        # 启动处理循环
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        # 注册信号处理器回调
        self.signal_processor.add_action_callback(self._handle_microstructure_signal)
        
        self.log_info("集成HFT信号处理器已启动")
        
    async def stop(self):
        """停止集成信号处理器"""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # 停止依赖组件
        await self.signal_processor.stop()
        await self.latency_monitor.stop()
        
        self.log_info("集成HFT信号处理器已停止")
        
    async def process_market_data(self, market_data: MarketData) -> Optional[List[OrderRequest]]:
        """
        处理市场数据并生成交易订单
        
        Args:
            market_data: 市场数据
            
        Returns:
            生成的订单请求列表，如果处理失败则返回None
        """
        start_time = time.perf_counter()
        symbol = market_data.symbol
        
        try:
            self.stats.total_signals_received += 1
            
            # 步骤1: 延迟检查
            active_source = self.latency_monitor.get_active_data_source(symbol)
            if active_source:
                is_fresh, latency_metrics = await self.latency_monitor.check_data_freshness(
                    symbol, market_data, active_source
                )
                
                if not is_fresh:
                    self._record_filter_result(FilterResult.REJECTED_LATENCY)
                    self.log_warning(f"数据延迟过高，跳过处理: {symbol}")
                    return None
            
            # 步骤2: 生成多维度信号
            market_data_dict = self._convert_market_data_to_dict(market_data)
            multidimensional_signal = await self.multidimensional_engine.generate_multidimensional_signal(
                symbol, market_data_dict
            )
            
            if multidimensional_signal is None:
                self.stats.processing_errors += 1
                return None
                
            # 步骤3: 信号过滤
            filter_result, reason = await self._filter_signal(multidimensional_signal, market_data)
            if filter_result != FilterResult.ACCEPTED:
                self._record_filter_result(filter_result)
                self.log_debug(f"信号被过滤: {symbol}, 原因: {reason}")
                return None
            
            # 步骤4: 生成订单
            orders = await self._generate_orders(multidimensional_signal, market_data)
            
            # 步骤5: 订单后处理
            if orders:
                await self._post_process_orders(orders, multidimensional_signal)
            
            # 更新统计
            self.stats.signals_processed += 1
            self.stats.orders_generated += len(orders) if orders else 0
            
            processing_time = (time.perf_counter() - start_time) * 1000
            self.stats.update_latency(processing_time)
            self.processing_times.append(processing_time)
            
            return orders
            
        except Exception as e:
            self.stats.processing_errors += 1
            await self._handle_processing_error(e, {"symbol": symbol, "market_data": market_data})
            return None
    
    async def _filter_signal(self, signal: MultiDimensionalSignal, market_data: MarketData) -> Tuple[FilterResult, str]:
        """
        信号过滤逻辑
        
        Returns:
            (filter_result, reason): 过滤结果和原因
        """
        symbol = signal.primary_signal.symbol
        
        try:
            # 1. 置信度过滤
            if signal.overall_confidence < self.min_confidence_threshold:
                return FilterResult.REJECTED_CONFIDENCE, f"置信度过低: {signal.overall_confidence:.3f}"
            
            # 2. 信号强度过滤
            signal_strength = abs(signal.primary_signal.confidence)
            if signal_strength < self.min_signal_strength:
                return FilterResult.REJECTED_STRENGTH, f"信号强度不足: {signal_strength:.3f}"
            
            # 3. 风险管理过滤
            if await self._check_risk_limits(signal, market_data):
                return FilterResult.REJECTED_RISK, "风险限制违规"
            
            # 4. 相关性过滤（避免过度集中）
            if await self._check_correlation_risk(symbol):
                return FilterResult.REJECTED_CORRELATION, "相关性风险过高"
            
            # 5. 每日订单限制
            if self._check_daily_limits():
                return FilterResult.REJECTED_RISK, "每日订单限制"
                
            return FilterResult.ACCEPTED, "通过所有过滤条件"
            
        except Exception as e:
            self.log_error(f"信号过滤时出错: {e}")
            return FilterResult.REJECTED_RISK, f"过滤异常: {str(e)}"
    
    async def _check_risk_limits(self, signal: MultiDimensionalSignal, market_data: MarketData) -> bool:
        """检查风险限制"""
        symbol = signal.primary_signal.symbol
        
        # 检查最大仓位限制
        current_position = self.active_positions.get(symbol, 0.0)
        recommended_size = signal.get_position_sizing_recommendation()
        
        if abs(current_position + recommended_size) > self.max_position_size:
            return True
            
        # 检查风险收益比
        if signal.risk_reward_ratio < 1.0:  # 风险收益比至少1:1
            return True
            
        # 检查波动率风险
        if signal.volatility_score > 0.8:  # 高波动率环境
            return True
            
        return False
    
    async def _check_correlation_risk(self, symbol: str) -> bool:
        """检查相关性风险"""
        # 简化实现：检查是否有过多同类资产的活跃仓位
        active_symbols = set(self.active_positions.keys())
        if len(active_symbols) > 5:  # 超过5个活跃仓位认为风险较高
            return True
        return False
    
    def _check_daily_limits(self) -> bool:
        """检查每日订单限制"""
        current_date = time.strftime("%Y-%m-%d")
        
        # 日期切换时重置计数器
        if current_date != self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = current_date
            
        return self.daily_order_count >= self.max_daily_orders
    
    async def _generate_orders(self, signal: MultiDimensionalSignal, market_data: MarketData) -> List[OrderRequest]:
        """生成交易订单"""
        orders = []
        symbol = signal.primary_signal.symbol
        primary_signal = signal.primary_signal
        
        try:
            # 确定订单方向
            if primary_signal._is_buy_signal():
                side = "buy"
            elif primary_signal._is_sell_signal():
                side = "sell" 
            else:
                return orders  # 中性信号不生成订单
            
            # 计算订单数量
            base_quantity = signal.get_position_sizing_recommendation(
                base_position_size=0.1,  # 基础仓位10%
                risk_tolerance=0.8
            )
            
            # 根据信号强度和市场状况调整
            quantity_adjustment = self._calculate_quantity_adjustment(signal, market_data)
            final_quantity = base_quantity * quantity_adjustment
            
            # 确定订单类型和价格
            order_type, price, stop_price = self._determine_order_params(signal, market_data)
            
            # 创建主要订单
            main_order = OrderRequest(
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=final_quantity,
                price=price,
                stop_price=stop_price,
                signal_id=f"{symbol}_{int(time.time()*1000)}",
                urgency_score=abs(signal.momentum_score),
                confidence=signal.overall_confidence,
                risk_level=self._assess_risk_level(signal),
                metadata={
                    "signal_type": primary_signal.signal_type.name,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "volatility_score": signal.volatility_score,
                    "market_regime": signal.market_regime
                }
            )
            
            orders.append(main_order)
            
            # 根据信号强度添加额外订单（如止损、获利了结）
            if signal.overall_confidence > 0.8:
                protective_orders = self._generate_protective_orders(signal, market_data, main_order)
                orders.extend(protective_orders)
                
        except Exception as e:
            self.log_error(f"生成订单时出错: {e}")
            await self._handle_processing_error(e, {"signal": signal, "market_data": market_data})
        
        return orders
    
    def _calculate_quantity_adjustment(self, signal: MultiDimensionalSignal, market_data: MarketData) -> float:
        """计算数量调整系数"""
        adjustment = 1.0
        
        # 基于成交量调整
        if signal.volume_score > 0.7:
            adjustment *= 1.2  # 高成交量时增加仓位
        elif signal.volume_score < 0.3:
            adjustment *= 0.8  # 低成交量时减少仓位
        
        # 基于波动率调整
        if signal.volatility_score > 0.8:
            adjustment *= 0.6  # 高波动率时减少仓位
        elif signal.volatility_score < 0.2:
            adjustment *= 1.1  # 低波动率时略增仓位
        
        # 基于信号质量调整
        quality_score = signal.signal_quality_score
        adjustment *= (0.5 + quality_score * 0.5)
        
        return max(0.1, min(2.0, adjustment))  # 限制在0.1-2.0之间
    
    def _determine_order_params(self, signal: MultiDimensionalSignal, market_data: MarketData) -> Tuple[OrderType, Optional[float], Optional[float]]:
        """确定订单参数"""
        primary_signal = signal.primary_signal
        
        # 根据紧急程度和市场状况选择订单类型
        if signal.volatility_score > 0.8:
            # 高波动率环境使用限价单
            order_type = OrderType.LIMIT
            if primary_signal._is_buy_signal():
                price = market_data.bid + (market_data.ask - market_data.bid) * 0.3
            else:
                price = market_data.ask - (market_data.ask - market_data.bid) * 0.3
        elif abs(signal.momentum_score) > 0.8 and signal.overall_confidence > 0.9:
            # 强烈信号使用市价单
            order_type = OrderType.MARKET
            price = None
        else:
            # 默认使用IOC限价单
            order_type = OrderType.IOC
            price = market_data.close
        
        # 计算止损价格
        stop_price = None
        if primary_signal.stop_loss > 0:
            stop_price = primary_signal.stop_loss
        
        return order_type, price, stop_price
    
    def _assess_risk_level(self, signal: MultiDimensionalSignal) -> str:
        """评估风险等级"""
        risk_factors = [
            signal.volatility_score,
            1.0 - signal.overall_confidence,
            abs(signal.momentum_score) - 0.5,  # 极端动量也是风险
        ]
        
        avg_risk = sum(max(0, factor) for factor in risk_factors) / len(risk_factors)
        
        if avg_risk > 0.7:
            return "high"
        elif avg_risk > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_protective_orders(self, signal: MultiDimensionalSignal, market_data: MarketData, main_order: OrderRequest) -> List[OrderRequest]:
        """生成保护性订单（止损、获利了结）"""
        protective_orders = []
        primary_signal = signal.primary_signal
        
        try:
            # 止损订单
            if primary_signal.stop_loss > 0:
                stop_side = "sell" if main_order.side == "buy" else "buy"
                stop_order = OrderRequest(
                    symbol=main_order.symbol,
                    order_type=OrderType.STOP,
                    side=stop_side,
                    quantity=main_order.quantity,
                    stop_price=primary_signal.stop_loss,
                    time_in_force="GTC",
                    signal_id=f"{main_order.signal_id}_stop",
                    urgency_score=0.9,
                    confidence=signal.overall_confidence,
                    risk_level="high",
                    metadata={"parent_order": main_order.signal_id, "order_purpose": "stop_loss"}
                )
                protective_orders.append(stop_order)
            
            # 获利了结订单
            if primary_signal.target_price > 0 and signal.risk_reward_ratio > 1.5:
                target_side = "sell" if main_order.side == "buy" else "buy"
                target_order = OrderRequest(
                    symbol=main_order.symbol,
                    order_type=OrderType.LIMIT,
                    side=target_side,
                    quantity=main_order.quantity * 0.5,  # 部分获利了结
                    price=primary_signal.target_price,
                    time_in_force="GTC",
                    signal_id=f"{main_order.signal_id}_target",
                    urgency_score=0.6,
                    confidence=signal.overall_confidence,
                    risk_level="low",
                    metadata={"parent_order": main_order.signal_id, "order_purpose": "take_profit"}
                )
                protective_orders.append(target_order)
                
        except Exception as e:
            self.log_error(f"生成保护性订单时出错: {e}")
        
        return protective_orders
    
    async def _post_process_orders(self, orders: List[OrderRequest], signal: MultiDimensionalSignal):
        """订单后处理"""
        symbol = signal.primary_signal.symbol
        
        # 更新订单历史
        if symbol not in self.order_history:
            self.order_history[symbol] = []
        
        self.order_history[symbol].extend(orders)
        
        # 更新每日订单计数
        self.daily_order_count += len(orders)
        
        # 更新活跃仓位（简化处理）
        total_quantity = sum(
            order.quantity if order.side == "buy" else -order.quantity 
            for order in orders
            if order.metadata.get("order_purpose") != "stop_loss"
        )
        
        current_position = self.active_positions.get(symbol, 0.0)
        self.active_positions[symbol] = current_position + total_quantity
        
        # 发送订单通知
        for order in orders:
            await self._notify_order_generated(order)
    
    async def _notify_order_generated(self, order: OrderRequest):
        """通知订单生成"""
        try:
            for callback in self.order_callbacks:
                callback(order)
            
            self.stats.orders_sent += 1
            
            self.log_info(
                f"订单生成: {order.symbol} {order.side} {order.quantity} @ {order.price or 'MARKET'}",
                order_id=order.signal_id,
                priority=order.priority.name,
                confidence=order.confidence
            )
            
        except Exception as e:
            self.log_error(f"订单通知时出错: {e}")
    
    async def _handle_microstructure_signal(self, action):
        """处理微观结构信号"""
        # 将微观结构信号整合到主处理流程
        # 这里可以添加额外的微观结构信号处理逻辑
        pass
    
    async def _processing_loop(self):
        """主处理循环"""
        while self._running:
            try:
                # 执行定期维护任务
                await self._periodic_maintenance()
                await asyncio.sleep(10)  # 每10秒执行一次维护
                
            except Exception as e:
                self.log_error(f"处理循环出错: {e}")
                await asyncio.sleep(1)
    
    async def _periodic_maintenance(self):
        """定期维护任务"""
        try:
            # 清理过期的信号缓存
            current_time = time.time()
            for symbol, signals in list(self.signal_cache.items()):
                self.signal_cache[symbol] = [
                    s for s in signals 
                    if current_time - s.primary_signal.timestamp.timestamp() < 300  # 保留5分钟内的信号
                ]
            
            # 清理过期的订单历史  
            for symbol, orders in list(self.order_history.items()):
                self.order_history[symbol] = orders[-100:]  # 只保留最近100个订单
                
        except Exception as e:
            self.log_error(f"定期维护出错: {e}")
    
    async def _handle_processing_error(self, error: Exception, context: Dict[str, Any]):
        """处理处理错误"""
        self.log_error(f"处理错误: {error}", **context)
        
        for callback in self.error_callbacks:
            try:
                callback(error, context)
            except Exception as cb_error:
                self.log_error(f"错误回调失败: {cb_error}")
    
    def _record_filter_result(self, result: FilterResult):
        """记录过滤结果"""
        self.stats.signals_filtered_out += 1
        
        if result.value not in self.stats.filter_breakdown:
            self.stats.filter_breakdown[result.value] = 0
        self.stats.filter_breakdown[result.value] += 1
    
    def _convert_market_data_to_dict(self, market_data: MarketData) -> Dict[str, Any]:
        """转换市场数据为字典格式"""
        return {
            "symbol": market_data.symbol,
            "open": [market_data.open],
            "high": [market_data.high], 
            "low": [market_data.low],
            "close": [market_data.close],
            "volume": [market_data.volume],
            "timestamp": market_data.timestamp.timestamp()
        }
    
    # 公共接口方法
    def add_order_callback(self, callback: Callable[[OrderRequest], None]):
        """添加订单回调"""
        self.order_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception, Dict[str, Any]], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def get_processing_stats(self) -> ProcessingStats:
        """获取处理统计信息"""
        return self.stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self._running,
            "stats": {
                "total_received": self.stats.total_signals_received,
                "processed": self.stats.signals_processed,
                "filtered_out": self.stats.signals_filtered_out,
                "orders_generated": self.stats.orders_generated,
                "processing_errors": self.stats.processing_errors,
                "avg_latency_ms": self.stats.avg_processing_latency_ms,
                "max_latency_ms": self.stats.max_processing_latency_ms,
                "filter_breakdown": dict(self.stats.filter_breakdown)
            },
            "active_positions": dict(self.active_positions),
            "daily_order_count": self.daily_order_count,
            "latency_monitor_health": self.latency_monitor.get_system_health(),
            "signal_processor_status": self.signal_processor.get_status()
        }
    
    def update_configuration(self, config: Dict[str, Any]):
        """更新配置"""
        if "max_latency_ms" in config:
            self.max_latency_ms = config["max_latency_ms"]
            
        if "min_confidence_threshold" in config:
            self.min_confidence_threshold = config["min_confidence_threshold"]
            
        if "min_signal_strength" in config:
            self.min_signal_strength = config["min_signal_strength"]
            
        if "max_position_size" in config:
            self.max_position_size = config["max_position_size"]
            
        if "max_daily_orders" in config:
            self.max_daily_orders = config["max_daily_orders"]
            
        self.log_info("配置已更新", **config)