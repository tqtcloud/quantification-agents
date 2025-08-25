import asyncio
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from decimal import Decimal

from src.core.models import MarketData, Order
from src.hft.orderbook_manager import OrderBookManager, OrderBookSnapshot
from src.hft.microstructure_analyzer import MicrostructureAnalyzer, MicrostructureSignal
from src.hft.execution_engine import FastExecutionEngine, ExecutionOrder, OrderType, SlippageConfig
from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, AlertEvent, AlertLevel
from src.utils.logger import LoggerMixin


@dataclass
class HFTConfig:
    """高频交易配置"""
    # 订单簿配置
    max_orderbook_levels: int = 50
    orderbook_history_size: int = 1000
    
    # 微观结构分析配置
    microstructure_lookback: int = 100
    min_signal_strength: float = 0.3
    
    # 执行引擎配置
    max_orders: int = 10000
    slippage_config: SlippageConfig = field(default_factory=SlippageConfig)
    
    # 性能配置
    update_interval_ms: float = 1.0  # 更新间隔（毫秒）
    latency_target_ms: float = 10.0  # 延迟目标（毫秒）
    
    # 延迟监控配置
    staleness_threshold_ms: float = 100.0  # 数据过期阈值（毫秒）
    latency_stats_window: int = 1000  # 延迟统计窗口大小
    alert_cooldown_seconds: float = 60.0  # 告警冷却时间
    enable_latency_monitoring: bool = True  # 是否启用延迟监控
    
    # 风险配置
    max_position_value: Decimal = field(default_factory=lambda: Decimal("50000"))
    max_daily_trades: int = 1000
    emergency_stop_loss: float = 0.05  # 5% 紧急止损


@dataclass
class HFTMetrics:
    """高频交易指标"""
    timestamp: float = field(default_factory=time.time)
    total_updates: int = 0
    avg_update_latency: float = 0.0
    max_update_latency: float = 0.0
    total_signals: int = 0
    active_signals: int = 0
    total_orders: int = 0
    filled_orders: int = 0
    avg_execution_latency: float = 0.0
    
    # 延迟监控指标
    data_freshness_checks: int = 0
    stale_data_detections: int = 0
    data_source_switches: int = 0
    avg_data_latency_ms: float = 0.0
    p99_data_latency_ms: float = 0.0
    total_slippage_bps: float = 0.0
    pnl: Decimal = field(default_factory=lambda: Decimal("0"))


class HFTEngine(LoggerMixin):
    """高频交易引擎
    
    整合订单簿管理、微观结构分析和快速执行引擎，
    提供低延迟的高频交易能力。
    """
    
    def __init__(self, config: Optional[HFTConfig] = None):
        self.config = config or HFTConfig()
        
        # 初始化核心组件
        self.orderbook_manager = OrderBookManager(
            max_levels=self.config.max_orderbook_levels,
            history_size=self.config.orderbook_history_size
        )
        
        self.microstructure_analyzer = MicrostructureAnalyzer(
            lookback_window=self.config.microstructure_lookback,
            min_signal_strength=self.config.min_signal_strength
        )
        
        self.execution_engine = FastExecutionEngine(
            orderbook_manager=self.orderbook_manager,
            max_orders=self.config.max_orders,
            slippage_config=self.config.slippage_config
        )
        
        # 延迟监控系统
        self.latency_monitor: Optional[LatencyMonitor] = None
        if self.config.enable_latency_monitoring:
            self.latency_monitor = LatencyMonitor(
                staleness_threshold_ms=self.config.staleness_threshold_ms,
                stats_window_size=self.config.latency_stats_window,
                alert_cooldown_seconds=self.config.alert_cooldown_seconds
            )
            # 注册告警回调
            self.latency_monitor.add_alert_callback(self._on_latency_alert)
        
        # 状态管理
        self._running = False
        self._symbols: List[str] = []
        self._update_task: Optional[asyncio.Task] = None
        
        # 性能监控
        self.metrics = HFTMetrics()
        self._update_times: List[float] = []
        
        # 回调函数
        self.signal_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        
        # 注册执行引擎回调
        self.execution_engine.add_execution_callback(self._on_order_executed)
        
    async def initialize(self, symbols: List[str], data_sources: Optional[List[DataSourceConfig]] = None):
        """初始化HFT引擎"""
        self._symbols = symbols.copy()
        
        # 初始化各组件
        await self.orderbook_manager.initialize(symbols)
        await self.microstructure_analyzer.initialize(symbols)
        
        # 初始化延迟监控
        if self.latency_monitor and data_sources:
            await self.latency_monitor.initialize(data_sources)
            # 设置默认数据源（优先级最高的）
            if data_sources:
                primary_source = min(data_sources, key=lambda x: x.priority)
                for symbol in symbols:
                    self.latency_monitor.set_active_data_source(symbol, primary_source.name)
        
        self.log_info(f"HFT Engine initialized for {len(symbols)} symbols: {symbols}")
        
    async def start(self):
        """启动HFT引擎"""
        if self._running:
            return
            
        self._running = True
        
        # 启动执行引擎
        await self.execution_engine.start()
        
        # 启动延迟监控
        if self.latency_monitor:
            await self.latency_monitor.start()
        
        # 启动更新循环
        self._update_task = asyncio.create_task(self._update_loop())
        
        self.log_info("HFT Engine started")
        
    async def stop(self):
        """停止HFT引擎"""
        self._running = False
        
        # 停止更新循环
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # 停止延迟监控
        if self.latency_monitor:
            await self.latency_monitor.stop()
        
        # 停止执行引擎
        await self.execution_engine.stop()
        
        # 清理资源
        await self.orderbook_manager.cleanup()
        
        self.log_info("HFT Engine stopped")
        
    async def update_market_data(self, symbol: str, market_data: MarketData, data_source: Optional[str] = None) -> bool:
        """更新市场数据"""
        start_time = time.perf_counter()
        
        try:
            # 延迟监控检查
            is_fresh = True
            if self.latency_monitor and data_source:
                is_fresh, latency_metrics = await self.latency_monitor.check_data_freshness(
                    symbol, market_data, data_source
                )
                
                # 更新延迟指标
                self.metrics.data_freshness_checks += 1
                if not is_fresh:
                    self.metrics.stale_data_detections += 1
                    
                    # 尝试切换数据源
                    new_source = await self.latency_monitor.switch_data_source(
                        symbol, "Data staleness detected"
                    )
                    if new_source:
                        self.metrics.data_source_switches += 1
                        self.log_warning(f"Switched data source for {symbol}: {data_source} -> {new_source}")
                        # 这里应该通知上游系统使用新的数据源
                
                # 更新延迟统计
                latency_summary = self.latency_monitor.get_latency_summary(symbol)
                if latency_summary:
                    self.metrics.avg_data_latency_ms = latency_summary.get("avg_latency_ms", 0.0)
                    self.metrics.p99_data_latency_ms = latency_summary.get("p99_latency_ms", 0.0)
            
            # 如果数据不新鲜，可以选择跳过处理或使用降级策略
            if not is_fresh and self.config.staleness_threshold_ms > 0:
                self.log_debug(f"Skipping stale data for {symbol}")
                return False
            
            # 更新订单簿
            orderbook_updated = await self.orderbook_manager.update_orderbook(symbol, market_data)
            if not orderbook_updated:
                return False
            
            # 获取最新订单簿快照
            orderbook = self.orderbook_manager.get_orderbook(symbol)
            if not orderbook:
                return False
            
            # 微观结构分析
            signals = await self.microstructure_analyzer.update(symbol, orderbook, market_data)
            
            # 处理信号
            if signals:
                await self._process_signals(symbol, signals)
            
            # 更新性能指标
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_times.append(processing_time)
            
            # 保持统计数量
            if len(self._update_times) > 1000:
                self._update_times = self._update_times[-1000:]
            
            self.metrics.total_updates += 1
            self.metrics.avg_update_latency = sum(self._update_times) / len(self._update_times)
            self.metrics.max_update_latency = max(self._update_times)
            
            # 延迟警告
            if processing_time > self.config.latency_target_ms:
                self.log_warning(f"High latency detected: {processing_time:.2f}ms for {symbol}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Error updating market data for {symbol}: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: str, 
                         quantity: Decimal, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[Decimal] = None,
                         **kwargs) -> Optional[str]:
        """下单"""
        try:
            order = ExecutionOrder(
                order_id="",  # 由执行引擎生成
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )
            
            success = await self.execution_engine.submit_order(order)
            if success:
                self.metrics.total_orders += 1
                return order.order_id
            else:
                return None
                
        except Exception as e:
            self.log_error(f"Error placing order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        return await self.execution_engine.cancel_order(order_id)
    
    async def _update_loop(self):
        """更新循环"""
        while self._running:
            try:
                # 更新性能指标
                await self._update_metrics()
                
                # 检查紧急停止条件
                await self._check_emergency_stop()
                
                # 休眠
                await asyncio.sleep(self.config.update_interval_ms / 1000)
                
            except Exception as e:
                self.log_error(f"Error in update loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _process_signals(self, symbol: str, signals: List[MicrostructureSignal]):
        """处理微观结构信号"""
        for signal in signals:
            try:
                # 通知信号回调
                for callback in self.signal_callbacks:
                    await callback(signal)
                
                self.metrics.total_signals += 1
                
                # 基于信号类型的处理逻辑可以在这里扩展
                if signal.signal_type == "imbalance" and abs(signal.strength) > 0.7:
                    self.log_debug(f"Strong imbalance signal for {symbol}: {signal.strength:.3f}")
                elif signal.signal_type == "toxicity" and signal.strength > 0.8:
                    self.log_debug(f"High toxicity signal for {symbol}: {signal.strength:.3f}")
                elif signal.signal_type in ["price_anomaly", "volume_spike", "spread_anomaly"]:
                    self.log_debug(f"Anomaly detected for {symbol}: {signal.signal_type}")
                    
            except Exception as e:
                self.log_error(f"Error processing signal: {e}")
        
        # 更新活跃信号计数
        self.metrics.active_signals = sum(
            len(self.microstructure_analyzer.get_current_signals(sym)) 
            for sym in self._symbols
        )
    
    async def _update_metrics(self):
        """更新性能指标"""
        try:
            # 执行统计
            exec_stats = self.execution_engine.get_performance_stats()
            self.metrics.avg_execution_latency = exec_stats["execution_latency"]["avg_ms"]
            self.metrics.total_slippage_bps = exec_stats["slippage"]["avg_bps"]
            self.metrics.filled_orders = exec_stats["total_executed"]
            
            # 订单簿健康检查
            for symbol in self._symbols:
                health = self.orderbook_manager.get_orderbook_health(symbol)
                if health.get("status") == "stale":
                    self.log_warning(f"Stale orderbook data for {symbol}")
            
        except Exception as e:
            self.log_error(f"Error updating metrics: {e}")
    
    async def _check_emergency_stop(self):
        """检查紧急停止条件"""
        try:
            # 这里可以添加紧急停止逻辑
            # 比如：极高延迟、连接断开、异常损失等
            
            if self.metrics.avg_update_latency > self.config.latency_target_ms * 5:
                self.log_error("Critical latency detected - consider emergency stop")
            
        except Exception as e:
            self.log_error(f"Error in emergency stop check: {e}")
    
    async def _on_order_executed(self, order: ExecutionOrder):
        """订单执行回调"""
        try:
            # 通知订单回调
            for callback in self.order_callbacks:
                await callback(order)
                
            # 更新PnL（简化计算）
            if order.average_price and order.filled_quantity > 0:
                # 这里需要根据实际策略计算PnL
                pass
                
        except Exception as e:
            self.log_error(f"Error in order execution callback: {e}")
    
    async def _on_latency_alert(self, alert: AlertEvent):
        """延迟告警回调"""
        try:
            alert_msg = f"Latency Alert [{alert.level.value.upper()}]: {alert.message}"
            
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
                self.log_error(
                    alert_msg,
                    symbol=alert.symbol,
                    data_source=alert.data_source,
                    latency_ms=alert.latency_ms,
                    **alert.metadata
                )
            elif alert.level == AlertLevel.WARNING:
                self.log_warning(
                    alert_msg,
                    symbol=alert.symbol,
                    data_source=alert.data_source,
                    latency_ms=alert.latency_ms,
                    **alert.metadata
                )
            else:
                self.log_info(
                    alert_msg,
                    symbol=alert.symbol,
                    data_source=alert.data_source,
                    latency_ms=alert.latency_ms,
                    **alert.metadata
                )
                
            # 这里可以添加额外的告警处理逻辑
            # 比如发送邮件、短信、推送消息等
            
        except Exception as e:
            self.log_error(f"Error handling latency alert: {e}")
    
    def add_signal_callback(self, callback: Callable):
        """添加信号回调"""
        self.signal_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """添加订单回调"""
        self.order_callbacks.append(callback)
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """获取订单簿快照"""
        return self.orderbook_manager.get_orderbook(symbol)
    
    def get_current_signals(self, symbol: str) -> List[MicrostructureSignal]:
        """获取当前微观结构信号"""
        return self.microstructure_analyzer.get_current_signals(symbol)
    
    def get_active_orders(self, symbol: Optional[str] = None):
        """获取活跃订单"""
        return self.execution_engine.get_active_orders(symbol)
    
    def get_performance_metrics(self) -> HFTMetrics:
        """获取性能指标"""
        return self.metrics
    
    def get_orderbook_health(self, symbol: str) -> Dict[str, any]:
        """获取订单簿健康状态"""
        return self.orderbook_manager.get_orderbook_health(symbol)
    
    def get_latency_stats(self, symbol: str) -> Dict[str, float]:
        """获取延迟统计"""
        return self.orderbook_manager.get_latency_stats(symbol)
    
    def get_execution_stats(self, symbol: Optional[str] = None) -> Dict[str, any]:
        """获取执行统计"""
        return self.execution_engine.get_performance_stats(symbol)
    
    def get_microstructure_summary(self, symbol: str) -> Dict[str, any]:
        """获取微观结构分析摘要"""
        return self.microstructure_analyzer.get_analytics_summary(symbol)
    
    def get_system_status(self) -> Dict[str, any]:
        """获取系统状态"""
        status = {
            "running": self._running,
            "symbols": self._symbols,
            "metrics": {
                "total_updates": self.metrics.total_updates,
                "avg_latency_ms": self.metrics.avg_update_latency,
                "max_latency_ms": self.metrics.max_update_latency,
                "total_signals": self.metrics.total_signals,
                "active_signals": self.metrics.active_signals,
                "total_orders": self.metrics.total_orders,
                "filled_orders": self.metrics.filled_orders,
                "avg_execution_latency_ms": self.metrics.avg_execution_latency,
                "avg_slippage_bps": self.metrics.total_slippage_bps,
                # 延迟监控指标
                "data_freshness_checks": self.metrics.data_freshness_checks,
                "stale_data_detections": self.metrics.stale_data_detections,
                "data_source_switches": self.metrics.data_source_switches,
                "avg_data_latency_ms": self.metrics.avg_data_latency_ms,
                "p99_data_latency_ms": self.metrics.p99_data_latency_ms
            },
            "components": {
                "orderbook_manager": "active" if self._running else "stopped",
                "microstructure_analyzer": "active" if self._running else "stopped", 
                "execution_engine": "active" if self._running else "stopped",
                "latency_monitor": "active" if (self.latency_monitor and self._running) else "disabled"
            }
        }
        
        # 添加延迟监控健康状态
        if self.latency_monitor:
            status["latency_monitoring"] = self.latency_monitor.get_system_health()
        
        return status
    
    def get_latency_stats(self, symbol: Optional[str] = None) -> Dict[str, any]:
        """获取延迟统计信息"""
        if not self.latency_monitor:
            return {}
        
        if symbol:
            return self.latency_monitor.get_latency_summary(symbol)
        else:
            # 返回所有symbol的延迟统计
            result = {}
            for sym in self._symbols:
                stats = self.latency_monitor.get_latency_summary(sym)
                if stats:
                    result[sym] = stats
            return result
    
    def get_data_source_status(self) -> Dict[str, str]:
        """获取数据源状态"""
        if not self.latency_monitor:
            return {}
        
        return self.latency_monitor.get_system_health().get("data_sources", {})
    
    def get_active_data_sources(self) -> Dict[str, str]:
        """获取活跃数据源映射"""
        if not self.latency_monitor:
            return {}
        
        return self.latency_monitor.get_system_health().get("active_sources", {})
    
    async def update_data_source_status(self, data_source: str, status: str):
        """更新数据源状态"""
        if not self.latency_monitor:
            self.log_warning("Latency monitor not available")
            return
            
        from src.hft.latency_monitor import DataSourceStatus
        try:
            status_enum = DataSourceStatus(status)
            await self.latency_monitor.update_data_source_status(data_source, status_enum)
        except ValueError:
            self.log_error(f"Invalid data source status: {status}")
    
    async def switch_data_source(self, symbol: str, reason: str = "Manual switch") -> Optional[str]:
        """手动切换数据源"""
        if not self.latency_monitor:
            self.log_warning("Latency monitor not available")
            return None
        
        return await self.latency_monitor.switch_data_source(symbol, reason)