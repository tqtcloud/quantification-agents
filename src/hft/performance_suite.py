import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.hft.hft_engine import HFTEngine, HFTConfig
from src.hft.performance_optimizer import HFTPerformanceOptimizer, PerformanceConfig
from src.hft.network_optimizer import NetworkLatencyOptimizer, NetworkConfig
from src.hft.signal_processor import LatencySensitiveSignalProcessor
from src.hft.agents.arbitrage_agent import ArbitrageAgent, ArbitrageConfig
from src.hft.agents.market_making_agent import MarketMakingAgent, MarketMakingConfig
from src.utils.logger import LoggerMixin


@dataclass
class HFTSuiteConfig:
    """HFT性能套件配置"""
    # 核心引擎配置
    hft_config: HFTConfig = field(default_factory=HFTConfig)
    
    # 性能优化配置
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # 网络优化配置
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    
    # 策略配置
    arbitrage_config: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    market_making_config: MarketMakingConfig = field(default_factory=MarketMakingConfig)
    
    # 套件设置
    enable_arbitrage: bool = True
    enable_market_making: bool = True
    enable_signal_processor: bool = True
    
    # 监控设置
    monitoring_interval: float = 1.0
    performance_reporting_interval: float = 60.0


@dataclass
class SuiteMetrics:
    """套件指标"""
    uptime_seconds: float = 0.0
    total_latency_violations: int = 0
    avg_end_to_end_latency_ms: float = 0.0
    total_trades_executed: int = 0
    total_pnl: float = 0.0
    system_health_score: float = 100.0


class HFTPerformanceSuite(LoggerMixin):
    """高频交易性能套件
    
    集成所有HFT组件，提供一站式的高性能交易解决方案：
    1. 性能优化（uvloop, CPU亲和性, 内存池）
    2. 网络优化（零拷贝, 连接池）
    3. 信号处理（延迟敏感）
    4. 策略执行（套利, 做市）
    """
    
    def __init__(self, config: Optional[HFTSuiteConfig] = None):
        self.config = config or HFTSuiteConfig()
        
        # 核心组件
        self.performance_optimizer: Optional[HFTPerformanceOptimizer] = None
        self.network_optimizer: Optional[NetworkLatencyOptimizer] = None
        self.signal_processor: Optional[LatencySensitiveSignalProcessor] = None
        self.hft_engine: Optional[HFTEngine] = None
        
        # 策略Agent
        self.arbitrage_agent: Optional[ArbitrageAgent] = None
        self.market_making_agent: Optional[MarketMakingAgent] = None
        
        # 运行状态
        self._running = False
        self._symbols: List[str] = []
        self._start_time: float = 0.0
        
        # 监控
        self.metrics = SuiteMetrics()
        self._monitor_task: Optional[asyncio.Task] = None
        self._report_task: Optional[asyncio.Task] = None
        
        # 回调
        self._latency_callbacks: List = []
        
    async def initialize(self, symbols: List[str]):
        """初始化HFT性能套件"""
        self.log_info("Initializing HFT Performance Suite...")
        
        self._symbols = symbols.copy()
        
        try:
            # 1. 初始化性能优化器
            self.performance_optimizer = HFTPerformanceOptimizer(self.config.performance_config)
            await self.performance_optimizer.initialize()
            
            # 2. 初始化网络优化器
            self.network_optimizer = NetworkLatencyOptimizer(self.config.network_config)
            self.network_optimizer.set_performance_optimizer(self.performance_optimizer)
            
            # 3. 初始化信号处理器
            if self.config.enable_signal_processor:
                self.signal_processor = LatencySensitiveSignalProcessor()
                await self.signal_processor.start()
            
            # 4. 初始化HFT引擎
            self.hft_engine = HFTEngine(self.config.hft_config)
            await self.hft_engine.initialize(symbols)
            
            # 5. 初始化策略Agent
            if self.config.enable_arbitrage:
                self.arbitrage_agent = ArbitrageAgent(self.hft_engine, self.config.arbitrage_config)
                
            if self.config.enable_market_making:
                self.market_making_agent = MarketMakingAgent(self.hft_engine, self.config.market_making_config)
            
            # 6. 设置回调和集成
            await self._setup_integrations()
            
            self.log_info(f"HFT Performance Suite initialized for {len(symbols)} symbols")
            
        except Exception as e:
            self.log_error(f"Failed to initialize HFT suite: {e}")
            raise
    
    async def _setup_integrations(self):
        """设置组件集成"""
        if not self.hft_engine:
            return
            
        # 集成信号处理器
        if self.signal_processor:
            # 注册信号回调
            self.hft_engine.add_signal_callback(self._on_microstructure_signal)
            
            # 注册动作回调
            self.signal_processor.add_action_callback(self._on_signal_action)
        
        # 集成性能监控
        if self.performance_optimizer:
            # 注册延迟监控回调
            self._latency_callbacks.append(self._on_latency_measurement)
    
    async def start(self):
        """启动HFT性能套件"""
        if self._running:
            return
            
        self.log_info("Starting HFT Performance Suite...")
        self._running = True
        self._start_time = time.time()
        
        try:
            # 启动HFT引擎
            if self.hft_engine:
                await self.hft_engine.start()
            
            # 启动策略Agent
            if self.arbitrage_agent:
                await self.arbitrage_agent.start()
                
            if self.market_making_agent:
                await self.market_making_agent.start(self._symbols)
            
            # 启动监控任务
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            self._report_task = asyncio.create_task(self._reporting_loop())
            
            self.log_info("HFT Performance Suite started successfully")
            
        except Exception as e:
            self.log_error(f"Failed to start HFT suite: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止HFT性能套件"""
        if not self._running:
            return
            
        self.log_info("Stopping HFT Performance Suite...")
        self._running = False
        
        # 停止监控任务
        for task in [self._monitor_task, self._report_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 停止策略Agent
        if self.arbitrage_agent:
            await self.arbitrage_agent.stop()
            
        if self.market_making_agent:
            await self.market_making_agent.stop()
        
        # 停止核心组件
        if self.hft_engine:
            await self.hft_engine.stop()
            
        if self.signal_processor:
            await self.signal_processor.stop()
            
        if self.network_optimizer:
            await self.network_optimizer.shutdown()
            
        if self.performance_optimizer:
            await self.performance_optimizer.shutdown()
        
        self.log_info("HFT Performance Suite stopped")
    
    async def update_market_data(self, symbol: str, market_data):
        """更新市场数据"""
        if not self._running or not self.hft_engine:
            return False
            
        # 更新HFT引擎
        success = await self.hft_engine.update_market_data(symbol, market_data)
        
        if success:
            # 更新策略Agent
            if self.arbitrage_agent:
                await self.arbitrage_agent.update_market_data(symbol, market_data)
                
            if self.market_making_agent:
                await self.market_making_agent.update_market_data(symbol, market_data)
        
        return success
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._update_metrics()
                await self._check_health()
                await asyncio.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _reporting_loop(self):
        """报告循环"""
        while self._running:
            try:
                await self._generate_performance_report()
                await asyncio.sleep(self.config.performance_reporting_interval)
                
            except Exception as e:
                self.log_error(f"Error in reporting loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_metrics(self):
        """更新指标"""
        # 运行时间
        self.metrics.uptime_seconds = time.time() - self._start_time
        
        # HFT引擎指标
        if self.hft_engine:
            hft_metrics = self.hft_engine.get_performance_metrics()
            self.metrics.avg_end_to_end_latency_ms = hft_metrics.avg_update_latency
            
            # 检查延迟违规
            if hft_metrics.avg_update_latency > self.config.hft_config.latency_target_ms:
                self.metrics.total_latency_violations += 1
        
        # 策略指标
        total_pnl = 0.0
        
        if self.market_making_agent:
            mm_pnl = self.market_making_agent.get_pnl()
            total_pnl += float(mm_pnl.get("total", 0))
        
        self.metrics.total_pnl = total_pnl
    
    async def _check_health(self):
        """检查系统健康状态"""
        health_score = 100.0
        
        # 性能优化器健康检查
        if self.performance_optimizer:
            perf_report = self.performance_optimizer.get_performance_report()
            
            # CPU使用率检查
            cpu_usage = perf_report.get("current_metrics", {}).get("cpu_percent", 0)
            if cpu_usage > 80:
                health_score -= 10
            
            # 内存使用检查
            memory_mb = perf_report.get("current_metrics", {}).get("memory_mb", 0)
            if memory_mb > 1000:  # 1GB
                health_score -= 10
            
            # 事件循环延迟检查
            loop_latency = perf_report.get("current_metrics", {}).get("loop_latency_us", 0)
            if loop_latency > 1000:  # 1ms
                health_score -= 15
        
        # HFT引擎健康检查
        if self.hft_engine:
            for symbol in self._symbols:
                orderbook_health = self.hft_engine.get_orderbook_health(symbol)
                if orderbook_health.get("status") == "stale":
                    health_score -= 5
        
        # 信号处理器健康检查
        if self.signal_processor:
            signal_stats = self.signal_processor.get_signal_stats()
            processing_rate = signal_stats.get("processing_rate", 1.0)
            if processing_rate < 0.95:  # 95%处理率
                health_score -= 10
        
        self.metrics.system_health_score = max(0, health_score)
        
        # 健康警告
        if health_score < 80:
            self.log_warning(f"System health degraded: {health_score:.1f}%")
    
    async def _generate_performance_report(self):
        """生成性能报告"""
        report = {
            "timestamp": time.time(),
            "uptime_hours": self.metrics.uptime_seconds / 3600,
            "suite_metrics": {
                "uptime_seconds": self.metrics.uptime_seconds,
                "latency_violations": self.metrics.total_latency_violations,
                "avg_latency_ms": self.metrics.avg_end_to_end_latency_ms,
                "total_pnl": self.metrics.total_pnl,
                "health_score": self.metrics.system_health_score
            }
        }
        
        # 性能优化器报告
        if self.performance_optimizer:
            report["performance_optimizer"] = self.performance_optimizer.get_performance_report()
        
        # 网络优化器报告
        if self.network_optimizer:
            report["network_optimizer"] = self.network_optimizer.get_optimization_report()
        
        # HFT引擎报告
        if self.hft_engine:
            report["hft_engine"] = self.hft_engine.get_system_status()
        
        # 信号处理器报告
        if self.signal_processor:
            report["signal_processor"] = self.signal_processor.get_status()
        
        # 策略报告
        if self.arbitrage_agent:
            report["arbitrage_agent"] = self.arbitrage_agent.get_status()
            
        if self.market_making_agent:
            report["market_making_agent"] = self.market_making_agent.get_status()
        
        self.log_info(f"Performance Report - Health: {self.metrics.system_health_score:.1f}%, "
                     f"Latency: {self.metrics.avg_end_to_end_latency_ms:.2f}ms, "
                     f"PnL: {self.metrics.total_pnl:.2f}")
    
    async def _on_microstructure_signal(self, signal):
        """处理微观结构信号"""
        if self.signal_processor:
            await self.signal_processor.process_signal(signal)
    
    async def _on_signal_action(self, action):
        """处理信号动作"""
        # 这里可以实现基于信号的自动交易逻辑
        self.log_debug(f"Signal action: {action.action_type.value} for {action.symbol}")
    
    async def _on_latency_measurement(self, latency_ms: float, component: str):
        """处理延迟测量"""
        if latency_ms > self.config.hft_config.latency_target_ms:
            self.log_warning(f"High latency in {component}: {latency_ms:.2f}ms")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取综合状态"""
        status = {
            "running": self._running,
            "symbols": self._symbols,
            "uptime_seconds": self.metrics.uptime_seconds,
            "metrics": {
                "health_score": self.metrics.system_health_score,
                "avg_latency_ms": self.metrics.avg_end_to_end_latency_ms,
                "latency_violations": self.metrics.total_latency_violations,
                "total_pnl": self.metrics.total_pnl
            },
            "components": {
                "performance_optimizer": self.performance_optimizer is not None,
                "network_optimizer": self.network_optimizer is not None,
                "signal_processor": self.signal_processor is not None,
                "hft_engine": self.hft_engine is not None,
                "arbitrage_agent": self.arbitrage_agent is not None,
                "market_making_agent": self.market_making_agent is not None
            }
        }
        
        return status
    
    async def emergency_shutdown(self):
        """紧急关闭"""
        self.log_warning("Emergency shutdown initiated")
        
        # 快速停止所有交易活动
        if self.arbitrage_agent:
            self.arbitrage_agent._running = False
            
        if self.market_making_agent:
            self.market_making_agent._running = False
        
        # 取消所有挂单
        if self.hft_engine:
            for symbol in self._symbols:
                active_orders = self.hft_engine.get_active_orders(symbol)
                for order in active_orders:
                    await self.hft_engine.cancel_order(order.order_id)
        
        # 常规关闭
        await self.stop()
        
        self.log_warning("Emergency shutdown completed")