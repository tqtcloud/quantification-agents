"""
测试配置和通用fixture
"""
import asyncio
import pytest
import time
from typing import Generator, AsyncGenerator, List
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

from src.core.models import MarketData
from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, DataSourceStatus
from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor, OrderRequest, OrderType
from src.hft.smart_order_router import SmartOrderRouter, VenueInfo
from src.hft.fault_tolerance_manager import FaultToleranceManager


@pytest.fixture
def sample_market_data() -> List[MarketData]:
    """生成样本市场数据"""
    base_time = time.time() * 1000
    data = []
    for i in range(100):
        market_data = MarketData(
            symbol="BTCUSDT",
            timestamp=base_time + i * 100,  # 每100ms一个数据点
            open=50000.0 + np.random.normal(0, 100),
            high=50100.0 + np.random.normal(0, 100),
            low=49900.0 + np.random.normal(0, 100),
            close=50000.0 + np.random.normal(0, 100),
            volume=np.random.uniform(10, 1000),
            turnover=50000.0 * np.random.uniform(10, 1000),
        )
        data.append(market_data)
    return data


@pytest.fixture
def sample_data_sources() -> List[DataSourceConfig]:
    """生成样本数据源配置"""
    return [
        DataSourceConfig(
            name="binance",
            priority=1,
            max_latency_ms=50.0,
            timeout_ms=2000.0,
            retry_count=3,
            health_check_interval=30.0
        ),
        DataSourceConfig(
            name="okx",
            priority=2,
            max_latency_ms=80.0,
            timeout_ms=3000.0,
            retry_count=2,
            health_check_interval=30.0
        ),
        DataSourceConfig(
            name="bybit", 
            priority=3,
            max_latency_ms=100.0,
            timeout_ms=5000.0,
            retry_count=2,
            health_check_interval=30.0
        )
    ]


@pytest.fixture
async def latency_monitor(sample_data_sources) -> AsyncGenerator[LatencyMonitor, None]:
    """创建延迟监控器实例"""
    monitor = LatencyMonitor(
        staleness_threshold_ms=100.0,
        stats_window_size=1000,
        alert_cooldown_seconds=1.0  # 测试用较短冷却时间
    )
    await monitor.initialize(sample_data_sources)
    yield monitor
    await monitor.stop()


@pytest.fixture
def sample_venues() -> List[VenueInfo]:
    """生成样本交易场所信息"""
    return [
        VenueInfo(
            name="binance",
            priority=1,
            latency_ms=15.0,
            liquidity_score=0.95,
            fee_rate=0.001,
            min_order_size=0.001,
            max_order_size=1000.0,
            is_active=True,
            recent_fill_rate=0.98,
            average_slippage=0.0002
        ),
        VenueInfo(
            name="okx",
            priority=2,
            latency_ms=25.0,
            liquidity_score=0.85,
            fee_rate=0.0015,
            min_order_size=0.001,
            max_order_size=500.0,
            is_active=True,
            recent_fill_rate=0.95,
            average_slippage=0.0005
        ),
        VenueInfo(
            name="bybit",
            priority=3,
            latency_ms=35.0,
            liquidity_score=0.75,
            fee_rate=0.002,
            min_order_size=0.01,
            max_order_size=100.0,
            is_active=True,
            recent_fill_rate=0.92,
            average_slippage=0.0008
        )
    ]


@pytest.fixture
async def smart_order_router(sample_venues) -> AsyncGenerator[SmartOrderRouter, None]:
    """创建智能订单路由器实例"""
    router = SmartOrderRouter()
    await router.initialize(sample_venues)
    yield router
    await router.shutdown()


@pytest.fixture
async def integrated_signal_processor() -> AsyncGenerator[IntegratedHFTSignalProcessor, None]:
    """创建集成信号处理器实例"""
    processor = IntegratedHFTSignalProcessor()
    yield processor
    await processor.shutdown()


@pytest.fixture
async def fault_tolerance_manager() -> AsyncGenerator[FaultToleranceManager, None]:
    """创建容错管理器实例"""
    manager = FaultToleranceManager()
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def sample_order_requests() -> List[OrderRequest]:
    """生成样本订单请求"""
    return [
        OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=0.1,
            price=49950.0,
            urgency_score=0.8,
            confidence=0.85,
            risk_level="medium",
            signal_id="test_signal_1"
        ),
        OrderRequest(
            symbol="ETHUSDT",
            order_type=OrderType.MARKET,
            side="sell",
            quantity=1.0,
            urgency_score=0.9,
            confidence=0.9,
            risk_level="high",
            signal_id="test_signal_2"
        ),
        OrderRequest(
            symbol="ADAUSDT",
            order_type=OrderType.IOC,
            side="buy",
            quantity=1000.0,
            price=0.45,
            urgency_score=0.6,
            confidence=0.7,
            risk_level="low",
            signal_id="test_signal_3"
        )
    ]


@pytest.fixture
def mock_alert_callback() -> MagicMock:
    """创建模拟告警回调"""
    return MagicMock()


@pytest.fixture
async def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def performance_benchmark_config():
    """性能基准配置"""
    return {
        "target_latency_ms": 10.0,  # 目标延迟 < 10ms
        "target_throughput_tps": 10000,  # 目标吞吐量 > 10,000 TPS
        "max_memory_mb": 512,  # 最大内存使用 512MB
        "max_cpu_percent": 80,  # 最大CPU使用率 80%
        "test_duration_seconds": 60,  # 测试持续时间 60秒
        "warmup_seconds": 10,  # 预热时间 10秒
    }


class MockPerformanceCollector:
    """模拟性能数据收集器"""
    
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.metrics.append({
            'type': 'latency',
            'value': latency_ms,
            'timestamp': time.time()
        })
    
    def record_throughput(self, operations_count: int, time_window: float):
        """记录吞吐量"""
        tps = operations_count / time_window
        self.metrics.append({
            'type': 'throughput',
            'value': tps,
            'timestamp': time.time()
        })
    
    def record_memory_usage(self, memory_mb: float):
        """记录内存使用"""
        self.metrics.append({
            'type': 'memory',
            'value': memory_mb,
            'timestamp': time.time()
        })
    
    def record_cpu_usage(self, cpu_percent: float):
        """记录CPU使用率"""
        self.metrics.append({
            'type': 'cpu',
            'value': cpu_percent,
            'timestamp': time.time()
        })
    
    def get_summary(self) -> dict:
        """获取性能摘要"""
        latencies = [m['value'] for m in self.metrics if m['type'] == 'latency']
        throughputs = [m['value'] for m in self.metrics if m['type'] == 'throughput']
        memory_usages = [m['value'] for m in self.metrics if m['type'] == 'memory']
        cpu_usages = [m['value'] for m in self.metrics if m['type'] == 'cpu']
        
        return {
            'latency': {
                'avg': np.mean(latencies) if latencies else 0,
                'p95': np.percentile(latencies, 95) if len(latencies) >= 20 else 0,
                'p99': np.percentile(latencies, 99) if len(latencies) >= 100 else 0,
                'max': max(latencies) if latencies else 0,
                'count': len(latencies)
            },
            'throughput': {
                'avg': np.mean(throughputs) if throughputs else 0,
                'max': max(throughputs) if throughputs else 0,
                'count': len(throughputs)
            },
            'memory': {
                'avg': np.mean(memory_usages) if memory_usages else 0,
                'max': max(memory_usages) if memory_usages else 0,
                'count': len(memory_usages)
            },
            'cpu': {
                'avg': np.mean(cpu_usages) if cpu_usages else 0,
                'max': max(cpu_usages) if cpu_usages else 0,
                'count': len(cpu_usages)
            },
            'test_duration': time.time() - self.start_time
        }


@pytest.fixture
def performance_collector() -> MockPerformanceCollector:
    """创建性能收集器"""
    return MockPerformanceCollector()


# 用于integration tests的全局状态管理
class TestStateManager:
    """测试状态管理器"""
    
    def __init__(self):
        self.components = {}
        self.execution_history = []
        self.error_log = []
    
    def register_component(self, name: str, component):
        """注册组件"""
        self.components[name] = component
    
    def log_execution(self, component: str, action: str, result: any):
        """记录执行"""
        self.execution_history.append({
            'timestamp': time.time(),
            'component': component,
            'action': action,
            'result': result
        })
    
    def log_error(self, component: str, error: str, context: dict = None):
        """记录错误"""
        self.error_log.append({
            'timestamp': time.time(),
            'component': component,
            'error': error,
            'context': context or {}
        })
    
    def get_execution_summary(self) -> dict:
        """获取执行摘要"""
        return {
            'total_executions': len(self.execution_history),
            'total_errors': len(self.error_log),
            'error_rate': len(self.error_log) / max(len(self.execution_history), 1),
            'components_count': len(self.components)
        }


@pytest.fixture
def test_state_manager() -> TestStateManager:
    """创建测试状态管理器"""
    return TestStateManager()