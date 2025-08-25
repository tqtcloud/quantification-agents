"""
策略管理系统测试配置和Fixture
为策略测试提供专用的fixture和配置
"""

import asyncio
import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, List, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

from src.strategy.strategy_manager import (
    StrategyManager, 
    StrategyConfig, 
    StrategyType, 
    StrategyStatus
)
from src.strategy.resource_allocator import ResourceAllocator, ResourceLimit
from src.strategy.strategy_monitor import StrategyMonitor, MonitoringLevel
from src.strategy.signal_aggregator import SignalAggregator, AggregationStrategy, SignalInput
from src.strategy.conflict_resolver import ConflictResolver
from src.strategy.priority_manager import PriorityManager
from src.core.message_bus import MessageBus
from src.core.models.signals import TradingSignal, SignalStrength
from src.hft.hft_engine import HFTConfig
from src.agents.orchestrator import WorkflowConfig


# 测试配置常量
TEST_CONFIG = {
    'DEFAULT_TIMEOUT': 30,
    'STRATEGY_STARTUP_TIMEOUT': 5,
    'SIGNAL_PROCESSING_TIMEOUT': 1,
    'RESOURCE_CHECK_INTERVAL': 0.1,
    'MAX_TEST_STRATEGIES': 10,
    'TEST_SYMBOLS': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'],
    'PERFORMANCE_THRESHOLDS': {
        'latency_ms': 10.0,
        'throughput_tps': 1000.0,
        'memory_mb': 512.0,
        'cpu_percent': 80.0
    }
}


@pytest.fixture(scope="session")
def event_loop():
    """会话级事件循环"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
async def temp_config_dir() -> AsyncGenerator[str, None]:
    """临时配置目录"""
    temp_dir = tempfile.mkdtemp(prefix="strategy_test_")
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def message_bus() -> AsyncGenerator[MessageBus, None]:
    """消息总线fixture"""
    bus = MessageBus()
    await bus.initialize()
    try:
        yield bus
    finally:
        await bus.shutdown()


@pytest.fixture
async def resource_allocator() -> AsyncGenerator[ResourceAllocator, None]:
    """资源分配器fixture"""
    allocator = ResourceAllocator()
    await allocator.initialize()
    try:
        yield allocator
    finally:
        await allocator.cleanup()


@pytest.fixture
async def strategy_monitor(message_bus) -> AsyncGenerator[StrategyMonitor, None]:
    """策略监控器fixture"""
    monitor = StrategyMonitor(message_bus)
    await monitor.initialize()
    try:
        yield monitor
    finally:
        await monitor.shutdown()


@pytest.fixture
async def conflict_resolver() -> AsyncGenerator[ConflictResolver, None]:
    """冲突解决器fixture"""
    resolver = ConflictResolver()
    await resolver.initialize()
    try:
        yield resolver
    finally:
        await resolver.shutdown()


@pytest.fixture
async def priority_manager() -> AsyncGenerator[PriorityManager, None]:
    """优先级管理器fixture"""
    manager = PriorityManager()
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.shutdown()


@pytest.fixture
async def signal_aggregator(message_bus) -> AsyncGenerator[SignalAggregator, None]:
    """信号聚合器fixture"""
    aggregator = SignalAggregator(
        message_bus=message_bus,
        aggregation_strategy=AggregationStrategy.HYBRID_FUSION,
        enable_conflict_resolution=True
    )
    await aggregator.initialize()
    try:
        yield aggregator
    finally:
        await aggregator.shutdown()


@pytest.fixture
async def strategy_manager(message_bus, temp_config_dir) -> AsyncGenerator[StrategyManager, None]:
    """策略管理器fixture"""
    manager = StrategyManager(
        message_bus=message_bus,
        config_dir=temp_config_dir
    )
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.shutdown()


@pytest.fixture
def sample_hft_configs() -> List[HFTConfig]:
    """样本HFT配置"""
    configs = []
    
    for i, symbol in enumerate(TEST_CONFIG['TEST_SYMBOLS']):
        config = HFTConfig(
            symbol=symbol,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal(f"{1.0 + i * 0.5}"),
            latency_target_ms=5 + i,
            max_position=Decimal(f"{10.0 + i * 5}"),
            risk_limit=Decimal(f"{1000.0 + i * 500}")
        )
        configs.append(config)
    
    return configs


@pytest.fixture
def sample_ai_configs() -> List[WorkflowConfig]:
    """样本AI配置"""
    configs = []
    
    for i in range(len(TEST_CONFIG['TEST_SYMBOLS'])):
        config = WorkflowConfig(
            max_agents=3 + i,
            timeout_seconds=30 + i * 10,
            enable_parallel_execution=True,
            max_retries=2 + i
        )
        configs.append(config)
    
    return configs


@pytest.fixture
def sample_resource_limits() -> List[ResourceLimit]:
    """样本资源限制"""
    limits = []
    
    base_memory = 256
    base_cpu = 20.0
    
    for i in range(4):
        limit = ResourceLimit(
            memory_mb=base_memory * (i + 1),
            cpu_percent=base_cpu + i * 10,
            network_connections=50 + i * 25,
            storage_mb=512 + i * 256
        )
        limits.append(limit)
    
    return limits


@pytest.fixture
def sample_trading_signals() -> List[TradingSignal]:
    """样本交易信号"""
    signals = []
    
    for i, symbol in enumerate(TEST_CONFIG['TEST_SYMBOLS']):
        signal = TradingSignal(
            symbol=symbol,
            side="buy" if i % 2 == 0 else "sell",
            strength=SignalStrength.MEDIUM,
            price=Decimal(f"{50000 + i * 1000}"),
            quantity=Decimal(f"{0.1 + i * 0.05}"),
            timestamp=time.time() + i,
            source=f"test_source_{i}",
            confidence=0.8 + i * 0.05
        )
        signals.append(signal)
    
    return signals


@pytest.fixture
def sample_signal_inputs(sample_trading_signals) -> List[SignalInput]:
    """样本信号输入"""
    inputs = []
    
    for i, signal in enumerate(sample_trading_signals):
        signal_input = SignalInput(
            signal_id=f"test_signal_{i}_{int(time.time() * 1000)}",
            signal=signal,
            source_strategy_id=f"strategy_{i}",
            weight=0.5 + i * 0.1,
            priority=i + 1
        )
        inputs.append(signal_input)
    
    return inputs


@pytest.fixture
def mock_hft_engine():
    """模拟HFT引擎"""
    mock_engine = AsyncMock()
    mock_engine.start = AsyncMock(return_value=True)
    mock_engine.stop = AsyncMock(return_value=True)
    mock_engine.get_status = AsyncMock(return_value="running")
    mock_engine.get_performance_metrics = AsyncMock(return_value={
        'latency_ms': 3.5,
        'throughput_tps': 1500,
        'orders_processed': 10000,
        'success_rate': 0.98
    })
    return mock_engine


@pytest.fixture
def mock_ai_orchestrator():
    """模拟AI编排器"""
    mock_orchestrator = AsyncMock()
    mock_orchestrator.start_workflow = AsyncMock(return_value=True)
    mock_orchestrator.stop_workflow = AsyncMock(return_value=True)
    mock_orchestrator.get_workflow_status = AsyncMock(return_value="active")
    mock_orchestrator.get_agents_info = AsyncMock(return_value=[
        {'agent_id': 'agent_1', 'status': 'active', 'performance': 0.85},
        {'agent_id': 'agent_2', 'status': 'active', 'performance': 0.92}
    ])
    return mock_orchestrator


@pytest.fixture
def performance_monitor():
    """性能监控器"""
    class MockPerformanceMonitor:
        def __init__(self):
            self.metrics = []
            self.start_time = time.time()
        
        def record_latency(self, latency_ms: float):
            self.metrics.append({
                'type': 'latency',
                'value': latency_ms,
                'timestamp': time.time()
            })
        
        def record_throughput(self, operations: int, duration: float):
            tps = operations / duration if duration > 0 else 0
            self.metrics.append({
                'type': 'throughput',
                'value': tps,
                'timestamp': time.time()
            })
        
        def record_memory_usage(self, memory_mb: float):
            self.metrics.append({
                'type': 'memory',
                'value': memory_mb,
                'timestamp': time.time()
            })
        
        def get_summary(self) -> dict:
            import statistics
            
            latencies = [m['value'] for m in self.metrics if m['type'] == 'latency']
            throughputs = [m['value'] for m in self.metrics if m['type'] == 'throughput']
            memory_usages = [m['value'] for m in self.metrics if m['type'] == 'memory']
            
            return {
                'latency': {
                    'avg': statistics.mean(latencies) if latencies else 0,
                    'max': max(latencies) if latencies else 0,
                    'count': len(latencies)
                },
                'throughput': {
                    'avg': statistics.mean(throughputs) if throughputs else 0,
                    'max': max(throughputs) if throughputs else 0,
                    'count': len(throughputs)
                },
                'memory': {
                    'avg': statistics.mean(memory_usages) if memory_usages else 0,
                    'max': max(memory_usages) if memory_usages else 0,
                    'count': len(memory_usages)
                },
                'test_duration': time.time() - self.start_time
            }
    
    return MockPerformanceMonitor()


@pytest.fixture
def test_data_factory():
    """测试数据工厂"""
    class TestDataFactory:
        @staticmethod
        def create_strategy_config(
            strategy_type: StrategyType = StrategyType.HFT,
            symbol: str = "BTCUSDT",
            **kwargs
        ) -> StrategyConfig:
            """创建策略配置"""
            if strategy_type == StrategyType.HFT:
                engine_config = HFTConfig(
                    symbol=symbol,
                    min_order_size=Decimal("0.001"),
                    max_order_size=Decimal("1.0"),
                    latency_target_ms=5,
                    **kwargs
                )
            else:
                engine_config = WorkflowConfig(
                    max_agents=5,
                    timeout_seconds=30,
                    enable_parallel_execution=True,
                    **kwargs
                )
            
            return StrategyConfig(
                name=f"test_strategy_{int(time.time() * 1000)}",
                strategy_type=strategy_type,
                engine_config=engine_config,
                enabled=True,
                auto_start=False
            )
        
        @staticmethod
        def create_trading_signal(
            symbol: str = "BTCUSDT",
            side: str = "buy",
            **kwargs
        ) -> TradingSignal:
            """创建交易信号"""
            return TradingSignal(
                symbol=symbol,
                side=side,
                strength=SignalStrength.MEDIUM,
                price=Decimal("50000"),
                quantity=Decimal("0.1"),
                timestamp=time.time(),
                source="test_factory",
                confidence=0.8,
                **kwargs
            )
        
        @staticmethod
        def create_signal_input(
            signal: TradingSignal = None,
            strategy_id: str = "test_strategy",
            **kwargs
        ) -> SignalInput:
            """创建信号输入"""
            if signal is None:
                signal = TestDataFactory.create_trading_signal()
            
            return SignalInput(
                signal_id=f"signal_{int(time.time() * 1000000)}",
                signal=signal,
                source_strategy_id=strategy_id,
                weight=1.0,
                priority=1,
                **kwargs
            )
        
        @staticmethod
        def create_resource_limit(**kwargs) -> ResourceLimit:
            """创建资源限制"""
            return ResourceLimit(
                memory_mb=512,
                cpu_percent=25.0,
                network_connections=100,
                storage_mb=1024,
                **kwargs
            )
    
    return TestDataFactory


@pytest.fixture
def error_injection_helper():
    """错误注入辅助工具"""
    class ErrorInjectionHelper:
        def __init__(self):
            self.injected_errors = []
        
        def inject_async_error(self, target_func, error_type=Exception, error_message="Injected error"):
            """注入异步函数错误"""
            async def error_side_effect(*args, **kwargs):
                self.injected_errors.append({
                    'function': target_func.__name__,
                    'error_type': error_type.__name__,
                    'timestamp': time.time()
                })
                raise error_type(error_message)
            
            return error_side_effect
        
        def inject_sync_error(self, target_func, error_type=Exception, error_message="Injected error"):
            """注入同步函数错误"""
            def error_side_effect(*args, **kwargs):
                self.injected_errors.append({
                    'function': target_func.__name__,
                    'error_type': error_type.__name__,
                    'timestamp': time.time()
                })
                raise error_type(error_message)
            
            return error_side_effect
        
        def get_error_stats(self) -> dict:
            """获取错误统计"""
            return {
                'total_errors': len(self.injected_errors),
                'error_types': list(set(e['error_type'] for e in self.injected_errors)),
                'affected_functions': list(set(e['function'] for e in self.injected_errors))
            }
    
    return ErrorInjectionHelper()


@pytest.fixture
async def integration_test_env(
    strategy_manager,
    signal_aggregator,
    strategy_monitor,
    resource_allocator
):
    """集成测试环境"""
    class IntegrationTestEnvironment:
        def __init__(self):
            self.strategy_manager = strategy_manager
            self.signal_aggregator = signal_aggregator
            self.strategy_monitor = strategy_monitor
            self.resource_allocator = resource_allocator
            self.created_strategies = []
            self.test_state = {}
        
        async def create_test_strategy(
            self,
            strategy_type: StrategyType = StrategyType.HFT,
            **config_kwargs
        ) -> str:
            """创建测试策略"""
            if strategy_type == StrategyType.HFT:
                config = HFTConfig(
                    symbol="BTCUSDT",
                    min_order_size=Decimal("0.001"),
                    max_order_size=Decimal("1.0"),
                    latency_target_ms=5,
                    **config_kwargs
                )
            else:
                config = WorkflowConfig(
                    max_agents=5,
                    timeout_seconds=30,
                    enable_parallel_execution=True,
                    **config_kwargs
                )
            
            strategy_id = await self.strategy_manager.create_strategy(
                name=f"integration_test_{len(self.created_strategies)}",
                strategy_type=strategy_type,
                config=config
            )
            
            if strategy_id:
                self.created_strategies.append(strategy_id)
            
            return strategy_id
        
        async def start_all_strategies(self) -> List[bool]:
            """启动所有策略"""
            results = []
            for strategy_id in self.created_strategies:
                result = await self.strategy_manager.start_strategy(strategy_id)
                results.append(result)
            return results
        
        async def stop_all_strategies(self) -> List[bool]:
            """停止所有策略"""
            results = []
            for strategy_id in self.created_strategies:
                try:
                    result = await self.strategy_manager.stop_strategy(strategy_id)
                    results.append(result)
                except Exception:
                    results.append(False)
            return results
        
        async def cleanup(self):
            """清理测试环境"""
            await self.stop_all_strategies()
            
            # 移除所有创建的策略
            for strategy_id in self.created_strategies:
                try:
                    await self.strategy_manager.remove_strategy(strategy_id)
                except Exception:
                    pass
            
            self.created_strategies.clear()
            self.test_state.clear()
        
        def get_test_summary(self) -> dict:
            """获取测试摘要"""
            return {
                'strategies_created': len(self.created_strategies),
                'test_state': self.test_state,
                'timestamp': time.time()
            }
    
    env = IntegrationTestEnvironment()
    try:
        yield env
    finally:
        await env.cleanup()


# 性能测试辅助工具
@pytest.fixture
def benchmark_helper():
    """基准测试辅助工具"""
    class BenchmarkHelper:
        def __init__(self):
            self.benchmarks = {}
        
        async def benchmark_async_function(self, func, *args, iterations=100, **kwargs):
            """基准测试异步函数"""
            import statistics
            
            durations = []
            results = []
            
            # 预热
            for _ in range(min(10, iterations // 10)):
                try:
                    await func(*args, **kwargs)
                except Exception:
                    pass
            
            # 正式测试
            for _ in range(iterations):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    results.append(e)
                
                duration = time.time() - start_time
                durations.append(duration * 1000)  # 转换为毫秒
            
            benchmark_data = {
                'iterations': iterations,
                'avg_duration_ms': statistics.mean(durations),
                'median_duration_ms': statistics.median(durations),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'success_count': len([r for r in results if not isinstance(r, Exception)]),
                'error_count': len([r for r in results if isinstance(r, Exception)])
            }
            
            func_name = func.__name__ if hasattr(func, '__name__') else str(func)
            self.benchmarks[func_name] = benchmark_data
            
            return benchmark_data
        
        def get_benchmark_summary(self) -> dict:
            """获取基准测试摘要"""
            return {
                'total_benchmarks': len(self.benchmarks),
                'benchmarks': self.benchmarks
            }
    
    return BenchmarkHelper()


# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "integration: 集成测试"
    )
    config.addinivalue_line(
        "markers", "performance: 性能测试"
    )
    config.addinivalue_line(
        "markers", "reliability: 可靠性测试"
    )
    config.addinivalue_line(
        "markers", "slow: 慢速测试"
    )
    config.addinivalue_line(
        "markers", "unit: 单元测试"
    )


# 测试跳过条件
def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    import sys
    
    # 跳过条件
    skip_slow = pytest.mark.skip(reason="跳过慢速测试")
    skip_integration = pytest.mark.skip(reason="跳过集成测试")
    
    for item in items:
        # 如果运行在CI环境中，跳过某些测试
        if "CI" in os.environ or "--fast" in sys.argv:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
        
        # 如果没有指定集成测试，跳过集成测试
        if "--integration" not in sys.argv:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)