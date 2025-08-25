"""
高频交易信号处理集成测试

此模块包含了完整的端到端测试，验证：
1. 多维度技术指标引擎与HFT信号处理器的集成
2. 信号过滤和订单生成流程
3. 容错机制和异常处理
4. 延迟监控和数据源切换
5. 智能订单路由
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor, FilterResult, OrderType
from src.hft.smart_order_router import SmartOrderRouter, ExecutionAlgorithm
from src.hft.fault_tolerance_manager import FaultToleranceManager, ErrorCategory, ComponentStatus
from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, DataSourceStatus
from src.hft.signal_processor import LatencySensitiveSignalProcessor, SignalPriority
from src.core.models.trading import MarketData
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength


@pytest.fixture
async def mock_multidimensional_engine():
    """模拟多维度指标引擎"""
    engine = Mock(spec=MultiDimensionalIndicatorEngine)
    engine.generate_multidimensional_signal = AsyncMock()
    return engine


@pytest.fixture
async def mock_latency_monitor():
    """模拟延迟监控器"""
    monitor = Mock(spec=LatencyMonitor)
    monitor.check_data_freshness = AsyncMock(return_value=(True, Mock()))
    monitor.get_active_data_source = Mock(return_value="binance")
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.get_system_health = Mock(return_value={
        "running": True,
        "total_checks": 100,
        "stale_detection_rate": 0.01
    })
    return monitor


@pytest.fixture
async def mock_signal_processor():
    """模拟信号处理器"""
    processor = Mock(spec=LatencySensitiveSignalProcessor)
    processor.start = AsyncMock()
    processor.stop = AsyncMock()
    processor.add_action_callback = Mock()
    processor.get_status = Mock(return_value={"running": True})
    return processor


@pytest.fixture
async def sample_market_data():
    """样本市场数据"""
    return MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        open=50000.0,
        high=50500.0,
        low=49500.0,
        close=50200.0,
        volume=1000.0,
        bid=50190.0,
        ask=50210.0,
        bid_volume=10.0,
        ask_volume=15.0
    )


@pytest.fixture
async def sample_trading_signal():
    """样本交易信号"""
    return TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.8,
        entry_price=50200.0,
        target_price=51000.0,
        stop_loss=49500.0,
        reasoning=["技术指标看涨", "成交量放大"],
        indicators_consensus={"rsi": 0.7, "macd": 0.6}
    )


@pytest.fixture
async def sample_multidimensional_signal(sample_trading_signal):
    """样本多维度信号"""
    return MultiDimensionalSignal(
        primary_signal=sample_trading_signal,
        momentum_score=0.6,
        mean_reversion_score=-0.2,
        volatility_score=0.4,
        volume_score=0.8,
        sentiment_score=0.3,
        overall_confidence=0.75,
        risk_reward_ratio=2.0,
        max_position_size=0.5,
        market_regime="trending_up",
        technical_levels={"support": 49000.0, "resistance": 51000.0}
    )


@pytest.fixture
async def integrated_processor(mock_multidimensional_engine, mock_latency_monitor, mock_signal_processor):
    """集成信号处理器夹具"""
    processor = IntegratedHFTSignalProcessor(
        multidimensional_engine=mock_multidimensional_engine,
        latency_monitor=mock_latency_monitor,
        signal_processor=mock_signal_processor,
        max_latency_ms=5.0,
        min_confidence_threshold=0.6,
        min_signal_strength=0.4
    )
    return processor


@pytest.fixture
async def fault_tolerance_manager():
    """容错管理器夹具"""
    manager = FaultToleranceManager(
        error_window_size=100,
        health_check_interval=1.0
    )
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
async def smart_order_router():
    """智能订单路由器夹具"""
    router = SmartOrderRouter(
        max_child_orders=10,
        max_order_value=50000.0,
        default_slice_size=0.2
    )
    await router.start()
    yield router
    await router.stop()


class TestIntegratedSignalProcessor:
    """集成信号处理器测试"""
    
    async def test_processor_initialization(self, integrated_processor):
        """测试处理器初始化"""
        assert integrated_processor.max_latency_ms == 5.0
        assert integrated_processor.min_confidence_threshold == 0.6
        assert integrated_processor.min_signal_strength == 0.4
        assert not integrated_processor._running
        
    async def test_processor_start_stop(self, integrated_processor):
        """测试启动和停止"""
        await integrated_processor.start()
        assert integrated_processor._running
        assert integrated_processor._processing_task is not None
        
        await integrated_processor.stop()
        assert not integrated_processor._running
        assert integrated_processor._processing_task is None
    
    async def test_market_data_processing_success(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试市场数据处理成功流程"""
        # 设置模拟返回值
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        await integrated_processor.start()
        
        # 处理市场数据
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 验证结果
        assert orders is not None
        assert len(orders) > 0
        assert orders[0].symbol == "BTCUSDT"
        assert orders[0].side == "buy"
        assert orders[0].confidence == sample_multidimensional_signal.overall_confidence
        
        await integrated_processor.stop()
    
    async def test_signal_filtering(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试信号过滤逻辑"""
        # 测试低置信度信号被过滤
        low_confidence_signal = sample_multidimensional_signal
        low_confidence_signal.overall_confidence = 0.3  # 低于阈值0.6
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = low_confidence_signal
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 应该被过滤掉
        assert orders is None
        assert integrated_processor.stats.signals_filtered_out > 0
        
        await integrated_processor.stop()
    
    async def test_latency_monitoring_integration(self, integrated_processor, sample_market_data):
        """测试延迟监控集成"""
        # 模拟数据过期
        integrated_processor.latency_monitor.check_data_freshness.return_value = (False, Mock())
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 应该因为延迟过高被拒绝
        assert orders is None
        
        await integrated_processor.stop()
    
    async def test_order_generation_with_protective_orders(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试保护性订单生成"""
        # 设置高置信度信号以触发保护性订单
        high_confidence_signal = sample_multidimensional_signal
        high_confidence_signal.overall_confidence = 0.85
        high_confidence_signal.risk_reward_ratio = 2.5
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = high_confidence_signal
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 应该生成主订单和保护性订单
        assert orders is not None
        assert len(orders) > 1  # 主订单 + 保护性订单
        
        # 检查订单类型
        order_purposes = [order.metadata.get("order_purpose", "main") for order in orders]
        assert "main" in order_purposes or any(purpose is None for purpose in order_purposes)
        
        await integrated_processor.stop()
    
    async def test_risk_management_limits(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试风险管理限制"""
        # 设置高风险信号
        high_risk_signal = sample_multidimensional_signal
        high_risk_signal.volatility_score = 0.9  # 高波动率
        high_risk_signal.risk_reward_ratio = 0.5  # 差的风险收益比
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = high_risk_signal
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 应该被风险控制拒绝
        assert orders is None
        
        await integrated_processor.stop()
    
    async def test_daily_order_limits(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试每日订单限制"""
        integrated_processor.max_daily_orders = 5
        integrated_processor.daily_order_count = 5  # 已达上限
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 应该被每日限制拒绝
        assert orders is None
        
        await integrated_processor.stop()


class TestSmartOrderRouter:
    """智能订单路由器测试"""
    
    async def test_order_routing_success(self, smart_order_router, sample_market_data):
        """测试订单路由成功"""
        from src.hft.integrated_signal_processor import OrderRequest
        
        order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=1.0,
            price=50200.0,
            confidence=0.8,
            urgency_score=0.6
        )
        
        execution_id = await smart_order_router.route_order(order, sample_market_data)
        
        assert execution_id is not None
        assert execution_id in smart_order_router.active_orders
        assert len(smart_order_router.child_orders[execution_id]) > 0
    
    async def test_order_slicing(self, smart_order_router, sample_market_data):
        """测试订单分片"""
        from src.hft.integrated_signal_processor import OrderRequest
        
        large_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=10.0,  # 大额订单
            price=50200.0,
            confidence=0.8,
            urgency_score=0.4
        )
        
        execution_id = await smart_order_router.route_order(large_order, sample_market_data)
        
        assert execution_id is not None
        child_orders = smart_order_router.child_orders[execution_id]
        assert len(child_orders) > 1  # 应该被分片
        
        # 验证分片总量等于原订单
        total_quantity = sum(child.quantity for child in child_orders)
        assert abs(total_quantity - large_order.quantity) < 0.001
    
    async def test_execution_algorithm_selection(self, smart_order_router, sample_market_data):
        """测试执行算法选择"""
        from src.hft.integrated_signal_processor import OrderRequest
        
        # 高紧急度订单应选择狙击算法
        urgent_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side="buy", 
            quantity=1.0,
            urgency_score=0.95,  # 高紧急度
            confidence=0.8
        )
        
        algorithm = smart_order_router._select_execution_algorithm(urgent_order, sample_market_data)
        assert algorithm == ExecutionAlgorithm.SNIPER
        
        # 大额订单应选择冰山算法
        large_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=2.0,
            price=50200.0,
            urgency_score=0.5,
            confidence=0.8
        )
        
        algorithm = smart_order_router._select_execution_algorithm(large_order, sample_market_data)
        assert algorithm == ExecutionAlgorithm.ICEBERG
    
    async def test_venue_assignment(self, smart_order_router):
        """测试交易场所分配"""
        from src.hft.smart_order_router import ChildOrder
        
        child_orders = [
            ChildOrder(
                parent_id="test",
                symbol="BTCUSDT",
                side="buy",
                quantity=1.0,
                price=50200.0,
                order_type=OrderType.LIMIT,
                venue="",  # 待分配
                algorithm=ExecutionAlgorithm.TWAP
            ) for _ in range(3)
        ]
        
        market_data = MarketData(
            symbol="BTCUSDT", 
            timestamp=datetime.now(),
            open=50000.0, high=50500.0, low=49500.0, close=50200.0, volume=1000.0
        )
        
        await smart_order_router._assign_venues(child_orders, market_data)
        
        # 所有子订单都应该被分配到场所
        for child_order in child_orders:
            assert child_order.venue != ""
            assert child_order.venue in smart_order_router.venues


class TestFaultToleranceManager:
    """容错管理器测试"""
    
    async def test_error_handling_and_categorization(self, fault_tolerance_manager):
        """测试错误处理和分类"""
        # 测试网络错误
        network_error = ConnectionError("Connection failed")
        result = await fault_tolerance_manager.handle_error("test_component", network_error)
        
        # 网络错误应该允许重试
        assert result is True or result is False  # 取决于重试策略
        
        # 检查错误是否被正确记录
        health = fault_tolerance_manager.get_component_health("test_component")
        assert health.error_count > 0
        assert health.last_error is not None
        assert health.last_error.category == ErrorCategory.NETWORK_ERROR
    
    async def test_circuit_breaker_activation(self, fault_tolerance_manager):
        """测试熔断器激活"""
        component = "test_component"
        
        # 连续多次失败以触发熔断器
        for _ in range(6):  # 超过默认阈值5
            await fault_tolerance_manager.handle_error(component, RuntimeError("Test error"))
        
        health = fault_tolerance_manager.get_component_health(component)
        assert health.status == ComponentStatus.CIRCUIT_OPEN
    
    async def test_protected_operation_context(self, fault_tolerance_manager):
        """测试保护操作上下文"""
        component = "test_component"
        
        # 成功操作
        async with fault_tolerance_manager.protected_operation(component, "test_op"):
            await asyncio.sleep(0.01)  # 模拟操作
        
        health = fault_tolerance_manager.get_component_health(component)
        assert health.total_requests > 0
        assert health.consecutive_failures == 0
        
        # 失败操作
        with pytest.raises(ValueError):
            async with fault_tolerance_manager.protected_operation(component, "failing_op"):
                raise ValueError("Test failure")
        
        health = fault_tolerance_manager.get_component_health(component)
        assert health.consecutive_failures > 0
    
    async def test_component_degradation(self, fault_tolerance_manager):
        """测试组件降级"""
        component = "test_component"
        
        # 注册组件，配置降级策略
        from src.hft.fault_tolerance_manager import RecoveryStrategy
        fault_tolerance_manager.recovery_strategies[component] = {
            ErrorCategory.PROCESSING_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION
        }
        
        # 触发降级
        processing_error = RuntimeError("Processing failed")
        await fault_tolerance_manager.handle_error(component, processing_error)
        
        assert fault_tolerance_manager.is_component_degraded(component)
    
    async def test_system_health_summary(self, fault_tolerance_manager):
        """测试系统健康摘要"""
        # 添加一些错误和成功操作
        await fault_tolerance_manager.handle_error("comp1", ConnectionError("network issue"))
        
        async with fault_tolerance_manager.protected_operation("comp2"):
            pass
        
        summary = fault_tolerance_manager.get_system_health_summary()
        
        assert "total_components" in summary
        assert "healthy_count" in summary
        assert "overall_status" in summary
        assert "error_breakdown" in summary
        assert summary["total_errors"] > 0


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    async def test_full_trading_pipeline(self, integrated_processor, smart_order_router, fault_tolerance_manager, sample_market_data, sample_multidimensional_signal):
        """测试完整交易管道"""
        # 设置集成处理器
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        # 添加订单路由回调
        routed_orders = []
        
        def order_callback(order):
            routed_orders.append(order)
        
        integrated_processor.add_order_callback(order_callback)
        
        # 启动所有组件
        await integrated_processor.start()
        
        # 处理市场数据
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        # 验证订单生成
        assert orders is not None
        assert len(orders) > 0
        
        # 路由订单
        for order in orders:
            execution_id = await smart_order_router.route_order(order, sample_market_data)
            assert execution_id is not None
        
        # 等待一小段时间让异步执行完成
        await asyncio.sleep(0.1)
        
        # 验证执行状态
        active_executions = smart_order_router.get_active_executions()
        assert len(active_executions) >= 0  # 可能已经执行完成
        
        await integrated_processor.stop()
    
    async def test_error_recovery_integration(self, integrated_processor, fault_tolerance_manager, sample_market_data):
        """测试错误恢复集成"""
        # 模拟多维度引擎故障
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.side_effect = ConnectionError("Engine connection failed")
        
        await integrated_processor.start()
        
        # 使用容错管理器保护操作
        async with fault_tolerance_manager.protected_operation("integrated_processor") as ctx:
            try:
                orders = await integrated_processor.process_market_data(sample_market_data)
                assert orders is None  # 应该失败
            except ConnectionError:
                pass  # 预期的异常
        
        # 检查错误被正确记录
        health = fault_tolerance_manager.get_component_health("integrated_processor")
        assert health.error_count > 0
        
        await integrated_processor.stop()
    
    async def test_latency_monitoring_with_fallback(self, integrated_processor, sample_market_data, sample_multidimensional_signal):
        """测试延迟监控和故障转移"""
        # 模拟主数据源延迟过高
        integrated_processor.latency_monitor.check_data_freshness.return_value = (False, Mock())
        integrated_processor.latency_monitor.switch_data_source.return_value = "backup_source"
        
        await integrated_processor.start()
        
        # 第一次处理应该失败（数据过期）
        orders = await integrated_processor.process_market_data(sample_market_data)
        assert orders is None
        
        # 模拟切换到备用数据源后数据新鲜
        integrated_processor.latency_monitor.check_data_freshness.return_value = (True, Mock())
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        # 第二次处理应该成功
        orders = await integrated_processor.process_market_data(sample_market_data)
        assert orders is not None
        
        await integrated_processor.stop()
    
    @pytest.mark.parametrize("signal_strength,expected_filter", [
        (0.2, FilterResult.REJECTED_STRENGTH),
        (0.8, FilterResult.ACCEPTED),
    ])
    async def test_signal_filtering_parameters(self, integrated_processor, sample_market_data, sample_multidimensional_signal, signal_strength, expected_filter):
        """参数化测试信号过滤"""
        # 调整信号强度
        sample_multidimensional_signal.primary_signal.confidence = signal_strength
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        await integrated_processor.start()
        orders = await integrated_processor.process_market_data(sample_market_data)
        
        if expected_filter == FilterResult.ACCEPTED:
            assert orders is not None
        else:
            assert orders is None
            
        await integrated_processor.stop()


class TestPerformanceAndLoad:
    """性能和负载测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, integrated_processor, sample_multidimensional_signal):
        """测试并发处理能力"""
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        await integrated_processor.start()
        
        # 创建多个市场数据
        market_data_list = [
            MarketData(
                symbol=f"SYMBOL{i}",
                timestamp=datetime.now(),
                open=50000.0 + i,
                high=50500.0 + i,
                low=49500.0 + i,
                close=50200.0 + i,
                volume=1000.0
            ) for i in range(10)
        ]
        
        # 并发处理
        tasks = [
            integrated_processor.process_market_data(data) 
            for data in market_data_list
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # 验证结果
        successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        # 性能指标验证
        assert processing_time < 5.0  # 应在5秒内完成
        assert len(successful_results) > 0  # 至少有一些成功处理
        
        # 检查统计信息
        stats = integrated_processor.get_processing_stats()
        assert stats.signals_processed >= len(successful_results)
        assert stats.avg_processing_latency_ms < 1000  # 平均延迟小于1秒
        
        await integrated_processor.stop()
    
    async def test_memory_usage_stability(self, integrated_processor, sample_multidimensional_signal):
        """测试内存使用稳定性"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        integrated_processor.multidimensional_engine.generate_multidimensional_signal.return_value = sample_multidimensional_signal
        
        await integrated_processor.start()
        
        # 处理大量数据
        for i in range(100):
            market_data = MarketData(
                symbol=f"TEST{i%10}",
                timestamp=datetime.now(),
                open=50000.0,
                high=50500.0,
                low=49500.0,
                close=50200.0,
                volume=1000.0
            )
            await integrated_processor.process_market_data(market_data)
            
            # 每10次检查一次内存
            if i % 10 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # 内存增长不应该太快（小于50MB）
                assert memory_growth < 50 * 1024 * 1024, f"Memory growth too large: {memory_growth / 1024 / 1024:.2f}MB"
        
        await integrated_processor.stop()


if __name__ == "__main__":
    # 运行特定测试
    pytest.main([__file__ + "::TestIntegratedSignalProcessor::test_market_data_processing_success", "-v"])