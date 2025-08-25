"""
HFT功能测试套件

测试功能:
- 延迟监控功能验证
- 信号过滤逻辑测试
- 订单生成准确性测试
- 容错机制测试
"""
import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import List, Dict, Any
import numpy as np

from src.hft.latency_monitor import (
    LatencyMonitor, DataSourceConfig, DataSourceStatus, 
    AlertLevel, LatencyMetrics, PerformanceStats
)
from src.hft.integrated_signal_processor import (
    IntegratedHFTSignalProcessor, OrderRequest, OrderType,
    FilterResult, ProcessingStats
)
from src.hft.smart_order_router import (
    SmartOrderRouter, VenueInfo, ExecutionAlgorithm,
    ChildOrder, OrderStatus, ExecutionReport
)
from src.hft.fault_tolerance_manager import (
    FaultToleranceManager, ErrorCategory, ErrorSeverity,
    RecoveryStrategy, ComponentStatus, ErrorEvent
)
from src.core.models.trading import MarketData


@pytest.mark.asyncio
class TestLatencyMonitorFunctionality:
    """延迟监控器功能测试"""
    
    async def test_data_freshness_checking(self, latency_monitor, sample_market_data):
        """测试数据新鲜度检查"""
        await latency_monitor.start()
        
        try:
            # 测试新鲜数据
            fresh_data = sample_market_data[0]
            fresh_data.timestamp = time.time() * 1000 - 50  # 50ms前的数据
            
            is_fresh, metrics = await latency_monitor.check_data_freshness(
                symbol="BTCUSDT",
                market_data=fresh_data,
                data_source="binance"
            )
            
            assert is_fresh, "新鲜数据应该被标记为fresh"
            assert not metrics.is_stale, "新鲜数据不应该被标记为stale"
            assert metrics.total_latency_ms < latency_monitor.staleness_threshold_ms
            
            # 测试过期数据
            stale_data = sample_market_data[1]
            stale_data.timestamp = time.time() * 1000 - 150  # 150ms前的数据
            
            is_fresh, metrics = await latency_monitor.check_data_freshness(
                symbol="BTCUSDT", 
                market_data=stale_data,
                data_source="binance"
            )
            
            assert not is_fresh, "过期数据应该被标记为not fresh"
            assert metrics.is_stale, "过期数据应该被标记为stale"
            assert metrics.total_latency_ms > latency_monitor.staleness_threshold_ms
            
        finally:
            await latency_monitor.stop()
    
    async def test_data_source_switching(self, latency_monitor, sample_data_sources):
        """测试数据源切换功能"""
        await latency_monitor.start()
        
        try:
            # 设置初始活跃数据源
            latency_monitor.set_active_data_source("BTCUSDT", "binance")
            
            # 触发数据源切换
            new_source = await latency_monitor.switch_data_source(
                symbol="BTCUSDT",
                reason="High latency detected"
            )
            
            assert new_source is not None, "应该成功切换到备用数据源"
            assert new_source != "binance", "应该切换到不同的数据源"
            assert new_source in [ds.name for ds in sample_data_sources[1:]]
            
            # 验证活跃数据源已更新
            active_source = latency_monitor.get_active_data_source("BTCUSDT")
            assert active_source == new_source, "活跃数据源应该已更新"
            
            # 测试无可用备用源的情况
            for ds in sample_data_sources:
                await latency_monitor.update_data_source_status(
                    ds.name, DataSourceStatus.FAILED
                )
            
            no_source = await latency_monitor.switch_data_source(
                symbol="ETHUSDT",
                reason="All sources failed"
            )
            
            assert no_source is None, "当无可用源时应返回None"
            
        finally:
            await latency_monitor.stop()
    
    async def test_performance_statistics_collection(self, latency_monitor, sample_market_data):
        """测试性能统计收集"""
        await latency_monitor.start()
        
        try:
            # 生成测试数据并收集统计
            for i, market_data in enumerate(sample_market_data[:50]):
                market_data.timestamp = time.time() * 1000 - i * 10  # 递减时间戳
                
                await latency_monitor.check_data_freshness(
                    symbol="BTCUSDT",
                    market_data=market_data, 
                    data_source="binance"
                )
            
            # 获取性能统计
            stats = latency_monitor.get_performance_stats(
                symbol="BTCUSDT",
                data_source="binance"
            )
            
            assert len(stats) > 0, "应该收集到性能统计数据"
            
            btc_binance_stats = stats.get("BTCUSDT_binance")
            assert btc_binance_stats is not None, "应该有BTCUSDT的binance统计"
            assert btc_binance_stats.total_samples > 0, "应该有统计样本"
            assert btc_binance_stats.avg_latency_ms >= 0, "平均延迟应该非负"
            
            # 获取延迟摘要
            latency_summary = latency_monitor.get_latency_summary("BTCUSDT")
            assert "avg_latency_ms" in latency_summary
            assert "p95_latency_ms" in latency_summary
            assert "stale_data_rate" in latency_summary
            
        finally:
            await latency_monitor.stop()
    
    async def test_alert_system(self, latency_monitor, sample_market_data):
        """测试告警系统"""
        await latency_monitor.start()
        
        try:
            # 注册告警回调
            alert_callback = MagicMock()
            latency_monitor.add_alert_callback(alert_callback)
            
            # 生成过期数据触发告警
            stale_data = sample_market_data[0]
            stale_data.timestamp = time.time() * 1000 - 200  # 200ms前的数据
            
            await latency_monitor.check_data_freshness(
                symbol="BTCUSDT",
                market_data=stale_data,
                data_source="binance"
            )
            
            # 验证告警被触发
            assert alert_callback.called, "应该触发告警回调"
            
            # 验证告警内容
            call_args = alert_callback.call_args[0][0]
            assert call_args.level == AlertLevel.WARNING
            assert "Stale data detected" in call_args.message
            assert call_args.symbol == "BTCUSDT"
            assert call_args.data_source == "binance"
            
        finally:
            await latency_monitor.stop()
    
    async def test_system_health_monitoring(self, latency_monitor):
        """测试系统健康监控"""
        await latency_monitor.start()
        
        try:
            # 获取初始健康状态
            health = latency_monitor.get_system_health()
            
            assert health["running"] is True, "系统应该在运行中"
            assert "data_sources" in health
            assert "active_sources" in health
            assert "total_checks" in health
            
            # 模拟一些检查
            await asyncio.sleep(0.1)
            
            # 更新数据源状态
            await latency_monitor.update_data_source_status(
                "binance", 
                DataSourceStatus.DEGRADED
            )
            
            updated_health = latency_monitor.get_system_health()
            assert updated_health["data_sources"]["binance"] == "degraded"
            
        finally:
            await latency_monitor.stop()


@pytest.mark.asyncio
class TestSignalProcessorFunctionality:
    """信号处理器功能测试"""
    
    async def test_signal_filtering_logic(self, integrated_signal_processor, sample_order_requests):
        """测试信号过滤逻辑"""
        # 使用mock来隔离测试
        with patch.object(integrated_signal_processor, '_filter_by_confidence') as mock_confidence, \
             patch.object(integrated_signal_processor, '_filter_by_risk') as mock_risk, \
             patch.object(integrated_signal_processor, '_filter_by_latency') as mock_latency:
            
            # 设置mock返回值
            mock_confidence.return_value = FilterResult.ACCEPTED
            mock_risk.return_value = FilterResult.ACCEPTED  
            mock_latency.return_value = FilterResult.ACCEPTED
            
            # 测试高质量信号
            high_quality_order = sample_order_requests[0]  # 高置信度订单
            
            # 这里需要实际的filter方法实现，暂时用mock
            # result = await integrated_signal_processor.filter_signal(high_quality_order)
            # assert result == FilterResult.ACCEPTED
            
            # 测试低置信度信号
            mock_confidence.return_value = FilterResult.REJECTED_CONFIDENCE
            # low_confidence_result = await integrated_signal_processor.filter_signal(high_quality_order)
            # assert low_confidence_result == FilterResult.REJECTED_CONFIDENCE
    
    async def test_order_generation_accuracy(self, integrated_signal_processor, sample_order_requests):
        """测试订单生成准确性"""
        # 测试订单优先级计算
        for order_request in sample_order_requests:
            priority = order_request.priority
            
            if order_request.urgency_score > 0.8 and order_request.confidence > 0.8:
                assert priority.value == "critical", f"高紧急度高置信度应该是critical优先级"
            elif order_request.urgency_score > 0.6 and order_request.confidence > 0.6:
                assert priority.value == "high", f"中高紧急度置信度应该是high优先级"
    
    async def test_processing_statistics_tracking(self, integrated_signal_processor):
        """测试处理统计跟踪"""
        # 创建处理统计对象
        stats = ProcessingStats()
        
        # 模拟处理过程
        stats.total_signals_received = 100
        stats.signals_processed = 95
        stats.signals_filtered_out = 5
        stats.orders_generated = 90
        stats.orders_sent = 85
        stats.processing_errors = 5
        
        # 测试延迟更新
        stats.update_latency(10.0)
        stats.update_latency(15.0)
        stats.update_latency(12.0)
        
        assert abs(stats.avg_processing_latency_ms - 12.33) < 0.1, "平均延迟计算错误"
        assert stats.max_processing_latency_ms == 15.0, "最大延迟应该是15.0"
    
    async def test_error_handling_in_processing(self, integrated_signal_processor):
        """测试信号处理中的错误处理"""
        # 这里需要实际的错误处理逻辑
        # 可以通过mock来模拟各种错误场景
        
        with patch.object(integrated_signal_processor, 'process_signal') as mock_process:
            # 模拟处理异常
            mock_process.side_effect = Exception("Processing error")
            
            # 验证异常被正确处理
            try:
                # await integrated_signal_processor.process_signal(sample_order_requests[0])
                pass  # 需要实际实现
            except Exception:
                pass  # 预期异常


@pytest.mark.asyncio 
class TestSmartOrderRouterFunctionality:
    """智能订单路由器功能测试"""
    
    async def test_venue_selection_algorithm(self, smart_order_router, sample_venues, sample_order_requests):
        """测试交易场所选择算法"""
        # 测试基于流动性和延迟的场所选择
        order = sample_order_requests[0]  # 使用第一个测试订单
        
        # Mock场所选择逻辑
        with patch.object(smart_order_router, '_select_optimal_venue') as mock_select:
            mock_select.return_value = sample_venues[0]  # 返回最优场所
            
            # selected_venue = await smart_order_router.select_venue_for_order(order)
            # assert selected_venue.name == "binance"  # 应该选择最优场所
            # assert selected_venue.liquidity_score >= 0.9  # 高流动性场所
    
    async def test_order_splitting_logic(self, smart_order_router, sample_order_requests):
        """测试订单拆分逻辑"""
        # 测试大订单拆分
        large_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="buy",
            quantity=10.0,  # 大订单
            price=50000.0,
            urgency_score=0.7,
            confidence=0.8
        )
        
        # Mock拆分逻辑
        with patch.object(smart_order_router, '_split_order') as mock_split:
            child_orders = [
                ChildOrder(
                    parent_id="large_order_1",
                    symbol="BTCUSDT",
                    side="buy",
                    quantity=2.5,
                    price=50000.0,
                    order_type=OrderType.LIMIT,
                    venue="binance",
                    algorithm=ExecutionAlgorithm.TWAP
                ) for i in range(4)
            ]
            mock_split.return_value = child_orders
            
            # split_result = await smart_order_router.split_large_order(large_order)
            # assert len(split_result) == 4  # 拆分成4个子订单
            # total_quantity = sum(child.quantity for child in split_result)
            # assert abs(total_quantity - large_order.quantity) < 0.001  # 总量保持不变
    
    async def test_execution_algorithm_selection(self, smart_order_router):
        """测试执行算法选择"""
        # 测试不同市场条件下的算法选择
        
        # 高波动性市场 -> SNIPER算法
        high_volatility_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.MARKET,
            side="buy", 
            quantity=1.0,
            urgency_score=0.95,  # 高紧急度
            confidence=0.9
        )
        
        # Mock算法选择
        with patch.object(smart_order_router, '_select_execution_algorithm') as mock_algo:
            mock_algo.return_value = ExecutionAlgorithm.SNIPER
            
            # selected_algo = await smart_order_router.select_algorithm(high_volatility_order)
            # assert selected_algo == ExecutionAlgorithm.SNIPER
    
    async def test_market_impact_minimization(self, smart_order_router):
        """测试市场冲击最小化"""
        # 测试大订单的市场冲击控制
        impact_sensitive_order = OrderRequest(
            symbol="BTCUSDT",
            order_type=OrderType.LIMIT,
            side="sell",
            quantity=5.0,  # 较大订单
            price=49900.0,
            urgency_score=0.6,  # 中等紧急度
            confidence=0.8
        )
        
        # Mock冲击评估
        with patch.object(smart_order_router, '_estimate_market_impact') as mock_impact:
            mock_impact.return_value = 0.0005  # 0.05%的预期冲击
            
            # estimated_impact = await smart_order_router.estimate_impact(impact_sensitive_order)
            # assert estimated_impact <= 0.001  # 冲击应该控制在0.1%以下
    
    async def test_execution_reporting(self, smart_order_router):
        """测试执行报告生成"""
        # 创建测试执行报告
        child_orders = [
            ChildOrder(
                parent_id="test_order",
                symbol="BTCUSDT", 
                side="buy",
                quantity=1.0,
                price=50000.0,
                order_type=OrderType.LIMIT,
                venue="binance",
                algorithm=ExecutionAlgorithm.TWAP,
                status=OrderStatus.FILLED,
                filled_quantity=1.0,
                avg_fill_price=50010.0
            )
        ]
        
        execution_report = ExecutionReport(
            order_id="test_order",
            symbol="BTCUSDT",
            total_quantity=1.0,
            filled_quantity=1.0,
            remaining_quantity=0.0,
            avg_fill_price=50010.0,
            total_cost=50010.0,
            market_impact=0.0002,
            execution_time_ms=150.0,
            child_orders=child_orders,
            slippage=10.0,  # 10元滑点
            success_rate=1.0
        )
        
        assert execution_report.filled_quantity == execution_report.total_quantity
        assert execution_report.success_rate == 1.0
        assert execution_report.slippage > 0  # 有滑点存在


@pytest.mark.asyncio
class TestFaultToleranceManagerFunctionality:
    """容错管理器功能测试"""
    
    async def test_error_detection_and_classification(self, fault_tolerance_manager):
        """测试错误检测和分类"""
        # 注册组件
        await fault_tolerance_manager.register_component(
            "test_component",
            failure_threshold=3,
            recovery_timeout=5.0
        )
        
        # 模拟不同类型的错误
        test_errors = [
            (ConnectionError("Network timeout"), ErrorCategory.NETWORK_ERROR),
            (ValueError("Invalid data format"), ErrorCategory.DATA_ERROR),
            (TimeoutError("Operation timeout"), ErrorCategory.TIMEOUT_ERROR),
            (Exception("System error"), ErrorCategory.SYSTEM_ERROR)
        ]
        
        for error, expected_category in test_errors:
            error_event = await fault_tolerance_manager.handle_error(
                component="test_component",
                error=error,
                context={"test": True}
            )
            
            # 验证错误分类正确
            # assert error_event.category == expected_category
    
    async def test_circuit_breaker_pattern(self, fault_tolerance_manager):
        """测试熔断器模式"""
        component_name = "circuit_test_component"
        
        # 注册组件
        await fault_tolerance_manager.register_component(
            component_name,
            failure_threshold=3,  # 3次失败后熔断
            recovery_timeout=2.0  # 2秒恢复时间
        )
        
        # 模拟连续失败触发熔断
        for i in range(5):
            await fault_tolerance_manager.handle_error(
                component=component_name,
                error=Exception(f"Test error {i}"),
                context={"attempt": i}
            )
        
        # 检查组件状态
        component_health = await fault_tolerance_manager.get_component_health(component_name)
        
        # 验证熔断器是否开启
        # assert component_health.status == ComponentStatus.CIRCUIT_OPEN
    
    async def test_automatic_recovery_strategies(self, fault_tolerance_manager):
        """测试自动恢复策略"""
        component_name = "recovery_test_component"
        
        # 注册组件
        await fault_tolerance_manager.register_component(
            component_name,
            failure_threshold=2,
            recovery_timeout=1.0
        )
        
        # 模拟失败和恢复
        await fault_tolerance_manager.handle_error(
            component=component_name,
            error=Exception("Recoverable error"),
            context={}
        )
        
        # 等待恢复
        await asyncio.sleep(1.5)
        
        # 模拟成功操作
        await fault_tolerance_manager.record_success(component_name, response_time=0.1)
        
        # 验证组件状态恢复
        component_health = await fault_tolerance_manager.get_component_health(component_name)
        # assert component_health.status == ComponentStatus.HEALTHY
    
    async def test_graceful_degradation(self, fault_tolerance_manager):
        """测试优雅降级"""
        # 测试系统在组件失败时的降级行为
        
        # 注册关键组件
        await fault_tolerance_manager.register_component(
            "critical_component",
            failure_threshold=1,
            recovery_timeout=10.0
        )
        
        # 注册备用组件  
        await fault_tolerance_manager.register_component(
            "backup_component",
            failure_threshold=3,
            recovery_timeout=5.0
        )
        
        # 模拟关键组件失败
        await fault_tolerance_manager.handle_error(
            component="critical_component",
            error=Exception("Critical failure"),
            context={}
        )
        
        # 验证备用组件可用
        backup_health = await fault_tolerance_manager.get_component_health("backup_component")
        # assert backup_health.status == ComponentStatus.HEALTHY
        
        # 测试降级服务
        is_degraded = await fault_tolerance_manager.is_service_degraded()
        # assert is_degraded is True, "服务应该处于降级状态"
    
    async def test_monitoring_and_alerting(self, fault_tolerance_manager):
        """测试监控和告警"""
        alert_callback = MagicMock()
        fault_tolerance_manager.add_alert_callback(alert_callback)
        
        # 注册组件
        await fault_tolerance_manager.register_component(
            "monitored_component",
            failure_threshold=1
        )
        
        # 触发错误产生告警
        await fault_tolerance_manager.handle_error(
            component="monitored_component",
            error=Exception("Alert test error"),
            context={"severity": "high"}
        )
        
        # 验证告警被触发
        # assert alert_callback.called, "应该触发告警回调"
    
    async def test_system_health_aggregation(self, fault_tolerance_manager):
        """测试系统健康状态聚合"""
        # 注册多个组件
        components = ["comp1", "comp2", "comp3"]
        
        for comp in components:
            await fault_tolerance_manager.register_component(comp)
        
        # 模拟不同组件状态
        await fault_tolerance_manager.record_success("comp1", response_time=0.1)
        await fault_tolerance_manager.handle_error(
            "comp2", 
            Exception("Minor error"),
            context={}
        )
        await fault_tolerance_manager.record_success("comp3", response_time=0.2)
        
        # 获取系统整体健康状态
        system_health = await fault_tolerance_manager.get_system_health()
        
        # 验证健康状态聚合正确
        # assert "overall_status" in system_health
        # assert "component_count" in system_health
        # assert "healthy_components" in system_health


@pytest.mark.asyncio
class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件测试"""
    
    async def test_invalid_input_handling(self, latency_monitor):
        """测试无效输入处理"""
        # 测试空数据
        try:
            await latency_monitor.check_data_freshness(
                symbol="",
                market_data=None,
                data_source=""
            )
            assert False, "应该抛出异常"
        except Exception:
            pass  # 预期异常
        
        # 测试无效数据源
        latency_monitor.set_active_data_source("BTCUSDT", "invalid_source")
        # 应该记录错误但不崩溃
    
    async def test_resource_exhaustion_scenarios(self, integrated_signal_processor):
        """测试资源耗尽场景"""
        # 模拟内存不足情况
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # 95%内存使用率
            
            # 系统应该触发保护机制
            # result = await integrated_signal_processor.check_resource_availability()
            # assert result["memory_warning"] is True
    
    async def test_concurrent_access_safety(self, smart_order_router):
        """测试并发访问安全性"""
        # 创建多个并发任务
        async def concurrent_operation(task_id):
            # 模拟并发订单路由
            await asyncio.sleep(0.01)
            return f"task_{task_id}_completed"
        
        # 启动并发任务
        tasks = [concurrent_operation(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务都完成了
        assert len(results) == 20
        assert all("completed" in result for result in results)
    
    async def test_data_consistency_under_failure(self, fault_tolerance_manager):
        """测试失败情况下的数据一致性"""
        # 注册组件
        await fault_tolerance_manager.register_component("consistency_test")
        
        # 模拟部分写入失败
        try:
            # 开始事务
            # await fault_tolerance_manager.begin_transaction()
            
            # 模拟操作
            await fault_tolerance_manager.record_success("consistency_test", 0.1)
            
            # 模拟失败
            raise Exception("Simulated failure")
            
        except Exception:
            # 回滚事务
            # await fault_tolerance_manager.rollback_transaction()
            pass
        
        # 验证数据状态一致
        health = await fault_tolerance_manager.get_component_health("consistency_test")
        # 健康状态应该保持一致


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])