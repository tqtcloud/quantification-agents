"""
双策略管理和隔离系统集成测试套件
测试HFT和AI策略的完整集成流程
"""

import asyncio
import pytest
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
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


class TestDualStrategyIntegration:
    """双策略集成测试类"""

    @pytest.fixture
    async def temp_config_dir(self):
        """临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    async def message_bus(self):
        """消息总线fixture"""
        bus = MessageBus()
        await bus.initialize()
        yield bus
        await bus.shutdown()

    @pytest.fixture
    async def resource_allocator(self):
        """资源分配器fixture"""
        allocator = ResourceAllocator()
        await allocator.initialize()
        yield allocator
        await allocator.cleanup()

    @pytest.fixture
    async def strategy_monitor(self, message_bus):
        """策略监控器fixture"""
        monitor = StrategyMonitor(message_bus)
        await monitor.initialize()
        yield monitor
        await monitor.shutdown()

    @pytest.fixture
    async def signal_aggregator(self, message_bus):
        """信号聚合器fixture"""
        aggregator = SignalAggregator(
            message_bus=message_bus,
            aggregation_strategy=AggregationStrategy.HYBRID_FUSION,
            enable_conflict_resolution=True
        )
        await aggregator.initialize()
        yield aggregator
        await aggregator.shutdown()

    @pytest.fixture
    async def strategy_manager(self, message_bus, temp_config_dir):
        """策略管理器fixture"""
        manager = StrategyManager(
            message_bus=message_bus,
            config_dir=temp_config_dir
        )
        await manager.initialize()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    def hft_config(self):
        """HFT配置"""
        return HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("1.0"),
            latency_target_ms=5,
            max_position=Decimal("10.0"),
            risk_limit=Decimal("1000.0")
        )

    @pytest.fixture
    def ai_config(self):
        """AI策略配置"""
        return WorkflowConfig(
            max_agents=5,
            timeout_seconds=30,
            enable_parallel_execution=True
        )

    @pytest.mark.asyncio
    async def test_dual_strategy_lifecycle(self, strategy_manager, hft_config, ai_config):
        """测试双策略完整生命周期"""
        # 1. 创建HFT策略
        hft_strategy_id = await strategy_manager.create_strategy(
            name="test_hft",
            strategy_type=StrategyType.HFT,
            config=hft_config
        )
        assert hft_strategy_id is not None
        
        # 2. 创建AI策略
        ai_strategy_id = await strategy_manager.create_strategy(
            name="test_ai",
            strategy_type=StrategyType.AI_AGENT,
            config=ai_config
        )
        assert ai_strategy_id is not None
        
        # 3. 启动双策略
        hft_start_result = await strategy_manager.start_strategy(hft_strategy_id)
        assert hft_start_result is True
        
        ai_start_result = await strategy_manager.start_strategy(ai_strategy_id)
        assert ai_start_result is True
        
        # 等待策略启动完成
        await asyncio.sleep(2)
        
        # 4. 验证策略状态
        hft_status = await strategy_manager.get_strategy_status(hft_strategy_id)
        assert hft_status == StrategyStatus.RUNNING
        
        ai_status = await strategy_manager.get_strategy_status(ai_strategy_id)
        assert ai_status == StrategyStatus.RUNNING
        
        # 5. 验证资源分配
        resource_info = await strategy_manager.get_resource_allocation()
        assert len(resource_info) == 2
        assert hft_strategy_id in resource_info
        assert ai_strategy_id in resource_info
        
        # 6. 停止策略
        hft_stop_result = await strategy_manager.stop_strategy(hft_strategy_id)
        assert hft_stop_result is True
        
        ai_stop_result = await strategy_manager.stop_strategy(ai_strategy_id)
        assert ai_stop_result is True
        
        # 等待策略停止完成
        await asyncio.sleep(1)
        
        # 7. 验证最终状态
        hft_final_status = await strategy_manager.get_strategy_status(hft_strategy_id)
        assert hft_final_status == StrategyStatus.STOPPED
        
        ai_final_status = await strategy_manager.get_strategy_status(ai_strategy_id)
        assert ai_final_status == StrategyStatus.STOPPED

    @pytest.mark.asyncio
    async def test_resource_isolation(self, strategy_manager, resource_allocator, hft_config, ai_config):
        """测试资源隔离效果"""
        # 设置资源限制
        hft_limits = ResourceLimit(
            memory_mb=512,
            cpu_percent=30.0,
            network_connections=100
        )
        ai_limits = ResourceLimit(
            memory_mb=1024,
            cpu_percent=50.0,
            network_connections=50
        )
        
        # 创建策略并设置资源限制
        hft_strategy_id = await strategy_manager.create_strategy(
            name="hft_isolated",
            strategy_type=StrategyType.HFT,
            config=hft_config,
            resource_limits=hft_limits
        )
        
        ai_strategy_id = await strategy_manager.create_strategy(
            name="ai_isolated",
            strategy_type=StrategyType.AI_AGENT,
            config=ai_config,
            resource_limits=ai_limits
        )
        
        # 启动策略
        await strategy_manager.start_strategy(hft_strategy_id)
        await strategy_manager.start_strategy(ai_strategy_id)
        
        # 等待稳定运行
        await asyncio.sleep(3)
        
        # 验证资源隔离
        hft_resource_usage = await resource_allocator.get_strategy_resource_usage(hft_strategy_id)
        ai_resource_usage = await resource_allocator.get_strategy_resource_usage(ai_strategy_id)
        
        # HFT策略资源使用应在限制内
        assert hft_resource_usage.memory_mb <= hft_limits.memory_mb
        assert hft_resource_usage.cpu_percent <= hft_limits.cpu_percent
        
        # AI策略资源使用应在限制内
        assert ai_resource_usage.memory_mb <= ai_limits.memory_mb
        assert ai_resource_usage.cpu_percent <= ai_limits.cpu_percent
        
        # 验证两策略的资源使用互相独立
        total_memory = hft_resource_usage.memory_mb + ai_resource_usage.memory_mb
        total_cpu = hft_resource_usage.cpu_percent + ai_resource_usage.cpu_percent
        
        # 总资源使用应小于系统限制
        assert total_memory <= hft_limits.memory_mb + ai_limits.memory_mb
        assert total_cpu <= hft_limits.cpu_percent + ai_limits.cpu_percent

    @pytest.mark.asyncio
    async def test_signal_aggregation_end_to_end(self, signal_aggregator, strategy_manager, hft_config, ai_config):
        """测试信号聚合端到端流程"""
        # 创建并启动双策略
        hft_strategy_id = await strategy_manager.create_strategy(
            name="hft_signal_test",
            strategy_type=StrategyType.HFT,
            config=hft_config
        )
        
        ai_strategy_id = await strategy_manager.create_strategy(
            name="ai_signal_test", 
            strategy_type=StrategyType.AI_AGENT,
            config=ai_config
        )
        
        await strategy_manager.start_strategy(hft_strategy_id)
        await strategy_manager.start_strategy(ai_strategy_id)
        
        # 等待策略启动
        await asyncio.sleep(1)
        
        # 模拟HFT信号
        hft_signal = TradingSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=SignalStrength.STRONG,
            price=Decimal("50000.0"),
            quantity=Decimal("0.1"),
            timestamp=time.time(),
            source="hft_engine",
            confidence=0.9
        )
        
        # 模拟AI信号
        ai_signal = TradingSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=SignalStrength.MEDIUM,
            price=Decimal("50050.0"),
            quantity=Decimal("0.2"),
            timestamp=time.time(),
            source="ai_agent",
            confidence=0.8
        )
        
        # 添加信号输入
        hft_input = SignalInput(
            signal_id="hft_001",
            signal=hft_signal,
            source_strategy_id=hft_strategy_id,
            weight=0.6,
            priority=1
        )
        
        ai_input = SignalInput(
            signal_id="ai_001",
            signal=ai_signal,
            source_strategy_id=ai_strategy_id,
            weight=0.4,
            priority=2
        )
        
        # 执行信号聚合
        aggregation_result = await signal_aggregator.aggregate_signals([hft_input, ai_input])
        
        # 验证聚合结果
        assert aggregation_result is not None
        assert aggregation_result.symbol == "BTCUSDT"
        assert aggregation_result.side == "buy"
        
        # 价格应该在两个信号之间
        assert Decimal("50000.0") <= aggregation_result.price <= Decimal("50050.0")
        
        # 数量应该合理聚合
        assert aggregation_result.quantity > Decimal("0")
        
        # 信心度应该反映聚合质量
        assert 0.7 <= aggregation_result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self, signal_aggregator, strategy_manager, hft_config, ai_config):
        """测试冲突检测和解决"""
        # 创建策略
        hft_strategy_id = await strategy_manager.create_strategy(
            name="hft_conflict_test",
            strategy_type=StrategyType.HFT,
            config=hft_config
        )
        
        ai_strategy_id = await strategy_manager.create_strategy(
            name="ai_conflict_test",
            strategy_type=StrategyType.AI_AGENT,
            config=ai_config
        )
        
        # 创建冲突信号（一个买入，一个卖出）
        hft_signal = TradingSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=SignalStrength.STRONG,
            price=Decimal("50000.0"),
            quantity=Decimal("0.1"),
            timestamp=time.time(),
            source="hft_engine",
            confidence=0.9
        )
        
        ai_signal = TradingSignal(
            symbol="BTCUSDT",
            side="sell",  # 相反方向
            strength=SignalStrength.STRONG,
            price=Decimal("50000.0"),
            quantity=Decimal("0.15"),
            timestamp=time.time(),
            source="ai_agent",
            confidence=0.85
        )
        
        hft_input = SignalInput(
            signal_id="hft_conflict",
            signal=hft_signal,
            source_strategy_id=hft_strategy_id,
            weight=0.5,
            priority=1
        )
        
        ai_input = SignalInput(
            signal_id="ai_conflict",
            signal=ai_signal,
            source_strategy_id=ai_strategy_id,
            weight=0.5,
            priority=1
        )
        
        # 执行聚合（应该检测到冲突）
        result = await signal_aggregator.aggregate_signals([hft_input, ai_input])
        
        # 验证冲突处理结果
        if result is not None:
            # 如果返回结果，应该是经过冲突解决的
            assert result.symbol == "BTCUSDT"
            # 数量应该反映冲突解决后的净值
            assert result.quantity >= Decimal("0")
        
        # 获取冲突统计
        conflict_stats = await signal_aggregator.get_conflict_statistics()
        assert conflict_stats["total_conflicts"] > 0
        assert conflict_stats["resolved_conflicts"] >= 0

    @pytest.mark.asyncio
    async def test_monitoring_and_alerts(self, strategy_monitor, strategy_manager, hft_config, ai_config):
        """测试监控告警功能"""
        # 设置告警回调
        alerts_received = []
        
        async def alert_callback(alert_data):
            alerts_received.append(alert_data)
        
        strategy_monitor.set_alert_callback(alert_callback)
        
        # 创建策略
        strategy_id = await strategy_manager.create_strategy(
            name="monitored_strategy",
            strategy_type=StrategyType.HFT,
            config=hft_config
        )
        
        # 启动监控
        await strategy_monitor.start_monitoring(
            strategy_id=strategy_id,
            monitoring_level=MonitoringLevel.DETAILED
        )
        
        # 启动策略
        await strategy_manager.start_strategy(strategy_id)
        
        # 等待监控数据收集
        await asyncio.sleep(3)
        
        # 验证监控数据
        monitoring_data = await strategy_monitor.get_strategy_metrics(strategy_id)
        assert monitoring_data is not None
        assert "performance" in monitoring_data
        assert "resource_usage" in monitoring_data
        assert "status" in monitoring_data
        
        # 模拟异常情况触发告警
        await strategy_monitor.trigger_resource_alert(
            strategy_id=strategy_id,
            alert_type="high_memory_usage",
            threshold=80.0,
            current_value=90.0
        )
        
        # 等待告警处理
        await asyncio.sleep(1)
        
        # 验证告警
        assert len(alerts_received) > 0
        alert = alerts_received[0]
        assert alert["strategy_id"] == strategy_id
        assert alert["alert_type"] == "high_memory_usage"

    @pytest.mark.asyncio
    async def test_strategy_priority_management(self, strategy_manager, hft_config, ai_config):
        """测试策略优先级动态调整"""
        # 创建多个策略
        strategies = []
        
        for i in range(3):
            strategy_id = await strategy_manager.create_strategy(
                name=f"priority_test_{i}",
                strategy_type=StrategyType.HFT if i % 2 == 0 else StrategyType.AI_AGENT,
                config=hft_config if i % 2 == 0 else ai_config,
                priority=i + 1
            )
            strategies.append(strategy_id)
        
        # 启动策略
        for strategy_id in strategies:
            await strategy_manager.start_strategy(strategy_id)
        
        await asyncio.sleep(2)
        
        # 获取初始优先级
        initial_priorities = {}
        for strategy_id in strategies:
            priority = await strategy_manager.get_strategy_priority(strategy_id)
            initial_priorities[strategy_id] = priority
        
        # 动态调整优先级
        new_priority = 10
        await strategy_manager.update_strategy_priority(strategies[0], new_priority)
        
        # 验证优先级更新
        updated_priority = await strategy_manager.get_strategy_priority(strategies[0])
        assert updated_priority == new_priority
        
        # 验证优先级影响资源分配
        resource_info = await strategy_manager.get_resource_allocation()
        highest_priority_strategy = strategies[0]
        assert resource_info[highest_priority_strategy]["priority"] == new_priority

    @pytest.mark.asyncio
    async def test_strategy_failure_isolation(self, strategy_manager, hft_config, ai_config):
        """测试策略故障隔离"""
        # 创建双策略
        hft_strategy_id = await strategy_manager.create_strategy(
            name="hft_isolation_test",
            strategy_type=StrategyType.HFT,
            config=hft_config
        )
        
        ai_strategy_id = await strategy_manager.create_strategy(
            name="ai_isolation_test",
            strategy_type=StrategyType.AI_AGENT,
            config=ai_config
        )
        
        # 启动策略
        await strategy_manager.start_strategy(hft_strategy_id)
        await strategy_manager.start_strategy(ai_strategy_id)
        
        # 等待稳定运行
        await asyncio.sleep(2)
        
        # 验证双策略都在运行
        hft_status = await strategy_manager.get_strategy_status(hft_strategy_id)
        ai_status = await strategy_manager.get_strategy_status(ai_strategy_id)
        assert hft_status == StrategyStatus.RUNNING
        assert ai_status == StrategyStatus.RUNNING
        
        # 模拟HFT策略故障
        with patch.object(strategy_manager, '_handle_strategy_error') as mock_error_handler:
            # 触发策略错误
            await strategy_manager.handle_strategy_error(
                strategy_id=hft_strategy_id,
                error_type="execution_error",
                error_message="模拟故障",
                auto_recover=True
            )
            
            # 等待错误处理
            await asyncio.sleep(1)
            
            # 验证错误处理被调用
            mock_error_handler.assert_called_once()
        
        # 验证AI策略仍在正常运行（故障隔离）
        ai_status_after_hft_error = await strategy_manager.get_strategy_status(ai_strategy_id)
        assert ai_status_after_hft_error == StrategyStatus.RUNNING
        
        # 验证HFT策略可能已停止或在恢复中
        hft_status_after_error = await strategy_manager.get_strategy_status(hft_strategy_id)
        assert hft_status_after_error in [
            StrategyStatus.ERROR,
            StrategyStatus.STOPPED,
            StrategyStatus.RUNNING  # 如果自动恢复成功
        ]

    @pytest.mark.asyncio
    async def test_concurrent_strategy_operations(self, strategy_manager, hft_config, ai_config):
        """测试并发策略操作"""
        # 创建多个策略
        strategy_ids = []
        create_tasks = []
        
        for i in range(5):
            task = strategy_manager.create_strategy(
                name=f"concurrent_test_{i}",
                strategy_type=StrategyType.HFT if i % 2 == 0 else StrategyType.AI_AGENT,
                config=hft_config if i % 2 == 0 else ai_config
            )
            create_tasks.append(task)
        
        # 并发创建策略
        strategy_ids = await asyncio.gather(*create_tasks)
        assert len(strategy_ids) == 5
        assert all(sid is not None for sid in strategy_ids)
        
        # 并发启动策略
        start_tasks = [
            strategy_manager.start_strategy(strategy_id)
            for strategy_id in strategy_ids
        ]
        start_results = await asyncio.gather(*start_tasks, return_exceptions=True)
        
        # 验证启动结果（允许部分失败，但不应该有异常）
        successful_starts = sum(1 for result in start_results if result is True)
        assert successful_starts >= 3  # 至少3个成功启动
        
        # 等待稳定
        await asyncio.sleep(2)
        
        # 并发停止策略
        stop_tasks = [
            strategy_manager.stop_strategy(strategy_id)
            for strategy_id in strategy_ids
        ]
        stop_results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # 验证停止结果
        successful_stops = sum(1 for result in stop_results if result is True)
        assert successful_stops >= successful_starts  # 启动成功的都应该能成功停止

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, strategy_manager, signal_aggregator, strategy_monitor):
        """测试组件间数据一致性"""
        # 创建策略
        strategy_id = await strategy_manager.create_strategy(
            name="consistency_test",
            strategy_type=StrategyType.HFT,
            config=HFTConfig(
                symbol="BTCUSDT",
                min_order_size=Decimal("0.001"),
                max_order_size=Decimal("1.0"),
                latency_target_ms=5
            )
        )
        
        # 启动策略和监控
        await strategy_manager.start_strategy(strategy_id)
        await strategy_monitor.start_monitoring(strategy_id)
        
        await asyncio.sleep(2)
        
        # 从不同组件获取策略信息
        manager_info = await strategy_manager.get_strategy_info(strategy_id)
        monitor_info = await strategy_monitor.get_strategy_metrics(strategy_id)
        
        # 验证基本信息一致性
        assert manager_info["strategy_id"] == strategy_id
        assert manager_info["status"] == StrategyStatus.RUNNING
        
        # 验证监控数据包含策略信息
        assert monitor_info["strategy_id"] == strategy_id
        assert monitor_info["status"] == "running"
        
        # 验证时间戳合理性
        current_time = time.time()
        assert abs(manager_info["last_update"] - current_time) < 60  # 1分钟内
        assert abs(monitor_info["last_update"] - current_time) < 60

    @pytest.mark.asyncio
    async def test_system_recovery_after_shutdown(self, message_bus, temp_config_dir):
        """测试系统重启后的恢复"""
        # 第一次启动：创建并运行策略
        manager1 = StrategyManager(message_bus=message_bus, config_dir=temp_config_dir)
        await manager1.initialize()
        
        strategy_id = await manager1.create_strategy(
            name="recovery_test",
            strategy_type=StrategyType.HFT,
            config=HFTConfig(
                symbol="BTCUSDT",
                min_order_size=Decimal("0.001")
            )
        )
        
        await manager1.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 验证策略运行
        status1 = await manager1.get_strategy_status(strategy_id)
        assert status1 == StrategyStatus.RUNNING
        
        # 模拟系统关闭
        await manager1.shutdown()
        
        # 第二次启动：验证恢复
        manager2 = StrategyManager(message_bus=message_bus, config_dir=temp_config_dir)
        await manager2.initialize()
        
        # 验证策略配置恢复
        strategies = await manager2.list_strategies()
        assert len(strategies) == 1
        assert strategies[0]["strategy_id"] == strategy_id
        assert strategies[0]["name"] == "recovery_test"
        
        # 清理
        await manager2.shutdown()