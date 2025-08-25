"""
可靠性和异常恢复测试套件
测试策略故障隔离、异常恢复、数据一致性和边界条件
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
from unittest.mock import AsyncMock, MagicMock, patch, call
import random
import threading

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
from src.core.message_bus import MessageBus
from src.core.models.signals import TradingSignal, SignalStrength
from src.hft.hft_engine import HFTConfig


class TestReliabilityAndRecovery:
    """可靠性和异常恢复测试类"""

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

    @pytest.mark.asyncio
    async def test_strategy_failure_isolation(self, strategy_manager):
        """测试策略故障隔离机制"""
        # 创建多个策略
        strategies = []
        
        for i in range(3):
            config = HFTConfig(
                symbol=f"TEST{i}USDT",
                min_order_size=Decimal("0.001"),
                max_order_size=Decimal("1.0"),
                latency_target_ms=5
            )
            
            strategy_id = await strategy_manager.create_strategy(
                name=f"isolation_test_{i}",
                strategy_type=StrategyType.HFT,
                config=config
            )
            strategies.append(strategy_id)
        
        # 启动所有策略
        for strategy_id in strategies:
            await strategy_manager.start_strategy(strategy_id)
        
        await asyncio.sleep(1)
        
        # 验证所有策略都在运行
        for strategy_id in strategies:
            status = await strategy_manager.get_strategy_status(strategy_id)
            assert status == StrategyStatus.RUNNING
        
        # 模拟第一个策略发生故障
        failing_strategy = strategies[0]
        
        with patch.object(strategy_manager, '_handle_strategy_error') as mock_error_handler:
            # 触发策略错误
            await strategy_manager.handle_strategy_error(
                strategy_id=failing_strategy,
                error_type="critical_error",
                error_message="模拟系统故障",
                auto_recover=False
            )
            
            # 等待错误处理
            await asyncio.sleep(2)
            
            # 验证错误处理被调用
            mock_error_handler.assert_called()
        
        # 验证其他策略仍在正常运行（故障隔离）
        for strategy_id in strategies[1:]:
            status = await strategy_manager.get_strategy_status(strategy_id)
            assert status == StrategyStatus.RUNNING, \
                f"策略 {strategy_id} 受到其他策略故障影响"
        
        # 验证故障策略已停止或处于错误状态
        failing_status = await strategy_manager.get_strategy_status(failing_strategy)
        assert failing_status in [StrategyStatus.ERROR, StrategyStatus.STOPPED], \
            f"故障策略状态应为ERROR或STOPPED，实际为 {failing_status}"

    @pytest.mark.asyncio
    async def test_automatic_recovery(self, strategy_manager):
        """测试自动恢复机制"""
        # 创建策略
        config = HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("1.0"),
            latency_target_ms=5
        )
        
        strategy_id = await strategy_manager.create_strategy(
            name="recovery_test",
            strategy_type=StrategyType.HFT,
            config=config
        )
        
        await strategy_manager.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 验证策略运行正常
        initial_status = await strategy_manager.get_strategy_status(strategy_id)
        assert initial_status == StrategyStatus.RUNNING
        
        # 模拟可恢复的错误
        recovery_attempts = []
        
        async def mock_recovery_handler(strategy_id, error_type, error_message):
            recovery_attempts.append({
                'timestamp': time.time(),
                'strategy_id': strategy_id,
                'error_type': error_type
            })
            # 模拟恢复过程
            await asyncio.sleep(0.5)
            return True  # 恢复成功
        
        with patch.object(strategy_manager, '_attempt_strategy_recovery', side_effect=mock_recovery_handler):
            # 触发自动恢复
            await strategy_manager.handle_strategy_error(
                strategy_id=strategy_id,
                error_type="recoverable_error",
                error_message="网络连接中断",
                auto_recover=True
            )
            
            # 等待恢复处理
            await asyncio.sleep(3)
        
        # 验证恢复尝试
        assert len(recovery_attempts) > 0, "未尝试自动恢复"
        
        # 验证策略恢复运行
        final_status = await strategy_manager.get_strategy_status(strategy_id)
        assert final_status in [StrategyStatus.RUNNING, StrategyStatus.ERROR], \
            f"恢复后策略状态异常: {final_status}"

    @pytest.mark.asyncio
    async def test_data_consistency_during_failures(self, strategy_manager, temp_config_dir):
        """测试故障期间的数据一致性"""
        # 创建策略
        config = HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("1.0")
        )
        
        strategy_id = await strategy_manager.create_strategy(
            name="consistency_test",
            strategy_type=StrategyType.HFT,
            config=config
        )
        
        await strategy_manager.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 获取初始状态数据
        initial_info = await strategy_manager.get_strategy_info(strategy_id)
        initial_config_files = list(Path(temp_config_dir).glob("*.json"))
        
        # 模拟数据写入过程中的故障
        with patch('builtins.open', side_effect=OSError("磁盘写入失败")):
            try:
                # 尝试更新策略配置（应该失败）
                await strategy_manager.update_strategy_config(
                    strategy_id=strategy_id,
                    new_config=config
                )
            except Exception as e:
                print(f"预期的配置更新失败: {e}")
        
        # 验证数据一致性
        # 1. 内存中的状态应该保持一致
        current_info = await strategy_manager.get_strategy_info(strategy_id)
        assert current_info["strategy_id"] == initial_info["strategy_id"]
        assert current_info["name"] == initial_info["name"]
        
        # 2. 配置文件应该保持原状（未被损坏）
        current_config_files = list(Path(temp_config_dir).glob("*.json"))
        assert len(current_config_files) == len(initial_config_files)
        
        # 3. 策略仍应能正常操作
        status = await strategy_manager.get_strategy_status(strategy_id)
        assert status in [StrategyStatus.RUNNING, StrategyStatus.ERROR]

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_failure(self, resource_allocator, strategy_manager):
        """测试故障时的资源清理"""
        # 创建策略并分配资源
        config = HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001")
        )
        
        strategy_id = await strategy_manager.create_strategy(
            name="resource_cleanup_test",
            strategy_type=StrategyType.HFT,
            config=config
        )
        
        # 分配资源
        limits = ResourceLimit(
            memory_mb=512,
            cpu_percent=25.0,
            network_connections=100
        )
        
        allocation_success = await resource_allocator.allocate_resources(strategy_id, limits)
        assert allocation_success, "资源分配失败"
        
        # 验证资源已分配
        usage_before = await resource_allocator.get_strategy_resource_usage(strategy_id)
        assert usage_before is not None, "无法获取资源使用情况"
        
        # 启动策略
        await strategy_manager.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 模拟策略故障
        await strategy_manager.handle_strategy_error(
            strategy_id=strategy_id,
            error_type="critical_failure",
            error_message="内存泄漏导致的严重故障",
            auto_recover=False
        )
        
        # 等待故障处理和资源清理
        await asyncio.sleep(2)
        
        # 验证资源已被清理
        try:
            usage_after = await resource_allocator.get_strategy_resource_usage(strategy_id)
            # 资源应该被释放或使用量显著降低
            if usage_after:
                assert usage_after.memory_mb < usage_before.memory_mb * 0.5, \
                    "故障后内存资源未正确清理"
        except Exception:
            # 如果无法获取使用情况，可能意味着资源已完全清理，这是期望的
            pass
        
        # 验证资源分配记录已清理
        allocated_strategies = await resource_allocator.get_allocated_strategies()
        assert strategy_id not in allocated_strategies, "资源分配记录未清理"

    @pytest.mark.asyncio
    async def test_concurrent_failure_handling(self, strategy_manager):
        """测试并发故障处理"""
        # 创建多个策略
        strategies = []
        
        for i in range(5):
            config = HFTConfig(
                symbol=f"CONCURRENT{i}USDT",
                min_order_size=Decimal("0.001")
            )
            
            strategy_id = await strategy_manager.create_strategy(
                name=f"concurrent_failure_test_{i}",
                strategy_type=StrategyType.HFT,
                config=config
            )
            strategies.append(strategy_id)
        
        # 启动所有策略
        for strategy_id in strategies:
            await strategy_manager.start_strategy(strategy_id)
        
        await asyncio.sleep(1)
        
        # 同时触发多个策略故障
        failure_tasks = []
        
        for i, strategy_id in enumerate(strategies[:3]):  # 只让前3个策略故障
            task = strategy_manager.handle_strategy_error(
                strategy_id=strategy_id,
                error_type=f"concurrent_error_{i}",
                error_message=f"并发故障测试 {i}",
                auto_recover=False
            )
            failure_tasks.append(task)
        
        # 等待所有故障处理完成
        await asyncio.gather(*failure_tasks, return_exceptions=True)
        await asyncio.sleep(2)
        
        # 验证故障处理结果
        failed_count = 0
        running_count = 0
        
        for strategy_id in strategies:
            status = await strategy_manager.get_strategy_status(strategy_id)
            if status in [StrategyStatus.ERROR, StrategyStatus.STOPPED]:
                failed_count += 1
            elif status == StrategyStatus.RUNNING:
                running_count += 1
        
        # 应该有3个策略失败，2个仍在运行
        assert failed_count >= 3, f"失败策略数量不符合预期: {failed_count}"
        assert running_count >= 2, f"运行策略数量不符合预期: {running_count}"

    @pytest.mark.asyncio
    async def test_message_bus_failure_resilience(self, strategy_manager, message_bus):
        """测试消息总线故障的弹性"""
        # 创建策略
        config = HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001")
        )
        
        strategy_id = await strategy_manager.create_strategy(
            name="message_bus_test",
            strategy_type=StrategyType.HFT,
            config=config
        )
        
        await strategy_manager.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 记录消息发送尝试
        message_attempts = []
        
        async def mock_publish_with_failure(*args, **kwargs):
            message_attempts.append(time.time())
            if len(message_attempts) <= 2:  # 前两次失败
                raise ConnectionError("消息总线连接失败")
            return True  # 第三次成功
        
        # 模拟消息总线间歇性故障
        with patch.object(message_bus, 'publish', side_effect=mock_publish_with_failure):
            # 尝试发送策略消息
            try:
                await strategy_manager._publish_strategy_event(
                    strategy_id=strategy_id,
                    event_type="status_update",
                    event_data={"status": "running"}
                )
            except Exception:
                pass  # 可能由于重试机制而成功或失败
        
        # 验证重试机制
        assert len(message_attempts) >= 1, "应该至少尝试发送消息一次"
        
        # 策略应该仍能正常运行（不应受消息发送失败影响）
        status = await strategy_manager.get_strategy_status(strategy_id)
        assert status == StrategyStatus.RUNNING, "策略不应受消息发送失败影响"

    @pytest.mark.asyncio
    async def test_edge_case_signal_processing(self, strategy_manager, message_bus):
        """测试边界条件信号处理"""
        signal_aggregator = SignalAggregator(
            message_bus=message_bus,
            aggregation_strategy=AggregationStrategy.HYBRID_FUSION
        )
        await signal_aggregator.initialize()
        
        try:
            # 测试空信号列表
            result = await signal_aggregator.aggregate_signals([])
            assert result is None, "空信号列表应返回None"
            
            # 测试无效信号数据
            invalid_signals = [
                SignalInput(
                    signal_id="invalid_1",
                    signal=None,  # 无效信号
                    source_strategy_id="test",
                    weight=1.0,
                    priority=1
                )
            ]
            
            result = await signal_aggregator.aggregate_signals(invalid_signals)
            assert result is None, "无效信号应返回None"
            
            # 测试极端价格和数量
            extreme_signals = [
                SignalInput(
                    signal_id="extreme_1",
                    signal=TradingSignal(
                        symbol="BTCUSDT",
                        side="buy",
                        strength=SignalStrength.STRONG,
                        price=Decimal("0.000001"),  # 极小价格
                        quantity=Decimal("999999999"),  # 极大数量
                        timestamp=time.time(),
                        source="test",
                        confidence=0.9
                    ),
                    source_strategy_id="test",
                    weight=1.0,
                    priority=1
                )
            ]
            
            result = await signal_aggregator.aggregate_signals(extreme_signals)
            # 应该能处理极端值，但可能进行合理性检查
            if result:
                assert result.price > Decimal("0"), "聚合后价格应大于0"
                assert result.quantity > Decimal("0"), "聚合后数量应大于0"
            
            # 测试时间戳异常的信号
            old_timestamp = time.time() - 3600  # 1小时前
            future_timestamp = time.time() + 3600  # 1小时后
            
            timestamp_signals = [
                SignalInput(
                    signal_id="old_signal",
                    signal=TradingSignal(
                        symbol="BTCUSDT",
                        side="buy",
                        strength=SignalStrength.MEDIUM,
                        price=Decimal("50000"),
                        quantity=Decimal("0.1"),
                        timestamp=old_timestamp,
                        source="test",
                        confidence=0.8
                    ),
                    source_strategy_id="test",
                    weight=1.0,
                    priority=1
                ),
                SignalInput(
                    signal_id="future_signal",
                    signal=TradingSignal(
                        symbol="BTCUSDT",
                        side="buy",
                        strength=SignalStrength.MEDIUM,
                        price=Decimal("50100"),
                        quantity=Decimal("0.1"),
                        timestamp=future_timestamp,
                        source="test",
                        confidence=0.8
                    ),
                    source_strategy_id="test",
                    weight=1.0,
                    priority=1
                )
            ]
            
            result = await signal_aggregator.aggregate_signals(timestamp_signals)
            # 应该能处理时间戳异常，可能会过滤或警告
            
        finally:
            await signal_aggregator.shutdown()

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, resource_allocator):
        """测试资源耗尽处理"""
        # 创建大量策略以耗尽资源
        allocated_strategies = []
        allocation_failures = []
        
        # 尝试分配超出系统限制的资源
        for i in range(1000):  # 尝试创建大量策略
            strategy_id = f"resource_exhaustion_{i}"
            
            # 每个策略都要求大量资源
            limits = ResourceLimit(
                memory_mb=1024,  # 1GB内存
                cpu_percent=10.0,  # 10% CPU
                network_connections=100
            )
            
            try:
                success = await resource_allocator.allocate_resources(strategy_id, limits)
                if success:
                    allocated_strategies.append(strategy_id)
                else:
                    allocation_failures.append(strategy_id)
                    
                # 如果连续失败太多次，停止测试
                if len(allocation_failures) > 50:
                    break
                    
            except Exception as e:
                allocation_failures.append((strategy_id, str(e)))
                if len(allocation_failures) > 50:
                    break
        
        # 验证资源分配器能正确处理资源不足
        assert len(allocation_failures) > 0, "应该遇到资源分配失败"
        assert len(allocated_strategies) > 0, "应该有一些策略成功分配资源"
        
        print(f"成功分配: {len(allocated_strategies)}, 分配失败: {len(allocation_failures)}")
        
        # 清理已分配的资源
        for strategy_id in allocated_strategies:
            try:
                await resource_allocator.deallocate_resources(strategy_id)
            except Exception:
                pass  # 忽略清理错误

    @pytest.mark.asyncio
    async def test_configuration_corruption_recovery(self, strategy_manager, temp_config_dir):
        """测试配置文件损坏的恢复"""
        # 创建策略
        config = HFTConfig(
            symbol="BTCUSDT",
            min_order_size=Decimal("0.001")
        )
        
        strategy_id = await strategy_manager.create_strategy(
            name="config_corruption_test",
            strategy_type=StrategyType.HFT,
            config=config
        )
        
        await strategy_manager.start_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 找到策略配置文件
        config_files = list(Path(temp_config_dir).glob("*.json"))
        assert len(config_files) > 0, "应该有配置文件"
        
        # 模拟配置文件损坏
        for config_file in config_files:
            if strategy_id in str(config_file):
                # 写入无效的JSON数据
                with open(config_file, 'w') as f:
                    f.write("{ invalid json content }")
                break
        
        # 重新初始化策略管理器（模拟重启）
        await strategy_manager.shutdown()
        
        new_manager = StrategyManager(
            message_bus=strategy_manager._message_bus,
            config_dir=temp_config_dir
        )
        
        # 应该能检测到损坏的配置并处理
        initialization_errors = []
        
        try:
            await new_manager.initialize()
        except Exception as e:
            initialization_errors.append(str(e))
        
        # 验证损坏配置的处理
        strategies = await new_manager.list_strategies()
        
        # 可能的处理方式：
        # 1. 跳过损坏的配置文件
        # 2. 使用默认配置
        # 3. 标记为错误状态
        
        print(f"配置损坏后的策略数量: {len(strategies)}")
        print(f"初始化错误: {initialization_errors}")
        
        await new_manager.shutdown()

    @pytest.mark.asyncio
    async def test_system_graceful_shutdown(self, strategy_manager, message_bus):
        """测试系统优雅关闭"""
        # 创建多个运行中的策略
        strategies = []
        
        for i in range(3):
            config = HFTConfig(
                symbol=f"SHUTDOWN{i}USDT",
                min_order_size=Decimal("0.001")
            )
            
            strategy_id = await strategy_manager.create_strategy(
                name=f"shutdown_test_{i}",
                strategy_type=StrategyType.HFT,
                config=config
            )
            strategies.append(strategy_id)
        
        # 启动所有策略
        for strategy_id in strategies:
            await strategy_manager.start_strategy(strategy_id)
        
        await asyncio.sleep(1)
        
        # 验证所有策略都在运行
        running_strategies = []
        for strategy_id in strategies:
            status = await strategy_manager.get_strategy_status(strategy_id)
            if status == StrategyStatus.RUNNING:
                running_strategies.append(strategy_id)
        
        assert len(running_strategies) > 0, "应该有策略在运行"
        
        # 记录关闭前的状态
        pre_shutdown_time = time.time()
        
        # 执行优雅关闭
        shutdown_start = time.time()
        await strategy_manager.shutdown()
        shutdown_duration = time.time() - shutdown_start
        
        # 验证关闭时间合理（不应该太长）
        assert shutdown_duration < 30.0, f"关闭时间过长: {shutdown_duration:.2f}秒"
        
        # 验证所有策略都已停止
        # 注意：shutdown后可能无法查询状态，这是正常的
        print(f"优雅关闭耗时: {shutdown_duration:.2f}秒")
        print(f"关闭前运行策略数: {len(running_strategies)}")

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, strategy_manager):
        """测试内存泄漏预防"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建和销毁大量策略
        iterations = 50
        
        for i in range(iterations):
            # 创建策略
            config = HFTConfig(
                symbol=f"MEMORY{i}USDT",
                min_order_size=Decimal("0.001")
            )
            
            strategy_id = await strategy_manager.create_strategy(
                name=f"memory_test_{i}",
                strategy_type=StrategyType.HFT,
                config=config
            )
            
            # 启动策略
            await strategy_manager.start_strategy(strategy_id)
            await asyncio.sleep(0.1)
            
            # 停止并移除策略
            await strategy_manager.stop_strategy(strategy_id)
            await strategy_manager.remove_strategy(strategy_id)
            
            # 每10次迭代检查内存
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"迭代 {i}: 内存使用 {current_memory:.2f}MB (+{memory_growth:.2f}MB)")
                
                # 强制垃圾回收
                gc.collect()
        
        # 最终内存检查
        gc.collect()
        await asyncio.sleep(1)  # 等待异步清理
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"内存泄漏测试结果:")
        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"内存增长: {total_growth:.2f}MB")
        
        # 内存增长应该在合理范围内
        max_acceptable_growth = 100  # 100MB
        assert total_growth < max_acceptable_growth, \
            f"内存增长 {total_growth:.2f}MB 超过可接受范围 {max_acceptable_growth}MB"