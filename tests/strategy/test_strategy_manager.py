"""
策略管理器测试用例
测试双策略管理和隔离系统的完整功能
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from decimal import Decimal

from src.strategy.strategy_manager import (
    StrategyManager, 
    StrategyConfig, 
    StrategyType, 
    StrategyStatus
)
from src.strategy.resource_allocator import ResourceAllocator, ResourceLimit
from src.strategy.strategy_monitor import StrategyMonitor, MonitoringLevel
from src.strategy.config_manager import StrategyConfigManager
from src.hft.hft_engine import HFTConfig
from src.agents.orchestrator import WorkflowConfig
from src.core.message_bus import MessageBus


class TestStrategyManager:
    """策略管理器测试类"""
    
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
    async def strategy_manager(self, message_bus):
        """策略管理器fixture"""
        manager = StrategyManager(message_bus)
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    def hft_config(self):
        """HFT配置fixture"""
        return StrategyConfig(
            strategy_id="test_hft",
            strategy_type=StrategyType.HFT,
            name="测试HFT策略",
            description="用于测试的高频交易策略",
            max_memory_mb=2048,
            max_cpu_percent=50.0,
            max_network_connections=200,
            hft_config=HFTConfig(
                max_orderbook_levels=50,
                update_interval_ms=1.0,
                latency_target_ms=10.0
            )
        )
    
    @pytest.fixture
    def ai_config(self):
        """AI Agent配置fixture"""
        return StrategyConfig(
            strategy_id="test_ai",
            strategy_type=StrategyType.AI_AGENT,
            name="测试AI策略",
            description="用于测试的AI智能策略",
            max_memory_mb=1024,
            max_cpu_percent=25.0,
            max_network_connections=50,
            workflow_config=WorkflowConfig(
                max_parallel_agents=6,
                timeout_seconds=300
            )
        )
    
    async def test_manager_initialization(self, strategy_manager):
        """测试管理器初始化"""
        assert strategy_manager._is_running
        assert strategy_manager._resource_allocator is not None
        assert strategy_manager._strategy_monitor is not None
        assert len(strategy_manager.strategies) == 0
    
    async def test_register_hft_strategy(self, strategy_manager, hft_config):
        """测试注册HFT策略"""
        # 注册策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        assert strategy_id == "test_hft"
        
        # 验证策略已注册
        assert strategy_id in strategy_manager.strategies
        assert strategy_id in strategy_manager._strategy_types[StrategyType.HFT]
        
        # 验证策略状态
        instance = strategy_manager.strategies[strategy_id]
        assert instance.config == hft_config
        assert instance.metrics.strategy_id == strategy_id
        assert instance.metrics.strategy_type == StrategyType.HFT
        assert instance.metrics.status == StrategyStatus.IDLE
    
    async def test_register_ai_strategy(self, strategy_manager, ai_config):
        """测试注册AI策略"""
        # 注册策略
        strategy_id = await strategy_manager.register_strategy(ai_config)
        assert strategy_id == "test_ai"
        
        # 验证策略已注册
        assert strategy_id in strategy_manager.strategies
        assert strategy_id in strategy_manager._strategy_types[StrategyType.AI_AGENT]
        
        # 验证策略状态
        instance = strategy_manager.strategies[strategy_id]
        assert instance.config == ai_config
        assert instance.metrics.strategy_type == StrategyType.AI_AGENT
    
    async def test_start_hft_strategy(self, strategy_manager, hft_config):
        """测试启动HFT策略"""
        # 注册并启动策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        success = await strategy_manager.start_strategy(strategy_id)
        
        assert success
        
        # 验证策略状态
        instance = strategy_manager.strategies[strategy_id]
        assert instance.metrics.status == StrategyStatus.RUNNING
        assert instance.engine is not None
        assert instance.task is not None
        assert instance.health_check_task is not None
        
        # 清理
        await strategy_manager.stop_strategy(strategy_id)
    
    async def test_start_ai_strategy(self, strategy_manager, ai_config):
        """测试启动AI策略"""
        # 注册并启动策略
        strategy_id = await strategy_manager.register_strategy(ai_config)
        success = await strategy_manager.start_strategy(strategy_id)
        
        assert success
        
        # 验证策略状态
        instance = strategy_manager.strategies[strategy_id]
        assert instance.metrics.status == StrategyStatus.RUNNING
        assert instance.engine is not None
        
        # 清理
        await strategy_manager.stop_strategy(strategy_id)
    
    async def test_stop_strategy(self, strategy_manager, hft_config):
        """测试停止策略"""
        # 注册并启动策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_manager.start_strategy(strategy_id)
        
        # 停止策略
        success = await strategy_manager.stop_strategy(strategy_id)
        assert success
        
        # 验证策略状态
        instance = strategy_manager.strategies[strategy_id]
        assert instance.metrics.status == StrategyStatus.STOPPED
    
    async def test_pause_resume_strategy(self, strategy_manager, hft_config):
        """测试暂停和恢复策略"""
        # 注册并启动策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_manager.start_strategy(strategy_id)
        
        # 暂停策略
        success = await strategy_manager.pause_strategy(strategy_id)
        assert success
        
        instance = strategy_manager.strategies[strategy_id]
        assert instance.metrics.status == StrategyStatus.PAUSED
        
        # 恢复策略
        success = await strategy_manager.resume_strategy(strategy_id)
        assert success
        assert instance.metrics.status == StrategyStatus.RUNNING
        
        # 清理
        await strategy_manager.stop_strategy(strategy_id)
    
    async def test_restart_strategy(self, strategy_manager, hft_config):
        """测试重启策略"""
        # 注册并启动策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_manager.start_strategy(strategy_id)
        
        # 记录重启前的计数
        instance = strategy_manager.strategies[strategy_id]
        initial_restart_count = instance.metrics.restart_count
        
        # 重启策略
        success = await strategy_manager.restart_strategy(strategy_id)
        assert success
        
        # 验证重启计数增加
        assert instance.metrics.restart_count == initial_restart_count + 1
        assert instance.metrics.status == StrategyStatus.RUNNING
        
        # 清理
        await strategy_manager.stop_strategy(strategy_id)
    
    async def test_unregister_strategy(self, strategy_manager, hft_config):
        """测试注销策略"""
        # 注册策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        
        # 注销策略
        success = await strategy_manager.unregister_strategy(strategy_id)
        assert success
        
        # 验证策略已被移除
        assert strategy_id not in strategy_manager.strategies
        assert strategy_id not in strategy_manager._strategy_types[StrategyType.HFT]
    
    async def test_dual_strategy_isolation(self, strategy_manager, hft_config, ai_config):
        """测试双策略隔离"""
        # 注册两种类型的策略
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        
        # 启动两个策略
        await strategy_manager.start_strategy(hft_id)
        await strategy_manager.start_strategy(ai_id)
        
        # 验证隔离组
        resource_allocator = strategy_manager._resource_allocator
        isolation_groups = resource_allocator.get_isolation_groups()
        
        assert hft_id in isolation_groups['hft']
        assert ai_id in isolation_groups['ai_agent']
        
        # 验证资源分配
        hft_status = await resource_allocator.get_resource_status(hft_id)
        ai_status = await resource_allocator.get_resource_status(ai_id)
        
        assert hft_status['isolation_group'] == 'hft'
        assert ai_status['isolation_group'] == 'ai_agent'
        
        # 清理
        await strategy_manager.stop_strategy(hft_id)
        await strategy_manager.stop_strategy(ai_id)
    
    async def test_get_strategy_status(self, strategy_manager, hft_config):
        """测试获取策略状态"""
        strategy_id = await strategy_manager.register_strategy(hft_config)
        
        status = strategy_manager.get_strategy_status(strategy_id)
        assert status is not None
        assert status['strategy_id'] == strategy_id
        assert status['strategy_type'] == StrategyType.HFT.value
        assert status['status'] == StrategyStatus.IDLE.value
        assert 'metrics' in status
        assert 'config' in status
    
    async def test_list_strategies(self, strategy_manager, hft_config, ai_config):
        """测试列出策略"""
        # 注册两个策略
        await strategy_manager.register_strategy(hft_config)
        await strategy_manager.register_strategy(ai_config)
        
        # 列出所有策略
        all_strategies = strategy_manager.list_strategies()
        assert len(all_strategies) == 2
        
        # 按类型列出
        hft_strategies = strategy_manager.list_strategies(StrategyType.HFT)
        ai_strategies = strategy_manager.list_strategies(StrategyType.AI_AGENT)
        
        assert len(hft_strategies) == 1
        assert len(ai_strategies) == 1
        assert hft_strategies[0]['strategy_type'] == 'hft'
        assert ai_strategies[0]['strategy_type'] == 'ai_agent'
    
    async def test_get_system_status(self, strategy_manager, hft_config, ai_config):
        """测试获取系统状态"""
        # 注册并启动策略
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        await strategy_manager.start_strategy(hft_id)
        await strategy_manager.start_strategy(ai_id)
        
        # 获取系统状态
        status = strategy_manager.get_system_status()
        
        assert status['manager_status'] == 'running'
        assert status['total_strategies'] == 2
        assert status['running_strategies'] == 2
        assert status['strategies_by_type']['hft'] == 1
        assert status['strategies_by_type']['ai_agent'] == 1
        assert 'system_metrics' in status
        
        # 清理
        await strategy_manager.stop_strategy(hft_id)
        await strategy_manager.stop_strategy(ai_id)
    
    async def test_callback_registration(self, strategy_manager, hft_config):
        """测试回调注册"""
        strategy_id = await strategy_manager.register_strategy(hft_config)
        
        # 注册回调
        callback_called = False
        
        def test_callback(instance, **kwargs):
            nonlocal callback_called
            callback_called = True
        
        strategy_manager.register_callback(strategy_id, 'on_start', test_callback)
        
        # 启动策略触发回调
        await strategy_manager.start_strategy(strategy_id)
        
        # 等待回调执行
        await asyncio.sleep(0.1)
        assert callback_called
        
        # 清理
        await strategy_manager.stop_strategy(strategy_id)
    
    async def test_error_handling(self, strategy_manager):
        """测试错误处理"""
        # 测试启动不存在的策略
        success = await strategy_manager.start_strategy("non_existent")
        assert not success
        
        # 测试停止不存在的策略
        success = await strategy_manager.stop_strategy("non_existent")
        assert not success
        
        # 测试获取不存在策略的状态
        status = strategy_manager.get_strategy_status("non_existent")
        assert status is None


class TestResourceAllocator:
    """资源分配器测试类"""
    
    @pytest.fixture
    async def resource_allocator(self):
        """资源分配器fixture"""
        allocator = ResourceAllocator()
        await allocator.initialize()
        yield allocator
        await allocator.shutdown()
    
    async def test_allocator_initialization(self, resource_allocator):
        """测试分配器初始化"""
        assert resource_allocator.system_resources is not None
        assert resource_allocator._is_monitoring
        assert len(resource_allocator.allocations) == 0
    
    async def test_resource_allocation(self, resource_allocator, hft_config):
        """测试资源分配"""
        # 分配资源
        success = await resource_allocator.allocate_resources(hft_config)
        assert success
        
        # 验证分配记录
        assert hft_config.strategy_id in resource_allocator.allocations
        allocation = resource_allocator.allocations[hft_config.strategy_id]
        
        assert allocation.strategy_id == hft_config.strategy_id
        assert allocation.resource_limit.memory_mb == hft_config.max_memory_mb
        assert allocation.resource_limit.cpu_percent == hft_config.max_cpu_percent
        assert allocation.isolation_group == 'hft'
    
    async def test_resource_release(self, resource_allocator, hft_config):
        """测试资源释放"""
        # 先分配
        await resource_allocator.allocate_resources(hft_config)
        
        # 释放资源
        success = await resource_allocator.release_resources(hft_config.strategy_id)
        assert success
        
        # 验证资源已释放
        assert hft_config.strategy_id not in resource_allocator.allocations
    
    async def test_resource_availability_check(self, resource_allocator, hft_config):
        """测试资源可用性检查"""
        # 检查资源可用性
        available = await resource_allocator.check_resources_available(hft_config)
        assert available  # 初始状态应该有足够资源
    
    async def test_resource_monitoring(self, resource_allocator, hft_config):
        """测试资源监控"""
        # 分配资源
        await resource_allocator.allocate_resources(hft_config)
        
        # 模拟进程ID
        process_ids = {12345, 12346}
        
        # 更新资源使用
        success = await resource_allocator.update_resource_usage(
            hft_config.strategy_id, 
            process_ids
        )
        assert success
        
        # 验证进程ID已记录
        allocation = resource_allocator.allocations[hft_config.strategy_id]
        assert allocation.process_ids == process_ids
    
    async def test_get_resource_status(self, resource_allocator, hft_config):
        """测试获取资源状态"""
        # 分配资源
        await resource_allocator.allocate_resources(hft_config)
        
        # 获取特定策略资源状态
        status = await resource_allocator.get_resource_status(hft_config.strategy_id)
        assert status['strategy_id'] == hft_config.strategy_id
        assert 'resource_limit' in status
        assert 'current_usage' in status
        assert 'utilization_rate' in status
        
        # 获取系统整体状态
        system_status = await resource_allocator.get_resource_status()
        assert 'system_resources' in system_status
        assert 'allocated_resources' in system_status
        assert 'total_allocations' in system_status
    
    async def test_isolation_groups(self, resource_allocator, hft_config, ai_config):
        """测试隔离组"""
        # 分配两种类型策略的资源
        await resource_allocator.allocate_resources(hft_config)
        await resource_allocator.allocate_resources(ai_config)
        
        # 获取隔离组信息
        isolation_groups = resource_allocator.get_isolation_groups()
        
        assert hft_config.strategy_id in isolation_groups['hft']
        assert ai_config.strategy_id in isolation_groups['ai_agent']


class TestStrategyMonitor:
    """策略监控器测试类"""
    
    @pytest.fixture
    async def strategy_manager(self):
        """策略管理器fixture"""
        manager = StrategyManager()
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    async def strategy_monitor(self, strategy_manager):
        """策略监控器fixture"""
        monitor = StrategyMonitor(strategy_manager, MonitoringLevel.DETAILED)
        await monitor.initialize()
        yield monitor
        await monitor.shutdown()
    
    async def test_monitor_initialization(self, strategy_monitor):
        """测试监控器初始化"""
        assert strategy_monitor._is_monitoring
        assert len(strategy_monitor.alert_rules) > 0  # 应该有默认告警规则
        assert len(strategy_monitor.metrics) == 0
    
    async def test_add_strategy_monitoring(self, strategy_manager, strategy_monitor, hft_config):
        """测试添加策略监控"""
        # 注册策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        
        # 添加监控
        await strategy_monitor.add_strategy_monitoring(strategy_id)
        
        # 验证监控已添加
        assert strategy_id in strategy_monitor.metrics
        metrics = strategy_monitor.metrics[strategy_id]
        assert metrics.strategy_id == strategy_id
        assert metrics.strategy_type == StrategyType.HFT.value
    
    async def test_update_metrics(self, strategy_manager, strategy_monitor, hft_config):
        """测试更新指标"""
        # 注册策略并添加监控
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_monitor.add_strategy_monitoring(strategy_id)
        
        # 更新各种指标
        strategy_monitor.update_metric(strategy_id, 'cpu_usage', 45.5)
        strategy_monitor.update_metric(strategy_id, 'memory_usage', 1200.0)
        strategy_monitor.update_metric(strategy_id, 'total_trades', 100)
        strategy_monitor.update_metric(strategy_id, 'successful_trades', 95)
        
        # 验证指标已更新
        metrics = strategy_monitor.metrics[strategy_id]
        assert list(metrics.cpu_usage)[-1] == 45.5
        assert list(metrics.memory_usage)[-1] == 1200.0
        assert metrics.total_trades == 100
        assert metrics.successful_trades == 95
        assert metrics.get_success_rate() == 0.95
    
    async def test_get_strategy_metrics(self, strategy_manager, strategy_monitor, hft_config):
        """测试获取策略指标"""
        # 注册策略并添加监控
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_monitor.add_strategy_monitoring(strategy_id)
        
        # 更新一些指标
        strategy_monitor.update_metric(strategy_id, 'cpu_usage', 30.0)
        strategy_monitor.update_metric(strategy_id, 'total_trades', 50)
        
        # 获取指标
        metrics = strategy_monitor.get_strategy_metrics(strategy_id)
        assert metrics is not None
        assert metrics['strategy_id'] == strategy_id
        assert metrics['current_cpu_usage'] == 30.0
        assert metrics['total_trades'] == 50
        assert 'custom_metrics' in metrics
    
    async def test_get_system_metrics(self, strategy_manager, strategy_monitor, hft_config, ai_config):
        """测试获取系统指标"""
        # 注册两个策略并添加监控
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        await strategy_monitor.add_strategy_monitoring(hft_id)
        await strategy_monitor.add_strategy_monitoring(ai_id)
        
        # 获取系统指标
        metrics = strategy_monitor.get_system_metrics()
        
        assert metrics['total_strategies'] == 2
        assert metrics['monitoring_level'] == MonitoringLevel.DETAILED.value
        assert 'total_trades' in metrics
        assert 'total_errors' in metrics
    
    async def test_alert_system(self, strategy_manager, strategy_monitor, hft_config):
        """测试告警系统"""
        # 注册策略并添加监控
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_monitor.add_strategy_monitoring(strategy_id)
        
        # 更新指标触发告警（CPU使用率过高）
        strategy_monitor.update_metric(strategy_id, 'cpu_usage', 85.0)
        
        # 等待告警处理
        await asyncio.sleep(0.1)
        
        # 检查是否有活跃告警
        active_alerts = strategy_monitor.get_active_alerts(strategy_id)
        
        # 可能会有高CPU使用率告警
        if active_alerts:
            alert = active_alerts[0]
            assert alert['strategy_id'] == strategy_id
            assert 'cpu_usage' in alert['message']
    
    async def test_remove_strategy_monitoring(self, strategy_manager, strategy_monitor, hft_config):
        """测试移除策略监控"""
        # 注册策略并添加监控
        strategy_id = await strategy_manager.register_strategy(hft_config)
        await strategy_monitor.add_strategy_monitoring(strategy_id)
        
        # 验证监控已添加
        assert strategy_id in strategy_monitor.metrics
        
        # 移除监控
        await strategy_monitor.remove_strategy_monitoring(strategy_id)
        
        # 验证监控已移除
        assert strategy_id not in strategy_monitor.metrics


class TestConfigManager:
    """配置管理器测试类"""
    
    @pytest.fixture
    async def temp_config_dir(self):
        """临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def config_manager(self, temp_config_dir):
        """配置管理器fixture"""
        manager = StrategyConfigManager(temp_config_dir)
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    async def test_config_manager_initialization(self, config_manager, temp_config_dir):
        """测试配置管理器初始化"""
        config_dir = Path(temp_config_dir)
        
        # 验证目录结构
        assert (config_dir / "hft").exists()
        assert (config_dir / "ai_agent").exists()
        assert (config_dir / "templates").exists()
        assert (config_dir / "backups").exists()
        
        # 验证模板文件
        assert (config_dir / "templates" / "hft_template.yaml").exists()
        assert (config_dir / "templates" / "ai_agent_template.yaml").exists()
    
    async def test_create_default_config(self, config_manager):
        """测试创建默认配置"""
        # 创建HFT默认配置
        hft_config = await config_manager.create_default_config("test_hft", StrategyType.HFT)
        assert hft_config.strategy_id == "test_hft"
        assert hft_config.strategy_type == StrategyType.HFT
        assert hft_config.hft_config is not None
        
        # 创建AI Agent默认配置
        ai_config = await config_manager.create_default_config("test_ai", StrategyType.AI_AGENT)
        assert ai_config.strategy_id == "test_ai"
        assert ai_config.strategy_type == StrategyType.AI_AGENT
        assert ai_config.workflow_config is not None
    
    async def test_save_load_config(self, config_manager, hft_config):
        """测试保存和加载配置"""
        # 保存配置
        success = await config_manager.save_config(hft_config)
        assert success
        
        # 加载配置
        loaded_config = await config_manager.load_config(hft_config.strategy_id)
        assert loaded_config is not None
        assert loaded_config.strategy_id == hft_config.strategy_id
        assert loaded_config.name == hft_config.name
        assert loaded_config.strategy_type == hft_config.strategy_type
    
    async def test_list_configs(self, config_manager, hft_config, ai_config):
        """测试列出配置"""
        # 保存两个配置
        await config_manager.save_config(hft_config)
        await config_manager.save_config(ai_config)
        
        # 列出所有配置
        all_configs = config_manager.list_configs()
        assert len(all_configs) == 2
        
        # 按类型列出
        hft_configs = config_manager.list_configs(StrategyType.HFT)
        ai_configs = config_manager.list_configs(StrategyType.AI_AGENT)
        
        assert len(hft_configs) == 1
        assert len(ai_configs) == 1
    
    async def test_update_config(self, config_manager, hft_config):
        """测试更新配置"""
        # 保存原始配置
        await config_manager.save_config(hft_config)
        
        # 更新配置
        updates = {
            'name': '更新后的HFT策略',
            'max_memory_mb': 4096
        }
        success = await config_manager.update_config(hft_config.strategy_id, updates)
        assert success
        
        # 验证更新
        updated_config = config_manager.get_config(hft_config.strategy_id)
        assert updated_config.name == '更新后的HFT策略'
        assert updated_config.max_memory_mb == 4096
    
    async def test_delete_config(self, config_manager, hft_config):
        """测试删除配置"""
        # 保存配置
        await config_manager.save_config(hft_config)
        
        # 验证配置存在
        assert config_manager.get_config(hft_config.strategy_id) is not None
        
        # 删除配置
        success = await config_manager.delete_config(hft_config.strategy_id)
        assert success
        
        # 验证配置已删除
        assert config_manager.get_config(hft_config.strategy_id) is None


class TestIntegration:
    """集成测试类"""
    
    @pytest.fixture
    async def temp_config_dir(self):
        """临时配置目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def full_system(self, temp_config_dir):
        """完整系统fixture"""
        # 创建消息总线
        message_bus = MessageBus()
        await message_bus.initialize()
        
        # 创建策略管理器
        strategy_manager = StrategyManager(message_bus)
        await strategy_manager.initialize()
        
        # 创建配置管理器
        config_manager = StrategyConfigManager(temp_config_dir)
        await config_manager.initialize()
        
        yield {
            'message_bus': message_bus,
            'strategy_manager': strategy_manager,
            'config_manager': config_manager
        }
        
        # 清理
        await strategy_manager.shutdown()
        await config_manager.shutdown()
        await message_bus.shutdown()
    
    async def test_end_to_end_workflow(self, full_system):
        """测试端到端工作流"""
        strategy_manager = full_system['strategy_manager']
        config_manager = full_system['config_manager']
        
        # 1. 创建配置
        hft_config = await config_manager.create_default_config("e2e_hft", StrategyType.HFT)
        ai_config = await config_manager.create_default_config("e2e_ai", StrategyType.AI_AGENT)
        
        # 2. 注册策略
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        
        assert hft_id == "e2e_hft"
        assert ai_id == "e2e_ai"
        
        # 3. 启动策略
        hft_success = await strategy_manager.start_strategy(hft_id)
        ai_success = await strategy_manager.start_strategy(ai_id)
        
        assert hft_success
        assert ai_success
        
        # 4. 验证策略隔离
        resource_allocator = strategy_manager._resource_allocator
        isolation_groups = resource_allocator.get_isolation_groups()
        
        assert hft_id in isolation_groups['hft']
        assert ai_id in isolation_groups['ai_agent']
        
        # 5. 检查系统状态
        system_status = strategy_manager.get_system_status()
        assert system_status['running_strategies'] == 2
        
        # 6. 停止策略
        await strategy_manager.stop_strategy(hft_id)
        await strategy_manager.stop_strategy(ai_id)
        
        # 7. 注销策略
        await strategy_manager.unregister_strategy(hft_id)
        await strategy_manager.unregister_strategy(ai_id)
        
        # 8. 验证清理
        assert len(strategy_manager.strategies) == 0
    
    async def test_resource_isolation_verification(self, full_system):
        """测试资源隔离验证"""
        strategy_manager = full_system['strategy_manager']
        config_manager = full_system['config_manager']
        
        # 创建具有不同资源需求的策略
        hft_config = await config_manager.create_default_config("isolation_hft", StrategyType.HFT)
        hft_config.max_memory_mb = 2048
        hft_config.max_cpu_percent = 50.0
        
        ai_config = await config_manager.create_default_config("isolation_ai", StrategyType.AI_AGENT)
        ai_config.max_memory_mb = 1024
        ai_config.max_cpu_percent = 25.0
        
        # 注册并启动策略
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        
        await strategy_manager.start_strategy(hft_id)
        await strategy_manager.start_strategy(ai_id)
        
        # 验证资源分配
        resource_allocator = strategy_manager._resource_allocator
        
        hft_status = await resource_allocator.get_resource_status(hft_id)
        ai_status = await resource_allocator.get_resource_status(ai_id)
        
        # 验证不同的隔离组
        assert hft_status['isolation_group'] != ai_status['isolation_group']
        
        # 验证资源限制
        assert hft_status['resource_limit']['memory_mb'] == 2048
        assert ai_status['resource_limit']['memory_mb'] == 1024
        
        # 清理
        await strategy_manager.stop_strategy(hft_id)
        await strategy_manager.stop_strategy(ai_id)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])