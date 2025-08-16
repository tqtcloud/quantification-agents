"""系统集成测试"""

import pytest
import asyncio
import time
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.core.system_orchestrator import SystemOrchestrator, ComponentInfo, ComponentStatus
from src.core.config_manager import ConfigManager
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.alert_manager import AlertManager, AlertSeverity
from main import TradingSystem


class MockComponent:
    """模拟组件"""
    
    def __init__(self, name: str, fail_on_start: bool = False, fail_on_stop: bool = False):
        self.name = name
        self.fail_on_start = fail_on_start
        self.fail_on_stop = fail_on_stop
        self.started = False
        self.stopped = False
    
    async def start(self):
        if self.fail_on_start:
            raise Exception(f"组件 {self.name} 启动失败")
        self.started = True
        await asyncio.sleep(0.1)  # 模拟启动时间
    
    async def stop(self):
        if self.fail_on_stop:
            raise Exception(f"组件 {self.name} 停止失败")
        self.stopped = True
        await asyncio.sleep(0.1)  # 模拟停止时间
    
    def health_check(self):
        return self.started and not self.stopped


class TestSystemOrchestrator:
    """系统编排器测试"""
    
    @pytest.fixture
    def orchestrator(self):
        """创建测试用的系统编排器"""
        return SystemOrchestrator()
    
    @pytest.mark.asyncio
    async def test_component_registration(self, orchestrator):
        """测试组件注册"""
        component = MockComponent("test_component")
        
        orchestrator.register_component(
            "test_component",
            component,
            startup_func=component.start,
            shutdown_func=component.stop,
            health_check_func=component.health_check
        )
        
        assert "test_component" in orchestrator.components
        assert orchestrator.get_component("test_component") == component
        assert orchestrator.get_component_status("test_component") == ComponentStatus.UNINITIALIZED
    
    @pytest.mark.asyncio
    async def test_simple_startup_shutdown(self, orchestrator):
        """测试简单的启动和关闭"""
        component = MockComponent("test_component")
        
        orchestrator.register_component(
            "test_component",
            component,
            startup_func=component.start,
            shutdown_func=component.stop
        )
        
        # 启动
        await orchestrator.start_all()
        assert component.started
        assert orchestrator.get_component_status("test_component") == ComponentStatus.RUNNING
        
        # 关闭
        await orchestrator.stop_all()
        assert component.stopped
        assert orchestrator.get_component_status("test_component") == ComponentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_dependency_order(self, orchestrator):
        """测试依赖顺序"""
        component_a = MockComponent("component_a")
        component_b = MockComponent("component_b")
        component_c = MockComponent("component_c")
        
        start_order = []
        
        async def track_start(name):
            start_order.append(name)
            if name == "component_a":
                await component_a.start()
            elif name == "component_b":
                await component_b.start()
            elif name == "component_c":
                await component_c.start()
        
        # B 依赖 A，C 依赖 B
        orchestrator.register_component(
            "component_a", component_a,
            startup_func=lambda: track_start("component_a"),
            shutdown_func=component_a.stop
        )
        orchestrator.register_component(
            "component_b", component_b,
            dependencies=["component_a"],
            startup_func=lambda: track_start("component_b"),
            shutdown_func=component_b.stop
        )
        orchestrator.register_component(
            "component_c", component_c,
            dependencies=["component_b"],
            startup_func=lambda: track_start("component_c"),
            shutdown_func=component_c.stop
        )
        
        await orchestrator.start_all()
        
        # 验证启动顺序：A -> B -> C
        assert start_order == ["component_a", "component_b", "component_c"]
        
        await orchestrator.stop_all()
    
    @pytest.mark.asyncio
    async def test_startup_failure_rollback(self, orchestrator):
        """测试启动失败时的回滚"""
        component_a = MockComponent("component_a")
        component_b = MockComponent("component_b", fail_on_start=True)  # B 启动失败
        component_c = MockComponent("component_c")
        
        orchestrator.register_component("component_a", component_a, startup_func=component_a.start, shutdown_func=component_a.stop)
        orchestrator.register_component("component_b", component_b, dependencies=["component_a"], startup_func=component_b.start, shutdown_func=component_b.stop)
        orchestrator.register_component("component_c", component_c, dependencies=["component_b"], startup_func=component_c.start, shutdown_func=component_c.stop)
        
        # 启动应该失败
        with pytest.raises(RuntimeError):
            await orchestrator.start_all()
        
        # A 应该已启动，B 启动失败，C 不应启动
        assert component_a.started
        assert not component_b.started
        assert not component_c.started
        
        # 系统应该尝试清理
        assert component_a.stopped  # A 应该被停止
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, orchestrator):
        """测试循环依赖检测"""
        component_a = MockComponent("component_a")
        component_b = MockComponent("component_b")
        
        orchestrator.register_component("component_a", component_a, dependencies=["component_b"])
        orchestrator.register_component("component_b", component_b, dependencies=["component_a"])
        
        with pytest.raises(ValueError, match="循环依赖"):
            await orchestrator.start_all()


class TestConfigManager:
    """配置管理器测试"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """创建临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """创建测试用的配置管理器"""
        return ConfigManager(temp_config_dir)
    
    def test_config_initialization(self, config_manager):
        """测试配置初始化"""
        # 应该创建默认环境配置
        assert "development" in config_manager.environment_configs
        assert "testing" in config_manager.environment_configs
        assert "production" in config_manager.environment_configs
        
        # 默认环境应该是 development
        assert config_manager.current_environment == "development"
    
    def test_config_get_set(self, config_manager):
        """测试配置获取和设置"""
        # 测试获取配置
        trading_mode = config_manager.get("trading.mode")
        assert trading_mode is not None
        
        # 测试设置配置
        config_manager.set("trading.max_position_size", 2000.0)
        assert config_manager.get("trading.max_position_size") == 2000.0
    
    def test_environment_switching(self, config_manager):
        """测试环境切换"""
        # 切换到测试环境
        config_manager.switch_environment("testing")
        assert config_manager.current_environment == "testing"
        
        # 切换到不存在的环境应该失败
        with pytest.raises(ValueError):
            config_manager.switch_environment("nonexistent")
    
    def test_config_hot_reload(self, config_manager):
        """测试配置热更新"""
        callback_called = False
        new_config = None
        
        def config_callback(config):
            nonlocal callback_called, new_config
            callback_called = True
            new_config = config
        
        config_manager.add_change_callback(config_callback)
        
        # 设置新配置值
        config_manager.set("test.value", "new_value")
        
        # 验证回调被调用
        assert callback_called
        assert new_config is not None
        assert config_manager.get("test.value") == "new_value"


class TestSystemMonitor:
    """系统监控器测试"""
    
    @pytest.fixture
    def monitor(self):
        """创建测试用的系统监控器"""
        return SystemMonitor(metrics_retention_minutes=60)
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """测试监控生命周期"""
        assert not monitor._monitoring
        
        # 启动监控
        monitor.start_monitoring(interval_seconds=1)
        assert monitor._monitoring
        
        # 等待收集一些指标
        await asyncio.sleep(2)
        
        # 停止监控
        await monitor.stop_monitoring()
        assert not monitor._monitoring
    
    def test_metric_collection(self, monitor):
        """测试指标收集"""
        system_metrics = monitor._collect_system_metrics()
        app_metrics = monitor._collect_application_metrics()
        
        # 验证系统指标
        assert 0 <= system_metrics.cpu_usage <= 100
        assert 0 <= system_metrics.memory_usage <= 100
        assert system_metrics.memory_available >= 0
        assert 0 <= system_metrics.disk_usage <= 100
        assert system_metrics.disk_free >= 0
        
        # 验证应用指标
        assert app_metrics.trading_system_status in ["running", "stopped", "error"]
        assert app_metrics.active_agents >= 0
        assert app_metrics.active_strategies >= 0
    
    def test_health_score_calculation(self, monitor):
        """测试健康评分计算"""
        # 添加一些模拟指标
        from src.monitoring.system_monitor import SystemMetrics, ApplicationMetrics
        
        system_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_available=2.0,
            disk_usage=70.0,
            disk_free=10.0,
            network_io={},
            process_count=100
        )
        
        app_metrics = ApplicationMetrics(
            timestamp=datetime.now(),
            trading_system_status="running",
            active_agents=5,
            active_strategies=2,
            active_orders=10,
            total_positions=3,
            api_requests_per_minute=50,
            websocket_connections=8,
            database_connections=5,
            error_count=1,
            warning_count=2
        )
        
        monitor.system_metrics.append(system_metrics)
        monitor.app_metrics.append(app_metrics)
        
        health_score = monitor.get_system_health_score()
        assert 0 <= health_score <= 100


class TestAlertManager:
    """告警管理器测试"""
    
    @pytest.fixture
    def alert_manager(self):
        """创建测试用的告警管理器"""
        return AlertManager()
    
    @pytest.mark.asyncio
    async def test_manual_alert_creation(self, alert_manager):
        """测试手动创建告警"""
        alert_id = await alert_manager.create_manual_alert(
            title="测试告警",
            message="这是一个测试告警",
            severity=AlertSeverity.WARNING
        )
        
        assert alert_id in alert_manager.alerts
        alert = alert_manager.alerts[alert_id]
        assert alert.title == "测试告警"
        assert alert.severity == AlertSeverity.WARNING
    
    def test_alert_resolution(self, alert_manager):
        """测试告警解决"""
        # 手动添加一个告警
        from src.monitoring.alert_manager import Alert, AlertStatus
        
        alert = Alert(
            id="test_alert",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.WARNING,
            source="test",
            timestamp=datetime.now()
        )
        alert_manager.alerts["test_alert"] = alert
        
        # 解决告警
        success = alert_manager.resolve_alert("test_alert", "已修复")
        assert success
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolution_note == "已修复"
    
    @pytest.mark.asyncio
    async def test_alert_rule_triggering(self, alert_manager):
        """测试告警规则触发"""
        from src.monitoring.system_monitor import SystemMetrics, ApplicationMetrics
        
        # 创建会触发告警的指标
        system_metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=95.0,  # 高CPU使用率
            memory_usage=60.0,
            memory_available=2.0,
            disk_usage=70.0,
            disk_free=10.0,
            network_io={},
            process_count=100
        )
        
        app_metrics = ApplicationMetrics(
            timestamp=datetime.now(),
            trading_system_status="running",
            active_agents=5,
            active_strategies=2,
            active_orders=10,
            total_positions=3,
            api_requests_per_minute=50,
            websocket_connections=8,
            database_connections=5,
            error_count=1,
            warning_count=2
        )
        
        initial_alert_count = len(alert_manager.get_active_alerts())
        
        # 检查告警
        await alert_manager.check_alerts(system_metrics, app_metrics)
        
        # 应该触发CPU使用率告警
        new_alert_count = len(alert_manager.get_active_alerts())
        assert new_alert_count > initial_alert_count


class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_trading_system_lifecycle(self):
        """测试交易系统完整生命周期"""
        trading_system = TradingSystem()
        
        try:
            # 初始化组件
            await trading_system.initialize_components()
            assert trading_system.components_initialized
            
            # 启动系统
            await trading_system.start()
            assert trading_system.running
            
            # 验证组件状态
            status = trading_system.get_status()
            assert status["running"]
            assert "components" in status
            
            # 等待一段时间确保系统稳定运行
            await asyncio.sleep(2)
            
            # 停止系统
            await trading_system.stop()
            assert not trading_system.running
            
        except Exception as e:
            # 确保在测试失败时也能清理资源
            await trading_system.stop()
            raise
    
    @pytest.mark.asyncio
    async def test_trading_mode_switching(self):
        """测试不同交易模式切换"""
        from src.config import settings
        
        original_mode = settings.trading_mode
        
        try:
            # 测试各种交易模式
            for mode in ["paper", "backtest", "live"]:
                settings.trading_mode = mode
                trading_system = TradingSystem()
                
                await trading_system.initialize_components()
                await trading_system.start()
                
                status = trading_system.get_status()
                assert status["mode"] == mode
                
                await trading_system.stop()
                
        finally:
            settings.trading_mode = original_mode
    
    @pytest.mark.asyncio
    async def test_system_error_recovery(self):
        """测试系统错误恢复"""
        trading_system = TradingSystem()
        
        # 模拟组件启动失败
        with patch.object(trading_system, 'start_paper_trading', side_effect=Exception("模拟错误")):
            with pytest.raises(Exception):
                await trading_system.start()
            
            # 系统应该已经停止
            assert not trading_system.running
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting_integration(self):
        """测试监控和告警集成"""
        from src.monitoring.system_monitor import system_monitor
        from src.monitoring.alert_manager import alert_manager
        
        # 启动监控
        system_monitor.start_monitoring(interval_seconds=1)
        
        try:
            # 等待收集一些指标
            await asyncio.sleep(3)
            
            # 验证指标收集
            current_metrics = system_monitor.get_current_metrics()
            assert current_metrics["system"] is not None
            assert current_metrics["application"] is not None
            
            # 验证健康评分
            health_score = system_monitor.get_system_health_score()
            assert 0 <= health_score <= 100
            
            # 验证告警统计
            alert_stats = alert_manager.get_alert_stats()
            assert "active_count" in alert_stats
            assert "total_alerts_24h" in alert_stats
            
        finally:
            await system_monitor.stop_monitoring()
    
    @pytest.mark.asyncio 
    async def test_concurrent_operations(self):
        """测试并发操作"""
        trading_system = TradingSystem()
        
        async def start_stop_cycle():
            await trading_system.initialize_components()
            await trading_system.start()
            await asyncio.sleep(1)
            await trading_system.stop()
        
        # 测试多个启动停止周期不会造成冲突
        tasks = [start_stop_cycle() for _ in range(3)]
        
        # 只有第一个应该成功启动，其他的应该检测到已启动状态
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证没有严重错误
        for result in results:
            if isinstance(result, Exception):
                # 这里可能会有一些预期的错误，如"系统已在运行"
                assert "已在运行" in str(result) or "未在运行" in str(result)


class TestPerformanceAndStress:
    """性能和压力测试"""
    
    @pytest.mark.asyncio
    async def test_component_startup_performance(self):
        """测试组件启动性能"""
        orchestrator = SystemOrchestrator()
        
        # 注册大量组件
        components = []
        for i in range(50):
            component = MockComponent(f"component_{i}")
            components.append(component)
            orchestrator.register_component(
                f"component_{i}",
                component,
                startup_func=component.start,
                shutdown_func=component.stop
            )
        
        # 测量启动时间
        start_time = time.time()
        await orchestrator.start_all()
        startup_time = time.time() - start_time
        
        # 启动时间应该在合理范围内（这里设为10秒）
        assert startup_time < 10.0
        
        # 验证所有组件都已启动
        for component in components:
            assert component.started
        
        # 测量关闭时间
        start_time = time.time()
        await orchestrator.stop_all()
        shutdown_time = time.time() - start_time
        
        # 关闭时间也应该在合理范围内
        assert shutdown_time < 10.0
        
        # 验证所有组件都已停止
        for component in components:
            assert component.stopped
    
    @pytest.mark.asyncio
    async def test_monitoring_performance(self):
        """测试监控性能"""
        monitor = SystemMonitor(metrics_retention_minutes=60)
        
        # 启动高频监控
        monitor.start_monitoring(interval_seconds=0.1)
        
        try:
            # 运行一段时间
            await asyncio.sleep(2)
            
            # 验证指标收集效率
            assert len(monitor.system_metrics) > 10  # 应该收集到多个指标点
            assert len(monitor.app_metrics) > 10
            
            # 测试指标检索性能
            start_time = time.time()
            for _ in range(100):
                monitor.get_current_metrics()
                monitor.get_system_health_score()
            retrieval_time = time.time() - start_time
            
            # 100次检索应该很快完成
            assert retrieval_time < 1.0
            
        finally:
            await monitor.stop_monitoring()


# 运行测试的示例
if __name__ == "__main__":
    pytest.main([__file__, "-v"])