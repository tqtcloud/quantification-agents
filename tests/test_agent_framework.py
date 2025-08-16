"""
Agent框架测试
测试Agent基类扩展功能、注册系统和通信机制
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.agents.base import (
    BaseAgent, AgentConfig, AgentState, AgentPerformanceMetrics,
    AnalysisAgent, ExecutionAgent, RiskAgent
)
from src.agents.registry import (
    AgentRegistry, AgentHealthInfo, HealthStatus, 
    AgentRegistration, create_agent_registry
)
from src.core.message_bus import MessageBus, Message
from src.core.models import TradingState, Signal


class MockTestAgent(BaseAgent):
    """测试用Agent实现"""
    
    def __init__(self, config: AgentConfig, message_bus=None):
        super().__init__(config, message_bus)
        self.analysis_count = 0
        self.test_data = {}
    
    async def _initialize(self):
        """测试初始化"""
        await asyncio.sleep(0.01)  # 模拟初始化工作
        self.test_data["initialized"] = True
    
    async def _shutdown(self):
        """测试关闭"""
        await asyncio.sleep(0.01)  # 模拟关闭工作
        self.test_data["shutdown"] = True
    
    async def analyze(self, state: TradingState):
        """测试分析"""
        self.analysis_count += 1
        await asyncio.sleep(0.001)  # 模拟分析工作
        
        # 返回测试信号
        return [Signal(
            source=self.name,
            symbol="BTCUSDT",
            action="BUY",
            strength=0.5,
            confidence=0.8,
            reason="Test signal"
        )]


class TestAgentState:
    """Agent状态测试"""
    
    def test_agent_state_enum(self):
        """测试状态枚举"""
        assert AgentState.CREATED.value == "created"
        assert AgentState.RUNNING.value == "running"
        assert AgentState.ERROR.value == "error"


class TestAgentPerformanceMetrics:
    """Agent性能指标测试"""
    
    def test_metrics_creation(self):
        """测试指标创建"""
        metrics = AgentPerformanceMetrics()
        assert metrics.signals_generated == 0
        assert metrics.error_count == 0
        assert metrics.avg_processing_time == 0.0
    
    def test_update_processing_time(self):
        """测试处理时间更新"""
        metrics = AgentPerformanceMetrics()
        
        # 添加处理时间
        metrics.update_processing_time(0.005)
        metrics.update_processing_time(0.010)
        metrics.update_processing_time(0.003)
        
        assert len(metrics.processing_times) == 3
        assert metrics.avg_processing_time > 0
        assert metrics.max_processing_time == 0.010
        assert metrics.min_processing_time == 0.003
    
    def test_update_signals_per_second(self):
        """测试每秒信号数更新"""
        metrics = AgentPerformanceMetrics()
        metrics.signals_generated = 100
        
        # 模拟运行了10秒
        metrics.start_time = time.time() - 10
        metrics.update_signals_per_second()
        
        assert abs(metrics.signals_per_second - 10.0) < 0.01  # 允许小的浮点误差
        assert metrics.uptime_seconds >= 10.0


class TestAgentConfig:
    """Agent配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AgentConfig(name="test_agent")
        
        assert config.name == "test_agent"
        assert config.enabled is True
        assert config.priority == 0
        assert config.max_processing_time == 5.0
        assert config.heartbeat_interval == 30.0
        assert config.config_update_enabled is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AgentConfig(
            name="custom_agent",
            enabled=False,
            priority=50,
            max_processing_time=10.0,
            heartbeat_interval=60.0,
            parameters={"param1": "value1"}
        )
        
        assert config.name == "custom_agent"
        assert config.enabled is False
        assert config.priority == 50
        assert config.max_processing_time == 10.0
        assert config.parameters["param1"] == "value1"


class TestBaseAgent:
    """BaseAgent测试"""
    
    @pytest.fixture
    async def mock_message_bus(self):
        """模拟消息总线"""
        bus = MagicMock(spec=MessageBus)
        bus.create_publisher = MagicMock()
        bus.create_subscriber = MagicMock()
        bus.publish = MagicMock()
        
        # 模拟订阅者
        subscriber = MagicMock()
        subscriber.subscribe = AsyncMock()
        bus.create_subscriber.return_value = subscriber
        
        return bus
    
    @pytest.fixture
    async def test_agent(self, mock_message_bus):
        """创建测试Agent"""
        config = AgentConfig(
            name="test_agent",
            heartbeat_interval=1.0  # 缩短心跳间隔用于测试
        )
        agent = MockTestAgent(config, mock_message_bus)
        yield agent
        
        # 清理
        if agent._initialized:
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, test_agent):
        """测试Agent生命周期"""
        # 初始状态
        assert test_agent.get_state() == AgentState.CREATED
        assert not test_agent._initialized
        
        # 初始化
        await test_agent.initialize()
        assert test_agent._initialized
        assert test_agent.get_state() == AgentState.RUNNING
        assert test_agent.test_data.get("initialized") is True
        
        # 关闭
        await test_agent.shutdown()
        assert not test_agent._initialized
        assert test_agent.get_state() == AgentState.SHUTDOWN
        assert test_agent.test_data.get("shutdown") is True
    
    @pytest.mark.asyncio
    async def test_state_management(self, test_agent):
        """测试状态管理"""
        # 测试状态变化
        await test_agent._set_state(AgentState.INITIALIZING, "测试初始化")
        assert test_agent.get_state() == AgentState.INITIALIZING
        
        # 检查状态历史
        history = test_agent.get_state_history(limit=1)
        assert len(history) == 1
        assert history[0]["state"] == "initializing"
        assert history[0]["reason"] == "测试初始化"
    
    @pytest.mark.asyncio
    async def test_analyze_with_monitoring(self, test_agent):
        """测试带监控的分析方法"""
        await test_agent.initialize()
        
        # 创建模拟交易状态
        trading_state = MagicMock(spec=TradingState)
        
        # 执行分析
        signals = await test_agent.analyze_with_monitoring(trading_state)
        
        # 验证结果
        assert len(signals) == 1
        assert signals[0].source == "test_agent"
        assert test_agent.analysis_count == 1
        
        # 验证性能指标更新
        metrics = test_agent.get_performance_metrics()
        assert metrics.signals_generated == 1
        assert metrics.avg_processing_time > 0
    
    @pytest.mark.asyncio
    async def test_config_update(self, test_agent):
        """测试配置更新"""
        await test_agent.initialize()
        
        # 更新配置
        await test_agent.update_config(
            enabled=False,
            priority=100,
            max_processing_time=2.0
        )
        
        # 验证配置更新
        assert test_agent.enabled is False
        assert test_agent.priority == 100
        assert test_agent.config.max_processing_time == 2.0
    
    @pytest.mark.asyncio
    async def test_message_communication(self, test_agent):
        """测试消息通信"""
        await test_agent.initialize()
        
        # 测试发送消息
        await test_agent.send_message_to_agent("target_agent", "test_message", {"data": "test"})
        
        # 验证消息发布调用
        test_agent.message_bus.publish.assert_called()
        
        # 测试广播消息
        await test_agent.broadcast_message("broadcast_test", {"info": "broadcast"})
        
        # 验证广播消息发布
        assert test_agent.message_bus.publish.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_message_handler_registration(self, test_agent):
        """测试消息处理器注册"""
        handler_called = False
        
        async def test_handler(from_agent, data):
            nonlocal handler_called
            handler_called = True
        
        # 注册处理器
        test_agent.register_message_handler("test_message", test_handler)
        
        # 模拟接收消息
        message = Message(
            topic="agent.test_agent.message",
            data={
                "from_agent": "sender",
                "message_type": "test_message",
                "data": {"test": "data"}
            }
        )
        
        await test_agent._handle_agent_message(message)
        
        # 验证处理器被调用
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, test_agent):
        """测试性能监控"""
        await test_agent.initialize()
        
        # 等待一小段时间让监控任务运行
        await asyncio.sleep(0.1)
        
        # 验证监控任务存在
        assert test_agent._monitoring_task is not None
        assert not test_agent._monitoring_task.done()
        
        # 验证心跳任务存在
        assert test_agent._heartbeat_task is not None
        assert not test_agent._heartbeat_task.done()


class TestAgentHealthInfo:
    """Agent健康信息测试"""
    
    def test_health_info_creation(self):
        """测试健康信息创建"""
        health = AgentHealthInfo(agent_name="test_agent")
        
        assert health.agent_name == "test_agent"
        assert health.status == HealthStatus.UNKNOWN
        assert health.error_count == 0
        assert health.consecutive_failures == 0
    
    def test_is_alive_check(self):
        """测试存活检查"""
        health = AgentHealthInfo(agent_name="test_agent")
        
        # 新创建的健康信息应该是死的（没有心跳）
        assert not health.is_alive
        
        # 设置最近的心跳
        health.last_heartbeat = time.time()
        assert health.is_alive
        
        # 设置过期的心跳
        health.last_heartbeat = time.time() - 70  # 70秒前
        assert not health.is_alive
    
    def test_heartbeat_age(self):
        """测试心跳年龄计算"""
        health = AgentHealthInfo(agent_name="test_agent")
        health.last_heartbeat = time.time() - 30  # 30秒前
        
        age = health.heartbeat_age_seconds
        assert 29 <= age <= 31  # 允许一些时间误差


class TestAgentRegistry:
    """Agent注册表测试"""
    
    @pytest.fixture
    async def mock_message_bus(self):
        """模拟消息总线"""
        bus = MagicMock(spec=MessageBus)
        bus.create_publisher = MagicMock()
        bus.create_subscriber = MagicMock()
        bus.publish = MagicMock()
        
        # 模拟订阅者
        subscriber = MagicMock()
        subscriber.subscribe = AsyncMock()
        bus.create_subscriber.return_value = subscriber
        
        return bus
    
    @pytest.fixture
    async def registry(self, mock_message_bus):
        """创建注册表实例"""
        registry = AgentRegistry(
            message_bus=mock_message_bus,
            health_check_interval=1.0  # 缩短检查间隔用于测试
        )
        await registry.initialize()
        yield registry
        await registry.shutdown()
    
    @pytest.fixture
    def test_agents(self):
        """创建测试Agent"""
        agents = []
        for i in range(3):
            config = AgentConfig(name=f"test_agent_{i}")
            agent = MockTestAgent(config)
            agents.append(agent)
        return agents
    
    @pytest.mark.asyncio
    async def test_registry_initialization(self, mock_message_bus):
        """测试注册表初始化"""
        registry = AgentRegistry(message_bus=mock_message_bus)
        await registry.initialize()
        
        # 验证初始化
        assert registry._running
        assert registry._health_check_task is not None
        
        await registry.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, registry, test_agents):
        """测试Agent注册"""
        agent = test_agents[0]
        
        # 注册Agent
        success = await registry.register_agent(agent, group="test_group", tags={"test", "analysis"})
        assert success
        
        # 验证注册
        registered_agent = registry.get_agent(agent.name)
        assert registered_agent is agent
        
        # 验证Agent已初始化
        assert agent._initialized
        
        # 验证组分配
        group_agents = registry.get_agents_by_group("test_group")
        assert agent in group_agents
        
        # 验证标签分配
        tag_agents = registry.get_agents_by_tag("test")
        assert agent in tag_agents
    
    @pytest.mark.asyncio
    async def test_agent_unregistration(self, registry, test_agents):
        """测试Agent注销"""
        agent = test_agents[0]
        
        # 先注册
        await registry.register_agent(agent)
        assert registry.get_agent(agent.name) is not None
        
        # 注销
        success = await registry.unregister_agent(agent.name)
        assert success
        
        # 验证注销
        assert registry.get_agent(agent.name) is None
        assert not agent._initialized
    
    @pytest.mark.asyncio
    async def test_dependency_management(self, registry, test_agents):
        """测试依赖管理"""
        agent1, agent2, agent3 = test_agents
        
        # 注册第一个Agent
        await registry.register_agent(agent1)
        
        # 注册依赖于第一个Agent的第二个Agent
        success = await registry.register_agent(agent2, dependencies={agent1.name})
        assert success
        
        # 尝试注册依赖不存在Agent的第三个Agent
        success = await registry.register_agent(agent3, dependencies={"nonexistent_agent"})
        assert not success
    
    @pytest.mark.asyncio
    async def test_config_update(self, registry, test_agents):
        """测试配置更新"""
        agent = test_agents[0]
        await registry.register_agent(agent)
        
        # 更新配置
        success = await registry.update_agent_config(agent.name, {
            "enabled": False,
            "priority": 50
        })
        assert success
        
        # 验证配置更新
        assert not agent.enabled
        assert agent.priority == 50
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, registry, test_agents):
        """测试健康监控"""
        agent = test_agents[0]
        await registry.register_agent(agent)
        
        # 获取健康信息
        health = registry.get_agent_health(agent.name)
        assert health is not None
        assert health.agent_name == agent.name
        
        # 模拟心跳
        heartbeat_message = Message(
            topic="agent.test_agent_0.heartbeat",
            data={
                "agent_name": agent.name,
                "timestamp": time.time(),
                "state": "running"
            }
        )
        await registry._handle_heartbeat(heartbeat_message)
        
        # 验证心跳更新
        assert health.is_alive
    
    @pytest.mark.asyncio
    async def test_event_handling(self, registry):
        """测试事件处理"""
        event_received = False
        event_data = None
        
        async def event_handler(data):
            nonlocal event_received, event_data
            event_received = True
            event_data = data
        
        # 添加事件处理器
        registry.add_event_handler("agent_registered", event_handler)
        
        # 注册Agent触发事件
        config = AgentConfig(name="event_test_agent")
        agent = MockTestAgent(config)
        await registry.register_agent(agent)
        
        # 等待事件处理
        await asyncio.sleep(0.1)
        
        # 验证事件处理
        assert event_received
        assert event_data["agent_name"] == "event_test_agent"
    
    def test_registry_stats(self, registry):
        """测试注册表统计"""
        stats = registry.get_registry_stats()
        
        assert "total_agents" in stats
        assert "healthy_agents" in stats
        assert "unhealthy_agents" in stats
        assert "groups" in stats
        assert stats["total_agents"] == 0  # 新注册表应该没有Agent


class TestAgentRegistryIntegration:
    """Agent注册表集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle(self):
        """测试完整的Agent生命周期"""
        # 创建消息总线
        mock_bus = MagicMock(spec=MessageBus)
        mock_bus.create_publisher = MagicMock()
        mock_bus.create_subscriber = MagicMock()
        mock_bus.publish = MagicMock()
        
        subscriber = MagicMock()
        subscriber.subscribe = AsyncMock()
        mock_bus.create_subscriber.return_value = subscriber
        
        # 创建注册表
        registry = AgentRegistry(
            message_bus=mock_bus,
            health_check_interval=0.5
        )
        
        try:
            await registry.initialize()
            
            # 创建Agent
            config = AgentConfig(name="lifecycle_test_agent")
            agent = MockTestAgent(config, mock_bus)
            
            # 注册Agent
            await registry.register_agent(agent, group="test_group")
            
            # 验证Agent运行
            assert agent.get_state() == AgentState.RUNNING
            
            # 执行一些操作
            trading_state = MagicMock(spec=TradingState)
            signals = await agent.analyze_with_monitoring(trading_state)
            assert len(signals) == 1
            
            # 更新配置
            await registry.update_agent_config(agent.name, {"priority": 100})
            assert agent.priority == 100
            
            # 检查健康状态
            health = registry.get_agent_health(agent.name)
            assert health is not None
            
            # 注销Agent
            await registry.unregister_agent(agent.name)
            assert registry.get_agent(agent.name) is None
            
        finally:
            await registry.shutdown()
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """测试故障恢复"""
        mock_bus = MagicMock(spec=MessageBus)
        mock_bus.create_publisher = MagicMock()
        mock_bus.create_subscriber = MagicMock()
        mock_bus.publish = MagicMock()
        
        subscriber = MagicMock()
        subscriber.subscribe = AsyncMock()
        mock_bus.create_subscriber.return_value = subscriber
        
        registry = AgentRegistry(
            message_bus=mock_bus,
            health_check_interval=0.1,
            max_consecutive_failures=2
        )
        
        try:
            await registry.initialize()
            
            # 创建Agent
            config = AgentConfig(name="recovery_test_agent")
            agent = MockTestAgent(config, mock_bus)
            await registry.register_agent(agent)
            
            # 模拟Agent进入错误状态
            await agent._set_state(AgentState.ERROR, "测试错误")
            
            # 等待健康检查和恢复
            await asyncio.sleep(0.3)
            
            # 验证恢复（需要检查Agent是否恢复到正常状态）
            # 这里的具体验证取决于恢复策略的实现
            
        finally:
            await registry.shutdown()


class TestAnalysisAgent:
    """分析Agent测试"""
    
    @pytest.mark.asyncio
    async def test_analysis_agent_creation(self):
        """测试分析Agent创建"""
        config = AgentConfig(name="analysis_agent")
        
        class TestAnalysisAgent(AnalysisAgent):
            async def analyze(self, state):
                return []
        
        agent = TestAnalysisAgent(config)
        await agent.initialize()
        
        assert isinstance(agent, AnalysisAgent)
        assert isinstance(agent, BaseAgent)
        
        await agent.shutdown()


class TestExecutionAgent:
    """执行Agent测试"""
    
    @pytest.mark.asyncio
    async def test_execution_agent_creation(self):
        """测试执行Agent创建"""
        config = AgentConfig(name="execution_agent")
        
        class TestExecutionAgent(ExecutionAgent):
            async def execute(self, signal, state):
                return None
        
        agent = TestExecutionAgent(config)
        await agent.initialize()
        
        assert isinstance(agent, ExecutionAgent)
        assert isinstance(agent, BaseAgent)
        
        # 执行Agent不应该生成信号
        trading_state = MagicMock(spec=TradingState)
        signals = await agent.analyze(trading_state)
        assert len(signals) == 0
        
        await agent.shutdown()


class TestRiskAgent:
    """风险Agent测试"""
    
    @pytest.mark.asyncio
    async def test_risk_agent_creation(self):
        """测试风险Agent创建"""
        config = AgentConfig(name="risk_agent")
        
        class TestRiskAgent(RiskAgent):
            async def check_risk(self, order, state):
                return True
            
            async def adjust_position_size(self, order, state):
                return order
        
        agent = TestRiskAgent(config)
        await agent.initialize()
        
        assert isinstance(agent, RiskAgent)
        assert isinstance(agent, BaseAgent)
        
        await agent.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])