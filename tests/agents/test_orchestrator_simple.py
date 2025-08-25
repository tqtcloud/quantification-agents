"""
简化的MultiAgentOrchestrator测试
测试基本的编排功能，避免复杂的依赖
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.agents.orchestrator import (
    MultiAgentOrchestrator, WorkflowConfig, WorkflowStatus
)
from src.agents.models import AgentState, MarketDataState
from src.agents.result_aggregator import ResultAggregator


class TestOrchestratorBasic:
    """基本编排器测试"""
    
    @pytest.fixture
    def simple_config(self):
        """创建简单配置"""
        return WorkflowConfig(
            max_parallel_agents=2,
            enable_checkpointing=False,
            timeout_seconds=30,
            retry_failed_nodes=False,
            aggregation_method="weighted_voting",
            consensus_threshold=0.6
        )
    
    @pytest.fixture
    def simple_state(self):
        """创建简单状态"""
        return AgentState(
            session_id="test",
            agent_id="test",
            state_version=1,
            market_data={
                "TEST/USDT": MarketDataState(
                    symbol="TEST/USDT",
                    price=100.0,
                    volume=1000.0,
                    bid=99.0,
                    ask=101.0,
                    spread=2.0,
                    volatility=0.02,
                    trend="up",
                    momentum=0.5,
                    timestamp=datetime.now(),
                    metadata={}
                )
            },
            news_data=[],
            social_sentiment={},
            analyst_opinions=[],
            confidence_scores={},
            risk_assessment=None,
            portfolio_recommendations=[],
            final_decision=None,
            reasoning_chain=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={}
        )
    
    def test_orchestrator_creation(self, simple_config):
        """测试编排器创建"""
        orchestrator = MultiAgentOrchestrator(config=simple_config)
        
        assert orchestrator is not None
        assert orchestrator.config == simple_config
        assert orchestrator.workflow_status == WorkflowStatus.IDLE
        assert orchestrator.current_workflow_id is None
        
    def test_workflow_config(self):
        """测试工作流配置"""
        config = WorkflowConfig()
        
        assert config.max_parallel_agents == 6
        assert config.enable_checkpointing == True
        assert config.timeout_seconds == 300
        assert config.aggregation_method == "weighted_voting"
        
    def test_result_aggregator_creation(self):
        """测试结果聚合器创建"""
        aggregator = ResultAggregator(
            aggregation_method="weighted_voting",
            consensus_threshold=0.6
        )
        
        assert aggregator is not None
        assert aggregator.aggregation_method == "weighted_voting"
        assert aggregator.consensus_threshold == 0.6
        
    def test_workflow_status_enum(self):
        """测试工作流状态枚举"""
        assert WorkflowStatus.IDLE.value == "idle"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        
    def test_initial_state_creation(self, simple_config):
        """测试初始状态创建"""
        orchestrator = MultiAgentOrchestrator(config=simple_config)
        initial_state = orchestrator._create_initial_state()
        
        assert initial_state is not None
        assert 'session_id' in initial_state
        assert 'agent_id' in initial_state
        assert initial_state['agent_id'] == "orchestrator"
        assert initial_state['state_version'] == 1
        
    @pytest.mark.asyncio
    async def test_callback_registration(self, simple_config):
        """测试回调注册"""
        orchestrator = MultiAgentOrchestrator(config=simple_config)
        
        callback_called = False
        
        def test_callback(state, error=None):
            nonlocal callback_called
            callback_called = True
        
        orchestrator.register_callback("test_event", test_callback)
        
        assert "test_event" in orchestrator.node_callbacks
        assert test_callback in orchestrator.node_callbacks["test_event"]
        
    def test_get_workflow_status(self, simple_config):
        """测试获取工作流状态"""
        orchestrator = MultiAgentOrchestrator(config=simple_config)
        status = orchestrator.get_workflow_status()
        
        assert 'current_workflow_id' in status
        assert 'status' in status
        assert status['status'] == WorkflowStatus.IDLE.value
        assert status['execution_history_count'] == 0
        
    def test_update_config(self, simple_config):
        """测试更新配置"""
        orchestrator = MultiAgentOrchestrator(config=simple_config)
        
        new_config = WorkflowConfig(
            max_parallel_agents=10,
            aggregation_method="majority_voting"
        )
        
        orchestrator.update_config(new_config)
        
        assert orchestrator.config.max_parallel_agents == 10
        assert orchestrator.config.aggregation_method == "majority_voting"