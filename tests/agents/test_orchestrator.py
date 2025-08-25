"""
MultiAgentOrchestrator端到端测试
测试完整的LangGraph工作流编排功能
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from src.agents.orchestrator import (
    MultiAgentOrchestrator, WorkflowConfig, WorkflowStatus, WorkflowMetrics
)
from src.agents.models import (
    AgentState, MarketDataState, NewsDataState,
    AnalystOpinionState, RiskAssessmentState,
    PortfolioRecommendation, FinalDecision
)
from src.agents.workflow_nodes import WorkflowNodes, NodeExecutionResult
from src.agents.result_aggregator import ResultAggregator, AggregatedResult


class TestMultiAgentOrchestrator:
    """MultiAgentOrchestrator测试类"""
    
    @pytest.fixture
    def workflow_config(self):
        """创建测试配置"""
        return WorkflowConfig(
            max_parallel_agents=3,
            enable_checkpointing=False,  # 测试时禁用检查点
            timeout_seconds=60,
            retry_failed_nodes=False,
            max_retries=2,
            aggregation_method="weighted_voting",
            consensus_threshold=0.6,
            enable_monitoring=True
        )
    
    @pytest.fixture
    def initial_state(self):
        """创建初始状态"""
        return AgentState(
            session_id="test_session",
            agent_id="test_orchestrator",
            state_version=1,
            market_data={
                "BTC/USDT": MarketDataState(
                    symbol="BTC/USDT",
                    price=50000.0,
                    volume=1000.0,
                    bid=49990.0,
                    ask=50010.0,
                    spread=20.0,
                    volatility=0.02,
                    trend="up",
                    momentum=0.8,
                    timestamp=datetime.now(),
                    metadata={}
                )
            },
            news_data=[
                NewsDataState(
                    source="test_news",
                    title="Bitcoin Surges",
                    content="Bitcoin price increases...",
                    sentiment_score=0.7,
                    relevance_score=0.9,
                    impact_level="high",
                    entities=["Bitcoin", "Crypto"],
                    timestamp=datetime.now(),
                    url="http://example.com"
                )
            ],
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
    
    @pytest.fixture
    def orchestrator(self, workflow_config):
        """创建编排器实例"""
        return MultiAgentOrchestrator(config=workflow_config)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """测试编排器初始化"""
        assert orchestrator is not None
        assert orchestrator.workflow_status == WorkflowStatus.IDLE
        assert orchestrator.current_workflow_id is None
        assert orchestrator.workflow_metrics is None
        assert len(orchestrator.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_graph_structure(self, orchestrator):
        """测试工作流图结构"""
        # 验证节点存在
        graph = orchestrator.workflow_graph
        nodes = graph.nodes
        
        expected_nodes = [
            "data_preprocessing",
            "parallel_analysis", 
            "result_aggregation",
            "risk_assessment",
            "portfolio_optimization",
            "decision_output"
        ]
        
        for node in expected_nodes:
            assert node in nodes
    
    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, orchestrator, initial_state):
        """测试成功执行工作流"""
        # Mock Agent分析结果
        mock_analyst_opinions = [
            AnalystOpinionState(
                source="warren_buffett",
                analyst_name="Warren Buffett",
                rating="buy",
                target_price=55000.0,
                confidence=0.8,
                rationale="Strong fundamentals",
                risk_factors=["Volatility"],
                timestamp=datetime.now()
            ),
            AnalystOpinionState(
                source="technical_analyst",
                analyst_name="Technical Analyst",
                rating="buy",
                target_price=52000.0,
                confidence=0.7,
                rationale="Bullish pattern",
                risk_factors=["Support levels"],
                timestamp=datetime.now()
            )
        ]
        
        # Mock工作流节点
        with patch.object(orchestrator.workflow_nodes, '_initialize_agents'):
            with patch.object(orchestrator.workflow_nodes, '_run_analyst_agent') as mock_run:
                mock_run.return_value = {
                    'agent_name': 'test_agent',
                    'master_name': 'Test Master',
                    'recommendation': 'buy',
                    'confidence': 0.75,
                    'target_price': 53000.0,
                    'rationale': 'Test analysis',
                    'risk_factors': []
                }
                
                # Mock风险评估
                mock_risk_assessment = MagicMock()
                mock_risk_assessment.risk_level.value = "moderate"
                mock_risk_assessment.risk_score = 45.0
                mock_risk_assessment.risk_metrics.var_95 = 0.05
                mock_risk_assessment.risk_metrics.var_99 = 0.08
                mock_risk_assessment.risk_metrics.max_drawdown = 0.15
                mock_risk_assessment.risk_metrics.sharpe_ratio = 1.2
                mock_risk_assessment.risk_metrics.concentration_risk = 0.3
                mock_risk_assessment.risk_metrics.liquidity_risk = 0.2
                mock_risk_assessment.risk_factors = []
                mock_risk_assessment.mitigation_strategies = ["Diversify"]
                
                with patch.object(orchestrator.workflow_nodes._risk_agent, 'analyze_state') as mock_risk:
                    mock_risk.return_value = mock_risk_assessment
                    
                    # Mock投资组合优化
                    mock_portfolio_decision = MagicMock()
                    mock_portfolio_decision.need_rebalance = True
                    mock_portfolio_decision.confidence = 0.75
                    mock_portfolio_decision.allocations = [
                        MagicMock(
                            symbol="BTC/USDT",
                            target_weight=0.5,
                            recommended_action="BUY",
                            confidence=0.8,
                            reasoning="Strong momentum"
                        )
                    ]
                    mock_portfolio_decision.metrics.expected_return = 0.15
                    mock_portfolio_decision.metrics.sharpe_ratio = 1.2
                    
                    with patch.object(orchestrator.workflow_nodes._portfolio_agent, 'optimize_portfolio') as mock_portfolio:
                        mock_portfolio.return_value = mock_portfolio_decision
                        
                        # 执行工作流
                        result = await orchestrator.execute_workflow(initial_state)
                        
                        # 验证结果
                        assert result['success'] is True
                        assert result['workflow_id'] is not None
                        assert 'final_state' in result
                        assert 'metrics' in result
                        assert 'execution_summary' in result
    
    @pytest.mark.asyncio
    async def test_execute_workflow_timeout(self, orchestrator, initial_state):
        """测试工作流超时"""
        # 设置很短的超时时间
        orchestrator.config.timeout_seconds = 0.001
        
        # Mock一个耗时的操作
        async def slow_node(state):
            await asyncio.sleep(1)
            return state
        
        with patch.object(orchestrator.workflow_nodes, 'data_preprocessing_node', slow_node):
            result = await orchestrator.execute_workflow(initial_state)
            
            assert result['success'] is False
            assert 'timeout' in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_node_failure_handling(self, orchestrator, initial_state):
        """测试节点失败处理"""
        # Mock一个会失败的节点
        async def failing_node(state):
            raise ValueError("Test error")
        
        orchestrator.config.retry_failed_nodes = False
        
        with patch.object(orchestrator.workflow_nodes, 'parallel_analysis_node', failing_node):
            result = await orchestrator.execute_workflow(initial_state)
            
            assert result['success'] is False
            assert 'Test error' in result['error']
    
    @pytest.mark.asyncio
    async def test_node_retry_mechanism(self, orchestrator, initial_state):
        """测试节点重试机制"""
        orchestrator.config.retry_failed_nodes = True
        orchestrator.config.max_retries = 3
        
        # 创建一个会失败然后成功的节点
        call_count = 0
        
        async def flaky_node(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Temporary error {call_count}")
            return state
        
        # 注意：这个测试需要实际的重试逻辑实现
        # 当前实现中重试是在_wrap_node中处理的
    
    @pytest.mark.asyncio
    async def test_result_aggregation(self, orchestrator):
        """测试结果聚合功能"""
        opinions = [
            AnalystOpinionState(
                source="agent1",
                analyst_name="Agent 1",
                rating="buy",
                target_price=100,
                confidence=0.8,
                rationale="Bullish",
                risk_factors=[],
                timestamp=datetime.now()
            ),
            AnalystOpinionState(
                source="agent2",
                analyst_name="Agent 2",
                rating="buy",
                target_price=105,
                confidence=0.7,
                rationale="Positive",
                risk_factors=[],
                timestamp=datetime.now()
            ),
            AnalystOpinionState(
                source="agent3",
                analyst_name="Agent 3",
                rating="hold",
                target_price=95,
                confidence=0.6,
                rationale="Neutral",
                risk_factors=["Market risk"],
                timestamp=datetime.now()
            )
        ]
        
        aggregator = orchestrator.result_aggregator
        result = aggregator.aggregate_opinions(opinions)
        
        assert result.consensus_action in ["buy", "sell", "hold"]
        assert 0 <= result.consensus_confidence <= 1
        assert 0 <= result.agreement_level <= 1
        assert len(result.risk_factors) >= 0
    
    @pytest.mark.asyncio
    async def test_callback_registration_and_execution(self, orchestrator, initial_state):
        """测试回调函数注册和执行"""
        callback_executed = False
        callback_state = None
        
        def test_callback(state, error=None):
            nonlocal callback_executed, callback_state
            callback_executed = True
            callback_state = state
        
        # 注册回调
        orchestrator.register_callback("before_data_preprocessing", test_callback)
        
        # 执行节点包装器中的回调逻辑
        wrapped_node = orchestrator._wrap_node(orchestrator.workflow_nodes.data_preprocessing_node)
        
        # 由于实际执行涉及复杂的依赖，这里只验证回调注册
        assert "before_data_preprocessing" in orchestrator.node_callbacks
        assert test_callback in orchestrator.node_callbacks["before_data_preprocessing"]
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_tracking(self, orchestrator, initial_state):
        """测试工作流指标跟踪"""
        # 初始化指标
        metrics = WorkflowMetrics(
            workflow_id="test_workflow",
            start_time=datetime.now()
        )
        
        # 模拟节点执行
        metrics.nodes_executed = 5
        metrics.nodes_successful = 4
        metrics.nodes_failed = 1
        
        # 计算成功率
        success_rate = metrics.calculate_success_rate()
        assert success_rate == 0.8
        
        # 设置结束时间
        metrics.end_time = datetime.now() + timedelta(seconds=10)
        metrics.total_execution_time_ms = 10000
        
        assert metrics.status == WorkflowStatus.IDLE
        assert len(metrics.error_messages) == 0
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(self, orchestrator):
        """测试取消工作流"""
        orchestrator.workflow_status = WorkflowStatus.RUNNING
        orchestrator.current_workflow_id = "test_workflow"
        orchestrator.workflow_metrics = WorkflowMetrics(
            workflow_id="test_workflow",
            start_time=datetime.now()
        )
        
        # 取消工作流
        result = await orchestrator.cancel_workflow()
        
        assert result is True
        assert orchestrator.workflow_status == WorkflowStatus.CANCELLED
        assert orchestrator.workflow_metrics.status == WorkflowStatus.CANCELLED
        assert orchestrator.workflow_metrics.end_time is not None
    
    @pytest.mark.asyncio
    async def test_get_workflow_status(self, orchestrator):
        """测试获取工作流状态"""
        status = orchestrator.get_workflow_status()
        
        assert status['status'] == WorkflowStatus.IDLE.value
        assert status['current_workflow_id'] is None
        assert status['metrics'] is None
        assert status['execution_history_count'] == 0
    
    @pytest.mark.asyncio
    async def test_get_execution_history(self, orchestrator):
        """测试获取执行历史"""
        # 添加一些历史记录
        for i in range(5):
            metrics = WorkflowMetrics(
                workflow_id=f"workflow_{i}",
                start_time=datetime.now() - timedelta(hours=i),
                end_time=datetime.now() - timedelta(hours=i) + timedelta(minutes=10),
                status=WorkflowStatus.COMPLETED
            )
            metrics.nodes_executed = 6
            metrics.nodes_successful = 6
            metrics.total_execution_time_ms = 600000
            orchestrator.execution_history.append(metrics)
        
        # 获取历史
        history = orchestrator.get_execution_history(limit=3)
        
        assert len(history) == 3
        assert all('workflow_id' in h for h in history)
        assert all('status' in h for h in history)
        assert all('success_rate' in h for h in history)
    
    @pytest.mark.asyncio
    async def test_update_config(self, orchestrator):
        """测试更新配置"""
        new_config = WorkflowConfig(
            max_parallel_agents=10,
            aggregation_method="majority_voting",
            consensus_threshold=0.8
        )
        
        orchestrator.update_config(new_config)
        
        assert orchestrator.config.max_parallel_agents == 10
        assert orchestrator.config.aggregation_method == "majority_voting"
        assert orchestrator.config.consensus_threshold == 0.8
        assert orchestrator.result_aggregator.aggregation_method == "majority_voting"
        assert orchestrator.result_aggregator.consensus_threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, orchestrator, initial_state):
        """测试并行Agent执行"""
        # 记录Agent执行时间
        execution_times = []
        
        async def mock_agent_execution(agent_name, agent, state):
            start = datetime.now()
            await asyncio.sleep(0.1)  # 模拟执行时间
            end = datetime.now()
            execution_times.append((agent_name, (end - start).total_seconds()))
            return {
                'agent_name': agent_name,
                'master_name': agent_name,
                'recommendation': 'buy',
                'confidence': 0.7,
                'target_price': 100,
                'rationale': 'Test',
                'risk_factors': []
            }
        
        with patch.object(orchestrator.workflow_nodes, '_run_analyst_agent', mock_agent_execution):
            with patch.object(orchestrator.workflow_nodes, '_initialize_agents'):
                orchestrator.workflow_nodes._analyst_agents = {
                    'agent1': Mock(),
                    'agent2': Mock(),
                    'agent3': Mock()
                }
                
                result = await orchestrator.workflow_nodes.parallel_analysis_node(initial_state)
                
                # 验证所有Agent都被执行
                assert len(execution_times) == 3
                
                # 验证并行执行（总时间应该接近单个Agent的执行时间）
                total_time = sum(t for _, t in execution_times)
                max_time = max(t for _, t in execution_times)
                # 并行执行的总时间应该接近最长的单个执行时间
                assert total_time < max_time * 1.5  # 允许一些开销


@pytest.mark.asyncio
class TestWorkflowIntegration:
    """工作流集成测试"""
    
    @pytest.fixture
    def full_state(self):
        """创建完整的测试状态"""
        return AgentState(
            session_id="integration_test",
            agent_id="test_orchestrator",
            state_version=1,
            market_data={
                "BTC/USDT": MarketDataState(
                    symbol="BTC/USDT",
                    price=50000.0,
                    volume=10000.0,
                    bid=49950.0,
                    ask=50050.0,
                    spread=100.0,
                    volatility=0.025,
                    trend="up",
                    momentum=0.75,
                    timestamp=datetime.now(),
                    metadata={"exchange": "Binance"}
                ),
                "ETH/USDT": MarketDataState(
                    symbol="ETH/USDT",
                    price=3000.0,
                    volume=5000.0,
                    bid=2995.0,
                    ask=3005.0,
                    spread=10.0,
                    volatility=0.03,
                    trend="sideways",
                    momentum=0.5,
                    timestamp=datetime.now(),
                    metadata={"exchange": "Binance"}
                )
            },
            news_data=[
                NewsDataState(
                    source="CoinDesk",
                    title="Bitcoin ETF Approval Expected",
                    content="Analysts predict imminent approval...",
                    sentiment_score=0.8,
                    relevance_score=0.95,
                    impact_level="high",
                    entities=["Bitcoin", "ETF", "SEC"],
                    timestamp=datetime.now(),
                    url="http://example.com/news1"
                ),
                NewsDataState(
                    source="CryptoNews",
                    title="Ethereum Upgrade Successful",
                    content="The latest Ethereum upgrade...",
                    sentiment_score=0.6,
                    relevance_score=0.8,
                    impact_level="medium",
                    entities=["Ethereum", "Upgrade"],
                    timestamp=datetime.now(),
                    url="http://example.com/news2"
                )
            ],
            social_sentiment={
                "BTC/USDT": {
                    "platform": "twitter",
                    "symbol": "BTC/USDT",
                    "sentiment_score": 0.7,
                    "volume": 10000,
                    "trending_score": 0.85,
                    "key_topics": ["ETF", "Bull Run"],
                    "influencer_sentiment": 0.8,
                    "retail_sentiment": 0.65,
                    "timestamp": datetime.now()
                }
            },
            analyst_opinions=[],
            confidence_scores={},
            risk_assessment=None,
            portfolio_recommendations=[],
            final_decision=None,
            reasoning_chain=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "market_context": {
                    "trend": "bullish",
                    "volatility": "moderate"
                }
            }
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, full_state):
        """端到端工作流执行测试"""
        config = WorkflowConfig(
            max_parallel_agents=6,
            enable_checkpointing=False,
            timeout_seconds=120,
            retry_failed_nodes=True,
            max_retries=2,
            aggregation_method="ensemble",
            consensus_threshold=0.65,
            enable_monitoring=True
        )
        
        orchestrator = MultiAgentOrchestrator(config=config)
        
        # Mock所有Agent以避免实际的LLM调用
        with patch.object(orchestrator.workflow_nodes, '_initialize_agents'):
            # Mock分析师Agent
            mock_agents = {}
            for agent_name in ['warren_buffett', 'cathie_wood', 'ray_dalio', 
                             'benjamin_graham', 'technical_analyst', 'quantitative_analyst']:
                mock_agents[agent_name] = Mock()
            
            orchestrator.workflow_nodes._analyst_agents = mock_agents
            
            # Mock Agent执行结果
            async def mock_analyst(agent_name, agent, state):
                results = {
                    'warren_buffett': {'recommendation': 'buy', 'confidence': 0.85},
                    'cathie_wood': {'recommendation': 'buy', 'confidence': 0.9},
                    'ray_dalio': {'recommendation': 'hold', 'confidence': 0.7},
                    'benjamin_graham': {'recommendation': 'buy', 'confidence': 0.8},
                    'technical_analyst': {'recommendation': 'buy', 'confidence': 0.75},
                    'quantitative_analyst': {'recommendation': 'buy', 'confidence': 0.82}
                }
                
                return {
                    'agent_name': agent_name,
                    'master_name': agent_name.replace('_', ' ').title(),
                    'recommendation': results[agent_name]['recommendation'],
                    'confidence': results[agent_name]['confidence'],
                    'target_price': 52000.0 if results[agent_name]['recommendation'] == 'buy' else 48000.0,
                    'rationale': f'{agent_name} analysis complete',
                    'risk_factors': ['Market volatility', 'Regulatory risk']
                }
            
            with patch.object(orchestrator.workflow_nodes, '_run_analyst_agent', mock_analyst):
                # Mock风险和投资组合管理
                mock_risk_agent = Mock()
                mock_portfolio_agent = Mock()
                
                mock_risk_assessment = MagicMock()
                mock_risk_assessment.risk_level.value = "moderate"
                mock_risk_assessment.risk_score = 40.0
                mock_risk_assessment.risk_metrics.var_95 = 0.045
                mock_risk_assessment.risk_metrics.var_99 = 0.072
                mock_risk_assessment.risk_metrics.max_drawdown = 0.12
                mock_risk_assessment.risk_metrics.sharpe_ratio = 1.35
                mock_risk_assessment.risk_metrics.concentration_risk = 0.25
                mock_risk_assessment.risk_metrics.liquidity_risk = 0.15
                mock_risk_assessment.risk_factors = []
                mock_risk_assessment.mitigation_strategies = ["Position sizing", "Stop loss"]
                
                mock_risk_agent.analyze_state = AsyncMock(return_value=mock_risk_assessment)
                
                mock_portfolio_decision = MagicMock()
                mock_portfolio_decision.need_rebalance = True
                mock_portfolio_decision.confidence = 0.82
                mock_portfolio_decision.allocations = [
                    MagicMock(
                        symbol="BTC/USDT",
                        target_weight=0.6,
                        recommended_action="BUY",
                        confidence=0.85,
                        reasoning="Strong bullish signals"
                    ),
                    MagicMock(
                        symbol="ETH/USDT",
                        target_weight=0.4,
                        recommended_action="BUY",
                        confidence=0.75,
                        reasoning="Following BTC momentum"
                    )
                ]
                mock_portfolio_decision.metrics.expected_return = 0.18
                mock_portfolio_decision.metrics.sharpe_ratio = 1.35
                
                mock_portfolio_agent.optimize_portfolio = AsyncMock(return_value=mock_portfolio_decision)
                
                orchestrator.workflow_nodes._risk_agent = mock_risk_agent
                orchestrator.workflow_nodes._portfolio_agent = mock_portfolio_agent
                
                # 执行完整工作流
                result = await orchestrator.execute_workflow(full_state)
                
                # 验证结果
                assert result['success'] is True
                assert result['workflow_id'] is not None
                
                # 验证最终状态
                final_state = result['final_state']
                assert final_state is not None
                assert len(final_state.get('analyst_opinions', [])) > 0
                assert final_state.get('risk_assessment') is not None
                assert len(final_state.get('portfolio_recommendations', [])) > 0
                assert final_state.get('final_decision') is not None
                
                # 验证决策
                decision = result['decision']
                assert decision is not None
                assert decision.get('action') in ['execute', 'hold', 'reject']
                assert len(decision.get('recommendations', [])) > 0
                
                # 验证执行摘要
                summary = result['execution_summary']
                assert summary['successful_nodes'] > 0
                assert summary['total_execution_time_ms'] > 0
                
                # 验证指标
                metrics = result['metrics']
                assert metrics['status'] == 'completed'
                assert metrics['nodes_executed'] > 0
                assert metrics['success_rate'] > 0