"""
投资大师Agent集成测试
测试15个专业分析师Agent的功能
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from src.agents.agent_registry import AgentRegistry, AgentOrchestrator
from src.agents.investment_masters import (
    WarrenBuffettAgent,
    CathieWoodAgent,
    RayDalioAgent,
    BenjaminGrahamAgent,
    TechnicalAnalystAgent,
    QuantitativeAnalystAgent,
)
from src.agents.base_agent import InvestmentMasterConfig
from src.agents.enums import InvestmentStyle, AnalysisType
from src.core.models import TradingState, MarketData, Signal


class TestInvestmentMasters:
    """投资大师Agent测试类"""
    
    @pytest.fixture
    def trading_state(self):
        """创建测试用的交易状态"""
        state = TradingState()
        
        # 添加测试数据
        symbols = ["AAPL", "TSLA", "BTC-USD", "SPY", "GLD"]
        for symbol in symbols:
            state.market_data[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000000.0
            )
        
        state.active_symbols = symbols
        return state
    
    @pytest.fixture
    def portfolio(self):
        """创建测试用的投资组合"""
        return {
            "total_value": 1000000,
            "cash": 200000,
            "positions": {
                "AAPL": {"value": 300000, "shares": 2000, "avg_cost": 140},
                "TSLA": {"value": 200000, "shares": 800, "avg_cost": 230},
                "SPY": {"value": 300000, "shares": 700, "avg_cost": 420}
            }
        }
    
    @pytest.fixture
    def registry(self):
        """创建Agent注册器"""
        return AgentRegistry()
    
    @pytest.mark.asyncio
    async def test_warren_buffett_agent(self, trading_state, portfolio):
        """测试Warren Buffett Agent"""
        agent = WarrenBuffettAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.action in ["BUY", "SELL", "HOLD"]
        assert 0 <= decision.confidence <= 1
        assert decision.reasoning_chain is not None
        assert len(decision.reasoning_chain) > 0
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {"sentiment": "neutral"})
        
        assert evaluation is not None
        assert "overall_score" in evaluation
        assert "recommendations" in evaluation
        assert "buffett_wisdom" in evaluation
    
    @pytest.mark.asyncio
    async def test_cathie_wood_agent(self, trading_state, portfolio):
        """测试Cathie Wood Agent"""
        agent = CathieWoodAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.action in ["BUY", "SELL", "HOLD"]
        assert decision.metadata.get("innovation_platforms") is not None
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {"volatility": "high"})
        
        assert evaluation is not None
        assert "innovation_exposure" in evaluation
        assert "ark_perspective" in evaluation
    
    @pytest.mark.asyncio
    async def test_ray_dalio_agent(self, trading_state, portfolio):
        """测试Ray Dalio Agent"""
        agent = RayDalioAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.action in ["BUY", "SELL", "HOLD", "REBALANCE", "REDUCE_RISK"]
        assert decision.metadata.get("allocation") is not None
        assert decision.metadata.get("economic_environment") is not None
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {"trend": "neutral"})
        
        assert evaluation is not None
        assert "diversification_score" in evaluation
        assert "risk_balance" in evaluation
    
    @pytest.mark.asyncio
    async def test_benjamin_graham_agent(self, trading_state, portfolio):
        """测试Benjamin Graham Agent"""
        agent = BenjaminGrahamAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.metadata.get("graham_score") is not None
        assert decision.metadata.get("investment_type") in ["defensive", "enterprising"]
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {"sentiment": "fearful"})
        
        assert evaluation is not None
        assert "safety_score" in evaluation
        assert "graham_wisdom" in evaluation
    
    @pytest.mark.asyncio
    async def test_technical_analyst_agent(self, trading_state, portfolio):
        """测试技术分析专家Agent"""
        agent = TechnicalAnalystAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.metadata.get("stop_loss") is not None
        assert decision.metadata.get("take_profit") is not None
        assert decision.metadata.get("entry_point") is not None
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {"trend": "bullish"})
        
        assert evaluation is not None
        assert "trend_alignment" in evaluation
        assert "technical_advice" in evaluation
    
    @pytest.mark.asyncio
    async def test_quantitative_analyst_agent(self, trading_state, portfolio):
        """测试量化分析师Agent"""
        agent = QuantitativeAnalystAgent()
        
        # 测试决策
        decision = await agent.make_investment_decision(trading_state, portfolio)
        
        assert decision is not None
        assert decision.metadata.get("sharpe_ratio") is not None
        assert decision.metadata.get("statistical_significance") is not None
        assert decision.metadata.get("factor_exposures") is not None
        
        # 测试组合评估
        evaluation = await agent.evaluate_portfolio(portfolio, {})
        
        assert evaluation is not None
        assert "sharpe_ratio" in evaluation
        assert "quant_advice" in evaluation
    
    @pytest.mark.asyncio
    async def test_agent_registry(self, registry, trading_state):
        """测试Agent注册器"""
        # 测试列出可用Agent
        available = registry.list_available_agents()
        assert len(available) > 0
        assert "warren_buffett" in available
        
        # 测试创建Agent
        agent = registry.create_agent("warren_buffett")
        assert agent is not None
        assert isinstance(agent, WarrenBuffettAgent)
        
        # 测试获取统计信息
        stats = registry.get_agent_statistics("warren_buffett")
        assert stats is not None
        assert "total_calls" in stats
    
    @pytest.mark.asyncio
    async def test_agent_orchestrator(self, registry, trading_state, portfolio):
        """测试Agent编排器"""
        orchestrator = AgentOrchestrator(registry)
        
        # 测试共识决策
        agent_names = ["warren_buffett", "technical_analyst", "quantitative_analyst"]
        
        consensus = await orchestrator.get_consensus_decision(
            trading_state,
            agent_names,
            method="confidence_weighted",
            portfolio=portfolio
        )
        
        assert consensus is not None
        assert consensus["action"] in ["BUY", "SELL", "HOLD"]
        assert "confidence" in consensus
        assert consensus["participants"] == len(agent_names)
    
    @pytest.mark.asyncio
    async def test_different_consensus_methods(self, registry, trading_state, portfolio):
        """测试不同的共识方法"""
        orchestrator = AgentOrchestrator(registry)
        agent_names = ["warren_buffett", "cathie_wood", "ray_dalio"]
        
        methods = ["majority_vote", "weighted_average", "confidence_weighted", "ensemble"]
        
        for method in methods:
            consensus = await orchestrator.get_consensus_decision(
                trading_state,
                agent_names,
                method=method,
                portfolio=portfolio
            )
            
            assert consensus is not None
            assert consensus["method"] == method
            assert consensus["action"] in ["BUY", "SELL", "HOLD", "REBALANCE", "REDUCE_RISK"]
    
    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, registry):
        """测试Agent性能跟踪"""
        # 创建Agent并生成信号
        agent = registry.create_agent("technical_analyst")
        
        # 模拟信号
        signals = [
            Signal(source="test", symbol="AAPL", action="BUY", strength=0.8, confidence=0.7),
            Signal(source="test", symbol="TSLA", action="SELL", strength=0.6, confidence=0.5)
        ]
        
        # 更新统计
        registry.update_agent_statistics("technical_analyst", signals)
        
        # 获取更新后的统计
        stats = registry.get_agent_statistics("technical_analyst")
        assert stats["total_signals"] >= len(signals)
        assert stats["avg_confidence"] > 0
    
    def test_agent_config_management(self, registry):
        """测试Agent配置管理"""
        # 创建自定义配置
        config = InvestmentMasterConfig(
            name="custom_buffett",
            master_name="Warren Buffett",
            investment_style=InvestmentStyle.VALUE,
            llm_temperature=0.2,
            risk_tolerance="conservative"
        )
        
        # 设置配置
        registry.set_agent_config("warren_buffett", config)
        
        # 获取配置
        retrieved = registry.get_agent_config("warren_buffett")
        assert retrieved is not None
        assert retrieved.llm_temperature == 0.2
        assert retrieved.risk_tolerance == "conservative"
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, trading_state, portfolio):
        """测试并行Agent执行"""
        agents = [
            WarrenBuffettAgent(),
            CathieWoodAgent(),
            RayDalioAgent(),
            TechnicalAnalystAgent(),
            QuantitativeAnalystAgent()
        ]
        
        # 并行执行所有Agent
        tasks = []
        for agent in agents:
            tasks.append(agent.make_investment_decision(trading_state, portfolio))
        
        decisions = await asyncio.gather(*tasks)
        
        # 验证所有决策
        assert len(decisions) == len(agents)
        for decision in decisions:
            assert decision is not None
            assert decision.action in ["BUY", "SELL", "HOLD", "REBALANCE", "REDUCE_RISK", "EXECUTE_ARBITRAGE"]
    
    @pytest.mark.asyncio
    async def test_agent_insights_generation(self, trading_state):
        """测试Agent洞察生成"""
        agent = WarrenBuffettAgent()
        
        # 生成洞察
        insights = await agent.generate_insights(trading_state)
        
        assert insights is not None
        assert len(insights) > 0
        
        for insight in insights:
            assert insight.master_name == "Warren Buffett"
            assert insight.investment_style == InvestmentStyle.VALUE
            assert insight.confidence_score >= 0 and insight.confidence_score <= 1
    
    def test_registry_state_persistence(self, registry, tmp_path):
        """测试注册器状态持久化"""
        # 创建一些Agent
        registry.create_agent("warren_buffett")
        registry.create_agent("technical_analyst")
        
        # 保存状态
        filepath = tmp_path / "registry_state.json"
        registry.save_registry_state(str(filepath))
        
        # 验证文件存在
        assert filepath.exists()
        
        # 创建新注册器并加载状态
        new_registry = AgentRegistry()
        new_registry.load_registry_state(str(filepath))
        
        # 验证统计信息被恢复
        stats = new_registry.get_agent_statistics("warren_buffett")
        assert stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])