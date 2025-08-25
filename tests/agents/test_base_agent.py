"""
投资大师Agent基类测试
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_agent import (
    InvestmentMasterAgent,
    InvestmentMasterConfig,
    MasterInsight
)
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.llm_client import LLMResponse
from src.agents.prompt_templates import PromptTemplate, InvestmentStylePrompts
from src.core.models import MarketData, TradingState, Signal


class TestInvestmentMasterAgent:
    """投资大师Agent测试类"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return InvestmentMasterConfig(
            name="test_master",
            master_name="Warren Buffett",
            investment_style=InvestmentStyle.VALUE,
            specialty=["stocks", "long-term"],
            llm_provider="local",
            llm_model="test-model",
            llm_temperature=0.7,
            llm_max_tokens=2000,
            enable_caching=True,
            cache_ttl_seconds=300,
            max_retries=3
        )
    
    @pytest.fixture
    def trading_state(self):
        """创建测试交易状态"""
        state = TradingState()
        state.timestamp = datetime.now()
        state.active_symbols = ["BTC/USDT", "ETH/USDT"]
        state.market_data = {
            "BTC/USDT": MarketData(
                symbol="BTC/USDT",
                timestamp=datetime.now(),
                open=50000,
                high=51000,
                low=49000,
                close=50500,
                volume=1000000
            ),
            "ETH/USDT": MarketData(
                symbol="ETH/USDT",
                timestamp=datetime.now(),
                open=3000,
                high=3100,
                low=2900,
                close=3050,
                volume=500000
            )
        }
        state.indicators = {
            "BTC/USDT": {
                "RSI": 55,
                "MACD": 100,
                "MA_20": 49500
            },
            "ETH/USDT": {
                "RSI": 60,
                "MACD": 50,
                "MA_20": 2950
            }
        }
        return state
    
    @pytest.fixture
    async def agent(self, config):
        """创建测试Agent"""
        
        class TestMasterAgent(InvestmentMasterAgent):
            """测试用投资大师Agent"""
            
            async def make_investment_decision(self, state, portfolio):
                """实现抽象方法"""
                return {
                    "action": "BUY",
                    "confidence": 0.8,
                    "reasoning": "Test decision"
                }
            
            async def evaluate_portfolio(self, portfolio, market_conditions):
                """实现抽象方法"""
                return {
                    "score": 0.75,
                    "recommendations": ["Rebalance"]
                }
        
        agent = TestMasterAgent(config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """测试Agent初始化"""
        assert agent.master_name == "Warren Buffett"
        assert agent.investment_style == InvestmentStyle.VALUE
        assert agent.llm_client is not None
        assert agent.prompt_template is not None
        assert agent.style_prompts is not None
    
    @pytest.mark.asyncio
    async def test_personality_setup(self, agent):
        """测试个性化配置"""
        # 检查默认个性特征
        assert agent.master_config.personality_traits is not None
        assert "patience" in agent.master_config.personality_traits
        
        # 检查默认指标
        assert agent.master_config.favorite_indicators is not None
        assert "PE" in agent.master_config.favorite_indicators
    
    @pytest.mark.asyncio
    async def test_generate_insights(self, agent, trading_state):
        """测试生成洞察"""
        # Mock LLM响应
        mock_response = LLMResponse(
            content=json.dumps({
                "analysis_type": "market_trend",
                "conclusion": "Market shows strong value opportunity",
                "confidence": 0.75,
                "key_points": ["Low PE ratio", "Strong fundamentals"],
                "recommendations": [{"action": "BUY", "reason": "Undervalued"}],
                "risks": ["Market volatility"],
                "time_horizon": "long"
            }),
            model="test-model",
            provider="local",
            tokens_used=100,
            response_time=0.5
        )
        
        with patch.object(agent, '_call_llm', return_value=mock_response):
            insights = await agent.generate_insights(trading_state)
            
            assert len(insights) == 2  # 两个交易对
            assert all(isinstance(i, MasterInsight) for i in insights)
            
            # 检查第一个洞察
            insight = insights[0]
            assert insight.master_name == "Warren Buffett"
            assert insight.investment_style == InvestmentStyle.VALUE
            assert insight.confidence_score == 0.75
            assert len(insight.key_points) == 2
            assert len(insight.recommendations) == 1
    
    @pytest.mark.asyncio
    async def test_analyze_symbol(self, agent, trading_state):
        """测试分析单个交易对"""
        mock_response = LLMResponse(
            content=json.dumps({
                "analysis_type": "market_trend",
                "conclusion": "Bullish trend",
                "confidence": 0.8,
                "key_points": ["Strong support"],
                "recommendations": [{"action": "BUY"}],
                "risks": [],
                "time_horizon": "medium"
            }),
            model="test-model",
            provider="local",
            tokens_used=100,
            response_time=0.5
        )
        
        with patch.object(agent, '_call_llm', return_value=mock_response):
            insight = await agent._analyze_symbol("BTC/USDT", trading_state)
            
            assert insight.master_name == "Warren Buffett"
            assert insight.main_conclusion == "Bullish trend"
            assert insight.confidence_score == 0.8
            assert insight.metadata["symbol"] == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_convert_insights_to_signals(self, agent):
        """测试将洞察转换为信号"""
        insights = [
            MasterInsight(
                master_name="Warren Buffett",
                investment_style=InvestmentStyle.VALUE,
                analysis_type=AnalysisType.MARKET_TREND,
                main_conclusion="Strong buy signal",
                confidence_score=0.85,
                key_points=["Undervalued", "Strong fundamentals"],
                recommendations=[
                    {"action": "BUY", "reason": "Value opportunity"},
                    {"action": "HOLD", "reason": "Long-term investment"}
                ],
                risk_warnings=["Volatility"],
                time_horizon="long",
                metadata={"symbol": "BTC/USDT"}
            )
        ]
        
        signals = agent._convert_insights_to_signals(insights)
        
        assert len(signals) == 2  # 两个推荐
        assert all(isinstance(s, Signal) for s in signals)
        
        # 检查第一个信号
        signal = signals[0]
        assert signal.symbol == "BTC/USDT"
        assert signal.action == "BUY"
        assert signal.confidence == 0.85
        assert signal.metadata["master_name"] == "Warren Buffett"
    
    @pytest.mark.asyncio
    async def test_caching(self, agent, trading_state):
        """测试缓存功能"""
        mock_response = LLMResponse(
            content=json.dumps({
                "analysis_type": "market_trend",
                "conclusion": "Test",
                "confidence": 0.7,
                "key_points": [],
                "recommendations": [],
                "risks": [],
                "time_horizon": "medium"
            }),
            model="test-model",
            provider="local",
            tokens_used=100,
            response_time=0.5
        )
        
        with patch.object(agent, '_call_llm', return_value=mock_response) as mock_llm:
            # 第一次调用
            insights1 = await agent.generate_insights(trading_state)
            assert mock_llm.call_count == 2  # 两个交易对
            
            # 第二次调用（应该使用缓存）
            insights2 = await agent.generate_insights(trading_state)
            assert mock_llm.call_count == 2  # 没有新的LLM调用
            
            # 检查缓存统计
            assert agent._cache_hits == 2
            assert agent._cache_misses == 2
    
    @pytest.mark.asyncio
    async def test_fallback_insight(self, agent):
        """测试降级洞察"""
        fallback = agent._generate_fallback_insight("BTC/USDT")
        
        assert fallback.master_name == "Warren Buffett"
        assert fallback.confidence_score == 0.3
        assert fallback.metadata["fallback"] is True
        assert "HOLD" in str(fallback.recommendations)
    
    @pytest.mark.asyncio
    async def test_parse_natural_language(self, agent):
        """测试自然语言解析"""
        content = """
        Overall conclusion: The market shows strong bullish momentum.
        
        Key point: Support levels are holding well.
        Important note: Volume is increasing significantly.
        
        I recommend buying at current levels.
        You should consider taking profits at resistance.
        
        Risk warning: Volatility may increase.
        Caution: External factors could impact prices.
        
        Confidence level: 75%
        """
        
        result = agent._parse_natural_language(content)
        
        assert "conclusion" in result["conclusion"].lower()
        assert result["confidence"] == 0.75
        assert len(result["key_points"]) >= 2
        assert len(result["recommendations"]) >= 2
        assert len(result["risks"]) >= 2
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, agent):
        """测试性能指标"""
        # 模拟更新性能指标
        agent._update_performance_metrics(1.5, 150)
        agent._update_performance_metrics(2.0, 200)
        
        assert agent._total_tokens_used == 350
        assert agent._avg_response_time == 1.75
        
        # 获取分析摘要
        summary = await agent.get_analysis_summary()
        
        assert summary["master_name"] == "Warren Buffett"
        assert summary["performance"]["total_tokens"] == 350
        assert summary["performance"]["avg_response_time"] == 1.75
    
    @pytest.mark.asyncio
    async def test_market_data_preparation(self, agent, trading_state):
        """测试市场数据准备"""
        data = agent._prepare_market_data("BTC/USDT", trading_state)
        
        assert data["symbol"] == "BTC/USDT"
        assert "price_data" in data
        assert data["price_data"]["current"] == 50500
        assert data["price_data"]["change_pct"] == 1.0  # (50500-50000)/50000 * 100
        assert "volume_data" in data
        assert data["volume_data"]["current"] == 1000000
        assert "indicators" in data
        assert data["indicators"]["RSI"] == 55
    
    @pytest.mark.asyncio
    async def test_signal_strength_calculation(self, agent):
        """测试信号强度计算"""
        insight = MasterInsight(
            master_name="Test",
            investment_style=InvestmentStyle.VALUE,
            analysis_type=AnalysisType.MARKET_TREND,
            main_conclusion="Test",
            confidence_score=0.8,
            key_points=[],
            recommendations=[],
            risk_warnings=["Risk1", "Risk2"],  # 两个风险
            time_horizon="long"
        )
        
        strength = agent._calculate_signal_strength(insight)
        
        # 0.8 - 0.2 (风险) * 1.2 (长期) = 0.72
        assert strength == pytest.approx(0.72, rel=0.01)
    
    @pytest.mark.asyncio
    async def test_recommendation_mapping(self, agent):
        """测试推荐动作映射"""
        assert agent._map_recommendation_to_action({"action": "buy now"}) == "BUY"
        assert agent._map_recommendation_to_action({"action": "sell position"}) == "SELL"
        assert agent._map_recommendation_to_action({"action": "hold and wait"}) == "HOLD"
        assert agent._map_recommendation_to_action({"action": "unclear"}) == "NEUTRAL"
    
    @pytest.mark.asyncio
    async def test_config_update(self, agent):
        """测试配置更新"""
        await agent.update_master_config(
            llm_temperature=0.9,
            risk_tolerance="aggressive",
            time_horizon="short"
        )
        
        assert agent.master_config.llm_temperature == 0.9
        assert agent.master_config.risk_tolerance == "aggressive"
        assert agent.master_config.time_horizon == "short"


class TestPromptTemplates:
    """提示词模板测试类"""
    
    def test_prompt_template_initialization(self):
        """测试提示词模板初始化"""
        template = PromptTemplate()
        assert "market_analysis" in template.templates
        assert "portfolio_evaluation" in template.templates
        assert "risk_assessment" in template.templates
    
    def test_build_analysis_prompt(self):
        """测试构建分析提示词"""
        template = PromptTemplate()
        
        market_data = {
            "price_data": {"current": 50000, "change_pct": 2.5},
            "volume_data": {"current": 1000000},
            "indicators": {"RSI": 55, "MACD": 100}
        }
        
        prompt = template.build_analysis_prompt(
            symbol="BTC/USDT",
            market_data=market_data,
            analysis_type=AnalysisType.MARKET_TREND,
            time_horizon="medium",
            risk_tolerance="moderate",
            favorite_indicators=["RSI", "MACD"]
        )
        
        assert "BTC/USDT" in prompt
        assert "market_trend" in prompt
        assert "RSI: 55" in prompt
        assert "MACD: 100" in prompt
    
    def test_investment_style_prompts(self):
        """测试投资风格提示词"""
        style_prompts = InvestmentStylePrompts()
        
        # 测试获取系统提示词
        prompt = style_prompts.get_system_prompt(
            InvestmentStyle.VALUE,
            "Warren Buffett"
        )
        
        assert "value investor" in prompt
        assert "Warren Buffett" in prompt
        assert "Be fearful when others are greedy" in prompt
    
    def test_master_personalities(self):
        """测试投资大师个性"""
        style_prompts = InvestmentStylePrompts()
        
        # 测试获取决策风格
        decision_style = style_prompts.get_decision_style("George Soros")
        
        assert decision_style["style"] == InvestmentStyle.MACRO
        assert decision_style["traits"]["reflexivity"] == "core_belief"
        assert decision_style["risk_tolerance"] == "calculated"
    
    def test_format_master_response(self):
        """测试格式化大师响应"""
        style_prompts = InvestmentStylePrompts()
        
        response = style_prompts.format_master_response(
            "Warren Buffett",
            "This is a great value opportunity.",
            0.85
        )
        
        assert "value opportunity" in response
        assert "As I always say" in response


class TestLLMIntegration:
    """LLM集成测试"""
    
    @pytest.mark.asyncio
    async def test_llm_retry_mechanism(self, config):
        """测试LLM重试机制"""
        
        class TestAgent(InvestmentMasterAgent):
            async def make_investment_decision(self, state, portfolio):
                return {}
            
            async def evaluate_portfolio(self, portfolio, market_conditions):
                return {}
        
        agent = TestAgent(config)
        await agent.initialize()
        
        # 模拟LLM调用失败
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API error")
            return LLMResponse(
                content='{"confidence": 0.7}',
                model="test",
                provider="test",
                tokens_used=100,
                response_time=0.5
            )
        
        agent.llm_client.generate = mock_generate
        
        # 应该在第3次成功
        response = await agent._call_llm("test prompt")
        assert call_count == 3
        assert response.content == '{"confidence": 0.7}'
    
    @pytest.mark.asyncio
    async def test_llm_timeout_handling(self, config):
        """测试LLM超时处理"""
        config.llm_timeout = 0.1  # 100ms超时
        
        class TestAgent(InvestmentMasterAgent):
            async def make_investment_decision(self, state, portfolio):
                return {}
            
            async def evaluate_portfolio(self, portfolio, market_conditions):
                return {}
        
        agent = TestAgent(config)
        await agent.initialize()
        
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(1)  # 模拟慢响应
            return LLMResponse("", "test", "test", 0, 0)
        
        agent.llm_client.generate = slow_generate
        
        with pytest.raises(Exception):
            await agent._call_llm("test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])