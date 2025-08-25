"""
投资大师Agent演示
展示如何使用InvestmentMasterAgent基类创建投资大师
"""

import asyncio
import json
from datetime import datetime

from src.agents.base_agent import (
    InvestmentMasterAgent,
    InvestmentMasterConfig,
    MasterInsight
)
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision
from src.core.models import MarketData, TradingState
from src.utils.logger import setup_logging


# 创建具体的投资大师Agent
class WarrenBuffettAgent(InvestmentMasterAgent):
    """沃伦·巴菲特投资大师Agent"""
    
    async def make_investment_decision(self, state: TradingState, portfolio: dict) -> dict:
        """做出投资决策"""
        # 基于价值投资理念做决策
        insights = await self.generate_insights(state)
        
        decision = {
            "decision_id": "buffett_001",
            "action": "BUY" if insights and insights[0].confidence_score > 0.7 else "HOLD",
            "confidence": insights[0].confidence_score if insights else 0.5,
            "reasoning": "Based on fundamental value analysis",
            "expected_return": 0.15,  # 期望15%年回报
            "risk_level": "low",
            "time_horizon": "long",
            "position_size": 0.2  # 建议20%仓位
        }
        
        return decision
    
    async def evaluate_portfolio(self, portfolio: dict, market_conditions: dict) -> dict:
        """评估投资组合"""
        evaluation = {
            "overall_score": 0.75,
            "diversification": "adequate",
            "risk_level": "moderate",
            "recommendations": [
                "Consider adding more value stocks",
                "Reduce exposure to overvalued growth stocks",
                "Maintain cash reserves for opportunities"
            ],
            "rebalancing_needed": False
        }
        
        return evaluation


class GeorgeSorosAgent(InvestmentMasterAgent):
    """乔治·索罗斯投资大师Agent"""
    
    async def make_investment_decision(self, state: TradingState, portfolio: dict) -> dict:
        """做出投资决策（宏观策略）"""
        insights = await self.generate_insights(state)
        
        # 索罗斯风格：寻找市场错误定价
        decision = {
            "decision_id": "soros_001",
            "action": "SHORT" if self._detect_market_bubble(state) else "NEUTRAL",
            "confidence": 0.65,
            "reasoning": "Market shows signs of reflexivity",
            "expected_return": 0.30,  # 高风险高回报
            "risk_level": "high",
            "time_horizon": "short",
            "position_size": 0.1  # 较小仓位但高杠杆
        }
        
        return decision
    
    def _detect_market_bubble(self, state: TradingState) -> bool:
        """检测市场泡沫（简化版）"""
        # 这里是简化的泡沫检测逻辑
        for symbol, data in state.market_data.items():
            if data.close > data.open * 1.1:  # 涨幅超过10%
                return True
        return False
    
    async def evaluate_portfolio(self, portfolio: dict, market_conditions: dict) -> dict:
        """评估投资组合（宏观视角）"""
        return {
            "overall_score": 0.68,
            "macro_alignment": "partially aligned",
            "currency_exposure": "balanced",
            "recommendations": [
                "Increase hedge positions",
                "Consider currency diversification",
                "Monitor central bank policies"
            ],
            "rebalancing_needed": True
        }


async def main():
    """主函数"""
    # 设置日志
    setup_logging()
    
    print("=" * 60)
    print("Investment Master Agent Demo")
    print("=" * 60)
    
    # 创建配置
    buffett_config = InvestmentMasterConfig(
        name="buffett_agent",
        master_name="Warren Buffett",
        investment_style=InvestmentStyle.VALUE,
        specialty=["stocks", "long-term", "value investing"],
        llm_provider="local",  # 使用本地模拟
        llm_model="mock",
        analysis_depth="comprehensive",
        risk_tolerance="conservative",
        time_horizon="long",
        personality_traits={
            "patience": "extreme",
            "discipline": "high",
            "contrarian": "moderate"
        }
    )
    
    soros_config = InvestmentMasterConfig(
        name="soros_agent",
        master_name="George Soros",
        investment_style=InvestmentStyle.MACRO,
        specialty=["forex", "macro", "reflexivity"],
        llm_provider="local",
        llm_model="mock",
        analysis_depth="quick",
        risk_tolerance="aggressive",
        time_horizon="short",
        personality_traits={
            "flexibility": "extreme",
            "risk_taking": "high",
            "contrarian": "high"
        }
    )
    
    # 创建Agent实例
    buffett = WarrenBuffettAgent(buffett_config)
    soros = GeorgeSorosAgent(soros_config)
    
    # 初始化
    await buffett.initialize()
    await soros.initialize()
    
    print("\n✅ Agents initialized successfully")
    
    # 创建模拟交易状态
    state = TradingState()
    state.timestamp = datetime.now()
    state.active_symbols = ["BTC/USDT", "ETH/USDT"]
    state.market_data = {
        "BTC/USDT": MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            open=50000,
            high=51500,
            low=49500,
            close=51000,
            volume=1500000
        ),
        "ETH/USDT": MarketData(
            symbol="ETH/USDT",
            timestamp=datetime.now(),
            open=3000,
            high=3050,
            low=2950,
            close=2980,
            volume=800000
        )
    }
    
    # 模拟投资组合
    portfolio = {
        "total_value": 100000,
        "positions": {
            "BTC/USDT": {"quantity": 0.5, "value": 25500},
            "ETH/USDT": {"quantity": 5, "value": 14900},
            "USDT": {"quantity": 59600, "value": 59600}
        }
    }
    
    print("\n📊 Market Analysis:")
    print("-" * 40)
    
    # Buffett的分析
    print("\n🎩 Warren Buffett's Analysis:")
    buffett_insights = await buffett.generate_insights(state)
    for insight in buffett_insights:
        print(f"  Symbol: {insight.metadata.get('symbol')}")
        print(f"  Conclusion: {insight.main_conclusion}")
        print(f"  Confidence: {insight.confidence_score:.2%}")
        print(f"  Time Horizon: {insight.time_horizon}")
    
    buffett_decision = await buffett.make_investment_decision(state, portfolio)
    print(f"\n  Decision: {buffett_decision['action']}")
    print(f"  Reasoning: {buffett_decision['reasoning']}")
    
    # Soros的分析
    print("\n💼 George Soros's Analysis:")
    soros_insights = await soros.generate_insights(state)
    for insight in soros_insights:
        print(f"  Symbol: {insight.metadata.get('symbol')}")
        print(f"  Conclusion: {insight.main_conclusion}")
        print(f"  Confidence: {insight.confidence_score:.2%}")
        print(f"  Time Horizon: {insight.time_horizon}")
    
    soros_decision = await soros.make_investment_decision(state, portfolio)
    print(f"\n  Decision: {soros_decision['action']}")
    print(f"  Reasoning: {soros_decision['reasoning']}")
    
    # 投资组合评估
    print("\n📈 Portfolio Evaluation:")
    print("-" * 40)
    
    buffett_eval = await buffett.evaluate_portfolio(portfolio, {})
    print("\n🎩 Buffett's Evaluation:")
    print(f"  Overall Score: {buffett_eval['overall_score']:.2f}")
    print(f"  Recommendations:")
    for rec in buffett_eval['recommendations'][:2]:
        print(f"    - {rec}")
    
    soros_eval = await soros.evaluate_portfolio(portfolio, {})
    print("\n💼 Soros's Evaluation:")
    print(f"  Overall Score: {soros_eval['overall_score']:.2f}")
    print(f"  Recommendations:")
    for rec in soros_eval['recommendations'][:2]:
        print(f"    - {rec}")
    
    # 性能统计
    print("\n📊 Performance Statistics:")
    print("-" * 40)
    
    buffett_stats = await buffett.get_analysis_summary()
    print(f"\nBuffett Agent:")
    print(f"  LLM Calls: {buffett_stats['performance']['llm_calls']}")
    print(f"  Cache Hit Rate: {buffett_stats['performance']['cache_hit_rate']:.2%}")
    
    soros_stats = await soros.get_analysis_summary()
    print(f"\nSoros Agent:")
    print(f"  LLM Calls: {soros_stats['performance']['llm_calls']}")
    print(f"  Cache Hit Rate: {soros_stats['performance']['cache_hit_rate']:.2%}")
    
    # 关闭Agent
    await buffett.shutdown()
    await soros.shutdown()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())