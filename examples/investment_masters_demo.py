"""
投资大师Agent集群演示
展示15个专业分析师Agent的协同决策
"""

import asyncio
from datetime import datetime
from typing import Dict, List
import json

from src.agents.agent_registry import AgentRegistry, AgentOrchestrator
from src.agents.investment_masters import (
    WarrenBuffettAgent,
    CathieWoodAgent,
    RayDalioAgent,
    BenjaminGrahamAgent,
    TechnicalAnalystAgent,
    QuantitativeAnalystAgent,
)
from src.core.models import TradingState, MarketData
from src.utils.logger import setup_logging


async def create_sample_market_data() -> TradingState:
    """创建示例市场数据"""
    state = TradingState()
    
    # 模拟不同类型的投资标的
    market_scenarios = {
        "AAPL": {  # 价值股示例
            "open": 150.0, "high": 155.0, "low": 149.0, "close": 153.0,
            "volume": 75000000, "pe_ratio": 25, "pb_ratio": 35
        },
        "TSLA": {  # 成长股示例
            "open": 240.0, "high": 250.0, "low": 235.0, "close": 245.0,
            "volume": 100000000, "pe_ratio": 60, "pb_ratio": 10
        },
        "JPM": {  # 价值银行股
            "open": 145.0, "high": 147.0, "low": 144.0, "close": 146.0,
            "volume": 12000000, "pe_ratio": 10, "pb_ratio": 1.2
        },
        "NVDA": {  # AI概念股
            "open": 450.0, "high": 465.0, "low": 445.0, "close": 460.0,
            "volume": 50000000, "pe_ratio": 65, "pb_ratio": 25
        },
        "GLD": {  # 黄金ETF
            "open": 180.0, "high": 181.0, "low": 179.0, "close": 180.5,
            "volume": 8000000, "pe_ratio": 0, "pb_ratio": 0
        }
    }
    
    for symbol, data in market_scenarios.items():
        market_data = MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )
        # 添加额外的元数据
        market_data.metadata = {
            "pe_ratio": data.get("pe_ratio", 0),
            "pb_ratio": data.get("pb_ratio", 0)
        }
        state.market_data[symbol] = market_data
    
    state.active_symbols = list(market_scenarios.keys())
    return state


async def demonstrate_individual_agents(state: TradingState):
    """演示各个投资大师Agent的独立决策"""
    print("\n" + "="*60)
    print("📊 各投资大师独立分析")
    print("="*60)
    
    portfolio = {
        "total_value": 1000000,
        "cash": 300000,
        "positions": {
            "AAPL": {"value": 200000, "shares": 1300},
            "TSLA": {"value": 150000, "shares": 600},
            "JPM": {"value": 100000, "shares": 680},
            "GLD": {"value": 250000, "shares": 1380}
        }
    }
    
    agents = [
        ("Warren Buffett", WarrenBuffettAgent()),
        ("Cathie Wood", CathieWoodAgent()),
        ("Ray Dalio", RayDalioAgent()),
        ("Benjamin Graham", BenjaminGrahamAgent()),
        ("Technical Analyst", TechnicalAnalystAgent()),
        ("Quantitative Analyst", QuantitativeAnalystAgent()),
    ]
    
    decisions = []
    
    for name, agent in agents:
        print(f"\n🎯 {name} 分析中...")
        
        try:
            decision = await agent.make_investment_decision(state, portfolio)
            
            print(f"   决策: {decision.action}")
            print(f"   信心度: {decision.confidence:.2%}")
            print(f"   时间跨度: {decision.time_horizon}")
            print(f"   风险评估: {decision.risk_assessment}")
            
            # 显示关键推理步骤
            if decision.reasoning_chain:
                print(f"   关键洞察:")
                for i, step in enumerate(decision.reasoning_chain[:2], 1):
                    print(f"      {i}. {step.thought}")
            
            decisions.append((name, decision))
            
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")
    
    return decisions


async def demonstrate_consensus_decision(state: TradingState):
    """演示共识决策机制"""
    print("\n" + "="*60)
    print("🤝 投资大师共识决策")
    print("="*60)
    
    registry = AgentRegistry()
    orchestrator = AgentOrchestrator(registry)
    
    portfolio = {
        "total_value": 1000000,
        "cash": 300000,
        "positions": {}
    }
    
    # 不同组合的Agent进行共识
    consensus_groups = [
        {
            "name": "价值投资组",
            "agents": ["warren_buffett", "benjamin_graham"],
            "method": "confidence_weighted"
        },
        {
            "name": "成长与创新组",
            "agents": ["cathie_wood"],
            "method": "majority_vote"
        },
        {
            "name": "量化技术组",
            "agents": ["technical_analyst", "quantitative_analyst"],
            "method": "weighted_average"
        },
        {
            "name": "全体共识",
            "agents": ["warren_buffett", "cathie_wood", "ray_dalio", 
                      "technical_analyst", "quantitative_analyst"],
            "method": "ensemble"
        }
    ]
    
    for group in consensus_groups:
        print(f"\n📈 {group['name']} ({group['method']})")
        print("-" * 40)
        
        try:
            consensus = await orchestrator.get_consensus_decision(
                state,
                group["agents"],
                method=group["method"],
                portfolio=portfolio
            )
            
            print(f"   共识决策: {consensus['action']}")
            print(f"   共识信心度: {consensus['confidence']:.2%}")
            print(f"   参与者数量: {consensus['participants']}")
            
            if "votes" in consensus:
                print(f"   投票分布: {consensus['votes']}")
            elif "scores" in consensus:
                print(f"   评分分布: {json.dumps(consensus['scores'], indent=6)}")
            
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")


async def demonstrate_portfolio_evaluation():
    """演示组合评估功能"""
    print("\n" + "="*60)
    print("💼 投资组合评估")
    print("="*60)
    
    # 创建一个示例组合
    portfolio = {
        "total_value": 2000000,
        "cash": 200000,
        "positions": {
            "AAPL": {
                "value": 500000,
                "shares": 3250,
                "avg_cost": 145,
                "unrealized_return": 0.055,
                "holding_period_days": 180
            },
            "TSLA": {
                "value": 300000,
                "shares": 1200,
                "avg_cost": 220,
                "unrealized_return": 0.136,
                "holding_period_days": 90
            },
            "BRK.B": {
                "value": 400000,
                "shares": 1150,
                "avg_cost": 340,
                "unrealized_return": 0.024,
                "holding_period_days": 365
            },
            "NVDA": {
                "value": 300000,
                "shares": 650,
                "avg_cost": 400,
                "unrealized_return": 0.154,
                "holding_period_days": 60
            },
            "GLD": {
                "value": 200000,
                "shares": 1100,
                "avg_cost": 175,
                "unrealized_return": 0.037,
                "holding_period_days": 120
            },
            "BONDS": {
                "value": 100000,
                "shares": 1000,
                "avg_cost": 98,
                "unrealized_return": 0.020,
                "holding_period_days": 200
            }
        },
        "performance": {
            "return": 0.082,
            "sharpe": 1.2,
            "max_drawdown": -0.15
        }
    }
    
    market_conditions = {
        "sentiment": "neutral",
        "volatility": "moderate",
        "trend": "sideways",
        "major_events": ["Fed meeting next week", "Earnings season"]
    }
    
    evaluators = [
        ("Warren Buffett", WarrenBuffettAgent()),
        ("Cathie Wood", CathieWoodAgent()),
        ("Ray Dalio", RayDalioAgent()),
        ("Technical Analyst", TechnicalAnalystAgent()),
        ("Quantitative Analyst", QuantitativeAnalystAgent()),
    ]
    
    for name, agent in evaluators:
        print(f"\n📊 {name} 的组合评估")
        print("-" * 40)
        
        try:
            evaluation = await agent.evaluate_portfolio(portfolio, market_conditions)
            
            print(f"   总体评分: {evaluation.get('overall_score', 0):.2f}")
            
            # 显示关键指标
            for key, value in evaluation.items():
                if key not in ["overall_score", "recommendations"] and isinstance(value, (int, float)):
                    print(f"   {key}: {value:.2f}")
            
            # 显示建议
            if "recommendations" in evaluation and evaluation["recommendations"]:
                print(f"   建议:")
                for rec in evaluation["recommendations"][:3]:
                    print(f"      • {rec}")
            
            # 显示特定的智慧/建议
            for wisdom_key in ["buffett_wisdom", "ark_perspective", "dalio_principles", 
                              "technical_advice", "quant_advice"]:
                if wisdom_key in evaluation:
                    print(f"   💡 {evaluation[wisdom_key]}")
                    break
            
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")


async def main():
    """主演示程序"""
    # 设置日志
    setup_logging("investment_masters_demo")
    
    print("\n" + "="*60)
    print("🏛️  量化交易系统 - 投资大师Agent集群演示")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建市场数据
    print("\n📈 准备市场数据...")
    state = await create_sample_market_data()
    print(f"   已加载 {len(state.active_symbols)} 个交易标的")
    
    # 1. 展示各个Agent的独立决策
    decisions = await demonstrate_individual_agents(state)
    
    # 2. 展示共识决策机制
    await demonstrate_consensus_decision(state)
    
    # 3. 展示组合评估
    await demonstrate_portfolio_evaluation()
    
    # 4. 总结
    print("\n" + "="*60)
    print("📋 演示总结")
    print("="*60)
    
    print("\n已实现的投资大师Agent:")
    print("  ✅ Warren Buffett - 价值投资，长期持有")
    print("  ✅ Cathie Wood - 颠覆性创新，高增长")
    print("  ✅ Ray Dalio - 全天候策略，风险平价")
    print("  ✅ Benjamin Graham - 安全边际，防御投资")
    print("  ✅ Technical Analyst - 技术分析，图表模式")
    print("  ✅ Quantitative Analyst - 量化策略，统计套利")
    
    print("\n核心功能:")
    print("  ✅ 独立决策分析")
    print("  ✅ 多种共识机制")
    print("  ✅ 组合评估建议")
    print("  ✅ Agent注册管理")
    print("  ✅ 性能统计跟踪")
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())