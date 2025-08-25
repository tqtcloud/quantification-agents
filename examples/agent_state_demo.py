"""
Agent状态管理系统演示
展示Agent间的状态共享、消息传递、共识机制等功能
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, Any

from src.agents.models import (
    AgentMessage, MarketDataState, RiskAssessmentState,
    PortfolioRecommendation, ReasoningStep, FinalDecision
)
from src.agents.state_manager import AgentStateManager, StateManagerConfig
from src.utils.logger import LoggerMixin


class DemoAgent(LoggerMixin):
    """演示Agent"""
    
    def __init__(self, name: str, state_manager: AgentStateManager):
        self.name = name
        self.state_manager = state_manager
        self.session_id = None
        
    async def initialize(self, session_id: str):
        """初始化Agent"""
        self.session_id = session_id
        
        # 注册消息处理器
        async def message_handler(message: AgentMessage):
            self.log_info(f"{self.name} received message: {message.message_type} from {message.sender_agent}")
            
        self.state_manager.register_message_handler(self.name, message_handler)
        
    async def analyze_market(self) -> Dict[str, Any]:
        """分析市场数据"""
        self.log_info(f"{self.name} analyzing market...")
        
        # 模拟市场分析
        analysis = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "price": 50000 + random.uniform(-1000, 1000),
                "volume": 1000000 * random.uniform(0.8, 1.2),
                "bid": 49950,
                "ask": 50050,
                "spread": 100,
                "volatility": random.uniform(0.01, 0.05),
                "trend": random.choice(["up", "down", "sideways"]),
                "momentum": random.uniform(-1, 1),
                "timestamp": datetime.now(),
                "metadata": {"source": self.name}
            }
        }
        
        # 更新状态
        await self.state_manager.update_state(
            self.session_id,
            {"market_data": analysis},
            self.name
        )
        
        # 设置共享内存
        await self.state_manager.set_shared_memory(
            f"market_analysis_{self.name}",
            analysis,
            self.name,
            ttl_seconds=300
        )
        
        return analysis
    
    async def evaluate_risk(self) -> RiskAssessmentState:
        """评估风险"""
        self.log_info(f"{self.name} evaluating risk...")
        
        # 从共享内存获取其他Agent的分析
        other_analyses = {}
        for agent_name in ["agent1", "agent2", "agent3"]:
            if agent_name != self.name:
                analysis = await self.state_manager.get_shared_memory(
                    f"market_analysis_{agent_name}",
                    self.name
                )
                if analysis:
                    other_analyses[agent_name] = analysis
        
        # 模拟风险评估
        risk_assessment: RiskAssessmentState = {
            "risk_level": random.choice(["low", "medium", "high"]),
            "var_95": random.uniform(0.02, 0.08),
            "var_99": random.uniform(0.05, 0.12),
            "max_drawdown": random.uniform(0.05, 0.15),
            "sharpe_ratio": random.uniform(0.5, 2.0),
            "exposure_ratio": random.uniform(0.3, 0.8),
            "concentration_risk": random.uniform(0.1, 0.5),
            "liquidity_risk": random.uniform(0.1, 0.3),
            "market_risk": random.uniform(0.2, 0.6),
            "operational_risk": random.uniform(0.05, 0.2),
            "risk_factors": [
                {"factor": "market_volatility", "impact": "high"},
                {"factor": "liquidity", "impact": "medium"}
            ],
            "mitigation_strategies": ["diversification", "stop_loss"],
            "timestamp": datetime.now()
        }
        
        # 更新状态
        await self.state_manager.update_state(
            self.session_id,
            {"risk_assessment": risk_assessment},
            self.name
        )
        
        return risk_assessment
    
    async def make_recommendation(self) -> PortfolioRecommendation:
        """生成投资建议"""
        self.log_info(f"{self.name} making recommendation...")
        
        # 获取当前状态
        state = await self.state_manager.get_state(self.session_id)
        
        # 基于状态生成建议
        confidence = random.uniform(0.6, 0.95)
        action = random.choice(["buy", "sell", "hold"])
        
        recommendation: PortfolioRecommendation = {
            "symbol": "BTC/USDT",
            "action": action,
            "position_size": random.uniform(0.01, 0.1),
            "entry_price": 50000,
            "stop_loss": 48000 if action == "buy" else 52000,
            "take_profit": 52000 if action == "buy" else 48000,
            "confidence": confidence,
            "rationale": f"{self.name} recommends {action} based on analysis",
            "risk_reward_ratio": random.uniform(1.5, 3.0),
            "expected_return": random.uniform(-0.1, 0.2),
            "holding_period": random.choice(["short", "medium", "long"]),
            "priority": random.randint(1, 10)
        }
        
        # 添加到推理链
        reasoning_step: ReasoningStep = {
            "step_id": len(state.get("reasoning_chain", [])) + 1,
            "agent_name": self.name,
            "action": "recommendation",
            "input_data": {"state": "analyzed"},
            "output_data": {"recommendation": action},
            "confidence": confidence,
            "reasoning": f"Based on market analysis and risk assessment",
            "timestamp": datetime.now(),
            "duration_ms": random.uniform(50, 200)
        }
        
        # 合并状态更新
        await self.state_manager.merge_states(
            self.session_id,
            {
                "portfolio_recommendations": [recommendation],
                "reasoning_chain": [reasoning_step],
                "confidence_scores": {self.name: confidence}
            },
            self.name
        )
        
        # 发送消息通知其他Agent
        message = AgentMessage(
            sender_agent=self.name,
            receiver_agent=None,  # 广播
            message_type="recommendation",
            payload={"action": action, "confidence": confidence}
        )
        await self.state_manager.send_message(message)
        
        return recommendation


async def demonstrate_consensus():
    """演示共识机制"""
    print("\n=== 共识机制演示 ===")
    
    # 创建状态管理器
    config = StateManagerConfig(
        enable_persistence=True,
        storage_path="data/demo_states",
        auto_cleanup_interval=0
    )
    state_manager = AgentStateManager(config)
    await state_manager.initialize()
    
    # 创建会话
    session_id = "consensus_demo"
    state_manager.create_initial_state(session_id)
    
    # 发起共识
    agents = ["agent1", "agent2", "agent3", "agent4", "agent5"]
    consensus_id = await state_manager.initiate_consensus(
        "Should we execute the trade?",
        agents,
        threshold=0.6
    )
    
    print(f"发起共识: {consensus_id}")
    print("参与Agent:", agents)
    
    # 模拟投票
    votes = [
        ("agent1", "approve", 0.9),
        ("agent2", "approve", 0.8),
        ("agent3", "approve", 0.7),
        ("agent4", "reject", 0.6),
        ("agent5", "approve", 0.85)
    ]
    
    for agent, vote, confidence in votes:
        await state_manager.submit_vote(consensus_id, agent, vote, confidence)
        print(f"  {agent} 投票: {vote} (置信度: {confidence})")
    
    # 获取结果
    result = await state_manager.get_consensus_result(consensus_id)
    
    print(f"\n共识结果:")
    print(f"  达成共识: {result.consensus_reached}")
    print(f"  最终决定: {result.final_decision}")
    print(f"  加权分数: {result.weighted_score:.2f}")
    if result.dissenting_opinions:
        print(f"  异议: {result.dissenting_opinions}")
    
    await state_manager.shutdown()


async def demonstrate_state_management():
    """演示状态管理"""
    print("\n=== 状态管理演示 ===")
    
    # 创建状态管理器
    config = StateManagerConfig(
        enable_persistence=True,
        storage_path="data/demo_states",
        checkpoint_interval=60,
        auto_cleanup_interval=0
    )
    state_manager = AgentStateManager(config)
    await state_manager.initialize()
    
    # 创建会话和Agents
    session_id = "demo_session"
    state_manager.create_initial_state(session_id)
    
    agents = []
    for i in range(1, 4):
        agent = DemoAgent(f"agent{i}", state_manager)
        await agent.initialize(session_id)
        agents.append(agent)
    
    print(f"创建会话: {session_id}")
    print(f"创建Agents: {[a.name for a in agents]}")
    
    # 各Agent执行分析
    print("\n1. 市场分析阶段:")
    for agent in agents:
        analysis = await agent.analyze_market()
        print(f"  {agent.name} 完成市场分析")
    
    # 风险评估
    print("\n2. 风险评估阶段:")
    for agent in agents:
        risk = await agent.evaluate_risk()
        print(f"  {agent.name} 风险评估: {risk['risk_level']}")
    
    # 生成建议
    print("\n3. 投资建议阶段:")
    for agent in agents:
        recommendation = await agent.make_recommendation()
        print(f"  {agent.name} 建议: {recommendation['action']} (置信度: {recommendation['confidence']:.2f})")
    
    # 创建检查点
    checkpoint_id = await state_manager.create_checkpoint(session_id, "demo_checkpoint", True)
    print(f"\n创建检查点: {checkpoint_id}")
    
    # 获取统计信息
    stats = state_manager.get_state_statistics(session_id)
    print("\n状态统计:")
    print(f"  状态版本: {stats['state_version']}")
    print(f"  历史记录: {stats['history_size']}")
    print(f"  市场数据: {stats['market_data_symbols']} 个符号")
    print(f"  推理步骤: {stats['reasoning_steps']}")
    print(f"  活跃Agent: {stats['active_agents']}")
    
    # 获取推理摘要
    reasoning = state_manager.get_reasoning_summary(session_id)
    print("\n推理链摘要:")
    for step in reasoning[-3:]:  # 显示最后3步
        print(f"  步骤{step['step_id']}: {step['agent_name']} - {step['action']} (置信度: {step['confidence']:.2f})")
    
    # 性能统计
    print("\nAgent性能:")
    for agent in agents:
        state_manager.update_agent_performance(
            agent.name,
            True,
            random.uniform(0.7, 0.9),
            random.uniform(50, 200),
            random.uniform(-100, 500)
        )
        
        perf = state_manager.get_agent_performance(agent.name)
        if perf:
            print(f"  {agent.name}: 成功率 {perf.win_rate:.2%}, 平均置信度 {perf.average_confidence:.2f}")
    
    # 保存状态
    await state_manager.save_state(session_id)
    print(f"\n状态已保存到磁盘")
    
    await state_manager.shutdown()


async def demonstrate_shared_memory():
    """演示共享内存"""
    print("\n=== 共享内存演示 ===")
    
    state_manager = AgentStateManager()
    await state_manager.initialize()
    
    # Agent1 设置共享数据
    print("Agent1 设置共享数据:")
    await state_manager.set_shared_memory(
        "market_signal",
        {"signal": "buy", "strength": 0.8},
        "agent1",
        ttl_seconds=60
    )
    print("  设置 market_signal = {'signal': 'buy', 'strength': 0.8}")
    
    # Agent2 读取共享数据
    data = await state_manager.get_shared_memory("market_signal", "agent2")
    print(f"\nAgent2 读取共享数据:")
    print(f"  market_signal = {data}")
    
    # 锁定机制演示
    print("\n锁定机制:")
    locked = await state_manager.lock_shared_memory("market_signal", "agent2")
    print(f"  Agent2 锁定: {locked}")
    
    locked = await state_manager.lock_shared_memory("market_signal", "agent3")
    print(f"  Agent3 尝试锁定: {locked} (应该失败)")
    
    unlocked = await state_manager.unlock_shared_memory("market_signal", "agent2")
    print(f"  Agent2 解锁: {unlocked}")
    
    locked = await state_manager.lock_shared_memory("market_signal", "agent3")
    print(f"  Agent3 再次锁定: {locked} (应该成功)")
    
    await state_manager.shutdown()


async def main():
    """主函数"""
    print("=" * 60)
    print("Agent状态管理系统演示")
    print("=" * 60)
    
    # 演示各个功能
    await demonstrate_state_management()
    await demonstrate_consensus()
    await demonstrate_shared_memory()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())