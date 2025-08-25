"""
Agent注册和管理系统
统一管理所有投资大师Agent的创建、配置和调度
"""

from typing import Any, Dict, List, Optional, Type
from datetime import datetime
import asyncio
import json

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig
from src.agents.investment_masters import (
    WarrenBuffettAgent,
    CathieWoodAgent,
    RayDalioAgent,
    BenjaminGrahamAgent,
    TechnicalAnalystAgent,
    QuantitativeAnalystAgent,
)
from src.core.models import TradingState, Signal
from src.utils.logger import LoggerMixin


class AgentRegistry(LoggerMixin):
    """Agent注册管理器"""
    
    def __init__(self):
        """初始化Agent注册器"""
        super().__init__()
        
        # Agent类注册表
        self._agent_classes: Dict[str, Type[InvestmentMasterAgent]] = {}
        
        # 活跃的Agent实例
        self._active_agents: Dict[str, InvestmentMasterAgent] = {}
        
        # Agent配置
        self._agent_configs: Dict[str, InvestmentMasterConfig] = {}
        
        # Agent性能统计
        self._agent_stats: Dict[str, Dict[str, Any]] = {}
        
        # 初始化内置Agent
        self._register_builtin_agents()
        
        self.log_info("Agent Registry initialized")
    
    def _register_builtin_agents(self):
        """注册内置的投资大师Agent"""
        builtin_agents = {
            # 价值投资类
            "warren_buffett": WarrenBuffettAgent,
            "benjamin_graham": BenjaminGrahamAgent,
            
            # 成长投资类
            "cathie_wood": CathieWoodAgent,
            
            # 宏观策略类
            "ray_dalio": RayDalioAgent,
            
            # 专业分析师类
            "technical_analyst": TechnicalAnalystAgent,
            "quantitative_analyst": QuantitativeAnalystAgent,
        }
        
        for name, agent_class in builtin_agents.items():
            self.register_agent_class(name, agent_class)
            self.log_debug(f"Registered builtin agent: {name}")
    
    def register_agent_class(self, name: str, agent_class: Type[InvestmentMasterAgent]):
        """注册Agent类"""
        if name in self._agent_classes:
            self.log_warning(f"Overwriting existing agent class: {name}")
        
        self._agent_classes[name] = agent_class
        self._agent_stats[name] = {
            "created_at": datetime.now().isoformat(),
            "total_calls": 0,
            "total_signals": 0,
            "avg_confidence": 0.0,
            "success_rate": 0.0
        }
    
    def create_agent(self, name: str, config: Optional[InvestmentMasterConfig] = None,
                    message_bus=None) -> InvestmentMasterAgent:
        """创建Agent实例"""
        if name not in self._agent_classes:
            raise ValueError(f"Unknown agent: {name}")
        
        agent_class = self._agent_classes[name]
        
        # 使用提供的配置或默认配置
        if config is None and name in self._agent_configs:
            config = self._agent_configs[name]
        
        # 创建Agent实例
        agent = agent_class(config=config, message_bus=message_bus)
        
        # 注册为活跃Agent
        agent_id = f"{name}_{datetime.now().timestamp()}"
        self._active_agents[agent_id] = agent
        
        # 更新统计
        self._agent_stats[name]["total_calls"] += 1
        
        self.log_info(f"Created agent instance: {agent_id}")
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[InvestmentMasterAgent]:
        """获取活跃的Agent实例"""
        return self._active_agents.get(agent_id)
    
    def list_available_agents(self) -> List[str]:
        """列出所有可用的Agent类型"""
        return list(self._agent_classes.keys())
    
    def list_active_agents(self) -> List[str]:
        """列出所有活跃的Agent实例"""
        return list(self._active_agents.keys())
    
    def set_agent_config(self, name: str, config: InvestmentMasterConfig):
        """设置Agent的默认配置"""
        self._agent_configs[name] = config
        self.log_debug(f"Set config for agent: {name}")
    
    def get_agent_config(self, name: str) -> Optional[InvestmentMasterConfig]:
        """获取Agent的默认配置"""
        return self._agent_configs.get(name)
    
    async def initialize_all_agents(self):
        """初始化所有活跃的Agent"""
        tasks = []
        for agent_id, agent in self._active_agents.items():
            if hasattr(agent, '_initialize'):
                tasks.append(agent._initialize())
        
        if tasks:
            await asyncio.gather(*tasks)
            self.log_info(f"Initialized {len(tasks)} agents")
    
    def remove_agent(self, agent_id: str) -> bool:
        """移除Agent实例"""
        if agent_id in self._active_agents:
            del self._active_agents[agent_id]
            self.log_info(f"Removed agent: {agent_id}")
            return True
        return False
    
    def clear_inactive_agents(self):
        """清理非活跃的Agent实例"""
        # 这里可以实现基于时间或其他条件的清理逻辑
        removed = 0
        for agent_id in list(self._active_agents.keys()):
            # 简单示例：清理所有Agent
            # 实际应该基于最后活动时间等条件
            pass
        
        if removed > 0:
            self.log_info(f"Cleared {removed} inactive agents")
    
    def get_agent_statistics(self, name: str) -> Dict[str, Any]:
        """获取Agent的统计信息"""
        return self._agent_stats.get(name, {})
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Agent的统计信息"""
        return self._agent_stats.copy()
    
    def update_agent_statistics(self, name: str, signals: List[Signal]):
        """更新Agent统计信息"""
        if name not in self._agent_stats:
            return
        
        stats = self._agent_stats[name]
        stats["total_signals"] += len(signals)
        
        if signals:
            avg_conf = sum(s.confidence for s in signals) / len(signals)
            # 更新移动平均
            alpha = 0.1  # 平滑系数
            stats["avg_confidence"] = (1 - alpha) * stats["avg_confidence"] + alpha * avg_conf
    
    def save_registry_state(self, filepath: str):
        """保存注册器状态"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "available_agents": self.list_available_agents(),
            "active_agents": self.list_active_agents(),
            "statistics": self.get_all_statistics(),
            "configs": {
                name: config.dict() if hasattr(config, 'dict') else {}
                for name, config in self._agent_configs.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.log_info(f"Saved registry state to {filepath}")
    
    def load_registry_state(self, filepath: str):
        """加载注册器状态"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # 恢复统计信息
            if "statistics" in state:
                self._agent_stats.update(state["statistics"])
            
            self.log_info(f"Loaded registry state from {filepath}")
            
        except Exception as e:
            self.log_error(f"Failed to load registry state: {e}")


class AgentOrchestrator(LoggerMixin):
    """Agent编排器 - 协调多个Agent的决策"""
    
    def __init__(self, registry: AgentRegistry):
        """初始化编排器"""
        super().__init__()
        self.registry = registry
        self.consensus_methods = {
            "majority_vote": self._majority_vote,
            "weighted_average": self._weighted_average,
            "confidence_weighted": self._confidence_weighted,
            "ensemble": self._ensemble_decision
        }
        self.log_info("Agent Orchestrator initialized")
    
    async def get_consensus_decision(self, state: TradingState, 
                                    agent_names: List[str],
                                    method: str = "confidence_weighted",
                                    portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """获取多个Agent的共识决策"""
        if method not in self.consensus_methods:
            raise ValueError(f"Unknown consensus method: {method}")
        
        # 创建并初始化Agent
        agents = []
        for name in agent_names:
            agent = self.registry.create_agent(name)
            agents.append((name, agent))
        
        # 并行获取所有Agent的决策
        tasks = []
        for name, agent in agents:
            if hasattr(agent, 'make_investment_decision'):
                tasks.append(agent.make_investment_decision(state, portfolio or {}))
        
        decisions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉异常
        valid_decisions = []
        for i, decision in enumerate(decisions):
            if not isinstance(decision, Exception):
                valid_decisions.append((agent_names[i], decision))
            else:
                self.log_error(f"Agent {agent_names[i]} failed: {decision}")
        
        # 应用共识方法
        consensus = self.consensus_methods[method](valid_decisions)
        
        # 清理临时Agent
        for name, agent in agents:
            # 实际实现中可能需要更复杂的清理逻辑
            pass
        
        return consensus
    
    def _majority_vote(self, decisions: List[tuple]) -> Dict[str, Any]:
        """多数投票法"""
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for name, decision in decisions:
            action = decision.action
            votes[action] = votes.get(action, 0) + 1
        
        # 找出得票最多的行动
        consensus_action = max(votes.items(), key=lambda x: x[1])[0]
        
        # 计算共识置信度
        total_votes = sum(votes.values())
        consensus_confidence = votes[consensus_action] / total_votes if total_votes > 0 else 0
        
        return {
            "action": consensus_action,
            "confidence": consensus_confidence,
            "method": "majority_vote",
            "votes": votes,
            "participants": len(decisions)
        }
    
    def _weighted_average(self, decisions: List[tuple]) -> Dict[str, Any]:
        """加权平均法"""
        action_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
        total_weight = 0
        
        for name, decision in decisions:
            weight = 1.0  # 可以基于Agent历史表现设置权重
            action_scores[decision.action] += weight
            total_weight += weight
        
        # 归一化
        for action in action_scores:
            action_scores[action] /= max(total_weight, 1)
        
        consensus_action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "action": consensus_action,
            "confidence": action_scores[consensus_action],
            "method": "weighted_average",
            "scores": action_scores,
            "participants": len(decisions)
        }
    
    def _confidence_weighted(self, decisions: List[tuple]) -> Dict[str, Any]:
        """置信度加权法"""
        action_confidence = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for name, decision in decisions:
            action = decision.action
            confidence = decision.confidence
            action_confidence[action] += confidence
        
        # 找出置信度最高的行动
        consensus_action = max(action_confidence.items(), key=lambda x: x[1])[0]
        
        # 归一化置信度
        total_confidence = sum(action_confidence.values())
        if total_confidence > 0:
            consensus_confidence = action_confidence[consensus_action] / total_confidence
        else:
            consensus_confidence = 0
        
        return {
            "action": consensus_action,
            "confidence": consensus_confidence,
            "method": "confidence_weighted",
            "confidence_scores": action_confidence,
            "participants": len(decisions)
        }
    
    def _ensemble_decision(self, decisions: List[tuple]) -> Dict[str, Any]:
        """集成决策法 - 结合多种方法"""
        # 应用多种共识方法
        majority = self._majority_vote(decisions)
        weighted = self._weighted_average(decisions)
        confidence = self._confidence_weighted(decisions)
        
        # 综合各方法的结果
        final_scores = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for method_result in [majority, weighted, confidence]:
            action = method_result["action"]
            conf = method_result["confidence"]
            final_scores[action] += conf
        
        # 归一化并选择最终行动
        total_score = sum(final_scores.values())
        if total_score > 0:
            for action in final_scores:
                final_scores[action] /= total_score
        
        consensus_action = max(final_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "action": consensus_action,
            "confidence": final_scores[consensus_action],
            "method": "ensemble",
            "ensemble_scores": final_scores,
            "sub_methods": {
                "majority": majority["action"],
                "weighted": weighted["action"],
                "confidence": confidence["action"]
            },
            "participants": len(decisions)
        }


# 全局注册器实例
agent_registry = AgentRegistry()
agent_orchestrator = AgentOrchestrator(agent_registry)