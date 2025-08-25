"""
工作流结果聚合器
智能整合多个Agent的分析结果
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import statistics

from src.agents.models import (
    AnalystOpinionState, AgentConsensus, PortfolioRecommendation,
    ConfidenceLevel, RiskLevel
)
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregationWeight:
    """聚合权重配置"""
    agent_name: str
    base_weight: float  # 基础权重
    performance_weight: float  # 基于历史表现的权重
    expertise_weight: float  # 基于专业领域的权重
    final_weight: float = field(init=False)
    
    def __post_init__(self):
        """计算最终权重"""
        self.final_weight = (
            self.base_weight * 0.3 + 
            self.performance_weight * 0.4 + 
            self.expertise_weight * 0.3
        )


@dataclass
class AggregatedResult:
    """聚合结果"""
    consensus_action: str  # buy/sell/hold
    consensus_confidence: float
    weighted_score: float
    agreement_level: float  # 0-1, 1表示完全一致
    dissenting_opinions: List[Dict[str, Any]]
    key_insights: List[str]
    risk_factors: List[str]
    aggregation_method: str
    timestamp: datetime = field(default_factory=datetime.now)


class ResultAggregator:
    """结果聚合器"""
    
    def __init__(self, 
                 aggregation_method: str = "weighted_voting",
                 consensus_threshold: float = 0.6,
                 min_agreement_level: float = 0.5):
        """
        初始化聚合器
        
        Args:
            aggregation_method: 聚合方法 (weighted_voting/majority_voting/bayesian/ensemble)
            consensus_threshold: 共识阈值
            min_agreement_level: 最小一致性水平
        """
        self.aggregation_method = aggregation_method
        self.consensus_threshold = consensus_threshold
        self.min_agreement_level = min_agreement_level
        
        # Agent权重配置（可从配置文件加载）
        self.agent_weights = self._initialize_agent_weights()
        
        # 历史性能追踪
        self.performance_history: Dict[str, List[float]] = {}
    
    def _initialize_agent_weights(self) -> Dict[str, AggregationWeight]:
        """初始化Agent权重"""
        weights = {
            # 价值投资大师 - 长期稳健
            'warren_buffett': AggregationWeight(
                agent_name='warren_buffett',
                base_weight=0.9,
                performance_weight=0.85,
                expertise_weight=0.95
            ),
            'benjamin_graham': AggregationWeight(
                agent_name='benjamin_graham',
                base_weight=0.85,
                performance_weight=0.80,
                expertise_weight=0.90
            ),
            
            # 成长投资专家 - 创新导向
            'cathie_wood': AggregationWeight(
                agent_name='cathie_wood',
                base_weight=0.75,
                performance_weight=0.70,
                expertise_weight=0.85
            ),
            
            # 宏观策略大师 - 全局视角
            'ray_dalio': AggregationWeight(
                agent_name='ray_dalio',
                base_weight=0.85,
                performance_weight=0.80,
                expertise_weight=0.90
            ),
            
            # 技术分析师 - 短期趋势
            'technical_analyst': AggregationWeight(
                agent_name='technical_analyst',
                base_weight=0.70,
                performance_weight=0.75,
                expertise_weight=0.70
            ),
            
            # 量化分析师 - 数据驱动
            'quantitative_analyst': AggregationWeight(
                agent_name='quantitative_analyst',
                base_weight=0.80,
                performance_weight=0.85,
                expertise_weight=0.80
            ),
        }
        
        return weights
    
    def aggregate_opinions(self, 
                          opinions: List[AnalystOpinionState],
                          market_context: Optional[Dict[str, Any]] = None) -> AggregatedResult:
        """
        聚合多个分析师的观点
        
        Args:
            opinions: 分析师观点列表
            market_context: 市场上下文信息
            
        Returns:
            聚合结果
        """
        if not opinions:
            raise ValueError("没有可聚合的观点")
        
        logger.info(f"开始聚合 {len(opinions)} 个分析师观点...")
        
        # 根据聚合方法选择策略
        if self.aggregation_method == "weighted_voting":
            result = self._weighted_voting_aggregation(opinions, market_context)
        elif self.aggregation_method == "majority_voting":
            result = self._majority_voting_aggregation(opinions)
        elif self.aggregation_method == "bayesian":
            result = self._bayesian_aggregation(opinions, market_context)
        elif self.aggregation_method == "ensemble":
            result = self._ensemble_aggregation(opinions, market_context)
        else:
            # 默认使用加权投票
            result = self._weighted_voting_aggregation(opinions, market_context)
        
        # 提取关键洞察和风险因素
        result.key_insights = self._extract_key_insights(opinions)
        result.risk_factors = self._extract_risk_factors(opinions)
        
        logger.info(f"聚合完成: 动作={result.consensus_action}, 置信度={result.consensus_confidence:.2f}")
        
        return result
    
    def _weighted_voting_aggregation(self, 
                                    opinions: List[AnalystOpinionState],
                                    market_context: Optional[Dict[str, Any]] = None) -> AggregatedResult:
        """加权投票聚合"""
        action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        total_weight = 0.0
        opinion_details = []
        
        for opinion in opinions:
            # 获取Agent权重
            agent_weight = self.agent_weights.get(
                opinion['source'],
                AggregationWeight(opinion['source'], 0.5, 0.5, 0.5)
            )
            
            # 计算调整后的权重（考虑置信度）
            adjusted_weight = agent_weight.final_weight * opinion['confidence']
            
            # 累加动作分数
            action = opinion['rating'].lower()
            if action in action_scores:
                action_scores[action] += adjusted_weight
                total_weight += adjusted_weight
            
            opinion_details.append({
                'agent': opinion['source'],
                'action': action,
                'confidence': opinion['confidence'],
                'weight': adjusted_weight
            })
        
        # 归一化分数
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight
        
        # 确定共识动作
        consensus_action = max(action_scores, key=action_scores.get)
        consensus_confidence = action_scores[consensus_action]
        
        # 计算一致性水平
        agreement_level = self._calculate_agreement_level(opinions)
        
        # 识别异议观点
        dissenting_opinions = [
            detail for detail in opinion_details
            if detail['action'] != consensus_action and detail['confidence'] > 0.6
        ]
        
        return AggregatedResult(
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            weighted_score=consensus_confidence,
            agreement_level=agreement_level,
            dissenting_opinions=dissenting_opinions,
            key_insights=[],
            risk_factors=[],
            aggregation_method="weighted_voting"
        )
    
    def _majority_voting_aggregation(self, opinions: List[AnalystOpinionState]) -> AggregatedResult:
        """多数投票聚合"""
        # 统计各动作的投票数
        action_votes = Counter(opinion['rating'].lower() for opinion in opinions)
        
        # 找出最多投票的动作
        consensus_action = action_votes.most_common(1)[0][0]
        vote_count = action_votes[consensus_action]
        
        # 计算置信度（基于投票比例）
        consensus_confidence = vote_count / len(opinions)
        
        # 计算平均置信度
        avg_confidence = statistics.mean(opinion['confidence'] for opinion in opinions)
        
        # 计算一致性水平
        agreement_level = self._calculate_agreement_level(opinions)
        
        # 识别异议观点
        dissenting_opinions = [
            {
                'agent': opinion['source'],
                'action': opinion['rating'].lower(),
                'rationale': opinion['rationale']
            }
            for opinion in opinions
            if opinion['rating'].lower() != consensus_action
        ]
        
        return AggregatedResult(
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence * avg_confidence,
            weighted_score=consensus_confidence,
            agreement_level=agreement_level,
            dissenting_opinions=dissenting_opinions,
            key_insights=[],
            risk_factors=[],
            aggregation_method="majority_voting"
        )
    
    def _bayesian_aggregation(self, 
                             opinions: List[AnalystOpinionState],
                             market_context: Optional[Dict[str, Any]] = None) -> AggregatedResult:
        """贝叶斯聚合（考虑先验概率）"""
        # 设置先验概率（可基于历史数据）
        prior_probs = {'buy': 0.3, 'sell': 0.2, 'hold': 0.5}
        
        # 如果有市场上下文，调整先验概率
        if market_context:
            market_trend = market_context.get('trend', 'neutral')
            if market_trend == 'bullish':
                prior_probs = {'buy': 0.5, 'sell': 0.1, 'hold': 0.4}
            elif market_trend == 'bearish':
                prior_probs = {'buy': 0.1, 'sell': 0.5, 'hold': 0.4}
        
        # 计算后验概率
        posterior_probs = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
        
        for action in posterior_probs:
            # 似然函数：给定动作下，观察到当前观点的概率
            likelihood = 1.0
            for opinion in opinions:
                if opinion['rating'].lower() == action:
                    likelihood *= opinion['confidence']
                else:
                    likelihood *= (1 - opinion['confidence'] * 0.3)  # 惩罚不一致
            
            # 后验概率 ∝ 先验概率 × 似然
            posterior_probs[action] = prior_probs[action] * likelihood
        
        # 归一化
        total_prob = sum(posterior_probs.values())
        if total_prob > 0:
            for action in posterior_probs:
                posterior_probs[action] /= total_prob
        
        # 选择最高后验概率的动作
        consensus_action = max(posterior_probs, key=posterior_probs.get)
        consensus_confidence = posterior_probs[consensus_action]
        
        # 计算一致性水平
        agreement_level = self._calculate_agreement_level(opinions)
        
        return AggregatedResult(
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            weighted_score=consensus_confidence,
            agreement_level=agreement_level,
            dissenting_opinions=[],
            key_insights=[],
            risk_factors=[],
            aggregation_method="bayesian"
        )
    
    def _ensemble_aggregation(self, 
                            opinions: List[AnalystOpinionState],
                            market_context: Optional[Dict[str, Any]] = None) -> AggregatedResult:
        """集成聚合（结合多种方法）"""
        # 使用多种聚合方法
        weighted_result = self._weighted_voting_aggregation(opinions, market_context)
        majority_result = self._majority_voting_aggregation(opinions)
        bayesian_result = self._bayesian_aggregation(opinions, market_context)
        
        # 综合各方法的结果
        action_scores = {
            'buy': 0.0,
            'sell': 0.0,
            'hold': 0.0
        }
        
        # 加权平均各方法的结果
        methods_weights = {
            'weighted': 0.4,
            'majority': 0.3,
            'bayesian': 0.3
        }
        
        for action in action_scores:
            if weighted_result.consensus_action == action:
                action_scores[action] += methods_weights['weighted'] * weighted_result.consensus_confidence
            if majority_result.consensus_action == action:
                action_scores[action] += methods_weights['majority'] * majority_result.consensus_confidence
            if bayesian_result.consensus_action == action:
                action_scores[action] += methods_weights['bayesian'] * bayesian_result.consensus_confidence
        
        # 确定最终动作
        consensus_action = max(action_scores, key=action_scores.get)
        consensus_confidence = action_scores[consensus_action]
        
        # 平均一致性水平
        agreement_level = statistics.mean([
            weighted_result.agreement_level,
            majority_result.agreement_level,
            bayesian_result.agreement_level
        ])
        
        return AggregatedResult(
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            weighted_score=consensus_confidence,
            agreement_level=agreement_level,
            dissenting_opinions=weighted_result.dissenting_opinions,
            key_insights=[],
            risk_factors=[],
            aggregation_method="ensemble"
        )
    
    def _calculate_agreement_level(self, opinions: List[AnalystOpinionState]) -> float:
        """计算观点一致性水平"""
        if len(opinions) <= 1:
            return 1.0
        
        # 统计各动作的分布
        action_counts = Counter(opinion['rating'].lower() for opinion in opinions)
        
        # 计算熵作为不一致性度量
        total = len(opinions)
        entropy = 0.0
        for count in action_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # 将熵转换为一致性分数（0-1）
        max_entropy = np.log2(3)  # 三种动作的最大熵
        agreement = 1.0 - (entropy / max_entropy)
        
        return agreement
    
    def _extract_key_insights(self, opinions: List[AnalystOpinionState]) -> List[str]:
        """提取关键洞察"""
        insights = []
        
        # 提取高置信度的理由
        for opinion in opinions:
            if opinion['confidence'] > 0.7 and opinion['rationale']:
                # 简化理由为关键点
                insight = f"{opinion['analyst_name']}: {opinion['rationale'][:100]}"
                insights.append(insight)
        
        # 限制洞察数量
        return insights[:5]
    
    def _extract_risk_factors(self, opinions: List[AnalystOpinionState]) -> List[str]:
        """提取风险因素"""
        all_risks = []
        
        for opinion in opinions:
            if 'risk_factors' in opinion:
                all_risks.extend(opinion['risk_factors'])
        
        # 去重并返回最常见的风险因素
        risk_counter = Counter(all_risks)
        return [risk for risk, _ in risk_counter.most_common(5)]
    
    def build_consensus(self, 
                       opinions: List[AnalystOpinionState],
                       topic: str = "trading_decision") -> AgentConsensus:
        """
        构建Agent共识
        
        Args:
            opinions: 分析师观点列表
            topic: 共识主题
            
        Returns:
            Agent共识对象
        """
        # 聚合观点
        aggregated = self.aggregate_opinions(opinions)
        
        # 构建投票记录
        votes = {
            opinion['source']: opinion['rating'].lower()
            for opinion in opinions
        }
        
        # 构建置信度分数
        confidence_scores = {
            opinion['source']: opinion['confidence']
            for opinion in opinions
        }
        
        # 构建异议观点
        dissenting_opinions = {
            item['agent']: f"建议{item['action']}"
            for item in aggregated.dissenting_opinions
        }
        
        # 创建共识对象
        consensus = AgentConsensus(
            consensus_id=f"consensus_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            participating_agents=list(votes.keys()),
            topic=topic,
            votes=votes,
            confidence_scores=confidence_scores,
            weighted_score=aggregated.weighted_score,
            consensus_reached=aggregated.consensus_confidence >= self.consensus_threshold,
            consensus_threshold=self.consensus_threshold,
            final_decision=aggregated.consensus_action if aggregated.consensus_confidence >= self.consensus_threshold else None,
            dissenting_opinions=dissenting_opinions,
            timestamp=datetime.now()
        )
        
        return consensus
    
    def update_performance(self, agent_name: str, performance_score: float) -> None:
        """
        更新Agent性能记录
        
        Args:
            agent_name: Agent名称
            performance_score: 性能分数（0-1）
        """
        if agent_name not in self.performance_history:
            self.performance_history[agent_name] = []
        
        self.performance_history[agent_name].append(performance_score)
        
        # 保持最近100条记录
        if len(self.performance_history[agent_name]) > 100:
            self.performance_history[agent_name] = self.performance_history[agent_name][-100:]
        
        # 更新Agent权重中的性能权重
        if agent_name in self.agent_weights:
            recent_performance = statistics.mean(self.performance_history[agent_name][-10:])
            self.agent_weights[agent_name].performance_weight = recent_performance
            # 重新计算最终权重
            self.agent_weights[agent_name].__post_init__()
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return {
            'method': self.aggregation_method,
            'consensus_threshold': self.consensus_threshold,
            'min_agreement_level': self.min_agreement_level,
            'agent_weights': {
                name: {
                    'base': weight.base_weight,
                    'performance': weight.performance_weight,
                    'expertise': weight.expertise_weight,
                    'final': weight.final_weight
                }
                for name, weight in self.agent_weights.items()
            },
            'performance_history_summary': {
                name: {
                    'count': len(history),
                    'mean': statistics.mean(history) if history else 0,
                    'recent_mean': statistics.mean(history[-10:]) if history else 0
                }
                for name, history in self.performance_history.items()
            }
        }