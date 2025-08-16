"""
决策聚合器实现
综合多个Agent信号进行决策，实现信号权重和置信度计算，添加决策冲突解决机制
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.core.models import Signal, OrderSide, RiskMetrics
from src.core.state_management import TradingStateGraph
from src.utils.logger import LoggerMixin


class AggregationMethod(Enum):
    """聚合方法枚举"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RISK_ADJUSTED = "risk_adjusted"
    ENSEMBLE = "ensemble"


class ConflictResolution(Enum):
    """冲突解决策略"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_CONSENSUS = "weighted_consensus"
    CONSERVATIVE = "conservative"  # 有冲突时选择保守策略
    RISK_BASED = "risk_based"      # 基于风险评估解决冲突


@dataclass
class AgentWeight:
    """Agent权重配置"""
    agent_name: str
    base_weight: float = 1.0
    performance_weight: float = 1.0  # 基于历史表现的动态权重
    risk_adjustment: float = 1.0     # 风险调整系数
    confidence_threshold: float = 0.3  # 最低置信度阈值
    
    @property
    def effective_weight(self) -> float:
        """计算有效权重"""
        return self.base_weight * self.performance_weight * self.risk_adjustment


@dataclass
class DecisionContext:
    """决策上下文"""
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_tolerance: float = 0.5
    position_limits: Dict[str, float] = field(default_factory=dict)
    blacklist_symbols: List[str] = field(default_factory=list)
    max_position_size: float = 0.1  # 最大仓位比例
    max_daily_trades: int = 10
    
    
@dataclass
class AggregatedDecision:
    """聚合决策结果"""
    action: str  # BUY, SELL, HOLD
    symbol: str
    confidence: float
    strength: float
    reasoning: str
    contributing_signals: List[Signal]
    conflict_resolution: Optional[str] = None
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self._agent_performance: Dict[str, Dict[str, float]] = {}
    
    def update_performance(self, agent_name: str, signal: Signal, outcome: float):
        """更新agent性能"""
        if agent_name not in self._agent_performance:
            self._agent_performance[agent_name] = {
                "total_signals": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "total_confidence": 0.0
            }
        
        perf = self._agent_performance[agent_name]
        perf["total_signals"] += 1
        perf["total_confidence"] += signal.confidence
        perf["avg_confidence"] = perf["total_confidence"] / perf["total_signals"]
        
        # 简化的正确性判断（实际应用中需要更复杂的逻辑）
        if outcome > 0:  # 假设正面结果表示正确预测
            perf["correct_predictions"] += 1
        
        perf["accuracy"] = perf["correct_predictions"] / perf["total_signals"]
    
    def get_performance_weight(self, agent_name: str) -> float:
        """获取基于表现的权重"""
        if agent_name not in self._agent_performance:
            return 1.0
        
        perf = self._agent_performance[agent_name]
        accuracy = perf["accuracy"]
        avg_confidence = perf["avg_confidence"]
        
        # 综合准确率和平均置信度
        weight = (accuracy * 0.7) + (avg_confidence * 0.3)
        return max(0.1, min(2.0, weight))  # 限制在0.1-2.0之间
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, float]:
        """获取agent统计信息"""
        return self._agent_performance.get(agent_name, {})


class DecisionAggregator(LoggerMixin):
    """决策聚合器"""
    
    def __init__(self, 
                 aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED,
                 conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_CONSENSUS,
                 decision_context: DecisionContext = None):
        
        self.aggregation_method = aggregation_method
        self.conflict_resolution = conflict_resolution
        self.decision_context = decision_context or DecisionContext()
        
        # Agent权重配置
        self.agent_weights: Dict[str, AgentWeight] = {}
        
        # 性能跟踪
        self.performance_tracker = PerformanceTracker()
        
        # 决策历史
        self.decision_history: List[AggregatedDecision] = []
    
    def configure_agent_weight(self, 
                             agent_name: str,
                             base_weight: float = 1.0,
                             confidence_threshold: float = 0.3):
        """配置Agent权重"""
        self.agent_weights[agent_name] = AgentWeight(
            agent_name=agent_name,
            base_weight=base_weight,
            confidence_threshold=confidence_threshold
        )
        self.log_debug(f"Configured weight for {agent_name}: {base_weight}")
    
    def aggregate_decisions(self, state: TradingStateGraph) -> Dict[str, Any]:
        """聚合决策主入口"""
        signals = state.get("signals", [])
        risk_metrics = state.get("risk_metrics")
        
        if not signals:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "No signals available"
            }
        
        # 按交易对分组信号
        signals_by_symbol = self._group_signals_by_symbol(signals)
        
        aggregated_decisions = {}
        
        for symbol, symbol_signals in signals_by_symbol.items():
            decision = self._aggregate_symbol_signals(
                symbol, symbol_signals, risk_metrics
            )
            aggregated_decisions[symbol] = decision
        
        # 选择最佳决策
        best_decision = self._select_best_decision(aggregated_decisions)
        
        # 记录决策历史
        if best_decision:
            self.decision_history.append(best_decision)
        
        return self._format_decision_output(best_decision)
    
    def _group_signals_by_symbol(self, signals: List[Signal]) -> Dict[str, List[Signal]]:
        """按交易对分组信号"""
        grouped = {}
        for signal in signals:
            symbol = signal.symbol
            if symbol not in grouped:
                grouped[symbol] = []
            grouped[symbol].append(signal)
        
        return grouped
    
    def _aggregate_symbol_signals(self, 
                                 symbol: str,
                                 signals: List[Signal],
                                 risk_metrics: Optional[RiskMetrics]) -> AggregatedDecision:
        """聚合单个交易对的信号"""
        
        # 过滤低置信度信号
        filtered_signals = self._filter_signals(signals)
        
        if not filtered_signals:
            return AggregatedDecision(
                action="HOLD",
                symbol=symbol,
                confidence=0.0,
                strength=0.0,
                reasoning="No valid signals after filtering",
                contributing_signals=[]
            )
        
        # 检测冲突
        conflicts = self._detect_conflicts(filtered_signals)
        
        # 选择聚合方法
        if conflicts:
            decision = self._resolve_conflicts(symbol, filtered_signals, conflicts)
        else:
            decision = self._apply_aggregation_method(symbol, filtered_signals)
        
        # 风险调整
        if risk_metrics:
            decision = self._apply_risk_adjustment(decision, risk_metrics)
        
        return decision
    
    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """过滤信号"""
        filtered = []
        
        for signal in signals:
            agent_weight = self.agent_weights.get(signal.source)
            
            # 检查置信度阈值
            min_confidence = (agent_weight.confidence_threshold 
                            if agent_weight else 0.3)
            
            if signal.confidence >= min_confidence:
                filtered.append(signal)
        
        return filtered
    
    def _detect_conflicts(self, signals: List[Signal]) -> List[str]:
        """检测信号冲突"""
        conflicts = []
        
        buy_signals = [s for s in signals if s.action == OrderSide.BUY]
        sell_signals = [s for s in signals if s.action == OrderSide.SELL]
        
        if len(buy_signals) > 0 and len(sell_signals) > 0:
            conflicts.append("buy_sell_conflict")
        
        # 检测强度冲突
        strengths = [s.strength for s in signals]
        if len(strengths) > 1:
            max_strength = max(strengths)
            min_strength = min(strengths)
            if abs(max_strength - min_strength) > 0.5:
                conflicts.append("strength_conflict")
        
        return conflicts
    
    def _resolve_conflicts(self, 
                          symbol: str,
                          signals: List[Signal],
                          conflicts: List[str]) -> AggregatedDecision:
        """解决冲突"""
        
        if self.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            return self._resolve_by_highest_confidence(symbol, signals)
        
        elif self.conflict_resolution == ConflictResolution.WEIGHTED_CONSENSUS:
            return self._resolve_by_weighted_consensus(symbol, signals)
        
        elif self.conflict_resolution == ConflictResolution.CONSERVATIVE:
            return self._resolve_conservatively(symbol, signals)
        
        elif self.conflict_resolution == ConflictResolution.RISK_BASED:
            return self._resolve_by_risk(symbol, signals)
        
        else:
            # 默认使用加权共识
            return self._resolve_by_weighted_consensus(symbol, signals)
    
    def _resolve_by_highest_confidence(self, 
                                     symbol: str,
                                     signals: List[Signal]) -> AggregatedDecision:
        """基于最高置信度解决冲突"""
        best_signal = max(signals, key=lambda s: s.confidence)
        
        return AggregatedDecision(
            action=best_signal.action.value,
            symbol=symbol,
            confidence=best_signal.confidence,
            strength=best_signal.strength,
            reasoning=f"Highest confidence signal from {best_signal.source}",
            contributing_signals=[best_signal],
            conflict_resolution="highest_confidence"
        )
    
    def _resolve_by_weighted_consensus(self, 
                                     symbol: str,
                                     signals: List[Signal]) -> AggregatedDecision:
        """基于加权共识解决冲突"""
        
        # 计算加权分数
        total_buy_weight = 0.0
        total_sell_weight = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0
        weighted_strength = 0.0
        
        for signal in signals:
            weight = self._get_signal_weight(signal)
            total_weight += weight
            weighted_confidence += signal.confidence * weight
            weighted_strength += signal.strength * weight
            
            if signal.action == OrderSide.BUY:
                total_buy_weight += weight * signal.strength
            elif signal.action == OrderSide.SELL:
                total_sell_weight += weight * abs(signal.strength)
        
        if total_weight == 0:
            action = "HOLD"
            final_confidence = 0.0
            final_strength = 0.0
        else:
            weighted_confidence /= total_weight
            weighted_strength /= total_weight
            
            if total_buy_weight > total_sell_weight:
                action = "BUY"
                final_confidence = weighted_confidence * (total_buy_weight / (total_buy_weight + total_sell_weight))
                final_strength = weighted_strength
            elif total_sell_weight > total_buy_weight:
                action = "SELL"
                final_confidence = weighted_confidence * (total_sell_weight / (total_buy_weight + total_sell_weight))
                final_strength = -abs(weighted_strength)
            else:
                action = "HOLD"
                final_confidence = 0.0
                final_strength = 0.0
        
        return AggregatedDecision(
            action=action,
            symbol=symbol,
            confidence=final_confidence,
            strength=final_strength,
            reasoning=f"Weighted consensus from {len(signals)} signals",
            contributing_signals=signals,
            conflict_resolution="weighted_consensus"
        )
    
    def _resolve_conservatively(self, 
                               symbol: str,
                               signals: List[Signal]) -> AggregatedDecision:
        """保守解决冲突"""
        return AggregatedDecision(
            action="HOLD",
            symbol=symbol,
            confidence=0.0,
            strength=0.0,
            reasoning="Conservative resolution due to conflicting signals",
            contributing_signals=signals,
            conflict_resolution="conservative"
        )
    
    def _resolve_by_risk(self, 
                        symbol: str,
                        signals: List[Signal]) -> AggregatedDecision:
        """基于风险解决冲突"""
        
        # 计算每个信号的风险调整分数
        risk_adjusted_signals = []
        
        for signal in signals:
            risk_score = self._calculate_signal_risk(signal)
            adjusted_confidence = signal.confidence * (1 - risk_score)
            
            risk_adjusted_signals.append({
                "signal": signal,
                "risk_score": risk_score,
                "adjusted_confidence": adjusted_confidence
            })
        
        # 选择风险调整后置信度最高的信号
        best_adjusted = max(risk_adjusted_signals, key=lambda x: x["adjusted_confidence"])
        best_signal = best_adjusted["signal"]
        
        return AggregatedDecision(
            action=best_signal.action.value,
            symbol=symbol,
            confidence=best_adjusted["adjusted_confidence"],
            strength=best_signal.strength,
            reasoning=f"Risk-adjusted decision from {best_signal.source}",
            contributing_signals=[best_signal],
            conflict_resolution="risk_based",
            risk_assessment={"risk_score": best_adjusted["risk_score"]}
        )
    
    def _apply_aggregation_method(self, 
                                 symbol: str,
                                 signals: List[Signal]) -> AggregatedDecision:
        """应用聚合方法"""
        
        if self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(symbol, signals)
        
        elif self.aggregation_method == AggregationMethod.MAJORITY_VOTE:
            return self._majority_vote_aggregation(symbol, signals)
        
        elif self.aggregation_method == AggregationMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_aggregation(symbol, signals)
        
        elif self.aggregation_method == AggregationMethod.RISK_ADJUSTED:
            return self._risk_adjusted_aggregation(symbol, signals)
        
        elif self.aggregation_method == AggregationMethod.ENSEMBLE:
            return self._ensemble_aggregation(symbol, signals)
        
        else:
            # 默认使用置信度加权
            return self._confidence_weighted_aggregation(symbol, signals)
    
    def _weighted_average_aggregation(self, 
                                    symbol: str,
                                    signals: List[Signal]) -> AggregatedDecision:
        """加权平均聚合"""
        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            weight = self._get_signal_weight(signal)
            total_weight += weight
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight
        
        if total_weight == 0:
            action = "HOLD"
            final_confidence = 0.0
            final_strength = 0.0
        else:
            final_strength = weighted_strength / total_weight
            final_confidence = weighted_confidence / total_weight
            
            if final_strength > 0.1:
                action = "BUY"
            elif final_strength < -0.1:
                action = "SELL"
            else:
                action = "HOLD"
        
        return AggregatedDecision(
            action=action,
            symbol=symbol,
            confidence=final_confidence,
            strength=final_strength,
            reasoning=f"Weighted average of {len(signals)} signals",
            contributing_signals=signals
        )
    
    def _confidence_weighted_aggregation(self, 
                                       symbol: str,
                                       signals: List[Signal]) -> AggregatedDecision:
        """置信度加权聚合"""
        total_confidence = sum(s.confidence for s in signals)
        
        if total_confidence == 0:
            return AggregatedDecision(
                action="HOLD",
                symbol=symbol,
                confidence=0.0,
                strength=0.0,
                reasoning="No confidence in signals",
                contributing_signals=signals
            )
        
        weighted_strength = sum(s.strength * s.confidence for s in signals) / total_confidence
        avg_confidence = total_confidence / len(signals)
        
        if weighted_strength > 0.1:
            action = "BUY"
        elif weighted_strength < -0.1:
            action = "SELL"
        else:
            action = "HOLD"
        
        return AggregatedDecision(
            action=action,
            symbol=symbol,
            confidence=avg_confidence,
            strength=weighted_strength,
            reasoning=f"Confidence-weighted aggregation of {len(signals)} signals",
            contributing_signals=signals
        )
    
    def _get_signal_weight(self, signal: Signal) -> float:
        """获取信号权重"""
        agent_weight = self.agent_weights.get(signal.source)
        
        if agent_weight:
            # 更新性能权重
            perf_weight = self.performance_tracker.get_performance_weight(signal.source)
            agent_weight.performance_weight = perf_weight
            
            return agent_weight.effective_weight
        else:
            # 默认权重
            return 1.0
    
    def _calculate_signal_risk(self, signal: Signal) -> float:
        """计算信号风险"""
        # 简化的风险计算
        base_risk = 0.1
        
        # 基于强度的风险
        strength_risk = abs(signal.strength) * 0.1
        
        # 基于置信度的风险（低置信度高风险）
        confidence_risk = (1 - signal.confidence) * 0.2
        
        return min(1.0, base_risk + strength_risk + confidence_risk)
    
    def _apply_risk_adjustment(self, 
                              decision: AggregatedDecision,
                              risk_metrics: RiskMetrics) -> AggregatedDecision:
        """应用风险调整"""
        
        if not risk_metrics:
            return decision
        
        # 风险调整因子
        risk_factor = 1.0
        
        # 基于当前回撤调整
        if hasattr(risk_metrics, 'current_drawdown'):
            if risk_metrics.current_drawdown > 0.1:  # 10%回撤
                risk_factor *= 0.5
            elif risk_metrics.current_drawdown > 0.05:  # 5%回撤
                risk_factor *= 0.7
        
        # 基于杠杆率调整
        if hasattr(risk_metrics, 'leverage_ratio'):
            if risk_metrics.leverage_ratio > 3:
                risk_factor *= 0.6
            elif risk_metrics.leverage_ratio > 2:
                risk_factor *= 0.8
        
        # 应用风险调整
        decision.confidence *= risk_factor
        decision.strength *= risk_factor
        
        if decision.confidence < 0.3:
            decision.action = "HOLD"
            decision.reasoning += " (Risk-adjusted to HOLD)"
        
        decision.risk_assessment = {
            "risk_factor": risk_factor,
            "original_confidence": decision.confidence / risk_factor if risk_factor > 0 else 0
        }
        
        return decision
    
    def _select_best_decision(self, 
                            decisions: Dict[str, AggregatedDecision]) -> Optional[AggregatedDecision]:
        """选择最佳决策"""
        
        if not decisions:
            return None
        
        # 过滤掉HOLD决策
        actionable_decisions = {k: v for k, v in decisions.items() 
                              if v.action != "HOLD"}
        
        if not actionable_decisions:
            # 如果没有可操作的决策，返回置信度最高的HOLD决策
            return max(decisions.values(), key=lambda d: d.confidence)
        
        # 选择置信度最高的可操作决策
        best_decision = max(actionable_decisions.values(), 
                          key=lambda d: d.confidence * abs(d.strength))
        
        return best_decision
    
    def _format_decision_output(self, decision: Optional[AggregatedDecision]) -> Dict[str, Any]:
        """格式化决策输出"""
        
        if not decision:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "No valid decision"
            }
        
        return {
            "action": decision.action,
            "symbol": decision.symbol,
            "confidence": decision.confidence,
            "strength": decision.strength,
            "reason": decision.reasoning,
            "metadata": {
                "contributing_signals_count": len(decision.contributing_signals),
                "conflict_resolution": decision.conflict_resolution,
                "risk_assessment": decision.risk_assessment,
                "timestamp": decision.timestamp.isoformat()
            }
        }
    
    def update_agent_performance(self, agent_name: str, signal: Signal, outcome: float):
        """更新Agent性能"""
        self.performance_tracker.update_performance(agent_name, signal, outcome)
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return {
            "total_decisions": len(self.decision_history),
            "aggregation_method": self.aggregation_method.value,
            "conflict_resolution": self.conflict_resolution.value,
            "agent_weights": {name: weight.effective_weight 
                            for name, weight in self.agent_weights.items()},
            "recent_decisions": [
                {
                    "action": d.action,
                    "symbol": d.symbol,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self.decision_history[-10:]  # 最近10个决策
            ]
        }
    
    def _majority_vote_aggregation(self, symbol: str, signals: List[Signal]) -> AggregatedDecision:
        """多数投票聚合"""
        vote_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        confidences = {"BUY": [], "SELL": [], "HOLD": []}
        
        for signal in signals:
            action = signal.action.value if hasattr(signal.action, 'value') else str(signal.action)
            vote_counts[action] += 1
            confidences[action].append(signal.confidence)
        
        # 找出得票最多的动作
        winning_action = max(vote_counts, key=vote_counts.get)
        
        # 计算平均置信度
        if confidences[winning_action]:
            avg_confidence = np.mean(confidences[winning_action])
        else:
            avg_confidence = 0.0
        
        # 计算强度
        action_signals = [s for s in signals if str(s.action) == winning_action]
        avg_strength = np.mean([s.strength for s in action_signals]) if action_signals else 0.0
        
        return AggregatedDecision(
            action=winning_action,
            symbol=symbol,
            confidence=avg_confidence,
            strength=avg_strength,
            reasoning=f"Majority vote: {vote_counts[winning_action]} votes for {winning_action}",
            contributing_signals=signals
        )
    
    def _risk_adjusted_aggregation(self, symbol: str, signals: List[Signal]) -> AggregatedDecision:
        """风险调整聚合"""
        # 先进行置信度加权聚合
        base_decision = self._confidence_weighted_aggregation(symbol, signals)
        
        # 计算总体风险分数
        total_risk = sum(self._calculate_signal_risk(s) for s in signals) / len(signals)
        
        # 应用风险折扣
        risk_discount = 1 - (total_risk * 0.5)  # 最多减少50%
        
        base_decision.confidence *= risk_discount
        base_decision.strength *= risk_discount
        base_decision.reasoning = f"Risk-adjusted: {base_decision.reasoning}"
        base_decision.risk_assessment = {"total_risk": total_risk, "risk_discount": risk_discount}
        
        return base_decision
    
    def _ensemble_aggregation(self, symbol: str, signals: List[Signal]) -> AggregatedDecision:
        """集成聚合（结合多种方法）"""
        # 应用多种聚合方法
        methods = [
            self._weighted_average_aggregation,
            self._confidence_weighted_aggregation,
            self._majority_vote_aggregation
        ]
        
        results = []
        for method in methods:
            try:
                result = method(symbol, signals)
                results.append(result)
            except Exception as e:
                self.log_warning(f"Ensemble method failed: {e}")
        
        if not results:
            return self._confidence_weighted_aggregation(symbol, signals)
        
        # 计算集成结果
        actions = [r.action for r in results]
        confidences = [r.confidence for r in results]
        strengths = [r.strength for r in results]
        
        # 使用多数投票确定动作
        from collections import Counter
        action_counts = Counter(actions)
        final_action = action_counts.most_common(1)[0][0]
        
        # 平均置信度和强度
        final_confidence = np.mean(confidences)
        final_strength = np.mean(strengths)
        
        return AggregatedDecision(
            action=final_action,
            symbol=symbol,
            confidence=final_confidence,
            strength=final_strength,
            reasoning=f"Ensemble of {len(methods)} aggregation methods",
            contributing_signals=signals,
            metadata={"ensemble_results": [r.action for r in results]}
        )