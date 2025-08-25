"""
冲突解决器 (ConflictResolver)

实现信号冲突的检测、分类和解决逻辑，支持多种冲突类型和解决策略。
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict
import numpy as np

from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
from src.utils.logger import LoggerMixin


class ConflictType(Enum):
    """冲突类型枚举"""
    DIRECTION_CONFLICT = "direction_conflict"           # 买卖方向冲突
    STRENGTH_CONFLICT = "strength_conflict"             # 信号强度冲突
    TIME_CONFLICT = "time_conflict"                     # 时间窗口冲突
    POSITION_CONFLICT = "position_conflict"             # 仓位冲突
    RISK_CONFLICT = "risk_conflict"                     # 风险冲突
    CORRELATION_CONFLICT = "correlation_conflict"       # 相关性冲突
    VOLATILITY_CONFLICT = "volatility_conflict"         # 波动率冲突


class ConflictSeverity(Enum):
    """冲突严重性"""
    LOW = "low"           # 轻微冲突
    MEDIUM = "medium"     # 中等冲突
    HIGH = "high"         # 严重冲突
    CRITICAL = "critical" # 严重冲突


class ConflictResolutionStrategy(Enum):
    """冲突解决策略"""
    PRIORITY_WEIGHTED = "priority_weighted"           # 优先级加权
    CONFIDENCE_WEIGHTED = "confidence_weighted"       # 置信度加权
    QUALITY_WEIGHTED = "quality_weighted"             # 质量加权
    MAJORITY_VOTE = "majority_vote"                   # 多数投票
    EXPERT_OVERRIDE = "expert_override"               # 专家覆盖
    RISK_MINIMIZATION = "risk_minimization"           # 风险最小化
    CONSERVATIVE_APPROACH = "conservative_approach"    # 保守方法
    HYBRID_APPROACH = "hybrid_approach"               # 混合方法


@dataclass
class ConflictDetail:
    """冲突详情"""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_signals: List[str]  # 涉及的信号ID列表
    description: str
    confidence_impact: float     # 对置信度的影响
    risk_impact: float          # 对风险的影响
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictResolution:
    """冲突解决结果"""
    conflict_id: str
    strategy_used: ConflictResolutionStrategy
    resolved_signal: Optional[MultiDimensionalSignal]
    confidence_adjustment: float
    risk_adjustment: float
    reasoning: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConflictMatrix:
    """冲突矩阵"""
    signal_pairs: List[Tuple[str, str]]
    conflict_scores: Dict[Tuple[str, str], float]
    conflict_types: Dict[Tuple[str, str], List[ConflictType]]
    severity_matrix: Dict[Tuple[str, str], ConflictSeverity]
    

class ConflictResolver(LoggerMixin):
    """
    信号冲突解决器
    
    功能：
    1. 检测多维度信号冲突
    2. 评估冲突严重性
    3. 选择合适的解决策略
    4. 生成解决后的信号
    5. 记录冲突历史和统计
    """
    
    def __init__(self, 
                 default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.HYBRID_APPROACH,
                 conflict_threshold: float = 0.3,
                 time_window_minutes: int = 5):
        """
        初始化冲突解决器
        
        Args:
            default_strategy: 默认解决策略
            conflict_threshold: 冲突检测阈值
            time_window_minutes: 时间窗口（分钟）
        """
        super().__init__()
        
        self.default_strategy = default_strategy
        self.conflict_threshold = conflict_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        
        # 冲突检测配置
        self.direction_conflict_threshold = 0.5
        self.strength_conflict_threshold = 0.3
        self.correlation_threshold = 0.8
        self.volatility_threshold = 0.4
        
        # 冲突历史和统计
        self.conflict_history: List[ConflictDetail] = []
        self.resolution_history: List[ConflictResolution] = []
        self.conflict_stats: Dict[ConflictType, int] = defaultdict(int)
        
        # 专家规则权重配置
        self.strategy_weights = {
            ConflictResolutionStrategy.PRIORITY_WEIGHTED: {
                'priority': 0.4,
                'confidence': 0.3,
                'quality': 0.3
            },
            ConflictResolutionStrategy.CONFIDENCE_WEIGHTED: {
                'confidence': 0.6,
                'quality': 0.25,
                'priority': 0.15
            },
            ConflictResolutionStrategy.QUALITY_WEIGHTED: {
                'quality': 0.5,
                'confidence': 0.3,
                'priority': 0.2
            }
        }
        
        self.log_info("ConflictResolver初始化完成")
    
    def detect_conflicts(self, 
                        signals: List[MultiDimensionalSignal],
                        signal_priorities: Optional[Dict[str, float]] = None) -> List[ConflictDetail]:
        """
        检测信号间的冲突
        
        Args:
            signals: 待检测的信号列表
            signal_priorities: 信号优先级字典
            
        Returns:
            检测到的冲突列表
        """
        conflicts = []
        
        if len(signals) < 2:
            return conflicts
        
        try:
            # 按交易标的分组检测
            signals_by_symbol = defaultdict(list)
            for signal in signals:
                signals_by_symbol[signal.primary_signal.symbol].append(signal)
            
            for symbol, symbol_signals in signals_by_symbol.items():
                if len(symbol_signals) < 2:
                    continue
                
                # 两两检测冲突
                for i in range(len(symbol_signals)):
                    for j in range(i + 1, len(symbol_signals)):
                        signal1, signal2 = symbol_signals[i], symbol_signals[j]
                        
                        detected_conflicts = self._detect_pairwise_conflicts(
                            signal1, signal2, signal_priorities
                        )
                        conflicts.extend(detected_conflicts)
            
            # 记录统计
            for conflict in conflicts:
                self.conflict_stats[conflict.conflict_type] += 1
                
            self.conflict_history.extend(conflicts)
            
            self.log_info(f"检测到 {len(conflicts)} 个冲突")
            return conflicts
            
        except Exception as e:
            self.log_error(f"冲突检测失败: {e}")
            return conflicts
    
    def _detect_pairwise_conflicts(self, 
                                  signal1: MultiDimensionalSignal,
                                  signal2: MultiDimensionalSignal,
                                  priorities: Optional[Dict[str, float]] = None) -> List[ConflictDetail]:
        """检测两个信号间的冲突"""
        conflicts = []
        signal1_id = id(signal1)
        signal2_id = id(signal2)
        
        # 1. 方向冲突检测
        direction_conflict = self._check_direction_conflict(signal1, signal2)
        if direction_conflict:
            conflicts.append(ConflictDetail(
                conflict_id=f"direction_{signal1_id}_{signal2_id}",
                conflict_type=ConflictType.DIRECTION_CONFLICT,
                severity=direction_conflict,
                involved_signals=[str(signal1_id), str(signal2_id)],
                description=f"信号方向冲突：{signal1.primary_signal.signal_type} vs {signal2.primary_signal.signal_type}",
                confidence_impact=-0.2,
                risk_impact=0.3
            ))
        
        # 2. 强度冲突检测
        strength_conflict = self._check_strength_conflict(signal1, signal2)
        if strength_conflict:
            conflicts.append(ConflictDetail(
                conflict_id=f"strength_{signal1_id}_{signal2_id}",
                conflict_type=ConflictType.STRENGTH_CONFLICT,
                severity=strength_conflict,
                involved_signals=[str(signal1_id), str(signal2_id)],
                description=f"信号强度冲突：置信度差异 {abs(signal1.overall_confidence - signal2.overall_confidence):.2f}",
                confidence_impact=-0.1,
                risk_impact=0.15
            ))
        
        # 3. 时间冲突检测
        time_conflict = self._check_time_conflict(signal1, signal2)
        if time_conflict:
            conflicts.append(ConflictDetail(
                conflict_id=f"time_{signal1_id}_{signal2_id}",
                conflict_type=ConflictType.TIME_CONFLICT,
                severity=time_conflict,
                involved_signals=[str(signal1_id), str(signal2_id)],
                description="信号时间窗口冲突",
                confidence_impact=-0.05,
                risk_impact=0.1
            ))
        
        # 4. 风险冲突检测
        risk_conflict = self._check_risk_conflict(signal1, signal2)
        if risk_conflict:
            conflicts.append(ConflictDetail(
                conflict_id=f"risk_{signal1_id}_{signal2_id}",
                conflict_type=ConflictType.RISK_CONFLICT,
                severity=risk_conflict,
                involved_signals=[str(signal1_id), str(signal2_id)],
                description="风险收益比冲突",
                confidence_impact=-0.15,
                risk_impact=0.25
            ))
        
        # 5. 波动率冲突检测
        volatility_conflict = self._check_volatility_conflict(signal1, signal2)
        if volatility_conflict:
            conflicts.append(ConflictDetail(
                conflict_id=f"volatility_{signal1_id}_{signal2_id}",
                conflict_type=ConflictType.VOLATILITY_CONFLICT,
                severity=volatility_conflict,
                involved_signals=[str(signal1_id), str(signal2_id)],
                description=f"波动率评估冲突：{abs(signal1.volatility_score - signal2.volatility_score):.2f}",
                confidence_impact=-0.1,
                risk_impact=0.2
            ))
        
        return conflicts
    
    def _check_direction_conflict(self, 
                                 signal1: MultiDimensionalSignal,
                                 signal2: MultiDimensionalSignal) -> Optional[ConflictSeverity]:
        """检查方向冲突"""
        sig1_type = signal1.primary_signal.signal_type
        sig2_type = signal2.primary_signal.signal_type
        
        # 买卖方向判断
        buy_signals = {SignalStrength.WEAK_BUY, SignalStrength.BUY, SignalStrength.STRONG_BUY}
        sell_signals = {SignalStrength.WEAK_SELL, SignalStrength.SELL, SignalStrength.STRONG_SELL}
        
        sig1_is_buy = sig1_type in buy_signals
        sig1_is_sell = sig1_type in sell_signals
        sig2_is_buy = sig2_type in buy_signals
        sig2_is_sell = sig2_type in sell_signals
        
        if (sig1_is_buy and sig2_is_sell) or (sig1_is_sell and sig2_is_buy):
            # 根据信号强度确定冲突严重性
            sig1_strength = abs(sig1_type.value - SignalStrength.NEUTRAL.value)
            sig2_strength = abs(sig2_type.value - SignalStrength.NEUTRAL.value)
            
            avg_strength = (sig1_strength + sig2_strength) / 2
            if avg_strength >= 3:  # STRONG信号
                return ConflictSeverity.CRITICAL
            elif avg_strength >= 2:  # BUY/SELL信号
                return ConflictSeverity.HIGH
            else:  # WEAK信号
                return ConflictSeverity.MEDIUM
        
        return None
    
    def _check_strength_conflict(self, 
                               signal1: MultiDimensionalSignal,
                               signal2: MultiDimensionalSignal) -> Optional[ConflictSeverity]:
        """检查强度冲突"""
        confidence_diff = abs(signal1.overall_confidence - signal2.overall_confidence)
        
        if confidence_diff > self.strength_conflict_threshold:
            if confidence_diff > 0.6:
                return ConflictSeverity.HIGH
            elif confidence_diff > 0.4:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW
        
        return None
    
    def _check_time_conflict(self, 
                           signal1: MultiDimensionalSignal,
                           signal2: MultiDimensionalSignal) -> Optional[ConflictSeverity]:
        """检查时间冲突"""
        time_diff = abs((signal1.primary_signal.timestamp - signal2.primary_signal.timestamp).total_seconds())
        
        if time_diff < self.time_window.total_seconds():
            if time_diff < 60:  # 1分钟内
                return ConflictSeverity.HIGH
            elif time_diff < 180:  # 3分钟内
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW
        
        return None
    
    def _check_risk_conflict(self, 
                           signal1: MultiDimensionalSignal,
                           signal2: MultiDimensionalSignal) -> Optional[ConflictSeverity]:
        """检查风险冲突"""
        rr1 = signal1.risk_reward_ratio
        rr2 = signal2.risk_reward_ratio
        
        if rr1 == 0 or rr2 == 0:
            return None
        
        ratio_diff = abs(rr1 - rr2) / max(rr1, rr2)
        
        if ratio_diff > 0.5:  # 50%以上差异
            if ratio_diff > 0.8:
                return ConflictSeverity.HIGH
            else:
                return ConflictSeverity.MEDIUM
        
        return None
    
    def _check_volatility_conflict(self, 
                                 signal1: MultiDimensionalSignal,
                                 signal2: MultiDimensionalSignal) -> Optional[ConflictSeverity]:
        """检查波动率冲突"""
        vol_diff = abs(signal1.volatility_score - signal2.volatility_score)
        
        if vol_diff > self.volatility_threshold:
            if vol_diff > 0.6:
                return ConflictSeverity.HIGH
            elif vol_diff > 0.5:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW
        
        return None
    
    def resolve_conflicts(self, 
                         signals: List[MultiDimensionalSignal],
                         conflicts: List[ConflictDetail],
                         signal_priorities: Optional[Dict[str, float]] = None,
                         strategy: Optional[ConflictResolutionStrategy] = None) -> List[ConflictResolution]:
        """
        解决检测到的冲突
        
        Args:
            signals: 信号列表
            conflicts: 冲突列表
            signal_priorities: 信号优先级
            strategy: 解决策略
            
        Returns:
            冲突解决结果列表
        """
        if not conflicts:
            return []
        
        resolution_strategy = strategy or self.default_strategy
        resolutions = []
        
        try:
            # 按冲突严重性排序，优先解决严重冲突
            sorted_conflicts = sorted(conflicts, 
                                    key=lambda c: self._get_severity_weight(c.severity), 
                                    reverse=True)
            
            for conflict in sorted_conflicts:
                resolution = self._resolve_single_conflict(
                    conflict, signals, signal_priorities, resolution_strategy
                )
                
                if resolution:
                    resolutions.append(resolution)
            
            self.resolution_history.extend(resolutions)
            
            self.log_info(f"解决了 {len(resolutions)} 个冲突")
            return resolutions
            
        except Exception as e:
            self.log_error(f"冲突解决失败: {e}")
            return resolutions
    
    def _resolve_single_conflict(self, 
                               conflict: ConflictDetail,
                               signals: List[MultiDimensionalSignal],
                               priorities: Optional[Dict[str, float]],
                               strategy: ConflictResolutionStrategy) -> Optional[ConflictResolution]:
        """解决单个冲突"""
        try:
            # 获取涉及冲突的信号
            involved_signals = []
            signal_id_map = {str(id(sig)): sig for sig in signals}
            
            for signal_id in conflict.involved_signals:
                if signal_id in signal_id_map:
                    involved_signals.append(signal_id_map[signal_id])
            
            if len(involved_signals) < 2:
                return None
            
            # 根据策略解决冲突
            if strategy == ConflictResolutionStrategy.PRIORITY_WEIGHTED:
                resolved_signal = self._resolve_by_priority(involved_signals, priorities)
            elif strategy == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED:
                resolved_signal = self._resolve_by_confidence(involved_signals)
            elif strategy == ConflictResolutionStrategy.QUALITY_WEIGHTED:
                resolved_signal = self._resolve_by_quality(involved_signals)
            elif strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
                resolved_signal = self._resolve_by_majority_vote(involved_signals)
            elif strategy == ConflictResolutionStrategy.CONSERVATIVE_APPROACH:
                resolved_signal = self._resolve_by_conservative_approach(involved_signals)
            elif strategy == ConflictResolutionStrategy.RISK_MINIMIZATION:
                resolved_signal = self._resolve_by_risk_minimization(involved_signals)
            elif strategy == ConflictResolutionStrategy.HYBRID_APPROACH:
                resolved_signal = self._resolve_by_hybrid_approach(involved_signals, priorities)
            else:
                resolved_signal = self._resolve_by_confidence(involved_signals)  # 默认策略
            
            # 计算调整系数
            confidence_adjustment = conflict.confidence_impact
            risk_adjustment = conflict.risk_impact
            
            # 生成解决推理
            reasoning = self._generate_resolution_reasoning(
                conflict, involved_signals, resolved_signal, strategy
            )
            
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                strategy_used=strategy,
                resolved_signal=resolved_signal,
                confidence_adjustment=confidence_adjustment,
                risk_adjustment=risk_adjustment,
                reasoning=reasoning,
                metadata={
                    'original_signals_count': len(involved_signals),
                    'conflict_type': conflict.conflict_type.value,
                    'conflict_severity': conflict.severity.value
                }
            )
            
        except Exception as e:
            self.log_error(f"解决单个冲突失败: {e}")
            return None
    
    def _resolve_by_priority(self, 
                           signals: List[MultiDimensionalSignal],
                           priorities: Optional[Dict[str, float]]) -> MultiDimensionalSignal:
        """基于优先级解决冲突"""
        if not priorities:
            # 如果没有优先级，使用质量分数作为代替
            return max(signals, key=lambda s: s.signal_quality_score)
        
        # 选择优先级最高的信号
        best_signal = signals[0]
        best_priority = priorities.get(str(id(best_signal)), 0.0)
        
        for signal in signals[1:]:
            signal_priority = priorities.get(str(id(signal)), 0.0)
            if signal_priority > best_priority:
                best_signal = signal
                best_priority = signal_priority
        
        return best_signal
    
    def _resolve_by_confidence(self, signals: List[MultiDimensionalSignal]) -> MultiDimensionalSignal:
        """基于置信度解决冲突"""
        return max(signals, key=lambda s: s.overall_confidence)
    
    def _resolve_by_quality(self, signals: List[MultiDimensionalSignal]) -> MultiDimensionalSignal:
        """基于质量分数解决冲突"""
        return max(signals, key=lambda s: s.signal_quality_score)
    
    def _resolve_by_majority_vote(self, signals: List[MultiDimensionalSignal]) -> MultiDimensionalSignal:
        """基于多数投票解决冲突"""
        # 按信号方向分组
        buy_signals = []
        sell_signals = []
        neutral_signals = []
        
        for signal in signals:
            if signal.primary_signal._is_buy_signal():
                buy_signals.append(signal)
            elif signal.primary_signal._is_sell_signal():
                sell_signals.append(signal)
            else:
                neutral_signals.append(signal)
        
        # 选择数量最多的组
        groups = [buy_signals, sell_signals, neutral_signals]
        majority_group = max(groups, key=len)
        
        if not majority_group:
            return signals[0]
        
        # 在多数组中选择质量最高的信号
        return max(majority_group, key=lambda s: s.signal_quality_score)
    
    def _resolve_by_conservative_approach(self, signals: List[MultiDimensionalSignal]) -> MultiDimensionalSignal:
        """基于保守方法解决冲突"""
        # 选择风险最小的信号
        return min(signals, key=lambda s: s.volatility_score)
    
    def _resolve_by_risk_minimization(self, signals: List[MultiDimensionalSignal]) -> MultiDimensionalSignal:
        """基于风险最小化解决冲突"""
        # 综合考虑风险收益比和波动率
        def risk_score(signal):
            rr = signal.risk_reward_ratio
            vol = signal.volatility_score
            if rr > 0:
                return vol / rr  # 风险越小越好
            else:
                return float('inf')
        
        return min(signals, key=risk_score)
    
    def _resolve_by_hybrid_approach(self, 
                                  signals: List[MultiDimensionalSignal],
                                  priorities: Optional[Dict[str, float]]) -> MultiDimensionalSignal:
        """基于混合方法解决冲突"""
        def hybrid_score(signal):
            # 综合评分：质量 + 置信度 + 优先级
            quality = signal.signal_quality_score
            confidence = signal.overall_confidence
            priority = priorities.get(str(id(signal)), 0.5) if priorities else 0.5
            
            # 加权计算
            return quality * 0.4 + confidence * 0.4 + priority * 0.2
        
        return max(signals, key=hybrid_score)
    
    def _generate_resolution_reasoning(self, 
                                     conflict: ConflictDetail,
                                     involved_signals: List[MultiDimensionalSignal],
                                     resolved_signal: MultiDimensionalSignal,
                                     strategy: ConflictResolutionStrategy) -> List[str]:
        """生成解决推理"""
        reasoning = []
        
        reasoning.append(f"冲突类型: {conflict.conflict_type.value}")
        reasoning.append(f"冲突严重性: {conflict.severity.value}")
        reasoning.append(f"涉及信号数量: {len(involved_signals)}")
        reasoning.append(f"使用策略: {strategy.value}")
        
        if resolved_signal:
            reasoning.append(f"选择的信号置信度: {resolved_signal.overall_confidence:.2f}")
            reasoning.append(f"选择的信号质量分数: {resolved_signal.signal_quality_score:.2f}")
            reasoning.append(f"选择的信号方向: {resolved_signal.primary_signal.signal_type.name}")
        
        return reasoning
    
    def _get_severity_weight(self, severity: ConflictSeverity) -> int:
        """获取严重性权重"""
        weights = {
            ConflictSeverity.CRITICAL: 4,
            ConflictSeverity.HIGH: 3,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 1
        }
        return weights.get(severity, 1)
    
    def get_conflict_statistics(self) -> Dict[str, Any]:
        """获取冲突统计信息"""
        total_conflicts = sum(self.conflict_stats.values())
        
        stats = {
            'total_conflicts': total_conflicts,
            'conflicts_by_type': dict(self.conflict_stats),
            'resolution_count': len(self.resolution_history),
            'conflict_rate_by_type': {}
        }
        
        if total_conflicts > 0:
            for conflict_type, count in self.conflict_stats.items():
                stats['conflict_rate_by_type'][conflict_type.value] = count / total_conflicts
        
        # 最近24小时的冲突
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_conflicts = [c for c in self.conflict_history if c.detected_at > recent_threshold]
        stats['recent_24h_conflicts'] = len(recent_conflicts)
        
        return stats
    
    def update_configuration(self, 
                           conflict_threshold: Optional[float] = None,
                           time_window_minutes: Optional[int] = None,
                           default_strategy: Optional[ConflictResolutionStrategy] = None):
        """更新配置"""
        if conflict_threshold is not None:
            self.conflict_threshold = conflict_threshold
            
        if time_window_minutes is not None:
            self.time_window = timedelta(minutes=time_window_minutes)
            
        if default_strategy is not None:
            self.default_strategy = default_strategy
        
        self.log_info("ConflictResolver配置已更新")