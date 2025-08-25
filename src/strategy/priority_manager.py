"""
优先级管理器 (PriorityManager)

实现信号优先级的动态管理，支持基于历史表现、市场条件、
策略类型等因素的智能优先级调整。
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
import statistics

from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
from src.utils.logger import LoggerMixin


class PriorityCategory(Enum):
    """优先级类别"""
    ULTRA_HIGH = "ultra_high"     # 超高优先级 (0.9-1.0)
    HIGH = "high"                 # 高优先级 (0.7-0.9)
    MEDIUM_HIGH = "medium_high"   # 中高优先级 (0.6-0.7)
    MEDIUM = "medium"             # 中等优先级 (0.4-0.6)
    MEDIUM_LOW = "medium_low"     # 中低优先级 (0.3-0.4)
    LOW = "low"                   # 低优先级 (0.1-0.3)
    ULTRA_LOW = "ultra_low"       # 超低优先级 (0.0-0.1)


class PriorityUpdateTrigger(Enum):
    """优先级更新触发器"""
    TIME_BASED = "time_based"                 # 基于时间的更新
    PERFORMANCE_BASED = "performance_based"   # 基于表现的更新
    MARKET_CONDITION = "market_condition"     # 基于市场条件的更新
    VOLUME_BASED = "volume_based"            # 基于成交量的更新
    VOLATILITY_BASED = "volatility_based"    # 基于波动率的更新
    MANUAL_OVERRIDE = "manual_override"       # 手动覆盖


@dataclass
class SignalSource:
    """信号源信息"""
    source_id: str
    source_name: str
    source_type: str  # "HFT", "AI_AGENT", "TECHNICAL", "FUNDAMENTAL"
    base_priority: float
    current_priority: float
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """信号源性能指标"""
    accuracy_rate: float           # 准确率
    profit_factor: float          # 盈利因子
    sharpe_ratio: float           # 夏普比率
    max_drawdown: float           # 最大回撤
    avg_return: float             # 平均收益
    win_rate: float               # 胜率
    avg_holding_period: float     # 平均持有期
    total_trades: int             # 总交易数
    reliability_score: float       # 可靠性评分
    consistency_score: float       # 一致性评分


@dataclass
class MarketCondition:
    """市场条件"""
    volatility_level: str         # "low", "medium", "high"
    trend_strength: float         # 趋势强度 [0, 1]
    volume_profile: str           # "low", "normal", "high"
    market_stress: float          # 市场压力 [0, 1]
    sentiment_score: float        # 情绪分数 [-1, 1]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PriorityAdjustment:
    """优先级调整记录"""
    signal_source_id: str
    old_priority: float
    new_priority: float
    adjustment_reason: str
    trigger: PriorityUpdateTrigger
    market_condition: Optional[MarketCondition]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriorityManager(LoggerMixin):
    """
    信号优先级管理器
    
    功能：
    1. 管理信号源的优先级
    2. 基于历史表现动态调整优先级
    3. 根据市场条件优化优先级分配
    4. 提供优先级查询和统计接口
    5. 支持手动优先级覆盖
    """
    
    def __init__(self, 
                 default_priority: float = 0.5,
                 performance_window_hours: int = 24,
                 min_trades_for_adjustment: int = 10,
                 adjustment_sensitivity: float = 0.1):
        """
        初始化优先级管理器
        
        Args:
            default_priority: 默认优先级
            performance_window_hours: 性能评估窗口（小时）
            min_trades_for_adjustment: 调整所需的最小交易数
            adjustment_sensitivity: 调整敏感度
        """
        super().__init__()
        
        self.default_priority = default_priority
        self.performance_window = timedelta(hours=performance_window_hours)
        self.min_trades_for_adjustment = min_trades_for_adjustment
        self.adjustment_sensitivity = adjustment_sensitivity
        
        # 信号源管理
        self.signal_sources: Dict[str, SignalSource] = {}
        self.priority_cache: Dict[str, float] = {}
        
        # 历史记录
        self.adjustment_history: List[PriorityAdjustment] = []
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # 市场条件缓存
        self.current_market_condition: Optional[MarketCondition] = None
        self.market_condition_history: deque = deque(maxlen=100)
        
        # 优先级权重配置
        self.priority_weights = {
            'performance_score': 0.4,      # 历史表现权重
            'consistency_score': 0.25,     # 一致性权重
            'market_adaptation': 0.2,      # 市场适应性权重
            'recency_factor': 0.1,         # 时效性权重
            'volume_factor': 0.05          # 成交量权重
        }
        
        # 市场条件适应配置
        self.market_adaptation_config = {
            'high_volatility': {
                'HFT': 1.2,      # 高波动率时HFT优先级提升
                'AI_AGENT': 0.9,  # AI策略优先级稍降
            },
            'low_volatility': {
                'HFT': 0.8,       # 低波动率时HFT优先级降低
                'AI_AGENT': 1.1,  # AI策略优先级提升
            },
            'high_volume': {
                'HFT': 1.15,     # 高成交量时HFT优先级提升
                'TECHNICAL': 1.05
            },
            'trend_market': {
                'TECHNICAL': 1.1, # 趋势市场技术分析优先级提升
                'FUNDAMENTAL': 0.95
            }
        }
        
        self.log_info("PriorityManager初始化完成")
    
    def register_signal_source(self, 
                              source_id: str,
                              source_name: str,
                              source_type: str,
                              base_priority: Optional[float] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册信号源
        
        Args:
            source_id: 信号源ID
            source_name: 信号源名称
            source_type: 信号源类型
            base_priority: 基础优先级
            metadata: 元数据
            
        Returns:
            是否注册成功
        """
        try:
            if source_id in self.signal_sources:
                self.log_warning(f"信号源 {source_id} 已存在")
                return False
            
            priority = base_priority or self.default_priority
            
            signal_source = SignalSource(
                source_id=source_id,
                source_name=source_name,
                source_type=source_type,
                base_priority=priority,
                current_priority=priority,
                metadata=metadata or {}
            )
            
            self.signal_sources[source_id] = signal_source
            self.priority_cache[source_id] = priority
            
            self.log_info(f"成功注册信号源: {source_id} ({source_type}), 优先级: {priority:.3f}")
            return True
            
        except Exception as e:
            self.log_error(f"注册信号源失败: {e}")
            return False
    
    def unregister_signal_source(self, source_id: str) -> bool:
        """注销信号源"""
        try:
            if source_id not in self.signal_sources:
                self.log_warning(f"信号源 {source_id} 不存在")
                return False
            
            del self.signal_sources[source_id]
            self.priority_cache.pop(source_id, None)
            
            self.log_info(f"成功注销信号源: {source_id}")
            return True
            
        except Exception as e:
            self.log_error(f"注销信号源失败: {e}")
            return False
    
    def get_signal_priority(self, signal: MultiDimensionalSignal, source_id: str) -> float:
        """
        获取信号优先级
        
        Args:
            signal: 多维度信号
            source_id: 信号源ID
            
        Returns:
            信号优先级 [0, 1]
        """
        try:
            # 从缓存获取基础优先级
            base_priority = self.priority_cache.get(source_id, self.default_priority)
            
            # 动态调整因子
            adjustment_factors = self._calculate_adjustment_factors(signal, source_id)
            
            # 计算最终优先级
            final_priority = base_priority * adjustment_factors['total_factor']
            
            # 限制在合理范围内
            final_priority = max(0.0, min(1.0, final_priority))
            
            return final_priority
            
        except Exception as e:
            self.log_error(f"获取信号优先级失败: {e}")
            return self.default_priority
    
    def _calculate_adjustment_factors(self, 
                                    signal: MultiDimensionalSignal,
                                    source_id: str) -> Dict[str, float]:
        """计算调整因子"""
        factors = {
            'quality_factor': 1.0,
            'confidence_factor': 1.0,
            'market_factor': 1.0,
            'timing_factor': 1.0,
            'volume_factor': 1.0,
            'total_factor': 1.0
        }
        
        try:
            # 1. 信号质量因子
            quality_score = signal.signal_quality_score
            factors['quality_factor'] = 0.7 + (quality_score * 0.6)  # [0.7, 1.3]
            
            # 2. 置信度因子
            confidence = signal.overall_confidence
            factors['confidence_factor'] = 0.8 + (confidence * 0.4)  # [0.8, 1.2]
            
            # 3. 市场适应因子
            if self.current_market_condition:
                factors['market_factor'] = self._get_market_adaptation_factor(source_id)
            
            # 4. 时效性因子
            time_diff = (datetime.now() - signal.primary_signal.timestamp).total_seconds()
            if time_diff < 300:  # 5分钟内
                factors['timing_factor'] = 1.1
            elif time_diff < 600:  # 10分钟内
                factors['timing_factor'] = 1.0
            else:
                factors['timing_factor'] = 0.9
            
            # 5. 成交量因子
            volume_score = signal.volume_score
            if volume_score > 0.7:
                factors['volume_factor'] = 1.05
            elif volume_score < 0.3:
                factors['volume_factor'] = 0.95
            
            # 计算总因子
            factors['total_factor'] = (
                factors['quality_factor'] * 
                factors['confidence_factor'] * 
                factors['market_factor'] * 
                factors['timing_factor'] * 
                factors['volume_factor']
            )
            
        except Exception as e:
            self.log_error(f"计算调整因子失败: {e}")
        
        return factors
    
    def _get_market_adaptation_factor(self, source_id: str) -> float:
        """获取市场适应因子"""
        if not self.current_market_condition:
            return 1.0
        
        try:
            source = self.signal_sources.get(source_id)
            if not source:
                return 1.0
            
            source_type = source.source_type
            market_condition = self.current_market_condition
            
            adaptation_factor = 1.0
            
            # 根据波动率调整
            if market_condition.volatility_level == "high":
                adaptation_factor *= self.market_adaptation_config['high_volatility'].get(source_type, 1.0)
            elif market_condition.volatility_level == "low":
                adaptation_factor *= self.market_adaptation_config['low_volatility'].get(source_type, 1.0)
            
            # 根据成交量调整
            if market_condition.volume_profile == "high":
                adaptation_factor *= self.market_adaptation_config['high_volume'].get(source_type, 1.0)
            
            # 根据趋势强度调整
            if market_condition.trend_strength > 0.7:
                adaptation_factor *= self.market_adaptation_config['trend_market'].get(source_type, 1.0)
            
            return adaptation_factor
            
        except Exception as e:
            self.log_error(f"获取市场适应因子失败: {e}")
            return 1.0
    
    def update_performance(self, 
                          source_id: str,
                          performance_data: Dict[str, Any]) -> bool:
        """
        更新信号源性能数据
        
        Args:
            source_id: 信号源ID
            performance_data: 性能数据
            
        Returns:
            是否更新成功
        """
        try:
            if source_id not in self.signal_sources:
                self.log_warning(f"信号源 {source_id} 不存在")
                return False
            
            # 添加到历史记录
            performance_record = {
                'timestamp': datetime.now(),
                'data': performance_data
            }
            
            self.performance_history[source_id].append(performance_record)
            
            # 检查是否需要调整优先级
            if self._should_adjust_priority(source_id):
                self._adjust_priority_based_on_performance(source_id)
            
            return True
            
        except Exception as e:
            self.log_error(f"更新性能数据失败: {e}")
            return False
    
    def _should_adjust_priority(self, source_id: str) -> bool:
        """判断是否应该调整优先级"""
        try:
            # 检查是否有足够的性能数据
            performance_data = self.performance_history[source_id]
            if len(performance_data) < self.min_trades_for_adjustment:
                return False
            
            # 检查上次调整时间
            last_adjustment = None
            for adjustment in reversed(self.adjustment_history):
                if adjustment.signal_source_id == source_id:
                    last_adjustment = adjustment
                    break
            
            if last_adjustment:
                time_since_last = datetime.now() - last_adjustment.timestamp
                if time_since_last < timedelta(hours=1):  # 最少1小时间隔
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(f"判断调整条件失败: {e}")
            return False
    
    def _adjust_priority_based_on_performance(self, source_id: str) -> None:
        """基于性能调整优先级"""
        try:
            source = self.signal_sources[source_id]
            
            # 计算性能指标
            metrics = self._calculate_performance_metrics(source_id)
            if not metrics:
                return
            
            # 计算新的优先级
            new_priority = self._calculate_priority_from_metrics(source, metrics)
            
            # 检查调整幅度
            priority_change = abs(new_priority - source.current_priority)
            if priority_change < self.adjustment_sensitivity:
                return  # 变化太小，不调整
            
            # 记录调整
            adjustment = PriorityAdjustment(
                signal_source_id=source_id,
                old_priority=source.current_priority,
                new_priority=new_priority,
                adjustment_reason=f"基于性能指标自动调整 (准确率: {metrics.accuracy_rate:.2f}, 盈利因子: {metrics.profit_factor:.2f})",
                trigger=PriorityUpdateTrigger.PERFORMANCE_BASED,
                market_condition=self.current_market_condition,
                metadata={
                    'accuracy_rate': metrics.accuracy_rate,
                    'profit_factor': metrics.profit_factor,
                    'sharpe_ratio': metrics.sharpe_ratio
                }
            )
            
            # 应用调整
            source.current_priority = new_priority
            source.updated_at = datetime.now()
            self.priority_cache[source_id] = new_priority
            self.adjustment_history.append(adjustment)
            
            self.log_info(f"调整信号源 {source_id} 优先级: {adjustment.old_priority:.3f} -> {new_priority:.3f}")
            
        except Exception as e:
            self.log_error(f"基于性能调整优先级失败: {e}")
    
    def _calculate_performance_metrics(self, source_id: str) -> Optional[PerformanceMetrics]:
        """计算性能指标"""
        try:
            performance_data = list(self.performance_history[source_id])
            if not performance_data:
                return None
            
            # 提取数据
            returns = []
            win_count = 0
            total_trades = len(performance_data)
            
            for record in performance_data:
                data = record['data']
                return_val = data.get('return', 0.0)
                returns.append(return_val)
                if return_val > 0:
                    win_count += 1
            
            if not returns:
                return None
            
            # 计算指标
            avg_return = statistics.mean(returns)
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            # 计算夏普比率
            if len(returns) > 1:
                return_std = statistics.stdev(returns)
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            cumulative_returns = np.cumsum(returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = peak - cumulative_returns
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # 计算盈利因子
            positive_returns = [r for r in returns if r > 0]
            negative_returns = [r for r in returns if r < 0]
            
            if positive_returns and negative_returns:
                profit_factor = sum(positive_returns) / abs(sum(negative_returns))
            elif positive_returns:
                profit_factor = 2.0  # 只有盈利，设定高值
            else:
                profit_factor = 0.0
            
            # 计算可靠性和一致性分数
            accuracy_rate = win_rate
            reliability_score = min(1.0, accuracy_rate + (profit_factor * 0.1))
            consistency_score = max(0.0, 1.0 - (max_drawdown / max(abs(avg_return), 0.01)))
            
            return PerformanceMetrics(
                accuracy_rate=accuracy_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_return=avg_return,
                win_rate=win_rate,
                avg_holding_period=1.0,  # 简化处理
                total_trades=total_trades,
                reliability_score=reliability_score,
                consistency_score=consistency_score
            )
            
        except Exception as e:
            self.log_error(f"计算性能指标失败: {e}")
            return None
    
    def _calculate_priority_from_metrics(self, 
                                       source: SignalSource,
                                       metrics: PerformanceMetrics) -> float:
        """从性能指标计算优先级"""
        try:
            # 基础优先级
            base_priority = source.base_priority
            
            # 性能调整
            performance_score = (
                metrics.accuracy_rate * self.priority_weights['performance_score'] +
                min(metrics.profit_factor / 2.0, 1.0) * 0.3 +
                max(0, min(metrics.sharpe_ratio / 2.0, 1.0)) * 0.2 +
                (1.0 - min(metrics.max_drawdown, 1.0)) * 0.1
            )
            
            consistency_bonus = metrics.consistency_score * self.priority_weights['consistency_score']
            
            # 计算新优先级
            adjustment_factor = 0.5 + performance_score + consistency_bonus
            new_priority = base_priority * adjustment_factor
            
            # 限制范围
            return max(0.1, min(1.0, new_priority))
            
        except Exception as e:
            self.log_error(f"从指标计算优先级失败: {e}")
            return source.current_priority
    
    def update_market_condition(self, market_condition: MarketCondition) -> None:
        """更新市场条件"""
        try:
            self.current_market_condition = market_condition
            self.market_condition_history.append(market_condition)
            
            # 触发市场条件相关的优先级调整
            self._adjust_priorities_for_market_condition()
            
            self.log_info(f"更新市场条件: 波动率={market_condition.volatility_level}, 趋势强度={market_condition.trend_strength:.2f}")
            
        except Exception as e:
            self.log_error(f"更新市场条件失败: {e}")
    
    def _adjust_priorities_for_market_condition(self) -> None:
        """根据市场条件调整优先级"""
        try:
            if not self.current_market_condition:
                return
            
            for source_id, source in self.signal_sources.items():
                old_priority = source.current_priority
                adaptation_factor = self._get_market_adaptation_factor(source_id)
                
                if abs(adaptation_factor - 1.0) > 0.05:  # 显著变化
                    new_priority = old_priority * adaptation_factor
                    new_priority = max(0.1, min(1.0, new_priority))
                    
                    # 记录调整
                    adjustment = PriorityAdjustment(
                        signal_source_id=source_id,
                        old_priority=old_priority,
                        new_priority=new_priority,
                        adjustment_reason=f"市场条件变化调整 (因子: {adaptation_factor:.2f})",
                        trigger=PriorityUpdateTrigger.MARKET_CONDITION,
                        market_condition=self.current_market_condition
                    )
                    
                    source.current_priority = new_priority
                    source.updated_at = datetime.now()
                    self.priority_cache[source_id] = new_priority
                    self.adjustment_history.append(adjustment)
            
        except Exception as e:
            self.log_error(f"根据市场条件调整优先级失败: {e}")
    
    def set_manual_priority(self, 
                           source_id: str,
                           priority: float,
                           reason: str = "手动设置") -> bool:
        """手动设置优先级"""
        try:
            if source_id not in self.signal_sources:
                self.log_warning(f"信号源 {source_id} 不存在")
                return False
            
            if not 0 <= priority <= 1:
                self.log_error(f"优先级必须在0-1范围内: {priority}")
                return False
            
            source = self.signal_sources[source_id]
            old_priority = source.current_priority
            
            # 记录调整
            adjustment = PriorityAdjustment(
                signal_source_id=source_id,
                old_priority=old_priority,
                new_priority=priority,
                adjustment_reason=reason,
                trigger=PriorityUpdateTrigger.MANUAL_OVERRIDE
            )
            
            # 应用设置
            source.current_priority = priority
            source.updated_at = datetime.now()
            self.priority_cache[source_id] = priority
            self.adjustment_history.append(adjustment)
            
            self.log_info(f"手动设置信号源 {source_id} 优先级: {old_priority:.3f} -> {priority:.3f}")
            return True
            
        except Exception as e:
            self.log_error(f"手动设置优先级失败: {e}")
            return False
    
    def get_priority_rankings(self) -> List[Tuple[str, float]]:
        """获取优先级排名"""
        try:
            rankings = [(source_id, priority) for source_id, priority in self.priority_cache.items()]
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            self.log_error(f"获取优先级排名失败: {e}")
            return []
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """获取优先级统计信息"""
        try:
            if not self.priority_cache:
                return {}
            
            priorities = list(self.priority_cache.values())
            
            stats = {
                'total_sources': len(self.signal_sources),
                'avg_priority': statistics.mean(priorities),
                'median_priority': statistics.median(priorities),
                'priority_std': statistics.stdev(priorities) if len(priorities) > 1 else 0,
                'max_priority': max(priorities),
                'min_priority': min(priorities),
                'adjustments_24h': len([adj for adj in self.adjustment_history 
                                      if (datetime.now() - adj.timestamp).total_seconds() < 86400]),
                'total_adjustments': len(self.adjustment_history)
            }
            
            # 按类别统计
            category_stats = defaultdict(list)
            for source_id, priority in self.priority_cache.items():
                source_type = self.signal_sources[source_id].source_type
                category_stats[source_type].append(priority)
            
            stats['by_category'] = {}
            for category, priorities in category_stats.items():
                stats['by_category'][category] = {
                    'count': len(priorities),
                    'avg_priority': statistics.mean(priorities),
                    'max_priority': max(priorities),
                    'min_priority': min(priorities)
                }
            
            return stats
            
        except Exception as e:
            self.log_error(f"获取优先级统计失败: {e}")
            return {}
    
    def cleanup_history(self, days_to_keep: int = 7) -> None:
        """清理历史数据"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # 清理调整历史
            self.adjustment_history = [
                adj for adj in self.adjustment_history 
                if adj.timestamp > cutoff_time
            ]
            
            # 清理性能历史
            for source_id in self.performance_history:
                performance_data = list(self.performance_history[source_id])
                filtered_data = [
                    record for record in performance_data
                    if record['timestamp'] > cutoff_time
                ]
                self.performance_history[source_id] = deque(filtered_data, maxlen=1000)
            
            self.log_info(f"清理了 {days_to_keep} 天前的历史数据")
            
        except Exception as e:
            self.log_error(f"清理历史数据失败: {e}")