"""
量化交易系统信号数据模型

包含信号强度枚举、交易信号和多维度信号数据类，
提供完整的强类型信号数据结构支持。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class SignalStrength(Enum):
    """信号强度枚举类型
    
    定义交易信号的强度等级，从强烈卖出到强烈买入
    """
    STRONG_SELL = -1
    SELL = 0
    WEAK_SELL = 1
    NEUTRAL = 2
    WEAK_BUY = 3
    BUY = 4
    STRONG_BUY = 5


@dataclass
class TradingSignal:
    """基础交易信号数据类
    
    包含单一交易信号的所有核心信息，包括价格目标、
    止损设置、推理逻辑和技术指标共识
    """
    symbol: str                                 # 交易标的符号
    signal_type: SignalStrength                 # 信号类型和强度
    confidence: float                          # 置信度 [0-1]
    entry_price: float                         # 入场价格
    target_price: float                        # 目标价格
    stop_loss: float                           # 止损价格
    reasoning: List[str]                       # 信号推理过程
    indicators_consensus: Dict[str, float]     # 技术指标共识
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """数据验证"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在0-1之间，当前值: {self.confidence}")
        
        if self.entry_price <= 0:
            raise ValueError(f"入场价格必须大于0，当前值: {self.entry_price}")
        
        if self.target_price <= 0:
            raise ValueError(f"目标价格必须大于0，当前值: {self.target_price}")
        
        if self.stop_loss <= 0:
            raise ValueError(f"止损价格必须大于0，当前值: {self.stop_loss}")
        
        # 根据信号类型验证价格逻辑
        if self._is_buy_signal():
            if self.target_price <= self.entry_price:
                raise ValueError("买入信号的目标价格必须高于入场价格")
            if self.stop_loss >= self.entry_price:
                raise ValueError("买入信号的止损价格必须低于入场价格")
        elif self._is_sell_signal():
            if self.target_price >= self.entry_price:
                raise ValueError("卖出信号的目标价格必须低于入场价格")
            if self.stop_loss <= self.entry_price:
                raise ValueError("卖出信号的止损价格必须高于入场价格")
    
    def _is_buy_signal(self) -> bool:
        """判断是否为买入信号"""
        return self.signal_type in [
            SignalStrength.WEAK_BUY, 
            SignalStrength.BUY, 
            SignalStrength.STRONG_BUY
        ]
    
    def _is_sell_signal(self) -> bool:
        """判断是否为卖出信号"""
        return self.signal_type in [
            SignalStrength.WEAK_SELL, 
            SignalStrength.SELL, 
            SignalStrength.STRONG_SELL
        ]
    
    @property
    def risk_reward_ratio(self) -> float:
        """计算风险收益比"""
        if self._is_buy_signal():
            potential_profit = self.target_price - self.entry_price
            potential_loss = self.entry_price - self.stop_loss
        elif self._is_sell_signal():
            potential_profit = self.entry_price - self.target_price
            potential_loss = self.stop_loss - self.entry_price
        else:
            return 0.0
        
        if potential_loss <= 0:
            return float('inf')
        
        return potential_profit / potential_loss
    
    @property
    def is_valid(self) -> bool:
        """检查信号是否有效"""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False


@dataclass 
class MultiDimensionalSignal:
    """多维度信号数据类
    
    扩展基础交易信号，增加多个维度的市场分析，
    提供更全面的交易决策支持
    """
    primary_signal: TradingSignal              # 主要交易信号
    momentum_score: float                      # 动量维度 [-1, 1]
    mean_reversion_score: float               # 均值回归维度 [-1, 1]  
    volatility_score: float                   # 波动率维度 [0, 1]
    volume_score: float                       # 成交量维度 [0, 1]
    sentiment_score: float                    # 市场情绪维度 [-1, 1]
    overall_confidence: float                 # 综合置信度 [0, 1]
    risk_reward_ratio: float                  # 风险收益比
    max_position_size: float                  # 建议最大仓位 [0, 1]
    
    # 可选的额外信息
    market_regime: Optional[str] = None       # 市场状态
    correlation_matrix: Optional[Dict[str, float]] = None  # 相关性矩阵
    technical_levels: Optional[Dict[str, float]] = None    # 技术位
    macro_factors: Optional[Dict[str, float]] = None       # 宏观因子
    
    def __post_init__(self):
        """数据验证"""
        # 验证维度分数范围
        if not -1 <= self.momentum_score <= 1:
            raise ValueError(f"动量分数必须在-1到1之间，当前值: {self.momentum_score}")
        
        if not -1 <= self.mean_reversion_score <= 1:
            raise ValueError(f"均值回归分数必须在-1到1之间，当前值: {self.mean_reversion_score}")
        
        if not 0 <= self.volatility_score <= 1:
            raise ValueError(f"波动率分数必须在0到1之间，当前值: {self.volatility_score}")
        
        if not 0 <= self.volume_score <= 1:
            raise ValueError(f"成交量分数必须在0到1之间，当前值: {self.volume_score}")
        
        if not -1 <= self.sentiment_score <= 1:
            raise ValueError(f"情绪分数必须在-1到1之间，当前值: {self.sentiment_score}")
        
        if not 0 <= self.overall_confidence <= 1:
            raise ValueError(f"综合置信度必须在0到1之间，当前值: {self.overall_confidence}")
        
        if not 0 <= self.max_position_size <= 1:
            raise ValueError(f"最大仓位必须在0到1之间，当前值: {self.max_position_size}")
        
        if self.risk_reward_ratio < 0:
            raise ValueError(f"风险收益比不能为负数，当前值: {self.risk_reward_ratio}")
    
    @property
    def signal_quality_score(self) -> float:
        """计算信号质量综合评分
        
        基于各维度分数和置信度计算综合质量评分
        Returns:
            float: 信号质量评分 [0, 1]
        """
        # 权重配置
        weights = {
            'confidence': 0.3,
            'risk_reward': 0.2,
            'momentum': 0.15,
            'volume': 0.15,
            'volatility': 0.1,
            'sentiment': 0.1
        }
        
        # 风险收益比标准化 (假设好的比例是2:1以上)
        normalized_rr = min(self.risk_reward_ratio / 2.0, 1.0)
        
        # 波动率分数反转 (低波动率更好)
        volatility_adjusted = 1.0 - self.volatility_score
        
        # 动量和情绪取绝对值 (强烈的方向性更好)
        momentum_abs = abs(self.momentum_score)
        sentiment_abs = abs(self.sentiment_score)
        
        quality_score = (
            weights['confidence'] * self.overall_confidence +
            weights['risk_reward'] * normalized_rr +
            weights['momentum'] * momentum_abs +
            weights['volume'] * self.volume_score +
            weights['volatility'] * volatility_adjusted +
            weights['sentiment'] * sentiment_abs
        )
        
        return min(quality_score, 1.0)
    
    @property
    def signal_direction_consensus(self) -> float:
        """计算信号方向一致性
        
        评估各维度对信号方向的一致性
        Returns:
            float: 方向一致性 [-1, 1]，正值表示买入一致，负值表示卖出一致
        """
        # 获取主信号方向
        if self.primary_signal._is_buy_signal():
            primary_direction = 1
        elif self.primary_signal._is_sell_signal():
            primary_direction = -1
        else:
            primary_direction = 0
        
        # 各维度与主方向的一致性
        dimensions = [
            self.momentum_score,
            self.mean_reversion_score,
            self.sentiment_score
        ]
        
        consensus = 0.0
        for dim_score in dimensions:
            if primary_direction > 0 and dim_score > 0:
                consensus += abs(dim_score)
            elif primary_direction < 0 and dim_score < 0:
                consensus += abs(dim_score)
            else:
                consensus -= abs(dim_score)
        
        # 标准化到[-1, 1]
        return consensus / len(dimensions)
    
    def get_position_sizing_recommendation(self, 
                                         base_position_size: float = 1.0,
                                         risk_tolerance: float = 1.0) -> float:
        """获取仓位大小建议
        
        Args:
            base_position_size: 基础仓位大小
            risk_tolerance: 风险容忍度 [0, 1]
            
        Returns:
            float: 建议仓位大小
        """
        # 基于信号质量调整
        quality_adjustment = self.signal_quality_score
        
        # 基于风险收益比调整
        rr_adjustment = min(self.risk_reward_ratio / 2.0, 1.0)
        
        # 基于波动率调整 (高波动率降低仓位)
        volatility_adjustment = 1.0 - (self.volatility_score * 0.5)
        
        # 基于方向一致性调整
        consensus_adjustment = (abs(self.signal_direction_consensus) + 1) / 2
        
        # 综合调整系数
        adjustment_factor = (
            quality_adjustment * 0.3 +
            rr_adjustment * 0.25 +
            volatility_adjustment * 0.25 +
            consensus_adjustment * 0.2
        )
        
        # 应用风险容忍度和最大仓位限制
        recommended_size = (
            base_position_size * 
            adjustment_factor * 
            risk_tolerance * 
            self.max_position_size
        )
        
        return min(recommended_size, self.max_position_size)


class SignalAggregator:
    """信号聚合器
    
    用于处理多个信号的聚合和评估
    """
    
    @staticmethod
    def combine_signals(signals: List[MultiDimensionalSignal], 
                       weights: Optional[Dict[str, float]] = None) -> Optional[MultiDimensionalSignal]:
        """组合多个信号为单一信号
        
        Args:
            signals: 信号列表
            weights: 信号权重字典，键为信号源，值为权重
            
        Returns:
            MultiDimensionalSignal: 组合后的信号，如果无法组合则返回None
        """
        if not signals:
            return None
        
        if len(signals) == 1:
            return signals[0]
        
        # 检查所有信号是否针对同一标的
        symbol = signals[0].primary_signal.symbol
        if not all(s.primary_signal.symbol == symbol for s in signals):
            raise ValueError("只能组合同一标的的信号")
        
        # 设置默认权重
        if weights is None:
            weights = {f"signal_{i}": 1.0 for i in range(len(signals))}
        
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("权重总和必须大于0")
        
        # 加权平均各维度分数
        weighted_momentum = sum(s.momentum_score * weights.get(f"signal_{i}", 1.0) 
                               for i, s in enumerate(signals)) / total_weight
        
        weighted_mean_reversion = sum(s.mean_reversion_score * weights.get(f"signal_{i}", 1.0)
                                     for i, s in enumerate(signals)) / total_weight
        
        weighted_volatility = sum(s.volatility_score * weights.get(f"signal_{i}", 1.0)
                                 for i, s in enumerate(signals)) / total_weight
        
        weighted_volume = sum(s.volume_score * weights.get(f"signal_{i}", 1.0)
                             for i, s in enumerate(signals)) / total_weight
        
        weighted_sentiment = sum(s.sentiment_score * weights.get(f"signal_{i}", 1.0)
                               for i, s in enumerate(signals)) / total_weight
        
        # 选择置信度最高的信号作为主信号
        primary_signal = max(signals, key=lambda s: s.overall_confidence).primary_signal
        
        # 综合置信度为加权平均
        combined_confidence = sum(s.overall_confidence * weights.get(f"signal_{i}", 1.0)
                                 for i, s in enumerate(signals)) / total_weight
        
        # 风险收益比取平均
        avg_risk_reward = sum(s.risk_reward_ratio for s in signals) / len(signals)
        
        # 最大仓位取最小值（更保守）
        min_position_size = min(s.max_position_size for s in signals)
        
        return MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=weighted_momentum,
            mean_reversion_score=weighted_mean_reversion,
            volatility_score=weighted_volatility,
            volume_score=weighted_volume,
            sentiment_score=weighted_sentiment,
            overall_confidence=combined_confidence,
            risk_reward_ratio=avg_risk_reward,
            max_position_size=min_position_size
        )
    
    @staticmethod
    def filter_signals_by_quality(signals: List[MultiDimensionalSignal],
                                 min_quality_score: float = 0.6,
                                 min_confidence: float = 0.7) -> List[MultiDimensionalSignal]:
        """根据质量标准过滤信号
        
        Args:
            signals: 信号列表
            min_quality_score: 最小质量评分
            min_confidence: 最小置信度
            
        Returns:
            List[MultiDimensionalSignal]: 过滤后的信号列表
        """
        return [
            signal for signal in signals
            if (signal.signal_quality_score >= min_quality_score and
                signal.overall_confidence >= min_confidence)
        ]