"""
自动平仓策略实现

实现7种不同的平仓策略，每种策略都有独特的触发逻辑和风险管理机制。
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
import math

from .models import (
    PositionInfo, ClosingReason, ClosingAction, PositionCloseRequest,
    ATRInfo, VolatilityInfo, CorrelationRisk
)
from ..models.signals import MultiDimensionalSignal


logger = logging.getLogger(__name__)


class BaseClosingStrategy(ABC):
    """平仓策略基类"""
    
    def __init__(self, strategy_name: str, parameters: Optional[Dict[str, Any]] = None):
        self.strategy_name = strategy_name
        self.parameters = parameters or {}
        self.enabled = True
        self.priority = self.parameters.get('priority', 5)
        
        # 策略统计
        self.trigger_count = 0
        self.success_count = 0
        self.last_trigger_time: Optional[datetime] = None
    
    @abstractmethod
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """
        判断是否应该平仓
        
        Args:
            position: 仓位信息
            current_signal: 当前市场信号
            market_context: 市场环境上下文
            
        Returns:
            PositionCloseRequest: 平仓请求，如果不需要平仓则返回None
        """
        pass
    
    def enable(self) -> None:
        """启用策略"""
        self.enabled = True
        
    def disable(self) -> None:
        """禁用策略"""
        self.enabled = False
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """更新策略参数"""
        self.parameters.update(parameters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        return {
            'strategy_name': self.strategy_name,
            'trigger_count': self.trigger_count,
            'success_count': self.success_count,
            'success_rate': self.success_count / max(self.trigger_count, 1),
            'last_trigger_time': self.last_trigger_time,
            'enabled': self.enabled,
            'priority': self.priority
        }
    
    def _create_close_request(
        self, 
        position: PositionInfo,
        reason: ClosingReason,
        action: ClosingAction = ClosingAction.FULL_CLOSE,
        quantity_ratio: float = 1.0,
        urgency: str = "normal",
        target_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PositionCloseRequest:
        """创建平仓请求"""
        quantity_to_close = position.quantity * quantity_ratio
        
        return PositionCloseRequest(
            position_id=position.position_id,
            closing_reason=reason,
            action=action,
            quantity_to_close=quantity_to_close,
            target_price=target_price,
            urgency=urgency,
            metadata=metadata or {}
        )


class ProfitTargetStrategy(BaseClosingStrategy):
    """目标盈利平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'target_profit_pct': 5.0,      # 目标盈利百分比
            'partial_close_enabled': True,  # 是否启用分批平仓
            'first_partial_pct': 50.0,     # 第一次部分平仓比例
            'first_partial_target': 3.0,   # 第一次部分平仓目标
            'priority': 2
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("profit_target", default_params)
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否达到盈利目标"""
        if not self.enabled:
            return None
        
        target_pct = self.parameters['target_profit_pct']
        partial_enabled = self.parameters['partial_close_enabled']
        
        # 检查是否达到主要盈利目标
        if position.unrealized_pnl_pct >= target_pct:
            self.trigger_count += 1
            self.last_trigger_time = datetime.utcnow()
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.PROFIT_TARGET,
                action=ClosingAction.FULL_CLOSE,
                metadata={
                    'target_profit_pct': target_pct,
                    'actual_profit_pct': position.unrealized_pnl_pct,
                    'trigger_strategy': 'main_target'
                }
            )
        
        # 检查分批平仓条件
        if partial_enabled:
            first_partial_target = self.parameters['first_partial_target']
            first_partial_pct = self.parameters['first_partial_pct']
            
            # 检查是否已经部分平仓过
            already_partial_closed = position.metadata.get('partial_profit_closed', False)
            
            if (position.unrealized_pnl_pct >= first_partial_target and 
                not already_partial_closed):
                
                self.trigger_count += 1
                self.last_trigger_time = datetime.utcnow()
                
                return self._create_close_request(
                    position=position,
                    reason=ClosingReason.PROFIT_TARGET,
                    action=ClosingAction.PARTIAL_CLOSE,
                    quantity_ratio=first_partial_pct / 100.0,
                    metadata={
                        'partial_target_pct': first_partial_target,
                        'partial_close_ratio': first_partial_pct / 100.0,
                        'trigger_strategy': 'partial_target'
                    }
                )
        
        return None


class StopLossStrategy(BaseClosingStrategy):
    """止损平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'stop_loss_pct': -2.0,         # 止损百分比
            'emergency_stop_pct': -5.0,    # 紧急止损百分比
            'use_atr_stop': True,          # 使用ATR动态止损
            'atr_multiplier': 2.0,         # ATR倍数
            'priority': 1                  # 最高优先级
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("stop_loss", default_params)
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否触发止损"""
        if not self.enabled:
            return None
        
        stop_loss_pct = self.parameters['stop_loss_pct']
        emergency_stop_pct = self.parameters['emergency_stop_pct']
        
        # 检查紧急止损
        if position.unrealized_pnl_pct <= emergency_stop_pct:
            self.trigger_count += 1
            self.last_trigger_time = datetime.utcnow()
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.STOP_LOSS,
                action=ClosingAction.FULL_CLOSE,
                urgency="emergency",
                metadata={
                    'stop_type': 'emergency',
                    'stop_loss_pct': emergency_stop_pct,
                    'actual_loss_pct': position.unrealized_pnl_pct
                }
            )
        
        # 检查常规止损
        if position.unrealized_pnl_pct <= stop_loss_pct:
            self.trigger_count += 1
            self.last_trigger_time = datetime.utcnow()
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.STOP_LOSS,
                action=ClosingAction.FULL_CLOSE,
                urgency="high",
                metadata={
                    'stop_type': 'regular',
                    'stop_loss_pct': stop_loss_pct,
                    'actual_loss_pct': position.unrealized_pnl_pct
                }
            )
        
        # 检查ATR动态止损
        if self.parameters['use_atr_stop'] and market_context:
            atr_info = market_context.get('atr_info')
            if isinstance(atr_info, ATRInfo):
                dynamic_stop_price = self._calculate_atr_stop_price(position, atr_info)
                
                should_stop = False
                if position.is_long and position.current_price <= dynamic_stop_price:
                    should_stop = True
                elif position.is_short and position.current_price >= dynamic_stop_price:
                    should_stop = True
                
                if should_stop:
                    self.trigger_count += 1
                    self.last_trigger_time = datetime.utcnow()
                    
                    return self._create_close_request(
                        position=position,
                        reason=ClosingReason.STOP_LOSS,
                        action=ClosingAction.FULL_CLOSE,
                        urgency="high",
                        metadata={
                            'stop_type': 'atr_dynamic',
                            'atr_stop_price': dynamic_stop_price,
                            'current_atr': atr_info.current_atr,
                            'atr_multiplier': atr_info.atr_multiplier
                        }
                    )
        
        return None
    
    def _calculate_atr_stop_price(self, position: PositionInfo, atr_info: ATRInfo) -> float:
        """计算基于ATR的止损价格"""
        stop_distance = atr_info.dynamic_stop_distance
        
        if position.is_long:
            return position.entry_price - stop_distance
        else:  # short position
            return position.entry_price + stop_distance


class TrailingStopStrategy(BaseClosingStrategy):
    """跟踪止损平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'trailing_distance_pct': 1.5,   # 跟踪距离百分比
            'activation_profit_pct': 1.0,   # 激活跟踪止损的盈利百分比
            'use_atr_trailing': True,       # 使用ATR动态跟踪
            'atr_multiplier': 1.5,          # ATR跟踪倍数
            'max_trailing_distance_pct': 3.0, # 最大跟踪距离
            'priority': 2
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("trailing_stop", default_params)
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否触发跟踪止损"""
        if not self.enabled:
            return None
        
        activation_profit = self.parameters['activation_profit_pct']
        
        # 检查是否达到激活条件
        if position.unrealized_pnl_pct < activation_profit:
            return None
        
        trailing_distance_pct = self.parameters['trailing_distance_pct']
        
        # 使用ATR动态跟踪
        if self.parameters['use_atr_trailing'] and market_context:
            atr_info = market_context.get('atr_info')
            if isinstance(atr_info, ATRInfo):
                trailing_distance_pct = self._calculate_atr_trailing_distance(
                    position, atr_info
                )
        
        # 计算跟踪止损价格
        stop_price = self._calculate_trailing_stop_price(position, trailing_distance_pct)
        
        # 检查是否触发跟踪止损
        should_trigger = False
        if position.is_long and position.current_price <= stop_price:
            should_trigger = True
        elif position.is_short and position.current_price >= stop_price:
            should_trigger = True
        
        if should_trigger:
            self.trigger_count += 1
            self.last_trigger_time = datetime.utcnow()
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.TRAILING_STOP,
                action=ClosingAction.FULL_CLOSE,
                urgency="high",
                metadata={
                    'trailing_distance_pct': trailing_distance_pct,
                    'stop_price': stop_price,
                    'highest_price': position.highest_price,
                    'lowest_price': position.lowest_price
                }
            )
        
        return None
    
    def _calculate_trailing_stop_price(self, position: PositionInfo, trailing_pct: float) -> float:
        """计算跟踪止损价格"""
        if position.is_long:
            # 多头：从最高价向下跟踪
            return position.highest_price * (1 - trailing_pct / 100)
        else:
            # 空头：从最低价向上跟踪
            return position.lowest_price * (1 + trailing_pct / 100)
    
    def _calculate_atr_trailing_distance(self, position: PositionInfo, atr_info: ATRInfo) -> float:
        """基于ATR计算动态跟踪距离"""
        atr_distance_pct = (atr_info.current_atr / position.current_price) * 100 * self.parameters['atr_multiplier']
        max_distance = self.parameters['max_trailing_distance_pct']
        
        return min(atr_distance_pct, max_distance)


class TimeBasedStrategy(BaseClosingStrategy):
    """时间止损平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'max_hold_hours': 24,           # 最大持仓小时数
            'intraday_close_time': "15:30", # 日内平仓时间
            'weekend_close': True,          # 周末前平仓
            'force_close_before_events': True, # 重要事件前强制平仓
            'priority': 4
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("time_based", default_params)
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否触发时间止损"""
        if not self.enabled:
            return None
        
        current_time = datetime.utcnow()
        
        # 检查最大持仓时间
        max_hold_hours = self.parameters['max_hold_hours']
        if position.hold_duration >= timedelta(hours=max_hold_hours):
            self.trigger_count += 1
            self.last_trigger_time = current_time
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.TIME_BASED,
                action=ClosingAction.FULL_CLOSE,
                metadata={
                    'time_trigger': 'max_hold_time',
                    'hold_duration_hours': position.hold_duration.total_seconds() / 3600,
                    'max_hold_hours': max_hold_hours
                }
            )
        
        # 检查日内平仓时间
        intraday_close = self.parameters.get('intraday_close_time')
        if intraday_close:
            close_hour, close_minute = map(int, intraday_close.split(':'))
            if (current_time.hour == close_hour and 
                current_time.minute >= close_minute):
                
                self.trigger_count += 1
                self.last_trigger_time = current_time
                
                return self._create_close_request(
                    position=position,
                    reason=ClosingReason.TIME_BASED,
                    action=ClosingAction.FULL_CLOSE,
                    metadata={
                        'time_trigger': 'intraday_close',
                        'close_time': intraday_close
                    }
                )
        
        # 检查周末前平仓
        if self.parameters['weekend_close']:
            # 周五下午强制平仓
            if current_time.weekday() == 4 and current_time.hour >= 14:
                self.trigger_count += 1
                self.last_trigger_time = current_time
                
                return self._create_close_request(
                    position=position,
                    reason=ClosingReason.TIME_BASED,
                    action=ClosingAction.FULL_CLOSE,
                    metadata={
                        'time_trigger': 'weekend_close',
                        'current_weekday': current_time.weekday()
                    }
                )
        
        # 检查重要事件前平仓
        if (self.parameters['force_close_before_events'] and 
            market_context and 
            market_context.get('important_event_soon')):
            
            self.trigger_count += 1
            self.last_trigger_time = current_time
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.TIME_BASED,
                action=ClosingAction.FULL_CLOSE,
                urgency="high",
                metadata={
                    'time_trigger': 'important_event',
                    'event_info': market_context.get('event_info', {})
                }
            )
        
        return None


class TechnicalReversalStrategy(BaseClosingStrategy):
    """技术指标反转信号平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'reversal_threshold': -0.5,     # 反转信号阈值
            'confirmation_required': True,   # 需要确认信号
            'min_signal_strength': 0.6,     # 最小信号强度
            'check_momentum_reversal': True, # 检查动量反转
            'check_volume_confirmation': True, # 检查成交量确认
            'priority': 3
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("technical_reversal", default_params)
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否出现技术反转信号"""
        if not self.enabled or not current_signal:
            return None
        
        reversal_threshold = self.parameters['reversal_threshold']
        min_strength = self.parameters['min_signal_strength']
        
        # 检查信号强度是否足够
        if current_signal.overall_confidence < min_strength:
            return None
        
        # 检查方向反转
        position_direction = 1 if position.is_long else -1
        
        reversal_detected = False
        reversal_reasons = []
        
        # 检查动量反转
        if self.parameters['check_momentum_reversal']:
            momentum_signal = current_signal.momentum_score
            if (position_direction > 0 and momentum_signal < reversal_threshold) or \
               (position_direction < 0 and momentum_signal > -reversal_threshold):
                reversal_detected = True
                reversal_reasons.append('momentum_reversal')
        
        # 检查整体信号方向一致性
        direction_consensus = current_signal.signal_direction_consensus
        if (position_direction > 0 and direction_consensus < reversal_threshold) or \
           (position_direction < 0 and direction_consensus > -reversal_threshold):
            reversal_detected = True
            reversal_reasons.append('direction_consensus_reversal')
        
        # 检查成交量确认
        if self.parameters['check_volume_confirmation'] and reversal_detected:
            volume_score = current_signal.volume_score
            if volume_score < 0.4:  # 成交量不足，不确认反转
                return None
            reversal_reasons.append('volume_confirmed')
        
        # 需要确认信号时的额外检查
        if self.parameters['confirmation_required'] and reversal_detected:
            # 检查是否有多个指标确认反转
            if len(reversal_reasons) < 2:
                return None
        
        if reversal_detected:
            self.trigger_count += 1
            self.last_trigger_time = datetime.utcnow()
            
            # 根据反转强度决定平仓数量
            reversal_strength = abs(direction_consensus)
            close_ratio = min(reversal_strength * 2, 1.0)  # 最多全仓平仓
            action = ClosingAction.FULL_CLOSE if close_ratio >= 0.8 else ClosingAction.PARTIAL_CLOSE
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.TECHNICAL_REVERSAL,
                action=action,
                quantity_ratio=close_ratio,
                metadata={
                    'reversal_reasons': reversal_reasons,
                    'reversal_strength': reversal_strength,
                    'momentum_score': current_signal.momentum_score,
                    'direction_consensus': direction_consensus,
                    'volume_score': current_signal.volume_score
                }
            )
        
        return None


class SentimentStrategy(BaseClosingStrategy):
    """市场情绪剧烈变化平仓策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'sentiment_change_threshold': 0.4,  # 情绪变化阈值
            'extreme_sentiment_threshold': 0.8, # 极端情绪阈值
            'sentiment_window_minutes': 30,     # 情绪变化窗口（分钟）
            'fear_greed_threshold': 20,         # 恐惧贪婪指数阈值
            'priority': 5
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("sentiment_change", default_params)
        
        # 情绪历史跟踪
        self.sentiment_history: List[Tuple[datetime, float]] = []
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """判断是否因情绪变化平仓"""
        if not self.enabled or not current_signal:
            return None
        
        current_sentiment = current_signal.sentiment_score
        current_time = datetime.utcnow()
        
        # 添加到历史记录
        self.sentiment_history.append((current_time, current_sentiment))
        
        # 清理过期历史
        window_minutes = self.parameters['sentiment_window_minutes']
        cutoff_time = current_time - timedelta(minutes=window_minutes)
        self.sentiment_history = [
            (t, s) for t, s in self.sentiment_history if t > cutoff_time
        ]
        
        if len(self.sentiment_history) < 2:
            return None
        
        # 检查极端情绪
        extreme_threshold = self.parameters['extreme_sentiment_threshold']
        if abs(current_sentiment) >= extreme_threshold:
            self.trigger_count += 1
            self.last_trigger_time = current_time
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.SENTIMENT_CHANGE,
                action=ClosingAction.PARTIAL_CLOSE,
                quantity_ratio=0.5,  # 极端情绪时部分平仓
                metadata={
                    'trigger_type': 'extreme_sentiment',
                    'current_sentiment': current_sentiment,
                    'extreme_threshold': extreme_threshold
                }
            )
        
        # 检查情绪急剧变化
        change_threshold = self.parameters['sentiment_change_threshold']
        earliest_sentiment = self.sentiment_history[0][1]
        sentiment_change = abs(current_sentiment - earliest_sentiment)
        
        if sentiment_change >= change_threshold:
            # 检查变化方向是否对仓位不利
            position_direction = 1 if position.is_long else -1
            sentiment_direction = 1 if current_sentiment > earliest_sentiment else -1
            
            # 情绪变化对仓位不利
            if position_direction * sentiment_direction < 0:
                self.trigger_count += 1
                self.last_trigger_time = current_time
                
                return self._create_close_request(
                    position=position,
                    reason=ClosingReason.SENTIMENT_CHANGE,
                    action=ClosingAction.PARTIAL_CLOSE,
                    quantity_ratio=0.3,
                    metadata={
                        'trigger_type': 'sentiment_change',
                        'sentiment_change': sentiment_change,
                        'earlier_sentiment': earliest_sentiment,
                        'current_sentiment': current_sentiment
                    }
                )
        
        # 检查恐惧贪婪指数
        if market_context:
            fear_greed_index = market_context.get('fear_greed_index')
            if fear_greed_index is not None:
                fg_threshold = self.parameters['fear_greed_threshold']
                
                # 极度恐惧或极度贪婪
                if fear_greed_index <= fg_threshold or fear_greed_index >= (100 - fg_threshold):
                    self.trigger_count += 1
                    self.last_trigger_time = current_time
                    
                    return self._create_close_request(
                        position=position,
                        reason=ClosingReason.SENTIMENT_CHANGE,
                        action=ClosingAction.PARTIAL_CLOSE,
                        quantity_ratio=0.4,
                        metadata={
                            'trigger_type': 'fear_greed_extreme',
                            'fear_greed_index': fear_greed_index,
                            'threshold': fg_threshold
                        }
                    )
        
        return None


class DynamicTrailingStrategy(BaseClosingStrategy):
    """动态调整跟踪止损策略"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'base_trailing_pct': 2.0,       # 基础跟踪距离百分比
            'volatility_adjustment': True,   # 基于波动率调整
            'profit_acceleration': True,     # 盈利加速调整
            'correlation_adjustment': True,  # 基于相关性调整
            'min_trailing_pct': 0.5,        # 最小跟踪距离
            'max_trailing_pct': 5.0,        # 最大跟踪距离
            'adjustment_frequency_minutes': 5, # 调整频率（分钟）
            'priority': 3
        }
        if parameters:
            default_params.update(parameters)
        
        super().__init__("dynamic_trailing", default_params)
        
        # 动态参数跟踪
        self.last_adjustment_time: Optional[datetime] = None
        self.current_trailing_pct: Optional[float] = None
    
    async def should_close_position(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """动态调整跟踪止损并检查触发"""
        if not self.enabled:
            return None
        
        current_time = datetime.utcnow()
        
        # 检查是否需要调整参数
        adjustment_freq = self.parameters['adjustment_frequency_minutes']
        if (self.last_adjustment_time is None or 
            (current_time - self.last_adjustment_time).total_seconds() >= adjustment_freq * 60):
            
            self.current_trailing_pct = self._calculate_dynamic_trailing_distance(
                position, current_signal, market_context
            )
            self.last_adjustment_time = current_time
        
        if self.current_trailing_pct is None:
            self.current_trailing_pct = self.parameters['base_trailing_pct']
        
        # 计算动态跟踪止损价格
        stop_price = self._calculate_trailing_stop_price(position, self.current_trailing_pct)
        
        # 检查是否触发
        should_trigger = False
        if position.is_long and position.current_price <= stop_price:
            should_trigger = True
        elif position.is_short and position.current_price >= stop_price:
            should_trigger = True
        
        if should_trigger:
            self.trigger_count += 1
            self.last_trigger_time = current_time
            
            return self._create_close_request(
                position=position,
                reason=ClosingReason.TRAILING_STOP,
                action=ClosingAction.FULL_CLOSE,
                urgency="high",
                metadata={
                    'strategy_type': 'dynamic_trailing',
                    'dynamic_trailing_pct': self.current_trailing_pct,
                    'stop_price': stop_price,
                    'highest_price': position.highest_price,
                    'lowest_price': position.lowest_price
                }
            )
        
        return None
    
    def _calculate_dynamic_trailing_distance(
        self, 
        position: PositionInfo,
        current_signal: Optional[MultiDimensionalSignal],
        market_context: Optional[Dict[str, Any]]
    ) -> float:
        """计算动态跟踪止损距离"""
        base_pct = self.parameters['base_trailing_pct']
        min_pct = self.parameters['min_trailing_pct']
        max_pct = self.parameters['max_trailing_pct']
        
        # 基础距离
        trailing_pct = base_pct
        
        # 基于波动率调整
        if (self.parameters['volatility_adjustment'] and 
            market_context and 
            isinstance(market_context.get('volatility_info'), VolatilityInfo)):
            
            volatility_info = market_context['volatility_info']
            volatility_adjustment = volatility_info.current_volatility / volatility_info.avg_volatility
            
            # 高波动率增大跟踪距离，低波动率减小距离
            trailing_pct *= volatility_adjustment
        
        # 基于盈利状况调整
        if self.parameters['profit_acceleration'] and position.is_profitable:
            profit_ratio = position.unrealized_pnl_pct / 10.0  # 以10%为基准
            
            # 盈利越多，跟踪距离适度增加（保护利润）
            profit_adjustment = 1 + (profit_ratio * 0.2)
            trailing_pct *= profit_adjustment
        
        # 基于信号强度调整
        if current_signal:
            signal_strength = current_signal.overall_confidence
            # 信号强度高时减小跟踪距离（更紧密跟踪）
            signal_adjustment = 2 - signal_strength
            trailing_pct *= signal_adjustment
        
        # 基于相关性风险调整
        if (self.parameters['correlation_adjustment'] and 
            market_context and 
            isinstance(market_context.get('correlation_risk'), CorrelationRisk)):
            
            correlation_risk = market_context['correlation_risk']
            if correlation_risk.max_correlation > 0.7:
                # 高相关性时增加跟踪距离（降低风险）
                correlation_adjustment = 1 + (abs(correlation_risk.max_correlation) - 0.7) * 2
                trailing_pct *= correlation_adjustment
        
        # 应用限制
        trailing_pct = max(min_pct, min(trailing_pct, max_pct))
        
        return trailing_pct
    
    def _calculate_trailing_stop_price(self, position: PositionInfo, trailing_pct: float) -> float:
        """计算跟踪止损价格"""
        if position.is_long:
            return position.highest_price * (1 - trailing_pct / 100)
        else:
            return position.lowest_price * (1 + trailing_pct / 100)