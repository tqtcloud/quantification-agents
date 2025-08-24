"""
测试平仓策略

测试所有7种平仓策略的触发逻辑和参数配置。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.core.position.models import (
    PositionInfo, ClosingReason, ClosingAction, ATRInfo, VolatilityInfo, CorrelationRisk
)
from src.core.position.closing_strategies import (
    ProfitTargetStrategy, StopLossStrategy, TrailingStopStrategy, TimeBasedStrategy,
    TechnicalReversalStrategy, SentimentStrategy, DynamicTrailingStrategy
)
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength


@pytest.fixture
def sample_long_position():
    """创建示例多头仓位"""
    return PositionInfo(
        position_id="LONG_001",
        symbol="BTCUSDT",
        entry_price=50000.0,
        current_price=51000.0,
        quantity=0.5,
        side="long",
        entry_time=datetime.utcnow(),
        unrealized_pnl=500.0,
        unrealized_pnl_pct=2.0
    )


@pytest.fixture
def sample_short_position():
    """创建示例空头仓位"""
    return PositionInfo(
        position_id="SHORT_001",
        symbol="BTCUSDT",
        entry_price=50000.0,
        current_price=49000.0,
        quantity=0.5,
        side="short",
        entry_time=datetime.utcnow(),
        unrealized_pnl=500.0,
        unrealized_pnl_pct=2.0
    )


@pytest.fixture
def sample_signal():
    """创建示例信号"""
    trading_signal = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.8,
        entry_price=50000.0,
        target_price=52000.0,
        stop_loss=48000.0,
        reasoning=["Technical breakout", "Volume confirmation"],
        indicators_consensus={"ma": 0.7, "rsi": 0.6}
    )
    
    return MultiDimensionalSignal(
        primary_signal=trading_signal,
        momentum_score=0.6,
        mean_reversion_score=-0.2,
        volatility_score=0.4,
        volume_score=0.8,
        sentiment_score=0.3,
        overall_confidence=0.7,
        risk_reward_ratio=2.5,
        max_position_size=0.8
    )


class TestProfitTargetStrategy:
    """测试目标盈利策略"""
    
    @pytest.mark.asyncio
    async def test_profit_target_triggered(self, sample_long_position):
        """测试盈利目标触发"""
        strategy = ProfitTargetStrategy({"target_profit_pct": 3.0})
        
        # 设置盈利达到目标
        sample_long_position.update_price(51500.0)  # 3% profit
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.closing_reason == ClosingReason.PROFIT_TARGET
        assert request.action == ClosingAction.FULL_CLOSE
        assert strategy.trigger_count == 1
    
    @pytest.mark.asyncio
    async def test_profit_target_not_triggered(self, sample_long_position):
        """测试盈利目标未触发"""
        strategy = ProfitTargetStrategy({"target_profit_pct": 5.0})
        
        # 盈利未达到目标
        sample_long_position.update_price(51000.0)  # 2% profit < 5%
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is None
        assert strategy.trigger_count == 0
    
    @pytest.mark.asyncio
    async def test_partial_profit_target(self, sample_long_position):
        """测试部分盈利目标"""
        strategy = ProfitTargetStrategy({
            "target_profit_pct": 5.0,
            "partial_close_enabled": True,
            "first_partial_target": 3.0,
            "first_partial_pct": 50.0
        })
        
        # 达到部分平仓目标
        sample_long_position.update_price(51500.0)  # 3% profit
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.action == ClosingAction.PARTIAL_CLOSE
        assert request.quantity_to_close == 0.25  # 50% of 0.5
        assert request.metadata["trigger_strategy"] == "partial_target"
    
    @pytest.mark.asyncio
    async def test_disabled_strategy(self, sample_long_position):
        """测试禁用策略"""
        strategy = ProfitTargetStrategy({"target_profit_pct": 1.0})
        strategy.disable()
        
        sample_long_position.update_price(52000.0)  # 4% profit
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is None


class TestStopLossStrategy:
    """测试止损策略"""
    
    @pytest.mark.asyncio
    async def test_regular_stop_loss(self, sample_long_position):
        """测试常规止损"""
        strategy = StopLossStrategy({"stop_loss_pct": -3.0})
        
        # 设置亏损达到止损
        sample_long_position.update_price(48500.0)  # -3% loss
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.closing_reason == ClosingReason.STOP_LOSS
        assert request.urgency == "high"
        assert request.metadata["stop_type"] == "regular"
    
    @pytest.mark.asyncio
    async def test_emergency_stop_loss(self, sample_long_position):
        """测试紧急止损"""
        strategy = StopLossStrategy({
            "stop_loss_pct": -3.0,
            "emergency_stop_pct": -5.0
        })
        
        # 设置亏损达到紧急止损
        sample_long_position.update_price(47500.0)  # -5% loss
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.urgency == "emergency"
        assert request.metadata["stop_type"] == "emergency"
    
    @pytest.mark.asyncio
    async def test_atr_stop_loss(self, sample_long_position):
        """测试ATR动态止损"""
        strategy = StopLossStrategy({
            "stop_loss_pct": -3.0,
            "use_atr_stop": True,
            "atr_multiplier": 2.0
        })
        
        atr_info = ATRInfo(
            period=14,
            current_atr=500.0,  # ATR = 500
            atr_multiplier=2.0
        )
        
        market_context = {"atr_info": atr_info}
        
        # ATR止损价格 = 50000 - (500 * 2) = 49000
        sample_long_position.update_price(48900.0)  # 低于ATR止损价格
        
        request = await strategy.should_close_position(
            sample_long_position, 
            market_context=market_context
        )
        
        assert request is not None
        assert request.metadata["stop_type"] == "atr_dynamic"
        assert request.metadata["atr_stop_price"] == 49000.0


class TestTrailingStopStrategy:
    """测试跟踪止损策略"""
    
    @pytest.mark.asyncio
    async def test_trailing_stop_activation(self, sample_long_position):
        """测试跟踪止损激活"""
        strategy = TrailingStopStrategy({
            "trailing_distance_pct": 2.0,
            "activation_profit_pct": 1.0
        })
        
        # 盈利不足，未激活跟踪止损
        sample_long_position.update_price(50400.0)  # 0.8% profit < 1%
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is None
    
    @pytest.mark.asyncio
    async def test_trailing_stop_triggered(self, sample_long_position):
        """测试跟踪止损触发"""
        strategy = TrailingStopStrategy({
            "trailing_distance_pct": 2.0,
            "activation_profit_pct": 1.0
        })
        
        # 先设置最高价
        sample_long_position.update_price(52000.0)  # 4% profit, highest = 52000
        
        # 然后价格回落触发跟踪止损
        # 跟踪止损价格 = 52000 * (1 - 2/100) = 50960
        sample_long_position.update_price(50900.0)  # 低于跟踪止损价格
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.closing_reason == ClosingReason.TRAILING_STOP
        assert request.metadata["stop_price"] == 50960.0
    
    @pytest.mark.asyncio
    async def test_atr_trailing_stop(self, sample_long_position):
        """测试基于ATR的跟踪止损"""
        strategy = TrailingStopStrategy({
            "activation_profit_pct": 1.0,
            "use_atr_trailing": True,
            "atr_multiplier": 1.5,
            "max_trailing_distance_pct": 3.0
        })
        
        atr_info = ATRInfo(
            current_atr=800.0,  # ATR = 800
            atr_multiplier=1.5
        )
        
        market_context = {"atr_info": atr_info}
        
        # 设置最高价
        sample_long_position.update_price(52000.0)  # highest = 52000
        
        # ATR跟踪距离 = (800 / 52000) * 100 * 1.5 ≈ 2.3%
        # 跟踪止损价格 = 52000 * (1 - 2.3/100) ≈ 50804
        sample_long_position.update_price(50700.0)
        
        request = await strategy.should_close_position(
            sample_long_position,
            market_context=market_context
        )
        
        assert request is not None


class TestTimeBasedStrategy:
    """测试时间止损策略"""
    
    @pytest.mark.asyncio
    async def test_max_hold_time_trigger(self, sample_long_position):
        """测试最大持仓时间触发"""
        strategy = TimeBasedStrategy({"max_hold_hours": 2})
        
        # 修改入场时间为3小时前
        sample_long_position.entry_time = datetime.utcnow() - timedelta(hours=3)
        
        request = await strategy.should_close_position(sample_long_position)
        
        assert request is not None
        assert request.closing_reason == ClosingReason.TIME_BASED
        assert request.metadata["time_trigger"] == "max_hold_time"
    
    @pytest.mark.asyncio
    async def test_weekend_close(self, sample_long_position):
        """测试周末前平仓"""
        strategy = TimeBasedStrategy({
            "max_hold_hours": 24,
            "weekend_close": True
        })
        
        # 模拟周五下午
        with patch('src.core.position.closing_strategies.datetime') as mock_datetime:
            friday_afternoon = datetime(2023, 12, 15, 15, 0)  # 周五下午3点
            mock_datetime.utcnow.return_value = friday_afternoon
            
            request = await strategy.should_close_position(sample_long_position)
            
            assert request is not None
            assert request.metadata["time_trigger"] == "weekend_close"
    
    @pytest.mark.asyncio
    async def test_important_event_close(self, sample_long_position):
        """测试重要事件前平仓"""
        strategy = TimeBasedStrategy({
            "max_hold_hours": 24,
            "force_close_before_events": True
        })
        
        market_context = {
            "important_event_soon": True,
            "event_info": {"event": "FOMC Meeting", "time": "2023-12-20 20:00"}
        }
        
        request = await strategy.should_close_position(
            sample_long_position,
            market_context=market_context
        )
        
        assert request is not None
        assert request.metadata["time_trigger"] == "important_event"
        assert request.urgency == "high"


class TestTechnicalReversalStrategy:
    """测试技术反转策略"""
    
    @pytest.mark.asyncio
    async def test_momentum_reversal_long(self, sample_long_position, sample_signal):
        """测试多头动量反转"""
        strategy = TechnicalReversalStrategy({
            "reversal_threshold": -0.5,
            "min_signal_strength": 0.6,
            "check_momentum_reversal": True
        })
        
        # 创建反转信号（负动量）
        sample_signal.momentum_score = -0.6
        sample_signal.overall_confidence = 0.7
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is not None
        assert request.closing_reason == ClosingReason.TECHNICAL_REVERSAL
        assert "momentum_reversal" in request.metadata["reversal_reasons"]
    
    @pytest.mark.asyncio
    async def test_insufficient_signal_strength(self, sample_long_position, sample_signal):
        """测试信号强度不足"""
        strategy = TechnicalReversalStrategy({
            "reversal_threshold": -0.5,
            "min_signal_strength": 0.8
        })
        
        sample_signal.momentum_score = -0.6
        sample_signal.overall_confidence = 0.5  # 低于最小要求
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is None
    
    @pytest.mark.asyncio
    async def test_volume_confirmation_required(self, sample_long_position, sample_signal):
        """测试需要成交量确认"""
        strategy = TechnicalReversalStrategy({
            "reversal_threshold": -0.5,
            "min_signal_strength": 0.6,
            "check_volume_confirmation": True
        })
        
        sample_signal.momentum_score = -0.6
        sample_signal.volume_score = 0.3  # 成交量不足
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is None  # 成交量不足，不确认反转


class TestSentimentStrategy:
    """测试情绪变化策略"""
    
    @pytest.mark.asyncio
    async def test_extreme_sentiment(self, sample_long_position, sample_signal):
        """测试极端情绪"""
        strategy = SentimentStrategy({
            "extreme_sentiment_threshold": 0.8
        })
        
        sample_signal.sentiment_score = 0.9  # 极度贪婪
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is not None
        assert request.action == ClosingAction.PARTIAL_CLOSE
        assert request.quantity_to_close == 0.25  # 50% of 0.5
        assert request.metadata["trigger_type"] == "extreme_sentiment"
    
    @pytest.mark.asyncio
    async def test_sentiment_change(self, sample_long_position, sample_signal):
        """测试情绪变化"""
        strategy = SentimentStrategy({
            "sentiment_change_threshold": 0.4,
            "sentiment_window_minutes": 30
        })
        
        # 先添加历史情绪数据
        old_time = datetime.utcnow() - timedelta(minutes=20)
        strategy.sentiment_history.append((old_time, 0.8))  # 之前很乐观
        
        sample_signal.sentiment_score = 0.2  # 现在悲观
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is not None
        assert request.metadata["trigger_type"] == "sentiment_change"
        assert abs(request.metadata["sentiment_change"] - 0.6) < 0.01  # |0.2 - 0.8| = 0.6
    
    @pytest.mark.asyncio
    async def test_fear_greed_index(self, sample_long_position, sample_signal):
        """测试恐惧贪婪指数"""
        strategy = SentimentStrategy({
            "fear_greed_threshold": 20
        })
        
        market_context = {
            "fear_greed_index": 15  # 极度恐惧
        }
        
        request = await strategy.should_close_position(
            sample_long_position,
            current_signal=sample_signal,
            market_context=market_context
        )
        
        assert request is not None
        assert request.metadata["trigger_type"] == "fear_greed_extreme"


class TestDynamicTrailingStrategy:
    """测试动态跟踪策略"""
    
    @pytest.mark.asyncio
    async def test_dynamic_trailing_adjustment(self, sample_long_position, sample_signal):
        """测试动态跟踪调整"""
        strategy = DynamicTrailingStrategy({
            "base_trailing_pct": 2.0,
            "volatility_adjustment": True,
            "profit_acceleration": True
        })
        
        volatility_info = VolatilityInfo(
            current_volatility=0.06,
            avg_volatility=0.04,
            volatility_percentile=0.7
        )
        
        market_context = {"volatility_info": volatility_info}
        
        # 设置盈利状态
        sample_long_position.update_price(52000.0)  # 4% profit
        
        # 计算动态距离
        dynamic_distance = strategy._calculate_dynamic_trailing_distance(
            sample_long_position, sample_signal, market_context
        )
        
        # 基础2% * 波动率调整(1.5) * 盈利调整(1.08) * 信号调整(1.3) ≈ 4.2%
        assert dynamic_distance > 2.0  # 应该大于基础值
        assert dynamic_distance < 6.0  # 但在合理范围内
    
    @pytest.mark.asyncio
    async def test_trailing_trigger_with_dynamic_distance(self, sample_long_position, sample_signal):
        """测试动态距离下的跟踪止损触发"""
        strategy = DynamicTrailingStrategy({
            "base_trailing_pct": 2.0,
            "adjustment_frequency_minutes": 1
        })
        
        # 设置最高价和当前价
        sample_long_position.update_price(52000.0)  # highest = 52000
        sample_long_position.update_price(50500.0)  # 当前价格
        
        # 强制设置动态距离
        strategy.current_trailing_pct = 3.0
        
        # 计算止损价格 = 52000 * (1 - 3/100) = 50440
        # 当前价格 50500 > 50440，不应触发
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is None
        
        # 价格继续下跌
        sample_long_position.update_price(50400.0)  # 低于止损价格
        
        request = await strategy.should_close_position(
            sample_long_position, 
            current_signal=sample_signal
        )
        
        assert request is not None
        assert request.metadata["strategy_type"] == "dynamic_trailing"


if __name__ == "__main__":
    pytest.main([__file__])