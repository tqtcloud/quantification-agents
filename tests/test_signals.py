"""
信号数据模型测试

测试SignalStrength枚举、TradingSignal和MultiDimensionalSignal数据类的
功能和验证逻辑
"""

import pytest
from datetime import datetime
from src.core.models.signals import (
    SignalStrength,
    TradingSignal,
    MultiDimensionalSignal,
    SignalAggregator,
)


class TestSignalStrength:
    """测试信号强度枚举"""
    
    def test_signal_strength_values(self):
        """测试信号强度枚举值"""
        assert SignalStrength.STRONG_SELL.value == -1
        assert SignalStrength.SELL.value == 0
        assert SignalStrength.WEAK_SELL.value == 1
        assert SignalStrength.NEUTRAL.value == 2
        assert SignalStrength.WEAK_BUY.value == 3
        assert SignalStrength.BUY.value == 4
        assert SignalStrength.STRONG_BUY.value == 5


class TestTradingSignal:
    """测试基础交易信号数据类"""
    
    def test_valid_buy_signal_creation(self):
        """测试创建有效的买入信号"""
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=55000.0,
            stop_loss=48000.0,
            reasoning=["技术指标看涨", "成交量放大"],
            indicators_consensus={"RSI": 0.7, "MACD": 0.6}
        )
        
        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalStrength.BUY
        assert signal.confidence == 0.8
        assert signal._is_buy_signal() is True
        assert signal._is_sell_signal() is False
        assert signal.is_valid is True
    
    def test_valid_sell_signal_creation(self):
        """测试创建有效的卖出信号"""
        signal = TradingSignal(
            symbol="ETHUSDT",
            signal_type=SignalStrength.SELL,
            confidence=0.75,
            entry_price=3000.0,
            target_price=2800.0,
            stop_loss=3200.0,
            reasoning=["阻力位回落", "RSI超买"],
            indicators_consensus={"RSI": 0.2, "MACD": 0.3}
        )
        
        assert signal.symbol == "ETHUSDT" 
        assert signal.signal_type == SignalStrength.SELL
        assert signal._is_sell_signal() is True
        assert signal.is_valid is True
    
    def test_risk_reward_ratio_calculation(self):
        """测试风险收益比计算"""
        buy_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=55000.0,  # +5000 profit
            stop_loss=48000.0,     # -2000 loss
            reasoning=["测试"],
            indicators_consensus={}
        )
        
        expected_ratio = 5000 / 2000  # 2.5
        assert abs(buy_signal.risk_reward_ratio - expected_ratio) < 0.001
    
    def test_invalid_confidence_validation(self):
        """测试置信度验证"""
        with pytest.raises(ValueError, match="置信度必须在0-1之间"):
            TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.BUY,
                confidence=1.5,  # 无效的置信度
                entry_price=50000.0,
                target_price=55000.0,
                stop_loss=48000.0,
                reasoning=["测试"],
                indicators_consensus={}
            )
    
    def test_invalid_price_validation(self):
        """测试价格验证"""
        with pytest.raises(ValueError, match="入场价格必须大于0"):
            TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.BUY,
                confidence=0.8,
                entry_price=-100.0,  # 无效的价格
                target_price=55000.0,
                stop_loss=48000.0,
                reasoning=["测试"],
                indicators_consensus={}
            )
    
    def test_invalid_buy_signal_logic(self):
        """测试买入信号价格逻辑验证"""
        with pytest.raises(ValueError, match="买入信号的目标价格必须高于入场价格"):
            TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.BUY,
                confidence=0.8,
                entry_price=50000.0,
                target_price=45000.0,  # 目标价低于入场价
                stop_loss=48000.0,
                reasoning=["测试"],
                indicators_consensus={}
            )


class TestMultiDimensionalSignal:
    """测试多维度信号数据类"""
    
    def create_valid_primary_signal(self):
        """创建有效的主信号"""
        return TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=55000.0,
            stop_loss=48000.0,
            reasoning=["技术指标看涨"],
            indicators_consensus={"RSI": 0.7}
        )
    
    def test_valid_multidimensional_signal_creation(self):
        """测试创建有效的多维度信号"""
        primary_signal = self.create_valid_primary_signal()
        
        multi_signal = MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=0.7,
            mean_reversion_score=-0.2,
            volatility_score=0.3,
            volume_score=0.8,
            sentiment_score=0.6,
            overall_confidence=0.85,
            risk_reward_ratio=2.5,
            max_position_size=0.3
        )
        
        assert multi_signal.primary_signal.symbol == "BTCUSDT"
        assert multi_signal.momentum_score == 0.7
        assert multi_signal.overall_confidence == 0.85
    
    def test_signal_quality_score_calculation(self):
        """测试信号质量评分计算"""
        primary_signal = self.create_valid_primary_signal()
        
        multi_signal = MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=0.8,
            mean_reversion_score=0.0,
            volatility_score=0.2,  # 低波动率，好
            volume_score=0.9,
            sentiment_score=0.7,
            overall_confidence=0.9,
            risk_reward_ratio=3.0,  # 好的风险收益比
            max_position_size=0.5
        )
        
        quality_score = multi_signal.signal_quality_score
        assert 0 <= quality_score <= 1
        assert quality_score > 0.7  # 应该是高质量信号
    
    def test_signal_direction_consensus(self):
        """测试信号方向一致性"""
        primary_signal = self.create_valid_primary_signal()
        
        # 创建方向一致的多维度信号
        multi_signal = MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=0.8,      # 正向
            mean_reversion_score=0.6, # 正向
            volatility_score=0.2,
            volume_score=0.9,
            sentiment_score=0.7,     # 正向
            overall_confidence=0.9,
            risk_reward_ratio=2.5,
            max_position_size=0.5
        )
        
        consensus = multi_signal.signal_direction_consensus
        assert consensus > 0  # 买入方向一致性应该为正
    
    def test_position_sizing_recommendation(self):
        """测试仓位大小建议"""
        primary_signal = self.create_valid_primary_signal()
        
        multi_signal = MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=0.8,
            mean_reversion_score=0.6,
            volatility_score=0.2,
            volume_score=0.9,
            sentiment_score=0.7,
            overall_confidence=0.9,
            risk_reward_ratio=3.0,
            max_position_size=0.5
        )
        
        recommended_size = multi_signal.get_position_sizing_recommendation(
            base_position_size=1.0,
            risk_tolerance=0.8
        )
        
        assert 0 <= recommended_size <= multi_signal.max_position_size
        assert recommended_size > 0  # 应该推荐一定仓位
    
    def test_invalid_score_validation(self):
        """测试分数范围验证"""
        primary_signal = self.create_valid_primary_signal()
        
        with pytest.raises(ValueError, match="动量分数必须在-1到1之间"):
            MultiDimensionalSignal(
                primary_signal=primary_signal,
                momentum_score=2.0,  # 超出范围
                mean_reversion_score=0.0,
                volatility_score=0.3,
                volume_score=0.8,
                sentiment_score=0.6,
                overall_confidence=0.85,
                risk_reward_ratio=2.5,
                max_position_size=0.3
            )


class TestSignalAggregator:
    """测试信号聚合器"""
    
    def create_test_signals(self):
        """创建测试信号列表"""
        signals = []
        
        for i in range(3):
            primary_signal = TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.BUY,
                confidence=0.7 + i * 0.1,
                entry_price=50000.0,
                target_price=55000.0,
                stop_loss=48000.0,
                reasoning=[f"信号{i}"],
                indicators_consensus={"RSI": 0.7}
            )
            
            multi_signal = MultiDimensionalSignal(
                primary_signal=primary_signal,
                momentum_score=0.5 + i * 0.1,
                mean_reversion_score=0.2,
                volatility_score=0.3,
                volume_score=0.8,
                sentiment_score=0.6,
                overall_confidence=0.7 + i * 0.1,
                risk_reward_ratio=2.0 + i * 0.5,
                max_position_size=0.3 + i * 0.1
            )
            
            signals.append(multi_signal)
        
        return signals
    
    def test_combine_signals(self):
        """测试信号组合"""
        signals = self.create_test_signals()
        combined = SignalAggregator.combine_signals(signals)
        
        assert combined is not None
        assert combined.primary_signal.symbol == "BTCUSDT"
        # 验证是否选择了置信度最高的主信号
        assert combined.primary_signal.confidence == 0.9
    
    def test_combine_single_signal(self):
        """测试单一信号组合"""
        signals = self.create_test_signals()[:1]
        combined = SignalAggregator.combine_signals(signals)
        
        assert combined is signals[0]
    
    def test_combine_empty_signals(self):
        """测试空信号列表组合"""
        combined = SignalAggregator.combine_signals([])
        assert combined is None
    
    def test_filter_signals_by_quality(self):
        """测试按质量过滤信号"""
        signals = self.create_test_signals()
        
        # 设置较高的过滤标准
        filtered = SignalAggregator.filter_signals_by_quality(
            signals,
            min_quality_score=0.1,  # 相对宽松的质量要求
            min_confidence=0.8      # 较高的置信度要求
        )
        
        # 应该过滤掉置信度低于0.8的信号
        assert len(filtered) < len(signals)
        for signal in filtered:
            assert signal.overall_confidence >= 0.8