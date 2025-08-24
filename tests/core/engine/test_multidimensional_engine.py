"""
多维度技术指标引擎测试

测试多维度信号生成、时间框架一致性检查和综合信号评估功能
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
from datetime import datetime, timedelta

from src.core.engine.multidimensional_engine import (
    MultiDimensionalIndicatorEngine,
    DimensionScore,
    TimeFrameConsensus,
    MarketRegime
)
from src.core.models.signals import (
    TradingSignal, 
    MultiDimensionalSignal, 
    SignalStrength
)
from src.core.indicators.timeframe import TimeFrame


class TestMultiDimensionalIndicatorEngine:
    """多维度指标引擎测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建测试引擎实例"""
        return MultiDimensionalIndicatorEngine(max_workers=2)
    
    @pytest.fixture
    def sample_market_data(self):
        """创建样本市场数据"""
        np.random.seed(42)  # 固定随机种子确保可重现
        
        # 生成100个数据点的OHLCV数据
        n_points = 100
        base_price = 100.0
        
        # 生成价格序列（带趋势和噪声）
        trend = np.linspace(0, 10, n_points)
        noise = np.random.normal(0, 2, n_points)
        closes = base_price + trend + noise
        
        # 生成OHLC数据
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        highs = closes + np.random.uniform(0, 3, n_points)
        lows = closes - np.random.uniform(0, 3, n_points)
        
        # 确保OHLC逻辑正确
        for i in range(n_points):
            high_val = max(opens[i], closes[i]) + abs(highs[i] - closes[i])
            low_val = min(opens[i], closes[i]) - abs(lows[i] - closes[i])
            highs[i] = high_val
            lows[i] = max(low_val, 0.1)  # 确保价格为正
        
        # 生成成交量数据
        volumes = np.random.uniform(1000, 10000, n_points)
        
        return {
            'open': opens.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }
    
    @pytest.fixture
    def dimension_scores(self):
        """创建样本维度评分"""
        return DimensionScore(
            momentum_score=0.6,
            trend_score=0.8,
            volatility_score=0.4,
            volume_score=0.7,
            sentiment_score=0.5,
            momentum_confidence=0.9,
            trend_confidence=0.8,
            volatility_confidence=0.7,
            volume_confidence=0.6,
            sentiment_confidence=0.5
        )
    
    def test_engine_initialization(self, engine):
        """测试引擎初始化"""
        assert engine.max_workers == 2
        assert len(engine.timeframes) == 7
        assert engine.stats['signals_generated'] == 0
        
        # 检查指标组初始化
        assert len(engine.momentum_indicators) > 0
        assert len(engine.trend_indicators) > 0
        assert len(engine.volatility_indicators) > 0
        
        # 检查关键指标存在
        assert 'rsi_14' in engine.momentum_indicators
        assert 'macd' in engine.momentum_indicators
        assert 'sma_20' in engine.trend_indicators
        assert 'bb' in engine.volatility_indicators
    
    def test_validate_market_data(self, engine, sample_market_data):
        """测试市场数据验证"""
        # 有效数据
        assert engine._validate_market_data(sample_market_data) == True
        
        # 缺少必需字段
        invalid_data = sample_market_data.copy()
        del invalid_data['close']
        assert engine._validate_market_data(invalid_data) == False
        
        # 数据长度不足
        short_data = {key: value[:10] for key, value in sample_market_data.items()}
        assert engine._validate_market_data(short_data) == False
        
        # 数据长度不一致
        inconsistent_data = sample_market_data.copy()
        inconsistent_data['high'] = inconsistent_data['high'][:-1]
        assert engine._validate_market_data(inconsistent_data) == False
    
    @pytest.mark.asyncio
    async def test_calculate_dimension_scores(self, engine, sample_market_data):
        """测试维度评分计算"""
        scores = await engine._calculate_dimension_scores(
            "BTCUSDT", sample_market_data, TimeFrame.MINUTE_15
        )
        
        assert scores is not None
        assert isinstance(scores, DimensionScore)
        
        # 检查评分范围
        assert -1 <= scores.momentum_score <= 1
        assert -1 <= scores.trend_score <= 1
        assert 0 <= scores.volatility_score <= 1
        assert 0 <= scores.volume_score <= 1
        assert -1 <= scores.sentiment_score <= 1
        
        # 检查置信度范围
        assert 0 <= scores.momentum_confidence <= 1
        assert 0 <= scores.trend_confidence <= 1
        assert 0 <= scores.volatility_confidence <= 1
        assert 0 <= scores.volume_confidence <= 1
        assert 0 <= scores.sentiment_confidence <= 1
        
        # 检查综合评分
        overall_score = scores.overall_score
        assert -1 <= overall_score <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_momentum_dimension(self, engine, sample_market_data):
        """测试动量维度计算"""
        score, confidence = await engine._calculate_momentum_dimension(
            "BTCUSDT", sample_market_data
        )
        
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert -1 <= score <= 1
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_trend_dimension(self, engine, sample_market_data):
        """测试趋势维度计算"""
        score, confidence = await engine._calculate_trend_dimension(
            "BTCUSDT", sample_market_data
        )
        
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert -1 <= score <= 1
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_volatility_dimension(self, engine, sample_market_data):
        """测试波动率维度计算"""
        score, confidence = await engine._calculate_volatility_dimension(
            "BTCUSDT", sample_market_data
        )
        
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert 0 <= score <= 1
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_volume_dimension(self, engine, sample_market_data):
        """测试成交量维度计算"""
        score, confidence = await engine._calculate_volume_dimension(
            "BTCUSDT", sample_market_data
        )
        
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert 0 <= score <= 1
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_calculate_sentiment_dimension(self, engine, sample_market_data):
        """测试情绪维度计算"""
        score, confidence = await engine._calculate_sentiment_dimension(
            "BTCUSDT", sample_market_data
        )
        
        assert isinstance(score, (int, float))
        assert isinstance(confidence, (int, float))
        assert -1 <= score <= 1
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_check_timeframe_consistency(self, engine, sample_market_data):
        """测试多时间框架一致性检查"""
        timeframes = [TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.HOUR_1]
        
        consensus_results = await engine._check_timeframe_consistency(
            "BTCUSDT", sample_market_data, timeframes
        )
        
        assert isinstance(consensus_results, list)
        # 注意：由于聚合函数简化，可能返回空列表或有限结果
        
        for consensus in consensus_results:
            assert isinstance(consensus, TimeFrameConsensus)
            assert consensus.timeframe in timeframes
            assert -1 <= consensus.trend_direction <= 1
            assert 0 <= consensus.trend_strength <= 1
            assert 0 <= consensus.volatility <= 1
            assert 0 <= consensus.weight <= 1
    
    def test_estimate_atr(self, engine, sample_market_data):
        """测试ATR估算"""
        atr = engine._estimate_atr(sample_market_data)
        
        assert isinstance(atr, (int, float))
        assert atr > 0
        
        # 测试数据不足情况
        short_data = {key: value[:5] for key, value in sample_market_data.items()}
        short_atr = engine._estimate_atr(short_data)
        assert short_atr > 0
    
    @pytest.mark.asyncio
    async def test_generate_primary_signal(self, engine, sample_market_data, dimension_scores):
        """测试主要信号生成"""
        timeframe_consensus = []  # 简化测试，不使用多时间框架
        
        signal = await engine._generate_primary_signal(
            "BTCUSDT", sample_market_data, dimension_scores, timeframe_consensus
        )
        
        assert signal is not None
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "BTCUSDT"
        assert isinstance(signal.signal_type, SignalStrength)
        assert 0 <= signal.confidence <= 1
        assert signal.entry_price > 0
        assert signal.target_price > 0
        assert signal.stop_loss > 0
        assert len(signal.reasoning) > 0
        assert len(signal.indicators_consensus) > 0
    
    def test_generate_reasoning(self, engine, dimension_scores):
        """测试推理生成"""
        timeframe_consensus = []
        overall_score = 0.7
        
        reasoning = engine._generate_reasoning(
            dimension_scores, timeframe_consensus, overall_score
        )
        
        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert all(isinstance(r, str) for r in reasoning)
    
    def test_generate_indicators_consensus(self, engine, dimension_scores):
        """测试指标共识生成"""
        consensus = engine._generate_indicators_consensus(dimension_scores)
        
        assert isinstance(consensus, dict)
        assert 'momentum' in consensus
        assert 'trend' in consensus
        assert 'volatility' in consensus
        assert 'volume' in consensus
        assert 'sentiment' in consensus
        assert 'overall' in consensus
        
        # 检查数值类型
        for key, value in consensus.items():
            assert isinstance(value, (int, float))
    
    @pytest.mark.asyncio
    async def test_create_multidimensional_signal(self, engine, sample_market_data, dimension_scores):
        """测试多维度信号创建"""
        # 先生成主信号
        primary_signal = await engine._generate_primary_signal(
            "BTCUSDT", sample_market_data, dimension_scores, []
        )
        
        assert primary_signal is not None
        
        # 创建多维度信号
        multidimensional_signal = await engine._create_multidimensional_signal(
            primary_signal, dimension_scores, []
        )
        
        assert isinstance(multidimensional_signal, MultiDimensionalSignal)
        assert multidimensional_signal.primary_signal == primary_signal
        assert -1 <= multidimensional_signal.momentum_score <= 1
        assert -1 <= multidimensional_signal.mean_reversion_score <= 1
        assert 0 <= multidimensional_signal.volatility_score <= 1
        assert 0 <= multidimensional_signal.volume_score <= 1
        assert -1 <= multidimensional_signal.sentiment_score <= 1
        assert 0 <= multidimensional_signal.overall_confidence <= 1
        assert multidimensional_signal.risk_reward_ratio >= 0
        assert 0 <= multidimensional_signal.max_position_size <= 1
        
        # 检查可选字段
        assert multidimensional_signal.market_regime is not None
        assert multidimensional_signal.technical_levels is not None
    
    def test_determine_market_regime(self, engine, dimension_scores):
        """测试市场状态判断"""
        regime = engine._determine_market_regime(dimension_scores, [])
        
        assert isinstance(regime, MarketRegime)
        assert regime in list(MarketRegime)
    
    def test_calculate_technical_levels(self, engine):
        """测试技术位计算"""
        # 创建模拟主信号
        primary_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=100.0,
            target_price=110.0,
            stop_loss=95.0,
            reasoning=["测试"],
            indicators_consensus={}
        )
        
        levels = engine._calculate_technical_levels(primary_signal)
        
        assert isinstance(levels, dict)
        required_keys = ['support', 'resistance', 'entry', 'target', 'stop_loss']
        for key in required_keys:
            assert key in levels
            assert isinstance(levels[key], (int, float))
    
    def test_calculate_timeframe_weight(self, engine):
        """测试时间框架权重计算"""
        for timeframe in engine.timeframes:
            weight = engine._calculate_timeframe_weight(timeframe)
            assert isinstance(weight, (int, float))
            assert 0 <= weight <= 1
    
    @pytest.mark.asyncio
    async def test_generate_multidimensional_signal_success(self, engine, sample_market_data):
        """测试完整多维度信号生成成功案例"""
        signal = await engine.generate_multidimensional_signal(
            "BTCUSDT", sample_market_data, enable_multiframe_analysis=False
        )
        
        # 可能返回None（如果是中性信号），这是正常的
        if signal is not None:
            assert isinstance(signal, MultiDimensionalSignal)
            assert signal.primary_signal.symbol == "BTCUSDT"
            
            # 检查统计更新
            stats = engine.get_performance_stats()
            assert stats['signals_generated'] >= 1
            assert stats['avg_processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_generate_multidimensional_signal_invalid_data(self, engine):
        """测试无效数据的信号生成"""
        invalid_data = {'close': [1, 2, 3]}  # 数据不足
        
        signal = await engine.generate_multidimensional_signal(
            "BTCUSDT", invalid_data
        )
        
        assert signal is None
    
    @pytest.mark.asyncio 
    async def test_generate_multidimensional_signal_with_multiframe(self, engine, sample_market_data):
        """测试带多时间框架分析的信号生成"""
        timeframes = [TimeFrame.MINUTE_15, TimeFrame.HOUR_1]
        
        signal = await engine.generate_multidimensional_signal(
            "BTCUSDT", 
            sample_market_data,
            timeframes=timeframes,
            enable_multiframe_analysis=True
        )
        
        # 可能返回None，这是正常的
        if signal is not None:
            assert isinstance(signal, MultiDimensionalSignal)
    
    def test_performance_stats(self, engine):
        """测试性能统计"""
        # 初始状态
        stats = engine.get_performance_stats()
        expected_keys = ['signals_generated', 'avg_processing_time', 'error_count', 'cache_hits']
        
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
        
        # 重置统计
        engine.reset_stats()
        reset_stats = engine.get_performance_stats()
        assert reset_stats['signals_generated'] == 0
        assert reset_stats['avg_processing_time'] == 0.0
        assert reset_stats['error_count'] == 0
    
    def test_cleanup(self, engine):
        """测试资源清理"""
        # 不应该抛出异常
        engine.cleanup()
        
        # 再次调用也应该安全
        engine.cleanup()


class TestDimensionScore:
    """维度评分测试类"""
    
    def test_dimension_score_creation(self):
        """测试维度评分创建"""
        score = DimensionScore(
            momentum_score=0.5,
            trend_score=-0.3,
            volatility_score=0.8,
            volume_score=0.6,
            sentiment_score=0.2,
            momentum_confidence=0.9,
            trend_confidence=0.8,
            volatility_confidence=0.7,
            volume_confidence=0.6,
            sentiment_confidence=0.5
        )
        
        assert score.momentum_score == 0.5
        assert score.trend_score == -0.3
        assert score.volatility_score == 0.8
        assert score.volume_score == 0.6
        assert score.sentiment_score == 0.2
    
    def test_overall_score_calculation(self):
        """测试综合评分计算"""
        score = DimensionScore(
            momentum_score=0.8,
            trend_score=0.6,
            volatility_score=0.4,
            volume_score=0.7,
            sentiment_score=0.5
        )
        
        overall = score.overall_score
        assert isinstance(overall, (int, float))
        assert -1 <= overall <= 1
        
        # 大致验证加权计算
        expected = (0.8 * 0.25 + 0.6 * 0.25 + 0.4 * 0.15 + 0.7 * 0.20 + 0.5 * 0.15)
        assert abs(overall - expected) < 0.01
    
    def test_overall_confidence_calculation(self):
        """测试综合置信度计算"""
        score = DimensionScore(
            momentum_score=0.5,
            trend_score=0.5,
            volatility_score=0.5,
            volume_score=0.5,
            sentiment_score=0.5,
            momentum_confidence=0.9,
            trend_confidence=0.8,
            volatility_confidence=0.7,
            volume_confidence=0.6,
            sentiment_confidence=0.5
        )
        
        confidence = score.overall_confidence
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1
        
        # 应该是所有非零置信度的平均值
        expected = (0.9 + 0.8 + 0.7 + 0.6 + 0.5) / 5
        assert abs(confidence - expected) < 0.01


class TestTimeFrameConsensus:
    """时间框架共识测试类"""
    
    def test_consensus_creation(self):
        """测试共识创建"""
        consensus = TimeFrameConsensus(
            timeframe=TimeFrame.MINUTE_15,
            trend_direction=0.6,
            trend_strength=0.8,
            volatility=0.4,
            volume_profile=0.7,
            weight=0.2
        )
        
        assert consensus.timeframe == TimeFrame.MINUTE_15
        assert consensus.trend_direction == 0.6
        assert consensus.trend_strength == 0.8
        assert consensus.volatility == 0.4
        assert consensus.volume_profile == 0.7
        assert consensus.weight == 0.2
    
    def test_bullish_consensus(self):
        """测试看涨共识判断"""
        # 看涨共识
        bullish_consensus = TimeFrameConsensus(
            timeframe=TimeFrame.MINUTE_15,
            trend_direction=0.5,  # > 0.3
            trend_strength=0.6,   # > 0.5
            volatility=0.4,
            volume_profile=0.7,
            weight=0.2
        )
        assert bullish_consensus.bullish_consensus == True
        assert bullish_consensus.bearish_consensus == False
        
        # 不满足看涨条件
        not_bullish = TimeFrameConsensus(
            timeframe=TimeFrame.MINUTE_15,
            trend_direction=0.2,  # <= 0.3
            trend_strength=0.6,
            volatility=0.4,
            volume_profile=0.7,
            weight=0.2
        )
        assert not_bullish.bullish_consensus == False
    
    def test_bearish_consensus(self):
        """测试看跌共识判断"""
        # 看跌共识
        bearish_consensus = TimeFrameConsensus(
            timeframe=TimeFrame.MINUTE_15,
            trend_direction=-0.5,  # < -0.3
            trend_strength=0.6,    # > 0.5
            volatility=0.4,
            volume_profile=0.7,
            weight=0.2
        )
        assert bearish_consensus.bearish_consensus == True
        assert bearish_consensus.bullish_consensus == False
        
        # 不满足看跌条件
        not_bearish = TimeFrameConsensus(
            timeframe=TimeFrame.MINUTE_15,
            trend_direction=-0.2,  # >= -0.3
            trend_strength=0.6,
            volatility=0.4,
            volume_profile=0.7,
            weight=0.2
        )
        assert not_bearish.bearish_consensus == False


# 集成测试
class TestIntegration:
    """集成测试类"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_signal_generation(self):
        """端到端信号生成测试"""
        engine = MultiDimensionalIndicatorEngine(max_workers=2)
        
        # 创建更真实的市场数据
        np.random.seed(123)
        n_points = 200
        
        # 模拟上涨趋势
        trend = np.linspace(0, 20, n_points)
        noise = np.random.normal(0, 1, n_points)
        closes = 100 + trend + noise
        
        opens = np.roll(closes, 1)
        opens[0] = 100
        
        highs = closes + np.random.uniform(0.5, 2, n_points)
        lows = closes - np.random.uniform(0.5, 2, n_points)
        volumes = np.random.uniform(5000, 15000, n_points)
        
        market_data = {
            'open': opens.tolist(),
            'high': highs.tolist(), 
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }
        
        try:
            # 生成信号
            signal = await engine.generate_multidimensional_signal(
                "BTCUSDT", 
                market_data,
                enable_multiframe_analysis=True
            )
            
            # 验证结果
            if signal is not None:
                assert isinstance(signal, MultiDimensionalSignal)
                assert signal.primary_signal.symbol == "BTCUSDT"
                
                # 由于是上涨趋势，应该倾向于看涨
                # 但不做严格断言，因为技术指标可能有滞后性
                
                # 验证数据完整性
                assert signal.signal_quality_score >= 0
                assert signal.signal_direction_consensus is not None
                
                # 获取仓位建议
                position_size = signal.get_position_sizing_recommendation(
                    base_position_size=1.0,
                    risk_tolerance=0.8
                )
                assert 0 <= position_size <= 1
            
            # 检查统计
            stats = engine.get_performance_stats()
            assert stats['signals_generated'] >= 0
            assert stats['error_count'] == 0
            
        finally:
            engine.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.performance 
    async def test_performance_benchmark(self):
        """性能基准测试"""
        import time
        
        engine = MultiDimensionalIndicatorEngine(max_workers=4)
        
        # 创建大量数据
        np.random.seed(456)
        n_points = 1000
        
        closes = 100 + np.cumsum(np.random.normal(0, 1, n_points))
        opens = np.roll(closes, 1)
        opens[0] = 100
        
        highs = closes + np.random.uniform(0, 3, n_points)
        lows = closes - np.random.uniform(0, 3, n_points)
        volumes = np.random.uniform(1000, 20000, n_points)
        
        market_data = {
            'open': opens.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }
        
        try:
            # 性能测试
            start_time = time.time()
            
            tasks = []
            for i in range(5):  # 并发生成5个信号
                task = engine.generate_multidimensional_signal(
                    f"TEST{i}USDT", 
                    market_data,
                    enable_multiframe_analysis=False  # 简化以提高速度
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 性能断言
            assert processing_time < 10.0  # 应该在10秒内完成
            
            # 检查结果
            successful_results = [r for r in results if isinstance(r, (MultiDimensionalSignal, type(None)))]
            assert len(successful_results) == 5
            
            # 检查统计
            stats = engine.get_performance_stats()
            assert stats['avg_processing_time'] > 0
            
        finally:
            engine.cleanup()


if __name__ == '__main__':
    # 运行测试
    pytest.main([
        __file__, 
        '-v',
        '--tb=short',
        '-m', 'not integration and not performance'  # 默认跳过集成和性能测试
    ])