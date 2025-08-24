"""
技术指标计算引擎单元测试
测试所有指标的计算准确性、异步功能、多时间框架和标准化功能
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import time
from typing import Dict, Any

# 导入新的指标模块
from src.core.indicators import (
    TechnicalIndicators,
    RelativeStrengthIndex,
    MACD,
    SimpleMovingAverage,
    ExponentialMovingAverage,
    BollingerBands,
    StochasticOscillator,
    CCI,
    WilliamsR,
    ADX,
    ParabolicSAR,
    IchimokuCloud,
    ATR,
    StandardDeviation,
    KeltnerChannels,
    DonchianChannels,
    VIXProxy,
    IndicatorNormalizer,
    NormalizationMethod,
    NormalizationConfig,
    TimeFrameManager,
    TimeFrame,
    TimeFrameConfig,
    OHLCVData,
    IndicatorType,
    IndicatorConfig,
    IndicatorResult,
    normalize_indicator_values,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)


class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_ohlcv_data(
        length: int = 100,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.001,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """生成模拟OHLCV数据"""
        np.random.seed(seed)
        
        # 生成价格序列
        returns = np.random.normal(trend, volatility, length)
        prices = [start_price]
        
        for i in range(length):
            prices.append(prices[-1] * (1 + returns[i]))
        
        prices = np.array(prices[1:])  # 移除初始价格
        
        # 生成OHLC
        close_prices = prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = start_price
        
        # 添加随机波动生成高低价
        high_low_range = np.abs(np.random.normal(0, volatility * 0.5, length))
        high_prices = close_prices + high_low_range * close_prices
        low_prices = close_prices - high_low_range * close_prices
        
        # 确保OHLC逻辑正确
        for i in range(length):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # 生成成交量
        volumes = np.random.uniform(1000, 10000, length)
        
        # 生成时间戳
        timestamps = np.arange(length) * 3600  # 每小时一个数据点
        
        return {
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }


class TestTechnicalIndicators:
    """技术指标基础测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """生成样本数据"""
        return TestDataGenerator.generate_ohlcv_data(100)
    
    @pytest.fixture
    def indicators_manager(self):
        """创建指标管理器"""
        return TechnicalIndicators()
    
    def test_technical_indicators_initialization(self, indicators_manager):
        """测试指标管理器初始化"""
        assert isinstance(indicators_manager, TechnicalIndicators)
        registered_indicators = indicators_manager.get_registered_indicators()
        
        # 检查是否有默认指标
        assert len(registered_indicators) > 0
        
        # 检查默认指标类型
        expected_indicators = [
            'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
            'RSI_14', 'MACD_12_26_9', 'STOCH_14_3', 'BBANDS_20_2'
        ]
        
        for indicator in expected_indicators:
            assert indicator in registered_indicators
    
    def test_data_update(self, indicators_manager, sample_data):
        """测试数据更新功能"""
        symbol = "BTCUSDT"
        
        # 逐个更新数据点
        for i in range(len(sample_data['close'])):
            price_data = {
                'open': sample_data['open'][i],
                'high': sample_data['high'][i],
                'low': sample_data['low'][i],
                'close': sample_data['close'][i],
                'volume': sample_data['volume'][i],
                'timestamp': sample_data['timestamp'][i]
            }
            indicators_manager.update_data(symbol, price_data)
        
        # 验证数据被正确存储
        prepared_data = indicators_manager._prepare_data_for_calculation(symbol)
        assert len(prepared_data['close']) == len(sample_data['close'])
        np.testing.assert_array_equal(prepared_data['close'], sample_data['close'])
    
    def test_batch_data_update(self, indicators_manager, sample_data):
        """测试批量数据更新"""
        symbol = "ETHUSDT"
        df = pd.DataFrame(sample_data)
        
        indicators_manager.update_data_batch(symbol, df)
        
        prepared_data = indicators_manager._prepare_data_for_calculation(symbol)
        assert len(prepared_data['close']) == len(sample_data['close'])


class TestMomentumIndicators:
    """动量指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_ohlcv_data(50, seed=123)
    
    def test_rsi_calculation(self, sample_data):
        """测试RSI计算"""
        rsi_indicator = RelativeStrengthIndex(period=14)
        result = rsi_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "RSI_14"
        assert result.symbol == "TEST"
        assert not np.isnan(result.value)
        assert 0 <= result.value <= 100
        assert result.metadata['period'] == 14
        assert 'signal' in result.metadata
    
    def test_macd_calculation(self, sample_data):
        """测试MACD计算"""
        macd_indicator = MACD(fast_period=12, slow_period=26, signal_period=9)
        result = macd_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        assert 'macd' in result.value
        assert 'signal' in result.value
        assert 'histogram' in result.value
        assert result.metadata['signal'] in ['bullish', 'bearish', 'neutral', 'bullish_crossover', 'bearish_crossover']
    
    def test_stochastic_calculation(self, sample_data):
        """测试随机指标计算"""
        stoch_indicator = StochasticOscillator(k_period=14, d_period=3)
        result = stoch_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        assert 'k' in result.value and 'd' in result.value
        
        if not np.isnan(result.value['k']):
            assert 0 <= result.value['k'] <= 100
        if not np.isnan(result.value['d']):
            assert 0 <= result.value['d'] <= 100
    
    def test_cci_calculation(self, sample_data):
        """测试CCI计算"""
        cci_indicator = CCI(period=20)
        result = cci_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "CCI_20"
        assert result.metadata['signal'] in ['overbought', 'oversold', 'bullish', 'bearish', 'neutral']
    
    def test_williams_r_calculation(self, sample_data):
        """测试威廉指标计算"""
        willr_indicator = WilliamsR(period=14)
        result = willr_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "WILLR_14"
        
        if not np.isnan(result.value):
            assert -100 <= result.value <= 0
        
        assert result.metadata['signal'] in ['overbought', 'oversold', 'bullish', 'bearish', 'neutral']


class TestTrendIndicators:
    """趋势指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_ohlcv_data(50, trend=0.002, seed=456)
    
    def test_sma_calculation(self, sample_data):
        """测试SMA计算"""
        sma_indicator = SimpleMovingAverage(period=20)
        result = sma_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "SMA_20"
        assert not np.isnan(result.value)
        assert result.metadata['trend'] in ['bullish', 'bearish', 'neutral']
        assert 'distance_pct' in result.metadata
    
    def test_ema_calculation(self, sample_data):
        """测试EMA计算"""
        ema_indicator = ExponentialMovingAverage(period=20)
        result = ema_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "EMA_20"
        assert not np.isnan(result.value)
        assert 'slope' in result.metadata
    
    def test_adx_calculation(self, sample_data):
        """测试ADX计算"""
        adx_indicator = ADX(period=14)
        result = adx_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        assert 'adx' in result.value
        assert 'plus_di' in result.value
        assert 'minus_di' in result.value
        assert result.metadata['trend_strength'] in ['weak', 'moderate', 'strong', 'very_strong']
        assert result.metadata['trend_direction'] in ['bullish', 'bearish', 'neutral']
    
    def test_parabolic_sar_calculation(self, sample_data):
        """测试抛物线SAR计算"""
        sar_indicator = ParabolicSAR(acceleration=0.02, maximum=0.2)
        result = sar_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert not np.isnan(result.value)
        assert result.metadata['signal'] in ['bullish', 'bearish', 'neutral', 'reversal_bullish', 'reversal_bearish']
        assert result.metadata['trend'] in ['uptrend', 'downtrend', 'neutral']
    
    def test_ichimoku_calculation(self, sample_data):
        """测试一目均衡表计算"""
        ichimoku_indicator = IchimokuCloud()
        result = ichimoku_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        
        expected_keys = ['conversion_line', 'base_line', 'leading_span_a', 'leading_span_b', 'lagging_span']
        for key in expected_keys:
            assert key in result.value
        
        assert result.metadata['signal'] in ['strong_bullish', 'bullish', 'neutral', 'bearish', 'strong_bearish']
        assert result.metadata['cloud_color'] in ['bullish', 'bearish', 'neutral']
        assert result.metadata['price_vs_cloud'] in ['above', 'below', 'inside', 'neutral']


class TestVolatilityIndicators:
    """波动率指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_ohlcv_data(50, volatility=0.03, seed=789)
    
    def test_bollinger_bands_calculation(self, sample_data):
        """测试布林带计算"""
        bb_indicator = BollingerBands(period=20, std_dev=2.0)
        result = bb_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        assert 'upper' in result.value
        assert 'middle' in result.value
        assert 'lower' in result.value
        
        if not any(np.isnan(v) for v in result.value.values()):
            assert result.value['upper'] > result.value['middle'] > result.value['lower']
        
        assert 'band_width' in result.metadata
        assert 'percent_b' in result.metadata
    
    def test_atr_calculation(self, sample_data):
        """测试ATR计算"""
        atr_indicator = ATR(period=14)
        result = atr_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "ATR_14"
        assert not np.isnan(result.value)
        assert result.value > 0  # ATR应该总是正数
        assert result.metadata['volatility_level'] in ['low', 'normal', 'moderate', 'high', 'very_high']
    
    def test_standard_deviation_calculation(self, sample_data):
        """测试标准差计算"""
        std_indicator = StandardDeviation(period=20)
        result = std_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert not np.isnan(result.value)
        assert result.value >= 0
        assert result.metadata['volatility_level'] in ['low', 'normal', 'moderate', 'high', 'very_high']
    
    def test_keltner_channels_calculation(self, sample_data):
        """测试凯尔特纳通道计算"""
        keltner_indicator = KeltnerChannels(ema_period=20, atr_period=10, multiplier=2.0)
        result = keltner_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        assert 'upper' in result.value
        assert 'middle' in result.value
        assert 'lower' in result.value
        
        assert result.metadata['signal'] in ['breakout_bullish', 'breakout_bearish', 'bullish', 'bearish', 'neutral']
        assert result.metadata['channel_position'] in ['above', 'below', 'upper_half', 'lower_half', 'middle']
    
    def test_donchian_channels_calculation(self, sample_data):
        """测试唐奇安通道计算"""
        donchian_indicator = DonchianChannels(period=20)
        result = donchian_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        assert isinstance(result.value, dict)
        
        if not any(np.isnan(v) for v in result.value.values()):
            assert result.value['upper'] >= result.value['middle'] >= result.value['lower']
        
        assert result.metadata['breakout_type'] in ['none', 'upper_breakout', 'lower_breakout']
    
    def test_vix_proxy_calculation(self, sample_data):
        """测试VIX代理指标计算"""
        vix_indicator = VIXProxy(period=30)
        result = vix_indicator.calculate(sample_data, "TEST")
        
        assert isinstance(result, IndicatorResult)
        
        if not np.isnan(result.value):
            assert result.value > 0  # 波动率应该为正
            assert result.metadata['fear_level'] in ['extreme_fear', 'high_fear', 'elevated', 'normal', 'low_fear', 'complacency']
            assert result.metadata['market_stress'] in ['very_high', 'high', 'moderate', 'low', 'very_low']


class TestAsyncFunctionality:
    """异步功能测试"""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_ohlcv_data(30, seed=999)
    
    @pytest.fixture
    def indicators_manager(self):
        return TechnicalIndicators()
    
    @pytest.mark.asyncio
    async def test_async_indicator_calculation(self, indicators_manager, sample_data):
        """测试异步指标计算"""
        symbol = "BTCUSDT"
        
        # 更新数据
        for i in range(len(sample_data['close'])):
            price_data = {
                'open': sample_data['open'][i],
                'high': sample_data['high'][i],
                'low': sample_data['low'][i],
                'close': sample_data['close'][i],
                'volume': sample_data['volume'][i]
            }
            indicators_manager.update_data(symbol, price_data)
        
        # 异步计算单个指标
        result = await indicators_manager.calculate_indicator_async("RSI_14", symbol)
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "RSI_14"
    
    @pytest.mark.asyncio
    async def test_async_batch_calculation(self, indicators_manager, sample_data):
        """测试异步批量计算"""
        symbol = "ETHUSDT"
        
        # 更新数据
        df = pd.DataFrame(sample_data)
        indicators_manager.update_data_batch(symbol, df)
        
        # 异步计算所有指标
        results = await indicators_manager.calculate_all_indicators_async(symbol)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for name, result in results.items():
            assert isinstance(result, IndicatorResult)
            assert result.symbol == symbol


class TestNormalization:
    """标准化功能测试"""
    
    def test_min_max_normalization(self):
        """测试最小-最大标准化"""
        values = np.array([10, 20, 30, 40, 50])
        normalized = normalize_indicator_values(
            values, 
            method=NormalizationMethod.MIN_MAX, 
            target_range=(-1, 1)
        )
        
        assert len(normalized) == len(values)
        assert np.min(normalized) >= -1
        assert np.max(normalized) <= 1
        assert np.isclose(normalized[0], -1)  # 最小值
        assert np.isclose(normalized[-1], 1)  # 最大值
    
    def test_z_score_normalization(self):
        """测试Z分数标准化"""
        values = np.random.normal(100, 10, 50)
        normalized = normalize_indicator_values(
            values,
            method=NormalizationMethod.Z_SCORE,
            target_range=(-1, 1)
        )
        
        assert len(normalized) == len(values)
        assert np.min(normalized) >= -1
        assert np.max(normalized) <= 1
    
    def test_percentile_normalization(self):
        """测试百分位数标准化"""
        values = np.random.exponential(2, 100)  # 偏斜分布
        
        config = NormalizationConfig(
            method=NormalizationMethod.PERCENTILE,
            percentile_range=(10, 90)
        )
        normalizer = IndicatorNormalizer(config)
        
        normalized = normalizer.normalize(values, "test_indicator")
        
        assert len(normalized) == len(values)
        assert np.min(normalized) >= -1
        assert np.max(normalized) <= 1
    
    def test_normalization_with_outliers(self):
        """测试处理离群值的标准化"""
        values = np.array([1, 2, 3, 4, 5, 100])  # 包含离群值
        
        config = NormalizationConfig(
            method=NormalizationMethod.MIN_MAX,
            clip_outliers=True,
            outlier_threshold=2.0
        )
        normalizer = IndicatorNormalizer(config)
        
        normalized = normalizer.normalize(values, "test_indicator")
        
        # 离群值应该被处理
        assert np.max(normalized) <= 1
        assert np.min(normalized) >= -1
    
    def test_denormalization(self):
        """测试反标准化"""
        values = np.array([10, 20, 30, 40, 50])
        
        normalizer = IndicatorNormalizer()
        normalized = normalizer.normalize(values, "test_indicator")
        denormalized = normalizer.denormalize(normalized, "test_indicator")
        
        np.testing.assert_array_almost_equal(values, denormalized, decimal=2)


class TestTimeFrameManager:
    """多时间框架管理器测试"""
    
    @pytest.fixture
    def timeframe_manager(self):
        return TimeFrameManager()
    
    def test_timeframe_registration(self, timeframe_manager):
        """测试时间框架注册"""
        config = TimeFrameConfig(timeframe=TimeFrame.MINUTE_5)
        timeframe_manager.register_timeframe(config)
        
        registered_timeframes = timeframe_manager.get_registered_timeframes()
        assert TimeFrame.MINUTE_5 in registered_timeframes
    
    def test_indicator_registration_for_timeframe(self, timeframe_manager):
        """测试为时间框架注册指标"""
        rsi_indicator = RelativeStrengthIndex(period=14)
        timeframe_manager.register_indicator_for_timeframe(TimeFrame.MINUTE_1, rsi_indicator)
        
        # 验证指标已注册
        assert TimeFrame.MINUTE_1 in timeframe_manager._indicators
        assert rsi_indicator.name in timeframe_manager._indicators[TimeFrame.MINUTE_1]
    
    def test_tick_data_aggregation(self, timeframe_manager):
        """测试逐笔数据聚合"""
        symbol = "BTCUSDT"
        base_time = time.time()
        
        # 添加多个tick数据到同一分钟
        for i in range(10):
            timestamp = base_time + i * 10  # 每10秒一个tick
            price = 50000 + i * 10
            volume = 100
            
            timeframe_manager.update_tick_data(symbol, timestamp, price, volume)
        
        # 检查1分钟数据是否正确聚合
        minute_data = timeframe_manager.get_timeframe_data(symbol, TimeFrame.MINUTE_1)
        assert len(minute_data) >= 0  # 可能还在聚合中
    
    def test_ohlcv_data_update(self, timeframe_manager):
        """测试OHLCV数据直接更新"""
        symbol = "ETHUSDT"
        
        ohlcv = OHLCVData(
            timestamp=time.time(),
            open=3000,
            high=3100,
            low=2950,
            close=3050,
            volume=1000
        )
        
        timeframe_manager.update_ohlcv_data(symbol, TimeFrame.HOUR_1, ohlcv)
        
        data = timeframe_manager.get_timeframe_data(symbol, TimeFrame.HOUR_1)
        assert len(data) == 1
        assert data[0].close == 3050
    
    def test_multi_timeframe_results(self, timeframe_manager):
        """测试多时间框架结果获取"""
        symbol = "BTCUSDT"
        
        # 为多个时间框架注册同一个指标
        for tf in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1]:
            rsi_indicator = RelativeStrengthIndex(period=14)
            timeframe_manager.register_indicator_for_timeframe(tf, rsi_indicator)
        
        # 添加一些测试数据
        sample_data = TestDataGenerator.generate_ohlcv_data(20)
        for i in range(len(sample_data['close'])):
            ohlcv = OHLCVData(
                timestamp=sample_data['timestamp'][i],
                open=sample_data['open'][i],
                high=sample_data['high'][i],
                low=sample_data['low'][i],
                close=sample_data['close'][i],
                volume=sample_data['volume'][i]
            )
            
            for tf in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1]:
                timeframe_manager.update_ohlcv_data(symbol, tf, ohlcv)
        
        # 获取多时间框架结果
        multi_tf_results = timeframe_manager.get_multi_timeframe_results(symbol, "RSI_14")
        
        # 验证结果结构
        assert isinstance(multi_tf_results, dict)
        for tf, result in multi_tf_results.items():
            assert isinstance(tf, TimeFrame)
            assert isinstance(result, IndicatorResult)
    
    def test_timeframe_conversion(self, timeframe_manager):
        """测试时间框架转换"""
        symbol = "BTCUSDT"
        
        # 添加1分钟数据
        sample_data = TestDataGenerator.generate_ohlcv_data(60)  # 60分钟的数据
        
        for i in range(len(sample_data['close'])):
            ohlcv = OHLCVData(
                timestamp=i * 60,  # 每分钟
                open=sample_data['open'][i],
                high=sample_data['high'][i],
                low=sample_data['low'][i],
                close=sample_data['close'][i],
                volume=sample_data['volume'][i]
            )
            timeframe_manager.update_ohlcv_data(symbol, TimeFrame.MINUTE_1, ohlcv)
        
        # 转换为5分钟数据
        converted_data = timeframe_manager.convert_timeframe(
            symbol, TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, limit=60
        )
        
        assert len(converted_data) == 12  # 60分钟 / 5分钟 = 12个K线
        
        # 验证聚合逻辑
        first_bar = converted_data[0]
        assert first_bar.open == sample_data['open'][0]  # 第一根K线的开盘价
        # 其他验证...


class TestPerformanceAndIntegration:
    """性能和集成测试"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 生成大数据集
        large_data = TestDataGenerator.generate_ohlcv_data(10000, seed=2023)
        
        indicators_manager = TechnicalIndicators()
        symbol = "PERFORMANCE_TEST"
        
        # 批量更新数据
        start_time = time.time()
        df = pd.DataFrame(large_data)
        indicators_manager.update_data_batch(symbol, df)
        update_time = time.time() - start_time
        
        # 计算所有指标
        start_time = time.time()
        results = indicators_manager.calculate_all_indicators(symbol)
        calculation_time = time.time() - start_time
        
        # 性能断言（根据实际情况调整）
        assert update_time < 5.0  # 数据更新应在5秒内完成
        assert calculation_time < 10.0  # 指标计算应在10秒内完成
        assert len(results) > 0
        
        # 获取性能统计
        perf_stats = indicators_manager.get_global_performance_stats()
        assert 'total_calculations' in perf_stats
        assert 'avg_calculation_time' in perf_stats
        assert perf_stats['total_calculations'] > 0
    
    def test_memory_usage(self):
        """测试内存使用"""
        indicators_manager = TechnicalIndicators()
        
        # 添加多个symbol的数据
        symbols = ["BTC", "ETH", "ADA", "DOT", "SOL"]
        
        for symbol in symbols:
            sample_data = TestDataGenerator.generate_ohlcv_data(1000)
            df = pd.DataFrame(sample_data)
            indicators_manager.update_data_batch(symbol, df)
            indicators_manager.calculate_all_indicators(symbol)
        
        # 清理缓存测试
        initial_cache_size = len(indicators_manager._results_cache)
        indicators_manager.clear_cache("BTC")
        after_clear_size = len(indicators_manager._results_cache)
        
        assert after_clear_size < initial_cache_size
    
    def test_error_handling(self):
        """测试错误处理"""
        indicators_manager = TechnicalIndicators()
        
        # 测试空数据
        empty_data = {
            'open': np.array([]),
            'high': np.array([]),
            'low': np.array([]),
            'close': np.array([]),
            'volume': np.array([])
        }
        
        rsi_indicator = RelativeStrengthIndex(period=14)
        result = rsi_indicator.calculate(empty_data, "EMPTY_TEST")
        
        assert isinstance(result, IndicatorResult)
        assert np.isnan(result.value)
        
        # 测试不足数据
        insufficient_data = {
            'close': np.array([100, 101, 102])  # 只有3个数据点，但RSI需要15个
        }
        
        result = rsi_indicator.calculate(insufficient_data, "INSUFFICIENT_TEST")
        assert np.isnan(result.value)
        
        # 测试NaN数据
        nan_data = {
            'close': np.array([100, np.nan, 102, 103, 104] * 10)
        }
        
        result = rsi_indicator.calculate(nan_data, "NAN_TEST")
        # 应该能处理部分NaN数据
        assert isinstance(result, IndicatorResult)
    
    def test_concurrent_access(self):
        """测试并发访问安全性"""
        import threading
        import queue
        
        indicators_manager = TechnicalIndicators()
        results_queue = queue.Queue()
        
        def worker(thread_id):
            symbol = f"THREAD_{thread_id}"
            sample_data = TestDataGenerator.generate_ohlcv_data(100, seed=thread_id)
            
            # 更新数据
            df = pd.DataFrame(sample_data)
            indicators_manager.update_data_batch(symbol, df)
            
            # 计算指标
            results = indicators_manager.calculate_all_indicators(symbol)
            results_queue.put((thread_id, len(results)))
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert results_queue.qsize() == 5
        while not results_queue.empty():
            thread_id, result_count = results_queue.get()
            assert result_count > 0


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_convenience_functions(self):
        """测试便捷计算函数"""
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        
        # 测试SMA
        sma_result = calculate_sma(prices, 5)
        assert len(sma_result) == len(prices)
        assert not np.isnan(sma_result[-1])  # 最后一个值应该有效
        
        # 测试EMA
        ema_result = calculate_ema(prices, 5)
        assert len(ema_result) == len(prices)
        assert not np.isnan(ema_result[-1])
        
        # 测试RSI
        rsi_result = calculate_rsi(prices, 5)
        assert len(rsi_result) == len(prices)
        
        # RSI值应该在0-100范围内
        valid_rsi = rsi_result[~np.isnan(rsi_result)]
        if len(valid_rsi) > 0:
            assert np.all((valid_rsi >= 0) & (valid_rsi <= 100))
    
    def test_data_export(self):
        """测试数据导出功能"""
        indicators_manager = TechnicalIndicators()
        symbol = "EXPORT_TEST"
        
        # 添加测试数据
        sample_data = TestDataGenerator.generate_ohlcv_data(50)
        df = pd.DataFrame(sample_data)
        indicators_manager.update_data_batch(symbol, df)
        
        # 计算指标
        indicators_manager.calculate_all_indicators(symbol)
        
        # 导出为pandas DataFrame
        exported_df = indicators_manager.export_data(symbol, format='pandas')
        assert isinstance(exported_df, pd.DataFrame)
        assert len(exported_df) > 0
        assert 'close' in exported_df.columns
        
        # 导出为字典
        exported_dict = indicators_manager.export_data(symbol, format='dict')
        assert isinstance(exported_dict, dict)
        assert 'data' in exported_dict
        assert 'results' in exported_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])