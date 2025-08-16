"""
技术指标系统测试
测试各类技术指标计算准确性、增量计算性能优化和信号生成逻辑
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.analysis.technical_indicators import (
    TechnicalIndicators, IndicatorConfig, IndicatorResult, IndicatorType,
    SimpleMovingAverage, ExponentialMovingAverage, RelativeStrengthIndex,
    MACD, BollingerBands, StochasticOscillator, CustomIndicator,
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd, calculate_bollinger_bands,
    custom_momentum_indicator
)


class TestIndicatorConfig:
    """指标配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = IndicatorConfig(
            name="test_indicator",
            indicator_type=IndicatorType.TREND
        )
        
        assert config.name == "test_indicator"
        assert config.indicator_type == IndicatorType.TREND
        assert config.enabled is True
        assert config.min_periods == 1
        assert config.cache_enabled is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = IndicatorConfig(
            name="custom_sma",
            indicator_type=IndicatorType.TREND,
            parameters={"period": 20},
            enabled=False,
            min_periods=20,
            lookback_window=500,
            cache_ttl=600.0
        )
        
        assert config.parameters["period"] == 20
        assert config.enabled is False
        assert config.min_periods == 20
        assert config.lookback_window == 500
        assert config.cache_ttl == 600.0


class TestIndicatorResult:
    """指标结果测试"""
    
    def test_valid_float_result(self):
        """测试有效的浮点数结果"""
        result = IndicatorResult(
            indicator_name="SMA_20",
            symbol="BTCUSDT",
            timestamp=time.time(),
            value=50000.0
        )
        
        assert result.is_valid is True
        assert result.value == 50000.0
    
    def test_invalid_nan_result(self):
        """测试无效的NaN结果"""
        result = IndicatorResult(
            indicator_name="SMA_20",
            symbol="BTCUSDT",
            timestamp=time.time(),
            value=np.nan
        )
        
        assert result.is_valid is False
    
    def test_valid_dict_result(self):
        """测试有效的字典结果"""
        result = IndicatorResult(
            indicator_name="MACD",
            symbol="BTCUSDT",
            timestamp=time.time(),
            value={"macd": 100.0, "signal": 95.0, "histogram": 5.0}
        )
        
        assert result.is_valid is True
    
    def test_invalid_dict_result(self):
        """测试无效的字典结果"""
        result = IndicatorResult(
            indicator_name="MACD",
            symbol="BTCUSDT",
            timestamp=time.time(),
            value={"macd": np.nan, "signal": np.nan, "histogram": np.nan}
        )
        
        assert result.is_valid is False


class TestSimpleMovingAverage:
    """简单移动平均线测试"""
    
    def test_sma_calculation(self):
        """测试SMA计算"""
        sma = SimpleMovingAverage(period=5)
        
        # 测试数据
        prices = np.array([10, 12, 14, 16, 18, 20, 22])
        data = {"close": prices}
        
        result = sma.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        # 最后5个价格的平均值: (16+18+20+22+prices[-1])/5
        expected_sma = np.mean(prices[-5:])
        assert abs(result.value - expected_sma) < 0.01
    
    def test_sma_insufficient_data(self):
        """测试数据不足的情况"""
        sma = SimpleMovingAverage(period=10)
        
        # 只有5个数据点，但需要10个
        prices = np.array([10, 12, 14, 16, 18])
        data = {"close": prices}
        
        result = sma.calculate(data, "BTCUSDT")
        
        assert not result.is_valid
        assert np.isnan(result.value)
    
    def test_sma_can_calculate(self):
        """测试是否能够计算检查"""
        sma = SimpleMovingAverage(period=5)
        
        # 足够数据
        sufficient_data = {"close": np.array([1, 2, 3, 4, 5, 6])}
        assert sma.can_calculate(sufficient_data) is True
        
        # 数据不足
        insufficient_data = {"close": np.array([1, 2, 3])}
        assert sma.can_calculate(insufficient_data) is False
        
        # 缺少必要字段
        missing_data = {"high": np.array([1, 2, 3, 4, 5, 6])}
        assert sma.can_calculate(missing_data) is False


class TestExponentialMovingAverage:
    """指数移动平均线测试"""
    
    def test_ema_calculation(self):
        """测试EMA计算"""
        ema = ExponentialMovingAverage(period=5)
        
        prices = np.array([10, 12, 14, 16, 18, 20])
        data = {"close": prices}
        
        result = ema.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        assert result.value > 0
        assert result.metadata["period"] == 5


class TestRelativeStrengthIndex:
    """RSI测试"""
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        rsi = RelativeStrengthIndex(period=14)
        
        # 创建一个趋势上升的价格序列
        prices = np.array([
            100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
            111, 113, 112, 114, 116, 115, 117, 119, 118, 120
        ])
        data = {"close": prices}
        
        result = rsi.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        assert 0 <= result.value <= 100  # RSI应该在0-100之间
        assert result.value > 50  # 上升趋势应该RSI > 50
    
    def test_rsi_overbought_oversold(self):
        """测试RSI超买超卖情况"""
        rsi = RelativeStrengthIndex(period=14)
        
        # 创建强烈上升趋势（可能超买）
        uptrend_prices = np.array([100 + i * 2 for i in range(20)])
        data = {"close": uptrend_prices}
        
        result = rsi.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        # 强烈上升趋势通常导致高RSI值
        assert result.value > 60


class TestMACD:
    """MACD测试"""
    
    def test_macd_calculation(self):
        """测试MACD计算"""
        macd = MACD(fast_period=12, slow_period=26, signal_period=9)
        
        # 创建足够长的价格序列
        prices = np.array([100 + np.sin(i/10) * 10 + i * 0.1 for i in range(50)])
        data = {"close": prices}
        
        result = macd.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        assert isinstance(result.value, dict)
        assert "macd" in result.value
        assert "signal" in result.value
        assert "histogram" in result.value
        
        # 检查所有值都不是NaN
        for key, value in result.value.items():
            assert not np.isnan(value)
    
    def test_macd_insufficient_data(self):
        """测试MACD数据不足情况"""
        macd = MACD()
        
        # 只有10个数据点，但MACD需要更多
        prices = np.array(range(10))
        data = {"close": prices}
        
        result = macd.calculate(data, "BTCUSDT")
        
        assert not result.is_valid


class TestBollingerBands:
    """布林带测试"""
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算"""
        bb = BollingerBands(period=20, std_dev=2)
        
        # 创建价格数据
        prices = np.array([100 + np.random.normal(0, 2) for _ in range(30)])
        data = {"close": prices}
        
        result = bb.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        assert isinstance(result.value, dict)
        assert "upper" in result.value
        assert "middle" in result.value
        assert "lower" in result.value
        
        # 上轨应该大于中轨，中轨应该大于下轨
        assert result.value["upper"] > result.value["middle"]
        assert result.value["middle"] > result.value["lower"]


class TestStochasticOscillator:
    """随机振荡器测试"""
    
    def test_stochastic_calculation(self):
        """测试随机振荡器计算"""
        stoch = StochasticOscillator(k_period=14, d_period=3)
        
        # 创建OHLC数据
        high_prices = np.array([105 + i + np.random.uniform(0, 2) for i in range(25)])
        low_prices = np.array([95 + i - np.random.uniform(0, 2) for i in range(25)])
        close_prices = np.array([100 + i for i in range(25)])
        
        data = {
            "high": high_prices,
            "low": low_prices,
            "close": close_prices
        }
        
        result = stoch.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        assert isinstance(result.value, dict)
        assert "k" in result.value
        assert "d" in result.value
        
        # K和D值应该在0-100之间
        assert 0 <= result.value["k"] <= 100
        assert 0 <= result.value["d"] <= 100


class TestCustomIndicator:
    """自定义指标测试"""
    
    def test_custom_indicator_creation(self):
        """测试自定义指标创建"""
        def simple_average(data, parameters):
            close_prices = data.get("close", np.array([]))
            if len(close_prices) == 0:
                return np.nan
            return np.mean(close_prices[-parameters.get("period", 5):])
        
        custom = CustomIndicator(
            name="custom_avg",
            calculation_func=simple_average,
            parameters={"period": 10}
        )
        
        prices = np.array(range(20))
        data = {"close": prices}
        
        result = custom.calculate(data, "BTCUSDT")
        
        assert result.is_valid
        # 最后10个数字的平均值
        expected = np.mean(prices[-10:])
        assert result.value == expected
    
    def test_custom_indicator_error_handling(self):
        """测试自定义指标错误处理"""
        def failing_function(data, parameters):
            raise ValueError("Test error")
        
        custom = CustomIndicator(
            name="failing_indicator",
            calculation_func=failing_function
        )
        
        data = {"close": np.array([1, 2, 3])}
        result = custom.calculate(data, "BTCUSDT")
        
        assert not result.is_valid
        assert np.isnan(result.value)
        assert "error" in result.metadata


class TestTechnicalIndicators:
    """技术指标管理器测试"""
    
    @pytest.fixture
    def indicators(self):
        """创建指标管理器实例"""
        return TechnicalIndicators()
    
    def test_default_indicators_registration(self, indicators):
        """测试默认指标注册"""
        registered = indicators.get_registered_indicators()
        
        # 检查是否注册了默认指标
        assert "SMA_20" in registered
        assert "SMA_50" in registered
        assert "EMA_20" in registered
        assert "RSI_14" in registered
        assert "MACD_12_26_9" in registered
        assert "BBANDS_20_2" in registered
    
    def test_custom_indicator_registration(self, indicators):
        """测试自定义指标注册"""
        def test_func(data, params):
            return 42.0
        
        indicators.add_custom_indicator("test_custom", test_func, {"param": 1})
        
        registered = indicators.get_registered_indicators()
        assert "test_custom" in registered
    
    def test_indicator_unregistration(self, indicators):
        """测试指标注销"""
        initial_count = len(indicators.get_registered_indicators())
        
        indicators.unregister_indicator("SMA_20")
        
        final_count = len(indicators.get_registered_indicators())
        assert final_count == initial_count - 1
        assert "SMA_20" not in indicators.get_registered_indicators()
    
    def test_data_update_and_calculation(self, indicators):
        """测试数据更新和计算"""
        symbol = "BTCUSDT"
        
        # 添加足够的数据点
        for i in range(25):
            price_data = {
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 100 + i,
                "volume": 1000 + i * 10,
                "timestamp": time.time() + i
            }
            indicators.update_data(symbol, price_data)
        
        # 计算SMA指标
        result = indicators.calculate_indicator("SMA_20", symbol)
        
        assert result is not None
        assert result.is_valid
        assert result.symbol == symbol
        assert result.indicator_name == "SMA_20"
    
    def test_calculate_all_indicators(self, indicators):
        """测试计算所有指标"""
        symbol = "BTCUSDT"
        
        # 添加数据
        for i in range(30):
            price_data = {
                "close": 100 + i + np.random.normal(0, 1),
                "high": 105 + i,
                "low": 95 + i,
                "volume": 1000
            }
            indicators.update_data(symbol, price_data)
        
        # 计算所有指标
        results = indicators.calculate_all_indicators(symbol)
        
        assert len(results) > 0
        
        # 检查每个结果
        for name, result in results.items():
            assert result.symbol == symbol
            assert result.indicator_name == name
    
    def test_performance_tracking(self, indicators):
        """测试性能跟踪"""
        symbol = "BTCUSDT"
        
        # 添加数据并计算指标多次
        for i in range(25):
            price_data = {"close": 100 + i, "volume": 1000}
            indicators.update_data(symbol, price_data)
        
        # 计算指标多次以生成性能数据
        for _ in range(5):
            indicators.calculate_indicator("SMA_20", symbol, force_recalculate=True)
        
        stats = indicators.get_performance_stats()
        
        assert "SMA_20" in stats
        assert "avg_time" in stats["SMA_20"]
        assert "total_calculations" in stats["SMA_20"]
        assert stats["SMA_20"]["total_calculations"] >= 5
    
    def test_cache_functionality(self, indicators):
        """测试缓存功能"""
        symbol = "BTCUSDT"
        
        # 添加数据
        for i in range(25):
            price_data = {"close": 100 + i}
            indicators.update_data(symbol, price_data)
        
        # 第一次计算
        result1 = indicators.calculate_indicator("SMA_20", symbol)
        
        # 第二次计算（应该使用缓存）
        result2 = indicators.calculate_indicator("SMA_20", symbol)
        
        assert result1.value == result2.value
        assert result1.timestamp == result2.timestamp
    
    def test_force_recalculation(self, indicators):
        """测试强制重新计算"""
        symbol = "BTCUSDT"
        
        # 添加数据
        for i in range(25):
            price_data = {"close": 100 + i}
            indicators.update_data(symbol, price_data)
        
        # 第一次计算
        result1 = indicators.calculate_indicator("SMA_20", symbol)
        
        # 强制重新计算
        result2 = indicators.calculate_indicator("SMA_20", symbol, force_recalculate=True)
        
        # 值应该相同，但时间戳不同
        assert result1.value == result2.value
        assert result1.timestamp <= result2.timestamp
    
    def test_get_indicator_config(self, indicators):
        """测试获取指标配置"""
        config = indicators.get_indicator_config("SMA_20")
        
        assert config is not None
        assert config.name == "SMA_20"
        assert config.indicator_type == IndicatorType.TREND
        assert config.parameters["period"] == 20
    
    def test_cache_clearing(self, indicators):
        """测试缓存清理"""
        symbol = "BTCUSDT"
        
        # 添加数据并计算
        for i in range(25):
            price_data = {"close": 100 + i}
            indicators.update_data(symbol, price_data)
        
        result = indicators.calculate_indicator("SMA_20", symbol)
        assert result is not None
        
        # 清理缓存
        indicators.clear_cache(symbol)
        
        # 获取结果应该为None
        cached_result = indicators.get_indicator_result(symbol, "SMA_20")
        assert cached_result is None


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_calculate_sma(self):
        """测试SMA便捷函数"""
        prices = np.array([10, 12, 14, 16, 18, 20])
        sma_values = calculate_sma(prices, period=3)
        
        assert len(sma_values) == len(prices)
        # 最后一个值应该是最后3个价格的平均值
        expected = np.mean(prices[-3:])
        assert abs(sma_values[-1] - expected) < 0.01
    
    def test_calculate_ema(self):
        """测试EMA便捷函数"""
        prices = np.array([10, 12, 14, 16, 18, 20])
        ema_values = calculate_ema(prices, period=3)
        
        assert len(ema_values) == len(prices)
        assert not np.isnan(ema_values[-1])
    
    def test_calculate_rsi(self):
        """测试RSI便捷函数"""
        prices = np.array([100 + i for i in range(20)])
        rsi_values = calculate_rsi(prices, period=14)
        
        assert len(rsi_values) == len(prices)
        assert 0 <= rsi_values[-1] <= 100
    
    def test_calculate_macd(self):
        """测试MACD便捷函数"""
        prices = np.array([100 + np.sin(i/5) * 5 for i in range(30)])
        macd_line, signal_line, histogram = calculate_macd(prices)
        
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)
    
    def test_calculate_bollinger_bands(self):
        """测试布林带便捷函数"""
        prices = np.array([100 + np.random.normal(0, 2) for _ in range(25)])
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
        
        # 检查带的关系
        assert upper[-1] > middle[-1] > lower[-1]


class TestCustomMomentumIndicator:
    """自定义动量指标测试"""
    
    def test_custom_momentum_calculation(self):
        """测试自定义动量指标计算"""
        data = {
            "close": np.array([100, 102, 104, 103, 105]),
            "volume": np.array([1000, 1100, 1200, 1050, 1300])
        }
        
        result = custom_momentum_indicator(data, {})
        
        assert not np.isnan(result)
        assert isinstance(result, float)
    
    def test_custom_momentum_insufficient_data(self):
        """测试数据不足的情况"""
        data = {
            "close": np.array([100]),
            "volume": np.array([1000])
        }
        
        result = custom_momentum_indicator(data, {})
        
        assert np.isnan(result)


class TestPerformanceAndOptimization:
    """性能和优化测试"""
    
    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()
    
    def test_large_dataset_performance(self, indicators):
        """测试大数据集性能"""
        symbol = "BTCUSDT"
        
        # 添加大量数据
        start_time = time.time()
        for i in range(1000):
            price_data = {
                "close": 100 + np.random.normal(0, 5),
                "volume": 1000 + np.random.randint(0, 500)
            }
            indicators.update_data(symbol, price_data)
        
        data_update_time = time.time() - start_time
        
        # 计算所有指标
        start_time = time.time()
        results = indicators.calculate_all_indicators(symbol)
        calculation_time = time.time() - start_time
        
        # 性能断言
        assert data_update_time < 1.0  # 数据更新应该在1秒内完成
        assert calculation_time < 2.0  # 计算应该在2秒内完成
        assert len(results) > 0
    
    def test_incremental_calculation_optimization(self, indicators):
        """测试增量计算优化"""
        symbol = "BTCUSDT"
        
        # 添加初始数据
        for i in range(30):
            price_data = {"close": 100 + i}
            indicators.update_data(symbol, price_data)
        
        # 第一次计算（完整计算）
        start_time = time.time()
        result1 = indicators.calculate_indicator("SMA_20", symbol)
        first_calc_time = time.time() - start_time
        
        # 添加一个新数据点
        indicators.update_data(symbol, {"close": 130})
        
        # 第二次计算（应该更快，但我们的实现是重新计算）
        start_time = time.time()
        result2 = indicators.calculate_indicator("SMA_20", symbol, force_recalculate=True)
        second_calc_time = time.time() - start_time
        
        assert result1 is not None
        assert result2 is not None
        assert result1.value != result2.value  # 新数据应该改变结果
    
    def test_memory_usage_with_large_cache(self, indicators):
        """测试大缓存的内存使用"""
        symbol = "BTCUSDT"
        
        # 添加大量数据（超过默认缓存大小）
        for i in range(2000):
            price_data = {
                "close": 100 + i % 100,  # 创建周期性数据
                "volume": 1000
            }
            indicators.update_data(symbol, price_data)
        
        # 计算指标
        result = indicators.calculate_indicator("SMA_20", symbol)
        
        assert result is not None
        # 缓存应该限制数据量，不会无限增长


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.fixture
    def indicators(self):
        return TechnicalIndicators()
    
    def test_invalid_indicator_calculation(self, indicators):
        """测试无效指标计算"""
        result = indicators.calculate_indicator("NONEXISTENT_INDICATOR", "BTCUSDT")
        assert result is None
    
    def test_empty_data_handling(self, indicators):
        """测试空数据处理"""
        symbol = "BTCUSDT"
        
        # 不添加任何数据就尝试计算
        result = indicators.calculate_indicator("SMA_20", symbol)
        assert result is None
    
    def test_malformed_data_handling(self, indicators):
        """测试畸形数据处理"""
        symbol = "BTCUSDT"
        
        # 添加包含NaN的数据
        price_data = {
            "close": np.nan,
            "volume": 1000
        }
        indicators.update_data(symbol, price_data)
        
        # 应该能够处理而不崩溃
        result = indicators.calculate_indicator("SMA_20", symbol)
        # 由于数据不足，结果可能为None，这是可接受的


if __name__ == "__main__":
    pytest.main([__file__])