"""
技术指标工具函数
提供便捷的指标计算函数和辅助工具
"""

import numpy as np
from typing import Union, Tuple, Dict, Any, List, Optional

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    from ..mock_talib import MockTalib
    talib = MockTalib()


# ================================
# 便捷计算函数
# ================================

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均线"""
    return talib.SMA(prices, timeperiod=period)


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均线"""
    return talib.EMA(prices, timeperiod=period)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI"""
    return talib.RSI(prices, timeperiod=period)


def calculate_macd(
    prices: np.ndarray, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算MACD"""
    return talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)


def calculate_bollinger_bands(
    prices: np.ndarray, 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带"""
    return talib.BBANDS(prices, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)


def calculate_stochastic(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """计算随机指标"""
    return talib.STOCH(
        high, low, close,
        fastk_period=k_period,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=d_period,
        slowd_matype=0
    )


def calculate_cci(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """计算商品通道指数"""
    return talib.CCI(high, low, close, timeperiod=period)


def calculate_williams_r(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """计算威廉指标"""
    return talib.WILLR(high, low, close, timeperiod=period)


def calculate_atr(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """计算平均真实波动范围"""
    return talib.ATR(high, low, close, timeperiod=period)


def calculate_adx(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """计算平均方向指数"""
    return talib.ADX(high, low, close, timeperiod=period)


def calculate_sar(
    high: np.ndarray, 
    low: np.ndarray,
    acceleration: float = 0.02,
    maximum: float = 0.2
) -> np.ndarray:
    """计算抛物线SAR"""
    return talib.SAR(high, low, acceleration=acceleration, maximum=maximum)


# ================================
# 高级计算函数
# ================================

def calculate_price_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """计算价格通道"""
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(period - 1, len(high)):
        upper_channel[i] = np.max(high[i - period + 1:i + 1])
        lower_channel[i] = np.min(low[i - period + 1:i + 1])
    
    return upper_channel, lower_channel


def calculate_pivot_points(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> Dict[str, np.ndarray]:
    """计算枢轴点"""
    # 使用前一天的HLC计算当天的枢轴点
    pivot = (high + low + close) / 3
    
    # 阻力位和支撑位
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def calculate_volume_profile(
    prices: np.ndarray,
    volumes: np.ndarray,
    num_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """计算成交量分布"""
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have the same length")
    
    # 创建价格区间
    price_min, price_max = np.min(prices), np.max(prices)
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # 计算每个价格区间的成交量
    volume_profile = np.zeros(num_bins)
    
    for i in range(len(prices)):
        bin_index = np.digitize(prices[i], price_bins) - 1
        bin_index = np.clip(bin_index, 0, num_bins - 1)
        volume_profile[bin_index] += volumes[i]
    
    # 价格区间中点
    price_levels = (price_bins[:-1] + price_bins[1:]) / 2
    
    return price_levels, volume_profile


def calculate_market_structure(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_period: int = 5
) -> Dict[str, List[int]]:
    """计算市场结构（高点低点）"""
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_period, len(high) - swing_period):
        # 检查是否为摆动高点
        is_swing_high = True
        for j in range(i - swing_period, i + swing_period + 1):
            if j != i and high[j] >= high[i]:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append(i)
        
        # 检查是否为摆动低点
        is_swing_low = True
        for j in range(i - swing_period, i + swing_period + 1):
            if j != i and low[j] <= low[i]:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append(i)
    
    return {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows
    }


def calculate_fibonacci_retracement(
    high_price: float,
    low_price: float
) -> Dict[str, float]:
    """计算斐波那契回撤位"""
    diff = high_price - low_price
    
    # 斐波那契回撤比例
    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    retracement_levels = {}
    for level in fib_levels:
        retracement_levels[f"fib_{level:.3f}"] = high_price - (diff * level)
    
    return retracement_levels


def calculate_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray
) -> np.ndarray:
    """计算成交量加权平均价"""
    typical_price = (high + low + close) / 3
    vwap = np.full_like(close, np.nan)
    
    cumulative_volume = 0
    cumulative_pv = 0
    
    for i in range(len(close)):
        if not np.isnan(volume[i]) and volume[i] > 0:
            cumulative_pv += typical_price[i] * volume[i]
            cumulative_volume += volume[i]
            
            if cumulative_volume > 0:
                vwap[i] = cumulative_pv / cumulative_volume
    
    return vwap


def calculate_twap(
    prices: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """计算时间加权平均价"""
    twap = np.full_like(prices, np.nan)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        twap[i] = np.mean(window)
    
    return twap


# ================================
# 信号识别函数
# ================================

def identify_crossovers(
    fast_line: np.ndarray,
    slow_line: np.ndarray
) -> Tuple[List[int], List[int]]:
    """识别交叉信号"""
    bullish_crossovers = []
    bearish_crossovers = []
    
    for i in range(1, len(fast_line)):
        if (np.isnan(fast_line[i-1]) or np.isnan(slow_line[i-1]) or 
            np.isnan(fast_line[i]) or np.isnan(slow_line[i])):
            continue
        
        # 金叉
        if fast_line[i-1] <= slow_line[i-1] and fast_line[i] > slow_line[i]:
            bullish_crossovers.append(i)
        
        # 死叉
        elif fast_line[i-1] >= slow_line[i-1] and fast_line[i] < slow_line[i]:
            bearish_crossovers.append(i)
    
    return bullish_crossovers, bearish_crossovers


def identify_divergences(
    prices: np.ndarray,
    indicator: np.ndarray,
    swing_period: int = 5
) -> Dict[str, List[Tuple[int, int]]]:
    """识别背离信号"""
    # 找到摆动点
    market_structure = calculate_market_structure(
        prices, prices, prices, swing_period
    )
    
    swing_highs = market_structure['swing_highs']
    swing_lows = market_structure['swing_lows']
    
    bullish_divergences = []  # 底背离
    bearish_divergences = []  # 顶背离
    
    # 检查顶背离
    for i in range(1, len(swing_highs)):
        idx1, idx2 = swing_highs[i-1], swing_highs[i]
        
        if (prices[idx2] > prices[idx1] and 
            indicator[idx2] < indicator[idx1]):
            bearish_divergences.append((idx1, idx2))
    
    # 检查底背离
    for i in range(1, len(swing_lows)):
        idx1, idx2 = swing_lows[i-1], swing_lows[i]
        
        if (prices[idx2] < prices[idx1] and 
            indicator[idx2] > indicator[idx1]):
            bullish_divergences.append((idx1, idx2))
    
    return {
        'bullish_divergences': bullish_divergences,
        'bearish_divergences': bearish_divergences
    }


def identify_support_resistance(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_touches: int = 3,
    tolerance: float = 0.01
) -> Dict[str, List[float]]:
    """识别支撑阻力位"""
    # 找到关键价格水平
    key_levels = []
    
    # 添加摆动高低点
    structure = calculate_market_structure(high, low, close)
    for idx in structure['swing_highs']:
        key_levels.append(high[idx])
    for idx in structure['swing_lows']:
        key_levels.append(low[idx])
    
    # 聚类相近的价格水平
    support_levels = []
    resistance_levels = []
    
    for level in set(key_levels):
        touches = 0
        
        # 计算触及次数
        for i in range(len(close)):
            if (abs(low[i] - level) / level <= tolerance or 
                abs(high[i] - level) / level <= tolerance):
                touches += 1
        
        if touches >= min_touches:
            # 判断是支撑还是阻力
            above_count = sum(1 for price in close if price > level)
            below_count = sum(1 for price in close if price < level)
            
            if above_count > below_count:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
    
    return {
        'support_levels': sorted(support_levels),
        'resistance_levels': sorted(resistance_levels, reverse=True)
    }


# ================================
# 数据处理工具
# ================================

def smooth_data(
    data: np.ndarray,
    method: str = 'ema',
    period: int = 3
) -> np.ndarray:
    """数据平滑处理"""
    if method == 'sma':
        return talib.SMA(data, timeperiod=period)
    elif method == 'ema':
        return talib.EMA(data, timeperiod=period)
    elif method == 'gaussian':
        # 高斯滤波器
        from scipy import ndimage
        sigma = period / 3.0  # 标准差
        return ndimage.gaussian_filter1d(data, sigma)
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")


def remove_outliers(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0
) -> np.ndarray:
    """移除离群值"""
    if method == 'zscore':
        z_scores = np.abs((data - np.nanmean(data)) / np.nanstd(data))
        return np.where(z_scores <= threshold, data, np.nan)
    
    elif method == 'iqr':
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)
    
    else:
        raise ValueError(f"Unsupported outlier removal method: {method}")


def fill_missing_data(
    data: np.ndarray,
    method: str = 'forward'
) -> np.ndarray:
    """填充缺失数据"""
    if method == 'forward':
        # 前向填充
        result = data.copy()
        for i in range(1, len(result)):
            if np.isnan(result[i]) and not np.isnan(result[i-1]):
                result[i] = result[i-1]
        return result
    
    elif method == 'backward':
        # 后向填充
        result = data.copy()
        for i in range(len(result) - 2, -1, -1):
            if np.isnan(result[i]) and not np.isnan(result[i+1]):
                result[i] = result[i+1]
        return result
    
    elif method == 'linear':
        # 线性插值
        import pandas as pd
        return pd.Series(data).interpolate(method='linear').values
    
    else:
        raise ValueError(f"Unsupported fill method: {method}")


def calculate_correlation(
    x: np.ndarray,
    y: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """计算滚动相关系数"""
    correlation = np.full_like(x, np.nan)
    
    for i in range(period - 1, len(x)):
        window_x = x[i - period + 1:i + 1]
        window_y = y[i - period + 1:i + 1]
        
        # 过滤NaN值
        valid_indices = ~(np.isnan(window_x) | np.isnan(window_y))
        valid_x = window_x[valid_indices]
        valid_y = window_y[valid_indices]
        
        if len(valid_x) > 1:
            correlation[i] = np.corrcoef(valid_x, valid_y)[0, 1]
    
    return correlation


# 自定义指标示例
def custom_momentum_indicator(data: Dict[str, np.ndarray], parameters: Dict[str, Any]) -> float:
    """自定义动量指标示例"""
    close_prices = data.get("close", np.array([]))
    volume = data.get("volume", np.array([]))
    
    if len(close_prices) < 2 or len(volume) < 2:
        return np.nan
    
    # 计算价格动量和成交量动量的综合指标
    price_momentum = (close_prices[-1] / close_prices[-2] - 1) * 100
    volume_ratio = volume[-1] / np.mean(volume[-10:]) if len(volume) >= 10 else 1
    
    return price_momentum * volume_ratio