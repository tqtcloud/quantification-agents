"""
TA-Lib 模拟实现
当TA-Lib不可用时提供基本的技术指标计算功能
"""

import numpy as np
from typing import Tuple


class MockTalib:
    """TA-Lib 模拟实现类"""
    
    @staticmethod
    def SMA(prices: np.ndarray, timeperiod: int) -> np.ndarray:
        """简单移动平均线"""
        if len(prices) < timeperiod:
            return np.full_like(prices, np.nan)
        result = np.full_like(prices, np.nan, dtype=float)
        for i in range(timeperiod - 1, len(prices)):
            result[i] = np.mean(prices[i - timeperiod + 1:i + 1])
        return result
    
    @staticmethod
    def EMA(prices: np.ndarray, timeperiod: int) -> np.ndarray:
        """指数移动平均线"""
        if len(prices) < timeperiod:
            return np.full_like(prices, np.nan)
        result = np.full_like(prices, np.nan, dtype=float)
        alpha = 2.0 / (timeperiod + 1.0)
        result[timeperiod - 1] = np.mean(prices[:timeperiod])
        for i in range(timeperiod, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
        return result
    
    @staticmethod
    def RSI(prices: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """相对强弱指数"""
        if len(prices) < timeperiod + 1:
            return np.full_like(prices, np.nan)
        
        deltas = np.diff(prices)
        seed = deltas[:timeperiod]
        up = seed[seed >= 0].sum() / timeperiod
        down = -seed[seed < 0].sum() / timeperiod
        if down == 0:
            down = 1e-10
        rs = up / down
        rsi = np.full_like(prices, np.nan, dtype=float)
        rsi[timeperiod] = 100.0 - 100.0 / (1.0 + rs)
        
        for i in range(timeperiod + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (timeperiod - 1) + upval) / timeperiod
            down = (down * (timeperiod - 1) + downval) / timeperiod
            if down == 0:
                down = 1e-10
            rs = up / down
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    @staticmethod
    def MACD(prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, 
             signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        if len(prices) < slowperiod + signalperiod:
            nan_array = np.full_like(prices, np.nan)
            return nan_array, nan_array, nan_array
        
        ema_fast = MockTalib.EMA(prices, fastperiod)
        ema_slow = MockTalib.EMA(prices, slowperiod)
        macd_line = ema_fast - ema_slow
        
        full_signal = np.full_like(prices, np.nan)
        valid_macd_start = slowperiod - 1
        valid_macd_data = macd_line[valid_macd_start:]
        
        if len(valid_macd_data) >= signalperiod:
            signal_values = MockTalib.EMA(valid_macd_data, signalperiod)
            signal_start = valid_macd_start
            valid_signal_count = len(signal_values) - sum(np.isnan(signal_values))
            if valid_signal_count > 0:
                first_valid_signal = signalperiod - 1
                actual_start = signal_start + first_valid_signal
                actual_end = signal_start + len(signal_values)
                if actual_end <= len(full_signal):
                    full_signal[signal_start:actual_end] = signal_values
        
        histogram = macd_line - full_signal
        return macd_line, full_signal, histogram
    
    @staticmethod
    def BBANDS(prices: np.ndarray, timeperiod: int = 20, nbdevup: float = 2, 
               nbdevdn: float = 2, matype: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        if len(prices) < timeperiod:
            nan_array = np.full_like(prices, np.nan)
            return nan_array, nan_array, nan_array
        
        sma = MockTalib.SMA(prices, timeperiod)
        std = np.full_like(prices, np.nan, dtype=float)
        
        for i in range(timeperiod - 1, len(prices)):
            window = prices[i - timeperiod + 1:i + 1]
            std[i] = np.std(window, ddof=1)
        
        upper_band = sma + (std * nbdevup)
        lower_band = sma - (std * nbdevdn)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def STOCH(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
              fastk_period: int = 14, slowk_period: int = 3, slowk_matype: int = 0,
              slowd_period: int = 3, slowd_matype: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """随机指标"""
        if len(close) < fastk_period + slowk_period + slowd_period:
            nan_array = np.full_like(close, np.nan)
            return nan_array, nan_array
        
        k_values = np.full_like(close, np.nan, dtype=float)
        for i in range(fastk_period - 1, len(close)):
            window_high = np.max(high[i - fastk_period + 1:i + 1])
            window_low = np.min(low[i - fastk_period + 1:i + 1])
            if window_high != window_low:
                k_values[i] = 100 * (close[i] - window_low) / (window_high - window_low)
            else:
                k_values[i] = 50.0
        
        slowk = MockTalib.SMA(k_values, slowk_period)
        slowd = MockTalib.SMA(slowk, slowd_period)
        
        return slowk, slowd
    
    @staticmethod
    def CCI(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            timeperiod: int = 14) -> np.ndarray:
        """商品通道指数"""
        if len(close) < timeperiod:
            return np.full_like(close, np.nan)
        
        typical_price = (high + low + close) / 3.0
        result = np.full_like(close, np.nan, dtype=float)
        
        for i in range(timeperiod - 1, len(close)):
            window = typical_price[i - timeperiod + 1:i + 1]
            sma = np.mean(window)
            mad = np.mean(np.abs(window - sma))  # Mean Absolute Deviation
            if mad != 0:
                result[i] = (typical_price[i] - sma) / (0.015 * mad)
            else:
                result[i] = 0.0
        
        return result
    
    @staticmethod
    def WILLR(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
              timeperiod: int = 14) -> np.ndarray:
        """威廉指标"""
        if len(close) < timeperiod:
            return np.full_like(close, np.nan)
        
        result = np.full_like(close, np.nan, dtype=float)
        for i in range(timeperiod - 1, len(close)):
            highest_high = np.max(high[i - timeperiod + 1:i + 1])
            lowest_low = np.min(low[i - timeperiod + 1:i + 1])
            if highest_high != lowest_low:
                result[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                result[i] = -50.0
        
        return result
    
    @staticmethod
    def ADX(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            timeperiod: int = 14) -> np.ndarray:
        """平均方向指数"""
        if len(close) < timeperiod * 2:
            return np.full_like(close, np.nan)
        
        # 计算真实范围和方向移动
        tr = MockTalib._true_range(high, low, close)
        plus_dm = MockTalib._plus_dm(high, low)
        minus_dm = MockTalib._minus_dm(high, low)
        
        # 平滑化
        tr_smooth = MockTalib._wilder_smooth(tr, timeperiod)
        plus_dm_smooth = MockTalib._wilder_smooth(plus_dm, timeperiod)
        minus_dm_smooth = MockTalib._wilder_smooth(minus_dm, timeperiod)
        
        # 计算DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # 计算DX
        dx = np.full_like(close, np.nan, dtype=float)
        for i in range(len(dx)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # 计算ADX
        adx = MockTalib._wilder_smooth(dx, timeperiod)
        
        return adx
    
    @staticmethod
    def SAR(high: np.ndarray, low: np.ndarray, acceleration: float = 0.02, 
            maximum: float = 0.2) -> np.ndarray:
        """抛物线SAR"""
        if len(high) < 2:
            return np.full_like(high, np.nan)
        
        sar = np.full_like(high, np.nan, dtype=float)
        trend = np.ones(len(high))  # 1 for uptrend, -1 for downtrend
        af = acceleration
        ep = high[0]  # extreme point
        
        # 初始化
        sar[0] = low[0]
        trend[0] = 1
        
        for i in range(1, len(high)):
            if trend[i-1] == 1:  # uptrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if low[i] <= sar[i]:
                    # Trend reversal
                    trend[i] = -1
                    sar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    trend[i] = 1
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR should not be above previous two lows
                    sar[i] = min(sar[i], low[i-1])
                    if i > 1:
                        sar[i] = min(sar[i], low[i-2])
            
            else:  # downtrend
                sar[i] = sar[i-1] + af * (ep - sar[i-1])
                
                if high[i] >= sar[i]:
                    # Trend reversal
                    trend[i] = 1
                    sar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    trend[i] = -1
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
                    
                    # SAR should not be below previous two highs
                    sar[i] = max(sar[i], high[i-1])
                    if i > 1:
                        sar[i] = max(sar[i], high[i-2])
        
        return sar
    
    @staticmethod
    def ATR(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
            timeperiod: int = 14) -> np.ndarray:
        """平均真实范围"""
        tr = MockTalib._true_range(high, low, close)
        return MockTalib._wilder_smooth(tr, timeperiod)
    
    @staticmethod
    def PLUS_DI(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                timeperiod: int = 14) -> np.ndarray:
        """正方向指数"""
        tr = MockTalib._true_range(high, low, close)
        plus_dm = MockTalib._plus_dm(high, low)
        
        tr_smooth = MockTalib._wilder_smooth(tr, timeperiod)
        plus_dm_smooth = MockTalib._wilder_smooth(plus_dm, timeperiod)
        
        plus_di = np.full_like(high, np.nan, dtype=float)
        for i in range(len(plus_di)):
            if tr_smooth[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
        
        return plus_di
    
    @staticmethod
    def MINUS_DI(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 timeperiod: int = 14) -> np.ndarray:
        """负方向指数"""
        tr = MockTalib._true_range(high, low, close)
        minus_dm = MockTalib._minus_dm(high, low)
        
        tr_smooth = MockTalib._wilder_smooth(tr, timeperiod)
        minus_dm_smooth = MockTalib._wilder_smooth(minus_dm, timeperiod)
        
        minus_di = np.full_like(high, np.nan, dtype=float)
        for i in range(len(minus_di)):
            if tr_smooth[i] != 0:
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
        
        return minus_di
    
    @staticmethod
    def STDDEV(prices: np.ndarray, timeperiod: int = 5, nbdev: float = 1.0) -> np.ndarray:
        """标准差"""
        if len(prices) < timeperiod:
            return np.full_like(prices, np.nan)
        
        result = np.full_like(prices, np.nan, dtype=float)
        for i in range(timeperiod - 1, len(prices)):
            window = prices[i - timeperiod + 1:i + 1]
            result[i] = np.std(window, ddof=1) * nbdev
        
        return result
    
    # 辅助函数
    @staticmethod
    def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """真实范围"""
        tr = np.full_like(high, np.nan, dtype=float)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        return tr
    
    @staticmethod
    def _plus_dm(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """正方向移动"""
        plus_dm = np.full_like(high, 0.0, dtype=float)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
        
        return plus_dm
    
    @staticmethod
    def _minus_dm(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """负方向移动"""
        minus_dm = np.full_like(high, 0.0, dtype=float)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        return minus_dm
    
    @staticmethod
    def _wilder_smooth(values: np.ndarray, timeperiod: int) -> np.ndarray:
        """威尔德平滑"""
        if len(values) < timeperiod:
            return np.full_like(values, np.nan)
        
        result = np.full_like(values, np.nan, dtype=float)
        result[timeperiod - 1] = np.mean(values[:timeperiod])
        
        for i in range(timeperiod, len(values)):
            result[i] = (result[i-1] * (timeperiod - 1) + values[i]) / timeperiod
        
        return result