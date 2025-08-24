"""
波动率/风险指标实现
包括布林带、ATR、标准差、凯尔特纳通道、唐奇安通道等
"""

import numpy as np
import time
from typing import Dict, List

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    from ..mock_talib import MockTalib
    talib = MockTalib()

from .base import BaseIndicator, AsyncTechnicalIndicator, IndicatorConfig, IndicatorResult, IndicatorType


class BollingerBands(AsyncTechnicalIndicator):
    """布林带指标"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        config = IndicatorConfig(
            name=f"BBANDS_{period}_{std_dev}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": period, "std_dev": std_dev},
            min_periods=period
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算布林带"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan}
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            std_dev = self.parameters["std_dev"]
            close_prices = data["close"]
            
            upper_band, middle_band, lower_band = talib.BBANDS(
                close_prices,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0
            )
            
            latest_upper = upper_band[-1] if len(upper_band) > 0 else np.nan
            latest_middle = middle_band[-1] if len(middle_band) > 0 else np.nan
            latest_lower = lower_band[-1] if len(lower_band) > 0 else np.nan
            current_price = close_prices[-1]
            
            # 布林带信号判断
            bb_signal = "neutral"
            squeeze_detected = False
            
            if not (np.isnan(latest_upper) or np.isnan(latest_lower) or np.isnan(latest_middle)):
                band_width = (latest_upper - latest_lower) / latest_middle * 100
                
                # 价格位置判断
                if current_price > latest_upper:
                    bb_signal = "overbought"
                elif current_price < latest_lower:
                    bb_signal = "oversold"
                elif current_price > latest_middle:
                    bb_signal = "bullish"
                else:
                    bb_signal = "bearish"
                
                # 布林带收缩判断
                if len(upper_band) > period:
                    prev_width = (upper_band[-period] - lower_band[-period]) / middle_band[-period] * 100
                    if band_width < prev_width * 0.8:  # 带宽缩小20%以上
                        squeeze_detected = True
                
                # %B 计算
                percent_b = (current_price - latest_lower) / (latest_upper - latest_lower) if latest_upper != latest_lower else 0.5
            else:
                band_width = np.nan
                percent_b = np.nan
            
            result_value = {
                "upper": latest_upper,
                "middle": latest_middle,
                "lower": latest_lower
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    **self.parameters,
                    "signal": bb_signal,
                    "band_width": band_width,
                    "percent_b": percent_b,
                    "squeeze": squeeze_detected,
                    "current_price": current_price
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan},
                metadata={"error": str(e)}
            )


class ATR(AsyncTechnicalIndicator):
    """平均真实波动范围"""
    
    def __init__(self, period: int = 14):
        config = IndicatorConfig(
            name=f"ATR_{period}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": period},
            min_periods=period
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算ATR"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            high_prices = data["high"]
            low_prices = data["low"]
            close_prices = data["close"]
            
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            latest_value = atr_values[-1] if len(atr_values) > 0 else np.nan
            current_price = close_prices[-1]
            
            # ATR相对分析
            volatility_level = "normal"
            atr_percent = np.nan
            
            if not np.isnan(latest_value) and not np.isnan(current_price) and current_price != 0:
                atr_percent = latest_value / current_price * 100
                
                if atr_percent > 5:
                    volatility_level = "very_high"
                elif atr_percent > 3:
                    volatility_level = "high"
                elif atr_percent > 1.5:
                    volatility_level = "moderate"
                elif atr_percent < 0.5:
                    volatility_level = "low"
            
            # ATR趋势分析
            atr_trend = "neutral"
            if len(atr_values) > period:
                recent_avg = np.mean(atr_values[-period:])
                older_avg = np.mean(atr_values[-2*period:-period])
                
                if recent_avg > older_avg * 1.2:
                    atr_trend = "increasing"
                elif recent_avg < older_avg * 0.8:
                    atr_trend = "decreasing"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "volatility_level": volatility_level,
                    "atr_percent": atr_percent,
                    "atr_trend": atr_trend,
                    "current_price": current_price
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan,
                metadata={"error": str(e)}
            )


class StandardDeviation(AsyncTechnicalIndicator):
    """标准差指标"""
    
    def __init__(self, period: int = 20, nbdev: float = 1.0):
        config = IndicatorConfig(
            name=f"STDDEV_{period}_{nbdev}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": period, "nbdev": nbdev},
            min_periods=period
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算标准差"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            nbdev = self.parameters["nbdev"]
            close_prices = data["close"]
            
            std_values = talib.STDDEV(close_prices, timeperiod=period, nbdev=nbdev)
            latest_value = std_values[-1] if len(std_values) > 0 else np.nan
            current_price = close_prices[-1]
            
            # 标准差相对分析
            volatility_percentile = np.nan
            if len(std_values) > period * 2:
                # 计算历史百分位数
                historical_std = std_values[:-1]  # 排除当前值
                valid_std = historical_std[~np.isnan(historical_std)]
                if len(valid_std) > 0:
                    volatility_percentile = np.percentile(valid_std, np.sum(valid_std <= latest_value) / len(valid_std) * 100)
            
            # 波动性等级
            volatility_level = "normal"
            std_percent = np.nan
            
            if not np.isnan(latest_value) and not np.isnan(current_price) and current_price != 0:
                std_percent = latest_value / current_price * 100
                
                if std_percent > 4:
                    volatility_level = "very_high"
                elif std_percent > 2.5:
                    volatility_level = "high"
                elif std_percent > 1:
                    volatility_level = "moderate"
                elif std_percent < 0.5:
                    volatility_level = "low"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "nbdev": nbdev,
                    "volatility_level": volatility_level,
                    "std_percent": std_percent,
                    "volatility_percentile": volatility_percentile,
                    "current_price": current_price
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan,
                metadata={"error": str(e)}
            )


class KeltnerChannels(AsyncTechnicalIndicator):
    """凯尔特纳通道"""
    
    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        config = IndicatorConfig(
            name=f"KELTNER_{ema_period}_{atr_period}_{multiplier}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={
                "ema_period": ema_period,
                "atr_period": atr_period,
                "multiplier": multiplier
            },
            min_periods=max(ema_period, atr_period)
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算凯尔特纳通道"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan}
            )
        
        start_time = time.time()
        try:
            ema_period = self.parameters["ema_period"]
            atr_period = self.parameters["atr_period"]
            multiplier = self.parameters["multiplier"]
            
            high_prices = data["high"]
            low_prices = data["low"]
            close_prices = data["close"]
            
            # 中线：EMA
            middle_line = talib.EMA(close_prices, timeperiod=ema_period)
            
            # ATR
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
            
            # 上下通道
            upper_channel = middle_line + (atr_values * multiplier)
            lower_channel = middle_line - (atr_values * multiplier)
            
            latest_upper = upper_channel[-1] if len(upper_channel) > 0 else np.nan
            latest_middle = middle_line[-1] if len(middle_line) > 0 else np.nan
            latest_lower = lower_channel[-1] if len(lower_channel) > 0 else np.nan
            current_price = close_prices[-1]
            
            # 凯尔特纳通道信号判断
            keltner_signal = "neutral"
            channel_position = "middle"
            
            if not (np.isnan(latest_upper) or np.isnan(latest_lower) or np.isnan(latest_middle)):
                if current_price > latest_upper:
                    keltner_signal = "breakout_bullish"
                    channel_position = "above"
                elif current_price < latest_lower:
                    keltner_signal = "breakout_bearish"
                    channel_position = "below"
                elif current_price > latest_middle:
                    keltner_signal = "bullish"
                    channel_position = "upper_half"
                else:
                    keltner_signal = "bearish"
                    channel_position = "lower_half"
                
                # 通道宽度
                channel_width = (latest_upper - latest_lower) / latest_middle * 100 if latest_middle != 0 else np.nan
            else:
                channel_width = np.nan
            
            result_value = {
                "upper": latest_upper,
                "middle": latest_middle,
                "lower": latest_lower
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    **self.parameters,
                    "signal": keltner_signal,
                    "channel_position": channel_position,
                    "channel_width": channel_width,
                    "current_price": current_price
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan},
                metadata={"error": str(e)}
            )


class DonchianChannels(AsyncTechnicalIndicator):
    """唐奇安通道"""
    
    def __init__(self, period: int = 20):
        config = IndicatorConfig(
            name=f"DONCHIAN_{period}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": period},
            min_periods=period
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算唐奇安通道"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan}
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            high_prices = data["high"]
            low_prices = data["low"]
            
            # 计算最高和最低价通道
            upper_channel = np.full_like(high_prices, np.nan)
            lower_channel = np.full_like(low_prices, np.nan)
            
            for i in range(period - 1, len(high_prices)):
                upper_channel[i] = np.max(high_prices[i - period + 1:i + 1])
                lower_channel[i] = np.min(low_prices[i - period + 1:i + 1])
            
            # 中线：上下通道的中点
            middle_channel = (upper_channel + lower_channel) / 2
            
            latest_upper = upper_channel[-1] if len(upper_channel) > 0 else np.nan
            latest_middle = middle_channel[-1] if len(middle_channel) > 0 else np.nan
            latest_lower = lower_channel[-1] if len(lower_channel) > 0 else np.nan
            
            current_price = data["close"][-1] if "close" in data else (high_prices[-1] + low_prices[-1]) / 2
            
            # 唐奇安通道信号判断
            donchian_signal = "neutral"
            breakout_type = "none"
            
            if not (np.isnan(latest_upper) or np.isnan(latest_lower)):
                if current_price >= latest_upper:
                    donchian_signal = "breakout_bullish"
                    breakout_type = "upper_breakout"
                elif current_price <= latest_lower:
                    donchian_signal = "breakout_bearish"
                    breakout_type = "lower_breakout"
                elif current_price > latest_middle:
                    donchian_signal = "bullish"
                else:
                    donchian_signal = "bearish"
                
                # 通道宽度分析
                channel_width = (latest_upper - latest_lower) / latest_middle * 100 if latest_middle != 0 else np.nan
                
                # 价格在通道中的位置百分比
                channel_position = (current_price - latest_lower) / (latest_upper - latest_lower) if latest_upper != latest_lower else 0.5
            else:
                channel_width = np.nan
                channel_position = np.nan
            
            result_value = {
                "upper": latest_upper,
                "middle": latest_middle,
                "lower": latest_lower
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    "period": period,
                    "signal": donchian_signal,
                    "breakout_type": breakout_type,
                    "channel_width": channel_width,
                    "channel_position": channel_position,
                    "current_price": current_price
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"upper": np.nan, "middle": np.nan, "lower": np.nan},
                metadata={"error": str(e)}
            )


class VIXProxy(AsyncTechnicalIndicator):
    """VIX代理指标 - 基于期权隐含波动率的简化实现"""
    
    def __init__(self, period: int = 30, lookback: int = 252):
        config = IndicatorConfig(
            name=f"VIX_PROXY_{period}_{lookback}",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": period, "lookback": lookback},
            min_periods=max(period, 30),  # 需要足够的数据计算波动率
            normalize=True,
            normalize_range=(5, 80)  # VIX 典型范围
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算VIX代理指标"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            lookback = self.parameters["lookback"]
            close_prices = data["close"]
            
            # 计算日收益率
            returns = np.diff(np.log(close_prices))
            
            # 计算滚动波动率（年化）
            vix_proxy = np.full_like(close_prices, np.nan)
            
            for i in range(period - 1, len(returns)):
                window_returns = returns[max(0, i - period + 1):i + 1]
                if len(window_returns) >= period:
                    volatility = np.std(window_returns, ddof=1) * np.sqrt(252) * 100  # 年化百分比
                    vix_proxy[i + 1] = volatility  # +1 因为returns比prices短1
            
            latest_value = vix_proxy[-1] if len(vix_proxy) > 0 else np.nan
            
            # VIX水平分析
            fear_level = "normal"
            market_stress = "low"
            
            if not np.isnan(latest_value):
                if latest_value > 40:
                    fear_level = "extreme_fear"
                    market_stress = "very_high"
                elif latest_value > 30:
                    fear_level = "high_fear"
                    market_stress = "high"
                elif latest_value > 20:
                    fear_level = "elevated"
                    market_stress = "moderate"
                elif latest_value < 12:
                    fear_level = "complacency"
                    market_stress = "very_low"
                elif latest_value < 16:
                    fear_level = "low_fear"
            
            # 历史百分位数
            historical_percentile = np.nan
            if len(vix_proxy) > lookback:
                historical_vix = vix_proxy[-lookback:]
                valid_vix = historical_vix[~np.isnan(historical_vix)]
                if len(valid_vix) > 0:
                    historical_percentile = np.percentile(valid_vix, 
                                                       np.sum(valid_vix <= latest_value) / len(valid_vix) * 100)
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "fear_level": fear_level,
                    "market_stress": market_stress,
                    "historical_percentile": historical_percentile,
                    "volatility_regime": "high" if latest_value > 25 else "low"
                }
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan,
                metadata={"error": str(e)}
            )