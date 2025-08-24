"""
趋势指标实现
包括SMA、EMA、ADX、抛物线SAR、一目均衡表等
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


class SimpleMovingAverage(AsyncTechnicalIndicator):
    """简单移动平均线（SMA）"""
    
    def __init__(self, period: int = 20):
        config = IndicatorConfig(
            name=f"SMA_{period}",
            indicator_type=IndicatorType.TREND,
            parameters={"period": period},
            min_periods=period
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算SMA"""
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
            close_prices = data["close"]
            
            sma_values = talib.SMA(close_prices, timeperiod=period)
            latest_value = sma_values[-1] if len(sma_values) > 0 else np.nan
            current_price = close_prices[-1]
            
            # 趋势判断
            trend_signal = "neutral"
            if not np.isnan(latest_value) and not np.isnan(current_price):
                if current_price > latest_value:
                    trend_signal = "bullish"
                elif current_price < latest_value:
                    trend_signal = "bearish"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "trend": trend_signal,
                    "current_price": current_price,
                    "distance_pct": ((current_price - latest_value) / latest_value * 100) if not np.isnan(latest_value) and latest_value != 0 else np.nan
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


class ExponentialMovingAverage(AsyncTechnicalIndicator):
    """指数移动平均线（EMA）"""
    
    def __init__(self, period: int = 20):
        config = IndicatorConfig(
            name=f"EMA_{period}",
            indicator_type=IndicatorType.TREND,
            parameters={"period": period},
            min_periods=period
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算EMA"""
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
            close_prices = data["close"]
            
            ema_values = talib.EMA(close_prices, timeperiod=period)
            latest_value = ema_values[-1] if len(ema_values) > 0 else np.nan
            current_price = close_prices[-1]
            
            # 趋势判断和斜率计算
            trend_signal = "neutral"
            slope = np.nan
            if not np.isnan(latest_value) and not np.isnan(current_price):
                if current_price > latest_value:
                    trend_signal = "bullish"
                elif current_price < latest_value:
                    trend_signal = "bearish"
                
                # 计算EMA斜率
                if len(ema_values) > 1:
                    slope = ema_values[-1] - ema_values[-2]
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "trend": trend_signal,
                    "slope": slope,
                    "current_price": current_price,
                    "distance_pct": ((current_price - latest_value) / latest_value * 100) if not np.isnan(latest_value) and latest_value != 0 else np.nan
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


class ADX(AsyncTechnicalIndicator):
    """平均方向指数"""
    
    def __init__(self, period: int = 14):
        config = IndicatorConfig(
            name=f"ADX_{period}",
            indicator_type=IndicatorType.TREND,
            parameters={"period": period},
            min_periods=period * 2,
            normalize=True,
            normalize_range=(0, 100)
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算ADX"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"adx": np.nan, "plus_di": np.nan, "minus_di": np.nan}
            )
        
        start_time = time.time()
        try:
            period = self.parameters["period"]
            high_prices = data["high"]
            low_prices = data["low"]
            close_prices = data["close"]
            
            adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
            plus_di_values = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            minus_di_values = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
            
            latest_adx = adx_values[-1] if len(adx_values) > 0 else np.nan
            latest_plus_di = plus_di_values[-1] if len(plus_di_values) > 0 else np.nan
            latest_minus_di = minus_di_values[-1] if len(minus_di_values) > 0 else np.nan
            
            # ADX信号判断
            trend_strength = "weak"
            trend_direction = "neutral"
            
            if not np.isnan(latest_adx):
                if latest_adx > 50:
                    trend_strength = "very_strong"
                elif latest_adx > 25:
                    trend_strength = "strong"
                elif latest_adx > 15:
                    trend_strength = "moderate"
            
            if not (np.isnan(latest_plus_di) or np.isnan(latest_minus_di)):
                if latest_plus_di > latest_minus_di:
                    trend_direction = "bullish"
                else:
                    trend_direction = "bearish"
            
            result_value = {
                "adx": latest_adx,
                "plus_di": latest_plus_di,
                "minus_di": latest_minus_di
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    "period": period,
                    "trend_strength": trend_strength,
                    "trend_direction": trend_direction,
                    "directional_movement": latest_plus_di - latest_minus_di if not (np.isnan(latest_plus_di) or np.isnan(latest_minus_di)) else np.nan
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
                value={"adx": np.nan, "plus_di": np.nan, "minus_di": np.nan},
                metadata={"error": str(e)}
            )


class ParabolicSAR(AsyncTechnicalIndicator):
    """抛物线SAR"""
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        config = IndicatorConfig(
            name=f"SAR_{acceleration}_{maximum}",
            indicator_type=IndicatorType.TREND,
            parameters={"acceleration": acceleration, "maximum": maximum},
            min_periods=2
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算抛物线SAR"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan
            )
        
        start_time = time.time()
        try:
            acceleration = self.parameters["acceleration"]
            maximum = self.parameters["maximum"]
            high_prices = data["high"]
            low_prices = data["low"]
            
            sar_values = talib.SAR(high_prices, low_prices, acceleration=acceleration, maximum=maximum)
            latest_value = sar_values[-1] if len(sar_values) > 0 else np.nan
            current_price = data["close"][-1] if "close" in data else (high_prices[-1] + low_prices[-1]) / 2
            
            # SAR信号判断
            sar_signal = "neutral"
            trend = "neutral"
            
            if not np.isnan(latest_value) and not np.isnan(current_price):
                if current_price > latest_value:
                    sar_signal = "bullish"
                    trend = "uptrend"
                else:
                    sar_signal = "bearish"
                    trend = "downtrend"
                
                # 检查是否有趋势反转
                if len(sar_values) > 1:
                    prev_price = data["close"][-2] if "close" in data else (high_prices[-2] + low_prices[-2]) / 2
                    prev_sar = sar_values[-2]
                    
                    if not np.isnan(prev_sar) and not np.isnan(prev_price):
                        prev_above = prev_price > prev_sar
                        curr_above = current_price > latest_value
                        
                        if prev_above != curr_above:
                            sar_signal = "reversal_bullish" if curr_above else "reversal_bearish"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "acceleration": acceleration,
                    "maximum": maximum,
                    "signal": sar_signal,
                    "trend": trend,
                    "current_price": current_price,
                    "stop_distance": abs(current_price - latest_value) if not np.isnan(latest_value) and not np.isnan(current_price) else np.nan
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


class IchimokuCloud(AsyncTechnicalIndicator):
    """一目均衡表"""
    
    def __init__(self, conversion_period: int = 9, base_period: int = 26, 
                 leading_span_b_period: int = 52, displacement: int = 26):
        config = IndicatorConfig(
            name=f"ICHIMOKU_{conversion_period}_{base_period}_{leading_span_b_period}",
            indicator_type=IndicatorType.TREND,
            parameters={
                "conversion_period": conversion_period,
                "base_period": base_period,
                "leading_span_b_period": leading_span_b_period,
                "displacement": displacement
            },
            min_periods=max(conversion_period, base_period, leading_span_b_period)
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算一目均衡表"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={
                    "conversion_line": np.nan,
                    "base_line": np.nan,
                    "leading_span_a": np.nan,
                    "leading_span_b": np.nan,
                    "lagging_span": np.nan
                }
            )
        
        start_time = time.time()
        try:
            conv_period = self.parameters["conversion_period"]
            base_period = self.parameters["base_period"]
            span_b_period = self.parameters["leading_span_b_period"]
            displacement = self.parameters["displacement"]
            
            high_prices = data["high"]
            low_prices = data["low"]
            close_prices = data["close"]
            
            # 转换线 (Tenkan-sen) = (9期最高 + 9期最低) / 2
            conv_high = np.full_like(high_prices, np.nan)
            conv_low = np.full_like(low_prices, np.nan)
            for i in range(conv_period - 1, len(high_prices)):
                conv_high[i] = np.max(high_prices[i - conv_period + 1:i + 1])
                conv_low[i] = np.min(low_prices[i - conv_period + 1:i + 1])
            conversion_line = (conv_high + conv_low) / 2
            
            # 基准线 (Kijun-sen) = (26期最高 + 26期最低) / 2  
            base_high = np.full_like(high_prices, np.nan)
            base_low = np.full_like(low_prices, np.nan)
            for i in range(base_period - 1, len(high_prices)):
                base_high[i] = np.max(high_prices[i - base_period + 1:i + 1])
                base_low[i] = np.min(low_prices[i - base_period + 1:i + 1])
            base_line = (base_high + base_low) / 2
            
            # 先行带A (Senkou Span A) = (转换线 + 基准线) / 2, 前移26期
            leading_span_a = (conversion_line + base_line) / 2
            
            # 先行带B (Senkou Span B) = (52期最高 + 52期最低) / 2, 前移26期
            span_b_high = np.full_like(high_prices, np.nan)
            span_b_low = np.full_like(low_prices, np.nan)
            for i in range(span_b_period - 1, len(high_prices)):
                span_b_high[i] = np.max(high_prices[i - span_b_period + 1:i + 1])
                span_b_low[i] = np.min(low_prices[i - span_b_period + 1:i + 1])
            leading_span_b = (span_b_high + span_b_low) / 2
            
            # 滞后线 (Chikou Span) = 收盘价，后移26期
            lagging_span = close_prices
            
            # 获取最新值
            latest_conv = conversion_line[-1] if len(conversion_line) > 0 else np.nan
            latest_base = base_line[-1] if len(base_line) > 0 else np.nan
            latest_span_a = leading_span_a[-displacement] if len(leading_span_a) > displacement else np.nan
            latest_span_b = leading_span_b[-displacement] if len(leading_span_b) > displacement else np.nan
            latest_lagging = lagging_span[-displacement] if len(lagging_span) > displacement else np.nan
            
            current_price = close_prices[-1]
            
            # 一目均衡表信号判断
            ichimoku_signal = "neutral"
            cloud_color = "neutral"
            price_vs_cloud = "neutral"
            
            if not np.isnan(latest_span_a) and not np.isnan(latest_span_b):
                # 云的颜色
                if latest_span_a > latest_span_b:
                    cloud_color = "bullish"  # 绿云
                else:
                    cloud_color = "bearish"  # 红云
                
                # 价格相对于云的位置
                cloud_top = max(latest_span_a, latest_span_b)
                cloud_bottom = min(latest_span_a, latest_span_b)
                
                if current_price > cloud_top:
                    price_vs_cloud = "above"
                elif current_price < cloud_bottom:
                    price_vs_cloud = "below"
                else:
                    price_vs_cloud = "inside"
            
            # 综合信号
            if not (np.isnan(latest_conv) or np.isnan(latest_base)):
                if latest_conv > latest_base and cloud_color == "bullish" and price_vs_cloud == "above":
                    ichimoku_signal = "strong_bullish"
                elif latest_conv > latest_base and (cloud_color == "bullish" or price_vs_cloud == "above"):
                    ichimoku_signal = "bullish"
                elif latest_conv < latest_base and cloud_color == "bearish" and price_vs_cloud == "below":
                    ichimoku_signal = "strong_bearish"
                elif latest_conv < latest_base and (cloud_color == "bearish" or price_vs_cloud == "below"):
                    ichimoku_signal = "bearish"
            
            result_value = {
                "conversion_line": latest_conv,
                "base_line": latest_base,
                "leading_span_a": latest_span_a,
                "leading_span_b": latest_span_b,
                "lagging_span": latest_lagging
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    **self.parameters,
                    "signal": ichimoku_signal,
                    "cloud_color": cloud_color,
                    "price_vs_cloud": price_vs_cloud,
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
                value={
                    "conversion_line": np.nan,
                    "base_line": np.nan,
                    "leading_span_a": np.nan,
                    "leading_span_b": np.nan,
                    "lagging_span": np.nan
                },
                metadata={"error": str(e)}
            )