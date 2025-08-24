"""
动量指标实现
包括RSI、MACD、随机指标、CCI、威廉指标等
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


class RelativeStrengthIndex(AsyncTechnicalIndicator):
    """相对强弱指数（RSI）"""
    
    def __init__(self, period: int = 14):
        config = IndicatorConfig(
            name=f"RSI_{period}",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={"period": period},
            min_periods=period + 1,
            normalize=True,
            normalize_range=(0, 100)
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算RSI"""
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
            
            rsi_values = talib.RSI(close_prices, timeperiod=period)
            latest_value = rsi_values[-1] if len(rsi_values) > 0 else np.nan
            
            # RSI信号判断
            signal = "neutral"
            if not np.isnan(latest_value):
                if latest_value > 70:
                    signal = "overbought"
                elif latest_value < 30:
                    signal = "oversold"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "signal": signal,
                    "overbought_level": 70,
                    "oversold_level": 30
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


class MACD(AsyncTechnicalIndicator):
    """MACD指标"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        config = IndicatorConfig(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period
            },
            min_periods=slow_period + signal_period
        )
        super().__init__(config)
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算MACD"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"macd": np.nan, "signal": np.nan, "histogram": np.nan}
            )
        
        start_time = time.time()
        try:
            fast_period = self.parameters["fast_period"]
            slow_period = self.parameters["slow_period"]
            signal_period = self.parameters["signal_period"]
            close_prices = data["close"]
            
            macd_line, signal_line, histogram = talib.MACD(
                close_prices,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            
            latest_macd = macd_line[-1] if len(macd_line) > 0 else np.nan
            latest_signal = signal_line[-1] if len(signal_line) > 0 else np.nan
            latest_histogram = histogram[-1] if len(histogram) > 0 else np.nan
            
            # MACD信号判断
            macd_signal = "neutral"
            if not (np.isnan(latest_macd) or np.isnan(latest_signal)):
                if latest_macd > latest_signal:
                    if len(histogram) > 1 and histogram[-2] < latest_histogram:
                        macd_signal = "bullish_crossover"
                    else:
                        macd_signal = "bullish"
                else:
                    if len(histogram) > 1 and histogram[-2] > latest_histogram:
                        macd_signal = "bearish_crossover"
                    else:
                        macd_signal = "bearish"
            
            result_value = {
                "macd": latest_macd,
                "signal": latest_signal,
                "histogram": latest_histogram
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    **self.parameters,
                    "signal": macd_signal,
                    "crossover": latest_macd > latest_signal if not np.isnan(latest_macd) and not np.isnan(latest_signal) else None
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
                value={"macd": np.nan, "signal": np.nan, "histogram": np.nan},
                metadata={"error": str(e)}
            )


class StochasticOscillator(AsyncTechnicalIndicator):
    """随机振荡器"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3):
        config = IndicatorConfig(
            name=f"STOCH_{k_period}_{d_period}",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={"k_period": k_period, "d_period": d_period},
            min_periods=k_period + d_period,
            normalize=True,
            normalize_range=(0, 100)
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算随机振荡器"""
        if not self.can_calculate(data) or not self._validate_data(data):
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value={"k": np.nan, "d": np.nan}
            )
        
        start_time = time.time()
        try:
            k_period = self.parameters["k_period"]
            d_period = self.parameters["d_period"]
            
            high_prices = data["high"]
            low_prices = data["low"]
            close_prices = data["close"]
            
            slowk, slowd = talib.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=k_period,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )
            
            latest_k = slowk[-1] if len(slowk) > 0 else np.nan
            latest_d = slowd[-1] if len(slowd) > 0 else np.nan
            
            # 随机指标信号判断
            stoch_signal = "neutral"
            if not (np.isnan(latest_k) or np.isnan(latest_d)):
                if latest_k > 80 and latest_d > 80:
                    stoch_signal = "overbought"
                elif latest_k < 20 and latest_d < 20:
                    stoch_signal = "oversold"
                elif latest_k > latest_d:
                    stoch_signal = "bullish"
                else:
                    stoch_signal = "bearish"
            
            result_value = {
                "k": latest_k,
                "d": latest_d
            }
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata={
                    **self.parameters,
                    "signal": stoch_signal,
                    "overbought_level": 80,
                    "oversold_level": 20
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
                value={"k": np.nan, "d": np.nan},
                metadata={"error": str(e)}
            )


class CCI(AsyncTechnicalIndicator):
    """商品通道指数"""
    
    def __init__(self, period: int = 20):
        config = IndicatorConfig(
            name=f"CCI_{period}",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={"period": period},
            min_periods=period,
            normalize=True,
            normalize_range=(-300, 300)  # CCI typical range
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算CCI"""
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
            
            cci_values = talib.CCI(high_prices, low_prices, close_prices, timeperiod=period)
            latest_value = cci_values[-1] if len(cci_values) > 0 else np.nan
            
            # CCI信号判断
            cci_signal = "neutral"
            if not np.isnan(latest_value):
                if latest_value > 100:
                    cci_signal = "overbought"
                elif latest_value < -100:
                    cci_signal = "oversold"
                elif latest_value > 0:
                    cci_signal = "bullish"
                else:
                    cci_signal = "bearish"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "signal": cci_signal,
                    "overbought_level": 100,
                    "oversold_level": -100
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


class WilliamsR(AsyncTechnicalIndicator):
    """威廉指标 (%R)"""
    
    def __init__(self, period: int = 14):
        config = IndicatorConfig(
            name=f"WILLR_{period}",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={"period": period},
            min_periods=period,
            normalize=True,
            normalize_range=(-100, 0)
        )
        super().__init__(config)
    
    def _get_required_data_keys(self) -> List[str]:
        return ["high", "low", "close"]
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算威廉指标"""
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
            
            willr_values = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
            latest_value = willr_values[-1] if len(willr_values) > 0 else np.nan
            
            # Williams %R信号判断
            willr_signal = "neutral"
            if not np.isnan(latest_value):
                if latest_value > -20:
                    willr_signal = "overbought"
                elif latest_value < -80:
                    willr_signal = "oversold"
                elif latest_value > -50:
                    willr_signal = "bullish"
                else:
                    willr_signal = "bearish"
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=latest_value,
                metadata={
                    "period": period,
                    "signal": willr_signal,
                    "overbought_level": -20,
                    "oversold_level": -80
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