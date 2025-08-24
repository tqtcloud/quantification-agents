"""
Legacy Adapter - 向后兼容性适配器
为了保持与现有代码的兼容性，提供原有接口的适配实现
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union

# 导入新架构的组件
from .base import TechnicalIndicators as NewTechnicalIndicators
from .momentum import RelativeStrengthIndex, MACD, StochasticOscillator
from .trend import SimpleMovingAverage, ExponentialMovingAverage
from .volatility import BollingerBands
from .utils import (
    calculate_sma as new_calculate_sma,
    calculate_ema as new_calculate_ema,
    calculate_rsi as new_calculate_rsi,
    calculate_macd as new_calculate_macd,
    calculate_bollinger_bands as new_calculate_bollinger_bands
)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    from ..mock_talib import MockTalib
    talib = MockTalib()


class TechnicalIndicators:
    """
    Legacy TechnicalIndicators class for backward compatibility
    提供与原有系统相同的接口，内部使用新架构实现
    """
    
    def __init__(self):
        self._new_indicators = NewTechnicalIndicators()
        self._registered_indicators = {}
        
        # 注册默认指标以保持兼容性
        self._register_legacy_indicators()
    
    def _register_legacy_indicators(self):
        """注册传统指标以保持接口兼容"""
        # 趋势指标
        self._registered_indicators['SMA_20'] = SimpleMovingAverage(20)
        self._registered_indicators['SMA_50'] = SimpleMovingAverage(50)
        self._registered_indicators['EMA_20'] = ExponentialMovingAverage(20)
        self._registered_indicators['EMA_50'] = ExponentialMovingAverage(50)
        
        # 动量指标
        self._registered_indicators['RSI_14'] = RelativeStrengthIndex(14)
        self._registered_indicators['MACD_12_26_9'] = MACD()
        self._registered_indicators['STOCH_14_3'] = StochasticOscillator()
        
        # 波动率指标
        self._registered_indicators['BBANDS_20_2'] = BollingerBands()
        
        # 向新系统注册这些指标
        for indicator in self._registered_indicators.values():
            self._new_indicators.register_indicator(indicator)
    
    def register_indicator(self, indicator):
        """注册技术指标"""
        return self._new_indicators.register_indicator(indicator)
    
    def unregister_indicator(self, indicator_name: str):
        """注销技术指标"""
        return self._new_indicators.unregister_indicator(indicator_name)
    
    def add_custom_indicator(self, name: str, calculation_func, parameters: Dict[str, Any] = None):
        """添加自定义指标"""
        return self._new_indicators.add_custom_indicator(name, calculation_func, parameters)
    
    def update_data(self, symbol: str, price_data: Dict[str, float]):
        """更新价格数据"""
        return self._new_indicators.update_data(symbol, price_data)
    
    def calculate_indicator(self, indicator_name: str, symbol: str, force_recalculate: bool = False):
        """计算单个指标"""
        result = self._new_indicators.calculate_indicator(indicator_name, symbol, force_recalculate)
        
        # 转换为Legacy格式（如果需要）
        if result:
            return result
        return None
    
    def calculate_all_indicators(self, symbol: str, force_recalculate: bool = False):
        """计算所有指标"""
        return self._new_indicators.calculate_all_indicators(symbol, force_recalculate)
    
    def get_indicator_result(self, symbol: str, indicator_name: str):
        """获取指标计算结果"""
        return self._new_indicators.get_indicator_result(symbol, indicator_name)
    
    def get_all_results(self, symbol: str):
        """获取所有指标结果"""
        return self._new_indicators.get_all_results(symbol)
    
    def get_performance_stats(self):
        """获取性能统计"""
        return self._new_indicators.get_global_performance_stats()
    
    def clear_cache(self, symbol: str = None):
        """清理缓存"""
        return self._new_indicators.clear_cache(symbol)
    
    def get_registered_indicators(self):
        """获取已注册的指标列表"""
        return self._new_indicators.get_registered_indicators()
    
    def get_indicator_config(self, indicator_name: str):
        """获取指标配置"""
        return self._new_indicators.get_indicator_config(indicator_name)


# 保持原有便捷函数的接口不变
def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均线"""
    return new_calculate_sma(prices, period)


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均线"""
    return new_calculate_ema(prices, period)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI"""
    return new_calculate_rsi(prices, period)


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD"""
    return new_calculate_macd(prices, fast, slow, signal)


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: int = 2):
    """计算布林带"""
    return new_calculate_bollinger_bands(prices, period, std_dev)


def custom_momentum_indicator(data: Dict[str, np.ndarray], parameters: Dict[str, Any]) -> float:
    """自定义动量指标示例 - 保持原有接口"""
    from .utils import custom_momentum_indicator as new_custom_momentum
    return new_custom_momentum(data, parameters)


# 保持与原有系统兼容的类和枚举
try:
    from .base import IndicatorType, IndicatorConfig, IndicatorResult, BaseIndicator
    from .momentum import *
    from .trend import *  
    from .volatility import *
except ImportError:
    # 如果新模块导入失败，提供基本兼容性
    pass


# 向后兼容的导入别名
TechnicalIndicatorsLegacy = TechnicalIndicators