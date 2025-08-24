"""
技术指标计算引擎核心模块
提供完整的技术指标计算功能，支持异步计算、多时间框架和性能优化
"""

from .base import (
    BaseIndicator, 
    AsyncTechnicalIndicator,
    IndicatorType, 
    IndicatorConfig, 
    IndicatorResult,
    TechnicalIndicators
)
from .momentum import *
from .trend import *
from .volatility import *
from .normalizer import IndicatorNormalizer
from .timeframe import TimeFrameManager, TimeFrame, TimeFrameConfig, OHLCVData
from .utils import *

__all__ = [
    # 基础类
    'BaseIndicator',
    'AsyncTechnicalIndicator', 
    'IndicatorType',
    'IndicatorConfig',
    'IndicatorResult',
    'TechnicalIndicators',
    'IndicatorNormalizer',
    'TimeFrameManager',
    'TimeFrame',
    'TimeFrameConfig',
    'OHLCVData',
    
    # 动量指标
    'RelativeStrengthIndex',
    'MACD', 
    'StochasticOscillator',
    'CCI',
    'WilliamsR',
    
    # 趋势指标
    'SimpleMovingAverage',
    'ExponentialMovingAverage', 
    'ADX',
    'ParabolicSAR',
    'IchimokuCloud',
    
    # 波动率指标
    'BollingerBands',
    'ATR',
    'StandardDeviation',
    'KeltnerChannels',
    'DonchianChannels',
    
    # 工具函数
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bollinger_bands'
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Quantification Agents'