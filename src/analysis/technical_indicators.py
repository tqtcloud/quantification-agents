"""
技术指标计算引擎 - Legacy版本
为保持向后兼容性，该文件现在重定向到新的指标架构
建议新代码直接使用 src.core.indicators 模块
"""

# 导入新的指标架构
from src.core.indicators.legacy_adapter import (
    TechnicalIndicators,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    custom_momentum_indicator
)

# 保持兼容性导入
import numpy as np
import pandas as pd
import time

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    from src.core.mock_talib import MockTalib
    
    talib = MockTalib()

# 为了向后兼容，导入新架构的类
from src.core.indicators import (
    IndicatorType,
    IndicatorConfig,
    IndicatorResult,
    BaseIndicator,
    CustomIndicator
)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from collections import deque, defaultdict

from src.utils.logger import LoggerMixin


# 这些类现在从新架构中导入，此处保留空定义以保持兼容性


# BaseIndicator 现在从新架构导入，此处保留注释以保持文档


# SimpleMovingAverage 现在从 src.core.indicators.trend 导入
# 此处保留为兼容性，但建议使用新的导入方式
from src.core.indicators.trend import SimpleMovingAverage


# ExponentialMovingAverage 现在从 src.core.indicators.trend 导入
from src.core.indicators.trend import ExponentialMovingAverage


# RelativeStrengthIndex 现在从 src.core.indicators.momentum 导入
from src.core.indicators.momentum import RelativeStrengthIndex


# MACD 现在从 src.core.indicators.momentum 导入
from src.core.indicators.momentum import MACD


# BollingerBands 现在从 src.core.indicators.volatility 导入
from src.core.indicators.volatility import BollingerBands


# StochasticOscillator 现在从 src.core.indicators.momentum 导入
from src.core.indicators.momentum import StochasticOscillator


# CustomIndicator 现在从 src.core.indicators.base 导入
from src.core.indicators.base import CustomIndicator


# TechnicalIndicators 现在使用新架构的适配器
# 主要功能已经迁移到 src.core.indicators.legacy_adapter
# 此处保留类定义以保持向后兼容


# 便捷函数已经通过适配器导入，保持接口一致性
# 这些函数现在内部使用新的架构实现

# 所有便捷函数和自定义指标示例已经在文件顶部通过 legacy_adapter 导入