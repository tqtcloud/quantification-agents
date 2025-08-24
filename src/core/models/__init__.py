"""
量化交易系统核心数据模型包

提供完整的数据模型定义，包括：
- 信号相关模型 (signals.py)
"""

from .signals import (
    SignalStrength,
    TradingSignal,
    MultiDimensionalSignal,
    SignalAggregator,
)

__all__ = [
    # 信号相关
    "SignalStrength",
    "TradingSignal", 
    "MultiDimensionalSignal",
    "SignalAggregator",
]