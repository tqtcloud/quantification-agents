"""
量化交易系统核心数据模型包

提供完整的数据模型定义，包括：
- 信号相关模型 (signals.py)
- 交易相关模型 (trading.py)
"""

from .signals import (
    SignalStrength,
    TradingSignal,
    MultiDimensionalSignal,
    SignalAggregator,
)

from .trading import (
    MarketData,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    Position,
    Signal,
    RiskMetrics,
    TradingState,
)

__all__ = [
    # 信号相关
    "SignalStrength",
    "TradingSignal", 
    "MultiDimensionalSignal",
    "SignalAggregator",
    # 交易相关
    "MarketData",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Position",
    "Signal",
    "RiskMetrics",
    "TradingState",
]