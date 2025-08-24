"""
仓位管理模块

提供完整的仓位管理功能，包括自动平仓、仓位监控、风险控制等。
"""

from .models import (
    PositionInfo,
    ClosingStrategy, 
    ClosingReason,
    ClosingAction,
    PositionCloseRequest,
    PositionCloseResult
)

from .auto_position_closer import AutoPositionCloser
from .position_manager import PositionManager
from .closing_strategies import (
    ProfitTargetStrategy,
    StopLossStrategy,
    TrailingStopStrategy,
    TimeBasedStrategy,
    TechnicalReversalStrategy,
    SentimentStrategy,
    DynamicTrailingStrategy
)

__all__ = [
    'PositionInfo',
    'ClosingStrategy',
    'ClosingReason', 
    'ClosingAction',
    'PositionCloseRequest',
    'PositionCloseResult',
    'AutoPositionCloser',
    'PositionManager',
    'ProfitTargetStrategy',
    'StopLossStrategy', 
    'TrailingStopStrategy',
    'TimeBasedStrategy',
    'TechnicalReversalStrategy',
    'SentimentStrategy',
    'DynamicTrailingStrategy'
]