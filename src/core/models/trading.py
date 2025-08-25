"""
交易相关数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field as PydanticField


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    vwap: float = 0.0  # Volume Weighted Average Price
    trades_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """市值"""
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """盈亏百分比"""
        if self.average_price == 0:
            return 0.0
        return (self.current_price - self.average_price) / self.average_price * 100


@dataclass
class Signal:
    """交易信号"""
    source: str
    symbol: str
    action: str  # BUY, SELL, HOLD, etc.
    strength: float  # -1.0 到 1.0
    confidence: float  # 0.0 到 1.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskMetrics(BaseModel):
    """风险指标"""
    total_exposure: float = PydanticField(default=0.0, description="总敞口")
    margin_usage: float = PydanticField(default=0.0, description="保证金使用率")
    leverage: float = PydanticField(default=1.0, description="杠杆率")
    var_95: float = PydanticField(default=0.0, description="95% VaR")
    var_99: float = PydanticField(default=0.0, description="99% VaR")
    sharpe_ratio: float = PydanticField(default=0.0, description="夏普比率")
    max_drawdown: float = PydanticField(default=0.0, description="最大回撤")
    current_drawdown: float = PydanticField(default=0.0, description="当前回撤")
    win_rate: float = PydanticField(default=0.0, description="胜率")
    profit_factor: float = PydanticField(default=0.0, description="盈亏比")
    

@dataclass
class TradingState:
    """交易状态"""
    timestamp: datetime = field(default_factory=datetime.now)
    active_symbols: List[str] = field(default_factory=list)
    market_data: Dict[str, MarketData] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    open_orders: List[Order] = field(default_factory=list)
    pending_signals: List[Signal] = field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    account_balance: float = 10000.0
    available_balance: float = 10000.0
    total_equity: float = 10000.0
    indicators: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_sentiment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)