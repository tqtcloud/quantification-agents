from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTX = "GTX"  # Good Till Crossing


@dataclass
class MarketData:
    symbol: str
    timestamp: int
    price: float
    volume: float
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    mark_price: Optional[float] = None
    index_price: Optional[float] = None


@dataclass
class OrderBook:
    symbol: str
    timestamp: int
    bids: List[tuple[float, float]]  # [(price, volume), ...]
    asks: List[tuple[float, float]]  # [(price, volume), ...]
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        return 0.0


class Order(BaseModel):
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    price: Optional[float] = Field(default=None, gt=0)
    stop_price: Optional[float] = Field(default=None, gt=0)
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    close_position: bool = False
    position_side: PositionSide = PositionSide.BOTH
    status: OrderStatus = OrderStatus.NEW
    executed_qty: float = 0.0
    avg_price: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class Position(BaseModel):
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    margin: float
    leverage: int = 1
    liquidation_price: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    @property
    def pnl_percentage(self) -> float:
        if self.entry_price > 0:
            return ((self.mark_price - self.entry_price) / self.entry_price) * 100
        return 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.mark_price


class Signal(BaseModel):
    source: str  # Agent that generated the signal
    symbol: str
    action: OrderSide
    strength: float = Field(ge=-1.0, le=1.0)  # -1 (strong sell) to 1 (strong buy)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskMetrics(BaseModel):
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float  # Value at Risk (95% confidence)
    margin_usage: float
    leverage_ratio: float
    daily_pnl: float
    total_pnl: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StrategyState(BaseModel):
    strategy_id: str
    is_active: bool
    positions: List[Position] = Field(default_factory=list)
    pending_orders: List[Order] = Field(default_factory=list)
    signals: List[Signal] = Field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_update: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class TradingState:
    market_data: Dict[str, MarketData] = field(default_factory=dict)
    order_books: Dict[str, OrderBook] = field(default_factory=dict)
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    strategies: Dict[str, StrategyState] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)