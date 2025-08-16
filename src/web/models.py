"""Web API 数据模型"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from decimal import Decimal
from enum import Enum


class TradingMode(str, Enum):
    """交易模式"""
    PAPER = "paper"
    LIVE = "live"


class StrategyStatus(str, Enum):
    """策略状态"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# 响应模型
class PositionResponse(BaseModel):
    """仓位响应模型"""
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    margin: float
    percentage: float
    trading_mode: TradingMode
    last_updated: datetime


class OrderResponse(BaseModel):
    """订单响应模型"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    filled_quantity: float
    status: OrderStatus
    trading_mode: TradingMode
    created_time: datetime
    updated_time: datetime


class PerformanceResponse(BaseModel):
    """性能响应模型"""
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    trading_mode: TradingMode
    period_start: datetime
    period_end: datetime


class SystemHealthResponse(BaseModel):
    """系统健康响应模型"""
    status: str
    uptime_seconds: float
    cpu_usage: float
    memory_usage_mb: float
    disk_usage_gb: float
    network_latency_ms: float
    active_connections: int
    last_check: datetime


class StrategyResponse(BaseModel):
    """策略响应模型"""
    strategy_id: str
    name: str
    status: StrategyStatus
    trading_mode: TradingMode
    config: Dict[str, Any]
    performance: Dict[str, float]
    created_time: datetime
    last_updated: datetime


class AgentStatusResponse(BaseModel):
    """Agent状态响应模型"""
    agent_id: str
    name: str
    status: str
    health_score: float
    last_activity: datetime
    metrics: Dict[str, float]


class MarketDataResponse(BaseModel):
    """市场数据响应模型"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    change_24h: float
    timestamp: datetime


# 请求模型
class StrategyControlRequest(BaseModel):
    """策略控制请求模型"""
    action: str = Field(..., description="操作类型: start, stop, pause, resume")
    trading_mode: Optional[TradingMode] = Field(default=TradingMode.PAPER, description="交易模式")
    config: Optional[Dict[str, Any]] = Field(default=None, description="策略配置")


class OrderRequest(BaseModel):
    """下单请求模型"""
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    trading_mode: TradingMode = TradingMode.PAPER


class SystemConfigRequest(BaseModel):
    """系统配置请求模型"""
    config_key: str
    config_value: Any
    trading_mode: Optional[TradingMode] = None


# WebSocket 消息模型
class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    type: str
    timestamp: datetime
    data: Dict[str, Any]


class MarketDataMessage(WebSocketMessage):
    """市场数据消息"""
    type: str = "market_data"
    symbol: str
    price: float
    volume: float


class OrderUpdateMessage(WebSocketMessage):
    """订单更新消息"""
    type: str = "order_update"
    order_id: str
    status: str
    filled_quantity: float


class SystemEventMessage(WebSocketMessage):
    """系统事件消息"""
    type: str = "system_event"
    event_type: str
    severity: str
    message: str


class PerformanceUpdateMessage(WebSocketMessage):
    """性能更新消息"""
    type: str = "performance_update"
    trading_mode: TradingMode
    pnl: float
    trades: int


# 批量操作模型
class BatchOrderRequest(BaseModel):
    """批量下单请求"""
    orders: List[OrderRequest]
    trading_mode: TradingMode = TradingMode.PAPER


class BatchOrderResponse(BaseModel):
    """批量下单响应"""
    success_count: int
    failed_count: int
    orders: List[OrderResponse]
    errors: List[str]


# 错误响应模型
class ErrorResponse(BaseModel):
    """错误响应模型"""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# API元数据模型
class APIInfoResponse(BaseModel):
    """API信息响应"""
    version: str
    name: str
    description: str
    docs_url: str
    health_check_url: str
    websocket_url: str
    supported_trading_modes: List[TradingMode]
    supported_exchanges: List[str]