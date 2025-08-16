"""
标准消息格式定义
定义系统中各种消息类型的数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import time

from src.core.models import OrderSide, OrderStatus, OrderType


class MessageType(Enum):
    """消息类型枚举"""
    # 市场数据
    MARKET_TICK = "market.tick"
    MARKET_ORDERBOOK = "market.orderbook"
    MARKET_KLINE = "market.kline"
    MARKET_TRADE = "market.trade"
    
    # 技术分析信号
    SIGNAL_TECHNICAL = "signal.technical"
    SIGNAL_SENTIMENT = "signal.sentiment"
    SIGNAL_FUNDAMENTAL = "signal.fundamental"
    
    # 交易相关
    ORDER_NEW = "order.new"
    ORDER_UPDATE = "order.update"
    ORDER_CANCEL = "order.cancel"
    ORDER_FILL = "order.fill"
    
    # 仓位管理
    POSITION_UPDATE = "position.update"
    POSITION_CLOSE = "position.close"
    
    # 风险管理
    RISK_ALERT = "risk.alert"
    RISK_UPDATE = "risk.update"
    
    # 系统事件
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_HEALTH = "system.health"
    
    # Agent通信
    AGENT_REGISTER = "agent.register"
    AGENT_UNREGISTER = "agent.unregister"
    AGENT_STATUS = "agent.status"
    AGENT_CONFIG = "agent.config"


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"


class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketTickMessage:
    """市场Tick数据消息"""
    symbol: str
    price: float
    volume: float
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    timestamp: float = field(default_factory=time.time)
    exchange: str = "binance"
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"market.{self.symbol.lower()}.tick"


@dataclass
class MarketOrderBookMessage:
    """订单簿消息"""
    symbol: str
    bids: List[List[float]]  # [[price, volume], ...]
    asks: List[List[float]]  # [[price, volume], ...]
    last_update_id: int
    timestamp: float = field(default_factory=time.time)
    exchange: str = "binance"
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"market.{self.symbol.lower()}.orderbook"


@dataclass
class MarketKlineMessage:
    """K线数据消息"""
    symbol: str
    interval: str  # 1m, 5m, 1h, 1d等
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    open_time: int
    close_time: int
    timestamp: float = field(default_factory=time.time)
    exchange: str = "binance"
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"market.{self.symbol.lower()}.kline.{self.interval}"


@dataclass
class TechnicalSignalMessage:
    """技术分析信号消息"""
    symbol: str
    indicator: str  # RSI, MACD, MA等
    signal_type: SignalType
    strength: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    price: float
    value: float  # 指标值
    timeframe: str = "1m"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"signal.technical.{self.indicator.lower()}.{self.symbol.lower()}"


@dataclass
class SentimentSignalMessage:
    """情绪分析信号消息"""
    symbol: str
    source: str  # twitter, reddit, news等
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0-1.0
    signal_type: SignalType
    timestamp: float = field(default_factory=time.time)
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"signal.sentiment.{self.source}.{self.symbol.lower()}"


@dataclass
class OrderMessage:
    """订单消息"""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"order.{self.status.value.lower()}.{self.symbol.lower()}"


@dataclass
class PositionMessage:
    """仓位消息"""
    symbol: str
    side: OrderSide  # LONG/SHORT
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    margin: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"position.update.{self.symbol.lower()}"


@dataclass
class RiskAlertMessage:
    """风险警告消息"""
    risk_type: str  # drawdown, exposure, volatility等
    level: RiskLevel
    symbol: Optional[str]
    current_value: float
    threshold: float
    description: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        symbol_part = f".{self.symbol.lower()}" if self.symbol else ""
        return f"risk.alert.{self.level.value}{symbol_part}"


@dataclass
class SystemMessage:
    """系统消息"""
    event_type: str  # start, stop, error, health
    component: str  # agent名称或组件名称
    status: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"system.{self.event_type}.{self.component}"


@dataclass
class AgentMessage:
    """Agent通信消息"""
    agent_id: str
    event_type: str  # register, unregister, status, config
    agent_type: str  # technical, sentiment, risk, execution等
    status: str = "active"
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_topic(self) -> str:
        """生成消息主题"""
        return f"agent.{self.event_type}.{self.agent_type}"


class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.routing_rules: Dict[str, List[str]] = {}
        self.filters: Dict[str, Callable[[Any], bool]] = {}
    
    def add_route(self, from_topic: str, to_topics: List[str]):
        """添加路由规则"""
        self.routing_rules[from_topic] = to_topics
    
    def add_filter(self, topic_pattern: str, filter_func: Callable[[Any], bool]):
        """添加消息过滤器"""
        self.filters[topic_pattern] = filter_func
    
    def route_message(self, message: Any) -> List[str]:
        """路由消息，返回目标主题列表"""
        if hasattr(message, 'to_topic'):
            source_topic = message.to_topic()
        else:
            return []
        
        # 检查路由规则
        target_topics = []
        for pattern, targets in self.routing_rules.items():
            if self._match_pattern(source_topic, pattern):
                target_topics.extend(targets)
        
        # 应用过滤器
        filtered_topics = []
        for topic in target_topics:
            for pattern, filter_func in self.filters.items():
                if self._match_pattern(topic, pattern):
                    if filter_func(message):
                        filtered_topics.append(topic)
                    break
            else:
                filtered_topics.append(topic)
        
        return filtered_topics
    
    def _match_pattern(self, topic: str, pattern: str) -> bool:
        """匹配主题模式（支持通配符）"""
        if pattern == "*":
            return True
        
        if "*" not in pattern:
            return topic == pattern
        
        # 简单的通配符匹配
        pattern_parts = pattern.split("*")
        if len(pattern_parts) == 2:
            prefix, suffix = pattern_parts
            return topic.startswith(prefix) and topic.endswith(suffix)
        
        return False


class TopicBuilder:
    """主题构建器"""
    
    @staticmethod
    def market_tick(symbol: str) -> str:
        """市场tick数据主题"""
        return f"market.{symbol.lower()}.tick"
    
    @staticmethod
    def market_orderbook(symbol: str) -> str:
        """订单簿主题"""
        return f"market.{symbol.lower()}.orderbook"
    
    @staticmethod
    def market_kline(symbol: str, interval: str) -> str:
        """K线数据主题"""
        return f"market.{symbol.lower()}.kline.{interval}"
    
    @staticmethod
    def technical_signal(indicator: str, symbol: str) -> str:
        """技术信号主题"""
        return f"signal.technical.{indicator.lower()}.{symbol.lower()}"
    
    @staticmethod
    def sentiment_signal(source: str, symbol: str) -> str:
        """情绪信号主题"""
        return f"signal.sentiment.{source}.{symbol.lower()}"
    
    @staticmethod
    def order_event(status: str, symbol: str) -> str:
        """订单事件主题"""
        return f"order.{status}.{symbol.lower()}"
    
    @staticmethod
    def position_update(symbol: str) -> str:
        """仓位更新主题"""
        return f"position.update.{symbol.lower()}"
    
    @staticmethod
    def risk_alert(level: str, symbol: str = None) -> str:
        """风险警告主题"""
        symbol_part = f".{symbol.lower()}" if symbol else ""
        return f"risk.alert.{level}{symbol_part}"
    
    @staticmethod
    def system_event(event_type: str, component: str) -> str:
        """系统事件主题"""
        return f"system.{event_type}.{component}"
    
    @staticmethod
    def agent_event(event_type: str, agent_type: str) -> str:
        """Agent事件主题"""
        return f"agent.{event_type}.{agent_type}"


# 全局路由器实例
message_router = MessageRouter()

# 默认路由规则
message_router.add_route("market.*", ["storage.market", "analysis.technical"])
message_router.add_route("signal.*", ["decision.aggregator", "risk.monitor"])
message_router.add_route("order.*", ["storage.orders", "risk.monitor"])
message_router.add_route("risk.alert.*", ["notification.system", "decision.override"])
message_router.add_route("system.*", ["monitoring.system", "logging.system"])