"""
WebSocket数据模型定义

定义WebSocket系统的核心数据结构，包括连接信息、
订阅管理、消息类型等。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
import json
from uuid import uuid4


class MessageType(Enum):
    """消息类型枚举"""
    # 控制消息
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    ERROR = "error"
    
    # 数据推送
    TRADING_SIGNAL = "trading_signal"
    STRATEGY_STATUS = "strategy_status"
    MARKET_DATA = "market_data"
    SYSTEM_MONITOR = "system_monitor"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_METRICS = "performance_metrics"


class SubscriptionType(Enum):
    """订阅类型枚举"""
    TRADING_SIGNALS = "trading_signals"
    STRATEGY_STATUS = "strategy_status"
    MARKET_DATA = "market_data"
    SYSTEM_MONITOR = "system_monitor"
    ORDER_UPDATES = "order_updates"
    POSITION_UPDATES = "position_updates"
    RISK_ALERTS = "risk_alerts"
    PERFORMANCE_METRICS = "performance_metrics"
    ALL = "all"


class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    authenticated_at: Optional[datetime] = None
    subscriptions: Set[SubscriptionType] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self) -> None:
        """更新最后活动时间"""
        self.last_activity = datetime.now()
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """检查连接是否活跃"""
        return (datetime.now() - self.last_activity).total_seconds() < timeout_seconds
    
    def add_subscription(self, subscription_type: SubscriptionType) -> None:
        """添加订阅"""
        self.subscriptions.add(subscription_type)
    
    def remove_subscription(self, subscription_type: SubscriptionType) -> None:
        """移除订阅"""
        self.subscriptions.discard(subscription_type)


@dataclass
class SubscriptionInfo:
    """订阅信息"""
    subscription_id: str = field(default_factory=lambda: str(uuid4()))
    connection_id: str = ""
    subscription_type: SubscriptionType = SubscriptionType.ALL
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    
    def matches_filter(self, data: Dict[str, Any]) -> bool:
        """检查数据是否匹配过滤条件"""
        if not self.filters:
            return True
        
        for key, value in self.filters.items():
            if key not in data:
                return False
            
            if isinstance(value, list):
                if data[key] not in value:
                    return False
            else:
                if data[key] != value:
                    return False
        
        return True
    
    def update_message_stats(self) -> None:
        """更新消息统计"""
        self.last_message_at = datetime.now()
        self.message_count += 1


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = MessageType.PING
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    connection_id: Optional[str] = None
    subscription_id: Optional[str] = None
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps({
            'message_id': self.message_id,
            'type': self.message_type.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'connection_id': self.connection_id,
            'subscription_id': self.subscription_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """从JSON字符串创建消息"""
        data = json.loads(json_str)
        return cls(
            message_id=data.get('message_id', str(uuid4())),
            message_type=MessageType(data['type']),
            data=data.get('data', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            connection_id=data.get('connection_id'),
            subscription_id=data.get('subscription_id')
        )
    
    @classmethod
    def create_control_message(cls, msg_type: MessageType, data: Dict[str, Any] = None) -> 'WebSocketMessage':
        """创建控制消息"""
        return cls(
            message_type=msg_type,
            data=data or {}
        )
    
    @classmethod
    def create_data_message(cls, msg_type: MessageType, data: Dict[str, Any]) -> 'WebSocketMessage':
        """创建数据消息"""
        return cls(
            message_type=msg_type,
            data=data
        )


@dataclass
class BroadcastMessage:
    """广播消息"""
    message: WebSocketMessage
    target_subscriptions: Set[SubscriptionType] = field(default_factory=set)
    target_connections: Set[str] = field(default_factory=set)
    filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0=普通, 1=高, 2=紧急
    created_at: datetime = field(default_factory=datetime.now)
    
    def should_send_to_connection(self, connection_info: ConnectionInfo, subscription_info: SubscriptionInfo) -> bool:
        """判断是否应该发送给指定连接"""
        # 检查目标连接
        if self.target_connections and connection_info.connection_id not in self.target_connections:
            return False
        
        # 检查订阅类型
        if self.target_subscriptions:
            if not any(sub in connection_info.subscriptions for sub in self.target_subscriptions):
                return False
        
        # 检查过滤条件
        if self.filters and not subscription_info.matches_filter(self.message.data):
            return False
        
        return True


@dataclass
class ConnectionStats:
    """连接统计"""
    total_connections: int = 0
    active_connections: int = 0
    authenticated_connections: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_errors: int = 0
    reconnections: int = 0
    average_latency_ms: float = 0.0
    peak_connections: int = 0
    peak_time: Optional[datetime] = None
    
    def update_peak_connections(self, current_connections: int) -> None:
        """更新峰值连接数"""
        if current_connections > self.peak_connections:
            self.peak_connections = current_connections
            self.peak_time = datetime.now()


@dataclass
class SubscriptionStats:
    """订阅统计"""
    subscription_type: SubscriptionType
    subscriber_count: int = 0
    total_messages: int = 0
    messages_per_second: float = 0.0
    average_message_size: int = 0
    last_message_time: Optional[datetime] = None
    error_count: int = 0


@dataclass
class ReconnectionConfig:
    """重连配置"""
    enabled: bool = True
    max_attempts: int = 5
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter_enabled: bool = True
    
    def get_delay_ms(self, attempt: int) -> int:
        """计算重连延迟"""
        if attempt <= 0:
            return self.initial_delay_ms
        
        delay = min(
            self.initial_delay_ms * (self.backoff_multiplier ** (attempt - 1)),
            self.max_delay_ms
        )
        
        # 添加抖动避免惊群效应
        if self.jitter_enabled:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return int(delay)


@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 1000
    message_queue_size: int = 10000
    ping_interval: int = 30
    ping_timeout: int = 10
    close_timeout: int = 10
    compression_enabled: bool = True
    compression_threshold: int = 1024
    max_message_size: int = 1024 * 1024  # 1MB
    connection_timeout: int = 300
    auth_required: bool = True
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    reconnection: ReconnectionConfig = field(default_factory=ReconnectionConfig)
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.port <= 0 or self.port > 65535:
            errors.append("端口号必须在1-65535范围内")
        
        if self.max_connections <= 0:
            errors.append("最大连接数必须大于0")
        
        if self.message_queue_size <= 0:
            errors.append("消息队列大小必须大于0")
        
        if self.ping_interval <= 0:
            errors.append("心跳间隔必须大于0")
        
        if self.max_message_size <= 0:
            errors.append("最大消息大小必须大于0")
        
        return errors