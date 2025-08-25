"""
WebSocket实时推送系统

提供高性能的WebSocket连接管理和实时数据推送服务。

主要模块:
- websocket_manager: 核心WebSocket管理器
- connection_manager: 连接生命周期管理
- subscription_manager: 订阅和主题管理
- message_broadcaster: 消息广播和路由
- models: WebSocket数据模型

特性:
- 高性能WebSocket连接管理
- 多种订阅类型支持
- 实时数据推送（延迟<100ms）
- 连接池管理和负载均衡
- 自动重连和故障转移
- 用户认证和权限控制
"""

from .websocket_manager import WebSocketManager
from .connection_manager import ConnectionManager
from .subscription_manager import SubscriptionManager
from .message_broadcaster import MessageBroadcaster
from .models import (
    ConnectionInfo,
    SubscriptionInfo,
    MessageType,
    WebSocketMessage,
    BroadcastMessage,
    ConnectionStatus,
    SubscriptionType,
    WebSocketConfig
)

__all__ = [
    'WebSocketManager',
    'ConnectionManager', 
    'SubscriptionManager',
    'MessageBroadcaster',
    'ConnectionInfo',
    'SubscriptionInfo',
    'MessageType',
    'WebSocketMessage',
    'BroadcastMessage',
    'ConnectionStatus',
    'SubscriptionType',
    'WebSocketConfig'
]