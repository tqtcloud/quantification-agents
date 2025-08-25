"""
WebSocket管理器

核心WebSocket管理器，整合连接管理、订阅管理和消息广播功能，
提供统一的WebSocket服务接口。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Union
import json
import weakref
import signal

import websockets
from websockets.server import WebSocketServerProtocol, serve
from websockets.exceptions import ConnectionClosed, WebSocketException

from .connection_manager import ConnectionManager
from .subscription_manager import SubscriptionManager
from .message_broadcaster import MessageBroadcaster, BroadcastStrategy
from .models import (
    WebSocketConfig,
    WebSocketMessage,
    MessageType,
    SubscriptionType,
    ConnectionInfo,
    ConnectionStatus,
    BroadcastMessage
)


logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket管理器
    
    整合所有WebSocket功能的核心管理器：
    - 连接管理和生命周期
    - 订阅管理和主题路由
    - 消息广播和推送
    - 认证和权限控制
    - 性能监控和统计
    - 故障恢复和重连
    """
    
    def __init__(self, config: Optional[WebSocketConfig] = None, auth_manager=None):
        self.config = config or WebSocketConfig()
        self.auth_manager = auth_manager
        
        # 核心组件
        self.connection_manager = ConnectionManager(self.config)
        self.subscription_manager = SubscriptionManager()
        self.message_broadcaster = MessageBroadcaster(
            connection_manager=self.connection_manager,
            subscription_manager=self.subscription_manager
        )
        
        # WebSocket服务器
        self._server: Optional[websockets.WebSocketServer] = None
        self._server_task: Optional[asyncio.Task] = None
        
        # 状态管理
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # 外部集成
        self._trading_api = None
        self._strategy_manager = None
        self._signal_aggregator = None
        
        # 事件回调
        self._startup_callbacks: List[Callable] = []
        self._shutdown_callbacks: List[Callable] = []
        
        # 设置组件回调
        self._setup_callbacks()
        
        logger.info("WebSocket管理器初始化完成")
    
    def _setup_callbacks(self) -> None:
        """设置内部组件回调"""
        # 连接事件回调
        self.connection_manager.add_connect_callback(self._on_connection_established)
        self.connection_manager.add_disconnect_callback(self._on_connection_closed)
        self.connection_manager.add_authenticate_callback(self._on_connection_authenticated)
        
        # 订阅事件回调
        self.subscription_manager.add_subscribe_callback(self._on_subscription_created)
        self.subscription_manager.add_unsubscribe_callback(self._on_subscription_removed)
        
        # 消息处理器
        self.connection_manager.add_message_handler(MessageType.SUBSCRIBE, self._handle_subscribe)
        self.connection_manager.add_message_handler(MessageType.UNSUBSCRIBE, self._handle_unsubscribe)
        
        # 认证处理器
        if self.auth_manager:
            self.connection_manager.set_auth_handler(self._authenticate_connection)
        
        # 权限检查器
        self.subscription_manager.set_permission_checker(self._check_subscription_permission)
    
    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """启动WebSocket服务器
        
        Args:
            host: 绑定主机地址
            port: 绑定端口
        """
        if self._running:
            logger.warning("WebSocket服务器已在运行")
            return
        
        # 验证配置
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"配置错误: {', '.join(config_errors)}")
        
        # 使用参数或配置中的地址
        server_host = host or self.config.host
        server_port = port or self.config.port
        
        logger.info(f"启动WebSocket服务器 {server_host}:{server_port}")
        
        try:
            # 启动核心组件
            await self.connection_manager.start()
            await self.message_broadcaster.start()
            
            # 配置服务器参数
            server_kwargs = {
                'compression': 'deflate' if self.config.compression_enabled else None,
                'ping_interval': self.config.ping_interval,
                'ping_timeout': self.config.ping_timeout,
                'close_timeout': self.config.close_timeout,
                'max_size': self.config.max_message_size,
                'max_queue': self.config.message_queue_size
            }
            
            # 启动WebSocket服务器
            self._server = await serve(
                self.connection_manager.handle_connection,
                server_host,
                server_port,
                **server_kwargs
            )
            
            self._running = True
            
            # 启动监控任务
            self._server_task = asyncio.create_task(self._monitor_server())
            
            # 触发启动回调
            for callback in self._startup_callbacks:
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"启动回调执行失败: {e}")
            
            logger.info(f"WebSocket服务器启动成功，监听 {server_host}:{server_port}")
            
            # 设置广播策略
            self._configure_broadcast_strategies()
            
            # 注册信号处理
            self._setup_signal_handlers()
            
        except Exception as e:
            logger.error(f"启动WebSocket服务器失败: {e}")
            await self._cleanup()
            raise
    
    async def stop(self) -> None:
        """停止WebSocket服务器"""
        if not self._running:
            logger.warning("WebSocket服务器未运行")
            return
        
        logger.info("停止WebSocket服务器...")
        
        # 设置停止标志
        self._running = False
        self._shutdown_event.set()
        
        try:
            # 触发关闭回调
            for callback in self._shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    logger.error(f"关闭回调执行失败: {e}")
            
            # 停止服务器
            if self._server:
                self._server.close()
                await self._server.wait_closed()
            
            # 取消监控任务
            if self._server_task:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
            
            # 停止核心组件
            await self.message_broadcaster.stop()
            await self.connection_manager.stop()
            
        except Exception as e:
            logger.error(f"停止WebSocket服务器异常: {e}")
        finally:
            await self._cleanup()
        
        logger.info("WebSocket服务器已停止")
    
    async def wait_until_stopped(self) -> None:
        """等待服务器停止"""
        await self._shutdown_event.wait()
    
    def set_trading_api(self, trading_api) -> None:
        """设置交易API接口"""
        self._trading_api = trading_api
        logger.info("设置交易API接口")
    
    def set_strategy_manager(self, strategy_manager) -> None:
        """设置策略管理器"""
        self._strategy_manager = strategy_manager
        logger.info("设置策略管理器")
    
    def set_signal_aggregator(self, signal_aggregator) -> None:
        """设置信号聚合器"""
        self._signal_aggregator = signal_aggregator
        logger.info("设置信号聚合器")
    
    def add_startup_callback(self, callback: Callable) -> None:
        """添加启动回调"""
        self._startup_callbacks.append(callback)
    
    def add_shutdown_callback(self, callback: Callable) -> None:
        """添加关闭回调"""
        self._shutdown_callbacks.append(callback)
    
    async def broadcast_trading_signal(self, signal_data: Dict[str, Any]) -> int:
        """广播交易信号
        
        Args:
            signal_data: 交易信号数据
            
        Returns:
            成功发送的连接数
        """
        message = WebSocketMessage.create_data_message(
            MessageType.TRADING_SIGNAL,
            {
                'timestamp': datetime.now().isoformat(),
                'signal': signal_data,
                'source': 'signal_aggregator'
            }
        )
        
        return await self.message_broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.TRADING_SIGNALS,
            priority=1  # 交易信号为高优先级
        )
    
    async def broadcast_strategy_status(self, strategy_id: str, status_data: Dict[str, Any]) -> int:
        """广播策略状态
        
        Args:
            strategy_id: 策略ID
            status_data: 状态数据
            
        Returns:
            成功发送的连接数
        """
        message = WebSocketMessage.create_data_message(
            MessageType.STRATEGY_STATUS,
            {
                'timestamp': datetime.now().isoformat(),
                'strategy_id': strategy_id,
                'status': status_data,
                'source': 'strategy_manager'
            }
        )
        
        return await self.message_broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.STRATEGY_STATUS
        )
    
    async def broadcast_market_data(self, market_data: Dict[str, Any]) -> int:
        """广播市场数据
        
        Args:
            market_data: 市场数据
            
        Returns:
            成功发送的连接数
        """
        message = WebSocketMessage.create_data_message(
            MessageType.MARKET_DATA,
            {
                'timestamp': datetime.now().isoformat(),
                'data': market_data,
                'source': 'market_data'
            }
        )
        
        return await self.message_broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.MARKET_DATA
        )
    
    async def broadcast_risk_alert(self, alert_data: Dict[str, Any]) -> int:
        """广播风险警报
        
        Args:
            alert_data: 风险警报数据
            
        Returns:
            成功发送的连接数
        """
        message = WebSocketMessage.create_data_message(
            MessageType.RISK_ALERT,
            {
                'timestamp': datetime.now().isoformat(),
                'alert': alert_data,
                'source': 'risk_manager',
                'severity': alert_data.get('severity', 'medium')
            }
        )
        
        return await self.message_broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.RISK_ALERTS,
            priority=2  # 风险警报为紧急优先级
        )
    
    async def broadcast_system_monitor(self, monitor_data: Dict[str, Any]) -> int:
        """广播系统监控数据
        
        Args:
            monitor_data: 监控数据
            
        Returns:
            成功发送的连接数
        """
        message = WebSocketMessage.create_data_message(
            MessageType.SYSTEM_MONITOR,
            {
                'timestamp': datetime.now().isoformat(),
                'metrics': monitor_data,
                'source': 'system_monitor'
            }
        )
        
        return await self.message_broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.SYSTEM_MONITOR
        )
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return self.connection_manager.get_connection_count()
    
    def get_subscription_stats(self) -> Dict:
        """获取订阅统计"""
        return self.subscription_manager.get_subscription_stats()
    
    def get_broadcast_stats(self) -> Dict:
        """获取广播统计"""
        return self.message_broadcaster.get_stats()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'server_status': 'running' if self._running else 'stopped',
            'uptime': (datetime.now() - datetime.now()).total_seconds() if self._running else 0,
            'connections': self.connection_manager.get_stats(),
            'subscriptions': self.subscription_manager.get_subscription_stats(),
            'broadcasts': self.message_broadcaster.get_stats(),
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'max_connections': self.config.max_connections,
                'compression_enabled': self.config.compression_enabled
            }
        }
    
    async def _on_connection_established(self, connection_id: str) -> None:
        """连接建立回调"""
        logger.debug(f"连接建立: {connection_id}")
    
    async def _on_connection_closed(self, connection_id: str) -> None:
        """连接关闭回调"""
        logger.debug(f"连接关闭: {connection_id}")
        
        # 清理订阅
        await self.subscription_manager.unsubscribe_all(connection_id)
        
        # 清理广播数据
        self.message_broadcaster.clear_connection_data(connection_id)
    
    async def _on_connection_authenticated(self, connection_id: str, user_id: str) -> None:
        """连接认证成功回调"""
        logger.info(f"连接认证成功: {connection_id}, 用户: {user_id}")
    
    async def _on_subscription_created(self, connection_id: str, subscription_info) -> None:
        """订阅创建回调"""
        logger.debug(f"创建订阅: {connection_id} -> {subscription_info.subscription_type.value}")
    
    async def _on_subscription_removed(self, connection_id: str, subscription_id: str) -> None:
        """订阅移除回调"""
        logger.debug(f"移除订阅: {connection_id} -> {subscription_id}")
    
    async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage) -> None:
        """处理订阅消息"""
        response = await self.subscription_manager.handle_subscribe_message(connection_id, message)
        await self.connection_manager.send_message(connection_id, response)
    
    async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage) -> None:
        """处理取消订阅消息"""
        response = await self.subscription_manager.handle_unsubscribe_message(connection_id, message)
        await self.connection_manager.send_message(connection_id, response)
    
    async def _authenticate_connection(self, connection_id: str, auth_data: Dict[str, Any]) -> bool:
        """认证连接"""
        if not self.auth_manager:
            logger.warning("认证管理器未设置，跳过认证")
            return True
        
        try:
            # 使用认证管理器验证
            token = auth_data.get('token')
            if not token:
                logger.warning(f"连接 {connection_id} 缺少认证令牌")
                return False
            
            # 验证令牌（这里需要根据实际的auth_manager接口调整）
            # 假设auth_manager有verify_token方法
            if hasattr(self.auth_manager, 'verify_token'):
                user_info = await self.auth_manager.verify_token(token)
                if user_info:
                    logger.info(f"连接 {connection_id} 认证成功，用户: {user_info.get('user_id')}")
                    return True
            
            logger.warning(f"连接 {connection_id} 认证失败")
            return False
            
        except Exception as e:
            logger.error(f"认证异常: {e}")
            return False
    
    async def _check_subscription_permission(self, connection_id: str, 
                                           subscription_type: SubscriptionType, 
                                           filters: Dict[str, Any]) -> bool:
        """检查订阅权限"""
        # 获取连接信息
        connection_info = self.connection_manager.get_connection_info(connection_id)
        if not connection_info:
            return False
        
        # 检查认证状态
        if self.config.auth_required and connection_info.status != ConnectionStatus.AUTHENTICATED:
            return False
        
        # 根据订阅类型和用户权限检查
        # 这里可以根据实际需求实现更复杂的权限逻辑
        return True
    
    def _configure_broadcast_strategies(self) -> None:
        """配置广播策略"""
        # 交易信号使用优先级队列
        self.message_broadcaster.set_strategy(
            SubscriptionType.TRADING_SIGNALS, 
            BroadcastStrategy.PRIORITY_QUEUE
        )
        
        # 风险警报立即发送
        self.message_broadcaster.set_strategy(
            SubscriptionType.RISK_ALERTS, 
            BroadcastStrategy.IMMEDIATE
        )
        
        # 市场数据使用批量发送
        self.message_broadcaster.set_strategy(
            SubscriptionType.MARKET_DATA, 
            BroadcastStrategy.BATCHED
        )
        
        # 系统监控使用限流发送
        self.message_broadcaster.set_strategy(
            SubscriptionType.SYSTEM_MONITOR, 
            BroadcastStrategy.RATE_LIMITED
        )
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，开始优雅关闭")
            asyncio.create_task(self.stop())
        
        # 注册信号处理
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _monitor_server(self) -> None:
        """监控服务器状态"""
        logger.info("启动服务器监控任务")
        
        while self._running and not self._shutdown_event.is_set():
            try:
                # 定期清理过期订阅
                await self.subscription_manager.cleanup_expired_subscriptions()
                
                # 记录统计信息
                stats = self.get_system_stats()
                logger.debug(f"系统状态: 连接数={stats['connections']['active_connections']}, "
                           f"订阅总数={self.subscription_manager.get_total_subscriptions()}")
                
                # 等待下一次检查
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"服务器监控异常: {e}")
        
        logger.info("服务器监控任务已停止")
    
    async def _cleanup(self) -> None:
        """清理资源"""
        self._server = None
        self._server_task = None
        self._running = False