"""
连接管理器

负责WebSocket连接的生命周期管理，包括连接建立、
认证、心跳检测、连接清理等功能。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
import weakref
import json
from uuid import uuid4

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .models import (
    ConnectionInfo, 
    ConnectionStatus, 
    WebSocketMessage, 
    MessageType,
    ConnectionStats,
    WebSocketConfig,
    ReconnectionConfig
)


logger = logging.getLogger(__name__)


class ConnectionManager:
    """连接管理器
    
    管理所有WebSocket连接的生命周期，包括：
    - 连接建立和断开
    - 连接认证和权限管理
    - 心跳检测和连接保活
    - 连接状态监控和统计
    - 连接清理和资源回收
    """
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self._connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._connection_info: Dict[str, ConnectionInfo] = {}
        self._stats = ConnectionStats()
        self._auth_handler: Optional[Callable] = None
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 连接事件回调
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_authenticate_callbacks: List[Callable] = []
        
        logger.info("连接管理器初始化完成")
    
    async def start(self) -> None:
        """启动连接管理器"""
        logger.info("启动连接管理器...")
        
        # 启动心跳任务
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("连接管理器启动成功")
    
    async def stop(self) -> None:
        """停止连接管理器"""
        logger.info("停止连接管理器...")
        
        # 设置停止事件
        self._shutdown_event.set()
        
        # 取消后台任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # 关闭所有连接
        await self._close_all_connections()
        
        logger.info("连接管理器已停止")
    
    def set_auth_handler(self, handler: Callable[[str, Dict[str, Any]], bool]) -> None:
        """设置认证处理器
        
        Args:
            handler: 认证处理函数，接受连接ID和认证数据，返回是否认证成功
        """
        self._auth_handler = handler
    
    def add_message_handler(self, message_type: MessageType, handler: Callable) -> None:
        """添加消息处理器"""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
    
    def add_connect_callback(self, callback: Callable[[str], None]) -> None:
        """添加连接建立回调"""
        self._on_connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[str], None]) -> None:
        """添加连接断开回调"""
        self._on_disconnect_callbacks.append(callback)
    
    def add_authenticate_callback(self, callback: Callable[[str, str], None]) -> None:
        """添加认证成功回调"""
        self._on_authenticate_callbacks.append(callback)
    
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """处理新的WebSocket连接"""
        connection_id = str(uuid4())
        
        # 检查连接数限制
        if len(self._connections) >= self.config.max_connections:
            logger.warning(f"连接数达到上限 {self.config.max_connections}，拒绝新连接")
            await websocket.close(code=1013, reason="服务器忙")
            return
        
        # 创建连接信息
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            client_ip=websocket.remote_address[0] if websocket.remote_address else None,
            user_agent=websocket.request_headers.get('User-Agent'),
            status=ConnectionStatus.CONNECTED
        )
        
        # 注册连接
        self._connections[connection_id] = websocket
        self._connection_info[connection_id] = connection_info
        self._stats.total_connections += 1
        self._stats.active_connections += 1
        self._stats.update_peak_connections(self._stats.active_connections)
        
        logger.info(f"新连接建立: {connection_id} from {connection_info.client_ip}")
        
        # 触发连接回调
        for callback in self._on_connect_callbacks:
            try:
                await callback(connection_id)
            except Exception as e:
                logger.error(f"连接回调执行失败: {e}")
        
        try:
            # 发送欢迎消息
            welcome_msg = WebSocketMessage.create_control_message(
                MessageType.PING,
                {"connection_id": connection_id, "message": "连接成功"}
            )
            await self.send_message(connection_id, welcome_msg)
            
            # 处理消息
            async for message in websocket:
                await self._handle_message(connection_id, message)
                
        except ConnectionClosed:
            logger.info(f"连接 {connection_id} 正常关闭")
        except WebSocketException as e:
            logger.warning(f"连接 {connection_id} WebSocket异常: {e}")
            self._stats.connection_errors += 1
        except Exception as e:
            logger.error(f"连接 {connection_id} 处理异常: {e}")
            self._stats.connection_errors += 1
        finally:
            await self._cleanup_connection(connection_id)
    
    async def send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """发送消息给指定连接
        
        Args:
            connection_id: 连接ID
            message: 要发送的消息
            
        Returns:
            是否发送成功
        """
        websocket = self._connections.get(connection_id)
        if not websocket:
            logger.warning(f"连接 {connection_id} 不存在")
            return False
        
        try:
            message_json = message.to_json()
            await websocket.send(message_json)
            
            # 更新统计
            self._stats.messages_sent += 1
            self._stats.bytes_sent += len(message_json.encode('utf-8'))
            
            # 更新连接活动时间
            if connection_id in self._connection_info:
                self._connection_info[connection_id].update_activity()
            
            return True
            
        except ConnectionClosed:
            logger.warning(f"连接 {connection_id} 已关闭，无法发送消息")
            await self._cleanup_connection(connection_id)
            return False
        except Exception as e:
            logger.error(f"发送消息到 {connection_id} 失败: {e}")
            return False
    
    async def broadcast_message(self, message: WebSocketMessage, 
                              target_connections: Optional[Set[str]] = None) -> int:
        """广播消息给指定或所有连接
        
        Args:
            message: 要广播的消息
            target_connections: 目标连接ID集合，None表示所有连接
            
        Returns:
            成功发送的连接数
        """
        if target_connections is None:
            target_connections = set(self._connections.keys())
        
        success_count = 0
        tasks = []
        
        for connection_id in target_connections:
            if connection_id in self._connections:
                tasks.append(self.send_message(connection_id, message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
        
        logger.debug(f"广播消息给 {len(tasks)} 个连接，成功 {success_count} 个")
        return success_count
    
    async def close_connection(self, connection_id: str, code: int = 1000, reason: str = "正常关闭") -> None:
        """关闭指定连接"""
        websocket = self._connections.get(connection_id)
        if websocket:
            try:
                await websocket.close(code=code, reason=reason)
            except Exception as e:
                logger.error(f"关闭连接 {connection_id} 异常: {e}")
        
        await self._cleanup_connection(connection_id)
    
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """获取连接信息"""
        return self._connection_info.get(connection_id)
    
    def get_active_connections(self) -> List[str]:
        """获取活跃连接列表"""
        return list(self._connections.keys())
    
    def get_connection_count(self) -> int:
        """获取当前连接数"""
        return len(self._connections)
    
    def get_stats(self) -> ConnectionStats:
        """获取连接统计"""
        self._stats.active_connections = len(self._connections)
        return self._stats
    
    async def _handle_message(self, connection_id: str, message_data: str) -> None:
        """处理收到的消息"""
        try:
            # 解析消息
            message = WebSocketMessage.from_json(message_data)
            message.connection_id = connection_id
            
            # 更新统计
            self._stats.messages_received += 1
            self._stats.bytes_received += len(message_data.encode('utf-8'))
            
            # 更新连接活动时间
            if connection_id in self._connection_info:
                self._connection_info[connection_id].update_activity()
            
            logger.debug(f"收到消息从 {connection_id}: {message.message_type.value}")
            
            # 处理特殊消息类型
            if message.message_type == MessageType.PING:
                await self._handle_ping(connection_id, message)
            elif message.message_type == MessageType.AUTH:
                await self._handle_auth(connection_id, message)
            else:
                # 转发给注册的处理器
                handlers = self._message_handlers.get(message.message_type, [])
                for handler in handlers:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"消息处理器异常: {e}")
            
        except json.JSONDecodeError as e:
            logger.error(f"消息解析失败: {e}")
            error_msg = WebSocketMessage.create_control_message(
                MessageType.ERROR,
                {"error": "消息格式错误"}
            )
            await self.send_message(connection_id, error_msg)
        except Exception as e:
            logger.error(f"处理消息异常: {e}")
    
    async def _handle_ping(self, connection_id: str, message: WebSocketMessage) -> None:
        """处理PING消息"""
        pong_msg = WebSocketMessage.create_control_message(
            MessageType.PONG,
            {"timestamp": datetime.now().isoformat()}
        )
        await self.send_message(connection_id, pong_msg)
    
    async def _handle_auth(self, connection_id: str, message: WebSocketMessage) -> None:
        """处理认证消息"""
        if not self._auth_handler:
            logger.warning(f"连接 {connection_id} 尝试认证但未设置认证处理器")
            return
        
        try:
            # 执行认证
            auth_result = await self._auth_handler(connection_id, message.data)
            
            connection_info = self._connection_info.get(connection_id)
            if connection_info:
                if auth_result:
                    connection_info.status = ConnectionStatus.AUTHENTICATED
                    connection_info.authenticated_at = datetime.now()
                    connection_info.user_id = message.data.get('user_id')
                    
                    # 触发认证成功回调
                    for callback in self._on_authenticate_callbacks:
                        try:
                            await callback(connection_id, connection_info.user_id or "")
                        except Exception as e:
                            logger.error(f"认证回调执行失败: {e}")
                    
                    # 发送认证成功消息
                    auth_msg = WebSocketMessage.create_control_message(
                        MessageType.AUTH,
                        {"success": True, "user_id": connection_info.user_id}
                    )
                    await self.send_message(connection_id, auth_msg)
                    
                    logger.info(f"连接 {connection_id} 认证成功，用户: {connection_info.user_id}")
                else:
                    # 认证失败
                    error_msg = WebSocketMessage.create_control_message(
                        MessageType.ERROR,
                        {"error": "认证失败"}
                    )
                    await self.send_message(connection_id, error_msg)
                    
                    # 关闭连接
                    await self.close_connection(connection_id, code=1008, reason="认证失败")
                    
        except Exception as e:
            logger.error(f"认证处理异常: {e}")
            error_msg = WebSocketMessage.create_control_message(
                MessageType.ERROR,
                {"error": "认证处理异常"}
            )
            await self.send_message(connection_id, error_msg)
    
    async def _heartbeat_loop(self) -> None:
        """心跳检测循环"""
        logger.info("启动心跳检测任务")
        
        while not self._shutdown_event.is_set():
            try:
                # 发送心跳消息
                ping_msg = WebSocketMessage.create_control_message(
                    MessageType.PING,
                    {"timestamp": datetime.now().isoformat()}
                )
                
                # 检查所有连接的活跃状态
                inactive_connections = []
                for connection_id, connection_info in self._connection_info.items():
                    if not connection_info.is_active(self.config.connection_timeout):
                        inactive_connections.append(connection_id)
                    else:
                        # 发送心跳
                        await self.send_message(connection_id, ping_msg)
                
                # 清理非活跃连接
                for connection_id in inactive_connections:
                    logger.info(f"清理非活跃连接: {connection_id}")
                    await self.close_connection(connection_id, code=1001, reason="连接超时")
                
                # 等待下一个心跳周期
                await asyncio.sleep(self.config.ping_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳检测异常: {e}")
                await asyncio.sleep(1)
        
        logger.info("心跳检测任务已停止")
    
    async def _cleanup_loop(self) -> None:
        """清理任务循环"""
        logger.info("启动连接清理任务")
        
        while not self._shutdown_event.is_set():
            try:
                # 定期清理统计数据和日志
                await asyncio.sleep(60)  # 每分钟执行一次
                
                # 可以在这里添加其他清理逻辑
                logger.debug(f"当前连接数: {len(self._connections)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理任务异常: {e}")
        
        logger.info("连接清理任务已停止")
    
    async def _cleanup_connection(self, connection_id: str) -> None:
        """清理连接资源"""
        # 移除连接
        if connection_id in self._connections:
            del self._connections[connection_id]
        
        # 触发断开回调
        if connection_id in self._connection_info:
            for callback in self._on_disconnect_callbacks:
                try:
                    await callback(connection_id)
                except Exception as e:
                    logger.error(f"断开回调执行失败: {e}")
            
            del self._connection_info[connection_id]
        
        # 更新统计
        if self._stats.active_connections > 0:
            self._stats.active_connections -= 1
        
        logger.debug(f"连接 {connection_id} 资源已清理")
    
    async def _close_all_connections(self) -> None:
        """关闭所有连接"""
        logger.info(f"关闭所有连接 ({len(self._connections)} 个)")
        
        tasks = []
        for connection_id in list(self._connections.keys()):
            tasks.append(self.close_connection(connection_id, code=1001, reason="服务器关闭"))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("所有连接已关闭")