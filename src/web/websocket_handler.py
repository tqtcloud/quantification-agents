"""WebSocket实时数据推送处理器"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum

from .models import (
    WebSocketMessage, MarketDataMessage, OrderUpdateMessage,
    SystemEventMessage, PerformanceUpdateMessage, TradingMode
)

logger = logging.getLogger(__name__)

class ConnectionType(str, Enum):
    """连接类型"""
    MARKET_DATA = "market_data"
    ORDER_UPDATES = "order_updates" 
    SYSTEM_EVENTS = "system_events"
    PERFORMANCE_UPDATES = "performance_updates"
    ALL = "all"

class WebSocketHandler:
    """WebSocket处理器"""
    
    def __init__(self):
        self.router = APIRouter()
        self.active_connections: Dict[str, List[WebSocket]] = {
            ConnectionType.MARKET_DATA: [],
            ConnectionType.ORDER_UPDATES: [],
            ConnectionType.SYSTEM_EVENTS: [],
            ConnectionType.PERFORMANCE_UPDATES: [],
            ConnectionType.ALL: []
        }
        self._setup_routes()
        self._running = False
        self._broadcast_tasks: List[asyncio.Task] = []
        
    def _setup_routes(self):
        """设置WebSocket路由"""
        
        @self.router.websocket("/market")
        async def market_data_websocket(websocket: WebSocket):
            """市场数据WebSocket连接"""
            await self.connect(websocket, ConnectionType.MARKET_DATA)
            try:
                while True:
                    # 保持连接活跃
                    data = await websocket.receive_text()
                    # 可以处理客户端发送的订阅请求等
                    await self._handle_client_message(websocket, data, ConnectionType.MARKET_DATA)
            except WebSocketDisconnect:
                await self.disconnect(websocket, ConnectionType.MARKET_DATA)
        
        @self.router.websocket("/orders")
        async def order_updates_websocket(websocket: WebSocket):
            """订单更新WebSocket连接"""
            await self.connect(websocket, ConnectionType.ORDER_UPDATES)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self._handle_client_message(websocket, data, ConnectionType.ORDER_UPDATES)
            except WebSocketDisconnect:
                await self.disconnect(websocket, ConnectionType.ORDER_UPDATES)
        
        @self.router.websocket("/system")
        async def system_events_websocket(websocket: WebSocket):
            """系统事件WebSocket连接"""
            await self.connect(websocket, ConnectionType.SYSTEM_EVENTS)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self._handle_client_message(websocket, data, ConnectionType.SYSTEM_EVENTS)
            except WebSocketDisconnect:
                await self.disconnect(websocket, ConnectionType.SYSTEM_EVENTS)
        
        @self.router.websocket("/performance")
        async def performance_updates_websocket(websocket: WebSocket):
            """性能更新WebSocket连接"""
            await self.connect(websocket, ConnectionType.PERFORMANCE_UPDATES)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self._handle_client_message(websocket, data, ConnectionType.PERFORMANCE_UPDATES)
            except WebSocketDisconnect:
                await self.disconnect(websocket, ConnectionType.PERFORMANCE_UPDATES)
        
        @self.router.websocket("/all")
        async def all_updates_websocket(websocket: WebSocket):
            """全部更新WebSocket连接"""
            await self.connect(websocket, ConnectionType.ALL)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self._handle_client_message(websocket, data, ConnectionType.ALL)
            except WebSocketDisconnect:
                await self.disconnect(websocket, ConnectionType.ALL)
    
    async def connect(self, websocket: WebSocket, connection_type: ConnectionType):
        """接受WebSocket连接"""
        await websocket.accept()
        self.active_connections[connection_type].append(websocket)
        logger.info(f"WebSocket连接已建立: {connection_type}, 当前连接数: {len(self.active_connections[connection_type])}")
        
        # 发送欢迎消息
        welcome_message = WebSocketMessage(
            type="connection_established",
            timestamp=datetime.now(),
            data={
                "connection_type": connection_type,
                "message": f"已连接到 {connection_type} 数据流",
                "server_time": datetime.now().isoformat()
            }
        )
        await websocket.send_text(welcome_message.json())
    
    async def disconnect(self, websocket: WebSocket, connection_type: ConnectionType):
        """断开WebSocket连接"""
        if websocket in self.active_connections[connection_type]:
            self.active_connections[connection_type].remove(websocket)
        logger.info(f"WebSocket连接已断开: {connection_type}, 剩余连接数: {len(self.active_connections[connection_type])}")
    
    async def _handle_client_message(self, websocket: WebSocket, data: str, connection_type: ConnectionType):
        """处理客户端消息"""
        try:
            message = json.loads(data)
            message_type = message.get("type")
            
            if message_type == "ping":
                # 心跳检测
                pong_message = WebSocketMessage(
                    type="pong",
                    timestamp=datetime.now(),
                    data={"server_time": datetime.now().isoformat()}
                )
                await websocket.send_text(pong_message.json())
            
            elif message_type == "subscribe":
                # 处理订阅请求
                symbols = message.get("symbols", [])
                await self._handle_subscription(websocket, symbols, connection_type)
            
            elif message_type == "unsubscribe":
                # 处理取消订阅请求
                symbols = message.get("symbols", [])
                await self._handle_unsubscription(websocket, symbols, connection_type)
                
        except json.JSONDecodeError:
            error_message = WebSocketMessage(
                type="error",
                timestamp=datetime.now(),
                data={"error": "Invalid JSON format"}
            )
            await websocket.send_text(error_message.json())
        except Exception as e:
            logger.error(f"处理客户端消息时出错: {e}")
    
    async def _handle_subscription(self, websocket: WebSocket, symbols: List[str], connection_type: ConnectionType):
        """处理订阅请求"""
        # TODO: 实现具体的订阅逻辑
        response = WebSocketMessage(
            type="subscription_confirmed",
            timestamp=datetime.now(),
            data={
                "symbols": symbols,
                "connection_type": connection_type,
                "message": f"已订阅 {len(symbols)} 个数据流"
            }
        )
        await websocket.send_text(response.json())
    
    async def _handle_unsubscription(self, websocket: WebSocket, symbols: List[str], connection_type: ConnectionType):
        """处理取消订阅请求"""
        # TODO: 实现具体的取消订阅逻辑
        response = WebSocketMessage(
            type="unsubscription_confirmed",
            timestamp=datetime.now(),
            data={
                "symbols": symbols,
                "connection_type": connection_type,
                "message": f"已取消订阅 {len(symbols)} 个数据流"
            }
        )
        await websocket.send_text(response.json())
    
    async def broadcast_market_data(self, symbol: str, price: float, volume: float, **kwargs):
        """广播市场数据"""
        message = MarketDataMessage(
            timestamp=datetime.now(),
            data={
                "symbol": symbol,
                "price": price,
                "volume": volume,
                **kwargs
            },
            symbol=symbol,
            price=price,
            volume=volume
        )
        
        await self._broadcast_to_connections(message, [ConnectionType.MARKET_DATA, ConnectionType.ALL])
    
    async def broadcast_order_update(self, order_id: str, status: str, filled_quantity: float, **kwargs):
        """广播订单更新"""
        message = OrderUpdateMessage(
            timestamp=datetime.now(),
            data={
                "order_id": order_id,
                "status": status,
                "filled_quantity": filled_quantity,
                **kwargs
            },
            order_id=order_id,
            status=status,
            filled_quantity=filled_quantity
        )
        
        await self._broadcast_to_connections(message, [ConnectionType.ORDER_UPDATES, ConnectionType.ALL])
    
    async def broadcast_system_event(self, event_type: str, severity: str, message: str, **kwargs):
        """广播系统事件"""
        event_message = SystemEventMessage(
            timestamp=datetime.now(),
            data={
                "event_type": event_type,
                "severity": severity,
                "message": message,
                **kwargs
            },
            event_type=event_type,
            severity=severity,
            message=message
        )
        
        await self._broadcast_to_connections(event_message, [ConnectionType.SYSTEM_EVENTS, ConnectionType.ALL])
    
    async def broadcast_performance_update(self, trading_mode: TradingMode, pnl: float, trades: int, **kwargs):
        """广播性能更新"""
        message = PerformanceUpdateMessage(
            timestamp=datetime.now(),
            data={
                "trading_mode": trading_mode,
                "pnl": pnl,
                "trades": trades,
                **kwargs
            },
            trading_mode=trading_mode,
            pnl=pnl,
            trades=trades
        )
        
        await self._broadcast_to_connections(message, [ConnectionType.PERFORMANCE_UPDATES, ConnectionType.ALL])
    
    async def _broadcast_to_connections(self, message: WebSocketMessage, connection_types: List[ConnectionType]):
        """向指定类型的连接广播消息"""
        message_text = message.json()
        
        for connection_type in connection_types:
            connections = self.active_connections[connection_type].copy()
            if not connections:
                continue
                
            # 并发发送消息，移除断开的连接
            disconnected = []
            tasks = []
            
            for websocket in connections:
                tasks.append(self._send_safe(websocket, message_text))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        disconnected.append(connections[i])
            
            # 清理断开的连接
            for websocket in disconnected:
                if websocket in self.active_connections[connection_type]:
                    self.active_connections[connection_type].remove(websocket)
                    logger.info(f"移除断开的WebSocket连接: {connection_type}")
    
    async def _send_safe(self, websocket: WebSocket, message: str):
        """安全发送消息，处理连接断开的情况"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"发送WebSocket消息失败: {e}")
            raise e
    
    async def initialize(self):
        """初始化WebSocket处理器"""
        logger.info("正在初始化WebSocket处理器...")
        self._running = True
        
        # 启动模拟数据广播任务（用于测试）
        self._broadcast_tasks = [
            asyncio.create_task(self._demo_market_data_broadcaster()),
            asyncio.create_task(self._demo_system_monitor())
        ]
        
        logger.info("WebSocket处理器初始化完成")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("正在清理WebSocket处理器资源...")
        self._running = False
        
        # 取消广播任务
        for task in self._broadcast_tasks:
            task.cancel()
        
        # 关闭所有连接
        for connection_type, connections in self.active_connections.items():
            for websocket in connections.copy():
                try:
                    await websocket.close()
                except:
                    pass
            connections.clear()
        
        logger.info("WebSocket处理器资源清理完成")
    
    async def _demo_market_data_broadcaster(self):
        """演示市场数据广播（仅用于测试）"""
        import random
        
        while self._running:
            try:
                # 模拟市场数据
                symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
                for symbol in symbols:
                    if self.active_connections[ConnectionType.MARKET_DATA] or self.active_connections[ConnectionType.ALL]:
                        base_price = 66000.0 if symbol == "BTCUSDT" else (3500.0 if symbol == "ETHUSDT" else 0.5)
                        price = base_price * (1 + random.uniform(-0.001, 0.001))
                        volume = random.uniform(100, 1000)
                        
                        await self.broadcast_market_data(
                            symbol=symbol,
                            price=price,
                            volume=volume,
                            bid=price - 0.5,
                            ask=price + 0.5
                        )
                
                await asyncio.sleep(1)  # 每秒发送一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"模拟市场数据广播出错: {e}")
                await asyncio.sleep(1)
    
    async def _demo_system_monitor(self):
        """演示系统监控（仅用于测试）"""
        import random
        
        while self._running:
            try:
                if self.active_connections[ConnectionType.SYSTEM_EVENTS] or self.active_connections[ConnectionType.ALL]:
                    # 模拟系统事件
                    events = [
                        ("strategy_update", "info", "策略参数已更新"),
                        ("order_executed", "info", "订单执行完成"),
                        ("risk_warning", "warning", "风险指标接近阈值"),
                        ("system_health", "info", "系统运行正常")
                    ]
                    
                    event_type, severity, message = random.choice(events)
                    await self.broadcast_system_event(event_type, severity, message)
                
                if self.active_connections[ConnectionType.PERFORMANCE_UPDATES] or self.active_connections[ConnectionType.ALL]:
                    # 模拟性能更新
                    await self.broadcast_performance_update(
                        trading_mode=TradingMode.PAPER,
                        pnl=random.uniform(-100, 200),
                        trades=random.randint(1, 10),
                        win_rate=random.uniform(0.5, 0.8)
                    )
                
                await asyncio.sleep(5)  # 每5秒发送一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"模拟系统监控出错: {e}")
                await asyncio.sleep(5)