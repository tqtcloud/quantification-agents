"""
WebSocket集成测试套件
测试连接管理、订阅机制和消息推送
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.websocket.websocket_manager import WebSocketManager
from src.websocket.connection_manager import ConnectionManager
from src.websocket.subscription_manager import SubscriptionManager
from src.websocket.message_broadcaster import MessageBroadcaster
from src.websocket.models import (
    WebSocketMessage, 
    MessageType, 
    SubscribeRequest,
    UnsubscribeRequest,
    AuthRequest,
    MarketDataMessage,
    SystemNotification
)
from src.core.database import DatabaseManager


class MockWebSocketConnection:
    """模拟WebSocket连接"""
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or f"test_client_{int(time.time())}"
        self.messages = []
        self.closed = False
        self.close_code = None
        self.close_reason = None
    
    async def send(self, message: str):
        """发送消息"""
        if self.closed:
            raise ConnectionClosed(None, None)
        self.messages.append(message)
    
    async def close(self, code: int = 1000, reason: str = ""):
        """关闭连接"""
        self.closed = True
        self.close_code = code
        self.close_reason = reason
    
    def get_messages(self) -> List[dict]:
        """获取接收到的消息"""
        return [json.loads(msg) for msg in self.messages]
    
    def clear_messages(self):
        """清空消息"""
        self.messages.clear()


class TestConnectionManager:
    """连接管理器集成测试"""
    
    @pytest_asyncio.fixture
    async def connection_manager(self):
        """创建连接管理器"""
        config = {
            'max_connections': 1000,
            'heartbeat_interval': 30,
            'cleanup_interval': 60
        }
        
        manager = ConnectionManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, connection_manager):
        """测试连接生命周期管理"""
        manager = connection_manager
        
        # 1. 添加连接
        ws_mock = MockWebSocketConnection("client_001")
        client_id = await manager.add_connection(ws_mock, {"user_id": "user_001"})
        
        assert client_id == "client_001"
        assert manager.get_connection_count() == 1
        assert manager.is_connected(client_id)
        
        # 2. 获取连接信息
        connection_info = manager.get_connection_info(client_id)
        assert connection_info is not None
        assert connection_info['client_id'] == client_id
        assert connection_info['metadata']['user_id'] == "user_001"
        assert 'connected_at' in connection_info
        
        # 3. 移除连接
        await manager.remove_connection(client_id)
        assert manager.get_connection_count() == 0
        assert not manager.is_connected(client_id)
    
    @pytest.mark.asyncio
    async def test_multiple_connections(self, connection_manager):
        """测试多连接管理"""
        manager = connection_manager
        client_ids = []
        
        # 添加多个连接
        for i in range(5):
            ws_mock = MockWebSocketConnection(f"client_{i:03d}")
            client_id = await manager.add_connection(
                ws_mock, 
                {"user_id": f"user_{i:03d}"}
            )
            client_ids.append(client_id)
        
        assert manager.get_connection_count() == 5
        
        # 验证所有连接都存在
        for client_id in client_ids:
            assert manager.is_connected(client_id)
        
        # 批量移除连接
        for client_id in client_ids[:3]:
            await manager.remove_connection(client_id)
        
        assert manager.get_connection_count() == 2
        
        # 获取活跃连接列表
        active_connections = manager.get_active_connections()
        assert len(active_connections) == 2
        
        for connection in active_connections:
            assert connection['client_id'] in client_ids[3:]
    
    @pytest.mark.asyncio
    async def test_connection_limits(self, connection_manager):
        """测试连接数量限制"""
        # 创建限制连接数的管理器
        config = {'max_connections': 3}
        limited_manager = ConnectionManager(config)
        await limited_manager.initialize()
        
        try:
            # 添加到达限制的连接数
            for i in range(3):
                ws_mock = MockWebSocketConnection(f"client_{i}")
                await limited_manager.add_connection(ws_mock, {})
            
            assert limited_manager.get_connection_count() == 3
            
            # 尝试添加超出限制的连接
            ws_mock = MockWebSocketConnection("client_extra")
            
            with pytest.raises(Exception) as exc_info:
                await limited_manager.add_connection(ws_mock, {})
            
            assert "connection limit" in str(exc_info.value).lower()
            
        finally:
            await limited_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, connection_manager):
        """测试心跳机制"""
        manager = connection_manager
        
        # 添加连接
        ws_mock = MockWebSocketConnection("heartbeat_client")
        client_id = await manager.add_connection(ws_mock, {})
        
        # 触发心跳检查
        await manager._send_heartbeat(client_id)
        
        # 验证心跳消息
        messages = ws_mock.get_messages()
        assert len(messages) == 1
        assert messages[0]['type'] == 'heartbeat'
        assert 'timestamp' in messages[0]
        
        # 模拟心跳响应
        await manager._handle_heartbeat_response(client_id)
        
        # 验证连接仍然活跃
        assert manager.is_connected(client_id)


class TestSubscriptionManager:
    """订阅管理器集成测试"""
    
    @pytest_asyncio.fixture
    async def subscription_manager(self):
        """创建订阅管理器"""
        config = {
            'max_subscriptions_per_client': 50,
            'supported_channels': [
                'market_data',
                'order_updates',
                'system_notifications',
                'strategy_signals'
            ]
        }
        
        manager = SubscriptionManager(config)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle(self, subscription_manager):
        """测试订阅生命周期"""
        manager = subscription_manager
        client_id = "test_client"
        
        # 1. 添加订阅
        subscription_data = {
            'channel': 'market_data',
            'symbol': 'BTCUSDT',
            'interval': '1m'
        }
        
        success = await manager.add_subscription(client_id, subscription_data)
        assert success
        
        # 验证订阅
        subscriptions = manager.get_client_subscriptions(client_id)
        assert len(subscriptions) == 1
        assert subscriptions[0]['channel'] == 'market_data'
        assert subscriptions[0]['symbol'] == 'BTCUSDT'
        
        # 2. 获取订阅者
        subscribers = manager.get_channel_subscribers('market_data')
        assert client_id in subscribers
        
        # 3. 移除订阅
        success = await manager.remove_subscription(
            client_id, 
            'market_data', 
            {'symbol': 'BTCUSDT'}
        )
        assert success
        
        # 验证订阅已移除
        subscriptions = manager.get_client_subscriptions(client_id)
        assert len(subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, subscription_manager):
        """测试多重订阅"""
        manager = subscription_manager
        client_id = "multi_sub_client"
        
        # 添加多个订阅
        subscriptions = [
            {'channel': 'market_data', 'symbol': 'BTCUSDT'},
            {'channel': 'market_data', 'symbol': 'ETHUSDT'},
            {'channel': 'order_updates', 'user_id': 'user123'},
            {'channel': 'system_notifications'}
        ]
        
        for sub_data in subscriptions:
            success = await manager.add_subscription(client_id, sub_data)
            assert success
        
        # 验证所有订阅
        client_subs = manager.get_client_subscriptions(client_id)
        assert len(client_subs) == 4
        
        # 验证频道订阅者
        market_data_subs = manager.get_channel_subscribers('market_data')
        order_update_subs = manager.get_channel_subscribers('order_updates')
        system_notif_subs = manager.get_channel_subscribers('system_notifications')
        
        assert client_id in market_data_subs
        assert client_id in order_update_subs
        assert client_id in system_notif_subs
        
        # 移除特定订阅
        success = await manager.remove_subscription(
            client_id, 
            'market_data', 
            {'symbol': 'BTCUSDT'}
        )
        assert success
        
        # 验证只有特定订阅被移除
        remaining_subs = manager.get_client_subscriptions(client_id)
        assert len(remaining_subs) == 3
        
        # 验证ETHUSDT订阅仍然存在
        eth_sub_exists = any(
            sub['channel'] == 'market_data' and sub.get('symbol') == 'ETHUSDT'
            for sub in remaining_subs
        )
        assert eth_sub_exists
    
    @pytest.mark.asyncio
    async def test_subscription_limits(self, subscription_manager):
        """测试订阅数量限制"""
        # 创建限制订阅数的管理器
        config = {
            'max_subscriptions_per_client': 3,
            'supported_channels': ['market_data', 'order_updates']
        }
        limited_manager = SubscriptionManager(config)
        await limited_manager.initialize()
        
        try:
            client_id = "limited_client"
            
            # 添加到达限制的订阅数
            for i in range(3):
                success = await limited_manager.add_subscription(
                    client_id, 
                    {'channel': 'market_data', 'symbol': f'SYMBOL{i}'}
                )
                assert success
            
            # 尝试添加超出限制的订阅
            success = await limited_manager.add_subscription(
                client_id, 
                {'channel': 'market_data', 'symbol': 'EXTRA_SYMBOL'}
            )
            assert not success  # 应该失败
            
            # 验证订阅数量没有超过限制
            subscriptions = limited_manager.get_client_subscriptions(client_id)
            assert len(subscriptions) == 3
            
        finally:
            await limited_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_client_cleanup(self, subscription_manager):
        """测试客户端清理"""
        manager = subscription_manager
        client_id = "cleanup_client"
        
        # 添加多个订阅
        for i in range(3):
            await manager.add_subscription(
                client_id, 
                {'channel': 'market_data', 'symbol': f'SYMBOL{i}'}
            )
        
        # 验证订阅存在
        assert len(manager.get_client_subscriptions(client_id)) == 3
        assert client_id in manager.get_channel_subscribers('market_data')
        
        # 清理客户端所有订阅
        await manager.cleanup_client_subscriptions(client_id)
        
        # 验证订阅已清理
        assert len(manager.get_client_subscriptions(client_id)) == 0
        assert client_id not in manager.get_channel_subscribers('market_data')


class TestMessageBroadcaster:
    """消息广播器集成测试"""
    
    @pytest_asyncio.fixture
    async def message_broadcaster(self):
        """创建消息广播器"""
        config = {
            'max_queue_size': 1000,
            'broadcast_timeout': 5.0,
            'retry_attempts': 3
        }
        
        broadcaster = MessageBroadcaster(config)
        await broadcaster.initialize()
        
        yield broadcaster
        
        await broadcaster.shutdown()
    
    @pytest.mark.asyncio
    async def test_single_client_broadcast(self, message_broadcaster):
        """测试单客户端广播"""
        broadcaster = message_broadcaster
        
        # 创建模拟连接
        ws_mock = MockWebSocketConnection("broadcast_client")
        client_id = "broadcast_client"
        
        # 注册连接
        await broadcaster.register_connection(client_id, ws_mock)
        
        # 发送消息
        message = MarketDataMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.5,
            timestamp=datetime.now()
        )
        
        await broadcaster.send_to_client(client_id, message.dict())
        
        # 验证消息
        messages = ws_mock.get_messages()
        assert len(messages) == 1
        assert messages[0]['type'] == 'market_data'
        assert messages[0]['symbol'] == 'BTCUSDT'
        assert messages[0]['price'] == 50000.0
    
    @pytest.mark.asyncio
    async def test_channel_broadcast(self, message_broadcaster):
        """测试频道广播"""
        broadcaster = message_broadcaster
        
        # 创建多个模拟连接
        connections = {}
        for i in range(3):
            client_id = f"client_{i}"
            ws_mock = MockWebSocketConnection(client_id)
            connections[client_id] = ws_mock
            await broadcaster.register_connection(client_id, ws_mock)
        
        # 广播频道消息
        message = SystemNotification(
            message="System maintenance scheduled",
            level="warning",
            timestamp=datetime.now()
        )
        
        client_ids = list(connections.keys())
        await broadcaster.broadcast_to_clients(client_ids, message.dict())
        
        # 验证所有客户端都收到消息
        for client_id, ws_mock in connections.items():
            messages = ws_mock.get_messages()
            assert len(messages) == 1
            assert messages[0]['type'] == 'system_notification'
            assert messages[0]['level'] == 'warning'
    
    @pytest.mark.asyncio
    async def test_failed_message_handling(self, message_broadcaster):
        """测试失败消息处理"""
        broadcaster = message_broadcaster
        
        # 创建一个会失败的连接
        class FailingWebSocket:
            async def send(self, message):
                raise WebSocketException("Connection failed")
        
        failing_ws = FailingWebSocket()
        client_id = "failing_client"
        
        await broadcaster.register_connection(client_id, failing_ws)
        
        # 发送消息（应该失败但不抛出异常）
        message = {"type": "test", "data": "test_data"}
        
        # 这应该不会抛出异常
        await broadcaster.send_to_client(client_id, message)
        
        # 验证连接被标记为失败或移除
        # 这里可以检查内部状态或错误日志
    
    @pytest.mark.asyncio
    async def test_message_queue_overflow(self, message_broadcaster):
        """测试消息队列溢出处理"""
        # 创建小队列的广播器
        config = {'max_queue_size': 2, 'broadcast_timeout': 0.1}
        small_queue_broadcaster = MessageBroadcaster(config)
        await small_queue_broadcaster.initialize()
        
        try:
            # 创建慢速连接
            class SlowWebSocket:
                def __init__(self):
                    self.messages = []
                
                async def send(self, message):
                    await asyncio.sleep(0.2)  # 模拟慢速发送
                    self.messages.append(message)
            
            slow_ws = SlowWebSocket()
            client_id = "slow_client"
            
            await small_queue_broadcaster.register_connection(client_id, slow_ws)
            
            # 快速发送多条消息
            messages = [
                {"type": "test", "data": f"message_{i}"} 
                for i in range(5)
            ]
            
            # 并发发送消息
            tasks = [
                small_queue_broadcaster.send_to_client(client_id, msg) 
                for msg in messages
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 验证队列处理（一些消息可能被丢弃）
            # 这里主要验证系统不会崩溃
            
        finally:
            await small_queue_broadcaster.shutdown()


class TestWebSocketManagerIntegration:
    """WebSocket管理器完整集成测试"""
    
    @pytest_asyncio.fixture
    async def websocket_manager(self):
        """创建WebSocket管理器"""
        config = {
            'host': '127.0.0.1',
            'port': 8765,
            'max_connections': 100,
            'heartbeat_interval': 30,
            'connection_timeout': 60,
            'max_subscriptions_per_client': 20
        }
        
        database = DatabaseManager('sqlite:///:memory:')
        await database.initialize()
        
        manager = WebSocketManager(config, database)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_full_client_session(self, websocket_manager):
        """测试完整的客户端会话"""
        manager = websocket_manager
        
        # 模拟客户端连接
        ws_mock = MockWebSocketConnection("session_client")
        
        # 1. 建立连接
        await manager.handle_connection(ws_mock, {"user_agent": "test_client"})
        
        # 2. 认证
        auth_message = AuthRequest(
            token="test_token",
            timestamp=datetime.now()
        )
        
        await manager.handle_message(
            ws_mock.client_id, 
            auth_message.dict()
        )
        
        # 3. 订阅
        subscribe_message = SubscribeRequest(
            channel="market_data",
            symbol="BTCUSDT",
            interval="1m",
            timestamp=datetime.now()
        )
        
        await manager.handle_message(
            ws_mock.client_id, 
            subscribe_message.dict()
        )
        
        # 4. 接收广播消息
        market_data = MarketDataMessage(
            symbol="BTCUSDT",
            price=45000.0,
            volume=2.0,
            timestamp=datetime.now()
        )
        
        await manager.broadcast_market_data("BTCUSDT", market_data.dict())
        
        # 验证客户端收到消息
        messages = ws_mock.get_messages()
        
        # 应该收到认证响应、订阅确认和市场数据
        assert len(messages) >= 2
        
        # 查找市场数据消息
        market_data_received = False
        for message in messages:
            if message.get('type') == 'market_data':
                assert message['symbol'] == 'BTCUSDT'
                assert message['price'] == 45000.0
                market_data_received = True
                break
        
        assert market_data_received
        
        # 5. 取消订阅
        unsubscribe_message = UnsubscribeRequest(
            channel="market_data",
            symbol="BTCUSDT",
            timestamp=datetime.now()
        )
        
        await manager.handle_message(
            ws_mock.client_id,
            unsubscribe_message.dict()
        )
        
        # 6. 断开连接
        await manager.handle_disconnection(ws_mock.client_id)
        
        # 验证连接已清理
        assert not manager.connection_manager.is_connected(ws_mock.client_id)
    
    @pytest.mark.asyncio
    async def test_concurrent_clients(self, websocket_manager):
        """测试并发客户端处理"""
        manager = websocket_manager
        clients = []
        
        # 创建多个并发客户端
        for i in range(10):
            ws_mock = MockWebSocketConnection(f"concurrent_client_{i}")
            clients.append(ws_mock)
            
            # 并发建立连接
            asyncio.create_task(
                manager.handle_connection(ws_mock, {"client_id": i})
            )
        
        # 等待所有连接建立
        await asyncio.sleep(0.1)
        
        # 验证所有连接
        for ws_mock in clients:
            assert manager.connection_manager.is_connected(ws_mock.client_id)
        
        # 并发发送广播消息
        system_message = SystemNotification(
            message="System update notification",
            level="info",
            timestamp=datetime.now()
        )
        
        await manager.broadcast_system_notification(system_message.dict())
        
        # 验证所有客户端收到消息
        await asyncio.sleep(0.1)
        
        for ws_mock in clients:
            messages = ws_mock.get_messages()
            notification_received = any(
                msg.get('type') == 'system_notification'
                for msg in messages
            )
            assert notification_received
        
        # 清理连接
        for ws_mock in clients:
            await manager.handle_disconnection(ws_mock.client_id)
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, websocket_manager):
        """测试错误恢复机制"""
        manager = websocket_manager
        
        # 创建连接
        ws_mock = MockWebSocketConnection("error_client")
        await manager.handle_connection(ws_mock, {})
        
        # 发送无效消息
        invalid_messages = [
            {"invalid": "message"},  # 缺少type字段
            {"type": "unknown_type"},  # 未知消息类型
            {"type": "subscribe"},  # 缺少必需字段
            "not_json"  # 无效JSON
        ]
        
        for invalid_msg in invalid_messages:
            try:
                await manager.handle_message(ws_mock.client_id, invalid_msg)
            except Exception:
                # 错误应该被优雅处理，不应该崩溃
                pass
        
        # 验证连接仍然有效
        assert manager.connection_manager.is_connected(ws_mock.client_id)
        
        # 发送有效消息应该仍然工作
        valid_message = SystemNotification(
            message="Test notification",
            level="info",
            timestamp=datetime.now()
        )
        
        await manager.broadcast_system_notification(valid_message.dict())
        
        # 验证能收到有效消息
        messages = ws_mock.get_messages()
        valid_message_received = any(
            msg.get('type') == 'system_notification'
            for msg in messages
        )
        assert valid_message_received


if __name__ == "__main__":
    # 运行WebSocket集成测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])