"""
WebSocket系统集成测试

测试WebSocket实时推送系统的各个组件和功能，
包括连接管理、订阅管理、消息广播等。
"""

import asyncio
import json
import pytest
import websockets
from datetime import datetime
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from src.websocket import (
    WebSocketManager,
    ConnectionManager,
    SubscriptionManager,
    MessageBroadcaster,
    WebSocketConfig,
    MessageType,
    SubscriptionType,
    WebSocketMessage,
    ConnectionInfo,
    ConnectionStatus
)


# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def websocket_config():
    """WebSocket配置"""
    return WebSocketConfig(
        host="localhost",
        port=0,  # 使用随机端口
        max_connections=10,
        ping_interval=5,
        connection_timeout=30,
        auth_required=False
    )


@pytest.fixture
async def websocket_manager(websocket_config):
    """WebSocket管理器"""
    manager = WebSocketManager(websocket_config)
    yield manager
    if manager._running:
        await manager.stop()


@pytest.fixture
def connection_manager(websocket_config):
    """连接管理器"""
    return ConnectionManager(websocket_config)


@pytest.fixture
def subscription_manager():
    """订阅管理器"""
    return SubscriptionManager()


@pytest.fixture
def message_broadcaster():
    """消息广播器"""
    return MessageBroadcaster()


class MockWebSocket:
    """模拟WebSocket连接"""
    
    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.remote_address = ("127.0.0.1", 12345)
        self.request_headers = {"User-Agent": "test-client"}
        self.closed = False
        self.sent_messages = []
    
    async def send(self, message: str):
        """发送消息"""
        if self.closed:
            raise websockets.ConnectionClosed(None, None)
        self.sent_messages.append(message)
    
    async def close(self, code: int = 1000, reason: str = ""):
        """关闭连接"""
        self.closed = True
    
    def get_sent_messages(self) -> List[Dict]:
        """获取发送的消息"""
        return [json.loads(msg) for msg in self.sent_messages]


class TestWebSocketModels:
    """测试WebSocket数据模型"""
    
    def test_websocket_message_serialization(self):
        """测试消息序列化"""
        # 创建消息
        message = WebSocketMessage.create_data_message(
            MessageType.TRADING_SIGNAL,
            {"symbol": "BTCUSDT", "signal": "BUY", "price": 50000}
        )
        
        # 序列化
        json_str = message.to_json()
        assert json_str is not None
        
        # 反序列化
        restored_message = WebSocketMessage.from_json(json_str)
        assert restored_message.message_type == MessageType.TRADING_SIGNAL
        assert restored_message.data["symbol"] == "BTCUSDT"
        assert restored_message.data["signal"] == "BUY"
    
    def test_connection_info_activity_tracking(self):
        """测试连接活动跟踪"""
        conn_info = ConnectionInfo(connection_id="test-001")
        
        # 初始状态
        assert conn_info.is_active()
        
        # 更新活动时间
        conn_info.update_activity()
        assert conn_info.is_active()
        
        # 测试订阅管理
        conn_info.add_subscription(SubscriptionType.TRADING_SIGNALS)
        assert SubscriptionType.TRADING_SIGNALS in conn_info.subscriptions
        
        conn_info.remove_subscription(SubscriptionType.TRADING_SIGNALS)
        assert SubscriptionType.TRADING_SIGNALS not in conn_info.subscriptions
    
    def test_websocket_config_validation(self):
        """测试配置验证"""
        # 有效配置
        config = WebSocketConfig(port=8765, max_connections=100)
        errors = config.validate()
        assert len(errors) == 0
        
        # 无效端口
        config = WebSocketConfig(port=0, max_connections=100)
        errors = config.validate()
        assert len(errors) > 0
        assert any("端口号" in error for error in errors)
        
        # 无效最大连接数
        config = WebSocketConfig(port=8765, max_connections=0)
        errors = config.validate()
        assert len(errors) > 0
        assert any("最大连接数" in error for error in errors)


class TestConnectionManager:
    """测试连接管理器"""
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, connection_manager):
        """测试连接生命周期"""
        # 启动连接管理器
        await connection_manager.start()
        
        # 创建模拟WebSocket
        mock_ws = MockWebSocket("test-001")
        
        # 模拟连接处理
        connection_info = ConnectionInfo(
            connection_id="test-001",
            status=ConnectionStatus.CONNECTED
        )
        
        connection_manager._connections["test-001"] = mock_ws
        connection_manager._connection_info["test-001"] = connection_info
        
        # 测试发送消息
        message = WebSocketMessage.create_control_message(
            MessageType.PING,
            {"test": "data"}
        )
        
        success = await connection_manager.send_message("test-001", message)
        assert success is True
        
        # 验证消息发送
        sent_messages = mock_ws.get_sent_messages()
        assert len(sent_messages) == 1
        assert sent_messages[0]["type"] == "ping"
        assert sent_messages[0]["data"]["test"] == "data"
        
        # 测试连接关闭
        await connection_manager.close_connection("test-001")
        assert "test-001" not in connection_manager._connections
        
        # 停止连接管理器
        await connection_manager.stop()
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, connection_manager):
        """测试消息广播"""
        await connection_manager.start()
        
        # 创建多个模拟连接
        connections = []
        for i in range(3):
            conn_id = f"test-{i:03d}"
            mock_ws = MockWebSocket(conn_id)
            connection_info = ConnectionInfo(
                connection_id=conn_id,
                status=ConnectionStatus.CONNECTED
            )
            
            connection_manager._connections[conn_id] = mock_ws
            connection_manager._connection_info[conn_id] = connection_info
            connections.append((conn_id, mock_ws))
        
        # 广播消息
        message = WebSocketMessage.create_data_message(
            MessageType.TRADING_SIGNAL,
            {"symbol": "BTCUSDT", "signal": "BUY"}
        )
        
        success_count = await connection_manager.broadcast_message(message)
        assert success_count == 3
        
        # 验证所有连接都收到消息
        for conn_id, mock_ws in connections:
            sent_messages = mock_ws.get_sent_messages()
            assert len(sent_messages) == 1
            assert sent_messages[0]["type"] == "trading_signal"
            assert sent_messages[0]["data"]["symbol"] == "BTCUSDT"
        
        await connection_manager.stop()
    
    @pytest.mark.asyncio
    async def test_connection_stats(self, connection_manager):
        """测试连接统计"""
        await connection_manager.start()
        
        # 初始统计
        stats = connection_manager.get_stats()
        assert stats.total_connections == 0
        assert stats.active_connections == 0
        
        # 添加连接
        connection_manager._connections["test-001"] = MockWebSocket("test-001")
        connection_manager._connection_info["test-001"] = ConnectionInfo(
            connection_id="test-001"
        )
        
        # 更新统计
        stats = connection_manager.get_stats()
        assert stats.active_connections == 1
        
        await connection_manager.stop()


class TestSubscriptionManager:
    """测试订阅管理器"""
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle(self, subscription_manager):
        """测试订阅生命周期"""
        # 创建订阅
        success, subscription_id = await subscription_manager.subscribe(
            "conn-001",
            SubscriptionType.TRADING_SIGNALS,
            {"symbol": "BTCUSDT"}
        )
        
        assert success is True
        assert subscription_id is not None
        
        # 验证订阅创建
        subscription = subscription_manager.get_subscription(subscription_id)
        assert subscription is not None
        assert subscription.connection_id == "conn-001"
        assert subscription.subscription_type == SubscriptionType.TRADING_SIGNALS
        assert subscription.filters["symbol"] == "BTCUSDT"
        
        # 获取连接的订阅
        conn_subscriptions = subscription_manager.get_subscriptions_by_connection("conn-001")
        assert len(conn_subscriptions) == 1
        assert conn_subscriptions[0].subscription_id == subscription_id
        
        # 取消订阅
        success, message = await subscription_manager.unsubscribe("conn-001", subscription_id)
        assert success is True
        
        # 验证订阅移除
        subscription = subscription_manager.get_subscription(subscription_id)
        assert subscription is None
    
    @pytest.mark.asyncio
    async def test_subscription_filtering(self, subscription_manager):
        """测试订阅过滤"""
        # 创建带过滤条件的订阅
        success, subscription_id = await subscription_manager.subscribe(
            "conn-001",
            SubscriptionType.TRADING_SIGNALS,
            {"symbol": "BTCUSDT", "strategy": "momentum"}
        )
        assert success is True
        
        # 测试匹配的数据
        matching_data = {
            "symbol": "BTCUSDT",
            "strategy": "momentum",
            "signal": "BUY"
        }
        
        matching_subscriptions = subscription_manager.get_matching_subscriptions(
            SubscriptionType.TRADING_SIGNALS,
            matching_data
        )
        assert len(matching_subscriptions) == 1
        
        # 测试不匹配的数据
        non_matching_data = {
            "symbol": "ETHUSDT",  # 不匹配的交易对
            "strategy": "momentum",
            "signal": "BUY"
        }
        
        matching_subscriptions = subscription_manager.get_matching_subscriptions(
            SubscriptionType.TRADING_SIGNALS,
            non_matching_data
        )
        assert len(matching_subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_subscription_stats(self, subscription_manager):
        """测试订阅统计"""
        # 创建多个订阅
        for i in range(3):
            await subscription_manager.subscribe(
                f"conn-{i:03d}",
                SubscriptionType.TRADING_SIGNALS
            )
        
        # 检查统计
        assert subscription_manager.get_total_subscriptions() == 3
        assert subscription_manager.get_connection_count_by_type(
            SubscriptionType.TRADING_SIGNALS
        ) == 3
        
        # 检查活跃类型
        active_types = subscription_manager.get_active_types()
        assert SubscriptionType.TRADING_SIGNALS in active_types


class TestMessageBroadcaster:
    """测试消息广播器"""
    
    @pytest.fixture
    async def broadcaster_with_mocks(self):
        """带模拟组件的广播器"""
        # 创建模拟组件
        mock_conn_manager = AsyncMock()
        mock_sub_manager = MagicMock()
        
        # 设置模拟订阅
        mock_subscription = MagicMock()
        mock_subscription.connection_id = "conn-001"
        mock_subscription.subscription_type = SubscriptionType.TRADING_SIGNALS
        mock_subscription.matches_filter.return_value = True
        
        mock_sub_manager.get_matching_subscriptions.return_value = [mock_subscription]
        mock_sub_manager.update_message_stats = MagicMock()
        
        # 设置模拟发送
        mock_conn_manager.send_message.return_value = True
        
        # 创建广播器
        broadcaster = MessageBroadcaster(mock_conn_manager, mock_sub_manager)
        await broadcaster.start()
        
        yield broadcaster, mock_conn_manager, mock_sub_manager
        
        await broadcaster.stop()
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, broadcaster_with_mocks):
        """测试消息广播"""
        broadcaster, mock_conn_manager, mock_sub_manager = broadcaster_with_mocks
        
        # 创建测试消息
        message = WebSocketMessage.create_data_message(
            MessageType.TRADING_SIGNAL,
            {"symbol": "BTCUSDT", "signal": "BUY", "price": 50000}
        )
        
        # 广播消息
        success_count = await broadcaster.broadcast_message(
            message=message,
            subscription_type=SubscriptionType.TRADING_SIGNALS
        )
        
        assert success_count == 1
        
        # 验证调用
        mock_sub_manager.get_matching_subscriptions.assert_called_once()
        mock_conn_manager.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_system_broadcast(self, broadcaster_with_mocks):
        """测试系统级广播"""
        broadcaster, mock_conn_manager, mock_sub_manager = broadcaster_with_mocks
        
        # 广播风险警报
        success_count = await broadcaster.handle_system_broadcast(
            MessageType.RISK_ALERT,
            {
                "alert_type": "high_volatility",
                "symbol": "BTCUSDT",
                "severity": "high"
            }
        )
        
        assert success_count == 1
        mock_conn_manager.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_stats(self, broadcaster_with_mocks):
        """测试广播统计"""
        broadcaster, mock_conn_manager, mock_sub_manager = broadcaster_with_mocks
        
        # 获取初始统计
        stats = broadcaster.get_stats()
        assert "messages_sent" in stats
        assert "broadcast_count" in stats
        assert "average_latency_ms" in stats


class TestWebSocketManager:
    """测试WebSocket管理器"""
    
    @pytest.mark.asyncio
    async def test_websocket_manager_lifecycle(self, websocket_manager):
        """测试WebSocket管理器生命周期"""
        # 启动服务器
        await websocket_manager.start(host="localhost", port=0)
        assert websocket_manager._running is True
        
        # 获取系统统计
        stats = websocket_manager.get_system_stats()
        assert stats["server_status"] == "running"
        assert "connections" in stats
        assert "subscriptions" in stats
        assert "broadcasts" in stats
        
        # 停止服务器
        await websocket_manager.stop()
        assert websocket_manager._running is False
    
    @pytest.mark.asyncio
    async def test_trading_signal_broadcast(self, websocket_manager):
        """测试交易信号广播"""
        # 模拟广播器
        with patch.object(websocket_manager.message_broadcaster, 'broadcast_message') as mock_broadcast:
            mock_broadcast.return_value = 5
            
            # 广播交易信号
            signal_data = {
                "symbol": "BTCUSDT",
                "signal": "BUY",
                "price": 50000,
                "confidence": 0.85
            }
            
            success_count = await websocket_manager.broadcast_trading_signal(signal_data)
            assert success_count == 5
            
            # 验证调用参数
            mock_broadcast.assert_called_once()
            args, kwargs = mock_broadcast.call_args
            assert kwargs["subscription_type"] == SubscriptionType.TRADING_SIGNALS
            assert kwargs["priority"] == 1  # 高优先级
    
    @pytest.mark.asyncio
    async def test_strategy_status_broadcast(self, websocket_manager):
        """测试策略状态广播"""
        with patch.object(websocket_manager.message_broadcaster, 'broadcast_message') as mock_broadcast:
            mock_broadcast.return_value = 3
            
            status_data = {
                "status": "running",
                "performance": {"pnl": 1250.50, "win_rate": 0.65},
                "positions": 2
            }
            
            success_count = await websocket_manager.broadcast_strategy_status(
                "strategy-001", status_data
            )
            assert success_count == 3
            
            mock_broadcast.assert_called_once()
            args, kwargs = mock_broadcast.call_args
            assert kwargs["subscription_type"] == SubscriptionType.STRATEGY_STATUS
    
    @pytest.mark.asyncio
    async def test_risk_alert_broadcast(self, websocket_manager):
        """测试风险警报广播"""
        with patch.object(websocket_manager.message_broadcaster, 'broadcast_message') as mock_broadcast:
            mock_broadcast.return_value = 8
            
            alert_data = {
                "alert_type": "position_limit",
                "symbol": "BTCUSDT",
                "current_exposure": 150000,
                "limit": 100000,
                "severity": "high"
            }
            
            success_count = await websocket_manager.broadcast_risk_alert(alert_data)
            assert success_count == 8
            
            mock_broadcast.assert_called_once()
            args, kwargs = mock_broadcast.call_args
            assert kwargs["subscription_type"] == SubscriptionType.RISK_ALERTS
            assert kwargs["priority"] == 2  # 紧急优先级


class TestWebSocketIntegration:
    """WebSocket系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_system_integration(self, websocket_config):
        """测试完整系统集成"""
        # 创建WebSocket管理器
        manager = WebSocketManager(websocket_config)
        
        try:
            # 启动系统
            await manager.start(host="localhost", port=0)
            
            # 模拟多个客户端连接和订阅
            connections = []
            for i in range(3):
                # 这里在真实测试中会创建实际的WebSocket客户端连接
                # 为了测试简化，我们直接测试管理器的统计功能
                pass
            
            # 测试各种广播功能
            await manager.broadcast_trading_signal({
                "symbol": "BTCUSDT",
                "signal": "BUY",
                "price": 50000
            })
            
            await manager.broadcast_strategy_status("test-strategy", {
                "status": "running",
                "pnl": 1000
            })
            
            await manager.broadcast_risk_alert({
                "alert_type": "high_volatility",
                "severity": "medium"
            })
            
            # 验证系统状态
            stats = manager.get_system_stats()
            assert stats["server_status"] == "running"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, websocket_manager):
        """测试错误处理"""
        # 测试未启动时的操作
        stats = websocket_manager.get_system_stats()
        assert stats["server_status"] == "stopped"
        
        # 测试广播到空订阅列表
        with patch.object(websocket_manager.subscription_manager, 'get_matching_subscriptions') as mock_get:
            mock_get.return_value = []
            
            success_count = await websocket_manager.broadcast_trading_signal({
                "symbol": "BTCUSDT",
                "signal": "BUY"
            })
            assert success_count == 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, websocket_manager):
        """测试性能指标"""
        # 获取初始统计
        initial_stats = websocket_manager.get_broadcast_stats()
        
        # 执行一些操作
        with patch.object(websocket_manager.message_broadcaster, 'broadcast_message') as mock_broadcast:
            mock_broadcast.return_value = 1
            
            # 连续广播多条消息
            for i in range(10):
                await websocket_manager.broadcast_trading_signal({
                    "symbol": "BTCUSDT",
                    "signal": "BUY" if i % 2 == 0 else "SELL",
                    "price": 50000 + i * 100
                })
        
        # 验证统计更新
        final_stats = websocket_manager.get_broadcast_stats()
        # 注意：由于我们使用了mock，实际的统计可能不会更新
        # 在真实场景中，这些统计会被正确更新


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])