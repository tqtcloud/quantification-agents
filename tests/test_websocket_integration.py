"""WebSocket集成测试"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from src.web.websocket_handler import WebSocketHandler, ConnectionType
from src.web.models import TradingMode


class MockWebSocket:
    """模拟WebSocket连接"""
    
    def __init__(self):
        self.messages: List[str] = []
        self.closed = False
        self.accepted = False
    
    async def accept(self):
        """接受连接"""
        self.accepted = True
    
    async def send_text(self, message: str):
        """发送文本消息"""
        if not self.closed:
            self.messages.append(message)
    
    async def receive_text(self) -> str:
        """接收文本消息"""
        return json.dumps({"type": "ping"})
    
    async def close(self):
        """关闭连接"""
        self.closed = True


class TestWebSocketIntegration:
    """WebSocket集成测试"""
    
    @pytest.fixture
    async def websocket_handler(self):
        """创建WebSocket处理器"""
        handler = WebSocketHandler()
        await handler.initialize()
        yield handler
        await handler.cleanup()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, websocket_handler):
        """测试WebSocket连接生命周期"""
        mock_websocket = MockWebSocket()
        
        # 测试连接建立
        await websocket_handler.connect(mock_websocket, ConnectionType.MARKET_DATA)
        assert mock_websocket.accepted
        assert mock_websocket in websocket_handler.active_connections[ConnectionType.MARKET_DATA]
        assert len(mock_websocket.messages) == 1  # 欢迎消息
        
        # 验证欢迎消息格式
        welcome_message = json.loads(mock_websocket.messages[0])
        assert welcome_message["type"] == "connection_established"
        assert welcome_message["data"]["connection_type"] == ConnectionType.MARKET_DATA
        
        # 测试连接断开
        await websocket_handler.disconnect(mock_websocket, ConnectionType.MARKET_DATA)
        assert mock_websocket not in websocket_handler.active_connections[ConnectionType.MARKET_DATA]
    
    @pytest.mark.asyncio
    async def test_market_data_broadcast(self, websocket_handler):
        """测试市场数据广播"""
        # 创建多个连接
        market_websocket = MockWebSocket()
        all_websocket = MockWebSocket()
        
        await websocket_handler.connect(market_websocket, ConnectionType.MARKET_DATA)
        await websocket_handler.connect(all_websocket, ConnectionType.ALL)
        
        # 广播市场数据
        await websocket_handler.broadcast_market_data(
            symbol="BTCUSDT",
            price=66000.0,
            volume=1500.0,
            bid=65995.0,
            ask=66005.0
        )
        
        # 验证两个连接都收到消息
        assert len(market_websocket.messages) == 2  # 欢迎消息 + 市场数据
        assert len(all_websocket.messages) == 2     # 欢迎消息 + 市场数据
        
        # 验证消息内容
        market_data_message = json.loads(market_websocket.messages[1])
        assert market_data_message["type"] == "market_data"
        assert market_data_message["symbol"] == "BTCUSDT"
        assert market_data_message["price"] == 66000.0
        assert market_data_message["volume"] == 1500.0
    
    @pytest.mark.asyncio
    async def test_order_update_broadcast(self, websocket_handler):
        """测试订单更新广播"""
        order_websocket = MockWebSocket()
        await websocket_handler.connect(order_websocket, ConnectionType.ORDER_UPDATES)
        
        # 广播订单更新
        await websocket_handler.broadcast_order_update(
            order_id="order_001",
            status="FILLED",
            filled_quantity=0.1,
            symbol="BTCUSDT"
        )
        
        # 验证消息
        assert len(order_websocket.messages) == 2
        order_message = json.loads(order_websocket.messages[1])
        assert order_message["type"] == "order_update"
        assert order_message["order_id"] == "order_001"
        assert order_message["status"] == "FILLED"
        assert order_message["filled_quantity"] == 0.1
    
    @pytest.mark.asyncio
    async def test_system_event_broadcast(self, websocket_handler):
        """测试系统事件广播"""
        system_websocket = MockWebSocket()
        await websocket_handler.connect(system_websocket, ConnectionType.SYSTEM_EVENTS)
        
        # 广播系统事件
        await websocket_handler.broadcast_system_event(
            event_type="strategy_update",
            severity="info",
            message="策略参数已更新",
            strategy_id="strategy_001"
        )
        
        # 验证消息
        assert len(system_websocket.messages) == 2
        event_message = json.loads(system_websocket.messages[1])
        assert event_message["type"] == "system_event"
        assert event_message["event_type"] == "strategy_update"
        assert event_message["severity"] == "info"
        assert event_message["message"] == "策略参数已更新"
    
    @pytest.mark.asyncio
    async def test_performance_update_broadcast(self, websocket_handler):
        """测试性能更新广播"""
        performance_websocket = MockWebSocket()
        await websocket_handler.connect(performance_websocket, ConnectionType.PERFORMANCE_UPDATES)
        
        # 广播性能更新
        await websocket_handler.broadcast_performance_update(
            trading_mode=TradingMode.PAPER,
            pnl=150.0,
            trades=10,
            win_rate=0.7
        )
        
        # 验证消息
        assert len(performance_websocket.messages) == 2
        perf_message = json.loads(performance_websocket.messages[1])
        assert perf_message["type"] == "performance_update"
        assert perf_message["trading_mode"] == TradingMode.PAPER
        assert perf_message["pnl"] == 150.0
        assert perf_message["trades"] == 10
    
    @pytest.mark.asyncio
    async def test_multiple_connection_types(self, websocket_handler):
        """测试多种连接类型同时工作"""
        # 创建不同类型的连接
        websockets = {}
        connection_types = [
            ConnectionType.MARKET_DATA,
            ConnectionType.ORDER_UPDATES,
            ConnectionType.SYSTEM_EVENTS,
            ConnectionType.PERFORMANCE_UPDATES,
            ConnectionType.ALL
        ]
        
        for conn_type in connection_types:
            ws = MockWebSocket()
            await websocket_handler.connect(ws, conn_type)
            websockets[conn_type] = ws
        
        # 广播不同类型的消息
        await websocket_handler.broadcast_market_data("BTCUSDT", 66000.0, 1500.0)
        await websocket_handler.broadcast_order_update("order_001", "FILLED", 0.1)
        await websocket_handler.broadcast_system_event("test", "info", "测试消息")
        await websocket_handler.broadcast_performance_update(TradingMode.PAPER, 100.0, 5)
        
        # 验证各连接收到正确的消息
        # MARKET_DATA连接应该收到: 欢迎消息 + 市场数据
        assert len(websockets[ConnectionType.MARKET_DATA].messages) == 2
        
        # ORDER_UPDATES连接应该收到: 欢迎消息 + 订单更新
        assert len(websockets[ConnectionType.ORDER_UPDATES].messages) == 2
        
        # SYSTEM_EVENTS连接应该收到: 欢迎消息 + 系统事件
        assert len(websockets[ConnectionType.SYSTEM_EVENTS].messages) == 2
        
        # PERFORMANCE_UPDATES连接应该收到: 欢迎消息 + 性能更新
        assert len(websockets[ConnectionType.PERFORMANCE_UPDATES].messages) == 2
        
        # ALL连接应该收到: 欢迎消息 + 所有4种消息类型
        assert len(websockets[ConnectionType.ALL].messages) == 5
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, websocket_handler):
        """测试连接错误处理"""
        # 创建一个会发送失败的WebSocket模拟
        class FailingWebSocket(MockWebSocket):
            async def send_text(self, message: str):
                raise Exception("Connection failed")
        
        failing_ws = FailingWebSocket()
        normal_ws = MockWebSocket()
        
        await websocket_handler.connect(failing_ws, ConnectionType.MARKET_DATA)
        await websocket_handler.connect(normal_ws, ConnectionType.MARKET_DATA)
        
        # 广播消息，应该自动移除失败的连接
        await websocket_handler.broadcast_market_data("BTCUSDT", 66000.0, 1500.0)
        
        # 验证失败的连接被移除，正常连接仍然工作
        assert failing_ws not in websocket_handler.active_connections[ConnectionType.MARKET_DATA]
        assert normal_ws in websocket_handler.active_connections[ConnectionType.MARKET_DATA]
        assert len(normal_ws.messages) == 2  # 欢迎消息 + 市场数据
    
    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, websocket_handler):
        """测试并发广播"""
        # 创建多个连接
        websockets = []
        for i in range(10):
            ws = MockWebSocket()
            await websocket_handler.connect(ws, ConnectionType.ALL)
            websockets.append(ws)
        
        # 并发广播多个消息
        tasks = []
        for i in range(20):
            task = asyncio.create_task(
                websocket_handler.broadcast_market_data(f"SYMBOL{i}", 100.0 + i, 1000.0)
            )
            tasks.append(task)
        
        # 等待所有广播完成
        await asyncio.gather(*tasks)
        
        # 验证所有连接都收到了所有消息
        for ws in websockets:
            assert len(ws.messages) == 21  # 欢迎消息 + 20个市场数据消息
    
    @pytest.mark.asyncio
    async def test_subscription_handling(self, websocket_handler):
        """测试订阅处理"""
        mock_websocket = MockWebSocket()
        
        # 模拟订阅消息处理
        subscribe_data = json.dumps({
            "type": "subscribe",
            "symbols": ["BTCUSDT", "ETHUSDT"]
        })
        
        await websocket_handler._handle_client_message(
            mock_websocket, subscribe_data, ConnectionType.MARKET_DATA
        )
        
        # 验证订阅确认消息
        assert len(mock_websocket.messages) == 1
        confirm_message = json.loads(mock_websocket.messages[0])
        assert confirm_message["type"] == "subscription_confirmed"
        assert confirm_message["data"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    
    @pytest.mark.asyncio
    async def test_heartbeat_handling(self, websocket_handler):
        """测试心跳处理"""
        mock_websocket = MockWebSocket()
        
        # 发送ping消息
        ping_data = json.dumps({"type": "ping"})
        await websocket_handler._handle_client_message(
            mock_websocket, ping_data, ConnectionType.MARKET_DATA
        )
        
        # 验证pong响应
        assert len(mock_websocket.messages) == 1
        pong_message = json.loads(mock_websocket.messages[0])
        assert pong_message["type"] == "pong"
        assert "server_time" in pong_message["data"]
    
    @pytest.mark.asyncio
    async def test_invalid_message_handling(self, websocket_handler):
        """测试无效消息处理"""
        mock_websocket = MockWebSocket()
        
        # 发送无效JSON
        await websocket_handler._handle_client_message(
            mock_websocket, "invalid json", ConnectionType.MARKET_DATA
        )
        
        # 验证错误响应
        assert len(mock_websocket.messages) == 1
        error_message = json.loads(mock_websocket.messages[0])
        assert error_message["type"] == "error"
        assert "Invalid JSON format" in error_message["data"]["error"]
    
    @pytest.mark.asyncio
    async def test_websocket_cleanup(self, websocket_handler):
        """测试WebSocket清理"""
        # 创建一些连接
        websockets = []
        for conn_type in [ConnectionType.MARKET_DATA, ConnectionType.ORDER_UPDATES]:
            ws = MockWebSocket()
            await websocket_handler.connect(ws, conn_type)
            websockets.append(ws)
        
        # 执行清理
        await websocket_handler.cleanup()
        
        # 验证所有连接都被关闭和清理
        for ws in websockets:
            assert ws.closed
        
        for connections in websocket_handler.active_connections.values():
            assert len(connections) == 0


class TestWebSocketPerformance:
    """WebSocket性能测试"""
    
    @pytest.mark.asyncio
    async def test_broadcast_performance(self):
        """测试广播性能"""
        handler = WebSocketHandler()
        await handler.initialize()
        
        try:
            # 创建大量连接
            websockets = []
            for i in range(100):
                ws = MockWebSocket()
                await handler.connect(ws, ConnectionType.MARKET_DATA)
                websockets.append(ws)
            
            # 测试广播性能
            start_time = time.time()
            
            for i in range(50):
                await handler.broadcast_market_data(f"SYMBOL{i}", 100.0 + i, 1000.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 计算性能指标
            total_messages = 100 * 50  # 100个连接 * 50个广播
            throughput = total_messages / duration
            
            print(f"广播性能: {throughput:.2f} 消息/秒")
            print(f"广播延迟: {duration/50:.4f} 秒/广播")
            
            # 性能应该足够好（这里设定一个合理的阈值）
            assert throughput > 1000  # 至少1000消息/秒
            
            # 验证所有连接都收到了所有消息
            for ws in websockets:
                assert len(ws.messages) == 51  # 欢迎消息 + 50个市场数据
                
        finally:
            await handler.cleanup()
    
    @pytest.mark.asyncio
    async def test_connection_scalability(self):
        """测试连接扩展性"""
        handler = WebSocketHandler()
        await handler.initialize()
        
        try:
            # 逐步增加连接数量并测试性能
            connection_counts = [10, 50, 100, 200]
            
            for count in connection_counts:
                # 创建连接
                websockets = []
                for i in range(count):
                    ws = MockWebSocket()
                    await handler.connect(ws, ConnectionType.ALL)
                    websockets.append(ws)
                
                # 测试广播时间
                start_time = time.time()
                await handler.broadcast_market_data("BTCUSDT", 66000.0, 1500.0)
                end_time = time.time()
                
                broadcast_time = end_time - start_time
                print(f"{count}个连接的广播时间: {broadcast_time:.4f}秒")
                
                # 广播时间应该保持在合理范围内
                assert broadcast_time < 0.1  # 100ms内完成
                
                # 清理连接
                for ws in websockets:
                    await handler.disconnect(ws, ConnectionType.ALL)
                    
        finally:
            await handler.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        handler = WebSocketHandler()
        await handler.initialize()
        
        try:
            # 创建大量连接和消息
            websockets = []
            for i in range(500):
                ws = MockWebSocket()
                await handler.connect(ws, ConnectionType.ALL)
                websockets.append(ws)
            
            # 发送大量消息
            for i in range(100):
                await handler.broadcast_market_data(f"SYMBOL{i}", 100.0, 1000.0)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"内存使用增加: {memory_increase:.2f}MB")
            
            # 内存增长应该在合理范围内
            assert memory_increase < 50  # 不超过50MB
            
        finally:
            await handler.cleanup()


# 运行测试的示例
if __name__ == "__main__":
    pytest.main([__file__, "-v"])