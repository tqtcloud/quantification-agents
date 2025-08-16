"""Web API测试"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.web.api import app
from src.web.models import (
    TradingMode, StrategyStatus, OrderStatus,
    StrategyControlRequest, OrderRequest, SystemConfigRequest
)

# 创建测试客户端
client = TestClient(app)

class TestWebAPI:
    """Web API测试类"""
    
    def test_health_check(self):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_api_info(self):
        """测试API信息接口"""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        
        assert data["version"] == "1.0.0"
        assert data["name"] == "Crypto Quantitative Trading System API"
        assert data["docs_url"] == "/api/docs"
        assert data["health_check_url"] == "/health"
        assert data["websocket_url"] == "/ws"
        assert TradingMode.PAPER in data["supported_trading_modes"]
        assert TradingMode.LIVE in data["supported_trading_modes"]
        assert "binance" in data["supported_exchanges"]
    
    def test_get_positions(self):
        """测试获取仓位列表"""
        # 测试无筛选条件
        response = client.get("/api/positions")
        assert response.status_code == 200
        positions = response.json()
        assert isinstance(positions, list)
        
        # 测试按交易模式筛选
        response = client.get("/api/positions", params={"trading_mode": TradingMode.PAPER})
        assert response.status_code == 200
        
        # 测试按交易对筛选
        response = client.get("/api/positions", params={"symbol": "BTCUSDT"})
        assert response.status_code == 200
    
    def test_get_position(self):
        """测试获取单个仓位"""
        symbol = "BTCUSDT"
        response = client.get(f"/api/positions/{symbol}", params={"trading_mode": TradingMode.PAPER})
        assert response.status_code == 200
        
        position = response.json()
        assert position["symbol"] == symbol
        assert position["trading_mode"] == TradingMode.PAPER
        assert "size" in position
        assert "entry_price" in position
        assert "unrealized_pnl" in position
    
    def test_get_orders(self):
        """测试获取订单列表"""
        # 测试无筛选条件
        response = client.get("/api/orders")
        assert response.status_code == 200
        orders = response.json()
        assert isinstance(orders, list)
        
        # 测试带筛选条件
        params = {
            "trading_mode": TradingMode.PAPER,
            "symbol": "BTCUSDT",
            "status": OrderStatus.PENDING,
            "limit": 50
        }
        response = client.get("/api/orders", params=params)
        assert response.status_code == 200
    
    def test_get_order(self):
        """测试获取单个订单"""
        order_id = "test_order_001"
        response = client.get(f"/api/orders/{order_id}")
        assert response.status_code == 200
        
        order = response.json()
        assert order["order_id"] == order_id
        assert "symbol" in order
        assert "status" in order
        assert "quantity" in order
    
    def test_create_order(self):
        """测试创建订单"""
        order_data = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": 0.1,
            "price": 65000.0,
            "trading_mode": TradingMode.PAPER
        }
        
        response = client.post("/api/orders", json=order_data)
        assert response.status_code == 200
        
        order = response.json()
        assert order["symbol"] == order_data["symbol"]
        assert order["side"] == order_data["side"]
        assert order["status"] == OrderStatus.SUBMITTED
        assert "order_id" in order
    
    def test_create_batch_orders(self):
        """测试批量创建订单"""
        batch_data = {
            "trading_mode": TradingMode.PAPER,
            "orders": [
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "order_type": "LIMIT",
                    "quantity": 0.1,
                    "price": 65000.0
                },
                {
                    "symbol": "ETHUSDT",
                    "side": "SELL",
                    "order_type": "MARKET",
                    "quantity": 1.0
                }
            ]
        }
        
        response = client.post("/api/orders/batch", json=batch_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["success_count"] >= 0
        assert result["failed_count"] >= 0
        assert isinstance(result["orders"], list)
        assert isinstance(result["errors"], list)
    
    def test_cancel_order(self):
        """测试取消订单"""
        order_id = "test_order_001"
        response = client.delete(f"/api/orders/{order_id}")
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
    
    def test_get_performance(self):
        """测试获取性能指标"""
        params = {
            "trading_mode": TradingMode.PAPER,
            "period_days": 30
        }
        
        response = client.get("/api/performance", params=params)
        assert response.status_code == 200
        
        performance = response.json()
        assert "total_pnl" in performance
        assert "daily_pnl" in performance
        assert "win_rate" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
        assert "total_trades" in performance
        assert performance["trading_mode"] == TradingMode.PAPER
    
    def test_get_performance_history(self):
        """测试获取历史性能数据"""
        start_date = (datetime.now() - timedelta(days=30)).isoformat()
        end_date = datetime.now().isoformat()
        
        params = {
            "trading_mode": TradingMode.PAPER,
            "start_date": start_date,
            "end_date": end_date
        }
        
        response = client.get("/api/performance/history", params=params)
        assert response.status_code == 200
        
        history = response.json()
        assert isinstance(history, list)
    
    def test_get_strategies(self):
        """测试获取策略列表"""
        response = client.get("/api/strategies")
        assert response.status_code == 200
        
        strategies = response.json()
        assert isinstance(strategies, list)
        
        if strategies:
            strategy = strategies[0]
            assert "strategy_id" in strategy
            assert "name" in strategy
            assert "status" in strategy
            assert "trading_mode" in strategy
    
    def test_get_strategy(self):
        """测试获取单个策略"""
        strategy_id = "strategy_001"
        response = client.get(f"/api/strategies/{strategy_id}")
        assert response.status_code == 200
        
        strategy = response.json()
        assert strategy["strategy_id"] == strategy_id
        assert "name" in strategy
        assert "status" in strategy
    
    def test_control_strategy(self):
        """测试策略控制"""
        strategy_id = "strategy_001"
        control_data = {
            "action": "start",
            "trading_mode": TradingMode.PAPER,
            "config": {"period": 14}
        }
        
        response = client.post(f"/api/strategies/{strategy_id}/control", json=control_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
    
    def test_update_strategy_config(self):
        """测试更新策略配置"""
        strategy_id = "strategy_001"
        config_data = {"period": 21, "threshold": 0.8}
        
        response = client.put(f"/api/strategies/{strategy_id}/config", json=config_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
    
    def test_get_agents(self):
        """测试获取Agent状态列表"""
        response = client.get("/api/agents")
        assert response.status_code == 200
        
        agents = response.json()
        assert isinstance(agents, list)
        
        if agents:
            agent = agents[0]
            assert "agent_id" in agent
            assert "name" in agent
            assert "status" in agent
            assert "health_score" in agent
    
    def test_get_agent_status(self):
        """测试获取单个Agent状态"""
        agent_id = "technical_analysis_agent"
        response = client.get(f"/api/agents/{agent_id}")
        assert response.status_code == 200
        
        agent = response.json()
        assert agent["agent_id"] == agent_id
        assert "status" in agent
        assert "health_score" in agent
    
    def test_get_system_health(self):
        """测试获取系统健康状态"""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        
        health = response.json()
        assert health["status"] == "healthy"
        assert "uptime_seconds" in health
        assert "cpu_usage" in health
        assert "memory_usage_mb" in health
        assert "network_latency_ms" in health
    
    def test_get_system_config(self):
        """测试获取系统配置"""
        response = client.get("/api/system/config")
        assert response.status_code == 200
        
        config = response.json()
        assert isinstance(config, dict)
        assert "max_position_size" in config
    
    def test_update_system_config(self):
        """测试更新系统配置"""
        config_data = {
            "config_key": "max_position_size",
            "config_value": 0.2,
            "trading_mode": TradingMode.PAPER
        }
        
        response = client.post("/api/system/config", json=config_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
    
    def test_get_market_data(self):
        """测试获取市场数据"""
        # 测试无筛选条件
        response = client.get("/api/market/data")
        assert response.status_code == 200
        
        market_data = response.json()
        assert isinstance(market_data, list)
        
        # 测试指定交易对
        params = {
            "symbols": "BTCUSDT,ETHUSDT",
            "limit": 5
        }
        response = client.get("/api/market/data", params=params)
        assert response.status_code == 200
        
        market_data = response.json()
        assert len(market_data) <= 5
        
        if market_data:
            data = market_data[0]
            assert "symbol" in data
            assert "price" in data
            assert "volume" in data
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试不存在的端点
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # 测试不存在的资源
        response = client.get("/api/positions/INVALID")
        assert response.status_code == 404
        
        # 测试无效的订单数据
        invalid_order = {
            "symbol": "",  # 空符号
            "side": "INVALID",  # 无效方向
            "order_type": "LIMIT",
            "quantity": -1  # 负数量
        }
        response = client.post("/api/orders", json=invalid_order)
        assert response.status_code == 422  # 数据验证错误
    
    def test_api_response_formats(self):
        """测试API响应格式"""
        # 测试JSON响应
        response = client.get("/api/positions")
        assert response.headers["content-type"] == "application/json"
        
        # 测试数据结构
        positions = response.json()
        if positions:
            position = positions[0]
            required_fields = [
                "symbol", "side", "size", "entry_price", 
                "mark_price", "unrealized_pnl", "margin", 
                "percentage", "trading_mode", "last_updated"
            ]
            for field in required_fields:
                assert field in position


class TestWebSocketAPI:
    """WebSocket API测试类"""
    
    @pytest.mark.asyncio
    async def test_websocket_market_data_connection(self):
        """测试市场数据WebSocket连接"""
        with client.websocket_connect("/ws/market") as websocket:
            # 接收欢迎消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection_established"
            assert message["data"]["connection_type"] == "market_data"
            
            # 发送ping测试心跳
            websocket.send_text(json.dumps({"type": "ping"}))
            
            # 接收pong响应
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_order_updates_connection(self):
        """测试订单更新WebSocket连接"""
        with client.websocket_connect("/ws/orders") as websocket:
            # 接收欢迎消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection_established"
            assert message["data"]["connection_type"] == "order_updates"
    
    @pytest.mark.asyncio
    async def test_websocket_system_events_connection(self):
        """测试系统事件WebSocket连接"""
        with client.websocket_connect("/ws/system") as websocket:
            # 接收欢迎消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection_established"
            assert message["data"]["connection_type"] == "system_events"
    
    @pytest.mark.asyncio
    async def test_websocket_performance_updates_connection(self):
        """测试性能更新WebSocket连接"""
        with client.websocket_connect("/ws/performance") as websocket:
            # 接收欢迎消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection_established"
            assert message["data"]["connection_type"] == "performance_updates"
    
    @pytest.mark.asyncio
    async def test_websocket_all_updates_connection(self):
        """测试全部更新WebSocket连接"""
        with client.websocket_connect("/ws/all") as websocket:
            # 接收欢迎消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "connection_established"
            assert message["data"]["connection_type"] == "all"
    
    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """测试WebSocket订阅功能"""
        with client.websocket_connect("/ws/market") as websocket:
            # 跳过欢迎消息
            websocket.receive_text()
            
            # 发送订阅请求
            subscribe_message = {
                "type": "subscribe",
                "symbols": ["BTCUSDT", "ETHUSDT"]
            }
            websocket.send_text(json.dumps(subscribe_message))
            
            # 接收订阅确认
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "subscription_confirmed"
            assert message["data"]["symbols"] == ["BTCUSDT", "ETHUSDT"]
    
    @pytest.mark.asyncio
    async def test_websocket_unsubscription(self):
        """测试WebSocket取消订阅功能"""
        with client.websocket_connect("/ws/market") as websocket:
            # 跳过欢迎消息
            websocket.receive_text()
            
            # 发送取消订阅请求
            unsubscribe_message = {
                "type": "unsubscribe",
                "symbols": ["BTCUSDT"]
            }
            websocket.send_text(json.dumps(unsubscribe_message))
            
            # 接收取消订阅确认
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "unsubscription_confirmed"
            assert message["data"]["symbols"] == ["BTCUSDT"]
    
    @pytest.mark.asyncio
    async def test_websocket_invalid_message(self):
        """测试WebSocket无效消息处理"""
        with client.websocket_connect("/ws/market") as websocket:
            # 跳过欢迎消息
            websocket.receive_text()
            
            # 发送无效JSON
            websocket.send_text("invalid json")
            
            # 接收错误消息
            data = websocket.receive_text()
            message = json.loads(data)
            assert message["type"] == "error"
            assert "Invalid JSON format" in message["data"]["error"]


class TestWebAPIPerformance:
    """Web API性能测试类"""
    
    def test_concurrent_api_requests(self):
        """测试并发API请求"""
        import concurrent.futures
        import time
        
        def make_request():
            response = client.get("/api/positions")
            return response.status_code == 200
        
        # 并发执行50个请求
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 所有请求都应该成功
        assert all(results)
        # 并发请求应该在合理时间内完成（这里设为10秒）
        assert duration < 10.0
        
        print(f"50个并发请求完成时间: {duration:.2f}秒")
    
    def test_api_response_time(self):
        """测试API响应时间"""
        import time
        
        endpoints = [
            "/api/positions",
            "/api/orders",
            "/api/performance",
            "/api/strategies",
            "/api/agents",
            "/api/system/health",
            "/api/market/data"
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()
            
            duration = end_time - start_time
            
            assert response.status_code == 200
            # 每个请求应该在1秒内完成
            assert duration < 1.0
            
            print(f"{endpoint} 响应时间: {duration:.3f}秒")
    
    def test_large_data_handling(self):
        """测试大数据量处理"""
        # 测试获取大量订单
        response = client.get("/api/orders", params={"limit": 1000})
        assert response.status_code == 200
        
        # 测试批量订单创建（模拟大量订单）
        orders = []
        for i in range(10):  # 创建10个订单
            orders.append({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "order_type": "LIMIT",
                "quantity": 0.1,
                "price": 65000.0 + i
            })
        
        batch_data = {
            "trading_mode": TradingMode.PAPER,
            "orders": orders
        }
        
        response = client.post("/api/orders/batch", json=batch_data)
        assert response.status_code == 200
        
        result = response.json()
        assert result["success_count"] + result["failed_count"] == len(orders)
    
    @pytest.mark.asyncio
    async def test_websocket_connection_stability(self):
        """测试WebSocket连接稳定性"""
        connection_count = 5
        connections = []
        
        try:
            # 创建多个WebSocket连接
            for i in range(connection_count):
                websocket = client.websocket_connect("/ws/all")
                websocket.__enter__()
                connections.append(websocket)
                
                # 接收欢迎消息
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["type"] == "connection_established"
            
            # 等待一段时间，确保连接稳定
            await asyncio.sleep(2)
            
            # 测试每个连接是否仍然活跃
            for i, websocket in enumerate(connections):
                websocket.send_text(json.dumps({"type": "ping"}))
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["type"] == "pong"
                
        finally:
            # 清理连接
            for websocket in connections:
                try:
                    websocket.__exit__(None, None, None)
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self):
        """测试WebSocket消息吞吐量"""
        import time
        
        with client.websocket_connect("/ws/market") as websocket:
            # 跳过欢迎消息
            websocket.receive_text()
            
            # 发送大量ping消息并计时
            message_count = 100
            start_time = time.time()
            
            for i in range(message_count):
                websocket.send_text(json.dumps({"type": "ping"}))
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["type"] == "pong"
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 计算消息处理速率
            throughput = message_count / duration
            print(f"WebSocket消息吞吐量: {throughput:.2f} 消息/秒")
            
            # 应该能够处理至少10消息/秒
            assert throughput > 10.0


# 运行测试的示例
if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])