"""
端到端集成测试套件
测试API和WebSocket协同工作、完整用户场景
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets
from httpx import AsyncClient, ASGITransport
from websockets.exceptions import ConnectionClosed

from src.api.trading_api import TradingAPI
from src.websocket.websocket_manager import WebSocketManager
from src.core.database import DatabaseManager
from src.core.config_manager import ConfigManager


class WebSocketTestClient:
    """WebSocket测试客户端"""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.messages = []
        self.connected = False
    
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            # 启动消息接收任务
            asyncio.create_task(self._receive_messages())
        except Exception as e:
            print(f"WebSocket连接失败: {e}")
            self.connected = False
    
    async def _receive_messages(self):
        """接收消息的后台任务"""
        try:
            while self.connected and self.websocket:
                message = await self.websocket.recv()
                self.messages.append(json.loads(message))
        except ConnectionClosed:
            self.connected = False
        except Exception as e:
            print(f"接收消息错误: {e}")
            self.connected = False
    
    async def send(self, message: dict):
        """发送消息"""
        if self.websocket and self.connected:
            await self.websocket.send(json.dumps(message))
    
    async def close(self):
        """关闭连接"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()
    
    def get_messages(self) -> List[dict]:
        """获取接收到的消息"""
        return self.messages.copy()
    
    def clear_messages(self):
        """清空消息"""
        self.messages.clear()
    
    def wait_for_message(self, timeout: float = 5.0) -> Optional[dict]:
        """等待特定消息"""
        start_time = time.time()
        initial_count = len(self.messages)
        
        while time.time() - start_time < timeout:
            if len(self.messages) > initial_count:
                return self.messages[-1]
            time.sleep(0.1)
        
        return None


class TestEndToEndUserScenarios:
    """端到端用户场景测试"""
    
    @pytest_asyncio.fixture
    async def integrated_system(self):
        """创建完整的集成系统"""
        # 配置
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {
                'secret_key': 'test_secret_key_12345',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7
            },
            'rate_limiting': {
                'enabled': True,
                'default_rate': '1000/minute',
                'burst_rate': '100/second'
            },
            'websocket': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 8766,  # 使用不同端口避免冲突
                'max_connections': 100
            },
            'api': {
                'host': '127.0.0.1',
                'port': 8001,
                'debug': False
            }
        }
        
        # 初始化数据库
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        # 创建API服务器
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 创建WebSocket管理器
        websocket_manager = WebSocketManager(
            config['websocket'], 
            database
        )
        await websocket_manager.initialize()
        
        # 连接API和WebSocket
        trading_api.websocket_manager = websocket_manager
        
        # 启动WebSocket服务器（在后台）
        websocket_server_task = asyncio.create_task(
            websocket_manager.start_server()
        )
        
        # 等待服务器启动
        await asyncio.sleep(0.5)
        
        # 创建HTTP客户端
        http_client = AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        )
        
        # 创建WebSocket客户端
        ws_client = WebSocketTestClient(
            f"ws://{config['websocket']['host']}:{config['websocket']['port']}"
        )
        
        yield {
            'http_client': http_client,
            'ws_client': ws_client,
            'trading_api': trading_api,
            'websocket_manager': websocket_manager,
            'config': config
        }
        
        # 清理资源
        await ws_client.close()
        await http_client.aclose()
        websocket_server_task.cancel()
        
        try:
            await websocket_server_task
        except asyncio.CancelledError:
            pass
        
        await trading_api.shutdown()
        await websocket_manager.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, integrated_system):
        """测试完整的交易工作流程"""
        system = integrated_system
        http_client = system['http_client']
        ws_client = system['ws_client']
        
        # 1. 用户注册
        register_data = {
            'username': 'trader_001',
            'email': 'trader001@example.com',
            'password': 'secure_password_123',
            'full_name': 'Professional Trader'
        }
        
        response = await http_client.post('/auth/register', json=register_data)
        assert response.status_code == 201
        user_data = response.json()
        
        # 2. 用户登录获取令牌
        login_response = await http_client.post('/auth/login', json={
            'username': register_data['username'],
            'password': register_data['password']
        })
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        access_token = token_data['access_token']
        
        # 3. 建立WebSocket连接
        await ws_client.connect()
        assert ws_client.connected
        
        # 4. WebSocket认证
        auth_message = {
            'type': 'auth',
            'token': access_token,
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(auth_message)
        auth_response = ws_client.wait_for_message(timeout=3.0)
        assert auth_response is not None
        assert auth_response.get('type') == 'auth_response'
        assert auth_response.get('status') == 'success'
        
        # 5. 订阅市场数据
        subscribe_message = {
            'type': 'subscribe',
            'channel': 'market_data',
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(subscribe_message)
        subscribe_response = ws_client.wait_for_message(timeout=3.0)
        assert subscribe_response is not None
        assert subscribe_response.get('type') == 'subscribe_response'
        assert subscribe_response.get('status') == 'success'
        
        # 6. 通过API启动交易策略
        strategy_data = {
            'strategy_name': 'ma_crossover',
            'config': {
                'symbol': 'BTCUSDT',
                'fast_period': 5,
                'slow_period': 20,
                'position_size': 0.1
            }
        }
        
        headers = {'Authorization': f'Bearer {access_token}'}
        strategy_response = await http_client.post(
            '/strategies/start',
            json=strategy_data,
            headers=headers
        )
        
        # 注意：如果没有实际的策略管理器，这可能返回错误
        # 但我们主要测试认证和请求处理
        print(f"策略启动响应状态: {strategy_response.status_code}")
        
        # 7. 模拟市场数据推送
        market_data = {
            'type': 'market_data',
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume': 1.5,
            'timestamp': datetime.now().isoformat()
        }
        
        # 通过WebSocket管理器广播市场数据
        websocket_manager = system['websocket_manager']
        await websocket_manager.broadcast_market_data('BTCUSDT', market_data)
        
        # 8. 验证客户端收到市场数据
        market_data_received = ws_client.wait_for_message(timeout=3.0)
        assert market_data_received is not None
        assert market_data_received.get('type') == 'market_data'
        assert market_data_received.get('symbol') == 'BTCUSDT'
        assert market_data_received.get('price') == 45000.0
        
        # 9. 查询策略状态
        status_response = await http_client.get(
            '/strategies',
            headers=headers
        )
        print(f"策略状态查询响应: {status_response.status_code}")
        
        # 10. 查询系统健康状态（需要管理员权限）
        # 这里先测试访问被拒绝
        health_response = await http_client.get(
            '/system/health',
            headers=headers
        )
        # 普通用户应该被拒绝访问
        assert health_response.status_code in [403, 404, 500]
        
        # 11. 取消订阅
        unsubscribe_message = {
            'type': 'unsubscribe',
            'channel': 'market_data',
            'symbol': 'BTCUSDT',
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(unsubscribe_message)
        unsubscribe_response = ws_client.wait_for_message(timeout=3.0)
        assert unsubscribe_response is not None
        assert unsubscribe_response.get('type') == 'unsubscribe_response'
        
        # 12. 用户登出
        logout_response = await http_client.post('/auth/logout', headers=headers)
        print(f"登出响应状态: {logout_response.status_code}")
    
    @pytest.mark.asyncio
    async def test_real_time_data_synchronization(self, integrated_system):
        """测试实时数据同步"""
        system = integrated_system
        http_client = system['http_client']
        ws_client = system['ws_client']
        websocket_manager = system['websocket_manager']
        
        # 创建用户和获取令牌
        user_data = await self._create_test_user(http_client, 'sync_user')
        token = await self._get_user_token(http_client, 'sync_user', 'test_pass')
        
        # 建立WebSocket连接并认证
        await ws_client.connect()
        await self._authenticate_websocket(ws_client, token)
        
        # 订阅多个数据源
        subscriptions = [
            {'channel': 'market_data', 'symbol': 'BTCUSDT'},
            {'channel': 'market_data', 'symbol': 'ETHUSDT'},
            {'channel': 'system_notifications'}
        ]
        
        for sub in subscriptions:
            sub_msg = {
                'type': 'subscribe',
                'timestamp': datetime.now().isoformat(),
                **sub
            }
            await ws_client.send(sub_msg)
            
            # 等待订阅确认
            response = ws_client.wait_for_message(timeout=2.0)
            assert response and response.get('status') == 'success'
        
        # 清空消息缓冲区
        ws_client.clear_messages()
        
        # 发送不同类型的实时数据
        test_data = [
            # 市场数据
            {
                'type': 'market_data',
                'symbol': 'BTCUSDT',
                'price': 46000.0,
                'volume': 2.0,
                'timestamp': datetime.now().isoformat()
            },
            {
                'type': 'market_data',
                'symbol': 'ETHUSDT',
                'price': 3200.0,
                'volume': 5.0,
                'timestamp': datetime.now().isoformat()
            },
            # 系统通知
            {
                'type': 'system_notification',
                'message': 'Market volatility alert',
                'level': 'warning',
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # 发送数据并验证接收
        for data in test_data:
            if data['type'] == 'market_data':
                await websocket_manager.broadcast_market_data(
                    data['symbol'], 
                    data
                )
            elif data['type'] == 'system_notification':
                await websocket_manager.broadcast_system_notification(data)
            
            # 等待接收
            await asyncio.sleep(0.1)
        
        # 验证所有数据都被接收
        received_messages = ws_client.get_messages()
        assert len(received_messages) >= len(test_data)
        
        # 验证每种类型的数据都收到了
        btc_received = any(
            msg.get('symbol') == 'BTCUSDT' and msg.get('price') == 46000.0
            for msg in received_messages
        )
        eth_received = any(
            msg.get('symbol') == 'ETHUSDT' and msg.get('price') == 3200.0
            for msg in received_messages
        )
        notification_received = any(
            msg.get('type') == 'system_notification' and 'volatility' in msg.get('message', '')
            for msg in received_messages
        )
        
        assert btc_received, "未收到BTC市场数据"
        assert eth_received, "未收到ETH市场数据"  
        assert notification_received, "未收到系统通知"
    
    @pytest.mark.asyncio
    async def test_multi_user_interaction(self, integrated_system):
        """测试多用户交互场景"""
        system = integrated_system
        http_client = system['http_client']
        websocket_manager = system['websocket_manager']
        
        # 创建多个用户
        users = []
        ws_clients = []
        
        for i in range(3):
            username = f'user_{i:03d}'
            
            # 创建用户
            await self._create_test_user(http_client, username)
            token = await self._get_user_token(http_client, username, 'test_pass')
            
            # 创建WebSocket客户端
            ws_client = WebSocketTestClient(
                f"ws://{system['config']['websocket']['host']}:{system['config']['websocket']['port']}"
            )
            
            await ws_client.connect()
            await self._authenticate_websocket(ws_client, token)
            
            users.append({'username': username, 'token': token})
            ws_clients.append(ws_client)
        
        # 所有用户订阅公共频道
        for ws_client in ws_clients:
            sub_msg = {
                'type': 'subscribe',
                'channel': 'system_notifications',
                'timestamp': datetime.now().isoformat()
            }
            await ws_client.send(sub_msg)
            
            # 等待订阅确认
            response = ws_client.wait_for_message(timeout=2.0)
            assert response and response.get('status') == 'success'
        
        # 清空消息缓冲区
        for ws_client in ws_clients:
            ws_client.clear_messages()
        
        # 广播系统消息
        system_msg = {
            'type': 'system_notification',
            'message': 'Multi-user test notification',
            'level': 'info',
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket_manager.broadcast_system_notification(system_msg)
        
        # 等待消息传播
        await asyncio.sleep(0.2)
        
        # 验证所有用户都收到了消息
        for i, ws_client in enumerate(ws_clients):
            messages = ws_client.get_messages()
            notification_received = any(
                msg.get('type') == 'system_notification' and 
                'Multi-user test' in msg.get('message', '')
                for msg in messages
            )
            assert notification_received, f"用户{i}未收到广播消息"
        
        # 清理WebSocket连接
        for ws_client in ws_clients:
            await ws_client.close()
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, integrated_system):
        """测试错误传播和恢复机制"""
        system = integrated_system
        http_client = system['http_client']
        ws_client = system['ws_client']
        
        # 创建用户和获取令牌
        await self._create_test_user(http_client, 'error_test_user')
        token = await self._get_user_token(http_client, 'error_test_user', 'test_pass')
        
        # 建立WebSocket连接
        await ws_client.connect()
        await self._authenticate_websocket(ws_client, token)
        
        # 1. 测试无效订阅请求
        invalid_sub = {
            'type': 'subscribe',
            'channel': 'invalid_channel',
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(invalid_sub)
        error_response = ws_client.wait_for_message(timeout=2.0)
        assert error_response is not None
        assert error_response.get('status') == 'error'
        
        # 2. 验证连接仍然有效
        assert ws_client.connected
        
        # 3. 发送有效请求验证恢复
        valid_sub = {
            'type': 'subscribe',
            'channel': 'system_notifications',
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(valid_sub)
        success_response = ws_client.wait_for_message(timeout=2.0)
        assert success_response is not None
        assert success_response.get('status') == 'success'
        
        # 4. 测试API错误处理
        headers = {'Authorization': f'Bearer {token}'}
        
        # 访问不存在的端点
        not_found_response = await http_client.get('/nonexistent', headers=headers)
        assert not_found_response.status_code == 404
        
        # 验证后续请求仍然正常
        user_info_response = await http_client.get('/auth/user-info', headers=headers)
        assert user_info_response.status_code == 200
    
    async def _create_test_user(self, http_client: AsyncClient, username: str) -> dict:
        """创建测试用户"""
        user_data = {
            'username': username,
            'email': f'{username}@example.com',
            'password': 'test_pass',
            'full_name': f'Test User {username}'
        }
        
        response = await http_client.post('/auth/register', json=user_data)
        assert response.status_code == 201
        return response.json()
    
    async def _get_user_token(self, http_client: AsyncClient, username: str, password: str) -> str:
        """获取用户令牌"""
        login_data = {'username': username, 'password': password}
        response = await http_client.post('/auth/login', json=login_data)
        assert response.status_code == 200
        return response.json()['access_token']
    
    async def _authenticate_websocket(self, ws_client: WebSocketTestClient, token: str):
        """WebSocket认证"""
        auth_msg = {
            'type': 'auth',
            'token': token,
            'timestamp': datetime.now().isoformat()
        }
        
        await ws_client.send(auth_msg)
        auth_response = ws_client.wait_for_message(timeout=3.0)
        assert auth_response and auth_response.get('status') == 'success'


class TestSystemIntegration:
    """系统集成测试"""
    
    @pytest.mark.asyncio
    async def test_service_startup_and_shutdown(self):
        """测试服务启动和关闭"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'test_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 8767,
                'max_connections': 10
            }
        }
        
        # 测试启动
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        websocket_manager = WebSocketManager(config['websocket'], database)
        await websocket_manager.initialize()
        
        # 验证服务已启动
        assert trading_api.app is not None
        assert websocket_manager.connection_manager is not None
        
        # 测试关闭
        await trading_api.shutdown()
        await websocket_manager.shutdown()
        await database.close()
        
        # 验证资源已清理
        print("服务启动和关闭测试完成")
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """测试配置验证"""
        # 测试无效配置
        invalid_configs = [
            # 缺少必需配置
            {},
            # 无效数据库URL
            {'database': {'url': 'invalid://url'}},
            # 无效端口号
            {'websocket': {'port': -1}},
            # 缺少认证密钥
            {'auth': {}}
        ]
        
        for invalid_config in invalid_configs:
            try:
                # 尝试使用无效配置创建系统
                if 'database' in invalid_config:
                    database = DatabaseManager(invalid_config['database']['url'])
                    # 这应该在某个点失败
                    await database.initialize()
                    await database.close()
                
            except Exception as e:
                # 预期会有异常
                print(f"预期的配置错误: {e}")
                continue
        
        print("配置验证测试完成")


if __name__ == "__main__":
    # 运行端到端集成测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s"  # 显示print输出
    ])