"""
API集成测试套件
测试认证、权限、速率限制和参数验证
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.api.trading_api import TradingAPI
from src.api.auth_manager import AuthManager
from src.api.rate_limiter import RateLimiter
from src.api.request_validator import RequestValidator
from src.core.database import DatabaseManager


class TestAPIAuthentication:
    """API认证集成测试"""
    
    @pytest_asyncio.fixture
    async def api_client(self):
        """创建API客户端"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {
                'secret_key': 'test_secret_key',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7
            },
            'rate_limiting': {
                'enabled': True,
                'default_rate': '100/minute',
                'burst_rate': '10/second'
            },
            'websocket': {
                'enabled': True,
                'host': '0.0.0.0',
                'port': 8765
            }
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(
            config=config,
            database=database
        )
        
        await trading_api.initialize()
        
        # 使用异步客户端进行测试
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            yield client, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_user_registration_and_authentication(self, api_client):
        """测试用户注册和认证流程"""
        client, trading_api = api_client
        
        # 1. 用户注册
        register_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'test_password_123',
            'full_name': 'Test User'
        }
        
        response = await client.post('/auth/register', json=register_data)
        assert response.status_code == status.HTTP_201_CREATED
        
        user_data = response.json()
        assert user_data['username'] == register_data['username']
        assert user_data['email'] == register_data['email']
        assert 'id' in user_data
        
        # 2. 用户登录
        login_data = {
            'username': register_data['username'],
            'password': register_data['password']
        }
        
        response = await client.post('/auth/login', json=login_data)
        assert response.status_code == status.HTTP_200_OK
        
        token_data = response.json()
        assert 'access_token' in token_data
        assert 'refresh_token' in token_data
        assert token_data['token_type'] == 'bearer'
        
        access_token = token_data['access_token']
        refresh_token = token_data['refresh_token']
        
        # 3. 使用访问令牌访问受保护的资源
        headers = {'Authorization': f'Bearer {access_token}'}
        response = await client.get('/auth/user-info', headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        user_info = response.json()
        assert user_info['username'] == register_data['username']
        
        # 4. 刷新令牌
        refresh_data = {'refresh_token': refresh_token}
        response = await client.post('/auth/refresh', json=refresh_data)
        assert response.status_code == status.HTTP_200_OK
        
        new_token_data = response.json()
        assert 'access_token' in new_token_data
        assert new_token_data['access_token'] != access_token  # 新令牌应该不同
    
    @pytest.mark.asyncio
    async def test_invalid_authentication_attempts(self, api_client):
        """测试无效认证尝试"""
        client, _ = api_client
        
        # 1. 无效的登录凭据
        invalid_login_data = {
            'username': 'nonexistent_user',
            'password': 'wrong_password'
        }
        
        response = await client.post('/auth/login', json=invalid_login_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # 2. 无效的访问令牌
        headers = {'Authorization': 'Bearer invalid_token'}
        response = await client.get('/auth/user-info', headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # 3. 缺失的认证头
        response = await client.get('/auth/user-info')
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # 4. 无效的刷新令牌
        refresh_data = {'refresh_token': 'invalid_refresh_token'}
        response = await client.post('/auth/refresh', json=refresh_data)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_token_expiration(self, api_client):
        """测试令牌过期处理"""
        client, trading_api = api_client
        
        # 创建短期过期的令牌配置
        auth_manager = trading_api.auth_manager
        
        # 注册和登录用户
        register_data = {
            'username': 'expiry_test_user',
            'email': 'expiry@example.com',
            'password': 'test_password_123'
        }
        
        await client.post('/auth/register', json=register_data)
        
        login_response = await client.post('/auth/login', json={
            'username': register_data['username'],
            'password': register_data['password']
        })
        
        token_data = login_response.json()
        access_token = token_data['access_token']
        
        # 模拟令牌过期
        with patch.object(auth_manager, 'verify_token') as mock_verify:
            mock_verify.side_effect = Exception("Token has expired")
            
            headers = {'Authorization': f'Bearer {access_token}'}
            response = await client.get('/auth/user-info', headers=headers)
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAPIRateLimit:
    """API速率限制集成测试"""
    
    @pytest_asyncio.fixture
    async def rate_limited_client(self):
        """创建启用速率限制的API客户端"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {
                'secret_key': 'test_secret_key',
                'access_token_expire_minutes': 30
            },
            'rate_limiting': {
                'enabled': True,
                'default_rate': '5/minute',  # 严格的速率限制用于测试
                'burst_rate': '2/second'
            },
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 创建测试用户和令牌
        await self._create_test_user(trading_api)
        token = await self._get_test_token(trading_api)
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            yield client, token, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    async def _create_test_user(self, trading_api):
        """创建测试用户"""
        user_data = {
            'username': 'rate_test_user',
            'email': 'rate@example.com',
            'password': 'test_password_123'
        }
        
        await trading_api.auth_manager.create_user(**user_data)
    
    async def _get_test_token(self, trading_api):
        """获取测试令牌"""
        login_data = {
            'username': 'rate_test_user',
            'password': 'test_password_123'
        }
        
        user = await trading_api.auth_manager.authenticate_user(
            login_data['username'],
            login_data['password']
        )
        
        token_data = await trading_api.auth_manager.create_tokens(user)
        return token_data['access_token']
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, rate_limited_client):
        """测试速率限制执行"""
        client, token, _ = rate_limited_client
        headers = {'Authorization': f'Bearer {token}'}
        
        # 发送多个请求以触发速率限制
        success_count = 0
        rate_limited_count = 0
        
        for i in range(10):
            response = await client.get('/auth/user-info', headers=headers)
            
            if response.status_code == status.HTTP_200_OK:
                success_count += 1
            elif response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                rate_limited_count += 1
                
                # 检查速率限制响应头
                assert 'X-RateLimit-Limit' in response.headers
                assert 'X-RateLimit-Remaining' in response.headers
                assert 'Retry-After' in response.headers
            
            # 小延迟以避免过快请求
            await asyncio.sleep(0.1)
        
        # 应该有一些请求成功，一些被速率限制
        assert success_count > 0
        assert rate_limited_count > 0
        
        print(f"成功请求: {success_count}, 被限制请求: {rate_limited_count}")
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limited_client):
        """测试速率限制重置"""
        client, token, _ = rate_limited_client
        headers = {'Authorization': f'Bearer {token}'}
        
        # 触发速率限制
        for _ in range(10):
            await client.get('/auth/user-info', headers=headers)
        
        # 等待速率限制窗口重置
        await asyncio.sleep(65)  # 等待超过1分钟
        
        # 现在应该可以成功请求
        response = await client.get('/auth/user-info', headers=headers)
        assert response.status_code == status.HTTP_200_OK


class TestAPIRequestValidation:
    """API请求验证集成测试"""
    
    @pytest_asyncio.fixture
    async def validation_client(self):
        """创建用于验证测试的API客户端"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'test_secret_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            yield client, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_request_validation_errors(self, validation_client):
        """测试请求验证错误"""
        client, _ = validation_client
        
        # 1. 无效的注册数据
        invalid_register_data = [
            # 缺失必需字段
            {'username': 'test'},
            # 无效邮箱
            {'username': 'test', 'email': 'invalid_email', 'password': 'pass123'},
            # 密码太短
            {'username': 'test', 'email': 'test@example.com', 'password': '123'},
            # 空值
            {'username': '', 'email': 'test@example.com', 'password': 'pass123'}
        ]
        
        for invalid_data in invalid_register_data:
            response = await client.post('/auth/register', json=invalid_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
            
            error_data = response.json()
            assert 'detail' in error_data
            assert isinstance(error_data['detail'], list)
    
    @pytest.mark.asyncio
    async def test_malformed_json_handling(self, validation_client):
        """测试格式错误的JSON处理"""
        client, _ = validation_client
        
        # 发送格式错误的JSON
        response = await client.post(
            '/auth/register',
            data='{"invalid": json}',  # 格式错误的JSON
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAPIPermissions:
    """API权限控制集成测试"""
    
    @pytest_asyncio.fixture
    async def permission_client(self):
        """创建用于权限测试的API客户端"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'test_secret_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 创建不同权限级别的用户
        users = await self._create_test_users(trading_api)
        tokens = await self._get_user_tokens(trading_api, users)
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            yield client, tokens, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    async def _create_test_users(self, trading_api):
        """创建测试用户"""
        users = []
        
        # 普通用户
        user_data = {
            'username': 'regular_user',
            'email': 'regular@example.com',
            'password': 'test_password_123'
        }
        user = await trading_api.auth_manager.create_user(**user_data)
        users.append(('regular', user))
        
        # 管理员用户
        admin_data = {
            'username': 'admin_user',
            'email': 'admin@example.com',
            'password': 'admin_password_123'
        }
        admin_user = await trading_api.auth_manager.create_user(**admin_data)
        # 设置管理员权限
        admin_user.permissions = ['admin', 'trading', 'monitoring']
        users.append(('admin', admin_user))
        
        return users
    
    async def _get_user_tokens(self, trading_api, users):
        """获取用户令牌"""
        tokens = {}
        
        for user_type, user in users:
            token_data = await trading_api.auth_manager.create_tokens(user)
            tokens[user_type] = token_data['access_token']
        
        return tokens
    
    @pytest.mark.asyncio
    async def test_admin_only_endpoints(self, permission_client):
        """测试仅管理员可访问的端点"""
        client, tokens, _ = permission_client
        
        admin_headers = {'Authorization': f'Bearer {tokens["admin"]}'}
        regular_headers = {'Authorization': f'Bearer {tokens["regular"]}'}
        
        # 管理员应该可以访问系统健康状态
        response = await client.get('/system/health', headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        
        # 普通用户应该被拒绝访问
        response = await client.get('/system/health', headers=regular_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_trading_permissions(self, permission_client):
        """测试交易权限"""
        client, tokens, _ = permission_client
        
        admin_headers = {'Authorization': f'Bearer {tokens["admin"]}'}
        
        # 策略操作需要交易权限
        strategy_data = {
            'strategy_name': 'test_strategy',
            'config': {'param1': 'value1'}
        }
        
        response = await client.post('/strategies/start', json=strategy_data, headers=admin_headers)
        # 由于没有实际的策略管理器，这里可能返回500或404，但不应该是403
        assert response.status_code != status.HTTP_403_FORBIDDEN


class TestAPIErrorHandling:
    """API错误处理集成测试"""
    
    @pytest_asyncio.fixture
    async def error_client(self):
        """创建用于错误处理测试的API客户端"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'test_secret_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            yield client, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_404_handling(self, error_client):
        """测试404错误处理"""
        client, _ = error_client
        
        response = await client.get('/nonexistent-endpoint')
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        error_data = response.json()
        assert 'detail' in error_data
        assert 'error_code' in error_data
        assert 'timestamp' in error_data
    
    @pytest.mark.asyncio
    async def test_method_not_allowed_handling(self, error_client):
        """测试方法不允许错误处理"""
        client, _ = error_client
        
        # 使用错误的HTTP方法
        response = await client.post('/auth/user-info')
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self, error_client):
        """测试服务器错误处理"""
        client, trading_api = error_client
        
        # 模拟服务器内部错误
        with patch.object(trading_api.auth_manager, 'create_user') as mock_create:
            mock_create.side_effect = Exception("Database connection failed")
            
            register_data = {
                'username': 'test_user',
                'email': 'test@example.com',
                'password': 'test_password_123'
            }
            
            response = await client.post('/auth/register', json=register_data)
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            
            error_data = response.json()
            assert 'detail' in error_data
            assert 'error_code' in error_data
            assert 'request_id' in error_data


if __name__ == "__main__":
    # 运行特定测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])