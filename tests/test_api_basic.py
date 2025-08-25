"""
基础API测试

测试API的基本功能，用于验证系统集成
"""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def test_import_api_components():
    """测试API组件导入"""
    try:
        from src.api.trading_api import TradingAPI
        from src.api.auth_manager import AuthenticationManager
        from src.api.rate_limiter import RateLimiter
        from src.api.request_validator import RequestValidator
        from src.api.models.auth_models import UserRole, Permission
        from src.api.models.request_models import StrategyStartRequest
        from src.api.models.response_models import APIResponse
        from src.api.models.error_models import ValidationError
        
        assert True, "All API components imported successfully"
        
    except ImportError as e:
        pytest.fail(f"Failed to import API components: {e}")


def test_config_creation():
    """测试配置创建"""
    try:
        from src.config import Config
        
        config = Config({
            'api': {
                'host': '127.0.0.1',
                'port': 8000
            }
        })
        
        assert config.get('api.host') == '127.0.0.1'
        assert config.get('api.port') == 8000
        
    except Exception as e:
        pytest.fail(f"Failed to create config: {e}")


@pytest.mark.asyncio
async def test_api_creation():
    """测试API实例创建"""
    try:
        from src.api.trading_api import TradingAPI
        from src.config import Config
        
        config = Config({
            'api': {
                'host': '127.0.0.1',
                'port': 8000,
                'debug': True
            },
            'auth': {
                'jwt_secret_key': 'test_secret_key'
            }
        })
        
        # 模拟数据库
        with patch('src.api.trading_api.Database') as mock_db:
            mock_db.return_value.initialize = AsyncMock()
            mock_db.return_value.close = AsyncMock()
            
            api = TradingAPI(config)
            assert api is not None
            assert api.host == '127.0.0.1'
            assert api.port == 8000
            
    except Exception as e:
        pytest.fail(f"Failed to create API instance: {e}")


def test_auth_models():
    """测试认证模型"""
    try:
        from src.api.models.auth_models import LoginRequest, UserRole, Permission
        
        # 测试登录请求模型
        login_req = LoginRequest(username="test", password="password123")
        assert login_req.username == "test"
        assert login_req.password == "password123"
        
        # 测试枚举
        assert UserRole.ADMIN == "admin"
        assert Permission.STRATEGY_START == "strategy:start"
        
    except Exception as e:
        pytest.fail(f"Failed to test auth models: {e}")


def test_request_models():
    """测试请求模型"""
    try:
        from src.api.models.request_models import StrategyStartRequest, StrategyType
        
        req = StrategyStartRequest(
            strategy_id="test_strategy",
            strategy_type=StrategyType.HFT
        )
        
        assert req.strategy_id == "test_strategy"
        assert req.strategy_type == StrategyType.HFT
        
    except Exception as e:
        pytest.fail(f"Failed to test request models: {e}")


def test_response_models():
    """测试响应模型"""
    try:
        from src.api.models.response_models import APIResponse, ResponseStatus
        
        response = APIResponse(
            status=ResponseStatus.SUCCESS,
            message="Test message",
            data={"test": "data"}
        )
        
        assert response.status == ResponseStatus.SUCCESS
        assert response.message == "Test message"
        assert response.data == {"test": "data"}
        
    except Exception as e:
        pytest.fail(f"Failed to test response models: {e}")


def test_error_models():
    """测试错误模型"""
    try:
        from src.api.models.error_models import ValidationError, ErrorCode
        
        error = ValidationError(
            message="Test validation error",
            field_errors=[]
        )
        
        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.message == "Test validation error"
        
    except Exception as e:
        pytest.fail(f"Failed to test error models: {e}")


@pytest.mark.asyncio  
async def test_auth_manager_creation():
    """测试认证管理器创建"""
    try:
        from src.api.auth_manager import AuthenticationManager
        from src.config import Config
        
        config = Config({
            'auth': {
                'jwt_secret_key': 'test_secret_key',
                'access_token_expire_minutes': 30
            }
        })
        
        # 模拟数据库
        mock_database = Mock()
        
        auth_manager = AuthenticationManager(config, mock_database)
        assert auth_manager is not None
        assert auth_manager.jwt_secret_key == 'test_secret_key'
        assert auth_manager.access_token_expire_minutes == 30
        
    except Exception as e:
        pytest.fail(f"Failed to create auth manager: {e}")


def test_rate_limiter_creation():
    """测试速率限制器创建"""
    try:
        from src.api.rate_limiter import RateLimiter
        from src.config import Config
        
        config = Config({
            'rate_limit': {
                'global': {'requests_per_minute': 1000},
                'per_ip': {'requests_per_minute': 100}
            }
        })
        
        rate_limiter = RateLimiter(config)
        assert rate_limiter is not None
        
    except Exception as e:
        pytest.fail(f"Failed to create rate limiter: {e}")


def test_request_validator_creation():
    """测试请求验证器创建"""
    try:
        from src.api.request_validator import RequestValidator
        from src.config import Config
        
        config = Config({
            'validation': {
                'max_request_size': 10 * 1024 * 1024,
                'max_string_length': 10000
            }
        })
        
        validator = RequestValidator(config)
        assert validator is not None
        
    except Exception as e:
        pytest.fail(f"Failed to create request validator: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])