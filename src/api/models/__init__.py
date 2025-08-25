"""
API数据模型定义

定义所有API接口使用的数据模型，包括：
- 请求模型
- 响应模型
- 错误模型
- 认证模型
"""

from .auth_models import *
from .request_models import *
from .response_models import *
from .error_models import *

__all__ = [
    # 认证相关
    'LoginRequest',
    'TokenResponse',
    'UserInfo',
    'Permission',
    
    # 策略控制相关
    'StrategyStartRequest',
    'StrategyStopRequest',
    'StrategyStatusRequest',
    'StrategyConfigRequest',
    
    # 响应模型
    'APIResponse',
    'StrategyStatusResponse',
    'SystemHealthResponse',
    'SignalHistoryResponse',
    
    # 错误模型
    'APIError',
    'ValidationError',
    'AuthenticationError',
    'RateLimitError'
]