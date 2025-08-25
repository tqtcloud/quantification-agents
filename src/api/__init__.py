"""
量化交易系统API模块

提供RESTful API接口，包括：
- 交易策略控制接口
- 认证和权限管理
- 请求限流和参数验证
- API监控和审计

主要组件：
- TradingAPI: 核心API服务器
- AuthenticationManager: JWT认证和权限管理
- RateLimiter: 请求限流中间件
- RequestValidator: 参数验证和数据校验
"""

__all__ = [
    'TradingAPI',
    'AuthenticationManager',
    'RateLimiter',
    'RequestValidator'
]