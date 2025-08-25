"""
认证相关的数据模型
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"


class Permission(str, Enum):
    """权限枚举"""
    # 策略控制权限
    STRATEGY_START = "strategy:start"
    STRATEGY_STOP = "strategy:stop"
    STRATEGY_RESTART = "strategy:restart"
    STRATEGY_VIEW = "strategy:view"
    STRATEGY_CONFIG = "strategy:config"
    
    # 系统管理权限
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    
    # API权限
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"
    
    # 数据权限
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"


class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    remember_me: bool = Field(default=False, description="记住登录状态")


class TokenResponse(BaseModel):
    """Token响应模型"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: Optional[str] = Field(None, description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间(秒)")
    expires_at: datetime = Field(..., description="过期时间戳")
    permissions: List[Permission] = Field(default=[], description="用户权限列表")


class RefreshTokenRequest(BaseModel):
    """刷新令牌请求模型"""
    refresh_token: str = Field(..., description="刷新令牌")


class UserInfo(BaseModel):
    """用户信息模型"""
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    role: UserRole = Field(..., description="用户角色")
    permissions: List[Permission] = Field(default=[], description="权限列表")
    email: Optional[str] = Field(None, description="邮箱")
    created_at: datetime = Field(..., description="创建时间")
    last_login: Optional[datetime] = Field(None, description="最后登录时间")
    is_active: bool = Field(default=True, description="是否激活")
    metadata: Dict[str, Any] = Field(default={}, description="用户元数据")


class CreateUserRequest(BaseModel):
    """创建用户请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    role: UserRole = Field(..., description="用户角色")
    email: Optional[str] = Field(None, description="邮箱")
    permissions: List[Permission] = Field(default=[], description="权限列表")
    metadata: Dict[str, Any] = Field(default={}, description="用户元数据")


class UpdateUserRequest(BaseModel):
    """更新用户请求模型"""
    role: Optional[UserRole] = Field(None, description="用户角色")
    permissions: Optional[List[Permission]] = Field(None, description="权限列表")
    email: Optional[str] = Field(None, description="邮箱")
    is_active: Optional[bool] = Field(None, description="是否激活")
    metadata: Optional[Dict[str, Any]] = Field(None, description="用户元数据")


class PasswordChangeRequest(BaseModel):
    """修改密码请求模型"""
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")


class APIKeyRequest(BaseModel):
    """API密钥请求模型"""
    name: str = Field(..., description="密钥名称")
    permissions: List[Permission] = Field(..., description="权限列表")
    expires_in_days: Optional[int] = Field(30, description="过期天数")
    rate_limit: Optional[int] = Field(1000, description="速率限制(每小时)")


class APIKeyResponse(BaseModel):
    """API密钥响应模型"""
    key_id: str = Field(..., description="密钥ID")
    api_key: str = Field(..., description="API密钥")
    name: str = Field(..., description="密钥名称")
    permissions: List[Permission] = Field(..., description="权限列表")
    created_at: datetime = Field(..., description="创建时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    rate_limit: int = Field(..., description="速率限制")
    is_active: bool = Field(default=True, description="是否激活")


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str = Field(..., description="会话ID")
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    role: UserRole = Field(..., description="用户角色")
    permissions: List[Permission] = Field(..., description="权限列表")
    created_at: datetime = Field(..., description="创建时间")
    last_activity: datetime = Field(..., description="最后活动时间")
    ip_address: str = Field(..., description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    is_api_session: bool = Field(default=False, description="是否为API会话")


class LogoutRequest(BaseModel):
    """登出请求模型"""
    all_sessions: bool = Field(default=False, description="是否登出所有会话")


# 角色权限映射
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.STRATEGY_START,
        Permission.STRATEGY_STOP,
        Permission.STRATEGY_RESTART,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CONFIG,
        Permission.SYSTEM_ADMIN,
        Permission.SYSTEM_MONITOR,
        Permission.SYSTEM_CONFIG,
        Permission.API_ACCESS,
        Permission.API_ADMIN,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
    ],
    UserRole.TRADER: [
        Permission.STRATEGY_START,
        Permission.STRATEGY_STOP,
        Permission.STRATEGY_RESTART,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CONFIG,
        Permission.SYSTEM_MONITOR,
        Permission.API_ACCESS,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
    ],
    UserRole.VIEWER: [
        Permission.STRATEGY_VIEW,
        Permission.SYSTEM_MONITOR,
        Permission.API_ACCESS,
        Permission.DATA_READ,
    ],
    UserRole.API_USER: [
        Permission.API_ACCESS,
        Permission.DATA_READ,
    ]
}