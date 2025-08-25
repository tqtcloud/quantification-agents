"""
API错误处理相关数据模型
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ErrorCode(str, Enum):
    """错误代码枚举"""
    # 通用错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    OPERATION_FAILED = "OPERATION_FAILED"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # 认证和授权错误
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    TOKEN_INVALID = "TOKEN_INVALID"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    ACCOUNT_LOCKED = "ACCOUNT_LOCKED"
    ACCOUNT_DISABLED = "ACCOUNT_DISABLED"
    
    # 验证错误
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FIELD_VALUE = "INVALID_FIELD_VALUE"
    INVALID_FIELD_FORMAT = "INVALID_FIELD_FORMAT"
    FIELD_LENGTH_EXCEEDED = "FIELD_LENGTH_EXCEEDED"
    
    # 限流错误
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    CONCURRENT_LIMIT_EXCEEDED = "CONCURRENT_LIMIT_EXCEEDED"
    
    # 策略相关错误
    STRATEGY_NOT_FOUND = "STRATEGY_NOT_FOUND"
    STRATEGY_ALREADY_EXISTS = "STRATEGY_ALREADY_EXISTS"
    STRATEGY_ALREADY_RUNNING = "STRATEGY_ALREADY_RUNNING"
    STRATEGY_NOT_RUNNING = "STRATEGY_NOT_RUNNING"
    STRATEGY_CONFIG_INVALID = "STRATEGY_CONFIG_INVALID"
    STRATEGY_START_FAILED = "STRATEGY_START_FAILED"
    STRATEGY_STOP_FAILED = "STRATEGY_STOP_FAILED"
    
    # 系统错误
    SYSTEM_MAINTENANCE = "SYSTEM_MAINTENANCE"
    SYSTEM_OVERLOADED = "SYSTEM_OVERLOADED"
    COMPONENT_UNAVAILABLE = "COMPONENT_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    
    # 数据错误
    DATA_CORRUPTION = "DATA_CORRUPTION"
    DATA_INCONSISTENCY = "DATA_INCONSISTENCY"
    DATA_ACCESS_ERROR = "DATA_ACCESS_ERROR"
    
    # 配置错误
    CONFIG_ERROR = "CONFIG_ERROR"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"


class ErrorSeverity(str, Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class APIError(BaseModel):
    """API错误基础模型"""
    code: ErrorCode = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[str] = Field(None, description="详细错误信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")
    trace_id: Optional[str] = Field(None, description="跟踪ID")
    severity: ErrorSeverity = Field(ErrorSeverity.MEDIUM, description="错误严重程度")
    context: Dict[str, Any] = Field(default={}, description="错误上下文")
    suggestions: List[str] = Field(default=[], description="建议解决方案")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.dict(exclude_none=True)


class ValidationError(APIError):
    """验证错误模型"""
    code: ErrorCode = Field(default=ErrorCode.VALIDATION_ERROR, description="错误代码")
    field_errors: List[Dict[str, Any]] = Field(default=[], description="字段错误列表")
    
    def __init__(self, message: str, field_errors: List[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            field_errors=field_errors or [],
            **kwargs
        )


class AuthenticationError(APIError):
    """认证错误模型"""
    code: ErrorCode = Field(default=ErrorCode.AUTHENTICATION_FAILED, description="错误代码")
    auth_scheme: Optional[str] = Field(None, description="认证方案")
    realm: Optional[str] = Field(None, description="认证域")
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            code=ErrorCode.AUTHENTICATION_FAILED,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(APIError):
    """授权错误模型"""
    code: ErrorCode = Field(default=ErrorCode.INSUFFICIENT_PERMISSIONS, description="错误代码")
    required_permissions: List[str] = Field(default=[], description="所需权限")
    user_permissions: List[str] = Field(default=[], description="用户权限")
    
    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(
            code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class RateLimitError(APIError):
    """限流错误模型"""
    code: ErrorCode = Field(default=ErrorCode.RATE_LIMIT_EXCEEDED, description="错误代码")
    limit: int = Field(..., description="限制数量")
    remaining: int = Field(..., description="剩余数量")
    reset_time: datetime = Field(..., description="重置时间")
    retry_after: int = Field(..., description="重试间隔(秒)")
    
    def __init__(self, limit: int, remaining: int, reset_time: datetime, retry_after: int, **kwargs):
        super().__init__(
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded. Limit: {limit}, Remaining: {remaining}",
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ResourceNotFoundError(APIError):
    """资源未找到错误模型"""
    code: ErrorCode = Field(default=ErrorCode.RESOURCE_NOT_FOUND, description="错误代码")
    resource_type: str = Field(..., description="资源类型")
    resource_id: str = Field(..., description="资源ID")
    
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} with ID '{resource_id}' not found",
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs
        )


class ResourceConflictError(APIError):
    """资源冲突错误模型"""
    code: ErrorCode = Field(default=ErrorCode.RESOURCE_CONFLICT, description="错误代码")
    resource_type: str = Field(..., description="资源类型")
    resource_id: str = Field(..., description="资源ID")
    conflict_reason: str = Field(..., description="冲突原因")
    
    def __init__(self, resource_type: str, resource_id: str, conflict_reason: str, **kwargs):
        super().__init__(
            code=ErrorCode.RESOURCE_CONFLICT,
            message=f"Conflict with {resource_type} '{resource_id}': {conflict_reason}",
            resource_type=resource_type,
            resource_id=resource_id,
            conflict_reason=conflict_reason,
            **kwargs
        )


class StrategyError(APIError):
    """策略错误模型"""
    strategy_id: str = Field(..., description="策略ID")
    strategy_type: Optional[str] = Field(None, description="策略类型")
    operation: Optional[str] = Field(None, description="操作类型")
    
    def __init__(self, code: ErrorCode, message: str, strategy_id: str, **kwargs):
        super().__init__(
            code=code,
            message=message,
            strategy_id=strategy_id,
            **kwargs
        )


class SystemError(APIError):
    """系统错误模型"""
    component: Optional[str] = Field(None, description="组件名称")
    service: Optional[str] = Field(None, description="服务名称")
    system_state: Optional[str] = Field(None, description="系统状态")
    
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(
            code=code,
            message=message,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ConfigurationError(APIError):
    """配置错误模型"""
    code: ErrorCode = Field(default=ErrorCode.CONFIG_ERROR, description="错误代码")
    config_section: Optional[str] = Field(None, description="配置分组")
    config_key: Optional[str] = Field(None, description="配置键")
    expected_type: Optional[str] = Field(None, description="期望类型")
    actual_value: Optional[str] = Field(None, description="实际值")
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            code=ErrorCode.CONFIG_ERROR,
            message=message,
            **kwargs
        )


class DataError(APIError):
    """数据错误模型"""
    data_type: Optional[str] = Field(None, description="数据类型")
    data_id: Optional[str] = Field(None, description="数据ID")
    operation: Optional[str] = Field(None, description="操作类型")
    
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(
            code=code,
            message=message,
            **kwargs
        )


class TimeoutError(APIError):
    """超时错误模型"""
    code: ErrorCode = Field(default=ErrorCode.TIMEOUT_ERROR, description="错误代码")
    timeout_seconds: float = Field(..., description="超时时间(秒)")
    operation: str = Field(..., description="超时操作")
    
    def __init__(self, timeout_seconds: float, operation: str, **kwargs):
        super().__init__(
            code=ErrorCode.TIMEOUT_ERROR,
            message=f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            timeout_seconds=timeout_seconds,
            operation=operation,
            **kwargs
        )


class ErrorDetail(BaseModel):
    """错误详情模型"""
    field: Optional[str] = Field(None, description="字段名")
    value: Optional[Any] = Field(None, description="字段值")
    message: str = Field(..., description="错误消息")
    code: Optional[str] = Field(None, description="错误代码")
    location: Optional[str] = Field(None, description="错误位置")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: APIError = Field(..., description="错误信息")
    status: str = Field(default="error", description="响应状态")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    path: Optional[str] = Field(None, description="请求路径")
    method: Optional[str] = Field(None, description="请求方法")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class BatchErrorResponse(BaseModel):
    """批量错误响应模型"""
    errors: List[APIError] = Field(..., description="错误列表")
    total_errors: int = Field(..., description="总错误数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    partial_success: bool = Field(..., description="是否部分成功")


# 错误代码到HTTP状态码的映射
ERROR_CODE_HTTP_STATUS_MAP = {
    # 400 Bad Request
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.MISSING_REQUIRED_FIELD: 400,
    ErrorCode.INVALID_FIELD_VALUE: 400,
    ErrorCode.INVALID_FIELD_FORMAT: 400,
    ErrorCode.FIELD_LENGTH_EXCEEDED: 400,
    ErrorCode.CONFIG_VALIDATION_FAILED: 400,
    
    # 401 Unauthorized
    ErrorCode.AUTHENTICATION_FAILED: 401,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.TOKEN_INVALID: 401,
    
    # 403 Forbidden
    ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
    ErrorCode.ACCOUNT_LOCKED: 403,
    ErrorCode.ACCOUNT_DISABLED: 403,
    
    # 404 Not Found
    ErrorCode.RESOURCE_NOT_FOUND: 404,
    ErrorCode.STRATEGY_NOT_FOUND: 404,
    ErrorCode.CONFIG_NOT_FOUND: 404,
    
    # 409 Conflict
    ErrorCode.RESOURCE_CONFLICT: 409,
    ErrorCode.STRATEGY_ALREADY_EXISTS: 409,
    ErrorCode.STRATEGY_ALREADY_RUNNING: 409,
    
    # 422 Unprocessable Entity
    ErrorCode.STRATEGY_NOT_RUNNING: 422,
    ErrorCode.STRATEGY_CONFIG_INVALID: 422,
    
    # 429 Too Many Requests
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.CONCURRENT_LIMIT_EXCEEDED: 429,
    
    # 500 Internal Server Error
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.OPERATION_FAILED: 500,
    ErrorCode.STRATEGY_START_FAILED: 500,
    ErrorCode.STRATEGY_STOP_FAILED: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.DATA_CORRUPTION: 500,
    ErrorCode.DATA_INCONSISTENCY: 500,
    ErrorCode.CONFIG_ERROR: 500,
    
    # 502 Bad Gateway
    ErrorCode.COMPONENT_UNAVAILABLE: 502,
    ErrorCode.NETWORK_ERROR: 502,
    
    # 503 Service Unavailable
    ErrorCode.SYSTEM_MAINTENANCE: 503,
    ErrorCode.SYSTEM_OVERLOADED: 503,
    
    # 504 Gateway Timeout
    ErrorCode.TIMEOUT_ERROR: 504,
    
    # 507 Insufficient Storage
    ErrorCode.DATA_ACCESS_ERROR: 507,
}


def get_http_status_code(error_code: ErrorCode) -> int:
    """根据错误代码获取HTTP状态码"""
    return ERROR_CODE_HTTP_STATUS_MAP.get(error_code, 500)