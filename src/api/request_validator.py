"""
API请求验证器

提供全面的请求验证功能，包括：
- 数据类型验证
- 业务逻辑验证
- 安全性验证
- 自定义验证规则
"""

import re
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from decimal import Decimal, InvalidOperation
import structlog
from pydantic import BaseModel, ValidationError as PydanticValidationError
from fastapi import Request

from .models.error_models import (
    ValidationError, ErrorCode, APIError, ErrorDetail
)
from .models.auth_models import Permission, SessionInfo
from src.config import Config

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class ValidationRule:
    """验证规则基类"""
    
    def __init__(self, field_name: str, error_message: str = None):
        self.field_name = field_name
        self.error_message = error_message or f"Validation failed for field '{field_name}'"
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        """验证值，返回True表示验证通过"""
        raise NotImplementedError
    
    def get_error_detail(self, value: Any = None) -> ErrorDetail:
        """获取错误详情"""
        return ErrorDetail(
            field=self.field_name,
            value=value,
            message=self.error_message,
            code="VALIDATION_FAILED"
        )


class RequiredRule(ValidationRule):
    """必填字段验证"""
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        return value is not None and value != ""


class LengthRule(ValidationRule):
    """长度验证"""
    
    def __init__(self, field_name: str, min_length: int = None, max_length: int = None):
        self.min_length = min_length
        self.max_length = max_length
        
        error_msg = f"Length validation failed for '{field_name}'"
        if min_length and max_length:
            error_msg += f" (must be between {min_length} and {max_length} characters)"
        elif min_length:
            error_msg += f" (must be at least {min_length} characters)"
        elif max_length:
            error_msg += f" (must be at most {max_length} characters)"
        
        super().__init__(field_name, error_msg)
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        length = len(str(value))
        
        if self.min_length and length < self.min_length:
            return False
        
        if self.max_length and length > self.max_length:
            return False
        
        return True


class RegexRule(ValidationRule):
    """正则表达式验证"""
    
    def __init__(self, field_name: str, pattern: str, error_message: str = None):
        self.pattern = re.compile(pattern)
        super().__init__(field_name, error_message)
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        return bool(self.pattern.match(str(value)))


class RangeRule(ValidationRule):
    """数值范围验证"""
    
    def __init__(self, field_name: str, min_value: float = None, max_value: float = None):
        self.min_value = min_value
        self.max_value = max_value
        
        error_msg = f"Range validation failed for '{field_name}'"
        if min_value is not None and max_value is not None:
            error_msg += f" (must be between {min_value} and {max_value})"
        elif min_value is not None:
            error_msg += f" (must be at least {min_value})"
        elif max_value is not None:
            error_msg += f" (must be at most {max_value})"
        
        super().__init__(field_name, error_msg)
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False
        
        if self.min_value is not None and num_value < self.min_value:
            return False
        
        if self.max_value is not None and num_value > self.max_value:
            return False
        
        return True


class EmailRule(ValidationRule):
    """邮箱格式验证"""
    
    def __init__(self, field_name: str):
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        super().__init__(field_name, f"Invalid email format for '{field_name}'")
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        return bool(self.email_pattern.match(str(value)))


class DateTimeRule(ValidationRule):
    """日期时间验证"""
    
    def __init__(self, field_name: str, min_date: datetime = None, max_date: datetime = None):
        self.min_date = min_date
        self.max_date = max_date
        super().__init__(field_name, f"Invalid datetime for '{field_name}'")
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return False
        elif isinstance(value, datetime):
            dt = value
        else:
            return False
        
        if self.min_date and dt < self.min_date:
            return False
        
        if self.max_date and dt > self.max_date:
            return False
        
        return True


class UUIDRule(ValidationRule):
    """UUID格式验证"""
    
    def __init__(self, field_name: str):
        super().__init__(field_name, f"Invalid UUID format for '{field_name}'")
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        try:
            uuid.UUID(str(value))
            return True
        except ValueError:
            return False


class ChoiceRule(ValidationRule):
    """选择值验证"""
    
    def __init__(self, field_name: str, choices: List[Any]):
        self.choices = choices
        super().__init__(field_name, f"Invalid choice for '{field_name}'. Must be one of: {choices}")
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        if value is None:
            return True
        
        return value in self.choices


class CustomRule(ValidationRule):
    """自定义验证规则"""
    
    def __init__(self, field_name: str, validator_func: Callable, error_message: str = None):
        self.validator_func = validator_func
        super().__init__(field_name, error_message)
    
    async def validate(self, value: Any, context: Dict[str, Any] = None) -> bool:
        try:
            if asyncio.iscoroutinefunction(self.validator_func):
                return await self.validator_func(value, context)
            else:
                return self.validator_func(value, context)
        except Exception as e:
            logger.error(f"Custom validation error for {self.field_name}: {e}")
            return False


class SecurityValidator:
    """安全验证器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.max_request_size = config.get('validation.max_request_size', 10 * 1024 * 1024)  # 10MB
        self.max_string_length = config.get('validation.max_string_length', 10000)
        self.max_array_length = config.get('validation.max_array_length', 1000)
        
        # 危险字符和模式
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Code execution
            r'setTimeout\s*\(',  # Code execution
            r'setInterval\s*\(',  # Code execution
            r'\bUNION\b.*\bSELECT\b',  # SQL injection
            r'\bOR\b.*\b1\s*=\s*1\b',  # SQL injection
            r'\bDROP\b.*\bTABLE\b',  # SQL injection
            r'\.\./',  # Path traversal
            r'\\x[0-9a-fA-F]{2}',  # Hex encoding
            r'%[0-9a-fA-F]{2}',  # URL encoding
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    async def validate_request_size(self, request: Request) -> bool:
        """验证请求大小"""
        try:
            content_length = request.headers.get('content-length')
            if content_length:
                size = int(content_length)
                return size <= self.max_request_size
            return True
        except (ValueError, TypeError):
            return False
    
    async def validate_content_security(self, data: Any) -> List[ErrorDetail]:
        """验证内容安全性"""
        errors = []
        await self._check_security_recursive("", data, errors)
        return errors
    
    async def _check_security_recursive(self, path: str, data: Any, errors: List[ErrorDetail]):
        """递归检查安全性"""
        if isinstance(data, str):
            if len(data) > self.max_string_length:
                errors.append(ErrorDetail(
                    field=path,
                    value=data[:100] + "..." if len(data) > 100 else data,
                    message=f"String length exceeds maximum allowed ({self.max_string_length})",
                    code="STRING_TOO_LONG"
                ))
            
            # 检查危险模式
            for pattern in self.compiled_patterns:
                if pattern.search(data):
                    errors.append(ErrorDetail(
                        field=path,
                        value=data[:100] + "..." if len(data) > 100 else data,
                        message="Potentially malicious content detected",
                        code="SECURITY_VIOLATION"
                    ))
                    break
        
        elif isinstance(data, (list, tuple)):
            if len(data) > self.max_array_length:
                errors.append(ErrorDetail(
                    field=path,
                    value=f"Array with {len(data)} items",
                    message=f"Array length exceeds maximum allowed ({self.max_array_length})",
                    code="ARRAY_TOO_LONG"
                ))
            
            for i, item in enumerate(data[:self.max_array_length]):
                await self._check_security_recursive(f"{path}[{i}]", item, errors)
        
        elif isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, str):
                    continue
                
                new_path = f"{path}.{key}" if path else key
                await self._check_security_recursive(new_path, value, errors)


class RequestValidator:
    """请求验证器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.security_validator = SecurityValidator(config)
        
        # 验证规则存储
        self.field_rules: Dict[str, List[ValidationRule]] = {}
        self.global_rules: List[ValidationRule] = []
        self.model_validators: Dict[str, Type[BaseModel]] = {}
        
        # 业务验证规则
        self._setup_business_rules()
        
        logger.info("Request validator initialized")
    
    def _setup_business_rules(self):
        """设置业务验证规则"""
        # 策略ID验证
        self.add_field_rule("strategy_id", RequiredRule("strategy_id"))
        self.add_field_rule("strategy_id", LengthRule("strategy_id", min_length=1, max_length=100))
        self.add_field_rule("strategy_id", RegexRule("strategy_id", r'^[a-zA-Z0-9_-]+$'))
        
        # 用户名验证
        self.add_field_rule("username", RequiredRule("username"))
        self.add_field_rule("username", LengthRule("username", min_length=3, max_length=50))
        self.add_field_rule("username", RegexRule("username", r'^[a-zA-Z0-9_-]+$'))
        
        # 密码验证
        self.add_field_rule("password", RequiredRule("password"))
        self.add_field_rule("password", LengthRule("password", min_length=8, max_length=128))
        
        # 邮箱验证
        self.add_field_rule("email", EmailRule("email"))
        
        # 数量验证
        self.add_field_rule("page", RangeRule("page", min_value=1))
        self.add_field_rule("page_size", RangeRule("page_size", min_value=1, max_value=100))
        
        # 置信度验证
        self.add_field_rule("confidence", RangeRule("confidence", min_value=0.0, max_value=1.0))
        self.add_field_rule("min_confidence", RangeRule("min_confidence", min_value=0.0, max_value=1.0))
    
    def add_field_rule(self, field_name: str, rule: ValidationRule):
        """添加字段验证规则"""
        if field_name not in self.field_rules:
            self.field_rules[field_name] = []
        self.field_rules[field_name].append(rule)
    
    def add_global_rule(self, rule: ValidationRule):
        """添加全局验证规则"""
        self.global_rules.append(rule)
    
    def register_model_validator(self, model_name: str, model_class: Type[BaseModel]):
        """注册模型验证器"""
        self.model_validators[model_name] = model_class
    
    async def validate_request(self, request: Request, data: Dict[str, Any], 
                             model_class: Type[T] = None, 
                             session: SessionInfo = None) -> T:
        """验证请求数据"""
        try:
            errors = []
            
            # 1. 安全验证
            if not await self.security_validator.validate_request_size(request):
                raise ValidationError("Request size too large")
            
            security_errors = await self.security_validator.validate_content_security(data)
            errors.extend(security_errors)
            
            # 2. Pydantic模型验证
            validated_data = None
            if model_class:
                try:
                    validated_data = model_class(**data)
                except PydanticValidationError as e:
                    for error in e.errors():
                        field_name = ".".join(str(x) for x in error['loc'])
                        errors.append(ErrorDetail(
                            field=field_name,
                            value=error.get('input'),
                            message=error['msg'],
                            code=error['type'].upper(),
                            location="body"
                        ))
            
            # 3. 自定义字段验证
            field_errors = await self._validate_fields(data)
            errors.extend(field_errors)
            
            # 4. 全局验证
            global_errors = await self._validate_global_rules(data)
            errors.extend(global_errors)
            
            # 5. 业务逻辑验证
            business_errors = await self._validate_business_logic(data, session)
            errors.extend(business_errors)
            
            if errors:
                raise ValidationError(
                    message="Validation failed",
                    field_errors=[error.dict() for error in errors]
                )
            
            return validated_data if validated_data else data
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Request validation failed", error=str(e))
            raise ValidationError(f"Validation error: {str(e)}")
    
    async def validate_pydantic_model(self, data: Dict[str, Any], model_class: Type[T]) -> T:
        """验证Pydantic模型"""
        try:
            return model_class(**data)
        except PydanticValidationError as e:
            errors = []
            for error in e.errors():
                field_name = ".".join(str(x) for x in error['loc'])
                errors.append(ErrorDetail(
                    field=field_name,
                    value=error.get('input'),
                    message=error['msg'],
                    code=error['type'].upper()
                ))
            
            raise ValidationError(
                message="Model validation failed",
                field_errors=[error.dict() for error in errors]
            )
    
    async def validate_json(self, json_str: str) -> Dict[str, Any]:
        """验证JSON格式"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"Invalid JSON format: {str(e)}",
                field_errors=[
                    ErrorDetail(
                        field="json_data",
                        message=f"JSON decode error: {str(e)}",
                        code="INVALID_JSON"
                    ).dict()
                ]
            )
    
    async def validate_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证查询参数"""
        errors = []
        validated_params = {}
        
        for key, value in params.items():
            if key in self.field_rules:
                for rule in self.field_rules[key]:
                    if not await rule.validate(value):
                        errors.append(rule.get_error_detail(value))
                        break
                else:
                    validated_params[key] = value
            else:
                validated_params[key] = value
        
        if errors:
            raise ValidationError(
                message="Query parameter validation failed",
                field_errors=[error.dict() for error in errors]
            )
        
        return validated_params
    
    async def validate_path_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证路径参数"""
        errors = []
        validated_params = {}
        
        for key, value in params.items():
            # 基本安全检查
            if not isinstance(value, (str, int, float)):
                errors.append(ErrorDetail(
                    field=key,
                    value=value,
                    message=f"Invalid path parameter type for '{key}'",
                    code="INVALID_TYPE"
                ))
                continue
            
            # 长度检查
            if isinstance(value, str) and len(value) > 255:
                errors.append(ErrorDetail(
                    field=key,
                    value=value,
                    message=f"Path parameter '{key}' too long",
                    code="PATH_PARAM_TOO_LONG"
                ))
                continue
            
            # 特殊字符检查
            if isinstance(value, str) and re.search(r'[<>"\']', value):
                errors.append(ErrorDetail(
                    field=key,
                    value=value,
                    message=f"Path parameter '{key}' contains invalid characters",
                    code="INVALID_CHARACTERS"
                ))
                continue
            
            validated_params[key] = value
        
        if errors:
            raise ValidationError(
                message="Path parameter validation failed",
                field_errors=[error.dict() for error in errors]
            )
        
        return validated_params
    
    async def _validate_fields(self, data: Dict[str, Any]) -> List[ErrorDetail]:
        """验证字段"""
        errors = []
        
        for field_name, value in data.items():
            if field_name in self.field_rules:
                for rule in self.field_rules[field_name]:
                    if not await rule.validate(value, data):
                        errors.append(rule.get_error_detail(value))
                        break  # 一个字段遇到第一个错误就停止
        
        return errors
    
    async def _validate_global_rules(self, data: Dict[str, Any]) -> List[ErrorDetail]:
        """验证全局规则"""
        errors = []
        
        for rule in self.global_rules:
            if not await rule.validate(data, data):
                errors.append(rule.get_error_detail(data))
        
        return errors
    
    async def _validate_business_logic(self, data: Dict[str, Any], 
                                     session: SessionInfo = None) -> List[ErrorDetail]:
        """验证业务逻辑"""
        errors = []
        
        # 策略相关验证
        if 'strategy_id' in data:
            strategy_id = data['strategy_id']
            if not await self._validate_strategy_exists(strategy_id):
                errors.append(ErrorDetail(
                    field="strategy_id",
                    value=strategy_id,
                    message=f"Strategy '{strategy_id}' not found",
                    code="STRATEGY_NOT_FOUND"
                ))
        
        # 权限验证
        if session:
            permission_errors = await self._validate_permissions(data, session)
            errors.extend(permission_errors)
        
        # 日期范围验证
        if 'start_time' in data and 'end_time' in data:
            start_time = data.get('start_time')
            end_time = data.get('end_time')
            
            if start_time and end_time:
                try:
                    if isinstance(start_time, str):
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    else:
                        start_dt = start_time
                    
                    if isinstance(end_time, str):
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    else:
                        end_dt = end_time
                    
                    if start_dt >= end_dt:
                        errors.append(ErrorDetail(
                            field="time_range",
                            value={"start_time": start_time, "end_time": end_time},
                            message="Start time must be before end time",
                            code="INVALID_TIME_RANGE"
                        ))
                    
                    # 检查时间范围是否合理
                    max_range = timedelta(days=30)
                    if end_dt - start_dt > max_range:
                        errors.append(ErrorDetail(
                            field="time_range",
                            value={"start_time": start_time, "end_time": end_time},
                            message=f"Time range too large (maximum {max_range.days} days)",
                            code="TIME_RANGE_TOO_LARGE"
                        ))
                
                except (ValueError, TypeError) as e:
                    errors.append(ErrorDetail(
                        field="time_range",
                        message=f"Invalid datetime format: {str(e)}",
                        code="INVALID_DATETIME"
                    ))
        
        return errors
    
    async def _validate_strategy_exists(self, strategy_id: str) -> bool:
        """验证策略是否存在（简化实现）"""
        # 实际实现应该查询策略管理器或数据库
        return True
    
    async def _validate_permissions(self, data: Dict[str, Any], 
                                  session: SessionInfo) -> List[ErrorDetail]:
        """验证权限"""
        errors = []
        
        # 根据操作类型检查权限
        if 'action' in data:
            action = data['action']
            required_permission = None
            
            if action == 'start':
                required_permission = Permission.STRATEGY_START
            elif action == 'stop':
                required_permission = Permission.STRATEGY_STOP
            elif action == 'restart':
                required_permission = Permission.STRATEGY_RESTART
            elif action in ['config', 'update']:
                required_permission = Permission.STRATEGY_CONFIG
            
            if required_permission and required_permission not in session.permissions:
                errors.append(ErrorDetail(
                    field="action",
                    value=action,
                    message=f"Insufficient permissions for action '{action}'",
                    code="INSUFFICIENT_PERMISSIONS"
                ))
        
        return errors
    
    def create_custom_validator(self, validator_func: Callable) -> CustomRule:
        """创建自定义验证器"""
        return CustomRule("custom", validator_func)
    
    async def validate_decimal(self, value: Any, precision: int = 8, scale: int = 2) -> Decimal:
        """验证和转换十进制数"""
        try:
            decimal_value = Decimal(str(value))
            
            # 检查精度和小数位数
            if decimal_value.as_tuple().digits.__len__() > precision:
                raise ValidationError(f"Decimal precision exceeds {precision} digits")
            
            if abs(decimal_value.as_tuple().exponent) > scale:
                raise ValidationError(f"Decimal scale exceeds {scale} digits")
            
            return decimal_value
            
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid decimal value: {str(e)}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证器摘要信息"""
        return {
            "field_rules_count": len(self.field_rules),
            "global_rules_count": len(self.global_rules),
            "model_validators_count": len(self.model_validators),
            "field_rules": {
                field: len(rules) for field, rules in self.field_rules.items()
            },
            "security_config": {
                "max_request_size": self.security_validator.max_request_size,
                "max_string_length": self.security_validator.max_string_length,
                "max_array_length": self.security_validator.max_array_length,
                "dangerous_patterns_count": len(self.security_validator.dangerous_patterns)
            }
        }