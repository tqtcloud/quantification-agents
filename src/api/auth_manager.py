"""
认证和权限管理器

提供JWT认证、权限验证、会话管理等功能
"""

import asyncio
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Any, Tuple
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import structlog
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound

from .models.auth_models import (
    UserRole, Permission, UserInfo, SessionInfo, APIKeyResponse,
    ROLE_PERMISSIONS, TokenResponse, CreateUserRequest
)
from .models.error_models import (
    AuthenticationError, AuthorizationError, ValidationError, 
    ResourceNotFoundError, ResourceConflictError, ErrorCode
)
# from src.core.database import Database  # 暂时注释掉，需要时启用
from src.config import Config

logger = structlog.get_logger(__name__)


class AuthenticationManager:
    """认证和权限管理器"""
    
    def __init__(self, config: Config, database=None):
        self.config = config
        self.database = database
        
        # JWT配置
        self.jwt_secret_key = config.get('auth.jwt_secret_key', self._generate_secret_key())
        self.jwt_algorithm = config.get('auth.jwt_algorithm', 'HS256')
        self.access_token_expire_minutes = config.get('auth.access_token_expire_minutes', 30)
        self.refresh_token_expire_days = config.get('auth.refresh_token_expire_days', 7)
        
        # 密码加密
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # 会话管理
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.revoked_tokens: Set[str] = set()
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # 速率限制
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.max_login_attempts = config.get('auth.max_login_attempts', 5)
        self.lockout_duration_minutes = config.get('auth.lockout_duration_minutes', 15)
        
        # 权限缓存
        self._permission_cache: Dict[str, Set[Permission]] = {}
        self._cache_expire_time: Dict[str, datetime] = {}
        self._cache_ttl_seconds = config.get('auth.permission_cache_ttl', 300)
        
        logger.info("Authentication manager initialized")
    
    def _generate_secret_key(self) -> str:
        """生成JWT密钥"""
        return secrets.token_urlsafe(32)
    
    async def initialize(self):
        """初始化认证管理器"""
        try:
            # 创建默认管理员账户
            await self._create_default_admin()
            
            # 清理过期会话
            await self._cleanup_expired_sessions()
            
            # 启动定期清理任务
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info("Authentication manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize authentication manager", error=str(e))
            raise
    
    async def _create_default_admin(self):
        """创建默认管理员账户"""
        try:
            admin_username = self.config.get('auth.default_admin_username', 'admin')
            admin_password = self.config.get('auth.default_admin_password', 'admin123')
            
            # 检查管理员是否已存在
            if await self._get_user_by_username(admin_username):
                return
            
            # 创建管理员账户
            user_data = CreateUserRequest(
                username=admin_username,
                password=admin_password,
                role=UserRole.ADMIN,
                email="admin@quantification.com",
                permissions=ROLE_PERMISSIONS[UserRole.ADMIN]
            )
            
            await self._create_user(user_data)
            logger.info(f"Default admin user '{admin_username}' created")
            
        except Exception as e:
            logger.error("Failed to create default admin user", error=str(e))
    
    async def authenticate_user(self, username: str, password: str, ip_address: str, 
                              user_agent: Optional[str] = None) -> TokenResponse:
        """用户认证"""
        try:
            # 检查登录尝试次数
            if await self._is_account_locked(username, ip_address):
                raise AuthenticationError(
                    code=ErrorCode.ACCOUNT_LOCKED,
                    message="Account temporarily locked due to too many failed login attempts",
                    context={'username': username, 'ip_address': ip_address}
                )
            
            # 验证用户凭据
            user = await self._verify_credentials(username, password)
            if not user:
                await self._record_failed_login(username, ip_address)
                raise AuthenticationError(message="Invalid username or password")
            
            # 检查用户状态
            if not user.is_active:
                raise AuthenticationError(
                    code=ErrorCode.ACCOUNT_DISABLED,
                    message="Account is disabled"
                )
            
            # 生成Token
            access_token = await self._generate_access_token(user)
            refresh_token = await self._generate_refresh_token(user)
            
            # 创建会话
            session = await self._create_session(user, ip_address, user_agent, False)
            
            # 更新最后登录时间
            await self._update_last_login(user.user_id)
            
            # 清除登录失败记录
            await self._clear_failed_logins(username, ip_address)
            
            logger.info(
                "User authenticated successfully", 
                user_id=user.user_id, 
                username=user.username,
                ip_address=ip_address
            )
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=self.access_token_expire_minutes * 60,
                expires_at=datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
                permissions=user.permissions
            )
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise AuthenticationError(message="Authentication failed")
    
    async def authenticate_api_key(self, api_key: str, ip_address: str) -> SessionInfo:
        """API密钥认证"""
        try:
            # 验证API密钥
            key_info = await self._verify_api_key(api_key)
            if not key_info:
                raise AuthenticationError(message="Invalid API key")
            
            # 检查密钥是否激活和未过期
            if not key_info.get('is_active'):
                raise AuthenticationError(
                    code=ErrorCode.ACCOUNT_DISABLED,
                    message="API key is disabled"
                )
            
            if key_info.get('expires_at') and datetime.utcnow() > key_info['expires_at']:
                raise AuthenticationError(
                    code=ErrorCode.TOKEN_EXPIRED,
                    message="API key has expired"
                )
            
            # 检查速率限制
            await self._check_api_key_rate_limit(api_key, key_info)
            
            # 获取用户信息
            user = await self._get_user_by_id(key_info['user_id'])
            if not user or not user.is_active:
                raise AuthenticationError(message="Associated user account is disabled")
            
            # 创建API会话
            session = await self._create_session(user, ip_address, None, True)
            
            logger.info(
                "API key authenticated successfully",
                user_id=user.user_id,
                api_key_id=key_info['key_id'],
                ip_address=ip_address
            )
            
            return session
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error("API key authentication failed", error=str(e))
            raise AuthenticationError(message="API key authentication failed")
    
    async def verify_token(self, token: str) -> SessionInfo:
        """验证访问令牌"""
        try:
            # 检查令牌是否已被撤销
            if token in self.revoked_tokens:
                raise AuthenticationError(
                    code=ErrorCode.TOKEN_INVALID,
                    message="Token has been revoked"
                )
            
            # 解码JWT令牌
            payload = jwt.decode(
                token, 
                self.jwt_secret_key, 
                algorithms=[self.jwt_algorithm]
            )
            
            # 验证令牌内容
            user_id = payload.get('sub')
            session_id = payload.get('session_id')
            
            if not user_id or not session_id:
                raise AuthenticationError(
                    code=ErrorCode.TOKEN_INVALID,
                    message="Invalid token format"
                )
            
            # 检查会话是否存在
            session = self.active_sessions.get(session_id)
            if not session:
                raise AuthenticationError(
                    code=ErrorCode.TOKEN_INVALID,
                    message="Session not found"
                )
            
            # 更新会话活动时间
            session.last_activity = datetime.utcnow()
            
            return session
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(
                code=ErrorCode.TOKEN_EXPIRED,
                message="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise AuthenticationError(
                code=ErrorCode.TOKEN_INVALID,
                message="Invalid token"
            )
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            raise AuthenticationError(message="Token verification failed")
    
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """刷新访问令牌"""
        try:
            # 解码刷新令牌
            payload = jwt.decode(
                refresh_token,
                self.jwt_secret_key,
                algorithms=[self.jwt_algorithm]
            )
            
            user_id = payload.get('sub')
            token_type = payload.get('type')
            
            if token_type != 'refresh':
                raise AuthenticationError(
                    code=ErrorCode.TOKEN_INVALID,
                    message="Invalid token type"
                )
            
            # 获取用户信息
            user = await self._get_user_by_id(user_id)
            if not user or not user.is_active:
                raise AuthenticationError(message="User account is disabled")
            
            # 生成新的访问令牌
            access_token = await self._generate_access_token(user)
            
            logger.info("Token refreshed successfully", user_id=user_id)
            
            return TokenResponse(
                access_token=access_token,
                expires_in=self.access_token_expire_minutes * 60,
                expires_at=datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
                permissions=user.permissions
            )
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(
                code=ErrorCode.TOKEN_EXPIRED,
                message="Refresh token has expired"
            )
        except jwt.InvalidTokenError:
            raise AuthenticationError(
                code=ErrorCode.TOKEN_INVALID,
                message="Invalid refresh token"
            )
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise AuthenticationError(message="Token refresh failed")
    
    async def check_permission(self, session: SessionInfo, permission: Permission) -> bool:
        """检查用户权限"""
        try:
            # 获取用户权限（使用缓存）
            permissions = await self._get_user_permissions_cached(session.user_id)
            
            # 检查权限
            has_permission = permission in permissions
            
            if not has_permission:
                logger.warning(
                    "Permission denied",
                    user_id=session.user_id,
                    username=session.username,
                    permission=permission,
                    user_permissions=list(permissions)
                )
            
            return has_permission
            
        except Exception as e:
            logger.error("Permission check failed", error=str(e))
            return False
    
    async def require_permission(self, session: SessionInfo, permission: Permission):
        """要求特定权限（抛出异常）"""
        if not await self.check_permission(session, permission):
            raise AuthorizationError(
                message=f"Permission '{permission}' is required",
                required_permissions=[permission],
                user_permissions=list(session.permissions)
            )
    
    async def logout(self, session_id: str, all_sessions: bool = False):
        """用户登出"""
        try:
            if all_sessions:
                # 登出所有会话
                user_sessions = [
                    s for s in self.active_sessions.values() 
                    if s.session_id == session_id
                ]
                
                if user_sessions:
                    user_id = user_sessions[0].user_id
                    sessions_to_remove = [
                        s.session_id for s in self.active_sessions.values()
                        if s.user_id == user_id
                    ]
                    
                    for sid in sessions_to_remove:
                        self.active_sessions.pop(sid, None)
                        
                    logger.info(
                        "All user sessions logged out", 
                        user_id=user_id,
                        session_count=len(sessions_to_remove)
                    )
            else:
                # 登出当前会话
                session = self.active_sessions.pop(session_id, None)
                if session:
                    logger.info(
                        "User session logged out",
                        user_id=session.user_id,
                        session_id=session_id
                    )
            
        except Exception as e:
            logger.error("Logout failed", error=str(e))
            raise
    
    async def create_api_key(self, user_id: str, name: str, permissions: List[Permission],
                           expires_in_days: Optional[int] = None, 
                           rate_limit: int = 1000) -> APIKeyResponse:
        """创建API密钥"""
        try:
            # 验证用户存在
            user = await self._get_user_by_id(user_id)
            if not user:
                raise ResourceNotFoundError("user", user_id)
            
            # 生成API密钥
            key_id = str(uuid.uuid4())
            api_key = self._generate_api_key()
            
            # 设置过期时间
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # 保存API密钥信息
            key_info = {
                'key_id': key_id,
                'user_id': user_id,
                'name': name,
                'permissions': permissions,
                'created_at': datetime.utcnow(),
                'expires_at': expires_at,
                'rate_limit': rate_limit,
                'is_active': True,
                'usage_count': 0,
                'last_used': None
            }
            
            # 存储API密钥（实际应用中应该存储到数据库）
            self.api_keys[self._hash_api_key(api_key)] = key_info
            
            logger.info(
                "API key created",
                user_id=user_id,
                key_id=key_id,
                name=name
            )
            
            return APIKeyResponse(
                key_id=key_id,
                api_key=api_key,
                name=name,
                permissions=permissions,
                created_at=key_info['created_at'],
                expires_at=expires_at,
                rate_limit=rate_limit
            )
            
        except Exception as e:
            logger.error("Failed to create API key", error=str(e))
            raise
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """撤销API密钥"""
        try:
            # 查找并撤销API密钥
            for key_hash, key_info in self.api_keys.items():
                if key_info['key_id'] == key_id:
                    key_info['is_active'] = False
                    logger.info("API key revoked", key_id=key_id)
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to revoke API key", error=str(e))
            raise
    
    # ===== 私有方法 =====
    
    def _hash_api_key(self, api_key: str) -> str:
        """对API密钥进行哈希"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _generate_api_key(self) -> str:
        """生成API密钥"""
        return f"qat_{secrets.token_urlsafe(32)}"
    
    async def _generate_access_token(self, user: UserInfo) -> str:
        """生成访问令牌"""
        payload = {
            'sub': user.user_id,
            'username': user.username,
            'role': user.role,
            'permissions': [p.value for p in user.permissions],
            'type': 'access',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        }
        
        return jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
    
    async def _generate_refresh_token(self, user: UserInfo) -> str:
        """生成刷新令牌"""
        payload = {
            'sub': user.user_id,
            'type': 'refresh',
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        }
        
        return jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
    
    async def _create_session(self, user: UserInfo, ip_address: str, 
                            user_agent: Optional[str], is_api_session: bool) -> SessionInfo:
        """创建会话"""
        session_id = str(uuid.uuid4())
        
        session = SessionInfo(
            session_id=session_id,
            user_id=user.user_id,
            username=user.username,
            role=user.role,
            permissions=user.permissions,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
            is_api_session=is_api_session
        )
        
        self.active_sessions[session_id] = session
        return session
    
    async def _verify_credentials(self, username: str, password: str) -> Optional[UserInfo]:
        """验证用户凭据"""
        # 实际应用中应该从数据库获取
        # 这里使用简化实现
        user = await self._get_user_by_username(username)
        if not user:
            return None
        
        # 验证密码（实际应用中密码应该是哈希存储的）
        if not self.pwd_context.verify(password, user.password_hash):
            return None
        
        return user
    
    async def _get_user_by_username(self, username: str) -> Optional[UserInfo]:
        """根据用户名获取用户信息"""
        # 实际实现应该查询数据库
        # 这里返回默认管理员用户
        if username == self.config.get('auth.default_admin_username', 'admin'):
            admin_password = self.config.get('auth.default_admin_password', 'admin123')
            password_hash = self.pwd_context.hash(admin_password)
            
            return UserInfo(
                user_id="admin-001",
                username=username,
                role=UserRole.ADMIN,
                permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
                email="admin@quantification.com",
                created_at=datetime.utcnow(),
                password_hash=password_hash
            )
        
        return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[UserInfo]:
        """根据用户ID获取用户信息"""
        # 实际实现应该查询数据库
        if user_id == "admin-001":
            return await self._get_user_by_username('admin')
        return None
    
    async def _create_user(self, user_data: CreateUserRequest) -> UserInfo:
        """创建用户"""
        # 实际实现应该保存到数据库
        logger.info(f"User creation simulated: {user_data.username}")
        return UserInfo(
            user_id=str(uuid.uuid4()),
            username=user_data.username,
            role=user_data.role,
            permissions=user_data.permissions,
            email=user_data.email,
            created_at=datetime.utcnow()
        )
    
    async def _update_last_login(self, user_id: str):
        """更新最后登录时间"""
        # 实际实现应该更新数据库
        pass
    
    async def _get_user_permissions_cached(self, user_id: str) -> Set[Permission]:
        """获取用户权限（带缓存）"""
        # 检查缓存是否有效
        if (user_id in self._permission_cache and 
            user_id in self._cache_expire_time and
            datetime.utcnow() < self._cache_expire_time[user_id]):
            return self._permission_cache[user_id]
        
        # 从数据库获取权限
        user = await self._get_user_by_id(user_id)
        if not user:
            return set()
        
        permissions = set(user.permissions)
        
        # 更新缓存
        self._permission_cache[user_id] = permissions
        self._cache_expire_time[user_id] = datetime.utcnow() + timedelta(
            seconds=self._cache_ttl_seconds
        )
        
        return permissions
    
    async def _is_account_locked(self, username: str, ip_address: str) -> bool:
        """检查账户是否被锁定"""
        key = f"{username}:{ip_address}"
        if key not in self.login_attempts:
            return False
        
        # 清理过期的登录尝试记录
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.lockout_duration_minutes)
        self.login_attempts[key] = [
            attempt for attempt in self.login_attempts[key]
            if attempt > cutoff_time
        ]
        
        return len(self.login_attempts[key]) >= self.max_login_attempts
    
    async def _record_failed_login(self, username: str, ip_address: str):
        """记录登录失败"""
        key = f"{username}:{ip_address}"
        if key not in self.login_attempts:
            self.login_attempts[key] = []
        
        self.login_attempts[key].append(datetime.utcnow())
    
    async def _clear_failed_logins(self, username: str, ip_address: str):
        """清除登录失败记录"""
        key = f"{username}:{ip_address}"
        self.login_attempts.pop(key, None)
    
    async def _verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        key_hash = self._hash_api_key(api_key)
        key_info = self.api_keys.get(key_hash)
        
        if key_info:
            # 更新使用统计
            key_info['usage_count'] += 1
            key_info['last_used'] = datetime.utcnow()
        
        return key_info
    
    async def _check_api_key_rate_limit(self, api_key: str, key_info: Dict[str, Any]):
        """检查API密钥速率限制"""
        # 简化实现，实际应该使用滑动窗口或令牌桶算法
        rate_limit = key_info.get('rate_limit', 1000)
        # 这里可以实现更复杂的速率限制逻辑
        pass
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        try:
            current_time = datetime.utcnow()
            session_timeout = timedelta(hours=24)  # 24小时会话超时
            
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if current_time - session.last_activity > session_timeout
            ]
            
            for session_id in expired_sessions:
                self.active_sessions.pop(session_id, None)
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
    
    async def _periodic_cleanup(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时运行一次
                await self._cleanup_expired_sessions()
                
                # 清理过期的登录尝试记录
                cutoff_time = datetime.utcnow() - timedelta(
                    minutes=self.lockout_duration_minutes * 2
                )
                
                keys_to_remove = []
                for key, attempts in self.login_attempts.items():
                    self.login_attempts[key] = [
                        attempt for attempt in attempts if attempt > cutoff_time
                    ]
                    if not self.login_attempts[key]:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    self.login_attempts.pop(key, None)
                
                # 清理权限缓存
                current_time = datetime.utcnow()
                expired_cache_keys = [
                    user_id for user_id, expire_time in self._cache_expire_time.items()
                    if current_time >= expire_time
                ]
                
                for user_id in expired_cache_keys:
                    self._permission_cache.pop(user_id, None)
                    self._cache_expire_time.pop(user_id, None)
                
            except Exception as e:
                logger.error("Periodic cleanup failed", error=str(e))
    
    async def get_active_sessions(self) -> List[SessionInfo]:
        """获取活跃会话列表"""
        return list(self.active_sessions.values())
    
    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        return self.active_sessions.get(session_id)
    
    async def invalidate_user_sessions(self, user_id: str):
        """使用户所有会话失效"""
        sessions_to_remove = [
            session_id for session_id, session in self.active_sessions.items()
            if session.user_id == user_id
        ]
        
        for session_id in sessions_to_remove:
            self.active_sessions.pop(session_id, None)
        
        logger.info(
            "User sessions invalidated",
            user_id=user_id,
            session_count=len(sessions_to_remove)
        )