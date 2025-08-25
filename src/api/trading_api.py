"""
量化交易系统核心API服务

提供完整的RESTful API接口，包括：
- 交易策略控制
- 认证和权限管理
- 系统监控和健康检查
- 信号查询和订阅
- WebSocket实时通信
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn

from .models.auth_models import (
    LoginRequest, TokenResponse, UserInfo, SessionInfo, Permission, 
    CreateUserRequest, RefreshTokenRequest, APIKeyRequest
)
from .models.request_models import (
    StrategyStartRequest, StrategyStopRequest, StrategyRestartRequest,
    StrategyStatusRequest, StrategyConfigRequest, SystemHealthRequest,
    SignalHistoryRequest, PaginationRequest, TimeRange
)
from .models.response_models import (
    APIResponse, ResponseStatus, StrategyStatusResponse, SystemHealthResponse,
    SignalHistoryResponse, PaginatedResponse, StrategyActionResponse,
    BatchActionResponse, StrategyInstance, SignalData
)
from .models.error_models import (
    APIError, ErrorResponse, AuthenticationError, AuthorizationError,
    ValidationError, ResourceNotFoundError, RateLimitError,
    get_http_status_code, ErrorCode
)
from .auth_manager import AuthenticationManager
from .rate_limiter import RateLimiter, rate_limit_middleware
from .request_validator import RequestValidator

from src.strategy.strategy_manager import StrategyManager
from src.strategy.signal_aggregator import SignalAggregator
from src.config import Config
# from src.core.database import Database  # 暂时注释掉

logger = structlog.get_logger(__name__)


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}  # connection_id -> subscription_info
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection established: {connection_id}")
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        self.active_connections.pop(connection_id, None)
        self.subscriptions.pop(connection_id, None)
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def subscribe(self, connection_id: str, channels: List[str], filters: Dict[str, Any] = None):
        """订阅频道"""
        if connection_id in self.active_connections:
            self.subscriptions[connection_id] = {
                'channels': channels,
                'filters': filters or {},
                'subscribed_at': datetime.utcnow()
            }
    
    async def broadcast(self, channel: str, message: Dict[str, Any]):
        """广播消息到订阅了指定频道的连接"""
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            subscription = self.subscriptions.get(connection_id)
            if not subscription or channel not in subscription['channels']:
                continue
            
            try:
                await websocket.send_json({
                    'channel': channel,
                    'data': message,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # 清理断开的连接
        for connection_id in disconnected_connections:
            await self.disconnect(connection_id)


class TradingAPI:
    """量化交易API服务器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.database = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.signal_aggregator: Optional[SignalAggregator] = None
        self.auth_manager: Optional[AuthenticationManager] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.request_validator: Optional[RequestValidator] = None
        
        # 使用新的WebSocket管理器
        from ..websocket import WebSocketManager, WebSocketConfig
        ws_config = WebSocketConfig(
            host=config.get('websocket.host', '0.0.0.0'),
            port=config.get('websocket.port', 8765),
            max_connections=config.get('websocket.max_connections', 1000),
            ping_interval=config.get('websocket.ping_interval', 30),
            connection_timeout=config.get('websocket.connection_timeout', 300),
            auth_required=config.get('websocket.auth_required', True),
            compression_enabled=config.get('websocket.compression_enabled', True)
        )
        self.websocket_manager = WebSocketManager(ws_config)
        
        # 服务器配置
        self.host = config.get('api.host', '0.0.0.0')
        self.port = config.get('api.port', 8000)
        self.debug = config.get('api.debug', False)
        self.enable_docs = config.get('api.enable_docs', True)
        
        # 创建FastAPI应用
        self.app = self._create_app()
        
        # 安全认证
        self.security = HTTPBearer()
        
        logger.info("Trading API initialized with WebSocket support")
    
    def _create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动时初始化
            await self.initialize()
            yield
            # 关闭时清理
            await self.shutdown()
        
        app = FastAPI(
            title="Quantification Trading API",
            description="量化交易系统REST API",
            version="1.0.0",
            docs_url="/docs" if self.enable_docs else None,
            redoc_url="/redoc" if self.enable_docs else None,
            openapi_url="/openapi.json" if self.enable_docs else None,
            lifespan=lifespan
        )
        
        # 添加中间件
        self._add_middleware(app)
        
        # 添加路由
        self._add_routes(app)
        
        # 异常处理
        self._add_exception_handlers(app)
        
        return app
    
    def _add_middleware(self, app: FastAPI):
        """添加中间件"""
        # CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('api.cors.allow_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip压缩
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # 请求ID中间件
        @app.middleware("http")
        async def add_request_id(request: Request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        
        # 速率限制中间件
        @app.middleware("http")
        async def rate_limit_middleware_wrapper(request: Request, call_next):
            if self.rate_limiter:
                # 获取当前会话（如果存在）
                session = getattr(request.state, 'session', None)
                return await rate_limit_middleware(request, call_next, self.rate_limiter, session)
            return await call_next(request)
        
        # 认证中间件
        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # 跳过不需要认证的路径
            public_paths = ["/", "/health", "/docs", "/redoc", "/openapi.json", "/auth/login", "/auth/refresh"]
            if any(request.url.path.startswith(path) for path in public_paths):
                return await call_next(request)
            
            try:
                # 尝试获取认证信息
                session = await self._get_current_session(request)
                request.state.session = session
                
                return await call_next(request)
            
            except AuthenticationError as e:
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(
                        error=e,
                        path=str(request.url.path),
                        method=request.method
                    ).dict()
                )
    
    def _add_routes(self, app: FastAPI):
        """添加API路由"""
        
        # ===== 根路由 =====
        @app.get("/")
        async def root():
            return {
                "name": "Quantification Trading API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # ===== 健康检查 =====
        @app.get("/health", response_model=APIResponse[SystemHealthResponse])
        async def health_check(request: SystemHealthRequest = Depends()):
            return await self.get_system_health(request)
        
        # ===== 认证相关路由 =====
        @app.post("/auth/login", response_model=APIResponse[TokenResponse])
        async def login(request: Request, login_data: LoginRequest):
            return await self.authenticate_user(request, login_data)
        
        @app.post("/auth/refresh", response_model=APIResponse[TokenResponse])
        async def refresh_token(refresh_data: RefreshTokenRequest):
            return await self.refresh_access_token(refresh_data)
        
        @app.post("/auth/logout")
        async def logout(request: Request):
            return await self.logout_user(request)
        
        @app.get("/auth/me", response_model=APIResponse[UserInfo])
        async def get_current_user(request: Request):
            return await self.get_user_info(request)
        
        @app.post("/auth/api-keys", response_model=APIResponse[Dict[str, Any]])
        async def create_api_key(request: Request, api_key_data: APIKeyRequest):
            return await self.create_api_key(request, api_key_data)
        
        # ===== 策略控制路由 =====
        @app.post("/strategies/{strategy_id}/start", response_model=APIResponse[StrategyActionResponse])
        async def start_strategy(request: Request, strategy_id: str, start_data: StrategyStartRequest):
            return await self.start_strategy(request, strategy_id, start_data)
        
        @app.post("/strategies/{strategy_id}/stop", response_model=APIResponse[StrategyActionResponse])
        async def stop_strategy(request: Request, strategy_id: str, stop_data: StrategyStopRequest):
            return await self.stop_strategy(request, strategy_id, stop_data)
        
        @app.post("/strategies/{strategy_id}/restart", response_model=APIResponse[StrategyActionResponse])
        async def restart_strategy(request: Request, strategy_id: str, restart_data: StrategyRestartRequest):
            return await self.restart_strategy(request, strategy_id, restart_data)
        
        @app.get("/strategies", response_model=APIResponse[PaginatedResponse[StrategyInstance]])
        async def list_strategies(request: Request, 
                                 page: int = 1, 
                                 page_size: int = 20,
                                 status_filter: Optional[str] = None):
            return await self.list_strategies(request, page, page_size, status_filter)
        
        @app.get("/strategies/{strategy_id}/status", response_model=APIResponse[StrategyInstance])
        async def get_strategy_status(request: Request, strategy_id: str):
            return await self.get_strategy_status(request, strategy_id)
        
        @app.put("/strategies/{strategy_id}/config", response_model=APIResponse[Dict[str, Any]])
        async def update_strategy_config(request: Request, strategy_id: str, config_data: StrategyConfigRequest):
            return await self.update_strategy_config(request, strategy_id, config_data)
        
        # ===== 信号相关路由 =====
        @app.get("/signals/history", response_model=APIResponse[PaginatedResponse[SignalData]])
        async def get_signal_history(request: Request,
                                   page: int = 1,
                                   page_size: int = 50,
                                   strategy_ids: Optional[str] = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None):
            return await self.get_signal_history(request, page, page_size, strategy_ids, start_time, end_time)
        
        @app.get("/signals/aggregation/statistics")
        async def get_aggregation_statistics(request: Request):
            return await self.get_aggregation_statistics(request)
        
        # ===== 系统管理路由 =====
        @app.get("/system/status", response_model=APIResponse[SystemHealthResponse])
        async def get_system_status(request: Request):
            return await self.get_system_health(request, SystemHealthRequest(detailed=True))
        
        @app.get("/system/metrics")
        async def get_system_metrics(request: Request):
            return await self.get_system_metrics(request)
        
        # ===== WebSocket路由 =====
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket连接端点"""
            await self.websocket_manager.connection_manager.handle_connection(websocket, "/ws")
        
        # ===== WebSocket管理路由 =====
        @app.get("/websocket/stats")
        async def get_websocket_stats(request: Request):
            """获取WebSocket统计信息"""
            await self._require_permission(request, Permission.SYSTEM_MONITOR)
            
            stats = self.websocket_manager.get_system_stats()
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="WebSocket statistics retrieved successfully",
                data=stats
            )
        
        @app.get("/websocket/connections")
        async def list_websocket_connections(request: Request):
            """列出WebSocket连接"""
            await self._require_permission(request, Permission.SYSTEM_MONITOR)
            
            connections = []
            for conn_id in self.websocket_manager.connection_manager.get_active_connections():
                conn_info = self.websocket_manager.connection_manager.get_connection_info(conn_id)
                if conn_info:
                    connections.append({
                        "connection_id": conn_info.connection_id,
                        "user_id": conn_info.user_id,
                        "client_ip": conn_info.client_ip,
                        "status": conn_info.status.value,
                        "connected_at": conn_info.connected_at.isoformat(),
                        "last_activity": conn_info.last_activity.isoformat(),
                        "subscriptions": [sub.value for sub in conn_info.subscriptions]
                    })
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="WebSocket connections retrieved successfully",
                data=connections
            )
        
        @app.post("/websocket/broadcast")
        async def broadcast_message(request: Request, broadcast_data: dict):
            """手动广播消息（管理员功能）"""
            await self._require_permission(request, Permission.SYSTEM_ADMIN)
            
            from ..websocket.models import MessageType, SubscriptionType
            
            try:
                message_type = MessageType(broadcast_data.get("message_type", "system_monitor"))
                subscription_type = SubscriptionType(broadcast_data.get("subscription_type", "system_monitor"))
                data = broadcast_data.get("data", {})
                priority = broadcast_data.get("priority", 0)
                
                success_count = await self.websocket_manager.message_broadcaster.handle_system_broadcast(
                    message_type, data
                )
                
                return APIResponse(
                    status=ResponseStatus.SUCCESS,
                    message=f"Message broadcasted to {success_count} connections",
                    data={"success_count": success_count}
                )
                
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
                return APIResponse(
                    status=ResponseStatus.ERROR,
                    message="Failed to broadcast message",
                    data={"error": str(e)}
                )
    
    def _add_exception_handlers(self, app: FastAPI):
        """添加异常处理器"""
        
        @app.exception_handler(AuthenticationError)
        async def auth_exception_handler(request: Request, exc: AuthenticationError):
            return JSONResponse(
                status_code=get_http_status_code(exc.code),
                content=ErrorResponse(
                    error=exc,
                    path=str(request.url.path),
                    method=request.method
                ).dict()
            )
        
        @app.exception_handler(AuthorizationError)
        async def authz_exception_handler(request: Request, exc: AuthorizationError):
            return JSONResponse(
                status_code=get_http_status_code(exc.code),
                content=ErrorResponse(
                    error=exc,
                    path=str(request.url.path),
                    method=request.method
                ).dict()
            )
        
        @app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            return JSONResponse(
                status_code=get_http_status_code(exc.code),
                content=ErrorResponse(
                    error=exc,
                    path=str(request.url.path),
                    method=request.method
                ).dict()
            )
        
        @app.exception_handler(RateLimitError)
        async def rate_limit_exception_handler(request: Request, exc: RateLimitError):
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error=exc,
                    path=str(request.url.path),
                    method=request.method
                ).dict(),
                headers={
                    "Retry-After": str(exc.retry_after),
                    "X-RateLimit-Limit": str(exc.limit),
                    "X-RateLimit-Remaining": str(exc.remaining)
                }
            )
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            error = APIError(
                code=ErrorCode.INTERNAL_ERROR,
                message=exc.detail,
                context={"status_code": exc.status_code}
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error=error,
                    path=str(request.url.path),
                    method=request.method
                ).dict()
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            
            error = APIError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Internal server error",
                details=str(exc) if self.debug else None
            )
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error=error,
                    path=str(request.url.path),
                    method=request.method
                ).dict()
            )
    
    async def initialize(self):
        """初始化API服务"""
        try:
            logger.info("Initializing Trading API...")
            
            # 初始化数据库（暂时跳过）
            # self.database = Database(self.config)
            # await self.database.initialize()
            
            # 初始化认证管理器
            self.auth_manager = AuthenticationManager(self.config, self.database)
            await self.auth_manager.initialize()
            
            # 初始化速率限制器
            self.rate_limiter = RateLimiter(self.config)
            
            # 初始化请求验证器
            self.request_validator = RequestValidator(self.config)
            
            # 启动WebSocket管理器
            await self.websocket_manager.start()
            
            # 设置WebSocket集成
            if self.auth_manager:
                self.websocket_manager.set_auth_manager(self.auth_manager)
            
            # 这里应该从依赖注入或服务注册中心获取这些组件
            # 暂时使用简化实现
            logger.warning("Strategy Manager and Signal Aggregator need to be injected")
            
            logger.info("Trading API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Trading API: {e}")
            raise
    
    async def shutdown(self):
        """关闭API服务"""
        try:
            logger.info("Shutting down Trading API...")
            
            # 停止WebSocket管理器
            await self.websocket_manager.stop()
            
            # 关闭数据库连接
            if self.database:
                await self.database.close()
            
            logger.info("Trading API shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during API shutdown: {e}")
    
    async def _get_current_session(self, request: Request) -> SessionInfo:
        """获取当前会话"""
        # 尝试从Authorization头获取Token
        authorization = request.headers.get('Authorization')
        if authorization and authorization.startswith('Bearer '):
            token = authorization[7:]
            return await self.auth_manager.verify_token(token)
        
        # 尝试从API Key头获取
        api_key = request.headers.get('X-API-Key')
        if api_key:
            client_ip = request.client.host if request.client else "unknown"
            return await self.auth_manager.authenticate_api_key(api_key, client_ip)
        
        raise AuthenticationError("No valid authentication found")
    
    async def _require_permission(self, request: Request, permission: Permission):
        """要求特定权限"""
        session = request.state.session
        await self.auth_manager.require_permission(session, permission)
    
    # ===== API端点实现 =====
    
    async def authenticate_user(self, request: Request, login_data: LoginRequest) -> APIResponse[TokenResponse]:
        """用户认证"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get('User-Agent')
            
            token_response = await self.auth_manager.authenticate_user(
                login_data.username,
                login_data.password,
                client_ip,
                user_agent
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Authentication successful",
                data=token_response
            )
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    async def refresh_access_token(self, refresh_data: RefreshTokenRequest) -> APIResponse[TokenResponse]:
        """刷新访问令牌"""
        try:
            token_response = await self.auth_manager.refresh_token(refresh_data.refresh_token)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Token refreshed successfully",
                data=token_response
            )
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise
    
    async def logout_user(self, request: Request) -> APIResponse[Dict[str, Any]]:
        """用户登出"""
        try:
            session = request.state.session
            await self.auth_manager.logout(session.session_id)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Logout successful",
                data={"session_id": session.session_id}
            )
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            raise
    
    async def get_user_info(self, request: Request) -> APIResponse[UserInfo]:
        """获取用户信息"""
        try:
            session = request.state.session
            user = await self.auth_manager._get_user_by_id(session.user_id)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="User info retrieved successfully",
                data=user
            )
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            raise
    
    async def create_api_key(self, request: Request, api_key_data: APIKeyRequest) -> APIResponse[Dict[str, Any]]:
        """创建API密钥"""
        try:
            await self._require_permission(request, Permission.API_ADMIN)
            session = request.state.session
            
            api_key_response = await self.auth_manager.create_api_key(
                session.user_id,
                api_key_data.name,
                api_key_data.permissions,
                api_key_data.expires_in_days,
                api_key_data.rate_limit
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="API key created successfully",
                data=api_key_response.dict()
            )
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise
    
    async def start_strategy(self, request: Request, strategy_id: str, 
                           start_data: StrategyStartRequest) -> APIResponse[StrategyActionResponse]:
        """启动策略"""
        try:
            await self._require_permission(request, Permission.STRATEGY_START)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            # 验证策略ID
            if start_data.strategy_id != strategy_id:
                raise ValidationError("Strategy ID mismatch")
            
            # 启动策略
            success = await self.strategy_manager.start_strategy(
                strategy_id,
                start_data.config,
                start_data.force
            )
            
            response = StrategyActionResponse(
                strategy_id=strategy_id,
                action="start",
                success=success,
                message="Strategy started successfully" if success else "Failed to start strategy"
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS if success else ResponseStatus.ERROR,
                message=response.message,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Failed to start strategy {strategy_id}: {e}")
            raise
    
    async def stop_strategy(self, request: Request, strategy_id: str,
                          stop_data: StrategyStopRequest) -> APIResponse[StrategyActionResponse]:
        """停止策略"""
        try:
            await self._require_permission(request, Permission.STRATEGY_STOP)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            success = await self.strategy_manager.stop_strategy(
                strategy_id,
                stop_data.force,
                stop_data.save_state
            )
            
            response = StrategyActionResponse(
                strategy_id=strategy_id,
                action="stop",
                success=success,
                message="Strategy stopped successfully" if success else "Failed to stop strategy"
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS if success else ResponseStatus.ERROR,
                message=response.message,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Failed to stop strategy {strategy_id}: {e}")
            raise
    
    async def restart_strategy(self, request: Request, strategy_id: str,
                             restart_data: StrategyRestartRequest) -> APIResponse[StrategyActionResponse]:
        """重启策略"""
        try:
            await self._require_permission(request, Permission.STRATEGY_RESTART)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            success = await self.strategy_manager.restart_strategy(
                strategy_id,
                restart_data.config,
                restart_data.preserve_state
            )
            
            response = StrategyActionResponse(
                strategy_id=strategy_id,
                action="restart",
                success=success,
                message="Strategy restarted successfully" if success else "Failed to restart strategy"
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS if success else ResponseStatus.ERROR,
                message=response.message,
                data=response
            )
            
        except Exception as e:
            logger.error(f"Failed to restart strategy {strategy_id}: {e}")
            raise
    
    async def list_strategies(self, request: Request, page: int, page_size: int,
                            status_filter: Optional[str] = None) -> APIResponse[PaginatedResponse[StrategyInstance]]:
        """获取策略列表"""
        try:
            await self._require_permission(request, Permission.STRATEGY_VIEW)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            # 获取策略列表
            strategies_data = await self.strategy_manager.list_strategies()
            
            # 转换为响应模型
            strategies = []
            for strategy_data in strategies_data:
                strategy = StrategyInstance(
                    strategy_id=strategy_data['strategy_id'],
                    name=strategy_data.get('name', strategy_data['strategy_id']),
                    strategy_type=strategy_data['strategy_type'],
                    status=strategy_data['status'],
                    description=strategy_data.get('description'),
                    created_at=strategy_data.get('created_at', datetime.utcnow()),
                    updated_at=strategy_data.get('updated_at', datetime.utcnow())
                )
                strategies.append(strategy)
            
            # 应用状态过滤
            if status_filter:
                strategies = [s for s in strategies if s.status == status_filter]
            
            # 分页
            total = len(strategies)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_strategies = strategies[start_idx:end_idx]
            
            paginated_response = PaginatedResponse.create(
                paginated_strategies, total, page, page_size
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Strategies retrieved successfully",
                data=paginated_response
            )
            
        except Exception as e:
            logger.error(f"Failed to list strategies: {e}")
            raise
    
    async def get_strategy_status(self, request: Request, strategy_id: str) -> APIResponse[StrategyInstance]:
        """获取策略状态"""
        try:
            await self._require_permission(request, Permission.STRATEGY_VIEW)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            strategy_data = await self.strategy_manager.get_strategy_status(strategy_id)
            
            if not strategy_data:
                raise ResourceNotFoundError("strategy", strategy_id)
            
            strategy = StrategyInstance(
                strategy_id=strategy_data['strategy_id'],
                name=strategy_data.get('name', strategy_data['strategy_id']),
                strategy_type=strategy_data['strategy_type'],
                status=strategy_data['status'],
                description=strategy_data.get('description'),
                created_at=strategy_data.get('created_at', datetime.utcnow()),
                updated_at=strategy_data.get('updated_at', datetime.utcnow()),
                started_at=strategy_data.get('started_at'),
                stopped_at=strategy_data.get('stopped_at'),
                error_message=strategy_data.get('error_message')
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Strategy status retrieved successfully",
                data=strategy
            )
            
        except Exception as e:
            logger.error(f"Failed to get strategy status for {strategy_id}: {e}")
            raise
    
    async def update_strategy_config(self, request: Request, strategy_id: str,
                                   config_data: StrategyConfigRequest) -> APIResponse[Dict[str, Any]]:
        """更新策略配置"""
        try:
            await self._require_permission(request, Permission.STRATEGY_CONFIG)
            
            if not self.strategy_manager:
                raise HTTPException(status_code=503, detail="Strategy manager not available")
            
            # 这里应该实现配置更新逻辑
            # 简化实现
            result = {
                "strategy_id": strategy_id,
                "config_updated": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Strategy configuration updated successfully",
                data=result
            )
            
        except Exception as e:
            logger.error(f"Failed to update strategy config for {strategy_id}: {e}")
            raise
    
    async def get_signal_history(self, request: Request, page: int, page_size: int,
                               strategy_ids: Optional[str], start_time: Optional[datetime],
                               end_time: Optional[datetime]) -> APIResponse[PaginatedResponse[SignalData]]:
        """获取信号历史"""
        try:
            await self._require_permission(request, Permission.DATA_READ)
            
            if not self.signal_aggregator:
                raise HTTPException(status_code=503, detail="Signal aggregator not available")
            
            # 解析策略ID列表
            strategy_id_list = None
            if strategy_ids:
                strategy_id_list = [s.strip() for s in strategy_ids.split(',')]
            
            # 获取信号历史（简化实现）
            signals = []
            
            # 创建模拟信号数据
            for i in range(min(page_size, 10)):
                signal = SignalData(
                    signal_id=f"signal_{i}",
                    strategy_id=strategy_id_list[0] if strategy_id_list else "demo_strategy",
                    signal_type="trade",
                    action="buy",
                    confidence=0.85,
                    strength=0.75,
                    symbol="BTCUSDT",
                    price=50000.0,
                    volume=0.1,
                    timestamp=datetime.utcnow() - timedelta(minutes=i)
                )
                signals.append(signal)
            
            paginated_response = PaginatedResponse.create(
                signals, len(signals), page, page_size
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Signal history retrieved successfully",
                data=paginated_response
            )
            
        except Exception as e:
            logger.error(f"Failed to get signal history: {e}")
            raise
    
    async def get_aggregation_statistics(self, request: Request) -> APIResponse[Dict[str, Any]]:
        """获取聚合统计信息"""
        try:
            await self._require_permission(request, Permission.DATA_READ)
            
            if not self.signal_aggregator:
                raise HTTPException(status_code=503, detail="Signal aggregator not available")
            
            stats = await self.signal_aggregator.get_aggregation_statistics()
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="Aggregation statistics retrieved successfully",
                data=stats.dict() if hasattr(stats, 'dict') else stats
            )
            
        except Exception as e:
            logger.error(f"Failed to get aggregation statistics: {e}")
            raise
    
    async def get_system_health(self, request: SystemHealthRequest) -> APIResponse[SystemHealthResponse]:
        """获取系统健康状态"""
        try:
            import psutil
            
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            from ..models.response_models import ResourceUsage, ComponentHealth
            
            resource_usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_percent=memory.percent,
                disk_usage_gb=disk.used / (1024 * 1024 * 1024),
                disk_percent=(disk.used / disk.total) * 100
            )
            
            # 检查组件健康状态
            components = []
            
            # 数据库组件
            db_status = "healthy" if self.database else "unavailable"
            components.append(ComponentHealth(
                component_name="database",
                status=db_status,
                uptime_seconds=0.0  # 应该从实际组件获取
            ))
            
            # 策略管理器组件
            sm_status = "healthy" if self.strategy_manager else "unavailable"
            components.append(ComponentHealth(
                component_name="strategy_manager",
                status=sm_status,
                uptime_seconds=0.0
            ))
            
            # 信号聚合器组件
            sa_status = "healthy" if self.signal_aggregator else "unavailable"
            components.append(ComponentHealth(
                component_name="signal_aggregator",
                status=sa_status,
                uptime_seconds=0.0
            ))
            
            overall_status = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
            
            health_response = SystemHealthResponse(
                overall_status=overall_status,
                uptime_seconds=0.0,  # 应该跟踪实际启动时间
                version="1.0.0",
                environment="development",
                components=components,
                resource_usage=resource_usage,
                active_strategies=0,  # 应该从策略管理器获取
                total_requests=self.rate_limiter.stats['total_requests'] if self.rate_limiter else 0,
                error_rate=0.0,  # 应该计算实际错误率
                average_response_time_ms=0.0  # 应该计算实际响应时间
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="System health retrieved successfully",
                data=health_response
            )
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            raise
    
    async def get_system_metrics(self, request: Request) -> APIResponse[Dict[str, Any]]:
        """获取系统指标"""
        try:
            await self._require_permission(request, Permission.SYSTEM_MONITOR)
            
            # 获取各种系统指标
            metrics = {
                "api": {
                    "total_requests": self.rate_limiter.stats['total_requests'] if self.rate_limiter else 0,
                    "blocked_requests": self.rate_limiter.stats['blocked_requests'] if self.rate_limiter else 0,
                    "active_connections": len(self.websocket_manager.active_connections),
                    "active_subscriptions": len(self.websocket_manager.subscriptions)
                },
                "auth": {
                    "active_sessions": len(self.auth_manager.active_sessions) if self.auth_manager else 0,
                    "api_keys": len(self.auth_manager.api_keys) if self.auth_manager else 0
                },
                "strategies": {
                    "total_strategies": 0,  # 应该从策略管理器获取
                    "running_strategies": 0,
                    "stopped_strategies": 0
                }
            }
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                message="System metrics retrieved successfully",
                data=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            raise
    
    async def _handle_websocket_message(self, connection_id: str, data: Dict[str, Any]):
        """处理WebSocket消息"""
        try:
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                channels = data.get('channels', [])
                filters = data.get('filters', {})
                await self.websocket_manager.subscribe(connection_id, channels, filters)
                
                # 发送订阅确认
                websocket = self.websocket_manager.active_connections.get(connection_id)
                if websocket:
                    await websocket.send_json({
                        'type': 'subscription_confirmed',
                        'channels': channels,
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            elif message_type == 'unsubscribe':
                await self.websocket_manager.disconnect(connection_id)
            
            elif message_type == 'ping':
                # 发送pong响应
                websocket = self.websocket_manager.active_connections.get(connection_id)
                if websocket:
                    await websocket.send_json({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message from {connection_id}: {e}")
            await self.websocket_manager.disconnect(connection_id)
    
    def set_strategy_manager(self, strategy_manager: StrategyManager):
        """设置策略管理器"""
        self.strategy_manager = strategy_manager
        
        # 将策略管理器集成到WebSocket管理器
        self.websocket_manager.set_strategy_manager(strategy_manager)
        
        # 设置策略状态变化回调，实时推送状态更新
        if hasattr(strategy_manager, 'add_status_change_callback'):
            strategy_manager.add_status_change_callback(self._on_strategy_status_change)
        
        logger.info("Strategy manager attached to API with WebSocket integration")
    
    def set_signal_aggregator(self, signal_aggregator: SignalAggregator):
        """设置信号聚合器"""
        self.signal_aggregator = signal_aggregator
        
        # 将信号聚合器集成到WebSocket管理器
        self.websocket_manager.set_signal_aggregator(signal_aggregator)
        
        # 设置信号回调，实时推送交易信号
        if hasattr(signal_aggregator, 'add_signal_callback'):
            signal_aggregator.add_signal_callback(self._on_trading_signal)
        
        logger.info("Signal aggregator attached to API with WebSocket integration")
    
    
    async def _on_strategy_status_change(self, strategy_id: str, status_data: Dict[str, Any]):
        """策略状态变化回调，实时推送策略状态更新"""
        try:
            await self.websocket_manager.broadcast_strategy_status(strategy_id, status_data)
            logger.debug(f"Broadcasted strategy status update for {strategy_id}")
        except Exception as e:
            logger.error(f"Failed to broadcast strategy status update: {e}")
    
    async def _on_trading_signal(self, signal_data: Dict[str, Any]):
        """交易信号回调，实时推送交易信号"""
        try:
            await self.websocket_manager.broadcast_trading_signal(signal_data)
            logger.debug(f"Broadcasted trading signal: {signal_data.get('symbol', 'Unknown')} {signal_data.get('signal', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to broadcast trading signal: {e}")
    
    async def broadcast_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """广播系统事件"""
        try:
            from ..websocket.models import MessageType
            
            # 根据事件类型选择合适的消息类型
            message_type_mapping = {
                'risk_alert': MessageType.RISK_ALERT,
                'market_data': MessageType.MARKET_DATA,
                'system_monitor': MessageType.SYSTEM_MONITOR,
                'order_update': MessageType.ORDER_UPDATE,
                'position_update': MessageType.POSITION_UPDATE,
                'performance_metrics': MessageType.PERFORMANCE_METRICS
            }
            
            message_type = message_type_mapping.get(event_type, MessageType.SYSTEM_MONITOR)
            
            if message_type == MessageType.RISK_ALERT:
                await self.websocket_manager.broadcast_risk_alert(event_data)
            elif message_type == MessageType.MARKET_DATA:
                await self.websocket_manager.broadcast_market_data(event_data)
            elif message_type == MessageType.SYSTEM_MONITOR:
                await self.websocket_manager.broadcast_system_monitor(event_data)
            else:
                # 通用系统广播
                success_count = await self.websocket_manager.message_broadcaster.handle_system_broadcast(
                    message_type, event_data
                )
                logger.debug(f"Broadcasted {event_type} event to {success_count} connections")
            
        except Exception as e:
            logger.error(f"Failed to broadcast system event {event_type}: {e}")

    def run(self, **kwargs):
        """运行API服务器"""
        config = {
            "host": kwargs.get("host", self.host),
            "port": kwargs.get("port", self.port),
            "log_level": "debug" if self.debug else "info",
            "access_log": True,
            "loop": "uvloop" if not kwargs.get("reload") else "asyncio",
            "reload": kwargs.get("reload", False),
            "workers": kwargs.get("workers", 1)
        }
        
        logger.info(f"Starting Trading API server on {config['host']}:{config['port']}")
        uvicorn.run(self.app, **config)