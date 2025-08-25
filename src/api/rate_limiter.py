"""
API速率限制中间件

提供多种限流策略，包括：
- 基于IP的限流
- 基于用户的限流
- 基于API密钥的限流
- 自适应限流
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from .models.auth_models import SessionInfo
from .models.error_models import RateLimitError, ErrorCode
from src.config import Config

logger = structlog.get_logger(__name__)


class TokenBucket:
    """令牌桶算法实现"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """消耗令牌"""
        async with self._lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def _refill(self):
        """填充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        
        # 计算应该添加的令牌数量
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def get_available_tokens(self) -> int:
        """获取可用令牌数量"""
        async with self._lock:
            await self._refill()
            return int(self.tokens)
    
    async def get_refill_time(self, required_tokens: int) -> float:
        """获取填充所需令牌的时间"""
        if required_tokens <= await self.get_available_tokens():
            return 0.0
        
        missing_tokens = required_tokens - await self.get_available_tokens()
        return missing_tokens / self.refill_rate


class SlidingWindowCounter:
    """滑动窗口计数器"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # 窗口大小（秒）
        self.max_requests = max_requests
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> Tuple[bool, int, float]:
        """检查是否允许请求"""
        async with self._lock:
            now = time.time()
            
            # 清理过期请求
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                # 计算重置时间
                reset_time = self.requests[0] + self.window_size - now
                return False, len(self.requests), reset_time
            
            # 记录当前请求
            self.requests.append(now)
            return True, len(self.requests), 0.0
    
    async def get_current_count(self) -> int:
        """获取当前窗口内的请求数"""
        async with self._lock:
            now = time.time()
            # 清理过期请求
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            return len(self.requests)


class AdaptiveRateLimiter:
    """自适应速率限制器"""
    
    def __init__(self, base_limit: int, window_size: int = 60):
        self.base_limit = base_limit
        self.window_size = window_size
        self.current_limit = base_limit
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # 调整间隔（秒）
        self.counter = SlidingWindowCounter(window_size, self.current_limit)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> Tuple[bool, int, float]:
        """检查是否允许请求"""
        await self._adjust_limit()
        return await self.counter.is_allowed()
    
    async def record_success(self):
        """记录成功请求"""
        async with self._lock:
            self.success_count += 1
    
    async def record_error(self):
        """记录错误请求"""
        async with self._lock:
            self.error_count += 1
    
    async def _adjust_limit(self):
        """动态调整限制"""
        async with self._lock:
            now = time.time()
            if now - self.last_adjustment < self.adjustment_interval:
                return
            
            total_requests = self.success_count + self.error_count
            if total_requests == 0:
                return
            
            error_rate = self.error_count / total_requests
            
            # 根据错误率调整限制
            if error_rate > 0.1:  # 错误率超过10%，降低限制
                new_limit = max(self.base_limit // 2, int(self.current_limit * 0.8))
            elif error_rate < 0.01:  # 错误率低于1%，提高限制
                new_limit = min(self.base_limit * 2, int(self.current_limit * 1.2))
            else:
                new_limit = self.current_limit
            
            if new_limit != self.current_limit:
                self.current_limit = new_limit
                self.counter = SlidingWindowCounter(self.window_size, self.current_limit)
                logger.info(
                    "Rate limit adjusted",
                    old_limit=self.current_limit,
                    new_limit=new_limit,
                    error_rate=error_rate
                )
            
            # 重置计数器
            self.success_count = 0
            self.error_count = 0
            self.last_adjustment = now


class RateLimitRule:
    """速率限制规则"""
    
    def __init__(self, 
                 limit: int,
                 window: int,
                 burst_limit: Optional[int] = None,
                 adaptive: bool = False):
        self.limit = limit
        self.window = window
        self.burst_limit = burst_limit or limit * 2
        self.adaptive = adaptive
        
        if adaptive:
            self.limiter = AdaptiveRateLimiter(limit, window)
        else:
            self.limiter = SlidingWindowCounter(window, limit)
        
        self.burst_limiter = TokenBucket(self.burst_limit, limit / window)


class RateLimiter:
    """API速率限制器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 默认限制规则
        self.default_rules = {
            'global': RateLimitRule(
                limit=config.get('rate_limit.global.requests_per_minute', 1000),
                window=60,
                adaptive=True
            ),
            'per_ip': RateLimitRule(
                limit=config.get('rate_limit.per_ip.requests_per_minute', 100),
                window=60
            ),
            'per_user': RateLimitRule(
                limit=config.get('rate_limit.per_user.requests_per_minute', 200),
                window=60
            ),
            'per_api_key': RateLimitRule(
                limit=config.get('rate_limit.per_api_key.requests_per_minute', 500),
                window=60
            ),
            'auth': RateLimitRule(
                limit=config.get('rate_limit.auth.requests_per_minute', 10),
                window=60
            )
        }
        
        # 动态限制规则存储
        self.ip_limiters: Dict[str, Dict[str, RateLimitRule]] = defaultdict(dict)
        self.user_limiters: Dict[str, Dict[str, RateLimitRule]] = defaultdict(dict)
        self.api_key_limiters: Dict[str, Dict[str, RateLimitRule]] = defaultdict(dict)
        self.path_limiters: Dict[str, RateLimitRule] = {}
        
        # 白名单和黑名单
        self.ip_whitelist = set(config.get('rate_limit.ip_whitelist', []))
        self.ip_blacklist = set(config.get('rate_limit.ip_blacklist', []))
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'last_reset': datetime.utcnow()
        }
        
        logger.info("Rate limiter initialized")
    
    async def check_rate_limit(self, request: Request, session: Optional[SessionInfo] = None) -> Optional[RateLimitError]:
        """检查速率限制"""
        try:
            client_ip = self._get_client_ip(request)
            path = request.url.path
            method = request.method
            
            # 更新统计
            self.stats['total_requests'] += 1
            
            # 检查IP黑名单
            if client_ip in self.ip_blacklist:
                self.stats['blocked_requests'] += 1
                return RateLimitError(
                    limit=0,
                    remaining=0,
                    reset_time=datetime.utcnow() + timedelta(hours=24),
                    retry_after=86400,
                    message="IP address is blacklisted"
                )
            
            # 检查IP白名单
            if client_ip in self.ip_whitelist:
                return None
            
            # 全局限制检查
            if not await self._check_global_limit():
                return await self._create_rate_limit_error('global')
            
            # IP限制检查
            ip_key = f"{client_ip}"
            if not await self._check_limit('per_ip', ip_key):
                return await self._create_rate_limit_error('per_ip', ip_key)
            
            # 路径特定限制检查
            path_key = f"{method}:{path}"
            if path_key in self.path_limiters:
                if not await self._check_custom_limit(path_key):
                    return await self._create_rate_limit_error('custom', path_key)
            
            # 用户限制检查
            if session and not session.is_api_session:
                user_key = f"{session.user_id}"
                if not await self._check_limit('per_user', user_key):
                    return await self._create_rate_limit_error('per_user', user_key)
            
            # API密钥限制检查
            if session and session.is_api_session:
                api_key = request.headers.get('X-API-Key')
                if api_key:
                    api_key_hash = self._hash_api_key(api_key)
                    if not await self._check_limit('per_api_key', api_key_hash):
                        return await self._create_rate_limit_error('per_api_key', api_key_hash)
            
            # 特殊路径限制（如认证端点）
            if self._is_auth_endpoint(path):
                auth_key = f"auth:{client_ip}"
                if not await self._check_limit('auth', auth_key):
                    return await self._create_rate_limit_error('auth', auth_key)
            
            return None
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            # 在错误情况下，允许请求通过，但记录错误
            return None
    
    async def record_request_result(self, request: Request, success: bool):
        """记录请求结果（用于自适应限流）"""
        try:
            # 更新全局自适应限制器
            global_rule = self.default_rules.get('global')
            if global_rule and hasattr(global_rule.limiter, 'record_success'):
                if success:
                    await global_rule.limiter.record_success()
                else:
                    await global_rule.limiter.record_error()
            
        except Exception as e:
            logger.error("Failed to record request result", error=str(e))
    
    def add_path_limit(self, method: str, path: str, limit: int, window: int = 60):
        """为特定路径添加限制"""
        path_key = f"{method}:{path}"
        self.path_limiters[path_key] = RateLimitRule(limit, window)
        logger.info(f"Added rate limit for {path_key}: {limit} requests per {window}s")
    
    def remove_path_limit(self, method: str, path: str):
        """移除特定路径的限制"""
        path_key = f"{method}:{path}"
        if path_key in self.path_limiters:
            del self.path_limiters[path_key]
            logger.info(f"Removed rate limit for {path_key}")
    
    def add_ip_to_whitelist(self, ip: str):
        """添加IP到白名单"""
        self.ip_whitelist.add(ip)
        logger.info(f"Added IP {ip} to whitelist")
    
    def add_ip_to_blacklist(self, ip: str):
        """添加IP到黑名单"""
        self.ip_blacklist.add(ip)
        logger.info(f"Added IP {ip} to blacklist")
    
    def remove_ip_from_whitelist(self, ip: str):
        """从白名单移除IP"""
        self.ip_whitelist.discard(ip)
        logger.info(f"Removed IP {ip} from whitelist")
    
    def remove_ip_from_blacklist(self, ip: str):
        """从黑名单移除IP"""
        self.ip_blacklist.discard(ip)
        logger.info(f"Removed IP {ip} from blacklist")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取速率限制统计信息"""
        return {
            'total_requests': self.stats['total_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'block_rate': (
                self.stats['blocked_requests'] / max(self.stats['total_requests'], 1)
            ),
            'active_ip_limiters': len(self.ip_limiters),
            'active_user_limiters': len(self.user_limiters),
            'active_api_key_limiters': len(self.api_key_limiters),
            'path_limiters': len(self.path_limiters),
            'ip_whitelist_size': len(self.ip_whitelist),
            'ip_blacklist_size': len(self.ip_blacklist),
            'last_reset': self.stats['last_reset']
        }
    
    async def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'last_reset': datetime.utcnow()
        }
    
    # ===== 私有方法 =====
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 检查代理头
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # 使用连接IP
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_auth_endpoint(self, path: str) -> bool:
        """检查是否为认证端点"""
        auth_endpoints = ['/auth/login', '/auth/refresh', '/auth/register']
        return any(path.startswith(endpoint) for endpoint in auth_endpoints)
    
    def _hash_api_key(self, api_key: str) -> str:
        """对API密钥进行哈希"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    async def _check_global_limit(self) -> bool:
        """检查全局限制"""
        global_rule = self.default_rules.get('global')
        if not global_rule:
            return True
        
        if hasattr(global_rule.limiter, 'is_allowed'):
            allowed, _, _ = await global_rule.limiter.is_allowed()
            return allowed
        else:
            allowed, _, _ = await global_rule.limiter.is_allowed()
            return allowed
    
    async def _check_limit(self, limit_type: str, key: str) -> bool:
        """检查特定类型的限制"""
        rule = self.default_rules.get(limit_type)
        if not rule:
            return True
        
        # 获取或创建限制器
        limiters = getattr(self, f"{limit_type.split('_')[1]}_limiters", {})
        if key not in limiters:
            limiters[key] = RateLimitRule(
                rule.limit, 
                rule.window, 
                rule.burst_limit,
                rule.adaptive
            )
        
        limiter = limiters[key].limiter
        
        if hasattr(limiter, 'is_allowed'):
            allowed, _, _ = await limiter.is_allowed()
        else:
            allowed, _, _ = await limiter.is_allowed()
        
        # 检查突发限制
        if allowed:
            burst_limiter = limiters[key].burst_limiter
            allowed = await burst_limiter.consume()
        
        return allowed
    
    async def _check_custom_limit(self, path_key: str) -> bool:
        """检查自定义路径限制"""
        rule = self.path_limiters.get(path_key)
        if not rule:
            return True
        
        allowed, _, _ = await rule.limiter.is_allowed()
        return allowed
    
    async def _create_rate_limit_error(self, limit_type: str, key: str = None) -> RateLimitError:
        """创建速率限制错误"""
        self.stats['blocked_requests'] += 1
        
        if limit_type == 'global':
            rule = self.default_rules.get('global')
            reset_time = datetime.utcnow() + timedelta(seconds=60)
            retry_after = 60
        elif limit_type in ['per_ip', 'per_user', 'per_api_key', 'auth']:
            rule = self.default_rules.get(limit_type)
            reset_time = datetime.utcnow() + timedelta(seconds=rule.window if rule else 60)
            retry_after = rule.window if rule else 60
        elif limit_type == 'custom' and key:
            rule = self.path_limiters.get(key)
            reset_time = datetime.utcnow() + timedelta(seconds=rule.window if rule else 60)
            retry_after = rule.window if rule else 60
        else:
            reset_time = datetime.utcnow() + timedelta(seconds=60)
            retry_after = 60
            rule = None
        
        limit = rule.limit if rule else 0
        
        # 尝试获取剩余次数
        remaining = 0
        if key and limit_type in ['per_ip', 'per_user', 'per_api_key']:
            limiters = getattr(self, f"{limit_type.split('_')[1]}_limiters", {})
            if key in limiters:
                limiter = limiters[key].limiter
                if hasattr(limiter, 'get_current_count'):
                    current = await limiter.get_current_count()
                    remaining = max(0, limit - current)
        
        return RateLimitError(
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )


# FastAPI中间件函数
async def rate_limit_middleware(
    request: Request,
    call_next: Callable,
    rate_limiter: RateLimiter,
    session: Optional[SessionInfo] = None
) -> Response:
    """FastAPI速率限制中间件"""
    try:
        # 检查速率限制
        rate_limit_error = await rate_limiter.check_rate_limit(request, session)
        
        if rate_limit_error:
            # 返回速率限制错误
            return JSONResponse(
                status_code=429,
                content={
                    "error": rate_limit_error.to_dict(),
                    "status": "error",
                    "message": "Rate limit exceeded"
                },
                headers={
                    "Retry-After": str(rate_limit_error.retry_after),
                    "X-RateLimit-Limit": str(rate_limit_error.limit),
                    "X-RateLimit-Remaining": str(rate_limit_error.remaining),
                    "X-RateLimit-Reset": str(int(rate_limit_error.reset_time.timestamp()))
                }
            )
        
        # 处理请求
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # 记录请求结果
        success = response.status_code < 400
        await rate_limiter.record_request_result(request, success)
        
        # 添加速率限制头
        response.headers["X-RateLimit-Limit"] = "1000"  # 示例值
        response.headers["X-RateLimit-Remaining"] = "999"  # 示例值
        response.headers["X-Request-ID"] = str(uuid.uuid4())
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}s"
        
        return response
        
    except Exception as e:
        logger.error("Rate limit middleware error", error=str(e))
        # 在中间件错误时，允许请求继续
        return await call_next(request)