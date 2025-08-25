"""
容错和异常处理管理器

此模块实现了高频交易系统的容错机制，包括：
1. 异常检测和分类
2. 自动恢复策略
3. 熔断器模式
4. 降级服务
5. 监控和告警
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from collections import deque, defaultdict
import traceback
import threading
from contextlib import asynccontextmanager

from src.utils.logger import LoggerMixin


class ErrorCategory(Enum):
    """错误分类"""
    NETWORK_ERROR = "network_error"
    DATA_ERROR = "data_error"
    PROCESSING_ERROR = "processing_error"
    ORDER_ERROR = "order_error"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"


class ComponentStatus(Enum):
    """组件状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"
    RECOVERING = "recovering"


@dataclass
class ErrorEvent:
    """错误事件"""
    component: str
    error_type: type
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_str: str = ""
    
    def __post_init__(self):
        if not self.traceback_str:
            self.traceback_str = traceback.format_exc()


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: ComponentStatus = ComponentStatus.HEALTHY
    error_count: int = 0
    last_error: Optional[ErrorEvent] = None
    last_success: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    circuit_breaker_open_time: Optional[float] = None
    
    # 健康指标
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    recovery_timeout: float = 30.0  # 恢复超时（秒）
    success_threshold: int = 3  # 半开状态下的成功阈值
    monitoring_window: float = 60.0  # 监控窗口（秒）


@dataclass
class RetryPolicy:
    """重试策略"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


class FaultToleranceManager(LoggerMixin):
    """
    容错和异常处理管理器
    
    核心功能：
    1. 异常监控和分类
    2. 自动恢复机制
    3. 熔断器保护
    4. 服务降级
    5. 健康状态管理
    """
    
    def __init__(self,
                 error_window_size: int = 1000,
                 health_check_interval: float = 10.0):
        """
        初始化容错管理器
        
        Args:
            error_window_size: 错误历史窗口大小
            health_check_interval: 健康检查间隔（秒）
        """
        super().__init__()
        
        self.error_window_size = error_window_size
        self.health_check_interval = health_check_interval
        
        # 错误历史和统计
        self.error_history: deque = deque(maxlen=error_window_size)
        self.error_counts: Dict[ErrorCategory, int] = defaultdict(int)
        self.component_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 组件健康状态
        self.component_health: Dict[str, ComponentHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerConfig] = {}
        
        # 恢复策略映射
        self.recovery_strategies: Dict[str, Dict[ErrorCategory, RecoveryStrategy]] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # 降级服务
        self.fallback_handlers: Dict[str, Callable] = {}
        self.degradation_modes: Dict[str, bool] = {}  # component -> is_degraded
        
        # 运行状态
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # 事件回调
        self.error_callbacks: List[Callable[[ErrorEvent], None]] = []
        self.recovery_callbacks: List[Callable[[str, ComponentStatus], None]] = []
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 默认配置
        self._setup_default_configurations()
    
    def _setup_default_configurations(self):
        """设置默认配置"""
        # 默认熔断器配置
        default_circuit_breaker = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=3,
            monitoring_window=60.0
        )
        
        # 默认重试策略
        default_retry_policy = RetryPolicy(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # 为主要组件设置默认配置
        components = [
            "multidimensional_engine",
            "latency_monitor", 
            "signal_processor",
            "order_router",
            "market_data_feed"
        ]
        
        for component in components:
            self.circuit_breakers[component] = default_circuit_breaker
            self.retry_policies[component] = default_retry_policy
            self.component_health[component] = ComponentHealth(name=component)
            
            # 默认恢复策略
            self.recovery_strategies[component] = {
                ErrorCategory.NETWORK_ERROR: RecoveryStrategy.RETRY,
                ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
                ErrorCategory.RATE_LIMIT_ERROR: RecoveryStrategy.CIRCUIT_BREAK,
                ErrorCategory.DATA_ERROR: RecoveryStrategy.FALLBACK,
                ErrorCategory.PROCESSING_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
                ErrorCategory.SYSTEM_ERROR: RecoveryStrategy.CIRCUIT_BREAK,
                ErrorCategory.ORDER_ERROR: RecoveryStrategy.FAIL_FAST
            }
    
    async def start(self):
        """启动容错管理器"""
        if self._running:
            return
            
        self._running = True
        
        # 启动健康检查任务
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.log_info("容错管理器已启动")
    
    async def stop(self):
        """停止容错管理器"""
        self._running = False
        
        # 停止健康检查
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 停止恢复任务
        for task in self._recovery_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self._recovery_tasks.values(), return_exceptions=True)
        self._recovery_tasks.clear()
        
        self.log_info("容错管理器已停止")
    
    @asynccontextmanager
    async def protected_operation(self, component: str, operation_name: str = ""):
        """
        保护操作上下文管理器
        
        用法:
        async with fault_manager.protected_operation("component_name"):
            # 执行可能失败的操作
            result = await some_operation()
        """
        start_time = time.time()
        operation_id = f"{component}_{operation_name}_{int(time.time()*1000)}"
        
        try:
            # 检查熔断器状态
            if not await self._check_circuit_breaker(component):
                raise RuntimeError(f"Circuit breaker open for {component}")
            
            yield
            
            # 操作成功
            await self._record_success(component, start_time)
            
        except Exception as e:
            # 操作失败
            error_event = self._create_error_event(component, e, operation_name)
            await self._handle_error(error_event)
            raise
    
    async def handle_error(self, component: str, error: Exception, context: Dict[str, Any] = None) -> bool:
        """
        处理错误
        
        Returns:
            bool: 是否应该重试
        """
        error_event = self._create_error_event(component, error, context=context or {})
        return await self._handle_error(error_event)
    
    def _create_error_event(self, component: str, error: Exception, operation: str = "", context: Dict[str, Any] = None) -> ErrorEvent:
        """创建错误事件"""
        # 错误分类
        category = self._categorize_error(error)
        severity = self._assess_severity(error, category)
        
        return ErrorEvent(
            component=component,
            error_type=type(error),
            error_message=str(error),
            category=category,
            severity=severity,
            context={
                "operation": operation,
                **(context or {})
            }
        )
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """错误分类"""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        if "network" in error_msg or "connection" in error_msg:
            return ErrorCategory.NETWORK_ERROR
        elif "timeout" in error_msg or "timed out" in error_msg:
            return ErrorCategory.TIMEOUT_ERROR
        elif "rate limit" in error_msg or "429" in error_msg:
            return ErrorCategory.RATE_LIMIT_ERROR
        elif "data" in error_msg or "parse" in error_msg:
            return ErrorCategory.DATA_ERROR
        elif "order" in error_msg or "trade" in error_msg:
            return ErrorCategory.ORDER_ERROR
        elif "config" in error_msg or "setting" in error_msg:
            return ErrorCategory.CONFIGURATION_ERROR
        elif error_type in ["systemexit", "keyboardinterrupt", "memoryerror"]:
            return ErrorCategory.SYSTEM_ERROR
        else:
            return ErrorCategory.PROCESSING_ERROR
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """评估错误严重程度"""
        error_type = type(error).__name__
        
        # 系统级错误最严重
        if category == ErrorCategory.SYSTEM_ERROR:
            return ErrorSeverity.CRITICAL
        
        # 订单错误在交易系统中是高优先级
        if category == ErrorCategory.ORDER_ERROR:
            return ErrorSeverity.HIGH
        
        # 配置错误可能影响系统稳定性
        if category == ErrorCategory.CONFIGURATION_ERROR:
            return ErrorSeverity.HIGH
        
        # 网络和超时错误通常是中等严重
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # 其他错误默认为低严重程度
        return ErrorSeverity.LOW
    
    async def _handle_error(self, error_event: ErrorEvent) -> bool:
        """处理错误事件"""
        with self._lock:
            # 记录错误
            self.error_history.append(error_event)
            self.error_counts[error_event.category] += 1
            self.component_errors[error_event.component].append(error_event)
            
            # 更新组件健康状态
            health = self.component_health[error_event.component]
            health.error_count += 1
            health.failed_requests += 1
            health.consecutive_failures += 1
            health.last_error = error_event
            
            # 计算成功率
            if health.total_requests > 0:
                health.success_rate = (health.total_requests - health.failed_requests) / health.total_requests
        
        # 触发错误回调
        for callback in self.error_callbacks:
            try:
                callback(error_event)
            except Exception as cb_error:
                self.log_error(f"错误回调失败: {cb_error}")
        
        # 记录日志
        self._log_error_event(error_event)
        
        # 确定恢复策略
        strategy = self._get_recovery_strategy(error_event.component, error_event.category)
        
        # 执行恢复策略
        return await self._execute_recovery_strategy(error_event, strategy)
    
    def _get_recovery_strategy(self, component: str, category: ErrorCategory) -> RecoveryStrategy:
        """获取恢复策略"""
        component_strategies = self.recovery_strategies.get(component, {})
        return component_strategies.get(category, RecoveryStrategy.FAIL_FAST)
    
    async def _execute_recovery_strategy(self, error_event: ErrorEvent, strategy: RecoveryStrategy) -> bool:
        """执行恢复策略"""
        component = error_event.component
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._should_retry(component, error_event)
            
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            await self._activate_circuit_breaker(component)
            return False
            
        elif strategy == RecoveryStrategy.FALLBACK:
            await self._activate_fallback(component)
            return False
            
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            await self._activate_degradation_mode(component)
            return False
            
        elif strategy == RecoveryStrategy.FAIL_FAST:
            return False
            
        elif strategy == RecoveryStrategy.IGNORE:
            return True
            
        return False
    
    async def _should_retry(self, component: str, error_event: ErrorEvent) -> bool:
        """判断是否应该重试"""
        health = self.component_health[component]
        retry_policy = self.retry_policies.get(component, RetryPolicy())
        
        # 检查连续失败次数
        if health.consecutive_failures >= retry_policy.max_attempts:
            self.log_warning(f"组件 {component} 连续失败次数达到上限，停止重试")
            return False
        
        # 检查错误类型是否适合重试
        non_retryable_categories = {
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.SYSTEM_ERROR
        }
        
        if error_event.category in non_retryable_categories:
            return False
            
        return True
    
    async def _activate_circuit_breaker(self, component: str):
        """激活熔断器"""
        health = self.component_health[component]
        health.status = ComponentStatus.CIRCUIT_OPEN
        health.circuit_breaker_open_time = time.time()
        
        self.log_warning(f"组件 {component} 熔断器已激活")
        
        # 启动自动恢复任务
        if component not in self._recovery_tasks:
            self._recovery_tasks[component] = asyncio.create_task(
                self._circuit_breaker_recovery(component)
            )
    
    async def _activate_fallback(self, component: str):
        """激活后备方案"""
        if component in self.fallback_handlers:
            try:
                await self.fallback_handlers[component]()
                self.log_info(f"组件 {component} 后备方案已激活")
            except Exception as e:
                self.log_error(f"后备方案执行失败: {component}, {e}")
        else:
            self.log_warning(f"组件 {component} 没有配置后备方案")
    
    async def _activate_degradation_mode(self, component: str):
        """激活降级模式"""
        self.degradation_modes[component] = True
        health = self.component_health[component]
        health.status = ComponentStatus.DEGRADED
        
        self.log_info(f"组件 {component} 已进入降级模式")
        
        # 触发恢复回调
        for callback in self.recovery_callbacks:
            try:
                callback(component, ComponentStatus.DEGRADED)
            except Exception as e:
                self.log_error(f"恢复回调失败: {e}")
    
    async def _check_circuit_breaker(self, component: str) -> bool:
        """检查熔断器状态"""
        health = self.component_health.get(component)
        if not health:
            return True
            
        if health.status == ComponentStatus.CIRCUIT_OPEN:
            if health.circuit_breaker_open_time:
                circuit_config = self.circuit_breakers.get(component, CircuitBreakerConfig())
                elapsed = time.time() - health.circuit_breaker_open_time
                
                if elapsed >= circuit_config.recovery_timeout:
                    # 进入半开状态
                    health.status = ComponentStatus.RECOVERING
                    self.log_info(f"组件 {component} 熔断器进入半开状态")
                    return True
                    
            return False
            
        return True
    
    async def _record_success(self, component: str, start_time: float):
        """记录成功操作"""
        with self._lock:
            health = self.component_health[component]
            health.total_requests += 1
            health.consecutive_failures = 0  # 重置连续失败计数
            health.last_success = time.time()
            
            # 更新响应时间
            response_time = time.time() - start_time
            if health.total_requests == 1:
                health.avg_response_time = response_time
            else:
                health.avg_response_time = (
                    (health.avg_response_time * (health.total_requests - 1) + response_time)
                    / health.total_requests
                )
            
            # 更新成功率
            health.success_rate = (health.total_requests - health.failed_requests) / health.total_requests
            
            # 处理熔断器恢复
            if health.status == ComponentStatus.RECOVERING:
                circuit_config = self.circuit_breakers.get(component, CircuitBreakerConfig())
                if health.consecutive_failures == 0:  # 成功操作
                    successful_ops = getattr(health, '_recovery_successes', 0) + 1
                    setattr(health, '_recovery_successes', successful_ops)
                    
                    if successful_ops >= circuit_config.success_threshold:
                        health.status = ComponentStatus.HEALTHY
                        health.circuit_breaker_open_time = None
                        delattr(health, '_recovery_successes')
                        self.log_info(f"组件 {component} 熔断器已恢复")
            elif health.status == ComponentStatus.DEGRADED:
                # 检查是否可以退出降级模式
                if health.success_rate > 0.95 and health.consecutive_failures == 0:
                    health.status = ComponentStatus.HEALTHY
                    self.degradation_modes[component] = False
                    self.log_info(f"组件 {component} 已退出降级模式")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.log_error(f"健康检查出错: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()
        
        for component, health in self.component_health.items():
            # 检查长时间无活动的组件
            if current_time - health.last_success > 300:  # 5分钟无成功操作
                if health.status == ComponentStatus.HEALTHY:
                    health.status = ComponentStatus.UNHEALTHY
                    self.log_warning(f"组件 {component} 长时间无响应，标记为不健康")
    
    async def _circuit_breaker_recovery(self, component: str):
        """熔断器自动恢复"""
        try:
            circuit_config = self.circuit_breakers.get(component, CircuitBreakerConfig())
            await asyncio.sleep(circuit_config.recovery_timeout)
            
            health = self.component_health[component]
            if health.status == ComponentStatus.CIRCUIT_OPEN:
                health.status = ComponentStatus.RECOVERING
                self.log_info(f"组件 {component} 熔断器开始恢复尝试")
                
        except asyncio.CancelledError:
            pass
        finally:
            self._recovery_tasks.pop(component, None)
    
    def _log_error_event(self, error_event: ErrorEvent):
        """记录错误事件日志"""
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.log_error(
                f"严重错误: {error_event.component} - {error_event.error_message}",
                category=error_event.category.value,
                traceback=error_event.traceback_str
            )
        elif error_event.severity == ErrorSeverity.HIGH:
            self.log_error(
                f"高优先级错误: {error_event.component} - {error_event.error_message}",
                category=error_event.category.value
            )
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.log_warning(
                f"中等错误: {error_event.component} - {error_event.error_message}",
                category=error_event.category.value
            )
        else:
            self.log_debug(
                f"低优先级错误: {error_event.component} - {error_event.error_message}",
                category=error_event.category.value
            )
    
    # 公共接口
    def register_component(self, component: str, 
                         circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                         retry_policy: Optional[RetryPolicy] = None,
                         recovery_strategies: Optional[Dict[ErrorCategory, RecoveryStrategy]] = None):
        """注册组件"""
        self.component_health[component] = ComponentHealth(name=component)
        
        if circuit_breaker_config:
            self.circuit_breakers[component] = circuit_breaker_config
            
        if retry_policy:
            self.retry_policies[component] = retry_policy
            
        if recovery_strategies:
            self.recovery_strategies[component] = recovery_strategies
    
    def register_fallback_handler(self, component: str, handler: Callable):
        """注册后备处理器"""
        self.fallback_handlers[component] = handler
    
    def add_error_callback(self, callback: Callable[[ErrorEvent], None]):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str, ComponentStatus], None]):
        """添加恢复回调"""
        self.recovery_callbacks.append(callback)
    
    def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """获取组件健康状态"""
        return self.component_health.get(component)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """获取系统健康摘要"""
        healthy = sum(1 for h in self.component_health.values() if h.status == ComponentStatus.HEALTHY)
        degraded = sum(1 for h in self.component_health.values() if h.status == ComponentStatus.DEGRADED)
        unhealthy = sum(1 for h in self.component_health.values() if h.status in [ComponentStatus.UNHEALTHY, ComponentStatus.CIRCUIT_OPEN])
        
        return {
            "total_components": len(self.component_health),
            "healthy_count": healthy,
            "degraded_count": degraded,
            "unhealthy_count": unhealthy,
            "overall_status": self._calculate_overall_status(),
            "total_errors": len(self.error_history),
            "error_breakdown": dict(self.error_counts),
            "component_status": {
                name: health.status.value 
                for name, health in self.component_health.items()
            }
        }
    
    def _calculate_overall_status(self) -> str:
        """计算整体状态"""
        if not self.component_health:
            return "unknown"
            
        statuses = [health.status for health in self.component_health.values()]
        
        if ComponentStatus.CIRCUIT_OPEN in statuses or ComponentStatus.UNHEALTHY in statuses:
            return "unhealthy"
        elif ComponentStatus.DEGRADED in statuses:
            return "degraded"  
        elif all(status == ComponentStatus.HEALTHY for status in statuses):
            return "healthy"
        else:
            return "mixed"
    
    def is_component_degraded(self, component: str) -> bool:
        """检查组件是否处于降级模式"""
        return self.degradation_modes.get(component, False)
    
    def force_component_recovery(self, component: str):
        """强制组件恢复"""
        if component in self.component_health:
            health = self.component_health[component]
            health.status = ComponentStatus.HEALTHY
            health.consecutive_failures = 0
            health.circuit_breaker_open_time = None
            self.degradation_modes[component] = False
            
            self.log_info(f"组件 {component} 已强制恢复")
        else:
            self.log_warning(f"未找到组件: {component}")