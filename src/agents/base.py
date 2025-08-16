import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict, deque

from pydantic import BaseModel, Field

from src.core.models import MarketData, Order, Position, Signal, TradingState
from src.core.message_bus import MessageBus, Message, MessagePriority
from src.utils.logger import LoggerMixin


class AgentState(Enum):
    """Agent状态枚举"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class AgentPerformanceMetrics:
    """Agent性能指标"""
    signals_generated: int = 0
    signals_per_second: float = 0.0
    avg_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    error_count: int = 0
    last_active_time: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000), init=False)
    start_time: float = field(default_factory=time.time, init=False)
    
    def update_processing_time(self, processing_time: float):
        """更新处理时间统计"""
        self.processing_times.append(processing_time)
        self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.last_active_time = time.time()
        
    def update_signals_per_second(self):
        """更新每秒信号数统计"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        if elapsed > 0:
            self.signals_per_second = self.signals_generated / elapsed
        self.uptime_seconds = elapsed


class AgentConfig(BaseModel):
    """Base configuration for all agents."""
    name: str
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # 新增配置
    max_processing_time: float = Field(default=5.0, description="最大处理时间(秒)")
    heartbeat_interval: float = Field(default=30.0, description="心跳间隔(秒)")
    config_update_enabled: bool = Field(default=True, description="是否启用配置热更新")
    communication_enabled: bool = Field(default=True, description="是否启用Agent间通信")
    performance_monitoring: bool = Field(default=True, description="是否启用性能监控")


class BaseAgent(ABC, LoggerMixin):
    """Abstract base class for all trading agents."""
    
    def __init__(
        self, 
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None
    ):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority
        self._initialized = False
        
        # 状态管理
        self._state = AgentState.CREATED
        self._state_history: List[tuple] = []  # (timestamp, state, reason)
        self._state_lock = asyncio.Lock()
        
        # 消息通信
        self.message_bus = message_bus
        self.publisher = None
        self.subscriber = None
        self._message_handlers: Dict[str, Callable] = {}
        
        # 性能监控
        self.metrics = AgentPerformanceMetrics()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 配置热更新
        self._config_watchers: List[Callable] = []
        self._last_config_update = time.time()
        
        # 任务管理
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return
        
        await self._set_state(AgentState.INITIALIZING, "开始初始化")
        
        try:
            self.log_info(f"Initializing agent: {self.name}")
            
            # 初始化消息通信
            if self.message_bus and self.config.communication_enabled:
                await self._setup_communication()
            
            # 启动性能监控
            if self.config.performance_monitoring:
                await self._start_monitoring()
            
            # 启动心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Agent特定初始化
            await self._initialize()
            
            self._initialized = True
            await self._set_state(AgentState.RUNNING, "初始化完成")
            self.log_info(f"Agent initialized: {self.name}")
            
        except Exception as e:
            await self._set_state(AgentState.ERROR, f"初始化失败: {e}")
            self.log_error(f"Failed to initialize agent {self.name}: {e}")
            raise
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        if not self._initialized:
            return
        
        await self._set_state(AgentState.SHUTTING_DOWN, "开始关闭")
        
        try:
            self.log_info(f"Shutting down agent: {self.name}")
            
            # 设置关闭事件
            self._shutdown_event.set()
            
            # 停止监控任务
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # 停止心跳任务
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # 停止所有运行中的任务
            for task_name, task in self._running_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    self.log_debug(f"Cancelled task: {task_name}")
            
            # Agent特定关闭
            await self._shutdown()
            
            self._initialized = False
            await self._set_state(AgentState.SHUTDOWN, "关闭完成")
            self.log_info(f"Agent shutdown: {self.name}")
            
        except Exception as e:
            await self._set_state(AgentState.ERROR, f"关闭失败: {e}")
            self.log_error(f"Failed to shutdown agent {self.name}: {e}")
            raise
    
    async def _shutdown(self) -> None:
        """Agent-specific shutdown logic."""
        pass
    
    # 状态管理方法
    async def _set_state(self, new_state: AgentState, reason: str = ""):
        """设置Agent状态"""
        async with self._state_lock:
            old_state = self._state
            self._state = new_state
            timestamp = time.time()
            self._state_history.append((timestamp, new_state.value, reason))
            
            # 保留最近100条历史记录
            if len(self._state_history) > 100:
                self._state_history = self._state_history[-100:]
            
            self.log_debug(f"State changed: {old_state.value} -> {new_state.value}, reason: {reason}")
            
            # 发布状态变化事件
            if self.publisher:
                await self._publish_message(
                    f"agent.{self.name}.state",
                    {
                        "agent_name": self.name,
                        "old_state": old_state.value,
                        "new_state": new_state.value,
                        "reason": reason,
                        "timestamp": timestamp
                    }
                )
    
    def get_state(self) -> AgentState:
        """获取当前状态"""
        return self._state
    
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取状态历史"""
        recent_history = self._state_history[-limit:] if limit > 0 else self._state_history
        return [
            {
                "timestamp": timestamp,
                "state": state,
                "reason": reason
            }
            for timestamp, state, reason in recent_history
        ]
    
    # Agent间通信方法
    async def _setup_communication(self):
        """设置消息通信"""
        if not self.message_bus:
            return
        
        try:
            # 创建发布者
            self.publisher = self.message_bus.create_publisher(f"agent_{self.name}")
            
            # 创建订阅者
            self.subscriber = self.message_bus.create_subscriber(f"agent_{self.name}_sub")
            
            # 订阅Agent间通信消息
            await self.subscriber.subscribe(f"agent.{self.name}.message", self._handle_agent_message)
            await self.subscriber.subscribe(f"agent.{self.name}.config", self._handle_config_update)
            await self.subscriber.subscribe("agent.broadcast.*", self._handle_broadcast_message)
            
            self.log_debug("Communication setup complete")
            
        except Exception as e:
            self.log_error(f"Failed to setup communication: {e}")
            raise
    
    async def _publish_message(self, topic: str, data: Any, priority: MessagePriority = MessagePriority.NORMAL):
        """发布消息"""
        if not self.publisher:
            return
        
        try:
            self.message_bus.publish(
                f"agent_{self.name}",
                topic,
                data,
                priority=priority
            )
        except Exception as e:
            self.log_warning(f"Failed to publish message to {topic}: {e}")
    
    async def send_message_to_agent(self, target_agent: str, message_type: str, data: Any):
        """向指定Agent发送消息"""
        topic = f"agent.{target_agent}.message"
        message_data = {
            "from_agent": self.name,
            "message_type": message_type,
            "data": data,
            "timestamp": time.time()
        }
        await self._publish_message(topic, message_data)
        self.log_debug(f"Sent message to {target_agent}: {message_type}")
    
    async def broadcast_message(self, message_type: str, data: Any):
        """广播消息给所有Agent"""
        topic = f"agent.broadcast.{message_type}"
        message_data = {
            "from_agent": self.name,
            "message_type": message_type,
            "data": data,
            "timestamp": time.time()
        }
        await self._publish_message(topic, message_data)
        self.log_debug(f"Broadcast message: {message_type}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self._message_handlers[message_type] = handler
        self.log_debug(f"Registered message handler: {message_type}")
    
    async def _handle_agent_message(self, message: Message):
        """处理Agent消息"""
        try:
            data = message.data
            message_type = data.get("message_type")
            from_agent = data.get("from_agent")
            
            if message_type in self._message_handlers:
                await self._message_handlers[message_type](from_agent, data.get("data"))
            else:
                self.log_debug(f"No handler for message type: {message_type}")
        except Exception as e:
            self.log_error(f"Error handling agent message: {e}")
    
    async def _handle_broadcast_message(self, message: Message):
        """处理广播消息"""
        try:
            data = message.data
            from_agent = data.get("from_agent")
            
            # 忽略自己发送的广播消息
            if from_agent == self.name:
                return
            
            message_type = data.get("message_type")
            if message_type in self._message_handlers:
                await self._message_handlers[message_type](from_agent, data.get("data"))
        except Exception as e:
            self.log_error(f"Error handling broadcast message: {e}")
    
    # 配置热更新方法
    async def _handle_config_update(self, message: Message):
        """处理配置更新"""
        if not self.config.config_update_enabled:
            return
        
        try:
            config_data = message.data
            await self._update_config(config_data)
            self._last_config_update = time.time()
            self.log_info("Configuration updated successfully")
            
            # 通知配置观察者
            for watcher in self._config_watchers:
                try:
                    await watcher(self.config)
                except Exception as e:
                    self.log_warning(f"Config watcher failed: {e}")
                    
        except Exception as e:
            self.log_error(f"Failed to update configuration: {e}")
    
    async def _update_config(self, config_data: Dict[str, Any]):
        """更新配置"""
        for key, value in config_data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.log_debug(f"Updated config: {key} = {value}")
        
        # 更新实例属性
        self.enabled = self.config.enabled
        self.priority = self.config.priority
    
    def add_config_watcher(self, watcher: Callable):
        """添加配置观察者"""
        self._config_watchers.append(watcher)
    
    async def update_config(self, **kwargs):
        """更新配置（程序化方式）"""
        config_data = kwargs
        await self._update_config(config_data)
        self.log_info(f"Config updated programmatically: {list(kwargs.keys())}")
    
    # 性能监控方法
    async def _start_monitoring(self):
        """启动性能监控"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.log_debug("Performance monitoring started")
    
    async def _monitoring_loop(self):
        """性能监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10.0)  # 每10秒更新一次
                
                # 更新性能指标
                self.metrics.update_signals_per_second()
                
                # 记录性能指标
                if self.metrics.signals_generated > 0:
                    self.log_debug(
                        "Performance metrics",
                        signals_generated=self.metrics.signals_generated,
                        signals_per_second=round(self.metrics.signals_per_second, 2),
                        avg_processing_time=round(self.metrics.avg_processing_time * 1000, 3),
                        error_count=self.metrics.error_count,
                        uptime_hours=round(self.metrics.uptime_seconds / 3600, 2)
                    )
                
                # 发布性能指标
                if self.publisher:
                    await self._publish_message(
                        f"agent.{self.name}.metrics",
                        {
                            "agent_name": self.name,
                            "metrics": {
                                "signals_generated": self.metrics.signals_generated,
                                "signals_per_second": self.metrics.signals_per_second,
                                "avg_processing_time": self.metrics.avg_processing_time,
                                "error_count": self.metrics.error_count,
                                "uptime_seconds": self.metrics.uptime_seconds,
                                "last_active_time": self.metrics.last_active_time
                            },
                            "timestamp": time.time()
                        }
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {e}")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # 发送心跳
                if self.publisher:
                    await self._publish_message(
                        f"agent.{self.name}.heartbeat",
                        {
                            "agent_name": self.name,
                            "state": self._state.value,
                            "timestamp": time.time(),
                            "uptime": self.metrics.uptime_seconds
                        }
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Error in heartbeat loop: {e}")
    
    def get_performance_metrics(self) -> AgentPerformanceMetrics:
        """获取性能指标"""
        return self.metrics
    
    # 增强的analyze方法
    async def analyze_with_monitoring(self, state: TradingState) -> List[Signal]:
        """带性能监控的analyze方法"""
        if not self.enabled or self._state != AgentState.RUNNING:
            return []
        
        start_time = time.time()
        
        try:
            # 执行分析
            signals = await self.analyze(state)
            
            # 更新性能指标
            processing_time = time.time() - start_time
            self.metrics.update_processing_time(processing_time)
            self.metrics.signals_generated += len(signals)
            
            # 检查处理时间是否超过阈值
            if processing_time > self.config.max_processing_time:
                self.log_warning(
                    f"Processing time exceeded threshold: {processing_time:.3f}s > {self.config.max_processing_time}s"
                )
            
            return signals
            
        except Exception as e:
            self.metrics.error_count += 1
            self.log_error(f"Error in analyze: {e}")
            await self._set_state(AgentState.ERROR, f"分析错误: {e}")
            raise
    
    @abstractmethod
    async def analyze(self, state: TradingState) -> List[Signal]:
        """
        Analyze the current trading state and generate signals.
        
        Args:
            state: Current trading state
            
        Returns:
            List of trading signals
        """
        pass
    
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal before execution.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        return (
            signal.strength != 0 and
            abs(signal.strength) >= self.config.parameters.get("min_signal_strength", 0.3) and
            signal.confidence >= self.config.parameters.get("min_confidence", 0.5)
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class AnalysisAgent(BaseAgent):
    """Base class for analysis agents that generate signals."""
    
    async def _initialize(self) -> None:
        """Initialize analysis agent."""
        pass
    
    @abstractmethod
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Analyze market and generate signals."""
        pass


class ExecutionAgent(BaseAgent):
    """Base class for execution agents that manage orders."""
    
    async def _initialize(self) -> None:
        """Initialize execution agent."""
        pass
    
    @abstractmethod
    async def execute(self, signal: Signal, state: TradingState) -> Optional[Order]:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal to execute
            state: Current trading state
            
        Returns:
            Order if execution successful, None otherwise
        """
        pass
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Execution agents don't generate signals."""
        return []


class RiskAgent(BaseAgent):
    """Base class for risk management agents."""
    
    async def _initialize(self) -> None:
        """Initialize risk agent."""
        pass
    
    @abstractmethod
    async def check_risk(self, order: Order, state: TradingState) -> bool:
        """
        Check if an order passes risk management rules.
        
        Args:
            order: Order to check
            state: Current trading state
            
        Returns:
            True if order is allowed, False otherwise
        """
        pass
    
    @abstractmethod
    async def adjust_position_size(self, order: Order, state: TradingState) -> Order:
        """
        Adjust order size based on risk management rules.
        
        Args:
            order: Order to adjust
            state: Current trading state
            
        Returns:
            Adjusted order
        """
        pass
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Risk agents may generate risk-related signals."""
        signals = []
        
        # Check for high-risk conditions
        if state.risk_metrics:
            if state.risk_metrics.margin_usage > 0.8:
                signals.append(Signal(
                    source=self.name,
                    symbol="ALL",
                    action="REDUCE_EXPOSURE",
                    strength=-0.8,
                    confidence=0.9,
                    reason="High margin usage detected",
                    metadata={"margin_usage": state.risk_metrics.margin_usage}
                ))
            
            if state.risk_metrics.current_drawdown > 0.1:
                signals.append(Signal(
                    source=self.name,
                    symbol="ALL",
                    action="STOP_TRADING",
                    strength=-1.0,
                    confidence=0.95,
                    reason="Maximum drawdown exceeded",
                    metadata={"drawdown": state.risk_metrics.current_drawdown}
                ))
        
        return signals