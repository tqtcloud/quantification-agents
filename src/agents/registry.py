"""
Agent注册和发现系统
管理Agent生命周期、健康检查和故障恢复
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict

from src.agents.base import BaseAgent, AgentState, AgentConfig
from src.core.message_bus import MessageBus, Message, MessagePriority
from src.utils.logger import LoggerMixin


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class AgentHealthInfo:
    """Agent健康信息"""
    agent_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: float = 0.0
    last_activity: float = 0.0
    error_count: int = 0
    consecutive_failures: int = 0
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_alive(self) -> bool:
        """检查Agent是否存活"""
        current_time = time.time()
        # 如果超过60秒没有心跳，认为Agent已死
        return (current_time - self.last_heartbeat) < 60.0
    
    @property
    def heartbeat_age_seconds(self) -> float:
        """心跳年龄（秒）"""
        return time.time() - self.last_heartbeat


@dataclass
class AgentRegistration:
    """Agent注册信息"""
    agent: BaseAgent
    config: AgentConfig
    registration_time: float = field(default_factory=time.time)
    health_info: AgentHealthInfo = field(init=False)
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.health_info = AgentHealthInfo(agent_name=self.agent.name)


class AgentRegistry(LoggerMixin):
    """Agent注册表"""
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        health_check_interval: float = 30.0,
        max_consecutive_failures: int = 3,
        failure_recovery_enabled: bool = True
    ):
        self.message_bus = message_bus
        self.health_check_interval = health_check_interval
        self.max_consecutive_failures = max_consecutive_failures
        self.failure_recovery_enabled = failure_recovery_enabled
        
        # Agent注册表
        self._agents: Dict[str, AgentRegistration] = {}
        self._agent_groups: Dict[str, Set[str]] = defaultdict(set)
        self._agent_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # 健康检查
        self._health_check_task: Optional[asyncio.Task] = None
        self._recovery_handlers: Dict[str, Callable] = {}
        
        # 消息通信
        self.publisher = None
        self.subscriber = None
        
        # 事件处理器
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 运行状态
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """初始化注册表"""
        if self.message_bus:
            await self._setup_communication()
        
        # 启动健康检查
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._running = True
        
        self.log_info("AgentRegistry initialized")
    
    async def shutdown(self):
        """关闭注册表"""
        self._running = False
        self._shutdown_event.set()
        
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有注册的Agent
        for agent_name in list(self._agents.keys()):
            await self.unregister_agent(agent_name, shutdown_agent=True)
        
        self.log_info("AgentRegistry shutdown complete")
    
    async def _setup_communication(self):
        """设置消息通信"""
        try:
            # 创建发布者和订阅者
            self.publisher = self.message_bus.create_publisher("agent_registry")
            self.subscriber = self.message_bus.create_subscriber("agent_registry_sub")
            
            # 订阅Agent事件
            await self.subscriber.subscribe("agent.*.heartbeat", self._handle_heartbeat)
            await self.subscriber.subscribe("agent.*.state", self._handle_state_change)
            await self.subscriber.subscribe("agent.*.metrics", self._handle_metrics_update)
            
            self.log_debug("Registry communication setup complete")
            
        except Exception as e:
            self.log_error(f"Failed to setup registry communication: {e}")
            raise
    
    async def register_agent(
        self,
        agent: BaseAgent,
        group: Optional[str] = None,
        dependencies: Optional[Set[str]] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """注册Agent"""
        try:
            agent_name = agent.name
            
            if agent_name in self._agents:
                self.log_warning(f"Agent {agent_name} already registered")
                return False
            
            # 创建注册信息
            registration = AgentRegistration(
                agent=agent,
                config=agent.config,
                dependencies=dependencies or set(),
                tags=tags or set()
            )
            
            # 检查依赖
            if dependencies:
                missing_deps = dependencies - set(self._agents.keys())
                if missing_deps:
                    self.log_error(f"Missing dependencies for {agent_name}: {missing_deps}")
                    return False
                
                self._agent_dependencies[agent_name] = dependencies
            
            # 注册Agent
            self._agents[agent_name] = registration
            
            # 添加到组
            if group:
                self._agent_groups[group].add(agent_name)
            
            # 初始化Agent
            if not agent._initialized:
                await agent.initialize()
            
            # 发布注册事件
            await self._publish_event("agent_registered", {
                "agent_name": agent_name,
                "group": group,
                "dependencies": list(dependencies or []),
                "tags": list(tags or [])
            })
            
            self.log_info(f"Agent {agent_name} registered successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to register agent {agent.name}: {e}")
            return False
    
    async def unregister_agent(self, agent_name: str, shutdown_agent: bool = True) -> bool:
        """注销Agent"""
        try:
            if agent_name not in self._agents:
                self.log_warning(f"Agent {agent_name} not found")
                return False
            
            registration = self._agents[agent_name]
            
            # 检查依赖关系
            dependents = self._get_dependents(agent_name)
            if dependents:
                self.log_warning(f"Agent {agent_name} has dependents: {dependents}")
                # 可以选择强制注销或拒绝注销
            
            # 关闭Agent
            if shutdown_agent and registration.agent._initialized:
                await registration.agent.shutdown()
            
            # 从注册表移除
            del self._agents[agent_name]
            
            # 从组中移除
            for group_agents in self._agent_groups.values():
                group_agents.discard(agent_name)
            
            # 清理依赖关系
            if agent_name in self._agent_dependencies:
                del self._agent_dependencies[agent_name]
            
            # 发布注销事件
            await self._publish_event("agent_unregistered", {
                "agent_name": agent_name
            })
            
            self.log_info(f"Agent {agent_name} unregistered successfully")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to unregister agent {agent_name}: {e}")
            return False
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """获取Agent实例"""
        registration = self._agents.get(agent_name)
        return registration.agent if registration else None
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """获取Agent配置"""
        registration = self._agents.get(agent_name)
        return registration.config if registration else None
    
    def get_agents_by_group(self, group: str) -> List[BaseAgent]:
        """按组获取Agent"""
        agent_names = self._agent_groups.get(group, set())
        return [self._agents[name].agent for name in agent_names if name in self._agents]
    
    def get_agents_by_tag(self, tag: str) -> List[BaseAgent]:
        """按标签获取Agent"""
        agents = []
        for registration in self._agents.values():
            if tag in registration.tags:
                agents.append(registration.agent)
        return agents
    
    def get_healthy_agents(self) -> List[BaseAgent]:
        """获取健康的Agent"""
        healthy_agents = []
        for registration in self._agents.values():
            if (registration.health_info.status == HealthStatus.HEALTHY and 
                registration.health_info.is_alive):
                healthy_agents.append(registration.agent)
        return healthy_agents
    
    def get_agent_health(self, agent_name: str) -> Optional[AgentHealthInfo]:
        """获取Agent健康信息"""
        registration = self._agents.get(agent_name)
        return registration.health_info if registration else None
    
    def get_all_agents_health(self) -> Dict[str, AgentHealthInfo]:
        """获取所有Agent健康信息"""
        return {
            name: registration.health_info 
            for name, registration in self._agents.items()
        }
    
    def _get_dependents(self, agent_name: str) -> Set[str]:
        """获取依赖于指定Agent的其他Agent"""
        dependents = set()
        for dependent, dependencies in self._agent_dependencies.items():
            if agent_name in dependencies:
                dependents.add(dependent)
        return dependents
    
    async def update_agent_config(self, agent_name: str, config_updates: Dict[str, Any]) -> bool:
        """更新Agent配置"""
        try:
            registration = self._agents.get(agent_name)
            if not registration:
                self.log_error(f"Agent {agent_name} not found")
                return False
            
            # 更新配置
            await registration.agent.update_config(**config_updates)
            
            # 发布配置更新事件
            await self._publish_event("agent_config_updated", {
                "agent_name": agent_name,
                "updates": config_updates
            })
            
            self.log_info(f"Config updated for agent {agent_name}: {list(config_updates.keys())}")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to update config for agent {agent_name}: {e}")
            return False
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        current_time = time.time()
        
        for agent_name, registration in self._agents.items():
            health_info = registration.health_info
            agent = registration.agent
            
            try:
                # 检查心跳
                if not health_info.is_alive:
                    health_info.status = HealthStatus.CRITICAL
                    health_info.consecutive_failures += 1
                    
                    self.log_warning(
                        f"Agent {agent_name} heartbeat timeout: "
                        f"{health_info.heartbeat_age_seconds:.1f}s ago"
                    )
                    
                    # 尝试恢复
                    if self.failure_recovery_enabled:
                        await self._attempt_recovery(agent_name, "heartbeat_timeout")
                    
                elif health_info.consecutive_failures > 0:
                    # 重置连续失败计数
                    health_info.consecutive_failures = 0
                    if health_info.status == HealthStatus.CRITICAL:
                        health_info.status = HealthStatus.HEALTHY
                        self.log_info(f"Agent {agent_name} recovered")
                
                # 检查Agent状态
                agent_state = agent.get_state()
                if agent_state == AgentState.ERROR:
                    health_info.status = HealthStatus.CRITICAL
                    health_info.error_count += 1
                    
                    if self.failure_recovery_enabled:
                        await self._attempt_recovery(agent_name, "error_state")
                
                elif agent_state == AgentState.RUNNING:
                    if health_info.status != HealthStatus.CRITICAL:
                        health_info.status = HealthStatus.HEALTHY
                
                # 检查性能指标
                metrics = agent.get_performance_metrics()
                if metrics.error_count > health_info.error_count:
                    health_info.error_count = metrics.error_count
                    if health_info.status == HealthStatus.HEALTHY:
                        health_info.status = HealthStatus.WARNING
                
                # 更新活动时间
                if metrics.last_active_time > health_info.last_activity:
                    health_info.last_activity = metrics.last_active_time
                
            except Exception as e:
                self.log_error(f"Health check failed for agent {agent_name}: {e}")
                health_info.status = HealthStatus.UNKNOWN
                health_info.consecutive_failures += 1
    
    async def _attempt_recovery(self, agent_name: str, failure_reason: str):
        """尝试恢复故障Agent"""
        try:
            registration = self._agents.get(agent_name)
            if not registration:
                return
            
            agent = registration.agent
            health_info = registration.health_info
            
            # 检查是否超过最大失败次数
            if health_info.consecutive_failures >= self.max_consecutive_failures:
                self.log_error(
                    f"Agent {agent_name} exceeded max failures ({self.max_consecutive_failures}), "
                    "removing from registry"
                )
                await self.unregister_agent(agent_name)
                return
            
            self.log_info(f"Attempting recovery for agent {agent_name}, reason: {failure_reason}")
            
            # 尝试重启Agent
            try:
                # 关闭Agent
                if agent._initialized:
                    await agent.shutdown()
                
                # 重新初始化
                await agent.initialize()
                
                # 重置健康状态
                health_info.status = HealthStatus.HEALTHY
                health_info.consecutive_failures = 0
                
                # 发布恢复事件
                await self._publish_event("agent_recovered", {
                    "agent_name": agent_name,
                    "failure_reason": failure_reason
                })
                
                self.log_info(f"Agent {agent_name} recovered successfully")
                
            except Exception as e:
                self.log_error(f"Recovery failed for agent {agent_name}: {e}")
                health_info.consecutive_failures += 1
        
        except Exception as e:
            self.log_error(f"Error in recovery attempt for {agent_name}: {e}")
    
    async def _handle_heartbeat(self, message: Message):
        """处理心跳消息"""
        try:
            data = message.data
            agent_name = data.get("agent_name")
            
            if agent_name in self._agents:
                health_info = self._agents[agent_name].health_info
                health_info.last_heartbeat = data.get("timestamp", time.time())
                
                # 更新健康状态
                if health_info.status == HealthStatus.CRITICAL:
                    health_info.status = HealthStatus.HEALTHY
                    self.log_info(f"Agent {agent_name} heartbeat resumed")
        
        except Exception as e:
            self.log_error(f"Error handling heartbeat: {e}")
    
    async def _handle_state_change(self, message: Message):
        """处理状态变化消息"""
        try:
            data = message.data
            agent_name = data.get("agent_name")
            new_state = data.get("new_state")
            
            if agent_name in self._agents:
                # 发布状态变化事件
                await self._publish_event("agent_state_changed", data)
        
        except Exception as e:
            self.log_error(f"Error handling state change: {e}")
    
    async def _handle_metrics_update(self, message: Message):
        """处理性能指标更新"""
        try:
            data = message.data
            agent_name = data.get("agent_name")
            metrics = data.get("metrics", {})
            
            if agent_name in self._agents:
                health_info = self._agents[agent_name].health_info
                
                # 更新自定义指标
                health_info.custom_metrics.update(metrics)
                health_info.last_activity = metrics.get("last_active_time", health_info.last_activity)
        
        except Exception as e:
            self.log_error(f"Error handling metrics update: {e}")
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """发布事件"""
        if not self.publisher:
            return
        
        try:
            self.message_bus.publish(
                "agent_registry",
                f"registry.{event_type}",
                {
                    "event_type": event_type,
                    "data": data,
                    "timestamp": time.time()
                },
                priority=MessagePriority.NORMAL
            )
            
            # 调用事件处理器
            handlers = self._event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    self.log_warning(f"Event handler failed for {event_type}: {e}")
        
        except Exception as e:
            self.log_warning(f"Failed to publish event {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器"""
        self._event_handlers[event_type].append(handler)
        self.log_debug(f"Added event handler for: {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """移除事件处理器"""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
                self.log_debug(f"Removed event handler for: {event_type}")
            except ValueError:
                pass
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        healthy_count = sum(
            1 for reg in self._agents.values() 
            if reg.health_info.status == HealthStatus.HEALTHY
        )
        
        return {
            "total_agents": len(self._agents),
            "healthy_agents": healthy_count,
            "unhealthy_agents": len(self._agents) - healthy_count,
            "groups": {group: len(agents) for group, agents in self._agent_groups.items()},
            "dependencies": dict(self._agent_dependencies),
            "health_check_interval": self.health_check_interval,
            "max_consecutive_failures": self.max_consecutive_failures,
            "failure_recovery_enabled": self.failure_recovery_enabled
        }


# 全局注册表实例
agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """获取全局Agent注册表"""
    global agent_registry
    if agent_registry is None:
        raise RuntimeError("Agent registry not initialized. Call create_agent_registry() first.")
    return agent_registry


def create_agent_registry(
    message_bus: Optional[MessageBus] = None,
    **kwargs
) -> AgentRegistry:
    """创建全局Agent注册表"""
    global agent_registry
    agent_registry = AgentRegistry(message_bus=message_bus, **kwargs)
    return agent_registry