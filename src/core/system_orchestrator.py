"""系统编排器 - 管理组件启动、关闭和依赖关系"""

import asyncio
import signal
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ComponentStatus(Enum):
    """组件状态枚举"""
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """组件信息"""
    name: str
    instance: Any = None
    status: ComponentStatus = ComponentStatus.UNINITIALIZED
    dependencies: List[str] = field(default_factory=list)
    startup_func: Optional[Callable] = None
    shutdown_func: Optional[Callable] = None
    health_check_func: Optional[Callable] = None
    start_time: Optional[float] = None
    error: Optional[Exception] = None


class SystemOrchestrator:
    """系统编排器 - 负责管理所有组件的生命周期"""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.shutdown_event = asyncio.Event()
        self.startup_completed = asyncio.Event()
        self.shutdown_completed = asyncio.Event()
        self._running = False
        self._startup_tasks: List[asyncio.Task] = []
        self._shutdown_tasks: List[asyncio.Task] = []
        self._health_check_task: Optional[asyncio.Task] = None
        
        # 注册信号处理器
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，开始优雅关闭")
            self.initiate_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 在Windows上添加额外的信号处理
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def register_component(
        self,
        name: str,
        instance: Any,
        dependencies: List[str] = None,
        startup_func: Optional[Callable] = None,
        shutdown_func: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None
    ):
        """注册组件"""
        if name in self.components:
            raise ValueError(f"组件 {name} 已存在")
        
        component = ComponentInfo(
            name=name,
            instance=instance,
            dependencies=dependencies or [],
            startup_func=startup_func or getattr(instance, 'start', None),
            shutdown_func=shutdown_func or getattr(instance, 'stop', None),
            health_check_func=health_check_func or getattr(instance, 'health_check', None)
        )
        
        self.components[name] = component
        logger.debug(f"注册组件: {name}")
    
    def unregister_component(self, name: str):
        """注销组件"""
        if name in self.components:
            del self.components[name]
            logger.debug(f"注销组件: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """获取组件实例"""
        component = self.components.get(name)
        return component.instance if component else None
    
    def get_component_status(self, name: str) -> Optional[ComponentStatus]:
        """获取组件状态"""
        component = self.components.get(name)
        return component.status if component else None
    
    def _resolve_startup_order(self) -> List[str]:
        """解析组件启动顺序（拓扑排序）"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node: str):
            if node in temp_visited:
                raise ValueError(f"检测到循环依赖: {node}")
            if node not in visited:
                temp_visited.add(node)
                component = self.components.get(node)
                if component:
                    for dep in component.dependencies:
                        if dep not in self.components:
                            raise ValueError(f"组件 {node} 依赖的组件 {dep} 不存在")
                        visit(dep)
                temp_visited.remove(node)
                visited.add(node)
                result.append(node)
        
        for component_name in self.components:
            visit(component_name)
        
        return result
    
    def _resolve_shutdown_order(self) -> List[str]:
        """解析组件关闭顺序（启动顺序的逆序）"""
        startup_order = self._resolve_startup_order()
        return list(reversed(startup_order))
    
    async def _start_component(self, name: str) -> bool:
        """启动单个组件"""
        component = self.components[name]
        
        try:
            logger.info(f"启动组件: {name}")
            component.status = ComponentStatus.STARTING
            component.start_time = time.time()
            
            # 检查依赖组件是否已启动
            for dep in component.dependencies:
                dep_component = self.components[dep]
                if dep_component.status != ComponentStatus.RUNNING:
                    raise RuntimeError(f"依赖组件 {dep} 未运行")
            
            # 执行启动函数
            if component.startup_func:
                if asyncio.iscoroutinefunction(component.startup_func):
                    await component.startup_func()
                else:
                    component.startup_func()
            
            component.status = ComponentStatus.RUNNING
            logger.info(f"组件 {name} 启动成功，耗时 {time.time() - component.start_time:.2f}秒")
            return True
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error = e
            logger.error(f"组件 {name} 启动失败: {e}")
            return False
    
    async def _stop_component(self, name: str) -> bool:
        """停止单个组件"""
        component = self.components[name]
        
        try:
            logger.info(f"停止组件: {name}")
            component.status = ComponentStatus.STOPPING
            
            # 执行关闭函数
            if component.shutdown_func:
                if asyncio.iscoroutinefunction(component.shutdown_func):
                    await component.shutdown_func()
                else:
                    component.shutdown_func()
            
            component.status = ComponentStatus.STOPPED
            logger.info(f"组件 {name} 已停止")
            return True
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error = e
            logger.error(f"组件 {name} 停止失败: {e}")
            return False
    
    async def start_all(self, timeout: float = 60.0):
        """启动所有组件"""
        if self._running:
            logger.warning("系统已在运行")
            return
        
        logger.info("开始启动系统所有组件")
        start_time = time.time()
        
        try:
            startup_order = self._resolve_startup_order()
            logger.info(f"组件启动顺序: {startup_order}")
            
            # 按依赖顺序启动组件
            for name in startup_order:
                success = await asyncio.wait_for(
                    self._start_component(name), 
                    timeout=timeout
                )
                if not success:
                    raise RuntimeError(f"组件 {name} 启动失败")
            
            self._running = True
            self.startup_completed.set()
            
            # 启动健康检查
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            total_time = time.time() - start_time
            logger.info(f"系统启动完成，总耗时 {total_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            # 启动失败时尝试清理已启动的组件
            await self.stop_all()
            raise
    
    async def stop_all(self, timeout: float = 30.0):
        """停止所有组件"""
        if not self._running:
            logger.warning("系统未在运行")
            return
        
        logger.info("开始停止系统所有组件")
        start_time = time.time()
        
        try:
            # 停止健康检查
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            shutdown_order = self._resolve_shutdown_order()
            logger.info(f"组件关闭顺序: {shutdown_order}")
            
            # 按逆依赖顺序停止组件
            for name in shutdown_order:
                await asyncio.wait_for(
                    self._stop_component(name), 
                    timeout=timeout
                )
            
            self._running = False
            self.shutdown_completed.set()
            
            total_time = time.time() - start_time
            logger.info(f"系统关闭完成，总耗时 {total_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"系统关闭过程中出现错误: {e}")
            raise
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                for name, component in self.components.items():
                    if component.status == ComponentStatus.RUNNING and component.health_check_func:
                        try:
                            if asyncio.iscoroutinefunction(component.health_check_func):
                                is_healthy = await component.health_check_func()
                            else:
                                is_healthy = component.health_check_func()
                            
                            if not is_healthy:
                                logger.warning(f"组件 {name} 健康检查失败")
                                component.status = ComponentStatus.ERROR
                        except Exception as e:
                            logger.error(f"组件 {name} 健康检查异常: {e}")
                            component.status = ComponentStatus.ERROR
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环出现异常: {e}")
                await asyncio.sleep(5)
    
    def initiate_shutdown(self):
        """触发系统关闭"""
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            logger.info("收到关闭信号，系统将开始关闭")
    
    async def wait_for_shutdown(self):
        """等待关闭信号"""
        await self.shutdown_event.wait()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self._running,
            "startup_completed": self.startup_completed.is_set(),
            "shutdown_completed": self.shutdown_completed.is_set(),
            "components": {
                name: {
                    "status": component.status.value,
                    "start_time": component.start_time,
                    "error": str(component.error) if component.error else None,
                    "dependencies": component.dependencies
                }
                for name, component in self.components.items()
            }
        }
    
    @asynccontextmanager
    async def lifecycle(self):
        """系统生命周期上下文管理器"""
        try:
            await self.start_all()
            yield self
        finally:
            await self.stop_all()


# 全局系统编排器实例
system_orchestrator = SystemOrchestrator()