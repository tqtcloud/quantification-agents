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
        # 延迟创建 asyncio 对象，避免事件循环不匹配
        self.shutdown_event = None
        self.startup_completed = None
        self.shutdown_completed = None
        self._running = False
        self._startup_tasks: List[asyncio.Task] = []
        self._shutdown_tasks: List[asyncio.Task] = []
        self._health_check_task: Optional[asyncio.Task] = None
        self._async_objects_initialized = False

    def _ensure_async_objects(self):
        """确保 asyncio 对象在正确的事件循环中创建"""
        if not self._async_objects_initialized:
            self.shutdown_event = asyncio.Event()
            self.startup_completed = asyncio.Event()
            self.shutdown_completed = asyncio.Event()
            self._async_objects_initialized = True
            
            # 现在设置信号处理器
            self._setup_signal_handlers()
        
        # 延迟设置信号处理器，避免在 asyncio 对象未创建时触发
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，开始优雅关闭")
            # 安全地设置关闭事件
            if self.shutdown_event and not self.shutdown_event.is_set():
                # 使用 call_soon_threadsafe 确保线程安全
                try:
                    loop = asyncio.get_event_loop()
                    if loop and not loop.is_closed():
                        loop.call_soon_threadsafe(self.shutdown_event.set)
                except RuntimeError:
                    # 如果无法获取事件循环，直接设置
                    self.shutdown_event.set()
        
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
    
    async def _start_component(self, name: str, timeout: float = 30.0) -> bool:
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
            
            # 执行启动函数（带超时控制）
            if component.startup_func:
                try:
                    if asyncio.iscoroutinefunction(component.startup_func):
                        await asyncio.wait_for(component.startup_func(), timeout=timeout)
                    else:
                        component.startup_func()
                except asyncio.TimeoutError:
                    raise RuntimeError(f"组件 {name} 启动超时 ({timeout}秒)")
            
            component.status = ComponentStatus.RUNNING
            logger.info(f"组件 {name} 启动成功，耗时 {time.time() - component.start_time:.2f}秒")
            return True
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error = e
            logger.error(f"组件 {name} 启动失败: {e}")
            return False
    
    async def _stop_component(self, name: str, timeout: float = 30.0) -> bool:
        """停止单个组件"""
        component = self.components[name]
        
        try:
            logger.info(f"停止组件: {name}")
            component.status = ComponentStatus.STOPPING
            
            # 执行关闭函数（带超时控制）
            if component.shutdown_func:
                try:
                    if asyncio.iscoroutinefunction(component.shutdown_func):
                        await asyncio.wait_for(component.shutdown_func(), timeout=timeout)
                    else:
                        component.shutdown_func()
                except asyncio.TimeoutError:
                    logger.warning(f"组件 {name} 停止超时 ({timeout}秒)，强制停止")
            
            component.status = ComponentStatus.STOPPED
            logger.info(f"组件 {name} 已停止")
            return True
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.error = e
            logger.error(f"组件 {name} 停止失败: {e}")
            return False

    async def _start_health_check(self):
        """启动健康检查任务"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("健康检查任务已在运行")
            return
        
        logger.info("启动系统健康检查")
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _stop_health_check(self):
        """停止健康检查任务"""
        if self._health_check_task and not self._health_check_task.done():
            logger.info("停止系统健康检查")
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
    
    async def _health_check_loop(self):
        """健康检查循环"""
        try:
            while True:
                await asyncio.sleep(30)  # 每30秒检查一次
                
                if not self._running:
                    break
                
                # 检查所有组件状态
                failed_components = []
                for name, component in self.components.items():
                    if component.status == ComponentStatus.ERROR:
                        failed_components.append(name)
                
                if failed_components:
                    logger.warning(f"发现失败的组件: {failed_components}")
                    # 可以在这里添加重启逻辑或告警
                
        except asyncio.CancelledError:
            logger.info("健康检查任务被取消")
        except Exception as e:
            logger.error(f"健康检查出错: {e}")
    
    async def start_all(self, timeout: float = 60.0):
        """启动所有组件"""
        # 确保 asyncio 对象已创建
        self._ensure_async_objects()
        
        if self._running:
            logger.warning("系统已经在运行中")
            return
        
        logger.info("开始启动系统组件")
        start_time = time.time()
        
        try:
            # 重置事件
            self.startup_completed.clear()
            self.shutdown_completed.clear()
            
            # 解析启动顺序
            startup_order = self._resolve_startup_order()
            logger.info(f"组件启动顺序: {startup_order}")
            
            # 按依赖顺序启动组件
            for component_name in startup_order:
                await self._start_component(component_name, timeout)
            
            # 启动健康检查
            await self._start_health_check()
            
            self._running = True
            self.startup_completed.set()
            
            elapsed = time.time() - start_time
            logger.info(f"所有组件启动完成，耗时: {elapsed:.2f}秒")
            
        except asyncio.TimeoutError:
            logger.error(f"系统启动超时 ({timeout}秒)")
            await self.stop_all()
            raise
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            await self.stop_all()
            raise
    
    async def stop_all(self, timeout: float = 30.0):
        """停止所有组件"""
        # 确保 asyncio 对象已创建
        self._ensure_async_objects()
        
        if not self._running:
            logger.info("系统未在运行，无需停止")
            return
        
        logger.info("开始停止系统组件")
        start_time = time.time()
        
        try:
            self.shutdown_completed.clear()
            
            # 停止健康检查
            await self._stop_health_check()
            
            # 按启动顺序的反序停止组件
            startup_order = self._resolve_startup_order()
            shutdown_order = list(reversed(startup_order))
            
            logger.info(f"组件停止顺序: {shutdown_order}")
            
            for component_name in shutdown_order:
                await self._stop_component(component_name, timeout)
            
            self._running = False
            self.shutdown_completed.set()
            
            elapsed = time.time() - start_time
            logger.info(f"所有组件停止完成，耗时: {elapsed:.2f}秒")
            
        except asyncio.TimeoutError:
            logger.error(f"系统停止超时 ({timeout}秒)")
            # 强制停止所有任务
            for task in self._startup_tasks + self._shutdown_tasks:
                if not task.done():
                    task.cancel()
            raise
        except Exception as e:
            logger.error(f"系统停止时出错: {e}")
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
        # 确保 asyncio 对象已创建
        if not self._async_objects_initialized:
            logger.warning("Asyncio 对象未初始化，跳过关闭信号")
            return
            
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()
            logger.info("收到关闭信号，系统将开始关闭")
    
    async def wait_for_shutdown(self):
        """等待关闭信号"""
        # 确保 asyncio 对象已创建
        self._ensure_async_objects()
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