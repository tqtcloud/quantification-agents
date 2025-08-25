"""
资源分配器 (ResourceAllocator)
实现策略资源隔离和分配管理系统
"""

import asyncio
import psutil
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from src.utils.logger import LoggerMixin


class ResourceType(Enum):
    """资源类型"""
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"


class AllocationStatus(Enum):
    """分配状态"""
    ALLOCATED = "allocated"
    PENDING = "pending"
    RELEASED = "released"
    INSUFFICIENT = "insufficient"
    ERROR = "error"


@dataclass
class ResourceLimit:
    """资源限制"""
    memory_mb: int = 1024
    cpu_percent: float = 25.0
    network_connections: int = 100
    storage_mb: int = 1024
    gpu_percent: float = 0.0
    
    def __post_init__(self):
        """验证资源限制"""
        if self.memory_mb <= 0:
            raise ValueError("内存限制必须大于0")
        if not 0 < self.cpu_percent <= 100:
            raise ValueError("CPU限制必须在0-100之间")
        if self.network_connections <= 0:
            raise ValueError("网络连接限制必须大于0")


@dataclass
class ResourceUsage:
    """资源使用情况"""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    network_connections: int = 0
    storage_mb: float = 0.0
    gpu_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memory_mb': self.memory_mb,
            'cpu_percent': self.cpu_percent,
            'network_connections': self.network_connections,
            'storage_mb': self.storage_mb,
            'gpu_percent': self.gpu_percent,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ResourceAllocation:
    """资源分配记录"""
    strategy_id: str
    strategy_type: str
    resource_limit: ResourceLimit
    current_usage: ResourceUsage = field(default_factory=ResourceUsage)
    allocation_time: datetime = field(default_factory=datetime.now)
    status: AllocationStatus = AllocationStatus.PENDING
    process_ids: Set[int] = field(default_factory=set)
    isolation_group: Optional[str] = None
    priority: int = 1
    
    def get_utilization_rate(self) -> Dict[str, float]:
        """获取资源利用率"""
        return {
            'memory': min(self.current_usage.memory_mb / self.resource_limit.memory_mb, 1.0) if self.resource_limit.memory_mb > 0 else 0.0,
            'cpu': min(self.current_usage.cpu_percent / self.resource_limit.cpu_percent, 1.0) if self.resource_limit.cpu_percent > 0 else 0.0,
            'network': min(self.current_usage.network_connections / self.resource_limit.network_connections, 1.0) if self.resource_limit.network_connections > 0 else 0.0,
            'storage': min(self.current_usage.storage_mb / self.resource_limit.storage_mb, 1.0) if self.resource_limit.storage_mb > 0 else 0.0,
            'gpu': min(self.current_usage.gpu_percent / self.resource_limit.gpu_percent, 1.0) if self.resource_limit.gpu_percent > 0 else 0.0
        }
    
    def is_over_limit(self) -> Dict[str, bool]:
        """检查是否超过限制"""
        utilization = self.get_utilization_rate()
        return {resource: rate > 1.0 for resource, rate in utilization.items()}


@dataclass
class SystemResources:
    """系统资源状态"""
    total_memory_mb: float
    available_memory_mb: float
    total_cpu_cores: int
    cpu_usage_percent: float
    total_network_connections: int
    available_storage_mb: float
    gpu_count: int = 0
    gpu_memory_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_current(cls) -> 'SystemResources':
        """获取当前系统资源状态"""
        try:
            memory = psutil.virtual_memory()
            storage = psutil.disk_usage('/')
            
            # 尝试获取网络连接数，如果失败则使用默认值
            try:
                network_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, PermissionError):
                network_connections = 0  # 无法获取时使用默认值
            
            return cls(
                total_memory_mb=memory.total / 1024 / 1024,
                available_memory_mb=memory.available / 1024 / 1024,
                total_cpu_cores=psutil.cpu_count(),
                cpu_usage_percent=psutil.cpu_percent(interval=0.1),  # 减少interval避免阻塞
                total_network_connections=network_connections,
                available_storage_mb=storage.free / 1024 / 1024
            )
        except Exception as e:
            # 如果完全失败，返回默认值
            return cls(
                total_memory_mb=8192,  # 默认8GB
                available_memory_mb=4096,  # 默认4GB可用
                total_cpu_cores=4,  # 默认4核
                cpu_usage_percent=0.0,
                total_network_connections=0,
                available_storage_mb=100000  # 默认100GB
            )


class ResourceAllocator(LoggerMixin):
    """
    资源分配器
    
    负责管理策略的资源分配、隔离和监控，
    确保不同策略之间的资源完全隔离
    """
    
    def __init__(self):
        """初始化资源分配器"""
        # 分配记录
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # 系统资源状态
        self.system_resources: Optional[SystemResources] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        # 资源池管理
        self._reserved_resources: ResourceUsage = ResourceUsage()
        self._allocated_resources: ResourceUsage = ResourceUsage()
        
        # 隔离组管理
        self._isolation_groups: Dict[str, Set[str]] = {
            'hft': set(),      # 高频交易策略组
            'ai_agent': set()   # AI策略组
        }
        
        # 监控配置
        self._monitor_interval: int = 5  # 监控间隔（秒）
        self._is_monitoring: bool = False
        
        # 线程锁
        self._allocation_lock = asyncio.Lock()
        
        self.log_info("资源分配器初始化完成")
    
    async def initialize(self):
        """初始化资源分配器"""
        try:
            # 获取初始系统资源状态
            self.system_resources = SystemResources.get_current()
            
            # 启动资源监控
            await self.start_monitoring()
            
            self.log_info(f"资源分配器初始化完成，系统资源: 内存 {self.system_resources.total_memory_mb:.0f}MB, "
                         f"CPU {self.system_resources.total_cpu_cores}核, 网络连接 {self.system_resources.total_network_connections}")
            
        except Exception as e:
            self.log_error(f"资源分配器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭资源分配器"""
        try:
            # 停止监控
            await self.stop_monitoring()
            
            # 释放所有分配
            allocation_ids = list(self.allocations.keys())
            for strategy_id in allocation_ids:
                await self.release_resources(strategy_id)
            
            self.log_info("资源分配器已关闭")
            
        except Exception as e:
            self.log_error(f"关闭资源分配器失败: {e}")
    
    async def allocate_resources(self, config) -> bool:
        """
        分配资源给策略
        
        Args:
            config: 策略配置
            
        Returns:
            是否成功分配
        """
        async with self._allocation_lock:
            try:
                strategy_id = config.strategy_id
                strategy_type = config.strategy_type.value
                
                # 检查是否已经分配
                if strategy_id in self.allocations:
                    self.log_warning(f"策略 {strategy_id} 资源已分配")
                    return True
                
                # 创建资源限制
                resource_limit = ResourceLimit(
                    memory_mb=config.max_memory_mb,
                    cpu_percent=config.max_cpu_percent,
                    network_connections=config.max_network_connections
                )
                
                # 检查资源可用性
                if not await self._check_resource_availability(resource_limit):
                    self.log_error(f"策略 {strategy_id} 资源不足，无法分配")
                    return False
                
                # 确定隔离组
                isolation_group = self._get_isolation_group(config.strategy_type)
                
                # 创建分配记录
                allocation = ResourceAllocation(
                    strategy_id=strategy_id,
                    strategy_type=strategy_type,
                    resource_limit=resource_limit,
                    isolation_group=isolation_group,
                    priority=config.priority,
                    status=AllocationStatus.ALLOCATED
                )
                
                # 记录分配
                self.allocations[strategy_id] = allocation
                self._isolation_groups[isolation_group].add(strategy_id)
                
                # 更新已分配资源
                self._update_allocated_resources()
                
                self.log_info(f"为策略 {strategy_id} 分配资源: 内存 {resource_limit.memory_mb}MB, "
                             f"CPU {resource_limit.cpu_percent}%, 网络 {resource_limit.network_connections}连接")
                
                return True
                
            except Exception as e:
                self.log_error(f"分配资源失败: {e}")
                return False
    
    async def release_resources(self, strategy_id: str) -> bool:
        """
        释放策略资源
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功释放
        """
        async with self._allocation_lock:
            try:
                if strategy_id not in self.allocations:
                    self.log_warning(f"策略 {strategy_id} 没有资源分配记录")
                    return False
                
                allocation = self.allocations[strategy_id]
                
                # 从隔离组移除
                if allocation.isolation_group:
                    self._isolation_groups[allocation.isolation_group].discard(strategy_id)
                
                # 清理进程资源限制
                await self._cleanup_process_limits(allocation)
                
                # 更新分配状态
                allocation.status = AllocationStatus.RELEASED
                
                # 从分配表移除
                del self.allocations[strategy_id]
                
                # 更新已分配资源
                self._update_allocated_resources()
                
                self.log_info(f"释放策略 {strategy_id} 的资源")
                return True
                
            except Exception as e:
                self.log_error(f"释放资源失败: {e}")
                return False
    
    async def check_resources_available(self, config) -> bool:
        """
        检查资源是否可用
        
        Args:
            config: 策略配置
            
        Returns:
            资源是否可用
        """
        try:
            resource_limit = ResourceLimit(
                memory_mb=config.max_memory_mb,
                cpu_percent=config.max_cpu_percent,
                network_connections=config.max_network_connections
            )
            
            return await self._check_resource_availability(resource_limit)
            
        except Exception as e:
            self.log_error(f"检查资源可用性失败: {e}")
            return False
    
    async def update_resource_usage(self, strategy_id: str, process_ids: Set[int]) -> bool:
        """
        更新策略资源使用情况
        
        Args:
            strategy_id: 策略ID
            process_ids: 进程ID集合
            
        Returns:
            是否成功更新
        """
        try:
            if strategy_id not in self.allocations:
                return False
            
            allocation = self.allocations[strategy_id]
            allocation.process_ids = process_ids
            
            # 计算实际资源使用
            usage = await self._calculate_process_resource_usage(process_ids)
            allocation.current_usage = usage
            
            # 检查是否超限
            over_limits = allocation.is_over_limit()
            if any(over_limits.values()):
                self.log_warning(f"策略 {strategy_id} 资源使用超限: {over_limits}")
            
            return True
            
        except Exception as e:
            self.log_error(f"更新资源使用失败: {e}")
            return False
    
    async def get_resource_status(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取资源状态
        
        Args:
            strategy_id: 策略ID，None表示获取全部
            
        Returns:
            资源状态信息
        """
        try:
            if strategy_id:
                # 获取指定策略的资源状态
                if strategy_id not in self.allocations:
                    return {}
                
                allocation = self.allocations[strategy_id]
                return {
                    'strategy_id': strategy_id,
                    'strategy_type': allocation.strategy_type,
                    'status': allocation.status.value,
                    'isolation_group': allocation.isolation_group,
                    'priority': allocation.priority,
                    'resource_limit': {
                        'memory_mb': allocation.resource_limit.memory_mb,
                        'cpu_percent': allocation.resource_limit.cpu_percent,
                        'network_connections': allocation.resource_limit.network_connections
                    },
                    'current_usage': allocation.current_usage.to_dict(),
                    'utilization_rate': allocation.get_utilization_rate(),
                    'over_limit': allocation.is_over_limit(),
                    'allocation_time': allocation.allocation_time.isoformat()
                }
            else:
                # 获取系统整体资源状态
                return {
                    'system_resources': {
                        'total_memory_mb': self.system_resources.total_memory_mb if self.system_resources else 0,
                        'available_memory_mb': self.system_resources.available_memory_mb if self.system_resources else 0,
                        'total_cpu_cores': self.system_resources.total_cpu_cores if self.system_resources else 0,
                        'cpu_usage_percent': self.system_resources.cpu_usage_percent if self.system_resources else 0,
                        'total_network_connections': self.system_resources.total_network_connections if self.system_resources else 0
                    },
                    'allocated_resources': self._allocated_resources.to_dict(),
                    'total_allocations': len(self.allocations),
                    'isolation_groups': {
                        group: len(strategies) for group, strategies in self._isolation_groups.items()
                    },
                    'allocations': {
                        strategy_id: {
                            'status': allocation.status.value,
                            'type': allocation.strategy_type,
                            'group': allocation.isolation_group,
                            'priority': allocation.priority
                        }
                        for strategy_id, allocation in self.allocations.items()
                    }
                }
                
        except Exception as e:
            self.log_error(f"获取资源状态失败: {e}")
            return {}
    
    async def start_monitoring(self):
        """启动资源监控"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
        self.log_info("资源监控已启动")
    
    async def stop_monitoring(self):
        """停止资源监控"""
        self._is_monitoring = False
        
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass
        
        self.log_info("资源监控已停止")
    
    async def monitor_system_resources(self):
        """监控系统资源"""
        try:
            # 更新系统资源状态
            self.system_resources = SystemResources.get_current()
            
            # 更新所有策略的资源使用情况
            for strategy_id, allocation in self.allocations.items():
                if allocation.process_ids:
                    usage = await self._calculate_process_resource_usage(allocation.process_ids)
                    allocation.current_usage = usage
            
            # 更新已分配资源统计
            self._update_allocated_resources()
            
            # 检查资源异常
            await self._check_resource_anomalies()
            
        except Exception as e:
            self.log_error(f"监控系统资源失败: {e}")
    
    def get_isolation_groups(self) -> Dict[str, List[str]]:
        """获取隔离组信息"""
        return {
            group: list(strategies) 
            for group, strategies in self._isolation_groups.items()
        }
    
    async def set_resource_limits(self, strategy_id: str, resource_limit: ResourceLimit) -> bool:
        """
        设置资源限制
        
        Args:
            strategy_id: 策略ID
            resource_limit: 新的资源限制
            
        Returns:
            是否成功设置
        """
        try:
            if strategy_id not in self.allocations:
                self.log_error(f"策略 {strategy_id} 不存在")
                return False
            
            allocation = self.allocations[strategy_id]
            old_limit = allocation.resource_limit
            
            # 检查新限制的可用性
            if not await self._check_resource_availability(resource_limit, exclude_strategy=strategy_id):
                self.log_error(f"策略 {strategy_id} 新资源限制超出系统可用资源")
                return False
            
            # 更新资源限制
            allocation.resource_limit = resource_limit
            
            # 应用到进程
            if allocation.process_ids:
                await self._apply_process_limits(allocation)
            
            self.log_info(f"更新策略 {strategy_id} 资源限制: 内存 {old_limit.memory_mb}->{resource_limit.memory_mb}MB, "
                         f"CPU {old_limit.cpu_percent}->{resource_limit.cpu_percent}%")
            
            return True
            
        except Exception as e:
            self.log_error(f"设置资源限制失败: {e}")
            return False
    
    async def _resource_monitor_loop(self):
        """资源监控循环"""
        while self._is_monitoring:
            try:
                await self.monitor_system_resources()
                await asyncio.sleep(self._monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"资源监控循环错误: {e}")
                await asyncio.sleep(self._monitor_interval)
    
    async def _check_resource_availability(self, resource_limit: ResourceLimit, exclude_strategy: Optional[str] = None) -> bool:
        """检查资源可用性"""
        try:
            if not self.system_resources:
                self.system_resources = SystemResources.get_current()
            
            # 计算当前已分配资源（排除指定策略）
            allocated_memory = 0.0
            allocated_cpu = 0.0
            allocated_connections = 0
            
            for strategy_id, allocation in self.allocations.items():
                if strategy_id == exclude_strategy:
                    continue
                if allocation.status == AllocationStatus.ALLOCATED:
                    allocated_memory += allocation.resource_limit.memory_mb
                    allocated_cpu += allocation.resource_limit.cpu_percent
                    allocated_connections += allocation.resource_limit.network_connections
            
            # 检查内存
            required_memory = allocated_memory + resource_limit.memory_mb
            available_memory = self.system_resources.available_memory_mb
            if required_memory > available_memory * 0.9:  # 保留10%缓冲
                self.log_warning(f"内存不足: 需要 {required_memory:.0f}MB, 可用 {available_memory:.0f}MB")
                return False
            
            # 检查CPU
            required_cpu = allocated_cpu + resource_limit.cpu_percent
            if required_cpu > self.system_resources.total_cpu_cores * 100 * 0.8:  # 保留20%缓冲
                self.log_warning(f"CPU不足: 需要 {required_cpu:.1f}%, 总计 {self.system_resources.total_cpu_cores * 100}%")
                return False
            
            # 检查网络连接
            if allocated_connections + resource_limit.network_connections > 10000:  # 系统连接数限制
                self.log_warning(f"网络连接数不足")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"检查资源可用性失败: {e}")
            return False
    
    async def _calculate_process_resource_usage(self, process_ids: Set[int]) -> ResourceUsage:
        """计算进程资源使用情况"""
        usage = ResourceUsage()
        
        try:
            total_memory = 0.0
            total_cpu = 0.0
            total_connections = 0
            
            for pid in process_ids:
                try:
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    total_memory += memory_info.rss / 1024 / 1024  # MB
                    total_cpu += process.cpu_percent()
                    
                    # 网络连接可能需要特殊权限，安全处理
                    try:
                        total_connections += len(process.connections())
                    except (psutil.AccessDenied, PermissionError):
                        pass  # 无权限时跳过连接统计
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    continue
            
            usage.memory_mb = total_memory
            usage.cpu_percent = total_cpu
            usage.network_connections = total_connections
            usage.last_updated = datetime.now()
            
        except Exception as e:
            self.log_error(f"计算进程资源使用失败: {e}")
        
        return usage
    
    def _get_isolation_group(self, strategy_type) -> str:
        """获取隔离组"""
        if strategy_type.value == 'hft':
            return 'hft'
        elif strategy_type.value == 'ai_agent':
            return 'ai_agent'
        else:
            return 'default'
    
    def _update_allocated_resources(self):
        """更新已分配资源统计"""
        total_memory = 0.0
        total_cpu = 0.0
        total_connections = 0
        
        for allocation in self.allocations.values():
            if allocation.status == AllocationStatus.ALLOCATED:
                total_memory += allocation.current_usage.memory_mb
                total_cpu += allocation.current_usage.cpu_percent
                total_connections += allocation.current_usage.network_connections
        
        self._allocated_resources = ResourceUsage(
            memory_mb=total_memory,
            cpu_percent=total_cpu,
            network_connections=total_connections,
            last_updated=datetime.now()
        )
    
    async def _check_resource_anomalies(self):
        """检查资源异常"""
        try:
            for strategy_id, allocation in self.allocations.items():
                if allocation.status != AllocationStatus.ALLOCATED:
                    continue
                
                # 检查资源使用是否超限
                over_limits = allocation.is_over_limit()
                if any(over_limits.values()):
                    over_resources = [resource for resource, over in over_limits.items() if over]
                    self.log_warning(f"策略 {strategy_id} 资源使用超限: {over_resources}")
                
                # 检查资源使用率异常
                utilization = allocation.get_utilization_rate()
                for resource, rate in utilization.items():
                    if rate > 0.95:  # 使用率超过95%
                        self.log_warning(f"策略 {strategy_id} {resource} 使用率过高: {rate:.1%}")
            
        except Exception as e:
            self.log_error(f"检查资源异常失败: {e}")
    
    async def _apply_process_limits(self, allocation: ResourceAllocation):
        """应用进程资源限制"""
        try:
            # 这里可以实现具体的进程资源限制
            # 例如使用cgroups、systemd、或其他系统级资源控制机制
            self.log_debug(f"应用进程资源限制: {allocation.strategy_id}")
        except Exception as e:
            self.log_error(f"应用进程资源限制失败: {e}")
    
    async def _cleanup_process_limits(self, allocation: ResourceAllocation):
        """清理进程资源限制"""
        try:
            # 清理资源限制
            self.log_debug(f"清理进程资源限制: {allocation.strategy_id}")
        except Exception as e:
            self.log_error(f"清理进程资源限制失败: {e}")