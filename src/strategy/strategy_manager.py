"""
策略管理器 (StrategyManager)
实现高频交易和AI策略的双策略管理和隔离系统
"""

import asyncio
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import uuid
import psutil
import threading

from src.hft.hft_engine import HFTEngine, HFTConfig
from src.core.message_bus import MessageBus
from src.utils.logger import LoggerMixin

# 可选导入，避免依赖问题
try:
    from src.agents.orchestrator import MultiAgentOrchestrator, WorkflowConfig
except ImportError:
    MultiAgentOrchestrator = None
    WorkflowConfig = None


class StrategyType(Enum):
    """策略类型"""
    HFT = "hft"          # 高频交易策略
    AI_AGENT = "ai_agent"  # AI智能策略


class StrategyStatus(Enum):
    """策略状态"""
    IDLE = "idle"                  # 空闲
    INITIALIZING = "initializing"   # 初始化中
    RUNNING = "running"            # 运行中
    PAUSED = "paused"             # 暂停
    STOPPING = "stopping"         # 停止中
    STOPPED = "stopped"           # 已停止
    ERROR = "error"               # 错误状态
    TERMINATED = "terminated"     # 已终止


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str = ""
    
    # 资源配置
    max_memory_mb: int = 1024      # 最大内存使用（MB）
    max_cpu_percent: float = 25.0  # 最大CPU使用率（%）
    max_network_connections: int = 100  # 最大网络连接数
    priority: int = 1              # 优先级（1-10，数字越大优先级越高）
    
    # 特定配置
    hft_config: Optional[HFTConfig] = None
    workflow_config: Optional[Any] = None  # WorkflowConfig可能不可用
    
    # 运行配置
    auto_restart: bool = True      # 自动重启
    max_restarts: int = 5         # 最大重启次数
    restart_delay_seconds: int = 30  # 重启延迟
    health_check_interval: int = 10  # 健康检查间隔（秒）
    
    # 风险控制
    max_daily_loss: Optional[Decimal] = None  # 每日最大损失
    max_position_value: Optional[Decimal] = None  # 最大仓位价值
    emergency_stop_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """策略性能指标"""
    strategy_id: str
    strategy_type: StrategyType
    status: StrategyStatus
    
    # 运行统计
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    restart_count: int = 0
    error_count: int = 0
    
    # 资源使用
    current_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    current_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    network_connections: int = 0
    
    # 交易统计（如果适用）
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    daily_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # 自定义指标
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyInstance:
    """策略实例"""
    config: StrategyConfig
    metrics: StrategyMetrics
    
    # 核心组件
    engine: Optional[Union[HFTEngine, Any]] = None  # MultiAgentOrchestrator可能不可用
    task: Optional[asyncio.Task] = None
    thread: Optional[threading.Thread] = None
    process_id: Optional[int] = None
    
    # 控制组件
    stop_event: Optional[asyncio.Event] = None
    health_check_task: Optional[asyncio.Task] = None
    
    # 回调函数
    callbacks: Dict[str, List[Callable]] = field(default_factory=lambda: {
        'on_start': [],
        'on_stop': [],
        'on_error': [],
        'on_restart': [],
        'on_metrics_update': []
    })


class StrategyManager(LoggerMixin):
    """
    策略管理器
    
    负责管理高频交易策略和AI智能策略的生命周期，
    实现完全隔离的双策略运行环境
    """
    
    def __init__(self, message_bus: Optional[MessageBus] = None):
        """
        初始化策略管理器
        
        Args:
            message_bus: 消息总线实例
        """
        self.message_bus = message_bus
        
        # 策略实例管理
        self.strategies: Dict[str, StrategyInstance] = {}
        self._strategy_types: Dict[StrategyType, Set[str]] = {
            StrategyType.HFT: set(),
            StrategyType.AI_AGENT: set()
        }
        
        # 全局控制
        self._is_running = False
        self._global_stop_event: Optional[asyncio.Event] = None
        self._management_task: Optional[asyncio.Task] = None
        
        # 资源管理
        self._resource_allocator = None  # 将在后面实现
        self._strategy_monitor = None    # 将在后面实现
        
        # 统计信息
        self._start_time = datetime.now()
        self._total_strategies_created = 0
        self._total_strategies_terminated = 0
        
        self.log_info("策略管理器初始化完成")
    
    async def initialize(self):
        """初始化策略管理器"""
        if self._is_running:
            self.log_warning("策略管理器已在运行")
            return
        
        try:
            # 导入资源分配器和监控器（避免循环导入）
            from .resource_allocator import ResourceAllocator
            from .strategy_monitor import StrategyMonitor
            
            self._resource_allocator = ResourceAllocator()
            self._strategy_monitor = StrategyMonitor(self)
            
            # 初始化组件
            await self._resource_allocator.initialize()
            await self._strategy_monitor.initialize()
            
            # 设置全局停止事件
            self._global_stop_event = asyncio.Event()
            
            # 启动管理任务
            self._management_task = asyncio.create_task(self._management_loop())
            
            self._is_running = True
            self.log_info("策略管理器初始化完成")
            
        except Exception as e:
            self.log_error(f"策略管理器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭策略管理器"""
        if not self._is_running:
            return
        
        self.log_info("开始关闭策略管理器...")
        
        try:
            # 设置全局停止标志
            if self._global_stop_event:
                self._global_stop_event.set()
            
            # 停止所有策略
            await self._stop_all_strategies()
            
            # 停止管理任务
            if self._management_task:
                self._management_task.cancel()
                try:
                    await self._management_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭资源管理器
            if self._resource_allocator:
                await self._resource_allocator.shutdown()
            
            if self._strategy_monitor:
                await self._strategy_monitor.shutdown()
            
            self._is_running = False
            self.log_info("策略管理器已关闭")
            
        except Exception as e:
            self.log_error(f"策略管理器关闭失败: {e}")
            raise
    
    async def register_strategy(self, config: StrategyConfig) -> str:
        """
        注册策略
        
        Args:
            config: 策略配置
            
        Returns:
            策略ID
        """
        try:
            # 验证配置
            self._validate_strategy_config(config)
            
            # 检查策略是否已存在
            if config.strategy_id in self.strategies:
                raise ValueError(f"策略 {config.strategy_id} 已存在")
            
            # 创建策略指标
            metrics = StrategyMetrics(
                strategy_id=config.strategy_id,
                strategy_type=config.strategy_type,
                status=StrategyStatus.IDLE
            )
            
            # 创建策略实例
            instance = StrategyInstance(
                config=config,
                metrics=metrics
            )
            
            # 注册策略
            self.strategies[config.strategy_id] = instance
            self._strategy_types[config.strategy_type].add(config.strategy_id)
            self._total_strategies_created += 1
            
            # 分配资源
            if self._resource_allocator:
                await self._resource_allocator.allocate_resources(config)
            
            self.log_info(f"策略 {config.strategy_id} 注册成功，类型: {config.strategy_type.value}")
            return config.strategy_id
            
        except Exception as e:
            self.log_error(f"注册策略失败: {e}")
            raise
    
    async def unregister_strategy(self, strategy_id: str) -> bool:
        """
        注销策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功注销
        """
        try:
            if strategy_id not in self.strategies:
                self.log_warning(f"策略 {strategy_id} 不存在")
                return False
            
            instance = self.strategies[strategy_id]
            
            # 如果策略正在运行，先停止它
            if instance.metrics.status == StrategyStatus.RUNNING:
                await self.stop_strategy(strategy_id, force=True)
            
            # 释放资源
            if self._resource_allocator:
                await self._resource_allocator.release_resources(strategy_id)
            
            # 从注册表中移除
            strategy_type = instance.config.strategy_type
            self._strategy_types[strategy_type].discard(strategy_id)
            del self.strategies[strategy_id]
            
            self._total_strategies_terminated += 1
            
            self.log_info(f"策略 {strategy_id} 已注销")
            return True
            
        except Exception as e:
            self.log_error(f"注销策略失败: {e}")
            return False
    
    async def start_strategy(self, strategy_id: str) -> bool:
        """
        启动策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            是否成功启动
        """
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"策略 {strategy_id} 不存在")
            
            instance = self.strategies[strategy_id]
            
            # 检查状态
            if instance.metrics.status == StrategyStatus.RUNNING:
                self.log_warning(f"策略 {strategy_id} 已在运行")
                return True
            
            # 更新状态为初始化中
            instance.metrics.status = StrategyStatus.INITIALIZING
            instance.metrics.start_time = datetime.now()
            
            # 检查资源可用性
            if self._resource_allocator:
                if not await self._resource_allocator.check_resources_available(instance.config):
                    raise RuntimeError(f"策略 {strategy_id} 资源不足，无法启动")
            
            # 根据策略类型启动对应的引擎
            if instance.config.strategy_type == StrategyType.HFT:
                success = await self._start_hft_strategy(instance)
            elif instance.config.strategy_type == StrategyType.AI_AGENT:
                success = await self._start_ai_agent_strategy(instance)
            else:
                raise ValueError(f"未知策略类型: {instance.config.strategy_type}")
            
            if success:
                instance.metrics.status = StrategyStatus.RUNNING
                instance.metrics.restart_count = 0
                
                # 启动健康检查
                instance.health_check_task = asyncio.create_task(
                    self._health_check_loop(instance)
                )
                
                # 通知回调
                await self._execute_callbacks(instance, 'on_start')
                
                self.log_info(f"策略 {strategy_id} 启动成功")
                return True
            else:
                instance.metrics.status = StrategyStatus.ERROR
                self.log_error(f"策略 {strategy_id} 启动失败")
                return False
                
        except Exception as e:
            if strategy_id in self.strategies:
                self.strategies[strategy_id].metrics.status = StrategyStatus.ERROR
                self.strategies[strategy_id].metrics.error_count += 1
            
            self.log_error(f"启动策略 {strategy_id} 失败: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: str, force: bool = False) -> bool:
        """
        停止策略
        
        Args:
            strategy_id: 策略ID
            force: 是否强制停止
            
        Returns:
            是否成功停止
        """
        try:
            if strategy_id not in self.strategies:
                self.log_warning(f"策略 {strategy_id} 不存在")
                return False
            
            instance = self.strategies[strategy_id]
            
            # 检查状态
            if instance.metrics.status in [StrategyStatus.STOPPED, StrategyStatus.TERMINATED]:
                self.log_warning(f"策略 {strategy_id} 已停止")
                return True
            
            # 更新状态
            instance.metrics.status = StrategyStatus.STOPPING
            
            try:
                # 停止健康检查
                if instance.health_check_task:
                    instance.health_check_task.cancel()
                
                # 设置停止事件
                if instance.stop_event:
                    instance.stop_event.set()
                
                # 停止引擎
                if instance.engine:
                    if isinstance(instance.engine, HFTEngine):
                        await instance.engine.stop()
                    elif MultiAgentOrchestrator and isinstance(instance.engine, MultiAgentOrchestrator):
                        await instance.engine.cancel_workflow()
                
                # 取消任务
                if instance.task and not instance.task.done():
                    instance.task.cancel()
                    if not force:
                        try:
                            await asyncio.wait_for(instance.task, timeout=10.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                
                # 更新状态
                instance.metrics.status = StrategyStatus.STOPPED
                
                # 计算运行时间
                if instance.metrics.start_time:
                    instance.metrics.uptime_seconds += (
                        datetime.now() - instance.metrics.start_time
                    ).total_seconds()
                
                # 通知回调
                await self._execute_callbacks(instance, 'on_stop')
                
                self.log_info(f"策略 {strategy_id} 已停止")
                return True
                
            except Exception as e:
                self.log_error(f"停止策略 {strategy_id} 时发生错误: {e}")
                if force:
                    instance.metrics.status = StrategyStatus.TERMINATED
                    return True
                else:
                    instance.metrics.status = StrategyStatus.ERROR
                    return False
                    
        except Exception as e:
            self.log_error(f"停止策略 {strategy_id} 失败: {e}")
            return False
    
    async def pause_strategy(self, strategy_id: str) -> bool:
        """暂停策略"""
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"策略 {strategy_id} 不存在")
            
            instance = self.strategies[strategy_id]
            if instance.metrics.status != StrategyStatus.RUNNING:
                self.log_warning(f"策略 {strategy_id} 未在运行")
                return False
            
            instance.metrics.status = StrategyStatus.PAUSED
            self.log_info(f"策略 {strategy_id} 已暂停")
            return True
            
        except Exception as e:
            self.log_error(f"暂停策略失败: {e}")
            return False
    
    async def resume_strategy(self, strategy_id: str) -> bool:
        """恢复策略"""
        try:
            if strategy_id not in self.strategies:
                raise ValueError(f"策略 {strategy_id} 不存在")
            
            instance = self.strategies[strategy_id]
            if instance.metrics.status != StrategyStatus.PAUSED:
                self.log_warning(f"策略 {strategy_id} 未暂停")
                return False
            
            instance.metrics.status = StrategyStatus.RUNNING
            self.log_info(f"策略 {strategy_id} 已恢复")
            return True
            
        except Exception as e:
            self.log_error(f"恢复策略失败: {e}")
            return False
    
    async def restart_strategy(self, strategy_id: str) -> bool:
        """重启策略"""
        try:
            self.log_info(f"重启策略 {strategy_id}")
            
            # 先停止策略
            await self.stop_strategy(strategy_id)
            
            # 等待一段时间
            if strategy_id in self.strategies:
                delay = self.strategies[strategy_id].config.restart_delay_seconds
                await asyncio.sleep(delay)
            
            # 重新启动
            success = await self.start_strategy(strategy_id)
            
            if success and strategy_id in self.strategies:
                instance = self.strategies[strategy_id]
                instance.metrics.restart_count += 1
                await self._execute_callbacks(instance, 'on_restart')
            
            return success
            
        except Exception as e:
            self.log_error(f"重启策略失败: {e}")
            return False
    
    def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """获取策略状态"""
        if strategy_id not in self.strategies:
            return None
        
        instance = self.strategies[strategy_id]
        return {
            'strategy_id': strategy_id,
            'strategy_type': instance.config.strategy_type.value,
            'status': instance.metrics.status.value,
            'name': instance.config.name,
            'description': instance.config.description,
            'metrics': self._get_strategy_metrics_dict(instance.metrics),
            'config': self._get_strategy_config_dict(instance.config)
        }
    
    def list_strategies(self, strategy_type: Optional[StrategyType] = None) -> List[Dict[str, Any]]:
        """列出所有策略"""
        strategies = []
        
        for strategy_id, instance in self.strategies.items():
            if strategy_type is None or instance.config.strategy_type == strategy_type:
                strategies.append({
                    'strategy_id': strategy_id,
                    'strategy_type': instance.config.strategy_type.value,
                    'name': instance.config.name,
                    'status': instance.metrics.status.value,
                    'uptime_seconds': instance.metrics.uptime_seconds,
                    'restart_count': instance.metrics.restart_count,
                    'error_count': instance.metrics.error_count
                })
        
        return strategies
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        total_strategies = len(self.strategies)
        running_strategies = len([s for s in self.strategies.values() 
                                 if s.metrics.status == StrategyStatus.RUNNING])
        
        return {
            'manager_status': 'running' if self._is_running else 'stopped',
            'start_time': self._start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'total_strategies': total_strategies,
            'running_strategies': running_strategies,
            'strategies_by_type': {
                strategy_type.value: len(strategy_ids)
                for strategy_type, strategy_ids in self._strategy_types.items()
            },
            'total_created': self._total_strategies_created,
            'total_terminated': self._total_strategies_terminated,
            'system_metrics': self._get_system_metrics()
        }
    
    def register_callback(self, strategy_id: str, event: str, callback: Callable):
        """注册回调函数"""
        if strategy_id in self.strategies:
            if event in self.strategies[strategy_id].callbacks:
                self.strategies[strategy_id].callbacks[event].append(callback)
    
    async def _start_hft_strategy(self, instance: StrategyInstance) -> bool:
        """启动HFT策略"""
        try:
            # 创建HFT引擎
            hft_config = instance.config.hft_config or HFTConfig()
            instance.engine = HFTEngine(hft_config)
            
            # 初始化引擎
            symbols = ['BTCUSDT', 'ETHUSDT']  # 默认交易对，实际应该从配置获取
            await instance.engine.initialize(symbols)
            
            # 启动引擎
            await instance.engine.start()
            
            # 创建停止事件
            instance.stop_event = asyncio.Event()
            
            # 创建监控任务
            instance.task = asyncio.create_task(
                self._hft_strategy_loop(instance)
            )
            
            return True
            
        except Exception as e:
            self.log_error(f"启动HFT策略失败: {e}")
            return False
    
    async def _start_ai_agent_strategy(self, instance: StrategyInstance) -> bool:
        """启动AI Agent策略"""
        try:
            if MultiAgentOrchestrator is None:
                self.log_error("MultiAgentOrchestrator不可用，请安装langgraph依赖")
                return False
            
            # 创建AI Agent编排器
            workflow_config = instance.config.workflow_config or WorkflowConfig()
            instance.engine = MultiAgentOrchestrator(workflow_config)
            
            # 创建停止事件
            instance.stop_event = asyncio.Event()
            
            # 创建监控任务
            instance.task = asyncio.create_task(
                self._ai_agent_strategy_loop(instance)
            )
            
            return True
            
        except Exception as e:
            self.log_error(f"启动AI Agent策略失败: {e}")
            return False
    
    async def _hft_strategy_loop(self, instance: StrategyInstance):
        """HFT策略运行循环"""
        try:
            while not instance.stop_event.is_set():
                # 这里实现HFT策略的主要逻辑
                # 实际实现中应该处理市场数据、生成信号、执行交易等
                
                # 更新指标
                await self._update_strategy_metrics(instance)
                
                # 短暂休眠
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self.log_info(f"HFT策略 {instance.config.strategy_id} 已被取消")
        except Exception as e:
            self.log_error(f"HFT策略 {instance.config.strategy_id} 运行错误: {e}")
            instance.metrics.status = StrategyStatus.ERROR
            instance.metrics.error_count += 1
    
    async def _ai_agent_strategy_loop(self, instance: StrategyInstance):
        """AI Agent策略运行循环"""
        try:
            while not instance.stop_event.is_set():
                # 这里实现AI Agent策略的主要逻辑
                # 可以定期执行工作流、处理市场分析等
                
                # 执行工作流（演示）
                if MultiAgentOrchestrator and isinstance(instance.engine, MultiAgentOrchestrator):
                    try:
                        result = await instance.engine.execute_workflow()
                        if result.get('success'):
                            instance.metrics.successful_trades += 1
                        instance.metrics.total_trades += 1
                    except Exception as e:
                        self.log_error(f"AI Agent工作流执行失败: {e}")
                        instance.metrics.error_count += 1
                
                # 更新指标
                await self._update_strategy_metrics(instance)
                
                # 休眠更长时间（AI策略通常不需要高频运行）
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            self.log_info(f"AI Agent策略 {instance.config.strategy_id} 已被取消")
        except Exception as e:
            self.log_error(f"AI Agent策略 {instance.config.strategy_id} 运行错误: {e}")
            instance.metrics.status = StrategyStatus.ERROR
            instance.metrics.error_count += 1
    
    async def _health_check_loop(self, instance: StrategyInstance):
        """健康检查循环"""
        try:
            interval = instance.config.health_check_interval
            
            while instance.metrics.status == StrategyStatus.RUNNING:
                # 检查引擎状态
                engine_healthy = await self._check_engine_health(instance)
                
                # 检查资源使用
                resource_healthy = await self._check_resource_usage(instance)
                
                # 检查业务指标
                business_healthy = await self._check_business_metrics(instance)
                
                if not (engine_healthy and resource_healthy and business_healthy):
                    self.log_warning(f"策略 {instance.config.strategy_id} 健康检查失败")
                    
                    # 根据配置决定是否自动重启
                    if instance.config.auto_restart and instance.metrics.restart_count < instance.config.max_restarts:
                        self.log_info(f"自动重启策略 {instance.config.strategy_id}")
                        asyncio.create_task(self.restart_strategy(instance.config.strategy_id))
                        break
                    else:
                        instance.metrics.status = StrategyStatus.ERROR
                        await self._execute_callbacks(instance, 'on_error')
                        break
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_error(f"健康检查失败: {e}")
    
    async def _management_loop(self):
        """管理循环"""
        try:
            while not self._global_stop_event.is_set():
                # 更新所有策略指标
                for instance in self.strategies.values():
                    if instance.metrics.status == StrategyStatus.RUNNING:
                        await self._update_strategy_metrics(instance)
                
                # 检查系统资源
                if self._resource_allocator:
                    await self._resource_allocator.monitor_system_resources()
                
                await asyncio.sleep(5)  # 每5秒更新一次
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_error(f"管理循环错误: {e}")
    
    async def _stop_all_strategies(self):
        """停止所有策略"""
        stop_tasks = []
        
        for strategy_id in list(self.strategies.keys()):
            task = asyncio.create_task(self.stop_strategy(strategy_id, force=True))
            stop_tasks.append(task)
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    def _validate_strategy_config(self, config: StrategyConfig):
        """验证策略配置"""
        if not config.strategy_id:
            raise ValueError("策略ID不能为空")
        
        if not config.name:
            raise ValueError("策略名称不能为空")
        
        if config.max_memory_mb <= 0:
            raise ValueError("最大内存必须大于0")
        
        if not 0 < config.max_cpu_percent <= 100:
            raise ValueError("CPU使用率必须在0-100之间")
    
    async def _update_strategy_metrics(self, instance: StrategyInstance):
        """更新策略指标"""
        try:
            # 更新基本指标
            instance.metrics.last_update_time = datetime.now()
            
            # 获取进程资源使用情况
            try:
                if instance.process_id:
                    process = psutil.Process(instance.process_id)
                    memory_info = process.memory_info()
                    instance.metrics.current_memory_mb = memory_info.rss / 1024 / 1024
                    instance.metrics.current_cpu_percent = process.cpu_percent()
                    
                    # 网络连接可能需要特殊权限，安全处理
                    try:
                        instance.metrics.network_connections = len(process.connections())
                    except (psutil.AccessDenied, PermissionError):
                        instance.metrics.network_connections = 0
                    
                    # 更新最大值
                    instance.metrics.max_memory_mb = max(
                        instance.metrics.max_memory_mb,
                        instance.metrics.current_memory_mb
                    )
                    instance.metrics.max_cpu_percent = max(
                        instance.metrics.max_cpu_percent,
                        instance.metrics.current_cpu_percent
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            
            # 更新引擎特定指标
            if instance.engine:
                if isinstance(instance.engine, HFTEngine):
                    hft_metrics = instance.engine.get_performance_metrics()
                    instance.metrics.custom_metrics['hft'] = {
                        'total_updates': hft_metrics.total_updates,
                        'avg_latency_ms': hft_metrics.avg_update_latency,
                        'total_signals': hft_metrics.total_signals,
                        'total_orders': hft_metrics.total_orders,
                        'filled_orders': hft_metrics.filled_orders
                    }
                elif MultiAgentOrchestrator and isinstance(instance.engine, MultiAgentOrchestrator):
                    workflow_status = instance.engine.get_workflow_status()
                    instance.metrics.custom_metrics['ai_agent'] = {
                        'workflow_status': workflow_status.get('status'),
                        'execution_history_count': workflow_status.get('execution_history_count', 0)
                    }
            
            # 通知回调
            await self._execute_callbacks(instance, 'on_metrics_update')
            
        except Exception as e:
            self.log_error(f"更新策略指标失败: {e}")
    
    async def _check_engine_health(self, instance: StrategyInstance) -> bool:
        """检查引擎健康状态"""
        try:
            if not instance.engine:
                return False
            
            if isinstance(instance.engine, HFTEngine):
                status = instance.engine.get_system_status()
                return status.get('running', False)
            elif MultiAgentOrchestrator and isinstance(instance.engine, MultiAgentOrchestrator):
                status = instance.engine.get_workflow_status()
                return status.get('status') not in ['failed', 'cancelled']
            
            return True
            
        except Exception as e:
            self.log_error(f"检查引擎健康状态失败: {e}")
            return False
    
    async def _check_resource_usage(self, instance: StrategyInstance) -> bool:
        """检查资源使用情况"""
        try:
            # 检查内存使用
            if instance.metrics.current_memory_mb > instance.config.max_memory_mb:
                self.log_warning(f"策略 {instance.config.strategy_id} 内存使用超限")
                return False
            
            # 检查CPU使用
            if instance.metrics.current_cpu_percent > instance.config.max_cpu_percent:
                self.log_warning(f"策略 {instance.config.strategy_id} CPU使用超限")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"检查资源使用失败: {e}")
            return False
    
    async def _check_business_metrics(self, instance: StrategyInstance) -> bool:
        """检查业务指标"""
        try:
            # 检查每日损失限制
            if (instance.config.max_daily_loss and 
                instance.metrics.daily_pnl < -abs(instance.config.max_daily_loss)):
                self.log_warning(f"策略 {instance.config.strategy_id} 超过每日最大损失限制")
                return False
            
            # 检查错误率
            if instance.metrics.total_trades > 0:
                error_rate = instance.metrics.error_count / instance.metrics.total_trades
                if error_rate > 0.1:  # 10%错误率阈值
                    self.log_warning(f"策略 {instance.config.strategy_id} 错误率过高: {error_rate:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(f"检查业务指标失败: {e}")
            return True  # 业务指标检查失败时不影响健康状态
    
    async def _execute_callbacks(self, instance: StrategyInstance, event: str, **kwargs):
        """执行回调函数"""
        try:
            callbacks = instance.callbacks.get(event, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(instance, **kwargs)
                    else:
                        callback(instance, **kwargs)
                except Exception as e:
                    self.log_error(f"回调执行失败: {e}")
        except Exception as e:
            self.log_error(f"执行回调失败: {e}")
    
    def _get_strategy_metrics_dict(self, metrics: StrategyMetrics) -> Dict[str, Any]:
        """获取策略指标字典"""
        return {
            'status': metrics.status.value,
            'start_time': metrics.start_time.isoformat() if metrics.start_time else None,
            'last_update_time': metrics.last_update_time.isoformat() if metrics.last_update_time else None,
            'uptime_seconds': metrics.uptime_seconds,
            'restart_count': metrics.restart_count,
            'error_count': metrics.error_count,
            'current_memory_mb': metrics.current_memory_mb,
            'max_memory_mb': metrics.max_memory_mb,
            'current_cpu_percent': metrics.current_cpu_percent,
            'max_cpu_percent': metrics.max_cpu_percent,
            'network_connections': metrics.network_connections,
            'total_trades': metrics.total_trades,
            'successful_trades': metrics.successful_trades,
            'total_pnl': float(metrics.total_pnl),
            'daily_pnl': float(metrics.daily_pnl),
            'custom_metrics': metrics.custom_metrics
        }
    
    def _get_strategy_config_dict(self, config: StrategyConfig) -> Dict[str, Any]:
        """获取策略配置字典"""
        return {
            'strategy_id': config.strategy_id,
            'strategy_type': config.strategy_type.value,
            'name': config.name,
            'description': config.description,
            'max_memory_mb': config.max_memory_mb,
            'max_cpu_percent': config.max_cpu_percent,
            'max_network_connections': config.max_network_connections,
            'priority': config.priority,
            'auto_restart': config.auto_restart,
            'max_restarts': config.max_restarts,
            'restart_delay_seconds': config.restart_delay_seconds,
            'health_check_interval': config.health_check_interval
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'active_connections': len(psutil.net_connections()),
                'processes_count': len(psutil.pids())
            }
        except Exception as e:
            self.log_error(f"获取系统指标失败: {e}")
            return {}