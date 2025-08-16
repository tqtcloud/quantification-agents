"""系统性能监控器"""

import asyncio
import psutil
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

from ..utils.logger import get_logger
from ..core.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    disk_free: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_available': self.memory_available,
            'disk_usage': self.disk_usage,
            'disk_free': self.disk_free,
            'network_io': self.network_io,
            'process_count': self.process_count,
            'load_average': self.load_average
        }


@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: datetime
    trading_system_status: str
    active_agents: int
    active_strategies: int
    active_orders: int
    total_positions: int
    api_requests_per_minute: int
    websocket_connections: int
    database_connections: int
    error_count: int
    warning_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'trading_system_status': self.trading_system_status,
            'active_agents': self.active_agents,
            'active_strategies': self.active_strategies,
            'active_orders': self.active_orders,
            'total_positions': self.total_positions,
            'api_requests_per_minute': self.api_requests_per_minute,
            'websocket_connections': self.websocket_connections,
            'database_connections': self.database_connections,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }


class SystemMonitor:
    """系统性能监控器"""
    
    def __init__(self, metrics_retention_minutes: int = 1440):  # 24小时
        self.metrics_retention_minutes = metrics_retention_minutes
        self.system_metrics: deque = deque(maxlen=metrics_retention_minutes)
        self.app_metrics: deque = deque(maxlen=metrics_retention_minutes)
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 监控回调
        self.metric_callbacks: List[Callable] = []
        
        # 性能计数器
        self._last_network_io = None
        self._api_request_counter = 0
        self._error_counter = 0
        self._warning_counter = 0
        
        # 重置计数器的时间戳
        self._last_reset_time = datetime.now()
    
    def start_monitoring(self, interval_seconds: int = 60):
        """开始监控"""
        if self._monitoring:
            logger.warning("监控已在运行")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval_seconds))
        logger.info(f"开始系统监控，间隔 {interval_seconds} 秒")
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("系统监控已停止")
    
    async def _monitoring_loop(self, interval_seconds: int):
        """监控循环"""
        while self._monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                app_metrics = self._collect_application_metrics()
                
                # 存储指标
                with self._lock:
                    self.system_metrics.append(system_metrics)
                    self.app_metrics.append(app_metrics)
                
                # 调用回调
                await self._notify_callbacks(system_metrics, app_metrics)
                
                # 检查是否需要重置计数器
                self._check_reset_counters()
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024 * 1024 * 1024)  # GB
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        disk_free = disk.free / (1024 * 1024 * 1024)  # GB
        
        # 网络IO
        network_io = psutil.net_io_counters()
        network_data = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }
        
        # 进程数
        process_count = len(psutil.pids())
        
        # 负载平均值（Unix系统）
        load_average = []
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            # Windows系统不支持getloadavg
            pass
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            disk_free=disk_free,
            network_io=network_data,
            process_count=process_count,
            load_average=load_average
        )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """收集应用指标"""
        # 这里需要从实际的系统组件获取数据
        # 目前使用模拟数据
        
        return ApplicationMetrics(
            timestamp=datetime.now(),
            trading_system_status="running",
            active_agents=5,
            active_strategies=2,
            active_orders=10,
            total_positions=3,
            api_requests_per_minute=self._api_request_counter,
            websocket_connections=8,
            database_connections=5,
            error_count=self._error_counter,
            warning_count=self._warning_counter
        )
    
    def _check_reset_counters(self):
        """检查是否需要重置计数器"""
        now = datetime.now()
        if now - self._last_reset_time >= timedelta(minutes=1):
            self._api_request_counter = 0
            self._error_counter = 0
            self._warning_counter = 0
            self._last_reset_time = now
    
    async def _notify_callbacks(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """通知回调函数"""
        for callback in self.metric_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(system_metrics, app_metrics)
                else:
                    callback(system_metrics, app_metrics)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")
    
    def add_metric_callback(self, callback: Callable):
        """添加指标回调"""
        self.metric_callbacks.append(callback)
    
    def remove_metric_callback(self, callback: Callable):
        """移除指标回调"""
        if callback in self.metric_callbacks:
            self.metric_callbacks.remove(callback)
    
    def increment_api_requests(self, count: int = 1):
        """增加API请求计数"""
        self._api_request_counter += count
    
    def increment_errors(self, count: int = 1):
        """增加错误计数"""
        self._error_counter += count
    
    def increment_warnings(self, count: int = 1):
        """增加警告计数"""
        self._warning_counter += count
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        with self._lock:
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            latest_app = self.app_metrics[-1] if self.app_metrics else None
        
        return {
            'system': latest_system.to_dict() if latest_system else None,
            'application': latest_app.to_dict() if latest_app else None,
            'monitoring_status': self._monitoring
        }
    
    def get_metrics_history(self, minutes: int = 60) -> Dict[str, List[Dict]]:
        """获取历史指标"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            system_history = [
                m.to_dict() for m in self.system_metrics 
                if m.timestamp >= cutoff_time
            ]
            app_history = [
                m.to_dict() for m in self.app_metrics 
                if m.timestamp >= cutoff_time
            ]
        
        return {
            'system': system_history,
            'application': app_history
        }
    
    def get_system_health_score(self) -> float:
        """计算系统健康评分 (0-100)"""
        if not self.system_metrics or not self.app_metrics:
            return 0.0
        
        with self._lock:
            latest_system = self.system_metrics[-1]
            latest_app = self.app_metrics[-1]
        
        # 基础分数
        score = 100.0
        
        # CPU使用率影响 (权重: 25%)
        if latest_system.cpu_usage > 90:
            score -= 25
        elif latest_system.cpu_usage > 80:
            score -= 15
        elif latest_system.cpu_usage > 70:
            score -= 10
        
        # 内存使用率影响 (权重: 25%)
        if latest_system.memory_usage > 95:
            score -= 25
        elif latest_system.memory_usage > 85:
            score -= 15
        elif latest_system.memory_usage > 75:
            score -= 10
        
        # 磁盘使用率影响 (权重: 15%)
        if latest_system.disk_usage > 95:
            score -= 15
        elif latest_system.disk_usage > 90:
            score -= 10
        elif latest_system.disk_usage > 85:
            score -= 5
        
        # 应用错误率影响 (权重: 20%)
        error_rate = latest_app.error_count
        if error_rate > 10:
            score -= 20
        elif error_rate > 5:
            score -= 15
        elif error_rate > 2:
            score -= 10
        
        # 交易系统状态影响 (权重: 15%)
        if latest_app.trading_system_status != "running":
            score -= 15
        
        return max(0.0, min(100.0, score))
    
    def export_metrics(self, format: str = 'json') -> str:
        """导出指标数据"""
        data = {
            'export_time': datetime.now().isoformat(),
            'system_metrics': [m.to_dict() for m in self.system_metrics],
            'application_metrics': [m.to_dict() for m in self.app_metrics]
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局系统监控器实例
system_monitor = SystemMonitor()