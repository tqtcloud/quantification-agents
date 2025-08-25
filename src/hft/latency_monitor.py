"""
高频交易延迟监控系统

实现数据新鲜度检查、备用数据源切换、延迟性能监控和告警、性能指标收集等功能。
适用于高频交易环境中的低延迟要求。
"""

import asyncio
import time
import statistics
from enum import Enum
from typing import Dict, List, Optional, Callable, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from decimal import Decimal

from src.core.models import MarketData
from src.utils.logger import LoggerMixin


class DataSourceStatus(Enum):
    """数据源状态"""
    ACTIVE = "active"
    INACTIVE = "inactive" 
    DEGRADED = "degraded"
    FAILED = "failed"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    priority: int  # 优先级，数字越小优先级越高
    max_latency_ms: float = 100.0  # 最大允许延迟（毫秒）
    timeout_ms: float = 5000.0  # 超时时间（毫秒）
    retry_count: int = 3  # 重试次数
    health_check_interval: float = 30.0  # 健康检查间隔（秒）


@dataclass 
class LatencyMetrics:
    """延迟指标"""
    symbol: str
    timestamp: float
    data_timestamp: float
    processing_latency_ms: float  # 处理延迟
    network_latency_ms: float  # 网络延迟
    total_latency_ms: float  # 总延迟
    data_source: str
    is_stale: bool = False


@dataclass
class PerformanceStats:
    """性能统计"""
    symbol: str
    data_source: str
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    stale_data_count: int = 0
    total_samples: int = 0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class AlertEvent:
    """告警事件"""
    level: AlertLevel
    message: str
    symbol: str
    data_source: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LatencyMonitor(LoggerMixin):
    """
    延迟监控系统
    
    核心功能：
    1. 数据新鲜度检查机制 - 检查市场数据是否超过延迟阈值
    2. 备用数据源切换逻辑 - 当延迟超标时自动切换
    3. 延迟性能监控和告警 - 持续监控系统延迟并发送告警
    4. 性能指标收集 - 统计平均延迟、99%分位数等关键指标
    """
    
    def __init__(self, 
                 staleness_threshold_ms: float = 100.0,
                 stats_window_size: int = 1000,
                 alert_cooldown_seconds: float = 60.0):
        """
        初始化延迟监控器
        
        Args:
            staleness_threshold_ms: 数据过期阈值（毫秒）
            stats_window_size: 统计窗口大小
            alert_cooldown_seconds: 告警冷却时间
        """
        self.staleness_threshold_ms = staleness_threshold_ms
        self.stats_window_size = stats_window_size
        self.alert_cooldown_seconds = alert_cooldown_seconds
        
        # 数据源管理
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.data_source_status: Dict[str, DataSourceStatus] = {}
        self.active_sources: Dict[str, str] = {}  # symbol -> active_data_source
        
        # 延迟统计
        self.latency_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=stats_window_size)
        )  # (symbol, data_source) -> latency_metrics
        self.performance_stats: Dict[Tuple[str, str], PerformanceStats] = {}
        
        # 告警管理
        self.alert_callbacks: List[Callable[[AlertEvent], None]] = []
        self.last_alert_time: Dict[Tuple[str, str], float] = {}  # 告警冷却
        
        # 状态管理
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 性能监控
        self.total_checks = 0
        self.stale_detections = 0
        self.source_switches = 0
        
    async def initialize(self, data_sources: List[DataSourceConfig]):
        """
        初始化数据源配置
        
        Args:
            data_sources: 数据源配置列表
        """
        for source in data_sources:
            self.data_sources[source.name] = source
            self.data_source_status[source.name] = DataSourceStatus.ACTIVE
        
        # 按优先级排序
        self._sorted_sources = sorted(
            data_sources, 
            key=lambda x: x.priority
        )
        
        self.log_info(
            f"Latency monitor initialized with {len(data_sources)} data sources",
            sources=[s.name for s in data_sources]
        )
    
    async def start(self):
        """启动延迟监控"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        self.log_info("Latency monitor started")
    
    async def stop(self):
        """停止延迟监控"""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.log_info("Latency monitor stopped")
    
    async def check_data_freshness(self, 
                                  symbol: str, 
                                  market_data: MarketData,
                                  data_source: str) -> Tuple[bool, LatencyMetrics]:
        """
        检查数据新鲜度
        
        Args:
            symbol: 交易对
            market_data: 市场数据
            data_source: 数据源名称
            
        Returns:
            (is_fresh, latency_metrics): 是否新鲜和延迟指标
        """
        current_time = time.time() * 1000  # 转换为毫秒
        data_timestamp = market_data.timestamp
        
        # 计算各种延迟
        total_latency = current_time - data_timestamp
        network_latency = self._estimate_network_latency(data_source)
        processing_latency = total_latency - network_latency
        
        # 创建延迟指标
        metrics = LatencyMetrics(
            symbol=symbol,
            timestamp=current_time,
            data_timestamp=data_timestamp,
            processing_latency_ms=processing_latency,
            network_latency_ms=network_latency,
            total_latency_ms=total_latency,
            data_source=data_source,
            is_stale=total_latency > self.staleness_threshold_ms
        )
        
        # 更新统计
        await self._update_statistics(metrics)
        
        # 检查是否需要告警
        if metrics.is_stale:
            await self._trigger_alert(
                AlertLevel.WARNING,
                f"Stale data detected for {symbol}",
                symbol,
                data_source,
                total_latency,
                {"threshold_ms": self.staleness_threshold_ms}
            )
        
        self.total_checks += 1
        if metrics.is_stale:
            self.stale_detections += 1
        
        is_fresh = not metrics.is_stale
        return is_fresh, metrics
    
    async def switch_data_source(self, symbol: str, reason: str) -> Optional[str]:
        """
        切换到备用数据源
        
        Args:
            symbol: 交易对
            reason: 切换原因
            
        Returns:
            new_data_source: 新的数据源名称，None表示无可用源
        """
        current_source = self.active_sources.get(symbol)
        
        # 寻找最佳备用数据源
        best_source = None
        for source in self._sorted_sources:
            if (source.name != current_source and 
                self.data_source_status[source.name] == DataSourceStatus.ACTIVE):
                best_source = source.name
                break
        
        if best_source is None:
            await self._trigger_alert(
                AlertLevel.CRITICAL,
                f"No backup data source available for {symbol}",
                symbol,
                current_source or "unknown",
                0,
                {"reason": reason}
            )
            return None
        
        # 执行切换
        old_source = self.active_sources.get(symbol, "unknown")
        self.active_sources[symbol] = best_source
        self.source_switches += 1
        
        await self._trigger_alert(
            AlertLevel.WARNING,
            f"Data source switched for {symbol}: {old_source} -> {best_source}",
            symbol,
            best_source,
            0,
            {"reason": reason, "old_source": old_source}
        )
        
        self.log_info(
            f"Data source switched for {symbol}",
            old_source=old_source,
            new_source=best_source,
            reason=reason
        )
        
        return best_source
    
    def get_active_data_source(self, symbol: str) -> Optional[str]:
        """获取当前活跃的数据源"""
        return self.active_sources.get(symbol)
    
    def set_active_data_source(self, symbol: str, data_source: str):
        """设置活跃的数据源"""
        if data_source in self.data_sources:
            self.active_sources[symbol] = data_source
        else:
            self.log_error(f"Unknown data source: {data_source}")
    
    def get_performance_stats(self, 
                            symbol: Optional[str] = None,
                            data_source: Optional[str] = None) -> Dict[str, PerformanceStats]:
        """
        获取性能统计
        
        Args:
            symbol: 可选的交易对过滤
            data_source: 可选的数据源过滤
            
        Returns:
            性能统计字典
        """
        result = {}
        
        for (s, ds), stats in self.performance_stats.items():
            if symbol and s != symbol:
                continue
            if data_source and ds != data_source:
                continue
            
            key = f"{s}_{ds}"
            result[key] = stats
            
        return result
    
    def get_latency_summary(self, symbol: str) -> Dict[str, float]:
        """获取延迟摘要"""
        active_source = self.active_sources.get(symbol)
        if not active_source:
            return {}
        
        key = (symbol, active_source)
        stats = self.performance_stats.get(key)
        
        if not stats:
            return {}
            
        return {
            "avg_latency_ms": stats.avg_latency_ms,
            "p95_latency_ms": stats.p95_latency_ms,
            "p99_latency_ms": stats.p99_latency_ms,
            "max_latency_ms": stats.max_latency_ms,
            "stale_data_rate": stats.stale_data_count / max(stats.total_samples, 1),
            "error_rate": stats.error_count / max(stats.total_samples, 1)
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        return {
            "running": self._running,
            "total_checks": self.total_checks,
            "stale_detection_rate": self.stale_detections / max(self.total_checks, 1),
            "source_switches": self.source_switches,
            "data_sources": {
                name: status.value 
                for name, status in self.data_source_status.items()
            },
            "active_sources": dict(self.active_sources)
        }
    
    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    async def update_data_source_status(self, 
                                      data_source: str, 
                                      status: DataSourceStatus):
        """更新数据源状态"""
        if data_source not in self.data_sources:
            self.log_error(f"Unknown data source: {data_source}")
            return
        
        old_status = self.data_source_status[data_source]
        self.data_source_status[data_source] = status
        
        if old_status != status:
            self.log_info(
                f"Data source status changed: {data_source}",
                old_status=old_status.value,
                new_status=status.value
            )
            
            # 如果数据源失效，触发相关symbol的切换
            if status == DataSourceStatus.FAILED:
                await self._handle_source_failure(data_source)
    
    async def _update_statistics(self, metrics: LatencyMetrics):
        """更新统计信息"""
        key = (metrics.symbol, metrics.data_source)
        
        # 添加到历史记录
        self.latency_history[key].append(metrics)
        
        # 更新性能统计
        if key not in self.performance_stats:
            self.performance_stats[key] = PerformanceStats(
                symbol=metrics.symbol,
                data_source=metrics.data_source
            )
        
        stats = self.performance_stats[key]
        history = list(self.latency_history[key])
        
        if history:
            latencies = [m.total_latency_ms for m in history]
            
            stats.avg_latency_ms = statistics.mean(latencies)
            stats.min_latency_ms = min(latencies)
            stats.max_latency_ms = max(latencies)
            
            # 计算分位数
            if len(latencies) >= 20:  # 至少需要20个样本计算分位数
                sorted_latencies = sorted(latencies)
                stats.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                stats.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            stats.stale_data_count = sum(1 for m in history if m.is_stale)
            stats.total_samples = len(history)
            stats.last_update = time.time()
    
    def _estimate_network_latency(self, data_source: str) -> float:
        """估算网络延迟（简化实现）"""
        # 这里可以实现更复杂的网络延迟估算逻辑
        # 比如ping测试、历史网络延迟统计等
        source_config = self.data_sources.get(data_source)
        if source_config:
            return source_config.max_latency_ms * 0.3  # 假设网络延迟是最大延迟的30%
        return 10.0  # 默认10ms网络延迟
    
    async def _trigger_alert(self, 
                           level: AlertLevel,
                           message: str,
                           symbol: str,
                           data_source: str,
                           latency_ms: float,
                           metadata: Optional[Dict] = None):
        """触发告警"""
        # 检查告警冷却
        key = (symbol, data_source)
        current_time = time.time()
        
        if (key in self.last_alert_time and 
            current_time - self.last_alert_time[key] < self.alert_cooldown_seconds):
            return
        
        self.last_alert_time[key] = current_time
        
        # 创建告警事件
        alert = AlertEvent(
            level=level,
            message=message,
            symbol=symbol,
            data_source=data_source,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        # 记录日志
        if level == AlertLevel.CRITICAL:
            self.log_error(message, **alert.metadata)
        elif level == AlertLevel.ERROR:
            self.log_error(message, **alert.metadata)
        elif level == AlertLevel.WARNING:
            self.log_warning(message, **alert.metadata)
        else:
            self.log_info(message, **alert.metadata)
        
        # 通知回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.log_error(f"Error in alert callback: {e}")
    
    async def _handle_source_failure(self, failed_source: str):
        """处理数据源失效"""
        affected_symbols = [
            symbol for symbol, source in self.active_sources.items() 
            if source == failed_source
        ]
        
        for symbol in affected_symbols:
            await self.switch_data_source(
                symbol, 
                f"Data source failure: {failed_source}"
            )
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._health_check()
                await asyncio.sleep(30)  # 每30秒执行一次健康检查
            except Exception as e:
                self.log_error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check(self):
        """健康检查"""
        for source_name, config in self.data_sources.items():
            try:
                # 这里可以实现具体的健康检查逻辑
                # 比如ping测试、连接测试等
                
                # 检查性能统计，判断数据源健康状况
                source_stats = [
                    stats for (symbol, ds), stats in self.performance_stats.items()
                    if ds == source_name and 
                       time.time() - stats.last_update < 300  # 5分钟内有更新
                ]
                
                if source_stats:
                    avg_error_rate = sum(
                        s.error_count / max(s.total_samples, 1) for s in source_stats
                    ) / len(source_stats)
                    
                    if avg_error_rate > 0.1:  # 错误率超过10%
                        await self.update_data_source_status(
                            source_name, 
                            DataSourceStatus.DEGRADED
                        )
                    elif avg_error_rate > 0.5:  # 错误率超过50%
                        await self.update_data_source_status(
                            source_name, 
                            DataSourceStatus.FAILED
                        )
                    else:
                        await self.update_data_source_status(
                            source_name, 
                            DataSourceStatus.ACTIVE
                        )
                        
            except Exception as e:
                self.log_error(f"Health check failed for {source_name}: {e}")
                await self.update_data_source_status(
                    source_name, 
                    DataSourceStatus.FAILED
                )