"""
策略监控器 (StrategyMonitor)
实现策略运行状态实时监控和健康检查系统
"""

import asyncio
import json
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from decimal import Decimal
import statistics
import logging

from src.utils.logger import LoggerMixin


class MonitoringLevel(Enum):
    """监控级别"""
    BASIC = "basic"           # 基础监控
    DETAILED = "detailed"     # 详细监控
    COMPREHENSIVE = "comprehensive"  # 全面监控


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"       # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"   # 直方图
    SUMMARY = "summary"      # 汇总


@dataclass
class MetricValue:
    """指标值"""
    value: Union[int, float, str]
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels
        }


@dataclass
class MonitoringMetrics:
    """监控指标"""
    strategy_id: str
    strategy_type: str
    
    # 基础运行指标
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None
    status_changes: List[str] = field(default_factory=list)
    error_count: int = 0
    restart_count: int = 0
    
    # 性能指标
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    network_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # 业务指标
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    daily_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # 延迟指标
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # 自定义指标
    custom_metrics: Dict[str, MetricValue] = field(default_factory=dict)
    
    # 统计计算
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_trades == 0:
            return 0.0
        return self.successful_trades / self.total_trades
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.total_trades == 0:
            return 0.0
        return self.failed_trades / self.total_trades
    
    def get_avg_cpu_usage(self) -> float:
        """获取平均CPU使用率"""
        if not self.cpu_usage:
            return 0.0
        return statistics.mean(self.cpu_usage)
    
    def get_avg_memory_usage(self) -> float:
        """获取平均内存使用"""
        if not self.memory_usage:
            return 0.0
        return statistics.mean(self.memory_usage)


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 条件表达式，如 "> 80", "< 0.9", "== ERROR"
    threshold: Union[int, float, str]
    alert_level: AlertLevel
    enabled: bool = True
    cooldown_seconds: int = 300  # 告警冷却时间
    last_alert_time: Optional[datetime] = None
    alert_count: int = 0


@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule_id: str
    strategy_id: str
    level: AlertLevel
    title: str
    message: str
    metric_value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'strategy_id': self.strategy_id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_time': self.resolved_time.isoformat() if self.resolved_time else None
        }


class StrategyMonitor(LoggerMixin):
    """
    策略监控器
    
    负责监控策略运行状态、性能指标、业务指标，
    并提供告警和统计分析功能
    """
    
    def __init__(self, strategy_manager, monitoring_level: MonitoringLevel = MonitoringLevel.DETAILED):
        """
        初始化策略监控器
        
        Args:
            strategy_manager: 策略管理器实例
            monitoring_level: 监控级别
        """
        self.strategy_manager = strategy_manager
        self.monitoring_level = monitoring_level
        
        # 监控数据
        self.metrics: Dict[str, MonitoringMetrics] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # 监控任务
        self._monitor_tasks: Dict[str, asyncio.Task] = {}
        self._is_monitoring: bool = False
        self._monitor_interval: int = 5  # 监控间隔（秒）
        
        # 告警回调
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 初始化默认告警规则
        self._initialize_default_alert_rules()
        
        self.log_info(f"策略监控器初始化完成，监控级别: {monitoring_level.value}")
    
    async def initialize(self):
        """初始化监控器"""
        try:
            # 启动监控
            await self.start_monitoring()
            self.log_info("策略监控器初始化完成")
        except Exception as e:
            self.log_error(f"策略监控器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭监控器"""
        try:
            await self.stop_monitoring()
            self.log_info("策略监控器已关闭")
        except Exception as e:
            self.log_error(f"关闭策略监控器失败: {e}")
    
    async def start_monitoring(self):
        """启动监控"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        
        # 为每个策略启动监控任务
        for strategy_id in self.strategy_manager.strategies:
            await self._start_strategy_monitoring(strategy_id)
        
        self.log_info("策略监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        
        # 停止所有监控任务
        for strategy_id, task in self._monitor_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._monitor_tasks.clear()
        self.log_info("策略监控已停止")
    
    async def add_strategy_monitoring(self, strategy_id: str):
        """添加策略监控"""
        if strategy_id in self.metrics:
            self.log_warning(f"策略 {strategy_id} 已在监控中")
            return
        
        # 获取策略信息
        if strategy_id not in self.strategy_manager.strategies:
            self.log_error(f"策略 {strategy_id} 不存在")
            return
        
        instance = self.strategy_manager.strategies[strategy_id]
        
        # 初始化监控指标
        self.metrics[strategy_id] = MonitoringMetrics(
            strategy_id=strategy_id,
            strategy_type=instance.config.strategy_type.value,
            last_heartbeat=datetime.now()
        )
        
        # 启动监控任务
        if self._is_monitoring:
            await self._start_strategy_monitoring(strategy_id)
        
        self.log_info(f"添加策略 {strategy_id} 监控")
    
    async def remove_strategy_monitoring(self, strategy_id: str):
        """移除策略监控"""
        # 停止监控任务
        if strategy_id in self._monitor_tasks:
            self._monitor_tasks[strategy_id].cancel()
            try:
                await self._monitor_tasks[strategy_id]
            except asyncio.CancelledError:
                pass
            del self._monitor_tasks[strategy_id]
        
        # 移除监控数据
        if strategy_id in self.metrics:
            del self.metrics[strategy_id]
        
        # 解决相关告警
        await self._resolve_strategy_alerts(strategy_id)
        
        self.log_info(f"移除策略 {strategy_id} 监控")
    
    def update_metric(self, strategy_id: str, metric_name: str, value: Any, labels: Optional[Dict[str, str]] = None):
        """
        更新指标
        
        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            value: 指标值
            labels: 标签
        """
        try:
            if strategy_id not in self.metrics:
                self.log_warning(f"策略 {strategy_id} 未在监控中")
                return
            
            metrics = self.metrics[strategy_id]
            
            # 更新心跳
            metrics.last_heartbeat = datetime.now()
            
            # 更新具体指标
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )
            
            if metric_name == 'cpu_usage':
                metrics.cpu_usage.append(float(value))
            elif metric_name == 'memory_usage':
                metrics.memory_usage.append(float(value))
            elif metric_name == 'network_usage':
                metrics.network_usage.append(int(value))
            elif metric_name == 'total_trades':
                metrics.total_trades = int(value)
            elif metric_name == 'successful_trades':
                metrics.successful_trades = int(value)
            elif metric_name == 'failed_trades':
                metrics.failed_trades = int(value)
            elif metric_name == 'total_pnl':
                metrics.total_pnl = Decimal(str(value))
            elif metric_name == 'daily_pnl':
                metrics.daily_pnl = Decimal(str(value))
            elif metric_name == 'avg_latency_ms':
                metrics.avg_latency_ms = float(value)
            elif metric_name == 'max_latency_ms':
                metrics.max_latency_ms = float(value)
            elif metric_name == 'error_count':
                metrics.error_count = int(value)
            elif metric_name == 'restart_count':
                metrics.restart_count = int(value)
            else:
                # 自定义指标
                metrics.custom_metrics[metric_name] = metric_value
            
            # 检查告警规则
            asyncio.create_task(self._check_alert_rules(strategy_id, metric_name, value))
            
        except Exception as e:
            self.log_error(f"更新指标失败: {e}")
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """获取策略指标"""
        if strategy_id not in self.metrics:
            return None
        
        metrics = self.metrics[strategy_id]
        
        return {
            'strategy_id': strategy_id,
            'strategy_type': metrics.strategy_type,
            'uptime_seconds': metrics.uptime_seconds,
            'last_heartbeat': metrics.last_heartbeat.isoformat() if metrics.last_heartbeat else None,
            'status_changes_count': len(metrics.status_changes),
            'error_count': metrics.error_count,
            'restart_count': metrics.restart_count,
            
            # 性能指标
            'avg_cpu_usage': metrics.get_avg_cpu_usage(),
            'avg_memory_usage': metrics.get_avg_memory_usage(),
            'current_cpu_usage': list(metrics.cpu_usage)[-1] if metrics.cpu_usage else 0.0,
            'current_memory_usage': list(metrics.memory_usage)[-1] if metrics.memory_usage else 0.0,
            'network_connections': list(metrics.network_usage)[-1] if metrics.network_usage else 0,
            
            # 业务指标
            'total_trades': metrics.total_trades,
            'successful_trades': metrics.successful_trades,
            'failed_trades': metrics.failed_trades,
            'success_rate': metrics.get_success_rate(),
            'error_rate': metrics.get_error_rate(),
            'total_pnl': float(metrics.total_pnl),
            'daily_pnl': float(metrics.daily_pnl),
            
            # 延迟指标
            'avg_latency_ms': metrics.avg_latency_ms,
            'max_latency_ms': metrics.max_latency_ms,
            'p95_latency_ms': metrics.p95_latency_ms,
            'p99_latency_ms': metrics.p99_latency_ms,
            
            # 自定义指标
            'custom_metrics': {
                name: metric.to_dict() 
                for name, metric in metrics.custom_metrics.items()
            }
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统级监控指标"""
        total_strategies = len(self.metrics)
        active_strategies = 0
        total_trades = 0
        total_errors = 0
        total_pnl = Decimal("0")
        
        for metrics in self.metrics.values():
            if metrics.last_heartbeat and (datetime.now() - metrics.last_heartbeat).seconds < 60:
                active_strategies += 1
            
            total_trades += metrics.total_trades
            total_errors += metrics.error_count
            total_pnl += metrics.total_pnl
        
        return {
            'total_strategies': total_strategies,
            'active_strategies': active_strategies,
            'inactive_strategies': total_strategies - active_strategies,
            'total_trades': total_trades,
            'total_errors': total_errors,
            'total_pnl': float(total_pnl),
            'active_alerts': len(self.active_alerts),
            'total_alert_history': len(self.alert_history),
            'monitoring_level': self.monitoring_level.value,
            'uptime_seconds': (datetime.now() - self.strategy_manager._start_time).total_seconds() if hasattr(self.strategy_manager, '_start_time') else 0
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.rule_id] = rule
        self.log_info(f"添加告警规则: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.log_info(f"移除告警规则: {rule_id}")
    
    def get_active_alerts(self, strategy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        alerts = []
        for alert in self.active_alerts.values():
            if strategy_id is None or alert.strategy_id == strategy_id:
                alerts.append(alert.to_dict())
        return alerts
    
    def get_alert_history(self, strategy_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警历史"""
        alerts = []
        count = 0
        
        # 按时间倒序
        for alert in sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True):
            if count >= limit:
                break
            
            if strategy_id is None or alert.strategy_id == strategy_id:
                alerts.append(alert.to_dict())
                count += 1
        
        return alerts
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """注册告警回调"""
        self.alert_callbacks.append(callback)
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_time = datetime.now()
            
            # 移到历史记录
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            self.log_info(f"告警已解决: {alert.title}")
            return True
            
        except Exception as e:
            self.log_error(f"解决告警失败: {e}")
            return False
    
    async def _start_strategy_monitoring(self, strategy_id: str):
        """启动单个策略监控"""
        if strategy_id in self._monitor_tasks:
            return
        
        self._monitor_tasks[strategy_id] = asyncio.create_task(
            self._strategy_monitor_loop(strategy_id)
        )
    
    async def _strategy_monitor_loop(self, strategy_id: str):
        """策略监控循环"""
        try:
            while self._is_monitoring:
                await self._collect_strategy_metrics(strategy_id)
                await asyncio.sleep(self._monitor_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_error(f"策略 {strategy_id} 监控循环错误: {e}")
    
    async def _collect_strategy_metrics(self, strategy_id: str):
        """收集策略指标"""
        try:
            # 检查策略是否还存在
            if strategy_id not in self.strategy_manager.strategies:
                await self.remove_strategy_monitoring(strategy_id)
                return
            
            instance = self.strategy_manager.strategies[strategy_id]
            metrics = self.metrics.get(strategy_id)
            
            if not metrics:
                return
            
            # 更新运行时间
            if instance.metrics.start_time:
                metrics.uptime_seconds = (datetime.now() - instance.metrics.start_time).total_seconds()
            
            # 从策略实例获取指标
            self.update_metric(strategy_id, 'cpu_usage', instance.metrics.current_cpu_percent)
            self.update_metric(strategy_id, 'memory_usage', instance.metrics.current_memory_mb)
            self.update_metric(strategy_id, 'network_usage', instance.metrics.network_connections)
            self.update_metric(strategy_id, 'total_trades', instance.metrics.total_trades)
            self.update_metric(strategy_id, 'successful_trades', instance.metrics.successful_trades)
            self.update_metric(strategy_id, 'error_count', instance.metrics.error_count)
            self.update_metric(strategy_id, 'total_pnl', float(instance.metrics.total_pnl))
            self.update_metric(strategy_id, 'daily_pnl', float(instance.metrics.daily_pnl))
            
            # 获取引擎特定指标
            if instance.engine:
                if hasattr(instance.engine, 'get_performance_metrics'):
                    engine_metrics = instance.engine.get_performance_metrics()
                    if hasattr(engine_metrics, 'avg_update_latency'):
                        self.update_metric(strategy_id, 'avg_latency_ms', engine_metrics.avg_update_latency)
                    if hasattr(engine_metrics, 'max_update_latency'):
                        self.update_metric(strategy_id, 'max_latency_ms', engine_metrics.max_update_latency)
            
        except Exception as e:
            self.log_error(f"收集策略 {strategy_id} 指标失败: {e}")
    
    async def _check_alert_rules(self, strategy_id: str, metric_name: str, value: Any):
        """检查告警规则"""
        try:
            current_time = datetime.now()
            
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                if rule.metric_name != metric_name:
                    continue
                
                # 检查冷却时间
                if (rule.last_alert_time and 
                    (current_time - rule.last_alert_time).total_seconds() < rule.cooldown_seconds):
                    continue
                
                # 评估条件
                if self._evaluate_condition(value, rule.condition, rule.threshold):
                    await self._trigger_alert(strategy_id, rule, value)
                    
        except Exception as e:
            self.log_error(f"检查告警规则失败: {e}")
    
    def _evaluate_condition(self, value: Any, condition: str, threshold: Any) -> bool:
        """评估条件"""
        try:
            if condition.startswith('>'):
                return float(value) > float(threshold)
            elif condition.startswith('<'):
                return float(value) < float(threshold)
            elif condition.startswith('=='):
                return str(value) == str(threshold)
            elif condition.startswith('!='):
                return str(value) != str(threshold)
            elif condition.startswith('>='):
                return float(value) >= float(threshold)
            elif condition.startswith('<='):
                return float(value) <= float(threshold)
            else:
                return False
        except Exception:
            return False
    
    async def _trigger_alert(self, strategy_id: str, rule: AlertRule, metric_value: Any):
        """触发告警"""
        try:
            # 生成告警ID
            alert_id = f"{rule.rule_id}_{strategy_id}_{int(time.time())}"
            
            # 创建告警
            alert = Alert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                strategy_id=strategy_id,
                level=rule.alert_level,
                title=f"{rule.name} - {strategy_id}",
                message=f"策略 {strategy_id} 的 {rule.metric_name} {rule.condition} {rule.threshold}，当前值: {metric_value}",
                metric_value=metric_value,
                threshold=rule.threshold
            )
            
            # 记录告警
            self.active_alerts[alert_id] = alert
            rule.last_alert_time = datetime.now()
            rule.alert_count += 1
            
            # 执行回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.log_error(f"告警回调执行失败: {e}")
            
            # 记录日志
            log_level = {
                AlertLevel.INFO: self.log_info,
                AlertLevel.WARNING: self.log_warning,
                AlertLevel.ERROR: self.log_error,
                AlertLevel.CRITICAL: self.log_error
            }
            log_level[rule.alert_level](f"触发告警: {alert.message}")
            
        except Exception as e:
            self.log_error(f"触发告警失败: {e}")
    
    async def _resolve_strategy_alerts(self, strategy_id: str):
        """解决策略相关的所有告警"""
        alert_ids_to_resolve = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.strategy_id == strategy_id
        ]
        
        for alert_id in alert_ids_to_resolve:
            await self.resolve_alert(alert_id)
    
    def _initialize_default_alert_rules(self):
        """初始化默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="CPU使用率过高",
                description="CPU使用率超过80%",
                metric_name="cpu_usage",
                condition="> 80",
                threshold=80.0,
                alert_level=AlertLevel.WARNING,
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="内存使用过高",
                description="内存使用超过1GB",
                metric_name="memory_usage",
                condition="> 1024",
                threshold=1024.0,
                alert_level=AlertLevel.WARNING,
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="high_error_rate",
                name="错误率过高",
                description="错误次数超过10次",
                metric_name="error_count",
                condition="> 10",
                threshold=10,
                alert_level=AlertLevel.ERROR,
                cooldown_seconds=600
            ),
            AlertRule(
                rule_id="high_latency",
                name="延迟过高",
                description="平均延迟超过100ms",
                metric_name="avg_latency_ms",
                condition="> 100",
                threshold=100.0,
                alert_level=AlertLevel.WARNING,
                cooldown_seconds=300
            ),
            AlertRule(
                rule_id="negative_daily_pnl",
                name="每日PnL为负",
                description="每日PnL小于-1000",
                metric_name="daily_pnl",
                condition="< -1000",
                threshold=-1000.0,
                alert_level=AlertLevel.ERROR,
                cooldown_seconds=3600
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)