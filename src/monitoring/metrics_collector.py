"""指标收集器"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """指标点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


class MetricsCollector:
    """指标收集器 - 收集和聚合自定义业务指标"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=retention_hours * 60))  # 每分钟一个点
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        self._collecting = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """记录计数器指标"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=self.counters[key],
            labels=labels or {}
        )
        self.metrics[key].append(metric_point)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """设置仪表指标"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        self.metrics[key].append(metric_point)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录直方图指标"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        # 保持直方图大小
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-500:]
        
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        self.metrics[key].append(metric_point)
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """生成指标键"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取计数器值"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取仪表值"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """获取直方图统计"""
        key = self._make_key(name, labels)
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        values = sorted(values)
        count = len(values)
        
        return {
            'count': count,
            'sum': sum(values),
            'min': values[0],
            'max': values[-1],
            'mean': sum(values) / count,
            'p50': values[count // 2],
            'p90': values[int(count * 0.9)],
            'p95': values[int(count * 0.95)],
            'p99': values[int(count * 0.99)]
        }
    
    def get_metric_history(self, name: str, labels: Dict[str, str] = None, 
                          hours: int = 1) -> List[MetricPoint]:
        """获取指标历史"""
        key = self._make_key(name, labels)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            point for point in self.metrics.get(key, [])
            if point.timestamp >= cutoff_time
        ]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {
                key: self.get_histogram_stats(key.split('{')[0], 
                    self._parse_labels(key) if '{' in key else None)
                for key in self.histograms.keys()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _parse_labels(self, key: str) -> Dict[str, str]:
        """从键中解析标签"""
        if '{' not in key:
            return {}
        
        label_part = key.split('{')[1].rstrip('}')
        labels = {}
        
        for label_pair in label_part.split(','):
            if '=' in label_pair:
                k, v = label_pair.split('=', 1)
                labels[k] = v
        
        return labels
    
    def start_collecting(self):
        """开始收集"""
        if self._collecting:
            return
        
        self._collecting = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("指标收集器已启动")
    
    async def stop_collecting(self):
        """停止收集"""
        if not self._collecting:
            return
        
        self._collecting = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("指标收集器已停止")
    
    async def _cleanup_loop(self):
        """清理循环 - 定期清理过期指标"""
        while self._collecting:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                self._cleanup_expired_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"指标清理循环异常: {e}")
    
    def _cleanup_expired_metrics(self):
        """清理过期指标"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        cleaned_count = 0
        
        for key, metric_queue in self.metrics.items():
            original_size = len(metric_queue)
            
            # 移除过期的指标点
            while metric_queue and metric_queue[0].timestamp < cutoff_time:
                metric_queue.popleft()
            
            cleaned_count += original_size - len(metric_queue)
        
        if cleaned_count > 0:
            logger.debug(f"清理了 {cleaned_count} 个过期指标点")
    
    def export_metrics(self, format: str = 'json') -> str:
        """导出指标"""
        if format.lower() == 'json':
            return json.dumps(self.get_all_metrics(), indent=2, ensure_ascii=False)
        elif format.lower() == 'prometheus':
            return self._export_prometheus()
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_prometheus(self) -> str:
        """导出Prometheus格式"""
        lines = []
        
        # 导出计数器
        for key, value in self.counters.items():
            name, labels = self._split_key(key)
            label_str = self._format_prometheus_labels(labels)
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{label_str} {value}")
        
        # 导出仪表
        for key, value in self.gauges.items():
            name, labels = self._split_key(key)
            label_str = self._format_prometheus_labels(labels)
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{label_str} {value}")
        
        # 导出直方图
        for key, values in self.histograms.items():
            name, labels = self._split_key(key)
            stats = self.get_histogram_stats(name, labels)
            label_str = self._format_prometheus_labels(labels)
            
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count{label_str} {stats.get('count', 0)}")
            lines.append(f"{name}_sum{label_str} {stats.get('sum', 0)}")
            
            for percentile in [50, 90, 95, 99]:
                p_key = f"p{percentile}"
                if p_key in stats:
                    lines.append(f"{name}{{quantile=\"0.{percentile:02d}\"{self._add_labels(labels)}} {stats[p_key]}")
        
        return '\n'.join(lines)
    
    def _split_key(self, key: str) -> tuple:
        """分割键为名称和标签"""
        if '{' in key:
            name = key.split('{')[0]
            labels = self._parse_labels(key)
            return name, labels
        return key, {}
    
    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """格式化Prometheus标签"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"
    
    def _add_labels(self, labels: Dict[str, str]) -> str:
        """添加标签到Prometheus格式"""
        if not labels:
            return ""
        
        label_pairs = [f',{k}="{v}"' for k, v in sorted(labels.items())]
        return "".join(label_pairs)


# 全局指标收集器实例
metrics_collector = MetricsCollector()