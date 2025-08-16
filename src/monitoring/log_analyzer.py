"""日志分析器"""

import asyncio
import re
import os
from typing import Dict, List, Any, Optional, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import json

from ..utils.logger import get_logger
from ..core.config_manager import config_manager

logger = get_logger(__name__)


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: str
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class LogPattern:
    """日志模式"""
    name: str
    pattern: Pattern
    level: str
    description: str
    action: Optional[str] = None  # 匹配时的动作


@dataclass
class LogStats:
    """日志统计"""
    total_count: int
    level_counts: Dict[str, int]
    error_rate: float
    warning_rate: float
    recent_errors: List[LogEntry]
    recent_warnings: List[LogEntry]
    pattern_matches: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_count': self.total_count,
            'level_counts': self.level_counts,
            'error_rate': self.error_rate,
            'warning_rate': self.warning_rate,
            'recent_errors': [e.to_dict() for e in self.recent_errors],
            'recent_warnings': [w.to_dict() for w in self.recent_warnings],
            'pattern_matches': self.pattern_matches
        }


class LogAnalyzer:
    """日志分析器 - 实时分析日志并检测异常模式"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.log_entries: deque = deque(maxlen=max_entries)
        self.patterns: List[LogPattern] = []
        self.pattern_matches: Dict[str, int] = defaultdict(int)
        
        self._analyzing = False
        self._analysis_task: Optional[asyncio.Task] = None
        self._file_watchers: Dict[str, asyncio.Task] = {}
        
        # 统计计数器
        self.level_counts: Dict[str, int] = defaultdict(int)
        self.hourly_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # 设置默认模式
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """设置默认日志模式"""
        patterns = [
            LogPattern(
                name="database_error",
                pattern=re.compile(r"database.*error|connection.*failed|sql.*error", re.IGNORECASE),
                level="ERROR",
                description="数据库错误",
                action="alert"
            ),
            LogPattern(
                name="api_error",
                pattern=re.compile(r"api.*error|http.*[45]\d\d|request.*failed", re.IGNORECASE),
                level="ERROR",
                description="API错误",
                action="alert"
            ),
            LogPattern(
                name="trading_error",
                pattern=re.compile(r"trading.*error|order.*failed|execution.*error", re.IGNORECASE),
                level="ERROR",
                description="交易错误",
                action="alert"
            ),
            LogPattern(
                name="memory_warning",
                pattern=re.compile(r"memory.*warning|out.*of.*memory|memory.*usage", re.IGNORECASE),
                level="WARNING",
                description="内存警告",
                action="monitor"
            ),
            LogPattern(
                name="performance_warning",
                pattern=re.compile(r"slow.*query|timeout|performance.*degraded", re.IGNORECASE),
                level="WARNING",
                description="性能警告",
                action="monitor"
            ),
            LogPattern(
                name="security_alert",
                pattern=re.compile(r"unauthorized|authentication.*failed|security.*breach", re.IGNORECASE),
                level="CRITICAL",
                description="安全警报",
                action="immediate_alert"
            )
        ]
        
        for pattern in patterns:
            self.add_pattern(pattern)
    
    def add_pattern(self, pattern: LogPattern):
        """添加日志模式"""
        self.patterns.append(pattern)
        logger.debug(f"添加日志模式: {pattern.name}")
    
    def remove_pattern(self, pattern_name: str):
        """移除日志模式"""
        self.patterns = [p for p in self.patterns if p.name != pattern_name]
        logger.debug(f"移除日志模式: {pattern_name}")
    
    def add_log_entry(self, timestamp: datetime, level: str, message: str, 
                     source: str = "system", metadata: Dict[str, Any] = None):
        """添加日志条目"""
        entry = LogEntry(
            timestamp=timestamp,
            level=level.upper(),
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self.log_entries.append(entry)
        self.level_counts[level.upper()] += 1
        
        # 按小时统计
        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
        self.hourly_counts[hour_key][level.upper()] += 1
        
        # 分析模式匹配
        self._analyze_entry(entry)
    
    def _analyze_entry(self, entry: LogEntry):
        """分析单个日志条目"""
        for pattern in self.patterns:
            if pattern.pattern.search(entry.message):
                self.pattern_matches[pattern.name] += 1
                
                # 执行模式动作
                if pattern.action:
                    self._execute_pattern_action(pattern, entry)
                
                logger.debug(f"日志模式匹配: {pattern.name} - {entry.message[:100]}")
    
    def _execute_pattern_action(self, pattern: LogPattern, entry: LogEntry):
        """执行模式动作"""
        if pattern.action == "alert":
            # 发送普通告警
            asyncio.create_task(self._send_log_alert(pattern, entry, "warning"))
        elif pattern.action == "immediate_alert":
            # 发送紧急告警
            asyncio.create_task(self._send_log_alert(pattern, entry, "critical"))
        elif pattern.action == "monitor":
            # 记录到监控系统
            logger.info(f"日志监控触发: {pattern.name} - {entry.message}")
    
    async def _send_log_alert(self, pattern: LogPattern, entry: LogEntry, severity: str):
        """发送日志告警"""
        try:
            from .alert_manager import alert_manager, AlertSeverity
            
            severity_map = {
                "info": AlertSeverity.INFO,
                "warning": AlertSeverity.WARNING,
                "error": AlertSeverity.ERROR,
                "critical": AlertSeverity.CRITICAL
            }
            
            await alert_manager.create_manual_alert(
                title=f"日志模式检测: {pattern.name}",
                message=f"检测到日志模式 '{pattern.name}': {entry.message}",
                severity=severity_map.get(severity, AlertSeverity.WARNING),
                source="log_analyzer",
                metadata={
                    "pattern_name": pattern.name,
                    "log_level": entry.level,
                    "log_source": entry.source,
                    "pattern_description": pattern.description
                }
            )
        except Exception as e:
            logger.error(f"发送日志告警失败: {e}")
    
    def start_analyzing(self, log_files: List[str] = None):
        """开始分析"""
        if self._analyzing:
            return
        
        self._analyzing = True
        
        # 启动文件监听
        if log_files:
            for log_file in log_files:
                self._file_watchers[log_file] = asyncio.create_task(
                    self._watch_log_file(log_file)
                )
        
        # 启动分析任务
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("日志分析器已启动")
    
    async def stop_analyzing(self):
        """停止分析"""
        if not self._analyzing:
            return
        
        self._analyzing = False
        
        # 停止文件监听
        for task in self._file_watchers.values():
            task.cancel()
        
        if self._file_watchers:
            await asyncio.gather(*self._file_watchers.values(), return_exceptions=True)
        
        self._file_watchers.clear()
        
        # 停止分析任务
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("日志分析器已停止")
    
    async def _watch_log_file(self, log_file: str):
        """监听日志文件"""
        try:
            file_path = Path(log_file)
            if not file_path.exists():
                logger.warning(f"日志文件不存在: {log_file}")
                return
            
            # 获取文件初始大小
            last_size = file_path.stat().st_size
            
            while self._analyzing:
                try:
                    current_size = file_path.stat().st_size
                    
                    if current_size > last_size:
                        # 文件有新内容
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_lines = f.readlines()
                        
                        for line in new_lines:
                            self._parse_log_line(line.strip(), log_file)
                        
                        last_size = current_size
                    
                    await asyncio.sleep(1)  # 每秒检查一次
                    
                except FileNotFoundError:
                    logger.warning(f"日志文件被删除: {log_file}")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"监听日志文件 {log_file} 时出错: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"监听日志文件任务异常: {e}")
    
    def _parse_log_line(self, line: str, source: str):
        """解析日志行"""
        if not line:
            return
        
        try:
            # 尝试解析不同的日志格式
            # 格式1: 2024-01-01 12:00:00 [INFO] message
            match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\[(\w+)\]\s*(.*)', line)
            if match:
                timestamp_str, level, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                self.add_log_entry(timestamp, level, message, source)
                return
            
            # 格式2: INFO:2024-01-01 12:00:00:message
            match = re.match(r'(\w+):(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}):(.*)', line)
            if match:
                level, timestamp_str, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                self.add_log_entry(timestamp, level, message, source)
                return
            
            # 格式3: JSON格式
            if line.startswith('{') and line.endswith('}'):
                try:
                    log_data = json.loads(line)
                    timestamp = datetime.fromisoformat(log_data.get('timestamp', datetime.now().isoformat()))
                    level = log_data.get('level', 'INFO')
                    message = log_data.get('message', line)
                    metadata = {k: v for k, v in log_data.items() 
                              if k not in ['timestamp', 'level', 'message']}
                    self.add_log_entry(timestamp, level, message, source, metadata)
                    return
                except json.JSONDecodeError:
                    pass
            
            # 默认处理：当作普通消息
            self.add_log_entry(datetime.now(), "INFO", line, source)
            
        except Exception as e:
            logger.debug(f"解析日志行失败: {e} - {line}")
    
    async def _analysis_loop(self):
        """分析循环"""
        while self._analyzing:
            try:
                # 定期清理统计数据
                await asyncio.sleep(300)  # 5分钟执行一次
                self._cleanup_old_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"日志分析循环异常: {e}")
    
    def _cleanup_old_stats(self):
        """清理旧的统计数据"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_hour = cutoff_time.strftime("%Y-%m-%d %H:00")
        
        # 清理小时统计
        keys_to_remove = [
            key for key in self.hourly_counts.keys()
            if key < cutoff_hour
        ]
        
        for key in keys_to_remove:
            del self.hourly_counts[key]
    
    def get_log_stats(self, hours: int = 24) -> LogStats:
        """获取日志统计"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 获取指定时间范围内的日志
        recent_logs = [
            entry for entry in self.log_entries
            if entry.timestamp >= cutoff_time
        ]
        
        # 计算统计
        level_counts = defaultdict(int)
        for entry in recent_logs:
            level_counts[entry.level] += 1
        
        total_count = len(recent_logs)
        error_count = level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)
        warning_count = level_counts.get('WARNING', 0)
        
        error_rate = (error_count / total_count * 100) if total_count > 0 else 0
        warning_rate = (warning_count / total_count * 100) if total_count > 0 else 0
        
        # 获取最近的错误和警告
        recent_errors = [
            entry for entry in recent_logs
            if entry.level in ['ERROR', 'CRITICAL']
        ][-10:]  # 最近10个错误
        
        recent_warnings = [
            entry for entry in recent_logs
            if entry.level == 'WARNING'
        ][-10:]  # 最近10个警告
        
        return LogStats(
            total_count=total_count,
            level_counts=dict(level_counts),
            error_rate=error_rate,
            warning_rate=warning_rate,
            recent_errors=recent_errors,
            recent_warnings=recent_warnings,
            pattern_matches=dict(self.pattern_matches)
        )
    
    def get_hourly_trends(self, hours: int = 24) -> Dict[str, Dict[str, int]]:
        """获取小时趋势"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_hour = cutoff_time.strftime("%Y-%m-%d %H:00")
        
        return {
            hour: counts for hour, counts in self.hourly_counts.items()
            if hour >= cutoff_hour
        }
    
    def search_logs(self, query: str, level: str = None, 
                   hours: int = 24, limit: int = 100) -> List[LogEntry]:
        """搜索日志"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        pattern = re.compile(query, re.IGNORECASE)
        
        results = []
        for entry in reversed(self.log_entries):
            if entry.timestamp < cutoff_time:
                break
            
            if level and entry.level != level.upper():
                continue
            
            if pattern.search(entry.message):
                results.append(entry)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def export_logs(self, format: str = 'json', hours: int = 24) -> str:
        """导出日志"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_logs = [
            entry for entry in self.log_entries
            if entry.timestamp >= cutoff_time
        ]
        
        if format.lower() == 'json':
            data = {
                'export_time': datetime.now().isoformat(),
                'stats': self.get_log_stats(hours).to_dict(),
                'logs': [entry.to_dict() for entry in recent_logs]
            }
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif format.lower() == 'csv':
            lines = ['timestamp,level,source,message']
            for entry in recent_logs:
                line = f'"{entry.timestamp.isoformat()}","{entry.level}","{entry.source}","{entry.message}"'
                lines.append(line)
            return '\n'.join(lines)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局日志分析器实例
log_analyzer = LogAnalyzer()