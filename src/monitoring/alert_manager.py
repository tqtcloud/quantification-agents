"""告警管理器"""

import asyncio
import smtplib
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp

from ..utils.logger import get_logger
from ..core.config_manager import config_manager

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """告警信息"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metadata': self.metadata,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_note': self.resolution_note
        }


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: Callable
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class AlertChannel:
    """告警通道基类"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        raise NotImplementedError


class EmailAlertChannel(AlertChannel):
    """邮件告警通道"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
告警详情:
- 标题: {alert.title}
- 严重级别: {alert.severity.value}
- 来源: {alert.source}
- 时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 消息: {alert.message}

元数据:
{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 异步发送邮件
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email, msg
            )
            
            logger.info(f"邮件告警发送成功: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")
            return False
    
    def _send_email(self, msg):
        """同步发送邮件"""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)


class WebhookAlertChannel(AlertChannel):
    """Webhook告警通道"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook告警发送成功: {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook告警发送失败，状态码: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Webhook告警发送失败: {e}")
            return False


class LogAlertChannel(AlertChannel):
    """日志告警通道"""
    
    async def send_alert(self, alert: Alert) -> bool:
        """记录告警到日志"""
        try:
            log_message = f"[ALERT] {alert.severity.value.upper()} - {alert.title}: {alert.message}"
            
            if alert.severity == AlertSeverity.CRITICAL:
                logger.critical(log_message)
            elif alert.severity == AlertSeverity.ERROR:
                logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            return True
            
        except Exception as e:
            logger.error(f"日志告警记录失败: {e}")
            return False


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        
        # 默认添加日志告警通道
        self.add_channel(LogAlertChannel())
        
        # 设置默认告警规则
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认告警规则"""
        # CPU使用率告警
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda system_metrics, app_metrics: system_metrics.cpu_usage > 90,
            severity=AlertSeverity.WARNING,
            message_template="CPU使用率过高: {cpu_usage:.1f}%",
            cooldown_minutes=5
        ))
        
        # 内存使用率告警
        self.add_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda system_metrics, app_metrics: system_metrics.memory_usage > 85,
            severity=AlertSeverity.WARNING,
            message_template="内存使用率过高: {memory_usage:.1f}%",
            cooldown_minutes=5
        ))
        
        # 磁盘空间告警
        self.add_rule(AlertRule(
            name="low_disk_space",
            condition=lambda system_metrics, app_metrics: system_metrics.disk_usage > 90,
            severity=AlertSeverity.ERROR,
            message_template="磁盘空间不足: {disk_usage:.1f}%",
            cooldown_minutes=10
        ))
        
        # 错误率告警
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda system_metrics, app_metrics: app_metrics.error_count > 5,
            severity=AlertSeverity.ERROR,
            message_template="错误率过高: {error_count} 个错误/分钟",
            cooldown_minutes=3
        ))
        
        # 交易系统状态告警
        self.add_rule(AlertRule(
            name="trading_system_down",
            condition=lambda system_metrics, app_metrics: app_metrics.trading_system_status != "running",
            severity=AlertSeverity.CRITICAL,
            message_template="交易系统状态异常: {trading_system_status}",
            cooldown_minutes=1
        ))
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.name] = rule
        logger.debug(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.debug(f"移除告警规则: {rule_name}")
    
    def add_channel(self, channel: AlertChannel):
        """添加告警通道"""
        self.alert_channels.append(channel)
        logger.debug(f"添加告警通道: {type(channel).__name__}")
    
    def remove_channel(self, channel: AlertChannel):
        """移除告警通道"""
        if channel in self.alert_channels:
            self.alert_channels.remove(channel)
            logger.debug(f"移除告警通道: {type(channel).__name__}")
    
    async def check_alerts(self, system_metrics, app_metrics):
        """检查告警条件"""
        now = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # 检查冷却时间
            if (rule.last_triggered and 
                now - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                continue
            
            try:
                # 检查告警条件
                if rule.condition(system_metrics, app_metrics):
                    await self._trigger_alert(rule, system_metrics, app_metrics)
                    rule.last_triggered = now
                    
            except Exception as e:
                logger.error(f"检查告警规则 {rule_name} 时出错: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, system_metrics, app_metrics):
        """触发告警"""
        # 生成告警ID
        alert_id = f"{rule.name}_{int(datetime.now().timestamp())}"
        
        # 格式化消息
        context = {
            **system_metrics.to_dict(),
            **app_metrics.to_dict()
        }
        message = rule.message_template.format(**context)
        
        # 创建告警
        alert = Alert(
            id=alert_id,
            title=f"系统告警: {rule.name}",
            message=message,
            severity=rule.severity,
            source="system_monitor",
            timestamp=datetime.now(),
            metadata={
                'rule_name': rule.name,
                'system_metrics': system_metrics.to_dict(),
                'app_metrics': app_metrics.to_dict()
            }
        )
        
        # 存储告警
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 保持历史记录大小
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
        
        logger.warning(f"触发告警: {alert.title} - {alert.message}")
        
        # 发送告警
        await self._send_alert(alert)
    
    async def _send_alert(self, alert: Alert):
        """发送告警到所有通道"""
        send_tasks = []
        for channel in self.alert_channels:
            send_tasks.append(channel.send_alert(alert))
        
        if send_tasks:
            results = await asyncio.gather(*send_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.info(f"告警发送完成: {success_count}/{len(send_tasks)} 个通道成功")
    
    async def create_manual_alert(self, title: str, message: str, 
                                severity: AlertSeverity = AlertSeverity.INFO,
                                source: str = "manual", metadata: Dict = None):
        """手动创建告警"""
        alert_id = f"manual_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        await self._send_alert(alert)
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolution_note: str = None):
        """解决告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolution_note = resolution_note
            
            logger.info(f"告警已解决: {alert_id}")
            return True
        
        return False
    
    def suppress_alert(self, alert_id: str):
        """抑制告警"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            logger.info(f"告警已抑制: {alert_id}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [
            alert for alert in self.alerts.values() 
            if alert.status == AlertStatus.ACTIVE
        ]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_time
        ]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(hours=24)
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in recent_alerts if a.severity == severity
            ])
        
        return {
            'active_count': len(active_alerts),
            'total_alerts_24h': len(recent_alerts),
            'severity_breakdown_24h': severity_counts,
            'rules_count': len(self.alert_rules),
            'channels_count': len(self.alert_channels),
            'last_alert': recent_alerts[-1].timestamp.isoformat() if recent_alerts else None
        }
    
    def export_alerts(self, format: str = 'json') -> str:
        """导出告警数据"""
        data = {
            'export_time': datetime.now().isoformat(),
            'active_alerts': [alert.to_dict() for alert in self.get_active_alerts()],
            'alert_history': [alert.to_dict() for alert in self.alert_history],
            'alert_rules': {
                name: {
                    'name': rule.name,
                    'severity': rule.severity.value,
                    'message_template': rule.message_template,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'enabled': rule.enabled,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for name, rule in self.alert_rules.items()
            }
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


# 全局告警管理器实例
alert_manager = AlertManager()