"""
消息可靠性保证机制
包括消息确认、重试、持久化等功能
"""

import asyncio
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import json
import sqlite3
from pathlib import Path

from src.config import settings
from src.utils.logger import LoggerMixin
from src.core.message_bus import Message, MessagePriority


class AckStatus(Enum):
    """确认状态"""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = "immediate"      # 立即重试
    LINEAR = "linear"           # 线性退避
    EXPONENTIAL = "exponential" # 指数退避
    FIXED_INTERVAL = "fixed"    # 固定间隔


@dataclass
class MessageAck:
    """消息确认信息"""
    message_id: str
    status: AckStatus = AckStatus.PENDING
    attempts: int = 0
    last_attempt: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class RetryConfig:
    """重试配置"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class MessagePersistence(LoggerMixin):
    """消息持久化存储"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Path(settings.data_directory) / "messages.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_tables()
    
    def _init_tables(self):
        """初始化数据库表"""
        with self.lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    data TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl REAL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    status TEXT DEFAULT 'pending',
                    created_at REAL
                )
            """)
            
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS message_acks (
                    message_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    last_attempt REAL,
                    created_at REAL,
                    expires_at REAL,
                    error_message TEXT
                )
            """)
            self.conn.commit()
    
    def store_message(self, message: Message) -> bool:
        """存储消息"""
        try:
            with self.lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO messages 
                    (id, topic, data, priority, timestamp, ttl, retry_count, max_retries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.message_id,
                    message.topic,
                    json.dumps(message.data),
                    message.priority.value,
                    message.timestamp,
                    message.ttl,
                    message.retry_count,
                    message.max_retries
                ))
                self.conn.commit()
                return True
        except Exception as e:
            self.log_error(f"Failed to store message {message.message_id}: {e}")
            return False
    
    def load_message(self, message_id: str) -> Optional[Message]:
        """加载消息"""
        try:
            with self.lock:
                cursor = self.conn.execute("""
                    SELECT topic, data, priority, timestamp, ttl, retry_count, max_retries
                    FROM messages WHERE id = ?
                """, (message_id,))
                row = cursor.fetchone()
                
                if row:
                    topic, data_json, priority, timestamp, ttl, retry_count, max_retries = row
                    return Message(
                        topic=topic,
                        data=json.loads(data_json),
                        timestamp=timestamp,
                        message_id=message_id,
                        priority=MessagePriority(priority),
                        retry_count=retry_count,
                        max_retries=max_retries,
                        ttl=ttl
                    )
                return None
        except Exception as e:
            self.log_error(f"Failed to load message {message_id}: {e}")
            return None
    
    def get_pending_messages(self, limit: int = 100) -> List[Message]:
        """获取待处理的消息"""
        try:
            with self.lock:
                cursor = self.conn.execute("""
                    SELECT id, topic, data, priority, timestamp, ttl, retry_count, max_retries
                    FROM messages 
                    WHERE status = 'pending' AND retry_count < max_retries
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                """, (limit,))
                
                messages = []
                for row in cursor.fetchall():
                    message_id, topic, data_json, priority, timestamp, ttl, retry_count, max_retries = row
                    message = Message(
                        topic=topic,
                        data=json.loads(data_json),
                        timestamp=timestamp,
                        message_id=message_id,
                        priority=MessagePriority(priority),
                        retry_count=retry_count,
                        max_retries=max_retries,
                        ttl=ttl
                    )
                    messages.append(message)
                
                return messages
        except Exception as e:
            self.log_error(f"Failed to get pending messages: {e}")
            return []
    
    def update_message_status(self, message_id: str, status: str, retry_count: int = None) -> bool:
        """更新消息状态"""
        try:
            with self.lock:
                if retry_count is not None:
                    self.conn.execute("""
                        UPDATE messages SET status = ?, retry_count = ? WHERE id = ?
                    """, (status, retry_count, message_id))
                else:
                    self.conn.execute("""
                        UPDATE messages SET status = ? WHERE id = ?
                    """, (status, message_id))
                self.conn.commit()
                return True
        except Exception as e:
            self.log_error(f"Failed to update message status {message_id}: {e}")
            return False
    
    def store_ack(self, ack: MessageAck) -> bool:
        """存储确认信息"""
        try:
            with self.lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO message_acks 
                    (message_id, status, attempts, last_attempt, created_at, expires_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ack.message_id,
                    ack.status.value,
                    ack.attempts,
                    ack.last_attempt,
                    ack.created_at,
                    ack.expires_at,
                    ack.error_message
                ))
                self.conn.commit()
                return True
        except Exception as e:
            self.log_error(f"Failed to store ack {ack.message_id}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """清理过期消息"""
        try:
            current_time = time.time()
            with self.lock:
                # 清理过期消息
                cursor = self.conn.execute("""
                    DELETE FROM messages 
                    WHERE (ttl IS NOT NULL AND timestamp + ttl < ?) 
                       OR (created_at < ? - 86400 * 7)  -- 7天前的消息
                """, (current_time, current_time))
                deleted_messages = cursor.rowcount
                
                # 清理过期确认
                cursor = self.conn.execute("""
                    DELETE FROM message_acks 
                    WHERE (expires_at IS NOT NULL AND expires_at < ?)
                       OR (created_at < ? - 86400 * 7)
                """, (current_time, current_time))
                deleted_acks = cursor.rowcount
                
                self.conn.commit()
                
                total_deleted = deleted_messages + deleted_acks
                if total_deleted > 0:
                    self.log_info(f"Cleaned up {deleted_messages} messages and {deleted_acks} acks")
                
                return total_deleted
        except Exception as e:
            self.log_error(f"Failed to cleanup expired data: {e}")
            return 0
    
    def close(self):
        """关闭数据库连接"""
        with self.lock:
            self.conn.close()


class MessageReliabilityManager(LoggerMixin):
    """消息可靠性管理器"""
    
    def __init__(self, persistence: MessagePersistence = None):
        self.persistence = persistence or MessagePersistence()
        self.pending_acks: Dict[str, MessageAck] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.default_retry_config = RetryConfig()
        
        # 回调函数
        self.ack_callbacks: Dict[str, Callable[[MessageAck], None]] = {}
        self.retry_callbacks: Dict[str, Callable[[Message], None]] = {}
        
        # 工作线程
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_acked': 0,
            'messages_failed': 0,
            'retries_attempted': 0,
            'messages_expired': 0
        }
        
        self.start_worker()
    
    def set_retry_config(self, topic_pattern: str, config: RetryConfig):
        """设置重试配置"""
        self.retry_configs[topic_pattern] = config
    
    def get_retry_config(self, topic: str) -> RetryConfig:
        """获取重试配置"""
        for pattern, config in self.retry_configs.items():
            if self._match_pattern(topic, pattern):
                return config
        return self.default_retry_config
    
    def send_reliable_message(self, message: Message, 
                            ack_timeout: float = 30.0,
                            ack_callback: Callable[[MessageAck], None] = None) -> str:
        """发送可靠消息"""
        # 存储消息
        self.persistence.store_message(message)
        
        # 创建确认记录
        ack = MessageAck(
            message_id=message.message_id,
            expires_at=time.time() + ack_timeout
        )
        self.pending_acks[message.message_id] = ack
        self.persistence.store_ack(ack)
        
        # 注册回调
        if ack_callback:
            self.ack_callbacks[message.message_id] = ack_callback
        
        self.stats['messages_sent'] += 1
        self.log_debug(f"Sent reliable message {message.message_id} to topic {message.topic}")
        
        return message.message_id
    
    def acknowledge_message(self, message_id: str, success: bool = True, 
                          error_message: str = None):
        """确认消息"""
        if message_id not in self.pending_acks:
            self.log_warning(f"Acknowledgment for unknown message {message_id}")
            return
        
        ack = self.pending_acks[message_id]
        ack.status = AckStatus.ACKNOWLEDGED if success else AckStatus.FAILED
        ack.error_message = error_message
        ack.last_attempt = time.time()
        
        # 更新持久化存储
        self.persistence.store_ack(ack)
        self.persistence.update_message_status(
            message_id, 
            'completed' if success else 'failed'
        )
        
        # 调用回调
        if message_id in self.ack_callbacks:
            try:
                self.ack_callbacks[message_id](ack)
                del self.ack_callbacks[message_id]
            except Exception as e:
                self.log_error(f"Error in ack callback for {message_id}: {e}")
        
        # 移除pending记录
        del self.pending_acks[message_id]
        
        if success:
            self.stats['messages_acked'] += 1
        else:
            self.stats['messages_failed'] += 1
        
        self.log_debug(f"Message {message_id} acknowledged: {success}")
    
    def retry_message(self, message: Message, config: RetryConfig = None) -> bool:
        """重试消息"""
        config = config or self.get_retry_config(message.topic)
        
        if message.retry_count >= config.max_attempts:
            self.log_warning(f"Message {message.message_id} exceeded max retries")
            self.persistence.update_message_status(message.message_id, 'failed')
            return False
        
        # 计算延迟
        delay = self._calculate_retry_delay(message.retry_count, config)
        
        # 增加重试次数
        message.increment_retry()
        self.persistence.update_message_status(
            message.message_id, 
            'pending', 
            message.retry_count
        )
        
        # 调用重试回调
        if message.topic in self.retry_callbacks:
            try:
                # 延迟执行重试
                def delayed_retry():
                    time.sleep(delay)
                    self.retry_callbacks[message.topic](message)
                
                threading.Thread(target=delayed_retry, daemon=True).start()
            except Exception as e:
                self.log_error(f"Error in retry callback for {message.message_id}: {e}")
                return False
        
        self.stats['retries_attempted'] += 1
        self.log_info(f"Retrying message {message.message_id} (attempt {message.retry_count})")
        
        return True
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """计算重试延迟"""
        if config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.initial_delay * (config.backoff_multiplier ** attempt)
        else:  # FIXED_INTERVAL
            delay = config.initial_delay
        
        # 应用最大延迟限制
        delay = min(delay, config.max_delay)
        
        # 添加抖动
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _match_pattern(self, topic: str, pattern: str) -> bool:
        """匹配主题模式"""
        if pattern == "*":
            return True
        if "*" not in pattern:
            return topic == pattern
        
        # 简单通配符匹配
        if pattern.endswith("*"):
            return topic.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return topic.endswith(pattern[1:])
        
        return False
    
    def start_worker(self):
        """启动工作线程"""
        if self.worker_thread is not None:
            return
        
        def worker():
            while not self.stop_event.wait(5.0):  # 每5秒检查一次
                try:
                    # 检查过期的确认
                    current_time = time.time()
                    expired_acks = []
                    
                    for message_id, ack in list(self.pending_acks.items()):
                        if ack.expires_at and current_time > ack.expires_at:
                            expired_acks.append(message_id)
                    
                    # 处理过期确认
                    for message_id in expired_acks:
                        ack = self.pending_acks[message_id]
                        ack.status = AckStatus.EXPIRED
                        
                        # 尝试重试
                        message = self.persistence.load_message(message_id)
                        if message and not message.is_expired():
                            self.retry_message(message)
                        else:
                            self.persistence.update_message_status(message_id, 'expired')
                            self.stats['messages_expired'] += 1
                        
                        del self.pending_acks[message_id]
                    
                    # 清理过期数据
                    self.persistence.cleanup_expired()
                    
                except Exception as e:
                    self.log_error(f"Error in reliability worker: {e}")
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self.log_info("Message reliability worker started")
    
    def stop_worker(self):
        """停止工作线程"""
        if self.worker_thread:
            self.stop_event.set()
            self.worker_thread.join(timeout=10.0)
            self.worker_thread = None
            self.log_info("Message reliability worker stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'pending_acks': len(self.pending_acks),
            'retry_configs': len(self.retry_configs)
        }
    
    def close(self):
        """关闭管理器"""
        self.stop_worker()
        self.persistence.close()


# 全局可靠性管理器实例
# reliability_manager = MessageReliabilityManager()  # 延迟初始化，避免import时错误