"""
消息可靠性测试
测试消息确认、重试、持久化等功能
"""

import asyncio
import os
import pytest
import sqlite3
import tempfile
import time
import threading
from unittest.mock import MagicMock, patch

from src.core.message_reliability import (
    AckStatus, RetryStrategy, MessageAck, RetryConfig,
    MessagePersistence, MessageReliabilityManager
)
from src.core.message_bus import Message, MessagePriority


class TestAckStatus:
    """确认状态测试"""
    
    def test_ack_status_enum(self):
        """测试确认状态枚举"""
        assert AckStatus.PENDING.value == "pending"
        assert AckStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AckStatus.FAILED.value == "failed"
        assert AckStatus.EXPIRED.value == "expired"


class TestRetryStrategy:
    """重试策略测试"""
    
    def test_retry_strategy_enum(self):
        """测试重试策略枚举"""
        assert RetryStrategy.IMMEDIATE.value == "immediate"
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.FIXED_INTERVAL.value == "fixed"


class TestMessageAck:
    """消息确认测试"""
    
    def test_message_ack_creation(self):
        """测试消息确认创建"""
        ack = MessageAck(message_id="test_123")
        
        assert ack.message_id == "test_123"
        assert ack.status == AckStatus.PENDING
        assert ack.attempts == 0
        assert ack.last_attempt > 0
        assert ack.created_at > 0
        assert ack.expires_at is None
        assert ack.error_message is None
    
    def test_message_ack_with_expiry(self):
        """测试带过期时间的消息确认"""
        expire_time = time.time() + 300  # 5分钟后过期
        ack = MessageAck(
            message_id="test_expire",
            expires_at=expire_time,
            error_message="Connection timeout"
        )
        
        assert ack.expires_at == expire_time
        assert ack.error_message == "Connection timeout"


class TestRetryConfig:
    """重试配置测试"""
    
    def test_default_retry_config(self):
        """测试默认重试配置"""
        config = RetryConfig()
        
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
    
    def test_custom_retry_config(self):
        """测试自定义重试配置"""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            backoff_multiplier=1.5,
            jitter=False
        )
        
        assert config.strategy == RetryStrategy.LINEAR
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.backoff_multiplier == 1.5
        assert config.jitter is False


class TestMessagePersistence:
    """消息持久化测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时数据库文件
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_messages.db")
        self.persistence = MessagePersistence(self.db_path)
    
    def teardown_method(self):
        """测试后清理"""
        self.persistence.close()
        # 清理临时文件
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_store_and_load_message(self):
        """测试存储和加载消息"""
        # 创建测试消息
        message = Message(
            topic="test.persistence",
            data={"test": True, "value": 123},
            priority=MessagePriority.HIGH,
            ttl=300.0
        )
        
        # 存储消息
        success = self.persistence.store_message(message)
        assert success is True
        
        # 加载消息
        loaded_message = self.persistence.load_message(message.message_id)
        
        # 验证消息内容
        assert loaded_message is not None
        assert loaded_message.topic == message.topic
        assert loaded_message.data == message.data
        assert loaded_message.priority == message.priority
        assert loaded_message.ttl == message.ttl
        assert loaded_message.message_id == message.message_id
    
    def test_load_nonexistent_message(self):
        """测试加载不存在的消息"""
        result = self.persistence.load_message("nonexistent_id")
        assert result is None
    
    def test_get_pending_messages(self):
        """测试获取待处理消息"""
        # 创建多个测试消息
        messages = []
        for i in range(5):
            message = Message(
                topic=f"test.pending.{i}",
                data={"index": i},
                priority=MessagePriority.HIGH if i % 2 == 0 else MessagePriority.NORMAL
            )
            messages.append(message)
            self.persistence.store_message(message)
        
        # 获取待处理消息
        pending = self.persistence.get_pending_messages(limit=10)
        
        # 验证消息数量和排序（优先级高的在前）
        assert len(pending) == 5
        assert pending[0].priority == MessagePriority.HIGH
    
    def test_update_message_status(self):
        """测试更新消息状态"""
        # 创建并存储测试消息
        message = Message(topic="test.status", data={"test": True})
        self.persistence.store_message(message)
        
        # 更新消息状态
        success = self.persistence.update_message_status(
            message.message_id, 
            "completed", 
            retry_count=2
        )
        assert success is True
        
        # 验证状态已更新（通过直接查询数据库）
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT status, retry_count FROM messages WHERE id = ?",
            (message.message_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == "completed"
        assert row[1] == 2
    
    def test_store_and_load_ack(self):
        """测试存储和加载确认信息"""
        # 创建确认信息
        ack = MessageAck(
            message_id="test_ack_123",
            status=AckStatus.ACKNOWLEDGED,
            attempts=2,
            expires_at=time.time() + 300,
            error_message="Test error"
        )
        
        # 存储确认信息
        success = self.persistence.store_ack(ack)
        assert success is True
        
        # 通过直接查询验证存储
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT status, attempts, error_message FROM message_acks WHERE message_id = ?",
            (ack.message_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == "acknowledged"
        assert row[1] == 2
        assert row[2] == "Test error"
    
    def test_cleanup_expired(self):
        """测试清理过期数据"""
        # 创建过期消息
        expired_message = Message(
            topic="test.expired",
            data={"expired": True},
            ttl=0.001  # 1毫秒后过期
        )
        self.persistence.store_message(expired_message)
        
        # 创建过期确认
        expired_ack = MessageAck(
            message_id="expired_ack",
            expires_at=time.time() - 3600  # 1小时前过期
        )
        self.persistence.store_ack(expired_ack)
        
        # 等待消息过期
        time.sleep(0.01)
        
        # 清理过期数据
        deleted_count = self.persistence.cleanup_expired()
        
        # 验证清理结果
        assert deleted_count >= 1  # 至少删除了过期确认
    
    def test_database_error_handling(self):
        """测试数据库错误处理"""
        # 关闭数据库连接以模拟错误
        self.persistence.conn.close()
        
        # 尝试存储消息（应该失败）
        message = Message(topic="test.error", data={})
        result = self.persistence.store_message(message)
        assert result is False
        
        # 尝试加载消息（应该返回None）
        result = self.persistence.load_message("any_id")
        assert result is None


class TestMessageReliabilityManager:
    """消息可靠性管理器测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 创建临时数据库
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reliability.db")
        
        # 创建持久化实例
        persistence = MessagePersistence(self.db_path)
        self.manager = MessageReliabilityManager(persistence)
    
    def teardown_method(self):
        """测试后清理"""
        self.manager.close()
        # 清理临时文件
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_send_reliable_message(self):
        """测试发送可靠消息"""
        message = Message(topic="test.reliable", data={"test": True})
        
        # 发送可靠消息
        message_id = self.manager.send_reliable_message(
            message, 
            ack_timeout=30.0
        )
        
        # 验证消息ID
        assert message_id == message.message_id
        
        # 验证pending acks中有记录
        assert message_id in self.manager.pending_acks
        
        # 验证确认信息
        ack = self.manager.pending_acks[message_id]
        assert ack.status == AckStatus.PENDING
        assert ack.expires_at > time.time()
    
    def test_acknowledge_message_success(self):
        """测试成功确认消息"""
        message = Message(topic="test.ack", data={"test": True})
        
        # 发送消息
        message_id = self.manager.send_reliable_message(message)
        
        # 确认消息成功
        self.manager.acknowledge_message(message_id, success=True)
        
        # 验证消息已从pending中移除
        assert message_id not in self.manager.pending_acks
        
        # 验证统计信息
        stats = self.manager.get_stats()
        assert stats['messages_acked'] == 1
        assert stats['messages_sent'] == 1
    
    def test_acknowledge_message_failure(self):
        """测试失败确认消息"""
        message = Message(topic="test.fail", data={"test": True})
        
        # 发送消息
        message_id = self.manager.send_reliable_message(message)
        
        # 确认消息失败
        self.manager.acknowledge_message(
            message_id, 
            success=False, 
            error_message="Processing failed"
        )
        
        # 验证消息已从pending中移除
        assert message_id not in self.manager.pending_acks
        
        # 验证统计信息
        stats = self.manager.get_stats()
        assert stats['messages_failed'] == 1
    
    def test_acknowledge_unknown_message(self):
        """测试确认未知消息"""
        # 确认不存在的消息不应该抛出异常
        self.manager.acknowledge_message("unknown_id")
        
        # 统计信息不应该改变
        stats = self.manager.get_stats()
        assert stats['messages_acked'] == 0
        assert stats['messages_failed'] == 0
    
    def test_retry_message(self):
        """测试重试消息"""
        message = Message(
            topic="test.retry", 
            data={"test": True},
            max_retries=2
        )
        
        # 设置重试回调
        retry_called = []
        def retry_callback(msg):
            retry_called.append(msg.message_id)
        
        self.manager.retry_callbacks["test.retry"] = retry_callback
        
        # 重试消息
        success = self.manager.retry_message(message)
        assert success is True
        assert message.retry_count == 1
        
        # 等待重试回调执行
        time.sleep(0.3)  # 增加等待时间
        assert message.message_id in retry_called
    
    def test_retry_message_max_attempts(self):
        """测试达到最大重试次数"""
        message = Message(
            topic="test.max_retry", 
            data={"test": True},
            max_retries=1
        )
        message.retry_count = 1  # 手动设置已经重试过1次
        
        # 重试应该失败
        success = self.manager.retry_message(message)
        assert success is False
    
    def test_retry_config_management(self):
        """测试重试配置管理"""
        # 设置重试配置
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR,
            max_attempts=5,
            initial_delay=0.5
        )
        self.manager.set_retry_config("high_priority.*", config)
        
        # 获取配置
        retrieved_config = self.manager.get_retry_config("high_priority.test")
        assert retrieved_config.strategy == RetryStrategy.LINEAR
        assert retrieved_config.max_attempts == 5
        
        # 获取不匹配模式的配置（应返回默认配置）
        default_config = self.manager.get_retry_config("normal.test")
        assert default_config.strategy == RetryStrategy.EXPONENTIAL  # 默认值
    
    def test_calculate_retry_delay(self):
        """测试重试延迟计算"""
        # 测试立即重试
        config = RetryConfig(strategy=RetryStrategy.IMMEDIATE)
        delay = self.manager._calculate_retry_delay(1, config)
        assert delay == 0.0
        
        # 测试线性退避
        config = RetryConfig(strategy=RetryStrategy.LINEAR, initial_delay=1.0, jitter=False)
        delay = self.manager._calculate_retry_delay(2, config)  # attempt=2
        assert delay == 3.0  # initial_delay * (attempt + 1) = 1.0 * 3
        
        # 测试指数退避
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL, 
            initial_delay=1.0, 
            backoff_multiplier=2.0,
            jitter=False
        )
        delay = self.manager._calculate_retry_delay(2, config)  # attempt=2
        assert delay == 4.0  # initial_delay * backoff_multiplier^attempt = 1.0 * 2^2
        
        # 测试固定间隔
        config = RetryConfig(strategy=RetryStrategy.FIXED_INTERVAL, initial_delay=5.0)
        delay = self.manager._calculate_retry_delay(10, config)
        assert delay == 5.0
        
        # 测试最大延迟限制
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_multiplier=2.0
        )
        delay = self.manager._calculate_retry_delay(10, config)  # 2^10 = 1024，但最大为10
        assert delay <= 10.0
    
    def test_worker_thread_management(self):
        """测试工作线程管理"""
        # 验证工作线程已启动
        assert self.manager.worker_thread is not None
        assert self.manager.worker_thread.is_alive()
        
        # 停止工作线程
        self.manager.stop_worker()
        
        # 验证线程已停止
        assert self.manager.worker_thread is None or not self.manager.worker_thread.is_alive()
    
    def test_expired_message_handling(self):
        """测试过期消息处理"""
        # 创建很快过期的消息
        message = Message(
            topic="test.expire",
            data={"test": True},
            ttl=0.001  # 1毫秒后过期
        )
        
        # 发送消息
        message_id = self.manager.send_reliable_message(message, ack_timeout=0.01)
        
        # 等待消息和确认过期
        time.sleep(0.05)
        
        # 手动触发工作线程逻辑（模拟定期检查）
        current_time = time.time()
        expired_acks = []
        
        for msg_id, ack in list(self.manager.pending_acks.items()):
            if ack.expires_at and current_time > ack.expires_at:
                expired_acks.append(msg_id)
        
        # 验证有过期的确认
        assert len(expired_acks) > 0
        assert message_id in expired_acks
    
    def test_stats_collection(self):
        """测试统计信息收集"""
        initial_stats = self.manager.get_stats()
        
        # 发送消息
        message = Message(topic="test.stats", data={"test": True})
        self.manager.send_reliable_message(message)
        
        # 确认消息
        self.manager.acknowledge_message(message.message_id, success=True)
        
        # 检查统计信息
        final_stats = self.manager.get_stats()
        
        assert final_stats['messages_sent'] == initial_stats['messages_sent'] + 1
        assert final_stats['messages_acked'] == initial_stats['messages_acked'] + 1
        assert 'pending_acks' in final_stats
        assert 'retry_configs' in final_stats
    
    def test_pattern_matching(self):
        """测试模式匹配"""
        # 测试通配符匹配
        assert self.manager._match_pattern("market.btc.tick", "market.*")
        assert self.manager._match_pattern("signal.rsi", "signal.*")
        
        # 测试前缀匹配
        assert self.manager._match_pattern("market.data", "market*")
        
        # 测试后缀匹配
        assert self.manager._match_pattern("data.tick", "*tick")
        
        # 测试精确匹配
        assert self.manager._match_pattern("exact.match", "exact.match")
        assert not self.manager._match_pattern("exact.match", "different")


class TestMessageReliabilityIntegration:
    """消息可靠性集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        persistence = MessagePersistence(self.db_path)
        self.manager = MessageReliabilityManager(persistence)
    
    def teardown_method(self):
        """测试后清理"""
        self.manager.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_complete_message_lifecycle(self):
        """测试完整的消息生命周期"""
        # 1. 发送可靠消息
        message = Message(topic="lifecycle.test", data={"step": 1})
        callback_called = []
        
        def ack_callback(ack):
            callback_called.append(ack.message_id)
        
        message_id = self.manager.send_reliable_message(
            message,
            ack_timeout=30.0,
            ack_callback=ack_callback
        )
        
        # 2. 验证消息已存储
        loaded_message = self.manager.persistence.load_message(message_id)
        assert loaded_message is not None
        assert loaded_message.topic == message.topic
        
        # 3. 确认消息
        self.manager.acknowledge_message(message_id, success=True)
        
        # 4. 验证回调被调用
        assert message_id in callback_called
        
        # 5. 验证统计信息
        stats = self.manager.get_stats()
        assert stats['messages_sent'] >= 1
        assert stats['messages_acked'] >= 1


if __name__ == "__main__":
    pytest.main([__file__])