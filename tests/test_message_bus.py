"""
ZeroMQ消息总线测试
测试发布订阅、序列化、错误处理等功能
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from src.core.message_bus import (
    Message, MessageBus, MessagePriority, MessageSerializer,
    Publisher, Subscriber, AsyncPublisher, AsyncSubscriber
)


class TestMessage:
    """Message类测试"""
    
    def test_message_creation(self):
        """测试消息创建"""
        data = {"price": 50000.0, "volume": 1.5}
        message = Message(topic="test.topic", data=data)
        
        assert message.topic == "test.topic"
        assert message.data == data
        assert message.priority == MessagePriority.NORMAL
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert message.ttl is None
        assert message.timestamp > 0
        assert len(message.message_id) > 0
    
    def test_message_expiry(self):
        """测试消息过期检查"""
        # 创建不过期的消息
        message = Message(topic="test", data={})
        assert not message.is_expired()
        
        # 创建已过期的消息
        message_expired = Message(topic="test", data={}, ttl=0.001)
        time.sleep(0.01)
        assert message_expired.is_expired()
    
    def test_message_retry(self):
        """测试消息重试逻辑"""
        message = Message(topic="test", data={}, max_retries=2)
        
        # 初始状态
        assert message.should_retry()
        assert message.retry_count == 0
        
        # 第一次重试
        message.increment_retry()
        assert message.should_retry()
        assert message.retry_count == 1
        
        # 第二次重试
        message.increment_retry()
        assert not message.should_retry()
        assert message.retry_count == 2


class TestMessageSerializer:
    """消息序列化器测试"""
    
    def test_serialization_round_trip(self):
        """测试序列化往返"""
        original = Message(
            topic="market.btcusdt.tick",
            data={"price": 50000.0, "volume": 1.5},
            priority=MessagePriority.HIGH,
            ttl=300.0,
            metadata={"exchange": "binance"}
        )
        
        # 序列化
        serialized = MessageSerializer.serialize(original)
        assert isinstance(serialized, bytes)
        
        # 反序列化
        deserialized = MessageSerializer.deserialize(serialized)
        
        # 验证数据一致性
        assert deserialized.topic == original.topic
        assert deserialized.data == original.data
        assert deserialized.priority == original.priority
        assert deserialized.ttl == original.ttl
        assert deserialized.metadata == original.metadata
        assert deserialized.message_id == original.message_id
    
    def test_serialization_error(self):
        """测试序列化错误处理"""
        # 创建不可序列化的数据
        message = Message(topic="test", data=lambda x: x)
        
        with pytest.raises(ValueError, match="Message serialization failed"):
            MessageSerializer.serialize(message)
    
    def test_deserialization_error(self):
        """测试反序列化错误处理"""
        invalid_data = b"invalid msgpack data"
        
        with pytest.raises(ValueError, match="Message deserialization failed"):
            MessageSerializer.deserialize(invalid_data)


class TestMessageBus:
    """消息总线测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.message_bus = MessageBus(base_port=15555)
    
    def teardown_method(self):
        """测试后清理"""
        self.message_bus.close()
    
    def test_create_publisher(self):
        """测试创建发布者"""
        publisher = self.message_bus.create_publisher("test_pub", port_offset=1)
        assert isinstance(publisher, Publisher)
        assert "test_pub" in self.message_bus.publishers
        
        # 重复创建应返回同一实例
        publisher2 = self.message_bus.create_publisher("test_pub", port_offset=1)
        assert publisher is publisher2
    
    def test_create_subscriber(self):
        """测试创建订阅者"""
        subscriber = self.message_bus.create_subscriber("test_sub", target_port=15556)
        assert isinstance(subscriber, Subscriber)
        assert "test_sub" in self.message_bus.subscribers
        
        # 重复创建应返回同一实例
        subscriber2 = self.message_bus.create_subscriber("test_sub", target_port=15556)
        assert subscriber is subscriber2
    
    def test_publish_subscribe(self):
        """测试发布订阅功能"""
        # 创建发布者和订阅者
        publisher = self.message_bus.create_publisher("pub", port_offset=2)
        subscriber = self.message_bus.create_subscriber("sub", target_port=15557)
        
        # 等待连接建立
        time.sleep(0.1)
        
        # 订阅主题
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
        
        subscriber.subscribe("test.topic", message_handler)
        
        # 发布消息
        test_data = {"price": 50000.0}
        self.message_bus.publish("pub", "test.topic", test_data)
        
        # 接收消息
        time.sleep(0.1)
        message = subscriber.receive(timeout=1000)
        
        # 验证消息
        assert message is not None
        assert message.topic == "test.topic"
        assert message.data == test_data
    
    def test_publish_error_handling(self):
        """测试发布错误处理"""
        # 尝试使用不存在的发布者
        with pytest.raises(ValueError, match="Publisher 'nonexistent' not found"):
            self.message_bus.publish("nonexistent", "topic", {})
    
    def test_stats_tracking(self):
        """测试统计信息跟踪"""
        initial_stats = self.message_bus.get_stats()
        
        # 创建发布者并发布消息
        publisher = self.message_bus.create_publisher("stats_pub", port_offset=3)
        self.message_bus.publish("stats_pub", "test", {})
        
        updated_stats = self.message_bus.get_stats()
        
        # 验证统计信息更新
        assert updated_stats['messages_sent'] == initial_stats['messages_sent'] + 1
        assert updated_stats['publishers_count'] > 0


class TestAsyncMessageBus:
    """异步消息总线测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.message_bus = MessageBus(base_port=16555)
    
    def teardown_method(self):
        """测试后清理"""
        self.message_bus.close()
    
    @pytest.mark.asyncio
    async def test_async_publish_subscribe(self):
        """测试异步发布订阅"""
        # 创建异步发布者和订阅者
        publisher = self.message_bus.create_async_publisher("async_pub", port_offset=1)
        subscriber = self.message_bus.create_async_subscriber("async_sub", target_port=16556)
        
        # 等待连接建立
        await asyncio.sleep(0.1)
        
        # 订阅主题
        received_messages = []
        
        async def async_handler(message):
            received_messages.append(message)
        
        subscriber.subscribe("async.test", async_handler)
        
        # 异步发布消息
        test_data = {"async": True, "value": 123}
        await self.message_bus.async_publish("async_pub", "async.test", test_data)
        
        # 异步接收消息
        await asyncio.sleep(0.1)
        message = await subscriber.receive(timeout=1000)
        
        # 验证消息
        assert message is not None
        assert message.topic == "async.test"
        assert message.data == test_data


class TestRetryMechanism:
    """重试机制测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.message_bus = MessageBus(base_port=17555)
    
    def teardown_method(self):
        """测试后清理"""
        self.message_bus.close()
    
    def test_retry_queue(self):
        """测试重试队列"""
        # 创建一个会失败的发布者（不绑定端口）
        with patch.object(self.message_bus, 'publishers', {}):
            # 尝试发布消息到不存在的发布者
            with pytest.raises(ValueError):
                self.message_bus.publish("nonexistent", "test", {})
    
    def test_retry_worker(self):
        """测试重试工作线程"""
        # 启动重试工作线程
        self.message_bus.start_retry_worker()
        
        # 验证线程已启动
        assert self.message_bus.retry_thread is not None
        assert self.message_bus.retry_thread.is_alive()
        
        # 关闭时验证线程停止
        self.message_bus.close()
        assert not self.message_bus.retry_thread.is_alive()


class TestMessageExpiry:
    """消息过期测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.message_bus = MessageBus(base_port=18555)
    
    def teardown_method(self):
        """测试后清理"""
        self.message_bus.close()
    
    def test_expired_message_skipped(self):
        """测试过期消息被跳过"""
        publisher = self.message_bus.create_publisher("exp_pub", port_offset=1)
        
        # 创建已过期的消息
        expired_message = Message(
            topic="test.expired",
            data={"test": True},
            ttl=0.001  # 1毫秒后过期
        )
        
        # 等待消息过期
        time.sleep(0.01)
        
        # 发布过期消息不应该抛出异常，应该被跳过
        publisher.publish(expired_message)


class TestConcurrency:
    """并发测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.message_bus = MessageBus(base_port=19555)
    
    def teardown_method(self):
        """测试后清理"""
        self.message_bus.close()
    
    def test_concurrent_publish(self):
        """测试并发发布"""
        publisher = self.message_bus.create_publisher("conc_pub", port_offset=1)
        
        def publish_messages(start_id, count):
            for i in range(count):
                self.message_bus.publish(
                    "conc_pub", 
                    "concurrent.test", 
                    {"id": start_id + i}
                )
        
        # 创建多个线程并发发布
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=publish_messages, 
                args=(i * 100, 50)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证统计信息
        stats = self.message_bus.get_stats()
        assert stats['messages_sent'] >= 150


class TestResourceCleanup:
    """资源清理测试"""
    
    def test_message_bus_close(self):
        """测试消息总线关闭"""
        message_bus = MessageBus(base_port=21555)
        
        # 创建资源
        publisher = message_bus.create_publisher("cleanup_pub")
        subscriber = message_bus.create_subscriber("cleanup_sub")
        async_publisher = message_bus.create_async_publisher("cleanup_async_pub", port_offset=1)
        async_subscriber = message_bus.create_async_subscriber("cleanup_async_sub", target_port=21556)
        
        # 启动重试工作线程
        message_bus.start_retry_worker()
        
        # 关闭消息总线
        message_bus.close()
        
        # 验证资源已关闭
        assert not publisher.is_connected
        assert not subscriber.is_connected
        assert not async_publisher.is_connected
        assert not async_subscriber.is_connected
        
        # 验证工作线程已停止
        if message_bus.retry_thread:
            assert not message_bus.retry_thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__])