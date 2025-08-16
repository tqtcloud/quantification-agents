"""
ZeroMQ消息总线实现
支持发布订阅、请求响应等多种消息模式
"""

import asyncio
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import weakref

import msgpack
import zmq
import zmq.asyncio

from src.config import settings
from src.utils.logger import LoggerMixin


class MessagePattern(Enum):
    """消息模式"""
    PUB_SUB = "pub_sub"          # 发布-订阅
    PUSH_PULL = "push_pull"      # 推-拉
    REQ_REP = "req_rep"          # 请求-响应
    DEALER_ROUTER = "dealer_router"  # 经销商-路由器


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    """标准消息格式"""
    topic: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[float] = None  # 消息生存时间（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查消息是否已过期"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """增加重试次数"""
        self.retry_count += 1


class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize(message: Message) -> bytes:
        """序列化消息"""
        try:
            data = {
                'topic': message.topic,
                'data': message.data,
                'timestamp': message.timestamp,
                'message_id': message.message_id,
                'priority': message.priority.value,
                'retry_count': message.retry_count,
                'max_retries': message.max_retries,
                'ttl': message.ttl,
                'metadata': message.metadata
            }
            return msgpack.packb(data, use_bin_type=True)
        except Exception as e:
            raise ValueError(f"Message serialization failed: {e}")
    
    @staticmethod
    def deserialize(data: bytes) -> Message:
        """反序列化消息"""
        try:
            msg_data = msgpack.unpackb(data, raw=False)
            return Message(
                topic=msg_data['topic'],
                data=msg_data['data'],
                timestamp=msg_data['timestamp'],
                message_id=msg_data['message_id'],
                priority=MessagePriority(msg_data['priority']),
                retry_count=msg_data['retry_count'],
                max_retries=msg_data['max_retries'],
                ttl=msg_data.get('ttl'),
                metadata=msg_data.get('metadata', {})
            )
        except Exception as e:
            raise ValueError(f"Message deserialization failed: {e}")


class Publisher(LoggerMixin):
    """ZeroMQ发布者"""
    
    def __init__(self, context: zmq.Context, address: str):
        self.context = context
        self.address = address
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 1000)  # 1秒延迟关闭
        self.socket.bind(address)
        self.is_connected = True
        self.log_info(f"Publisher bound to {address}")
    
    def publish(self, message: Message):
        """发布消息"""
        if not self.is_connected:
            raise RuntimeError("Publisher is not connected")
        
        if message.is_expired():
            self.log_warning(f"Message {message.message_id} expired, skipping")
            return
        
        try:
            # 发送主题和消息数据
            serialized = MessageSerializer.serialize(message)
            self.socket.send_multipart([
                message.topic.encode('utf-8'),
                serialized
            ])
            self.log_debug(f"Published message to topic: {message.topic}")
        except Exception as e:
            self.log_error(f"Failed to publish message: {e}")
            raise
    
    def close(self):
        """关闭发布者"""
        if self.is_connected:
            self.socket.close()
            self.is_connected = False
            self.log_info("Publisher closed")


class Subscriber(LoggerMixin):
    """ZeroMQ订阅者"""
    
    def __init__(self, context: zmq.Context, address: str):
        self.context = context
        self.address = address
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        self.is_connected = True
        self.subscriptions: Set[str] = set()
        self.handlers: Dict[str, List[Callable]] = {}
        self.log_info(f"Subscriber connected to {address}")
    
    def subscribe(self, topic: str, handler: Optional[Callable[[Message], None]] = None):
        """订阅主题"""
        if topic not in self.subscriptions:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            self.subscriptions.add(topic)
            self.log_info(f"Subscribed to topic: {topic}")
        
        if handler:
            if topic not in self.handlers:
                self.handlers[topic] = []
            self.handlers[topic].append(handler)
    
    def unsubscribe(self, topic: str):
        """取消订阅主题"""
        if topic in self.subscriptions:
            self.socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)
            self.subscriptions.remove(topic)
            if topic in self.handlers:
                del self.handlers[topic]
            self.log_info(f"Unsubscribed from topic: {topic}")
    
    def receive(self, timeout: Optional[int] = None) -> Optional[Message]:
        """接收消息"""
        if not self.is_connected:
            return None
        
        try:
            if timeout:
                if self.socket.poll(timeout) == 0:
                    return None
            
            topic_bytes, message_bytes = self.socket.recv_multipart(zmq.NOBLOCK)
            topic = topic_bytes.decode('utf-8')
            message = MessageSerializer.deserialize(message_bytes)
            
            # 调用处理器
            if topic in self.handlers:
                for handler in self.handlers[topic]:
                    try:
                        handler(message)
                    except Exception as e:
                        self.log_error(f"Handler error for topic {topic}: {e}")
            
            return message
        except zmq.Again:
            return None
        except Exception as e:
            self.log_error(f"Failed to receive message: {e}")
            return None
    
    def close(self):
        """关闭订阅者"""
        if self.is_connected:
            self.socket.close()
            self.is_connected = False
            self.log_info("Subscriber closed")


class AsyncPublisher(LoggerMixin):
    """异步ZeroMQ发布者"""
    
    def __init__(self, context: zmq.asyncio.Context, address: str):
        self.context = context
        self.address = address
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 1000)
        self.socket.bind(address)
        self.is_connected = True
        self.log_info(f"Async Publisher bound to {address}")
    
    async def publish(self, message: Message):
        """异步发布消息"""
        if not self.is_connected:
            raise RuntimeError("Publisher is not connected")
        
        if message.is_expired():
            self.log_warning(f"Message {message.message_id} expired, skipping")
            return
        
        try:
            serialized = MessageSerializer.serialize(message)
            await self.socket.send_multipart([
                message.topic.encode('utf-8'),
                serialized
            ])
            self.log_debug(f"Published message to topic: {message.topic}")
        except Exception as e:
            self.log_error(f"Failed to publish message: {e}")
            raise
    
    def close(self):
        """关闭发布者"""
        if self.is_connected:
            self.socket.close()
            self.is_connected = False
            self.log_info("Async Publisher closed")


class AsyncSubscriber(LoggerMixin):
    """异步ZeroMQ订阅者"""
    
    def __init__(self, context: zmq.asyncio.Context, address: str):
        self.context = context
        self.address = address
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(address)
        self.is_connected = True
        self.subscriptions: Set[str] = set()
        self.handlers: Dict[str, List[Callable]] = {}
        self.log_info(f"Async Subscriber connected to {address}")
    
    def subscribe(self, topic: str, handler: Optional[Callable[[Message], Any]] = None):
        """订阅主题"""
        if topic not in self.subscriptions:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            self.subscriptions.add(topic)
            self.log_info(f"Subscribed to topic: {topic}")
        
        if handler:
            if topic not in self.handlers:
                self.handlers[topic] = []
            self.handlers[topic].append(handler)
    
    def unsubscribe(self, topic: str):
        """取消订阅主题"""
        if topic in self.subscriptions:
            self.socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)
            self.subscriptions.remove(topic)
            if topic in self.handlers:
                del self.handlers[topic]
            self.log_info(f"Unsubscribed from topic: {topic}")
    
    async def receive(self, timeout: Optional[int] = None) -> Optional[Message]:
        """异步接收消息"""
        if not self.is_connected:
            return None
        
        try:
            if timeout:
                if await self.socket.poll(timeout) == 0:
                    return None
            
            topic_bytes, message_bytes = await self.socket.recv_multipart()
            topic = topic_bytes.decode('utf-8')
            message = MessageSerializer.deserialize(message_bytes)
            
            # 调用处理器
            if topic in self.handlers:
                for handler in self.handlers[topic]:
                    try:
                        result = handler(message)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        self.log_error(f"Handler error for topic {topic}: {e}")
            
            return message
        except Exception as e:
            self.log_error(f"Failed to receive message: {e}")
            return None
    
    def close(self):
        """关闭订阅者"""
        if self.is_connected:
            self.socket.close()
            self.is_connected = False
            self.log_info("Async Subscriber closed")


class MessageBus(LoggerMixin):
    """消息总线 - ZeroMQ封装类"""
    
    def __init__(self, base_port: int = None):
        self.base_port = base_port or settings.zmq_pub_port
        self.context = zmq.Context()
        self.async_context = zmq.asyncio.Context()
        
        # 发布者和订阅者管理
        self.publishers: Dict[str, Publisher] = {}
        self.subscribers: Dict[str, Subscriber] = {}
        self.async_publishers: Dict[str, AsyncPublisher] = {}
        self.async_subscribers: Dict[str, AsyncSubscriber] = {}
        
        # 消息重试队列
        self.retry_queue: List[Message] = []
        self.retry_thread: Optional[threading.Thread] = None
        self.stop_retry = threading.Event()
        
        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'messages_retried': 0
        }
        
        self.log_info("MessageBus initialized")
    
    def create_publisher(self, name: str, port_offset: int = 0) -> Publisher:
        """创建发布者"""
        if name in self.publishers:
            return self.publishers[name]
        
        address = f"tcp://*:{self.base_port + port_offset}"
        publisher = Publisher(self.context, address)
        self.publishers[name] = publisher
        return publisher
    
    def create_subscriber(self, name: str, target_port: int = None) -> Subscriber:
        """创建订阅者"""
        if name in self.subscribers:
            return self.subscribers[name]
        
        port = target_port or self.base_port
        address = f"tcp://localhost:{port}"
        subscriber = Subscriber(self.context, address)
        self.subscribers[name] = subscriber
        return subscriber
    
    def create_async_publisher(self, name: str, port_offset: int = 0) -> AsyncPublisher:
        """创建异步发布者"""
        if name in self.async_publishers:
            return self.async_publishers[name]
        
        address = f"tcp://*:{self.base_port + port_offset}"
        publisher = AsyncPublisher(self.async_context, address)
        self.async_publishers[name] = publisher
        return publisher
    
    def create_async_subscriber(self, name: str, target_port: int = None) -> AsyncSubscriber:
        """创建异步订阅者"""
        if name in self.async_subscribers:
            return self.async_subscribers[name]
        
        port = target_port or self.base_port
        address = f"tcp://localhost:{port}"
        subscriber = AsyncSubscriber(self.async_context, address)
        self.async_subscribers[name] = subscriber
        return subscriber
    
    def publish(self, publisher_name: str, topic: str, data: Any, 
                priority: MessagePriority = MessagePriority.NORMAL,
                ttl: Optional[float] = None, **metadata):
        """发布消息"""
        if publisher_name not in self.publishers:
            raise ValueError(f"Publisher '{publisher_name}' not found")
        
        message = Message(
            topic=topic,
            data=data,
            priority=priority,
            ttl=ttl,
            metadata=metadata
        )
        
        try:
            self.publishers[publisher_name].publish(message)
            self.stats['messages_sent'] += 1
        except Exception as e:
            self.stats['messages_failed'] += 1
            self.log_error(f"Failed to publish message: {e}")
            
            # 添加到重试队列
            if message.should_retry():
                message.increment_retry()
                self.retry_queue.append(message)
                self.stats['messages_retried'] += 1
            raise
    
    async def async_publish(self, publisher_name: str, topic: str, data: Any,
                           priority: MessagePriority = MessagePriority.NORMAL,
                           ttl: Optional[float] = None, **metadata):
        """异步发布消息"""
        if publisher_name not in self.async_publishers:
            raise ValueError(f"Async Publisher '{publisher_name}' not found")
        
        message = Message(
            topic=topic,
            data=data,
            priority=priority,
            ttl=ttl,
            metadata=metadata
        )
        
        try:
            await self.async_publishers[publisher_name].publish(message)
            self.stats['messages_sent'] += 1
        except Exception as e:
            self.stats['messages_failed'] += 1
            self.log_error(f"Failed to publish message: {e}")
            
            # 添加到重试队列
            if message.should_retry():
                message.increment_retry()
                self.retry_queue.append(message)
                self.stats['messages_retried'] += 1
            raise
    
    def subscribe(self, subscriber_name: str, topic: str, 
                  handler: Optional[Callable[[Message], None]] = None):
        """订阅消息"""
        if subscriber_name not in self.subscribers:
            raise ValueError(f"Subscriber '{subscriber_name}' not found")
        
        self.subscribers[subscriber_name].subscribe(topic, handler)
    
    def start_retry_worker(self):
        """启动重试工作线程"""
        if self.retry_thread is not None:
            return
        
        def retry_worker():
            while not self.stop_retry.wait(1.0):
                if not self.retry_queue:
                    continue
                
                # 处理重试队列
                messages_to_retry = []
                while self.retry_queue:
                    message = self.retry_queue.pop(0)
                    if not message.is_expired() and message.should_retry():
                        messages_to_retry.append(message)
                
                # 重试发送消息
                for message in messages_to_retry:
                    try:
                        # 找到合适的发布者重试
                        for publisher in self.publishers.values():
                            publisher.publish(message)
                            self.stats['messages_sent'] += 1
                            break
                    except Exception as e:
                        message.increment_retry()
                        if message.should_retry():
                            self.retry_queue.append(message)
                        else:
                            self.log_error(f"Message {message.message_id} retry failed: {e}")
        
        self.retry_thread = threading.Thread(target=retry_worker, daemon=True)
        self.retry_thread.start()
        self.log_info("Retry worker started")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'publishers_count': len(self.publishers),
            'subscribers_count': len(self.subscribers),
            'retry_queue_size': len(self.retry_queue),
            'uptime': time.time()
        }
    
    def close(self):
        """关闭消息总线"""
        # 停止重试线程
        if self.retry_thread:
            self.stop_retry.set()
            self.retry_thread.join(timeout=5.0)
        
        # 关闭所有发布者和订阅者
        for publisher in self.publishers.values():
            publisher.close()
        
        for subscriber in self.subscribers.values():
            subscriber.close()
        
        for publisher in self.async_publishers.values():
            publisher.close()
        
        for subscriber in self.async_subscribers.values():
            subscriber.close()
        
        # 关闭ZeroMQ上下文
        self.context.term()
        self.async_context.term()
        
        self.log_info("MessageBus closed")


# 全局消息总线实例
message_bus = MessageBus()


@contextmanager
def get_publisher(name: str, port_offset: int = 0):
    """获取发布者的上下文管理器"""
    publisher = message_bus.create_publisher(name, port_offset)
    try:
        yield publisher
    finally:
        pass  # 不立即关闭，由消息总线统一管理


@contextmanager
def get_subscriber(name: str, target_port: int = None):
    """获取订阅者的上下文管理器"""
    subscriber = message_bus.create_subscriber(name, target_port)
    try:
        yield subscriber
    finally:
        pass  # 不立即关闭，由消息总线统一管理