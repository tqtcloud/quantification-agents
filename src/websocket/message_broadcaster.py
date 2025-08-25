"""
消息广播器

负责消息的路由和广播，支持各种推送策略，
包括实时推送、批量推送、优先级队列等。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict, deque
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import heapq

from .models import (
    WebSocketMessage,
    BroadcastMessage,
    MessageType,
    SubscriptionType,
    ConnectionInfo,
    SubscriptionInfo
)


logger = logging.getLogger(__name__)


class BroadcastStrategy(Enum):
    """广播策略"""
    IMMEDIATE = "immediate"        # 立即发送
    BATCHED = "batched"           # 批量发送
    PRIORITY_QUEUE = "priority"   # 优先级队列
    RATE_LIMITED = "rate_limited" # 限流发送


@dataclass
class MessageBuffer:
    """消息缓冲区"""
    messages: deque = field(default_factory=deque)
    last_flush: datetime = field(default_factory=datetime.now)
    max_size: int = 1000
    max_age_seconds: int = 5
    
    def add_message(self, message: BroadcastMessage) -> None:
        """添加消息到缓冲区"""
        if len(self.messages) >= self.max_size:
            self.messages.popleft()  # 移除最旧的消息
        self.messages.append(message)
    
    def should_flush(self) -> bool:
        """判断是否应该刷新缓冲区"""
        if not self.messages:
            return False
        
        # 检查消息数量
        if len(self.messages) >= self.max_size:
            return True
        
        # 检查时间
        age = (datetime.now() - self.last_flush).total_seconds()
        return age >= self.max_age_seconds
    
    def flush(self) -> List[BroadcastMessage]:
        """刷新缓冲区并返回消息列表"""
        messages = list(self.messages)
        self.messages.clear()
        self.last_flush = datetime.now()
        return messages


@dataclass
class PriorityMessage:
    """优先级消息"""
    priority: int
    timestamp: float
    message: BroadcastMessage
    
    def __lt__(self, other):
        # 优先级高的排在前面，时间早的排在前面
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


@dataclass
class RateLimiter:
    """速率限制器"""
    max_messages: int = 100
    window_seconds: int = 60
    message_times: deque = field(default_factory=deque)
    
    def can_send(self) -> bool:
        """检查是否可以发送消息"""
        now = time.time()
        
        # 清理过期时间戳
        while self.message_times and self.message_times[0] < now - self.window_seconds:
            self.message_times.popleft()
        
        # 检查是否超过限制
        return len(self.message_times) < self.max_messages
    
    def record_message(self) -> None:
        """记录消息发送"""
        self.message_times.append(time.time())


class MessageBroadcaster:
    """消息广播器
    
    负责消息的路由和广播，支持多种推送策略：
    - 实时推送：立即发送消息
    - 批量推送：缓冲消息后批量发送
    - 优先级推送：根据优先级排序发送
    - 限流推送：控制发送速率
    """
    
    def __init__(self, connection_manager=None, subscription_manager=None):
        self.connection_manager = connection_manager
        self.subscription_manager = subscription_manager
        
        # 广播策略
        self._default_strategy = BroadcastStrategy.IMMEDIATE
        self._type_strategies: Dict[SubscriptionType, BroadcastStrategy] = {}
        
        # 消息缓冲
        self._message_buffers: Dict[str, MessageBuffer] = defaultdict(MessageBuffer)  # connection_id -> buffer
        
        # 优先级队列
        self._priority_queue: List[PriorityMessage] = []
        self._priority_queue_lock = asyncio.Lock()
        
        # 速率限制
        self._rate_limiters: Dict[str, RateLimiter] = defaultdict(RateLimiter)  # connection_id -> limiter
        
        # 统计信息
        self._stats = {
            'messages_sent': 0,
            'messages_dropped': 0,
            'bytes_sent': 0,
            'broadcast_count': 0,
            'average_latency_ms': 0.0,
            'last_broadcast_time': None
        }
        
        # 后台任务
        self._flush_task: Optional[asyncio.Task] = None
        self._priority_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 消息处理器
        self._message_processors: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        logger.info("消息广播器初始化完成")
    
    async def start(self) -> None:
        """启动消息广播器"""
        logger.info("启动消息广播器...")
        
        # 启动缓冲刷新任务
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        # 启动优先级队列处理任务
        self._priority_task = asyncio.create_task(self._priority_loop())
        
        logger.info("消息广播器启动成功")
    
    async def stop(self) -> None:
        """停止消息广播器"""
        logger.info("停止消息广播器...")
        
        # 设置停止事件
        self._shutdown_event.set()
        
        # 取消后台任务
        if self._flush_task:
            self._flush_task.cancel()
        if self._priority_task:
            self._priority_task.cancel()
        
        # 刷新所有缓冲区
        await self._flush_all_buffers()
        
        logger.info("消息广播器已停止")
    
    def set_strategy(self, subscription_type: SubscriptionType, strategy: BroadcastStrategy) -> None:
        """设置订阅类型的广播策略"""
        self._type_strategies[subscription_type] = strategy
        logger.info(f"设置 {subscription_type.value} 的广播策略为 {strategy.value}")
    
    def add_message_processor(self, message_type: MessageType, processor: Callable) -> None:
        """添加消息处理器"""
        self._message_processors[message_type].append(processor)
    
    async def broadcast_message(self, message: WebSocketMessage, 
                              subscription_type: SubscriptionType,
                              target_connections: Optional[Set[str]] = None,
                              filters: Optional[Dict[str, Any]] = None,
                              priority: int = 0) -> int:
        """广播消息
        
        Args:
            message: 要广播的消息
            subscription_type: 订阅类型
            target_connections: 目标连接ID集合，None表示所有订阅该类型的连接
            filters: 额外过滤条件
            priority: 消息优先级（0=普通, 1=高, 2=紧急）
            
        Returns:
            成功发送的连接数
        """
        start_time = time.time()
        
        try:
            # 获取匹配的订阅
            if target_connections:
                # 按指定连接过滤
                subscriptions = []
                for connection_id in target_connections:
                    conn_subscriptions = self.subscription_manager.get_subscriptions_by_connection(connection_id)
                    for sub in conn_subscriptions:
                        if sub.subscription_type == subscription_type:
                            if not filters or sub.matches_filter(message.data):
                                subscriptions.append(sub)
            else:
                # 获取所有匹配订阅
                subscriptions = self.subscription_manager.get_matching_subscriptions(
                    subscription_type, message.data
                )
            
            if not subscriptions:
                logger.debug(f"没有找到 {subscription_type.value} 类型的匹配订阅")
                return 0
            
            # 创建广播消息
            broadcast_msg = BroadcastMessage(
                message=message,
                target_subscriptions={subscription_type},
                target_connections=target_connections or set(),
                filters=filters or {},
                priority=priority
            )
            
            # 根据策略处理消息
            strategy = self._type_strategies.get(subscription_type, self._default_strategy)
            success_count = await self._process_broadcast(broadcast_msg, subscriptions, strategy)
            
            # 更新统计
            self._stats['broadcast_count'] += 1
            self._stats['last_broadcast_time'] = datetime.now()
            
            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000
            self._update_latency_stats(latency_ms)
            
            logger.debug(f"广播消息 {message.message_type.value} 给 {len(subscriptions)} 个订阅，成功 {success_count} 个")
            return success_count
            
        except Exception as e:
            logger.error(f"广播消息异常: {e}")
            return 0
    
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """发送消息给指定连接
        
        Args:
            connection_id: 连接ID
            message: 消息
            
        Returns:
            是否发送成功
        """
        if not self.connection_manager:
            logger.warning("连接管理器未设置")
            return False
        
        # 检查速率限制
        rate_limiter = self._rate_limiters[connection_id]
        if not rate_limiter.can_send():
            logger.warning(f"连接 {connection_id} 触发速率限制，消息被丢弃")
            self._stats['messages_dropped'] += 1
            return False
        
        # 发送消息
        success = await self.connection_manager.send_message(connection_id, message)
        
        if success:
            # 记录发送
            rate_limiter.record_message()
            self._stats['messages_sent'] += 1
            
            # 计算消息大小
            message_size = len(message.to_json().encode('utf-8'))
            self._stats['bytes_sent'] += message_size
            
            # 更新订阅统计
            if self.subscription_manager:
                # 这里可以根据消息类型映射到订阅类型
                subscription_type = self._get_subscription_type_from_message(message)
                if subscription_type:
                    self.subscription_manager.update_message_stats(subscription_type, message_size)
        
        return success
    
    async def _process_broadcast(self, broadcast_msg: BroadcastMessage, 
                               subscriptions: List[SubscriptionInfo],
                               strategy: BroadcastStrategy) -> int:
        """处理广播消息"""
        if strategy == BroadcastStrategy.IMMEDIATE:
            return await self._broadcast_immediate(broadcast_msg, subscriptions)
        elif strategy == BroadcastStrategy.BATCHED:
            return await self._broadcast_batched(broadcast_msg, subscriptions)
        elif strategy == BroadcastStrategy.PRIORITY_QUEUE:
            return await self._broadcast_priority(broadcast_msg, subscriptions)
        elif strategy == BroadcastStrategy.RATE_LIMITED:
            return await self._broadcast_rate_limited(broadcast_msg, subscriptions)
        else:
            return await self._broadcast_immediate(broadcast_msg, subscriptions)
    
    async def _broadcast_immediate(self, broadcast_msg: BroadcastMessage, 
                                 subscriptions: List[SubscriptionInfo]) -> int:
        """立即广播"""
        success_count = 0
        tasks = []
        
        for subscription in subscriptions:
            tasks.append(self.send_to_connection(subscription.connection_id, broadcast_msg.message))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result is True)
        
        return success_count
    
    async def _broadcast_batched(self, broadcast_msg: BroadcastMessage, 
                               subscriptions: List[SubscriptionInfo]) -> int:
        """批量广播（添加到缓冲区）"""
        for subscription in subscriptions:
            buffer = self._message_buffers[subscription.connection_id]
            buffer.add_message(broadcast_msg)
        
        # 返回添加到缓冲区的订阅数
        return len(subscriptions)
    
    async def _broadcast_priority(self, broadcast_msg: BroadcastMessage, 
                                subscriptions: List[SubscriptionInfo]) -> int:
        """优先级广播（添加到优先级队列）"""
        async with self._priority_queue_lock:
            priority_msg = PriorityMessage(
                priority=broadcast_msg.priority,
                timestamp=time.time(),
                message=broadcast_msg
            )
            heapq.heappush(self._priority_queue, priority_msg)
        
        return len(subscriptions)
    
    async def _broadcast_rate_limited(self, broadcast_msg: BroadcastMessage, 
                                    subscriptions: List[SubscriptionInfo]) -> int:
        """限流广播"""
        success_count = 0
        
        for subscription in subscriptions:
            rate_limiter = self._rate_limiters[subscription.connection_id]
            if rate_limiter.can_send():
                success = await self.send_to_connection(subscription.connection_id, broadcast_msg.message)
                if success:
                    success_count += 1
                    rate_limiter.record_message()
            else:
                self._stats['messages_dropped'] += 1
        
        return success_count
    
    async def _flush_loop(self) -> None:
        """缓冲区刷新循环"""
        logger.info("启动缓冲区刷新任务")
        
        while not self._shutdown_event.is_set():
            try:
                await self._flush_ready_buffers()
                await asyncio.sleep(1)  # 每秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓冲区刷新异常: {e}")
        
        logger.info("缓冲区刷新任务已停止")
    
    async def _priority_loop(self) -> None:
        """优先级队列处理循环"""
        logger.info("启动优先级队列处理任务")
        
        while not self._shutdown_event.is_set():
            try:
                async with self._priority_queue_lock:
                    if self._priority_queue:
                        priority_msg = heapq.heappop(self._priority_queue)
                        
                        # 处理高优先级消息
                        broadcast_msg = priority_msg.message
                        subscriptions = []
                        
                        # 获取匹配的订阅
                        for subscription_type in broadcast_msg.target_subscriptions:
                            type_subscriptions = self.subscription_manager.get_matching_subscriptions(
                                subscription_type, broadcast_msg.message.data
                            )
                            subscriptions.extend(type_subscriptions)
                        
                        if subscriptions:
                            await self._broadcast_immediate(broadcast_msg, subscriptions)
                
                await asyncio.sleep(0.1)  # 高频检查
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"优先级队列处理异常: {e}")
        
        logger.info("优先级队列处理任务已停止")
    
    async def _flush_ready_buffers(self) -> None:
        """刷新准备好的缓冲区"""
        for connection_id, buffer in self._message_buffers.items():
            if buffer.should_flush():
                messages = buffer.flush()
                
                for broadcast_msg in messages:
                    await self.send_to_connection(connection_id, broadcast_msg.message)
    
    async def _flush_all_buffers(self) -> None:
        """刷新所有缓冲区"""
        for connection_id, buffer in self._message_buffers.items():
            messages = buffer.flush()
            
            for broadcast_msg in messages:
                await self.send_to_connection(connection_id, broadcast_msg.message)
        
        logger.info("所有消息缓冲区已刷新")
    
    def _get_subscription_type_from_message(self, message: WebSocketMessage) -> Optional[SubscriptionType]:
        """从消息类型推断订阅类型"""
        message_to_subscription = {
            MessageType.TRADING_SIGNAL: SubscriptionType.TRADING_SIGNALS,
            MessageType.STRATEGY_STATUS: SubscriptionType.STRATEGY_STATUS,
            MessageType.MARKET_DATA: SubscriptionType.MARKET_DATA,
            MessageType.SYSTEM_MONITOR: SubscriptionType.SYSTEM_MONITOR,
            MessageType.ORDER_UPDATE: SubscriptionType.ORDER_UPDATES,
            MessageType.POSITION_UPDATE: SubscriptionType.POSITION_UPDATES,
            MessageType.RISK_ALERT: SubscriptionType.RISK_ALERTS,
            MessageType.PERFORMANCE_METRICS: SubscriptionType.PERFORMANCE_METRICS,
        }
        return message_to_subscription.get(message.message_type)
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """更新延迟统计"""
        current_avg = self._stats['average_latency_ms']
        count = self._stats['broadcast_count']
        
        if count == 1:
            self._stats['average_latency_ms'] = latency_ms
        else:
            # 计算移动平均
            self._stats['average_latency_ms'] = (current_avg * (count - 1) + latency_ms) / count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'buffer_count': len(self._message_buffers),
            'priority_queue_size': len(self._priority_queue),
            'active_rate_limiters': len(self._rate_limiters)
        }
    
    def clear_connection_data(self, connection_id: str) -> None:
        """清理连接相关数据"""
        # 清理缓冲区
        if connection_id in self._message_buffers:
            del self._message_buffers[connection_id]
        
        # 清理速率限制器
        if connection_id in self._rate_limiters:
            del self._rate_limiters[connection_id]
        
        logger.debug(f"清理连接 {connection_id} 的广播数据")
    
    async def handle_system_broadcast(self, message_type: MessageType, data: Dict[str, Any]) -> int:
        """处理系统级广播
        
        Args:
            message_type: 消息类型
            data: 消息数据
            
        Returns:
            成功发送的连接数
        """
        # 创建系统消息
        message = WebSocketMessage.create_data_message(message_type, data)
        
        # 根据消息类型确定订阅类型
        subscription_type = self._get_subscription_type_from_message(message)
        if not subscription_type:
            logger.warning(f"未知的消息类型: {message_type.value}")
            return 0
        
        # 广播消息
        return await self.broadcast_message(
            message=message,
            subscription_type=subscription_type,
            priority=1 if message_type == MessageType.RISK_ALERT else 0
        )