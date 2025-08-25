"""
订阅管理器

管理WebSocket连接的订阅和主题，包括订阅创建、
取消、过滤、统计等功能。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from .models import (
    SubscriptionInfo,
    SubscriptionType,
    SubscriptionStats,
    WebSocketMessage,
    MessageType,
    ConnectionInfo
)


logger = logging.getLogger(__name__)


@dataclass
class SubscriptionFilter:
    """订阅过滤器"""
    symbols: Optional[Set[str]] = None
    strategies: Optional[Set[str]] = None
    risk_levels: Optional[Set[str]] = None
    user_ids: Optional[Set[str]] = None
    custom_filters: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """检查数据是否匹配过滤条件"""
        # 检查交易品种
        if self.symbols and data.get('symbol') not in self.symbols:
            return False
        
        # 检查策略
        if self.strategies and data.get('strategy') not in self.strategies:
            return False
        
        # 检查风险等级
        if self.risk_levels and data.get('risk_level') not in self.risk_levels:
            return False
        
        # 检查用户ID
        if self.user_ids and data.get('user_id') not in self.user_ids:
            return False
        
        # 检查自定义过滤器
        for key, value in self.custom_filters.items():
            if key not in data or data[key] != value:
                return False
        
        return True


class SubscriptionManager:
    """订阅管理器
    
    管理所有WebSocket连接的订阅，包括：
    - 订阅的创建和取消
    - 主题管理和路由
    - 订阅过滤和权限控制
    - 订阅统计和监控
    """
    
    def __init__(self):
        # 订阅存储
        self._subscriptions: Dict[str, SubscriptionInfo] = {}
        self._connection_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # connection_id -> subscription_ids
        self._type_subscriptions: Dict[SubscriptionType, Set[str]] = defaultdict(set)  # type -> subscription_ids
        
        # 统计信息
        self._stats: Dict[SubscriptionType, SubscriptionStats] = {}
        
        # 事件回调
        self._on_subscribe_callbacks: List[Callable] = []
        self._on_unsubscribe_callbacks: List[Callable] = []
        
        # 权限检查器
        self._permission_checker: Optional[Callable] = None
        
        logger.info("订阅管理器初始化完成")
    
    def set_permission_checker(self, checker: Callable[[str, SubscriptionType, Dict[str, Any]], bool]) -> None:
        """设置权限检查器
        
        Args:
            checker: 权限检查函数，接受连接ID、订阅类型和过滤条件，返回是否有权限
        """
        self._permission_checker = checker
    
    def add_subscribe_callback(self, callback: Callable[[str, SubscriptionInfo], None]) -> None:
        """添加订阅回调"""
        self._on_subscribe_callbacks.append(callback)
    
    def add_unsubscribe_callback(self, callback: Callable[[str, str], None]) -> None:
        """添加取消订阅回调"""
        self._on_unsubscribe_callbacks.append(callback)
    
    async def subscribe(self, connection_id: str, subscription_type: SubscriptionType, 
                       filters: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """创建订阅
        
        Args:
            connection_id: 连接ID
            subscription_type: 订阅类型
            filters: 过滤条件
            
        Returns:
            (是否成功, 订阅ID或错误信息)
        """
        try:
            # 权限检查
            if self._permission_checker:
                if not await self._permission_checker(connection_id, subscription_type, filters or {}):
                    return False, "权限不足"
            
            # 创建订阅
            subscription = SubscriptionInfo(
                connection_id=connection_id,
                subscription_type=subscription_type,
                filters=filters or {}
            )
            
            # 存储订阅
            self._subscriptions[subscription.subscription_id] = subscription
            self._connection_subscriptions[connection_id].add(subscription.subscription_id)
            self._type_subscriptions[subscription_type].add(subscription.subscription_id)
            
            # 更新统计
            self._update_subscription_stats(subscription_type, 1)
            
            # 触发回调
            for callback in self._on_subscribe_callbacks:
                try:
                    await callback(connection_id, subscription)
                except Exception as e:
                    logger.error(f"订阅回调执行失败: {e}")
            
            logger.info(f"连接 {connection_id} 订阅 {subscription_type.value}: {subscription.subscription_id}")
            return True, subscription.subscription_id
            
        except Exception as e:
            logger.error(f"创建订阅失败: {e}")
            return False, str(e)
    
    async def unsubscribe(self, connection_id: str, subscription_id: str) -> Tuple[bool, str]:
        """取消订阅
        
        Args:
            connection_id: 连接ID
            subscription_id: 订阅ID
            
        Returns:
            (是否成功, 错误信息)
        """
        try:
            subscription = self._subscriptions.get(subscription_id)
            if not subscription:
                return False, "订阅不存在"
            
            if subscription.connection_id != connection_id:
                return False, "权限不足"
            
            # 移除订阅
            subscription_type = subscription.subscription_type
            del self._subscriptions[subscription_id]
            self._connection_subscriptions[connection_id].discard(subscription_id)
            self._type_subscriptions[subscription_type].discard(subscription_id)
            
            # 更新统计
            self._update_subscription_stats(subscription_type, -1)
            
            # 触发回调
            for callback in self._on_unsubscribe_callbacks:
                try:
                    await callback(connection_id, subscription_id)
                except Exception as e:
                    logger.error(f"取消订阅回调执行失败: {e}")
            
            logger.info(f"连接 {connection_id} 取消订阅: {subscription_id}")
            return True, "取消订阅成功"
            
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False, str(e)
    
    async def unsubscribe_by_type(self, connection_id: str, subscription_type: SubscriptionType) -> int:
        """根据类型取消订阅
        
        Args:
            connection_id: 连接ID
            subscription_type: 订阅类型
            
        Returns:
            取消的订阅数量
        """
        subscription_ids = list(self._connection_subscriptions.get(connection_id, set()))
        canceled_count = 0
        
        for subscription_id in subscription_ids:
            subscription = self._subscriptions.get(subscription_id)
            if subscription and subscription.subscription_type == subscription_type:
                success, _ = await self.unsubscribe(connection_id, subscription_id)
                if success:
                    canceled_count += 1
        
        return canceled_count
    
    async def unsubscribe_all(self, connection_id: str) -> int:
        """取消连接的所有订阅
        
        Args:
            connection_id: 连接ID
            
        Returns:
            取消的订阅数量
        """
        subscription_ids = list(self._connection_subscriptions.get(connection_id, set()))
        canceled_count = 0
        
        for subscription_id in subscription_ids:
            success, _ = await self.unsubscribe(connection_id, subscription_id)
            if success:
                canceled_count += 1
        
        # 清理连接订阅记录
        if connection_id in self._connection_subscriptions:
            del self._connection_subscriptions[connection_id]
        
        logger.info(f"连接 {connection_id} 取消了 {canceled_count} 个订阅")
        return canceled_count
    
    def get_subscriptions_by_type(self, subscription_type: SubscriptionType) -> List[SubscriptionInfo]:
        """获取指定类型的所有订阅"""
        subscription_ids = self._type_subscriptions.get(subscription_type, set())
        return [self._subscriptions[sub_id] for sub_id in subscription_ids if sub_id in self._subscriptions]
    
    def get_subscriptions_by_connection(self, connection_id: str) -> List[SubscriptionInfo]:
        """获取指定连接的所有订阅"""
        subscription_ids = self._connection_subscriptions.get(connection_id, set())
        return [self._subscriptions[sub_id] for sub_id in subscription_ids if sub_id in self._subscriptions]
    
    def get_subscription(self, subscription_id: str) -> Optional[SubscriptionInfo]:
        """获取订阅信息"""
        return self._subscriptions.get(subscription_id)
    
    def get_matching_subscriptions(self, subscription_type: SubscriptionType, 
                                 data: Dict[str, Any]) -> List[SubscriptionInfo]:
        """获取匹配数据的订阅列表
        
        Args:
            subscription_type: 订阅类型
            data: 要匹配的数据
            
        Returns:
            匹配的订阅列表
        """
        subscriptions = self.get_subscriptions_by_type(subscription_type)
        matching_subscriptions = []
        
        for subscription in subscriptions:
            if subscription.matches_filter(data):
                matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    def get_connection_count_by_type(self, subscription_type: SubscriptionType) -> int:
        """获取指定类型的订阅连接数"""
        subscription_ids = self._type_subscriptions.get(subscription_type, set())
        connection_ids = set()
        
        for sub_id in subscription_ids:
            subscription = self._subscriptions.get(sub_id)
            if subscription:
                connection_ids.add(subscription.connection_id)
        
        return len(connection_ids)
    
    def get_subscription_stats(self, subscription_type: Optional[SubscriptionType] = None) -> Dict[SubscriptionType, SubscriptionStats]:
        """获取订阅统计"""
        if subscription_type:
            return {subscription_type: self._stats.get(subscription_type, SubscriptionStats(subscription_type))}
        return self._stats.copy()
    
    def get_total_subscriptions(self) -> int:
        """获取总订阅数"""
        return len(self._subscriptions)
    
    def get_active_types(self) -> Set[SubscriptionType]:
        """获取有活跃订阅的类型"""
        return set(sub_type for sub_type, sub_ids in self._type_subscriptions.items() if sub_ids)
    
    def update_message_stats(self, subscription_type: SubscriptionType, message_size: int) -> None:
        """更新消息统计"""
        if subscription_type not in self._stats:
            self._stats[subscription_type] = SubscriptionStats(subscription_type)
        
        stats = self._stats[subscription_type]
        stats.total_messages += 1
        stats.last_message_time = datetime.now()
        
        # 更新平均消息大小
        if stats.total_messages == 1:
            stats.average_message_size = message_size
        else:
            stats.average_message_size = int(
                (stats.average_message_size * (stats.total_messages - 1) + message_size) / stats.total_messages
            )
    
    def _update_subscription_stats(self, subscription_type: SubscriptionType, delta: int) -> None:
        """更新订阅统计"""
        if subscription_type not in self._stats:
            self._stats[subscription_type] = SubscriptionStats(subscription_type)
        
        self._stats[subscription_type].subscriber_count = max(0, 
            self._stats[subscription_type].subscriber_count + delta)
    
    async def cleanup_expired_subscriptions(self, max_age_hours: int = 24) -> int:
        """清理过期订阅
        
        Args:
            max_age_hours: 订阅最大存活时间（小时）
            
        Returns:
            清理的订阅数量
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        expired_subscriptions = []
        
        for subscription_id, subscription in self._subscriptions.items():
            if subscription.created_at < cutoff_time:
                expired_subscriptions.append(subscription_id)
        
        cleaned_count = 0
        for subscription_id in expired_subscriptions:
            subscription = self._subscriptions.get(subscription_id)
            if subscription:
                success, _ = await self.unsubscribe(subscription.connection_id, subscription_id)
                if success:
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个过期订阅")
        
        return cleaned_count
    
    async def handle_subscribe_message(self, connection_id: str, message: WebSocketMessage) -> WebSocketMessage:
        """处理订阅消息
        
        Args:
            connection_id: 连接ID
            message: 订阅消息
            
        Returns:
            响应消息
        """
        try:
            data = message.data
            subscription_type_str = data.get('type')
            if not subscription_type_str:
                return WebSocketMessage.create_control_message(
                    MessageType.ERROR,
                    {"error": "缺少订阅类型"}
                )
            
            try:
                subscription_type = SubscriptionType(subscription_type_str)
            except ValueError:
                return WebSocketMessage.create_control_message(
                    MessageType.ERROR,
                    {"error": f"无效的订阅类型: {subscription_type_str}"}
                )
            
            filters = data.get('filters', {})
            success, result = await self.subscribe(connection_id, subscription_type, filters)
            
            if success:
                return WebSocketMessage.create_control_message(
                    MessageType.SUBSCRIBE,
                    {
                        "success": True,
                        "subscription_id": result,
                        "type": subscription_type.value
                    }
                )
            else:
                return WebSocketMessage.create_control_message(
                    MessageType.ERROR,
                    {"error": result}
                )
            
        except Exception as e:
            logger.error(f"处理订阅消息异常: {e}")
            return WebSocketMessage.create_control_message(
                MessageType.ERROR,
                {"error": "订阅处理异常"}
            )
    
    async def handle_unsubscribe_message(self, connection_id: str, message: WebSocketMessage) -> WebSocketMessage:
        """处理取消订阅消息
        
        Args:
            connection_id: 连接ID
            message: 取消订阅消息
            
        Returns:
            响应消息
        """
        try:
            data = message.data
            subscription_id = data.get('subscription_id')
            subscription_type_str = data.get('type')
            
            if subscription_id:
                # 取消指定订阅
                success, result = await self.unsubscribe(connection_id, subscription_id)
                return WebSocketMessage.create_control_message(
                    MessageType.UNSUBSCRIBE,
                    {
                        "success": success,
                        "subscription_id": subscription_id,
                        "message": result
                    }
                )
            elif subscription_type_str:
                # 取消指定类型的所有订阅
                try:
                    subscription_type = SubscriptionType(subscription_type_str)
                    count = await self.unsubscribe_by_type(connection_id, subscription_type)
                    return WebSocketMessage.create_control_message(
                        MessageType.UNSUBSCRIBE,
                        {
                            "success": True,
                            "type": subscription_type.value,
                            "count": count,
                            "message": f"取消了 {count} 个订阅"
                        }
                    )
                except ValueError:
                    return WebSocketMessage.create_control_message(
                        MessageType.ERROR,
                        {"error": f"无效的订阅类型: {subscription_type_str}"}
                    )
            else:
                # 取消所有订阅
                count = await self.unsubscribe_all(connection_id)
                return WebSocketMessage.create_control_message(
                    MessageType.UNSUBSCRIBE,
                    {
                        "success": True,
                        "count": count,
                        "message": f"取消了所有 {count} 个订阅"
                    }
                )
            
        except Exception as e:
            logger.error(f"处理取消订阅消息异常: {e}")
            return WebSocketMessage.create_control_message(
                MessageType.ERROR,
                {"error": "取消订阅处理异常"}
            )