"""
环形缓冲区热数据管理
高性能内存中数据缓存，支持线程安全和数据过期
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Generic, TypeVar, Iterator, Union
from threading import RLock
import weakref

from src.utils.logger import LoggerMixin

T = TypeVar('T')


@dataclass
class BufferItem(Generic[T]):
    """缓冲区数据项"""
    data: T
    timestamp: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """检查数据是否过期"""
        if self.expires_at is None:
            return False
        
        current = current_time if current_time is not None else time.time()
        return current >= self.expires_at


class RingBuffer(Generic[T], LoggerMixin):
    """
    高性能环形缓冲区
    
    特性:
    - 线程安全的数据访问
    - 自动数据过期和清理
    - 高性能循环覆盖
    - 支持多种访问模式
    """
    
    def __init__(
        self,
        capacity: int,
        ttl_seconds: Optional[float] = None,
        auto_cleanup: bool = True,
        cleanup_interval: float = 60.0,
        cleanup_threshold: float = 0.8
    ):
        """
        初始化环形缓冲区
        
        Args:
            capacity: 缓冲区容量
            ttl_seconds: 数据生存时间（秒），None表示不过期
            auto_cleanup: 是否自动清理过期数据
            cleanup_interval: 自动清理间隔（秒）
            cleanup_threshold: 触发清理的阈值（使用率）
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self.cleanup_threshold = cleanup_threshold
        
        # 使用 deque 作为底层存储，maxlen 自动实现环形行为
        self._buffer: deque[BufferItem[T]] = deque(maxlen=capacity)
        self._lock = RLock()
        
        # 统计信息
        self._total_inserts = 0
        self._total_expired = 0
        self._total_evicted = 0
        
        # 清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        if auto_cleanup:
            self._start_cleanup_thread()
    
    def __len__(self) -> int:
        """获取当前有效数据数量"""
        with self._lock:
            return len(self._buffer)
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass  # 忽略析构时的错误
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self._cleanup_thread is not None:
            return
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name=f"RingBuffer-Cleanup-{id(self)}"
        )
        self._cleanup_thread.start()
        self.log_debug("Started cleanup thread")
    
    def _cleanup_worker(self):
        """清理工作线程"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                current_usage = len(self._buffer) / self.capacity
                if current_usage >= self.cleanup_threshold:
                    cleaned = self.cleanup_expired()
                    if cleaned > 0:
                        self.log_debug(f"Cleaned {cleaned} expired items")
            except Exception as e:
                self.log_error(f"Error in cleanup worker: {e}")
    
    def put(
        self, 
        data: T, 
        timestamp: Optional[float] = None,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加数据到缓冲区
        
        Args:
            data: 要存储的数据
            timestamp: 数据时间戳，默认为当前时间
            ttl_seconds: 数据TTL，覆盖默认设置
            metadata: 元数据
        
        Returns:
            是否成功添加
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 计算过期时间
        expires_at = None
        if ttl_seconds is not None:
            expires_at = timestamp + ttl_seconds
        elif self.ttl_seconds is not None:
            expires_at = timestamp + self.ttl_seconds
        
        item = BufferItem(
            data=data,
            timestamp=timestamp,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        with self._lock:
            # deque 的 append 会自动处理容量限制
            if len(self._buffer) == self.capacity:
                self._total_evicted += 1
            
            self._buffer.append(item)
            self._total_inserts += 1
        
        return True
    
    def get_latest(self, count: int = 1) -> List[BufferItem[T]]:
        """
        获取最新的N个数据项
        
        Args:
            count: 要获取的数据项数量
        
        Returns:
            最新的数据项列表，按时间倒序排列
        """
        with self._lock:
            # 过滤未过期的数据
            valid_items = [
                item for item in self._buffer 
                if not item.is_expired()
            ]
            
            # 返回最新的 count 个项目
            return list(reversed(valid_items[-count:]))
    
    def get_range(
        self, 
        start_time: float, 
        end_time: float,
        include_expired: bool = False
    ) -> List[BufferItem[T]]:
        """
        获取指定时间范围内的数据
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            include_expired: 是否包含过期数据
        
        Returns:
            时间范围内的数据项列表
        """
        with self._lock:
            result = []
            for item in self._buffer:
                if start_time <= item.timestamp <= end_time:
                    if include_expired or not item.is_expired():
                        result.append(item)
            
            return sorted(result, key=lambda x: x.timestamp)
    
    def get_since(
        self, 
        since_time: float,
        include_expired: bool = False
    ) -> List[BufferItem[T]]:
        """
        获取指定时间点之后的所有数据
        
        Args:
            since_time: 起始时间戳
            include_expired: 是否包含过期数据
        
        Returns:
            指定时间之后的数据项列表
        """
        return self.get_range(since_time, float('inf'), include_expired)
    
    def peek_latest(self) -> Optional[BufferItem[T]]:
        """
        查看最新的数据项，不移除
        
        Returns:
            最新的数据项，如果缓冲区为空则返回None
        """
        with self._lock:
            if not self._buffer:
                return None
            
            # 从最新开始查找未过期的数据
            for item in reversed(self._buffer):
                if not item.is_expired():
                    return item
            
            return None
    
    def cleanup_expired(self, current_time: Optional[float] = None) -> int:
        """
        清理过期数据
        
        Args:
            current_time: 当前时间，用于过期检查
        
        Returns:
            清理的数据项数量
        """
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            initial_size = len(self._buffer)
            
            # 创建新的 deque，只保留未过期的数据
            new_buffer = deque(maxlen=self.capacity)
            
            for item in self._buffer:
                if not item.is_expired(current_time):
                    new_buffer.append(item)
                else:
                    self._total_expired += 1
            
            self._buffer = new_buffer
            cleaned_count = initial_size - len(self._buffer)
            
            if cleaned_count > 0:
                self.log_debug(f"Cleaned {cleaned_count} expired items")
            
            return cleaned_count
    
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self._buffer.clear()
            self.log_debug("Buffer cleared")
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        with self._lock:
            return len(self._buffer) == self.capacity
    
    def usage_ratio(self) -> float:
        """获取缓冲区使用率"""
        with self._lock:
            return len(self._buffer) / self.capacity
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓冲区统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for item in self._buffer 
                if item.is_expired(current_time)
            )
            
            return {
                'capacity': self.capacity,
                'current_size': len(self._buffer),
                'usage_ratio': self.usage_ratio(),
                'expired_count': expired_count,
                'total_inserts': self._total_inserts,
                'total_expired': self._total_expired,
                'total_evicted': self._total_evicted,
                'ttl_seconds': self.ttl_seconds,
                'auto_cleanup': self.auto_cleanup
            }
    
    def __iter__(self) -> Iterator[BufferItem[T]]:
        """迭代器支持"""
        with self._lock:
            # 返回有效数据的副本
            valid_items = [
                item for item in self._buffer 
                if not item.is_expired()
            ]
            return iter(valid_items)
    
    def close(self):
        """关闭缓冲区，停止后台线程"""
        if self._cleanup_thread is not None:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5.0)
            self._cleanup_thread = None
            self.log_debug("Cleanup thread stopped")


class MultiChannelRingBuffer(LoggerMixin):
    """
    多通道环形缓冲区
    
    支持按不同通道（如交易对）分别管理数据
    """
    
    def __init__(
        self,
        capacity_per_channel: int,
        ttl_seconds: Optional[float] = None,
        auto_cleanup: bool = True,
        cleanup_interval: float = 60.0,
        max_channels: int = 1000
    ):
        """
        初始化多通道缓冲区
        
        Args:
            capacity_per_channel: 每个通道的容量
            ttl_seconds: 数据TTL
            auto_cleanup: 是否自动清理
            cleanup_interval: 清理间隔
            max_channels: 最大通道数
        """
        self.capacity_per_channel = capacity_per_channel
        self.ttl_seconds = ttl_seconds
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        self.max_channels = max_channels
        
        self._channels: Dict[str, RingBuffer[T]] = {}
        self._lock = RLock()
        
        # 使用弱引用来自动清理未使用的通道
        self._channel_refs: Dict[str, Any] = {}
    
    def get_channel(self, channel_id: str) -> RingBuffer[T]:
        """
        获取或创建通道
        
        Args:
            channel_id: 通道标识符
        
        Returns:
            对应的环形缓冲区
        """
        with self._lock:
            if channel_id not in self._channels:
                if len(self._channels) >= self.max_channels:
                    raise RuntimeError(f"Maximum channels ({self.max_channels}) exceeded")
                
                self._channels[channel_id] = RingBuffer(
                    capacity=self.capacity_per_channel,
                    ttl_seconds=self.ttl_seconds,
                    auto_cleanup=self.auto_cleanup,
                    cleanup_interval=self.cleanup_interval
                )
                
                self.log_debug(f"Created new channel: {channel_id}")
            
            return self._channels[channel_id]
    
    def put(
        self,
        channel_id: str,
        data: T,
        timestamp: Optional[float] = None,
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        向指定通道添加数据
        
        Args:
            channel_id: 通道标识符
            data: 数据
            timestamp: 时间戳
            ttl_seconds: TTL
            metadata: 元数据
        
        Returns:
            是否成功添加
        """
        channel = self.get_channel(channel_id)
        return channel.put(data, timestamp, ttl_seconds, metadata)
    
    def get_latest(self, channel_id: str, count: int = 1) -> List[BufferItem[T]]:
        """获取指定通道的最新数据"""
        if channel_id not in self._channels:
            return []
        
        return self._channels[channel_id].get_latest(count)
    
    def get_all_channels(self) -> List[str]:
        """获取所有通道ID"""
        with self._lock:
            return list(self._channels.keys())
    
    def cleanup_expired_all(self) -> Dict[str, int]:
        """清理所有通道的过期数据"""
        result = {}
        with self._lock:
            for channel_id, buffer in self._channels.items():
                cleaned = buffer.cleanup_expired()
                if cleaned > 0:
                    result[channel_id] = cleaned
        
        return result
    
    def get_total_stats(self) -> Dict[str, Any]:
        """获取所有通道的统计信息"""
        with self._lock:
            total_size = sum(len(buf) for buf in self._channels.values())
            total_capacity = len(self._channels) * self.capacity_per_channel
            
            channel_stats = {}
            for channel_id, buffer in self._channels.items():
                channel_stats[channel_id] = buffer.get_stats()
            
            return {
                'total_channels': len(self._channels),
                'total_size': total_size,
                'total_capacity': total_capacity,
                'overall_usage_ratio': total_size / total_capacity if total_capacity > 0 else 0,
                'channel_stats': channel_stats
            }
    
    def close(self):
        """关闭所有通道"""
        with self._lock:
            for buffer in self._channels.values():
                buffer.close()
            
            self._channels.clear()
            self.log_info("All channels closed")


# 便利函数和类型别名
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    MarketDataBuffer = RingBuffer[Dict[str, Any]]
    TickDataBuffer = MultiChannelRingBuffer[Dict[str, Any]]
else:
    MarketDataBuffer = RingBuffer
    TickDataBuffer = MultiChannelRingBuffer


def create_market_data_buffer(
    capacity: int = 10000,
    ttl_minutes: float = 60.0,
    auto_cleanup: bool = True
) -> MarketDataBuffer:
    """
    创建市场数据缓冲区的便利函数
    
    Args:
        capacity: 缓冲区容量
        ttl_minutes: 数据TTL（分钟）
        auto_cleanup: 是否自动清理
    
    Returns:
        配置好的市场数据缓冲区
    """
    return RingBuffer(
        capacity=capacity,
        ttl_seconds=ttl_minutes * 60,
        auto_cleanup=auto_cleanup,
        cleanup_interval=30.0,  # 30秒清理一次
        cleanup_threshold=0.8   # 80%使用率触发清理
    )


def create_tick_data_buffer(
    capacity_per_symbol: int = 5000,
    ttl_minutes: float = 30.0,
    max_symbols: int = 100
) -> TickDataBuffer:
    """
    创建tick数据缓冲区的便利函数
    
    Args:
        capacity_per_symbol: 每个交易对的容量
        ttl_minutes: 数据TTL（分钟）
        max_symbols: 最大交易对数量
    
    Returns:
        配置好的tick数据缓冲区
    """
    return MultiChannelRingBuffer(
        capacity_per_channel=capacity_per_symbol,
        ttl_seconds=ttl_minutes * 60,
        auto_cleanup=True,
        cleanup_interval=20.0,  # 20秒清理一次
        max_channels=max_symbols
    )