"""
高性能内存缓存池实现

替代Redis的轻量级内存缓存系统，专为量化交易低延迟需求设计。
支持Redis主要操作、TTL机制、LRU淘汰、并发安全和性能监控。
"""

import time
import json
import asyncio
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import weakref
import pickle
from contextlib import contextmanager

import structlog

logger = structlog.get_logger(__name__)


class DataType(Enum):
    """支持的数据类型"""
    STRING = "string"
    HASH = "hash" 
    LIST = "list"
    SET = "set"
    ZSET = "zset"  # 有序集合


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    data_type: DataType
    expire_at: Optional[float] = None  # Unix时间戳
    access_time: float = field(default_factory=time.time)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expire_at is None:
            return False
        return time.time() > self.expire_at
    
    def touch(self) -> None:
        """更新访问时间和计数"""
        self.access_time = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """缓存配置"""
    max_memory: int = 128 * 1024 * 1024  # 最大内存128MB
    max_keys: int = 1000000  # 最大键数量
    default_ttl: Optional[int] = None  # 默认TTL秒数
    eviction_policy: str = "lru"  # 淘汰策略：lru, lfu, random
    cleanup_interval: int = 60  # 过期清理间隔秒数
    enable_stats: bool = True  # 启用统计
    thread_safe: bool = True  # 线程安全
    compress_threshold: int = 1024  # 压缩阈值字节数
    serialize_complex_types: bool = True  # 序列化复杂类型


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_keys: int = 0
    memory_usage: int = 0
    key_count: int = 0
    operations: int = 0
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_rate
    
    def reset(self) -> None:
        """重置统计"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_keys = 0
        self.operations = 0


class MemoryCachePool:
    """
    高性能内存缓存池
    
    提供Redis兼容的API接口，支持多种数据类型和高级功能：
    - 基础操作：get/set/delete/exists/keys等
    - TTL支持：expire/ttl/persist
    - 数据类型：string/hash/list/set/zset
    - 并发安全：线程锁保护
    - 内存管理：LRU/LFU淘汰策略
    - 性能监控：命中率、内存使用等统计
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._data: Dict[str, CacheEntry] = {}
        self._stats = CacheStats()
        
        # 线程安全
        if self.config.thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = threading.RLock()  # 保持一致的接口
        
        # LRU访问顺序跟踪
        self._access_order: OrderedDict[str, float] = OrderedDict()
        
        # 过期清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = False
        
        # 内存使用估算
        self._estimated_memory = 0
        
        logger.info("MemoryCachePool initialized", 
                   max_memory=self.config.max_memory,
                   max_keys=self.config.max_keys,
                   eviction_policy=self.config.eviction_policy)
    
    @contextmanager
    def _thread_safe(self):
        """线程安全上下文管理器"""
        if self.config.thread_safe:
            with self._lock:
                yield
        else:
            yield
    
    def _update_stats(self, hit: bool = False, miss: bool = False, 
                     eviction: bool = False, expired: bool = False) -> None:
        """更新统计信息"""
        if not self.config.enable_stats:
            return
            
        if hit:
            self._stats.hits += 1
        if miss:
            self._stats.misses += 1
        if eviction:
            self._stats.evictions += 1
        if expired:
            self._stats.expired_keys += 1
        
        self._stats.operations += 1
        self._stats.key_count = len(self._data)
        self._stats.memory_usage = self._estimated_memory
    
    def _estimate_memory(self, key: str, entry: CacheEntry) -> int:
        """估算内存使用"""
        try:
            # 简单的内存估算
            key_size = len(key.encode('utf-8'))
            
            if entry.data_type == DataType.STRING:
                if isinstance(entry.value, str):
                    value_size = len(entry.value.encode('utf-8'))
                else:
                    value_size = len(str(entry.value).encode('utf-8'))
            else:
                # 复杂类型使用pickle序列化大小估算
                value_size = len(pickle.dumps(entry.value))
            
            return key_size + value_size + 100  # 额外开销
        except Exception:
            return 1024  # 默认1KB
    
    def _cleanup_expired_keys(self) -> int:
        """清理过期键，返回清理数量"""
        expired_count = 0
        current_time = time.time()
        expired_keys = []
        
        with self._thread_safe():
            for key, entry in self._data.items():
                if entry.expire_at and current_time > entry.expire_at:
                    expired_keys.append(key)
        
        for key in expired_keys:
            with self._thread_safe():
                if key in self._data:
                    entry = self._data[key]
                    self._estimated_memory -= self._estimate_memory(key, entry)
                    del self._data[key]
                    self._access_order.pop(key, None)
                    expired_count += 1
                    self._update_stats(expired=True)
        
        if expired_count > 0:
            logger.debug("Cleaned up expired keys", count=expired_count)
        
        return expired_count
    
    def _evict_keys(self, target_count: int = 1) -> int:
        """根据淘汰策略移除键"""
        if not self._data:
            return 0
        
        evicted_count = 0
        
        with self._thread_safe():
            if self.config.eviction_policy == "lru":
                # LRU：移除最近最少使用的
                keys_to_evict = []
                for key in self._access_order:
                    if key in self._data:
                        keys_to_evict.append(key)
                        if len(keys_to_evict) >= target_count:
                            break
            
            elif self.config.eviction_policy == "lfu":
                # LFU：移除使用频率最低的
                key_access_counts = [(key, entry.access_count) 
                                   for key, entry in self._data.items()]
                key_access_counts.sort(key=lambda x: x[1])
                keys_to_evict = [key for key, _ in key_access_counts[:target_count]]
            
            else:  # random
                import random
                keys_to_evict = random.sample(list(self._data.keys()), 
                                            min(target_count, len(self._data)))
            
            # 执行淘汰
            for key in keys_to_evict:
                if key in self._data:
                    entry = self._data[key]
                    self._estimated_memory -= self._estimate_memory(key, entry)
                    del self._data[key]
                    self._access_order.pop(key, None)
                    evicted_count += 1
                    self._update_stats(eviction=True)
        
        if evicted_count > 0:
            logger.debug("Evicted keys", 
                        count=evicted_count, 
                        policy=self.config.eviction_policy)
        
        return evicted_count
    
    def _check_memory_limits(self) -> None:
        """检查并处理内存限制"""
        # 检查键数量限制
        if len(self._data) >= self.config.max_keys:
            evict_count = max(1, int(self.config.max_keys * 0.1))  # 淘汰10%
            self._evict_keys(evict_count)
        
        # 检查内存限制
        if self._estimated_memory >= self.config.max_memory:
            # 淘汰直到内存使用降到90%
            target_memory = int(self.config.max_memory * 0.9)
            while self._estimated_memory > target_memory and self._data:
                self._evict_keys(max(1, int(len(self._data) * 0.05)))  # 每次淘汰5%
    
    async def start_cleanup_task(self) -> None:
        """启动异步过期清理任务"""
        if self._cleanup_task is not None:
            return
        
        async def cleanup_loop():
            while not self._stop_cleanup:
                try:
                    self._cleanup_expired_keys()
                    await asyncio.sleep(self.config.cleanup_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Cleanup task error", error=str(e))
                    await asyncio.sleep(5)  # 错误后短暂等待
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Cleanup task started", 
                   interval=self.config.cleanup_interval)
    
    async def stop_cleanup_task(self) -> None:
        """停止清理任务"""
        if self._cleanup_task:
            self._stop_cleanup = True
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Cleanup task stopped")
    
    # ==================== Redis兼容API ====================
    
    def set(self, key: str, value: Any, ex: Optional[int] = None, 
            px: Optional[int] = None, nx: bool = False, xx: bool = False) -> bool:
        """
        设置键值
        
        Args:
            key: 键名
            value: 值
            ex: 过期时间(秒)
            px: 过期时间(毫秒) 
            nx: 仅当键不存在时设置
            xx: 仅当键存在时设置
        """
        with self._thread_safe():
            exists = key in self._data and not self._data[key].is_expired()
            
            if nx and exists:
                return False
            if xx and not exists:
                return False
            
            # 计算过期时间
            expire_at = None
            if ex:
                expire_at = time.time() + ex
            elif px:
                expire_at = time.time() + (px / 1000.0)
            elif self.config.default_ttl:
                expire_at = time.time() + self.config.default_ttl
            
            # 创建缓存条目
            entry = CacheEntry(
                value=value,
                data_type=DataType.STRING,
                expire_at=expire_at
            )
            
            # 移除旧条目的内存占用
            if key in self._data:
                old_entry = self._data[key]
                self._estimated_memory -= self._estimate_memory(key, old_entry)
            
            # 设置新条目
            self._data[key] = entry
            self._access_order[key] = entry.access_time
            self._access_order.move_to_end(key)
            
            # 更新内存估算
            self._estimated_memory += self._estimate_memory(key, entry)
            
            # 检查限制
            self._check_memory_limits()
            
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """获取键值"""
        with self._thread_safe():
            if key not in self._data:
                self._update_stats(miss=True)
                return None
            
            entry = self._data[key]
            
            if entry.is_expired():
                # 过期则删除
                del self._data[key]
                self._access_order.pop(key, None)
                self._estimated_memory -= self._estimate_memory(key, entry)
                self._update_stats(miss=True, expired=True)
                return None
            
            # 更新访问信息
            entry.touch()
            self._access_order[key] = entry.access_time
            self._access_order.move_to_end(key)
            
            self._update_stats(hit=True)
            return entry.value
    
    def delete(self, *keys: str) -> int:
        """删除键，返回删除的数量"""
        deleted_count = 0
        
        with self._thread_safe():
            for key in keys:
                if key in self._data:
                    entry = self._data[key]
                    self._estimated_memory -= self._estimate_memory(key, entry)
                    del self._data[key]
                    self._access_order.pop(key, None)
                    deleted_count += 1
        
        return deleted_count
    
    def exists(self, *keys: str) -> int:
        """检查键是否存在，返回存在的数量"""
        count = 0
        current_time = time.time()
        
        with self._thread_safe():
            for key in keys:
                if (key in self._data and 
                    (self._data[key].expire_at is None or 
                     current_time <= self._data[key].expire_at)):
                    count += 1
        
        return count
    
    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的键列表"""
        import fnmatch
        
        result = []
        current_time = time.time()
        
        with self._thread_safe():
            for key in self._data:
                entry = self._data[key]
                # 检查过期
                if (entry.expire_at is None or current_time <= entry.expire_at):
                    if fnmatch.fnmatch(key, pattern):
                        result.append(key)
        
        return result
    
    def expire(self, key: str, seconds: int) -> bool:
        """设置键过期时间"""
        with self._thread_safe():
            if key not in self._data or self._data[key].is_expired():
                return False
            
            self._data[key].expire_at = time.time() + seconds
            return True
    
    def ttl(self, key: str) -> int:
        """获取键的剩余生存时间"""
        with self._thread_safe():
            if key not in self._data:
                return -2  # 键不存在
            
            entry = self._data[key]
            if entry.expire_at is None:
                return -1  # 无过期时间
            
            remaining = entry.expire_at - time.time()
            return max(0, int(remaining))
    
    def persist(self, key: str) -> bool:
        """移除键的过期时间"""
        with self._thread_safe():
            if key not in self._data or self._data[key].is_expired():
                return False
            
            self._data[key].expire_at = None
            return True
    
    def flushall(self) -> bool:
        """清空所有数据"""
        with self._thread_safe():
            self._data.clear()
            self._access_order.clear()
            self._estimated_memory = 0
            if self.config.enable_stats:
                self._stats.reset()
        
        logger.info("Cache flushed")
        return True
    
    def dbsize(self) -> int:
        """获取键的数量"""
        return len([k for k, v in self._data.items() if not v.is_expired()])
    
    # ==================== Hash操作 ====================
    
    def hset(self, name: str, key: str = None, value: Any = None, 
             mapping: Dict[str, Any] = None, **kwargs) -> int:
        """设置hash字段"""
        if key is not None and value is not None:
            kwargs[key] = value
        if mapping:
            kwargs.update(mapping)
        
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                self._data[name] = CacheEntry(
                    value={}, 
                    data_type=DataType.HASH
                )
                self._access_order[name] = time.time()
            
            entry = self._data[name]
            if entry.data_type != DataType.HASH:
                raise TypeError(f"Key {name} is not a hash")
            
            old_size = len(entry.value)
            entry.value.update(kwargs)
            entry.touch()
            self._access_order.move_to_end(name)
            
            return len(entry.value) - old_size
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        """获取hash字段值"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return None
            
            entry = self._data[name]
            if entry.data_type != DataType.HASH:
                return None
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return entry.value.get(key)
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """获取hash所有字段"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return {}
            
            entry = self._data[name]
            if entry.data_type != DataType.HASH:
                return {}
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return dict(entry.value)
    
    def hdel(self, name: str, *keys: str) -> int:
        """删除hash字段"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return 0
            
            entry = self._data[name]
            if entry.data_type != DataType.HASH:
                return 0
            
            deleted_count = 0
            for key in keys:
                if key in entry.value:
                    del entry.value[key]
                    deleted_count += 1
            
            entry.touch()
            return deleted_count
    
    # ==================== List操作 ====================
    
    def lpush(self, name: str, *values: Any) -> int:
        """从左侧推入列表"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                self._data[name] = CacheEntry(
                    value=[], 
                    data_type=DataType.LIST
                )
                self._access_order[name] = time.time()
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST:
                raise TypeError(f"Key {name} is not a list")
            
            # 从左侧插入（反序）
            for value in reversed(values):
                entry.value.insert(0, value)
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return len(entry.value)
    
    def rpush(self, name: str, *values: Any) -> int:
        """从右侧推入列表"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                self._data[name] = CacheEntry(
                    value=[], 
                    data_type=DataType.LIST
                )
                self._access_order[name] = time.time()
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST:
                raise TypeError(f"Key {name} is not a list")
            
            entry.value.extend(values)
            entry.touch()
            self._access_order.move_to_end(name)
            
            return len(entry.value)
    
    def lpop(self, name: str) -> Optional[Any]:
        """从左侧弹出元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return None
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST or not entry.value:
                return None
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return entry.value.pop(0)
    
    def rpop(self, name: str) -> Optional[Any]:
        """从右侧弹出元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return None
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST or not entry.value:
                return None
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return entry.value.pop()
    
    def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """获取列表范围元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return []
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST:
                return []
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            # Redis风格的索引处理
            length = len(entry.value)
            if start < 0:
                start += length
            if end < 0:
                end += length
            
            start = max(0, start)
            end = min(length - 1, end)
            
            if start > end:
                return []
            
            return entry.value[start:end+1]
    
    def llen(self, name: str) -> int:
        """获取列表长度"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return 0
            
            entry = self._data[name]
            if entry.data_type != DataType.LIST:
                return 0
            
            return len(entry.value)
    
    # ==================== Set操作 ====================
    
    def sadd(self, name: str, *values: Any) -> int:
        """添加集合元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                self._data[name] = CacheEntry(
                    value=set(), 
                    data_type=DataType.SET
                )
                self._access_order[name] = time.time()
            
            entry = self._data[name]
            if entry.data_type != DataType.SET:
                raise TypeError(f"Key {name} is not a set")
            
            old_size = len(entry.value)
            entry.value.update(values)
            entry.touch()
            self._access_order.move_to_end(name)
            
            return len(entry.value) - old_size
    
    def srem(self, name: str, *values: Any) -> int:
        """移除集合元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return 0
            
            entry = self._data[name]
            if entry.data_type != DataType.SET:
                return 0
            
            removed_count = 0
            for value in values:
                if value in entry.value:
                    entry.value.remove(value)
                    removed_count += 1
            
            entry.touch()
            return removed_count
    
    def smembers(self, name: str) -> Set[Any]:
        """获取集合所有元素"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return set()
            
            entry = self._data[name]
            if entry.data_type != DataType.SET:
                return set()
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return set(entry.value)
    
    def sismember(self, name: str, value: Any) -> bool:
        """检查元素是否在集合中"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return False
            
            entry = self._data[name]
            if entry.data_type != DataType.SET:
                return False
            
            entry.touch()
            self._access_order.move_to_end(name)
            
            return value in entry.value
    
    def scard(self, name: str) -> int:
        """获取集合大小"""
        with self._thread_safe():
            if name not in self._data or self._data[name].is_expired():
                return 0
            
            entry = self._data[name]
            if entry.data_type != DataType.SET:
                return 0
            
            return len(entry.value)
    
    # ==================== 统计和监控 ====================
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        stats_copy = CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            expired_keys=self._stats.expired_keys,
            memory_usage=self._estimated_memory,
            key_count=len(self._data),
            operations=self._stats.operations
        )
        return stats_copy
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._thread_safe():
            self._stats.reset()
    
    def info(self) -> Dict[str, Any]:
        """获取缓存详细信息"""
        stats = self.get_stats()
        
        return {
            "version": "1.0.0",
            "config": {
                "max_memory": self.config.max_memory,
                "max_keys": self.config.max_keys,
                "default_ttl": self.config.default_ttl,
                "eviction_policy": self.config.eviction_policy,
                "cleanup_interval": self.config.cleanup_interval,
            },
            "stats": {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions,
                "expired_keys": stats.expired_keys,
                "key_count": stats.key_count,
                "memory_usage": stats.memory_usage,
                "operations": stats.operations,
            },
            "runtime": {
                "uptime": time.time() - self._data.get("__start_time__", time.time()),
                "cleanup_running": self._cleanup_task is not None,
            }
        }
    
    def debug_keys(self) -> List[Dict[str, Any]]:
        """调试：获取所有键的详细信息"""
        result = []
        current_time = time.time()
        
        with self._thread_safe():
            for key, entry in self._data.items():
                ttl = -1
                if entry.expire_at:
                    ttl = max(0, int(entry.expire_at - current_time))
                
                result.append({
                    "key": key,
                    "type": entry.data_type.value,
                    "ttl": ttl,
                    "access_count": entry.access_count,
                    "access_time": entry.access_time,
                    "created_at": entry.created_at,
                    "memory_estimate": self._estimate_memory(key, entry),
                    "is_expired": entry.is_expired()
                })
        
        return result
    
    def __enter__(self):
        """上下文管理器支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理资源"""
        if self._cleanup_task:
            asyncio.create_task(self.stop_cleanup_task())
    
    def __del__(self):
        """析构函数"""
        try:
            if self._cleanup_task:
                self._stop_cleanup = True
        except Exception:
            pass


# 全局默认缓存实例
_default_cache: Optional[MemoryCachePool] = None


def get_default_cache() -> MemoryCachePool:
    """获取全局默认缓存实例"""
    global _default_cache
    if _default_cache is None:
        _default_cache = MemoryCachePool()
    return _default_cache


def set_default_cache(cache: MemoryCachePool) -> None:
    """设置全局默认缓存实例"""
    global _default_cache
    _default_cache = cache