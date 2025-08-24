"""
内存缓存模块

提供高性能的内存缓存实现，替代Redis中间件，支持：
- 基础Redis操作接口
- TTL和过期数据自动清理
- 线程安全并发访问
- LRU淘汰策略
- 多数据类型支持
- 性能监控统计
"""

from .memory_cache import MemoryCachePool, CacheStats, CacheConfig

__all__ = ["MemoryCachePool", "CacheStats", "CacheConfig"]