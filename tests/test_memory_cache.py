"""
内存缓存系统测试套件

全面测试MemoryCachePool的功能、性能和并发安全性。
包括：基础操作、TTL机制、数据类型支持、淘汰策略、并发测试等。
"""

import asyncio
import time
import threading
import pytest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.core.cache import MemoryCachePool, CacheConfig, CacheStats, DataType


class TestMemoryCacheBasics:
    """基础功能测试"""
    
    @pytest.fixture
    def cache(self):
        """创建测试缓存实例"""
        config = CacheConfig(
            max_memory=1024 * 1024,  # 1MB
            max_keys=1000,
            cleanup_interval=1
        )
        return MemoryCachePool(config)
    
    def test_cache_initialization(self, cache):
        """测试缓存初始化"""
        assert cache.config.max_memory == 1024 * 1024
        assert cache.config.max_keys == 1000
        assert cache.dbsize() == 0
        
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.key_count == 0
    
    def test_basic_string_operations(self, cache):
        """测试基础字符串操作"""
        # 设置和获取
        assert cache.set("key1", "value1") is True
        assert cache.get("key1") == "value1"
        
        # 键存在性检查
        assert cache.exists("key1") == 1
        assert cache.exists("nonexistent") == 0
        
        # 删除
        assert cache.delete("key1") == 1
        assert cache.get("key1") is None
        assert cache.exists("key1") == 0
    
    def test_multiple_data_types(self, cache):
        """测试多种数据类型"""
        # 字符串
        cache.set("string_key", "string_value")
        assert cache.get("string_key") == "string_value"
        
        # 数字
        cache.set("int_key", 42)
        assert cache.get("int_key") == 42
        
        cache.set("float_key", 3.14)
        assert cache.get("float_key") == 3.14
        
        # 复杂对象
        complex_obj = {"nested": {"list": [1, 2, 3], "dict": {"a": "b"}}}
        cache.set("complex_key", complex_obj)
        result = cache.get("complex_key")
        assert result == complex_obj
    
    def test_set_options(self, cache):
        """测试set操作的各种选项"""
        # nx选项（仅当键不存在时设置）
        assert cache.set("nx_key", "value1", nx=True) is True
        assert cache.set("nx_key", "value2", nx=True) is False
        assert cache.get("nx_key") == "value1"
        
        # xx选项（仅当键存在时设置）
        assert cache.set("xx_key", "value1", xx=True) is False
        cache.set("xx_key", "value1")
        assert cache.set("xx_key", "value2", xx=True) is True
        assert cache.get("xx_key") == "value2"
    
    def test_keys_pattern_matching(self, cache):
        """测试键模式匹配"""
        # 设置测试数据
        cache.set("user:1:name", "alice")
        cache.set("user:2:name", "bob")
        cache.set("user:1:email", "alice@test.com")
        cache.set("product:1:name", "widget")
        
        # 通配符匹配
        user_keys = cache.keys("user:*")
        assert len(user_keys) == 3
        assert all(key.startswith("user:") for key in user_keys)
        
        name_keys = cache.keys("*:name")
        assert len(name_keys) == 3
        assert all(key.endswith(":name") for key in name_keys)
        
        # 精确匹配
        exact_keys = cache.keys("user:1:*")
        assert len(exact_keys) == 2
        assert all(key.startswith("user:1:") for key in exact_keys)
    
    def test_database_size(self, cache):
        """测试数据库大小统计"""
        assert cache.dbsize() == 0
        
        cache.set("key1", "value1")
        assert cache.dbsize() == 1
        
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert cache.dbsize() == 3
        
        cache.delete("key1", "key2")
        assert cache.dbsize() == 1
        
        cache.flushall()
        assert cache.dbsize() == 0


class TestTTLMechanism:
    """TTL（生存时间）机制测试"""
    
    @pytest.fixture
    def cache(self):
        config = CacheConfig(cleanup_interval=0.1)  # 快速清理
        return MemoryCachePool(config)
    
    def test_expire_with_seconds(self, cache):
        """测试秒级过期时间"""
        cache.set("temp_key", "temp_value", ex=1)  # 1秒过期
        assert cache.get("temp_key") == "temp_value"
        assert cache.ttl("temp_key") <= 1
        
        time.sleep(1.1)
        assert cache.get("temp_key") is None
        assert cache.ttl("temp_key") == -2  # 键不存在
    
    def test_expire_with_milliseconds(self, cache):
        """测试毫秒级过期时间"""
        cache.set("temp_key", "temp_value", px=500)  # 500毫秒过期
        assert cache.get("temp_key") == "temp_value"
        
        time.sleep(0.6)
        assert cache.get("temp_key") is None
    
    def test_expire_command(self, cache):
        """测试expire命令"""
        cache.set("persist_key", "persist_value")
        assert cache.ttl("persist_key") == -1  # 无过期时间
        
        # 设置过期时间
        assert cache.expire("persist_key", 2) is True
        assert 1 <= cache.ttl("persist_key") <= 2
        
        # 对不存在的键设置过期时间
        assert cache.expire("nonexistent", 10) is False
    
    def test_persist_command(self, cache):
        """测试persist命令"""
        cache.set("expire_key", "expire_value", ex=10)
        assert cache.ttl("expire_key") > 0
        
        # 移除过期时间
        assert cache.persist("expire_key") is True
        assert cache.ttl("expire_key") == -1
        
        # 对不存在的键执行persist
        assert cache.persist("nonexistent") is False
    
    @pytest.mark.asyncio
    async def test_automatic_cleanup(self, cache):
        """测试自动清理过期键"""
        # 启动清理任务
        await cache.start_cleanup_task()
        
        # 设置一些会过期的键
        cache.set("expire1", "value1", ex=1)
        cache.set("expire2", "value2", px=800)
        cache.set("persist", "persist_value")
        
        assert cache.dbsize() == 3
        
        # 等待过期
        await asyncio.sleep(1.2)
        
        # 检查自动清理结果
        assert cache.get("expire1") is None
        assert cache.get("expire2") is None
        assert cache.get("persist") == "persist_value"
        
        await cache.stop_cleanup_task()


class TestHashOperations:
    """Hash数据类型操作测试"""
    
    @pytest.fixture
    def cache(self):
        return MemoryCachePool()
    
    def test_hash_basic_operations(self, cache):
        """测试Hash基础操作"""
        # hset和hget
        assert cache.hset("user:1", name="alice", age=25) == 2
        assert cache.hget("user:1", "name") == "alice"
        assert cache.hget("user:1", "age") == 25
        assert cache.hget("user:1", "nonexistent") is None
        
        # hgetall
        user_data = cache.hgetall("user:1")
        assert user_data == {"name": "alice", "age": 25}
        
        # 更新现有字段
        assert cache.hset("user:1", age=26) == 0  # 未添加新字段
        assert cache.hget("user:1", "age") == 26
    
    def test_hash_deletion(self, cache):
        """测试Hash字段删除"""
        cache.hset("user:2", name="bob", age=30, city="NYC")
        
        # 删除单个字段
        assert cache.hdel("user:2", "age") == 1
        assert cache.hget("user:2", "age") is None
        assert cache.hget("user:2", "name") == "bob"
        
        # 删除多个字段
        assert cache.hdel("user:2", "name", "city", "nonexistent") == 2
        assert cache.hgetall("user:2") == {}
    
    def test_hash_on_non_hash_key(self, cache):
        """测试在非Hash键上执行Hash操作"""
        cache.set("string_key", "string_value")
        
        # 应该引发类型错误或返回空
        assert cache.hget("string_key", "field") is None
        
        with pytest.raises(TypeError):
            cache.hset("string_key", field="value")


class TestListOperations:
    """List数据类型操作测试"""
    
    @pytest.fixture
    def cache(self):
        return MemoryCachePool()
    
    def test_list_push_operations(self, cache):
        """测试List push操作"""
        # lpush
        assert cache.lpush("list1", "a") == 1
        assert cache.lpush("list1", "b", "c") == 3  # [c, b, a]
        
        # rpush 
        assert cache.rpush("list1", "d", "e") == 5  # [c, b, a, d, e]
        
        # 验证顺序
        full_list = cache.lrange("list1", 0, -1)
        assert full_list == ["c", "b", "a", "d", "e"]
    
    def test_list_pop_operations(self, cache):
        """测试List pop操作"""
        cache.rpush("list2", "1", "2", "3", "4")
        
        # lpop
        assert cache.lpop("list2") == "1"
        assert cache.llen("list2") == 3
        
        # rpop
        assert cache.rpop("list2") == "4"
        assert cache.llen("list2") == 2
        
        # 弹出剩余元素
        assert cache.lpop("list2") == "2"
        assert cache.rpop("list2") == "3"
        assert cache.llen("list2") == 0
        
        # 空列表弹出
        assert cache.lpop("list2") is None
        assert cache.rpop("list2") is None
    
    def test_list_range_operations(self, cache):
        """测试List范围操作"""
        cache.rpush("list3", *range(10))  # [0,1,2,3,4,5,6,7,8,9]
        
        # 正向索引
        assert cache.lrange("list3", 0, 2) == [0, 1, 2]
        assert cache.lrange("list3", 5, 7) == [5, 6, 7]
        
        # 负向索引
        assert cache.lrange("list3", -3, -1) == [7, 8, 9]
        assert cache.lrange("list3", 0, -1) == list(range(10))
        
        # 超出范围
        assert cache.lrange("list3", 20, 30) == []
        assert cache.lrange("list3", 5, 2) == []  # start > end
    
    def test_list_length(self, cache):
        """测试List长度"""
        assert cache.llen("nonexistent") == 0
        
        cache.lpush("list4", "a", "b", "c")
        assert cache.llen("list4") == 3
        
        cache.lpop("list4")
        assert cache.llen("list4") == 2


class TestSetOperations:
    """Set数据类型操作测试"""
    
    @pytest.fixture
    def cache(self):
        return MemoryCachePool()
    
    def test_set_basic_operations(self, cache):
        """测试Set基础操作"""
        # 添加元素
        assert cache.sadd("set1", "a", "b", "c") == 3
        assert cache.sadd("set1", "b", "d") == 1  # b已存在
        
        # 检查成员
        assert cache.sismember("set1", "a") is True
        assert cache.sismember("set1", "z") is False
        
        # 获取所有成员
        members = cache.smembers("set1")
        assert members == {"a", "b", "c", "d"}
        
        # 集合大小
        assert cache.scard("set1") == 4
    
    def test_set_removal(self, cache):
        """测试Set元素移除"""
        cache.sadd("set2", "x", "y", "z", "w")
        
        # 移除存在的元素
        assert cache.srem("set2", "x", "y") == 2
        assert cache.scard("set2") == 2
        
        # 移除不存在的元素
        assert cache.srem("set2", "nonexistent", "z") == 1
        assert cache.scard("set2") == 1
        
        remaining = cache.smembers("set2")
        assert remaining == {"w"}
    
    def test_set_on_nonexistent_key(self, cache):
        """测试不存在键的Set操作"""
        assert cache.sismember("nonexistent_set", "value") is False
        assert cache.smembers("nonexistent_set") == set()
        assert cache.scard("nonexistent_set") == 0


class TestEvictionPolicies:
    """淘汰策略测试"""
    
    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        config = CacheConfig(
            max_keys=3,
            eviction_policy="lru"
        )
        cache = MemoryCachePool(config)
        
        # 填满缓存
        cache.set("key1", "value1")
        cache.set("key2", "value2")  
        cache.set("key3", "value3")
        
        # 访问key1，使其成为最近使用
        cache.get("key1")
        
        # 添加新键，应该淘汰key2（最久未使用）
        cache.set("key4", "value4")
        
        assert cache.exists("key1") == 1  # 最近访问，保留
        assert cache.exists("key2") == 0  # 最久未使用，被淘汰
        assert cache.exists("key3") == 1  # 保留
        assert cache.exists("key4") == 1  # 新添加
    
    def test_lfu_eviction(self):
        """测试LFU淘汰策略"""
        config = CacheConfig(
            max_keys=3,
            eviction_policy="lfu"
        )
        cache = MemoryCachePool(config)
        
        # 设置初始数据
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # key1访问多次，key2访问一次，key3不访问
        cache.get("key1")
        cache.get("key1") 
        cache.get("key1")  # key1访问次数最多
        cache.get("key2")  # key2访问一次
        
        # 添加新键，应该淘汰key3（访问次数最少）
        cache.set("key4", "value4")
        
        assert cache.exists("key1") == 1  # 访问次数多，保留
        assert cache.exists("key2") == 1  # 有访问，保留
        assert cache.exists("key3") == 0  # 无访问，被淘汰
        assert cache.exists("key4") == 1  # 新添加
    
    def test_memory_limit_eviction(self):
        """测试内存限制触发的淘汰"""
        config = CacheConfig(
            max_memory=1024,  # 1KB限制
            eviction_policy="lru"
        )
        cache = MemoryCachePool(config)
        
        # 添加大量数据直到触发内存限制
        large_value = "x" * 200  # 200字节的值
        initial_keys = []
        
        for i in range(10):
            key = f"large_key_{i}"
            cache.set(key, large_value)
            initial_keys.append(key)
        
        # 检查是否发生了淘汰
        remaining_keys = [key for key in initial_keys if cache.exists(key)]
        assert len(remaining_keys) < len(initial_keys)
        
        # 验证内存使用在限制内
        stats = cache.get_stats()
        assert stats.memory_usage <= config.max_memory


class TestConcurrency:
    """并发安全测试"""
    
    def test_thread_safety_basic_operations(self):
        """测试基础操作的线程安全"""
        cache = MemoryCachePool(CacheConfig(thread_safe=True))
        
        def worker(thread_id):
            """工作线程函数"""
            results = []
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                
                # 设置
                cache.set(key, value)
                
                # 获取
                retrieved = cache.get(key)
                results.append(retrieved == value)
                
                # 删除部分键
                if i % 2 == 0:
                    cache.delete(key)
            
            return results
        
        # 启动多个线程
        threads = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            
            # 等待所有线程完成
            for future in as_completed(futures):
                results = future.result()
                # 所有操作应该成功
                assert all(results)
    
    def test_concurrent_set_get(self):
        """测试并发set/get操作"""
        cache = MemoryCachePool(CacheConfig(thread_safe=True))
        
        def setter():
            """设置线程"""
            for i in range(1000):
                cache.set(f"key_{i}", f"value_{i}")
        
        def getter():
            """获取线程"""
            successful_gets = 0
            for i in range(1000):
                value = cache.get(f"key_{i}")
                if value == f"value_{i}":
                    successful_gets += 1
            return successful_gets
        
        # 启动设置线程和获取线程
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 一个设置线程
            set_future = executor.submit(setter)
            
            # 多个获取线程
            get_futures = [executor.submit(getter) for _ in range(3)]
            
            # 等待设置完成
            set_future.result()
            
            # 等待获取完成，统计成功率
            total_successful = sum(future.result() for future in get_futures)
            # 由于并发，不是所有get都能成功，但应该有相当数量成功
            assert total_successful > 0
    
    def test_concurrent_hash_operations(self):
        """测试Hash操作的并发安全"""
        cache = MemoryCachePool(CacheConfig(thread_safe=True))
        
        def hash_worker(worker_id):
            """Hash操作工作线程"""
            hash_key = f"hash_{worker_id}"
            
            # 设置多个字段
            for i in range(50):
                cache.hset(hash_key, **{f"field_{i}": f"value_{i}"})
            
            # 获取字段
            successful_gets = 0
            for i in range(50):
                value = cache.hget(hash_key, f"field_{i}")
                if value == f"value_{i}":
                    successful_gets += 1
            
            return successful_gets
        
        # 启动多个Hash工作线程
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(hash_worker, i) for i in range(4)]
            
            results = [future.result() for future in futures]
            
            # 每个线程都应该成功操作自己的Hash
            for successful_count in results:
                assert successful_count == 50  # 所有字段操作都应该成功


class TestStatistics:
    """统计功能测试"""
    
    @pytest.fixture
    def cache(self):
        config = CacheConfig(enable_stats=True)
        return MemoryCachePool(config)
    
    def test_hit_miss_statistics(self, cache):
        """测试命中/未命中统计"""
        initial_stats = cache.get_stats()
        assert initial_stats.hits == 0
        assert initial_stats.misses == 0
        
        # 设置和获取（命中）
        cache.set("key1", "value1")
        cache.get("key1")  # 命中
        
        # 获取不存在的键（未命中）
        cache.get("nonexistent")  # 未命中
        
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5
        assert stats.miss_rate == 0.5
    
    def test_operation_counting(self, cache):
        """测试操作计数"""
        initial_ops = cache.get_stats().operations
        
        # 执行多种操作
        cache.set("key1", "value1")  # +1
        cache.get("key1")            # +1
        cache.get("nonexistent")     # +1
        cache.exists("key1")         # 不计入operations
        cache.delete("key1")         # 不计入operations
        
        stats = cache.get_stats()
        # 应该增加3个操作（set, get hit, get miss）
        assert stats.operations == initial_ops + 3
    
    def test_memory_usage_tracking(self, cache):
        """测试内存使用统计"""
        initial_memory = cache.get_stats().memory_usage
        assert initial_memory == 0
        
        # 添加一些数据
        cache.set("key1", "a" * 100)  # 100字节字符串
        cache.set("key2", "b" * 200)  # 200字节字符串
        
        stats = cache.get_stats()
        # 内存使用应该增加（包含键名和值）
        assert stats.memory_usage > initial_memory
        assert stats.key_count == 2
        
        # 删除数据
        cache.delete("key1", "key2")
        stats = cache.get_stats()
        assert stats.key_count == 0
    
    def test_stats_reset(self, cache):
        """测试统计重置"""
        # 执行一些操作产生统计数据
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")
        
        stats = cache.get_stats()
        assert stats.hits > 0
        assert stats.misses > 0
        assert stats.operations > 0
        
        # 重置统计
        cache.reset_stats()
        
        stats = cache.get_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.operations == 0
        # 注意：key_count和memory_usage不会重置


class TestCacheInfo:
    """缓存信息和调试功能测试"""
    
    @pytest.fixture
    def cache(self):
        return MemoryCachePool()
    
    def test_info_command(self, cache):
        """测试info命令"""
        info = cache.info()
        
        # 验证基本结构
        assert "version" in info
        assert "config" in info
        assert "stats" in info
        assert "runtime" in info
        
        # 验证配置信息
        config_info = info["config"]
        assert "max_memory" in config_info
        assert "max_keys" in config_info
        assert "eviction_policy" in config_info
        
        # 验证统计信息
        stats_info = info["stats"]
        assert "hits" in stats_info
        assert "misses" in stats_info
        assert "hit_rate" in stats_info
    
    def test_debug_keys(self, cache):
        """测试debug_keys调试功能"""
        # 添加不同类型的数据
        cache.set("string_key", "string_value", ex=10)
        cache.hset("hash_key", field1="value1")
        cache.lpush("list_key", "item1")
        
        debug_info = cache.debug_keys()
        
        # 应该有3个键的调试信息
        assert len(debug_info) == 3
        
        # 验证调试信息结构
        for key_info in debug_info:
            assert "key" in key_info
            assert "type" in key_info
            assert "ttl" in key_info
            assert "access_count" in key_info
            assert "memory_estimate" in key_info
        
        # 找到string_key的信息验证TTL
        string_key_info = next(info for info in debug_info if info["key"] == "string_key")
        assert string_key_info["type"] == "string"
        assert 0 <= string_key_info["ttl"] <= 10  # TTL应该在合理范围内


class TestPerformance:
    """性能测试"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        cache = MemoryCachePool(CacheConfig(
            max_keys=100000,
            max_memory=50 * 1024 * 1024  # 50MB
        ))
        
        # 性能测试：插入大量数据
        start_time = time.time()
        for i in range(10000):
            cache.set(f"perf_key_{i}", f"value_{i}")
        insert_time = time.time() - start_time
        
        print(f"插入10000条记录耗时: {insert_time:.3f}秒")
        assert insert_time < 5.0  # 应该在5秒内完成
        
        # 性能测试：读取数据
        start_time = time.time()
        hit_count = 0
        for i in range(10000):
            if cache.get(f"perf_key_{i}") is not None:
                hit_count += 1
        read_time = time.time() - start_time
        
        print(f"读取10000条记录耗时: {read_time:.3f}秒, 命中率: {hit_count/10000:.2%}")
        assert read_time < 3.0  # 读取应该更快
        assert hit_count == 10000  # 所有数据都应该命中
    
    def test_concurrent_performance(self):
        """测试并发性能"""
        cache = MemoryCachePool(CacheConfig(
            thread_safe=True,
            max_keys=50000
        ))
        
        def performance_worker(worker_id, operation_count):
            """性能测试工作线程"""
            start_time = time.time()
            
            for i in range(operation_count):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                cache.set(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
            
            return time.time() - start_time
        
        # 启动多个工作线程
        operation_count = 1000
        thread_count = 4
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(performance_worker, i, operation_count) 
                for i in range(thread_count)
            ]
            
            # 等待所有线程完成
            thread_times = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        avg_thread_time = sum(thread_times) / len(thread_times)
        
        print(f"并发性能测试 - 总时间: {total_time:.3f}秒, 平均线程时间: {avg_thread_time:.3f}秒")
        print(f"每秒操作数: {(operation_count * thread_count * 2) / total_time:.0f} ops/sec")
        
        # 性能断言（具体数值可根据实际情况调整）
        assert total_time < 10.0  # 总时间不应超过10秒
        assert avg_thread_time < 8.0  # 平均线程时间不应超过8秒


@pytest.mark.asyncio 
class TestAsyncIntegration:
    """异步集成测试"""
    
    async def test_cleanup_task_lifecycle(self):
        """测试清理任务的生命周期"""
        cache = MemoryCachePool(CacheConfig(cleanup_interval=0.1))
        
        # 启动清理任务
        await cache.start_cleanup_task()
        assert cache._cleanup_task is not None
        assert not cache._cleanup_task.done()
        
        # 添加过期数据
        cache.set("temp1", "value1", ex=1)
        cache.set("temp2", "value2", px=500)
        
        # 等待清理
        await asyncio.sleep(1.2)
        
        # 验证过期数据被清理
        assert cache.get("temp1") is None
        assert cache.get("temp2") is None
        
        # 停止清理任务
        await cache.stop_cleanup_task()
        assert cache._cleanup_task is None
    
    async def test_context_manager_usage(self):
        """测试上下文管理器用法"""
        config = CacheConfig(cleanup_interval=0.1)
        
        async with MemoryCachePool(config) as cache:
            await cache.start_cleanup_task()
            
            cache.set("test_key", "test_value")
            assert cache.get("test_key") == "test_value"
            
        # 上下文退出后，清理任务应该被停止


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])