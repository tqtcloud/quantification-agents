"""
内存缓存性能基准测试

专门用于性能和并发压力测试，模拟量化交易系统的高频访问模式。
包括：吞吐量测试、延迟测试、内存使用测试、并发压力测试等。
"""

import time
import threading
import asyncio
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
import pytest

from src.core.cache import MemoryCachePool, CacheConfig


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, cache: MemoryCachePool):
        self.cache = cache
        self.results: Dict[str, List[float]] = {}
    
    def time_operation(self, name: str, operation_func, iterations: int = 10000):
        """计时操作执行"""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            operation_func()
            end = time.perf_counter()
            times.append((end - start) * 1000000)  # 转换为微秒
        
        self.results[name] = times
        
        return {
            'avg_us': statistics.mean(times),
            'min_us': min(times),
            'max_us': max(times),
            'p50_us': statistics.median(times),
            'p95_us': self._percentile(times, 95),
            'p99_us': self._percentile(times, 99),
            'ops_per_sec': 1000000 / statistics.mean(times)  # 每秒操作数
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_results(self):
        """打印性能结果"""
        print("\n" + "="*60)
        print("内存缓存性能基准测试结果")
        print("="*60)
        
        for operation, times in self.results.items():
            stats = {
                'avg_us': statistics.mean(times),
                'min_us': min(times),
                'max_us': max(times),
                'p95_us': self._percentile(times, 95),
                'p99_us': self._percentile(times, 99),
                'ops_per_sec': 1000000 / statistics.mean(times)
            }
            
            print(f"\n{operation}:")
            print(f"  平均延迟: {stats['avg_us']:.2f} μs")
            print(f"  最小延迟: {stats['min_us']:.2f} μs")
            print(f"  最大延迟: {stats['max_us']:.2f} μs") 
            print(f"  P95延迟:  {stats['p95_us']:.2f} μs")
            print(f"  P99延迟:  {stats['p99_us']:.2f} μs")
            print(f"  吞吐量:   {stats['ops_per_sec']:.0f} ops/sec")


class TestBasicOperationBenchmarks:
    """基础操作性能测试"""
    
    def test_set_operation_performance(self):
        """测试SET操作性能"""
        cache = MemoryCachePool(CacheConfig(max_keys=100000))
        benchmark = PerformanceBenchmark(cache)
        
        counter = 0
        def set_operation():
            nonlocal counter
            cache.set(f"bench_key_{counter}", f"value_{counter}")
            counter += 1
        
        result = benchmark.time_operation("SET操作", set_operation, 10000)
        
        # 性能要求：平均延迟小于100微秒，吞吐量大于10000 ops/sec
        assert result['avg_us'] < 100, f"SET平均延迟过高: {result['avg_us']:.2f}μs"
        assert result['ops_per_sec'] > 10000, f"SET吞吐量过低: {result['ops_per_sec']:.0f}ops/sec"
        
        benchmark.print_results()
    
    def test_get_operation_performance(self):
        """测试GET操作性能"""
        cache = MemoryCachePool(CacheConfig(max_keys=100000))
        
        # 预填充数据
        for i in range(10000):
            cache.set(f"get_bench_key_{i}", f"value_{i}")
        
        benchmark = PerformanceBenchmark(cache)
        
        counter = 0
        def get_operation():
            nonlocal counter
            cache.get(f"get_bench_key_{counter % 10000}")
            counter += 1
        
        result = benchmark.time_operation("GET操作", get_operation, 10000)
        
        # GET应该比SET更快
        assert result['avg_us'] < 50, f"GET平均延迟过高: {result['avg_us']:.2f}μs"
        assert result['ops_per_sec'] > 20000, f"GET吞吐量过低: {result['ops_per_sec']:.0f}ops/sec"
        
        benchmark.print_results()
    
    def test_mixed_operations_performance(self):
        """测试混合操作性能（模拟实际使用场景）"""
        cache = MemoryCachePool(CacheConfig(max_keys=50000))
        benchmark = PerformanceBenchmark(cache)
        
        # 预填充一些数据
        for i in range(1000):
            cache.set(f"mixed_key_{i}", f"value_{i}")
        
        counter = 0
        def mixed_operation():
            nonlocal counter
            key = f"mixed_key_{counter % 2000}"
            
            # 70%读，20%写，10%删除（模拟量化交易场景）
            operation = random.choices(
                ['get', 'set', 'delete'], 
                weights=[70, 20, 10]
            )[0]
            
            if operation == 'get':
                cache.get(key)
            elif operation == 'set':
                cache.set(key, f"new_value_{counter}")
            else:  # delete
                cache.delete(key)
            
            counter += 1
        
        result = benchmark.time_operation("混合操作", mixed_operation, 10000)
        
        # 混合操作性能要求
        assert result['avg_us'] < 150, f"混合操作平均延迟过高: {result['avg_us']:.2f}μs"
        assert result['p99_us'] < 1000, f"P99延迟过高: {result['p99_us']:.2f}μs"
        
        benchmark.print_results()


class TestDataTypePerformance:
    """不同数据类型操作性能测试"""
    
    def test_hash_operations_performance(self):
        """测试Hash操作性能"""
        cache = MemoryCachePool(CacheConfig(max_keys=10000))
        benchmark = PerformanceBenchmark(cache)
        
        # 预创建一些hash
        for i in range(100):
            cache.hset(f"hash_{i}", field1="value1", field2="value2")
        
        counter = 0
        def hash_operation():
            nonlocal counter
            hash_key = f"hash_{counter % 100}"
            field = f"field_{counter % 10}"
            
            # 50%读，50%写
            if counter % 2 == 0:
                cache.hget(hash_key, field)
            else:
                cache.hset(hash_key, **{field: f"value_{counter}"})
            counter += 1
        
        result = benchmark.time_operation("Hash操作", hash_operation, 5000)
        
        # Hash操作可能稍慢，但仍需保持高性能
        assert result['avg_us'] < 200, f"Hash操作平均延迟过高: {result['avg_us']:.2f}μs"
        
        benchmark.print_results()
    
    def test_list_operations_performance(self):
        """测试List操作性能"""
        cache = MemoryCachePool(CacheConfig(max_keys=10000))
        benchmark = PerformanceBenchmark(cache)
        
        # 预创建一些列表
        for i in range(100):
            cache.lpush(f"list_{i}", *[f"item_{j}" for j in range(10)])
        
        counter = 0
        def list_operation():
            nonlocal counter
            list_key = f"list_{counter % 100}"
            
            # 平均分配各种操作
            op = counter % 4
            if op == 0:
                cache.lpush(list_key, f"new_item_{counter}")
            elif op == 1:
                cache.rpush(list_key, f"new_item_{counter}")
            elif op == 2:
                cache.lpop(list_key)
            else:
                cache.lrange(list_key, 0, 5)
            
            counter += 1
        
        result = benchmark.time_operation("List操作", list_operation, 5000)
        
        # List操作性能要求
        assert result['avg_us'] < 300, f"List操作平均延迟过高: {result['avg_us']:.2f}μs"
        
        benchmark.print_results()


class TestConcurrencyBenchmarks:
    """并发性能测试"""
    
    def test_concurrent_reads(self):
        """测试并发读性能"""
        cache = MemoryCachePool(CacheConfig(thread_safe=True, max_keys=100000))
        
        # 预填充数据
        for i in range(10000):
            cache.set(f"read_key_{i}", f"value_{i}")
        
        def reader_worker(worker_id: int, iterations: int) -> Tuple[float, int]:
            """读工作线程"""
            start_time = time.perf_counter()
            successful_reads = 0
            
            for i in range(iterations):
                key = f"read_key_{(worker_id * 1000 + i) % 10000}"
                if cache.get(key) is not None:
                    successful_reads += 1
            
            end_time = time.perf_counter()
            return end_time - start_time, successful_reads
        
        # 启动多个并发读线程
        thread_count = 8
        iterations_per_thread = 5000
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(reader_worker, i, iterations_per_thread)
                for i in range(thread_count)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        total_operations = thread_count * iterations_per_thread
        total_successful = sum(successful for _, successful in results)
        
        throughput = total_operations / total_time
        success_rate = total_successful / total_operations
        
        print(f"\n并发读测试结果:")
        print(f"  线程数: {thread_count}")
        print(f"  总操作数: {total_operations}")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  吞吐量: {throughput:.0f} ops/sec")
        print(f"  成功率: {success_rate:.2%}")
        
        # 并发读性能要求
        assert throughput > 50000, f"并发读吞吐量过低: {throughput:.0f}ops/sec"
        assert success_rate > 0.99, f"并发读成功率过低: {success_rate:.2%}"
    
    def test_concurrent_writes(self):
        """测试并发写性能"""
        cache = MemoryCachePool(CacheConfig(
            thread_safe=True,
            max_keys=100000,
            eviction_policy="lru"
        ))
        
        def writer_worker(worker_id: int, iterations: int) -> Tuple[float, int]:
            """写工作线程"""
            start_time = time.perf_counter()
            successful_writes = 0
            
            for i in range(iterations):
                key = f"write_worker_{worker_id}_key_{i}"
                value = f"value_{worker_id}_{i}"
                if cache.set(key, value):
                    successful_writes += 1
            
            end_time = time.perf_counter()
            return end_time - start_time, successful_writes
        
        # 启动多个并发写线程
        thread_count = 4  # 写操作通常比读操作更重
        iterations_per_thread = 2500
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(writer_worker, i, iterations_per_thread)
                for i in range(thread_count)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        total_operations = thread_count * iterations_per_thread
        total_successful = sum(successful for _, successful in results)
        
        throughput = total_operations / total_time
        success_rate = total_successful / total_operations
        
        print(f"\n并发写测试结果:")
        print(f"  线程数: {thread_count}")
        print(f"  总操作数: {total_operations}")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  吞吐量: {throughput:.0f} ops/sec")
        print(f"  成功率: {success_rate:.2%}")
        
        # 并发写性能要求
        assert throughput > 15000, f"并发写吞吐量过低: {throughput:.0f}ops/sec"
        assert success_rate > 0.99, f"并发写成功率过低: {success_rate:.2%}"
    
    def test_mixed_concurrent_operations(self):
        """测试混合并发操作"""
        cache = MemoryCachePool(CacheConfig(
            thread_safe=True,
            max_keys=50000,
            eviction_policy="lru"
        ))
        
        # 预填充数据
        for i in range(5000):
            cache.set(f"mixed_key_{i}", f"initial_value_{i}")
        
        def mixed_worker(worker_id: int, iterations: int) -> Dict[str, int]:
            """混合操作工作线程"""
            counters = {'get': 0, 'set': 0, 'delete': 0, 'hash': 0}
            
            for i in range(iterations):
                # 根据工作线程类型执行不同比例的操作
                if worker_id % 3 == 0:  # 读密集型
                    operation = random.choices(
                        ['get', 'set', 'delete', 'hash'],
                        weights=[80, 10, 5, 5]
                    )[0]
                elif worker_id % 3 == 1:  # 写密集型
                    operation = random.choices(
                        ['get', 'set', 'delete', 'hash'],
                        weights=[40, 40, 10, 10]
                    )[0]
                else:  # 平衡型
                    operation = random.choices(
                        ['get', 'set', 'delete', 'hash'],
                        weights=[60, 25, 10, 5]
                    )[0]
                
                key = f"mixed_key_{(worker_id * 1000 + i) % 10000}"
                
                if operation == 'get':
                    cache.get(key)
                    counters['get'] += 1
                elif operation == 'set':
                    cache.set(key, f"new_value_{worker_id}_{i}")
                    counters['set'] += 1
                elif operation == 'delete':
                    cache.delete(key)
                    counters['delete'] += 1
                else:  # hash
                    hash_key = f"hash_{key}"
                    cache.hset(hash_key, field=f"value_{i}")
                    counters['hash'] += 1
            
            return counters
        
        # 启动混合工作线程
        thread_count = 6
        iterations_per_thread = 2000
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(mixed_worker, i, iterations_per_thread)
                for i in range(thread_count)
            ]
            
            results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        total_operations = thread_count * iterations_per_thread
        
        # 汇总操作统计
        total_counters = {'get': 0, 'set': 0, 'delete': 0, 'hash': 0}
        for result in results:
            for op, count in result.items():
                total_counters[op] += count
        
        throughput = total_operations / total_time
        
        print(f"\n混合并发操作测试结果:")
        print(f"  线程数: {thread_count}")
        print(f"  总操作数: {total_operations}")
        print(f"  总耗时: {total_time:.3f}s")
        print(f"  总吞吐量: {throughput:.0f} ops/sec")
        print("  操作分布:")
        for op, count in total_counters.items():
            print(f"    {op.upper()}: {count} ({count/total_operations:.1%})")
        
        # 获取最终缓存统计
        cache_stats = cache.get_stats()
        print(f"  最终统计:")
        print(f"    键数量: {cache_stats.key_count}")
        print(f"    命中率: {cache_stats.hit_rate:.2%}")
        print(f"    内存使用: {cache_stats.memory_usage//1024}KB")
        
        # 混合并发操作性能要求
        assert throughput > 20000, f"混合并发吞吐量过低: {throughput:.0f}ops/sec"
        assert cache_stats.hit_rate > 0.3, f"命中率过低: {cache_stats.hit_rate:.2%}"


class TestMemoryUsageBenchmarks:
    """内存使用基准测试"""
    
    def test_memory_efficiency(self):
        """测试内存使用效率"""
        cache = MemoryCachePool(CacheConfig(
            max_memory=10 * 1024 * 1024,  # 10MB
            max_keys=100000
        ))
        
        # 测试不同大小数据的内存效率
        test_cases = [
            ("小对象(50字节)", "x" * 50, 1000),
            ("中等对象(500字节)", "y" * 500, 500),
            ("大对象(5KB)", "z" * 5000, 100),
        ]
        
        print(f"\n内存效率测试:")
        
        for case_name, value, count in test_cases:
            # 清空缓存
            cache.flushall()
            
            # 记录初始内存
            initial_memory = cache.get_stats().memory_usage
            
            # 插入测试数据
            start_time = time.perf_counter()
            for i in range(count):
                cache.set(f"mem_test_{i}", value)
            insert_time = time.perf_counter() - start_time
            
            # 检查内存使用
            final_stats = cache.get_stats()
            actual_memory = final_stats.memory_usage - initial_memory
            theoretical_memory = len(value) * count  # 理论最小内存
            
            efficiency = theoretical_memory / actual_memory if actual_memory > 0 else 0
            
            print(f"  {case_name}:")
            print(f"    对象数量: {count}")
            print(f"    理论内存: {theoretical_memory//1024}KB")
            print(f"    实际内存: {actual_memory//1024}KB")
            print(f"    内存效率: {efficiency:.2%}")
            print(f"    插入耗时: {insert_time:.3f}s")
            print(f"    插入速率: {count/insert_time:.0f} objects/sec")
            
            # 内存效率要求（考虑键名、元数据等开销）
            assert efficiency > 0.5, f"{case_name}内存效率过低: {efficiency:.2%}"
    
    def test_memory_limit_enforcement(self):
        """测试内存限制执行"""
        memory_limit = 1024 * 1024  # 1MB限制
        cache = MemoryCachePool(CacheConfig(
            max_memory=memory_limit,
            eviction_policy="lru"
        ))
        
        # 持续添加数据直到触发内存限制
        large_value = "x" * 1000  # 1KB对象
        keys_added = 0
        
        while keys_added < 2000:  # 最多添加2000个对象（理论2MB）
            cache.set(f"limit_test_{keys_added}", large_value)
            keys_added += 1
            
            stats = cache.get_stats()
            # 检查内存使用是否超过限制
            if stats.memory_usage > memory_limit * 1.1:  # 允许10%缓冲
                break
        
        final_stats = cache.get_stats()
        
        print(f"\n内存限制执行测试:")
        print(f"  内存限制: {memory_limit//1024}KB")
        print(f"  实际内存: {final_stats.memory_usage//1024}KB")
        print(f"  添加对象: {keys_added}")
        print(f"  最终键数: {final_stats.key_count}")
        print(f"  淘汰次数: {final_stats.evictions}")
        
        # 内存限制应该被有效执行
        assert final_stats.memory_usage <= memory_limit * 1.2, "内存使用超出限制过多"
        assert final_stats.evictions > 0, "应该发生淘汰操作"


class TestLatencyDistribution:
    """延迟分布测试（专业性能分析）"""
    
    def test_operation_latency_distribution(self):
        """测试操作延迟分布"""
        cache = MemoryCachePool(CacheConfig(max_keys=50000))
        
        # 预填充数据
        for i in range(10000):
            cache.set(f"lat_key_{i}", f"value_{i}")
        
        # 收集延迟数据
        get_latencies = []
        set_latencies = []
        
        for i in range(5000):
            # 测试GET延迟
            start = time.perf_counter_ns()
            cache.get(f"lat_key_{i % 10000}")
            get_latencies.append((time.perf_counter_ns() - start) / 1000)  # 微秒
            
            # 测试SET延迟
            start = time.perf_counter_ns()
            cache.set(f"new_lat_key_{i}", f"new_value_{i}")
            set_latencies.append((time.perf_counter_ns() - start) / 1000)  # 微秒
        
        # 分析延迟分布
        def analyze_latencies(latencies: List[float], operation: str):
            latencies.sort()
            n = len(latencies)
            
            percentiles = {
                'P50': latencies[int(n * 0.5)],
                'P90': latencies[int(n * 0.9)],
                'P95': latencies[int(n * 0.95)],
                'P99': latencies[int(n * 0.99)],
                'P99.9': latencies[int(n * 0.999)],
            }
            
            print(f"\n{operation}操作延迟分布:")
            print(f"  平均延迟: {statistics.mean(latencies):.2f}μs")
            print(f"  最小延迟: {min(latencies):.2f}μs")
            print(f"  最大延迟: {max(latencies):.2f}μs")
            for p, value in percentiles.items():
                print(f"  {p}延迟: {value:.2f}μs")
            
            return percentiles
        
        get_percentiles = analyze_latencies(get_latencies, "GET")
        set_percentiles = analyze_latencies(set_latencies, "SET")
        
        # 延迟要求（针对高频交易系统）
        assert get_percentiles['P99'] < 100, f"GET P99延迟过高: {get_percentiles['P99']:.2f}μs"
        assert set_percentiles['P99'] < 200, f"SET P99延迟过高: {set_percentiles['P99']:.2f}μs"
        assert get_percentiles['P99.9'] < 500, f"GET P99.9延迟过高: {get_percentiles['P99.9']:.2f}μs"


@pytest.mark.asyncio
class TestAsyncPerformance:
    """异步性能测试"""
    
    async def test_cleanup_performance_impact(self):
        """测试清理任务对性能的影响"""
        cache = MemoryCachePool(CacheConfig(
            cleanup_interval=0.1,  # 高频清理
            max_keys=10000
        ))
        
        # 启动清理任务
        await cache.start_cleanup_task()
        
        # 在清理任务运行期间测试性能
        def mixed_operations():
            operations = 0
            start_time = time.perf_counter()
            
            for i in range(1000):
                # 混合操作，包括一些会过期的键
                if i % 3 == 0:
                    cache.set(f"temp_key_{i}", f"temp_value_{i}", ex=1)
                else:
                    cache.set(f"perm_key_{i}", f"perm_value_{i}")
                
                # 随机读取
                cache.get(f"perm_key_{i // 2}")
                operations += 2
            
            return operations / (time.perf_counter() - start_time)
        
        # 测试有清理任务时的性能
        throughput_with_cleanup = mixed_operations()
        
        # 等待一个清理周期
        await asyncio.sleep(0.2)
        
        # 停止清理任务
        await cache.stop_cleanup_task()
        
        # 测试无清理任务时的性能
        throughput_without_cleanup = mixed_operations()
        
        performance_impact = (throughput_without_cleanup - throughput_with_cleanup) / throughput_without_cleanup
        
        print(f"\n清理任务性能影响测试:")
        print(f"  有清理任务: {throughput_with_cleanup:.0f} ops/sec")
        print(f"  无清理任务: {throughput_without_cleanup:.0f} ops/sec")
        print(f"  性能影响: {performance_impact:.2%}")
        
        # 清理任务的性能影响应该在可接受范围内
        assert performance_impact < 0.2, f"清理任务性能影响过大: {performance_impact:.2%}"
        assert throughput_with_cleanup > 5000, f"清理期间吞吐量过低: {throughput_with_cleanup:.0f}ops/sec"


def run_comprehensive_benchmark():
    """运行综合性能基准测试"""
    print("开始运行内存缓存综合性能基准测试...")
    print("="*80)
    
    # 创建不同配置的缓存进行测试
    configs = [
        ("默认配置", CacheConfig()),
        ("高性能配置", CacheConfig(
            max_memory=50*1024*1024,  # 50MB
            max_keys=500000,
            eviction_policy="lru",
            thread_safe=True
        )),
        ("低内存配置", CacheConfig(
            max_memory=1*1024*1024,   # 1MB
            max_keys=10000,
            eviction_policy="lfu"
        ))
    ]
    
    all_results = {}
    
    for config_name, config in configs:
        print(f"\n测试配置: {config_name}")
        print("-" * 40)
        
        cache = MemoryCachePool(config)
        benchmark = PerformanceBenchmark(cache)
        
        # 基础操作测试
        counter = 0
        def set_op():
            nonlocal counter
            cache.set(f"bench_{counter}", f"value_{counter}")
            counter += 1
        
        counter = 0
        def get_op():
            nonlocal counter
            cache.get(f"bench_{counter % 1000}")
            counter += 1
        
        # 预填充数据用于GET测试
        for i in range(1000):
            cache.set(f"bench_{i}", f"value_{i}")
        
        # 执行基准测试
        results = {}
        results['SET'] = benchmark.time_operation("SET", set_op, 5000)
        results['GET'] = benchmark.time_operation("GET", get_op, 5000)
        
        all_results[config_name] = results
        
        # 打印当前配置结果
        for op_name, stats in results.items():
            print(f"{op_name}: {stats['avg_us']:.1f}μs, {stats['ops_per_sec']:.0f}ops/sec")
    
    # 打印对比结果
    print(f"\n配置对比结果:")
    print("="*80)
    print(f"{'配置':<15} {'SET延迟(μs)':<12} {'SET吞吐量':<12} {'GET延迟(μs)':<12} {'GET吞吐量'}")
    print("-" * 80)
    
    for config_name, results in all_results.items():
        set_stats = results['SET']
        get_stats = results['GET']
        print(f"{config_name:<15} {set_stats['avg_us']:<12.1f} "
              f"{set_stats['ops_per_sec']:<12.0f} {get_stats['avg_us']:<12.1f} "
              f"{get_stats['ops_per_sec']:.0f}")


if __name__ == "__main__":
    # 运行综合基准测试
    run_comprehensive_benchmark()
    
    # 运行pytest测试
    pytest.main([__file__, "-v", "-s"])