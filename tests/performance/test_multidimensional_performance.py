"""
多维度技术指标引擎性能测试

测试引擎在不同负载条件下的性能表现，包括延迟、吞吐量、内存使用等指标
"""

import asyncio
import time
import psutil
import numpy as np
import pytest
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import statistics
import gc

from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.core.indicators.timeframe import TimeFrame


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu = self.process.cpu_percent()
        
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        return {
            'elapsed_time': end_time - self.start_time,
            'memory_usage_mb': end_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'cpu_usage_percent': end_cpu,
            'cpu_increase_percent': end_cpu - self.start_cpu
        }


def generate_test_data(n_points: int = 1000, seed: int = None) -> Dict[str, List[float]]:
    """生成测试用的市场数据"""
    if seed:
        np.random.seed(seed)
    
    # 生成价格序列
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_points)  # 0.01% 平均回报，2%波动率
    prices = base_price * np.cumprod(1 + returns)
    
    # 生成OHLCV
    closes = prices
    opens = np.roll(closes, 1)
    opens[0] = base_price
    
    # 日内波动
    daily_ranges = np.random.uniform(0.005, 0.03, n_points) * closes
    highs = closes + daily_ranges * np.random.uniform(0.3, 1.0, n_points)
    lows = closes - daily_ranges * np.random.uniform(0.3, 1.0, n_points)
    
    # 确保价格逻辑
    for i in range(n_points):
        highs[i] = max(opens[i], closes[i], highs[i])
        lows[i] = min(opens[i], closes[i], lows[i])
    
    # 成交量
    base_volume = 50000
    volume_volatility = np.random.uniform(0.5, 2.0, n_points)
    volumes = base_volume * volume_volatility
    
    return {
        'open': opens.tolist(),
        'high': highs.tolist(),
        'low': lows.tolist(),
        'close': closes.tolist(),
        'volume': volumes.tolist()
    }


class TestMultiDimensionalEnginePerformance:
    """多维度引擎性能测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        return MultiDimensionalIndicatorEngine(max_workers=4)
    
    @pytest.fixture
    def large_dataset(self):
        """创建大数据集"""
        return generate_test_data(2000, seed=42)
    
    @pytest.fixture
    def medium_dataset(self):
        """创建中等数据集"""
        return generate_test_data(1000, seed=123)
    
    @pytest.fixture
    def small_dataset(self):
        """创建小数据集"""
        return generate_test_data(500, seed=456)
    
    @pytest.mark.performance
    def test_single_signal_latency(self, engine, medium_dataset):
        """测试单个信号生成延迟"""
        async def run_test():
            latencies = []
            
            # 预热
            await engine.generate_multidimensional_signal(
                "WARMUP/USDT", medium_dataset, enable_multiframe_analysis=False
            )
            
            # 测试延迟
            for i in range(20):
                start_time = time.perf_counter()
                
                signal = await engine.generate_multidimensional_signal(
                    f"TEST{i}/USDT", 
                    medium_dataset,
                    enable_multiframe_analysis=False
                )
                
                end_time = time.perf_counter()
                latency = end_time - start_time
                latencies.append(latency)
            
            return latencies
        
        latencies = asyncio.run(run_test())
        
        # 性能断言
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\n延迟统计:")
        print(f"平均延迟: {avg_latency:.3f}s")
        print(f"P95延迟: {p95_latency:.3f}s")
        print(f"P99延迟: {p99_latency:.3f}s")
        print(f"最小延迟: {min(latencies):.3f}s")
        print(f"最大延迟: {max(latencies):.3f}s")
        
        # 性能要求
        assert avg_latency < 2.0, f"平均延迟过高: {avg_latency:.3f}s"
        assert p95_latency < 3.0, f"P95延迟过高: {p95_latency:.3f}s"
    
    @pytest.mark.performance
    def test_concurrent_signals_throughput(self, engine, medium_dataset):
        """测试并发信号生成吞吐量"""
        async def run_concurrent_test(concurrency_level: int):
            start_time = time.perf_counter()
            
            # 创建并发任务
            tasks = []
            for i in range(concurrency_level):
                task = engine.generate_multidimensional_signal(
                    f"CONCURRENT{i}/USDT",
                    medium_dataset,
                    enable_multiframe_analysis=False
                )
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            # 统计成功和失败
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            return {
                'concurrency': concurrency_level,
                'elapsed_time': elapsed_time,
                'successful': successful,
                'failed': failed,
                'throughput': successful / elapsed_time if elapsed_time > 0 else 0
            }
        
        # 测试不同并发级别
        concurrency_levels = [1, 5, 10, 20]
        results = []
        
        for level in concurrency_levels:
            result = asyncio.run(run_concurrent_test(level))
            results.append(result)
            
            print(f"\n并发级别 {level}:")
            print(f"  总时间: {result['elapsed_time']:.3f}s")
            print(f"  成功: {result['successful']}/{level}")
            print(f"  失败: {result['failed']}/{level}")
            print(f"  吞吐量: {result['throughput']:.2f} signals/s")
        
        # 性能要求
        max_throughput = max(r['throughput'] for r in results)
        assert max_throughput > 2.0, f"最大吞吐量过低: {max_throughput:.2f} signals/s"
    
    @pytest.mark.performance
    def test_memory_usage_scaling(self, engine):
        """测试内存使用随数据规模的扩展性"""
        monitor = PerformanceMonitor()
        data_sizes = [200, 500, 1000, 2000]
        memory_results = []
        
        async def test_memory_with_size(size: int):
            # 强制垃圾回收
            gc.collect()
            
            monitor.start_monitoring()
            
            # 生成测试数据
            test_data = generate_test_data(size, seed=size)
            
            # 生成信号
            signal = await engine.generate_multidimensional_signal(
                f"MEMORY_TEST_{size}/USDT",
                test_data,
                enable_multiframe_analysis=True
            )
            
            metrics = monitor.get_metrics()
            
            return {
                'data_size': size,
                'memory_usage': metrics['memory_usage_mb'],
                'memory_increase': metrics['memory_increase_mb'],
                'processing_time': metrics['elapsed_time'],
                'signal_generated': signal is not None
            }
        
        for size in data_sizes:
            result = asyncio.run(test_memory_with_size(size))
            memory_results.append(result)
            
            print(f"\n数据规模 {size}:")
            print(f"  内存使用: {result['memory_usage']:.1f} MB")
            print(f"  内存增加: {result['memory_increase']:.1f} MB")
            print(f"  处理时间: {result['processing_time']:.3f}s")
            print(f"  信号生成: {'成功' if result['signal_generated'] else '失败'}")
        
        # 检查内存增长是否合理（应该大致线性）
        memory_increases = [r['memory_increase'] for r in memory_results]
        max_memory_increase = max(memory_increases)
        
        # 内存增长不应该超过100MB（这是一个保守的限制）
        assert max_memory_increase < 100, f"内存增长过高: {max_memory_increase:.1f}MB"
    
    @pytest.mark.performance
    def test_multiframe_analysis_overhead(self, engine, medium_dataset):
        """测试多时间框架分析的性能开销"""
        
        async def test_with_multiframe(enable: bool):
            times = []
            
            for i in range(10):
                start_time = time.perf_counter()
                
                signal = await engine.generate_multidimensional_signal(
                    f"MULTIFRAME_{enable}_{i}/USDT",
                    medium_dataset,
                    enable_multiframe_analysis=enable
                )
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            return {
                'multiframe_enabled': enable,
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0
            }
        
        # 测试开启和关闭多时间框架分析
        without_multiframe = asyncio.run(test_with_multiframe(False))
        with_multiframe = asyncio.run(test_with_multiframe(True))
        
        print(f"\n多时间框架分析性能对比:")
        print(f"不启用多时间框架:")
        print(f"  平均时间: {without_multiframe['avg_time']:.3f}s")
        print(f"  标准差: {without_multiframe['std_time']:.3f}s")
        
        print(f"启用多时间框架:")
        print(f"  平均时间: {with_multiframe['avg_time']:.3f}s")
        print(f"  标准差: {with_multiframe['std_time']:.3f}s")
        
        # 计算开销
        overhead = with_multiframe['avg_time'] - without_multiframe['avg_time']
        overhead_percent = (overhead / without_multiframe['avg_time']) * 100
        
        print(f"开销: {overhead:.3f}s ({overhead_percent:.1f}%)")
        
        # 开销不应该超过100%
        assert overhead_percent < 100, f"多时间框架分析开销过高: {overhead_percent:.1f}%"
    
    @pytest.mark.performance
    def test_worker_thread_scaling(self, medium_dataset):
        """测试工作线程数量对性能的影响"""
        
        async def test_with_workers(num_workers: int):
            engine = MultiDimensionalIndicatorEngine(max_workers=num_workers)
            
            try:
                start_time = time.perf_counter()
                
                # 创建大量并发任务
                tasks = []
                for i in range(50):  # 50个并发任务
                    task = engine.generate_multidimensional_signal(
                        f"WORKER_TEST_{num_workers}_{i}/USDT",
                        medium_dataset,
                        enable_multiframe_analysis=False
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                successful = sum(1 for r in results if not isinstance(r, Exception))
                throughput = successful / elapsed_time if elapsed_time > 0 else 0
                
                return {
                    'workers': num_workers,
                    'elapsed_time': elapsed_time,
                    'successful': successful,
                    'throughput': throughput
                }
                
            finally:
                engine.cleanup()
        
        worker_counts = [1, 2, 4, 8, 16]
        scaling_results = []
        
        for workers in worker_counts:
            result = asyncio.run(test_with_workers(workers))
            scaling_results.append(result)
            
            print(f"\n工作线程数 {workers}:")
            print(f"  总时间: {result['elapsed_time']:.3f}s")
            print(f"  成功任务: {result['successful']}/50")
            print(f"  吞吐量: {result['throughput']:.2f} signals/s")
        
        # 找到最优线程数
        best_result = max(scaling_results, key=lambda x: x['throughput'])
        print(f"\n最优配置: {best_result['workers']} 个工作线程")
        print(f"最大吞吐量: {best_result['throughput']:.2f} signals/s")
        
        # 验证多线程确实提供了性能提升
        single_thread_throughput = next(r['throughput'] for r in scaling_results if r['workers'] == 1)
        max_throughput = max(r['throughput'] for r in scaling_results)
        
        improvement = (max_throughput - single_thread_throughput) / single_thread_throughput * 100
        print(f"性能提升: {improvement:.1f}%")
        
        assert improvement > 10, f"多线程性能提升不足: {improvement:.1f}%"
    
    @pytest.mark.performance
    def test_error_handling_performance(self, engine):
        """测试错误处理对性能的影响"""
        
        # 创建各种类型的错误数据
        error_datasets = {
            'empty_data': {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []},
            'insufficient_data': {
                'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]
            },
            'invalid_data': {
                'open': [100, np.nan, 102], 
                'high': [101, np.inf, 103], 
                'low': [99, 98, 101], 
                'close': [100.5, 101.5, 102.5], 
                'volume': [1000, 1200, 1100]
            }
        }
        
        async def test_error_handling():
            error_times = []
            valid_times = []
            
            # 测试错误数据处理时间
            for error_type, data in error_datasets.items():
                for i in range(5):
                    start_time = time.perf_counter()
                    
                    signal = await engine.generate_multidimensional_signal(
                        f"ERROR_{error_type}_{i}/USDT",
                        data
                    )
                    
                    end_time = time.perf_counter()
                    error_times.append(end_time - start_time)
            
            # 测试正常数据处理时间作为对比
            valid_data = generate_test_data(200, seed=999)
            for i in range(15):  # 与错误数据测试次数相同
                start_time = time.perf_counter()
                
                signal = await engine.generate_multidimensional_signal(
                    f"VALID_{i}/USDT",
                    valid_data
                )
                
                end_time = time.perf_counter()
                valid_times.append(end_time - start_time)
            
            return error_times, valid_times
        
        error_times, valid_times = asyncio.run(test_error_handling())
        
        avg_error_time = statistics.mean(error_times)
        avg_valid_time = statistics.mean(valid_times)
        
        print(f"\n错误处理性能:")
        print(f"错误数据平均处理时间: {avg_error_time:.3f}s")
        print(f"正常数据平均处理时间: {avg_valid_time:.3f}s")
        print(f"错误处理开销: {(avg_error_time - avg_valid_time) * 1000:.1f}ms")
        
        # 错误处理不应该显著慢于正常处理（允许一定开销）
        overhead_ratio = avg_error_time / avg_valid_time if avg_valid_time > 0 else 1
        assert overhead_ratio < 2.0, f"错误处理开销过高: {overhead_ratio:.2f}x"
    
    def test_cleanup_performance(self, engine):
        """测试资源清理性能"""
        
        # 使用引擎生成一些信号以创建资源
        async def use_engine():
            data = generate_test_data(500, seed=111)
            tasks = []
            
            for i in range(20):
                task = engine.generate_multidimensional_signal(
                    f"CLEANUP_TEST_{i}/USDT",
                    data
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        asyncio.run(use_engine())
        
        # 测试清理时间
        start_time = time.perf_counter()
        engine.cleanup()
        end_time = time.perf_counter()
        
        cleanup_time = end_time - start_time
        
        print(f"\n资源清理性能:")
        print(f"清理时间: {cleanup_time:.3f}s")
        
        # 清理不应该花费太长时间
        assert cleanup_time < 5.0, f"资源清理时间过长: {cleanup_time:.3f}s"


# 基准测试
@pytest.mark.benchmark
class TestBenchmarks:
    """基准测试类"""
    
    def test_benchmark_signal_generation(self, benchmark):
        """基准测试：信号生成性能"""
        
        engine = MultiDimensionalIndicatorEngine(max_workers=4)
        test_data = generate_test_data(1000, seed=777)
        
        async def generate_signal():
            return await engine.generate_multidimensional_signal(
                "BENCHMARK/USDT",
                test_data,
                enable_multiframe_analysis=True
            )
        
        def sync_generate():
            return asyncio.run(generate_signal())
        
        try:
            # 使用pytest-benchmark进行基准测试
            result = benchmark(sync_generate)
            
            print(f"\n基准测试结果:")
            print(f"信号生成成功: {result is not None}")
            
        finally:
            engine.cleanup()


if __name__ == '__main__':
    # 运行性能测试
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'performance',
        '--durations=10'
    ])