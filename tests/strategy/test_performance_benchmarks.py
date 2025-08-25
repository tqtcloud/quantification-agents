"""
策略管理系统性能基准测试套件
测试信号聚合延迟、并发性能和资源使用效率
"""

import asyncio
import pytest
import time
import statistics
import psutil
import gc
from decimal import Decimal
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np

from src.strategy.strategy_manager import StrategyManager, StrategyType
from src.strategy.signal_aggregator import SignalAggregator, AggregationStrategy, SignalInput
from src.strategy.resource_allocator import ResourceAllocator, ResourceLimit
from src.core.message_bus import MessageBus
from src.core.models.signals import TradingSignal, SignalStrength
from src.hft.hft_engine import HFTConfig


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.latencies = []
        self.throughputs = []
        self.memory_snapshots = []
        self.cpu_snapshots = []
        self.start_time = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.memory_snapshots.clear()
        self.cpu_snapshots.clear()
        self.latencies.clear()
        self.throughputs.clear()
    
    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.latencies.append(latency_ms)
    
    def record_throughput(self, operations: int, duration_seconds: float):
        """记录吞吐量"""
        tps = operations / duration_seconds if duration_seconds > 0 else 0
        self.throughputs.append(tps)
    
    def take_resource_snapshot(self):
        """获取资源快照"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            self.memory_snapshots.append({
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent
            })
        except Exception:
            pass  # 忽略资源监控错误
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        duration = time.time() - self.start_time if self.start_time else 0
        
        # 延迟统计
        latency_stats = {}
        if self.latencies:
            latency_stats = {
                'avg_ms': statistics.mean(self.latencies),
                'median_ms': statistics.median(self.latencies),
                'p95_ms': np.percentile(self.latencies, 95),
                'p99_ms': np.percentile(self.latencies, 99),
                'max_ms': max(self.latencies),
                'min_ms': min(self.latencies),
                'count': len(self.latencies)
            }
        
        # 吞吐量统计
        throughput_stats = {}
        if self.throughputs:
            throughput_stats = {
                'avg_tps': statistics.mean(self.throughputs),
                'max_tps': max(self.throughputs),
                'min_tps': min(self.throughputs),
                'total_operations': sum(t * 1 for t in self.throughputs)  # 近似值
            }
        
        # 资源使用统计
        resource_stats = {}
        if self.memory_snapshots:
            memory_values = [s['memory_mb'] for s in self.memory_snapshots]
            cpu_values = [s['cpu_percent'] for s in self.memory_snapshots if s['cpu_percent'] > 0]
            
            resource_stats = {
                'memory': {
                    'avg_mb': statistics.mean(memory_values),
                    'max_mb': max(memory_values),
                    'min_mb': min(memory_values)
                },
                'cpu': {
                    'avg_percent': statistics.mean(cpu_values) if cpu_values else 0,
                    'max_percent': max(cpu_values) if cpu_values else 0
                }
            }
        
        return {
            'test_duration_seconds': duration,
            'latency': latency_stats,
            'throughput': throughput_stats,
            'resources': resource_stats,
            'total_snapshots': len(self.memory_snapshots)
        }


class TestPerformanceBenchmarks:
    """性能基准测试类"""
    
    # 性能目标常量
    TARGET_LATENCY_MS = 10.0  # 目标延迟 < 10ms
    TARGET_THROUGHPUT_TPS = 1000  # 目标吞吐量 > 1000 TPS
    MAX_MEMORY_MB = 512  # 最大内存使用 512MB
    MAX_CPU_PERCENT = 80  # 最大CPU使用率 80%
    
    @pytest.fixture
    async def performance_monitor(self):
        """性能监控器fixture"""
        monitor = PerformanceMonitor()
        yield monitor
    
    @pytest.fixture
    async def message_bus(self):
        """消息总线fixture"""
        bus = MessageBus()
        await bus.initialize()
        yield bus
        await bus.shutdown()
    
    @pytest.fixture
    async def signal_aggregator(self, message_bus):
        """信号聚合器fixture"""
        aggregator = SignalAggregator(
            message_bus=message_bus,
            aggregation_strategy=AggregationStrategy.HYBRID_FUSION
        )
        await aggregator.initialize()
        yield aggregator
        await aggregator.shutdown()

    @pytest.mark.asyncio
    async def test_signal_aggregation_latency(self, signal_aggregator, performance_monitor):
        """测试信号聚合延迟性能"""
        performance_monitor.start_monitoring()
        
        # 预热
        await self._warmup_signal_aggregation(signal_aggregator, 100)
        
        # 基准测试
        test_iterations = 1000
        for i in range(test_iterations):
            start_time = time.time()
            
            # 创建测试信号
            signals = self._create_test_signals(count=5)
            
            # 执行聚合
            result = await signal_aggregator.aggregate_signals(signals)
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            performance_monitor.record_latency(latency_ms)
            performance_monitor.take_resource_snapshot()
            
            # 每100次迭代强制垃圾回收
            if i % 100 == 0:
                gc.collect()
        
        # 分析结果
        report = performance_monitor.get_performance_report()
        
        # 断言性能目标
        assert report['latency']['avg_ms'] < self.TARGET_LATENCY_MS, \
            f"平均延迟 {report['latency']['avg_ms']:.2f}ms 超过目标 {self.TARGET_LATENCY_MS}ms"
        
        assert report['latency']['p95_ms'] < self.TARGET_LATENCY_MS * 2, \
            f"P95延迟 {report['latency']['p95_ms']:.2f}ms 超过目标 {self.TARGET_LATENCY_MS * 2}ms"
        
        # 打印详细报告
        print("\n=== 信号聚合延迟性能报告 ===")
        print(f"平均延迟: {report['latency']['avg_ms']:.2f}ms")
        print(f"P95延迟: {report['latency']['p95_ms']:.2f}ms")
        print(f"P99延迟: {report['latency']['p99_ms']:.2f}ms")
        print(f"最大延迟: {report['latency']['max_ms']:.2f}ms")
        print(f"测试次数: {report['latency']['count']}")

    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, signal_aggregator, performance_monitor):
        """测试并发信号处理性能"""
        performance_monitor.start_monitoring()
        
        async def process_signals_batch(batch_size: int = 10):
            """处理一批信号"""
            tasks = []
            for _ in range(batch_size):
                signals = self._create_test_signals(count=3)
                task = signal_aggregator.aggregate_signals(signals)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_count = len([r for r in results if not isinstance(r, Exception)])
            duration = end_time - start_time
            
            return successful_count, duration
        
        # 并发测试
        concurrent_levels = [1, 5, 10, 20, 50]
        results = {}
        
        for level in concurrent_levels:
            print(f"测试并发级别: {level}")
            
            # 多次测试求平均值
            level_throughputs = []
            for _ in range(5):
                count, duration = await process_signals_batch(level)
                if duration > 0:
                    throughput = count / duration
                    level_throughputs.append(throughput)
                    performance_monitor.record_throughput(count, duration)
                
                performance_monitor.take_resource_snapshot()
                await asyncio.sleep(0.1)  # 避免过度负载
            
            if level_throughputs:
                results[level] = {
                    'avg_throughput': statistics.mean(level_throughputs),
                    'max_throughput': max(level_throughputs)
                }
        
        # 分析结果
        report = performance_monitor.get_performance_report()
        
        # 验证吞吐量目标
        if report['throughput']:
            max_throughput = report['throughput']['max_tps']
            assert max_throughput > self.TARGET_THROUGHPUT_TPS * 0.1, \
                f"最大吞吐量 {max_throughput:.2f} TPS 低于最低目标"
        
        # 打印详细报告
        print("\n=== 并发信号处理性能报告 ===")
        for level, stats in results.items():
            print(f"并发级别 {level}: 平均吞吐量 {stats['avg_throughput']:.2f} TPS")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, signal_aggregator, performance_monitor):
        """测试负载下的内存使用"""
        performance_monitor.start_monitoring()
        
        # 基线内存使用
        performance_monitor.take_resource_snapshot()
        baseline_memory = performance_monitor.memory_snapshots[-1]['memory_mb']
        
        # 负载测试
        batch_size = 100
        batches = 50  # 总共处理5000个信号聚合操作
        
        for batch in range(batches):
            # 处理一批信号
            tasks = []
            for _ in range(batch_size):
                signals = self._create_test_signals(count=5)
                task = signal_aggregator.aggregate_signals(signals)
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 监控资源使用
            performance_monitor.take_resource_snapshot()
            
            # 每10批强制垃圾回收
            if batch % 10 == 0:
                gc.collect()
                performance_monitor.take_resource_snapshot()
                print(f"完成批次 {batch + 1}/{batches}")
        
        # 分析内存使用
        report = performance_monitor.get_performance_report()
        final_memory = report['resources']['memory']['max_mb']
        memory_growth = final_memory - baseline_memory
        
        # 验证内存使用目标
        assert final_memory < self.MAX_MEMORY_MB, \
            f"最大内存使用 {final_memory:.2f}MB 超过目标 {self.MAX_MEMORY_MB}MB"
        
        assert memory_growth < self.MAX_MEMORY_MB * 0.5, \
            f"内存增长 {memory_growth:.2f}MB 过大"
        
        print(f"\n=== 内存使用报告 ===")
        print(f"基线内存: {baseline_memory:.2f}MB")
        print(f"最大内存: {final_memory:.2f}MB")
        print(f"内存增长: {memory_growth:.2f}MB")
        print(f"平均内存: {report['resources']['memory']['avg_mb']:.2f}MB")

    @pytest.mark.asyncio
    async def test_resource_allocation_performance(self, performance_monitor):
        """测试资源分配器性能"""
        resource_allocator = ResourceAllocator()
        await resource_allocator.initialize()
        
        try:
            performance_monitor.start_monitoring()
            
            # 测试资源分配和释放的性能
            strategy_ids = [f"strategy_{i}" for i in range(100)]
            
            # 分配资源
            allocation_times = []
            for strategy_id in strategy_ids:
                start_time = time.time()
                
                limits = ResourceLimit(
                    memory_mb=256,
                    cpu_percent=10.0,
                    network_connections=50
                )
                
                success = await resource_allocator.allocate_resources(strategy_id, limits)
                
                end_time = time.time()
                
                if success:
                    allocation_times.append((end_time - start_time) * 1000)
                
                performance_monitor.take_resource_snapshot()
            
            # 释放资源
            deallocation_times = []
            for strategy_id in strategy_ids:
                start_time = time.time()
                
                await resource_allocator.deallocate_resources(strategy_id)
                
                end_time = time.time()
                deallocation_times.append((end_time - start_time) * 1000)
                
                performance_monitor.take_resource_snapshot()
            
            # 分析性能
            if allocation_times:
                avg_allocation_ms = statistics.mean(allocation_times)
                max_allocation_ms = max(allocation_times)
                
                # 资源分配应该很快
                assert avg_allocation_ms < 5.0, \
                    f"平均资源分配时间 {avg_allocation_ms:.2f}ms 过长"
                
                assert max_allocation_ms < 20.0, \
                    f"最大资源分配时间 {max_allocation_ms:.2f}ms 过长"
            
            if deallocation_times:
                avg_deallocation_ms = statistics.mean(deallocation_times)
                assert avg_deallocation_ms < 2.0, \
                    f"平均资源释放时间 {avg_deallocation_ms:.2f}ms 过长"
            
            print(f"\n=== 资源分配性能报告 ===")
            print(f"平均分配时间: {avg_allocation_ms:.2f}ms")
            print(f"最大分配时间: {max_allocation_ms:.2f}ms")
            print(f"平均释放时间: {avg_deallocation_ms:.2f}ms")
            print(f"成功分配: {len(allocation_times)}/100")
            
        finally:
            await resource_allocator.cleanup()

    @pytest.mark.asyncio
    async def test_strategy_manager_scalability(self, message_bus, performance_monitor):
        """测试策略管理器可扩展性"""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            strategy_manager = StrategyManager(message_bus=message_bus, config_dir=temp_dir)
            await strategy_manager.initialize()
            
            performance_monitor.start_monitoring()
            
            # 测试不同数量的策略
            strategy_counts = [10, 25, 50]
            
            for count in strategy_counts:
                print(f"测试 {count} 个策略的性能")
                
                # 创建策略
                strategy_ids = []
                create_start = time.time()
                
                for i in range(count):
                    config = HFTConfig(
                        symbol=f"SYMBOL{i}",
                        min_order_size=Decimal("0.001"),
                        max_order_size=Decimal("1.0"),
                        latency_target_ms=5
                    )
                    
                    strategy_id = await strategy_manager.create_strategy(
                        name=f"scale_test_{i}",
                        strategy_type=StrategyType.HFT,
                        config=config
                    )
                    
                    if strategy_id:
                        strategy_ids.append(strategy_id)
                
                create_time = time.time() - create_start
                performance_monitor.record_throughput(len(strategy_ids), create_time)
                
                # 启动策略
                start_time = time.time()
                start_tasks = [
                    strategy_manager.start_strategy(sid) 
                    for sid in strategy_ids[:min(10, len(strategy_ids))]  # 限制并发启动数量
                ]
                
                start_results = await asyncio.gather(*start_tasks, return_exceptions=True)
                start_duration = time.time() - start_time
                
                successful_starts = sum(1 for r in start_results if r is True)
                print(f"成功启动策略: {successful_starts}/{len(start_tasks)}")
                
                performance_monitor.take_resource_snapshot()
                
                # 清理策略
                for strategy_id in strategy_ids:
                    try:
                        await strategy_manager.stop_strategy(strategy_id)
                        await strategy_manager.remove_strategy(strategy_id)
                    except Exception:
                        pass  # 忽略清理错误
                
                await asyncio.sleep(1)  # 等待清理完成
                performance_monitor.take_resource_snapshot()
            
            report = performance_monitor.get_performance_report()
            
            print(f"\n=== 策略管理器可扩展性报告 ===")
            if report['throughput']:
                print(f"平均创建吞吐量: {report['throughput']['avg_tps']:.2f} strategies/sec")
                print(f"最大创建吞吐量: {report['throughput']['max_tps']:.2f} strategies/sec")
            
            if report['resources']:
                print(f"最大内存使用: {report['resources']['memory']['max_mb']:.2f}MB")
                print(f"最大CPU使用: {report['resources']['cpu']['max_percent']:.2f}%")
            
        finally:
            await strategy_manager.shutdown()
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_system_stability_under_stress(self, message_bus, signal_aggregator, performance_monitor):
        """测试系统压力下的稳定性"""
        performance_monitor.start_monitoring()
        
        # 持续压力测试
        stress_duration_seconds = 30  # 30秒压力测试
        start_time = time.time()
        
        error_count = 0
        success_count = 0
        
        while time.time() - start_time < stress_duration_seconds:
            try:
                # 并发执行多种操作
                tasks = []
                
                # 信号聚合任务
                for _ in range(10):
                    signals = self._create_test_signals(count=5)
                    task = signal_aggregator.aggregate_signals(signals)
                    tasks.append(task)
                
                # 执行任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 统计结果
                for result in results:
                    if isinstance(result, Exception):
                        error_count += 1
                    else:
                        success_count += 1
                
                performance_monitor.take_resource_snapshot()
                
                # 短暂休息
                await asyncio.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                print(f"压力测试异常: {e}")
        
        # 分析稳定性
        total_operations = success_count + error_count
        error_rate = error_count / total_operations if total_operations > 0 else 0
        
        report = performance_monitor.get_performance_report()
        
        # 稳定性断言
        assert error_rate < 0.05, f"错误率 {error_rate:.3f} 过高"  # 错误率应小于5%
        assert success_count > 1000, f"成功操作数 {success_count} 过低"  # 应至少完成1000次操作
        
        print(f"\n=== 系统稳定性报告 ===")
        print(f"测试持续时间: {stress_duration_seconds}秒")
        print(f"总操作数: {total_operations}")
        print(f"成功操作: {success_count}")
        print(f"失败操作: {error_count}")
        print(f"错误率: {error_rate:.3f}")
        print(f"平均吞吐量: {total_operations / stress_duration_seconds:.2f} ops/sec")

    def _create_test_signals(self, count: int = 3) -> List[SignalInput]:
        """创建测试信号"""
        signals = []
        
        for i in range(count):
            signal = TradingSignal(
                symbol=f"TEST{i}USDT",
                side="buy" if i % 2 == 0 else "sell",
                strength=SignalStrength.MEDIUM,
                price=Decimal(f"{50000 + i * 100}"),
                quantity=Decimal(f"{0.1 + i * 0.01}"),
                timestamp=time.time(),
                source=f"test_source_{i}",
                confidence=0.8 + i * 0.01
            )
            
            signal_input = SignalInput(
                signal_id=f"test_{i}_{int(time.time() * 1000000)}",
                signal=signal,
                source_strategy_id=f"strategy_{i}",
                weight=0.5 + i * 0.1,
                priority=i + 1
            )
            
            signals.append(signal_input)
        
        return signals

    async def _warmup_signal_aggregation(self, signal_aggregator, iterations: int = 100):
        """预热信号聚合器"""
        for _ in range(iterations):
            signals = self._create_test_signals(count=3)
            try:
                await signal_aggregator.aggregate_signals(signals)
            except Exception:
                pass  # 忽略预热错误
        
        # 强制垃圾回收
        gc.collect()


@pytest.mark.benchmark
class TestBenchmarkSuite:
    """基准测试套件（用于CI/CD）"""
    
    @pytest.mark.asyncio
    async def test_quick_performance_check(self):
        """快速性能检查（用于CI）"""
        bus = MessageBus()
        await bus.initialize()
        
        try:
            aggregator = SignalAggregator(
                message_bus=bus,
                aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE
            )
            await aggregator.initialize()
            
            # 快速性能测试
            latencies = []
            for _ in range(100):
                start_time = time.time()
                
                signals = [
                    SignalInput(
                        signal_id=f"quick_test_{int(time.time() * 1000000)}",
                        signal=TradingSignal(
                            symbol="BTCUSDT",
                            side="buy",
                            strength=SignalStrength.MEDIUM,
                            price=Decimal("50000"),
                            quantity=Decimal("0.1"),
                            timestamp=time.time(),
                            source="test",
                            confidence=0.8
                        ),
                        source_strategy_id="test_strategy",
                        weight=1.0,
                        priority=1
                    )
                ]
                
                await aggregator.aggregate_signals(signals)
                
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            
            # 基本性能断言
            avg_latency = statistics.mean(latencies)
            assert avg_latency < 50.0, f"平均延迟 {avg_latency:.2f}ms 超过阈值"
            
            print(f"快速性能检查通过 - 平均延迟: {avg_latency:.2f}ms")
            
            await aggregator.shutdown()
            
        finally:
            await bus.shutdown()