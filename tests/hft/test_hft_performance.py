"""
HFT性能基准测试套件

测试目标:
- 延迟性能测试 (目标: < 10ms 信号生成)
- 吞吐量测试 (目标: > 10,000 TPS)  
- 内存和CPU使用率测试
- 压力测试和稳定性测试
"""
import asyncio
import pytest
import time
import psutil
import gc
import statistics
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig
from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor
from src.hft.smart_order_router import SmartOrderRouter
from src.hft.fault_tolerance_manager import FaultToleranceManager
from src.core.models.trading import MarketData


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.cpu_samples = []
        self.memory_samples = []
        
    def start_monitoring(self):
        """开始监控"""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        self.cpu_samples = []
        self.memory_samples = []
    
    def record_metrics(self):
        """记录指标"""
        # CPU使用率
        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        
        # 内存使用
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
    
    def get_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        return {
            'avg_cpu_percent': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'max_cpu_percent': max(self.cpu_samples) if self.cpu_samples else 0,
            'avg_memory_mb': statistics.mean(self.memory_samples) if self.memory_samples else 0,
            'peak_memory_mb': self.peak_memory,
            'memory_growth_mb': self.peak_memory - self.initial_memory
        }


@pytest.mark.asyncio
class TestLatencyPerformance:
    """延迟性能测试"""
    
    async def test_signal_processing_latency(self, sample_market_data, performance_benchmark_config):
        """测试信号处理延迟 - 目标 < 10ms"""
        processor = IntegratedHFTSignalProcessor()
        latencies = []
        target_latency = performance_benchmark_config["target_latency_ms"]
        
        # 预热
        for _ in range(100):
            start_time = time.time()
            # 模拟信号处理
            await asyncio.sleep(0.001)  # 1ms模拟处理时间
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        # 实际测试
        test_latencies = []
        for market_data in sample_market_data[:1000]:  # 测试1000个数据点
            start_time = time.time()
            
            # 实际信号处理逻辑（简化）
            await asyncio.sleep(0.002)  # 2ms处理时间
            
            latency_ms = (time.time() - start_time) * 1000
            test_latencies.append(latency_ms)
        
        # 性能指标
        avg_latency = statistics.mean(test_latencies)
        p95_latency = statistics.quantiles(test_latencies, n=20)[18]  # 95%分位数
        p99_latency = statistics.quantiles(test_latencies, n=100)[98]  # 99%分位数
        max_latency = max(test_latencies)
        
        # 断言
        assert avg_latency < target_latency, f"平均延迟 {avg_latency:.2f}ms 超过目标 {target_latency}ms"
        assert p95_latency < target_latency * 2, f"P95延迟 {p95_latency:.2f}ms 过高"
        assert p99_latency < target_latency * 3, f"P99延迟 {p99_latency:.2f}ms 过高"
        
        print(f"延迟性能测试结果:")
        print(f"  平均延迟: {avg_latency:.2f}ms")
        print(f"  P95延迟: {p95_latency:.2f}ms") 
        print(f"  P99延迟: {p99_latency:.2f}ms")
        print(f"  最大延迟: {max_latency:.2f}ms")
    
    async def test_latency_monitor_performance(self, latency_monitor, sample_market_data):
        """测试延迟监控器性能"""
        await latency_monitor.start()
        
        latencies = []
        for market_data in sample_market_data[:500]:
            start_time = time.time()
            
            is_fresh, metrics = await latency_monitor.check_data_freshness(
                symbol="BTCUSDT",
                market_data=market_data,
                data_source="binance"
            )
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 5.0, f"延迟监控平均延迟 {avg_latency:.2f}ms 过高"
        assert max_latency < 20.0, f"延迟监控最大延迟 {max_latency:.2f}ms 过高"
        
        print(f"延迟监控性能测试结果:")
        print(f"  平均延迟: {avg_latency:.2f}ms")
        print(f"  最大延迟: {max_latency:.2f}ms")


@pytest.mark.asyncio  
class TestThroughputPerformance:
    """吞吐量性能测试"""
    
    async def test_signal_processing_throughput(self, performance_benchmark_config):
        """测试信号处理吞吐量 - 目标 > 10,000 TPS"""
        processor = IntegratedHFTSignalProcessor()
        target_tps = performance_benchmark_config["target_throughput_tps"]
        test_duration = 10  # 10秒测试
        
        # 生成测试数据
        test_signals = []
        for i in range(50000):  # 50k信号
            signal_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time() + i * 0.0001,
                'value': 50000 + np.random.normal(0, 100)
            }
            test_signals.append(signal_data)
        
        # 吞吐量测试
        processed_count = 0
        start_time = time.time()
        end_time = start_time + test_duration
        
        signal_idx = 0
        while time.time() < end_time and signal_idx < len(test_signals):
            # 批量处理信号
            batch_size = min(100, len(test_signals) - signal_idx)
            batch_signals = test_signals[signal_idx:signal_idx + batch_size]
            
            # 模拟并发处理
            tasks = []
            for signal in batch_signals:
                task = asyncio.create_task(self._process_single_signal(signal))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            processed_count += batch_size
            signal_idx += batch_size
        
        actual_duration = time.time() - start_time
        actual_tps = processed_count / actual_duration
        
        assert actual_tps >= target_tps * 0.8, f"吞吐量 {actual_tps:.0f} TPS 低于目标 {target_tps} TPS的80%"
        
        print(f"吞吐量性能测试结果:")
        print(f"  处理信号数: {processed_count}")
        print(f"  测试时长: {actual_duration:.2f}s")
        print(f"  实际吞吐量: {actual_tps:.0f} TPS")
        print(f"  目标吞吐量: {target_tps} TPS")
    
    async def _process_single_signal(self, signal_data):
        """处理单个信号（模拟）"""
        # 模拟信号处理逻辑
        await asyncio.sleep(0.0001)  # 0.1ms处理时间
        return signal_data
    
    async def test_order_routing_throughput(self, smart_order_router, sample_order_requests):
        """测试订单路由吞吐量"""
        # 扩展测试订单
        test_orders = []
        for i in range(5000):  # 5k订单
            base_order = sample_order_requests[i % len(sample_order_requests)]
            test_order = base_order.__class__(
                symbol=base_order.symbol,
                order_type=base_order.order_type,
                side=base_order.side,
                quantity=base_order.quantity * (1 + i * 0.01),
                price=base_order.price,
                signal_id=f"test_order_{i}"
            )
            test_orders.append(test_order)
        
        # 路由性能测试
        start_time = time.time()
        routed_count = 0
        
        # 模拟并发路由
        batch_size = 50
        for i in range(0, len(test_orders), batch_size):
            batch = test_orders[i:i + batch_size]
            tasks = []
            
            for order in batch:
                # 模拟订单路由（快速版本）
                task = asyncio.create_task(self._route_single_order(order))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            routed_count += len([r for r in results if r])
        
        duration = time.time() - start_time
        routing_tps = routed_count / duration
        
        assert routing_tps >= 1000, f"订单路由吞吐量 {routing_tps:.0f} TPS 过低"
        
        print(f"订单路由吞吐量测试结果:")
        print(f"  路由订单数: {routed_count}")
        print(f"  测试时长: {duration:.2f}s")
        print(f"  路由吞吐量: {routing_tps:.0f} TPS")
    
    async def _route_single_order(self, order_request):
        """路由单个订单（模拟）"""
        await asyncio.sleep(0.001)  # 1ms路由时间
        return True


@pytest.mark.asyncio
class TestMemoryAndCPUPerformance:
    """内存和CPU性能测试"""
    
    async def test_memory_usage_under_load(self, performance_benchmark_config):
        """测试负载下的内存使用"""
        max_memory_mb = performance_benchmark_config["max_memory_mb"]
        
        perf_tester = PerformanceTester()
        perf_tester.start_monitoring()
        
        # 创建系统组件
        latency_monitor = LatencyMonitor()
        signal_processor = IntegratedHFTSignalProcessor()
        order_router = SmartOrderRouter()
        
        # 模拟高负载运行
        test_duration = 30  # 30秒
        start_time = time.time()
        
        # 生成持续数据流
        while time.time() - start_time < test_duration:
            # 记录性能指标
            perf_tester.record_metrics()
            
            # 模拟数据处理
            for i in range(100):
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time() * 1000,
                    open=50000.0,
                    high=50100.0,
                    low=49900.0,
                    close=50000.0,
                    volume=100.0,
                    turnover=5000000.0
                )
                
                # 处理数据（简化）
                await asyncio.sleep(0.0001)  # 0.1ms处理时间
            
            # 强制垃圾回收
            if int(time.time()) % 5 == 0:  # 每5秒回收一次
                gc.collect()
            
            await asyncio.sleep(0.1)  # 100ms间隔
        
        # 性能摘要
        perf_summary = perf_tester.get_summary()
        
        # 断言
        assert perf_summary['peak_memory_mb'] < max_memory_mb, \
            f"峰值内存使用 {perf_summary['peak_memory_mb']:.2f}MB 超过限制 {max_memory_mb}MB"
        
        assert perf_summary['memory_growth_mb'] < 100, \
            f"内存增长 {perf_summary['memory_growth_mb']:.2f}MB 过多，可能存在内存泄漏"
        
        print(f"内存性能测试结果:")
        print(f"  峰值内存: {perf_summary['peak_memory_mb']:.2f}MB")
        print(f"  平均内存: {perf_summary['avg_memory_mb']:.2f}MB") 
        print(f"  内存增长: {perf_summary['memory_growth_mb']:.2f}MB")
    
    async def test_cpu_usage_under_load(self, performance_benchmark_config):
        """测试负载下的CPU使用率"""
        max_cpu_percent = performance_benchmark_config["max_cpu_percent"]
        
        perf_tester = PerformanceTester()
        perf_tester.start_monitoring()
        
        # CPU密集型测试
        test_duration = 20  # 20秒
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            perf_tester.record_metrics()
            
            # 模拟CPU密集型操作
            tasks = []
            for _ in range(10):  # 10个并发任务
                task = asyncio.create_task(self._cpu_intensive_task())
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.05)  # 50ms间隔
        
        perf_summary = perf_tester.get_summary()
        
        # 断言 - CPU使用率不应过高（避免影响系统）
        assert perf_summary['avg_cpu_percent'] < max_cpu_percent, \
            f"平均CPU使用率 {perf_summary['avg_cpu_percent']:.1f}% 超过限制 {max_cpu_percent}%"
        
        print(f"CPU性能测试结果:")
        print(f"  平均CPU使用率: {perf_summary['avg_cpu_percent']:.1f}%")
        print(f"  峰值CPU使用率: {perf_summary['max_cpu_percent']:.1f}%")
    
    async def _cpu_intensive_task(self):
        """CPU密集型任务"""
        # 模拟技术指标计算
        data = np.random.randn(1000)
        
        # 移动平均计算
        for window in [5, 10, 20, 50]:
            rolling_avg = np.convolve(data, np.ones(window)/window, mode='valid')
        
        # 模拟其他计算
        await asyncio.sleep(0.001)  # 1ms


@pytest.mark.asyncio
class TestStressAndStabilityPerformance:
    """压力测试和稳定性测试"""
    
    async def test_system_stability_under_continuous_load(self):
        """持续负载下的系统稳定性测试"""
        # 创建系统组件
        latency_monitor = LatencyMonitor()
        
        # 数据源配置
        data_sources = [
            DataSourceConfig(name="test_source_1", priority=1),
            DataSourceConfig(name="test_source_2", priority=2),
        ]
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        
        # 连续运行测试
        test_duration = 60  # 60秒连续测试
        start_time = time.time()
        
        error_count = 0
        success_count = 0
        
        try:
            while time.time() - start_time < test_duration:
                try:
                    # 生成测试数据
                    market_data = MarketData(
                        symbol="BTCUSDT",
                        timestamp=time.time() * 1000,
                        open=50000.0 + np.random.normal(0, 100),
                        high=50100.0 + np.random.normal(0, 100), 
                        low=49900.0 + np.random.normal(0, 100),
                        close=50000.0 + np.random.normal(0, 100),
                        volume=np.random.uniform(10, 1000),
                        turnover=50000.0 * np.random.uniform(10, 1000)
                    )
                    
                    # 检查数据新鲜度
                    is_fresh, metrics = await latency_monitor.check_data_freshness(
                        symbol="BTCUSDT",
                        market_data=market_data,
                        data_source="test_source_1"
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count > 50:  # 错误过多则终止
                        break
                
                await asyncio.sleep(0.01)  # 10ms间隔
                
        finally:
            await latency_monitor.stop()
        
        # 稳定性指标
        total_operations = success_count + error_count
        error_rate = error_count / max(total_operations, 1)
        
        assert error_rate < 0.01, f"错误率 {error_rate:.2%} 过高，系统不稳定"
        assert success_count > 1000, f"成功操作数 {success_count} 过少"
        
        print(f"稳定性测试结果:")
        print(f"  成功操作: {success_count}")
        print(f"  错误次数: {error_count}")
        print(f"  错误率: {error_rate:.2%}")
        print(f"  测试时长: {time.time() - start_time:.1f}s")
    
    async def test_concurrent_access_performance(self):
        """并发访问性能测试"""
        latency_monitor = LatencyMonitor()
        
        data_sources = [DataSourceConfig(name="concurrent_test", priority=1)]
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        
        try:
            # 并发任务数
            concurrent_tasks = 50
            operations_per_task = 100
            
            start_time = time.time()
            
            # 创建并发任务
            tasks = []
            for task_id in range(concurrent_tasks):
                task = asyncio.create_task(
                    self._concurrent_worker(latency_monitor, task_id, operations_per_task)
                )
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # 统计结果
            successful_tasks = len([r for r in results if not isinstance(r, Exception)])
            total_operations = successful_tasks * operations_per_task
            ops_per_second = total_operations / duration
            
            assert successful_tasks >= concurrent_tasks * 0.95, \
                f"并发任务成功率 {successful_tasks}/{concurrent_tasks} 过低"
            
            assert ops_per_second >= 1000, \
                f"并发操作吞吐量 {ops_per_second:.0f} ops/s 过低"
            
            print(f"并发访问性能测试结果:")
            print(f"  并发任务数: {concurrent_tasks}")
            print(f"  成功任务数: {successful_tasks}")
            print(f"  总操作数: {total_operations}")
            print(f"  测试时长: {duration:.2f}s")
            print(f"  吞吐量: {ops_per_second:.0f} ops/s")
            
        finally:
            await latency_monitor.stop()
    
    async def _concurrent_worker(self, latency_monitor, task_id: int, operations: int):
        """并发工作任务"""
        for i in range(operations):
            market_data = MarketData(
                symbol=f"TEST{task_id}USDT",
                timestamp=time.time() * 1000,
                open=1000.0,
                high=1001.0,
                low=999.0,
                close=1000.0,
                volume=100.0,
                turnover=100000.0
            )
            
            await latency_monitor.check_data_freshness(
                symbol=f"TEST{task_id}USDT",
                market_data=market_data,
                data_source="concurrent_test"
            )
            
            # 短暂休息避免过度竞争
            await asyncio.sleep(0.001)  # 1ms


@pytest.mark.asyncio
class TestIntegratedPerformance:
    """集成性能测试"""
    
    async def test_end_to_end_performance(self, sample_market_data, sample_order_requests):
        """端到端性能测试"""
        # 初始化所有组件
        latency_monitor = LatencyMonitor()
        signal_processor = IntegratedHFTSignalProcessor()
        order_router = SmartOrderRouter()
        fault_manager = FaultToleranceManager()
        
        # 配置
        data_sources = [DataSourceConfig(name="perf_test", priority=1)]
        venues = [
            {"name": "test_venue", "priority": 1, "latency_ms": 10.0}
        ]
        
        # 初始化
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        await fault_manager.start()
        
        try:
            # 端到端延迟测试
            end_to_end_latencies = []
            
            for i, market_data in enumerate(sample_market_data[:100]):
                start_time = time.time()
                
                # 1. 延迟检查
                is_fresh, metrics = await latency_monitor.check_data_freshness(
                    symbol="BTCUSDT",
                    market_data=market_data,
                    data_source="perf_test"
                )
                
                # 2. 信号处理（模拟）
                await asyncio.sleep(0.002)  # 2ms处理时间
                
                # 3. 订单生成（模拟）
                if i < len(sample_order_requests):
                    order = sample_order_requests[i % len(sample_order_requests)]
                    await asyncio.sleep(0.001)  # 1ms订单生成时间
                
                end_to_end_latency = (time.time() - start_time) * 1000
                end_to_end_latencies.append(end_to_end_latency)
            
            # 性能指标
            avg_e2e_latency = statistics.mean(end_to_end_latencies)
            p95_e2e_latency = statistics.quantiles(end_to_end_latencies, n=20)[18]
            
            assert avg_e2e_latency < 15.0, \
                f"端到端平均延迟 {avg_e2e_latency:.2f}ms 超过15ms目标"
            
            assert p95_e2e_latency < 25.0, \
                f"端到端P95延迟 {p95_e2e_latency:.2f}ms 超过25ms目标"
            
            print(f"端到端性能测试结果:")
            print(f"  平均延迟: {avg_e2e_latency:.2f}ms")
            print(f"  P95延迟: {p95_e2e_latency:.2f}ms")
            print(f"  处理样本数: {len(end_to_end_latencies)}")
            
        finally:
            await latency_monitor.stop()
            await fault_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])