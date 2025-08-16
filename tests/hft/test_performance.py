import pytest
import asyncio
import time
import statistics
from decimal import Decimal
from typing import List, Dict

from src.hft import (
    HFTEngine, HFTConfig,
    HFTPerformanceOptimizer, PerformanceConfig,
    OrderBookManager, MicrostructureAnalyzer,
    FastExecutionEngine, ExecutionOrder, OrderType,
    LatencySensitiveSignalProcessor
)
from src.core.models import MarketData
from src.core.ring_buffer import RingBuffer


class TestHFTPerformance:
    """HFT性能测试"""
    
    @pytest.fixture
    async def hft_engine(self):
        """HFT引擎测试夹具"""
        config = HFTConfig(latency_target_ms=5.0)
        engine = HFTEngine(config)
        await engine.initialize(["BTCUSDT", "ETHUSDT"])
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    async def performance_optimizer(self):
        """性能优化器测试夹具"""
        config = PerformanceConfig(
            use_uvloop=False,  # 测试环境不使用uvloop
            cpu_affinity_enabled=False,  # 避免测试环境权限问题
            gc_optimization=True
        )
        optimizer = HFTPerformanceOptimizer(config)
        await optimizer.initialize()
        yield optimizer
        await optimizer.shutdown()
    
    @pytest.mark.asyncio
    async def test_orderbook_update_latency(self, hft_engine):
        """测试订单簿更新延迟"""
        symbol = "BTCUSDT"
        latencies = []
        num_updates = 1000
        
        for i in range(num_updates):
            market_data = MarketData(
                symbol=symbol,
                timestamp=time.time(),
                price=50000.0 + i,
                volume=1.0,
                bid=50000.0 + i - 0.5,
                ask=50000.0 + i + 0.5,
                bid_volume=10.0,
                ask_volume=10.0
            )
            
            start_time = time.perf_counter()
            success = await hft_engine.update_market_data(symbol, market_data)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            assert success, f"Market data update failed at iteration {i}"
            latencies.append(latency_ms)
        
        # 性能断言
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        max_latency = max(latencies)
        
        print(f"Orderbook update latency stats:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        print(f"  P99: {p99_latency:.3f}ms")
        print(f"  Max: {max_latency:.3f}ms")
        
        # 延迟要求
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms exceeds 1ms target"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.3f}ms exceeds 2ms target"
        assert p99_latency < 5.0, f"P99 latency {p99_latency:.3f}ms exceeds 5ms target"
    
    @pytest.mark.asyncio
    async def test_signal_processing_latency(self):
        """测试信号处理延迟"""
        processor = LatencySensitiveSignalProcessor(latency_target_ms=1.0)
        await processor.start()
        
        try:
            from src.hft.microstructure_analyzer import MicrostructureSignal
            
            latencies = []
            num_signals = 1000
            
            for i in range(num_signals):
                signal = MicrostructureSignal(
                    signal_type="imbalance",
                    symbol="BTCUSDT",
                    timestamp=time.time(),
                    strength=0.5 + i * 0.001,
                    confidence=0.8,
                    metadata={}
                )
                
                start_time = time.perf_counter()
                success = await processor.process_signal(signal)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                assert success, f"Signal processing failed at iteration {i}"
                latencies.append(latency_ms)
            
            # 等待处理完成
            await asyncio.sleep(0.1)
            
            # 性能断言
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]
            
            print(f"Signal processing latency stats:")
            print(f"  Average: {avg_latency:.3f}ms")
            print(f"  P95: {p95_latency:.3f}ms")
            
            assert avg_latency < 0.1, f"Average signal processing latency {avg_latency:.3f}ms too high"
            assert p95_latency < 0.5, f"P95 signal processing latency {p95_latency:.3f}ms too high"
            
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execution_engine_latency(self, hft_engine):
        """测试执行引擎延迟"""
        execution_engine = hft_engine.execution_engine
        symbol = "BTCUSDT"
        
        # 先更新市场数据以建立订单簿
        market_data = MarketData(
            symbol=symbol,
            timestamp=time.time(),
            price=50000.0,
            volume=1.0,
            bid=49999.5,
            ask=50000.5,
            bid_volume=10.0,
            ask_volume=10.0
        )
        await hft_engine.update_market_data(symbol, market_data)
        
        latencies = []
        num_orders = 100
        
        for i in range(num_orders):
            order = ExecutionOrder(
                order_id="",
                symbol=symbol,
                side="buy",
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            
            start_time = time.perf_counter()
            success = await execution_engine.submit_order(order)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            assert success, f"Order submission failed at iteration {i}"
            latencies.append(latency_ms)
            
            # 清理订单
            if order.order_id:
                await execution_engine.cancel_order(order.order_id)
        
        # 性能断言
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"Order execution latency stats:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        
        assert avg_latency < 2.0, f"Average execution latency {avg_latency:.3f}ms too high"
        assert p95_latency < 5.0, f"P95 execution latency {p95_latency:.3f}ms too high"
    
    @pytest.mark.asyncio
    async def test_memory_pool_performance(self, performance_optimizer):
        """测试内存池性能"""
        memory_pool = performance_optimizer.memory_pool
        if not memory_pool:
            pytest.skip("Memory pool not available")
        
        # 测试缓冲区分配性能
        allocation_times = []
        num_allocations = 10000
        
        for i in range(num_allocations):
            start_time = time.perf_counter()
            buffer = memory_pool.acquire_buffer()
            allocation_time = (time.perf_counter() - start_time) * 1000000  # 微秒
            
            assert buffer is not None, f"Buffer allocation failed at iteration {i}"
            allocation_times.append(allocation_time)
            
            memory_pool.release_buffer(buffer)
        
        # 性能断言
        avg_allocation_time = statistics.mean(allocation_times)
        p95_allocation_time = statistics.quantiles(allocation_times, n=20)[18]
        
        print(f"Memory allocation performance:")
        print(f"  Average: {avg_allocation_time:.3f}μs")
        print(f"  P95: {p95_allocation_time:.3f}μs")
        
        assert avg_allocation_time < 1.0, f"Average allocation time {avg_allocation_time:.3f}μs too high"
        assert p95_allocation_time < 5.0, f"P95 allocation time {p95_allocation_time:.3f}μs too high"
    
    @pytest.mark.asyncio
    async def test_zero_copy_message_performance(self, performance_optimizer):
        """测试零拷贝消息性能"""
        num_messages = 10000
        creation_times = []
        parsing_times = []
        
        for i in range(num_messages):
            # 测试消息创建性能
            start_time = time.perf_counter()
            buffer = performance_optimizer.create_zero_copy_message(
                symbol="BTCUSDT",
                price=50000.0 + i,
                volume=1.0 + i * 0.001,
                timestamp=int(time.time() * 1000000)
            )
            creation_time = (time.perf_counter() - start_time) * 1000000  # 微秒
            
            assert buffer is not None, f"Message creation failed at iteration {i}"
            creation_times.append(creation_time)
            
            # 测试消息解析性能
            start_time = time.perf_counter()
            parsed = performance_optimizer.parse_zero_copy_message(buffer)
            parsing_time = (time.perf_counter() - start_time) * 1000000  # 微秒
            
            assert parsed is not None, f"Message parsing failed at iteration {i}"
            parsing_times.append(parsing_time)
            
            performance_optimizer.release_buffer(buffer)
        
        # 性能断言
        avg_creation_time = statistics.mean(creation_times)
        avg_parsing_time = statistics.mean(parsing_times)
        
        print(f"Zero-copy message performance:")
        print(f"  Creation: {avg_creation_time:.3f}μs")
        print(f"  Parsing: {avg_parsing_time:.3f}μs")
        
        assert avg_creation_time < 5.0, f"Average creation time {avg_creation_time:.3f}μs too high"
        assert avg_parsing_time < 5.0, f"Average parsing time {avg_parsing_time:.3f}μs too high"
    
    @pytest.mark.asyncio
    async def test_concurrent_performance(self, hft_engine):
        """测试并发性能"""
        symbol = "BTCUSDT"
        num_concurrent_tasks = 10
        updates_per_task = 100
        
        async def update_task(task_id: int) -> List[float]:
            """并发更新任务"""
            latencies = []
            
            for i in range(updates_per_task):
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=time.time(),
                    price=50000.0 + task_id * 1000 + i,
                    volume=1.0,
                    bid=50000.0 + task_id * 1000 + i - 0.5,
                    ask=50000.0 + task_id * 1000 + i + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                start_time = time.perf_counter()
                success = await hft_engine.update_market_data(symbol, market_data)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                assert success, f"Update failed in task {task_id}, iteration {i}"
                latencies.append(latency_ms)
            
            return latencies
        
        # 启动并发任务
        start_time = time.perf_counter()
        tasks = [update_task(i) for i in range(num_concurrent_tasks)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # 合并所有延迟数据
        all_latencies = []
        for task_latencies in results:
            all_latencies.extend(task_latencies)
        
        # 性能分析
        total_updates = num_concurrent_tasks * updates_per_task
        throughput = total_updates / total_time
        avg_latency = statistics.mean(all_latencies)
        
        print(f"Concurrent performance:")
        print(f"  Total updates: {total_updates}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} updates/s")
        print(f"  Average latency: {avg_latency:.3f}ms")
        
        # 性能断言
        assert throughput > 1000, f"Throughput {throughput:.1f} updates/s too low"
        assert avg_latency < 5.0, f"Average latency {avg_latency:.3f}ms too high under load"
    
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, hft_engine, performance_optimizer):
        """测试内存使用稳定性"""
        symbol = "BTCUSDT"
        num_iterations = 1000
        
        # 记录初始内存使用
        initial_report = performance_optimizer.get_performance_report()
        initial_memory = initial_report["current_metrics"]["memory_mb"]
        
        print(f"Initial memory usage: {initial_memory:.2f}MB")
        
        # 执行大量操作
        for i in range(num_iterations):
            market_data = MarketData(
                symbol=symbol,
                timestamp=time.time(),
                price=50000.0 + i,
                volume=1.0,
                bid=50000.0 + i - 0.5,
                ask=50000.0 + i + 0.5,
                bid_volume=10.0,
                ask_volume=10.0
            )
            
            await hft_engine.update_market_data(symbol, market_data)
            
            # 定期手动GC
            if i % 100 == 0:
                await performance_optimizer.manual_gc()
        
        # 最终GC
        await performance_optimizer.manual_gc()
        await asyncio.sleep(0.1)  # 让GC完成
        
        # 检查最终内存使用
        final_report = performance_optimizer.get_performance_report()
        final_memory = final_report["current_metrics"]["memory_mb"]
        
        memory_growth = final_memory - initial_memory
        memory_growth_percent = (memory_growth / initial_memory) * 100
        
        print(f"Final memory usage: {final_memory:.2f}MB")
        print(f"Memory growth: {memory_growth:.2f}MB ({memory_growth_percent:.1f}%)")
        
        # 内存稳定性断言
        assert memory_growth_percent < 50, f"Memory growth {memory_growth_percent:.1f}% too high"
        assert final_memory < initial_memory + 100, f"Absolute memory growth {memory_growth:.2f}MB too high"
    
    def test_performance_config_validation(self):
        """测试性能配置验证"""
        # 有效配置
        valid_config = PerformanceConfig(
            use_uvloop=True,
            cpu_affinity_enabled=True,
            memory_pool_size=1024*1024*100,
            thread_priority_boost=True
        )
        
        assert valid_config.use_uvloop is True
        assert valid_config.memory_pool_size > 0
        
        # 边界值测试
        boundary_config = PerformanceConfig(
            memory_pool_size=1024,  # 最小值
            max_worker_threads=1,
            scheduler_priority=1
        )
        
        assert boundary_config.memory_pool_size == 1024
        assert boundary_config.max_worker_threads == 1