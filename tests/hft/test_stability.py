import pytest
import asyncio
import time
import random
import threading
from decimal import Decimal
from typing import List, Dict
import gc

from src.hft import (
    HFTEngine, HFTConfig,
    HFTPerformanceOptimizer, PerformanceConfig,
    NetworkLatencyOptimizer, NetworkConfig,
    HFTPerformanceSuite, HFTSuiteConfig
)
from src.core.models import MarketData


class TestHFTStability:
    """HFT系统稳定性测试"""
    
    @pytest.fixture
    async def hft_suite(self):
        """HFT性能套件测试夹具"""
        config = HFTSuiteConfig(
            enable_arbitrage=True,
            enable_market_making=True,
            monitoring_interval=0.1,
            performance_reporting_interval=5.0
        )
        config.performance_config.use_uvloop = False  # 测试环境
        config.performance_config.cpu_affinity_enabled = False
        
        suite = HFTPerformanceSuite(config)
        await suite.initialize(["BTCUSDT", "ETHUSDT"])
        await suite.start()
        yield suite
        await suite.stop()
    
    @pytest.mark.asyncio
    async def test_long_running_stability(self, hft_suite):
        """测试长时间运行稳定性"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        test_duration = 30  # 30秒测试
        update_interval = 0.01  # 10ms更新间隔
        
        start_time = time.time()
        update_count = 0
        error_count = 0
        
        print(f"Starting {test_duration}s stability test...")
        
        while time.time() - start_time < test_duration:
            try:
                for symbol in symbols:
                    base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                    
                    # 添加随机波动
                    price_change = random.uniform(-10, 10)
                    current_price = base_price + price_change
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=time.time(),
                        price=current_price,
                        volume=random.uniform(0.1, 2.0),
                        bid=current_price - random.uniform(0.1, 1.0),
                        ask=current_price + random.uniform(0.1, 1.0),
                        bid_volume=random.uniform(5, 50),
                        ask_volume=random.uniform(5, 50)
                    )
                    
                    success = await hft_suite.update_market_data(symbol, market_data)
                    if success:
                        update_count += 1
                    else:
                        error_count += 1
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                error_count += 1
                print(f"Error during stability test: {e}")
        
        total_time = time.time() - start_time
        update_rate = update_count / total_time
        error_rate = error_count / (update_count + error_count) if (update_count + error_count) > 0 else 0
        
        # 获取最终状态
        final_status = hft_suite.get_comprehensive_status()
        
        print(f"Stability test results:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Updates: {update_count}")
        print(f"  Errors: {error_count}")
        print(f"  Update rate: {update_rate:.1f} updates/s")
        print(f"  Error rate: {error_rate:.1%}")
        print(f"  Final health score: {final_status['metrics']['health_score']:.1f}%")
        
        # 稳定性断言
        assert error_rate < 0.01, f"Error rate {error_rate:.1%} too high"
        assert update_rate > 50, f"Update rate {update_rate:.1f} too low"
        assert final_status["running"], "System should still be running"
        assert final_status["metrics"]["health_score"] > 70, "Health score too low"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, hft_suite):
        """测试内存泄露检测"""
        if not hft_suite.performance_optimizer:
            pytest.skip("Performance optimizer not available")
        
        # 记录初始内存
        initial_report = hft_suite.performance_optimizer.get_performance_report()
        initial_memory = initial_report["current_metrics"]["memory_mb"]
        
        print(f"Initial memory: {initial_memory:.2f}MB")
        
        # 执行大量操作
        num_cycles = 100
        for cycle in range(num_cycles):
            # 大量市场数据更新
            for i in range(100):
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time(),
                    price=50000.0 + i,
                    volume=1.0,
                    bid=50000.0 + i - 0.5,
                    ask=50000.0 + i + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                await hft_suite.update_market_data("BTCUSDT", market_data)
            
            # 定期手动GC和内存检查
            if cycle % 10 == 0:
                await hft_suite.performance_optimizer.manual_gc()
                
                current_report = hft_suite.performance_optimizer.get_performance_report()
                current_memory = current_report["current_metrics"]["memory_mb"]
                memory_growth = current_memory - initial_memory
                
                print(f"Cycle {cycle}: Memory = {current_memory:.2f}MB (growth: +{memory_growth:.2f}MB)")
        
        # 最终内存检查
        await hft_suite.performance_optimizer.manual_gc()
        await asyncio.sleep(0.1)  # 让GC完成
        
        final_report = hft_suite.performance_optimizer.get_performance_report()
        final_memory = final_report["current_metrics"]["memory_mb"]
        total_growth = final_memory - initial_memory
        growth_percent = (total_growth / initial_memory) * 100
        
        print(f"Memory leak test results:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Growth: {total_growth:.2f}MB ({growth_percent:.1f}%)")
        
        # 内存泄露检测断言
        assert growth_percent < 100, f"Memory growth {growth_percent:.1f}% indicates potential leak"
        assert total_growth < 200, f"Absolute memory growth {total_growth:.2f}MB too high"
    
    @pytest.mark.asyncio
    async def test_concurrent_stress(self, hft_suite):
        """测试并发压力"""
        num_concurrent_tasks = 20
        operations_per_task = 100
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        async def stress_task(task_id: int) -> Dict[str, int]:
            """并发压力任务"""
            success_count = 0
            error_count = 0
            
            for i in range(operations_per_task):
                try:
                    symbol = random.choice(symbols)
                    base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=time.time(),
                        price=base_price + random.uniform(-100, 100),
                        volume=random.uniform(0.1, 5.0),
                        bid=base_price + random.uniform(-101, -1),
                        ask=base_price + random.uniform(1, 101),
                        bid_volume=random.uniform(1, 100),
                        ask_volume=random.uniform(1, 100)
                    )
                    
                    success = await hft_suite.update_market_data(symbol, market_data)
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                    
                    # 随机短暂休眠
                    if random.random() < 0.1:
                        await asyncio.sleep(random.uniform(0.001, 0.01))
                        
                except Exception as e:
                    error_count += 1
            
            return {"task_id": task_id, "success": success_count, "errors": error_count}
        
        # 启动并发任务
        print(f"Starting {num_concurrent_tasks} concurrent stress tasks...")
        start_time = time.time()
        
        tasks = [stress_task(i) for i in range(num_concurrent_tasks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 统计结果
        total_success = 0
        total_errors = 0
        task_failures = 0
        
        for result in results:
            if isinstance(result, Exception):
                task_failures += 1
                print(f"Task failed with exception: {result}")
            else:
                total_success += result["success"]
                total_errors += result["errors"]
        
        total_operations = total_success + total_errors
        throughput = total_operations / total_time
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        
        print(f"Concurrent stress test results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Successful: {total_success}")
        print(f"  Errors: {total_errors}")
        print(f"  Task failures: {task_failures}")
        print(f"  Duration: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/s")
        print(f"  Error rate: {error_rate:.1%}")
        
        # 压力测试断言
        assert task_failures == 0, f"{task_failures} tasks failed completely"
        assert error_rate < 0.05, f"Error rate {error_rate:.1%} too high under stress"
        assert throughput > 500, f"Throughput {throughput:.1f} too low under stress"
        
        # 检查系统仍然健康
        final_status = hft_suite.get_comprehensive_status()
        assert final_status["running"], "System should still be running after stress test"
        assert final_status["metrics"]["health_score"] > 50, "Health score too low after stress"
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, hft_suite):
        """测试错误恢复能力"""
        symbol = "BTCUSDT"
        
        # 正常操作基线
        print("Establishing baseline...")
        baseline_success = 0
        for i in range(50):
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
            
            if await hft_suite.update_market_data(symbol, market_data):
                baseline_success += 1
        
        baseline_rate = baseline_success / 50
        print(f"Baseline success rate: {baseline_rate:.1%}")
        
        # 注入错误场景
        print("Testing error scenarios...")
        error_scenarios = [
            # 无效数据
            MarketData(
                symbol=symbol,
                timestamp=time.time(),
                price=float('nan'),  # NaN价格
                volume=1.0,
                bid=50000.0,
                ask=50001.0,
                bid_volume=10.0,
                ask_volume=10.0
            ),
            # 负价格
            MarketData(
                symbol=symbol,
                timestamp=time.time(),
                price=-100.0,
                volume=1.0,
                bid=-100.5,
                ask=-99.5,
                bid_volume=10.0,
                ask_volume=10.0
            ),
            # 极大数值
            MarketData(
                symbol=symbol,
                timestamp=time.time(),
                price=1e10,
                volume=1.0,
                bid=1e10 - 1,
                ask=1e10 + 1,
                bid_volume=10.0,
                ask_volume=10.0
            )
        ]
        
        error_handled_count = 0
        for i, bad_data in enumerate(error_scenarios):
            try:
                # 应该优雅地处理错误而不崩溃
                result = await hft_suite.update_market_data(symbol, bad_data)
                print(f"Error scenario {i+1}: handled gracefully (result: {result})")
                error_handled_count += 1
            except Exception as e:
                print(f"Error scenario {i+1}: unhandled exception: {e}")
        
        # 错误后恢复测试
        print("Testing recovery...")
        recovery_success = 0
        for i in range(50):
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
            
            if await hft_suite.update_market_data(symbol, market_data):
                recovery_success += 1
        
        recovery_rate = recovery_success / 50
        print(f"Recovery success rate: {recovery_rate:.1%}")
        
        # 错误恢复断言
        assert error_handled_count == len(error_scenarios), "Should handle all error scenarios gracefully"
        assert recovery_rate > 0.9, f"Recovery rate {recovery_rate:.1%} too low"
        assert abs(recovery_rate - baseline_rate) < 0.1, "Performance should recover to baseline"
        
        # 检查系统状态
        final_status = hft_suite.get_comprehensive_status()
        assert final_status["running"], "System should still be running after errors"
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, hft_suite):
        """测试资源耗尽处理"""
        if not hft_suite.performance_optimizer or not hft_suite.performance_optimizer.memory_pool:
            pytest.skip("Memory pool not available")
        
        memory_pool = hft_suite.performance_optimizer.memory_pool
        
        # 耗尽内存池
        print("Exhausting memory pool...")
        allocated_buffers = []
        
        try:
            # 分配所有可用缓冲区
            while True:
                buffer = memory_pool.acquire_buffer()
                if buffer is None:
                    break
                allocated_buffers.append(buffer)
            
            print(f"Allocated {len(allocated_buffers)} buffers")
            
            # 在资源耗尽情况下测试系统行为
            print("Testing behavior under resource exhaustion...")
            
            operations_success = 0
            operations_total = 100
            
            for i in range(operations_total):
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time(),
                    price=50000.0 + i,
                    volume=1.0,
                    bid=50000.0 + i - 0.5,
                    ask=50000.0 + i + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                try:
                    if await hft_suite.update_market_data("BTCUSDT", market_data):
                        operations_success += 1
                except Exception as e:
                    print(f"Operation {i} failed under resource exhaustion: {e}")
            
            success_rate = operations_success / operations_total
            print(f"Success rate under resource exhaustion: {success_rate:.1%}")
            
            # 应该能够继续工作，即使资源受限
            assert success_rate > 0.5, f"Success rate {success_rate:.1%} too low under resource pressure"
            
        finally:
            # 释放所有缓冲区
            for buffer in allocated_buffers:
                memory_pool.release_buffer(buffer)
        
        # 测试恢复
        print("Testing recovery after resource release...")
        recovery_success = 0
        for i in range(50):
            market_data = MarketData(
                symbol="BTCUSDT",
                timestamp=time.time(),
                price=50000.0 + i,
                volume=1.0,
                bid=50000.0 + i - 0.5,
                ask=50000.0 + i + 0.5,
                bid_volume=10.0,
                ask_volume=10.0
            )
            
            if await hft_suite.update_market_data("BTCUSDT", market_data):
                recovery_success += 1
        
        recovery_rate = recovery_success / 50
        print(f"Recovery rate after resource release: {recovery_rate:.1%}")
        
        assert recovery_rate > 0.9, f"Recovery rate {recovery_rate:.1%} too low"
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, hft_suite):
        """测试优雅关闭"""
        # 在活跃状态下启动关闭
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        # 启动持续的市场数据更新
        async def continuous_updates():
            count = 0
            while True:
                try:
                    for symbol in symbols:
                        base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=time.time(),
                            price=base_price + count * 0.1,
                            volume=1.0,
                            bid=base_price + count * 0.1 - 0.5,
                            ask=base_price + count * 0.1 + 0.5,
                            bid_volume=10.0,
                            ask_volume=10.0
                        )
                        
                        await hft_suite.update_market_data(symbol, market_data)
                        count += 1
                    
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass
        
        # 启动更新任务
        update_task = asyncio.create_task(continuous_updates())
        
        # 让系统运行一会儿
        await asyncio.sleep(1.0)
        
        # 检查活跃状态
        status_before = hft_suite.get_comprehensive_status()
        assert status_before["running"], "System should be running before shutdown"
        
        # 启动优雅关闭
        print("Initiating graceful shutdown...")
        shutdown_start = time.time()
        
        # 取消更新任务
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        
        # 执行关闭
        await hft_suite.stop()
        
        shutdown_time = time.time() - shutdown_start
        
        # 检查关闭后状态
        status_after = hft_suite.get_comprehensive_status()
        
        print(f"Graceful shutdown completed in {shutdown_time:.3f}s")
        print(f"Final status: {status_after}")
        
        # 优雅关闭断言
        assert not status_after["running"], "System should not be running after shutdown"
        assert shutdown_time < 10.0, f"Shutdown took too long: {shutdown_time:.3f}s"
        
        # 验证没有遗留的异步任务或资源
        # 这里可以添加更多的资源清理验证