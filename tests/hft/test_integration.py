import pytest
import asyncio
import time
from decimal import Decimal
import json

from src.hft import (
    HFTPerformanceSuite, HFTSuiteConfig,
    HFTConfig, PerformanceConfig, NetworkConfig,
    ArbitrageConfig, MarketMakingConfig
)
from src.core.models import MarketData


class TestHFTIntegration:
    """HFT系统集成测试"""
    
    @pytest.fixture
    async def full_hft_suite(self):
        """完整的HFT套件测试夹具"""
        # 配置完整的HFT套件
        config = HFTSuiteConfig()
        
        # HFT引擎配置
        config.hft_config = HFTConfig(
            latency_target_ms=10.0,
            update_interval_ms=1.0
        )
        
        # 性能优化配置（测试环境调整）
        config.performance_config = PerformanceConfig(
            use_uvloop=False,  # 避免测试环境问题
            cpu_affinity_enabled=False,
            memory_pool_size=1024*1024*10,  # 10MB
            gc_optimization=True,
            gc_disable_during_trading=False  # 测试时保持GC
        )
        
        # 网络配置
        config.network_config = NetworkConfig(
            tcp_nodelay=True,
            send_buffer_size=64*1024,
            recv_buffer_size=64*1024
        )
        
        # 套利配置
        config.arbitrage_config = ArbitrageConfig(
            min_profit_bps=5.0,  # 降低阈值用于测试
            min_confidence=0.6,
            max_position_value=Decimal("1000")
        )
        
        # 做市配置
        config.market_making_config = MarketMakingConfig(
            base_spread_bps=15.0,
            min_order_value=Decimal("10"),
            max_order_value=Decimal("100")
        )
        
        # 监控配置
        config.monitoring_interval = 0.1
        config.performance_reporting_interval = 2.0
        
        # 创建并初始化套件
        suite = HFTPerformanceSuite(config)
        await suite.initialize(["BTCUSDT", "ETHUSDT", "ETHBTC"])
        await suite.start()
        
        yield suite
        
        await suite.stop()
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_hft_suite):
        """测试端到端工作流程"""
        symbols = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
        
        print("Testing end-to-end HFT workflow...")
        
        # 1. 初始状态检查
        initial_status = full_hft_suite.get_comprehensive_status()
        assert initial_status["running"], "Suite should be running"
        assert len(initial_status["symbols"]) == 3, "Should track 3 symbols"
        
        print(f"Initial status: {initial_status['metrics']}")
        
        # 2. 市场数据注入和处理
        num_updates = 200
        start_time = time.time()
        
        for i in range(num_updates):
            # 生成相关的市场数据
            btc_price = 50000.0 + i * 0.5
            eth_price = 3000.0 + i * 0.3
            eth_btc_price = eth_price / btc_price  # 理论价格
            
            # 添加一些噪音制造套利机会
            if i % 20 == 0:
                eth_btc_price *= 0.98  # 偶尔制造2%的价差
            
            market_updates = [
                MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time(),
                    price=btc_price,
                    volume=1.0 + i * 0.01,
                    bid=btc_price - 0.5,
                    ask=btc_price + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                ),
                MarketData(
                    symbol="ETHUSDT",
                    timestamp=time.time(),
                    price=eth_price,
                    volume=1.0 + i * 0.01,
                    bid=eth_price - 0.3,
                    ask=eth_price + 0.3,
                    bid_volume=15.0,
                    ask_volume=15.0
                ),
                MarketData(
                    symbol="ETHBTC",
                    timestamp=time.time(),
                    price=eth_btc_price,
                    volume=1.0 + i * 0.01,
                    bid=eth_btc_price - 0.0001,
                    ask=eth_btc_price + 0.0001,
                    bid_volume=20.0,
                    ask_volume=20.0
                )
            ]
            
            # 并发更新所有市场数据
            update_tasks = [
                full_hft_suite.update_market_data(data.symbol, data)
                for data in market_updates
            ]
            results = await asyncio.gather(*update_tasks)
            
            # 验证更新成功
            assert all(results), f"Market data update failed at iteration {i}"
            
            # 短暂休眠模拟真实时间间隔
            await asyncio.sleep(0.001)
        
        processing_time = time.time() - start_time
        throughput = (num_updates * len(symbols)) / processing_time
        
        print(f"Market data processing:")
        print(f"  Updates: {num_updates * len(symbols)}")
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} updates/s")
        
        # 3. 等待策略处理
        await asyncio.sleep(1.0)
        
        # 4. 检查策略执行结果
        if full_hft_suite.arbitrage_agent:
            arb_status = full_hft_suite.arbitrage_agent.get_status()
            arb_stats = full_hft_suite.arbitrage_agent.get_statistics()
            print(f"Arbitrage results: {arb_stats}")
            
            assert arb_status["running"], "Arbitrage agent should be running"
        
        if full_hft_suite.market_making_agent:
            mm_status = full_hft_suite.market_making_agent.get_status()
            mm_stats = full_hft_suite.market_making_agent.get_statistics()
            mm_quotes = full_hft_suite.market_making_agent.get_active_quotes()
            print(f"Market making results: {mm_stats}")
            print(f"Active quotes: {len(mm_quotes)}")
            
            assert mm_status["running"], "Market making agent should be running"
        
        # 5. 性能指标验证
        final_status = full_hft_suite.get_comprehensive_status()
        metrics = final_status["metrics"]
        
        print(f"Final metrics: {metrics}")
        
        assert metrics["health_score"] > 80, f"Health score {metrics['health_score']:.1f} too low"
        assert metrics["avg_latency_ms"] < 20, f"Average latency {metrics['avg_latency_ms']:.2f}ms too high"
        assert throughput > 100, f"Throughput {throughput:.1f} too low"
    
    @pytest.mark.asyncio
    async def test_real_time_performance_monitoring(self, full_hft_suite):
        """测试实时性能监控"""
        monitoring_duration = 10  # 10秒监控
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        print(f"Starting {monitoring_duration}s real-time monitoring...")
        
        # 收集性能数据
        performance_snapshots = []
        
        async def performance_monitor():
            while True:
                try:
                    status = full_hft_suite.get_comprehensive_status()
                    performance_snapshots.append({
                        "timestamp": time.time(),
                        "health_score": status["metrics"]["health_score"],
                        "avg_latency_ms": status["metrics"]["avg_latency_ms"],
                        "uptime": status["uptime_seconds"]
                    })
                    await asyncio.sleep(0.5)  # 每500ms采样一次
                except asyncio.CancelledError:
                    break
        
        # 启动监控任务
        monitor_task = asyncio.create_task(performance_monitor())
        
        # 同时进行市场数据更新
        async def market_data_feed():
            i = 0
            while True:
                try:
                    for symbol in symbols:
                        base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=time.time(),
                            price=base_price + i * 0.1,
                            volume=1.0,
                            bid=base_price + i * 0.1 - 0.5,
                            ask=base_price + i * 0.1 + 0.5,
                            bid_volume=10.0,
                            ask_volume=10.0
                        )
                        
                        await full_hft_suite.update_market_data(symbol, market_data)
                        i += 1
                    
                    await asyncio.sleep(0.01)  # 10ms间隔
                except asyncio.CancelledError:
                    break
        
        # 启动数据源任务
        feed_task = asyncio.create_task(market_data_feed())
        
        # 运行指定时间
        await asyncio.sleep(monitoring_duration)
        
        # 停止任务
        monitor_task.cancel()
        feed_task.cancel()
        
        try:
            await asyncio.gather(monitor_task, feed_task)
        except asyncio.CancelledError:
            pass
        
        # 分析性能数据
        if performance_snapshots:
            health_scores = [s["health_score"] for s in performance_snapshots]
            latencies = [s["avg_latency_ms"] for s in performance_snapshots]
            
            avg_health = sum(health_scores) / len(health_scores)
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_health = min(health_scores)
            
            print(f"Performance monitoring results:")
            print(f"  Samples: {len(performance_snapshots)}")
            print(f"  Average health: {avg_health:.1f}%")
            print(f"  Min health: {min_health:.1f}%")
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            
            # 性能监控断言
            assert avg_health > 75, f"Average health {avg_health:.1f}% too low"
            assert min_health > 50, f"Minimum health {min_health:.1f}% too low"
            assert avg_latency < 15, f"Average latency {avg_latency:.2f}ms too high"
            assert max_latency < 50, f"Max latency {max_latency:.2f}ms too high"
    
    @pytest.mark.asyncio 
    async def test_component_interaction(self, full_hft_suite):
        """测试组件间交互"""
        # 测试各组件是否正确集成
        
        # 1. 检查所有组件都已初始化
        assert full_hft_suite.hft_engine is not None, "HFT engine should be initialized"
        assert full_hft_suite.performance_optimizer is not None, "Performance optimizer should be initialized"
        assert full_hft_suite.arbitrage_agent is not None, "Arbitrage agent should be initialized"
        assert full_hft_suite.market_making_agent is not None, "Market making agent should be initialized"
        
        # 2. 测试信号流
        # 市场数据 -> HFT引擎 -> 微观结构分析 -> 信号 -> 策略Agent
        
        symbol = "BTCUSDT"
        market_data = MarketData(
            symbol=symbol,
            timestamp=time.time(),
            price=50000.0,
            volume=10.0,  # 大成交量
            bid=49999.0,
            ask=50001.0,
            bid_volume=100.0,
            ask_volume=10.0  # 创建失衡
        )
        
        # 注入市场数据
        success = await full_hft_suite.update_market_data(symbol, market_data)
        assert success, "Market data update should succeed"
        
        # 等待信号处理
        await asyncio.sleep(0.1)
        
        # 3. 检查信号生成
        if full_hft_suite.hft_engine:
            signals = full_hft_suite.hft_engine.get_current_signals(symbol)
            print(f"Generated signals: {len(signals)}")
            
            for signal in signals:
                print(f"  Signal: {signal.signal_type} strength={signal.strength:.3f} confidence={signal.confidence:.3f}")
        
        # 4. 检查订单簿状态
        if full_hft_suite.hft_engine:
            orderbook = full_hft_suite.hft_engine.get_orderbook(symbol)
            assert orderbook is not None, "Orderbook should be available"
            
            orderbook_health = full_hft_suite.hft_engine.get_orderbook_health(symbol)
            assert orderbook_health["status"] != "no_data", "Orderbook should have data"
            
            print(f"Orderbook health: {orderbook_health}")
        
        # 5. 检查策略响应
        if full_hft_suite.market_making_agent:
            quotes = full_hft_suite.market_making_agent.get_active_quotes()
            inventory = full_hft_suite.market_making_agent.get_inventory()
            
            print(f"Market making quotes: {len(quotes)}")
            print(f"Market making inventory: {inventory}")
        
        if full_hft_suite.arbitrage_agent:
            opportunities = full_hft_suite.arbitrage_agent.get_active_opportunities()
            positions = full_hft_suite.arbitrage_agent.get_positions()
            
            print(f"Arbitrage opportunities: {len(opportunities)}")
            print(f"Arbitrage positions: {positions}")
    
    @pytest.mark.asyncio
    async def test_emergency_scenarios(self, full_hft_suite):
        """测试紧急场景处理"""
        # 1. 模拟极端市场条件
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        print("Testing emergency scenarios...")
        
        # 极端价格波动
        extreme_market_data = [
            MarketData(
                symbol="BTCUSDT",
                timestamp=time.time(),
                price=30000.0,  # 40%下跌
                volume=1000.0,  # 极大成交量
                bid=29000.0,
                ask=31000.0,    # 极大价差
                bid_volume=1.0,
                ask_volume=1.0
            ),
            MarketData(
                symbol="ETHUSDT", 
                timestamp=time.time(),
                price=1500.0,   # 50%下跌
                volume=2000.0,
                bid=1400.0,
                ask=1600.0,     # 极大价差
                bid_volume=0.1,
                ask_volume=0.1
            )
        ]
        
        # 注入极端数据
        for data in extreme_market_data:
            await full_hft_suite.update_market_data(data.symbol, data)
        
        await asyncio.sleep(0.5)
        
        # 检查系统响应
        status_after_extreme = full_hft_suite.get_comprehensive_status()
        print(f"Status after extreme conditions: {status_after_extreme['metrics']}")
        
        # 2. 测试紧急关闭
        print("Testing emergency shutdown...")
        
        # 记录关闭前状态
        pre_shutdown_status = full_hft_suite.get_comprehensive_status()
        assert pre_shutdown_status["running"], "Should be running before emergency shutdown"
        
        # 执行紧急关闭
        shutdown_start = time.time()
        await full_hft_suite.emergency_shutdown()
        shutdown_time = time.time() - shutdown_start
        
        # 检查关闭后状态
        post_shutdown_status = full_hft_suite.get_comprehensive_status()
        
        print(f"Emergency shutdown completed in {shutdown_time:.3f}s")
        print(f"Post-shutdown status: {post_shutdown_status}")
        
        # 紧急场景断言
        assert not post_shutdown_status["running"], "Should not be running after emergency shutdown"
        assert shutdown_time < 5.0, f"Emergency shutdown took too long: {shutdown_time:.3f}s"
        
        # 验证系统在极端条件下仍然响应
        assert status_after_extreme["metrics"]["health_score"] > 0, "Should maintain some health under extreme conditions"
    
    def test_configuration_validation(self):
        """测试配置验证"""
        # 测试默认配置
        default_config = HFTSuiteConfig()
        assert default_config.hft_config is not None
        assert default_config.performance_config is not None
        assert default_config.network_config is not None
        
        # 测试自定义配置
        custom_config = HFTSuiteConfig(
            enable_arbitrage=False,
            enable_market_making=True,
            monitoring_interval=2.0
        )
        
        assert custom_config.enable_arbitrage is False
        assert custom_config.enable_market_making is True
        assert custom_config.monitoring_interval == 2.0
        
        # 测试配置边界值
        boundary_config = HFTSuiteConfig()
        boundary_config.hft_config.latency_target_ms = 0.1  # 极低延迟要求
        boundary_config.performance_config.memory_pool_size = 1024  # 最小内存池
        
        assert boundary_config.hft_config.latency_target_ms == 0.1
        assert boundary_config.performance_config.memory_pool_size == 1024
    
    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, full_hft_suite):
        """测试综合指标收集"""
        # 运行系统一段时间收集各种指标
        runtime = 5  # 5秒
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        metrics_history = []
        
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < runtime:
            # 市场数据更新
            for symbol in symbols:
                base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=time.time(),
                    price=base_price + update_count * 0.1,
                    volume=1.0,
                    bid=base_price + update_count * 0.1 - 0.5,
                    ask=base_price + update_count * 0.1 + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                await full_hft_suite.update_market_data(symbol, market_data)
                update_count += 1
            
            # 收集指标
            status = full_hft_suite.get_comprehensive_status()
            metrics_history.append({
                "timestamp": time.time() - start_time,
                "health_score": status["metrics"]["health_score"],
                "avg_latency_ms": status["metrics"]["avg_latency_ms"],
                "total_pnl": status["metrics"]["total_pnl"]
            })
            
            await asyncio.sleep(0.1)
        
        # 分析收集的指标
        print(f"Collected {len(metrics_history)} metric snapshots over {runtime}s")
        
        if metrics_history:
            # 健康度分析
            health_scores = [m["health_score"] for m in metrics_history]
            avg_health = sum(health_scores) / len(health_scores)
            health_stability = len([h for h in health_scores if abs(h - avg_health) < 10]) / len(health_scores)
            
            # 延迟分析
            latencies = [m["avg_latency_ms"] for m in metrics_history]
            avg_latency = sum(latencies) / len(latencies)
            latency_stability = len([l for l in latencies if l < avg_latency * 1.5]) / len(latencies)
            
            print(f"Metrics analysis:")
            print(f"  Average health: {avg_health:.1f}%")
            print(f"  Health stability: {health_stability:.1%}")
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  Latency stability: {latency_stability:.1%}")
            
            # 指标收集断言
            assert avg_health > 70, f"Average health {avg_health:.1f}% too low"
            assert health_stability > 0.8, f"Health stability {health_stability:.1%} too low"
            assert avg_latency < 20, f"Average latency {avg_latency:.2f}ms too high"
            assert latency_stability > 0.9, f"Latency stability {latency_stability:.1%} too low"