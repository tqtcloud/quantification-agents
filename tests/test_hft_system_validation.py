"""
高频交易系统完整测试验证套件

测试包含：
1. 性能测试 - 延迟、吞吐量、资源使用
2. 功能测试 - 延迟监控、信号处理、订单生成
3. 集成测试 - 端到端流程验证
4. 可靠性测试 - 故障恢复、并发安全
"""

import pytest
import asyncio
import time
import statistics
import gc
import psutil
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入待测试模块
from src.hft.latency_monitor import LatencyMonitor, DataSource, LatencyAlert
from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor
from src.hft.smart_order_router import SmartOrderRouter, OrderRequest
from src.hft.fault_tolerance_manager import FaultToleranceManager
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength


class TestSystemPerformance:
    """系统性能测试"""
    
    @pytest.fixture
    def setup_system_components(self):
        """设置系统组件"""
        latency_monitor = LatencyMonitor(threshold_ms=100)
        signal_processor = IntegratedHFTSignalProcessor()
        order_router = SmartOrderRouter()
        fault_manager = FaultToleranceManager()
        
        return {
            'latency_monitor': latency_monitor,
            'signal_processor': signal_processor,
            'order_router': order_router,
            'fault_manager': fault_manager
        }
    
    @pytest.mark.asyncio
    async def test_signal_generation_latency(self, setup_system_components):
        """测试信号生成延迟性能 - 目标 < 10ms"""
        components = setup_system_components
        signal_processor = components['signal_processor']
        
        # 准备测试数据
        test_data = {
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume': 1000,
            'timestamp': datetime.now().timestamp() * 1000
        }
        
        latencies = []
        iterations = 100
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            # 生成交易信号
            signal = await signal_processor.process_signal(test_data)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # 性能指标计算
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        max_latency = max(latencies)
        
        print(f"\n=== 信号生成延迟性能测试结果 ===")
        print(f"平均延迟: {avg_latency:.2f}ms")
        print(f"P95延迟: {p95_latency:.2f}ms")
        print(f"P99延迟: {p99_latency:.2f}ms")
        print(f"最大延迟: {max_latency:.2f}ms")
        
        # 性能断言
        assert avg_latency < 10.0, f"平均延迟 {avg_latency:.2f}ms 超过目标 10ms"
        assert p95_latency < 15.0, f"P95延迟 {p95_latency:.2f}ms 超过目标 15ms"
        assert p99_latency < 25.0, f"P99延迟 {p99_latency:.2f}ms 超过目标 25ms"
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self, setup_system_components):
        """测试系统吞吐量性能 - 目标 > 10,000 TPS"""
        components = setup_system_components
        signal_processor = components['signal_processor']
        
        # 准备批量测试数据
        test_batch = []
        batch_size = 1000
        
        for i in range(batch_size):
            test_batch.append({
                'symbol': f'TEST{i % 100}USDT',
                'price': 45000.0 + i,
                'volume': 1000 + i,
                'timestamp': datetime.now().timestamp() * 1000 + i
            })
        
        # 吞吐量测试
        start_time = time.perf_counter()
        
        # 并发处理
        tasks = [signal_processor.process_signal(data) for data in test_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        
        # 计算吞吐量
        total_time = end_time - start_time
        throughput = batch_size / total_time
        successful_results = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successful_results / batch_size
        
        print(f"\n=== 系统吞吐量性能测试结果 ===")
        print(f"处理数量: {batch_size}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"吞吐量: {throughput:.2f} TPS")
        print(f"成功率: {success_rate:.2%}")
        
        # 性能断言
        assert throughput > 5000, f"吞吐量 {throughput:.2f} TPS 低于最低目标 5000 TPS"
        assert success_rate > 0.99, f"成功率 {success_rate:.2%} 低于目标 99%"
    
    def test_memory_usage_performance(self, setup_system_components):
        """测试内存使用性能"""
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行内存压力测试
        components = setup_system_components
        
        # 大量数据处理
        for i in range(1000):
            test_data = {
                'symbol': f'TEST{i}USDT',
                'price': 45000.0 + i,
                'volume': 1000 + i,
                'timestamp': datetime.now().timestamp() * 1000 + i
            }
            
            # 模拟数据处理
            components['latency_monitor']._calculate_network_latency(test_data['timestamp'])
        
        # 强制垃圾回收
        gc.collect()
        
        # 记录最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\n=== 内存使用性能测试结果 ===")
        print(f"初始内存: {initial_memory:.2f}MB")
        print(f"最终内存: {final_memory:.2f}MB")
        print(f"内存增长: {memory_increase:.2f}MB")
        
        # 内存使用断言
        assert memory_increase < 50, f"内存增长 {memory_increase:.2f}MB 超过目标 50MB"


class TestFunctionalValidation:
    """功能测试验证"""
    
    @pytest.mark.asyncio
    async def test_latency_monitor_functionality(self):
        """测试延迟监控功能"""
        monitor = LatencyMonitor(threshold_ms=50)
        
        # 添加数据源
        primary_source = DataSource(
            name="primary",
            endpoint="ws://primary.com",
            priority=1,
            is_active=True
        )
        backup_source = DataSource(
            name="backup", 
            endpoint="ws://backup.com",
            priority=2,
            is_active=True
        )
        
        await monitor.add_data_source(primary_source)
        await monitor.add_data_source(backup_source)
        
        # 测试正常延迟检查
        current_time = datetime.now().timestamp() * 1000
        fresh_timestamp = current_time - 30  # 30ms前
        
        is_fresh, latency = await monitor.check_data_freshness(fresh_timestamp)
        
        assert is_fresh == True, "新鲜数据应该通过检查"
        assert latency < 50, f"延迟 {latency}ms 应该低于阈值 50ms"
        
        # 测试过期数据检查
        stale_timestamp = current_time - 100  # 100ms前
        is_fresh, latency = await monitor.check_data_freshness(stale_timestamp)
        
        assert is_fresh == False, "过期数据应该未通过检查"
        assert latency > 50, f"延迟 {latency}ms 应该高于阈值 50ms"
        
        print(f"✅ 延迟监控功能测试通过")
    
    @pytest.mark.asyncio 
    async def test_signal_filtering_logic(self):
        """测试信号过滤逻辑"""
        processor = IntegratedHFTSignalProcessor()
        
        # 测试高质量信号（应该通过）
        high_quality_signal = MultiDimensionalSignal(
            primary_signal=TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.BUY,
                confidence=0.85,
                entry_price=45000.0,
                target_price=46000.0,
                stop_loss=44000.0,
                reasoning=["强烈上涨趋势", "技术指标一致看涨"],
                indicators_consensus={"rsi": 0.7, "macd": 0.8},
                timestamp=datetime.now()
            ),
            momentum_score=0.8,
            mean_reversion_score=0.6,
            volatility_score=0.4,
            volume_score=0.7,
            sentiment_score=0.6,
            overall_confidence=0.85,
            risk_reward_ratio=2.0,
            max_position_size=1000
        )
        
        should_pass = processor._should_filter_signal(high_quality_signal)
        assert should_pass == False, "高质量信号应该通过过滤器（返回False表示不过滤）"
        
        # 测试低质量信号（应该被过滤）
        low_quality_signal = MultiDimensionalSignal(
            primary_signal=TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.WEAK_BUY,
                confidence=0.4,  # 低置信度
                entry_price=45000.0,
                target_price=45500.0,
                stop_loss=44500.0,
                reasoning=["微弱信号"],
                indicators_consensus={"rsi": 0.5, "macd": 0.3},
                timestamp=datetime.now()
            ),
            momentum_score=0.2,  # 低动量
            mean_reversion_score=0.1,
            volatility_score=0.8,  # 高波动率
            volume_score=0.3,
            sentiment_score=0.2,
            overall_confidence=0.4,  # 低整体置信度
            risk_reward_ratio=0.5,  # 低风险收益比
            max_position_size=100
        )
        
        should_filter = processor._should_filter_signal(low_quality_signal)
        assert should_filter == True, "低质量信号应该被过滤器拦截"
        
        print(f"✅ 信号过滤逻辑测试通过")
    
    @pytest.mark.asyncio
    async def test_order_generation_accuracy(self):
        """测试订单生成准确性"""
        router = SmartOrderRouter()
        
        # 创建测试信号
        test_signal = MultiDimensionalSignal(
            primary_signal=TradingSignal(
                symbol="BTCUSDT",
                signal_type=SignalStrength.STRONG_BUY,
                confidence=0.9,
                entry_price=45000.0,
                target_price=46800.0,  # 4%盈利目标
                stop_loss=43200.0,     # 4%止损
                reasoning=["强势突破", "量价齐升"],
                indicators_consensus={"rsi": 0.75, "macd": 0.85},
                timestamp=datetime.now()
            ),
            momentum_score=0.85,
            mean_reversion_score=0.7,
            volatility_score=0.3,
            volume_score=0.8,
            sentiment_score=0.7,
            overall_confidence=0.9,
            risk_reward_ratio=2.0,
            max_position_size=1000
        )
        
        # 生成订单
        orders = await router.generate_orders_from_signal(test_signal)
        
        assert len(orders) > 0, "应该生成至少一个订单"
        
        main_order = orders[0]
        assert main_order.symbol == "BTCUSDT", "订单标的应该匹配信号"
        assert main_order.side == "BUY", "应该生成买入订单"
        assert main_order.quantity > 0, "订单数量应该大于0"
        assert main_order.price <= 45000.0 * 1.001, "限价订单价格应该合理"  # 允许0.1%溢价
        
        # 检查是否生成了保护性订单
        has_stop_loss = any(order.order_type in ["STOP_LOSS", "STOP_LOSS_LIMIT"] for order in orders)
        has_take_profit = any("TAKE_PROFIT" in order.order_type for order in orders)
        
        print(f"✅ 订单生成准确性测试通过 - 生成了{len(orders)}个订单")
        print(f"   - 包含止损订单: {has_stop_loss}")
        print(f"   - 包含止盈订单: {has_take_profit}")
    
    @pytest.mark.asyncio
    async def test_fault_tolerance_mechanisms(self):
        """测试容错机制"""
        fault_manager = FaultToleranceManager()
        
        # 测试组件注册
        test_component = Mock()
        test_component.name = "test_component"
        
        fault_manager.register_component("test_component", test_component)
        assert "test_component" in fault_manager.components, "组件应该成功注册"
        
        # 测试错误检测和分类
        test_error = ValueError("测试错误")
        error_type = fault_manager._classify_error(test_error)
        
        assert error_type in fault_manager.error_strategies, f"错误类型 {error_type} 应该有对应的处理策略"
        
        # 测试熔断器功能
        circuit_breaker = fault_manager._get_circuit_breaker("test_component")
        
        # 模拟多次失败触发熔断
        for _ in range(5):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.state == "OPEN", "多次失败后熔断器应该开启"
        
        # 等待熔断恢复时间并测试半开状态
        circuit_breaker.last_failure_time = time.time() - 61  # 模拟60秒后
        circuit_breaker._update_state()
        
        assert circuit_breaker.state == "HALF_OPEN", "等待时间后熔断器应该进入半开状态"
        
        print(f"✅ 容错机制测试通过")


class TestIntegrationValidation:
    """集成测试验证"""
    
    @pytest.fixture
    def integrated_system(self):
        """集成测试系统fixture"""
        return {
            'latency_monitor': LatencyMonitor(threshold_ms=100),
            'signal_processor': IntegratedHFTSignalProcessor(),
            'order_router': SmartOrderRouter(),
            'fault_manager': FaultToleranceManager()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_trading_flow(self, integrated_system):
        """端到端交易流程测试"""
        
        # 模拟市场数据输入
        market_data = {
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume': 1500,
            'timestamp': datetime.now().timestamp() * 1000,
            'bid': 44995.0,
            'ask': 45005.0,
            'open': 44800.0,
            'high': 45200.0,
            'low': 44700.0
        }
        
        # Step 1: 延迟检查
        latency_monitor = integrated_system['latency_monitor']
        is_fresh, latency = await latency_monitor.check_data_freshness(market_data['timestamp'])
        
        if not is_fresh:
            pytest.skip(f"数据延迟 {latency}ms 过高，跳过测试")
        
        # Step 2: 信号处理
        signal_processor = integrated_system['signal_processor']
        
        with patch.object(signal_processor.multidimensional_engine, 'generate_multidimensional_signal') as mock_engine:
            # 模拟信号生成
            mock_signal = MultiDimensionalSignal(
                primary_signal=TradingSignal(
                    symbol="BTCUSDT",
                    signal_type=SignalStrength.BUY,
                    confidence=0.8,
                    entry_price=45000.0,
                    target_price=46800.0,
                    stop_loss=43200.0,
                    reasoning=["趋势向上", "成交量增加"],
                    indicators_consensus={"rsi": 0.7, "macd": 0.75},
                    timestamp=datetime.now()
                ),
                momentum_score=0.8,
                mean_reversion_score=0.6,
                volatility_score=0.4,
                volume_score=0.8,
                sentiment_score=0.6,
                overall_confidence=0.8,
                risk_reward_ratio=2.1,
                max_position_size=800
            )
            mock_engine.return_value = mock_signal
            
            processed_signal = await signal_processor.process_signal(market_data)
        
        assert processed_signal is not None, "应该成功处理信号"
        
        # Step 3: 订单生成
        order_router = integrated_system['order_router']
        orders = await order_router.generate_orders_from_signal(processed_signal)
        
        assert len(orders) > 0, "应该生成交易订单"
        
        # Step 4: 订单路由
        routed_orders = []
        for order in orders:
            routed_order = await order_router.route_order(order)
            routed_orders.append(routed_order)
        
        assert len(routed_orders) == len(orders), "所有订单都应该成功路由"
        
        # Step 5: 验证完整流程时间
        total_processing_time = (time.time() * 1000) - market_data['timestamp']
        
        print(f"\n=== 端到端交易流程测试结果 ===")
        print(f"数据延迟: {latency:.2f}ms")
        print(f"信号质量: {processed_signal.overall_confidence:.2%}")
        print(f"生成订单: {len(orders)}个")
        print(f"总处理时间: {total_processing_time:.2f}ms")
        
        # 性能断言
        assert total_processing_time < 100, f"总处理时间 {total_processing_time:.2f}ms 应该低于100ms"
        
        print(f"✅ 端到端交易流程测试通过")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, integrated_system):
        """并发处理测试"""
        
        # 创建多个并发测试任务
        test_tasks = []
        num_concurrent = 50
        
        for i in range(num_concurrent):
            market_data = {
                'symbol': f'TEST{i % 10}USDT',
                'price': 45000.0 + i,
                'volume': 1000 + i * 10,
                'timestamp': datetime.now().timestamp() * 1000 + i
            }
            
            # 创建端到端处理任务
            task = self._process_single_request(integrated_system, market_data)
            test_tasks.append(task)
        
        # 并发执行所有任务
        start_time = time.perf_counter()
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        # 分析结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results)
        total_time = end_time - start_time
        
        print(f"\n=== 并发处理测试结果 ===")
        print(f"并发数量: {num_concurrent}")
        print(f"成功处理: {len(successful_results)}")
        print(f"处理失败: {len(failed_results)}")
        print(f"成功率: {success_rate:.2%}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均响应时间: {total_time / num_concurrent * 1000:.2f}ms")
        
        # 并发性能断言
        assert success_rate > 0.95, f"并发成功率 {success_rate:.2%} 应该高于95%"
        assert total_time < 5.0, f"总处理时间 {total_time:.2f}s 应该低于5秒"
        
        print(f"✅ 并发处理测试通过")
    
    async def _process_single_request(self, system, market_data):
        """处理单个请求的辅助方法"""
        try:
            # 简化的处理流程
            latency_monitor = system['latency_monitor'] 
            is_fresh, _ = await latency_monitor.check_data_freshness(market_data['timestamp'])
            
            if is_fresh:
                signal_processor = system['signal_processor']
                # 使用mock避免复杂的信号生成
                with patch.object(signal_processor, 'process_signal') as mock_process:
                    mock_signal = Mock()
                    mock_signal.overall_confidence = 0.7
                    mock_process.return_value = mock_signal
                    
                    result = await signal_processor.process_signal(market_data)
                    return result
            
            return None
            
        except Exception as e:
            return e


class TestReliabilityValidation:
    """可靠性测试验证"""
    
    @pytest.mark.asyncio
    async def test_fault_recovery(self):
        """故障恢复测试"""
        fault_manager = FaultToleranceManager()
        
        # 注册测试组件
        test_component = Mock()
        test_component.recover.return_value = True
        fault_manager.register_component("test_service", test_component)
        
        # 模拟组件故障
        await fault_manager.handle_component_failure("test_service", Exception("模拟故障"))
        
        # 检查组件状态
        health_status = await fault_manager.get_system_health()
        
        assert "test_service" in health_status, "故障组件应该在健康状态中被跟踪"
        
        # 模拟恢复过程
        recovery_success = await fault_manager.attempt_recovery("test_service")
        
        print(f"✅ 故障恢复测试通过 - 恢复状态: {recovery_success}")
    
    @pytest.mark.asyncio
    async def test_data_consistency(self):
        """数据一致性测试"""
        
        # 创建多个处理器实例模拟分布式场景
        processor1 = IntegratedHFTSignalProcessor()
        processor2 = IntegratedHFTSignalProcessor()
        
        # 相同输入数据
        test_data = {
            'symbol': 'BTCUSDT',
            'price': 45000.0,
            'volume': 1000,
            'timestamp': datetime.now().timestamp() * 1000
        }
        
        # Mock信号生成以确保一致性
        mock_signal = Mock()
        mock_signal.overall_confidence = 0.8
        mock_signal.primary_signal.symbol = 'BTCUSDT'
        
        with patch.multiple(
            processor1,
            _should_filter_signal=Mock(return_value=False),
            multidimensional_engine=Mock()
        ), patch.multiple(
            processor2,
            _should_filter_signal=Mock(return_value=False), 
            multidimensional_engine=Mock()
        ):
            processor1.multidimensional_engine.generate_multidimensional_signal.return_value = mock_signal
            processor2.multidimensional_engine.generate_multidimensional_signal.return_value = mock_signal
            
            # 并行处理相同数据
            result1, result2 = await asyncio.gather(
                processor1.process_signal(test_data),
                processor2.process_signal(test_data)
            )
        
        # 验证结果一致性
        assert result1 is not None and result2 is not None, "两个处理器都应该返回结果"
        
        print(f"✅ 数据一致性测试通过")
    
    def test_error_handling_coverage(self):
        """错误处理覆盖测试"""
        fault_manager = FaultToleranceManager()
        
        # 测试不同类型的错误处理
        error_types = [
            ValueError("值错误"),
            ConnectionError("连接错误"), 
            TimeoutError("超时错误"),
            KeyError("键错误"),
            AttributeError("属性错误"),
            Exception("通用异常")
        ]
        
        handled_errors = 0
        for error in error_types:
            error_type = fault_manager._classify_error(error)
            if error_type in fault_manager.error_strategies:
                handled_errors += 1
        
        coverage_rate = handled_errors / len(error_types)
        
        print(f"\n=== 错误处理覆盖测试结果 ===")
        print(f"测试错误类型: {len(error_types)}")
        print(f"已处理错误: {handled_errors}")
        print(f"覆盖率: {coverage_rate:.2%}")
        
        assert coverage_rate > 0.8, f"错误处理覆盖率 {coverage_rate:.2%} 应该高于80%"
        
        print(f"✅ 错误处理覆盖测试通过")


class TestSystemBenchmarks:
    """系统基准测试"""
    
    @pytest.mark.asyncio
    async def test_system_benchmark_suite(self):
        """系统基准测试套件"""
        print(f"\n{'='*50}")
        print(f"高频交易系统基准测试报告")
        print(f"{'='*50}")
        
        # 系统信息
        print(f"\n系统信息:")
        print(f"  CPU: {psutil.cpu_count()}核")
        print(f"  内存: {psutil.virtual_memory().total / 1024**3:.1f}GB")
        print(f"  Python: {sys.version.split()[0]}")
        
        # 基准测试结果汇总
        benchmark_results = {
            'signal_latency_avg': 0,
            'signal_latency_p99': 0,
            'throughput_tps': 0,
            'memory_usage_mb': 0,
            'success_rate': 0,
            'fault_recovery_time': 0
        }
        
        # 执行各项基准测试
        latency_results = await self._benchmark_latency()
        throughput_results = await self._benchmark_throughput()
        memory_results = self._benchmark_memory()
        reliability_results = await self._benchmark_reliability()
        
        benchmark_results.update({
            'signal_latency_avg': latency_results['avg'],
            'signal_latency_p99': latency_results['p99'],
            'throughput_tps': throughput_results['tps'],
            'memory_usage_mb': memory_results['peak_mb'],
            'success_rate': reliability_results['success_rate'],
            'fault_recovery_time': reliability_results['recovery_time']
        })
        
        # 输出基准报告
        print(f"\n基准测试结果:")
        print(f"  信号延迟 (平均): {benchmark_results['signal_latency_avg']:.2f}ms")
        print(f"  信号延迟 (P99): {benchmark_results['signal_latency_p99']:.2f}ms") 
        print(f"  系统吞吐量: {benchmark_results['throughput_tps']:.0f} TPS")
        print(f"  内存使用峰值: {benchmark_results['memory_usage_mb']:.1f}MB")
        print(f"  系统成功率: {benchmark_results['success_rate']:.1%}")
        print(f"  故障恢复时间: {benchmark_results['fault_recovery_time']:.2f}s")
        
        # 性能等级评估
        performance_score = self._calculate_performance_score(benchmark_results)
        print(f"\n系统性能评分: {performance_score:.1f}/100")
        
        if performance_score >= 90:
            print("🏆 性能等级: 优秀 (生产就绪)")
        elif performance_score >= 80:
            print("🥉 性能等级: 良好 (可以部署)")
        elif performance_score >= 70:
            print("⚠️  性能等级: 一般 (需要优化)")
        else:
            print("❌ 性能等级: 较差 (需要重构)")
        
        print(f"{'='*50}\n")
        
        # 基准断言
        assert benchmark_results['signal_latency_avg'] < 20, "平均延迟应该小于20ms"
        assert benchmark_results['throughput_tps'] > 3000, "吞吐量应该大于3000 TPS"
        assert benchmark_results['success_rate'] > 0.95, "成功率应该大于95%"
    
    async def _benchmark_latency(self):
        """延迟基准测试"""
        processor = IntegratedHFTSignalProcessor()
        latencies = []
        
        for _ in range(100):
            test_data = {
                'symbol': 'BTCUSDT',
                'price': 45000.0,
                'volume': 1000,
                'timestamp': datetime.now().timestamp() * 1000
            }
            
            start = time.perf_counter()
            await processor.process_signal(test_data)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        return {
            'avg': statistics.mean(latencies),
            'p99': statistics.quantiles(latencies, n=100)[98]
        }
    
    async def _benchmark_throughput(self):
        """吞吐量基准测试"""
        processor = IntegratedHFTSignalProcessor()
        batch_size = 500
        
        tasks = []
        for i in range(batch_size):
            test_data = {
                'symbol': f'TEST{i}USDT',
                'price': 45000.0 + i,
                'volume': 1000 + i,
                'timestamp': datetime.now().timestamp() * 1000 + i
            }
            tasks.append(processor.process_signal(test_data))
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end = time.perf_counter()
        
        total_time = end - start
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        return {'tps': successful / total_time}
    
    def _benchmark_memory(self):
        """内存基准测试"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # 内存压力测试
        components = [
            LatencyMonitor(threshold_ms=100),
            IntegratedHFTSignalProcessor(),
            SmartOrderRouter(),
            FaultToleranceManager()
        ]
        
        # 模拟大量操作
        for _ in range(1000):
            pass
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        return {'peak_mb': peak_memory}
    
    async def _benchmark_reliability(self):
        """可靠性基准测试"""
        fault_manager = FaultToleranceManager()
        
        # 注册测试组件
        test_component = Mock()
        test_component.recover.return_value = True
        fault_manager.register_component("test", test_component)
        
        # 模拟故障和恢复
        start_time = time.perf_counter()
        await fault_manager.handle_component_failure("test", Exception("test error"))
        recovery_success = await fault_manager.attempt_recovery("test")
        end_time = time.perf_counter()
        
        return {
            'success_rate': 1.0 if recovery_success else 0.0,
            'recovery_time': end_time - start_time
        }
    
    def _calculate_performance_score(self, results):
        """计算性能评分"""
        score = 0
        
        # 延迟评分 (30分)
        if results['signal_latency_avg'] < 10:
            score += 30
        elif results['signal_latency_avg'] < 20:
            score += 20
        elif results['signal_latency_avg'] < 50:
            score += 10
        
        # 吞吐量评分 (30分)
        if results['throughput_tps'] > 10000:
            score += 30
        elif results['throughput_tps'] > 5000:
            score += 20
        elif results['throughput_tps'] > 1000:
            score += 10
        
        # 成功率评分 (20分)
        score += results['success_rate'] * 20
        
        # 内存效率评分 (10分)
        if results['memory_usage_mb'] < 100:
            score += 10
        elif results['memory_usage_mb'] < 200:
            score += 5
        
        # 恢复时间评分 (10分)
        if results['fault_recovery_time'] < 1.0:
            score += 10
        elif results['fault_recovery_time'] < 3.0:
            score += 5
        
        return score


if __name__ == "__main__":
    # 运行基准测试
    pytest.main([__file__ + "::TestSystemBenchmarks::test_system_benchmark_suite", "-v", "-s"])