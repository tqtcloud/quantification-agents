"""
HFT集成测试套件

测试目标:
- 端到端交易流程测试  
- 多组件协作测试
- 真实场景模拟测试
- 边界条件测试
"""
import asyncio
import pytest
import time
import json
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch, call
from dataclasses import dataclass, field
import numpy as np

from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, DataSourceStatus
from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor, OrderRequest, OrderType
from src.hft.smart_order_router import SmartOrderRouter, VenueInfo, ExecutionAlgorithm
from src.hft.fault_tolerance_manager import FaultToleranceManager, ErrorCategory
from src.core.models.trading import MarketData
from src.core.models.signals import TradingSignal, SignalStrength


@dataclass
class IntegratedTestResult:
    """集成测试结果"""
    test_name: str
    success: bool
    execution_time_ms: float
    components_involved: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HFTSystemIntegrator:
    """HFT系统集成器 - 用于集成测试"""
    
    def __init__(self):
        self.latency_monitor: Optional[LatencyMonitor] = None
        self.signal_processor: Optional[IntegratedHFTSignalProcessor] = None
        self.order_router: Optional[SmartOrderRouter] = None
        self.fault_manager: Optional[FaultToleranceManager] = None
        
        self.execution_history: List[Dict] = []
        self.error_log: List[Dict] = []
        
    async def initialize(self, config: Dict[str, Any] = None):
        """初始化系统组件"""
        config = config or {}
        
        # 初始化延迟监控器
        self.latency_monitor = LatencyMonitor(
            staleness_threshold_ms=config.get('staleness_threshold_ms', 100.0),
            stats_window_size=config.get('stats_window_size', 1000)
        )
        
        data_sources = config.get('data_sources', [
            DataSourceConfig(name="binance", priority=1, max_latency_ms=50.0),
            DataSourceConfig(name="okx", priority=2, max_latency_ms=80.0)
        ])
        await self.latency_monitor.initialize(data_sources)
        await self.latency_monitor.start()
        
        # 初始化信号处理器
        self.signal_processor = IntegratedHFTSignalProcessor()
        
        # 初始化订单路由器
        self.order_router = SmartOrderRouter()
        venues = config.get('venues', [
            VenueInfo(
                name="binance", priority=1, latency_ms=15.0,
                liquidity_score=0.95, fee_rate=0.001,
                min_order_size=0.001, max_order_size=1000.0
            )
        ])
        await self.order_router.initialize(venues)
        
        # 初始化容错管理器
        self.fault_manager = FaultToleranceManager()
        await self.fault_manager.start()
        
        # 注册系统组件到容错管理器
        components = [
            ("latency_monitor", 3, 5.0),
            ("signal_processor", 5, 10.0),
            ("order_router", 3, 5.0)
        ]
        
        for comp_name, threshold, timeout in components:
            await self.fault_manager.register_component(
                comp_name, failure_threshold=threshold, recovery_timeout=timeout
            )
    
    async def shutdown(self):
        """关闭系统组件"""
        if self.latency_monitor:
            await self.latency_monitor.stop()
        if self.order_router:
            await self.order_router.shutdown()
        if self.fault_manager:
            await self.fault_manager.stop()
    
    async def process_market_data_to_order(self, market_data: MarketData) -> Optional[OrderRequest]:
        """完整的从市场数据到订单的处理流程"""
        start_time = time.time()
        
        try:
            # 1. 延迟检查
            is_fresh, latency_metrics = await self.latency_monitor.check_data_freshness(
                symbol=market_data.symbol,
                market_data=market_data,
                data_source="binance"
            )
            
            if not is_fresh:
                await self.fault_manager.handle_error(
                    "latency_monitor",
                    Exception("Stale data detected"),
                    {"symbol": market_data.symbol, "latency": latency_metrics.total_latency_ms}
                )
                return None
            
            # 2. 信号生成（模拟）
            signal = self._generate_trading_signal(market_data)
            
            # 3. 信号处理和过滤
            order_request = await self._process_signal_to_order(signal, market_data)
            
            if order_request:
                # 4. 记录成功执行
                await self.fault_manager.record_success(
                    "signal_processor",
                    response_time=(time.time() - start_time) * 1000
                )
                
                execution_record = {
                    'timestamp': time.time(),
                    'symbol': market_data.symbol,
                    'action': 'market_data_to_order',
                    'latency_ms': latency_metrics.total_latency_ms,
                    'order_generated': True,
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
                self.execution_history.append(execution_record)
            
            return order_request
            
        except Exception as e:
            await self.fault_manager.handle_error(
                "signal_processor",
                e,
                {"symbol": market_data.symbol, "processing_stage": "market_data_to_order"}
            )
            self.error_log.append({
                'timestamp': time.time(),
                'error': str(e),
                'symbol': market_data.symbol,
                'stage': 'market_data_to_order'
            })
            return None
    
    async def execute_order_workflow(self, order_request: OrderRequest) -> Dict[str, Any]:
        """执行完整的订单工作流"""
        start_time = time.time()
        result = {
            'success': False,
            'order_id': f"order_{int(time.time() * 1000)}",
            'execution_time_ms': 0,
            'child_orders': [],
            'errors': []
        }
        
        try:
            # 1. 订单路由决策
            routing_decision = await self._make_routing_decision(order_request)
            
            # 2. 订单拆分（如果需要）
            child_orders = await self._split_order_if_needed(order_request, routing_decision)
            
            # 3. 执行子订单
            execution_results = []
            for child_order in child_orders:
                child_result = await self._execute_child_order(child_order)
                execution_results.append(child_result)
            
            # 4. 聚合执行结果
            result['child_orders'] = execution_results
            result['success'] = all(r.get('success', False) for r in execution_results)
            result['execution_time_ms'] = (time.time() - start_time) * 1000
            
            # 5. 记录成功执行
            if result['success']:
                await self.fault_manager.record_success(
                    "order_router",
                    response_time=result['execution_time_ms']
                )
            
            return result
            
        except Exception as e:
            await self.fault_manager.handle_error(
                "order_router", 
                e,
                {"order_id": result['order_id'], "symbol": order_request.symbol}
            )
            result['errors'].append(str(e))
            return result
    
    def _generate_trading_signal(self, market_data: MarketData) -> TradingSignal:
        """生成交易信号（模拟）"""
        # 简单的价格变动信号
        price_change = (market_data.close - market_data.open) / market_data.open
        
        if price_change > 0.001:  # 0.1%涨幅
            action = "buy"
            strength = SignalStrength.STRONG if abs(price_change) > 0.005 else SignalStrength.MEDIUM
        elif price_change < -0.001:  # 0.1%跌幅  
            action = "sell"
            strength = SignalStrength.STRONG if abs(price_change) > 0.005 else SignalStrength.MEDIUM
        else:
            action = "hold"
            strength = SignalStrength.WEAK
        
        return TradingSignal(
            symbol=market_data.symbol,
            action=action,
            strength=strength,
            confidence=0.7 + abs(price_change) * 100,  # 基于价格变动的置信度
            timestamp=market_data.timestamp,
            metadata={'price_change': price_change}
        )
    
    async def _process_signal_to_order(self, signal: TradingSignal, market_data: MarketData) -> Optional[OrderRequest]:
        """将信号转换为订单请求"""
        if signal.action == "hold" or signal.strength == SignalStrength.WEAK:
            return None
        
        # 计算订单参数
        quantity = self._calculate_order_quantity(signal, market_data)
        price = self._calculate_order_price(signal, market_data)
        
        return OrderRequest(
            symbol=signal.symbol,
            order_type=OrderType.LIMIT if signal.strength != SignalStrength.STRONG else OrderType.MARKET,
            side=signal.action,
            quantity=quantity,
            price=price,
            urgency_score=0.8 if signal.strength == SignalStrength.STRONG else 0.6,
            confidence=signal.confidence,
            signal_id=f"signal_{int(signal.timestamp)}"
        )
    
    def _calculate_order_quantity(self, signal: TradingSignal, market_data: MarketData) -> float:
        """计算订单数量"""
        # 基于信号强度的简单数量计算
        base_quantity = 1.0
        
        if signal.strength == SignalStrength.STRONG:
            return base_quantity * 1.5
        elif signal.strength == SignalStrength.MEDIUM:
            return base_quantity
        else:
            return base_quantity * 0.5
    
    def _calculate_order_price(self, signal: TradingSignal, market_data: MarketData) -> Optional[float]:
        """计算订单价格"""
        if signal.action == "buy":
            # 买入时使用稍高于当前价格
            return market_data.close * 1.001  # 0.1%溢价
        elif signal.action == "sell":
            # 卖出时使用稍低于当前价格
            return market_data.close * 0.999  # 0.1%折扣
        return None
    
    async def _make_routing_decision(self, order_request: OrderRequest) -> Dict[str, Any]:
        """制定路由决策"""
        return {
            'primary_venue': 'binance',
            'backup_venues': ['okx'],
            'algorithm': ExecutionAlgorithm.TWAP if order_request.quantity > 5.0 else ExecutionAlgorithm.SNIPER,
            'split_required': order_request.quantity > 10.0
        }
    
    async def _split_order_if_needed(self, order_request: OrderRequest, routing_decision: Dict) -> List[Dict]:
        """根据需要拆分订单"""
        if not routing_decision.get('split_required', False):
            return [{
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': order_request.quantity,
                'price': order_request.price,
                'venue': routing_decision['primary_venue']
            }]
        
        # 拆分大订单
        chunk_size = order_request.quantity / 3  # 拆分为3个子订单
        return [
            {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': chunk_size,
                'price': order_request.price,
                'venue': routing_decision['primary_venue']
            } for _ in range(3)
        ]
    
    async def _execute_child_order(self, child_order: Dict) -> Dict[str, Any]:
        """执行子订单"""
        # 模拟订单执行
        await asyncio.sleep(0.01)  # 10ms执行时间
        
        # 模拟90%成功率
        success = np.random.random() > 0.1
        
        if success:
            return {
                'success': True,
                'filled_quantity': child_order['quantity'],
                'avg_fill_price': child_order['price'] * (1 + np.random.normal(0, 0.0001)),  # 微小滑点
                'execution_time_ms': 10 + np.random.normal(0, 2)
            }
        else:
            return {
                'success': False,
                'error': 'Simulated execution failure',
                'filled_quantity': 0
            }


@pytest.mark.asyncio
class TestEndToEndFlow:
    """端到端流程测试"""
    
    async def test_complete_trading_workflow(self, sample_market_data):
        """测试完整交易工作流"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            results = []
            successful_orders = 0
            
            # 处理一批市场数据
            for market_data in sample_market_data[:20]:
                # 从市场数据到订单
                order_request = await integrator.process_market_data_to_order(market_data)
                
                if order_request:
                    # 执行订单工作流
                    execution_result = await integrator.execute_order_workflow(order_request)
                    results.append(execution_result)
                    
                    if execution_result['success']:
                        successful_orders += 1
            
            # 验证结果
            assert len(results) > 0, "应该生成一些订单"
            success_rate = successful_orders / len(results)
            assert success_rate >= 0.8, f"成功率 {success_rate:.2%} 应该 >= 80%"
            
            # 验证执行历史
            assert len(integrator.execution_history) > 0, "应该记录执行历史"
            
            # 计算平均处理延迟
            processing_latencies = [
                record['processing_time_ms'] 
                for record in integrator.execution_history
            ]
            avg_latency = sum(processing_latencies) / len(processing_latencies)
            assert avg_latency < 50.0, f"平均处理延迟 {avg_latency:.2f}ms 过高"
            
            print(f"端到端测试结果:")
            print(f"  处理的数据点: {len(sample_market_data[:20])}")
            print(f"  生成的订单: {len(results)}")
            print(f"  成功的订单: {successful_orders}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  平均处理延迟: {avg_latency:.2f}ms")
            
        finally:
            await integrator.shutdown()
    
    async def test_high_frequency_data_stream_processing(self):
        """测试高频数据流处理"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 生成高频数据流
            stream_duration = 5  # 5秒
            data_frequency = 100  # 100Hz (每10ms一个数据点)
            
            processed_count = 0
            start_time = time.time()
            
            while time.time() - start_time < stream_duration:
                # 生成实时数据点
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time() * 1000,
                    open=50000.0 + np.random.normal(0, 50),
                    high=50100.0 + np.random.normal(0, 50),
                    low=49900.0 + np.random.normal(0, 50), 
                    close=50000.0 + np.random.normal(0, 50),
                    volume=np.random.uniform(1, 100),
                    turnover=50000.0 * np.random.uniform(1, 100)
                )
                
                # 处理数据
                order_request = await integrator.process_market_data_to_order(market_data)
                processed_count += 1
                
                # 控制频率
                await asyncio.sleep(1.0 / data_frequency)
            
            actual_duration = time.time() - start_time
            actual_frequency = processed_count / actual_duration
            
            assert actual_frequency >= data_frequency * 0.8, \
                f"实际处理频率 {actual_frequency:.1f}Hz 低于目标的80%"
            
            print(f"高频数据流测试结果:")
            print(f"  目标频率: {data_frequency}Hz")
            print(f"  实际频率: {actual_frequency:.1f}Hz")
            print(f"  处理数据点: {processed_count}")
            print(f"  测试时长: {actual_duration:.2f}s")
            
        finally:
            await integrator.shutdown()
    
    async def test_system_component_coordination(self):
        """测试系统组件协调"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 测试组件间的协调工作
            coordination_tests = [
                self._test_latency_monitor_signal_processor_coordination,
                self._test_signal_processor_order_router_coordination,
                self._test_fault_manager_system_coordination
            ]
            
            coordination_results = []
            for test_func in coordination_tests:
                result = await test_func(integrator)
                coordination_results.append(result)
            
            # 验证所有协调测试都通过
            assert all(result.success for result in coordination_results), \
                "所有组件协调测试都应该通过"
            
            print("组件协调测试结果:")
            for result in coordination_results:
                print(f"  {result.test_name}: {'通过' if result.success else '失败'}")
            
        finally:
            await integrator.shutdown()
    
    async def _test_latency_monitor_signal_processor_coordination(self, integrator: HFTSystemIntegrator) -> IntegratedTestResult:
        """测试延迟监控器和信号处理器协调"""
        start_time = time.time()
        
        try:
            # 生成过期数据
            stale_data = MarketData(
                symbol="BTCUSDT",
                timestamp=time.time() * 1000 - 200,  # 200ms前
                open=50000.0, high=50100.0, low=49900.0, close=50000.0,
                volume=100.0, turnover=5000000.0
            )
            
            # 处理过期数据
            order_request = await integrator.process_market_data_to_order(stale_data)
            
            # 验证信号处理器正确响应延迟监控器的过期数据警告
            assert order_request is None, "过期数据不应该生成订单"
            
            return IntegratedTestResult(
                test_name="延迟监控器-信号处理器协调",
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["latency_monitor", "signal_processor"]
            )
            
        except Exception as e:
            return IntegratedTestResult(
                test_name="延迟监控器-信号处理器协调",
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["latency_monitor", "signal_processor"],
                errors=[str(e)]
            )
    
    async def _test_signal_processor_order_router_coordination(self, integrator: HFTSystemIntegrator) -> IntegratedTestResult:
        """测试信号处理器和订单路由器协调"""
        start_time = time.time()
        
        try:
            # 生成大订单请求
            large_order = OrderRequest(
                symbol="BTCUSDT",
                order_type=OrderType.LIMIT,
                side="buy",
                quantity=15.0,  # 大订单
                price=50000.0,
                urgency_score=0.7,
                confidence=0.8
            )
            
            # 执行订单工作流
            execution_result = await integrator.execute_order_workflow(large_order)
            
            # 验证大订单被正确拆分和路由
            assert len(execution_result.get('child_orders', [])) > 1, "大订单应该被拆分"
            
            return IntegratedTestResult(
                test_name="信号处理器-订单路由器协调", 
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["signal_processor", "order_router"]
            )
            
        except Exception as e:
            return IntegratedTestResult(
                test_name="信号处理器-订单路由器协调",
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["signal_processor", "order_router"],
                errors=[str(e)]
            )
    
    async def _test_fault_manager_system_coordination(self, integrator: HFTSystemIntegrator) -> IntegratedTestResult:
        """测试容错管理器和系统协调"""
        start_time = time.time()
        
        try:
            # 模拟系统错误
            await integrator.fault_manager.handle_error(
                "latency_monitor",
                Exception("模拟错误"),
                {"test": True}
            )
            
            # 验证系统健康状态
            system_health = await integrator.fault_manager.get_system_health()
            assert system_health is not None, "应该能获取系统健康状态"
            
            return IntegratedTestResult(
                test_name="容错管理器-系统协调",
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["fault_manager", "system"]
            )
            
        except Exception as e:
            return IntegratedTestResult(
                test_name="容错管理器-系统协调",
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                components_involved=["fault_manager", "system"],
                errors=[str(e)]
            )


@pytest.mark.asyncio
class TestRealWorldScenarios:
    """真实世界场景测试"""
    
    async def test_market_volatility_scenario(self):
        """测试市场波动场景"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 生成高波动性市场数据
            volatile_data = []
            base_price = 50000.0
            
            for i in range(100):
                # 模拟高波动 - 大幅价格变动
                price_change = np.random.normal(0, 0.02)  # 2%标准差
                new_price = base_price * (1 + price_change)
                
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time() * 1000 + i * 100,
                    open=base_price,
                    high=max(base_price, new_price) * 1.001,
                    low=min(base_price, new_price) * 0.999,
                    close=new_price,
                    volume=np.random.uniform(50, 500),  # 高交易量
                    turnover=new_price * np.random.uniform(50, 500)
                )
                
                volatile_data.append(market_data)
                base_price = new_price
            
            # 处理高波动数据
            generated_orders = 0
            successful_orders = 0
            
            for market_data in volatile_data:
                order_request = await integrator.process_market_data_to_order(market_data)
                
                if order_request:
                    generated_orders += 1
                    execution_result = await integrator.execute_order_workflow(order_request)
                    
                    if execution_result['success']:
                        successful_orders += 1
            
            # 验证系统在高波动下的表现
            order_generation_rate = generated_orders / len(volatile_data)
            success_rate = successful_orders / max(generated_orders, 1)
            
            assert order_generation_rate >= 0.3, \
                f"高波动下订单生成率 {order_generation_rate:.2%} 应该 >= 30%"
            
            assert success_rate >= 0.7, \
                f"高波动下订单成功率 {success_rate:.2%} 应该 >= 70%"
            
            print(f"市场波动场景测试结果:")
            print(f"  处理数据点: {len(volatile_data)}")
            print(f"  生成订单: {generated_orders}")
            print(f"  订单生成率: {order_generation_rate:.2%}")
            print(f"  成功订单: {successful_orders}")
            print(f"  成功率: {success_rate:.2%}")
            
        finally:
            await integrator.shutdown()
    
    async def test_network_latency_scenario(self):
        """测试网络延迟场景"""
        integrator = HFTSystemIntegrator()
        
        try:
            # 配置高延迟环境
            config = {
                'staleness_threshold_ms': 50.0,  # 更严格的延迟要求
                'data_sources': [
                    DataSourceConfig(name="high_latency_source", priority=1, max_latency_ms=100.0)
                ]
            }
            
            await integrator.initialize(config)
            
            # 生成不同延迟的数据
            latency_scenarios = [10, 30, 60, 120, 200]  # 不同延迟(ms)
            results_by_latency = {}
            
            for target_latency in latency_scenarios:
                # 生成具有特定延迟的数据
                market_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time() * 1000 - target_latency,
                    open=50000.0, high=50100.0, low=49900.0, close=50050.0,
                    volume=100.0, turnover=5005000.0
                )
                
                order_request = await integrator.process_market_data_to_order(market_data)
                results_by_latency[target_latency] = order_request is not None
            
            # 验证延迟过滤效果
            assert results_by_latency[10] == True, "低延迟数据应该被处理"
            assert results_by_latency[30] == True, "中等延迟数据应该被处理"
            assert results_by_latency[120] == False, "高延迟数据应该被拒绝"
            assert results_by_latency[200] == False, "极高延迟数据应该被拒绝"
            
            print("网络延迟场景测试结果:")
            for latency, processed in results_by_latency.items():
                print(f"  {latency}ms延迟: {'处理' if processed else '拒绝'}")
            
        finally:
            await integrator.shutdown()
    
    async def test_system_overload_scenario(self):
        """测试系统过载场景"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 生成大量并发请求
            concurrent_requests = 100
            batch_size = 10
            
            processed_batches = 0
            total_processing_time = 0
            
            for batch_start in range(0, concurrent_requests, batch_size):
                batch_end = min(batch_start + batch_size, concurrent_requests)
                batch_requests = []
                
                # 创建批次请求
                for i in range(batch_start, batch_end):
                    market_data = MarketData(
                        symbol=f"TEST{i%5}USDT",  # 5个不同的交易对
                        timestamp=time.time() * 1000,
                        open=1000.0 + i, high=1001.0 + i,
                        low=999.0 + i, close=1000.5 + i,
                        volume=100.0, turnover=100000.0
                    )
                    batch_requests.append(
                        integrator.process_market_data_to_order(market_data)
                    )
                
                # 并发处理批次
                start_time = time.time()
                batch_results = await asyncio.gather(*batch_requests, return_exceptions=True)
                processing_time = time.time() - start_time
                
                # 统计结果
                successful_in_batch = len([
                    r for r in batch_results 
                    if not isinstance(r, Exception) and r is not None
                ])
                
                if successful_in_batch > 0:
                    processed_batches += 1
                    total_processing_time += processing_time
            
            # 计算性能指标
            avg_batch_time = total_processing_time / max(processed_batches, 1)
            throughput = (processed_batches * batch_size) / total_processing_time if total_processing_time > 0 else 0
            
            assert processed_batches >= concurrent_requests // batch_size * 0.8, \
                "至少80%的批次应该成功处理"
            
            assert avg_batch_time < 1.0, \
                f"平均批次处理时间 {avg_batch_time:.3f}s 应该 < 1秒"
            
            print(f"系统过载场景测试结果:")
            print(f"  并发请求数: {concurrent_requests}")
            print(f"  批次大小: {batch_size}")
            print(f"  处理的批次: {processed_batches}")
            print(f"  平均批次处理时间: {avg_batch_time:.3f}s")
            print(f"  吞吐量: {throughput:.1f} 请求/秒")
            
        finally:
            await integrator.shutdown()


@pytest.mark.asyncio
class TestBoundaryConditions:
    """边界条件测试"""
    
    async def test_zero_latency_data(self):
        """测试零延迟数据"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 生成零延迟数据（当前时间戳）
            zero_latency_data = MarketData(
                symbol="BTCUSDT",
                timestamp=time.time() * 1000,  # 当前时间戳
                open=50000.0, high=50100.0, low=49900.0, close=50050.0,
                volume=100.0, turnover=5005000.0
            )
            
            order_request = await integrator.process_market_data_to_order(zero_latency_data)
            
            # 零延迟数据应该被正常处理
            assert order_request is not None, "零延迟数据应该被处理"
            
        finally:
            await integrator.shutdown()
    
    async def test_maximum_order_size(self):
        """测试最大订单大小"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 创建最大大小订单
            max_order = OrderRequest(
                symbol="BTCUSDT",
                order_type=OrderType.LIMIT,
                side="buy",
                quantity=1000.0,  # 最大允许数量
                price=50000.0,
                urgency_score=0.5,
                confidence=0.8
            )
            
            execution_result = await integrator.execute_order_workflow(max_order)
            
            # 最大订单应该被拆分处理
            assert len(execution_result.get('child_orders', [])) > 1, \
                "最大订单应该被拆分为多个子订单"
            
        finally:
            await integrator.shutdown()
    
    async def test_minimum_order_size(self):
        """测试最小订单大小"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 创建最小大小订单
            min_order = OrderRequest(
                symbol="BTCUSDT", 
                order_type=OrderType.LIMIT,
                side="buy",
                quantity=0.001,  # 最小允许数量
                price=50000.0,
                urgency_score=0.5,
                confidence=0.8
            )
            
            execution_result = await integrator.execute_order_workflow(min_order)
            
            # 最小订单应该被正常处理
            assert execution_result['success'] or len(execution_result['errors']) == 0, \
                "最小订单应该被正常处理"
            
        finally:
            await integrator.shutdown()
    
    async def test_extreme_market_conditions(self):
        """测试极端市场条件"""
        integrator = HFTSystemIntegrator()
        
        try:
            await integrator.initialize()
            
            # 测试极端价格变动
            extreme_scenarios = [
                {"price_change": 0.1, "description": "10%涨幅"},  # 极端上涨
                {"price_change": -0.1, "description": "10%跌幅"},  # 极端下跌
                {"price_change": 0.0, "description": "无变动"},   # 无变动
                {"price_change": 0.5, "description": "50%涨幅"},  # 异常涨幅
            ]
            
            results = {}
            
            for scenario in extreme_scenarios:
                base_price = 50000.0
                new_price = base_price * (1 + scenario["price_change"])
                
                extreme_data = MarketData(
                    symbol="BTCUSDT",
                    timestamp=time.time() * 1000,
                    open=base_price,
                    high=max(base_price, new_price),
                    low=min(base_price, new_price),
                    close=new_price,
                    volume=1000.0,
                    turnover=new_price * 1000.0
                )
                
                order_request = await integrator.process_market_data_to_order(extreme_data)
                results[scenario["description"]] = order_request is not None
            
            # 验证极端条件下的行为
            assert results["10%涨幅"] == True, "大涨应该生成买入信号"
            assert results["10%跌幅"] == True, "大跌应该生成卖出信号"
            assert results["无变动"] == False, "无变动不应该生成信号"
            # 异常涨幅可能被风控拒绝，这是正常的
            
            print("极端市场条件测试结果:")
            for condition, processed in results.items():
                print(f"  {condition}: {'生成信号' if processed else '无信号'}")
            
        finally:
            await integrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])