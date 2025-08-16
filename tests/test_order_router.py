import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.execution.router import (
    OrderRouter, RoutingStrategy, OrderPriority, RoutingConfig, RoutingContext
)
from src.execution.algorithms import AlgorithmStatus
from src.agents.execution_agent import ExecutionAgent, ExecutionMode, ExecutionReport, ExecutionResult
from src.core.models import Order, OrderSide, OrderType, MarketData, OrderBook, Position


class TestOrderRouter:
    """订单路由器测试"""
    
    @pytest.fixture
    def router(self):
        """创建订单路由器实例"""
        return OrderRouter()
    
    @pytest.fixture
    def mock_execution_agent(self):
        """创建模拟执行Agent"""
        agent = MagicMock(spec=ExecutionAgent)
        agent.execute_order = AsyncMock()
        return agent
    
    @pytest.fixture
    def sample_order(self):
        """创建示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=50000.0,
            client_order_id="test_route_001"
        )
    
    @pytest.fixture
    def market_data(self):
        """创建市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=100.0,
            bid=49999.0,
            ask=50001.0,
            bid_volume=50.0,
            ask_volume=60.0
        )
    
    @pytest.fixture
    def order_book(self):
        """创建订单簿"""
        bids = [(49999.0, 100.0), (49998.0, 200.0), (49997.0, 300.0)]
        asks = [(50001.0, 150.0), (50002.0, 250.0), (50003.0, 350.0)]
        
        return OrderBook(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            bids=bids,
            asks=asks
        )
    
    def test_router_initialization(self, router):
        """测试路由器初始化"""
        assert len(router.execution_agents) == 0
        assert len(router.algorithms) == 0
        assert len(router.routing_history) == 0
        assert router.routing_stats["total_routed"] == 0
    
    def test_register_execution_agent(self, router, mock_execution_agent):
        """测试注册执行Agent"""
        router.register_execution_agent(ExecutionMode.PAPER, mock_execution_agent)
        
        assert ExecutionMode.PAPER in router.execution_agents
        assert router.execution_agents[ExecutionMode.PAPER] == mock_execution_agent
    
    @pytest.mark.asyncio
    async def test_analyze_routing_context(self, router, sample_order, market_data, order_book):
        """测试路由上下文分析"""
        config = RoutingConfig(priority=OrderPriority.HIGH)
        context = RoutingContext(
            order=sample_order,
            config=config,
            market_data=market_data,
            order_book=order_book
        )
        
        await router._analyze_routing_context(context)
        
        # 验证紧急程度评分
        assert context.urgency_score == 0.8  # HIGH priority
        
        # 验证市场冲击估算
        assert context.market_impact_estimate > 0
        
        # 验证流动性评分
        assert 0 <= context.liquidity_score <= 1
    
    def test_estimate_market_impact(self, router, sample_order, order_book):
        """测试市场冲击估算"""
        context = RoutingContext(
            order=sample_order,
            config=RoutingConfig(),
            order_book=order_book
        )
        
        impact = router._estimate_market_impact(context)
        
        # 买单应该消耗ask侧流动性
        assert impact > 0
        
        # 测试卖单
        sell_order = sample_order.copy()
        sell_order.side = OrderSide.SELL
        context.order = sell_order
        
        sell_impact = router._estimate_market_impact(context)
        assert sell_impact > 0
    
    def test_evaluate_liquidity(self, router, sample_order, order_book):
        """测试流动性评估"""
        context = RoutingContext(
            order=sample_order,
            config=RoutingConfig(),
            order_book=order_book
        )
        
        liquidity_score = router._evaluate_liquidity(context)
        
        assert 0 <= liquidity_score <= 1
        
        # 测试空订单簿
        empty_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            bids=[],
            asks=[]
        )
        context.order_book = empty_book
        
        empty_liquidity = router._evaluate_liquidity(context)
        assert empty_liquidity < liquidity_score
    
    @pytest.mark.asyncio
    async def test_smart_strategy_selection(self, router, market_data, order_book):
        """测试智能策略选择"""
        # 小订单 -> DIRECT
        small_order = Order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=0.01, price=50000.0, client_order_id="small_001"
        )
        
        context = RoutingContext(
            order=small_order,
            config=RoutingConfig(strategy=RoutingStrategy.SMART),
            market_data=market_data,
            order_book=order_book
        )
        await router._analyze_routing_context(context)
        
        strategy = await router._select_execution_strategy(context)
        assert strategy == RoutingStrategy.DIRECT
        
        # 紧急订单 -> DIRECT
        urgent_order = Order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=5.0, price=50000.0, client_order_id="urgent_001"
        )
        
        context = RoutingContext(
            order=urgent_order,
            config=RoutingConfig(strategy=RoutingStrategy.SMART, priority=OrderPriority.URGENT),
            market_data=market_data,
            order_book=order_book
        )
        await router._analyze_routing_context(context)
        
        strategy = await router._select_execution_strategy(context)
        assert strategy == RoutingStrategy.DIRECT
        
        # 大订单，高流动性 -> VWAP
        large_order = Order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=100.0, price=50000.0, client_order_id="large_001"
        )
        
        context = RoutingContext(
            order=large_order,
            config=RoutingConfig(strategy=RoutingStrategy.SMART),
            market_data=market_data,
            order_book=order_book
        )
        await router._analyze_routing_context(context)
        
        # 手动设置高流动性评分
        context.liquidity_score = 0.8
        context.market_impact_estimate = 0.005
        
        strategy = await router._select_execution_strategy(context)
        assert strategy == RoutingStrategy.VWAP
    
    @pytest.mark.asyncio
    async def test_direct_execution(self, router, mock_execution_agent, sample_order):
        """测试直接执行"""
        # 注册执行Agent
        router.register_execution_agent(ExecutionMode.LIVE, mock_execution_agent)
        
        # 模拟执行报告
        mock_report = ExecutionReport(
            order_id="12345",
            client_order_id=sample_order.client_order_id,
            result=ExecutionResult.SUCCESS,
            executed_qty=sample_order.quantity,
            avg_price=sample_order.price,
            commission=10.0,
            slippage=0.001,
            execution_time=1.5
        )
        mock_execution_agent.execute_order.return_value = mock_report
        
        # 创建路由上下文
        context = RoutingContext(
            order=sample_order,
            config=RoutingConfig()
        )
        
        # 执行直接路由
        report = await router._execute_direct(context)
        
        # 验证结果
        assert report == mock_report
        mock_execution_agent.execute_order.assert_called_once()
        
        # 验证滑点保护
        executed_order = mock_execution_agent.execute_order.call_args[0][0]
        if sample_order.side == OrderSide.BUY:
            assert executed_order.price > sample_order.price  # 买单价格被适当提高
        
    @pytest.mark.asyncio 
    async def test_algorithm_execution(self, router, sample_order):
        """测试算法执行"""
        # 创建路由上下文
        context = RoutingContext(
            order=sample_order,
            config=RoutingConfig(),
            urgency_score=0.3,
            liquidity_score=0.6,
            market_impact_estimate=0.003
        )
        
        # 测试TWAP算法创建
        algorithm = router._create_algorithm(RoutingStrategy.TWAP, context)
        assert algorithm is not None
        assert "twap" in algorithm.algorithm_id.lower()
        
        # 测试VWAP算法创建
        algorithm = router._create_algorithm(RoutingStrategy.VWAP, context)
        assert algorithm is not None
        assert "vwap" in algorithm.algorithm_id.lower()
        
        # 测试POV算法创建
        algorithm = router._create_algorithm(RoutingStrategy.POV, context)
        assert algorithm is not None
        assert "pov" in algorithm.algorithm_id.lower()
        
        # 测试IS算法创建
        algorithm = router._create_algorithm(RoutingStrategy.IS, context)
        assert algorithm is not None
        assert "is" in algorithm.algorithm_id.lower()
    
    @pytest.mark.asyncio
    async def test_full_routing_flow(self, router, mock_execution_agent, sample_order, market_data, order_book):
        """测试完整的路由流程"""
        # 注册执行Agent
        router.register_execution_agent(ExecutionMode.LIVE, mock_execution_agent)
        
        # 模拟执行报告
        mock_report = ExecutionReport(
            order_id="route_test_001",
            client_order_id=sample_order.client_order_id,
            result=ExecutionResult.SUCCESS,
            executed_qty=sample_order.quantity,
            avg_price=sample_order.price,
            commission=10.0,
            slippage=0.002,
            execution_time=2.0
        )
        mock_execution_agent.execute_order.return_value = mock_report
        
        # 配置直接执行策略
        config = RoutingConfig(strategy=RoutingStrategy.DIRECT)
        
        # 执行路由
        report = await router.route_order(sample_order, config)
        
        # 验证结果
        assert report == mock_report
        
        # 验证统计更新
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 1
        assert stats["successful_routes"] == 1
        assert stats["strategy_usage"][RoutingStrategy.DIRECT.value] == 1
        
        # 验证历史记录
        history = router.get_routing_history()
        assert len(history) == 1
        assert history[0]["symbol"] == sample_order.symbol
        assert history[0]["strategy"] == RoutingStrategy.DIRECT.value
    
    @pytest.mark.asyncio
    async def test_routing_failure_handling(self, router, sample_order):
        """测试路由失败处理"""
        # 没有注册执行Agent的情况
        config = RoutingConfig(strategy=RoutingStrategy.DIRECT)
        
        with pytest.raises(ValueError, match="No execution agent available"):
            await router.route_order(sample_order, config)
        
        # 验证失败统计
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 1
        assert stats["failed_routes"] == 1
    
    def test_routing_config_creation(self):
        """测试路由配置创建"""
        config = RoutingConfig(
            strategy=RoutingStrategy.VWAP,
            priority=OrderPriority.HIGH,
            max_slice_size=500.0,
            slippage_tolerance=0.01,
            timeout_seconds=600,
            enable_dark_pool=True
        )
        
        assert config.strategy == RoutingStrategy.VWAP
        assert config.priority == OrderPriority.HIGH
        assert config.max_slice_size == 500.0
        assert config.slippage_tolerance == 0.01
        assert config.timeout_seconds == 600
        assert config.enable_dark_pool is True
    
    def test_routing_context_creation(self, sample_order):
        """测试路由上下文创建"""
        config = RoutingConfig()
        context = RoutingContext(
            order=sample_order,
            config=config,
            urgency_score=0.7,
            market_impact_estimate=0.005,
            liquidity_score=0.6
        )
        
        assert context.order == sample_order
        assert context.config == config
        assert context.urgency_score == 0.7
        assert context.market_impact_estimate == 0.005
        assert context.liquidity_score == 0.6
    
    @pytest.mark.asyncio
    async def test_algorithm_lifecycle_management(self, router):
        """测试算法生命周期管理"""
        # 创建模拟算法
        mock_algorithm = MagicMock()
        mock_algorithm.stop = AsyncMock()
        mock_algorithm.pause = AsyncMock()
        mock_algorithm.resume = AsyncMock()
        
        algorithm_id = "test_algo_001"
        router.algorithms[algorithm_id] = mock_algorithm
        
        # 测试取消算法
        success = await router.cancel_algorithm(algorithm_id)
        assert success is True
        mock_algorithm.stop.assert_called_once()
        assert algorithm_id not in router.algorithms
        
        # 测试取消不存在的算法
        success = await router.cancel_algorithm("nonexistent")
        assert success is False
        
        # 重新添加算法测试暂停/恢复
        router.algorithms[algorithm_id] = mock_algorithm
        
        success = await router.pause_algorithm(algorithm_id)
        assert success is True
        mock_algorithm.pause.assert_called_once()
        
        success = await router.resume_algorithm(algorithm_id)
        assert success is True
        mock_algorithm.resume.assert_called_once()
    
    def test_routing_statistics(self, router):
        """测试路由统计"""
        # 初始统计
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 0
        assert stats["success_rate"] == 0
        
        # 模拟一些路由操作
        from src.execution.router import RoutingContext
        
        context = RoutingContext(
            order=Order(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, 
                       quantity=1.0, price=50000.0),
            config=RoutingConfig()
        )
        
        # 成功路由
        report = ExecutionReport(
            order_id="stat_test_001", client_order_id="test_001",
            result=ExecutionResult.SUCCESS, executed_qty=1.0, avg_price=50000.0,
            commission=5.0, slippage=0.001, execution_time=1.0
        )
        
        router._update_routing_stats(context, report, True)
        
        # 失败路由
        router._update_routing_stats(context, None, False)
        
        # 检查统计
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 2
        assert stats["successful_routes"] == 1
        assert stats["failed_routes"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["avg_execution_time"] == 1.0
        assert stats["avg_slippage"] == 0.001
    
    def test_routing_history_management(self, router, sample_order):
        """测试路由历史管理"""
        from src.execution.router import RoutingContext
        
        # 添加多条历史记录
        for i in range(5):
            context = RoutingContext(
                order=sample_order.copy(),
                config=RoutingConfig(strategy=RoutingStrategy.DIRECT),
                urgency_score=0.5,
                market_impact_estimate=0.001,
                liquidity_score=0.7
            )
            
            report = ExecutionReport(
                order_id=f"hist_test_{i:03d}",
                client_order_id=f"test_{i:03d}",
                result=ExecutionResult.SUCCESS,
                executed_qty=1.0,
                avg_price=50000.0,
                commission=5.0,
                slippage=0.001,
                execution_time=1.0
            )
            
            router._record_routing_history(context, RoutingStrategy.DIRECT, report)
        
        # 检查历史记录
        history = router.get_routing_history()
        assert len(history) == 5
        
        # 检查限制功能
        limited_history = router.get_routing_history(limit=3)
        assert len(limited_history) == 3
        
        # 验证历史记录内容
        for entry in history:
            assert "timestamp" in entry
            assert "symbol" in entry
            assert "strategy" in entry
            assert "execution_result" in entry
            assert entry["symbol"] == sample_order.symbol
            assert entry["strategy"] == RoutingStrategy.DIRECT.value
    
    def test_strategy_enum_completeness(self):
        """测试策略枚举完整性"""
        strategies = list(RoutingStrategy)
        
        # 确保包含所有预期策略
        expected_strategies = [
            RoutingStrategy.DIRECT,
            RoutingStrategy.SMART,
            RoutingStrategy.TWAP,
            RoutingStrategy.VWAP,
            RoutingStrategy.POV,
            RoutingStrategy.IS
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies
    
    def test_priority_enum_ordering(self):
        """测试优先级枚举顺序"""
        assert OrderPriority.LOW.value < OrderPriority.NORMAL.value
        assert OrderPriority.NORMAL.value < OrderPriority.HIGH.value
        assert OrderPriority.HIGH.value < OrderPriority.URGENT.value