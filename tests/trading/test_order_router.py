import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading.order_router import (
    TradingModeRouter, TradingMode, RouteDecision, RoutingContext, 
    RoutingResult, EnvironmentConfirmation
)
from src.trading.paper_trading_engine import PaperTradingEngine
from src.exchanges.binance_quant_order_manager import BinanceQuantOrderManager
from src.exchanges.binance import BinanceFuturesClient
from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData


class TestTradingModeRouter:
    """交易模式路由器测试"""

    @pytest.fixture
    async def mock_live_client(self):
        """模拟实盘客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        client.get_exchange_info = AsyncMock(return_value={"symbols": []})
        return client

    @pytest.fixture
    async def mock_paper_client(self):
        """模拟模拟盘客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        client.get_exchange_info = AsyncMock(return_value={"symbols": []})
        return client

    @pytest.fixture
    async def mock_paper_engine(self):
        """模拟虚拟盘引擎"""
        engine = MagicMock(spec=PaperTradingEngine)
        engine.execute_order = AsyncMock(return_value={
            "orderId": "paper_12345",
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "executedQty": "0.001",
            "avgPrice": "50000.0"
        })
        engine.cancel_order = AsyncMock(return_value={
            "status": "CANCELED",
            "clientOrderId": "test_order"
        })
        engine.get_order_status = AsyncMock(return_value={
            "status": "FILLED",
            "executedQty": "0.001"
        })
        return engine

    @pytest.fixture
    async def router(self, mock_live_client, mock_paper_client):
        """路由器实例"""
        router = TradingModeRouter(
            live_client=mock_live_client,
            paper_client=mock_paper_client
        )
        await router.initialize()
        return router

    @pytest.fixture
    def sample_order(self):
        """示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            client_order_id="test_order_001"
        )

    @pytest.fixture
    def sample_market_data(self):
        """示例市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow()
        )

    async def test_initialization(self, mock_live_client, mock_paper_client):
        """测试初始化"""
        router = TradingModeRouter(
            live_client=mock_live_client,
            paper_client=mock_paper_client
        )

        # 初始化前管理器应该为空
        assert router.live_order_manager is None
        assert router.paper_order_manager is None

        await router.initialize()

        # 初始化后应该创建管理器
        assert router.live_order_manager is not None
        assert router.paper_order_manager is not None
        assert isinstance(router.live_order_manager, BinanceQuantOrderManager)
        assert isinstance(router.paper_order_manager, BinanceQuantOrderManager)

    async def test_initialization_without_clients(self):
        """测试没有客户端的初始化"""
        router = TradingModeRouter()
        await router.initialize()

        assert router.live_order_manager is None
        assert router.paper_order_manager is None

    async def test_route_to_paper_trading(self, router, sample_order, sample_market_data):
        """测试路由到模拟盘交易"""
        # 模拟paper manager的place_order方法
        router.paper_order_manager.place_order = AsyncMock(return_value={
            "orderId": "paper_123",
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "clientOrderId": "test_order_001"
        })

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER,
            market_data=sample_market_data,
            account_balance=10000.0,
            risk_level="LOW"
        )

        assert isinstance(result, RoutingResult)
        assert result.decision == RouteDecision.PAPER_TRADING
        assert result.target_engine == "paper"
        assert result.execution_result is not None
        assert result.error_message is None
        assert result.routing_time > 0

        # 验证统计更新
        assert router.routing_stats["total_routes"] == 1
        assert router.routing_stats["paper_routes"] == 1
        assert router.routing_stats["live_routes"] == 0

    async def test_route_to_live_trading(self, router, sample_order, sample_market_data):
        """测试路由到实盘交易"""
        # 模拟live manager的place_order方法
        router.live_order_manager.place_order = AsyncMock(return_value={
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "test_order_001"
        })

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.LIVE,
            market_data=sample_market_data,
            account_balance=10000.0,
            risk_level="LOW"
        )

        assert result.decision == RouteDecision.LIVE_TRADING
        assert result.target_engine == "live"
        assert result.execution_result is not None
        assert result.error_message is None

        # 验证统计更新
        assert router.routing_stats["live_routes"] == 1

    async def test_route_with_paper_engine(self, router, sample_order, mock_paper_engine):
        """测试使用虚拟盘引擎路由"""
        router.set_paper_trading_engine(mock_paper_engine)

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER,
            risk_level="LOW"
        )

        assert result.decision == RouteDecision.PAPER_TRADING
        assert result.target_engine == "paper_engine"
        
        # 验证引擎被调用
        mock_paper_engine.execute_order.assert_called_once_with(sample_order)

    async def test_pre_route_checks_invalid_symbol(self, router):
        """测试预路由检查：无效交易对"""
        invalid_order = Order(
            symbol="",  # 空交易对
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )

        result = await router.route_order(
            order=invalid_order,
            trading_mode=TradingMode.PAPER
        )

        assert result.decision == RouteDecision.REJECTED
        assert "Missing order symbol" in result.error_message

    async def test_pre_route_checks_invalid_quantity(self, router, sample_order):
        """测试预路由检查：无效数量"""
        sample_order.quantity = 0  # 无效数量

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER
        )

        assert result.decision == RouteDecision.REJECTED
        assert "Invalid order quantity" in result.error_message

    async def test_pre_route_checks_insufficient_balance(self, router, sample_order):
        """测试预路由检查：余额不足"""
        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.LIVE,
            account_balance=100.0  # 余额不足
        )

        assert result.decision == RouteDecision.REJECTED
        assert "Insufficient account balance" in result.error_message

    async def test_pre_route_checks_manager_not_available(self, sample_order):
        """测试预路由检查：管理器不可用"""
        router = TradingModeRouter()  # 没有客户端
        await router.initialize()

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.LIVE
        )

        assert result.decision == RouteDecision.REJECTED
        assert "Live trading manager not available" in result.error_message

    async def test_environment_confirmation_required(self, router, sample_market_data):
        """测试环境确认需求"""
        # 大额订单需要确认
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,  # 市价单需要确认
            quantity=1.0,
            price=50000.0
        )

        # 模拟确认失败
        with patch.object(router.env_confirmation, 'confirm_environment', 
                         return_value=False) as mock_confirm:
            result = await router.route_order(
                order=large_order,
                trading_mode=TradingMode.LIVE,
                market_data=sample_market_data,
                account_balance=100000.0,
                risk_level="HIGH"
            )

            assert result.decision == RouteDecision.REJECTED
            assert result.confirmation_required is True
            assert "Environment confirmation failed" in result.error_message
            mock_confirm.assert_called_once()

    async def test_force_confirmation(self, router, sample_order, sample_market_data):
        """测试强制确认"""
        with patch.object(router.env_confirmation, 'confirm_environment', 
                         return_value=True) as mock_confirm:
            result = await router.route_order(
                order=sample_order,
                trading_mode=TradingMode.LIVE,
                market_data=sample_market_data,
                account_balance=10000.0,
                force_confirmation=True
            )

            mock_confirm.assert_called_once()
            # 如果确认成功，应该继续执行
            assert result.decision != RouteDecision.REJECTED or "confirmation" not in (result.error_message or "")

    async def test_routing_error_handling(self, router, sample_order):
        """测试路由错误处理"""
        # 模拟执行错误
        router.paper_order_manager.place_order = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER
        )

        assert result.decision == RouteDecision.REJECTED
        assert result.target_engine == "paper"
        assert "API Error" in result.error_message

    async def test_cancel_order_paper(self, router, mock_paper_engine):
        """测试取消模拟盘订单"""
        router.set_paper_trading_engine(mock_paper_engine)

        result = await router.cancel_order(
            symbol="BTCUSDT",
            trading_mode=TradingMode.PAPER,
            client_order_id="test_order"
        )

        assert result["status"] == "CANCELED"
        mock_paper_engine.cancel_order.assert_called_once_with("test_order")

    async def test_cancel_order_live(self, router):
        """测试取消实盘订单"""
        router.live_order_manager.cancel_order = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "orderId": 123456,
            "status": "CANCELED"
        })

        result = await router.cancel_order(
            symbol="BTCUSDT",
            trading_mode=TradingMode.LIVE,
            order_id=123456
        )

        assert result["status"] == "CANCELED"
        router.live_order_manager.cancel_order.assert_called_once()

    async def test_cancel_order_fallback(self, router):
        """测试取消订单回退机制"""
        # 没有paper engine，使用manager
        router.paper_order_manager.cancel_order = AsyncMock(return_value={
            "status": "CANCELED",
            "clientOrderId": "test"
        })

        result = await router.cancel_order(
            symbol="BTCUSDT",
            trading_mode=TradingMode.PAPER,
            client_order_id="test"
        )

        assert result["status"] == "CANCELED"

    async def test_cancel_order_simple_simulation(self, router):
        """测试简单模拟取消"""
        # 清空managers来测试简单模拟
        router.paper_order_manager = None
        router.paper_trading_engine = None

        result = await router.cancel_order(
            symbol="BTCUSDT",
            trading_mode=TradingMode.PAPER,
            client_order_id="test"
        )

        assert result["status"] == "CANCELED"
        assert result["clientOrderId"] == "test"

    async def test_get_order_status_paper(self, router, mock_paper_engine):
        """测试查询模拟盘订单状态"""
        router.set_paper_trading_engine(mock_paper_engine)

        result = await router.get_order_status(
            symbol="BTCUSDT",
            trading_mode=TradingMode.PAPER,
            client_order_id="test_order"
        )

        assert result["status"] == "FILLED"
        mock_paper_engine.get_order_status.assert_called_once_with("test_order")

    async def test_get_order_status_live(self, router):
        """测试查询实盘订单状态"""
        router.live_order_manager.get_order_status = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "orderId": 123456,
            "status": "FILLED",
            "executedQty": "0.001"
        })

        result = await router.get_order_status(
            symbol="BTCUSDT",
            trading_mode=TradingMode.LIVE,
            order_id=123456
        )

        assert result["status"] == "FILLED"
        router.live_order_manager.get_order_status.assert_called_once()

    async def test_routing_statistics(self, router, sample_order):
        """测试路由统计"""
        # 执行几次路由来生成统计数据
        router.paper_order_manager.place_order = AsyncMock(return_value={
            "orderId": "paper_123", "status": "FILLED"
        })
        
        # 模拟盘路由
        await router.route_order(sample_order, TradingMode.PAPER)
        await router.route_order(sample_order, TradingMode.PAPER)
        
        # 拒绝的路由
        invalid_order = Order(symbol="", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0)
        await router.route_order(invalid_order, TradingMode.PAPER)

        stats = router.get_routing_statistics()

        assert stats["total_routes"] == 3
        assert stats["paper_routes"] == 2
        assert stats["rejected_routes"] == 1
        assert stats["live_routes"] == 0
        assert stats["paper_route_ratio"] == 2/3
        assert stats["rejection_rate"] == 1/3
        assert "avg_routing_time" in stats

    async def test_routing_history(self, router, sample_order):
        """测试路由历史"""
        router.paper_order_manager.place_order = AsyncMock(return_value={
            "orderId": "paper_123", "status": "FILLED"
        })

        await router.route_order(sample_order, TradingMode.PAPER, risk_level="LOW")

        history = router.get_routing_history(limit=10)

        assert len(history) == 1
        assert history[0]["symbol"] == "BTCUSDT"
        assert history[0]["trading_mode"] == "paper"
        assert history[0]["decision"] == "paper_trading"
        assert history[0]["risk_level"] == "LOW"
        assert "timestamp" in history[0]
        assert "routing_time" in history[0]

    async def test_routing_history_limit(self, router, sample_order):
        """测试路由历史限制"""
        router.paper_order_manager.place_order = AsyncMock(return_value={
            "orderId": "paper_123", "status": "FILLED"
        })

        # 生成多条历史记录
        for i in range(5):
            sample_order.client_order_id = f"test_{i}"
            await router.route_order(sample_order, TradingMode.PAPER)

        # 测试限制
        history = router.get_routing_history(limit=3)
        assert len(history) == 3

        # 测试无限制
        history = router.get_routing_history(limit=0)
        assert len(history) == 5

    async def test_auto_confirm_whitelist(self, router, sample_order):
        """测试自动确认白名单"""
        router.add_auto_confirm_order("BTCUSDT", TradingMode.LIVE)

        # 白名单中的订单应该自动确认
        with patch.object(router.env_confirmation, 'requires_confirmation', 
                         return_value=True):
            confirmed = await router.env_confirmation.confirm_environment(
                RoutingContext(order=sample_order, trading_mode=TradingMode.LIVE)
            )
            assert confirmed is True

    async def test_cleanup(self, router):
        """测试清理"""
        # 添加一些历史数据
        router.routing_history.append({"test": "data"})

        # 模拟管理器清理
        router.live_order_manager.cleanup = AsyncMock()
        router.paper_order_manager.cleanup = AsyncMock()

        await router.cleanup()

        # 验证清理
        assert len(router.routing_history) == 0
        router.live_order_manager.cleanup.assert_called_once()
        router.paper_order_manager.cleanup.assert_called_once()

    async def test_simple_paper_fallback(self, router, sample_order):
        """测试简单模拟盘回退"""
        # 移除所有管理器和引擎
        router.paper_order_manager = None
        router.paper_trading_engine = None

        result = await router.route_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER
        )

        assert result.decision == RouteDecision.PAPER_TRADING
        assert result.target_engine == "simple_paper"
        assert result.execution_result is not None
        assert "orderId" in result.execution_result
        assert result.execution_result["status"] == "FILLED"


class TestEnvironmentConfirmation:
    """环境确认机制测试"""

    @pytest.fixture
    def env_confirmation(self):
        """环境确认实例"""
        return EnvironmentConfirmation()

    @pytest.fixture
    def sample_context(self):
        """示例上下文"""
        return RoutingContext(
            order=Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.0
            ),
            trading_mode=TradingMode.LIVE,
            risk_level="LOW"
        )

    def test_requires_confirmation_large_order(self, env_confirmation):
        """测试大额订单需要确认"""
        large_order_context = RoutingContext(
            order=Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,  # 50k USDT > 10k limit
                price=50000.0
            ),
            trading_mode=TradingMode.LIVE
        )

        assert env_confirmation.requires_confirmation(large_order_context) is True

    def test_requires_confirmation_high_risk_symbol(self, env_confirmation):
        """测试高风险交易对需要确认"""
        high_risk_context = RoutingContext(
            order=Order(
                symbol="BTCUSDT",  # 在高风险列表中
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=50000.0
            ),
            trading_mode=TradingMode.LIVE
        )

        assert env_confirmation.requires_confirmation(high_risk_context) is True

    def test_requires_confirmation_market_order(self, env_confirmation):
        """测试市价单需要确认"""
        market_order_context = RoutingContext(
            order=Order(
                symbol="ADAUSDT",  # 非高风险交易对
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,  # 市价单
                quantity=100.0,
                price=1.0  # 小额订单
            ),
            trading_mode=TradingMode.LIVE
        )

        assert env_confirmation.requires_confirmation(market_order_context) is True

    def test_no_confirmation_needed_paper(self, env_confirmation, sample_context):
        """测试模拟盘不需要确认"""
        sample_context.trading_mode = TradingMode.PAPER
        assert env_confirmation.requires_confirmation(sample_context) is False

    def test_no_confirmation_needed_small_order(self, env_confirmation):
        """测试小额订单不需要确认"""
        small_order_context = RoutingContext(
            order=Order(
                symbol="ADAUSDT",  # 非高风险交易对
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,  # 限价单
                quantity=100.0,
                price=1.0  # 100 USDT < 10k limit
            ),
            trading_mode=TradingMode.LIVE
        )

        assert env_confirmation.requires_confirmation(small_order_context) is False

    async def test_confirm_environment_low_risk(self, env_confirmation, sample_context):
        """测试低风险环境确认"""
        sample_context.risk_level = "LOW"
        confirmed = await env_confirmation.confirm_environment(sample_context)
        assert confirmed is True

    async def test_confirm_environment_medium_risk(self, env_confirmation):
        """测试中风险环境确认"""
        medium_risk_context = RoutingContext(
            order=Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,  # 中等金额
                price=50000.0
            ),
            trading_mode=TradingMode.LIVE,
            risk_level="MEDIUM"
        )

        confirmed = await env_confirmation.confirm_environment(medium_risk_context)
        assert confirmed is True  # 5k < 5k limit

        # 超过限额的中风险订单
        medium_risk_context.order.quantity = 0.2  # 10k > 5k limit
        confirmed = await env_confirmation.confirm_environment(medium_risk_context)
        assert confirmed is False

    async def test_confirm_environment_high_risk(self, env_confirmation, sample_context):
        """测试高风险环境确认"""
        sample_context.risk_level = "HIGH"
        confirmed = await env_confirmation.confirm_environment(sample_context)
        assert confirmed is False  # 高风险需要人工确认

    async def test_whitelist_auto_confirm(self, env_confirmation, sample_context):
        """测试白名单自动确认"""
        # 添加到白名单
        order_signature = f"{sample_context.order.symbol}_{sample_context.trading_mode.value}"
        env_confirmation.auto_confirm_whitelist.add(order_signature)

        confirmed = await env_confirmation.confirm_environment(sample_context)
        assert confirmed is True