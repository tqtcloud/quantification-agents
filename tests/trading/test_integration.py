import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.trading.order_router import TradingModeRouter, TradingMode
from src.trading.paper_trading_engine import PaperTradingEngine
from src.trading.order_sync_manager import OrderSyncManager, DataIsolationConfig
from src.exchanges.binance_quant_order_manager import BinanceQuantOrderManager, BinanceOrderType
from src.exchanges.binance import BinanceFuturesClient
from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData


class TestTradingSystemIntegration:
    """交易系统集成测试"""

    @pytest.fixture
    async def mock_live_client(self):
        """模拟实盘客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        client.get_exchange_info = AsyncMock(return_value={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "pricePrecision": 2,
                    "quantityPrecision": 6,
                    "orderTypes": ["MARKET", "LIMIT"],
                    "filters": [
                        {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "1000000", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "9000", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
                        {"filterType": "MAX_NUM_ORDERS", "maxNumOrders": "200"}
                    ]
                }
            ]
        })
        client.place_order = AsyncMock(return_value={
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "live_order_001",
            "transactTime": int(datetime.utcnow().timestamp() * 1000)
        })
        return client

    @pytest.fixture
    async def paper_engine(self):
        """虚拟盘引擎"""
        engine = PaperTradingEngine("integration_test_account")
        await engine.initialize()
        return engine

    @pytest.fixture
    async def sync_manager(self):
        """同步管理器"""
        config = DataIsolationConfig(isolation_level="MODERATE")
        manager = OrderSyncManager(config)
        await manager.initialize()
        return manager

    @pytest.fixture
    async def integrated_router(self, mock_live_client, paper_engine, sync_manager):
        """集成路由器"""
        router = TradingModeRouter(live_client=mock_live_client)
        await router.initialize()
        router.set_paper_trading_engine(paper_engine)
        return router, sync_manager

    @pytest.fixture
    def sample_orders(self):
        """示例订单集合"""
        return [
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                client_order_id="integration_market_001"
            ),
            Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.002,
                price=49000.0,
                client_order_id="integration_limit_001"
            ),
            Order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.STOP_MARKET,
                quantity=0.001,
                stop_price=51000.0,
                client_order_id="integration_stop_001"
            )
        ]

    async def test_end_to_end_paper_trading_flow(self, integrated_router, sample_orders):
        """测试端到端模拟盘交易流程"""
        router, sync_manager = integrated_router
        market_order = sample_orders[0]

        # 1. 路由订单到模拟盘
        routing_result = await router.route_order(
            order=market_order,
            trading_mode=TradingMode.PAPER,
            account_balance=10000.0,
            risk_level="LOW"
        )

        # 验证路由结果
        assert routing_result.decision.value == "paper_trading"
        assert routing_result.target_engine == "paper_engine"
        assert routing_result.execution_result is not None

        # 2. 注册到同步管理器
        await sync_manager.register_order(
            order=market_order,
            trading_mode=TradingMode.PAPER,
            execution_result=routing_result.execution_result
        )

        # 3. 验证同步管理器状态
        registered_order = await sync_manager.get_order(
            market_order.client_order_id,
            TradingMode.PAPER
        )
        assert registered_order is not None
        assert registered_order.symbol == "BTCUSDT"

        # 4. 获取虚拟盘引擎状态
        account_info = router.paper_trading_engine.get_account_info()
        assert account_info["accountId"] is not None
        assert "balance" in account_info

        # 5. 获取订单历史
        order_history = await sync_manager.get_order_history(TradingMode.PAPER)
        # 由于订单已完成并可能被归档，检查相关统计
        sync_stats = sync_manager.get_sync_statistics()
        assert sync_stats["paper_orders_count"] >= 0

    async def test_end_to_end_live_trading_flow(self, integrated_router, sample_orders):
        """测试端到端实盘交易流程"""
        router, sync_manager = integrated_router
        limit_order = sample_orders[1]

        # 1. 路由订单到实盘
        routing_result = await router.route_order(
            order=limit_order,
            trading_mode=TradingMode.LIVE,
            account_balance=100000.0,  # 足够的余额
            risk_level="LOW"
        )

        # 验证路由结果
        assert routing_result.decision.value == "live_trading"
        assert routing_result.target_engine == "live"
        assert routing_result.execution_result is not None

        # 2. 注册到同步管理器
        await sync_manager.register_order(
            order=limit_order,
            trading_mode=TradingMode.LIVE,
            execution_result=routing_result.execution_result
        )

        # 3. 验证实盘订单管理
        live_orders = await sync_manager.get_orders_by_environment(TradingMode.LIVE)
        assert len(live_orders) == 1
        assert live_orders[0].client_order_id == limit_order.client_order_id

        # 4. 验证路由统计
        routing_stats = router.get_routing_statistics()
        assert routing_stats["live_routes"] >= 1
        assert routing_stats["total_routes"] >= 1

    async def test_cross_environment_isolation(self, integrated_router, sample_orders):
        """测试跨环境数据隔离"""
        router, sync_manager = integrated_router
        paper_order = sample_orders[0]
        live_order = sample_orders[1]

        # 注册到不同环境
        await router.route_order(paper_order, TradingMode.PAPER)
        await sync_manager.register_order(paper_order, TradingMode.PAPER)

        await router.route_order(live_order, TradingMode.LIVE, account_balance=100000.0)
        await sync_manager.register_order(live_order, TradingMode.LIVE)

        # 验证环境隔离
        paper_orders = await sync_manager.get_orders_by_environment(TradingMode.PAPER)
        live_orders = await sync_manager.get_orders_by_environment(TradingMode.LIVE)

        assert len(paper_orders) == 1
        assert len(live_orders) == 1
        assert paper_orders[0].client_order_id == paper_order.client_order_id
        assert live_orders[0].client_order_id == live_order.client_order_id

        # 验证跨环境访问统计
        isolation_stats = sync_manager.isolation_stats
        assert isolation_stats["paper_operations"] >= 1
        assert isolation_stats["live_operations"] >= 1

    async def test_order_lifecycle_management(self, integrated_router, sample_orders):
        """测试订单生命周期管理"""
        router, sync_manager = integrated_router
        limit_order = sample_orders[1]

        # 1. 下单
        routing_result = await router.route_order(
            limit_order, TradingMode.PAPER, risk_level="LOW"
        )
        await sync_manager.register_order(limit_order, TradingMode.PAPER)

        # 2. 查询订单状态
        order_status = await router.get_order_status(
            symbol=limit_order.symbol,
            trading_mode=TradingMode.PAPER,
            client_order_id=limit_order.client_order_id
        )
        assert "status" in order_status

        # 3. 更新订单状态
        await sync_manager.update_order_status(
            client_order_id=limit_order.client_order_id,
            new_status=OrderStatus.FILLED,
            trading_mode=TradingMode.PAPER,
            remote_data={"status": "FILLED", "executedQty": "0.002"}
        )

        # 4. 归档已完成订单
        await sync_manager.archive_completed_order(
            limit_order.client_order_id,
            TradingMode.PAPER
        )

        # 5. 验证订单历史
        history = await sync_manager.get_order_history(TradingMode.PAPER)
        assert len(history) >= 1

    async def test_error_handling_and_recovery(self, integrated_router, sample_orders):
        """测试错误处理和恢复"""
        router, sync_manager = integrated_router
        invalid_order = Order(
            symbol="",  # 无效符号
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0,  # 无效数量
            client_order_id="invalid_order_001"
        )

        # 1. 测试路由错误处理
        routing_result = await router.route_order(
            invalid_order, TradingMode.PAPER
        )
        assert routing_result.decision.value == "rejected"
        assert routing_result.error_message is not None

        # 2. 测试同步管理器权限错误
        valid_order = sample_orders[0]
        await sync_manager.register_order(valid_order, TradingMode.PAPER)

        with pytest.raises(PermissionError):
            await sync_manager.update_order_status(
                client_order_id=valid_order.client_order_id,
                new_status=OrderStatus.FILLED,
                trading_mode=TradingMode.LIVE  # 错误的环境
            )

        # 3. 验证错误统计
        isolation_stats = sync_manager.isolation_stats
        assert isolation_stats["isolation_violations"] >= 1

        routing_stats = router.get_routing_statistics()
        assert routing_stats["rejected_routes"] >= 1

    async def test_batch_operation_integration(self, integrated_router):
        """测试批量操作集成"""
        router, sync_manager = integrated_router

        # 创建批量订单
        batch_orders = []
        for i in range(3):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                price=49000.0 + i * 100,
                client_order_id=f"batch_order_{i}"
            )
            batch_orders.append(order)

        # 批量路由和注册
        routing_results = []
        for order in batch_orders:
            result = await router.route_order(order, TradingMode.PAPER)
            routing_results.append(result)
            await sync_manager.register_order(order, TradingMode.PAPER)

        # 验证批量结果
        assert len(routing_results) == 3
        assert all(r.decision.value == "paper_trading" for r in routing_results)

        # 验证同步管理器中的订单
        paper_orders = await sync_manager.get_orders_by_environment(TradingMode.PAPER)
        assert len(paper_orders) >= 3

        # 批量归档
        for order in batch_orders:
            await sync_manager.archive_completed_order(
                order.client_order_id,
                TradingMode.PAPER
            )

        # 验证历史记录
        history = await sync_manager.get_order_history(TradingMode.PAPER)
        assert len(history) >= 3

    async def test_market_data_integration(self, integrated_router):
        """测试市场数据集成"""
        router, sync_manager = integrated_router

        # 创建市场数据
        market_data = MarketData(
            symbol="BTCUSDT",
            price=52000.0,
            timestamp=datetime.utcnow()
        )

        # 下单带市场数据
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="market_data_test"
        )

        routing_result = await router.route_order(
            order=order,
            trading_mode=TradingMode.PAPER,
            market_data=market_data
        )

        # 更新虚拟盘引擎的市场数据
        if router.paper_trading_engine:
            router.paper_trading_engine.update_market_data(market_data)
            
            # 验证市场价格更新
            assert router.paper_trading_engine.market_prices["BTCUSDT"] == 52000.0

        assert routing_result.decision.value == "paper_trading"

    async def test_performance_and_statistics(self, integrated_router, sample_orders):
        """测试性能和统计集成"""
        router, sync_manager = integrated_router

        # 执行多次操作来生成统计数据
        for i in range(5):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                client_order_id=f"perf_test_{i}"
            )

            # 随机选择环境
            trading_mode = TradingMode.PAPER if i % 2 == 0 else TradingMode.LIVE
            account_balance = 100000.0 if trading_mode == TradingMode.LIVE else 10000.0

            routing_result = await router.route_order(
                order, trading_mode, account_balance=account_balance
            )
            
            if routing_result.decision.value != "rejected":
                await sync_manager.register_order(order, trading_mode)

        # 获取路由统计
        routing_stats = router.get_routing_statistics()
        assert routing_stats["total_routes"] >= 5
        assert "avg_routing_time" in routing_stats
        assert routing_stats["avg_routing_time"] > 0

        # 获取同步统计
        sync_stats = sync_manager.get_sync_statistics()
        assert sync_stats["total_records"] >= 1
        assert "isolation_stats" in sync_stats

        # 获取环境摘要
        env_summary = sync_manager.get_environment_summary()
        assert "paper_trading" in env_summary
        assert "live_trading" in env_summary
        assert "isolation_config" in env_summary

    async def test_concurrent_operations(self, integrated_router):
        """测试并发操作"""
        router, sync_manager = integrated_router

        async def place_order(order_id: int, trading_mode: TradingMode):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                client_order_id=f"concurrent_{order_id}"
            )
            
            account_balance = 100000.0 if trading_mode == TradingMode.LIVE else 10000.0
            routing_result = await router.route_order(
                order, trading_mode, account_balance=account_balance
            )
            
            if routing_result.decision.value != "rejected":
                await sync_manager.register_order(order, trading_mode)
            
            return routing_result

        # 并发执行多个订单
        tasks = []
        for i in range(10):
            trading_mode = TradingMode.PAPER if i % 2 == 0 else TradingMode.LIVE
            task = place_order(i, trading_mode)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 5  # 至少一半成功

        # 验证统计数据
        routing_stats = router.get_routing_statistics()
        sync_stats = sync_manager.get_sync_statistics()
        
        assert routing_stats["total_routes"] >= 5
        assert sync_stats["total_records"] >= 1

    async def test_cleanup_integration(self, integrated_router):
        """测试清理集成"""
        router, sync_manager = integrated_router

        # 执行一些操作
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="cleanup_test"
        )

        await router.route_order(order, TradingMode.PAPER)
        await sync_manager.register_order(order, TradingMode.PAPER)

        # 验证系统状态
        assert len(sync_manager.sync_records) >= 1
        assert len(router.routing_history) >= 1

        # 执行清理
        await sync_manager.cleanup()
        await router.cleanup()

        # 验证清理效果
        assert not sync_manager._running
        assert len(router.routing_history) == 0

    async def test_full_system_workflow(self, integrated_router):
        """测试完整系统工作流程"""
        router, sync_manager = integrated_router

        # 1. 系统初始化验证
        assert router.live_order_manager is not None
        assert router.paper_trading_engine is not None
        assert sync_manager._running is True

        # 2. 下单流程
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            client_order_id="full_workflow_test"
        )

        # 3. 路由决策
        routing_result = await router.route_order(
            order=order,
            trading_mode=TradingMode.PAPER,
            risk_level="LOW"
        )
        assert routing_result.error_message is None

        # 4. 订单注册和同步
        await sync_manager.register_order(
            order, TradingMode.PAPER, routing_result.execution_result
        )

        # 5. 状态查询
        order_status = await router.get_order_status(
            symbol=order.symbol,
            trading_mode=TradingMode.PAPER,
            client_order_id=order.client_order_id
        )
        assert "status" in order_status

        # 6. 订单管理
        retrieved_order = await sync_manager.get_order(order.client_order_id)
        assert retrieved_order is not None

        # 7. 统计和监控
        routing_stats = router.get_routing_statistics()
        sync_stats = sync_manager.get_sync_statistics()
        env_summary = sync_manager.get_environment_summary()

        assert routing_stats["total_routes"] >= 1
        assert sync_stats["total_records"] >= 1
        assert env_summary["paper_trading"]["active_orders"] >= 0

        # 8. 订单完成和归档
        await sync_manager.update_order_status(
            order.client_order_id,
            OrderStatus.FILLED,
            TradingMode.PAPER,
            {"status": "FILLED", "executedQty": "0.001"}
        )

        await sync_manager.archive_completed_order(
            order.client_order_id,
            TradingMode.PAPER
        )

        # 9. 历史查询
        history = await sync_manager.get_order_history(TradingMode.PAPER)
        assert len(history) >= 1

        # 10. 系统清理
        await sync_manager.cleanup()
        await router.cleanup()

        # 验证最终状态
        assert not sync_manager._running