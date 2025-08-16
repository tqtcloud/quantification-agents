import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.exchanges.binance_quant_order_manager import (
    BinanceQuantOrderManager, BinanceOrderType, WorkingType, 
    SelfTradePreventionMode, OrderValidationRule, OrderRiskCheck
)
from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError
from src.core.models import Order, OrderSide, OrderType, OrderStatus, TimeInForce, PositionSide


class TestBinanceQuantOrderManager:
    """币安量化订单管理器测试"""

    @pytest.fixture
    async def mock_client(self):
        """模拟币安客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        
        # 模拟交易所信息
        client.get_exchange_info = AsyncMock(return_value={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "pricePrecision": 2,
                    "quantityPrecision": 6,
                    "orderTypes": ["MARKET", "LIMIT", "STOP", "STOP_MARKET"],
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "minPrice": "0.01",
                            "maxPrice": "1000000",
                            "tickSize": "0.01"
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.001",
                            "maxQty": "9000",
                            "stepSize": "0.001"
                        },
                        {
                            "filterType": "MIN_NOTIONAL",
                            "notional": "5.0"
                        },
                        {
                            "filterType": "MAX_NUM_ORDERS",
                            "maxNumOrders": "200"
                        }
                    ]
                }
            ]
        })
        
        return client

    @pytest.fixture
    async def order_manager(self, mock_client):
        """订单管理器实例"""
        manager = BinanceQuantOrderManager(mock_client)
        await manager.initialize()
        return manager

    @pytest.fixture
    def sample_order(self):
        """示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_order_001"
        )

    async def test_initialization(self, mock_client):
        """测试初始化"""
        manager = BinanceQuantOrderManager(mock_client)
        
        # 初始化前应该为空
        assert len(manager.symbol_configs) == 0
        
        await manager.initialize()
        
        # 初始化后应该有配置
        assert len(manager.symbol_configs) > 0
        assert "BTCUSDT" in manager.symbol_configs
        
        # 验证配置内容
        btc_config = manager.symbol_configs["BTCUSDT"]
        assert btc_config["status"] == "TRADING"
        assert btc_config["baseAsset"] == "BTC"
        assert btc_config["quoteAsset"] == "USDT"

    async def test_validation_rules_generation(self, order_manager):
        """测试验证规则生成"""
        rules = order_manager._get_validation_rules("BTCUSDT")
        
        assert isinstance(rules, OrderValidationRule)
        assert rules.min_quantity == 0.001
        assert rules.max_quantity == 9000
        assert rules.min_price == 0.01
        assert rules.max_price == 1000000
        assert rules.tick_size == 0.01
        assert rules.step_size == 0.001
        assert rules.min_notional == 5.0
        assert rules.max_num_orders == 200

    async def test_order_validation_success(self, order_manager, sample_order):
        """测试订单验证成功"""
        validation = await order_manager.validate_order(sample_order)
        
        assert isinstance(validation, OrderRiskCheck)
        assert validation.is_valid is True
        assert len(validation.violations) == 0
        assert validation.risk_score >= 0

    async def test_order_validation_failures(self, order_manager):
        """测试订单验证失败情况"""
        # 测试无效数量
        invalid_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.0001,  # 低于最小数量
            price=50000.0
        )
        
        validation = await order_manager.validate_order(invalid_order)
        assert validation.is_valid is False
        assert any("below minimum" in v for v in validation.violations)

        # 测试无效价格
        invalid_order.quantity = 0.001
        invalid_order.price = 0.001  # 低于最小价格
        
        validation = await order_manager.validate_order(invalid_order)
        assert validation.is_valid is False
        assert any("below minimum" in v for v in validation.violations)

        # 测试名义价值过低
        invalid_order.price = 1000.0  # 1000 * 0.001 = 1 < 5
        
        validation = await order_manager.validate_order(invalid_order)
        assert validation.is_valid is False
        assert any("Notional value" in v for v in validation.violations)

    async def test_place_order_success(self, order_manager, sample_order):
        """测试下单成功"""
        # 模拟API响应
        order_manager.client.place_order = AsyncMock(return_value={
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "test_order_001",
            "transactTime": 1609459200000
        })

        result = await order_manager.place_order(
            symbol=sample_order.symbol,
            side=sample_order.side,
            order_type=BinanceOrderType.LIMIT,
            quantity=sample_order.quantity,
            price=sample_order.price,
            client_order_id=sample_order.client_order_id
        )

        # 验证返回结果
        assert result["orderId"] == 123456789
        assert result["status"] == "NEW"
        assert result["clientOrderId"] == "test_order_001"

        # 验证订单已添加到活跃订单
        assert sample_order.client_order_id in order_manager.active_orders
        
        # 验证统计更新
        assert order_manager.daily_order_count == 1
        assert order_manager.order_stats["total_orders"] == 1

    async def test_place_order_validation_failure(self, order_manager):
        """测试下单验证失败"""
        with pytest.raises(ValueError, match="Order validation failed"):
            await order_manager.place_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=BinanceOrderType.LIMIT,
                quantity=0.0001,  # 无效数量
                price=50000.0
            )

    async def test_place_order_api_error(self, order_manager, sample_order):
        """测试API错误处理"""
        # 模拟API错误
        order_manager.client.place_order = AsyncMock(
            side_effect=BinanceAPIError("Insufficient balance", -2010)
        )

        with pytest.raises(BinanceAPIError):
            await order_manager.place_order(
                symbol=sample_order.symbol,
                side=sample_order.side,
                order_type=BinanceOrderType.LIMIT,
                quantity=sample_order.quantity,
                price=sample_order.price
            )

        # 验证统计更新
        assert order_manager.order_stats["failed_orders"] == 1
        assert order_manager.order_stats["rejected_orders"] == 1

    async def test_place_market_order(self, order_manager):
        """测试市价单"""
        order_manager.client.place_order = AsyncMock(return_value={
            "orderId": 123456790,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "clientOrderId": "market_order_001"
        })

        result = await order_manager.place_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=BinanceOrderType.MARKET,
            quantity=0.001
        )

        assert result["status"] == "FILLED"
        assert order_manager.order_stats["successful_orders"] == 1

    async def test_place_stop_order(self, order_manager):
        """测试止损单"""
        order_manager.client.place_order = AsyncMock(return_value={
            "orderId": 123456791,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "stop_order_001"
        })

        result = await order_manager.place_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=BinanceOrderType.STOP_MARKET,
            quantity=0.001,
            stop_price=49000.0
        )

        assert result["status"] == "NEW"
        
        # 验证API调用参数
        call_args = order_manager.client.place_order.call_args[1]
        assert call_args["stopPrice"] == "49000.0"
        assert call_args["type"] == "STOP_MARKET"

    async def test_place_trailing_stop_order(self, order_manager):
        """测试追踪止损单"""
        order_manager.client.place_order = AsyncMock(return_value={
            "orderId": 123456792,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "trailing_stop_001"
        })

        result = await order_manager.place_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=BinanceOrderType.TRAILING_STOP_MARKET,
            quantity=0.001,
            callback_rate=1.0,  # 1%回调率
            activation_price=51000.0
        )

        assert result["status"] == "NEW"
        
        # 验证特殊参数
        call_args = order_manager.client.place_order.call_args[1]
        assert call_args["callbackRate"] == "1.0"
        assert call_args["activationPrice"] == "51000.0"

    async def test_batch_orders(self, order_manager):
        """测试批量下单"""
        # 模拟每个订单的API响应
        order_manager.place_order = AsyncMock(side_effect=[
            {"orderId": 1001, "status": "NEW", "clientOrderId": "batch_001_0"},
            {"orderId": 1002, "status": "NEW", "clientOrderId": "batch_001_1"},
            {"orderId": 1003, "status": "NEW", "clientOrderId": "batch_001_2"}
        ])

        batch_orders = [
            {
                "symbol": "BTCUSDT",
                "side": OrderSide.BUY,
                "order_type": BinanceOrderType.LIMIT,
                "quantity": 0.001,
                "price": 50000.0
            },
            {
                "symbol": "BTCUSDT",
                "side": OrderSide.BUY,
                "order_type": BinanceOrderType.LIMIT,
                "quantity": 0.002,
                "price": 49500.0
            },
            {
                "symbol": "BTCUSDT",
                "side": OrderSide.SELL,
                "order_type": BinanceOrderType.LIMIT,
                "quantity": 0.001,
                "price": 50500.0
            }
        ]

        results = await order_manager.place_batch_orders(batch_orders)

        assert len(results) == 3
        assert all("orderId" in result for result in results)
        assert order_manager.place_order.call_count == 3

    async def test_batch_orders_size_limit(self, order_manager):
        """测试批量下单数量限制"""
        large_batch = [{"symbol": "BTCUSDT"}] * 6  # 超过5个限制

        with pytest.raises(ValueError, match="Batch size cannot exceed 5 orders"):
            await order_manager.place_batch_orders(large_batch)

    async def test_cancel_order(self, order_manager):
        """测试取消订单"""
        # 先添加一个活跃订单
        order_manager.active_orders["test_cancel"] = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            client_order_id="test_cancel"
        )

        order_manager.client.cancel_order = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "orderId": 123456789,
            "clientOrderId": "test_cancel",
            "status": "CANCELED"
        })

        result = await order_manager.cancel_order(
            symbol="BTCUSDT",
            client_order_id="test_cancel"
        )

        assert result["status"] == "CANCELED"
        assert "test_cancel" not in order_manager.active_orders

    async def test_cancel_all_orders(self, order_manager):
        """测试取消所有订单"""
        # 添加多个活跃订单
        order_manager.active_orders.update({
            "btc_1": Order(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.001),
            "btc_2": Order(symbol="BTCUSDT", side=OrderSide.SELL, order_type=OrderType.LIMIT, quantity=0.001),
            "eth_1": Order(symbol="ETHUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.01)
        })

        order_manager.client.cancel_all_orders = AsyncMock(return_value={
            "code": 200,
            "msg": "The operation of cancel all open order is done."
        })

        result = await order_manager.cancel_all_orders("BTCUSDT")

        assert result["code"] == 200
        
        # 验证只有BTCUSDT的订单被移除
        remaining_orders = [order.symbol for order in order_manager.active_orders.values()]
        assert "BTCUSDT" not in remaining_orders
        assert "ETHUSDT" in remaining_orders

    async def test_modify_order(self, order_manager):
        """测试修改订单"""
        order_manager.client.modify_order = AsyncMock(return_value={
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "price": "49500.0",
            "origQty": "0.002"
        })

        result = await order_manager.modify_order(
            symbol="BTCUSDT",
            order_id=123456789,
            side=OrderSide.BUY,
            quantity=0.002,
            price=49500.0
        )

        assert result["price"] == "49500.0"
        assert result["origQty"] == "0.002"

    async def test_get_order_status(self, order_manager):
        """测试查询订单状态"""
        # 添加活跃订单
        order_manager.active_orders["status_test"] = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            client_order_id="status_test"
        )

        order_manager.client.get_order = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "orderId": 123456789,
            "clientOrderId": "status_test",
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.0005",
            "avgPrice": "50000.0"
        })

        result = await order_manager.get_order_status(
            symbol="BTCUSDT",
            client_order_id="status_test"
        )

        assert result["status"] == "PARTIALLY_FILLED"
        assert result["executedQty"] == "0.0005"
        
        # 验证本地状态更新
        local_order = order_manager.active_orders["status_test"]
        assert local_order.status == OrderStatus.PARTIALLY_FILLED
        assert local_order.executed_qty == 0.0005
        assert local_order.avg_price == 50000.0

    async def test_get_order_status_completed_order(self, order_manager):
        """测试查询已完成订单状态"""
        order_manager.active_orders["completed_test"] = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            client_order_id="completed_test"
        )

        order_manager.client.get_order = AsyncMock(return_value={
            "symbol": "BTCUSDT",
            "orderId": 123456789,
            "clientOrderId": "completed_test",
            "status": "FILLED",
            "executedQty": "0.001",
            "avgPrice": "50000.0"
        })

        result = await order_manager.get_order_status(
            symbol="BTCUSDT",
            client_order_id="completed_test"
        )

        assert result["status"] == "FILLED"
        
        # 已完成的订单应该从活跃订单中移除
        assert "completed_test" not in order_manager.active_orders

    async def test_daily_stats_reset(self, order_manager):
        """测试日统计重置"""
        # 设置订单计数
        order_manager.daily_order_count = 50
        order_manager.last_reset_date = datetime(2023, 1, 1).date()

        # 触发重置检查
        order_manager._reset_daily_stats_if_needed()

        # 应该重置为0
        assert order_manager.daily_order_count == 0
        assert order_manager.last_reset_date == datetime.utcnow().date()

    async def test_order_statistics(self, order_manager):
        """测试订单统计"""
        # 设置统计数据
        order_manager.order_stats.update({
            "total_orders": 100,
            "successful_orders": 85,
            "failed_orders": 10,
            "rejected_orders": 5
        })
        order_manager.daily_order_count = 25
        order_manager.active_orders["test1"] = Order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.001
        )

        stats = order_manager.get_order_statistics()

        assert stats["total_orders"] == 100
        assert stats["successful_orders"] == 85
        assert stats["success_rate"] == 0.85
        assert stats["rejection_rate"] == 0.05
        assert stats["daily_order_count"] == 25
        assert stats["active_orders_count"] == 1

    async def test_risk_scoring(self, order_manager):
        """测试风险评分"""
        # 高风险订单（大额市价单）
        high_risk_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,  # 大数量
            price=50000.0
        )

        validation = await order_manager.validate_order(high_risk_order)
        assert validation.risk_score > 0.5
        assert len(validation.warnings) > 0

        # 低风险订单（小额限价单）
        low_risk_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0
        )

        validation = await order_manager.validate_order(low_risk_order)
        assert validation.risk_score < 0.3

    async def test_cleanup(self, order_manager):
        """测试资源清理"""
        # 添加一些数据
        order_manager.active_orders["test"] = Order(
            symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.001
        )
        order_manager.conditional_orders["cond"] = (
            Order(symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=0.001),
            MagicMock()
        )

        await order_manager.cleanup()

        # 验证数据已清理
        assert len(order_manager.active_orders) == 0
        assert len(order_manager.conditional_orders) == 0
        assert len(order_manager.batch_orders) == 0

    async def test_unknown_symbol_validation(self, order_manager):
        """测试未知交易对的验证"""
        unknown_order = Order(
            symbol="UNKNOWNUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=1000.0
        )

        # 应该使用默认规则，不应该失败
        validation = await order_manager.validate_order(unknown_order)
        assert validation.is_valid is True  # 使用默认规则应该通过

    async def test_precision_validation(self, order_manager):
        """测试精度验证"""
        # 测试价格精度
        invalid_price_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.001  # 超过tick_size精度
        )

        validation = await order_manager.validate_order(invalid_price_order)
        assert validation.is_valid is False
        assert any("tick size" in v for v in validation.violations)

        # 测试数量精度
        invalid_qty_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.0001,  # 不符合step_size
            price=50000.0
        )

        validation = await order_manager.validate_order(invalid_qty_order)
        assert validation.is_valid is False
        assert any("step size" in v for v in validation.violations)