import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.trading.paper_trading_engine import (
    PaperTradingEngine, VirtualAccount, VirtualPosition, VirtualTrade
)
from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData


class TestPaperTradingEngine:
    """虚拟盘交易引擎测试"""

    @pytest.fixture
    async def engine(self):
        """引擎实例"""
        engine = PaperTradingEngine("test_account")
        await engine.initialize()
        return engine

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
    def market_order(self):
        """市价单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="market_order_001"
        )

    @pytest.fixture
    def sample_market_data(self):
        """示例市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow()
        )

    async def test_initialization(self):
        """测试初始化"""
        engine = PaperTradingEngine("test_account_init")
        
        # 初始化前检查默认状态
        assert engine.account.account_id == "test_account_init"
        assert engine.account.initial_balance == 100000.0
        assert engine.account.current_balance == 100000.0
        assert len(engine.positions) == 0
        assert len(engine.active_orders) == 0
        
        await engine.initialize()
        
        # 初始化后状态
        assert engine.stats["peak_balance"] == 100000.0

    async def test_market_order_execution(self, engine, market_order):
        """测试市价单执行"""
        result = await engine.execute_order(market_order)
        
        # 验证返回结果
        assert "orderId" in result
        assert result["symbol"] == "BTCUSDT"
        assert result["status"] == "FILLED"
        assert result["executedQty"] == "0.001"
        assert "avgPrice" in result
        
        # 验证订单状态
        assert market_order.status == OrderStatus.FILLED
        assert market_order.executed_qty == 0.001
        assert market_order.order_id is not None
        
        # 验证统计更新
        assert engine.stats["total_orders"] == 1
        assert engine.stats["filled_orders"] == 1
        assert engine.stats["total_trades"] == 1
        
        # 验证仓位创建
        assert "BTCUSDT" in engine.positions
        position = engine.positions["BTCUSDT"]
        assert position.side == "LONG"
        assert position.size == 0.001

    async def test_limit_order_immediate_fill(self, engine):
        """测试限价单立即成交"""
        # 买单价格高于市价，应该立即成交
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=51000.0,  # 高于市价50000
            client_order_id="limit_buy_001"
        )
        
        result = await engine.execute_order(buy_order)
        
        assert result["status"] == "FILLED"
        assert buy_order.status == OrderStatus.FILLED

    async def test_limit_order_pending(self, engine):
        """测试限价单挂单"""
        # 买单价格低于市价，应该挂单
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0,  # 低于市价50000
            client_order_id="limit_pending_001"
        )
        
        result = await engine.execute_order(buy_order)
        
        assert result["status"] == "NEW"
        assert buy_order.status == OrderStatus.NEW
        assert buy_order.client_order_id in engine.active_orders

    async def test_stop_order_placement(self, engine):
        """测试止损单下单"""
        stop_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=0.001,
            stop_price=49000.0,
            client_order_id="stop_order_001"
        )
        
        result = await engine.execute_order(stop_order)
        
        assert result["status"] == "NEW"
        assert stop_order.status == OrderStatus.NEW
        assert stop_order.client_order_id in engine.active_orders

    async def test_slippage_calculation(self, engine, market_order):
        """测试滑点计算"""
        result = await engine.execute_order(market_order)
        
        executed_price = float(result["avgPrice"])
        market_price = 50000.0
        
        # 买单应该有向上滑点
        expected_price = market_price * (1 + engine.slippage_rate)
        assert abs(executed_price - expected_price) < 0.01

    async def test_commission_calculation(self, engine, market_order):
        """测试手续费计算"""
        initial_balance = engine.account.current_balance
        
        await engine.execute_order(market_order)
        
        # 计算预期手续费
        executed_price = 50000.0 * (1 + engine.slippage_rate)
        expected_commission = 0.001 * executed_price * engine.commission_rate
        
        # 验证余额扣除了手续费
        balance_change = initial_balance - engine.account.current_balance
        assert abs(balance_change - expected_commission) < 0.01
        
        # 验证统计
        assert engine.stats["total_commission"] > 0

    async def test_position_management_long(self, engine):
        """测试多头仓位管理"""
        # 开多头仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="long_001"
        )
        
        await engine.execute_order(buy_order)
        
        assert "BTCUSDT" in engine.positions
        position = engine.positions["BTCUSDT"]
        assert position.side == "LONG"
        assert position.size == 0.001
        assert position.entry_price > 0

    async def test_position_management_add_to_long(self, engine):
        """测试加仓"""
        # 第一次买入
        buy_order1 = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="long_001"
        )
        await engine.execute_order(buy_order1)
        
        original_position = engine.positions["BTCUSDT"]
        original_size = original_position.size
        original_entry = original_position.entry_price
        
        # 第二次买入（加仓）
        buy_order2 = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.002,
            client_order_id="long_002"
        )
        await engine.execute_order(buy_order2)
        
        # 验证仓位增加
        position = engine.positions["BTCUSDT"]
        assert position.size == original_size + 0.002
        # 成本价应该是加权平均
        assert position.entry_price != original_entry

    async def test_position_management_partial_close(self, engine):
        """测试部分平仓"""
        # 开仓
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.003,
            client_order_id="open_001"
        )
        await engine.execute_order(buy_order)
        
        # 部分平仓
        sell_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="close_001"
        )
        await engine.execute_order(sell_order)
        
        # 验证仓位减少
        position = engine.positions["BTCUSDT"]
        assert position.size == 0.002
        assert position.side == "LONG"

    async def test_position_management_full_close(self, engine):
        """测试完全平仓"""
        # 开仓
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="open_001"
        )
        await engine.execute_order(buy_order)
        
        # 完全平仓
        sell_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="close_001"
        )
        await engine.execute_order(sell_order)
        
        # 验证仓位已关闭
        assert "BTCUSDT" not in engine.positions

    async def test_position_management_reverse(self, engine):
        """测试反向开仓"""
        # 开多头仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="long_001"
        )
        await engine.execute_order(buy_order)
        
        # 反向开仓（卖出更多）
        sell_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.002,
            client_order_id="short_001"
        )
        await engine.execute_order(sell_order)
        
        # 应该变成空头仓位
        position = engine.positions["BTCUSDT"]
        assert position.side == "SHORT"
        assert position.size == 0.001

    async def test_insufficient_balance_rejection(self, engine):
        """测试余额不足拒绝"""
        # 尝试下一个超大订单
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,  # 需要约50万USDT，超过初始资金
            client_order_id="large_order_001"
        )
        
        result = await engine.execute_order(large_order)
        
        assert result["status"] == "REJECTED"
        assert "error" in result
        assert large_order.status == OrderStatus.REJECTED

    async def test_cancel_order(self, engine):
        """测试取消订单"""
        # 下一个挂单
        pending_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0,
            client_order_id="pending_001"
        )
        
        await engine.execute_order(pending_order)
        assert pending_order.client_order_id in engine.active_orders
        
        # 取消订单
        result = await engine.cancel_order("pending_001")
        
        assert result["status"] == "CANCELED"
        assert "pending_001" not in engine.active_orders
        
        # 验证订单在历史中
        canceled_order = next(
            (order for order in engine.order_history 
             if order.client_order_id == "pending_001"), 
            None
        )
        assert canceled_order is not None
        assert canceled_order.status == OrderStatus.CANCELED

    async def test_get_order_status_active(self, engine):
        """测试查询活跃订单状态"""
        # 下挂单
        pending_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0,
            client_order_id="status_test_001"
        )
        
        await engine.execute_order(pending_order)
        
        result = await engine.get_order_status("status_test_001")
        
        assert result["status"] == "NEW"
        assert result["clientOrderId"] == "status_test_001"
        assert result["symbol"] == "BTCUSDT"

    async def test_get_order_status_historical(self, engine):
        """测试查询历史订单状态"""
        # 执行并完成订单
        market_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="history_test_001"
        )
        
        await engine.execute_order(market_order)
        
        result = await engine.get_order_status("history_test_001")
        
        assert result["status"] == "FILLED"
        assert result["executedQty"] == "0.001"

    async def test_get_order_status_not_found(self, engine):
        """测试查询不存在的订单"""
        with pytest.raises(ValueError, match="Order not found"):
            await engine.get_order_status("nonexistent_order")

    async def test_market_data_update(self, engine, sample_market_data):
        """测试市场数据更新"""
        # 先建立仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="market_data_test"
        )
        await engine.execute_order(buy_order)
        
        # 更新市场数据
        new_market_data = MarketData(
            symbol="BTCUSDT",
            price=55000.0,  # 价格上涨
            timestamp=datetime.utcnow()
        )
        
        engine.update_market_data(new_market_data)
        
        # 验证价格更新
        assert engine.market_prices["BTCUSDT"] == 55000.0
        
        # 验证仓位盈亏更新
        position = engine.positions["BTCUSDT"]
        assert position.current_price == 55000.0
        assert position.unrealized_pnl > 0  # 应该有盈利

    async def test_process_pending_orders_limit(self, engine):
        """测试处理挂单：限价单"""
        # 下买单，价格低于市价
        pending_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=49000.0,
            client_order_id="pending_limit_001"
        )
        
        await engine.execute_order(pending_order)
        assert pending_order.status == OrderStatus.NEW
        
        # 模拟市价下跌到限价以下
        engine.market_prices["BTCUSDT"] = 48000.0
        
        # 处理挂单
        await engine.process_pending_orders()
        
        # 应该成交
        assert pending_order.status == OrderStatus.FILLED
        assert "pending_limit_001" not in engine.active_orders

    async def test_process_pending_orders_stop(self, engine):
        """测试处理挂单：止损单"""
        # 先建立多头仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="position_001"
        )
        await engine.execute_order(buy_order)
        
        # 下止损单
        stop_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=0.001,
            stop_price=49000.0,
            client_order_id="stop_001"
        )
        
        await engine.execute_order(stop_order)
        assert stop_order.status == OrderStatus.NEW
        
        # 模拟市价跌破止损价
        engine.market_prices["BTCUSDT"] = 48000.0
        
        # 处理挂单
        await engine.process_pending_orders()
        
        # 止损单应该触发
        assert stop_order.status == OrderStatus.FILLED

    async def test_get_account_info(self, engine):
        """测试获取账户信息"""
        # 建立一些仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="account_test_001"
        )
        await engine.execute_order(buy_order)
        
        account_info = engine.get_account_info()
        
        assert account_info["accountId"] == "test_account"
        assert "balance" in account_info
        assert "availableBalance" in account_info
        assert "totalPnL" in account_info
        assert "unrealizedPnL" in account_info
        assert "totalEquity" in account_info
        assert account_info["positions"] == 1
        assert account_info["activeOrders"] >= 0

    async def test_get_positions(self, engine):
        """测试获取持仓信息"""
        # 建立多个仓位
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            buy_order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                client_order_id=f"pos_{symbol}"
            )
            await engine.execute_order(buy_order)
        
        positions = engine.get_positions()
        
        assert len(positions) == 2
        for pos in positions:
            assert "symbol" in pos
            assert "side" in pos
            assert "size" in pos
            assert "entryPrice" in pos
            assert "currentPrice" in pos
            assert "unrealizedPnL" in pos

    async def test_get_trading_statistics(self, engine):
        """测试获取交易统计"""
        # 执行一些交易
        for i in range(3):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.001,
                client_order_id=f"stat_test_{i}"
            )
            await engine.execute_order(order)
        
        stats = engine.get_trading_statistics()
        
        assert stats["total_orders"] == 3
        assert stats["filled_orders"] == 3
        assert stats["totalTrades"] == 3
        assert "winRate" in stats
        assert "avgCommission" in stats
        assert "currentEquity" in stats
        assert "returnRate" in stats

    async def test_drawdown_calculation(self, engine):
        """测试回撤计算"""
        initial_peak = engine.stats["peak_balance"]
        
        # 执行一个会产生手续费的交易
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001,
            client_order_id="drawdown_test"
        )
        await engine.execute_order(order)
        
        # 由于手续费，余额应该略有下降
        current_equity = engine.account.current_balance + engine.account.total_pnl
        assert current_equity < initial_peak
        
        # 最大回撤应该被记录
        assert engine.stats["max_drawdown"] > 0

    async def test_virtual_position_pnl_update(self):
        """测试虚拟仓位盈亏更新"""
        position = VirtualPosition(
            symbol="BTCUSDT",
            side="LONG",
            size=0.001,
            entry_price=50000.0,
            current_price=50000.0
        )
        
        # 价格上涨，多头应该盈利
        position.update_pnl(55000.0)
        assert position.unrealized_pnl == 5.0  # (55000 - 50000) * 0.001
        
        # 价格下跌，多头应该亏损
        position.update_pnl(45000.0)
        assert position.unrealized_pnl == -5.0  # (45000 - 50000) * 0.001

    async def test_virtual_position_short_pnl(self):
        """测试空头仓位盈亏"""
        position = VirtualPosition(
            symbol="BTCUSDT",
            side="SHORT",
            size=0.001,
            entry_price=50000.0,
            current_price=50000.0
        )
        
        # 价格下跌，空头应该盈利
        position.update_pnl(45000.0)
        assert position.unrealized_pnl == 5.0  # (50000 - 45000) * 0.001
        
        # 价格上涨，空头应该亏损
        position.update_pnl(55000.0)
        assert position.unrealized_pnl == -5.0  # (50000 - 55000) * 0.001

    async def test_virtual_account_initialization(self):
        """测试虚拟账户初始化"""
        account = VirtualAccount("test_account", initial_balance=50000.0)
        
        assert account.account_id == "test_account"
        assert account.initial_balance == 50000.0
        assert account.current_balance == 50000.0
        assert account.available_balance == 50000.0
        assert account.total_pnl == 0.0

    async def test_virtual_trade_creation(self):
        """测试虚拟成交记录"""
        trade = VirtualTrade(
            trade_id="trade_001",
            order_id="order_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001,
            price=50000.0,
            commission=0.02
        )
        
        assert trade.trade_id == "trade_001"
        assert trade.symbol == "BTCUSDT"
        assert trade.quantity == 0.001
        assert trade.commission == 0.02
        assert isinstance(trade.timestamp, datetime)