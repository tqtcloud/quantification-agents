import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading.enhanced_paper_trading_engine import (
    EnhancedPaperTradingEngine, 
    EnhancedVirtualAccount, 
    EnhancedVirtualPosition,
    TradingParameters,
    MarketDepth,
    PositionMode
)
from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData


class TestEnhancedVirtualAccount:
    """测试增强虚拟账户"""
    
    def test_account_initialization(self):
        """测试账户初始化"""
        account = EnhancedVirtualAccount(
            account_id="test_001",
            initial_balance=Decimal("100000.0")
        )
        
        assert account.account_id == "test_001"
        assert account.initial_balance == Decimal("100000.0")
        assert account.current_balance == Decimal("100000.0")
        assert account.available_balance == Decimal("100000.0")
        assert account.margin_used == Decimal("0.0")
        assert account.total_pnl == Decimal("0.0")
        assert account.max_leverage == 20
        assert account.position_mode == PositionMode.ONE_WAY
        
    def test_daily_stats_reset(self):
        """测试日统计重置"""
        account = EnhancedVirtualAccount(
            account_id="test_001",
            daily_loss=Decimal("500.0"),
            last_reset_date=datetime.utcnow() - timedelta(days=1)
        )
        
        # 重置前
        assert account.daily_loss == Decimal("500.0")
        
        # 执行重置
        account.reset_daily_stats()
        
        # 重置后
        assert account.daily_loss == Decimal("0.0")
        assert account.last_reset_date.date() == datetime.utcnow().date()


class TestEnhancedVirtualPosition:
    """测试增强虚拟仓位"""
    
    def test_position_initialization(self):
        """测试仓位初始化"""
        position = EnhancedVirtualPosition(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            margin=Decimal("2500.0"),
            leverage=20
        )
        
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.size == Decimal("1.0")
        assert position.entry_price == Decimal("50000.0")
        assert position.margin == Decimal("2500.0")
        assert position.leverage == 20
        
    def test_long_position_pnl_update(self):
        """测试多头仓位盈亏更新"""
        position = EnhancedVirtualPosition(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            margin=Decimal("2500.0")
        )
        
        # 价格上涨
        position.update_pnl(Decimal("52000.0"))
        assert position.unrealized_pnl == Decimal("2000.0")
        assert position.current_price == Decimal("52000.0")
        
        # 价格下跌
        position.update_pnl(Decimal("48000.0"))
        assert position.unrealized_pnl == Decimal("-2000.0")
        
    def test_short_position_pnl_update(self):
        """测试空头仓位盈亏更新"""
        position = EnhancedVirtualPosition(
            symbol="BTCUSDT",
            side="SHORT",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            margin=Decimal("2500.0")
        )
        
        # 价格下跌（空头盈利）
        position.update_pnl(Decimal("48000.0"))
        assert position.unrealized_pnl == Decimal("2000.0")
        
        # 价格上涨（空头亏损）
        position.update_pnl(Decimal("52000.0"))
        assert position.unrealized_pnl == Decimal("-2000.0")
        
    def test_liquidation_price_calculation(self):
        """测试强平价格计算"""
        # 多头仓位
        long_position = EnhancedVirtualPosition(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            margin=Decimal("2500.0")
        )
        
        long_position._calculate_liquidation_price()
        # 强平价 = 开仓价 - (保证金 * 0.995) / 数量
        expected_liq_price = Decimal("50000.0") - (Decimal("2500.0") * Decimal("0.995"))
        assert long_position.liquidation_price == expected_liq_price
        
        # 空头仓位
        short_position = EnhancedVirtualPosition(
            symbol="BTCUSDT",
            side="SHORT",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0"),
            margin=Decimal("2500.0")
        )
        
        short_position._calculate_liquidation_price()
        # 强平价 = 开仓价 + (保证金 * 0.995) / 数量
        expected_liq_price = Decimal("50000.0") + (Decimal("2500.0") * Decimal("0.995"))
        assert short_position.liquidation_price == expected_liq_price


class TestMarketDepth:
    """测试市场深度"""
    
    def test_market_depth_initialization(self):
        """测试市场深度初始化"""
        bids = [(Decimal("50000.0"), Decimal("1.0")), (Decimal("49990.0"), Decimal("2.0"))]
        asks = [(Decimal("50010.0"), Decimal("1.5")), (Decimal("50020.0"), Decimal("1.8"))]
        
        depth = MarketDepth(symbol="BTCUSDT", bids=bids, asks=asks)
        
        assert depth.symbol == "BTCUSDT"
        assert depth.bids == bids
        assert depth.asks == asks
        
    def test_best_bid_ask(self):
        """测试最佳买卖价获取"""
        bids = [(Decimal("50000.0"), Decimal("1.0")), (Decimal("49990.0"), Decimal("2.0"))]
        asks = [(Decimal("50010.0"), Decimal("1.5")), (Decimal("50020.0"), Decimal("1.8"))]
        
        depth = MarketDepth(symbol="BTCUSDT", bids=bids, asks=asks)
        
        assert depth.get_best_bid() == Decimal("50000.0")
        assert depth.get_best_ask() == Decimal("50010.0")
        
    def test_mid_price(self):
        """测试中间价计算"""
        bids = [(Decimal("50000.0"), Decimal("1.0"))]
        asks = [(Decimal("50010.0"), Decimal("1.5"))]
        
        depth = MarketDepth(symbol="BTCUSDT", bids=bids, asks=asks)
        
        expected_mid = (Decimal("50000.0") + Decimal("50010.0")) / Decimal("2.0")
        assert depth.get_mid_price() == expected_mid


@pytest.mark.asyncio
class TestEnhancedPaperTradingEngine:
    """测试增强虚拟盘交易引擎"""
    
    @pytest.fixture
    async def trading_engine(self):
        """创建交易引擎实例"""
        engine = EnhancedPaperTradingEngine(
            account_id="test_engine",
            initial_balance=Decimal("100000.0")
        )
        await engine.initialize()
        return engine
        
    async def test_engine_initialization(self, trading_engine):
        """测试引擎初始化"""
        assert trading_engine.account.account_id == "test_engine"
        assert trading_engine.account.initial_balance == Decimal("100000.0")
        assert len(trading_engine.contract_configs) > 0
        assert "BTCUSDT" in trading_engine.contract_configs
        assert trading_engine.performance_stats["peak_balance"] == Decimal("100000.0")
        
    async def test_market_order_execution(self, trading_engine):
        """测试市价单执行"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1  # 减小数量避免超出风险限制
        )
        
        result = await trading_engine.execute_order(order)
        
        assert result["status"] == "FILLED"
        assert order.status == OrderStatus.FILLED
        assert order.executed_qty == 0.1
        assert order.avg_price > 0
        assert "BTCUSDT" in trading_engine.positions
        
        # 检查仓位
        position = trading_engine.positions["BTCUSDT"]
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.size == Decimal("0.1")
        
    async def test_limit_order_execution(self, trading_engine):
        """测试限价单执行"""
        # 设置市场价格
        trading_engine.market_prices["BTCUSDT"] = Decimal("50000.0")
        
        # 可立即成交的限价买单（价格合理，不会被风险控制拦截）
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,  # 减小数量避免超出仓位限制
            price=50100.0  # 高于市价，可成交
        )
        
        result = await trading_engine.execute_order(order)
        
        assert result["status"] == "FILLED"
        assert order.status == OrderStatus.FILLED
        assert order.avg_price == 50100.0  # 按限价成交
        
    async def test_limit_order_pending(self, trading_engine):
        """测试限价单挂单"""
        # 设置市场价格
        trading_engine.market_prices["BTCUSDT"] = Decimal("50000.0")
        
        # 无法立即成交的限价买单
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,  # 减小数量避免风险控制问题
            price=49000.0  # 低于市价，无法成交
        )
        
        result = await trading_engine.execute_order(order)
        
        assert result["status"] == "NEW"
        assert order.status == OrderStatus.NEW
        assert order.client_order_id in trading_engine.active_orders
        
    async def test_position_management(self, trading_engine):
        """测试仓位管理"""
        # 开多头仓位
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(buy_order)
        
        # 检查仓位
        assert "BTCUSDT" in trading_engine.positions
        position = trading_engine.positions["BTCUSDT"]
        assert position.side == "LONG"
        assert position.size == Decimal("0.1")
        
        # 加仓
        add_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.05
        )
        await trading_engine.execute_order(add_order)
        
        # 检查加仓后的仓位
        position = trading_engine.positions["BTCUSDT"]
        assert position.size == Decimal("0.15")
        
        # 部分平仓
        close_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.05
        )
        await trading_engine.execute_order(close_order)
        
        # 检查平仓后的仓位
        position = trading_engine.positions["BTCUSDT"]
        assert position.size == Decimal("0.1")
        
        # 完全平仓
        close_all_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(close_all_order)
        
        # 检查仓位是否已清空
        assert "BTCUSDT" not in trading_engine.positions
        
    async def test_risk_management(self, trading_engine):
        """测试风险管理"""
        # 尝试超出最大仓位限制的订单
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0  # 大数量，但会被余额限制拦截
        )
        
        # 模拟价格以计算所需保证金
        trading_engine.market_prices["BTCUSDT"] = Decimal("50000.0")
        
        result = await trading_engine.execute_order(large_order)
        
        # 应该被风险控制拒绝（由于余额不足）
        assert result["status"] == "REJECTED"
        assert "Insufficient available balance" in result["reason"]
        
    async def test_daily_loss_limit(self, trading_engine):
        """测试日亏损限制"""
        # 设置日亏损接近限制
        trading_engine.account.daily_loss = Decimal("999.0")  # 接近1000限制
        trading_engine.account.max_daily_loss = Decimal("1000.0")
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        result = await trading_engine.execute_order(order)
        assert result["status"] in ["FILLED", "NEW"]  # 应该仍可执行
        
        # 超出日亏损限制
        trading_engine.account.daily_loss = Decimal("1000.0")
        
        order2 = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        result2 = await trading_engine.execute_order(order2)
        assert result2["status"] == "REJECTED"
        assert "Daily loss limit exceeded" in result2["reason"]
        
    async def test_slippage_calculation(self, trading_engine):
        """测试滑点计算"""
        # 设置市场价格和深度
        market_price = Decimal("50000.0")
        trading_engine.market_prices["BTCUSDT"] = market_price
        
        # 创建模拟订单簿
        depth = MarketDepth(
            symbol="BTCUSDT",
            bids=[(Decimal("49995.0"), Decimal("10.0"))],
            asks=[(Decimal("50005.0"), Decimal("10.0"))]
        )
        trading_engine.market_depths["BTCUSDT"] = depth
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # 计算执行价格
        execution_price = await trading_engine._calculate_execution_price(order, market_price, depth)
        
        # 买单应该有正滑点（价格更高）
        assert execution_price > market_price
        
    async def test_commission_calculation(self, trading_engine):
        """测试手续费计算"""
        execution_price = Decimal("50000.0")
        
        # 市价单（taker）
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        commission = trading_engine._calculate_commission(order, execution_price, is_maker=False)
        expected_commission = Decimal("1.0") * execution_price * trading_engine.trading_params.taker_fee_rate
        assert commission == expected_commission
        
        # 限价单（maker）
        commission_maker = trading_engine._calculate_commission(order, execution_price, is_maker=True)
        expected_commission_maker = Decimal("1.0") * execution_price * trading_engine.trading_params.maker_fee_rate
        assert commission_maker == expected_commission_maker
        
    async def test_market_data_update(self, trading_engine):
        """测试市场数据更新"""
        # 先创建一个仓位
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(order)
        
        # 更新市场数据
        market_data = MarketData(
            symbol="BTCUSDT",
            price=52000.0,
            volume=1000.0,
            bid=51950.0,
            ask=52050.0,
            bid_volume=500.0,
            ask_volume=500.0,
            timestamp=datetime.utcnow()
        )
        
        await trading_engine.update_market_data(market_data)
        
        # 检查价格更新
        assert trading_engine.market_prices["BTCUSDT"] == Decimal("52000.0")
        
        # 检查仓位盈亏更新
        position = trading_engine.positions["BTCUSDT"]
        assert position.current_price == Decimal("52000.0")
        assert position.unrealized_pnl > 0  # 应该有盈利
        
    async def test_pending_order_processing(self, trading_engine):
        """测试挂单处理"""
        # 设置初始市场价格
        trading_engine.market_prices["BTCUSDT"] = Decimal("50000.0")
        
        # 创建挂单
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,  # 减小数量避免风险控制问题
            price=49500.0  # 低于市价的买单
        )
        
        await trading_engine.execute_order(order)
        assert order.status == OrderStatus.NEW
        
        # 价格下跌，应该触发成交
        trading_engine.market_prices["BTCUSDT"] = Decimal("49400.0")
        await trading_engine.process_pending_orders()
        
        # 检查订单是否成交
        assert order.status == OrderStatus.FILLED
        assert order.avg_price == 49500.0  # 按限价成交
        
    async def test_stop_order_processing(self, trading_engine):
        """测试止损单处理"""
        # 设置初始市场价格
        trading_engine.market_prices["BTCUSDT"] = Decimal("50000.0")
        
        # 创建止损单
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=1.0,
            stop_price=49000.0  # 止损价
        )
        
        await trading_engine.execute_order(order)
        assert order.status == OrderStatus.NEW
        
        # 价格跌破止损价
        trading_engine.market_prices["BTCUSDT"] = Decimal("48500.0")
        await trading_engine.process_pending_orders()
        
        # 检查止损单是否触发
        assert order.status == OrderStatus.FILLED
        
    async def test_liquidation_check(self, trading_engine):
        """测试强平检查"""
        # 创建一个有保证金的仓位
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        await trading_engine.execute_order(order)
        
        position = trading_engine.positions["BTCUSDT"]
        position.margin = Decimal("2500.0")  # 设置保证金
        position._calculate_liquidation_price()  # 计算强平价
        
        # 价格跌破强平价
        liquidation_price = position.liquidation_price
        trading_engine.market_prices["BTCUSDT"] = liquidation_price - Decimal("100.0")
        
        # 检查强平
        await trading_engine._check_liquidation("BTCUSDT")
        
        # 仓位应该被强平（这里需要mock强平逻辑或检查相关状态）
        
    async def test_account_info(self, trading_engine):
        """测试账户信息获取"""
        account_info = trading_engine.get_enhanced_account_info()
        
        assert account_info["accountId"] == "test_engine"
        assert account_info["balance"] == 100000.0
        assert "riskMetrics" in account_info
        assert "leverageRatio" in account_info["riskMetrics"]
        assert "marginUtilization" in account_info["riskMetrics"]
        
    async def test_performance_stats(self, trading_engine):
        """测试性能统计"""
        # 执行一些交易
        order1 = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(order1)
        
        order2 = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(order2)
        
        stats = trading_engine.get_enhanced_performance_stats()
        
        assert stats["totalTrades"] == len(trading_engine.trade_history)
        assert stats["filled_orders"] == 2
        assert "winRate" in stats
        assert "totalVolume" in stats
        assert "returnRate" in stats
        
    async def test_required_margin_calculation(self, trading_engine):
        """测试所需保证金计算"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0
        )
        
        required_margin = trading_engine._calculate_required_margin(order)
        
        # 保证金 = 名义价值 / 杠杆
        expected_margin = Decimal("50000.0") / Decimal("20")  # 默认20倍杠杆
        assert required_margin == expected_margin
        
    async def test_risk_metrics_calculation(self, trading_engine):
        """测试风险指标计算"""
        # 创建一些仓位
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        await trading_engine.execute_order(order)
        
        risk_metrics = trading_engine._calculate_risk_metrics()
        
        assert "portfolioValue" in risk_metrics
        assert "leverageRatio" in risk_metrics
        assert "marginUtilization" in risk_metrics
        assert "positionCount" in risk_metrics
        assert risk_metrics["positionCount"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])