import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.trading.enhanced_paper_trading_engine import (
    EnhancedPaperTradingEngine,
    TradingParameters,
    MarketDepth
)
from src.trading.market_simulation import (
    IntegratedMarketSimulator,
    MarketCondition,
    SlippageConfig,
    CommissionStructure
)
from src.trading.performance_analytics import (
    PerformanceTracker,
    TradeRecord,
    PerformancePeriod
)
from src.core.models import Order, OrderSide, OrderType, MarketData


@pytest.mark.asyncio
class TestVirtualTradingIntegration:
    """虚拟盘交易系统集成测试"""
    
    @pytest.fixture
    async def trading_system(self):
        """创建完整的虚拟盘交易系统"""
        # 创建增强交易引擎
        engine = EnhancedPaperTradingEngine(
            account_id="integration_test",
            initial_balance=Decimal("100000.0"),
            trading_params=TradingParameters(
                base_slippage_rate=Decimal("0.0001"),
                maker_fee_rate=Decimal("0.0002"),
                taker_fee_rate=Decimal("0.0004")
            )
        )
        await engine.initialize()
        
        # 创建市场模拟器
        market_simulator = IntegratedMarketSimulator()
        
        # 创建性能追踪器
        performance_tracker = PerformanceTracker(initial_balance=Decimal("100000.0"))
        
        return {
            "engine": engine,
            "market_simulator": market_simulator,
            "performance_tracker": performance_tracker
        }
    
    async def test_complete_trading_workflow(self, trading_system):
        """测试完整的交易流程"""
        engine = trading_system["engine"]
        market_simulator = trading_system["market_simulator"]
        performance_tracker = trading_system["performance_tracker"]
        
        # 1. 设置市场条件
        symbol = "BTCUSDT"
        base_price = Decimal("50000.0")
        market_simulator.update_market_condition(symbol, MarketCondition.NORMAL)
        
        # 2. 创建并执行买单
        buy_order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        buy_result = await engine.execute_order(buy_order)
        assert buy_result["status"] == "FILLED"
        
        # 3. 使用市场模拟器模拟执行
        sim_result = await market_simulator.simulate_order_execution(
            buy_order, base_price, user_id="integration_test"
        )
        
        assert "execution_price" in sim_result
        assert "commission" in sim_result
        assert "slippage_breakdown" in sim_result
        
        # 4. 记录交易到性能追踪器
        trade_record = TradeRecord(
            trade_id="integration_trade_001",
            order_id=buy_order.order_id,
            client_order_id=buy_order.client_order_id,
            symbol=symbol,
            side="BUY",
            quantity=Decimal(str(buy_order.quantity)),
            entry_price=Decimal(str(buy_order.avg_price)),
            entry_commission=Decimal(str(sim_result["commission"])),
            strategy_name="integration_test_strategy"
        )
        
        performance_tracker.record_trade(trade_record)
        
        # 5. 验证系统状态
        account_info = engine.get_enhanced_account_info()
        assert account_info["positions"] == 1
        
        performance_metrics = performance_tracker.calculate_performance_metrics()
        assert performance_metrics.total_trades == 1
        
        # 6. 更新市场价格并检查盈亏
        new_price = base_price * Decimal("1.05")  # 价格上涨5%
        market_data = MarketData(
            symbol=symbol, 
            price=float(new_price), 
            volume=1000.0,
            bid=float(new_price * Decimal("0.999")),
            ask=float(new_price * Decimal("1.001")),
            bid_volume=500.0,
            ask_volume=500.0,
            timestamp=datetime.utcnow()
        )
        await engine.update_market_data(market_data)
        
        # 检查仓位盈亏
        positions = engine.positions
        assert symbol in positions
        position = positions[symbol]
        assert position.unrealized_pnl > 0  # 应该有浮盈
        
        # 7. 平仓
        sell_order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        sell_result = await engine.execute_order(sell_order)
        assert sell_result["status"] == "FILLED"
        
        # 8. 验证最终状态
        final_account = engine.get_enhanced_account_info()
        assert final_account["positions"] == 0  # 仓位已平
        
    async def test_multiple_strategy_performance_tracking(self, trading_system):
        """测试多策略性能追踪"""
        performance_tracker = trading_system["performance_tracker"]
        
        # 策略A的交易
        strategy_a_trades = [
            TradeRecord("ta1", "oa1", "ca1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), Decimal("52000"), strategy_name="strategy_a", 
                       net_pnl=Decimal("1900")),
            TradeRecord("ta2", "oa2", "ca2", "ETHUSDT", "BUY", Decimal("10.0"), 
                       Decimal("3000"), Decimal("3100"), strategy_name="strategy_a",
                       net_pnl=Decimal("950"))
        ]
        
        # 策略B的交易
        strategy_b_trades = [
            TradeRecord("tb1", "ob1", "cb1", "BTCUSDT", "SELL", Decimal("0.5"), 
                       Decimal("51000"), Decimal("49000"), strategy_name="strategy_b",
                       net_pnl=Decimal("975")),
            TradeRecord("tb2", "ob2", "cb2", "ADAUSDT", "BUY", Decimal("1000.0"), 
                       Decimal("1.5"), Decimal("1.4"), strategy_name="strategy_b",
                       net_pnl=Decimal("-110"))
        ]
        
        # 记录所有交易
        for trade in strategy_a_trades + strategy_b_trades:
            performance_tracker.record_trade(trade)
        
        # 分析策略A性能
        metrics_a = performance_tracker.calculate_performance_metrics(strategy_name="strategy_a")
        assert metrics_a.total_trades == 2
        assert metrics_a.winning_trades == 2
        assert metrics_a.total_pnl == Decimal("2850")
        
        # 分析策略B性能
        metrics_b = performance_tracker.calculate_performance_metrics(strategy_name="strategy_b")
        assert metrics_b.total_trades == 2
        assert metrics_b.winning_trades == 1
        assert metrics_b.losing_trades == 1
        assert metrics_b.total_pnl == Decimal("865")
        
        # 生成性能报告
        report = performance_tracker.generate_performance_report()
        assert "strategy_analysis" in report
        assert "strategy_a" in report["strategy_analysis"]
        assert "strategy_b" in report["strategy_analysis"]
        
    async def test_risk_management_integration(self, trading_system):
        """测试风险管理集成"""
        engine = trading_system["engine"]
        
        # 设置较低的风险限制
        engine.account.max_daily_loss = Decimal("500.0")
        engine.account.max_position_size = Decimal("25000.0")  # $25k限制
        
        # 尝试超出仓位限制的订单
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0  # 按50k价格，会超出25k限制
        )
        
        result = await engine.execute_order(large_order)
        assert result["status"] == "REJECTED"
        assert "Position size exceeds maximum limit" in result["reason"]
        
        # 模拟日亏损累积
        engine.account.daily_loss = Decimal("450.0")
        
        normal_order = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # 这个订单应该还能执行
        result2 = await engine.execute_order(normal_order)
        assert result2["status"] in ["FILLED", "NEW"]
        
        # 设置日亏损接近限制
        engine.account.daily_loss = Decimal("500.0")
        
        another_order = Order(
            symbol="BNBUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        result3 = await engine.execute_order(another_order)
        assert result3["status"] == "REJECTED"
        assert "Daily loss limit exceeded" in result3["reason"]
        
    async def test_market_condition_impact(self, trading_system):
        """测试市场条件对交易的影响"""
        market_simulator = trading_system["market_simulator"]
        
        base_price = Decimal("50000.0")
        
        # 创建相同的订单
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5.0  # 较大订单以体现差异
        )
        
        # 正常市场条件
        market_simulator.update_market_condition("BTCUSDT", MarketCondition.NORMAL)
        normal_result = await market_simulator.simulate_order_execution(order, base_price)
        
        # 高波动市场条件
        market_simulator.update_market_condition("BTCUSDT", MarketCondition.VOLATILE)
        volatile_result = await market_simulator.simulate_order_execution(order, base_price)
        
        # 低流动性市场条件
        market_simulator.update_market_condition("BTCUSDT", MarketCondition.LOW_LIQUIDITY)
        low_liquidity_result = await market_simulator.simulate_order_execution(order, base_price)
        
        # 验证不同市场条件下的差异
        normal_slippage = normal_result["slippage_breakdown"]["total_slippage"]
        volatile_slippage = volatile_result["slippage_breakdown"]["total_slippage"]
        low_liquidity_slippage = low_liquidity_result["slippage_breakdown"]["total_slippage"]
        
        # 高波动和低流动性条件下滑点应该更大
        assert abs(volatile_slippage) >= abs(normal_slippage)
        assert abs(low_liquidity_slippage) >= abs(normal_slippage)
        
    async def test_commission_tier_progression(self, trading_system):
        """测试手续费等级递进"""
        market_simulator = trading_system["market_simulator"]
        
        base_price = Decimal("50000.0")
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,  # Maker订单
            quantity=1.0,
            price=49950.0  # 低于市价的买单
        )
        
        # VIP 0用户
        market_simulator.update_user_trading_volume("user_vip0", Decimal("0"))
        result_vip0 = await market_simulator.simulate_order_execution(order, base_price, "user_vip0")
        
        # VIP 2用户 
        market_simulator.update_user_trading_volume("user_vip2", Decimal("2500"))
        result_vip2 = await market_simulator.simulate_order_execution(order, base_price, "user_vip2")
        
        # VIP等级更高的用户手续费应该更低
        vip0_commission = result_vip0["commission"]
        vip2_commission = result_vip2["commission"]
        
        assert vip2_commission < vip0_commission
        
        # 检查费率信息
        vip0_breakdown = result_vip0["commission_breakdown"]
        vip2_breakdown = result_vip2["commission_breakdown"]
        
        assert vip0_breakdown["fee_tier"] == "VIP 0"
        assert vip2_breakdown["fee_tier"] == "VIP 2"
        
    async def test_bnb_discount_functionality(self, trading_system):
        """测试BNB抵扣功能"""
        market_simulator = trading_system["market_simulator"]
        
        base_price = Decimal("50000.0")
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # 不使用BNB抵扣
        result_no_bnb = await market_simulator.simulate_order_execution(order, base_price, "test_user")
        
        # 使用BNB抵扣
        market_simulator.configure_bnb_discount(Decimal("100"), enabled=True)
        result_with_bnb = await market_simulator.simulate_order_execution(order, base_price, "test_user")
        
        # 使用BNB抵扣后手续费应该更低
        no_bnb_commission = result_no_bnb["commission"]
        with_bnb_commission = result_with_bnb["commission"]
        
        assert with_bnb_commission < no_bnb_commission
        
        # 检查BNB抵扣明细
        bnb_breakdown = result_with_bnb["commission_breakdown"]
        assert bnb_breakdown["bnb_discount"] > 0
        
    async def test_performance_benchmark_comparison(self, trading_system):
        """测试性能基准比较"""
        performance_tracker = trading_system["performance_tracker"]
        
        # 创建一系列交易
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("2000")),  # +4%
            TradeRecord("t2", "o2", "c2", "ETHUSDT", "BUY", Decimal("10.0"), 
                       Decimal("3000"), net_pnl=Decimal("1500")),   # +5%
            TradeRecord("t3", "o3", "c3", "BTCUSDT", "SELL", Decimal("0.5"), 
                       Decimal("51000"), net_pnl=Decimal("-200")),  # -0.8%
        ]
        
        for trade in trades:
            performance_tracker.record_trade(trade)
        
        # 基准收益（如BTC指数收益）
        benchmark_returns = [
            Decimal("0.03"),   # 3%
            Decimal("0.04"),   # 4%
            Decimal("-0.01")   # -1%
        ]
        
        comparison = performance_tracker.compare_to_benchmark(benchmark_returns)
        
        # 验证基准比较结果
        assert "portfolio_return" in comparison
        assert "benchmark_return" in comparison
        assert "alpha" in comparison
        assert "beta" in comparison
        assert "information_ratio" in comparison
        
        # 组合收益应该是各笔交易收益的加权平均
        portfolio_return = comparison["portfolio_return"]
        benchmark_return = comparison["benchmark_return"]
        
        assert portfolio_return != 0
        assert benchmark_return != 0
        
    async def test_stress_testing_scenario(self, trading_system):
        """测试压力测试场景"""
        engine = trading_system["engine"]
        market_simulator = trading_system["market_simulator"]
        performance_tracker = trading_system["performance_tracker"]
        
        # 设置极端市场条件
        market_simulator.update_market_condition("BTCUSDT", MarketCondition.VOLATILE)
        
        # 快速执行多笔交易
        orders = []
        for i in range(10):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.1
            )
            orders.append(order)
        
        # 并发执行订单
        results = []
        for order in orders:
            result = await engine.execute_order(order)
            results.append(result)
            
            # 模拟市场数据更新
            price_change = Decimal("1.02") if order.side == OrderSide.BUY else Decimal("0.98")
            new_price = engine.market_prices.get("BTCUSDT", Decimal("50000")) * price_change
            market_data = MarketData(
                symbol="BTCUSDT", 
                price=float(new_price),
                volume=1000.0,
                bid=float(new_price * Decimal("0.999")),
                ask=float(new_price * Decimal("1.001")),
                bid_volume=500.0,
                ask_volume=500.0,
                timestamp=datetime.utcnow()
            )
            await engine.update_market_data(market_data)
        
        # 验证所有订单都得到处理
        filled_orders = [r for r in results if r["status"] == "FILLED"]
        assert len(filled_orders) > 0
        
        # 检查账户状态稳定性
        account_info = engine.get_enhanced_account_info()
        assert account_info["balance"] > 0
        assert abs(account_info["totalEquity"]) < 200000  # 权益在合理范围内
        
    async def test_data_consistency_across_modules(self, trading_system):
        """测试模块间数据一致性"""
        engine = trading_system["engine"]
        market_simulator = trading_system["market_simulator"]
        performance_tracker = trading_system["performance_tracker"]
        
        # 执行交易
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        engine_result = await engine.execute_order(order)
        sim_result = await market_simulator.simulate_order_execution(order, Decimal("50000"))
        
        # 创建性能记录
        trade_record = TradeRecord(
            trade_id="consistency_test",
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side="BUY",
            quantity=Decimal(str(order.quantity)),
            entry_price=Decimal(str(order.avg_price)),
            entry_commission=Decimal(str(sim_result["commission"]))
        )
        performance_tracker.record_trade(trade_record)
        
        # 验证数据一致性
        
        # 1. 订单数量一致性
        assert float(trade_record.quantity) == order.executed_qty
        
        # 2. 价格一致性（允许小误差）
        engine_price = Decimal(str(order.avg_price))
        sim_price = sim_result["execution_price"]
        price_diff = abs(engine_price - sim_price) / engine_price
        assert price_diff < Decimal("0.1")  # 价格差异小于10%
        
        # 3. 手续费计算一致性（基于相同费率）
        engine_volume = Decimal(str(order.executed_qty)) * Decimal(str(order.avg_price))
        expected_commission = engine_volume * engine.trading_params.taker_fee_rate
        
        # 验证引擎计算的手续费合理
        assert Decimal(str(sim_result["commission"])) > Decimal("0")
        
        # 4. 账户余额变化一致性
        initial_balance = Decimal("100000.0")
        current_balance = engine.account.current_balance
        commission_paid = engine.account.total_commission
        
        # 余额变化应该等于手续费支出（不考虑盈亏的情况下）
        balance_change = initial_balance - current_balance
        assert balance_change >= commission_paid  # 至少应该扣除手续费


if __name__ == "__main__":
    pytest.main([__file__, "-v"])