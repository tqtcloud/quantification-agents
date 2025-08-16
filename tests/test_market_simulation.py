import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch

from src.trading.market_simulation import (
    RealTimeSlippageCalculator,
    BinanceCommissionCalculator,
    MarketDelaySimulator,
    MarketMicrostructureSimulator,
    IntegratedMarketSimulator,
    SlippageConfig,
    CommissionStructure,
    MarketCondition,
    OrderBookSnapshot,
    MarketDepthLevel
)
from src.core.models import Order, OrderSide, OrderType


class TestSlippageConfig:
    """测试滑点配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SlippageConfig()
        
        assert config.base_slippage_bps == 1
        assert config.impact_coefficient == Decimal("0.1")
        assert config.impact_exponent == Decimal("0.6")
        assert config.volatility_multiplier == Decimal("2.0")
        assert config.avg_depth_usd == Decimal("100000")


class TestCommissionStructure:
    """测试手续费结构"""
    
    def test_default_commission_structure(self):
        """测试默认手续费结构"""
        structure = CommissionStructure()
        
        # 检查VIP等级费率
        assert len(structure.maker_tiers) == 5
        assert len(structure.taker_tiers) == 5
        
        # 检查VIP 0费率
        assert structure.maker_tiers[0] == (Decimal("0"), Decimal("0.0002"))
        assert structure.taker_tiers[0] == (Decimal("0"), Decimal("0.0004"))
        
        # 检查BNB折扣
        assert structure.bnb_discount == Decimal("0.1")


class TestOrderBookSnapshot:
    """测试订单簿快照"""
    
    def test_orderbook_initialization(self):
        """测试订单簿初始化"""
        bids = [
            MarketDepthLevel(Decimal("50000"), Decimal("1.0"), Decimal("1.0")),
            MarketDepthLevel(Decimal("49990"), Decimal("2.0"), Decimal("3.0"))
        ]
        asks = [
            MarketDepthLevel(Decimal("50010"), Decimal("1.5"), Decimal("1.5")),
            MarketDepthLevel(Decimal("50020"), Decimal("1.8"), Decimal("3.3"))
        ]
        
        orderbook = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow()
        )
        
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        
    def test_spread_calculation(self):
        """测试价差计算"""
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("1.0"), Decimal("1.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("1.5"), Decimal("1.5"))]
        
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        assert orderbook.get_spread() == Decimal("10")
        
    def test_mid_price_calculation(self):
        """测试中间价计算"""
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("1.0"), Decimal("1.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("1.5"), Decimal("1.5"))]
        
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        expected_mid = (Decimal("50000") + Decimal("50010")) / Decimal("2")
        assert orderbook.get_mid_price() == expected_mid
        
    def test_depth_at_price(self):
        """测试指定价格深度查询"""
        bids = [
            MarketDepthLevel(Decimal("50000"), Decimal("1.0"), Decimal("1.0")),
            MarketDepthLevel(Decimal("49990"), Decimal("2.0"), Decimal("3.0"))
        ]
        
        orderbook = OrderBookSnapshot("BTCUSDT", bids, [], datetime.utcnow())
        
        # 查询价格50000的买单深度（只包含价格>=50000的）
        depth = orderbook.get_depth_at_price(Decimal("50000"), "BUY")
        assert depth == Decimal("1.0")
        
        # 查询价格49995的买单深度（包含价格>=49995的）
        depth_lower = orderbook.get_depth_at_price(Decimal("49995"), "BUY")
        assert depth_lower == Decimal("1.0")  # 只有50000价格的档位符合
        
        # 查询价格49985的买单深度（所有档位都符合价格>=49985）
        depth_all = orderbook.get_depth_at_price(Decimal("49985"), "BUY")
        assert depth_all == Decimal("3.0")


class TestRealTimeSlippageCalculator:
    """测试实时滑点计算器"""
    
    def test_slippage_calculator_initialization(self):
        """测试滑点计算器初始化"""
        config = SlippageConfig()
        calculator = RealTimeSlippageCalculator(config)
        
        assert calculator.config == config
        assert isinstance(calculator.market_conditions, dict)
        assert isinstance(calculator.recent_trades, dict)
        assert isinstance(calculator.volatility_cache, dict)
        
    def test_basic_slippage_calculation(self):
        """测试基础滑点计算"""
        config = SlippageConfig(base_slippage_bps=2)
        calculator = RealTimeSlippageCalculator(config)
        
        # 创建订单和订单簿
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("10.0"), Decimal("10.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("10.0"), Decimal("10.0"))]
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        slippage_rate, breakdown = calculator.calculate_slippage(
            order, orderbook, MarketCondition.NORMAL
        )
        
        assert isinstance(slippage_rate, Decimal)
        assert "base_slippage" in breakdown
        assert "market_impact" in breakdown
        assert "total_slippage" in breakdown
        assert breakdown["market_condition"] == "normal"
        
    def test_market_impact_calculation(self):
        """测试市场冲击计算"""
        config = SlippageConfig(
            impact_coefficient=Decimal("0.1"),
            avg_depth_usd=Decimal("100000")
        )
        calculator = RealTimeSlippageCalculator(config)
        
        # 小订单
        small_order_size = Decimal("5000")  # $5k
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("10.0"), Decimal("10.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("10.0"), Decimal("10.0"))]
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        small_impact = calculator._calculate_market_impact(small_order_size, orderbook)
        
        # 大订单
        large_order_size = Decimal("50000")  # $50k
        large_impact = calculator._calculate_market_impact(large_order_size, orderbook)
        
        # 大订单的市场冲击应该更大
        assert large_impact > small_impact
        
    def test_volatility_adjustment(self):
        """测试波动性调整"""
        config = SlippageConfig(volatility_multiplier=Decimal("2.0"))
        calculator = RealTimeSlippageCalculator(config)
        
        # 正常市场条件
        normal_adjustment = calculator._calculate_volatility_adjustment(
            "BTCUSDT", MarketCondition.NORMAL
        )
        
        # 高波动市场条件
        volatile_adjustment = calculator._calculate_volatility_adjustment(
            "BTCUSDT", MarketCondition.VOLATILE
        )
        
        # 高波动条件下的调整应该更大
        assert volatile_adjustment > normal_adjustment
        
    def test_sell_order_slippage(self):
        """测试卖单滑点"""
        config = SlippageConfig()
        calculator = RealTimeSlippageCalculator(config)
        
        sell_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("10.0"), Decimal("10.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("10.0"), Decimal("10.0"))]
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        slippage_rate, _ = calculator.calculate_slippage(
            sell_order, orderbook, MarketCondition.NORMAL
        )
        
        # 卖单滑点应该是负数
        assert slippage_rate < 0


class TestBinanceCommissionCalculator:
    """测试币安手续费计算器"""
    
    def test_commission_calculator_initialization(self):
        """测试手续费计算器初始化"""
        structure = CommissionStructure()
        calculator = BinanceCommissionCalculator(structure)
        
        assert calculator.structure == structure
        assert isinstance(calculator.user_30d_volume, dict)
        assert calculator.bnb_balance == Decimal("0")
        assert calculator.use_bnb_discount == False
        
    def test_basic_commission_calculation(self):
        """测试基础手续费计算"""
        structure = CommissionStructure()
        calculator = BinanceCommissionCalculator(structure)
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        execution_price = Decimal("50000")
        
        # Taker手续费
        commission, breakdown = calculator.calculate_commission(
            order, execution_price, is_maker=False, user_id="test_user"
        )
        
        expected_commission = Decimal("1.0") * execution_price * Decimal("0.0004")  # VIP 0 taker
        assert commission == expected_commission
        assert breakdown["is_maker"] == False
        assert breakdown["fee_tier"] == "VIP 0"
        
        # Maker手续费
        commission_maker, breakdown_maker = calculator.calculate_commission(
            order, execution_price, is_maker=True, user_id="test_user"
        )
        
        expected_commission_maker = Decimal("1.0") * execution_price * Decimal("0.0002")  # VIP 0 maker
        assert commission_maker == expected_commission_maker
        assert breakdown_maker["is_maker"] == True
        
    def test_vip_tier_calculation(self):
        """测试VIP等级计算"""
        structure = CommissionStructure()
        calculator = BinanceCommissionCalculator(structure)
        
        # 设置不同的30日交易量
        volumes_and_tiers = [
            (Decimal("0"), "VIP 0"),
            (Decimal("250"), "VIP 1"),
            (Decimal("2500"), "VIP 2"),
            (Decimal("7500"), "VIP 3"),
            (Decimal("22500"), "VIP 4")
        ]
        
        for volume, expected_tier in volumes_and_tiers:
            tier = calculator._get_fee_tier(volume)
            assert tier == expected_tier
            
    def test_bnb_discount(self):
        """测试BNB抵扣"""
        structure = CommissionStructure()
        calculator = BinanceCommissionCalculator(structure)
        
        # 启用BNB抵扣并设置余额
        calculator.enable_bnb_discount(True)
        calculator.set_bnb_balance(Decimal("100"))
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        execution_price = Decimal("50000")
        
        commission, breakdown = calculator.calculate_commission(
            order, execution_price, is_maker=False, user_id="test_user"
        )
        
        # 应该有BNB折扣
        assert breakdown["bnb_discount"] > 0
        assert breakdown["final_commission"] < breakdown["base_commission"]
        
    def test_insufficient_bnb_balance(self):
        """测试BNB余额不足"""
        structure = CommissionStructure()
        calculator = BinanceCommissionCalculator(structure)
        
        # 启用BNB抵扣但余额不足
        calculator.enable_bnb_discount(True)
        calculator.set_bnb_balance(Decimal("1"))  # 很少的BNB
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0  # 大订单
        )
        
        execution_price = Decimal("50000")
        
        commission, breakdown = calculator.calculate_commission(
            order, execution_price, is_maker=False, user_id="test_user"
        )
        
        # BNB余额不足，折扣有限
        assert breakdown["bnb_discount"] >= 0
        assert calculator.bnb_balance >= 0  # 余额不应为负


@pytest.mark.asyncio
class TestMarketDelaySimulator:
    """测试市场延迟模拟器"""
    
    def test_delay_simulator_initialization(self):
        """测试延迟模拟器初始化"""
        simulator = MarketDelaySimulator()
        
        assert simulator.network_latency_ms == 50
        assert simulator.exchange_processing_ms == 20
        assert simulator.order_queue_delay_ms == 10
        
    async def test_execution_delay_simulation(self):
        """测试执行延迟模拟"""
        simulator = MarketDelaySimulator()
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        start_time = asyncio.get_event_loop().time()
        delay = await simulator.simulate_execution_delay(order, MarketCondition.NORMAL)
        end_time = asyncio.get_event_loop().time()
        
        # 实际延迟应该接近返回的延迟值
        actual_delay = end_time - start_time
        assert abs(actual_delay - delay) < 0.01  # 允许小误差
        
    async def test_market_condition_impact(self):
        """测试市场条件对延迟的影响"""
        simulator = MarketDelaySimulator()
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        # 正常市场条件
        normal_delay = await simulator.simulate_execution_delay(order, MarketCondition.NORMAL)
        
        # 高成交量市场条件
        high_volume_delay = await simulator.simulate_execution_delay(order, MarketCondition.HIGH_VOLUME)
        
        # 高成交量时延迟应该更长（平均而言）
        # 注意：由于有随机因素，单次测试可能不满足，这里仅做基本检查
        assert normal_delay > 0
        assert high_volume_delay > 0


class TestMarketMicrostructureSimulator:
    """测试市场微观结构模拟器"""
    
    def test_microstructure_simulator_initialization(self):
        """测试微观结构模拟器初始化"""
        simulator = MarketMicrostructureSimulator()
        
        assert simulator.order_arrival_rate == 10
        assert simulator.market_maker_spread == Decimal("0.0001")
        
    def test_realistic_orderbook_generation(self):
        """测试真实订单簿生成"""
        simulator = MarketMicrostructureSimulator()
        
        mid_price = Decimal("50000")
        orderbook = simulator.generate_realistic_orderbook("BTCUSDT", mid_price, depth_levels=10)
        
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 10
        assert len(orderbook.asks) == 10
        
        # 检查价格序列正确性
        for i in range(len(orderbook.bids) - 1):
            assert orderbook.bids[i].price > orderbook.bids[i + 1].price
            
        for i in range(len(orderbook.asks) - 1):
            assert orderbook.asks[i].price < orderbook.asks[i + 1].price
            
        # 检查最佳买卖价
        best_bid = orderbook.bids[0].price
        best_ask = orderbook.asks[0].price
        assert best_bid < mid_price < best_ask
        
    def test_market_impact_simulation(self):
        """测试市场冲击模拟"""
        simulator = MarketMicrostructureSimulator()
        
        # 生成初始订单簿
        mid_price = Decimal("50000")
        initial_orderbook = simulator.generate_realistic_orderbook("BTCUSDT", mid_price)
        
        # 创建大买单
        buy_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0  # 大订单
        )
        
        # 模拟市场冲击
        impacted_orderbook = simulator.simulate_market_impact(buy_order, initial_orderbook)
        
        # 检查卖单深度是否减少
        initial_ask_qty = sum(level.quantity for level in initial_orderbook.asks[:3])
        impacted_ask_qty = sum(level.quantity for level in impacted_orderbook.asks[:3])
        
        assert impacted_ask_qty < initial_ask_qty  # 流动性被消耗


@pytest.mark.asyncio
class TestIntegratedMarketSimulator:
    """测试集成市场模拟器"""
    
    def test_integrated_simulator_initialization(self):
        """测试集成模拟器初始化"""
        simulator = IntegratedMarketSimulator()
        
        assert isinstance(simulator.slippage_calculator, RealTimeSlippageCalculator)
        assert isinstance(simulator.commission_calculator, BinanceCommissionCalculator)
        assert isinstance(simulator.delay_simulator, MarketDelaySimulator)
        assert isinstance(simulator.microstructure_simulator, MarketMicrostructureSimulator)
        
    async def test_integrated_order_execution_simulation(self):
        """测试集成订单执行模拟"""
        simulator = IntegratedMarketSimulator()
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        base_price = Decimal("50000")
        
        result = await simulator.simulate_order_execution(order, base_price, user_id="test_user")
        
        # 检查返回结果完整性
        required_keys = [
            "execution_price", "commission", "execution_delay",
            "slippage_breakdown", "commission_breakdown",
            "orderbook_before", "orderbook_after", "trade_info"
        ]
        
        for key in required_keys:
            assert key in result
            
        # 检查执行价格合理性
        assert result["execution_price"] > 0
        assert result["commission"] > 0
        assert result["execution_delay"] > 0
        
    def test_maker_status_determination(self):
        """测试maker状态判断"""
        simulator = IntegratedMarketSimulator()
        
        bids = [MarketDepthLevel(Decimal("50000"), Decimal("1.0"), Decimal("1.0"))]
        asks = [MarketDepthLevel(Decimal("50010"), Decimal("1.0"), Decimal("1.0"))]
        orderbook = OrderBookSnapshot("BTCUSDT", bids, asks, datetime.utcnow())
        
        # 市价单总是taker
        market_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        assert simulator._determine_maker_status(market_order, orderbook) == False
        
        # 限价买单低于最佳卖价是maker
        limit_buy_maker = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50005.0  # 低于最佳卖价50010
        )
        assert simulator._determine_maker_status(limit_buy_maker, orderbook) == True
        
        # 限价买单高于最佳卖价是taker
        limit_buy_taker = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50015.0  # 高于最佳卖价50010
        )
        assert simulator._determine_maker_status(limit_buy_taker, orderbook) == False
        
    def test_market_condition_update(self):
        """测试市场条件更新"""
        simulator = IntegratedMarketSimulator()
        
        # 更新市场条件
        simulator.update_market_condition("BTCUSDT", MarketCondition.VOLATILE)
        
        assert simulator.current_market_conditions["BTCUSDT"] == MarketCondition.VOLATILE
        assert simulator.slippage_calculator.market_conditions["BTCUSDT"] == MarketCondition.VOLATILE
        
    def test_user_trading_volume_update(self):
        """测试用户交易量更新"""
        simulator = IntegratedMarketSimulator()
        
        # 更新用户交易量
        simulator.update_user_trading_volume("test_user", Decimal("5000"))
        
        assert simulator.commission_calculator.user_30d_volume["test_user"] == Decimal("5000")
        
    def test_bnb_discount_configuration(self):
        """测试BNB抵扣配置"""
        simulator = IntegratedMarketSimulator()
        
        # 配置BNB抵扣
        simulator.configure_bnb_discount(Decimal("100"), enabled=True)
        
        assert simulator.commission_calculator.bnb_balance == Decimal("100")
        assert simulator.commission_calculator.use_bnb_discount == True
        
    def test_simulation_stats(self):
        """测试模拟统计信息"""
        simulator = IntegratedMarketSimulator()
        
        # 设置一些状态
        simulator.update_market_condition("BTCUSDT", MarketCondition.VOLATILE)
        simulator.configure_bnb_discount(Decimal("50"), enabled=True)
        
        stats = simulator.get_simulation_stats()
        
        assert "market_conditions" in stats
        assert "slippage_config" in stats
        assert "commission_config" in stats
        assert "delay_config" in stats
        
        assert stats["market_conditions"]["BTCUSDT"] == "volatile"
        assert stats["commission_config"]["bnb_discount_enabled"] == True
        assert stats["commission_config"]["bnb_balance"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])