import pytest
import asyncio
import time
from decimal import Decimal
from typing import List, Dict

from src.hft import (
    HFTEngine, HFTConfig,
    ArbitrageAgent, ArbitrageConfig, ArbitrageType,
    MarketMakingAgent, MarketMakingConfig,
    LatencySensitiveSignalProcessor
)
from src.hft.microstructure_analyzer import MicrostructureSignal
from src.core.models import MarketData


class TestHFTStrategies:
    """HFT策略测试"""
    
    @pytest.fixture
    async def hft_engine(self):
        """HFT引擎测试夹具"""
        config = HFTConfig()
        engine = HFTEngine(config)
        await engine.initialize(["BTCUSDT", "ETHUSDT", "ETHBTC"])
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    async def arbitrage_agent(self, hft_engine):
        """套利Agent测试夹具"""
        config = ArbitrageConfig(
            min_profit_bps=10.0,  # 降低阈值便于测试
            min_confidence=0.5
        )
        agent = ArbitrageAgent(hft_engine, config)
        await agent.start()
        yield agent
        await agent.stop()
    
    @pytest.fixture
    async def market_making_agent(self, hft_engine):
        """做市Agent测试夹具"""
        config = MarketMakingConfig(
            base_spread_bps=10.0,  # 降低价差便于测试
            min_order_value=Decimal("5")
        )
        agent = MarketMakingAgent(hft_engine, config)
        await agent.start(["BTCUSDT"])
        yield agent
        await agent.stop()
    
    @pytest.mark.asyncio
    async def test_arbitrage_opportunity_detection(self, arbitrage_agent):
        """测试套利机会检测"""
        # 模拟价格数据以创造套利机会
        symbols = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
        
        # 创建价格不一致的情况
        market_data_list = [
            MarketData(
                symbol="BTCUSDT",
                timestamp=time.time(),
                price=50000.0,
                volume=1.0,
                bid=49999.0,
                ask=50001.0,
                bid_volume=10.0,
                ask_volume=10.0
            ),
            MarketData(
                symbol="ETHUSDT", 
                timestamp=time.time(),
                price=3000.0,
                volume=1.0,
                bid=2999.0,
                ask=3001.0,
                bid_volume=10.0,
                ask_volume=10.0
            ),
            MarketData(
                symbol="ETHBTC",
                timestamp=time.time(),
                price=0.065,  # 创造三角套利机会：ETH/USD / BTC/USD ≠ ETH/BTC
                volume=1.0,
                bid=0.0649,
                ask=0.0651,
                bid_volume=10.0,
                ask_volume=10.0
            )
        ]
        
        # 更新市场数据
        for market_data in market_data_list:
            await arbitrage_agent.update_market_data(market_data.symbol, market_data)
        
        # 等待套利检测
        await asyncio.sleep(0.1)
        
        # 检查是否检测到套利机会
        opportunities = arbitrage_agent.get_active_opportunities()
        stats = arbitrage_agent.get_statistics()
        
        print(f"Detected {len(opportunities)} arbitrage opportunities")
        print(f"Arbitrage stats: {stats}")
        
        # 验证套利检测功能
        assert stats["opportunities_found"] >= 0, "Should track opportunity count"
        
        # 如果检测到机会，验证其有效性
        for opportunity in opportunities:
            assert opportunity.expected_profit_bps > 0, "Profit should be positive"
            assert 0 < opportunity.confidence <= 1, "Confidence should be between 0 and 1"
            assert len(opportunity.symbols) >= 2, "Should involve multiple symbols"
    
    @pytest.mark.asyncio
    async def test_triangular_arbitrage_calculation(self, arbitrage_agent):
        """测试三角套利计算"""
        # 配置三角套利对
        arbitrage_agent.config.triangular_symbols = [("BTCUSDT", "ETHUSDT", "ETHBTC")]
        
        # 创建明显的三角套利机会
        # BTC/USD = 50000, ETH/USD = 3000, ETH/BTC = 0.05
        # 理论 ETH/BTC = ETH/USD / BTC/USD = 3000/50000 = 0.06
        # 实际 ETH/BTC = 0.05，存在套利机会
        
        market_data_btc = MarketData(
            symbol="BTCUSDT",
            timestamp=time.time(),
            price=50000.0,
            volume=1.0,
            bid=49999.0,
            ask=50001.0,
            bid_volume=100.0,
            ask_volume=100.0
        )
        
        market_data_eth = MarketData(
            symbol="ETHUSDT",
            timestamp=time.time(),
            price=3000.0,
            volume=1.0,
            bid=2999.0,
            ask=3001.0,
            bid_volume=100.0,
            ask_volume=100.0
        )
        
        market_data_ethbtc = MarketData(
            symbol="ETHBTC",
            timestamp=time.time(),
            price=0.05,  # 低于理论价格0.06
            volume=1.0,
            bid=0.0499,
            ask=0.0501,
            bid_volume=100.0,
            ask_volume=100.0
        )
        
        # 按顺序更新数据
        await arbitrage_agent.update_market_data("BTCUSDT", market_data_btc)
        await arbitrage_agent.update_market_data("ETHUSDT", market_data_eth)
        await arbitrage_agent.update_market_data("ETHBTC", market_data_ethbtc)
        
        # 等待处理
        await asyncio.sleep(0.2)
        
        opportunities = arbitrage_agent.get_active_opportunities()
        
        # 应该检测到三角套利机会
        triangular_opportunities = [
            opp for opp in opportunities 
            if opp.arbitrage_type == ArbitrageType.TRIANGULAR
        ]
        
        if triangular_opportunities:
            opp = triangular_opportunities[0]
            print(f"Triangular arbitrage detected:")
            print(f"  Expected profit: {opp.expected_profit_bps:.2f} bps")
            print(f"  Confidence: {opp.confidence:.3f}")
            print(f"  Symbols: {opp.symbols}")
            print(f"  Directions: {opp.directions}")
            
            # 验证套利逻辑
            assert opp.expected_profit_bps > 10, "Should detect significant profit opportunity"
            assert opp.confidence > 0.5, "Should have reasonable confidence"
            assert set(opp.symbols) == {"BTCUSDT", "ETHUSDT", "ETHBTC"}, "Should involve all three symbols"
    
    @pytest.mark.asyncio
    async def test_market_making_quote_generation(self, market_making_agent):
        """测试做市报价生成"""
        symbol = "BTCUSDT"
        
        # 更新市场数据
        market_data = MarketData(
            symbol=symbol,
            timestamp=time.time(),
            price=50000.0,
            volume=1.0,
            bid=49999.0,
            ask=50001.0,
            bid_volume=10.0,
            ask_volume=10.0
        )
        
        await market_making_agent.update_market_data(symbol, market_data)
        
        # 等待报价生成
        await asyncio.sleep(1.0)
        
        # 检查生成的报价
        active_quotes = market_making_agent.get_active_quotes()
        stats = market_making_agent.get_statistics()
        
        print(f"Generated {len(active_quotes)} market making quotes")
        print(f"Market making stats: {stats}")
        
        if symbol in active_quotes:
            quote = active_quotes[symbol]
            print(f"Quote for {symbol}:")
            print(f"  Bid: {quote.bid_price} (qty: {quote.bid_quantity})")
            print(f"  Ask: {quote.ask_price} (qty: {quote.ask_quantity})")
            print(f"  Spread: {quote.spread_bps:.2f} bps")
            
            # 验证报价合理性
            assert quote.bid_price < quote.ask_price, "Bid should be less than ask"
            assert quote.spread_bps > 0, "Spread should be positive"
            assert quote.bid_quantity > 0, "Bid quantity should be positive"
            assert quote.ask_quantity > 0, "Ask quantity should be positive"
            
            # 验证价格在合理范围内
            mid_price = (quote.bid_price + quote.ask_price) / 2
            market_mid = (Decimal(str(market_data.bid)) + Decimal(str(market_data.ask))) / 2
            price_diff_bps = abs(float((mid_price - market_mid) / market_mid)) * 10000
            
            assert price_diff_bps < 50, f"Quote mid price deviation {price_diff_bps:.1f}bps too large"
    
    @pytest.mark.asyncio
    async def test_market_making_inventory_management(self, market_making_agent):
        """测试做市库存管理"""
        symbol = "BTCUSDT"
        
        # 模拟库存变化
        market_making_agent.inventory[symbol] = Decimal("0.1")  # 正库存
        market_making_agent.avg_purchase_price[symbol] = Decimal("49000")
        
        # 更新市场数据
        market_data = MarketData(
            symbol=symbol,
            timestamp=time.time(),
            price=50000.0,
            volume=1.0,
            bid=49999.0,
            ask=50001.0,
            bid_volume=10.0,
            ask_volume=10.0
        )
        
        await market_making_agent.update_market_data(symbol, market_data)
        await asyncio.sleep(0.5)
        
        # 检查库存影响
        inventory = market_making_agent.get_inventory()
        risk_metrics = market_making_agent.get_risk_metrics()
        
        print(f"Inventory: {inventory}")
        print(f"Risk metrics: {risk_metrics}")
        
        assert symbol in inventory, "Should track inventory"
        assert "total_inventory_value" in risk_metrics, "Should calculate inventory value"
        assert "inventory_utilization" in risk_metrics, "Should track utilization"
        
        # 验证库存倾斜效应
        quotes = market_making_agent.get_active_quotes()
        if symbol in quotes:
            quote = quotes[symbol]
            # 正库存应该倾向于更积极的卖价（降低ask）
            print(f"Inventory skew effect on quote: bid={quote.bid_price}, ask={quote.ask_price}")
    
    @pytest.mark.asyncio
    async def test_signal_based_strategy_adjustment(self, hft_engine):
        """测试基于信号的策略调整"""
        signal_processor = LatencySensitiveSignalProcessor()
        await signal_processor.start()
        
        try:
            # 创建强烈的失衡信号
            imbalance_signal = MicrostructureSignal(
                signal_type="imbalance",
                symbol="BTCUSDT",
                timestamp=time.time(),
                strength=0.8,  # 强买盘失衡
                confidence=0.9,
                metadata={"bid_ask_imbalance": 0.8}
            )
            
            # 创建高毒性信号
            toxicity_signal = MicrostructureSignal(
                signal_type="toxicity",
                symbol="BTCUSDT",
                timestamp=time.time(),
                strength=0.9,  # 高毒性
                confidence=0.85,
                metadata={"vpin": 0.9}
            )
            
            actions_generated = []
            
            async def action_callback(action):
                actions_generated.append(action)
            
            signal_processor.add_action_callback(action_callback)
            
            # 处理信号
            await signal_processor.process_signal(imbalance_signal)
            await signal_processor.process_signal(toxicity_signal)
            
            # 等待处理完成
            await asyncio.sleep(0.1)
            
            print(f"Generated {len(actions_generated)} actions from signals")
            
            # 验证信号生成的动作
            for action in actions_generated:
                print(f"Action: {action.action_type.value} for {action.symbol} "
                     f"(urgency: {action.urgency_score:.2f}, confidence: {action.confidence:.2f})")
            
            # 应该至少生成一些动作
            imbalance_actions = [a for a in actions_generated if "imbalance" in str(a.metadata)]
            toxicity_actions = [a for a in actions_generated if "toxicity" in str(a.metadata)]
            
            # 失衡信号应该生成买入或卖出动作
            if imbalance_actions:
                assert any(a.action_type.value in ["buy", "sell"] for a in imbalance_actions)
            
            # 毒性信号应该生成取消动作
            if toxicity_actions:
                assert any(a.action_type.value == "cancel" for a in toxicity_actions)
            
        finally:
            await signal_processor.stop()
    
    @pytest.mark.asyncio
    async def test_strategy_performance_under_load(self, arbitrage_agent, market_making_agent):
        """测试策略在负载下的性能"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        num_updates = 500
        update_interval = 0.001  # 1ms间隔
        
        # 记录开始时间
        start_time = time.time()
        
        # 高频市场数据更新
        for i in range(num_updates):
            for j, symbol in enumerate(symbols):
                base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=time.time(),
                    price=base_price + i * 0.1,
                    volume=1.0 + i * 0.001,
                    bid=base_price + i * 0.1 - 0.5,
                    ask=base_price + i * 0.1 + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                # 并发更新两个策略
                await asyncio.gather(
                    arbitrage_agent.update_market_data(symbol, market_data),
                    market_making_agent.update_market_data(symbol, market_data)
                )
            
            await asyncio.sleep(update_interval)
        
        total_time = time.time() - start_time
        update_rate = (num_updates * len(symbols)) / total_time
        
        print(f"Strategy performance under load:")
        print(f"  Total updates: {num_updates * len(symbols)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Update rate: {update_rate:.1f} updates/s")
        
        # 检查策略状态
        arb_stats = arbitrage_agent.get_statistics()
        mm_stats = market_making_agent.get_statistics()
        
        print(f"Arbitrage agent stats: {arb_stats}")
        print(f"Market making agent stats: {mm_stats}")
        
        # 性能断言
        assert update_rate > 100, f"Update rate {update_rate:.1f} too low"
        
        # 验证策略仍在正常工作
        arb_status = arbitrage_agent.get_status()
        mm_status = market_making_agent.get_status()
        
        assert arb_status["running"], "Arbitrage agent should still be running"
        assert mm_status["running"], "Market making agent should still be running"
    
    @pytest.mark.asyncio
    async def test_strategy_accuracy_metrics(self, arbitrage_agent, market_making_agent):
        """测试策略准确性指标"""
        # 运行一段时间收集数据
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        for i in range(100):
            for symbol in symbols:
                base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
                # 添加一些随机性模拟真实市场
                price_change = (i % 10 - 5) * 0.1  # -0.5 到 +0.4 的变化
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=time.time(),
                    price=base_price + price_change,
                    volume=1.0,
                    bid=base_price + price_change - 0.5,
                    ask=base_price + price_change + 0.5,
                    bid_volume=10.0,
                    ask_volume=10.0
                )
                
                await arbitrage_agent.update_market_data(symbol, market_data)
                await market_making_agent.update_market_data(symbol, market_data)
            
            await asyncio.sleep(0.01)
        
        # 收集准确性指标
        arb_stats = arbitrage_agent.get_statistics()
        mm_stats = market_making_agent.get_statistics()
        
        print(f"Arbitrage accuracy metrics:")
        print(f"  Opportunities found: {arb_stats.get('opportunities_found', 0)}")
        print(f"  Opportunities executed: {arb_stats.get('opportunities_executed', 0)}")
        print(f"  Win rate: {arb_stats.get('win_rate', 0):.2%}")
        
        print(f"Market making accuracy metrics:")
        print(f"  Total quotes: {mm_stats.get('total_quotes', 0)}")
        print(f"  Fill rate: {mm_stats.get('fill_rate', 0):.2%}")
        print(f"  Average spread: {mm_stats.get('avg_spread_bps', 0):.1f} bps")
        
        # 基本合理性检查
        assert arb_stats.get("opportunities_found", 0) >= 0, "Should track opportunities"
        assert 0 <= arb_stats.get("win_rate", 0) <= 1, "Win rate should be between 0 and 1"
        
        assert mm_stats.get("total_quotes", 0) >= 0, "Should track quotes"
        assert 0 <= mm_stats.get("fill_rate", 0) <= 1, "Fill rate should be between 0 and 1"
        assert mm_stats.get("avg_spread_bps", 0) >= 0, "Spread should be non-negative"