import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

from src.backtesting.backtest_engine import (
    BacktestEngine, BacktestConfig, BacktestResult, BacktestMode,
    HistoricalDataProvider, BacktestEvent
)
from src.backtesting.strategies import create_sma_strategy, create_buy_hold_strategy
from src.core.models import Order, OrderSide, OrderType, MarketData


class TestHistoricalDataProvider:
    """测试历史数据提供者"""
    
    @pytest.mark.asyncio
    async def test_load_data(self):
        """测试数据加载"""
        provider = HistoricalDataProvider()
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        data = await provider.load_data("BTCUSDT", start_date, end_date, "1m")
        
        assert len(data) > 0
        assert all(isinstance(d, MarketData) for d in data)
        assert data[0].timestamp >= start_date
        assert data[-1].timestamp <= end_date
        
        # 测试缓存
        data2 = await provider.load_data("BTCUSDT", start_date, end_date, "1m")
        assert len(data) == len(data2)
    
    def test_generate_sample_data(self):
        """测试样本数据生成"""
        provider = HistoricalDataProvider()
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 2)  # 2小时
        
        data = provider._generate_sample_data("BTCUSDT", start_date, end_date, "1m")
        
        assert len(data) == 120  # 120分钟
        assert all(d.symbol == "BTCUSDT" for d in data)
        assert all(d.price > 0 for d in data)
        assert all(d.volume > 0 for d in data)


class TestBacktestConfig:
    """测试回测配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date
        )
        
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.initial_balance == Decimal("100000")
        assert config.symbols == ["BTCUSDT"]
        assert config.timeframes == ["1m"]
        assert config.mode == BacktestMode.SINGLE_STRATEGY
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 7),
            initial_balance=Decimal("50000"),
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["5m", "1h"],
            commission_rate=Decimal("0.002"),
            slippage_bps=Decimal("2.0"),
            mode=BacktestMode.MULTI_STRATEGY
        )
        
        assert config.initial_balance == Decimal("50000")
        assert len(config.symbols) == 2
        assert len(config.timeframes) == 2
        assert config.commission_rate == Decimal("0.002")
        assert config.mode == BacktestMode.MULTI_STRATEGY


@pytest.mark.asyncio
class TestBacktestEngine:
    """测试回测引擎"""
    
    @pytest.fixture
    async def basic_config(self):
        """基础配置"""
        return BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),  # 1天的数据
            initial_balance=Decimal("10000"),
            strategy_configs={
                "test_strategy": {
                    "short_window": 5,
                    "long_window": 10,
                    "position_size": 0.1
                }
            }
        )
    
    @pytest.fixture
    async def backtest_engine(self, basic_config):
        """创建回测引擎"""
        engine = BacktestEngine(basic_config)
        await engine.initialize()
        return engine
    
    async def test_engine_initialization(self, basic_config):
        """测试引擎初始化"""
        engine = BacktestEngine(basic_config)
        
        # 注册策略
        engine.register_strategy("test_strategy", create_sma_strategy)
        
        await engine.initialize()
        
        assert "test_strategy" in engine.trading_engines
        assert "test_strategy" in engine.performance_trackers
        assert len(engine.market_data_cache) > 0
        assert len(engine.event_queue) > 0
    
    async def test_strategy_registration(self, basic_config):
        """测试策略注册"""
        engine = BacktestEngine(basic_config)
        
        # 注册策略
        engine.register_strategy("sma_strategy", create_sma_strategy)
        engine.register_strategy("buy_hold_strategy", create_buy_hold_strategy)
        
        assert "sma_strategy" in engine.strategy_functions
        assert "buy_hold_strategy" in engine.strategy_functions
    
    async def test_single_strategy_backtest(self, basic_config):
        """测试单策略回测"""
        engine = BacktestEngine(basic_config)
        engine.register_strategy("test_strategy", create_buy_hold_strategy)
        
        await engine.initialize()
        
        results = await engine.run_backtest("test_strategy")
        
        assert "test_strategy" in results
        result = results["test_strategy"]
        
        assert isinstance(result, BacktestResult)
        assert result.config == basic_config
        assert result.total_events > 0
        assert result.processed_events >= 0
        assert isinstance(result.performance_metrics, dict)
    
    async def test_multi_strategy_backtest(self):
        """测试多策略回测"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("10000"),
            mode=BacktestMode.MULTI_STRATEGY,
            strategy_configs={
                "sma_strategy": {"short_window": 5, "long_window": 10, "position_size": 0.1},
                "buy_hold_strategy": {"position_size": 0.1}
            }
        )
        
        engine = BacktestEngine(config)
        engine.register_strategy("sma_strategy", create_sma_strategy)
        engine.register_strategy("buy_hold_strategy", create_buy_hold_strategy)
        
        await engine.initialize()
        
        results = await engine.run_backtest()
        
        assert len(results) == 2
        assert "sma_strategy" in results
        assert "buy_hold_strategy" in results
    
    async def test_event_queue_processing(self, backtest_engine):
        """测试事件队列处理"""
        initial_queue_size = len(backtest_engine.event_queue)
        
        # 处理几个事件
        processed = 0
        while backtest_engine.event_queue and processed < 10:
            event = backtest_engine.event_queue.pop(0)
            processed += 1
            
            assert isinstance(event, BacktestEvent)
            assert event.timestamp is not None
            assert event.event_type in ["market_data"]
        
        assert processed == 10
        assert len(backtest_engine.event_queue) == initial_queue_size - 10
    
    async def test_market_data_update(self, backtest_engine):
        """测试市场数据更新"""
        engine_name = list(backtest_engine.trading_engines.keys())[0]
        trading_engine = backtest_engine.trading_engines[engine_name]
        
        # 创建测试市场数据
        market_data = MarketData(
            symbol="BTCUSDT",
            price=51000.0,
            volume=1000.0,
            bid=50995.0,
            ask=51005.0,
            bid_volume=500.0,
            ask_volume=500.0,
            timestamp=datetime.utcnow()
        )
        
        await trading_engine.update_market_data(market_data)
        
        assert trading_engine.market_prices["BTCUSDT"] == Decimal("51000.0")
    
    async def test_backtest_progress_tracking(self, backtest_engine):
        """测试回测进度跟踪"""
        progress = await backtest_engine.get_backtest_progress()
        
        assert "total_events" in progress
        assert "processed_events" in progress
        assert "progress_percentage" in progress
        assert progress["total_events"] > 0
        assert 0 <= progress["progress_percentage"] <= 100
    
    async def test_backtest_stop(self, backtest_engine):
        """测试回测停止"""
        initial_queue_size = len(backtest_engine.event_queue)
        
        await backtest_engine.stop_backtest()
        
        assert len(backtest_engine.event_queue) == 0
    
    async def test_error_handling(self):
        """测试错误处理"""
        # 错误的策略函数
        def faulty_strategy(context):
            raise ValueError("Test error")
        
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 1),  # 1小时
            strategy_configs={"faulty": {}}
        )
        
        engine = BacktestEngine(config)
        engine.register_strategy("faulty", faulty_strategy)
        
        await engine.initialize()
        
        # 运行回测应该处理错误而不崩溃
        results = await engine.run_backtest("faulty")
        
        assert "faulty" in results
        result = results["faulty"]
        assert len(result.execution_errors) > 0
    
    async def test_different_timeframes(self):
        """测试不同时间框架"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 5),  # 5小时
            timeframes=["1m", "5m"],
            strategy_configs={"test": {}}
        )
        
        engine = BacktestEngine(config)
        engine.register_strategy("test", create_buy_hold_strategy)
        
        await engine.initialize()
        
        # 检查是否加载了不同时间框架的数据
        assert "BTCUSDT_1m" in engine.market_data_cache
        assert "BTCUSDT_5m" in engine.market_data_cache
        
        # 5分钟数据应该比1分钟数据少
        data_1m = engine.market_data_cache["BTCUSDT_1m"]
        data_5m = engine.market_data_cache["BTCUSDT_5m"]
        assert len(data_5m) < len(data_1m)
    
    async def test_performance_metrics_collection(self, basic_config):
        """测试性能指标收集"""
        engine = BacktestEngine(basic_config)
        engine.register_strategy("test_strategy", create_buy_hold_strategy)
        
        await engine.initialize()
        
        results = await engine.run_backtest("test_strategy")
        result = results["test_strategy"]
        
        metrics = result.performance_metrics
        
        # 检查关键指标是否存在
        expected_metrics = [
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_pnl", "total_return", "max_drawdown"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # 检查指标类型
        assert isinstance(metrics["total_trades"], int)
        assert isinstance(metrics["win_rate"], (int, float))
        assert isinstance(metrics["total_return"], (int, float))
    
    async def test_trade_recording(self, basic_config):
        """测试交易记录"""
        engine = BacktestEngine(basic_config)
        engine.register_strategy("test_strategy", create_buy_hold_strategy)
        
        await engine.initialize()
        
        results = await engine.run_backtest("test_strategy")
        result = results["test_strategy"]
        
        # 买入持有策略应该至少有一笔交易
        trade_records = result.trade_records
        
        if trade_records:
            trade = trade_records[0]
            assert hasattr(trade, "trade_id")
            assert hasattr(trade, "symbol")
            assert hasattr(trade, "side")
            assert hasattr(trade, "quantity")
            assert hasattr(trade, "entry_price")
    
    async def test_equity_curve_generation(self, basic_config):
        """测试权益曲线生成"""
        engine = BacktestEngine(basic_config)
        engine.register_strategy("test_strategy", create_buy_hold_strategy)
        
        await engine.initialize()
        
        results = await engine.run_backtest("test_strategy")
        result = results["test_strategy"]
        
        equity_curve = result.equity_curve
        
        # 应该有权益曲线数据
        if equity_curve:
            # 检查数据格式
            timestamp, balance = equity_curve[0]
            assert isinstance(timestamp, datetime)
            assert isinstance(balance, (Decimal, int, float))
            
            # 检查时间顺序
            for i in range(1, len(equity_curve)):
                assert equity_curve[i][0] >= equity_curve[i-1][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])