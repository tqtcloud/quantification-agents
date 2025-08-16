import pytest
import statistics
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading.performance_analytics import (
    PerformanceTracker,
    TradeRecord,
    PositionRecord,
    PerformanceMetrics,
    PerformancePeriod
)


class TestTradeRecord:
    """测试交易记录"""
    
    def test_trade_record_initialization(self):
        """测试交易记录初始化"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0")
        )
        
        assert trade.trade_id == "trade_001"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.quantity == Decimal("1.0")
        assert trade.entry_price == Decimal("50000.0")
        assert trade.exit_price is None
        assert trade.realized_pnl == Decimal("0")
        
    def test_pnl_calculation_buy_profit(self):
        """测试买单盈利计算"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("52000.0"),
            entry_commission=Decimal("20.0"),
            exit_commission=Decimal("20.8"),
            slippage_cost=Decimal("10.0")
        )
        
        trade.calculate_pnl()
        
        expected_gross_pnl = (Decimal("52000.0") - Decimal("50000.0")) * Decimal("1.0")
        expected_net_pnl = expected_gross_pnl - Decimal("20.0") - Decimal("20.8") - Decimal("10.0")
        
        assert trade.gross_pnl == expected_gross_pnl
        assert trade.net_pnl == expected_net_pnl
        assert trade.realized_pnl == expected_net_pnl
        
    def test_pnl_calculation_buy_loss(self):
        """测试买单亏损计算"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("48000.0"),
            entry_commission=Decimal("20.0"),
            exit_commission=Decimal("19.2")
        )
        
        trade.calculate_pnl()
        
        expected_gross_pnl = (Decimal("48000.0") - Decimal("50000.0")) * Decimal("1.0")
        expected_net_pnl = expected_gross_pnl - Decimal("20.0") - Decimal("19.2")
        
        assert trade.gross_pnl == expected_gross_pnl
        assert trade.net_pnl == expected_net_pnl
        assert trade.net_pnl < 0
        
    def test_pnl_calculation_sell_profit(self):
        """测试卖单盈利计算"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="SELL",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("48000.0"),  # 空头：价格下跌盈利
            entry_commission=Decimal("20.0"),
            exit_commission=Decimal("19.2")
        )
        
        trade.calculate_pnl()
        
        expected_gross_pnl = (Decimal("50000.0") - Decimal("48000.0")) * Decimal("1.0")
        expected_net_pnl = expected_gross_pnl - Decimal("20.0") - Decimal("19.2")
        
        assert trade.gross_pnl == expected_gross_pnl
        assert trade.net_pnl == expected_net_pnl
        assert trade.net_pnl > 0
        
    def test_trade_duration(self):
        """测试交易持续时间"""
        entry_time = datetime.utcnow()
        exit_time = entry_time + timedelta(hours=2, minutes=30)
        
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            entry_time=entry_time,
            exit_time=exit_time
        )
        
        duration = trade.get_duration()
        assert duration == timedelta(hours=2, minutes=30)
        
    def test_return_rate_calculation(self):
        """测试收益率计算"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("52000.0"),
            entry_commission=Decimal("20.0"),
            exit_commission=Decimal("20.8")
        )
        
        trade.calculate_pnl()
        return_rate = trade.get_return_rate()
        
        cost_basis = Decimal("50000.0") * Decimal("1.0") + Decimal("20.0")
        expected_return_rate = trade.net_pnl / cost_basis
        
        assert return_rate == expected_return_rate


class TestPositionRecord:
    """测试持仓记录"""
    
    def test_position_record_initialization(self):
        """测试持仓记录初始化"""
        position = PositionRecord(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            avg_entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0")
        )
        
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.size == Decimal("1.0")
        assert position.avg_entry_price == Decimal("50000.0")
        assert position.unrealized_pnl == Decimal("0")
        
    def test_long_position_unrealized_pnl_update(self):
        """测试多头仓位未实现盈亏更新"""
        position = PositionRecord(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            avg_entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0")
        )
        
        # 价格上涨
        position.update_unrealized_pnl(Decimal("52000.0"))
        expected_pnl = (Decimal("52000.0") - Decimal("50000.0")) * Decimal("1.0")
        assert position.unrealized_pnl == expected_pnl
        assert position.max_unrealized_profit == expected_pnl
        
        # 价格下跌
        position.update_unrealized_pnl(Decimal("48000.0"))
        expected_loss = (Decimal("48000.0") - Decimal("50000.0")) * Decimal("1.0")
        assert position.unrealized_pnl == expected_loss
        assert position.max_unrealized_loss == expected_loss
        
    def test_short_position_unrealized_pnl_update(self):
        """测试空头仓位未实现盈亏更新"""
        position = PositionRecord(
            symbol="BTCUSDT",
            side="SHORT",
            size=Decimal("1.0"),
            avg_entry_price=Decimal("50000.0"),
            current_price=Decimal("50000.0")
        )
        
        # 价格下跌（空头盈利）
        position.update_unrealized_pnl(Decimal("48000.0"))
        expected_pnl = (Decimal("50000.0") - Decimal("48000.0")) * Decimal("1.0")
        assert position.unrealized_pnl == expected_pnl
        assert position.max_unrealized_profit == expected_pnl
        
        # 价格上涨（空头亏损）
        position.update_unrealized_pnl(Decimal("52000.0"))
        expected_loss = (Decimal("50000.0") - Decimal("52000.0")) * Decimal("1.0")
        assert position.unrealized_pnl == expected_loss
        assert position.max_unrealized_loss == expected_loss


class TestPerformanceTracker:
    """测试性能追踪器"""
    
    @pytest.fixture
    def tracker(self):
        """创建性能追踪器实例"""
        return PerformanceTracker(initial_balance=Decimal("100000"))
        
    def test_tracker_initialization(self, tracker):
        """测试追踪器初始化"""
        assert tracker.initial_balance == Decimal("100000")
        assert tracker.current_balance == Decimal("100000")
        assert len(tracker.trade_records) == 0
        assert len(tracker.position_records) == 0
        
    def test_trade_recording(self, tracker):
        """测试交易记录"""
        trade = TradeRecord(
            trade_id="trade_001",
            order_id="order_001",
            client_order_id="client_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            exit_price=Decimal("52000.0"),
            strategy_name="test_strategy"
        )
        trade.calculate_pnl()
        
        tracker.record_trade(trade)
        
        assert len(tracker.trade_records) == 1
        assert trade in tracker.strategy_performance["test_strategy"]
        assert trade in tracker.symbol_performance["BTCUSDT"]
        assert tracker.current_balance > tracker.initial_balance
        
    def test_position_management(self, tracker):
        """测试持仓管理"""
        position = PositionRecord(
            symbol="BTCUSDT",
            side="LONG",
            size=Decimal("1.0"),
            avg_entry_price=Decimal("50000.0"),
            current_price=Decimal("52000.0")
        )
        
        # 添加持仓
        tracker.update_position("BTCUSDT", position)
        assert "BTCUSDT" in tracker.position_records
        
        # 移除持仓
        tracker.remove_position("BTCUSDT")
        assert "BTCUSDT" not in tracker.position_records
        
    def test_basic_metrics_calculation(self, tracker):
        """测试基础指标计算"""
        # 添加几笔交易
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), Decimal("52000"), net_pnl=Decimal("1950")),
            TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("52000"), Decimal("51000"), net_pnl=Decimal("-1050")),
            TradeRecord("t3", "o3", "c3", "ETHUSDT", "BUY", Decimal("10.0"), 
                       Decimal("3000"), Decimal("3200"), net_pnl=Decimal("1900"))
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
            
        metrics = tracker.calculate_performance_metrics()
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        expected_win_rate = Decimal("2") / Decimal("3")
        assert abs(metrics.win_rate - expected_win_rate) < Decimal("0.0001")
        assert metrics.total_pnl == Decimal("2800")  # 1950 - 1050 + 1900
        assert metrics.gross_profit == Decimal("3850")  # 1950 + 1900
        assert metrics.gross_loss == Decimal("1050")
        
    def test_strategy_filtering(self, tracker):
        """测试策略过滤"""
        # 添加不同策略的交易
        trade1 = TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                           Decimal("50000"), strategy_name="strategy_a", net_pnl=Decimal("1000"))
        trade2 = TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                           Decimal("50000"), strategy_name="strategy_b", net_pnl=Decimal("500"))
        
        tracker.record_trade(trade1)
        tracker.record_trade(trade2)
        
        # 测试策略A的指标
        metrics_a = tracker.calculate_performance_metrics(strategy_name="strategy_a")
        assert metrics_a.total_trades == 1
        assert metrics_a.total_pnl == Decimal("1000")
        
        # 测试策略B的指标
        metrics_b = tracker.calculate_performance_metrics(strategy_name="strategy_b")
        assert metrics_b.total_trades == 1
        assert metrics_b.total_pnl == Decimal("500")
        
    def test_symbol_filtering(self, tracker):
        """测试交易对过滤"""
        # 添加不同交易对的交易
        trade1 = TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                           Decimal("50000"), net_pnl=Decimal("1000"))
        trade2 = TradeRecord("t2", "o2", "c2", "ETHUSDT", "BUY", Decimal("10.0"), 
                           Decimal("3000"), net_pnl=Decimal("500"))
        
        tracker.record_trade(trade1)
        tracker.record_trade(trade2)
        
        # 测试BTC的指标
        metrics_btc = tracker.calculate_performance_metrics(symbol="BTCUSDT")
        assert metrics_btc.total_trades == 1
        assert metrics_btc.total_pnl == Decimal("1000")
        
        # 测试ETH的指标
        metrics_eth = tracker.calculate_performance_metrics(symbol="ETHUSDT")
        assert metrics_eth.total_trades == 1
        assert metrics_eth.total_pnl == Decimal("500")
        
    def test_time_period_filtering(self, tracker):
        """测试时间周期过滤"""
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        
        # 添加不同时间的交易
        trade1 = TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                           Decimal("50000"), entry_time=yesterday, net_pnl=Decimal("1000"))
        trade2 = TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                           Decimal("50000"), entry_time=now, net_pnl=Decimal("500"))
        
        tracker.record_trade(trade1)
        tracker.record_trade(trade2)
        
        # 测试今日指标
        daily_metrics = tracker.calculate_performance_metrics(PerformancePeriod.DAILY)
        assert daily_metrics.total_trades == 1  # 只有今天的交易
        assert daily_metrics.total_pnl == Decimal("500")
        
    def test_drawdown_calculation(self, tracker):
        """测试回撤计算"""
        # 创建一系列交易模拟回撤
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("5000")),  # +5000, 总计105000
            TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("3000")),  # +3000, 总计108000
            TradeRecord("t3", "o3", "c3", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("-6000")), # -6000, 总计102000
            TradeRecord("t4", "o4", "c4", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("-2000"))  # -2000, 总计100000
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
            
        max_dd, current_dd = tracker._calculate_drawdown(tracker.trade_records)
        
        # 最大回撤应该是从108000到100000
        expected_max_dd = (Decimal("108000") - Decimal("100000")) / Decimal("108000")
        assert abs(max_dd - expected_max_dd) < Decimal("0.0001")
        
    def test_sharpe_ratio_calculation(self, tracker):
        """测试夏普比率计算"""
        returns = [0.02, 0.01, -0.01, 0.03, 0.015, -0.005, 0.025]
        
        sharpe = tracker._calculate_sharpe_ratio(returns)
        
        # 基本验证夏普比率计算
        assert isinstance(sharpe, Decimal)
        assert sharpe != Decimal("0")  # 应该有有效的夏普比率
        
    def test_sortino_ratio_calculation(self, tracker):
        """测试索提诺比率计算"""
        returns = [0.02, 0.01, -0.01, 0.03, 0.015, -0.005, 0.025, -0.02]
        
        sortino = tracker._calculate_sortino_ratio(returns)
        
        # 基本验证索提诺比率计算
        assert isinstance(sortino, Decimal)
        
    def test_var_calculation(self, tracker):
        """测试VaR计算"""
        returns = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.03, 0.01, -0.03, 0.02, 0.01,
                  -0.01, 0.015, -0.025, 0.035, -0.015]
        
        var_95 = tracker._calculate_var(returns, 0.95)
        
        # VaR应该是正数且合理
        assert var_95 > Decimal("0")
        assert var_95 < Decimal("1")  # 应该小于100%
        
    def test_consecutive_wins_losses(self, tracker):
        """测试连续盈亏统计"""
        # 创建包含连续盈亏的交易序列
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("1000")),   # 盈利
            TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("500")),    # 盈利
            TradeRecord("t3", "o3", "c3", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("200")),    # 盈利
            TradeRecord("t4", "o4", "c4", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("-300")),   # 亏损
            TradeRecord("t5", "o5", "c5", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("-200")),   # 亏损
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
            
        max_wins, max_losses = tracker._calculate_streaks(tracker.trade_records)
        
        assert max_wins == 3    # 连续3次盈利
        assert max_losses == 2  # 连续2次亏损
        
    def test_benchmark_comparison(self, tracker):
        """测试基准比较"""
        # 添加一些交易
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("2000")),
            TradeRecord("t2", "o2", "c2", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), net_pnl=Decimal("1000"))
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
            
        # 基准收益
        benchmark_returns = [Decimal("0.01"), Decimal("0.005")]  # 1%, 0.5%
        
        comparison = tracker.compare_to_benchmark(benchmark_returns)
        
        assert "portfolio_return" in comparison
        assert "benchmark_return" in comparison
        assert "alpha" in comparison
        assert "beta" in comparison
        assert "information_ratio" in comparison
        
    def test_performance_report_generation(self, tracker):
        """测试性能报告生成"""
        # 添加交易和持仓
        trade = TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                          Decimal("50000"), strategy_name="test_strategy", net_pnl=Decimal("1000"))
        tracker.record_trade(trade)
        
        position = PositionRecord("ETHUSDT", "LONG", Decimal("10.0"), 
                                Decimal("3000"), Decimal("3100"))
        tracker.update_position("ETHUSDT", position)
        
        report = tracker.generate_performance_report(include_trades=True, include_positions=True)
        
        assert "report_timestamp" in report
        assert "metrics" in report
        assert "trades" in report
        assert "positions" in report
        assert "strategy_analysis" in report
        assert "symbol_analysis" in report
        
        # 检查策略分析
        assert "test_strategy" in report["strategy_analysis"]
        
    def test_csv_export(self, tracker, tmp_path):
        """测试CSV导出"""
        # 添加一些交易
        trades = [
            TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                       Decimal("50000"), Decimal("52000")),
            TradeRecord("t2", "o2", "c2", "ETHUSDT", "SELL", Decimal("10.0"), 
                       Decimal("3000"), Decimal("2900"))
        ]
        
        for trade in trades:
            tracker.record_trade(trade)
            
        # 导出到临时文件
        csv_file = tmp_path / "trades.csv"
        tracker.export_trades_to_csv(str(csv_file))
        
        # 验证文件存在且包含数据
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "trade_id" in content
        assert "BTCUSDT" in content
        assert "ETHUSDT" in content
        
    def test_realtime_metrics(self, tracker):
        """测试实时指标"""
        # 添加交易和持仓
        trade = TradeRecord("t1", "o1", "c1", "BTCUSDT", "BUY", Decimal("1.0"), 
                          Decimal("50000"), net_pnl=Decimal("1000"))
        tracker.record_trade(trade)
        
        position = PositionRecord("ETHUSDT", "LONG", Decimal("10.0"), 
                                Decimal("3000"), Decimal("3100"))
        position.update_unrealized_pnl(Decimal("3100"))
        tracker.update_position("ETHUSDT", position)
        
        metrics = tracker.get_realtime_metrics()
        
        assert "current_balance" in metrics
        assert "current_equity" in metrics
        assert "total_unrealized_pnl" in metrics
        assert "total_return" in metrics
        assert "active_positions" in metrics
        assert "total_trades" in metrics
        
        assert metrics["active_positions"] == 1
        assert metrics["total_trades"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])