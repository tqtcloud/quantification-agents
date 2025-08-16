import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.backtesting.performance_evaluator import (
    PerformanceEvaluator, BacktestComparison, BenchmarkComparison, RiskAnalysis
)
from src.backtesting.backtest_engine import BacktestResult, BacktestConfig
from src.trading.performance_analytics import TradeRecord, PerformanceMetrics


class TestPerformanceEvaluator:
    """测试性能评估器"""
    
    @pytest.fixture
    def sample_trade_records(self):
        """创建样本交易记录"""
        trades = []
        base_time = datetime(2024, 1, 1)
        
        # 创建10笔交易，有盈有亏
        for i in range(10):
            entry_time = base_time + timedelta(hours=i)
            exit_time = entry_time + timedelta(hours=1)
            
            # 交替盈亏
            if i % 2 == 0:
                entry_price = Decimal("50000")
                exit_price = Decimal("51000")  # 盈利
            else:
                entry_price = Decimal("50000")
                exit_price = Decimal("49500")  # 亏损
            
            trade = TradeRecord(
                trade_id=f"trade_{i}",
                order_id=f"order_{i}",
                client_order_id=f"client_{i}",
                symbol="BTCUSDT",
                side="BUY",
                quantity=Decimal("0.1"),
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_commission=Decimal("5.0"),
                exit_commission=Decimal("5.1")
            )
            
            trade.calculate_pnl()
            trades.append(trade)
        
        return trades
    
    @pytest.fixture
    def sample_backtest_result(self, sample_trade_records):
        """创建样本回测结果"""
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("100000")
        )
        
        # 模拟性能指标
        performance_metrics = {
            "total_trades": 10,
            "winning_trades": 5,
            "losing_trades": 5,
            "win_rate": 0.5,
            "total_pnl": 250.0,
            "total_return": 0.0025,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.2,
            "sortino_ratio": 1.5,
            "calmar_ratio": 0.8,
            "profit_factor": 1.1
        }
        
        # 模拟权益曲线
        equity_curve = []
        balance = Decimal("100000")
        for i, trade in enumerate(sample_trade_records):
            balance += trade.net_pnl
            equity_curve.append((trade.exit_time, balance))
        
        result = BacktestResult(
            config=config,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=3600,
            performance_metrics=performance_metrics,
            strategy_results={"test_strategy": performance_metrics},
            trade_records=sample_trade_records,
            daily_stats=[],
            equity_curve=equity_curve,
            total_events=1000,
            processed_events=1000
        )
        
        return result
    
    @pytest.fixture
    def evaluator(self, sample_backtest_result):
        """创建评估器"""
        evaluator = PerformanceEvaluator()
        evaluator.add_backtest_result("test_strategy", sample_backtest_result)
        return evaluator
    
    def test_add_backtest_result(self, sample_backtest_result):
        """测试添加回测结果"""
        evaluator = PerformanceEvaluator()
        
        evaluator.add_backtest_result("strategy1", sample_backtest_result)
        
        assert "strategy1" in evaluator.results_cache
        assert evaluator.results_cache["strategy1"] == sample_backtest_result
    
    def test_evaluate_single_strategy(self, evaluator):
        """测试单策略评估"""
        result = evaluator.results_cache["test_strategy"]
        evaluation = evaluator.evaluate_single_strategy(result)
        
        # 检查评估结果结构
        assert "basic_metrics" in evaluation
        assert "risk_analysis" in evaluation
        assert "time_analysis" in evaluation
        assert "trade_analysis" in evaluation
        assert "summary_score" in evaluation
        
        # 检查基础指标
        basic_metrics = evaluation["basic_metrics"]
        assert basic_metrics["total_trades"] == 10
        assert basic_metrics["win_rate"] == 0.5
        
        # 检查风险分析
        risk_analysis = evaluation["risk_analysis"]
        assert "var_95" in risk_analysis
        assert "max_drawdown" in risk_analysis
        assert "volatility" in risk_analysis
        
        # 检查交易分析
        trade_analysis = evaluation["trade_analysis"]
        assert "trade_count" in trade_analysis
        assert "max_consecutive_wins" in trade_analysis
        assert "max_consecutive_losses" in trade_analysis
        
        # 检查综合评分
        summary_score = evaluation["summary_score"]
        assert isinstance(summary_score, (int, float))
        assert 0 <= summary_score <= 100
    
    def test_risk_metrics_calculation(self, evaluator):
        """测试风险指标计算"""
        result = evaluator.results_cache["test_strategy"]
        risk_analysis = evaluator._calculate_risk_metrics(result)
        
        assert isinstance(risk_analysis, RiskAnalysis)
        assert risk_analysis.var_95 >= 0
        assert risk_analysis.var_99 >= 0
        assert risk_analysis.volatility >= 0
        assert isinstance(risk_analysis.max_drawdown_duration, int)
        assert risk_analysis.tail_ratio > 0
    
    def test_time_patterns_analysis(self, evaluator):
        """测试时间模式分析"""
        result = evaluator.results_cache["test_strategy"]
        time_analysis = evaluator._analyze_time_patterns(result)
        
        assert "monthly_returns" in time_analysis
        assert "weekday_analysis" in time_analysis
        assert "hourly_analysis" in time_analysis
        
        # 检查月度收益
        monthly_returns = time_analysis["monthly_returns"]
        if monthly_returns:
            for month, pnl in monthly_returns.items():
                assert isinstance(month, str)
                assert isinstance(pnl, (int, float))
        
        # 检查星期分析
        weekday_analysis = time_analysis["weekday_analysis"]
        for day, stats in weekday_analysis.items():
            assert isinstance(day, int)
            assert 0 <= day <= 6
            assert "avg_pnl" in stats
            assert "win_rate" in stats
            assert "trade_count" in stats
    
    def test_trade_quality_analysis(self, evaluator):
        """测试交易质量分析"""
        result = evaluator.results_cache["test_strategy"]
        trade_analysis = evaluator._analyze_trade_quality(result)
        
        assert trade_analysis["trade_count"] == 10
        assert "avg_duration_hours" in trade_analysis
        assert "max_consecutive_wins" in trade_analysis
        assert "max_consecutive_losses" in trade_analysis
        assert "max_single_win" in trade_analysis
        assert "max_single_loss" in trade_analysis
        
        # 检查连续统计
        assert trade_analysis["max_consecutive_wins"] >= 1
        assert trade_analysis["max_consecutive_losses"] >= 1
    
    def test_returns_series_extraction(self, evaluator):
        """测试收益率序列提取"""
        result = evaluator.results_cache["test_strategy"]
        returns = evaluator._extract_returns_series(result)
        
        assert len(returns) == 10  # 10笔交易
        assert all(isinstance(r, float) for r in returns)
        
        # 检查盈亏交替模式
        assert returns[0] > 0  # 第一笔盈利
        assert returns[1] < 0  # 第二笔亏损
    
    def test_compare_strategies(self):
        """测试策略比较"""
        evaluator = PerformanceEvaluator()
        
        # 创建两个不同的策略结果
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2)
        )
        
        # 策略1：表现较好
        result1 = BacktestResult(
            config=config,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=3600,
            performance_metrics={
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.05,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "calmar_ratio": 2.0,
                "sortino_ratio": 1.8
            },
            strategy_results={},
            trade_records=[],
            daily_stats=[],
            equity_curve=[],
            total_events=1000,
            processed_events=1000
        )
        
        # 策略2：表现较差
        result2 = BacktestResult(
            config=config,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=3600,
            performance_metrics={
                "total_return": 0.05,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.1,
                "win_rate": 0.4,
                "profit_factor": 1.2,
                "calmar_ratio": 0.5,
                "sortino_ratio": 0.9
            },
            strategy_results={},
            trade_records=[],
            daily_stats=[],
            equity_curve=[],
            total_events=1000,
            processed_events=1000
        )
        
        evaluator.add_backtest_result("strategy1", result1)
        evaluator.add_backtest_result("strategy2", result2)
        
        comparison = evaluator.compare_strategies()
        
        assert isinstance(comparison, BacktestComparison)
        assert len(comparison.strategy_names) == 2
        assert "strategy1" in comparison.strategy_names
        assert "strategy2" in comparison.strategy_names
        
        # 检查比较指标
        assert "strategy1" in comparison.comparison_metrics
        assert "strategy2" in comparison.comparison_metrics
        
        # 检查相对性能
        assert len(comparison.relative_performance) == 2
        
        # 策略1应该表现更好
        assert comparison.relative_performance["strategy1"] > comparison.relative_performance["strategy2"]
        
        # 检查风险调整排名
        ranking = comparison.risk_adjusted_ranking
        assert len(ranking) == 2
        assert ranking[0][0] == "strategy1"  # strategy1应该排第一
    
    def test_benchmark_comparison(self, evaluator):
        """测试基准比较"""
        # 创建基准收益率
        benchmark_returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012, -0.002, 0.008]
        
        comparison = evaluator.compare_to_benchmark(
            "test_strategy", 
            benchmark_returns, 
            "Market_Benchmark"
        )
        
        assert isinstance(comparison, BenchmarkComparison)
        assert comparison.benchmark_name == "Market_Benchmark"
        assert isinstance(comparison.strategy_return, float)
        assert isinstance(comparison.benchmark_return, float)
        assert isinstance(comparison.alpha, float)
        assert isinstance(comparison.beta, float)
        assert isinstance(comparison.information_ratio, float)
        assert isinstance(comparison.tracking_error, float)
    
    def test_correlation_matrix_calculation(self, evaluator):
        """测试相关系数矩阵计算"""
        # 创建两个收益率序列
        returns_series = {
            "strategy1": [0.01, -0.005, 0.02, -0.01, 0.015],
            "strategy2": [0.008, -0.003, 0.018, -0.012, 0.012]
        }
        
        correlation_matrix = evaluator._calculate_correlation_matrix(returns_series)
        
        assert "strategy1" in correlation_matrix
        assert "strategy2" in correlation_matrix
        assert correlation_matrix["strategy1"]["strategy1"] == 1.0
        assert correlation_matrix["strategy2"]["strategy2"] == 1.0
        
        # 检查对称性
        corr_12 = correlation_matrix["strategy1"]["strategy2"]
        corr_21 = correlation_matrix["strategy2"]["strategy1"]
        assert abs(corr_12 - corr_21) < 1e-10
    
    def test_relative_performance_calculation(self, evaluator):
        """测试相对性能计算"""
        metrics = {
            "strategy1": {
                "total_return": 0.1,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.05
            },
            "strategy2": {
                "total_return": 0.05,
                "sharpe_ratio": 0.8,
                "max_drawdown": 0.1
            }
        }
        
        relative_performance = evaluator._calculate_relative_performance(metrics)
        
        assert len(relative_performance) == 2
        assert "strategy1" in relative_performance
        assert "strategy2" in relative_performance
        
        # strategy1应该表现更好
        assert relative_performance["strategy1"] > relative_performance["strategy2"]
    
    def test_summary_score_calculation(self, evaluator):
        """测试综合评分计算"""
        metrics = {
            "total_return": 0.1,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.05,
            "win_rate": 0.6
        }
        
        risk_analysis = RiskAnalysis(
            var_95=0.02,
            var_99=0.04,
            cvar_95=0.025,
            max_drawdown=0.05,
            max_drawdown_duration=10,
            volatility=0.15,
            skewness=-0.1,
            kurtosis=0.5,
            tail_ratio=2.0
        )
        
        score = evaluator._calculate_summary_score(metrics, risk_analysis)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_generate_report(self, evaluator):
        """测试生成报告"""
        report = evaluator.generate_report("test_strategy")
        
        assert "strategy_name" in report
        assert "report_timestamp" in report
        assert "backtest_period" in report
        assert "performance_evaluation" in report
        assert "config" in report
        assert "execution_summary" in report
        
        assert report["strategy_name"] == "test_strategy"
        
        # 检查回测周期
        backtest_period = report["backtest_period"]
        assert "start_date" in backtest_period
        assert "end_date" in backtest_period
        assert "duration_days" in backtest_period
        
        # 检查性能评估
        performance_evaluation = report["performance_evaluation"]
        assert "basic_metrics" in performance_evaluation
        assert "risk_analysis" in performance_evaluation
        assert "summary_score" in performance_evaluation
        
        # 检查执行总结
        execution_summary = report["execution_summary"]
        assert "total_events" in execution_summary
        assert "processed_events" in execution_summary
        assert "duration_seconds" in execution_summary
    
    def test_kurtosis_calculation(self, evaluator):
        """测试峰度计算"""
        # 正态分布数据（峰度接近0）
        normal_data = np.random.normal(0, 1, 1000).tolist()
        kurtosis_normal = evaluator._calculate_kurtosis(normal_data)
        assert -1 < kurtosis_normal < 1  # 正态分布的超额峰度接近0
        
        # 尖峰数据
        peaked_data = [0] * 990 + [10] * 5 + [-10] * 5
        kurtosis_peaked = evaluator._calculate_kurtosis(peaked_data)
        assert kurtosis_peaked > 0  # 尖峰分布的超额峰度为正
    
    def test_max_drawdown_duration_calculation(self, evaluator):
        """测试最大回撤持续时间计算"""
        # 创建包含回撤的权益曲线
        base_time = datetime(2024, 1, 1)
        equity_curve = [
            (base_time + timedelta(days=i), Decimal(str(100000 + values)))
            for i, values in enumerate([0, 1000, 500, -500, -1000, -800, 0, 2000])
        ]
        
        result = BacktestResult(
            config=BacktestConfig(datetime(2024, 1, 1), datetime(2024, 1, 8)),
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=3600,
            performance_metrics={},
            strategy_results={},
            trade_records=[],
            daily_stats=[],
            equity_curve=equity_curve,
            total_events=1000,
            processed_events=1000
        )
        
        duration = evaluator._calculate_max_drawdown_duration(result)
        assert duration >= 0
        assert isinstance(duration, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])