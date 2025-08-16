import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestMode
from src.backtesting.performance_evaluator import PerformanceEvaluator
from src.backtesting.optimizer import ParameterOptimizer, OptimizationConfig, ParameterRange
from src.backtesting.strategies import create_sma_strategy, create_rsi_strategy, create_buy_hold_strategy
from src.core.models import Order, OrderSide, OrderType


@pytest.mark.asyncio
class TestBacktestIntegration:
    """回测系统集成测试"""
    
    async def test_end_to_end_single_strategy_backtest(self):
        """测试端到端单策略回测"""
        # 配置回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_balance=Decimal("100000"),
            symbols=["BTCUSDT"],
            timeframes=["1m"],
            strategy_configs={
                "sma_strategy": {
                    "short_window": 10,
                    "long_window": 20,
                    "position_size": 0.1
                }
            }
        )
        
        # 创建回测引擎
        engine = BacktestEngine(config)
        engine.register_strategy("sma_strategy", create_sma_strategy)
        
        # 初始化并运行回测
        await engine.initialize()
        results = await engine.run_backtest("sma_strategy")
        
        # 验证结果
        assert "sma_strategy" in results
        result = results["sma_strategy"]
        
        assert result.total_events > 0
        assert result.processed_events >= 0
        assert isinstance(result.performance_metrics, dict)
        assert "total_return" in result.performance_metrics
        assert "sharpe_ratio" in result.performance_metrics
        
        # 验证交易记录
        if result.trade_records:
            for trade in result.trade_records:
                assert trade.symbol == "BTCUSDT"
                assert trade.quantity > 0
                assert trade.entry_price > 0
    
    async def test_multi_strategy_comparison(self):
        """测试多策略比较"""
        # 配置多策略回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_balance=Decimal("50000"),
            mode=BacktestMode.MULTI_STRATEGY,
            strategy_configs={
                "sma_strategy": {
                    "short_window": 10,
                    "long_window": 20,
                    "position_size": 0.1
                },
                "buy_hold_strategy": {
                    "position_size": 0.1
                }
            }
        )
        
        # 创建回测引擎
        engine = BacktestEngine(config)
        engine.register_strategy("sma_strategy", create_sma_strategy)
        engine.register_strategy("buy_hold_strategy", create_buy_hold_strategy)
        
        # 运行回测
        await engine.initialize()
        results = await engine.run_backtest()
        
        # 验证结果
        assert len(results) == 2
        assert "sma_strategy" in results
        assert "buy_hold_strategy" in results
        
        # 使用性能评估器比较策略
        evaluator = PerformanceEvaluator()
        for name, result in results.items():
            evaluator.add_backtest_result(name, result)
        
        comparison = evaluator.compare_strategies()
        
        assert len(comparison.strategy_names) == 2
        assert len(comparison.comparison_metrics) == 2
        assert len(comparison.relative_performance) == 2
        assert len(comparison.risk_adjusted_ranking) == 2
    
    async def test_parameter_optimization_workflow(self):
        """测试参数优化工作流"""
        # 基础配置
        base_config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("50000")
        )
        
        # 参数范围
        parameter_ranges = [
            ParameterRange("short_window", "int", 5, 15, 5),  # [5, 10, 15]
            ParameterRange("long_window", "int", 20, 30, 10),  # [20, 30]
            ParameterRange("position_size", "float", 0.05, 0.15, 0.05)  # [0.05, 0.1, 0.15]
        ]
        
        # 优化配置
        optimization_config = OptimizationConfig(
            base_config=base_config,
            strategy_name="sma_strategy",
            strategy_function=create_sma_strategy,
            parameter_ranges=parameter_ranges,
            optimization_method="grid_search",
            objective_function="sharpe_ratio",
            max_combinations=10,  # 限制组合数量
            max_workers=1
        )
        
        # 运行优化
        optimizer = ParameterOptimizer(optimization_config)
        result = await optimizer.optimize()
        
        # 验证优化结果
        assert isinstance(result.best_parameters, dict)
        assert "short_window" in result.best_parameters
        assert "long_window" in result.best_parameters
        assert "position_size" in result.best_parameters
        assert isinstance(result.best_score, float)
        assert result.total_combinations > 0
        assert result.successful_runs >= 0
        
        # 验证最优参数在合理范围内
        assert 5 <= result.best_parameters["short_window"] <= 15
        assert 20 <= result.best_parameters["long_window"] <= 30
        assert 0.05 <= result.best_parameters["position_size"] <= 0.15
    
    async def test_performance_evaluation_workflow(self):
        """测试性能评估工作流"""
        # 运行单个策略回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_balance=Decimal("100000"),
            strategy_configs={
                "test_strategy": {
                    "short_window": 8,
                    "long_window": 21,
                    "position_size": 0.1
                }
            }
        )
        
        engine = BacktestEngine(config)
        engine.register_strategy("test_strategy", create_sma_strategy)
        
        await engine.initialize()
        results = await engine.run_backtest("test_strategy")
        
        # 性能评估
        evaluator = PerformanceEvaluator()
        evaluator.add_backtest_result("test_strategy", results["test_strategy"])
        
        # 单策略评估
        evaluation = evaluator.evaluate_single_strategy(results["test_strategy"])
        
        assert "basic_metrics" in evaluation
        assert "risk_analysis" in evaluation
        assert "time_analysis" in evaluation
        assert "trade_analysis" in evaluation
        assert "summary_score" in evaluation
        
        # 生成详细报告
        report = evaluator.generate_report("test_strategy")
        
        assert "strategy_name" in report
        assert "backtest_period" in report
        assert "performance_evaluation" in report
        assert report["strategy_name"] == "test_strategy"
        
        # 验证报告结构
        assert "start_date" in report["backtest_period"]
        assert "end_date" in report["backtest_period"]
        assert "duration_days" in report["backtest_period"]
    
    async def test_benchmark_comparison_workflow(self):
        """测试基准比较工作流"""
        # 运行策略回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("50000"),
            strategy_configs={
                "buy_hold": {"position_size": 0.1}
            }
        )
        
        engine = BacktestEngine(config)
        engine.register_strategy("buy_hold", create_buy_hold_strategy)
        
        await engine.initialize()
        results = await engine.run_backtest("buy_hold")
        
        # 创建基准收益率（模拟市场表现）
        benchmark_returns = []
        for i in range(10):  # 假设有10个交易日
            daily_return = 0.001 if i % 2 == 0 else -0.0005  # 简单的上下波动
            benchmark_returns.append(daily_return)
        
        # 基准比较
        evaluator = PerformanceEvaluator()
        evaluator.add_backtest_result("buy_hold", results["buy_hold"])
        
        comparison = evaluator.compare_to_benchmark("buy_hold", benchmark_returns, "Market_Index")
        
        assert comparison.benchmark_name == "Market_Index"
        assert isinstance(comparison.strategy_return, float)
        assert isinstance(comparison.benchmark_return, float)
        assert isinstance(comparison.alpha, float)
        assert isinstance(comparison.beta, float)
    
    async def test_data_consistency_across_components(self):
        """测试组件间数据一致性"""
        # 配置回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("100000"),
            strategy_configs={
                "consistency_test": {
                    "short_window": 5,
                    "long_window": 10,
                    "position_size": 0.1
                }
            }
        )
        
        # 运行回测
        engine = BacktestEngine(config)
        engine.register_strategy("consistency_test", create_sma_strategy)
        
        await engine.initialize()
        results = await engine.run_backtest("consistency_test")
        
        result = results["consistency_test"]
        
        # 验证配置一致性
        assert result.config.start_date == config.start_date
        assert result.config.end_date == config.end_date
        assert result.config.initial_balance == config.initial_balance
        
        # 验证事件处理一致性
        assert result.total_events > 0
        assert result.processed_events <= result.total_events
        
        # 验证性能指标一致性
        metrics = result.performance_metrics
        if metrics.get("total_trades", 0) > 0:
            assert metrics["winning_trades"] + metrics["losing_trades"] <= metrics["total_trades"]
            if metrics["total_trades"] > 0:
                calculated_win_rate = metrics["winning_trades"] / metrics["total_trades"]
                assert abs(calculated_win_rate - metrics["win_rate"]) < 0.01
        
        # 验证权益曲线一致性
        if result.equity_curve:
            # 权益曲线应该按时间顺序
            for i in range(1, len(result.equity_curve)):
                assert result.equity_curve[i][0] >= result.equity_curve[i-1][0]
    
    async def test_error_recovery_and_robustness(self):
        """测试错误恢复和稳健性"""
        # 测试无效配置的处理
        invalid_config = BacktestConfig(
            start_date=datetime(2024, 1, 2),  # 开始日期晚于结束日期
            end_date=datetime(2024, 1, 1),
            initial_balance=Decimal("100000")
        )
        
        engine = BacktestEngine(invalid_config)
        
        # 即使配置有问题，系统也应该能优雅处理
        try:
            await engine.initialize()
            # 如果初始化成功，数据应该为空或最小
            assert len(engine.market_data_cache) >= 0
        except Exception as e:
            # 或者抛出明确的错误
            assert isinstance(e, (ValueError, AssertionError))
        
        # 测试缺失策略的处理
        valid_config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            strategy_configs={"missing_strategy": {}}
        )
        
        engine = BacktestEngine(valid_config)
        await engine.initialize()
        
        # 运行不存在的策略应该报错
        with pytest.raises(ValueError):
            await engine.run_backtest("missing_strategy")
    
    async def test_performance_under_different_market_conditions(self):
        """测试不同市场条件下的性能"""
        base_config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            initial_balance=Decimal("50000")
        )
        
        # 测试不同策略在相同市场条件下的表现
        strategies = {
            "sma_conservative": {
                "strategy_func": create_sma_strategy,
                "params": {"short_window": 20, "long_window": 50, "position_size": 0.05}
            },
            "sma_aggressive": {
                "strategy_func": create_sma_strategy,
                "params": {"short_window": 5, "long_window": 10, "position_size": 0.2}
            },
            "buy_hold": {
                "strategy_func": create_buy_hold_strategy,
                "params": {"position_size": 0.1}
            }
        }
        
        results = {}
        
        for strategy_name, strategy_info in strategies.items():
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_balance=base_config.initial_balance,
                strategy_configs={strategy_name: strategy_info["params"]}
            )
            
            engine = BacktestEngine(config)
            engine.register_strategy(strategy_name, strategy_info["strategy_func"])
            
            await engine.initialize()
            strategy_results = await engine.run_backtest(strategy_name)
            results[strategy_name] = strategy_results[strategy_name]
        
        # 比较不同策略的表现
        evaluator = PerformanceEvaluator()
        for name, result in results.items():
            evaluator.add_backtest_result(name, result)
        
        comparison = evaluator.compare_strategies()
        
        # 验证比较结果
        assert len(comparison.strategy_names) == 3
        assert len(comparison.relative_performance) == 3
        
        # 检查风险调整排名
        ranking = comparison.risk_adjusted_ranking
        assert len(ranking) == 3
        
        # 每个策略都应该有排名
        ranked_strategies = [name for name, score in ranking]
        for strategy_name in strategies.keys():
            assert strategy_name in ranked_strategies
    
    async def test_memory_and_performance_efficiency(self):
        """测试内存和性能效率"""
        # 测试较长时间段的回测
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5),  # 5天数据
            initial_balance=Decimal("100000"),
            strategy_configs={
                "efficiency_test": {
                    "short_window": 10,
                    "long_window": 20,
                    "position_size": 0.1
                }
            }
        )
        
        start_time = datetime.utcnow()
        
        engine = BacktestEngine(config)
        engine.register_strategy("efficiency_test", create_sma_strategy)
        
        await engine.initialize()
        results = await engine.run_backtest("efficiency_test")
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        result = results["efficiency_test"]
        
        # 验证执行效率
        assert execution_time < 60  # 应该在60秒内完成
        assert result.total_events > 0
        assert result.processed_events > 0
        
        # 验证内存使用（简单检查数据结构不会无限增长）
        assert len(engine.market_data_cache) <= len(config.symbols) * len(config.timeframes)
        
        # 清理检查
        await engine.stop_backtest()
        assert len(engine.event_queue) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])