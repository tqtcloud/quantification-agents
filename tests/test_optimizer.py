import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List

from src.backtesting.optimizer import (
    ParameterOptimizer, OptimizationConfig, ParameterRange, OptimizationResult, optimize_strategy
)
from src.backtesting.backtest_engine import BacktestConfig
from src.backtesting.strategies import create_sma_strategy, create_buy_hold_strategy
from src.core.models import Order


class TestParameterRange:
    """测试参数范围"""
    
    def test_int_parameter_range(self):
        """测试整数参数范围"""
        param_range = ParameterRange(
            name="window_size",
            param_type="int",
            min_value=5,
            max_value=20,
            step=1
        )
        
        assert param_range.name == "window_size"
        assert param_range.param_type == "int"
        assert param_range.min_value == 5
        assert param_range.max_value == 20
        assert param_range.step == 1
    
    def test_float_parameter_range(self):
        """测试浮点数参数范围"""
        param_range = ParameterRange(
            name="position_size",
            param_type="float",
            min_value=0.05,
            max_value=0.5,
            step=0.05
        )
        
        assert param_range.name == "position_size"
        assert param_range.param_type == "float"
        assert param_range.min_value == 0.05
        assert param_range.max_value == 0.5
        assert param_range.step == 0.05
    
    def test_choice_parameter_range(self):
        """测试选择参数范围"""
        param_range = ParameterRange(
            name="strategy_type",
            param_type="choice",
            choices=["conservative", "moderate", "aggressive"]
        )
        
        assert param_range.name == "strategy_type"
        assert param_range.param_type == "choice"
        assert param_range.choices == ["conservative", "moderate", "aggressive"]
    
    def test_invalid_parameter_range(self):
        """测试无效参数范围"""
        # 缺少min_value和max_value
        with pytest.raises(ValueError):
            ParameterRange(
                name="test",
                param_type="int"
            )
        
        # choice类型缺少choices
        with pytest.raises(ValueError):
            ParameterRange(
                name="test",
                param_type="choice"
            )


class TestOptimizationConfig:
    """测试优化配置"""
    
    @pytest.fixture
    def base_config(self):
        """基础回测配置"""
        return BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            initial_balance=Decimal("10000")
        )
    
    @pytest.fixture
    def parameter_ranges(self):
        """参数范围"""
        return [
            ParameterRange("short_window", "int", 5, 15, 2),
            ParameterRange("long_window", "int", 20, 40, 5),
            ParameterRange("position_size", "float", 0.1, 0.3, 0.1)
        ]
    
    def test_default_optimization_config(self, base_config, parameter_ranges):
        """测试默认优化配置"""
        config = OptimizationConfig(
            base_config=base_config,
            strategy_name="test_strategy",
            strategy_function=create_sma_strategy,
            parameter_ranges=parameter_ranges
        )
        
        assert config.base_config == base_config
        assert config.strategy_name == "test_strategy"
        assert config.strategy_function == create_sma_strategy
        assert config.parameter_ranges == parameter_ranges
        assert config.optimization_method == "grid_search"
        assert config.objective_function == "sharpe_ratio"
        assert config.max_workers == 4
    
    def test_custom_optimization_config(self, base_config, parameter_ranges):
        """测试自定义优化配置"""
        config = OptimizationConfig(
            base_config=base_config,
            strategy_name="test_strategy",
            strategy_function=create_sma_strategy,
            parameter_ranges=parameter_ranges,
            optimization_method="random_search",
            objective_function="total_return",
            random_iterations=50,
            max_workers=2
        )
        
        assert config.optimization_method == "random_search"
        assert config.objective_function == "total_return"
        assert config.random_iterations == 50
        assert config.max_workers == 2


@pytest.mark.asyncio
class TestParameterOptimizer:
    """测试参数优化器"""
    
    @pytest.fixture
    def base_config(self):
        """基础配置"""
        return BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 6),  # 6小时数据
            initial_balance=Decimal("10000")
        )
    
    @pytest.fixture
    def parameter_ranges(self):
        """参数范围"""
        return [
            ParameterRange("short_window", "int", 3, 7, 2),  # [3, 5, 7]
            ParameterRange("long_window", "int", 10, 15, 5),  # [10, 15]
            ParameterRange("position_size", "float", 0.1, 0.2, 0.1)  # [0.1, 0.2]
        ]
    
    @pytest.fixture
    def optimization_config(self, base_config, parameter_ranges):
        """优化配置"""
        return OptimizationConfig(
            base_config=base_config,
            strategy_name="sma_strategy",
            strategy_function=create_sma_strategy,
            parameter_ranges=parameter_ranges,
            max_combinations=10,  # 限制组合数量
            max_workers=1  # 使用单线程避免复杂性
        )
    
    async def test_optimizer_initialization(self, optimization_config):
        """测试优化器初始化"""
        optimizer = ParameterOptimizer(optimization_config)
        
        assert optimizer.config == optimization_config
        assert len(optimizer.results) == 0
    
    def test_parameter_combinations_generation(self, optimization_config):
        """测试参数组合生成"""
        optimizer = ParameterOptimizer(optimization_config)
        
        combinations = optimizer._generate_parameter_combinations()
        
        # 应该有 3 * 2 * 2 = 12 个组合
        assert len(combinations) == 12
        
        # 检查组合内容
        for combo in combinations:
            assert "short_window" in combo
            assert "long_window" in combo
            assert "position_size" in combo
            assert combo["short_window"] in [3, 5, 7]
            assert combo["long_window"] in [10, 15]
            assert combo["position_size"] in [0.1, 0.2]
    
    def test_random_parameters_generation(self, optimization_config):
        """测试随机参数生成"""
        optimizer = ParameterOptimizer(optimization_config)
        
        for _ in range(10):
            params = optimizer._generate_random_parameters()
            
            assert "short_window" in params
            assert "long_window" in params
            assert "position_size" in params
            assert 3 <= params["short_window"] <= 7
            assert 10 <= params["long_window"] <= 15
            assert 0.1 <= params["position_size"] <= 0.2
    
    async def test_single_backtest_execution(self, optimization_config):
        """测试单个回测执行"""
        optimizer = ParameterOptimizer(optimization_config)
        
        test_params = {
            "short_window": 5,
            "long_window": 10,
            "position_size": 0.1
        }
        
        result = await optimizer._run_single_backtest(test_params)
        
        assert "parameters" in result
        assert "score" in result
        assert result["parameters"] == test_params
        assert isinstance(result["score"], (int, float, type(None)))
        
        if result["score"] is not None:
            assert "backtest_result" in result
            assert "performance_metrics" in result
    
    def test_objective_score_calculation(self, optimization_config):
        """测试目标函数分数计算"""
        optimizer = ParameterOptimizer(optimization_config)
        
        # 模拟回测结果
        from src.backtesting.backtest_engine import BacktestResult
        
        mock_result = BacktestResult(
            config=optimization_config.base_config,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            duration_seconds=3600,
            performance_metrics={
                "sharpe_ratio": 1.5,
                "total_return": 0.1,
                "calmar_ratio": 2.0,
                "max_drawdown": 0.05
            },
            strategy_results={},
            trade_records=[],
            daily_stats=[],
            equity_curve=[],
            total_events=1000,
            processed_events=1000
        )
        
        # 测试不同目标函数
        optimizer.config.objective_function = "sharpe_ratio"
        score1 = optimizer._calculate_objective_score(mock_result)
        assert score1 == 1.5
        
        optimizer.config.objective_function = "total_return"
        score2 = optimizer._calculate_objective_score(mock_result)
        assert score2 == 0.1
        
        optimizer.config.objective_function = "calmar_ratio"
        score3 = optimizer._calculate_objective_score(mock_result)
        assert score3 == 2.0
        
        # 测试复合分数
        optimizer.config.objective_function = "composite"
        score4 = optimizer._calculate_objective_score(mock_result)
        assert isinstance(score4, float)
    
    async def test_grid_search_optimization(self, optimization_config):
        """测试网格搜索优化"""
        optimization_config.optimization_method = "grid_search"
        optimization_config.max_combinations = 6  # 限制组合数量
        
        optimizer = ParameterOptimizer(optimization_config)
        
        result = await optimizer.optimize()
        
        assert isinstance(result, OptimizationResult)
        assert "short_window" in result.best_parameters
        assert "long_window" in result.best_parameters
        assert "position_size" in result.best_parameters
        assert isinstance(result.best_score, float)
        assert result.total_combinations <= 6
        assert result.successful_runs >= 0
        assert result.optimization_time > 0
    
    async def test_random_search_optimization(self, optimization_config):
        """测试随机搜索优化"""
        optimization_config.optimization_method = "random_search"
        optimization_config.random_iterations = 5
        
        optimizer = ParameterOptimizer(optimization_config)
        
        result = await optimizer.optimize()
        
        assert isinstance(result, OptimizationResult)
        assert result.total_combinations == 5
        assert len(result.all_results) == 5
    
    def test_parameter_sensitivity_calculation(self, optimization_config):
        """测试参数敏感性计算"""
        optimizer = ParameterOptimizer(optimization_config)
        
        # 模拟一些结果
        optimizer.results = [
            {"parameters": {"short_window": 3, "long_window": 10, "position_size": 0.1}, "score": 0.5},
            {"parameters": {"short_window": 5, "long_window": 10, "position_size": 0.1}, "score": 0.8},
            {"parameters": {"short_window": 7, "long_window": 10, "position_size": 0.1}, "score": 1.2},
            {"parameters": {"short_window": 3, "long_window": 15, "position_size": 0.1}, "score": 0.6},
            {"parameters": {"short_window": 5, "long_window": 15, "position_size": 0.1}, "score": 0.9},
            {"parameters": {"short_window": 7, "long_window": 15, "position_size": 0.1}, "score": 1.0},
        ]
        
        sensitivity = optimizer._calculate_parameter_sensitivity()
        
        assert isinstance(sensitivity, dict)
        assert "short_window" in sensitivity
        assert "long_window" in sensitivity
        assert "position_size" in sensitivity
        
        # short_window应该有较高的敏感性（因为它与score正相关）
        assert sensitivity["short_window"] > 0
    
    async def test_parallel_backtest_execution(self, optimization_config):
        """测试并行回测执行"""
        optimization_config.max_workers = 2
        
        optimizer = ParameterOptimizer(optimization_config)
        
        parameter_combinations = [
            {"short_window": 3, "long_window": 10, "position_size": 0.1},
            {"short_window": 5, "long_window": 10, "position_size": 0.1},
            {"short_window": 7, "long_window": 10, "position_size": 0.1}
        ]
        
        results = await optimizer._run_parallel_backtests(parameter_combinations)
        
        assert len(results) == 3
        for result in results:
            assert "parameters" in result
            assert "score" in result
    
    async def test_error_handling_in_optimization(self, optimization_config):
        """测试优化中的错误处理"""
        # 使用一个会出错的策略函数
        def faulty_strategy(context):
            raise ValueError("Test error")
        
        optimization_config.strategy_function = faulty_strategy
        
        optimizer = ParameterOptimizer(optimization_config)
        
        # 运行单个回测应该处理错误
        test_params = {"short_window": 5, "long_window": 10, "position_size": 0.1}
        result = await optimizer._run_single_backtest(test_params)
        
        assert result["score"] is None
        assert "error" in result
    
    async def test_optimization_result_structure(self, optimization_config):
        """测试优化结果结构"""
        optimization_config.max_combinations = 3
        
        optimizer = ParameterOptimizer(optimization_config)
        
        result = await optimizer.optimize()
        
        # 检查结果结构
        assert hasattr(result, "best_parameters")
        assert hasattr(result, "best_score")
        assert hasattr(result, "best_backtest_result")
        assert hasattr(result, "all_results")
        assert hasattr(result, "total_combinations")
        assert hasattr(result, "successful_runs")
        assert hasattr(result, "failed_runs")
        assert hasattr(result, "optimization_time")
        assert hasattr(result, "parameter_sensitivity")
        
        # 检查类型
        assert isinstance(result.best_parameters, dict)
        assert isinstance(result.best_score, (int, float))
        assert isinstance(result.all_results, list)
        assert isinstance(result.total_combinations, int)
        assert isinstance(result.optimization_time, float)
        assert isinstance(result.parameter_sensitivity, dict)


class TestOptimizationUtilities:
    """测试优化工具函数"""
    
    @pytest.mark.asyncio
    async def test_optimize_strategy_function(self):
        """测试便捷优化函数"""
        base_config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1, 3),  # 3小时
            initial_balance=Decimal("10000")
        )
        
        parameter_ranges = [
            ParameterRange("position_size", "float", 0.1, 0.2, 0.1)
        ]
        
        result = await optimize_strategy(
            strategy_function=create_buy_hold_strategy,
            parameter_ranges=parameter_ranges,
            base_config=base_config,
            strategy_name="test_buy_hold",
            method="grid_search",
            objective="total_return"
        )
        
        assert isinstance(result, OptimizationResult)
        assert "position_size" in result.best_parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])