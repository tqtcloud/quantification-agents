import asyncio
import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
import json

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult
from src.utils.logger import LoggerMixin


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    param_type: str  # "int", "float", "choice"
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    step: Optional[Union[int, float]] = None
    
    def __post_init__(self):
        if self.param_type in ["int", "float"] and (self.min_value is None or self.max_value is None):
            raise ValueError(f"Parameter {self.name}: min_value and max_value required for {self.param_type}")
        if self.param_type == "choice" and not self.choices:
            raise ValueError(f"Parameter {self.name}: choices required for choice type")


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 基础配置
    base_config: BacktestConfig
    strategy_name: str
    strategy_function: Callable
    
    # 参数定义
    parameter_ranges: List[ParameterRange]
    
    # 优化配置
    optimization_method: str = "grid_search"  # "grid_search", "random_search", "bayesian"
    objective_function: str = "sharpe_ratio"  # "sharpe_ratio", "total_return", "calmar_ratio", "custom"
    custom_objective: Optional[Callable] = None
    
    # 网格搜索配置
    max_combinations: int = 1000
    
    # 随机搜索配置
    random_iterations: int = 100
    
    # 贝叶斯优化配置
    bayesian_iterations: int = 50
    acquisition_function: str = "expected_improvement"  # "expected_improvement", "upper_confidence_bound"
    
    # 并行配置
    max_workers: int = 4
    use_multiprocessing: bool = False
    
    # 输出配置
    save_all_results: bool = True
    results_file: Optional[str] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    best_parameters: Dict[str, Any]
    best_score: float
    best_backtest_result: BacktestResult
    
    # 所有结果
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # 统计信息
    total_combinations: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    optimization_time: float = 0
    
    # 参数敏感性分析
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)


class ParameterOptimizer(LoggerMixin):
    """参数优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []
        
    async def optimize(self) -> OptimizationResult:
        """执行参数优化"""
        start_time = datetime.utcnow()
        self.log_info(f"Starting parameter optimization using {self.config.optimization_method}")
        
        if self.config.optimization_method == "grid_search":
            result = await self._grid_search_optimization()
        elif self.config.optimization_method == "random_search":
            result = await self._random_search_optimization()
        elif self.config.optimization_method == "bayesian":
            result = await self._bayesian_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        end_time = datetime.utcnow()
        result.optimization_time = (end_time - start_time).total_seconds()
        
        # 计算参数敏感性
        result.parameter_sensitivity = self._calculate_parameter_sensitivity()
        
        # 保存结果
        if self.config.results_file:
            await self._save_results(result)
        
        self.log_info(f"Optimization completed in {result.optimization_time:.2f} seconds")
        self.log_info(f"Best parameters: {result.best_parameters}")
        self.log_info(f"Best score: {result.best_score:.4f}")
        
        return result
    
    async def _grid_search_optimization(self) -> OptimizationResult:
        """网格搜索优化"""
        self.log_info("Running grid search optimization...")
        
        # 生成参数组合
        parameter_combinations = self._generate_parameter_combinations()
        
        # 限制组合数量
        if len(parameter_combinations) > self.config.max_combinations:
            self.log_warning(f"Too many combinations ({len(parameter_combinations)}), "
                           f"limiting to {self.config.max_combinations}")
            random.shuffle(parameter_combinations)
            parameter_combinations = parameter_combinations[:self.config.max_combinations]
        
        self.log_info(f"Testing {len(parameter_combinations)} parameter combinations")
        
        # 并行执行回测
        results = await self._run_parallel_backtests(parameter_combinations)
        
        # 找到最优结果
        best_result = max(results, key=lambda x: x["score"])
        
        return OptimizationResult(
            best_parameters=best_result["parameters"],
            best_score=best_result["score"],
            best_backtest_result=best_result["backtest_result"],
            all_results=results,
            total_combinations=len(parameter_combinations),
            successful_runs=len([r for r in results if r["score"] is not None]),
            failed_runs=len([r for r in results if r["score"] is None])
        )
    
    async def _random_search_optimization(self) -> OptimizationResult:
        """随机搜索优化"""
        self.log_info("Running random search optimization...")
        
        parameter_combinations = []
        for _ in range(self.config.random_iterations):
            params = self._generate_random_parameters()
            parameter_combinations.append(params)
        
        # 并行执行回测
        results = await self._run_parallel_backtests(parameter_combinations)
        
        # 找到最优结果
        valid_results = [r for r in results if r["score"] is not None]
        if not valid_results:
            raise RuntimeError("No valid results from random search")
            
        best_result = max(valid_results, key=lambda x: x["score"])
        
        return OptimizationResult(
            best_parameters=best_result["parameters"],
            best_score=best_result["score"],
            best_backtest_result=best_result["backtest_result"],
            all_results=results,
            total_combinations=len(parameter_combinations),
            successful_runs=len(valid_results),
            failed_runs=len(results) - len(valid_results)
        )
    
    async def _bayesian_optimization(self) -> OptimizationResult:
        """贝叶斯优化"""
        self.log_info("Running Bayesian optimization...")
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            self.log_error("scikit-optimize not installed, falling back to random search")
            return await self._random_search_optimization()
        
        # 定义搜索空间
        dimensions = []
        param_names = []
        
        for param_range in self.config.parameter_ranges:
            param_names.append(param_range.name)
            
            if param_range.param_type == "int":
                dimensions.append(Integer(param_range.min_value, param_range.max_value, name=param_range.name))
            elif param_range.param_type == "float":
                dimensions.append(Real(param_range.min_value, param_range.max_value, name=param_range.name))
            elif param_range.param_type == "choice":
                dimensions.append(Categorical(param_range.choices, name=param_range.name))
        
        # 定义目标函数
        @use_named_args(dimensions)
        async def objective(**params):
            try:
                result = await self._run_single_backtest(params)
                return -result["score"] if result["score"] is not None else 1e6  # 最小化负分数
            except Exception as e:
                self.log_error(f"Error in objective function: {e}")
                return 1e6
        
        # 包装异步函数为同步函数
        def sync_objective(**params):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(objective(**params))
            finally:
                loop.close()
        
        # 运行贝叶斯优化
        result = gp_minimize(
            func=sync_objective,
            dimensions=dimensions,
            n_calls=self.config.bayesian_iterations,
            acq_func=self.config.acquisition_function,
            random_state=42
        )
        
        # 提取最优参数
        best_params = dict(zip(param_names, result.x))
        
        # 运行最优参数的完整回测
        best_result = await self._run_single_backtest(best_params)
        
        # 收集所有结果
        all_results = []
        for i, (params, score) in enumerate(zip(result.x_iters, result.func_vals)):
            param_dict = dict(zip(param_names, params))
            all_results.append({
                "parameters": param_dict,
                "score": -score,  # 转回正分数
                "iteration": i
            })
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_result["score"],
            best_backtest_result=best_result["backtest_result"],
            all_results=all_results,
            total_combinations=self.config.bayesian_iterations,
            successful_runs=len([r for r in all_results if r["score"] > -1e5]),
            failed_runs=len([r for r in all_results if r["score"] <= -1e5])
        )
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        param_lists = []
        param_names = []
        
        for param_range in self.config.parameter_ranges:
            param_names.append(param_range.name)
            
            if param_range.param_type == "int":
                step = param_range.step or 1
                values = list(range(param_range.min_value, param_range.max_value + 1, step))
            elif param_range.param_type == "float":
                step = param_range.step or 0.1
                values = []
                val = param_range.min_value
                while val <= param_range.max_value:
                    values.append(round(val, 6))
                    val += step
            elif param_range.param_type == "choice":
                values = param_range.choices
            else:
                raise ValueError(f"Unknown parameter type: {param_range.param_type}")
            
            param_lists.append(values)
        
        # 生成笛卡尔积
        combinations = []
        for combo in itertools.product(*param_lists):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """生成随机参数"""
        params = {}
        
        for param_range in self.config.parameter_ranges:
            if param_range.param_type == "int":
                value = random.randint(param_range.min_value, param_range.max_value)
            elif param_range.param_type == "float":
                value = random.uniform(param_range.min_value, param_range.max_value)
            elif param_range.param_type == "choice":
                value = random.choice(param_range.choices)
            else:
                raise ValueError(f"Unknown parameter type: {param_range.param_type}")
            
            params[param_range.name] = value
        
        return params
    
    async def _run_parallel_backtests(self, parameter_combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并行运行回测"""
        if self.config.use_multiprocessing:
            # 使用进程池（适合CPU密集型任务）
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                tasks = [
                    executor.submit(self._run_single_backtest_sync, params)
                    for params in parameter_combinations
                ]
                results = []
                for task in tasks:
                    try:
                        result = task.result()
                        results.append(result)
                    except Exception as e:
                        self.log_error(f"Backtest failed: {e}")
                        results.append({"parameters": {}, "score": None, "error": str(e)})
        else:
            # 使用协程（适合I/O密集型任务）
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def run_with_semaphore(params):
                async with semaphore:
                    return await self._run_single_backtest(params)
            
            tasks = [run_with_semaphore(params) for params in parameter_combinations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.log_error(f"Backtest failed: {result}")
                    processed_results.append({
                        "parameters": parameter_combinations[i],
                        "score": None,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            results = processed_results
        
        return results
    
    def _run_single_backtest_sync(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """同步运行单个回测（用于进程池）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_single_backtest(parameters))
        finally:
            loop.close()
    
    async def _run_single_backtest(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个回测"""
        try:
            # 创建配置副本
            config = BacktestConfig(
                start_date=self.config.base_config.start_date,
                end_date=self.config.base_config.end_date,
                initial_balance=self.config.base_config.initial_balance,
                strategy_configs={self.config.strategy_name: parameters},
                symbols=self.config.base_config.symbols,
                timeframes=self.config.base_config.timeframes,
                commission_rate=self.config.base_config.commission_rate,
                slippage_bps=self.config.base_config.slippage_bps
            )
            
            # 创建回测引擎
            engine = BacktestEngine(config)
            engine.register_strategy(self.config.strategy_name, self.config.strategy_function)
            
            await engine.initialize()
            
            # 运行回测
            results = await engine.run_backtest(self.config.strategy_name)
            backtest_result = results[self.config.strategy_name]
            
            # 计算目标函数值
            score = self._calculate_objective_score(backtest_result)
            
            result = {
                "parameters": parameters,
                "score": score,
                "backtest_result": backtest_result,
                "performance_metrics": backtest_result.performance_metrics
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.log_error(f"Backtest failed for parameters {parameters}: {e}")
            return {
                "parameters": parameters,
                "score": None,
                "error": str(e)
            }
    
    def _calculate_objective_score(self, backtest_result: BacktestResult) -> float:
        """计算目标函数分数"""
        metrics = backtest_result.performance_metrics
        
        if self.config.objective_function == "sharpe_ratio":
            return metrics.get("sharpe_ratio", 0)
        elif self.config.objective_function == "total_return":
            return metrics.get("total_return", 0)
        elif self.config.objective_function == "calmar_ratio":
            return metrics.get("calmar_ratio", 0)
        elif self.config.objective_function == "custom" and self.config.custom_objective:
            return self.config.custom_objective(backtest_result)
        else:
            # 默认使用复合分数
            sharpe = metrics.get("sharpe_ratio", 0)
            total_return = metrics.get("total_return", 0)
            max_drawdown = abs(metrics.get("max_drawdown", 0))
            
            # 复合分数 = 夏普比率 * 0.5 + 总收益 * 0.3 - 最大回撤 * 0.2
            score = sharpe * 0.5 + total_return * 0.3 - max_drawdown * 0.2
            return score
    
    def _calculate_parameter_sensitivity(self) -> Dict[str, float]:
        """计算参数敏感性"""
        if len(self.results) < 10:
            return {}
        
        sensitivity = {}
        
        for param_range in self.config.parameter_ranges:
            param_name = param_range.name
            
            # 收集该参数的所有值和对应分数
            param_values = []
            scores = []
            
            for result in self.results:
                if result["score"] is not None:
                    param_values.append(result["parameters"].get(param_name))
                    scores.append(result["score"])
            
            if len(param_values) > 5:
                # 计算相关系数作为敏感性指标
                correlation = np.corrcoef(param_values, scores)[0, 1]
                sensitivity[param_name] = abs(correlation) if not np.isnan(correlation) else 0
            else:
                sensitivity[param_name] = 0
        
        return sensitivity
    
    async def _save_results(self, result: OptimizationResult):
        """保存优化结果"""
        output_data = {
            "optimization_config": {
                "method": self.config.optimization_method,
                "objective_function": self.config.objective_function,
                "parameter_ranges": [
                    {
                        "name": pr.name,
                        "type": pr.param_type,
                        "min_value": pr.min_value,
                        "max_value": pr.max_value,
                        "choices": pr.choices
                    }
                    for pr in self.config.parameter_ranges
                ]
            },
            "best_result": {
                "parameters": result.best_parameters,
                "score": result.best_score,
                "performance_metrics": result.best_backtest_result.performance_metrics
            },
            "optimization_stats": {
                "total_combinations": result.total_combinations,
                "successful_runs": result.successful_runs,
                "failed_runs": result.failed_runs,
                "optimization_time": result.optimization_time
            },
            "parameter_sensitivity": result.parameter_sensitivity
        }
        
        if self.config.save_all_results:
            output_data["all_results"] = [
                {
                    "parameters": r["parameters"],
                    "score": r["score"],
                    "performance_metrics": r.get("performance_metrics", {})
                }
                for r in result.all_results
                if r["score"] is not None
            ]
        
        with open(self.config.results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.log_info(f"Optimization results saved to {self.config.results_file}")


# 便捷函数
async def optimize_strategy(strategy_function: Callable, 
                          parameter_ranges: List[ParameterRange],
                          base_config: BacktestConfig,
                          strategy_name: str = "optimized_strategy",
                          method: str = "grid_search",
                          objective: str = "sharpe_ratio") -> OptimizationResult:
    """便捷的策略优化函数"""
    
    optimization_config = OptimizationConfig(
        base_config=base_config,
        strategy_name=strategy_name,
        strategy_function=strategy_function,
        parameter_ranges=parameter_ranges,
        optimization_method=method,
        objective_function=objective
    )
    
    optimizer = ParameterOptimizer(optimization_config)
    return await optimizer.optimize()