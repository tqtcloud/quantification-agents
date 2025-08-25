"""
投资组合优化引擎
实现高级优化算法和约束管理
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm
import cvxpy as cp

from src.utils.logger import LoggerMixin


class OptimizationType(Enum):
    """优化类型"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    EQUAL_WEIGHT = "equal_weight"
    KELLY_CRITERION = "kelly_criterion"
    CVaR_OPTIMIZATION = "cvar_optimization"


@dataclass
class OptimizationConstraints:
    """优化约束"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_turnover: Optional[float] = None
    min_positions: Optional[int] = None
    max_positions: Optional[int] = None
    sector_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    leverage_limit: float = 1.0
    transaction_costs: Dict[str, float] = field(default_factory=dict)
    holding_costs: Dict[str, float] = field(default_factory=dict)
    
    # 风险约束
    max_var: Optional[float] = None
    max_cvar: Optional[float] = None
    max_volatility: Optional[float] = None
    max_tracking_error: Optional[float] = None
    
    # 流动性约束
    max_adv_participation: Dict[str, float] = field(default_factory=dict)
    min_market_cap: Optional[float] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    allocations: Dict[str, float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    turnover: Optional[float] = None
    transaction_costs: Optional[float] = None
    
    success: bool = True
    message: str = ""
    optimization_time: float = 0.0
    iterations: int = 0
    
    # 额外指标
    diversification_ratio: Optional[float] = None
    effective_assets: Optional[int] = None
    concentration_index: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class OptimizationEngine(LoggerMixin):
    """高级优化引擎"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 use_cvxpy: bool = True):
        """初始化优化引擎"""
        self.risk_free_rate = risk_free_rate
        self.use_cvxpy = use_cvxpy
        
        # 优化器缓存
        self._covariance_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._returns_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        
        self.log_info("Optimization Engine initialized")
    
    async def optimize(self,
                      returns_data: Dict[str, List[float]],
                      optimization_type: OptimizationType,
                      constraints: OptimizationConstraints = None,
                      current_weights: Dict[str, float] = None,
                      benchmark_weights: Dict[str, float] = None,
                      views: Dict[str, float] = None,
                      confidence: Dict[str, float] = None) -> OptimizationResult:
        """执行优化"""
        start_time = datetime.now()
        
        try:
            # 准备数据
            symbols = list(returns_data.keys())
            returns_matrix = self._prepare_returns_matrix(returns_data)
            
            if returns_matrix.shape[0] < 2:
                return self._create_error_result("Insufficient historical data")
            
            # 计算期望收益和协方差矩阵
            expected_returns = self._calculate_expected_returns(
                returns_matrix, symbols, views, confidence
            )
            covariance_matrix = self._calculate_covariance_matrix(returns_matrix)
            
            # 执行优化
            if optimization_type == OptimizationType.MEAN_VARIANCE:
                result = await self._optimize_mean_variance(
                    expected_returns, covariance_matrix, symbols, constraints
                )
            elif optimization_type == OptimizationType.RISK_PARITY:
                result = await self._optimize_risk_parity(
                    covariance_matrix, symbols, constraints
                )
            elif optimization_type == OptimizationType.BLACK_LITTERMAN:
                result = await self._optimize_black_litterman(
                    expected_returns, covariance_matrix, symbols,
                    benchmark_weights, views, confidence, constraints
                )
            elif optimization_type == OptimizationType.MINIMUM_VARIANCE:
                result = await self._optimize_minimum_variance(
                    covariance_matrix, symbols, constraints
                )
            elif optimization_type == OptimizationType.MAXIMUM_DIVERSIFICATION:
                result = await self._optimize_maximum_diversification(
                    covariance_matrix, symbols, constraints
                )
            elif optimization_type == OptimizationType.KELLY_CRITERION:
                result = await self._optimize_kelly_criterion(
                    returns_matrix, symbols, constraints
                )
            elif optimization_type == OptimizationType.CVaR_OPTIMIZATION:
                result = await self._optimize_cvar(
                    returns_matrix, expected_returns, symbols, constraints
                )
            elif optimization_type == OptimizationType.EQUAL_WEIGHT:
                result = self._create_equal_weight_result(symbols)
            else:
                return self._create_error_result(f"Unsupported optimization type: {optimization_type}")
            
            # 计算额外指标
            self._calculate_additional_metrics(result, returns_matrix, covariance_matrix)
            
            # 记录性能
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time
            
            self.log_info(f"Optimization completed in {optimization_time:.3f}s")
            return result
            
        except Exception as e:
            self.log_error(f"Optimization failed: {e}")
            return self._create_error_result(str(e))
    
    async def _optimize_mean_variance(self,
                                     expected_returns: np.ndarray,
                                     covariance_matrix: np.ndarray,
                                     symbols: List[str],
                                     constraints: OptimizationConstraints = None) -> OptimizationResult:
        """均值方差优化"""
        n_assets = len(symbols)
        
        if self.use_cvxpy:
            # 使用CVXPY求解
            weights = cp.Variable(n_assets)
            
            # 目标函数：最大化夏普比率
            portfolio_return = expected_returns.T @ weights
            portfolio_risk = cp.quad_form(weights, covariance_matrix)
            
            # 约束条件
            constraints_list = [cp.sum(weights) == 1.0]
            
            # 添加权重约束
            if constraints:
                if constraints.min_weight > 0:
                    constraints_list.append(weights >= constraints.min_weight)
                else:
                    constraints_list.append(weights >= 0)
                
                if constraints.max_weight < 1:
                    constraints_list.append(weights <= constraints.max_weight)
                
                # 风险约束
                if constraints.max_volatility:
                    constraints_list.append(cp.sqrt(portfolio_risk) <= constraints.max_volatility)
            else:
                constraints_list.append(weights >= 0)
            
            # 定义问题（最大化夏普比率的替代形式）
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
            problem = cp.Problem(objective, constraints_list)
            
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    optimal_weights = weights.value
                    
                    # 清理小权重
                    optimal_weights[optimal_weights < 1e-6] = 0
                    optimal_weights = optimal_weights / optimal_weights.sum()
                    
                    allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0}
                    
                    return OptimizationResult(
                        allocations=allocations,
                        expected_return=float(expected_returns.T @ optimal_weights),
                        risk=float(np.sqrt(optimal_weights.T @ covariance_matrix @ optimal_weights)),
                        sharpe_ratio=self._calculate_sharpe_ratio(expected_returns, covariance_matrix, optimal_weights),
                        success=True,
                        message="Mean-variance optimization successful"
                    )
                else:
                    raise Exception(f"Optimization failed with status: {problem.status}")
                    
            except Exception as e:
                self.log_warning(f"CVXPY optimization failed: {e}, falling back to scipy")
        
        # 回退到scipy优化
        return await self._optimize_mean_variance_scipy(expected_returns, covariance_matrix, symbols, constraints)
    
    async def _optimize_mean_variance_scipy(self,
                                          expected_returns: np.ndarray,
                                          covariance_matrix: np.ndarray,
                                          symbols: List[str],
                                          constraints: OptimizationConstraints = None) -> OptimizationResult:
        """使用scipy的均值方差优化"""
        n_assets = len(symbols)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            # 最大化夏普比率（最小化负夏普比率）
            if portfolio_variance <= 0:
                return 1e6
            sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
            return -sharpe
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # 权重和为1
        ]
        
        # 边界条件
        if constraints:
            min_w = constraints.min_weight
            max_w = constraints.max_weight
        else:
            min_w, max_w = 0.0, 1.0
        
        bounds = tuple((min_w, max_w) for _ in range(n_assets))
        
        # 初始猜测
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_weights[optimal_weights < 1e-6] = 0
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0}
            
            return OptimizationResult(
                allocations=allocations,
                expected_return=float(np.dot(optimal_weights, expected_returns)),
                risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                sharpe_ratio=self._calculate_sharpe_ratio(expected_returns, covariance_matrix, optimal_weights),
                iterations=result.nit,
                success=True,
                message="Mean-variance optimization successful"
            )
        else:
            return self._create_error_result("Mean-variance optimization failed")
    
    async def _optimize_risk_parity(self,
                                   covariance_matrix: np.ndarray,
                                   symbols: List[str],
                                   constraints: OptimizationConstraints = None) -> OptimizationResult:
        """风险平价优化"""
        n_assets = len(symbols)
        
        def risk_contribution(weights):
            """计算风险贡献度"""
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            if portfolio_variance <= 0:
                return np.ones(n_assets) / n_assets
            
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            return contrib
        
        def objective(weights):
            """目标函数：最小化风险贡献的方差"""
            contrib = risk_contribution(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # 边界条件
        bounds = tuple((1e-6, 1) for _ in range(n_assets))
        if constraints and constraints.max_weight < 1:
            bounds = tuple((1e-6, constraints.max_weight) for _ in range(n_assets))
        
        # 初始猜测
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 1e-6}
            
            return OptimizationResult(
                allocations=allocations,
                expected_return=0.0,  # 风险平价不关注收益
                risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                sharpe_ratio=0.0,
                iterations=result.nit,
                success=True,
                message="Risk parity optimization successful"
            )
        else:
            return self._create_error_result("Risk parity optimization failed")
    
    async def _optimize_black_litterman(self,
                                       expected_returns: np.ndarray,
                                       covariance_matrix: np.ndarray,
                                       symbols: List[str],
                                       benchmark_weights: Dict[str, float] = None,
                                       views: Dict[str, float] = None,
                                       confidence: Dict[str, float] = None,
                                       constraints: OptimizationConstraints = None) -> OptimizationResult:
        """Black-Litterman优化"""
        n_assets = len(symbols)
        
        # 如果没有基准权重，使用市值加权或等权重
        if benchmark_weights is None:
            w_market = np.ones(n_assets) / n_assets
        else:
            w_market = np.array([benchmark_weights.get(symbol, 0) for symbol in symbols])
            w_market = w_market / w_market.sum()
        
        # 估算风险厌恶系数
        market_return = np.dot(w_market, expected_returns)
        market_variance = np.dot(w_market, np.dot(covariance_matrix, w_market))
        risk_aversion = market_return / market_variance if market_variance > 0 else 1.0
        
        # 隐含预期收益
        implied_returns = risk_aversion * np.dot(covariance_matrix, w_market)
        
        # 如果有观点，应用Black-Litterman公式
        if views and confidence:
            # 构建观点矩阵P和观点向量Q
            view_symbols = list(views.keys())
            P = np.zeros((len(view_symbols), n_assets))
            Q = np.zeros(len(view_symbols))
            Omega = np.zeros((len(view_symbols), len(view_symbols)))
            
            for i, symbol in enumerate(view_symbols):
                if symbol in symbols:
                    symbol_idx = symbols.index(symbol)
                    P[i, symbol_idx] = 1.0
                    Q[i] = views[symbol]
                    # 观点不确定性矩阵
                    view_confidence = confidence.get(symbol, 0.5)
                    Omega[i, i] = (1 - view_confidence) * covariance_matrix[symbol_idx, symbol_idx]
            
            # Black-Litterman公式
            tau = 0.05  # 不确定性标量
            
            try:
                # 计算新的期望收益
                M1 = np.linalg.inv(tau * covariance_matrix)
                M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
                M3 = np.dot(np.linalg.inv(tau * covariance_matrix), implied_returns)
                M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
                
                mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
                
                # 计算新的协方差矩阵
                cov_bl = np.linalg.inv(M1 + M2)
                
                # 使用新的期望收益和协方差进行优化
                return await self._optimize_mean_variance(mu_bl, cov_bl, symbols, constraints)
                
            except np.linalg.LinAlgError:
                self.log_warning("Black-Litterman matrix inversion failed, using implied returns")
                return await self._optimize_mean_variance(implied_returns, covariance_matrix, symbols, constraints)
        
        # 没有观点时，使用隐含收益进行优化
        return await self._optimize_mean_variance(implied_returns, covariance_matrix, symbols, constraints)
    
    async def _optimize_minimum_variance(self,
                                        covariance_matrix: np.ndarray,
                                        symbols: List[str],
                                        constraints: OptimizationConstraints = None) -> OptimizationResult:
        """最小方差优化"""
        n_assets = len(symbols)
        
        if self.use_cvxpy:
            try:
                weights = cp.Variable(n_assets)
                
                # 目标函数：最小化方差
                objective = cp.Minimize(cp.quad_form(weights, covariance_matrix))
                
                # 约束条件
                constraints_list = [
                    cp.sum(weights) == 1.0,
                    weights >= 0
                ]
                
                if constraints and constraints.max_weight < 1:
                    constraints_list.append(weights <= constraints.max_weight)
                
                problem = cp.Problem(objective, constraints_list)
                problem.solve(solver=cp.OSQP, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    optimal_weights = weights.value
                    optimal_weights[optimal_weights < 1e-6] = 0
                    optimal_weights = optimal_weights / optimal_weights.sum()
                    
                    allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0}
                    
                    return OptimizationResult(
                        allocations=allocations,
                        expected_return=0.0,
                        risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                        sharpe_ratio=0.0,
                        success=True,
                        message="Minimum variance optimization successful"
                    )
                    
            except Exception as e:
                self.log_warning(f"CVXPY minimum variance failed: {e}")
        
        # 回退到解析解
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            ones = np.ones((n_assets, 1))
            
            optimal_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            optimal_weights = optimal_weights.flatten()
            
            # 应用约束
            if constraints:
                optimal_weights = np.clip(optimal_weights, constraints.min_weight, constraints.max_weight)
                optimal_weights = optimal_weights / optimal_weights.sum()
            
            allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 1e-6}
            
            return OptimizationResult(
                allocations=allocations,
                expected_return=0.0,
                risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                sharpe_ratio=0.0,
                success=True,
                message="Minimum variance optimization successful (analytical solution)"
            )
            
        except np.linalg.LinAlgError:
            return self._create_error_result("Minimum variance optimization failed: singular matrix")
    
    async def _optimize_maximum_diversification(self,
                                              covariance_matrix: np.ndarray,
                                              symbols: List[str],
                                              constraints: OptimizationConstraints = None) -> OptimizationResult:
        """最大分散化优化"""
        n_assets = len(symbols)
        
        # 个股波动率
        individual_vols = np.sqrt(np.diag(covariance_matrix))
        
        def diversification_ratio(weights):
            """计算分散化比率"""
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        def objective(weights):
            """目标函数：最大化分散化比率"""
            return -diversification_ratio(weights)
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # 边界条件
        bounds = tuple((0, 1) for _ in range(n_assets))
        if constraints and constraints.max_weight < 1:
            bounds = tuple((0, constraints.max_weight) for _ in range(n_assets))
        
        # 初始猜测
        initial_weights = np.ones(n_assets) / n_assets
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_weights = optimal_weights / optimal_weights.sum()
            
            allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 1e-6}
            
            return OptimizationResult(
                allocations=allocations,
                expected_return=0.0,
                risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                sharpe_ratio=0.0,
                diversification_ratio=diversification_ratio(optimal_weights),
                iterations=result.nit,
                success=True,
                message="Maximum diversification optimization successful"
            )
        else:
            return self._create_error_result("Maximum diversification optimization failed")
    
    async def _optimize_kelly_criterion(self,
                                       returns_matrix: np.ndarray,
                                       symbols: List[str],
                                       constraints: OptimizationConstraints = None) -> OptimizationResult:
        """Kelly准则优化"""
        n_assets, n_periods = returns_matrix.shape
        
        if n_periods < 2:
            return self._create_error_result("Insufficient data for Kelly optimization")
        
        # 计算期望收益和协方差
        expected_returns = np.mean(returns_matrix, axis=1)
        covariance_matrix = np.cov(returns_matrix)
        
        try:
            # Kelly公式：f = C^-1 * μ，其中C是协方差矩阵，μ是期望收益
            inv_cov = np.linalg.inv(covariance_matrix)
            kelly_weights = np.dot(inv_cov, expected_returns)
            
            # 标准化权重（Kelly权重可能为负或大于1）
            if np.sum(kelly_weights) <= 0:
                # 如果总权重为负，使用等权重
                optimal_weights = np.ones(n_assets) / n_assets
            else:
                # 标准化权重
                optimal_weights = kelly_weights / np.sum(kelly_weights)
                
                # 应用约束
                if constraints:
                    optimal_weights = np.clip(optimal_weights, constraints.min_weight, constraints.max_weight)
                    optimal_weights = optimal_weights / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else np.ones(n_assets) / n_assets
                else:
                    optimal_weights = np.clip(optimal_weights, 0, 1)
                    optimal_weights = optimal_weights / np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else np.ones(n_assets) / n_assets
            
            allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 1e-6}
            
            return OptimizationResult(
                allocations=allocations,
                expected_return=float(np.dot(optimal_weights, expected_returns)),
                risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                sharpe_ratio=self._calculate_sharpe_ratio(expected_returns, covariance_matrix, optimal_weights),
                success=True,
                message="Kelly criterion optimization successful"
            )
            
        except np.linalg.LinAlgError:
            return self._create_error_result("Kelly optimization failed: singular covariance matrix")
    
    async def _optimize_cvar(self,
                            returns_matrix: np.ndarray,
                            expected_returns: np.ndarray,
                            symbols: List[str],
                            constraints: OptimizationConstraints = None,
                            alpha: float = 0.05) -> OptimizationResult:
        """CVaR优化"""
        n_assets, n_periods = returns_matrix.shape
        
        if not self.use_cvxpy:
            return self._create_error_result("CVaR optimization requires CVXPY")
        
        try:
            # 变量
            weights = cp.Variable(n_assets)
            
            # CVaR优化需要额外的变量
            var = cp.Variable()  # VaR
            u = cp.Variable(n_periods)  # 辅助变量
            
            # 投资组合收益序列
            portfolio_returns = returns_matrix.T @ weights
            
            # 约束条件
            constraints_list = [
                cp.sum(weights) == 1.0,
                weights >= 0,
                u >= 0,
                u >= -(portfolio_returns - var)
            ]
            
            if constraints and constraints.max_weight < 1:
                constraints_list.append(weights <= constraints.max_weight)
            
            # CVaR = VaR + (1/α) * E[max(-(r-VaR), 0)]
            cvar = var + (1.0/alpha) * cp.sum(u) / n_periods
            
            # 目标函数：最小化CVaR
            objective = cp.Minimize(cvar)
            
            problem = cp.Problem(objective, constraints_list)
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                optimal_weights[optimal_weights < 1e-6] = 0
                optimal_weights = optimal_weights / optimal_weights.sum()
                
                allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0}
                
                # 计算VaR和CVaR
                portfolio_returns_realized = returns_matrix.T @ optimal_weights
                var_value = np.percentile(portfolio_returns_realized, alpha * 100)
                cvar_value = np.mean(portfolio_returns_realized[portfolio_returns_realized <= var_value])
                
                return OptimizationResult(
                    allocations=allocations,
                    expected_return=float(np.dot(optimal_weights, expected_returns)),
                    risk=float(np.sqrt(np.dot(optimal_weights, np.dot(np.cov(returns_matrix), optimal_weights)))),
                    sharpe_ratio=self._calculate_sharpe_ratio(expected_returns, np.cov(returns_matrix), optimal_weights),
                    var_95=abs(var_value) if alpha == 0.05 else None,
                    cvar_95=abs(cvar_value) if alpha == 0.05 else None,
                    success=True,
                    message="CVaR optimization successful"
                )
            else:
                return self._create_error_result(f"CVaR optimization failed with status: {problem.status}")
                
        except Exception as e:
            return self._create_error_result(f"CVaR optimization failed: {str(e)}")
    
    def _create_equal_weight_result(self, symbols: List[str]) -> OptimizationResult:
        """创建等权重结果"""
        n_assets = len(symbols)
        weight = 1.0 / n_assets
        
        allocations = {symbol: weight for symbol in symbols}
        
        return OptimizationResult(
            allocations=allocations,
            expected_return=0.0,
            risk=0.0,
            sharpe_ratio=0.0,
            success=True,
            message="Equal weight allocation"
        )
    
    def _prepare_returns_matrix(self, returns_data: Dict[str, List[float]]) -> np.ndarray:
        """准备收益率矩阵"""
        if not returns_data:
            return np.array([])
        
        # 获取最短序列长度
        min_length = min(len(returns) for returns in returns_data.values())
        
        if min_length == 0:
            return np.array([])
        
        # 构建矩阵
        returns_matrix = []
        for symbol in sorted(returns_data.keys()):
            returns = returns_data[symbol][-min_length:]
            returns_matrix.append(returns)
        
        return np.array(returns_matrix)
    
    def _calculate_expected_returns(self,
                                   returns_matrix: np.ndarray,
                                   symbols: List[str],
                                   views: Dict[str, float] = None,
                                   confidence: Dict[str, float] = None) -> np.ndarray:
        """计算期望收益"""
        # 历史平均收益
        historical_returns = np.mean(returns_matrix, axis=1)
        
        # 如果有观点，进行调整
        if views:
            expected_returns = historical_returns.copy()
            for i, symbol in enumerate(symbols):
                if symbol in views:
                    view_return = views[symbol]
                    view_confidence = confidence.get(symbol, 0.5) if confidence else 0.5
                    
                    # 加权平均历史收益和观点
                    expected_returns[i] = (
                        (1 - view_confidence) * historical_returns[i] + 
                        view_confidence * view_return
                    )
            
            return expected_returns
        
        return historical_returns
    
    def _calculate_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """计算协方差矩阵"""
        cov_matrix = np.cov(returns_matrix)
        
        # 确保正定性
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvalues) < 0:
            # Ledoit-Wolf收缩估计
            n_assets = cov_matrix.shape[0]
            identity = np.eye(n_assets)
            shrinkage = 0.1
            cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * np.trace(cov_matrix) / n_assets * identity
        
        return cov_matrix
    
    def _calculate_sharpe_ratio(self,
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               weights: np.ndarray) -> float:
        """计算夏普比率"""
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        
        if portfolio_variance <= 0:
            return 0.0
        
        portfolio_std = np.sqrt(portfolio_variance)
        return (portfolio_return - self.risk_free_rate) / portfolio_std
    
    def _calculate_additional_metrics(self,
                                     result: OptimizationResult,
                                     returns_matrix: np.ndarray,
                                     covariance_matrix: np.ndarray):
        """计算额外指标"""
        if not result.success or not result.allocations:
            return
        
        weights = np.array(list(result.allocations.values()))
        
        # 分散化比率
        if len(weights) > 1:
            individual_vols = np.sqrt(np.diag(covariance_matrix)[:len(weights)])
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = result.risk
            
            if portfolio_vol > 0:
                result.diversification_ratio = weighted_avg_vol / portfolio_vol
        
        # 有效资产数量
        result.effective_assets = int(1 / np.sum(weights ** 2)) if len(weights) > 0 else 0
        
        # 集中度指数（HHI）
        result.concentration_index = np.sum(weights ** 2)
    
    def _create_error_result(self, message: str) -> OptimizationResult:
        """创建错误结果"""
        return OptimizationResult(
            allocations={},
            expected_return=0.0,
            risk=0.0,
            sharpe_ratio=0.0,
            success=False,
            message=message
        )
    
    def create_empty_result(self) -> OptimizationResult:
        """创建空结果"""
        return OptimizationResult(
            allocations={},
            expected_return=0.0,
            risk=0.0,
            sharpe_ratio=0.0,
            success=True,
            message="Empty portfolio"
        )
    
    async def calculate_efficient_frontier(self,
                                          returns_data: Dict[str, List[float]],
                                          n_points: int = 20) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        """计算有效前沿"""
        symbols = list(returns_data.keys())
        returns_matrix = self._prepare_returns_matrix(returns_data)
        
        if returns_matrix.shape[0] < 2:
            return [], [], []
        
        expected_returns = np.mean(returns_matrix, axis=1)
        covariance_matrix = np.cov(returns_matrix)
        
        # 计算最小和最大期望收益
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        risks = []
        returns = []
        portfolios = []
        
        for target_return in target_returns:
            try:
                # 为每个目标收益率求解最小方差投资组合
                result = await self._optimize_for_target_return(
                    expected_returns, covariance_matrix, symbols, target_return
                )
                
                if result.success:
                    risks.append(result.risk)
                    returns.append(result.expected_return)
                    portfolios.append(result.allocations)
                
            except Exception as e:
                self.log_warning(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        return returns, risks, portfolios
    
    async def _optimize_for_target_return(self,
                                         expected_returns: np.ndarray,
                                         covariance_matrix: np.ndarray,
                                         symbols: List[str],
                                         target_return: float) -> OptimizationResult:
        """为目标收益率优化最小方差投资组合"""
        n_assets = len(symbols)
        
        if self.use_cvxpy:
            try:
                weights = cp.Variable(n_assets)
                
                # 目标函数：最小化方差
                objective = cp.Minimize(cp.quad_form(weights, covariance_matrix))
                
                # 约束条件
                constraints_list = [
                    cp.sum(weights) == 1.0,
                    expected_returns.T @ weights == target_return,
                    weights >= 0
                ]
                
                problem = cp.Problem(objective, constraints_list)
                problem.solve(solver=cp.OSQP, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    optimal_weights = weights.value
                    optimal_weights[optimal_weights < 1e-6] = 0
                    
                    if optimal_weights.sum() > 0:
                        optimal_weights = optimal_weights / optimal_weights.sum()
                    
                    allocations = {symbol: weight for symbol, weight in zip(symbols, optimal_weights) if weight > 0}
                    
                    return OptimizationResult(
                        allocations=allocations,
                        expected_return=float(np.dot(optimal_weights, expected_returns)),
                        risk=float(np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))),
                        sharpe_ratio=self._calculate_sharpe_ratio(expected_returns, covariance_matrix, optimal_weights),
                        success=True,
                        message="Target return optimization successful"
                    )
                    
            except Exception as e:
                self.log_warning(f"CVXPY target return optimization failed: {e}")
        
        return self._create_error_result("Target return optimization failed")