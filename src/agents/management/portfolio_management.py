"""
投资组合管理Agent
实现投资组合优化、再平衡和决策融合功能
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import norm

from src.agents.base import BaseAgent, AgentConfig
from src.agents.models import (
    AgentState, PortfolioRecommendation, FinalDecision,
    ReasoningStep, AgentConsensus
)
from src.agents.state_manager import AgentStateManager
from src.agents.management.optimization import OptimizationEngine, OptimizationResult
from src.core.models import TradingState, Signal, Position
from src.utils.logger import LoggerMixin


@dataclass
class PortfolioAllocation:
    """投资组合配置"""
    symbol: str
    target_weight: float  # 目标权重
    current_weight: float  # 当前权重
    recommended_action: str  # BUY/SELL/HOLD
    rebalance_amount: float  # 需要调整的数量
    confidence: float
    reasoning: str


@dataclass
class PortfolioMetrics:
    """投资组合指标"""
    total_value: float
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: int  # 有效资产数量
    concentration_index: float  # HHI指数
    turnover_rate: float  # 换手率
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RebalanceDecision:
    """再平衡决策"""
    need_rebalance: bool
    allocations: List[PortfolioAllocation]
    estimated_cost: float  # 预估交易成本
    expected_improvement: float  # 预期改善
    reasoning: str
    constraints_satisfied: bool
    timestamp: datetime = field(default_factory=datetime.now)


class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.2,
                 min_position_size: float = 0.01):
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
    
    def optimize_markowitz(self, 
                           expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray,
                           risk_aversion: float = 1.0,
                           constraints: Optional[Dict] = None) -> np.ndarray:
        """马科维茨均值方差优化"""
        n_assets = len(expected_returns)
        
        # 目标函数：最大化效用 = 期望收益 - risk_aversion * 方差
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # 约束条件
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # 权重和为1
        ]
        
        # 添加自定义约束
        if constraints:
            if 'max_weight' in constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: constraints['max_weight'] - np.max(x)
                })
            if 'min_weight' in constraints:
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda x: np.min(x[x > 0]) - constraints['min_weight'] if np.any(x > 0) else 0
                })
        
        # 边界条件
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
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
            # 清理小权重
            weights = result.x
            weights[weights < self.min_position_size] = 0
            weights = weights / weights.sum() if weights.sum() > 0 else initial_weights
            return weights
        else:
            # 失败时返回等权重
            return initial_weights
    
    def optimize_risk_parity(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """风险平价优化"""
        n_assets = len(covariance_matrix)
        
        def risk_contribution(weights):
            """计算风险贡献"""
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            contrib = weights * marginal_contrib / np.sqrt(portfolio_variance)
            return contrib
        
        def objective(weights):
            """目标：最小化风险贡献的差异"""
            contrib = risk_contribution(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result.x if result.success else initial_weights
    
    def calculate_efficient_frontier(self,
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    n_portfolios: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """计算有效前沿"""
        n_assets = len(expected_returns)
        
        # 生成不同风险水平的投资组合
        risk_levels = np.linspace(0.01, 2.0, n_portfolios)
        
        frontier_returns = []
        frontier_risks = []
        
        for risk_aversion in risk_levels:
            weights = self.optimize_markowitz(
                expected_returns,
                covariance_matrix,
                risk_aversion
            )
            
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            frontier_returns.append(portfolio_return)
            frontier_risks.append(portfolio_risk)
        
        return np.array(frontier_returns), np.array(frontier_risks)


class PortfolioManagementAgent(BaseAgent):
    """投资组合管理Agent"""
    
    def __init__(self,
                 config: AgentConfig,
                 state_manager: Optional[AgentStateManager] = None,
                 message_bus=None):
        super().__init__(config, message_bus)
        
        self.state_manager = state_manager
        self.optimizer = PortfolioOptimizer(
            risk_free_rate=config.parameters.get("risk_free_rate", 0.02),
            transaction_cost=config.parameters.get("transaction_cost", 0.001),
            max_position_size=config.parameters.get("max_position_size", 0.2),
            min_position_size=config.parameters.get("min_position_size", 0.01)
        )
        
        # 优化引擎
        self.optimization_engine = OptimizationEngine()
        
        # 配置参数
        self.rebalance_threshold = config.parameters.get("rebalance_threshold", 0.05)
        self.min_rebalance_interval = config.parameters.get("min_rebalance_interval", 3600)
        self.risk_aversion = config.parameters.get("risk_aversion", 1.0)
        self.use_risk_parity = config.parameters.get("use_risk_parity", False)
        self.max_turnover = config.parameters.get("max_turnover", 0.5)
        
        # 历史数据
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.covariance_matrix: Optional[np.ndarray] = None
        
        # 当前投资组合
        self.current_allocations: Dict[str, float] = {}
        self.target_allocations: Dict[str, float] = {}
        self.last_rebalance_time = datetime.now()
        
        # 决策融合权重
        self.analyst_weights: Dict[str, float] = {}
        
        self.log_info("Portfolio Management Agent initialized")
    
    async def _initialize(self) -> None:
        """初始化Agent"""
        self.log_info("Initializing Portfolio Management Agent")
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """分析并生成投资组合信号"""
        # 更新历史数据
        self._update_returns_history(state)
        
        # 获取分析师意见
        analyst_opinions = await self._get_analyst_opinions(state)
        
        # 融合决策
        consensus = self._fuse_decisions(analyst_opinions)
        
        # 获取风险约束
        risk_constraints = await self._get_risk_constraints(state)
        
        # 优化投资组合
        optimization_result = await self._optimize_portfolio(
            state, consensus, risk_constraints
        )
        
        # 评估是否需要再平衡
        rebalance_decision = self._evaluate_rebalance(state, optimization_result)
        
        # 生成交易信号
        signals = self._generate_portfolio_signals(rebalance_decision, state)
        
        # 更新状态
        await self._update_portfolio_state(state, rebalance_decision)
        
        return signals
    
    async def _get_analyst_opinions(self, state: TradingState) -> List[Dict[str, Any]]:
        """获取分析师意见"""
        if not self.state_manager:
            return []
        
        agent_state = await self.state_manager.get_state(state.session_id)
        if not agent_state:
            return []
        
        return agent_state.get("analyst_opinions", [])
    
    def _fuse_decisions(self, opinions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """融合多个分析师的决策"""
        if not opinions:
            return {}
        
        # 按交易对分组意见
        symbol_opinions = {}
        for opinion in opinions:
            symbol = opinion.get("symbol")
            if symbol:
                if symbol not in symbol_opinions:
                    symbol_opinions[symbol] = []
                symbol_opinions[symbol].append(opinion)
        
        # 计算加权共识
        consensus = {}
        for symbol, symbol_ops in symbol_opinions.items():
            # 提取各分析师的建议
            recommendations = []
            confidences = []
            reasons = []
            
            for op in symbol_ops:
                rec_value = self._map_recommendation_to_value(op.get("recommendation"))
                conf = op.get("confidence", 0.5)
                reason = op.get("reasoning", "")
                
                recommendations.append(rec_value)
                confidences.append(conf)
                reasons.append(reason)
            
            # 计算加权平均
            if recommendations:
                weights = np.array(confidences)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                weighted_recommendation = np.average(recommendations, weights=weights)
                avg_confidence = np.mean(confidences)
                
                consensus[symbol] = {
                    "recommendation": weighted_recommendation,
                    "confidence": avg_confidence,
                    "num_analysts": len(recommendations),
                    "agreement_score": 1 - np.std(recommendations),
                    "reasons": reasons[:3]  # 保留前3个理由
                }
        
        return consensus
    
    async def _get_risk_constraints(self, state: TradingState) -> Dict[str, Any]:
        """获取风险约束"""
        constraints = {
            "max_position_size": self.optimizer.max_position_size,
            "min_position_size": self.optimizer.min_position_size,
            "max_volatility": 0.3,
            "max_drawdown": 0.2,
            "max_var_95": 0.05
        }
        
        # 从状态管理器获取风险评估
        if self.state_manager:
            agent_state = await self.state_manager.get_state(state.session_id)
            if agent_state and "risk_assessment" in agent_state:
                risk_assessment = agent_state["risk_assessment"]
                
                # 根据风险等级调整约束
                risk_level = risk_assessment.get("risk_level", "moderate")
                if risk_level == "extreme":
                    constraints["max_position_size"] *= 0.5
                    constraints["max_volatility"] *= 0.7
                elif risk_level == "high":
                    constraints["max_position_size"] *= 0.8
                    constraints["max_volatility"] *= 0.85
        
        return constraints
    
    async def _optimize_portfolio(self,
                                 state: TradingState,
                                 consensus: Dict[str, Any],
                                 constraints: Dict[str, Any]) -> OptimizationResult:
        """执行投资组合优化"""
        symbols = list(state.active_symbols)
        n_assets = len(symbols)
        
        if n_assets == 0:
            return self.optimization_engine.create_empty_result()
        
        # 准备优化输入
        expected_returns = np.zeros(n_assets)
        for i, symbol in enumerate(symbols):
            if symbol in consensus:
                # 使用共识预期收益
                expected_returns[i] = consensus[symbol]["recommendation"] * 0.1  # 缩放到合理范围
            else:
                # 使用历史平均收益
                if symbol in self.returns_history:
                    expected_returns[i] = np.mean(self.returns_history[symbol])
        
        # 计算协方差矩阵
        covariance_matrix = self._calculate_covariance_matrix(symbols)
        
        # 选择优化方法
        if self.use_risk_parity:
            # 风险平价优化
            optimal_weights = self.optimizer.optimize_risk_parity(covariance_matrix)
        else:
            # 马科维茨优化
            optimal_weights = self.optimizer.optimize_markowitz(
                expected_returns,
                covariance_matrix,
                self.risk_aversion,
                constraints
            )
        
        # 创建优化结果
        allocations = {}
        for i, symbol in enumerate(symbols):
            if optimal_weights[i] > self.optimizer.min_position_size:
                allocations[symbol] = optimal_weights[i]
        
        # 计算投资组合指标
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.optimizer.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        result = OptimizationResult(
            allocations=allocations,
            expected_return=portfolio_return,
            risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            success=True,
            message="Portfolio optimization completed"
        )
        
        self.target_allocations = allocations
        return result
    
    def _evaluate_rebalance(self, 
                           state: TradingState,
                           optimization_result: OptimizationResult) -> RebalanceDecision:
        """评估是否需要再平衡"""
        # 检查时间间隔
        time_since_last = (datetime.now() - self.last_rebalance_time).total_seconds()
        if time_since_last < self.min_rebalance_interval:
            return RebalanceDecision(
                need_rebalance=False,
                allocations=[],
                estimated_cost=0,
                expected_improvement=0,
                reasoning="Too soon since last rebalance",
                constraints_satisfied=True
            )
        
        # 计算当前权重
        current_weights = self._calculate_current_weights(state)
        target_weights = optimization_result.allocations
        
        # 计算偏差
        allocations = []
        total_deviation = 0
        estimated_turnover = 0
        
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            deviation = abs(target - current)
            total_deviation += deviation
            
            if deviation > self.rebalance_threshold:
                # 需要调整
                if target > current:
                    action = "BUY"
                    amount = target - current
                elif target < current:
                    action = "SELL"
                    amount = current - target
                else:
                    action = "HOLD"
                    amount = 0
                
                allocations.append(PortfolioAllocation(
                    symbol=symbol,
                    target_weight=target,
                    current_weight=current,
                    recommended_action=action,
                    rebalance_amount=amount,
                    confidence=0.8,
                    reasoning=f"Rebalance from {current:.2%} to {target:.2%}"
                ))
                
                estimated_turnover += abs(amount)
        
        # 检查换手率约束
        if estimated_turnover > self.max_turnover:
            # 缩减调整规模
            scale_factor = self.max_turnover / estimated_turnover
            for alloc in allocations:
                alloc.rebalance_amount *= scale_factor
            estimated_turnover = self.max_turnover
        
        # 估算交易成本
        estimated_cost = estimated_turnover * self.optimizer.transaction_cost
        
        # 估算预期改善
        expected_improvement = optimization_result.sharpe_ratio - self._calculate_current_sharpe(state)
        
        # 决定是否再平衡
        need_rebalance = (
            total_deviation > self.rebalance_threshold * 2 and
            expected_improvement > estimated_cost * 10  # 改善要大于成本的10倍
        )
        
        return RebalanceDecision(
            need_rebalance=need_rebalance,
            allocations=allocations,
            estimated_cost=estimated_cost,
            expected_improvement=expected_improvement,
            reasoning=f"Deviation: {total_deviation:.2%}, Expected improvement: {expected_improvement:.4f}",
            constraints_satisfied=estimated_turnover <= self.max_turnover
        )
    
    def _generate_portfolio_signals(self, 
                                   decision: RebalanceDecision,
                                   state: TradingState) -> List[Signal]:
        """生成投资组合调整信号"""
        signals = []
        
        if not decision.need_rebalance:
            return signals
        
        for allocation in decision.allocations:
            if allocation.recommended_action != "HOLD":
                # 计算信号强度
                strength = min(allocation.rebalance_amount * 5, 1.0)  # 缩放到[-1, 1]
                if allocation.recommended_action == "SELL":
                    strength = -strength
                
                signal = Signal(
                    source=f"{self.name}_portfolio",
                    symbol=allocation.symbol,
                    action=allocation.recommended_action,
                    strength=strength,
                    confidence=allocation.confidence,
                    reason=allocation.reasoning,
                    metadata={
                        "target_weight": allocation.target_weight,
                        "current_weight": allocation.current_weight,
                        "rebalance_amount": allocation.rebalance_amount,
                        "estimated_cost": decision.estimated_cost / len(decision.allocations),
                        "portfolio_optimization": True
                    }
                )
                
                # 简单验证信号 (避免异步调用在这个上下文中)
                if signal.symbol and signal.action and -1 <= signal.strength <= 1:
                    signals.append(signal)
        
        # 更新最后再平衡时间
        if signals:
            self.last_rebalance_time = datetime.now()
        
        return signals
    
    async def _update_portfolio_state(self, state: TradingState, decision: RebalanceDecision):
        """更新投资组合状态"""
        if not self.state_manager:
            return
        
        # 计算投资组合指标
        metrics = self._calculate_portfolio_metrics(state)
        
        # 创建投资组合推荐
        recommendations = []
        for alloc in decision.allocations:
            rec = PortfolioRecommendation(
                asset=alloc.symbol,
                allocation_percentage=alloc.target_weight * 100,
                confidence_score=alloc.confidence,
                risk_adjusted_return=0.0,  # 可以计算
                recommendation_source=self.name,
                rebalance_urgency="high" if decision.need_rebalance else "low",
                constraints_applied=["position_limits", "risk_constraints"]
            )
            recommendations.append(rec.__dict__)
        
        # 创建最终决策
        if decision.need_rebalance:
            final_decision = FinalDecision(
                action="REBALANCE",
                confidence=0.8,
                reasoning=decision.reasoning,
                risk_assessment="managed",
                expected_return=metrics.expected_return,
                position_sizes={a.symbol: a.target_weight for a in decision.allocations},
                stop_loss_levels={},
                take_profit_levels={},
                time_horizon="medium",
                execution_strategy="TWAP"
            )
        else:
            final_decision = None
        
        # 更新状态
        updates = {
            "portfolio_recommendations": recommendations,
            "portfolio_metrics": {
                "total_value": metrics.total_value,
                "expected_return": metrics.expected_return,
                "volatility": metrics.portfolio_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "diversification_ratio": metrics.diversification_ratio
            }
        }
        
        if final_decision:
            updates["final_decision"] = final_decision.__dict__
        
        await self.state_manager.update_state(
            state.session_id,
            updates,
            self.name
        )
    
    def _update_returns_history(self, state: TradingState):
        """更新收益率历史"""
        for symbol, market_data in state.market_data.items():
            if symbol not in self.returns_history:
                self.returns_history[symbol] = []
            
            # 简单收益率计算（实际应该使用对数收益率）
            if len(self.returns_history[symbol]) > 0:
                last_price = self.returns_history[symbol][-1]
                return_rate = (market_data.close - last_price) / last_price if last_price > 0 else 0
                self.returns_history[symbol].append(return_rate)
            else:
                self.returns_history[symbol].append(0)
            
            # 限制历史长度
            max_length = 252  # 一年的交易日
            if len(self.returns_history[symbol]) > max_length:
                self.returns_history[symbol] = self.returns_history[symbol][-max_length:]
    
    def _calculate_covariance_matrix(self, symbols: List[str]) -> np.ndarray:
        """计算协方差矩阵"""
        n_assets = len(symbols)
        
        # 收集收益率数据
        returns_matrix = []
        for symbol in symbols:
            if symbol in self.returns_history:
                returns_matrix.append(self.returns_history[symbol])
            else:
                # 没有历史数据时使用随机数据
                returns_matrix.append(np.random.normal(0, 0.01, 100).tolist())
        
        # 确保所有序列长度相同
        min_length = min(len(r) for r in returns_matrix)
        if min_length < 2:
            # 数据不足，返回单位矩阵
            return np.eye(n_assets) * 0.01
        
        returns_matrix = [r[-min_length:] for r in returns_matrix]
        returns_df = pd.DataFrame(returns_matrix).T
        
        # 计算协方差矩阵
        cov_matrix = returns_df.cov().values
        
        # 确保正定性
        eigenvalues = np.linalg.eigvals(cov_matrix)
        if np.min(eigenvalues) < 0:
            # 修正负特征值
            cov_matrix = cov_matrix + np.eye(n_assets) * abs(np.min(eigenvalues)) * 1.1
        
        self.covariance_matrix = cov_matrix
        return cov_matrix
    
    def _calculate_current_weights(self, state: TradingState) -> Dict[str, float]:
        """计算当前投资组合权重"""
        total_value = 0
        position_values = {}
        
        for symbol, position in state.positions.items():
            if symbol in state.market_data:
                value = abs(position.quantity * state.market_data[symbol].close)
                position_values[symbol] = value
                total_value += value
        
        if total_value == 0:
            return {}
        
        return {symbol: value / total_value for symbol, value in position_values.items()}
    
    def _calculate_current_sharpe(self, state: TradingState) -> float:
        """计算当前投资组合的夏普比率"""
        weights = self._calculate_current_weights(state)
        if not weights:
            return 0.0
        
        # 计算加权收益率
        portfolio_returns = []
        symbols = list(weights.keys())
        
        if not symbols:
            return 0.0
        
        # 获取最短的历史长度
        min_length = float('inf')
        for symbol in symbols:
            if symbol in self.returns_history:
                min_length = min(min_length, len(self.returns_history[symbol]))
        
        if min_length < 2:
            return 0.0
        
        # 计算投资组合收益率序列
        for i in range(int(min_length)):
            portfolio_return = 0
            for symbol in symbols:
                if symbol in self.returns_history:
                    portfolio_return += weights[symbol] * self.returns_history[symbol][i]
            portfolio_returns.append(portfolio_return)
        
        if not portfolio_returns:
            return 0.0
        
        # 计算夏普比率
        returns_array = np.array(portfolio_returns)
        excess_returns = returns_array - self.optimizer.risk_free_rate / 252
        
        if returns_array.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / returns_array.std()
    
    def _calculate_portfolio_metrics(self, state: TradingState) -> PortfolioMetrics:
        """计算投资组合指标"""
        # 计算总价值
        total_value = sum(
            abs(p.quantity * state.market_data[s].close)
            for s, p in state.positions.items()
            if s in state.market_data
        )
        
        # 计算权重
        weights = self._calculate_current_weights(state)
        
        # 计算预期收益和风险
        if weights:
            symbols = list(weights.keys())
            weights_array = np.array([weights[s] for s in symbols])
            
            # 简单预期收益（使用历史平均）
            expected_returns = []
            for symbol in symbols:
                if symbol in self.returns_history and self.returns_history[symbol]:
                    expected_returns.append(np.mean(self.returns_history[symbol]))
                else:
                    expected_returns.append(0)
            
            expected_return = np.dot(weights_array, expected_returns) * 252  # 年化
            
            # 计算组合波动率
            if self.covariance_matrix is not None and len(symbols) == len(self.covariance_matrix):
                portfolio_variance = np.dot(weights_array, np.dot(self.covariance_matrix, weights_array))
                portfolio_volatility = np.sqrt(portfolio_variance * 252)  # 年化
            else:
                portfolio_volatility = 0.2  # 默认值
            
            # 夏普比率
            sharpe_ratio = (expected_return - self.optimizer.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # 分散化比率
            individual_vols = [np.std(self.returns_history.get(s, [0])) * np.sqrt(252) for s in symbols]
            weighted_avg_vol = np.dot(weights_array, individual_vols)
            diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1
            
            # HHI集中度指数
            concentration_index = np.sum(weights_array ** 2)
            
            # 有效资产数量
            effective_assets = 1 / concentration_index if concentration_index > 0 else len(symbols)
            
        else:
            expected_return = 0
            portfolio_volatility = 0
            sharpe_ratio = 0
            diversification_ratio = 1
            concentration_index = 0
            effective_assets = 0
        
        return PortfolioMetrics(
            total_value=total_value,
            expected_return=expected_return,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            effective_assets=int(effective_assets),
            concentration_index=concentration_index,
            turnover_rate=0  # 需要历史数据计算
        )
    
    def _map_recommendation_to_value(self, recommendation: Any) -> float:
        """将推荐映射到数值"""
        if isinstance(recommendation, (int, float)):
            return float(recommendation)
        
        recommendation_str = str(recommendation).upper()
        
        mapping = {
            "STRONG_BUY": 1.0,
            "BUY": 0.5,
            "HOLD": 0.0,
            "SELL": -0.5,
            "STRONG_SELL": -1.0
        }
        
        return mapping.get(recommendation_str, 0.0)
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取投资组合摘要"""
        return {
            "current_allocations": self.current_allocations,
            "target_allocations": self.target_allocations,
            "last_rebalance": self.last_rebalance_time.isoformat(),
            "optimization_method": "risk_parity" if self.use_risk_parity else "mean_variance",
            "risk_aversion": self.risk_aversion,
            "rebalance_threshold": self.rebalance_threshold,
            "performance": {
                "sharpe_ratio": 0,  # 需要计算
                "total_return": 0,  # 需要计算
                "volatility": 0  # 需要计算
            }
        }