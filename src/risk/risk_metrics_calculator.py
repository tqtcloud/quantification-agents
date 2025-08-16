"""
风险指标计算引擎
实现实时风险指标计算（VAR、夏普比率等）、最大回撤监控和预警、动态止损止盈调整
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from src.core.models import (
    RiskMetrics, Position, Order, MarketData, TradingState
)
from src.utils.logger import LoggerMixin


class RiskCalculationMethod(Enum):
    """风险计算方法"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    timestamp: datetime
    total_value: float
    positions: Dict[str, float]  # symbol -> value
    pnl: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """风险计算配置"""
    var_confidence_level: float = 0.95
    var_time_horizon: int = 1  # 天
    lookback_days: int = 252   # 历史数据回看天数
    min_data_points: int = 30  # 最小数据点数
    sharpe_risk_free_rate: float = 0.02  # 无风险利率
    drawdown_window: int = 252  # 回撤计算窗口
    correlation_window: int = 60  # 相关性计算窗口


class RiskMetricsCalculator(LoggerMixin):
    """风险指标计算引擎"""
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        
        # 历史数据存储
        self.price_history: Dict[str, deque] = {}  # symbol -> price_history
        self.portfolio_history: deque = deque(maxlen=self.config.lookback_days)
        self.return_history: deque = deque(maxlen=self.config.lookback_days)
        
        # 实时指标
        self.current_metrics: Optional[RiskMetrics] = None
        self.last_calculation_time: Optional[datetime] = None
        
        # 回撤追踪
        self.peak_value = 0.0
        self.peak_date: Optional[datetime] = None
        self.drawdown_periods: List[Tuple[datetime, datetime, float]] = []
        
        self.log_info("RiskMetricsCalculator initialized")

    def calculate_risk_metrics(self, 
                             state: TradingState,
                             total_capital: float) -> RiskMetrics:
        """计算完整的风险指标"""
        
        try:
            # 更新历史数据
            self._update_price_history(state)
            portfolio_value = self._calculate_portfolio_value(state, total_capital)
            self._update_portfolio_history(portfolio_value)
            
            # 计算各项风险指标
            risk_metrics = RiskMetrics(
                total_exposure=self._calculate_total_exposure(state),
                max_drawdown=self._calculate_max_drawdown(),
                current_drawdown=self._calculate_current_drawdown(portfolio_value),
                sharpe_ratio=self._calculate_sharpe_ratio(),
                win_rate=self._calculate_win_rate(state),
                profit_factor=self._calculate_profit_factor(state),
                var_95=self._calculate_var(confidence=0.95),
                margin_usage=self._calculate_margin_usage(state, total_capital),
                leverage_ratio=self._calculate_leverage_ratio(state, total_capital),
                daily_pnl=self._calculate_daily_pnl(),
                total_pnl=self._calculate_total_pnl(portfolio_value, total_capital),
                timestamp=datetime.utcnow()
            )
            
            self.current_metrics = risk_metrics
            self.last_calculation_time = datetime.utcnow()
            
            self.log_debug(
                f"Risk metrics calculated - Drawdown: {risk_metrics.current_drawdown:.2%}, "
                f"Sharpe: {risk_metrics.sharpe_ratio:.2f}, VaR: {risk_metrics.var_95:.2f}"
            )
            
            return risk_metrics
            
        except Exception as e:
            self.log_error(f"Error calculating risk metrics: {e}")
            # 返回默认风险指标
            return self._get_default_risk_metrics()

    def calculate_var(self, 
                     returns: List[float], 
                     confidence: float = 0.95,
                     method: RiskCalculationMethod = RiskCalculationMethod.HISTORICAL) -> float:
        """计算风险价值（VaR）"""
        
        if not returns or len(returns) < self.config.min_data_points:
            return 0.0
        
        try:
            if method == RiskCalculationMethod.HISTORICAL:
                return self._calculate_historical_var(returns, confidence)
            elif method == RiskCalculationMethod.PARAMETRIC:
                return self._calculate_parametric_var(returns, confidence)
            elif method == RiskCalculationMethod.MONTE_CARLO:
                return self._calculate_monte_carlo_var(returns, confidence)
            else:
                return self._calculate_historical_var(returns, confidence)
                
        except Exception as e:
            self.log_error(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_conditional_var(self, 
                                returns: List[float], 
                                confidence: float = 0.95) -> float:
        """计算条件风险价值（CVaR/Expected Shortfall）"""
        
        if not returns or len(returns) < self.config.min_data_points:
            return 0.0
        
        try:
            var = self.calculate_var(returns, confidence)
            # CVaR是超过VaR的损失的平均值
            extreme_losses = [r for r in returns if r <= var]
            
            if extreme_losses:
                return np.mean(extreme_losses)
            else:
                return var
                
        except Exception as e:
            self.log_error(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_beta(self, 
                      asset_returns: List[float], 
                      market_returns: List[float]) -> float:
        """计算Beta系数"""
        
        if (not asset_returns or not market_returns or 
            len(asset_returns) != len(market_returns) or
            len(asset_returns) < self.config.min_data_points):
            return 1.0
        
        try:
            asset_returns_np = np.array(asset_returns)
            market_returns_np = np.array(market_returns)
            
            # 计算协方差和方差
            covariance = np.cov(asset_returns_np, market_returns_np)[0, 1]
            market_variance = np.var(market_returns_np)
            
            if market_variance == 0:
                return 1.0
            
            return covariance / market_variance
            
        except Exception as e:
            self.log_error(f"Error calculating beta: {e}")
            return 1.0

    def calculate_correlation_matrix(self, 
                                   symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """计算相关性矩阵"""
        
        correlation_matrix = {}
        
        try:
            # 准备收益率数据
            returns_data = {}
            min_length = float('inf')
            
            for symbol in symbols:
                if symbol in self.price_history:
                    prices = list(self.price_history[symbol])
                    if len(prices) >= 2:
                        returns = [
                            (prices[i] - prices[i-1]) / prices[i-1] 
                            for i in range(1, len(prices))
                        ]
                        returns_data[symbol] = returns
                        min_length = min(min_length, len(returns))
            
            # 截取相同长度的数据
            if min_length < self.config.min_data_points:
                self.log_warning("Insufficient data for correlation calculation")
                return correlation_matrix
            
            # 计算相关性
            for symbol1 in symbols:
                correlation_matrix[symbol1] = {}
                for symbol2 in symbols:
                    if symbol1 == symbol2:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    elif symbol1 in returns_data and symbol2 in returns_data:
                        returns1 = returns_data[symbol1][-min_length:]
                        returns2 = returns_data[symbol2][-min_length:]
                        
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        
                        correlation_matrix[symbol1][symbol2] = correlation
                    else:
                        correlation_matrix[symbol1][symbol2] = 0.0
            
            return correlation_matrix
            
        except Exception as e:
            self.log_error(f"Error calculating correlation matrix: {e}")
            return correlation_matrix

    def calculate_portfolio_volatility(self, 
                                     positions: Dict[str, float],
                                     correlation_matrix: Dict[str, Dict[str, float]],
                                     volatilities: Dict[str, float]) -> float:
        """计算投资组合波动率"""
        
        try:
            symbols = list(positions.keys())
            total_value = sum(abs(pos) for pos in positions.values())
            
            if total_value == 0:
                return 0.0
            
            # 计算权重
            weights = {symbol: positions[symbol] / total_value for symbol in symbols}
            
            # 计算投资组合方差
            portfolio_variance = 0.0
            
            for symbol1 in symbols:
                for symbol2 in symbols:
                    weight1 = weights.get(symbol1, 0)
                    weight2 = weights.get(symbol2, 0)
                    vol1 = volatilities.get(symbol1, 0)
                    vol2 = volatilities.get(symbol2, 0)
                    corr = correlation_matrix.get(symbol1, {}).get(symbol2, 0)
                    
                    portfolio_variance += weight1 * weight2 * vol1 * vol2 * corr
            
            return math.sqrt(max(0, portfolio_variance))
            
        except Exception as e:
            self.log_error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    def calculate_stop_loss_level(self, 
                                position: Position,
                                risk_percentage: float = 0.02,
                                atr_multiplier: float = 2.0) -> float:
        """计算止损水平"""
        
        try:
            symbol = position.symbol
            current_price = position.mark_price
            
            # 方法1：基于风险百分比
            risk_based_stop = current_price * (1 - risk_percentage) if position.side == PositionSide.LONG else current_price * (1 + risk_percentage)
            
            # 方法2：基于ATR（简化版本）
            atr_stop = self._calculate_atr_stop_loss(symbol, current_price, atr_multiplier, position.side)
            
            # 选择更保守的止损水平
            if position.side == PositionSide.LONG:
                return max(risk_based_stop, atr_stop) if atr_stop > 0 else risk_based_stop
            else:
                return min(risk_based_stop, atr_stop) if atr_stop > 0 else risk_based_stop
                
        except Exception as e:
            self.log_error(f"Error calculating stop loss level: {e}")
            return position.mark_price * 0.98 if position.side == PositionSide.LONG else position.mark_price * 1.02

    def calculate_take_profit_level(self, 
                                  position: Position,
                                  risk_reward_ratio: float = 2.0,
                                  stop_loss_level: float = None) -> float:
        """计算止盈水平"""
        
        try:
            current_price = position.mark_price
            
            if stop_loss_level:
                # 基于风险回报比
                risk_distance = abs(current_price - stop_loss_level)
                profit_distance = risk_distance * risk_reward_ratio
                
                if position.side == PositionSide.LONG:
                    return current_price + profit_distance
                else:
                    return current_price - profit_distance
            else:
                # 简单的百分比方法
                profit_percentage = 0.04  # 4%
                if position.side == PositionSide.LONG:
                    return current_price * (1 + profit_percentage)
                else:
                    return current_price * (1 - profit_percentage)
                    
        except Exception as e:
            self.log_error(f"Error calculating take profit level: {e}")
            return position.mark_price * 1.04 if position.side == PositionSide.LONG else position.mark_price * 0.96

    def _update_price_history(self, state: TradingState):
        """更新价格历史"""
        for symbol, market_data in state.market_data.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.config.lookback_days)
            
            self.price_history[symbol].append(market_data.price)

    def _calculate_portfolio_value(self, state: TradingState, total_capital: float) -> float:
        """计算投资组合价值"""
        portfolio_value = total_capital
        
        for position in state.positions.values():
            portfolio_value += position.unrealized_pnl + position.realized_pnl
        
        return portfolio_value

    def _update_portfolio_history(self, portfolio_value: float):
        """更新投资组合历史"""
        self.portfolio_history.append(portfolio_value)
        
        # 计算收益率
        if len(self.portfolio_history) >= 2:
            previous_value = self.portfolio_history[-2]
            if previous_value > 0:
                return_rate = (portfolio_value - previous_value) / previous_value
                self.return_history.append(return_rate)

    def _calculate_total_exposure(self, state: TradingState) -> float:
        """计算总敞口"""
        total_exposure = 0.0
        
        for position in state.positions.values():
            total_exposure += abs(position.quantity) * position.mark_price
        
        return total_exposure

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        values = list(self.portfolio_history)
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _calculate_current_drawdown(self, current_value: float) -> float:
        """计算当前回撤"""
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.peak_date = datetime.utcnow()
            return 0.0
        
        if self.peak_value > 0:
            return (self.peak_value - current_value) / self.peak_value
        
        return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.return_history) < self.config.min_data_points:
            return 0.0
        
        returns = list(self.return_history)
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # 年化
        daily_risk_free_rate = self.config.sharpe_risk_free_rate / 252
        excess_return = avg_return - daily_risk_free_rate
        
        # 年化夏普比率
        return (excess_return * math.sqrt(252)) / (std_return * math.sqrt(252))

    def _calculate_win_rate(self, state: TradingState) -> float:
        """计算胜率（简化实现）"""
        if not self.return_history:
            return 0.0
        
        positive_returns = [r for r in self.return_history if r > 0]
        return len(positive_returns) / len(self.return_history)

    def _calculate_profit_factor(self, state: TradingState) -> float:
        """计算盈利因子"""
        if not self.return_history:
            return 1.0
        
        positive_returns = [r for r in self.return_history if r > 0]
        negative_returns = [r for r in self.return_history if r < 0]
        
        total_profit = sum(positive_returns) if positive_returns else 0
        total_loss = abs(sum(negative_returns)) if negative_returns else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 1.0
        
        return total_profit / total_loss

    def _calculate_var(self, confidence: float) -> float:
        """计算VaR"""
        returns = list(self.return_history)
        return self.calculate_var(returns, confidence)

    def _calculate_historical_var(self, returns: List[float], confidence: float) -> float:
        """历史模拟法计算VaR"""
        if not returns:
            return 0.0
        
        percentile = (1 - confidence) * 100
        return np.percentile(returns, percentile)

    def _calculate_parametric_var(self, returns: List[float], confidence: float) -> float:
        """参数法计算VaR"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 使用正态分布假设
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        
        return mean_return + z_score * std_return

    def _calculate_monte_carlo_var(self, returns: List[float], confidence: float, simulations: int = 10000) -> float:
        """蒙特卡洛法计算VaR"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # 蒙特卡洛模拟
        simulated_returns = np.random.normal(mean_return, std_return, simulations)
        
        percentile = (1 - confidence) * 100
        return np.percentile(simulated_returns, percentile)

    def _calculate_margin_usage(self, state: TradingState, total_capital: float) -> float:
        """计算保证金使用率"""
        total_margin = sum(position.margin for position in state.positions.values())
        
        if total_capital == 0:
            return 0.0
        
        return total_margin / total_capital

    def _calculate_leverage_ratio(self, state: TradingState, total_capital: float) -> float:
        """计算杠杆率"""
        total_exposure = self._calculate_total_exposure(state)
        
        if total_capital == 0:
            return 0.0
        
        return total_exposure / total_capital

    def _calculate_daily_pnl(self) -> float:
        """计算当日盈亏"""
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return 0.0
        
        return self.portfolio_history[-1] - self.portfolio_history[-2]

    def _calculate_total_pnl(self, current_value: float, initial_capital: float) -> float:
        """计算总盈亏"""
        return current_value - initial_capital

    def _calculate_atr_stop_loss(self, 
                                symbol: str, 
                                current_price: float, 
                                multiplier: float,
                                side: PositionSide) -> float:
        """基于ATR计算止损水平（简化版本）"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 14:
            return 0.0
        
        # 简化的ATR计算
        prices = list(self.price_history[symbol])[-14:]  # 使用最近14个价格
        
        if len(prices) < 2:
            return 0.0
        
        # 计算真实波幅
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])  # 简化：用价格差代替高低价差
            true_ranges.append(high_low)
        
        if not true_ranges:
            return 0.0
        
        atr = np.mean(true_ranges)
        
        if side == PositionSide.LONG:
            return current_price - (atr * multiplier)
        else:
            return current_price + (atr * multiplier)

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """获取默认风险指标"""
        return RiskMetrics(
            total_exposure=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            var_95=0.0,
            margin_usage=0.0,
            leverage_ratio=0.0,
            daily_pnl=0.0,
            total_pnl=0.0,
            timestamp=datetime.utcnow()
        )

    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险指标摘要"""
        if not self.current_metrics:
            return {"status": "no_data"}
        
        return {
            "total_exposure": self.current_metrics.total_exposure,
            "current_drawdown": self.current_metrics.current_drawdown,
            "max_drawdown": self.current_metrics.max_drawdown,
            "sharpe_ratio": self.current_metrics.sharpe_ratio,
            "var_95": self.current_metrics.var_95,
            "margin_usage": self.current_metrics.margin_usage,
            "leverage_ratio": self.current_metrics.leverage_ratio,
            "last_calculation": self.last_calculation_time.isoformat() if self.last_calculation_time else None,
            "data_points": len(self.return_history)
        }