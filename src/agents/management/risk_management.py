"""
高级风险管理Agent
实现VaR计算、回撤分析、相关性分析等风险评估功能
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats

from src.agents.base import BaseAgent, AgentConfig
from src.agents.models import AgentState, RiskAssessmentState
from src.agents.state_manager import AgentStateManager
from src.core.models import TradingState, Signal, MarketData
from src.utils.logger import LoggerMixin


class RiskLevel(Enum):
    """风险等级"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    max_drawdown: float  # 最大回撤
    current_drawdown: float  # 当前回撤
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # 索提诺比率
    calmar_ratio: float  # 卡尔玛比率
    beta: float  # 贝塔系数
    correlation_risk: float  # 相关性风险
    concentration_risk: float  # 集中度风险
    liquidity_risk: float  # 流动性风险
    volatility: float  # 波动率
    downside_volatility: float  # 下行波动率
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskFactor:
    """风险因子"""
    factor_name: str
    factor_type: str  # market, credit, liquidity, operational
    impact_score: float  # -1 到 1
    confidence: float
    description: str
    mitigation_strategy: str


@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_level: RiskLevel
    risk_score: float  # 0-100
    risk_metrics: RiskMetrics
    risk_factors: List[RiskFactor]
    position_recommendations: Dict[str, float]  # symbol -> recommended position size
    stop_loss_levels: Dict[str, float]  # symbol -> stop loss price
    risk_warnings: List[str]
    mitigation_strategies: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class RiskModel:
    """风险模型基类"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.confidence_levels = [0.95, 0.99]
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算VaR (Value at Risk)"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算CVaR (Conditional Value at Risk)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> Tuple[float, float]:
        """计算最大回撤和当前回撤"""
        if len(prices) < 2:
            return 0.0, 0.0
        
        cumulative = np.cumprod(1 + prices)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        current_dd = drawdown[-1] if len(drawdown) > 0 else 0.0
        
        return abs(max_dd), abs(current_dd)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_correlation_matrix(self, returns_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """计算相关性矩阵"""
        if len(returns_dict) < 2:
            return np.array([[1.0]])
        
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr().values
    
    def assess_concentration_risk(self, positions: Dict[str, float]) -> float:
        """评估集中度风险 (使用HHI指数)"""
        if not positions:
            return 0.0
        
        total = sum(abs(v) for v in positions.values())
        if total == 0:
            return 0.0
        
        weights = [abs(v) / total for v in positions.values()]
        hhi = sum(w ** 2 for w in weights)
        
        # HHI范围从1/n (完全分散) 到 1 (完全集中)
        n = len(positions)
        min_hhi = 1 / n if n > 0 else 1
        
        # 标准化到0-1范围
        return (hhi - min_hhi) / (1 - min_hhi) if min_hhi < 1 else 0.0


class RiskManagementAgent(BaseAgent):
    """高级风险管理Agent"""
    
    def __init__(self, 
                 config: AgentConfig,
                 state_manager: Optional[AgentStateManager] = None,
                 message_bus=None):
        super().__init__(config, message_bus)
        
        self.state_manager = state_manager
        self.risk_model = RiskModel(lookback_period=config.parameters.get("lookback_period", 252))
        
        # 风险参数
        self.max_var_95 = config.parameters.get("max_var_95", 0.05)
        self.max_var_99 = config.parameters.get("max_var_99", 0.10)
        self.max_drawdown = config.parameters.get("max_drawdown", 0.20)
        self.max_concentration = config.parameters.get("max_concentration", 0.30)
        self.max_correlation = config.parameters.get("max_correlation", 0.70)
        self.min_sharpe = config.parameters.get("min_sharpe", 0.5)
        
        # 历史数据存储
        self.returns_history: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.position_history: Dict[str, List[float]] = {}
        
        # 风险评估缓存
        self.last_assessment: Optional[RiskAssessment] = None
        self.assessment_cache: Dict[str, RiskAssessment] = {}
        
        # 意见分歧度跟踪
        self.opinion_divergence: Dict[str, float] = {}
        
        self.log_info("Advanced Risk Management Agent initialized")
    
    async def _initialize(self) -> None:
        """初始化Agent"""
        self.log_info("Initializing Risk Management Agent")
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """分析风险并生成信号"""
        # 执行风险评估
        assessment = await self.assess_risk(state)
        
        # 生成风险调整信号
        signals = self._generate_risk_signals(assessment, state)
        
        # 更新状态管理器
        if self.state_manager:
            await self._update_state_manager(assessment, state.session_id)
        
        return signals
    
    async def assess_risk(self, state: TradingState) -> RiskAssessment:
        """综合风险评估"""
        # 更新历史数据
        self._update_history(state)
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(state)
        
        # 识别风险因子
        risk_factors = self._identify_risk_factors(state, risk_metrics)
        
        # 计算意见分歧度
        divergence = await self._calculate_opinion_divergence(state)
        
        # 评估整体风险水平
        risk_level, risk_score = self._evaluate_risk_level(risk_metrics, risk_factors, divergence)
        
        # 生成仓位建议
        position_recommendations = self._calculate_position_recommendations(
            state, risk_metrics, risk_level
        )
        
        # 计算止损水平
        stop_loss_levels = self._calculate_stop_loss_levels(state, risk_metrics)
        
        # 生成风险警告和缓解策略
        warnings = self._generate_risk_warnings(risk_metrics, risk_factors)
        strategies = self._generate_mitigation_strategies(risk_factors, risk_level)
        
        assessment = RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_metrics=risk_metrics,
            risk_factors=risk_factors,
            position_recommendations=position_recommendations,
            stop_loss_levels=stop_loss_levels,
            risk_warnings=warnings,
            mitigation_strategies=strategies
        )
        
        self.last_assessment = assessment
        return assessment
    
    def _update_history(self, state: TradingState):
        """更新历史数据"""
        for symbol, market_data in state.market_data.items():
            # 更新价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(market_data.close)
            
            # 限制历史长度
            max_length = self.risk_model.lookback_period
            if len(self.price_history[symbol]) > max_length:
                self.price_history[symbol] = self.price_history[symbol][-max_length:]
            
            # 计算收益率
            if len(self.price_history[symbol]) >= 2:
                returns = np.diff(np.log(self.price_history[symbol]))
                if symbol not in self.returns_history:
                    self.returns_history[symbol] = []
                self.returns_history[symbol] = list(returns)
    
    def _calculate_risk_metrics(self, state: TradingState) -> RiskMetrics:
        """计算综合风险指标"""
        # 合并所有收益率
        all_returns = []
        for returns in self.returns_history.values():
            all_returns.extend(returns)
        
        if not all_returns:
            return self._get_default_risk_metrics()
        
        returns_array = np.array(all_returns)
        
        # 计算VaR和CVaR
        var_95 = self.risk_model.calculate_var(returns_array, 0.95)
        var_99 = self.risk_model.calculate_var(returns_array, 0.99)
        cvar_95 = self.risk_model.calculate_cvar(returns_array, 0.95)
        cvar_99 = self.risk_model.calculate_cvar(returns_array, 0.99)
        
        # 计算回撤
        portfolio_values = self._calculate_portfolio_values(state)
        max_dd, current_dd = self.risk_model.calculate_max_drawdown(portfolio_values)
        
        # 计算比率
        sharpe = self.risk_model.calculate_sharpe_ratio(returns_array)
        sortino = self.risk_model.calculate_sortino_ratio(returns_array)
        calmar = abs(returns_array.mean() * 252 / max_dd) if max_dd != 0 else 0
        
        # 计算风险度量
        positions = {s: p.quantity for s, p in state.positions.items()}
        concentration = self.risk_model.assess_concentration_risk(positions)
        
        # 计算相关性风险
        correlation_risk = self._calculate_correlation_risk()
        
        # 计算流动性风险
        liquidity_risk = self._calculate_liquidity_risk(state)
        
        # 计算波动率
        volatility = returns_array.std() * np.sqrt(252) if len(returns_array) > 1 else 0
        downside_returns = returns_array[returns_array < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        return RiskMetrics(
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            cvar_99=abs(cvar_99),
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            beta=self._calculate_beta(returns_array),
            correlation_risk=correlation_risk,
            concentration_risk=concentration,
            liquidity_risk=liquidity_risk,
            volatility=volatility,
            downside_volatility=downside_vol
        )
    
    def _identify_risk_factors(self, state: TradingState, metrics: RiskMetrics) -> List[RiskFactor]:
        """识别关键风险因子"""
        risk_factors = []
        
        # 市场风险因子
        if metrics.volatility > 0.3:
            risk_factors.append(RiskFactor(
                factor_name="High Volatility",
                factor_type="market",
                impact_score=min(metrics.volatility / 0.5, 1.0),
                confidence=0.9,
                description=f"Market volatility ({metrics.volatility:.2%}) exceeds normal levels",
                mitigation_strategy="Reduce position sizes and increase diversification"
            ))
        
        # 回撤风险
        if metrics.current_drawdown > 0.1:
            risk_factors.append(RiskFactor(
                factor_name="Significant Drawdown",
                factor_type="market",
                impact_score=min(metrics.current_drawdown / 0.2, 1.0),
                confidence=1.0,
                description=f"Current drawdown ({metrics.current_drawdown:.2%}) is significant",
                mitigation_strategy="Consider reducing exposure or implementing stop-losses"
            ))
        
        # 集中度风险
        if metrics.concentration_risk > 0.5:
            risk_factors.append(RiskFactor(
                factor_name="Concentration Risk",
                factor_type="market",
                impact_score=metrics.concentration_risk,
                confidence=0.95,
                description="Portfolio is too concentrated in few positions",
                mitigation_strategy="Diversify across more assets"
            ))
        
        # 相关性风险
        if metrics.correlation_risk > 0.7:
            risk_factors.append(RiskFactor(
                factor_name="High Correlation",
                factor_type="market",
                impact_score=metrics.correlation_risk,
                confidence=0.85,
                description="Portfolio assets are highly correlated",
                mitigation_strategy="Add uncorrelated or negatively correlated assets"
            ))
        
        # 流动性风险
        if metrics.liquidity_risk > 0.3:
            risk_factors.append(RiskFactor(
                factor_name="Liquidity Risk",
                factor_type="liquidity",
                impact_score=metrics.liquidity_risk,
                confidence=0.8,
                description="Some positions may be difficult to exit quickly",
                mitigation_strategy="Reduce positions in illiquid assets"
            ))
        
        # VaR风险
        if metrics.var_95 > self.max_var_95:
            risk_factors.append(RiskFactor(
                factor_name="VaR Breach",
                factor_type="market",
                impact_score=min(metrics.var_95 / self.max_var_95, 1.0),
                confidence=0.95,
                description=f"95% VaR ({metrics.var_95:.2%}) exceeds limit",
                mitigation_strategy="Reduce overall portfolio risk"
            ))
        
        return risk_factors
    
    async def _calculate_opinion_divergence(self, state: TradingState) -> Dict[str, float]:
        """计算Agent意见分歧度"""
        divergence = {}
        
        if self.state_manager:
            # 获取当前状态
            agent_state = await self.state_manager.get_state(state.session_id)
            
            if agent_state and "analyst_opinions" in agent_state:
                opinions = agent_state["analyst_opinions"]
                
                # 按交易对分组意见
                symbol_opinions = {}
                for opinion in opinions:
                    symbol = opinion.get("symbol")
                    if symbol:
                        if symbol not in symbol_opinions:
                            symbol_opinions[symbol] = []
                        symbol_opinions[symbol].append(opinion.get("recommendation", 0))
                
                # 计算每个交易对的分歧度
                for symbol, recommendations in symbol_opinions.items():
                    if len(recommendations) > 1:
                        # 使用标准差作为分歧度指标
                        divergence[symbol] = np.std(recommendations)
                    else:
                        divergence[symbol] = 0.0
        
        self.opinion_divergence = divergence
        return divergence
    
    def _evaluate_risk_level(self, 
                            metrics: RiskMetrics, 
                            factors: List[RiskFactor],
                            divergence: Dict[str, float]) -> Tuple[RiskLevel, float]:
        """评估整体风险水平"""
        risk_score = 0.0
        weights = {
            "var": 0.15,
            "drawdown": 0.20,
            "volatility": 0.15,
            "concentration": 0.15,
            "correlation": 0.10,
            "liquidity": 0.10,
            "factors": 0.10,
            "divergence": 0.05
        }
        
        # VaR评分
        var_score = min(metrics.var_95 / self.max_var_95, 1.0) * 100
        risk_score += var_score * weights["var"]
        
        # 回撤评分
        dd_score = min(metrics.current_drawdown / self.max_drawdown, 1.0) * 100
        risk_score += dd_score * weights["drawdown"]
        
        # 波动率评分
        vol_score = min(metrics.volatility / 0.5, 1.0) * 100
        risk_score += vol_score * weights["volatility"]
        
        # 集中度评分
        conc_score = metrics.concentration_risk * 100
        risk_score += conc_score * weights["concentration"]
        
        # 相关性评分
        corr_score = metrics.correlation_risk * 100
        risk_score += corr_score * weights["correlation"]
        
        # 流动性评分
        liq_score = metrics.liquidity_risk * 100
        risk_score += liq_score * weights["liquidity"]
        
        # 风险因子评分
        if factors:
            factor_score = np.mean([f.impact_score for f in factors]) * 100
            risk_score += factor_score * weights["factors"]
        
        # 意见分歧度评分
        if divergence:
            div_score = min(np.mean(list(divergence.values())) * 20, 100)
            risk_score += div_score * weights["divergence"]
        
        # 确定风险等级
        if risk_score < 20:
            risk_level = RiskLevel.MINIMAL
        elif risk_score < 40:
            risk_level = RiskLevel.LOW
        elif risk_score < 60:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 80:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME
        
        return risk_level, risk_score
    
    def _calculate_position_recommendations(self,
                                           state: TradingState,
                                           metrics: RiskMetrics,
                                           risk_level: RiskLevel) -> Dict[str, float]:
        """计算推荐仓位大小"""
        recommendations = {}
        
        # 基础仓位大小（Kelly公式的保守版本）
        base_position_size = 0.02  # 2% 基础仓位
        
        # 根据风险水平调整
        risk_multipliers = {
            RiskLevel.MINIMAL: 1.5,
            RiskLevel.LOW: 1.2,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 0.5,
            RiskLevel.EXTREME: 0.2
        }
        
        risk_multiplier = risk_multipliers.get(risk_level, 1.0)
        
        for symbol in state.active_symbols:
            # 获取该交易对的特定风险
            symbol_volatility = self._get_symbol_volatility(symbol)
            symbol_divergence = self.opinion_divergence.get(symbol, 0)
            
            # 计算调整后的仓位
            position_size = base_position_size * risk_multiplier
            
            # 根据波动率调整
            if symbol_volatility > 0:
                vol_adjustment = min(0.2 / symbol_volatility, 1.5)
                position_size *= vol_adjustment
            
            # 根据意见分歧度调整
            if symbol_divergence > 0.3:
                position_size *= 0.7  # 高分歧度时减少仓位
            
            # 应用最大仓位限制
            position_size = min(position_size, 0.05)  # 单个仓位不超过5%
            
            recommendations[symbol] = position_size
        
        return recommendations
    
    def _calculate_stop_loss_levels(self, 
                                   state: TradingState,
                                   metrics: RiskMetrics) -> Dict[str, float]:
        """计算动态止损水平"""
        stop_loss_levels = {}
        
        for symbol, market_data in state.market_data.items():
            current_price = market_data.close
            
            # 获取该交易对的波动率
            symbol_volatility = self._get_symbol_volatility(symbol)
            
            # 基础止损百分比
            base_stop_loss = 0.02  # 2%
            
            # 根据波动率调整
            if symbol_volatility > 0:
                # 使用2倍标准差作为止损距离
                volatility_stop = symbol_volatility * 2
                stop_loss_pct = max(base_stop_loss, min(volatility_stop, 0.05))
            else:
                stop_loss_pct = base_stop_loss
            
            # 根据风险指标调整
            if metrics.current_drawdown > 0.1:
                stop_loss_pct *= 0.8  # 已有回撤时收紧止损
            
            stop_loss_levels[symbol] = current_price * (1 - stop_loss_pct)
        
        return stop_loss_levels
    
    def _generate_risk_signals(self, assessment: RiskAssessment, state: TradingState) -> List[Signal]:
        """生成风险调整信号"""
        signals = []
        
        for symbol in state.active_symbols:
            # 基础信号强度
            signal_strength = 0.0
            
            # 根据风险等级调整
            if assessment.risk_level == RiskLevel.EXTREME:
                # 极端风险时生成卖出信号
                signal_strength = -0.8
                action = "SELL"
                reason = "Extreme risk detected - reducing exposure"
            elif assessment.risk_level == RiskLevel.HIGH:
                # 高风险时减少仓位
                signal_strength = -0.4
                action = "REDUCE"
                reason = "High risk level - reducing position size"
            elif assessment.risk_level == RiskLevel.LOW:
                # 低风险时可以考虑增加仓位
                signal_strength = 0.2
                action = "HOLD"
                reason = "Low risk environment - maintaining positions"
            else:
                action = "HOLD"
                reason = "Moderate risk level"
            
            # 创建风险信号
            signal = Signal(
                source=f"{self.name}_risk",
                symbol=symbol,
                action=action,
                strength=signal_strength,
                confidence=0.8,
                reason=reason,
                metadata={
                    "risk_level": assessment.risk_level.value,
                    "risk_score": assessment.risk_score,
                    "recommended_position": assessment.position_recommendations.get(symbol, 0),
                    "stop_loss": assessment.stop_loss_levels.get(symbol, 0),
                    "risk_warnings": assessment.risk_warnings[:3]  # 前3个警告
                }
            )
            
            # 简单验证信号 (避免异步调用在这个上下文中)
            if signal.symbol and signal.action and -1 <= signal.strength <= 1:
                signals.append(signal)
        
        return signals
    
    async def _update_state_manager(self, assessment: RiskAssessment, session_id: str):
        """更新状态管理器中的风险评估"""
        if not self.state_manager:
            return
        
        risk_state = RiskAssessmentState(
            risk_level=assessment.risk_level.value,
            var_95=assessment.risk_metrics.var_95,
            var_99=assessment.risk_metrics.var_99,
            max_drawdown=assessment.risk_metrics.max_drawdown,
            sharpe_ratio=assessment.risk_metrics.sharpe_ratio,
            exposure_ratio=sum(assessment.position_recommendations.values()),
            concentration_risk=assessment.risk_metrics.concentration_risk,
            liquidity_risk=assessment.risk_metrics.liquidity_risk,
            market_risk=assessment.risk_score / 100,
            operational_risk=0.0,  # 可以扩展
            risk_factors=[{
                "name": f.factor_name,
                "type": f.factor_type,
                "impact": f.impact_score,
                "description": f.description
            } for f in assessment.risk_factors],
            mitigation_strategies=assessment.mitigation_strategies,
            timestamp=assessment.timestamp
        )
        
        await self.state_manager.update_state(
            session_id,
            {"risk_assessment": risk_state.__dict__},
            self.name
        )
    
    def _calculate_portfolio_values(self, state: TradingState) -> np.ndarray:
        """计算投资组合价值序列"""
        if not self.price_history:
            return np.array([1.0])
        
        # 简化：使用等权重计算
        min_length = min(len(prices) for prices in self.price_history.values())
        if min_length == 0:
            return np.array([1.0])
        
        portfolio_values = []
        for i in range(min_length):
            value = sum(prices[i] for prices in self.price_history.values())
            portfolio_values.append(value)
        
        if not portfolio_values:
            return np.array([1.0])
        
        # 标准化
        portfolio_values = np.array(portfolio_values)
        if portfolio_values[0] != 0:
            portfolio_values = portfolio_values / portfolio_values[0]
        
        return portfolio_values
    
    def _calculate_correlation_risk(self) -> float:
        """计算相关性风险"""
        if len(self.returns_history) < 2:
            return 0.0
        
        correlation_matrix = self.risk_model.calculate_correlation_matrix(self.returns_history)
        
        # 计算平均相关性（排除对角线）
        n = len(correlation_matrix)
        if n <= 1:
            return 0.0
        
        total_corr = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_corr += abs(correlation_matrix[i, j])
                count += 1
        
        return total_corr / count if count > 0 else 0.0
    
    def _calculate_liquidity_risk(self, state: TradingState) -> float:
        """计算流动性风险"""
        liquidity_scores = []
        
        for symbol, market_data in state.market_data.items():
            # 使用成交量作为流动性指标
            if market_data.volume > 0:
                # 假设日均成交量的倒数作为流动性风险
                # 这里简化处理，实际应该使用更复杂的流动性模型
                avg_volume = market_data.volume  # 应该使用历史平均
                position_size = state.positions.get(symbol, None)
                
                if position_size and avg_volume > 0:
                    # 仓位占日均成交量的比例
                    liquidity_score = abs(position_size.quantity) / avg_volume
                    liquidity_scores.append(min(liquidity_score, 1.0))
        
        return np.mean(liquidity_scores) if liquidity_scores else 0.0
    
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """计算贝塔系数"""
        # 简化：假设市场收益率
        market_returns = np.random.normal(0.0001, 0.01, len(returns))
        
        if len(returns) < 2:
            return 1.0
        
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """获取特定交易对的波动率"""
        if symbol not in self.returns_history:
            return 0.0
        
        returns = np.array(self.returns_history[symbol])
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def _generate_risk_warnings(self, metrics: RiskMetrics, factors: List[RiskFactor]) -> List[str]:
        """生成风险警告"""
        warnings = []
        
        if metrics.var_95 > self.max_var_95:
            warnings.append(f"95% VaR ({metrics.var_95:.2%}) exceeds limit ({self.max_var_95:.2%})")
        
        if metrics.max_drawdown > self.max_drawdown:
            warnings.append(f"Maximum drawdown ({metrics.max_drawdown:.2%}) exceeds limit")
        
        if metrics.sharpe_ratio < self.min_sharpe:
            warnings.append(f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) below minimum threshold")
        
        if metrics.concentration_risk > self.max_concentration:
            warnings.append("Portfolio concentration risk is too high")
        
        for factor in factors[:2]:  # 添加前2个最重要的风险因子
            warnings.append(f"{factor.factor_name}: {factor.description}")
        
        return warnings
    
    def _generate_mitigation_strategies(self, factors: List[RiskFactor], risk_level: RiskLevel) -> List[str]:
        """生成风险缓解策略"""
        strategies = []
        
        # 基础策略
        if risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            strategies.append("Immediately reduce overall portfolio exposure")
            strategies.append("Implement strict stop-loss orders on all positions")
        
        if risk_level == RiskLevel.MODERATE:
            strategies.append("Monitor positions closely and prepare to reduce if risk increases")
            strategies.append("Diversify into uncorrelated assets")
        
        # 根据风险因子添加特定策略
        for factor in factors:
            if factor.mitigation_strategy not in strategies:
                strategies.append(factor.mitigation_strategy)
        
        # 限制策略数量
        return strategies[:5]
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """获取默认风险指标"""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            beta=1.0,
            correlation_risk=0.0,
            concentration_risk=0.0,
            liquidity_risk=0.0,
            volatility=0.0,
            downside_volatility=0.0
        )
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险摘要"""
        if not self.last_assessment:
            return {"status": "No risk assessment available"}
        
        assessment = self.last_assessment
        return {
            "risk_level": assessment.risk_level.value,
            "risk_score": round(assessment.risk_score, 2),
            "metrics": {
                "var_95": round(assessment.risk_metrics.var_95, 4),
                "var_99": round(assessment.risk_metrics.var_99, 4),
                "max_drawdown": round(assessment.risk_metrics.max_drawdown, 4),
                "sharpe_ratio": round(assessment.risk_metrics.sharpe_ratio, 2),
                "volatility": round(assessment.risk_metrics.volatility, 4)
            },
            "top_risks": [f.factor_name for f in assessment.risk_factors[:3]],
            "warnings": assessment.risk_warnings[:3],
            "strategies": assessment.mitigation_strategies[:3],
            "timestamp": assessment.timestamp.isoformat()
        }