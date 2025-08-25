"""
量化分析师Agent
专注于数据驱动的系统化交易策略
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np
from scipy import stats

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class QuantitativeAnalystAgent(InvestmentMasterAgent):
    """量化分析师Agent"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化量化分析师Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="quantitative_analyst_agent",
                master_name="Quantitative Analyst",
                investment_style=InvestmentStyle.QUANTITATIVE,
                specialty=[
                    "统计套利",
                    "因子模型",
                    "算法交易",
                    "风险模型",
                    "高频交易",
                    "机器学习"
                ],
                llm_model="gpt-4",
                llm_temperature=0.3,  # 低温度，更确定性
                analysis_depth="comprehensive",
                risk_tolerance="moderate",
                time_horizon="short",  # 量化策略通常较短期
                personality_traits={
                    "mathematical": "extreme",
                    "systematic": "pure",
                    "data_driven": "absolute",
                    "emotion_free": "complete",
                    "backtesting_focus": "high",
                    "optimization": "continuous"
                },
                favorite_indicators=[
                    "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown",
                    "Beta", "Alpha", "Correlation", "Volatility",
                    "Value_at_Risk", "Information_Ratio", "Calmar_Ratio"
                ],
                avoid_sectors=[]  # 量化不限制行业
            )
        
        super().__init__(config, message_bus)
        
        # 量化参数
        self.quant_params = {
            "min_sharpe_ratio": 1.0,
            "max_drawdown_tolerance": 0.20,
            "min_sample_size": 100,
            "confidence_level": 0.95,
            "backtest_period_years": 3,
            "rebalance_frequency": "monthly",
            "position_sizing_method": "kelly_criterion"
        }
        
        # 因子库
        self.factors = {
            "value": ["PE", "PB", "PS", "PCF"],
            "momentum": ["Returns_1M", "Returns_3M", "Returns_6M", "Returns_12M"],
            "quality": ["ROE", "ROA", "Gross_Margin", "Debt_to_Equity"],
            "volatility": ["Realized_Vol", "GARCH_Vol", "IV"],
            "liquidity": ["Volume", "Bid_Ask_Spread", "Turnover"],
            "size": ["Market_Cap", "Enterprise_Value"]
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        量化风格：基于统计模型和回测验证
        """
        reasoning_steps = []
        
        # 步骤1：因子分析
        factor_step = await self._analyze_factors(state)
        reasoning_steps.append(factor_step)
        
        # 步骤2：统计套利机会
        arbitrage_step = await self._identify_arbitrage_opportunities(state)
        reasoning_steps.append(arbitrage_step)
        
        # 步骤3：风险模型评估
        risk_step = await self._evaluate_risk_models(state, portfolio)
        reasoning_steps.append(risk_step)
        
        # 步骤4：优化组合配置
        optimization_step = await self._optimize_portfolio_allocation(state, portfolio)
        reasoning_steps.append(optimization_step)
        
        # 步骤5：回测验证
        backtest_step = await self._validate_with_backtest(state, optimization_step)
        reasoning_steps.append(backtest_step)
        
        # 综合决策
        decision = self._synthesize_decision(reasoning_steps, portfolio)
        
        return FinalDecision(
            action=decision["action"],
            confidence=decision["confidence"],
            reasoning_chain=reasoning_steps,
            position_size=decision["position_size"],
            risk_assessment=decision["risk_assessment"],
            expected_return=decision["expected_return"],
            time_horizon=decision["time_horizon"],
            metadata={
                "investment_style": "Quantitative Systematic",
                "model_type": decision["model_type"],
                "sharpe_ratio": decision["sharpe_ratio"],
                "statistical_significance": decision["statistical_significance"],
                "factor_exposures": decision["factor_exposures"]
            }
        )
    
    async def _analyze_factors(self, state: TradingState) -> ReasoningStep:
        """因子分析"""
        factor_scores = {}
        
        for symbol in state.active_symbols:
            scores = {}
            
            # 计算各类因子得分
            for factor_type, factor_list in self.factors.items():
                factor_score = self._calculate_factor_score(symbol, factor_list, state)
                scores[factor_type] = factor_score
            
            # 计算综合因子得分
            composite_score = self._calculate_composite_factor_score(scores)
            
            factor_scores[symbol] = {
                "individual_scores": scores,
                "composite_score": composite_score,
                "rank": 0  # 将在后面计算排名
            }
        
        # 计算排名
        sorted_symbols = sorted(factor_scores.items(), 
                              key=lambda x: x[1]["composite_score"], 
                              reverse=True)
        for rank, (symbol, _) in enumerate(sorted_symbols):
            factor_scores[symbol]["rank"] = rank + 1
        
        top_quintile = len(factor_scores) // 5
        top_performers = [s for s, d in factor_scores.items() if d["rank"] <= top_quintile]
        
        return ReasoningStep(
            thought="因子分析 - 多因子模型筛选",
            action="factor_analysis",
            observation=f"顶部五分位标的: {len(top_performers)}",
            confidence=0.75,
            metadata={
                "factor_scores": factor_scores,
                "top_performers": top_performers
            }
        )
    
    async def _identify_arbitrage_opportunities(self, state: TradingState) -> ReasoningStep:
        """识别统计套利机会"""
        arbitrage_opportunities = []
        
        # 配对交易机会
        pairs = self._find_cointegrated_pairs(state)
        for pair in pairs:
            symbol1, symbol2 = pair
            spread = self._calculate_spread(symbol1, symbol2, state)
            z_score = self._calculate_z_score(spread)
            
            if abs(z_score) > 2:  # 2个标准差以外
                opportunities.append({
                    "type": "pairs_trading",
                    "symbols": pair,
                    "z_score": z_score,
                    "expected_return": abs(z_score) * 0.02,  # 简化计算
                    "confidence": min(abs(z_score) / 3, 1.0)
                })
        
        # 均值回归机会
        for symbol in state.active_symbols:
            if self._is_mean_reverting(symbol, state):
                deviation = self._calculate_deviation_from_mean(symbol, state)
                if abs(deviation) > 0.15:  # 15%偏离
                    arbitrage_opportunities.append({
                        "type": "mean_reversion",
                        "symbol": symbol,
                        "deviation": deviation,
                        "expected_return": abs(deviation) * 0.5,
                        "confidence": 0.6
                    })
        
        avg_confidence = np.mean([o["confidence"] for o in arbitrage_opportunities]) if arbitrage_opportunities else 0
        
        return ReasoningStep(
            thought="统计套利 - 寻找市场无效性",
            action="arbitrage_identification",
            observation=f"发现套利机会: {len(arbitrage_opportunities)}",
            confidence=avg_confidence,
            metadata={"opportunities": arbitrage_opportunities}
        )
    
    async def _evaluate_risk_models(self, state: TradingState, 
                                   portfolio: Dict[str, Any]) -> ReasoningStep:
        """评估风险模型"""
        risk_metrics = {
            "portfolio_volatility": self._calculate_portfolio_volatility(portfolio),
            "value_at_risk": self._calculate_var(portfolio, confidence=0.95),
            "conditional_var": self._calculate_cvar(portfolio, confidence=0.95),
            "max_drawdown": self._calculate_max_drawdown(portfolio),
            "beta": self._calculate_portfolio_beta(portfolio),
            "correlation_risk": self._assess_correlation_risk(portfolio)
        }
        
        # 风险评分
        risk_score = 1.0
        if risk_metrics["portfolio_volatility"] > 0.20:
            risk_score *= 0.8
        if risk_metrics["max_drawdown"] > self.quant_params["max_drawdown_tolerance"]:
            risk_score *= 0.7
        if risk_metrics["correlation_risk"] > 0.7:
            risk_score *= 0.9
        
        risk_acceptable = risk_score > 0.6
        
        return ReasoningStep(
            thought="风险模型评估 - 量化风险指标",
            action="risk_evaluation",
            observation=f"风险评分: {risk_score:.2f}, 可接受: {risk_acceptable}",
            confidence=risk_score,
            metadata={
                "risk_metrics": risk_metrics,
                "risk_acceptable": risk_acceptable
            }
        )
    
    async def _optimize_portfolio_allocation(self, state: TradingState, 
                                            portfolio: Dict[str, Any]) -> ReasoningStep:
        """优化组合配置"""
        # 获取因子得分高的标的
        top_symbols = self._get_top_factor_symbols(state)
        
        # 计算最优权重（简化的均值方差优化）
        optimal_weights = self._mean_variance_optimization(top_symbols, state)
        
        # Kelly准则调整仓位
        kelly_adjustments = {}
        for symbol, weight in optimal_weights.items():
            kelly_fraction = self._calculate_kelly_criterion(symbol, state)
            adjusted_weight = weight * min(kelly_fraction, 0.25)  # 限制最大25%
            kelly_adjustments[symbol] = adjusted_weight
        
        # 计算预期夏普比率
        expected_sharpe = self._calculate_expected_sharpe(kelly_adjustments, state)
        
        return ReasoningStep(
            thought="组合优化 - 均值方差优化与Kelly准则",
            action="portfolio_optimization",
            observation=f"预期夏普比率: {expected_sharpe:.2f}",
            confidence=min(expected_sharpe / 2, 1.0),
            metadata={
                "optimal_weights": optimal_weights,
                "kelly_adjusted": kelly_adjustments,
                "expected_sharpe": expected_sharpe
            }
        )
    
    async def _validate_with_backtest(self, state: TradingState, 
                                     optimization_step: ReasoningStep) -> ReasoningStep:
        """回测验证"""
        weights = optimization_step.metadata.get("kelly_adjusted", {})
        
        # 简化的回测
        backtest_results = {
            "annual_return": 0.12,  # 12%年化收益
            "annual_volatility": 0.15,  # 15%年化波动
            "sharpe_ratio": 0.8,
            "max_drawdown": 0.18,
            "win_rate": 0.55,
            "profit_factor": 1.4,
            "total_trades": 250
        }
        
        # 统计显著性检验
        t_statistic = backtest_results["sharpe_ratio"] * np.sqrt(backtest_results["total_trades"] / 252)
        p_value = 1 - stats.norm.cdf(t_statistic)
        statistically_significant = p_value < 0.05
        
        # 验证是否满足量化标准
        meets_criteria = (
            backtest_results["sharpe_ratio"] >= self.quant_params["min_sharpe_ratio"] and
            backtest_results["max_drawdown"] <= self.quant_params["max_drawdown_tolerance"] and
            statistically_significant
        )
        
        return ReasoningStep(
            thought="回测验证 - 历史数据验证策略有效性",
            action="backtest_validation",
            observation=f"夏普: {backtest_results['sharpe_ratio']:.2f}, 显著性: {statistically_significant}",
            confidence=0.7 if meets_criteria else 0.3,
            metadata={
                "backtest_results": backtest_results,
                "p_value": p_value,
                "meets_criteria": meets_criteria
            }
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析做出决策"""
        # 提取关键信息
        factor_data = reasoning_steps[0].metadata
        arbitrage_data = reasoning_steps[1].metadata
        risk_data = reasoning_steps[2].metadata
        optimization_data = reasoning_steps[3].metadata
        backtest_data = reasoning_steps[4].metadata
        
        # 检查是否满足量化标准
        backtest_meets_criteria = backtest_data.get("meets_criteria", False)
        risk_acceptable = risk_data.get("risk_acceptable", False)
        has_opportunities = len(arbitrage_data.get("opportunities", [])) > 0
        
        # 决策逻辑
        if backtest_meets_criteria and risk_acceptable:
            action = "REBALANCE"  # 调整到最优配置
            confidence = 0.8
            weights = optimization_data.get("kelly_adjusted", {})
            position_size = sum(weights.values()) / len(weights) if weights else 0
            model_type = "Multi-Factor with Statistical Arbitrage"
        elif has_opportunities and risk_acceptable:
            action = "EXECUTE_ARBITRAGE"
            confidence = 0.7
            position_size = 0.05  # 套利仓位
            model_type = "Statistical Arbitrage"
        else:
            action = "HOLD"
            confidence = 0.5
            position_size = 0
            model_type = "Risk Control Mode"
        
        return {
            "action": action,
            "confidence": confidence,
            "position_size": position_size,
            "risk_assessment": "Quantitative risk metrics based assessment",
            "expected_return": f"{backtest_data.get('backtest_results', {}).get('annual_return', 0):.1%} annually",
            "time_horizon": "1-3 months",
            "model_type": model_type,
            "sharpe_ratio": optimization_data.get("expected_sharpe", 0),
            "statistical_significance": backtest_data.get("p_value", 1) < 0.05,
            "factor_exposures": self._get_factor_exposures(factor_data)
        }
    
    def _calculate_factor_score(self, symbol: str, factors: List[str], 
                               state: TradingState) -> float:
        """计算因子得分"""
        # 简化：返回随机得分
        return np.random.uniform(-1, 1)
    
    def _calculate_composite_factor_score(self, scores: Dict[str, float]) -> float:
        """计算综合因子得分"""
        # 等权重平均
        return np.mean(list(scores.values()))
    
    def _find_cointegrated_pairs(self, state: TradingState) -> List[tuple]:
        """寻找协整对"""
        # 简化：返回模拟对
        pairs = []
        symbols = list(state.active_symbols)
        if len(symbols) >= 2:
            pairs.append((symbols[0], symbols[1]))
        return pairs
    
    def _calculate_spread(self, symbol1: str, symbol2: str, state: TradingState) -> float:
        """计算价差"""
        return np.random.normal(0, 1)
    
    def _calculate_z_score(self, spread: float) -> float:
        """计算Z分数"""
        return spread  # 简化：直接返回
    
    def _is_mean_reverting(self, symbol: str, state: TradingState) -> bool:
        """检查是否均值回归"""
        return np.random.random() < 0.3
    
    def _calculate_deviation_from_mean(self, symbol: str, state: TradingState) -> float:
        """计算偏离均值程度"""
        return np.random.uniform(-0.3, 0.3)
    
    def _calculate_portfolio_volatility(self, portfolio: Dict[str, Any]) -> float:
        """计算组合波动率"""
        return 0.15  # 简化：15%年化波动
    
    def _calculate_var(self, portfolio: Dict[str, Any], confidence: float) -> float:
        """计算VaR"""
        return 0.05  # 简化：5%的VaR
    
    def _calculate_cvar(self, portfolio: Dict[str, Any], confidence: float) -> float:
        """计算CVaR"""
        return 0.08  # 简化：8%的CVaR
    
    def _calculate_max_drawdown(self, portfolio: Dict[str, Any]) -> float:
        """计算最大回撤"""
        return 0.12  # 简化：12%回撤
    
    def _calculate_portfolio_beta(self, portfolio: Dict[str, Any]) -> float:
        """计算组合Beta"""
        return 0.9  # 简化
    
    def _assess_correlation_risk(self, portfolio: Dict[str, Any]) -> float:
        """评估相关性风险"""
        return 0.5  # 简化
    
    def _get_top_factor_symbols(self, state: TradingState) -> List[str]:
        """获取因子得分最高的标的"""
        return list(state.active_symbols)[:5]  # 简化：返回前5个
    
    def _mean_variance_optimization(self, symbols: List[str], state: TradingState) -> Dict[str, float]:
        """均值方差优化"""
        # 简化：等权重
        weight = 1.0 / len(symbols) if symbols else 0
        return {symbol: weight for symbol in symbols}
    
    def _calculate_kelly_criterion(self, symbol: str, state: TradingState) -> float:
        """计算Kelly准则仓位"""
        # Kelly = (p*b - q) / b
        # p: 获胜概率, q: 失败概率, b: 赔率
        p = 0.55  # 55%胜率
        b = 2.0   # 2:1赔率
        q = 1 - p
        kelly = (p * b - q) / b
        return max(0, min(kelly, 0.25))  # 限制在0-25%
    
    def _calculate_expected_sharpe(self, weights: Dict[str, float], state: TradingState) -> float:
        """计算预期夏普比率"""
        return 1.2  # 简化
    
    def _get_factor_exposures(self, factor_data: Dict[str, Any]) -> Dict[str, float]:
        """获取因子暴露"""
        exposures = {
            "value": 0.3,
            "momentum": 0.25,
            "quality": 0.2,
            "low_volatility": 0.15,
            "size": 0.1
        }
        return exposures
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        量化风格：数据驱动的系统化评估
        """
        evaluation = {
            "sharpe_ratio": 0.0,
            "information_ratio": 0.0,
            "factor_alignment": 0.0,
            "risk_efficiency": 0.0,
            "recommendations": []
        }
        
        # 计算夏普比率
        returns = portfolio.get("returns", [])
        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
            evaluation["sharpe_ratio"] = sharpe
            
            if sharpe < self.quant_params["min_sharpe_ratio"]:
                evaluation["recommendations"].append(f"提升夏普比率至{self.quant_params['min_sharpe_ratio']}")
        
        # 信息比率
        evaluation["information_ratio"] = 0.8  # 简化
        
        # 因子对齐度
        evaluation["factor_alignment"] = 0.7
        
        # 风险效率
        vol = self._calculate_portfolio_volatility(portfolio)
        evaluation["risk_efficiency"] = 1 / (1 + vol)
        
        # 总体评分
        evaluation["overall_score"] = (
            min(evaluation["sharpe_ratio"] / 2, 0.5) +  # 夏普贡献最多50%
            evaluation["information_ratio"] * 0.2 +
            evaluation["factor_alignment"] * 0.15 +
            evaluation["risk_efficiency"] * 0.15
        )
        
        # 量化建议
        evaluation["quant_advice"] = self._get_quant_advice(evaluation["overall_score"])
        
        return evaluation
    
    def _get_quant_advice(self, score: float) -> str:
        """获取量化建议"""
        if score >= 0.8:
            return "优秀的量化组合。继续系统化执行，避免人为干预。"
        elif score >= 0.6:
            return "组合表现良好。考虑增加因子分散化，优化风险调整收益。"
        elif score >= 0.4:
            return "需要改进。检查模型假设，考虑重新校准参数。"
        else:
            return "表现不佳。全面审查策略，可能需要重新设计模型。"