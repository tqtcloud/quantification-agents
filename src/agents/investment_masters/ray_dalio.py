"""
Ray Dalio投资大师Agent
实现全天候投资策略和风险平价理念
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class RayDalioAgent(InvestmentMasterAgent):
    """Ray Dalio投资大师Agent - Bridgewater创始人"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化Ray Dalio Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="ray_dalio_agent",
                master_name="Ray Dalio",
                investment_style=InvestmentStyle.MACRO,
                specialty=[
                    "全天候策略",
                    "风险平价",
                    "经济机器理论",
                    "债务周期分析",
                    "全球宏观配置"
                ],
                llm_model="gpt-4",
                llm_temperature=0.4,  # 系统化思维
                analysis_depth="comprehensive",
                risk_tolerance="moderate",
                time_horizon="long",
                personality_traits={
                    "systematic": "extreme",
                    "principle_based": "core",
                    "diversification": "extreme",
                    "transparency": "high",
                    "data_driven": "extreme",
                    "cycle_awareness": "expert"
                },
                favorite_indicators=[
                    "GDP_Growth", "Inflation_Rate", "Interest_Rates",
                    "Credit_Spreads", "Currency_Strength", "Commodity_Prices",
                    "Debt_to_GDP", "Yield_Curve", "VIX"
                ],
                avoid_sectors=[]  # Dalio投资所有资产类别
            )
        
        super().__init__(config, message_bus)
        
        # 全天候策略的四个经济环境
        self.economic_environments = {
            "growth_rising": {
                "probability": 0.25,
                "favorable_assets": ["stocks", "commodities", "corporate_credit", "EM_bonds"]
            },
            "growth_falling": {
                "probability": 0.25,
                "favorable_assets": ["nominal_bonds", "inflation_linked_bonds"]
            },
            "inflation_rising": {
                "probability": 0.25,
                "favorable_assets": ["inflation_linked_bonds", "commodities", "EM_bonds"]
            },
            "inflation_falling": {
                "probability": 0.25,
                "favorable_assets": ["stocks", "nominal_bonds"]
            }
        }
        
        # 风险平价目标权重
        self.risk_parity_targets = {
            "stocks": 0.30,
            "long_term_bonds": 0.40,
            "intermediate_bonds": 0.15,
            "commodities": 0.075,
            "gold": 0.075
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        Dalio风格：基于经济环境的系统化配置
        """
        reasoning_steps = []
        
        # 步骤1：分析经济机器状态
        economic_step = await self._analyze_economic_machine(state)
        reasoning_steps.append(economic_step)
        
        # 步骤2：识别债务周期位置
        debt_cycle_step = await self._identify_debt_cycle_position(state)
        reasoning_steps.append(debt_cycle_step)
        
        # 步骤3：评估市场环境
        environment_step = await self._assess_market_environment(state)
        reasoning_steps.append(environment_step)
        
        # 步骤4：计算风险平价配置
        risk_parity_step = await self._calculate_risk_parity_allocation(state, portfolio)
        reasoning_steps.append(risk_parity_step)
        
        # 步骤5：应用战术调整
        tactical_step = await self._apply_tactical_adjustments(state, risk_parity_step)
        reasoning_steps.append(tactical_step)
        
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
                "investment_style": "All Weather Strategy",
                "economic_environment": decision["environment"],
                "allocation": decision["allocation"],
                "principles_applied": decision["principles"]
            }
        )
    
    async def _analyze_economic_machine(self, state: TradingState) -> ReasoningStep:
        """分析经济机器状态"""
        economic_indicators = {
            "productivity_growth": 0.02,  # 长期2%增长
            "short_term_debt_cycle": "expansion",  # 5-8年周期
            "long_term_debt_cycle": "late_stage",  # 50-75年周期
            "credit_availability": "moderate",
            "money_velocity": "declining"
        }
        
        # 评估三大驱动力
        drivers = {
            "productivity": 0.6,  # 生产力增长
            "short_term_cycle": 0.7,  # 短期债务周期
            "long_term_cycle": 0.4  # 长期债务周期
        }
        
        machine_health = sum(drivers.values()) / len(drivers)
        
        return ReasoningStep(
            thought="分析经济机器 - 理解生产力增长和债务周期的相互作用",
            action="economic_machine_analysis",
            observation=f"经济机器健康度: {machine_health:.2f}",
            confidence=machine_health,
            metadata={
                "indicators": economic_indicators,
                "drivers": drivers
            }
        )
    
    async def _identify_debt_cycle_position(self, state: TradingState) -> ReasoningStep:
        """识别债务周期位置"""
        debt_metrics = {
            "total_debt_to_gdp": 2.5,  # 示例值
            "debt_service_ratio": 0.15,
            "interest_coverage": 3.0,
            "credit_growth": 0.05,
            "deleveraging_pressure": 0.3
        }
        
        # 判断周期阶段
        if debt_metrics["total_debt_to_gdp"] > 3.0:
            cycle_phase = "deleveraging"
            phase_score = 0.3
        elif debt_metrics["credit_growth"] > 0.10:
            cycle_phase = "bubble"
            phase_score = 0.4
        elif debt_metrics["credit_growth"] > 0.05:
            cycle_phase = "expansion"
            phase_score = 0.7
        else:
            cycle_phase = "contraction"
            phase_score = 0.5
        
        return ReasoningStep(
            thought="识别债务周期 - 确定我们在长期和短期债务周期中的位置",
            action="debt_cycle_identification",
            observation=f"债务周期阶段: {cycle_phase}",
            confidence=phase_score,
            metadata={
                "debt_metrics": debt_metrics,
                "cycle_phase": cycle_phase,
                "implications": self._get_cycle_implications(cycle_phase)
            }
        )
    
    async def _assess_market_environment(self, state: TradingState) -> ReasoningStep:
        """评估市场环境（四象限）"""
        # 简化的环境判断
        growth_indicator = 0.6  # 示例：基于GDP等指标
        inflation_indicator = 0.4  # 示例：基于CPI等指标
        
        if growth_indicator > 0.5 and inflation_indicator > 0.5:
            environment = "growth_rising_inflation_rising"
            favorable_assets = ["commodities", "real_estate", "tips"]
        elif growth_indicator > 0.5 and inflation_indicator <= 0.5:
            environment = "growth_rising_inflation_falling"
            favorable_assets = ["stocks", "corporate_bonds"]
        elif growth_indicator <= 0.5 and inflation_indicator > 0.5:
            environment = "growth_falling_inflation_rising"
            favorable_assets = ["commodities", "inflation_bonds"]
        else:
            environment = "growth_falling_inflation_falling"
            favorable_assets = ["government_bonds", "cash"]
        
        environment_score = 0.7  # 环境清晰度
        
        return ReasoningStep(
            thought="评估市场环境 - 识别增长和通胀的四种组合",
            action="environment_assessment",
            observation=f"当前环境: {environment}",
            confidence=environment_score,
            metadata={
                "environment": environment,
                "growth": growth_indicator,
                "inflation": inflation_indicator,
                "favorable_assets": favorable_assets
            }
        )
    
    async def _calculate_risk_parity_allocation(self, state: TradingState, 
                                               portfolio: Dict[str, Any]) -> ReasoningStep:
        """计算风险平价配置"""
        # 简化的风险平价计算
        asset_volatilities = {
            "stocks": 0.15,
            "bonds": 0.05,
            "commodities": 0.20,
            "gold": 0.12,
            "real_estate": 0.10
        }
        
        # 目标：每个资产贡献相等的风险
        total_inv_vol = sum(1/vol for vol in asset_volatilities.values())
        
        risk_parity_weights = {}
        for asset, vol in asset_volatilities.items():
            weight = (1/vol) / total_inv_vol
            risk_parity_weights[asset] = round(weight, 3)
        
        # 计算与当前组合的偏差
        current_weights = self._get_current_weights(portfolio)
        rebalancing_needed = self._calculate_rebalancing_need(current_weights, risk_parity_weights)
        
        return ReasoningStep(
            thought="计算风险平价 - 平衡各资产类别的风险贡献",
            action="risk_parity_calculation",
            observation=f"需要再平衡: {rebalancing_needed:.1%}",
            confidence=0.8,
            metadata={
                "target_weights": risk_parity_weights,
                "current_weights": current_weights,
                "rebalancing_needed": rebalancing_needed
            }
        )
    
    async def _apply_tactical_adjustments(self, state: TradingState, 
                                         risk_parity_step: ReasoningStep) -> ReasoningStep:
        """应用战术调整"""
        base_allocation = risk_parity_step.metadata.get("target_weights", {})
        environment_data = state.metadata if hasattr(state, 'metadata') else {}
        
        # 基于环境的战术调整
        adjusted_allocation = base_allocation.copy()
        
        # 示例调整逻辑
        if environment_data.get("volatility", "normal") == "high":
            # 高波动时增加避险资产
            if "bonds" in adjusted_allocation:
                adjusted_allocation["bonds"] *= 1.2
            if "stocks" in adjusted_allocation:
                adjusted_allocation["stocks"] *= 0.8
        
        # 归一化权重
        total_weight = sum(adjusted_allocation.values())
        for asset in adjusted_allocation:
            adjusted_allocation[asset] /= total_weight
        
        adjustment_magnitude = sum(
            abs(adjusted_allocation.get(asset, 0) - base_allocation.get(asset, 0))
            for asset in set(adjusted_allocation) | set(base_allocation)
        )
        
        return ReasoningStep(
            thought="应用战术调整 - 基于当前环境微调战略配置",
            action="tactical_adjustment",
            observation=f"调整幅度: {adjustment_magnitude:.1%}",
            confidence=0.7,
            metadata={
                "adjusted_allocation": adjusted_allocation,
                "base_allocation": base_allocation,
                "adjustment_reasons": ["volatility_management", "cycle_positioning"]
            }
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析做出决策"""
        # 提取关键信息
        economic_health = reasoning_steps[0].confidence
        cycle_position = reasoning_steps[1].metadata.get("cycle_phase", "unknown")
        environment = reasoning_steps[2].metadata.get("environment", "unknown")
        target_allocation = reasoning_steps[4].metadata.get("adjusted_allocation", {})
        
        # Dalio风格：系统化、多元化
        overall_confidence = np.mean([step.confidence for step in reasoning_steps])
        
        # 决定行动
        rebalancing_needed = reasoning_steps[3].metadata.get("rebalancing_needed", 0)
        
        if rebalancing_needed > 0.05:  # 需要5%以上的再平衡
            action = "REBALANCE"
            position_size = min(rebalancing_needed * 0.5, 0.10)  # 渐进调整
        elif overall_confidence < 0.4:
            action = "REDUCE_RISK"
            position_size = -0.05  # 减少风险暴露
        else:
            action = "HOLD"
            position_size = 0.0
        
        # 应用的原则
        principles = self._get_applied_principles(reasoning_steps)
        
        return {
            "action": action,
            "confidence": overall_confidence,
            "position_size": position_size,
            "risk_assessment": "Balanced risk through diversification and risk parity",
            "expected_return": "7-10% annually with lower volatility",
            "time_horizon": "All weather - perpetual",
            "environment": environment,
            "allocation": target_allocation,
            "principles": principles
        }
    
    def _get_cycle_implications(self, cycle_phase: str) -> List[str]:
        """获取周期阶段的含义"""
        implications = {
            "deleveraging": [
                "央行可能印钞",
                "通胀风险上升",
                "持有实物资产"
            ],
            "bubble": [
                "谨慎增加风险",
                "准备流动性",
                "关注信贷指标"
            ],
            "expansion": [
                "适度承担风险",
                "平衡配置",
                "监控过热迹象"
            ],
            "contraction": [
                "增加防御性",
                "持有现金",
                "等待机会"
            ]
        }
        return implications.get(cycle_phase, ["保持观察"])
    
    def _get_current_weights(self, portfolio: Dict[str, Any]) -> Dict[str, float]:
        """获取当前组合权重"""
        positions = portfolio.get("positions", {})
        total_value = portfolio.get("total_value", 0)
        
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in positions.items():
            asset_class = self._classify_asset(symbol)
            value = position.get("value", 0)
            weight = value / total_value
            
            if asset_class in weights:
                weights[asset_class] += weight
            else:
                weights[asset_class] = weight
        
        return weights
    
    def _classify_asset(self, symbol: str) -> str:
        """分类资产类别"""
        symbol_upper = symbol.upper()
        
        if "BOND" in symbol_upper or "TLT" in symbol_upper:
            return "bonds"
        elif "GOLD" in symbol_upper or "GLD" in symbol_upper:
            return "gold"
        elif "COMMODITY" in symbol_upper or "DJP" in symbol_upper:
            return "commodities"
        elif "REIT" in symbol_upper or "VNQ" in symbol_upper:
            return "real_estate"
        else:
            return "stocks"
    
    def _calculate_rebalancing_need(self, current: Dict[str, float], 
                                   target: Dict[str, float]) -> float:
        """计算再平衡需求"""
        all_assets = set(current.keys()) | set(target.keys())
        total_deviation = 0.0
        
        for asset in all_assets:
            current_weight = current.get(asset, 0)
            target_weight = target.get(asset, 0)
            total_deviation += abs(current_weight - target_weight)
        
        return total_deviation / 2  # 除以2因为买入和卖出被计算了两次
    
    def _get_applied_principles(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """获取应用的原则"""
        principles = [
            "1. 分散化是免费的午餐",
            "2. 平衡不同经济环境的风险",
            "3. 理解经济机器的运作",
            "4. 现金是垃圾（长期而言）",
            "5. 痛苦 + 反思 = 进步"
        ]
        
        # 根据分析添加特定原则
        cycle_phase = reasoning_steps[1].metadata.get("cycle_phase", "")
        if cycle_phase == "deleveraging":
            principles.append("6. 在去杠杆时期，现金和债券可能贬值")
        elif cycle_phase == "bubble":
            principles.append("6. 泡沫总是因为借贷过度而破裂")
        
        return principles[:5]  # 返回前5个最相关的原则
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        Dalio风格：追求全天候的稳健回报
        """
        evaluation = {
            "diversification_score": 0.0,
            "risk_balance": 0.0,
            "correlation_assessment": 0.0,
            "stress_test_result": 0.0,
            "recommendations": []
        }
        
        positions = portfolio.get("positions", {})
        
        # 评估分散化
        asset_classes = set()
        for symbol in positions.keys():
            asset_classes.add(self._classify_asset(symbol))
        
        if len(asset_classes) >= 4:
            evaluation["diversification_score"] = 0.9
        elif len(asset_classes) >= 2:
            evaluation["diversification_score"] = 0.6
        else:
            evaluation["diversification_score"] = 0.3
            evaluation["recommendations"].append("增加资产类别多样性")
        
        # 评估风险平衡
        current_weights = self._get_current_weights(portfolio)
        risk_balance = self._evaluate_risk_balance(current_weights)
        evaluation["risk_balance"] = risk_balance
        
        if risk_balance < 0.6:
            evaluation["recommendations"].append("调整配置以平衡各资产的风险贡献")
        
        # 相关性评估
        evaluation["correlation_assessment"] = 0.7  # 简化值
        
        # 压力测试
        stress_scenarios = {
            "2008_crisis": -0.30,
            "covid_crash": -0.20,
            "inflation_spike": -0.15,
            "growth_shock": -0.25
        }
        
        worst_case = min(stress_scenarios.values())
        evaluation["stress_test_result"] = 1 + worst_case  # 转换为0-1评分
        
        # 总体评分
        evaluation["overall_score"] = (
            evaluation["diversification_score"] * 0.30 +
            evaluation["risk_balance"] * 0.30 +
            evaluation["correlation_assessment"] * 0.20 +
            evaluation["stress_test_result"] * 0.20
        )
        
        # Dalio风格建议
        evaluation["dalio_principles"] = self._get_portfolio_principles(evaluation["overall_score"])
        
        return evaluation
    
    def _evaluate_risk_balance(self, weights: Dict[str, float]) -> float:
        """评估风险平衡度"""
        if not weights:
            return 0.0
        
        # 检查是否接近风险平价目标
        target = self.risk_parity_targets
        deviation = 0.0
        
        for asset in target:
            current = weights.get(asset, 0)
            target_weight = target[asset]
            deviation += abs(current - target_weight)
        
        # 转换为0-1评分
        balance_score = max(0, 1 - deviation)
        return balance_score
    
    def _get_portfolio_principles(self, score: float) -> str:
        """获取组合原则建议"""
        if score >= 0.8:
            return "优秀的全天候组合。保持当前的风险平价和多元化策略。"
        elif score >= 0.6:
            return "组合结构良好。考虑增加非相关资产以进一步降低风险。"
        elif score >= 0.4:
            return "需要改进风险平衡。记住：预测很难，所以要为各种情况做准备。"
        else:
            return "组合需要重构。应用全天候原则：在不同经济环境下都能表现良好。"