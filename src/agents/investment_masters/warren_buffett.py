"""
Warren Buffett投资大师Agent
实现价值投资理念，专注于长期持有优质公司
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class WarrenBuffettAgent(InvestmentMasterAgent):
    """Warren Buffett投资大师Agent"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化Warren Buffett Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="warren_buffett_agent",
                master_name="Warren Buffett",
                investment_style=InvestmentStyle.VALUE,
                specialty=[
                    "价值投资",
                    "长期持有",
                    "护城河分析",
                    "管理层评估",
                    "消费品行业"
                ],
                llm_model="gpt-4",
                llm_temperature=0.3,  # 保守稳健
                analysis_depth="comprehensive",
                risk_tolerance="conservative",
                time_horizon="long",
                personality_traits={
                    "patience": "extreme",
                    "conviction": "high",
                    "complexity_preference": "simple_business",
                    "holding_period": "forever",
                    "focus": "intrinsic_value",
                    "management_quality": "critical"
                },
                favorite_indicators=[
                    "PE", "PB", "ROE", "FCF",
                    "Debt_to_Equity", "Profit_Margin",
                    "Revenue_Growth", "Book_Value"
                ],
                avoid_sectors=["高科技", "生物技术", "加密货币"]
            )
        
        super().__init__(config, message_bus)
        
        # Buffett特有的投资准则
        self.investment_criteria = {
            "min_roe": 0.15,  # 最低ROE 15%
            "max_debt_to_equity": 0.5,  # 最大负债率50%
            "min_profit_margin": 0.10,  # 最低利润率10%
            "min_holding_period_years": 5,  # 最短持有期5年
            "max_pe_ratio": 25,  # 最大市盈率25
            "min_moat_score": 0.7  # 最低护城河评分70%
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        Buffett风格：只投资理解的业务，寻找有护城河的公司
        """
        reasoning_steps = []
        
        # 步骤1：评估业务理解度
        understanding_step = await self._evaluate_business_understanding(state)
        reasoning_steps.append(understanding_step)
        
        # 步骤2：分析护城河
        moat_step = await self._analyze_economic_moat(state)
        reasoning_steps.append(moat_step)
        
        # 步骤3：评估管理层
        management_step = await self._evaluate_management(state)
        reasoning_steps.append(management_step)
        
        # 步骤4：计算内在价值
        valuation_step = await self._calculate_intrinsic_value(state)
        reasoning_steps.append(valuation_step)
        
        # 步骤5：确定安全边际
        margin_step = await self._determine_margin_of_safety(state, valuation_step)
        reasoning_steps.append(margin_step)
        
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
                "investment_style": "Buffett Value Investing",
                "key_factors": decision["key_factors"],
                "warnings": decision["warnings"]
            }
        )
    
    async def _evaluate_business_understanding(self, state: TradingState) -> ReasoningStep:
        """评估对业务的理解程度"""
        understanding_score = 0.0
        details = []
        
        for symbol in state.active_symbols:
            # 检查是否在能力圈内
            if self._is_in_circle_of_competence(symbol):
                understanding_score += 1.0
                details.append(f"{symbol}: 在能力圈内，业务模式简单清晰")
            else:
                details.append(f"{symbol}: 超出能力圈，业务复杂或不熟悉")
        
        avg_score = understanding_score / max(len(state.active_symbols), 1)
        
        return ReasoningStep(
            thought="评估业务理解度 - 只投资能够理解的生意",
            action="business_understanding_analysis",
            observation=f"平均理解度评分: {avg_score:.2f}",
            confidence=avg_score,
            metadata={"details": details}
        )
    
    async def _analyze_economic_moat(self, state: TradingState) -> ReasoningStep:
        """分析经济护城河"""
        moat_factors = {
            "brand_power": 0.0,  # 品牌力量
            "network_effects": 0.0,  # 网络效应
            "cost_advantages": 0.0,  # 成本优势
            "switching_costs": 0.0,  # 转换成本
            "intangible_assets": 0.0  # 无形资产
        }
        
        # 基于市场数据评估护城河
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                # 简化的护城河评分逻辑
                if hasattr(data, 'metadata') and 'company_info' in data.metadata:
                    info = data.metadata['company_info']
                    moat_factors["brand_power"] = info.get('brand_score', 0.5)
                    moat_factors["cost_advantages"] = info.get('margin_stability', 0.5)
        
        moat_score = sum(moat_factors.values()) / len(moat_factors)
        
        return ReasoningStep(
            thought="分析经济护城河 - 寻找有持续竞争优势的公司",
            action="moat_analysis",
            observation=f"护城河评分: {moat_score:.2f}",
            confidence=moat_score,
            metadata={"moat_factors": moat_factors}
        )
    
    async def _evaluate_management(self, state: TradingState) -> ReasoningStep:
        """评估管理层质量"""
        management_score = 0.7  # 默认评分
        criteria = {
            "integrity": 0.8,  # 诚信
            "competence": 0.7,  # 能力
            "shareholder_friendly": 0.7,  # 股东友好
            "capital_allocation": 0.6  # 资本配置
        }
        
        avg_score = sum(criteria.values()) / len(criteria)
        
        return ReasoningStep(
            thought="评估管理层 - 寻找诚实有能力的管理团队",
            action="management_evaluation",
            observation=f"管理层评分: {avg_score:.2f}",
            confidence=avg_score,
            metadata={"criteria": criteria}
        )
    
    async def _calculate_intrinsic_value(self, state: TradingState) -> ReasoningStep:
        """计算内在价值"""
        valuations = {}
        
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                current_price = data.close
                
                # 简化的DCF模型
                estimated_fcf = current_price * 0.05  # 假设5%的自由现金流收益率
                growth_rate = 0.03  # 3%永续增长
                discount_rate = 0.10  # 10%折现率
                
                # 简化的内在价值计算
                intrinsic_value = estimated_fcf / (discount_rate - growth_rate)
                
                valuations[symbol] = {
                    "current_price": current_price,
                    "intrinsic_value": intrinsic_value,
                    "discount": (intrinsic_value - current_price) / current_price
                }
        
        avg_discount = sum(v["discount"] for v in valuations.values()) / max(len(valuations), 1)
        
        return ReasoningStep(
            thought="计算内在价值 - 评估真实价值vs市场价格",
            action="intrinsic_value_calculation",
            observation=f"平均折价率: {avg_discount:.2%}",
            confidence=0.6,
            metadata={"valuations": valuations}
        )
    
    async def _determine_margin_of_safety(self, state: TradingState, 
                                         valuation_step: ReasoningStep) -> ReasoningStep:
        """确定安全边际"""
        required_margin = 0.25  # 要求25%的安全边际
        valuations = valuation_step.metadata.get("valuations", {})
        
        safe_investments = []
        for symbol, val in valuations.items():
            if val["discount"] >= required_margin:
                safe_investments.append(symbol)
        
        safety_score = len(safe_investments) / max(len(valuations), 1)
        
        return ReasoningStep(
            thought="确定安全边际 - 只在价格远低于价值时买入",
            action="margin_of_safety_check",
            observation=f"符合安全边际的投资: {len(safe_investments)}/{len(valuations)}",
            confidence=safety_score,
            metadata={"safe_investments": safe_investments, "required_margin": required_margin}
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析做出决策"""
        # 提取各步骤的置信度
        understanding_conf = reasoning_steps[0].confidence
        moat_conf = reasoning_steps[1].confidence
        management_conf = reasoning_steps[2].confidence
        valuation_conf = reasoning_steps[3].confidence
        safety_conf = reasoning_steps[4].confidence
        
        # Buffett风格的决策逻辑：所有条件都要满足
        overall_confidence = min(understanding_conf, moat_conf, management_conf, safety_conf)
        
        # 决定行动
        if overall_confidence >= 0.7 and safety_conf >= 0.5:
            action = "BUY"
            position_size = min(overall_confidence * 0.15, 0.10)  # 最多10%仓位
        elif overall_confidence <= 0.3:
            action = "SELL"
            position_size = 0.0
        else:
            action = "HOLD"
            position_size = 0.0
        
        return {
            "action": action,
            "confidence": overall_confidence,
            "position_size": position_size,
            "risk_assessment": "Conservative - Focus on capital preservation",
            "expected_return": "15-20% annually over long term",
            "time_horizon": "10+ years",
            "key_factors": {
                "understanding": understanding_conf,
                "moat": moat_conf,
                "management": management_conf,
                "valuation": valuation_conf,
                "safety_margin": safety_conf
            },
            "warnings": self._generate_warnings(reasoning_steps)
        }
    
    def _generate_warnings(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        if reasoning_steps[0].confidence < 0.5:
            warnings.append("业务超出能力圈，不建议投资")
        
        if reasoning_steps[1].confidence < 0.6:
            warnings.append("护城河不够宽，竞争优势不明显")
        
        if reasoning_steps[4].confidence < 0.3:
            warnings.append("安全边际不足，当前价格偏高")
        
        return warnings
    
    def _is_in_circle_of_competence(self, symbol: str) -> bool:
        """判断是否在能力圈内"""
        # Buffett偏好的行业
        preferred_sectors = [
            "consumer", "retail", "food", "beverage",
            "insurance", "banking", "railroad", "utility"
        ]
        
        # 简化判断逻辑
        symbol_lower = symbol.lower()
        for sector in preferred_sectors:
            if sector in symbol_lower:
                return True
        
        # 避免的行业
        avoided = ["tech", "bio", "crypto", "defi"]
        for avoid in avoided:
            if avoid in symbol_lower:
                return False
        
        return False  # 默认不在能力圈内
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        Buffett风格：集中投资，长期持有
        """
        evaluation = {
            "overall_quality": 0.0,
            "concentration": 0.0,
            "moat_strength": 0.0,
            "valuation_attractiveness": 0.0,
            "recommendations": []
        }
        
        # 评估集中度（Buffett喜欢集中投资）
        positions = portfolio.get("positions", {})
        if len(positions) <= 10:
            evaluation["concentration"] = 1.0
            evaluation["recommendations"].append("保持集中投资策略")
        else:
            evaluation["concentration"] = 0.5
            evaluation["recommendations"].append("考虑减少持仓数量，集中投资最优秀的公司")
        
        # 评估持仓质量
        total_value = portfolio.get("total_value", 0)
        if total_value > 0:
            # 检查是否持有优质公司
            quality_score = self._assess_holdings_quality(positions)
            evaluation["overall_quality"] = quality_score
            
            if quality_score < 0.7:
                evaluation["recommendations"].append("提升持仓质量，卖出平庸公司")
        
        # 评估估值吸引力
        market_sentiment = market_conditions.get("sentiment", "neutral")
        if market_sentiment == "fearful":
            evaluation["valuation_attractiveness"] = 0.9
            evaluation["recommendations"].append("市场恐慌时是买入良机")
        elif market_sentiment == "greedy":
            evaluation["valuation_attractiveness"] = 0.3
            evaluation["recommendations"].append("市场贪婪时保持谨慎")
        else:
            evaluation["valuation_attractiveness"] = 0.6
        
        # 总体评分
        evaluation["overall_score"] = (
            evaluation["overall_quality"] * 0.4 +
            evaluation["concentration"] * 0.2 +
            evaluation["moat_strength"] * 0.2 +
            evaluation["valuation_attractiveness"] * 0.2
        )
        
        # Buffett式建议
        evaluation["buffett_wisdom"] = self._get_buffett_wisdom(evaluation["overall_score"])
        
        return evaluation
    
    def _assess_holdings_quality(self, positions: Dict[str, Any]) -> float:
        """评估持仓质量"""
        if not positions:
            return 0.0
        
        quality_scores = []
        for symbol, position in positions.items():
            score = 0.5  # 默认评分
            
            # 检查是否是优质公司特征
            if position.get("holding_period_days", 0) > 365:
                score += 0.2  # 长期持有加分
            
            if position.get("unrealized_return", 0) > 0.15:
                score += 0.2  # 盈利能力加分
            
            quality_scores.append(min(score, 1.0))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _get_buffett_wisdom(self, score: float) -> str:
        """获取Buffett风格的智慧建议"""
        if score >= 0.8:
            return "这是一个优秀的投资组合。记住：时间是优秀企业的朋友，平庸企业的敌人。"
        elif score >= 0.6:
            return "组合质量尚可。寻找那些十年后仍会繁荣的企业，而非下个季度的明星。"
        elif score >= 0.4:
            return "需要提升组合质量。买入你愿意持有十年的股票，即使股市明天关闭也不担心。"
        else:
            return "组合需要重大调整。记住：以合理价格买入优秀公司，远胜以便宜价格买入平庸公司。"