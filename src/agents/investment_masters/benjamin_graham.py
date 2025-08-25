"""
Benjamin Graham投资大师Agent
价值投资之父，安全边际理念创始人
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class BenjaminGrahamAgent(InvestmentMasterAgent):
    """Benjamin Graham投资大师Agent - 价值投资之父"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化Benjamin Graham Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="benjamin_graham_agent",
                master_name="Benjamin Graham",
                investment_style=InvestmentStyle.VALUE,
                specialty=[
                    "安全边际",
                    "基本面分析",
                    "防御型投资",
                    "净净营运资本",
                    "价值投资原理"
                ],
                llm_model="gpt-4",
                llm_temperature=0.2,  # 极度保守理性
                analysis_depth="comprehensive",
                risk_tolerance="conservative",
                time_horizon="long",
                personality_traits={
                    "analytical": "extreme",
                    "conservative": "extreme",
                    "margin_of_safety": "core_principle",
                    "emotional_discipline": "supreme",
                    "quantitative_focus": "high",
                    "speculation_aversion": "extreme"
                },
                favorite_indicators=[
                    "PE", "PB", "Current_Ratio", "Quick_Ratio",
                    "Debt_to_Equity", "Working_Capital", "Book_Value",
                    "Earnings_Stability", "Dividend_Record"
                ],
                avoid_sectors=["高成长科技", "投机性生物技术", "加密货币"]
            )
        
        super().__init__(config, message_bus)
        
        # Graham的经典投资标准
        self.graham_criteria = {
            "max_pe_ratio": 15,  # 最大市盈率
            "max_pb_ratio": 1.5,  # 最大市净率
            "pe_times_pb": 22.5,  # PE × PB ≤ 22.5
            "min_current_ratio": 2.0,  # 最小流动比率
            "max_debt_to_equity": 1.0,  # 最大负债权益比
            "min_earnings_growth": 0.03,  # 最小年化盈利增长3%
            "min_dividend_years": 5,  # 最少连续分红年数
            "max_price_to_ncav": 0.67  # 价格不超过净流动资产价值的67%
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        Graham风格：严格的定量分析，极度保守的安全边际
        """
        reasoning_steps = []
        
        # 步骤1：定量筛选
        quantitative_step = await self._quantitative_screening(state)
        reasoning_steps.append(quantitative_step)
        
        # 步骤2：安全边际分析
        margin_step = await self._analyze_margin_of_safety(state)
        reasoning_steps.append(margin_step)
        
        # 步骤3：财务稳健性评估
        financial_step = await self._assess_financial_strength(state)
        reasoning_steps.append(financial_step)
        
        # 步骤4：盈利稳定性分析
        earnings_step = await self._analyze_earnings_stability(state)
        reasoning_steps.append(earnings_step)
        
        # 步骤5：防御型vs进取型分类
        classification_step = await self._classify_investment_type(state)
        reasoning_steps.append(classification_step)
        
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
                "investment_style": "Graham Value Investing",
                "investment_type": decision["investment_type"],
                "graham_score": decision["graham_score"],
                "warnings": decision["warnings"]
            }
        )
    
    async def _quantitative_screening(self, state: TradingState) -> ReasoningStep:
        """定量筛选 - Graham的严格数字标准"""
        passed_stocks = []
        failed_stocks = []
        
        for symbol in state.active_symbols:
            score = 0
            max_score = len(self.graham_criteria)
            
            # 模拟评分（实际应从市场数据获取）
            if symbol in state.market_data:
                data = state.market_data[symbol]
                
                # PE比率检查
                pe = getattr(data, 'pe_ratio', 20) if hasattr(data, 'pe_ratio') else 20
                if pe <= self.graham_criteria["max_pe_ratio"]:
                    score += 1
                
                # PB比率检查
                pb = getattr(data, 'pb_ratio', 2) if hasattr(data, 'pb_ratio') else 2
                if pb <= self.graham_criteria["max_pb_ratio"]:
                    score += 1
                
                # Graham公式：PE × PB ≤ 22.5
                if pe * pb <= self.graham_criteria["pe_times_pb"]:
                    score += 2  # 这个标准更重要
            
            if score >= max_score * 0.6:  # 通过60%以上的标准
                passed_stocks.append(symbol)
            else:
                failed_stocks.append(symbol)
        
        pass_rate = len(passed_stocks) / max(len(state.active_symbols), 1)
        
        return ReasoningStep(
            thought="定量筛选 - 应用Graham的严格数字标准",
            action="quantitative_screening",
            observation=f"通过筛选: {len(passed_stocks)}/{len(state.active_symbols)}",
            confidence=pass_rate,
            metadata={
                "passed": passed_stocks,
                "failed": failed_stocks,
                "criteria": self.graham_criteria
            }
        )
    
    async def _analyze_margin_of_safety(self, state: TradingState) -> ReasoningStep:
        """分析安全边际 - Graham的核心理念"""
        margin_analysis = {}
        
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                current_price = data.close
                
                # 计算内在价值（简化的Graham公式）
                # V = EPS × (8.5 + 2g)
                # 其中g是预期增长率
                eps = current_price * 0.05  # 假设5%盈利收益率
                growth = 0.03  # 保守的3%增长
                intrinsic_value = eps * (8.5 + 2 * growth * 100)
                
                # 计算安全边际
                margin = (intrinsic_value - current_price) / intrinsic_value
                
                margin_analysis[symbol] = {
                    "current_price": current_price,
                    "intrinsic_value": intrinsic_value,
                    "margin_of_safety": margin,
                    "acceptable": margin >= 0.33  # Graham要求至少33%的安全边际
                }
        
        avg_margin = sum(m["margin_of_safety"] for m in margin_analysis.values()) / max(len(margin_analysis), 1)
        acceptable_count = sum(1 for m in margin_analysis.values() if m["acceptable"])
        
        return ReasoningStep(
            thought="安全边际分析 - 投资的中心概念",
            action="margin_of_safety_analysis",
            observation=f"平均安全边际: {avg_margin:.1%}, 可接受: {acceptable_count}/{len(margin_analysis)}",
            confidence=min(max(avg_margin, 0), 1),
            metadata={"margin_analysis": margin_analysis}
        )
    
    async def _assess_financial_strength(self, state: TradingState) -> ReasoningStep:
        """评估财务稳健性"""
        financial_scores = {}
        
        for symbol in state.active_symbols:
            scores = {
                "current_ratio": 0.7,  # 流动比率评分
                "debt_ratio": 0.6,  # 负债比率评分
                "working_capital": 0.8,  # 营运资本评分
                "book_value_growth": 0.5,  # 账面价值增长
                "dividend_consistency": 0.7  # 分红一致性
            }
            
            # 计算综合财务评分
            financial_score = sum(scores.values()) / len(scores)
            
            financial_scores[symbol] = {
                "score": financial_score,
                "details": scores,
                "rating": self._get_financial_rating(financial_score)
            }
        
        avg_score = sum(f["score"] for f in financial_scores.values()) / max(len(financial_scores), 1)
        
        return ReasoningStep(
            thought="财务稳健性评估 - 寻找财务强健的公司",
            action="financial_strength_assessment",
            observation=f"平均财务评分: {avg_score:.2f}",
            confidence=avg_score,
            metadata={"financial_scores": financial_scores}
        )
    
    async def _analyze_earnings_stability(self, state: TradingState) -> ReasoningStep:
        """分析盈利稳定性"""
        stability_analysis = {}
        
        for symbol in state.active_symbols:
            # 模拟盈利稳定性分析
            metrics = {
                "earnings_volatility": 0.15,  # 盈利波动率
                "positive_years": 8,  # 过去10年中盈利年数
                "average_growth": 0.05,  # 平均增长率
                "recession_performance": 0.7  # 衰退期表现
            }
            
            # 计算稳定性评分
            stability_score = 0.0
            if metrics["earnings_volatility"] < 0.2:
                stability_score += 0.3
            if metrics["positive_years"] >= 7:
                stability_score += 0.3
            if metrics["average_growth"] > 0.03:
                stability_score += 0.2
            if metrics["recession_performance"] > 0.5:
                stability_score += 0.2
            
            stability_analysis[symbol] = {
                "score": stability_score,
                "metrics": metrics,
                "stable": stability_score >= 0.7
            }
        
        avg_stability = sum(s["score"] for s in stability_analysis.values()) / max(len(stability_analysis), 1)
        
        return ReasoningStep(
            thought="盈利稳定性分析 - 寻找盈利可预测的公司",
            action="earnings_stability_analysis",
            observation=f"平均稳定性评分: {avg_stability:.2f}",
            confidence=avg_stability,
            metadata={"stability_analysis": stability_analysis}
        )
    
    async def _classify_investment_type(self, state: TradingState) -> ReasoningStep:
        """分类投资类型：防御型vs进取型"""
        classification = {}
        
        for symbol in state.active_symbols:
            # Graham的分类标准
            is_defensive = True
            reasons = []
            
            # 检查是否符合防御型投资者标准
            if symbol in state.market_data:
                data = state.market_data[symbol]
                
                # 大公司
                market_cap = getattr(data, 'market_cap', 0) if hasattr(data, 'market_cap') else 1e9
                if market_cap < 2e9:  # 小于20亿
                    is_defensive = False
                    reasons.append("市值过小")
                
                # 稳定分红
                # （简化判断）
                if not self._has_stable_dividends(symbol):
                    is_defensive = False
                    reasons.append("分红不稳定")
            
            classification[symbol] = {
                "type": "defensive" if is_defensive else "enterprising",
                "suitable_for": "conservative investors" if is_defensive else "aggressive investors",
                "reasons": reasons if not is_defensive else ["符合所有防御型标准"]
            }
        
        defensive_count = sum(1 for c in classification.values() if c["type"] == "defensive")
        defensive_ratio = defensive_count / max(len(classification), 1)
        
        return ReasoningStep(
            thought="投资类型分类 - 区分防御型和进取型投资",
            action="investment_type_classification",
            observation=f"防御型投资: {defensive_count}/{len(classification)}",
            confidence=defensive_ratio,
            metadata={"classification": classification}
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析做出决策"""
        # 提取各步骤评分
        quant_score = reasoning_steps[0].confidence
        margin_score = reasoning_steps[1].confidence
        financial_score = reasoning_steps[2].confidence
        stability_score = reasoning_steps[3].confidence
        defensive_ratio = reasoning_steps[4].confidence
        
        # Graham风格：所有标准都要满足
        graham_score = min(quant_score, margin_score, financial_score, stability_score)
        
        # 根据投资者类型调整
        if defensive_ratio > 0.7:
            investment_type = "defensive"
            required_score = 0.6  # 防御型投资者的标准
        else:
            investment_type = "enterprising"
            required_score = 0.5  # 进取型投资者可以接受稍低的标准
        
        # 决定行动
        if graham_score >= required_score and margin_score >= 0.33:
            action = "BUY"
            position_size = min(graham_score * 0.1, 0.05)  # 最多5%仓位，分散投资
        elif graham_score <= 0.3:
            action = "SELL"
            position_size = 0.0
        else:
            action = "HOLD"
            position_size = 0.0
        
        warnings = self._generate_warnings(reasoning_steps)
        
        return {
            "action": action,
            "confidence": graham_score,
            "position_size": position_size,
            "risk_assessment": "极度保守 - 安全边际是投资的中心概念",
            "expected_return": "年化10-15%（包含分红）",
            "time_horizon": "3-5年或更长",
            "investment_type": investment_type,
            "graham_score": graham_score,
            "warnings": warnings
        }
    
    def _generate_warnings(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        margin_data = reasoning_steps[1].metadata.get("margin_analysis", {})
        for symbol, analysis in margin_data.items():
            if not analysis.get("acceptable", False):
                warnings.append(f"{symbol}: 安全边际不足")
        
        if reasoning_steps[2].confidence < 0.6:
            warnings.append("财务稳健性不足")
        
        if reasoning_steps[3].confidence < 0.5:
            warnings.append("盈利稳定性差")
        
        return warnings
    
    def _get_financial_rating(self, score: float) -> str:
        """获取财务评级"""
        if score >= 0.8:
            return "A - 优秀"
        elif score >= 0.6:
            return "B - 良好"
        elif score >= 0.4:
            return "C - 一般"
        else:
            return "D - 较差"
    
    def _has_stable_dividends(self, symbol: str) -> bool:
        """检查是否有稳定分红（简化）"""
        # 实际应该检查历史分红记录
        return "BANK" in symbol.upper() or "UTIL" in symbol.upper()
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        Graham风格：防御型、分散化、注重安全
        """
        evaluation = {
            "safety_score": 0.0,
            "diversification": 0.0,
            "quality_score": 0.0,
            "margin_adequacy": 0.0,
            "recommendations": []
        }
        
        positions = portfolio.get("positions", {})
        
        # 评估安全性
        if len(positions) >= 10:  # Graham建议至少10-30只股票
            evaluation["diversification"] = min(len(positions) / 20, 1.0)
        else:
            evaluation["diversification"] = len(positions) / 10
            evaluation["recommendations"].append("增加持仓数量以分散风险（建议10-30只）")
        
        # 评估质量
        quality_score = 0.7  # 简化评分
        evaluation["quality_score"] = quality_score
        
        # 评估安全边际
        total_value = portfolio.get("total_value", 0)
        cash_position = portfolio.get("cash", 0)
        cash_ratio = cash_position / total_value if total_value > 0 else 0
        
        if cash_ratio > 0.25:  # 保持25%以上现金很保守
            evaluation["safety_score"] = 0.9
        elif cash_ratio > 0.10:
            evaluation["safety_score"] = 0.7
        else:
            evaluation["safety_score"] = 0.5
            evaluation["recommendations"].append("考虑增加现金储备")
        
        # 市场条件评估
        market_sentiment = market_conditions.get("sentiment", "neutral")
        if market_sentiment == "greedy":
            evaluation["recommendations"].append("市场贪婪时要格外谨慎，提高安全边际要求")
            evaluation["margin_adequacy"] = 0.4
        elif market_sentiment == "fearful":
            evaluation["recommendations"].append("市场恐慌可能带来机会，但要坚持价值标准")
            evaluation["margin_adequacy"] = 0.8
        else:
            evaluation["margin_adequacy"] = 0.6
        
        # 总体评分
        evaluation["overall_score"] = (
            evaluation["safety_score"] * 0.3 +
            evaluation["diversification"] * 0.2 +
            evaluation["quality_score"] * 0.3 +
            evaluation["margin_adequacy"] * 0.2
        )
        
        # Graham智慧
        evaluation["graham_wisdom"] = self._get_graham_wisdom(evaluation["overall_score"])
        
        return evaluation
    
    def _get_graham_wisdom(self, score: float) -> str:
        """获取Graham风格的智慧建议"""
        if score >= 0.8:
            return "组合符合防御型投资者标准。记住：投资最重要的是避免严重错误。"
        elif score >= 0.6:
            return "组合基本稳健。聪明的投资者是现实主义者，向乐观主义者卖出，从悲观主义者买入。"
        elif score >= 0.4:
            return "需要提高安全标准。市场先生的情绪不应影响你的投资决策。"
        else:
            return "组合风险较高。永远不要忘记：安全边际是投资的中心概念。"