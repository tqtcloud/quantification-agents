"""
Cathie Wood投资大师Agent
实现颠覆性创新投资理念，专注于技术革新和指数级增长
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class CathieWoodAgent(InvestmentMasterAgent):
    """Cathie Wood投资大师Agent - ARK Invest创始人"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化Cathie Wood Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="cathie_wood_agent",
                master_name="Cathie Wood",
                investment_style=InvestmentStyle.GROWTH,
                specialty=[
                    "颠覆性创新",
                    "人工智能",
                    "基因组学",
                    "自动驾驶",
                    "区块链技术",
                    "太空探索"
                ],
                llm_model="gpt-4",
                llm_temperature=0.7,  # 更开放的思维
                analysis_depth="comprehensive",
                risk_tolerance="aggressive",  # 高风险偏好
                time_horizon="long",  # 5-10年视角
                personality_traits={
                    "innovation_focus": "extreme",
                    "risk_appetite": "high",
                    "conviction": "very_high",
                    "volatility_tolerance": "extreme",
                    "research_depth": "comprehensive",
                    "optimism": "high"
                },
                favorite_indicators=[
                    "Revenue_Growth", "TAM_Expansion",  # 总可寻址市场扩张
                    "R&D_Spending", "Patent_Count",
                    "User_Growth", "Market_Share_Growth",
                    "Innovation_Score", "Disruption_Potential"
                ],
                avoid_sectors=["传统能源", "传统零售", "传统银行"]
            )
        
        super().__init__(config, message_bus)
        
        # ARK投资的五大创新平台
        self.innovation_platforms = {
            "artificial_intelligence": {
                "weight": 0.25,
                "keywords": ["AI", "machine learning", "deep learning", "neural"],
                "min_growth_rate": 0.30
            },
            "robotics": {
                "weight": 0.20,
                "keywords": ["robotics", "automation", "autonomous", "drone"],
                "min_growth_rate": 0.25
            },
            "genomics": {
                "weight": 0.20,
                "keywords": ["genomics", "CRISPR", "biotech", "gene therapy"],
                "min_growth_rate": 0.35
            },
            "blockchain": {
                "weight": 0.20,
                "keywords": ["blockchain", "crypto", "DeFi", "Web3"],
                "min_growth_rate": 0.40
            },
            "space": {
                "weight": 0.15,
                "keywords": ["space", "satellite", "aerospace", "orbital"],
                "min_growth_rate": 0.30
            }
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        Cathie Wood风格：寻找颠覆性创新，接受高波动性
        """
        reasoning_steps = []
        
        # 步骤1：评估创新潜力
        innovation_step = await self._evaluate_innovation_potential(state)
        reasoning_steps.append(innovation_step)
        
        # 步骤2：分析指数级增长潜力
        growth_step = await self._analyze_exponential_growth(state)
        reasoning_steps.append(growth_step)
        
        # 步骤3：评估技术采用曲线
        adoption_step = await self._evaluate_technology_adoption(state)
        reasoning_steps.append(adoption_step)
        
        # 步骤4：分析竞争格局
        competition_step = await self._analyze_competitive_landscape(state)
        reasoning_steps.append(competition_step)
        
        # 步骤5：预测未来场景
        future_step = await self._project_future_scenarios(state)
        reasoning_steps.append(future_step)
        
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
                "investment_style": "ARK Disruptive Innovation",
                "innovation_platforms": decision["platforms"],
                "key_thesis": decision["thesis"]
            }
        )
    
    async def _evaluate_innovation_potential(self, state: TradingState) -> ReasoningStep:
        """评估创新潜力"""
        innovation_scores = {}
        
        for symbol in state.active_symbols:
            score = 0.0
            platforms_matched = []
            
            # 检查是否匹配创新平台
            symbol_lower = symbol.lower()
            for platform, info in self.innovation_platforms.items():
                if any(keyword in symbol_lower for keyword in info["keywords"]):
                    score += info["weight"]
                    platforms_matched.append(platform)
            
            # 额外的创新指标
            if symbol in state.market_data:
                data = state.market_data[symbol]
                # 高成长性加分
                if hasattr(data, 'metadata'):
                    growth_rate = data.metadata.get('revenue_growth', 0)
                    if growth_rate > 0.50:  # 50%以上增长
                        score += 0.3
            
            innovation_scores[symbol] = {
                "score": min(score, 1.0),
                "platforms": platforms_matched
            }
        
        avg_score = sum(s["score"] for s in innovation_scores.values()) / max(len(innovation_scores), 1)
        
        return ReasoningStep(
            thought="评估创新潜力 - 寻找引领未来的颠覆性技术",
            action="innovation_assessment",
            observation=f"创新潜力评分: {avg_score:.2f}",
            confidence=avg_score,
            metadata={"innovation_scores": innovation_scores}
        )
    
    async def _analyze_exponential_growth(self, state: TradingState) -> ReasoningStep:
        """分析指数级增长潜力"""
        growth_analysis = {}
        
        for symbol in state.active_symbols:
            metrics = {
                "current_growth": 0.0,
                "growth_acceleration": 0.0,
                "tam_expansion": 0.0,  # 总可寻址市场扩张
                "network_effects": 0.0
            }
            
            if symbol in state.market_data:
                data = state.market_data[symbol]
                # 简化的增长分析
                price_change = (data.close - data.open) / data.open if data.open else 0
                metrics["current_growth"] = abs(price_change) * 10  # 放大短期变化
                
                # Wright's Law - 成本下降曲线
                metrics["growth_acceleration"] = 0.6  # 假设值
                metrics["tam_expansion"] = 0.7
                metrics["network_effects"] = 0.5
            
            growth_analysis[symbol] = {
                "score": sum(metrics.values()) / len(metrics),
                "metrics": metrics
            }
        
        avg_growth_score = sum(g["score"] for g in growth_analysis.values()) / max(len(growth_analysis), 1)
        
        return ReasoningStep(
            thought="分析指数级增长 - 识别具有S曲线增长潜力的机会",
            action="exponential_growth_analysis",
            observation=f"指数增长潜力: {avg_growth_score:.2f}",
            confidence=avg_growth_score * 0.8,
            metadata={"growth_analysis": growth_analysis}
        )
    
    async def _evaluate_technology_adoption(self, state: TradingState) -> ReasoningStep:
        """评估技术采用曲线位置"""
        adoption_stages = {
            "innovators": 0.025,  # 2.5%
            "early_adopters": 0.135,  # 13.5%
            "early_majority": 0.34,  # 34%
            "late_majority": 0.34,  # 34%
            "laggards": 0.16  # 16%
        }
        
        adoption_analysis = {}
        for symbol in state.active_symbols:
            # ARK偏好早期阶段
            stage = self._identify_adoption_stage(symbol, state)
            if stage in ["innovators", "early_adopters"]:
                score = 0.9
            elif stage == "early_majority":
                score = 0.6
            else:
                score = 0.3
            
            adoption_analysis[symbol] = {
                "stage": stage,
                "score": score,
                "potential_multiplier": self._calculate_potential_multiplier(stage)
            }
        
        avg_score = sum(a["score"] for a in adoption_analysis.values()) / max(len(adoption_analysis), 1)
        
        return ReasoningStep(
            thought="评估技术采用曲线 - 寻找处于早期爆发阶段的技术",
            action="adoption_curve_analysis",
            observation=f"采用曲线位置评分: {avg_score:.2f}",
            confidence=avg_score * 0.7,
            metadata={"adoption_analysis": adoption_analysis}
        )
    
    async def _analyze_competitive_landscape(self, state: TradingState) -> ReasoningStep:
        """分析竞争格局"""
        competitive_analysis = {}
        
        for symbol in state.active_symbols:
            factors = {
                "first_mover": 0.7,  # 先发优势
                "technology_moat": 0.6,  # 技术护城河
                "ecosystem_strength": 0.5,  # 生态系统
                "talent_concentration": 0.6  # 人才集中度
            }
            
            competitive_analysis[symbol] = {
                "score": sum(factors.values()) / len(factors),
                "factors": factors,
                "competitive_advantage": "Strong" if sum(factors.values()) / len(factors) > 0.6 else "Moderate"
            }
        
        avg_score = sum(c["score"] for c in competitive_analysis.values()) / max(len(competitive_analysis), 1)
        
        return ReasoningStep(
            thought="分析竞争格局 - 识别具有持续创新优势的领导者",
            action="competitive_analysis",
            observation=f"竞争优势评分: {avg_score:.2f}",
            confidence=avg_score * 0.8,
            metadata={"competitive_analysis": competitive_analysis}
        )
    
    async def _project_future_scenarios(self, state: TradingState) -> ReasoningStep:
        """预测未来场景"""
        scenarios = {
            "base_case": {
                "probability": 0.5,
                "return_multiple": 3,  # 3倍回报
                "timeframe_years": 5
            },
            "bull_case": {
                "probability": 0.3,
                "return_multiple": 10,  # 10倍回报
                "timeframe_years": 5
            },
            "bear_case": {
                "probability": 0.2,
                "return_multiple": 0.5,  # 损失50%
                "timeframe_years": 2
            }
        }
        
        # 计算期望回报
        expected_return = sum(
            s["probability"] * s["return_multiple"] 
            for s in scenarios.values()
        )
        
        return ReasoningStep(
            thought="预测未来场景 - 构建概率加权的回报预期",
            action="future_scenario_projection",
            observation=f"期望回报倍数: {expected_return:.1f}x",
            confidence=0.6,  # 未来预测的不确定性
            metadata={
                "scenarios": scenarios,
                "expected_return_multiple": expected_return
            }
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有分析做出决策"""
        # 提取各步骤的评分
        innovation_score = reasoning_steps[0].confidence
        growth_score = reasoning_steps[1].confidence
        adoption_score = reasoning_steps[2].confidence
        competitive_score = reasoning_steps[3].confidence
        future_return = reasoning_steps[4].metadata.get("expected_return_multiple", 1)
        
        # ARK风格：高信念度，愿意承担波动
        overall_score = (
            innovation_score * 0.3 +
            growth_score * 0.25 +
            adoption_score * 0.2 +
            competitive_score * 0.15 +
            min(future_return / 10, 1.0) * 0.1  # 未来回报影响
        )
        
        # 决策逻辑
        if overall_score >= 0.6 and innovation_score >= 0.5:
            action = "BUY"
            # ARK风格：高信念度时大仓位
            position_size = min(overall_score * 0.2, 0.15)  # 最多15%仓位
        elif overall_score <= 0.3:
            action = "SELL"
            position_size = 0.0
        else:
            action = "HOLD"
            position_size = 0.05  # 小仓位观察
        
        # 识别主要创新平台
        innovation_data = reasoning_steps[0].metadata.get("innovation_scores", {})
        main_platforms = []
        for symbol, data in innovation_data.items():
            main_platforms.extend(data.get("platforms", []))
        main_platforms = list(set(main_platforms))
        
        return {
            "action": action,
            "confidence": overall_score,
            "position_size": position_size,
            "risk_assessment": "High Risk/High Reward - Embrace volatility for long-term gains",
            "expected_return": f"{future_return:.1f}x over 5 years",
            "time_horizon": "5-10 years",
            "platforms": main_platforms,
            "thesis": self._generate_investment_thesis(reasoning_steps)
        }
    
    def _generate_investment_thesis(self, reasoning_steps: List[ReasoningStep]) -> str:
        """生成投资论述"""
        innovation_score = reasoning_steps[0].confidence
        growth_score = reasoning_steps[1].confidence
        
        if innovation_score >= 0.7 and growth_score >= 0.6:
            return "强烈的颠覆性创新机会，技术正处于指数增长的拐点"
        elif innovation_score >= 0.5:
            return "具有创新潜力，需要密切关注技术发展和市场采用"
        else:
            return "创新特征不明显，可能不符合ARK的投资标准"
    
    def _identify_adoption_stage(self, symbol: str, state: TradingState) -> str:
        """识别技术采用阶段"""
        # 简化逻辑：基于符号特征判断
        if "AI" in symbol.upper() or "TSLA" in symbol.upper():
            return "early_majority"
        elif "CRYPTO" in symbol.upper() or "GENE" in symbol.upper():
            return "early_adopters"
        elif "SPACE" in symbol.upper():
            return "innovators"
        else:
            return "early_majority"
    
    def _calculate_potential_multiplier(self, stage: str) -> float:
        """计算潜在回报倍数"""
        multipliers = {
            "innovators": 20.0,
            "early_adopters": 10.0,
            "early_majority": 5.0,
            "late_majority": 2.0,
            "laggards": 1.2
        }
        return multipliers.get(stage, 1.0)
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        ARK风格：集中于高信念度的创新主题
        """
        evaluation = {
            "innovation_exposure": 0.0,
            "growth_potential": 0.0,
            "volatility_assessment": 0.0,
            "theme_concentration": 0.0,
            "recommendations": []
        }
        
        positions = portfolio.get("positions", {})
        
        # 评估创新暴露度
        innovation_count = 0
        for symbol in positions.keys():
            if self._is_innovative_company(symbol):
                innovation_count += 1
        
        if len(positions) > 0:
            evaluation["innovation_exposure"] = innovation_count / len(positions)
        
        # 评估增长潜力
        avg_growth = 0.0
        for symbol, position in positions.items():
            # 假设的增长评分
            growth_score = position.get("expected_growth", 0.3)
            avg_growth += growth_score
        
        if len(positions) > 0:
            evaluation["growth_potential"] = avg_growth / len(positions)
        
        # 波动性评估（ARK接受高波动）
        volatility = market_conditions.get("volatility", "medium")
        if volatility == "high":
            evaluation["volatility_assessment"] = 0.8  # 高波动是机会
            evaluation["recommendations"].append("利用市场波动增加高信念度仓位")
        else:
            evaluation["volatility_assessment"] = 0.5
        
        # 主题集中度
        if len(positions) <= 15:  # ARK偏好集中投资
            evaluation["theme_concentration"] = 0.9
        else:
            evaluation["theme_concentration"] = 0.5
            evaluation["recommendations"].append("考虑集中于最高信念度的创新主题")
        
        # 总体评分
        evaluation["overall_score"] = (
            evaluation["innovation_exposure"] * 0.35 +
            evaluation["growth_potential"] * 0.30 +
            evaluation["volatility_assessment"] * 0.15 +
            evaluation["theme_concentration"] * 0.20
        )
        
        # ARK风格建议
        evaluation["ark_perspective"] = self._get_ark_perspective(evaluation["overall_score"])
        
        # 创新平台分配建议
        evaluation["platform_allocation"] = self._suggest_platform_allocation()
        
        return evaluation
    
    def _is_innovative_company(self, symbol: str) -> bool:
        """判断是否为创新公司"""
        innovative_keywords = [
            "AI", "ML", "ROBOT", "GENE", "CRISPR", "CRYPTO", "BLOCKCHAIN",
            "SPACE", "SATELLITE", "EV", "AUTONOMOUS", "BIOTECH", "QUANTUM"
        ]
        
        symbol_upper = symbol.upper()
        return any(keyword in symbol_upper for keyword in innovative_keywords)
    
    def _get_ark_perspective(self, score: float) -> str:
        """获取ARK风格的观点"""
        if score >= 0.8:
            return "组合充分暴露于颠覆性创新。继续寻找下一个特斯拉或比特币级别的机会。"
        elif score >= 0.6:
            return "创新暴露度良好。考虑增加对人工智能和基因组学的配置。"
        elif score >= 0.4:
            return "需要更多创新资产。传统价值股在技术颠覆面前可能成为价值陷阱。"
        else:
            return "组合过于传统。需要大幅增加对颠覆性创新平台的投资。"
    
    def _suggest_platform_allocation(self) -> Dict[str, float]:
        """建议创新平台配置"""
        return {
            "人工智能": 0.25,
            "机器人技术": 0.20,
            "基因组学": 0.20,
            "区块链": 0.20,
            "太空探索": 0.15
        }