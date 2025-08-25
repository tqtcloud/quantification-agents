"""
LangGraph工作流节点定义
定义工作流中的各个处理节点
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import traceback

# from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage  # 未使用
from langgraph.graph import StateGraph, END

from src.agents.models import (
    AgentState, MarketDataState, NewsDataState, 
    SocialSentimentState, AnalystOpinionState,
    RiskAssessmentState, PortfolioRecommendation,
    FinalDecision, ReasoningStep, AgentConsensus
)
from src.agents.state_manager import AgentStateManager
from src.agents.investment_masters import (
    WarrenBuffettAgent, CathieWoodAgent, RayDalioAgent,
    BenjaminGrahamAgent, TechnicalAnalystAgent, QuantitativeAnalystAgent
)
from src.agents.management import RiskManagementAgent, PortfolioManagementAgent
from src.agents.base_agent import InvestmentMasterConfig
from src.agents.enums import InvestmentStyle
from src.core.models import MarketData, TradingState
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeExecutionResult:
    """节点执行结果"""
    success: bool
    node_name: str
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowNodes:
    """工作流节点集合"""
    
    def __init__(self, state_manager: Optional[AgentStateManager] = None):
        """初始化工作流节点"""
        self.state_manager = state_manager or AgentStateManager()
        self.execution_history: List[NodeExecutionResult] = []
        
        # 初始化Agent池（延迟初始化）
        self._analyst_agents: Optional[Dict[str, Any]] = None
        self._risk_agent: Optional[RiskManagementAgent] = None
        self._portfolio_agent: Optional[PortfolioManagementAgent] = None
    
    def _initialize_agents(self) -> None:
        """延迟初始化Agent池"""
        if self._analyst_agents is None:
            self._analyst_agents = self._create_analyst_agents()
        if self._risk_agent is None:
            self._risk_agent = self._create_risk_agent()
        if self._portfolio_agent is None:
            self._portfolio_agent = self._create_portfolio_agent()
    
    def _create_analyst_agents(self) -> Dict[str, Any]:
        """创建分析师Agent池"""
        agents = {}
        
        # 价值投资类
        agents['warren_buffett'] = WarrenBuffettAgent(
            config=InvestmentMasterConfig(
                name="Warren Buffett Agent",  # 添加必需的name字段
                agent_id="warren_buffett",
                agent_name="Warren Buffett",
                master_name="Warren Buffett",
                investment_style=InvestmentStyle.VALUE,
                specialty=["value_investing", "fundamental_analysis"],
                analysis_depth="comprehensive"
            )
        )
        
        agents['benjamin_graham'] = BenjaminGrahamAgent(
            config=InvestmentMasterConfig(
                name="Benjamin Graham Agent",  # 添加必需的name字段
                agent_id="benjamin_graham",
                agent_name="Benjamin Graham",
                master_name="Benjamin Graham",
                investment_style=InvestmentStyle.VALUE,
                specialty=["value_investing", "margin_of_safety"],
                analysis_depth="comprehensive"
            )
        )
        
        # 成长投资类
        agents['cathie_wood'] = CathieWoodAgent(
            config=InvestmentMasterConfig(
                name="Cathie Wood Agent",  # 添加必需的name字段
                agent_id="cathie_wood",
                agent_name="Cathie Wood",
                master_name="Cathie Wood",
                investment_style=InvestmentStyle.GROWTH,
                specialty=["disruptive_innovation", "technology"],
                analysis_depth="comprehensive"
            )
        )
        
        # 宏观策略类
        agents['ray_dalio'] = RayDalioAgent(
            config=InvestmentMasterConfig(
                name="Ray Dalio Agent",  # 添加必需的name字段
                agent_id="ray_dalio",
                agent_name="Ray Dalio",
                master_name="Ray Dalio",
                investment_style=InvestmentStyle.MACRO,
                specialty=["macro_economics", "risk_parity"],
                analysis_depth="comprehensive"
            )
        )
        
        # 技术分析类
        agents['technical_analyst'] = TechnicalAnalystAgent(
            config=InvestmentMasterConfig(
                name="Technical Analyst Agent",  # 添加必需的name字段
                agent_id="technical_analyst",
                agent_name="Technical Analyst",
                master_name="Technical Analyst",
                investment_style=InvestmentStyle.TECHNICAL,
                specialty=["technical_analysis", "chart_patterns"],
                analysis_depth="standard"
            )
        )
        
        # 量化分析类
        agents['quantitative_analyst'] = QuantitativeAnalystAgent(
            config=InvestmentMasterConfig(
                name="Quantitative Analyst Agent",  # 添加必需的name字段
                agent_id="quantitative_analyst",
                agent_name="Quantitative Analyst",
                master_name="Quantitative Analyst",
                investment_style=InvestmentStyle.QUANTITATIVE,
                specialty=["quantitative_analysis", "statistical_arbitrage"],
                analysis_depth="comprehensive"
            )
        )
        
        return agents
    
    def _create_risk_agent(self) -> RiskManagementAgent:
        """创建风险管理Agent"""
        from src.agents.management import RiskManagementAgent, RiskManagementConfig
        
        return RiskManagementAgent(
            config=RiskManagementConfig(
                name="Risk Manager",  # 添加必需的name字段
                agent_id="risk_manager",
                agent_name="Risk Manager",
                var_confidence_level=0.95,
                max_position_size=0.2,
                max_portfolio_risk=0.15,
                stop_loss_threshold=0.05
            )
        )
    
    def _create_portfolio_agent(self) -> PortfolioManagementAgent:
        """创建投资组合管理Agent"""
        from src.agents.management import PortfolioManagementAgent, PortfolioManagementConfig
        
        return PortfolioManagementAgent(
            config=PortfolioManagementConfig(
                name="Portfolio Manager",  # 添加必需的name字段
                agent_id="portfolio_manager",
                agent_name="Portfolio Manager",
                risk_aversion=1.0,
                rebalance_threshold=0.05,
                min_position_size=0.01,
                max_position_size=0.2
            )
        )
    
    async def data_preprocessing_node(self, state: AgentState) -> AgentState:
        """
        数据预处理节点
        准备市场数据、新闻数据和社交情绪数据
        """
        start_time = datetime.now()
        
        try:
            logger.info("开始数据预处理...")
            
            # 验证必要的数据字段
            if not state.get('market_data'):
                state['market_data'] = {}
            if not state.get('news_data'):
                state['news_data'] = []
            if not state.get('social_sentiment'):
                state['social_sentiment'] = {}
            
            # 数据质量检查
            data_quality_score = self._assess_data_quality(state)
            
            # 数据标准化和清洗
            state = self._standardize_data(state)
            
            # 添加推理步骤
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="data_preprocessor",
                action="preprocess_data",
                input_data={"data_sources": list(state.get('market_data', {}).keys())},
                output_data={"quality_score": data_quality_score},
                confidence=data_quality_score,
                reasoning="数据预处理完成，质量评分: {:.2f}".format(data_quality_score),
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            if 'reasoning_chain' not in state:
                state['reasoning_chain'] = []
            state['reasoning_chain'].append(reasoning_step)
            
            # 记录执行结果
            result = NodeExecutionResult(
                success=True,
                node_name="data_preprocessing",
                output_data={"quality_score": data_quality_score},
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            
            logger.info(f"数据预处理完成，质量评分: {data_quality_score:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            result = NodeExecutionResult(
                success=False,
                node_name="data_preprocessing",
                output_data={},
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            raise
    
    async def parallel_analysis_node(self, state: AgentState) -> AgentState:
        """
        并行分析节点
        多个分析师Agent并行执行分析
        """
        start_time = datetime.now()
        
        try:
            logger.info("开始并行分析...")
            self._initialize_agents()
            
            # 准备分析任务
            analysis_tasks = []
            for agent_name, agent in self._analyst_agents.items():
                task = self._run_analyst_agent(agent_name, agent, state)
                analysis_tasks.append(task)
            
            # 并行执行所有分析任务
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # 处理分析结果
            successful_analyses = []
            failed_analyses = []
            analyst_opinions = []
            
            for agent_name, result in zip(self._analyst_agents.keys(), analysis_results):
                if isinstance(result, Exception):
                    failed_analyses.append({
                        'agent': agent_name,
                        'error': str(result)
                    })
                    logger.warning(f"Agent {agent_name} 分析失败: {str(result)}")
                else:
                    successful_analyses.append(result)
                    # 转换为AnalystOpinionState格式
                    opinion = AnalystOpinionState(
                        source=agent_name,
                        analyst_name=result.get('master_name', agent_name),
                        rating=result.get('recommendation', 'hold'),
                        target_price=result.get('target_price', 0.0),
                        confidence=result.get('confidence', 0.5),
                        rationale=result.get('rationale', ''),
                        risk_factors=result.get('risk_factors', []),
                        timestamp=datetime.now()
                    )
                    analyst_opinions.append(opinion)
            
            # 更新状态
            state['analyst_opinions'] = analyst_opinions
            state['confidence_scores'] = {
                opinion['analyst_name']: opinion['confidence'] 
                for opinion in analyst_opinions
            }
            
            # 添加推理步骤
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="parallel_analyzer",
                action="parallel_analysis",
                input_data={"agents": list(self._analyst_agents.keys())},
                output_data={
                    "successful": len(successful_analyses),
                    "failed": len(failed_analyses)
                },
                confidence=len(successful_analyses) / len(self._analyst_agents) if self._analyst_agents else 0,
                reasoning=f"并行分析完成: {len(successful_analyses)}个成功, {len(failed_analyses)}个失败",
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            state['reasoning_chain'].append(reasoning_step)
            
            # 记录执行结果
            result = NodeExecutionResult(
                success=True,
                node_name="parallel_analysis",
                output_data={
                    "successful_analyses": len(successful_analyses),
                    "failed_analyses": len(failed_analyses)
                },
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            
            logger.info(f"并行分析完成: {len(successful_analyses)}个成功")
            return state
            
        except Exception as e:
            logger.error(f"并行分析失败: {str(e)}")
            result = NodeExecutionResult(
                success=False,
                node_name="parallel_analysis",
                output_data={},
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            raise
    
    async def risk_assessment_node(self, state: AgentState) -> AgentState:
        """
        风险评估节点
        执行风险管理分析
        """
        start_time = datetime.now()
        
        try:
            logger.info("开始风险评估...")
            self._initialize_agents()
            
            # 执行风险评估
            risk_assessment = await self._risk_agent.analyze_state(state)
            
            # 更新状态
            state['risk_assessment'] = RiskAssessmentState(
                risk_level=risk_assessment.risk_level.value,
                var_95=risk_assessment.risk_metrics.var_95,
                var_99=risk_assessment.risk_metrics.var_99,
                max_drawdown=risk_assessment.risk_metrics.max_drawdown,
                sharpe_ratio=risk_assessment.risk_metrics.sharpe_ratio,
                exposure_ratio=0.0,  # 需要从实际持仓计算
                concentration_risk=risk_assessment.risk_metrics.concentration_risk,
                liquidity_risk=risk_assessment.risk_metrics.liquidity_risk,
                market_risk=risk_assessment.risk_score / 100.0,
                operational_risk=0.0,
                risk_factors=[
                    {
                        'name': factor.factor_name,
                        'type': factor.factor_type,
                        'impact': factor.impact_score
                    }
                    for factor in risk_assessment.risk_factors
                ],
                mitigation_strategies=risk_assessment.mitigation_strategies,
                timestamp=datetime.now()
            )
            
            # 添加推理步骤
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="risk_manager",
                action="risk_assessment",
                input_data={"analyst_count": len(state.get('analyst_opinions', []))},
                output_data={
                    "risk_level": risk_assessment.risk_level.value,
                    "risk_score": risk_assessment.risk_score
                },
                confidence=0.8,
                reasoning=f"风险评估完成: 风险等级={risk_assessment.risk_level.value}, 风险分数={risk_assessment.risk_score:.2f}",
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            state['reasoning_chain'].append(reasoning_step)
            
            # 记录执行结果
            result = NodeExecutionResult(
                success=True,
                node_name="risk_assessment",
                output_data={
                    "risk_level": risk_assessment.risk_level.value,
                    "risk_score": risk_assessment.risk_score
                },
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            
            logger.info(f"风险评估完成: {risk_assessment.risk_level.value}")
            return state
            
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            result = NodeExecutionResult(
                success=False,
                node_name="risk_assessment",
                output_data={},
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            raise
    
    async def portfolio_optimization_node(self, state: AgentState) -> AgentState:
        """
        投资组合优化节点
        生成最终投资决策
        """
        start_time = datetime.now()
        
        try:
            logger.info("开始投资组合优化...")
            self._initialize_agents()
            
            # 执行投资组合优化
            portfolio_decision = await self._portfolio_agent.optimize_portfolio(state)
            
            # 生成投资建议
            recommendations = []
            for allocation in portfolio_decision.allocations:
                recommendation = PortfolioRecommendation(
                    symbol=allocation.symbol,
                    action=allocation.recommended_action.lower(),
                    position_size=allocation.target_weight,
                    entry_price=0.0,  # 需要从市场数据获取
                    stop_loss=0.0,  # 需要根据风险管理计算
                    take_profit=0.0,  # 需要根据目标收益计算
                    confidence=allocation.confidence,
                    rationale=allocation.reasoning,
                    risk_reward_ratio=2.0,  # 默认风险收益比
                    expected_return=portfolio_decision.metrics.expected_return,
                    holding_period="medium",
                    priority=1
                )
                recommendations.append(recommendation)
            
            state['portfolio_recommendations'] = recommendations
            
            # 生成最终决策
            final_decision = FinalDecision(
                decision_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                action="execute" if portfolio_decision.need_rebalance else "hold",
                recommendations=recommendations,
                total_confidence=portfolio_decision.confidence,
                risk_adjusted_score=portfolio_decision.metrics.sharpe_ratio,
                execution_strategy="phased",  # 分阶段执行
                timing="delayed",  # 延迟执行以等待最佳时机
                conditions=[
                    f"Risk level <= {state['risk_assessment']['risk_level']}",
                    f"Confidence >= {portfolio_decision.confidence}"
                ],
                approved_by=["risk_manager", "portfolio_manager"],
                timestamp=datetime.now()
            )
            state['final_decision'] = final_decision
            
            # 添加推理步骤
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="portfolio_manager",
                action="portfolio_optimization",
                input_data={"recommendations_count": len(recommendations)},
                output_data={
                    "action": final_decision['action'],
                    "confidence": final_decision['total_confidence']
                },
                confidence=final_decision['total_confidence'],
                reasoning=f"投资组合优化完成: 动作={final_decision['action']}, 置信度={final_decision['total_confidence']:.2f}",
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            state['reasoning_chain'].append(reasoning_step)
            
            # 记录执行结果
            result = NodeExecutionResult(
                success=True,
                node_name="portfolio_optimization",
                output_data={
                    "action": final_decision['action'],
                    "recommendations": len(recommendations)
                },
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            
            logger.info(f"投资组合优化完成: {final_decision['action']}")
            return state
            
        except Exception as e:
            logger.error(f"投资组合优化失败: {str(e)}")
            result = NodeExecutionResult(
                success=False,
                node_name="portfolio_optimization",
                output_data={},
                error_message=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            self.execution_history.append(result)
            raise
    
    async def _run_analyst_agent(self, agent_name: str, agent: Any, state: AgentState) -> Dict[str, Any]:
        """运行单个分析师Agent"""
        try:
            # 准备Agent输入
            market_data = state.get('market_data', {})
            
            # 调用Agent分析方法
            # 注意：这里需要根据实际的Agent接口调整
            result = await agent.analyze_market(
                market_data=market_data,
                news_data=state.get('news_data', []),
                sentiment_data=state.get('social_sentiment', {})
            )
            
            return {
                'agent_name': agent_name,
                'master_name': agent.master_name,
                'recommendation': result.get('action', 'hold'),
                'confidence': result.get('confidence', 0.5),
                'target_price': result.get('target_price', 0.0),
                'rationale': result.get('analysis', ''),
                'risk_factors': result.get('risks', [])
            }
            
        except Exception as e:
            logger.error(f"Agent {agent_name} 执行失败: {str(e)}")
            raise
    
    def _assess_data_quality(self, state: AgentState) -> float:
        """评估数据质量"""
        quality_score = 1.0
        
        # 检查市场数据完整性
        market_data = state.get('market_data', {})
        if not market_data:
            quality_score *= 0.5
        else:
            for symbol, data in market_data.items():
                required_fields = ['price', 'volume', 'bid', 'ask']
                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    quality_score *= 0.9
        
        # 检查新闻数据
        news_data = state.get('news_data', [])
        if not news_data:
            quality_score *= 0.9
        
        # 检查社交情绪数据
        social_sentiment = state.get('social_sentiment', {})
        if not social_sentiment:
            quality_score *= 0.95
        
        return max(0.1, min(1.0, quality_score))
    
    def _standardize_data(self, state: AgentState) -> AgentState:
        """标准化数据格式"""
        # 确保所有必要字段存在
        if 'reasoning_chain' not in state:
            state['reasoning_chain'] = []
        
        if 'confidence_scores' not in state:
            state['confidence_scores'] = {}
        
        if 'metadata' not in state:
            state['metadata'] = {}
        
        # 添加时间戳
        if 'created_at' not in state:
            state['created_at'] = datetime.now()
        
        state['updated_at'] = datetime.now()
        
        return state
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        successful_nodes = [r for r in self.execution_history if r.success]
        failed_nodes = [r for r in self.execution_history if not r.success]
        
        total_time = sum(r.execution_time_ms for r in self.execution_history)
        
        return {
            'total_nodes_executed': len(self.execution_history),
            'successful_nodes': len(successful_nodes),
            'failed_nodes': len(failed_nodes),
            'total_execution_time_ms': total_time,
            'average_execution_time_ms': total_time / len(self.execution_history) if self.execution_history else 0,
            'node_details': [
                {
                    'node': r.node_name,
                    'success': r.success,
                    'time_ms': r.execution_time_ms,
                    'error': r.error_message
                }
                for r in self.execution_history
            ]
        }