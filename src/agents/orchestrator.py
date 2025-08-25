"""
MultiAgentOrchestrator - LangGraph工作流编排系统
实现多Agent协同工作的核心编排器
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import traceback

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver  # 可选功能，需要安装后启用
# from langgraph.prebuilt import ToolExecutor  # 未使用

from src.agents.models import AgentState
from src.agents.workflow_nodes import WorkflowNodes, NodeExecutionResult
from src.agents.result_aggregator import ResultAggregator
from src.agents.state_manager import AgentStateManager
from src.core.models import MarketData, TradingState
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """工作流状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_parallel_agents: int = 6  # 最大并行Agent数
    enable_checkpointing: bool = True  # 启用检查点
    checkpoint_interval: int = 5  # 检查点间隔（节点数）
    timeout_seconds: int = 300  # 工作流超时时间
    retry_failed_nodes: bool = True  # 重试失败节点
    max_retries: int = 3  # 最大重试次数
    aggregation_method: str = "weighted_voting"  # 聚合方法
    consensus_threshold: float = 0.6  # 共识阈值
    enable_monitoring: bool = True  # 启用监控
    log_level: str = "INFO"  # 日志级别


@dataclass 
class WorkflowMetrics:
    """工作流指标"""
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.IDLE
    nodes_executed: int = 0
    nodes_successful: int = 0
    nodes_failed: int = 0
    total_agents_invoked: int = 0
    average_node_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    checkpoints_created: int = 0
    memory_usage_mb: float = 0.0
    
    def calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.nodes_executed == 0:
            return 0.0
        return self.nodes_successful / self.nodes_executed


class MultiAgentOrchestrator:
    """
    多Agent编排器
    使用LangGraph实现复杂的工作流编排
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        初始化编排器
        
        Args:
            config: 工作流配置
        """
        self.config = config or WorkflowConfig()
        
        # 初始化组件
        self.state_manager = AgentStateManager()
        self.workflow_nodes = WorkflowNodes(self.state_manager)
        self.result_aggregator = ResultAggregator(
            aggregation_method=self.config.aggregation_method,
            consensus_threshold=self.config.consensus_threshold
        )
        
        # 初始化工作流图
        self.workflow_graph = self._build_workflow_graph()
        
        # 检查点管理
        self.checkpointer = None
        if self.config.enable_checkpointing:
            # self.checkpointer = SqliteSaver.from_conn_string(":memory:")  # 需要安装后启用
            pass
        
        # 编译工作流
        self.compiled_workflow = self.workflow_graph.compile(
            checkpointer=self.checkpointer
        )
        
        # 工作流状态
        self.current_workflow_id: Optional[str] = None
        self.workflow_metrics: Optional[WorkflowMetrics] = None
        self.workflow_status = WorkflowStatus.IDLE
        
        # 执行历史
        self.execution_history: List[WorkflowMetrics] = []
        
        # 回调函数
        self.node_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("MultiAgentOrchestrator 初始化完成")
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        构建工作流图
        定义节点和边的关系
        """
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点（使用不同的名称避免与状态属性冲突）
        workflow.add_node("preprocess_data", self._wrap_node(self.workflow_nodes.data_preprocessing_node))
        workflow.add_node("analyze_parallel", self._wrap_node(self.workflow_nodes.parallel_analysis_node))
        workflow.add_node("aggregate_results", self._wrap_node(self._result_aggregation_node))
        workflow.add_node("assess_risk", self._wrap_node(self.workflow_nodes.risk_assessment_node))
        workflow.add_node("optimize_portfolio", self._wrap_node(self.workflow_nodes.portfolio_optimization_node))
        workflow.add_node("output_decision", self._wrap_node(self._decision_output_node))
        
        # 设置入口点
        workflow.set_entry_point("preprocess_data")
        
        # 添加边（定义执行顺序）
        workflow.add_edge("preprocess_data", "analyze_parallel")
        workflow.add_edge("analyze_parallel", "aggregate_results")
        workflow.add_edge("aggregate_results", "assess_risk")
        workflow.add_edge("assess_risk", "optimize_portfolio")
        workflow.add_edge("optimize_portfolio", "output_decision")
        workflow.add_edge("output_decision", END)
        
        return workflow
    
    def _wrap_node(self, node_func: Callable) -> Callable:
        """
        包装节点函数，添加错误处理和监控
        
        Args:
            node_func: 原始节点函数
            
        Returns:
            包装后的节点函数
        """
        async def wrapped_node(state: AgentState) -> AgentState:
            node_name = node_func.__name__.replace('_node', '')
            start_time = time.time()
            
            try:
                # 执行前回调
                await self._execute_callbacks(f"before_{node_name}", state)
                
                # 记录节点开始执行
                logger.info(f"执行节点: {node_name}")
                
                # 执行节点
                if asyncio.iscoroutinefunction(node_func):
                    result_state = await node_func(state)
                else:
                    result_state = node_func(state)
                
                # 更新指标
                if self.workflow_metrics:
                    self.workflow_metrics.nodes_executed += 1
                    self.workflow_metrics.nodes_successful += 1
                
                # 执行后回调
                await self._execute_callbacks(f"after_{node_name}", result_state)
                
                # 记录执行时间
                execution_time = (time.time() - start_time) * 1000
                logger.info(f"节点 {node_name} 执行成功，耗时: {execution_time:.2f}ms")
                
                # 检查是否需要创建检查点
                if self.config.enable_checkpointing and self.workflow_metrics:
                    if self.workflow_metrics.nodes_executed % self.config.checkpoint_interval == 0:
                        await self._create_checkpoint(result_state)
                
                return result_state
                
            except Exception as e:
                # 记录错误
                error_msg = f"节点 {node_name} 执行失败: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                # 更新指标
                if self.workflow_metrics:
                    self.workflow_metrics.nodes_executed += 1
                    self.workflow_metrics.nodes_failed += 1
                    self.workflow_metrics.error_messages.append(error_msg)
                
                # 错误回调
                await self._execute_callbacks(f"error_{node_name}", state, error=e)
                
                # 决定是否重试
                if self.config.retry_failed_nodes:
                    for retry in range(self.config.max_retries):
                        logger.info(f"重试节点 {node_name} (尝试 {retry + 1}/{self.config.max_retries})")
                        try:
                            if asyncio.iscoroutinefunction(node_func):
                                result_state = await node_func(state)
                            else:
                                result_state = node_func(state)
                            
                            logger.info(f"节点 {node_name} 重试成功")
                            return result_state
                        except Exception as retry_error:
                            logger.error(f"重试失败: {str(retry_error)}")
                            if retry == self.config.max_retries - 1:
                                raise
                
                raise
        
        return wrapped_node
    
    async def _result_aggregation_node(self, state: AgentState) -> AgentState:
        """
        结果聚合节点
        聚合所有分析师的观点
        """
        start_time = datetime.now()
        
        try:
            logger.info("开始结果聚合...")
            
            # 获取所有分析师观点
            analyst_opinions = state.get('analyst_opinions', [])
            
            if not analyst_opinions:
                logger.warning("没有分析师观点可供聚合")
                return state
            
            # 聚合观点
            aggregated_result = self.result_aggregator.aggregate_opinions(
                opinions=analyst_opinions,
                market_context=state.get('metadata', {}).get('market_context')
            )
            
            # 构建共识
            consensus = self.result_aggregator.build_consensus(
                opinions=analyst_opinions,
                topic="trading_decision"
            )
            
            # 更新状态
            state['metadata'] = state.get('metadata', {})
            state['metadata']['aggregation_result'] = {
                'consensus_action': aggregated_result.consensus_action,
                'consensus_confidence': aggregated_result.consensus_confidence,
                'agreement_level': aggregated_result.agreement_level,
                'key_insights': aggregated_result.key_insights,
                'risk_factors': aggregated_result.risk_factors
            }
            state['metadata']['consensus'] = consensus.dict()
            
            # 添加推理步骤
            from src.agents.models import ReasoningStep
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="result_aggregator",
                action="aggregate_opinions",
                input_data={"opinions_count": len(analyst_opinions)},
                output_data={
                    "consensus_action": aggregated_result.consensus_action,
                    "consensus_confidence": aggregated_result.consensus_confidence,
                    "agreement_level": aggregated_result.agreement_level
                },
                confidence=aggregated_result.consensus_confidence,
                reasoning=f"聚合完成: {aggregated_result.consensus_action} (置信度: {aggregated_result.consensus_confidence:.2f})",
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            if 'reasoning_chain' not in state:
                state['reasoning_chain'] = []
            state['reasoning_chain'].append(reasoning_step)
            
            logger.info(f"结果聚合完成: {aggregated_result.consensus_action}")
            return state
            
        except Exception as e:
            logger.error(f"结果聚合失败: {str(e)}")
            raise
    
    async def _decision_output_node(self, state: AgentState) -> AgentState:
        """
        决策输出节点
        格式化并输出最终决策
        """
        start_time = datetime.now()
        
        try:
            logger.info("生成最终决策输出...")
            
            # 获取最终决策
            final_decision = state.get('final_decision')
            
            if not final_decision:
                logger.warning("没有生成最终决策")
                return state
            
            # 格式化输出
            decision_summary = {
                'decision_id': final_decision['decision_id'],
                'action': final_decision['action'],
                'recommendations': [
                    {
                        'symbol': rec['symbol'],
                        'action': rec['action'],
                        'position_size': rec['position_size'],
                        'confidence': rec['confidence']
                    }
                    for rec in final_decision.get('recommendations', [])
                ],
                'total_confidence': final_decision['total_confidence'],
                'risk_adjusted_score': final_decision['risk_adjusted_score'],
                'execution_strategy': final_decision['execution_strategy'],
                'timestamp': final_decision['timestamp'].isoformat() if isinstance(final_decision['timestamp'], datetime) else final_decision['timestamp']
            }
            
            # 添加到元数据
            state['metadata'] = state.get('metadata', {})
            state['metadata']['decision_summary'] = decision_summary
            
            # 记录推理链总结
            from src.agents.models import ReasoningStep
            reasoning_step = ReasoningStep(
                step_id=len(state.get('reasoning_chain', [])) + 1,
                agent_name="decision_output",
                action="finalize_decision",
                input_data={"decision_id": final_decision['decision_id']},
                output_data=decision_summary,
                confidence=final_decision['total_confidence'],
                reasoning=f"最终决策: {final_decision['action']} - {len(final_decision.get('recommendations', []))}个建议",
                timestamp=datetime.now(),
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            state['reasoning_chain'].append(reasoning_step)
            
            # 打印决策摘要
            logger.info("=" * 50)
            logger.info("最终决策摘要:")
            logger.info(f"决策ID: {decision_summary['decision_id']}")
            logger.info(f"动作: {decision_summary['action']}")
            logger.info(f"置信度: {decision_summary['total_confidence']:.2f}")
            logger.info(f"风险调整分数: {decision_summary['risk_adjusted_score']:.2f}")
            logger.info(f"建议数量: {len(decision_summary['recommendations'])}")
            logger.info("=" * 50)
            
            return state
            
        except Exception as e:
            logger.error(f"决策输出失败: {str(e)}")
            raise
    
    async def execute_workflow(self, 
                              initial_state: Optional[AgentState] = None,
                              workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的工作流
        
        Args:
            initial_state: 初始状态
            workflow_id: 工作流ID（用于恢复）
            
        Returns:
            执行结果
        """
        # 生成工作流ID
        if not workflow_id:
            workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_workflow_id = workflow_id
        
        # 初始化指标
        self.workflow_metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            start_time=datetime.now(),
            status=WorkflowStatus.RUNNING
        )
        
        # 设置初始状态
        if initial_state is None:
            initial_state = self._create_initial_state()
        
        logger.info(f"开始执行工作流: {workflow_id}")
        
        try:
            # 设置超时
            async def run_with_timeout():
                return await self.compiled_workflow.ainvoke(
                    initial_state,
                    {"configurable": {"thread_id": workflow_id}}
                )
            
            # 执行工作流
            if self.config.timeout_seconds > 0:
                final_state = await asyncio.wait_for(
                    run_with_timeout(),
                    timeout=self.config.timeout_seconds
                )
            else:
                final_state = await run_with_timeout()
            
            # 更新指标
            self.workflow_metrics.end_time = datetime.now()
            self.workflow_metrics.status = WorkflowStatus.COMPLETED
            self.workflow_metrics.total_execution_time_ms = (
                self.workflow_metrics.end_time - self.workflow_metrics.start_time
            ).total_seconds() * 1000
            
            # 保存执行历史
            self.execution_history.append(self.workflow_metrics)
            
            logger.info(f"工作流执行成功: {workflow_id}")
            
            # 返回结果
            return {
                'success': True,
                'workflow_id': workflow_id,
                'final_state': final_state,
                'metrics': self._get_metrics_summary(),
                'decision': final_state.get('final_decision'),
                'execution_summary': self.workflow_nodes.get_execution_summary()
            }
            
        except asyncio.TimeoutError:
            logger.error(f"工作流执行超时: {workflow_id}")
            self.workflow_metrics.status = WorkflowStatus.FAILED
            self.workflow_metrics.error_messages.append("Workflow timeout")
            
            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': 'Workflow timeout',
                'metrics': self._get_metrics_summary()
            }
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            
            if self.workflow_metrics:
                self.workflow_metrics.status = WorkflowStatus.FAILED
                self.workflow_metrics.error_messages.append(str(e))
            
            return {
                'success': False,
                'workflow_id': workflow_id,
                'error': str(e),
                'metrics': self._get_metrics_summary()
            }
        
        finally:
            self.workflow_status = WorkflowStatus.IDLE
    
    def _create_initial_state(self) -> AgentState:
        """创建初始状态"""
        return AgentState(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            agent_id="orchestrator",
            state_version=1,
            market_data={},
            news_data=[],
            social_sentiment={},
            analyst_opinions=[],
            confidence_scores={},
            risk_assessment=None,
            portfolio_recommendations=[],
            final_decision=None,
            reasoning_chain=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={}
        )
    
    async def _create_checkpoint(self, state: AgentState) -> None:
        """创建检查点"""
        try:
            if self.checkpointer and self.current_workflow_id:
                # 保存检查点
                checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 这里可以添加实际的检查点保存逻辑
                logger.info(f"创建检查点: {checkpoint_id}")
                
                if self.workflow_metrics:
                    self.workflow_metrics.checkpoints_created += 1
                    
        except Exception as e:
            logger.error(f"创建检查点失败: {str(e)}")
    
    async def _execute_callbacks(self, 
                                event: str, 
                                state: AgentState,
                                error: Optional[Exception] = None) -> None:
        """执行回调函数"""
        if event in self.node_callbacks:
            for callback in self.node_callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(state, error)
                    else:
                        callback(state, error)
                except Exception as e:
                    logger.error(f"回调执行失败: {str(e)}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册回调函数
        
        Args:
            event: 事件名称 (before_*, after_*, error_*)
            callback: 回调函数
        """
        if event not in self.node_callbacks:
            self.node_callbacks[event] = []
        self.node_callbacks[event].append(callback)
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.workflow_metrics:
            return {}
        
        return {
            'workflow_id': self.workflow_metrics.workflow_id,
            'status': self.workflow_metrics.status.value,
            'start_time': self.workflow_metrics.start_time.isoformat(),
            'end_time': self.workflow_metrics.end_time.isoformat() if self.workflow_metrics.end_time else None,
            'nodes_executed': self.workflow_metrics.nodes_executed,
            'nodes_successful': self.workflow_metrics.nodes_successful,
            'nodes_failed': self.workflow_metrics.nodes_failed,
            'success_rate': self.workflow_metrics.calculate_success_rate(),
            'total_execution_time_ms': self.workflow_metrics.total_execution_time_ms,
            'checkpoints_created': self.workflow_metrics.checkpoints_created,
            'error_messages': self.workflow_metrics.error_messages
        }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """获取工作流状态"""
        return {
            'current_workflow_id': self.current_workflow_id,
            'status': self.workflow_status.value,
            'metrics': self._get_metrics_summary() if self.workflow_metrics else None,
            'execution_history_count': len(self.execution_history)
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Args:
            limit: 返回的历史记录数量
            
        Returns:
            执行历史列表
        """
        history = []
        for metrics in self.execution_history[-limit:]:
            history.append({
                'workflow_id': metrics.workflow_id,
                'status': metrics.status.value,
                'start_time': metrics.start_time.isoformat(),
                'end_time': metrics.end_time.isoformat() if metrics.end_time else None,
                'success_rate': metrics.calculate_success_rate(),
                'total_execution_time_ms': metrics.total_execution_time_ms
            })
        return history
    
    async def cancel_workflow(self) -> bool:
        """
        取消当前运行的工作流
        
        Returns:
            是否成功取消
        """
        if self.workflow_status == WorkflowStatus.RUNNING:
            logger.info(f"取消工作流: {self.current_workflow_id}")
            self.workflow_status = WorkflowStatus.CANCELLED
            
            if self.workflow_metrics:
                self.workflow_metrics.status = WorkflowStatus.CANCELLED
                self.workflow_metrics.end_time = datetime.now()
            
            return True
        
        return False
    
    def update_config(self, config: WorkflowConfig) -> None:
        """
        更新工作流配置
        
        Args:
            config: 新的配置
        """
        self.config = config
        
        # 更新聚合器配置
        self.result_aggregator = ResultAggregator(
            aggregation_method=config.aggregation_method,
            consensus_threshold=config.consensus_threshold
        )
        
        logger.info("工作流配置已更新")