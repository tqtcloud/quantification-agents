"""
交易决策图实现
基于 LangGraph 实现多Agent协同决策流程，支持条件分支、循环控制和决策路径记录分析
"""

import time
from typing import Callable, Dict, List, Optional, Any
from functools import wraps

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from src.core.state_management import TradingStateGraph, TradingStateManager
from src.core.models import Signal, MarketData, RiskMetrics
from src.utils.logger import LoggerMixin


class GraphNode:
    """图节点包装器"""
    
    def __init__(self, 
                 name: str, 
                 agent: Any, 
                 method: str = "analyze",
                 condition_func: Optional[Callable] = None,
                 timeout: float = 30.0):
        self.name = name
        self.agent = agent
        self.method = method
        self.condition_func = condition_func
        self.timeout = timeout
    
    def execute(self, state: TradingStateGraph, state_manager: TradingStateManager) -> TradingStateGraph:
        """执行节点"""
        start_time = time.time()
        
        try:
            # 获取agent方法
            agent_method = getattr(self.agent, self.method)
            
            # 执行agent
            if hasattr(agent_method, '__call__'):
                result = agent_method(state)
            else:
                raise AttributeError(f"Method {self.method} not callable on {self.agent}")
            
            execution_time = time.time() - start_time
            
            # 更新状态
            updated_state = state_manager.update_state(
                state=state,
                node_name=self.name,
                agent_name=self.agent.__class__.__name__,
                input_data={"state_keys": list(state.keys())},
                output_data=result if isinstance(result, dict) else {"result": str(result)},
                execution_time=execution_time,
                success=True
            )
            
            return updated_state
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 记录错误
            updated_state = state_manager.update_state(
                state=state,
                node_name=self.name,
                agent_name=self.agent.__class__.__name__,
                input_data={"state_keys": list(state.keys())},
                output_data={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
            
            raise e
    
    def should_execute(self, state: TradingStateGraph) -> bool:
        """检查是否应该执行此节点"""
        if self.condition_func:
            return self.condition_func(state)
        return True


class TradingGraphBuilder(LoggerMixin):
    """交易图构建器"""
    
    def __init__(self, state_manager: TradingStateManager):
        self.state_manager = state_manager
        self.graph = StateGraph(TradingStateGraph)
        self.nodes: Dict[str, GraphNode] = {}
        self.conditions: Dict[str, Callable] = {}
        
    def add_node(self, 
                 name: str, 
                 agent: Any, 
                 method: str = "analyze",
                 condition_func: Optional[Callable] = None,
                 timeout: float = 30.0):
        """添加节点"""
        node = GraphNode(name, agent, method, condition_func, timeout)
        self.nodes[name] = node
        
        # 创建包装函数
        def node_wrapper(state: TradingStateGraph) -> TradingStateGraph:
            return node.execute(state, self.state_manager)
        
        self.graph.add_node(name, node_wrapper)
        self.log_debug(f"Added node: {name}")
    
    def add_edge(self, from_node: str, to_node: str):
        """添加边"""
        self.graph.add_edge(from_node, to_node)
        self.log_debug(f"Added edge: {from_node} -> {to_node}")
    
    def add_conditional_edge(self, 
                           from_node: str, 
                           condition_func: Callable,
                           condition_map: Dict[str, str]):
        """添加条件边"""
        self.graph.add_conditional_edges(from_node, condition_func, condition_map)
        self.log_debug(f"Added conditional edge from {from_node}: {condition_map}")
    
    def set_entry_point(self, node_name: str):
        """设置入口点"""
        self.graph.set_entry_point(node_name)
        self.log_debug(f"Set entry point: {node_name}")
    
    def set_finish_point(self, node_name: str):
        """设置结束点"""
        self.graph.add_edge(node_name, END)
        self.log_debug(f"Set finish point: {node_name}")
    
    def compile(self):
        """编译图"""
        compiled_graph = self.graph.compile()
        self.log_info("Trading graph compiled successfully")
        return compiled_graph


def create_data_collection_node(market_data_collector):
    """数据收集节点"""
    def collect_data(state: TradingStateGraph) -> TradingStateGraph:
        # 这里可以集成实际的数据收集逻辑
        updated_state = state.copy()
        updated_state["decision_context"]["data_collected"] = True
        return updated_state
    
    return collect_data


def create_technical_analysis_node(technical_agent):
    """技术分析节点"""
    def technical_analysis(state: TradingStateGraph) -> TradingStateGraph:
        try:
            # 调用技术分析agent
            if hasattr(technical_agent, 'analyze_state'):
                signals = technical_agent.analyze_state(state)
            else:
                # 后退兼容，使用已有的analyze方法
                from src.core.models import TradingState
                legacy_state = TradingState(
                    market_data=state["market_data"],
                    positions=state["positions"],
                    orders=state["orders"],
                    signals=state["signals"],
                    risk_metrics=state["risk_metrics"]
                )
                signals = technical_agent.analyze(legacy_state)
            
            updated_state = state.copy()
            if signals:
                updated_state["signals"].extend(signals)
                updated_state["agent_outputs"]["technical_analysis"] = {
                    "signals_count": len(signals),
                    "latest_signals": [s.model_dump() if hasattr(s, 'model_dump') else str(s) for s in signals[-3:]]
                }
            
            return updated_state
            
        except Exception as e:
            updated_state = state.copy()
            updated_state["agent_outputs"]["technical_analysis"] = {"error": str(e)}
            return updated_state
    
    return technical_analysis


def create_risk_assessment_node(risk_agent):
    """风险评估节点"""
    def risk_assessment(state: TradingStateGraph) -> TradingStateGraph:
        try:
            # 计算风险指标
            if hasattr(risk_agent, 'assess_risk'):
                risk_metrics = risk_agent.assess_risk(state)
                updated_state = state.copy()
                updated_state["risk_metrics"] = risk_metrics
                updated_state["agent_outputs"]["risk_assessment"] = {
                    "risk_level": "calculated",
                    "metrics_available": True
                }
            else:
                updated_state = state.copy()
                updated_state["agent_outputs"]["risk_assessment"] = {
                    "error": "Risk agent does not support assess_risk method"
                }
            
            return updated_state
            
        except Exception as e:
            updated_state = state.copy()
            updated_state["agent_outputs"]["risk_assessment"] = {"error": str(e)}
            return updated_state
    
    return risk_assessment


def create_decision_aggregation_node(decision_aggregator):
    """决策聚合节点"""
    def decision_aggregation(state: TradingStateGraph) -> TradingStateGraph:
        try:
            # 聚合所有agent的输出
            if hasattr(decision_aggregator, 'aggregate_decisions'):
                final_decision = decision_aggregator.aggregate_decisions(state)
                updated_state = state.copy()
                updated_state["decision_context"]["final_decision"] = final_decision
                updated_state["agent_outputs"]["decision_aggregation"] = {
                    "decision_made": True,
                    "decision_type": final_decision.get("action", "unknown")
                }
            else:
                updated_state = state.copy()
                updated_state["agent_outputs"]["decision_aggregation"] = {
                    "error": "Decision aggregator does not support aggregate_decisions method"
                }
            
            return updated_state
            
        except Exception as e:
            updated_state = state.copy()
            updated_state["agent_outputs"]["decision_aggregation"] = {"error": str(e)}
            return updated_state
    
    return decision_aggregation


def create_execution_node(execution_agent):
    """执行节点"""
    def execution(state: TradingStateGraph) -> TradingStateGraph:
        try:
            final_decision = state["decision_context"].get("final_decision")
            if not final_decision:
                updated_state = state.copy()
                updated_state["agent_outputs"]["execution"] = {"error": "No decision to execute"}
                return updated_state
            
            # 执行决策
            if hasattr(execution_agent, 'execute_decision'):
                execution_result = execution_agent.execute_decision(final_decision, state)
                updated_state = state.copy()
                updated_state["agent_outputs"]["execution"] = {
                    "executed": True,
                    "result": execution_result
                }
            else:
                updated_state = state.copy()
                updated_state["agent_outputs"]["execution"] = {
                    "error": "Execution agent does not support execute_decision method"
                }
            
            return updated_state
            
        except Exception as e:
            updated_state = state.copy()
            updated_state["agent_outputs"]["execution"] = {"error": str(e)}
            return updated_state
    
    return execution


# 条件函数
def has_market_data(state: TradingStateGraph) -> bool:
    """检查是否有市场数据"""
    return len(state["market_data"]) > 0


def has_signals(state: TradingStateGraph) -> bool:
    """检查是否有信号"""
    return len(state["signals"]) > 0


def should_execute_trade(state: TradingStateGraph) -> str:
    """决定是否应该执行交易"""
    final_decision = state["decision_context"].get("final_decision")
    
    if not final_decision:
        return "skip_execution"
    
    action = final_decision.get("action", "").upper()
    confidence = final_decision.get("confidence", 0)
    
    if action in ["BUY", "SELL"] and confidence > 0.6:
        return "execute"
    else:
        return "skip_execution"


def risk_level_check(state: TradingStateGraph) -> str:
    """风险级别检查"""
    risk_metrics = state.get("risk_metrics")
    
    if not risk_metrics:
        return "medium_risk"
    
    # 简化的风险评估逻辑
    if hasattr(risk_metrics, 'current_drawdown'):
        if risk_metrics.current_drawdown > 0.1:  # 10%回撤
            return "high_risk"
        elif risk_metrics.current_drawdown > 0.05:  # 5%回撤
            return "medium_risk"
        else:
            return "low_risk"
    
    return "medium_risk"


def create_trading_graph(agents: Dict[str, Any], 
                        state_manager: TradingStateManager = None) -> StateGraph:
    """
    创建交易决策图
    
    Args:
        agents: Agent字典，包含所需的各种agent
        state_manager: 状态管理器
    
    Returns:
        编译后的交易图
    """
    if not state_manager:
        from src.core.state_management import StateConfig
        state_manager = TradingStateManager(StateConfig())
    
    builder = TradingGraphBuilder(state_manager)
    
    # 添加节点
    if "market_data_collector" in agents:
        builder.add_node("collect_data", agents["market_data_collector"], "collect_market_data")
    
    if "technical_agent" in agents:
        builder.add_node("technical_analysis", agents["technical_agent"], "analyze")
    
    if "risk_agent" in agents:
        builder.add_node("risk_assessment", agents["risk_agent"], "assess_risk")
    
    if "decision_aggregator" in agents:
        builder.add_node("decision_aggregation", agents["decision_aggregator"], "aggregate")
    
    if "execution_agent" in agents:
        builder.add_node("execution", agents["execution_agent"], "execute")
    
    # 添加控制流节点
    builder.add_node("risk_check", lambda state: state, condition_func=lambda state: True)
    builder.add_node("skip_execution", lambda state: state)
    
    # 设置基本流程
    builder.set_entry_point("collect_data")
    
    # 添加基本边
    if "market_data_collector" in agents and "technical_agent" in agents:
        builder.add_edge("collect_data", "technical_analysis")
    
    if "technical_agent" in agents and "risk_agent" in agents:
        builder.add_edge("technical_analysis", "risk_assessment")
    
    if "risk_agent" in agents:
        builder.add_edge("risk_assessment", "risk_check")
    
    # 添加条件边
    risk_condition_map = {
        "low_risk": "decision_aggregation",
        "medium_risk": "decision_aggregation", 
        "high_risk": "skip_execution"
    }
    builder.add_conditional_edge("risk_check", risk_level_check, risk_condition_map)
    
    if "decision_aggregator" in agents:
        execution_condition_map = {
            "execute": "execution",
            "skip_execution": "skip_execution"
        }
        builder.add_conditional_edge("decision_aggregation", should_execute_trade, execution_condition_map)
    
    # 设置结束点
    builder.set_finish_point("execution")
    builder.set_finish_point("skip_execution")
    
    # 编译并返回图
    return builder.compile()


def create_simple_trading_graph(technical_agent, 
                               decision_aggregator = None,
                               state_manager: TradingStateManager = None) -> StateGraph:
    """
    创建简化的交易决策图，用于测试
    
    Args:
        technical_agent: 技术分析agent
        decision_aggregator: 决策聚合器（可选）
        state_manager: 状态管理器
    
    Returns:
        编译后的简化交易图
    """
    if not state_manager:
        from src.core.state_management import StateConfig
        state_manager = TradingStateManager(StateConfig())
    
    graph = StateGraph(TradingStateGraph)
    
    # 添加技术分析节点
    graph.add_node("technical_analysis", create_technical_analysis_node(technical_agent))
    
    # 添加决策节点
    if decision_aggregator:
        graph.add_node("decision_making", create_decision_aggregation_node(decision_aggregator))
        graph.add_edge("technical_analysis", "decision_making")
        graph.add_edge("decision_making", END)
        graph.set_entry_point("technical_analysis")
    else:
        graph.add_edge("technical_analysis", END)
        graph.set_entry_point("technical_analysis")
    
    return graph.compile()


class TradingGraphExecutor(LoggerMixin):
    """交易图执行器"""
    
    def __init__(self, 
                 compiled_graph: StateGraph,
                 state_manager: TradingStateManager):
        self.graph = compiled_graph
        self.state_manager = state_manager
    
    def execute(self, 
                initial_market_data: Dict[str, MarketData] = None,
                session_id: str = None) -> Dict[str, Any]:
        """执行交易图"""
        
        # 创建初始状态
        initial_state = self.state_manager.create_initial_state(
            market_data=initial_market_data,
            session_id=session_id
        )
        
        self.log_info(f"Starting graph execution for session {initial_state['session_id']}")
        
        try:
            # 执行图
            result = self.graph.invoke(initial_state)
            
            # 创建执行结果快照
            self.state_manager.create_snapshot(result, "Graph execution completed")
            
            # 分析执行路径
            path_analysis = self.state_manager.get_path_analysis()
            
            execution_result = {
                "success": True,
                "final_state": result,
                "path_analysis": path_analysis,
                "session_id": result["session_id"],
                "path_id": result["path_id"]
            }
            
            self.log_info(f"Graph execution completed successfully")
            return execution_result
            
        except Exception as e:
            self.log_error(f"Graph execution failed: {e}")
            
            # 创建错误快照
            error_state = initial_state.copy()
            error_state["metadata"]["execution_error"] = str(e)
            self.state_manager.create_snapshot(error_state, f"Graph execution failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "session_id": initial_state["session_id"],
                "path_id": initial_state["path_id"]
            }
    
    def execute_step_by_step(self, 
                           initial_market_data: Dict[str, MarketData] = None,
                           session_id: str = None):
        """分步执行交易图（用于调试）"""
        
        initial_state = self.state_manager.create_initial_state(
            market_data=initial_market_data,
            session_id=session_id
        )
        
        self.log_info(f"Starting step-by-step execution for session {initial_state['session_id']}")
        
        # 使用流式执行
        for step_result in self.graph.stream(initial_state):
            yield step_result
            
            # 为每步创建快照
            if isinstance(step_result, dict) and "current_step" in step_result:
                self.state_manager.create_snapshot(
                    step_result, 
                    f"Step: {step_result['current_step']}"
                )