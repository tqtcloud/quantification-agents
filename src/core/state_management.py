"""
LangGraph 状态管理模块
扩展现有状态模型支持复杂决策流程，实现状态序列化、持久化、回溯和调试功能
"""

import json
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
from uuid import uuid4

from langgraph.graph import StateGraph, add_messages
from pydantic import BaseModel, Field

from src.core.models import (
    TradingState, MarketData, OrderBook, Position, Order, Signal, 
    RiskMetrics, StrategyState
)
from src.utils.logger import LoggerMixin


class DecisionNode(BaseModel):
    """决策节点"""
    node_id: str
    node_name: str
    agent_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DecisionPath(BaseModel):
    """决策路径记录"""
    path_id: str
    session_id: str
    nodes: List[DecisionNode] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    final_decision: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    """状态快照"""
    snapshot_id: str
    session_id: str
    state_data: Dict[str, Any]
    path_context: Optional[str] = None
    node_context: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""


class TradingStateGraph(TypedDict):
    """LangGraph 状态图定义"""
    # 市场数据
    market_data: Dict[str, MarketData]
    order_books: Dict[str, OrderBook]
    
    # 交易状态
    positions: Dict[str, Position]
    orders: Dict[str, Order]
    signals: List[Signal]
    risk_metrics: Optional[RiskMetrics]
    strategies: Dict[str, StrategyState]
    
    # 决策流程状态
    current_step: str
    decision_context: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    
    # 路径追踪
    path_id: str
    session_id: str
    step_history: List[str]
    
    # 时间戳
    timestamp: datetime
    
    # 元数据
    metadata: Dict[str, Any]


@dataclass
class StateConfig:
    """状态管理配置"""
    enable_persistence: bool = True
    enable_snapshots: bool = True
    snapshot_frequency: int = 10  # 每10步保存一次快照
    max_history_size: int = 1000
    storage_path: str = "data/state_management"
    auto_backup: bool = True
    backup_interval: int = 3600  # 1小时


class TradingStateManager(LoggerMixin):
    """交易状态管理器"""
    
    def __init__(self, config: StateConfig = None):
        self.config = config or StateConfig()
        self._setup_storage()
        
        # 当前状态
        self._current_state: Optional[TradingStateGraph] = None
        
        # 历史记录
        self._decision_paths: Dict[str, DecisionPath] = {}
        self._snapshots: Dict[str, StateSnapshot] = {}
        
        # 会话管理
        self._current_session_id: Optional[str] = None
        self._current_path_id: Optional[str] = None
        
        # 性能监控
        self._step_count = 0
        self._last_backup_time = time.time()
    
    def _setup_storage(self):
        """设置存储目录"""
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.storage_path / "snapshots").mkdir(exist_ok=True)
        (self.storage_path / "paths").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)
    
    def create_initial_state(self, 
                           market_data: Dict[str, MarketData] = None,
                           session_id: str = None) -> TradingStateGraph:
        """创建初始状态"""
        session_id = session_id or str(uuid4())
        path_id = str(uuid4())
        
        initial_state: TradingStateGraph = {
            # 市场数据
            "market_data": market_data or {},
            "order_books": {},
            
            # 交易状态
            "positions": {},
            "orders": {},
            "signals": [],
            "risk_metrics": None,
            "strategies": {},
            
            # 决策流程状态
            "current_step": "START",
            "decision_context": {},
            "agent_outputs": {},
            
            # 路径追踪
            "path_id": path_id,
            "session_id": session_id,
            "step_history": ["START"],
            
            # 时间戳
            "timestamp": datetime.utcnow(),
            
            # 元数据
            "metadata": {}
        }
        
        self._current_state = initial_state
        self._current_session_id = session_id
        self._current_path_id = path_id
        
        # 创建决策路径记录
        self._decision_paths[path_id] = DecisionPath(
            path_id=path_id,
            session_id=session_id
        )
        
        self.log_info(f"Created initial state for session {session_id}")
        return initial_state
    
    def update_state(self, 
                    state: TradingStateGraph, 
                    node_name: str,
                    agent_name: str,
                    input_data: Dict[str, Any],
                    output_data: Dict[str, Any],
                    execution_time: float,
                    success: bool = True,
                    error_message: str = None) -> TradingStateGraph:
        """更新状态并记录决策节点"""
        
        # 更新基本状态信息
        updated_state = deepcopy(state)
        updated_state["current_step"] = node_name
        updated_state["timestamp"] = datetime.utcnow()
        updated_state["step_history"].append(node_name)
        
        # 记录agent输出
        updated_state["agent_outputs"][agent_name] = output_data
        
        # 创建决策节点记录
        decision_node = DecisionNode(
            node_id=str(uuid4()),
            node_name=node_name,
            agent_name=agent_name,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )
        
        # 添加到决策路径
        path_id = updated_state["path_id"]
        if path_id in self._decision_paths:
            self._decision_paths[path_id].nodes.append(decision_node)
            self._decision_paths[path_id].total_execution_time += execution_time
        
        # 更新当前状态
        self._current_state = updated_state
        self._step_count += 1
        
        # 检查是否需要创建快照
        if (self.config.enable_snapshots and 
            self._step_count % self.config.snapshot_frequency == 0):
            self.create_snapshot(updated_state, f"Auto snapshot at step {self._step_count}")
        
        # 检查是否需要备份
        current_time = time.time()
        if (self.config.auto_backup and 
            current_time - self._last_backup_time > self.config.backup_interval):
            self.backup_state()
            self._last_backup_time = current_time
        
        self.log_debug(f"Updated state: {node_name} by {agent_name}")
        return updated_state
    
    def create_snapshot(self, 
                       state: TradingStateGraph, 
                       description: str = "") -> str:
        """创建状态快照"""
        snapshot_id = str(uuid4())
        
        # 序列化状态数据
        serialized_state = self._serialize_state(state)
        
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            session_id=state["session_id"],
            state_data=serialized_state,
            path_context=state["path_id"],
            node_context=state["current_step"],
            description=description
        )
        
        self._snapshots[snapshot_id] = snapshot
        
        # 持久化快照
        if self.config.enable_persistence:
            self._save_snapshot(snapshot)
        
        self.log_info(f"Created snapshot {snapshot_id}: {description}")
        return snapshot_id
    
    def restore_from_snapshot(self, snapshot_id: str) -> Optional[TradingStateGraph]:
        """从快照恢复状态"""
        if snapshot_id not in self._snapshots:
            # 尝试从磁盘加载
            snapshot = self._load_snapshot(snapshot_id)
            if not snapshot:
                self.log_error(f"Snapshot {snapshot_id} not found")
                return None
            self._snapshots[snapshot_id] = snapshot
        
        snapshot = self._snapshots[snapshot_id]
        
        # 反序列化状态
        state = self._deserialize_state(snapshot.state_data)
        self._current_state = state
        
        self.log_info(f"Restored state from snapshot {snapshot_id}")
        return state
    
    def get_decision_path(self, path_id: str = None) -> Optional[DecisionPath]:
        """获取决策路径"""
        path_id = path_id or self._current_path_id
        return self._decision_paths.get(path_id)
    
    def get_path_analysis(self, path_id: str = None) -> Dict[str, Any]:
        """分析决策路径"""
        path = self.get_decision_path(path_id)
        if not path:
            return {}
        
        analysis = {
            "path_id": path.path_id,
            "session_id": path.session_id,
            "total_nodes": len(path.nodes),
            "total_execution_time": path.total_execution_time,
            "success_rate": sum(1 for node in path.nodes if node.success) / len(path.nodes) if path.nodes else 0,
            "avg_execution_time": path.total_execution_time / len(path.nodes) if path.nodes else 0,
            "node_breakdown": {}
        }
        
        # 按节点类型分析
        for node in path.nodes:
            if node.node_name not in analysis["node_breakdown"]:
                analysis["node_breakdown"][node.node_name] = {
                    "count": 0,
                    "total_time": 0,
                    "success_count": 0,
                    "avg_time": 0
                }
            
            breakdown = analysis["node_breakdown"][node.node_name]
            breakdown["count"] += 1
            breakdown["total_time"] += node.execution_time
            if node.success:
                breakdown["success_count"] += 1
            breakdown["avg_time"] = breakdown["total_time"] / breakdown["count"]
        
        return analysis
    
    def debug_state(self, state: TradingStateGraph = None) -> Dict[str, Any]:
        """调试状态信息"""
        state = state or self._current_state
        if not state:
            return {"error": "No current state"}
        
        debug_info = {
            "session_id": state["session_id"],
            "path_id": state["path_id"],
            "current_step": state["current_step"],
            "step_count": len(state["step_history"]),
            "step_history": state["step_history"][-10:],  # 最近10步
            "active_agents": list(state["agent_outputs"].keys()),
            "market_data_symbols": list(state["market_data"].keys()),
            "positions_count": len(state["positions"]),
            "orders_count": len(state["orders"]),
            "signals_count": len(state["signals"]),
            "timestamp": state["timestamp"].isoformat(),
            "memory_usage": self._get_memory_usage()
        }
        
        return debug_info
    
    def _serialize_state(self, state: TradingStateGraph) -> Dict[str, Any]:
        """序列化状态"""
        serialized = {}
        
        for key, value in state.items():
            if key == "timestamp":
                serialized[key] = value.isoformat()
            elif hasattr(value, 'model_dump'):  # Pydantic models
                serialized[key] = value.model_dump()
            elif hasattr(value, '__dict__'):  # Dataclass objects
                serialized[key] = asdict(value)
            elif isinstance(value, (list, dict)):
                serialized[key] = self._serialize_complex_object(value)
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_complex_object(self, obj: Union[List, Dict]) -> Union[List, Dict]:
        """序列化复杂对象"""
        if isinstance(obj, list):
            return [self._serialize_complex_object(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_complex_object(v) for k, v in obj.items()}
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return asdict(obj)
        else:
            return obj
    
    def _deserialize_state(self, serialized: Dict[str, Any]) -> TradingStateGraph:
        """反序列化状态"""
        # 这里需要根据具体的序列化格式进行反序列化
        # 简化实现，实际应用中需要更复杂的反序列化逻辑
        state = serialized.copy()
        
        # 恢复datetime对象
        if "timestamp" in state and isinstance(state["timestamp"], str):
            state["timestamp"] = datetime.fromisoformat(state["timestamp"])
        
        return state
    
    def _save_snapshot(self, snapshot: StateSnapshot):
        """保存快照到磁盘"""
        file_path = self.storage_path / "snapshots" / f"{snapshot.snapshot_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot.model_dump(), f, indent=2, default=str)
    
    def _load_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """从磁盘加载快照"""
        file_path = self.storage_path / "snapshots" / f"{snapshot_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return StateSnapshot(**data)
        except Exception as e:
            self.log_error(f"Failed to load snapshot {snapshot_id}: {e}")
            return None
    
    def backup_state(self):
        """备份当前状态"""
        if not self._current_state:
            return
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = self.storage_path / "backups" / f"state_backup_{timestamp}.pkl"
        
        backup_data = {
            "current_state": self._current_state,
            "decision_paths": self._decision_paths,
            "snapshots": {k: v.model_dump() for k, v in self._snapshots.items()},
            "session_id": self._current_session_id,
            "path_id": self._current_path_id,
            "step_count": self._step_count
        }
        
        try:
            with open(backup_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            self.log_info(f"State backed up to {backup_file}")
        except Exception as e:
            self.log_error(f"Failed to backup state: {e}")
    
    def restore_from_backup(self, backup_file: str) -> bool:
        """从备份恢复状态"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            backup_path = self.storage_path / "backups" / backup_file
        
        if not backup_path.exists():
            self.log_error(f"Backup file not found: {backup_file}")
            return False
        
        try:
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            self._current_state = backup_data["current_state"]
            self._decision_paths = backup_data["decision_paths"]
            self._snapshots = {k: StateSnapshot(**v) for k, v in backup_data["snapshots"].items()}
            self._current_session_id = backup_data["session_id"]
            self._current_path_id = backup_data["path_id"]
            self._step_count = backup_data["step_count"]
            
            self.log_info(f"State restored from backup: {backup_file}")
            return True
        except Exception as e:
            self.log_error(f"Failed to restore from backup: {e}")
            return False
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        import sys
        
        return {
            "decision_paths": len(self._decision_paths),
            "snapshots": len(self._snapshots),
            "current_state_size": sys.getsizeof(self._current_state) if self._current_state else 0
        }
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """清理旧数据"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        # 清理旧快照
        to_remove = []
        for snapshot_id, snapshot in self._snapshots.items():
            if snapshot.timestamp.timestamp() < cutoff_time:
                to_remove.append(snapshot_id)
        
        for snapshot_id in to_remove:
            del self._snapshots[snapshot_id]
            # 删除磁盘文件
            file_path = self.storage_path / "snapshots" / f"{snapshot_id}.json"
            if file_path.exists():
                file_path.unlink()
        
        self.log_info(f"Cleaned up {len(to_remove)} old snapshots")