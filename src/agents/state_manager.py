"""
Agent状态管理器
实现Agent间的状态共享、序列化、反序列化和持久化功能
"""

import asyncio
import json
import pickle
import time
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from src.agents.models import (
    AgentState, AgentMessage, StateTransition, AgentConsensus,
    SharedMemory, AgentPerformanceState, StateCheckpoint,
    DataQualityMetrics, ReasoningStep, FinalDecision,
    PortfolioRecommendation, RiskAssessmentState
)
from src.utils.logger import LoggerMixin


class StateManagerConfig(BaseModel):
    """状态管理器配置"""
    enable_persistence: bool = Field(default=True, description="是否启用持久化")
    storage_path: str = Field(default="data/agent_states", description="存储路径")
    max_state_history: int = Field(default=100, description="最大状态历史记录数")
    state_ttl_seconds: int = Field(default=3600, description="状态生存时间(秒)")
    enable_compression: bool = Field(default=True, description="是否启用压缩")
    checkpoint_interval: int = Field(default=300, description="检查点间隔(秒)")
    enable_encryption: bool = Field(default=False, description="是否启用加密")
    shared_memory_size_mb: int = Field(default=100, description="共享内存大小(MB)")
    consensus_timeout_seconds: int = Field(default=30, description="共识超时时间(秒)")
    enable_state_validation: bool = Field(default=True, description="是否启用状态验证")
    auto_cleanup_interval: int = Field(default=3600, description="自动清理间隔(秒)")


class AgentStateManager(LoggerMixin):
    """Agent状态管理器"""
    
    def __init__(self, config: StateManagerConfig = None):
        """初始化状态管理器"""
        self.config = config or StateManagerConfig()
        self._setup_storage()
        
        # 状态存储
        self._current_states: Dict[str, AgentState] = {}  # session_id -> state
        self._state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_state_history))
        self._state_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # 共享内存
        self._shared_memory: Dict[str, SharedMemory] = {}
        self._memory_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # 消息队列
        self._message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._message_handlers: Dict[str, List[callable]] = defaultdict(list)
        
        # 状态转换记录
        self._transitions: List[StateTransition] = []
        
        # 检查点管理
        self._checkpoints: Dict[str, StateCheckpoint] = {}
        self._last_checkpoint_time = time.time()
        
        # Agent性能追踪
        self._agent_performance: Dict[str, AgentPerformanceState] = {}
        
        # 共识管理
        self._active_consensus: Dict[str, AgentConsensus] = {}
        
        # 数据质量监控
        self._data_quality: Dict[str, DataQualityMetrics] = {}
        
        # 后台任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self.log_info("Agent State Manager initialized")
    
    def _setup_storage(self):
        """设置存储目录"""
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.storage_path / "states").mkdir(exist_ok=True)
        (self.storage_path / "checkpoints").mkdir(exist_ok=True)
        (self.storage_path / "transitions").mkdir(exist_ok=True)
        (self.storage_path / "performance").mkdir(exist_ok=True)
    
    async def initialize(self):
        """初始化管理器"""
        # 启动后台任务
        if self.config.auto_cleanup_interval > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.checkpoint_interval > 0:
            self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        
        self.log_info("State Manager initialized with background tasks")
    
    async def shutdown(self):
        """关闭管理器"""
        self._shutdown_event.set()
        
        # 停止后台任务
        tasks = []
        if self._cleanup_task:
            self._cleanup_task.cancel()
            tasks.append(self._cleanup_task)
        
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            tasks.append(self._checkpoint_task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 保存所有当前状态
        await self.save_all_states()
        
        self.log_info("State Manager shutdown complete")
    
    # ==================== 状态管理 ====================
    
    def create_initial_state(self, session_id: str = None, agent_id: str = None) -> AgentState:
        """创建初始状态"""
        session_id = session_id or str(uuid4())
        agent_id = agent_id or "system"
        
        now = datetime.now()
        initial_state: AgentState = {
            "session_id": session_id,
            "agent_id": agent_id,
            "state_version": 1,
            "market_data": {},
            "news_data": [],
            "social_sentiment": {},
            "analyst_opinions": [],
            "confidence_scores": {},
            "risk_assessment": {
                "risk_level": "medium",
                "var_95": 0.0,
                "var_99": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "exposure_ratio": 0.0,
                "concentration_risk": 0.0,
                "liquidity_risk": 0.0,
                "market_risk": 0.0,
                "operational_risk": 0.0,
                "risk_factors": [],
                "mitigation_strategies": [],
                "timestamp": now
            },
            "portfolio_recommendations": [],
            "final_decision": None,
            "reasoning_chain": [],
            "created_at": now,
            "updated_at": now,
            "metadata": {}
        }
        
        self._current_states[session_id] = initial_state
        self._state_history[session_id].append(deepcopy(initial_state))
        
        self.log_info(f"Created initial state for session {session_id}")
        return initial_state
    
    async def get_state(self, session_id: str) -> Optional[AgentState]:
        """获取当前状态"""
        async with self._state_locks[session_id]:
            return deepcopy(self._current_states.get(session_id))
    
    async def update_state(self, 
                          session_id: str,
                          updates: Dict[str, Any],
                          agent_id: str = None) -> AgentState:
        """更新状态"""
        async with self._state_locks[session_id]:
            if session_id not in self._current_states:
                raise ValueError(f"Session {session_id} not found")
            
            state = self._current_states[session_id]
            old_version = state["state_version"]
            
            # 记录状态转换
            changed_fields = list(updates.keys())
            transition = StateTransition(
                transition_id=str(uuid4()),
                from_state=f"v{old_version}",
                to_state=f"v{old_version + 1}",
                trigger_agent=agent_id or "unknown",
                trigger_event="state_update",
                changed_fields=changed_fields
            )
            self._transitions.append(transition)
            
            # 深度合并更新（对于字典类型的字段）
            for key, value in updates.items():
                if key in state:
                    if isinstance(state[key], dict) and isinstance(value, dict):
                        # 深度合并字典
                        self._deep_merge(state, {key: value})
                    else:
                        state[key] = value
                else:
                    state[key] = value
            
            state["state_version"] += 1
            state["updated_at"] = datetime.now()
            if agent_id:
                state["agent_id"] = agent_id
            
            # 保存历史
            self._state_history[session_id].append(deepcopy(state))
            
            # 验证状态
            if self.config.enable_state_validation:
                self._validate_state(state)
            
            self.log_debug(f"Updated state for session {session_id}, fields: {changed_fields}")
            return state
    
    async def merge_states(self, 
                          session_id: str,
                          partial_state: Dict[str, Any],
                          agent_id: str) -> AgentState:
        """合并部分状态更新"""
        async with self._state_locks[session_id]:
            if session_id not in self._current_states:
                self.create_initial_state(session_id, agent_id)
            
            state = self._current_states[session_id]
            
            # 深度合并状态
            self._deep_merge(state, partial_state)
            
            state["state_version"] += 1
            state["updated_at"] = datetime.now()
            state["agent_id"] = agent_id
            
            # 保存历史
            self._state_history[session_id].append(deepcopy(state))
            
            self.log_debug(f"Merged state for session {session_id} from agent {agent_id}")
            return state
    
    def _deep_merge(self, base: Dict, update: Dict):
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            elif key in base and isinstance(base[key], list) and isinstance(value, list):
                base[key].extend(value)
            else:
                base[key] = value
    
    def _validate_state(self, state: AgentState):
        """验证状态完整性"""
        required_fields = [
            "session_id", "agent_id", "state_version",
            "market_data", "risk_assessment", "reasoning_chain"
        ]
        
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Required field '{field}' missing in state")
        
        # 验证数据质量
        if state.get("market_data"):
            for symbol, data in state["market_data"].items():
                quality = self._assess_data_quality(data)
                self._data_quality[f"market_data_{symbol}"] = quality
    
    def _assess_data_quality(self, data: Dict[str, Any]) -> DataQualityMetrics:
        """评估数据质量"""
        missing_fields = []
        anomalies = []
        
        # 检查必需字段
        required = ["price", "volume", "timestamp"]
        for field in required:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        # 检查异常值
        if "price" in data and data["price"] is not None:
            if data["price"] <= 0:
                anomalies.append("negative_or_zero_price")
        
        completeness = 1.0 - (len(missing_fields) / len(required)) if required else 1.0
        accuracy = 1.0 if not anomalies else 0.8
        
        return DataQualityMetrics(
            data_source="market_data",
            completeness_score=completeness,
            accuracy_score=accuracy,
            timeliness_score=1.0,  # 假设实时数据
            consistency_score=1.0,
            overall_quality=(completeness + accuracy) / 2,
            missing_fields=missing_fields,
            anomalies_detected=anomalies
        )
    
    # ==================== 共享内存管理 ====================
    
    async def set_shared_memory(self,
                                key: str,
                                value: Any,
                                owner_agent: str,
                                ttl_seconds: int = None) -> SharedMemory:
        """设置共享内存"""
        async with self._memory_locks[key]:
            memory = SharedMemory(
                memory_id=str(uuid4()),
                key=key,
                value=value,
                owner_agent=owner_agent,
                ttl_seconds=ttl_seconds or self.config.state_ttl_seconds
            )
            
            self._shared_memory[key] = memory
            self.log_debug(f"Set shared memory: {key} by {owner_agent}")
            return memory
    
    async def get_shared_memory(self, key: str, agent_id: str) -> Optional[Any]:
        """获取共享内存"""
        async with self._memory_locks[key]:
            if key not in self._shared_memory:
                return None
            
            memory = self._shared_memory[key]
            
            # 检查TTL
            if memory.ttl_seconds:
                elapsed = (datetime.now() - memory.created_at).total_seconds()
                if elapsed > memory.ttl_seconds:
                    del self._shared_memory[key]
                    return None
            
            # 检查访问权限
            if memory.access_agents and agent_id not in memory.access_agents:
                if agent_id != memory.owner_agent:
                    self.log_warning(f"Access denied for agent {agent_id} to memory {key}")
                    return None
            
            # 更新访问信息
            memory.access_count += 1
            memory.last_accessed_by = agent_id
            memory.updated_at = datetime.now()
            
            return deepcopy(memory.value)
    
    async def lock_shared_memory(self, key: str, agent_id: str) -> bool:
        """锁定共享内存"""
        async with self._memory_locks[key]:
            if key not in self._shared_memory:
                return False
            
            memory = self._shared_memory[key]
            
            if memory.is_locked and memory.locked_by != agent_id:
                return False
            
            memory.is_locked = True
            memory.locked_by = agent_id
            return True
    
    async def unlock_shared_memory(self, key: str, agent_id: str) -> bool:
        """解锁共享内存"""
        async with self._memory_locks[key]:
            if key not in self._shared_memory:
                return False
            
            memory = self._shared_memory[key]
            
            if memory.locked_by != agent_id:
                return False
            
            memory.is_locked = False
            memory.locked_by = None
            return True
    
    # ==================== 消息传递 ====================
    
    async def send_message(self, message: AgentMessage):
        """发送消息"""
        if message.receiver_agent:
            # 点对点消息
            self._message_queues[message.receiver_agent].append(message)
            
            # 触发处理器
            handlers = self._message_handlers.get(message.receiver_agent, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.log_error(f"Message handler error: {e}")
        else:
            # 广播消息
            for agent_id in self._message_handlers.keys():
                if agent_id != message.sender_agent:
                    self._message_queues[agent_id].append(message)
        
        self.log_debug(f"Message sent from {message.sender_agent} to {message.receiver_agent or 'all'}")
    
    async def receive_messages(self, agent_id: str, max_count: int = 10) -> List[AgentMessage]:
        """接收消息"""
        queue = self._message_queues[agent_id]
        messages = []
        
        for _ in range(min(max_count, len(queue))):
            if queue:
                messages.append(queue.popleft())
        
        return messages
    
    def register_message_handler(self, agent_id: str, handler: callable):
        """注册消息处理器"""
        self._message_handlers[agent_id].append(handler)
    
    # ==================== 共识机制 ====================
    
    async def initiate_consensus(self,
                                 topic: str,
                                 participating_agents: List[str],
                                 threshold: float = 0.7) -> str:
        """发起共识"""
        consensus_id = str(uuid4())
        
        consensus = AgentConsensus(
            consensus_id=consensus_id,
            participating_agents=participating_agents,
            topic=topic,
            votes={},
            confidence_scores={},
            weighted_score=0.0,
            consensus_reached=False,
            consensus_threshold=threshold
        )
        
        self._active_consensus[consensus_id] = consensus
        
        # 通知参与的Agent
        for agent_id in participating_agents:
            message = AgentMessage(
                sender_agent="consensus_manager",
                receiver_agent=agent_id,
                message_type="consensus_request",
                payload={
                    "consensus_id": consensus_id,
                    "topic": topic
                },
                requires_response=True
            )
            await self.send_message(message)
        
        self.log_info(f"Initiated consensus {consensus_id} on topic: {topic}")
        return consensus_id
    
    async def submit_vote(self,
                         consensus_id: str,
                         agent_id: str,
                         vote: str,
                         confidence: float):
        """提交投票"""
        if consensus_id not in self._active_consensus:
            raise ValueError(f"Consensus {consensus_id} not found")
        
        consensus = self._active_consensus[consensus_id]
        
        if agent_id not in consensus.participating_agents:
            raise ValueError(f"Agent {agent_id} not participating in consensus")
        
        consensus.votes[agent_id] = vote
        consensus.confidence_scores[agent_id] = confidence
        
        # 检查是否达成共识
        if len(consensus.votes) == len(consensus.participating_agents):
            await self._evaluate_consensus(consensus_id)
    
    async def _evaluate_consensus(self, consensus_id: str):
        """评估共识"""
        consensus = self._active_consensus[consensus_id]
        
        # 计算加权分数
        vote_weights = defaultdict(float)
        total_weight = 0.0
        
        for agent_id, vote in consensus.votes.items():
            weight = consensus.confidence_scores[agent_id]
            vote_weights[vote] += weight
            total_weight += weight
        
        if total_weight > 0:
            # 找出最高票
            max_vote = max(vote_weights.items(), key=lambda x: x[1])
            consensus.weighted_score = max_vote[1] / total_weight
            
            if consensus.weighted_score >= consensus.consensus_threshold:
                consensus.consensus_reached = True
                consensus.final_decision = max_vote[0]
            else:
                # 记录异议
                for agent_id, vote in consensus.votes.items():
                    if vote != max_vote[0]:
                        consensus.dissenting_opinions[agent_id] = vote
        
        self.log_info(f"Consensus {consensus_id} evaluated: reached={consensus.consensus_reached}")
    
    async def get_consensus_result(self, consensus_id: str) -> Optional[AgentConsensus]:
        """获取共识结果"""
        return self._active_consensus.get(consensus_id)
    
    # ==================== 序列化和持久化 ====================
    
    def serialize_state(self, state: AgentState) -> bytes:
        """序列化状态"""
        # 转换datetime对象为字符串
        serializable_state = self._prepare_for_serialization(state)
        
        if self.config.enable_compression:
            import gzip
            json_str = json.dumps(serializable_state, default=str)
            return gzip.compress(json_str.encode())
        else:
            return json.dumps(serializable_state, default=str).encode()
    
    def deserialize_state(self, data: bytes) -> AgentState:
        """反序列化状态"""
        if self.config.enable_compression:
            import gzip
            json_str = gzip.decompress(data).decode()
        else:
            json_str = data.decode()
        
        state_dict = json.loads(json_str)
        return self._restore_from_serialization(state_dict)
    
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """准备序列化"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        else:
            return obj
    
    def _restore_from_serialization(self, obj: Any) -> Any:
        """从序列化恢复"""
        if isinstance(obj, str):
            # 尝试解析为datetime
            try:
                return datetime.fromisoformat(obj)
            except:
                return obj
        elif isinstance(obj, dict):
            return {k: self._restore_from_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_serialization(item) for item in obj]
        else:
            return obj
    
    async def save_state(self, session_id: str) -> bool:
        """保存状态到磁盘"""
        if not self.config.enable_persistence:
            return False
        
        try:
            state = self._current_states.get(session_id)
            if not state:
                return False
            
            # 序列化状态
            serialized = self.serialize_state(state)
            
            # 保存到文件
            file_path = self.storage_path / "states" / f"{session_id}.state"
            with open(file_path, 'wb') as f:
                f.write(serialized)
            
            self.log_debug(f"Saved state for session {session_id}")
            return True
        
        except Exception as e:
            self.log_error(f"Failed to save state: {e}")
            return False
    
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        """从磁盘加载状态"""
        if not self.config.enable_persistence:
            return None
        
        try:
            file_path = self.storage_path / "states" / f"{session_id}.state"
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                serialized = f.read()
            
            # 反序列化状态
            state = self.deserialize_state(serialized)
            
            # 恢复到内存
            self._current_states[session_id] = state
            
            self.log_debug(f"Loaded state for session {session_id}")
            return state
        
        except Exception as e:
            self.log_error(f"Failed to load state: {e}")
            return None
    
    async def save_all_states(self):
        """保存所有状态"""
        for session_id in self._current_states.keys():
            await self.save_state(session_id)
    
    # ==================== 检查点管理 ====================
    
    async def create_checkpoint(self, 
                               session_id: str,
                               trigger_event: str = "manual",
                               is_recovery_point: bool = False) -> str:
        """创建检查点"""
        checkpoint_id = str(uuid4())
        
        # 收集所有Agent状态
        agent_states = {}
        for agent_id, performance in self._agent_performance.items():
            agent_states[agent_id] = performance.model_dump()
        
        checkpoint = StateCheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            state_data=deepcopy(self._current_states.get(session_id, {})),
            agent_states=agent_states,
            trigger_event=trigger_event,
            is_recovery_point=is_recovery_point
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        
        # 持久化检查点
        if self.config.enable_persistence:
            await self._save_checkpoint(checkpoint)
        
        self.log_info(f"Created checkpoint {checkpoint_id} for session {session_id}")
        return checkpoint_id
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """从检查点恢复"""
        if checkpoint_id not in self._checkpoints:
            # 尝试从磁盘加载
            checkpoint = await self._load_checkpoint(checkpoint_id)
            if not checkpoint:
                return False
            self._checkpoints[checkpoint_id] = checkpoint
        
        checkpoint = self._checkpoints[checkpoint_id]
        
        # 恢复状态
        self._current_states[checkpoint.session_id] = deepcopy(checkpoint.state_data)
        
        # 恢复Agent性能状态
        for agent_id, state_data in checkpoint.agent_states.items():
            self._agent_performance[agent_id] = AgentPerformanceState(**state_data)
        
        self.log_info(f"Restored from checkpoint {checkpoint_id}")
        return True
    
    async def _save_checkpoint(self, checkpoint: StateCheckpoint):
        """保存检查点到磁盘"""
        file_path = self.storage_path / "checkpoints" / f"{checkpoint.checkpoint_id}.pkl"
        
        with open(file_path, 'wb') as f:
            pickle.dump(checkpoint.model_dump(), f)
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """从磁盘加载检查点"""
        file_path = self.storage_path / "checkpoints" / f"{checkpoint_id}.pkl"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return StateCheckpoint(**data)
        except Exception as e:
            self.log_error(f"Failed to load checkpoint: {e}")
            return None
    
    # ==================== 性能追踪 ====================
    
    def update_agent_performance(self,
                                agent_name: str,
                                decision_success: bool,
                                confidence: float,
                                processing_time_ms: float,
                                profit_loss: float = 0.0):
        """更新Agent性能"""
        if agent_name not in self._agent_performance:
            self._agent_performance[agent_name] = AgentPerformanceState(agent_name=agent_name)
        
        performance = self._agent_performance[agent_name]
        
        performance.total_decisions += 1
        if decision_success:
            performance.successful_decisions += 1
        else:
            performance.failed_decisions += 1
        
        # 更新平均值
        n = performance.total_decisions
        performance.average_confidence = (
            (performance.average_confidence * (n - 1) + confidence) / n
        )
        performance.average_processing_time_ms = (
            (performance.average_processing_time_ms * (n - 1) + processing_time_ms) / n
        )
        
        performance.total_profit_loss += profit_loss
        performance.win_rate = performance.successful_decisions / n if n > 0 else 0
        performance.last_decision_time = datetime.now()
    
    def get_agent_performance(self, agent_name: str) -> Optional[AgentPerformanceState]:
        """获取Agent性能"""
        return self._agent_performance.get(agent_name)
    
    def get_all_performances(self) -> Dict[str, AgentPerformanceState]:
        """获取所有Agent性能"""
        return deepcopy(self._agent_performance)
    
    # ==================== 后台任务 ====================
    
    async def _cleanup_loop(self):
        """清理循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.auto_cleanup_interval)
                await self._cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Cleanup loop error: {e}")
    
    async def _checkpoint_loop(self):
        """检查点循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                
                # 为所有活跃会话创建检查点
                for session_id in self._current_states.keys():
                    await self.create_checkpoint(session_id, "auto", False)
                
                self._last_checkpoint_time = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Checkpoint loop error: {e}")
    
    async def _cleanup_expired_data(self):
        """清理过期数据"""
        now = datetime.now()
        
        # 清理过期的共享内存
        expired_keys = []
        for key, memory in self._shared_memory.items():
            if memory.ttl_seconds:
                elapsed = (now - memory.created_at).total_seconds()
                if elapsed > memory.ttl_seconds:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._shared_memory[key]
        
        if expired_keys:
            self.log_info(f"Cleaned up {len(expired_keys)} expired shared memory entries")
        
        # 清理旧的状态转换记录
        if len(self._transitions) > 10000:
            self._transitions = self._transitions[-5000:]
        
        # 清理旧的共识记录
        expired_consensus = []
        for consensus_id, consensus in self._active_consensus.items():
            elapsed = (now - consensus.timestamp).total_seconds()
            if elapsed > 3600:  # 1小时后清理
                expired_consensus.append(consensus_id)
        
        for consensus_id in expired_consensus:
            del self._active_consensus[consensus_id]
    
    # ==================== 统计和分析 ====================
    
    def get_state_statistics(self, session_id: str) -> Dict[str, Any]:
        """获取状态统计"""
        state = self._current_states.get(session_id)
        if not state:
            return {}
        
        history = self._state_history.get(session_id, [])
        
        return {
            "session_id": session_id,
            "state_version": state.get("state_version", 0),
            "history_size": len(history),
            "market_data_symbols": len(state.get("market_data", {})),
            "news_count": len(state.get("news_data", [])),
            "recommendations_count": len(state.get("portfolio_recommendations", [])),
            "reasoning_steps": len(state.get("reasoning_chain", [])),
            "active_agents": len(state.get("confidence_scores", {})),
            "has_final_decision": state.get("final_decision") is not None,
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "transitions_count": len([t for t in self._transitions if t.trigger_agent == state.get("agent_id")])
        }
    
    def get_reasoning_summary(self, session_id: str) -> List[Dict[str, Any]]:
        """获取推理摘要"""
        state = self._current_states.get(session_id)
        if not state:
            return []
        
        reasoning_chain = state.get("reasoning_chain", [])
        
        summary = []
        for step in reasoning_chain:
            summary.append({
                "step_id": step.get("step_id"),
                "agent_name": step.get("agent_name"),
                "action": step.get("action"),
                "confidence": step.get("confidence"),
                "reasoning": step.get("reasoning"),
                "duration_ms": step.get("duration_ms")
            })
        
        return summary