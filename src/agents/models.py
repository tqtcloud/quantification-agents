"""
Agent状态管理数据模型
定义Agent间数据共享的TypedDict结构和相关模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """置信度级别"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskLevel(str, Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketDataState(TypedDict):
    """市场数据状态"""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    volatility: float
    trend: str  # up, down, sideways
    momentum: float
    timestamp: datetime
    metadata: Dict[str, Any]


class NewsDataState(TypedDict):
    """新闻数据状态"""
    source: str
    title: str
    content: str
    sentiment_score: float  # -1 到 1
    relevance_score: float  # 0 到 1
    impact_level: str  # low, medium, high
    entities: List[str]  # 相关实体(公司、人物等)
    timestamp: datetime
    url: Optional[str]


class SocialSentimentState(TypedDict):
    """社交情绪状态"""
    platform: str  # twitter, reddit, etc.
    symbol: str
    sentiment_score: float  # -1 到 1
    volume: int  # 讨论量
    trending_score: float  # 热度分数
    key_topics: List[str]
    influencer_sentiment: float  # 大V情绪
    retail_sentiment: float  # 散户情绪
    timestamp: datetime


class AnalystOpinionState(TypedDict):
    """分析师观点状态"""
    source: str
    analyst_name: str
    rating: str  # buy, hold, sell
    target_price: float
    confidence: float
    rationale: str
    risk_factors: List[str]
    timestamp: datetime


class RiskAssessmentState(TypedDict):
    """风险评估状态"""
    risk_level: str  # RiskLevel
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    exposure_ratio: float
    concentration_risk: float
    liquidity_risk: float
    market_risk: float
    operational_risk: float
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    timestamp: datetime


class PortfolioRecommendation(TypedDict):
    """投资组合建议"""
    symbol: str
    action: str  # buy, sell, hold
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    rationale: str
    risk_reward_ratio: float
    expected_return: float
    holding_period: str  # short, medium, long
    priority: int  # 1-10


class FinalDecision(TypedDict):
    """最终决策"""
    decision_id: str
    action: str  # execute, reject, modify
    recommendations: List[PortfolioRecommendation]
    total_confidence: float
    risk_adjusted_score: float
    execution_strategy: str
    timing: str  # immediate, delayed, scheduled
    conditions: List[str]  # 执行条件
    approved_by: List[str]  # 批准的Agent列表
    timestamp: datetime


class ReasoningStep(TypedDict):
    """推理步骤"""
    step_id: int
    agent_name: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    reasoning: str
    timestamp: datetime
    duration_ms: float


class AgentState(TypedDict):
    """
    Agent状态管理的核心数据结构
    用于Agent间的数据共享和状态传递
    """
    # 基础信息
    session_id: str
    agent_id: str
    state_version: int
    
    # 市场数据
    market_data: Dict[str, MarketDataState]
    
    # 新闻数据
    news_data: List[NewsDataState]
    
    # 社交情绪
    social_sentiment: Dict[str, SocialSentimentState]
    
    # 分析师观点
    analyst_opinions: List[AnalystOpinionState]
    
    # 置信度分数
    confidence_scores: Dict[str, float]  # agent_name -> confidence
    
    # 风险评估
    risk_assessment: RiskAssessmentState
    
    # 投资组合建议
    portfolio_recommendations: List[PortfolioRecommendation]
    
    # 最终决策
    final_decision: Optional[FinalDecision]
    
    # 推理链
    reasoning_chain: List[ReasoningStep]
    
    # 时间戳
    created_at: datetime
    updated_at: datetime
    
    # 元数据
    metadata: Dict[str, Any]


class AgentMessage(BaseModel):
    """Agent间消息"""
    message_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    sender_agent: str
    receiver_agent: Optional[str] = None  # None表示广播
    message_type: str
    payload: Dict[str, Any]
    priority: int = Field(default=5, ge=1, le=10)
    requires_response: bool = False
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class StateTransition(BaseModel):
    """状态转换记录"""
    transition_id: str
    from_state: Optional[str] = None
    to_state: str
    trigger_agent: str
    trigger_event: str
    changed_fields: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentConsensus(BaseModel):
    """Agent共识"""
    consensus_id: str
    participating_agents: List[str]
    topic: str
    votes: Dict[str, str]  # agent_name -> vote
    confidence_scores: Dict[str, float]  # agent_name -> confidence
    weighted_score: float
    consensus_reached: bool
    consensus_threshold: float = 0.7
    final_decision: Optional[str] = None
    dissenting_opinions: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SharedMemory(BaseModel):
    """共享内存模型"""
    memory_id: str
    key: str
    value: Any
    owner_agent: str
    access_agents: List[str] = Field(default_factory=list)  # 可访问的Agent列表
    ttl_seconds: Optional[int] = None  # 生存时间(秒)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed_by: Optional[str] = None
    is_locked: bool = False
    locked_by: Optional[str] = None


class AgentPerformanceState(BaseModel):
    """Agent性能状态"""
    agent_name: str
    total_decisions: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0
    average_confidence: float = 0.0
    average_processing_time_ms: float = 0.0
    total_profit_loss: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    last_decision_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class StateCheckpoint(BaseModel):
    """状态检查点"""
    checkpoint_id: str
    session_id: str
    state_data: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]  # agent_name -> state
    timestamp: datetime = Field(default_factory=datetime.now)
    trigger_event: str
    is_recovery_point: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataQualityMetrics(BaseModel):
    """数据质量指标"""
    data_source: str
    completeness_score: float  # 0-1 数据完整性
    accuracy_score: float  # 0-1 数据准确性
    timeliness_score: float  # 0-1 数据时效性
    consistency_score: float  # 0-1 数据一致性
    overall_quality: float  # 0-1 总体质量分数
    missing_fields: List[str]
    anomalies_detected: List[str]
    last_updated: datetime = Field(default_factory=datetime.now)
    validation_errors: List[str] = Field(default_factory=list)