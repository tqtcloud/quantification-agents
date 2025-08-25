"""
API响应数据模型
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

T = TypeVar('T')


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


class APIResponse(BaseModel, Generic[T]):
    """通用API响应模型"""
    status: ResponseStatus = Field(..., description="响应状态")
    message: str = Field(..., description="响应消息")
    data: Optional[T] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="响应时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")
    execution_time_ms: Optional[float] = Field(None, description="执行时间(毫秒)")
    warnings: List[str] = Field(default=[], description="警告信息")
    
    class Config:
        arbitrary_types_allowed = True


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    items: List[T] = Field(..., description="数据项列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    total_pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")
    
    @classmethod
    def create(cls, items: List[T], total: int, page: int, page_size: int) -> 'PaginatedResponse[T]':
        """创建分页响应"""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


# ===== 策略相关响应模型 =====

class StrategyStatus(str, Enum):
    """策略状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class StrategyMetrics(BaseModel):
    """策略指标模型"""
    total_signals: int = Field(default=0, description="总信号数")
    successful_trades: int = Field(default=0, description="成功交易数")
    failed_trades: int = Field(default=0, description="失败交易数")
    total_profit_loss: float = Field(default=0.0, description="总盈亏")
    win_rate: float = Field(default=0.0, description="胜率")
    average_return: float = Field(default=0.0, description="平均收益率")
    sharpe_ratio: Optional[float] = Field(None, description="夏普比率")
    max_drawdown: Optional[float] = Field(None, description="最大回撤")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")


class StrategyInstance(BaseModel):
    """策略实例模型"""
    strategy_id: str = Field(..., description="策略ID")
    name: str = Field(..., description="策略名称")
    strategy_type: str = Field(..., description="策略类型")
    status: StrategyStatus = Field(..., description="策略状态")
    description: Optional[str] = Field(None, description="策略描述")
    config: Optional[Dict[str, Any]] = Field(None, description="策略配置")
    metrics: Optional[StrategyMetrics] = Field(None, description="策略指标")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    started_at: Optional[datetime] = Field(None, description="启动时间")
    stopped_at: Optional[datetime] = Field(None, description="停止时间")
    error_message: Optional[str] = Field(None, description="错误信息")
    health_status: str = Field(default="unknown", description="健康状态")
    resource_usage: Optional[Dict[str, Any]] = Field(None, description="资源使用情况")


class StrategyStatusResponse(BaseModel):
    """策略状态响应模型"""
    strategies: List[StrategyInstance] = Field(..., description="策略实例列表")
    system_status: str = Field(..., description="系统状态")
    total_strategies: int = Field(..., description="总策略数")
    active_strategies: int = Field(..., description="活跃策略数")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")


class StrategyActionResponse(BaseModel):
    """策略操作响应模型"""
    strategy_id: str = Field(..., description="策略ID")
    action: str = Field(..., description="执行的操作")
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="操作消息")
    new_status: Optional[StrategyStatus] = Field(None, description="新状态")
    execution_time_ms: Optional[float] = Field(None, description="执行时间")
    warnings: List[str] = Field(default=[], description="警告信息")


class BatchActionResponse(BaseModel):
    """批量操作响应模型"""
    total_count: int = Field(..., description="总操作数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    results: List[StrategyActionResponse] = Field(..., description="操作结果列表")
    overall_success: bool = Field(..., description="整体是否成功")
    execution_time_ms: float = Field(..., description="总执行时间")


# ===== 信号相关响应模型 =====

class SignalData(BaseModel):
    """信号数据模型"""
    signal_id: str = Field(..., description="信号ID")
    strategy_id: str = Field(..., description="策略ID")
    signal_type: str = Field(..., description="信号类型")
    action: str = Field(..., description="信号动作")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    strength: float = Field(..., description="信号强度")
    symbol: str = Field(..., description="交易标的")
    price: Optional[float] = Field(None, description="价格")
    volume: Optional[float] = Field(None, description="数量")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")
    timestamp: datetime = Field(..., description="信号时间戳")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class SignalHistoryResponse(BaseModel):
    """信号历史响应模型"""
    signals: List[SignalData] = Field(..., description="信号列表")
    statistics: Dict[str, Any] = Field(default={}, description="统计信息")
    time_range: Dict[str, datetime] = Field(..., description="时间范围")


class SignalAggregationResponse(BaseModel):
    """信号聚合响应模型"""
    aggregated_signal: SignalData = Field(..., description="聚合信号")
    input_signals: List[SignalData] = Field(..., description="输入信号")
    aggregation_method: str = Field(..., description="聚合方法")
    aggregation_quality: float = Field(..., description="聚合质量")
    confidence_adjustment: float = Field(..., description="置信度调整")
    reasoning: str = Field(..., description="聚合推理")
    execution_time_ms: float = Field(..., description="执行时间")


class SignalSubscriptionResponse(BaseModel):
    """信号订阅响应模型"""
    subscription_id: str = Field(..., description="订阅ID")
    channels: List[str] = Field(..., description="订阅频道")
    filters: Dict[str, Any] = Field(default={}, description="过滤条件")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    is_active: bool = Field(default=True, description="是否激活")


# ===== 系统相关响应模型 =====

class ResourceUsage(BaseModel):
    """资源使用情况模型"""
    cpu_percent: float = Field(..., description="CPU使用率")
    memory_usage_mb: float = Field(..., description="内存使用量(MB)")
    memory_percent: float = Field(..., description="内存使用率")
    disk_usage_gb: Optional[float] = Field(None, description="磁盘使用量(GB)")
    disk_percent: Optional[float] = Field(None, description="磁盘使用率")
    network_io_mb: Optional[Dict[str, float]] = Field(None, description="网络IO(MB)")
    active_connections: Optional[int] = Field(None, description="活跃连接数")
    thread_count: Optional[int] = Field(None, description="线程数")


class ComponentHealth(BaseModel):
    """组件健康状态模型"""
    component_name: str = Field(..., description="组件名称")
    status: str = Field(..., description="状态")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    last_error: Optional[str] = Field(None, description="最后错误")
    last_error_time: Optional[datetime] = Field(None, description="最后错误时间")
    metrics: Dict[str, Any] = Field(default={}, description="组件指标")
    dependencies: List[str] = Field(default=[], description="依赖组件")


class SystemHealthResponse(BaseModel):
    """系统健康响应模型"""
    overall_status: str = Field(..., description="整体状态")
    uptime_seconds: float = Field(..., description="系统运行时间")
    version: str = Field(..., description="系统版本")
    environment: str = Field(..., description="运行环境")
    components: List[ComponentHealth] = Field(..., description="组件健康状态")
    resource_usage: ResourceUsage = Field(..., description="资源使用情况")
    active_strategies: int = Field(..., description="活跃策略数")
    total_requests: int = Field(..., description="总请求数")
    error_rate: float = Field(..., description="错误率")
    average_response_time_ms: float = Field(..., description="平均响应时间")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="最后更新时间")


class SystemMetricsResponse(BaseModel):
    """系统指标响应模型"""
    metrics: Dict[str, List[Dict[str, Any]]] = Field(..., description="指标数据")
    time_range: Dict[str, datetime] = Field(..., description="时间范围")
    aggregation: str = Field(..., description="聚合方式")
    interval: str = Field(..., description="时间间隔")
    metadata: Dict[str, Any] = Field(default={}, description="元数据")


class SystemConfigResponse(BaseModel):
    """系统配置响应模型"""
    config_sections: Dict[str, Dict[str, Any]] = Field(..., description="配置分组")
    last_updated: datetime = Field(..., description="最后更新时间")
    version: str = Field(..., description="配置版本")
    readonly_sections: List[str] = Field(default=[], description="只读配置分组")


# ===== 日志相关响应模型 =====

class LogEntry(BaseModel):
    """日志条目模型"""
    log_id: str = Field(..., description="日志ID")
    timestamp: datetime = Field(..., description="时间戳")
    level: str = Field(..., description="日志级别")
    component: str = Field(..., description="组件名称")
    message: str = Field(..., description="日志消息")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    request_id: Optional[str] = Field(None, description="请求ID")
    context: Dict[str, Any] = Field(default={}, description="上下文信息")
    stack_trace: Optional[str] = Field(None, description="堆栈跟踪")


class AuditLogEntry(BaseModel):
    """审计日志条目模型"""
    audit_id: str = Field(..., description="审计ID")
    timestamp: datetime = Field(..., description="时间戳")
    user_id: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    action: str = Field(..., description="操作类型")
    resource: str = Field(..., description="资源")
    resource_id: Optional[str] = Field(None, description="资源ID")
    ip_address: str = Field(..., description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    request_data: Optional[Dict[str, Any]] = Field(None, description="请求数据")
    response_status: int = Field(..., description="响应状态码")
    execution_time_ms: float = Field(..., description="执行时间")
    success: bool = Field(..., description="是否成功")
    error_message: Optional[str] = Field(None, description="错误信息")


# ===== 统计和分析相关响应模型 =====

class StatisticsResponse(BaseModel):
    """统计响应模型"""
    period: str = Field(..., description="统计周期")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    total_count: int = Field(..., description="总数")
    success_count: int = Field(..., description="成功数")
    error_count: int = Field(..., description="错误数")
    average_value: float = Field(..., description="平均值")
    min_value: float = Field(..., description="最小值")
    max_value: float = Field(..., description="最大值")
    percentiles: Dict[str, float] = Field(default={}, description="百分位数")
    breakdown: Dict[str, int] = Field(default={}, description="详细分解")


class PerformanceMetrics(BaseModel):
    """性能指标模型"""
    requests_per_second: float = Field(..., description="每秒请求数")
    average_response_time_ms: float = Field(..., description="平均响应时间")
    p95_response_time_ms: float = Field(..., description="95%响应时间")
    p99_response_time_ms: float = Field(..., description="99%响应时间")
    error_rate: float = Field(..., description="错误率")
    success_rate: float = Field(..., description="成功率")
    concurrent_users: int = Field(..., description="并发用户数")
    active_connections: int = Field(..., description="活跃连接数")
    throughput_mbps: Optional[float] = Field(None, description="吞吐量(Mbps)")


# ===== 文件和导入导出相关响应模型 =====

class FileUploadResponse(BaseModel):
    """文件上传响应模型"""
    file_id: str = Field(..., description="文件ID")
    filename: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小(字节)")
    file_type: str = Field(..., description="文件类型")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="上传时间")
    checksum: str = Field(..., description="文件校验和")
    storage_path: str = Field(..., description="存储路径")


class ExportResponse(BaseModel):
    """导出响应模型"""
    export_id: str = Field(..., description="导出ID")
    export_type: str = Field(..., description="导出类型")
    file_url: str = Field(..., description="文件URL")
    file_size: int = Field(..., description="文件大小(字节)")
    record_count: int = Field(..., description="记录数")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    format: str = Field(..., description="文件格式")


# ===== WebSocket相关响应模型 =====

class WebSocketMessage(BaseModel):
    """WebSocket消息模型"""
    message_type: str = Field(..., description="消息类型")
    channel: str = Field(..., description="频道")
    data: Any = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="时间戳")
    sequence: Optional[int] = Field(None, description="序列号")


class WebSocketSubscriptionStatus(BaseModel):
    """WebSocket订阅状态模型"""
    subscription_id: str = Field(..., description="订阅ID")
    active_channels: List[str] = Field(..., description="活跃频道")
    message_count: int = Field(..., description="消息数")
    last_message_time: Optional[datetime] = Field(None, description="最后消息时间")
    connection_time: datetime = Field(..., description="连接时间")
    client_info: Dict[str, Any] = Field(default={}, description="客户端信息")