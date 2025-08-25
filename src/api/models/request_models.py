"""
API请求数据模型
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class StrategyAction(str, Enum):
    """策略操作枚举"""
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RESTART = "restart"


class StrategyType(str, Enum):
    """策略类型枚举"""
    HFT = "hft"
    AI_AGENT = "ai_agent"


class TimeRange(BaseModel):
    """时间范围模型"""
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    
    @field_validator('end_time')
    def validate_end_time(cls, v, info):
        if v and info.data.get('start_time') and v <= info.data['start_time']:
            raise ValueError('end_time must be after start_time')
        return v


class PaginationRequest(BaseModel):
    """分页请求模型"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size


class SortRequest(BaseModel):
    """排序请求模型"""
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", pattern="^(asc|desc)$", description="排序方向")


class FilterRequest(BaseModel):
    """过滤请求模型"""
    filters: Dict[str, Any] = Field(default={}, description="过滤条件")


# ===== 策略控制相关请求模型 =====

class StrategyStartRequest(BaseModel):
    """策略启动请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    strategy_type: StrategyType = Field(..., description="策略类型")
    config: Optional[Dict[str, Any]] = Field(None, description="策略配置")
    force: bool = Field(False, description="是否强制启动")
    dry_run: bool = Field(False, description="是否为模拟运行")


class StrategyStopRequest(BaseModel):
    """策略停止请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    force: bool = Field(False, description="是否强制停止")
    save_state: bool = Field(True, description="是否保存状态")
    reason: Optional[str] = Field(None, description="停止原因")


class StrategyRestartRequest(BaseModel):
    """策略重启请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    config: Optional[Dict[str, Any]] = Field(None, description="新的策略配置")
    preserve_state: bool = Field(True, description="是否保持状态")


class StrategyPauseRequest(BaseModel):
    """策略暂停请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    reason: Optional[str] = Field(None, description="暂停原因")


class StrategyResumeRequest(BaseModel):
    """策略恢复请求模型"""
    strategy_id: str = Field(..., description="策略ID")


class StrategyConfigRequest(BaseModel):
    """策略配置请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    config: Dict[str, Any] = Field(..., description="策略配置")
    validate_only: bool = Field(False, description="是否仅验证配置")


class StrategyStatusRequest(BaseModel):
    """策略状态请求模型"""
    strategy_ids: Optional[List[str]] = Field(None, description="策略ID列表，为空时查询所有")
    include_metrics: bool = Field(True, description="是否包含指标信息")
    include_config: bool = Field(False, description="是否包含配置信息")


class StrategyListRequest(PaginationRequest, SortRequest, FilterRequest):
    """策略列表请求模型"""
    strategy_type: Optional[StrategyType] = Field(None, description="策略类型过滤")
    status_filter: Optional[List[str]] = Field(None, description="状态过滤")
    include_inactive: bool = Field(False, description="是否包含非活跃策略")


class StrategyRegistrationRequest(BaseModel):
    """策略注册请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    strategy_type: StrategyType = Field(..., description="策略类型")
    name: str = Field(..., description="策略名称")
    description: Optional[str] = Field(None, description="策略描述")
    config: Dict[str, Any] = Field(..., description="策略配置")
    auto_start: bool = Field(False, description="是否自动启动")


# ===== 信号相关请求模型 =====

class SignalHistoryRequest(PaginationRequest, SortRequest):
    """信号历史请求模型"""
    strategy_ids: Optional[List[str]] = Field(None, description="策略ID过滤")
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    signal_types: Optional[List[str]] = Field(None, description="信号类型过滤")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="最小置信度")


class SignalSubscriptionRequest(BaseModel):
    """信号订阅请求模型"""
    strategy_ids: Optional[List[str]] = Field(None, description="策略ID过滤")
    signal_types: Optional[List[str]] = Field(None, description="信号类型过滤")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="最小置信度")
    callback_url: Optional[str] = Field(None, description="回调URL")


class SignalAggregationRequest(BaseModel):
    """信号聚合请求模型"""
    signals: List[Dict[str, Any]] = Field(..., description="信号列表")
    aggregation_strategy: str = Field("weighted_average", description="聚合策略")
    config: Optional[Dict[str, Any]] = Field(None, description="聚合配置")


# ===== 系统监控相关请求模型 =====

class SystemHealthRequest(BaseModel):
    """系统健康检查请求模型"""
    include_strategies: bool = Field(True, description="是否包含策略状态")
    include_resources: bool = Field(True, description="是否包含资源使用情况")
    include_performance: bool = Field(True, description="是否包含性能指标")
    detailed: bool = Field(False, description="是否返回详细信息")


class SystemMetricsRequest(BaseModel):
    """系统指标请求模型"""
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    metric_types: Optional[List[str]] = Field(None, description="指标类型过滤")
    aggregation: str = Field("average", description="聚合方式")
    interval: str = Field("1m", description="时间间隔")


class SystemConfigRequest(BaseModel):
    """系统配置请求模型"""
    config_section: Optional[str] = Field(None, description="配置分组")
    include_sensitive: bool = Field(False, description="是否包含敏感配置")


class SystemConfigUpdateRequest(BaseModel):
    """系统配置更新请求模型"""
    config_section: str = Field(..., description="配置分组")
    config_data: Dict[str, Any] = Field(..., description="配置数据")
    validate_only: bool = Field(False, description="是否仅验证配置")


# ===== 日志和审计相关请求模型 =====

class LogQueryRequest(PaginationRequest, SortRequest):
    """日志查询请求模型"""
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    log_levels: Optional[List[str]] = Field(None, description="日志级别过滤")
    components: Optional[List[str]] = Field(None, description="组件过滤")
    search_text: Optional[str] = Field(None, description="搜索文本")
    user_id: Optional[str] = Field(None, description="用户ID过滤")


class AuditLogRequest(PaginationRequest, SortRequest):
    """审计日志请求模型"""
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    user_ids: Optional[List[str]] = Field(None, description="用户ID过滤")
    actions: Optional[List[str]] = Field(None, description="操作类型过滤")
    resources: Optional[List[str]] = Field(None, description="资源过滤")
    ip_addresses: Optional[List[str]] = Field(None, description="IP地址过滤")


# ===== 批量操作请求模型 =====

class BatchStrategyActionRequest(BaseModel):
    """批量策略操作请求模型"""
    strategy_ids: List[str] = Field(..., description="策略ID列表")
    action: StrategyAction = Field(..., description="操作类型")
    config: Optional[Dict[str, Any]] = Field(None, description="操作配置")
    force: bool = Field(False, description="是否强制执行")
    dry_run: bool = Field(False, description="是否为模拟运行")


class BulkUpdateRequest(BaseModel):
    """批量更新请求模型"""
    target_ids: List[str] = Field(..., description="目标ID列表")
    update_data: Dict[str, Any] = Field(..., description="更新数据")
    validate_only: bool = Field(False, description="是否仅验证")


# ===== 文件上传相关请求模型 =====

class FileUploadRequest(BaseModel):
    """文件上传请求模型"""
    file_type: str = Field(..., description="文件类型")
    description: Optional[str] = Field(None, description="文件描述")
    overwrite: bool = Field(False, description="是否覆盖已存在文件")


class ConfigImportRequest(BaseModel):
    """配置导入请求模型"""
    config_type: str = Field(..., description="配置类型")
    merge_mode: str = Field("replace", description="合并模式: replace, merge, append")
    validate_only: bool = Field(False, description="是否仅验证配置")


# ===== WebSocket相关请求模型 =====

class WebSocketSubscriptionRequest(BaseModel):
    """WebSocket订阅请求模型"""
    channels: List[str] = Field(..., description="订阅频道列表")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    throttle_ms: Optional[int] = Field(None, description="节流时间(毫秒)")


class WebSocketUnsubscriptionRequest(BaseModel):
    """WebSocket取消订阅请求模型"""
    channels: List[str] = Field(..., description="取消订阅频道列表")


# ===== 通用请求模型 =====

class HealthCheckRequest(BaseModel):
    """健康检查请求模型"""
    deep_check: bool = Field(False, description="是否进行深度检查")
    timeout: Optional[int] = Field(30, description="超时时间(秒)")


class ValidationRequest(BaseModel):
    """验证请求模型"""
    data: Any = Field(..., description="待验证数据")
    schema_type: str = Field(..., description="验证模式类型")
    strict: bool = Field(True, description="是否严格验证")


class ExportRequest(BaseModel):
    """导出请求模型"""
    export_type: str = Field(..., description="导出类型")
    time_range: Optional[TimeRange] = Field(None, description="时间范围")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    format: str = Field("json", description="导出格式")
    include_metadata: bool = Field(True, description="是否包含元数据")