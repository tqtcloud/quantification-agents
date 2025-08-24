"""
DuckDB存储系统的数据模型
定义历史数据存储的结构化模型和配置
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict, field_validator


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    SNAPPY = "snappy" 
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"


class PartitionStrategy(Enum):
    """分区策略枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    NONE = "none"


class DataCategory(Enum):
    """数据类别枚举"""
    MARKET_DATA = "market_data"
    TRADING_SIGNALS = "trading_signals"
    ORDER_RECORDS = "order_records"  
    RISK_METRICS = "risk_metrics"
    PERFORMANCE_DATA = "performance_data"
    TECHNICAL_INDICATORS = "technical_indicators"


@dataclass
class StorageConfig:
    """存储配置"""
    # 基本配置
    max_memory_gb: float = 4.0
    thread_count: int = 4
    checkpoint_threshold_gb: float = 2.0
    
    # 压缩配置
    default_compression: CompressionType = CompressionType.SNAPPY
    enable_compression: bool = True
    
    # 分区配置
    partition_strategy: PartitionStrategy = PartitionStrategy.MONTHLY
    enable_partitioning: bool = True
    
    # 缓存配置
    enable_result_cache: bool = True
    cache_size_mb: int = 512
    
    # 优化配置
    auto_optimize: bool = True
    optimize_interval_hours: int = 24
    
    # 导出配置
    export_format: str = "parquet"
    export_compression: CompressionType = CompressionType.SNAPPY


@dataclass
class DataRetentionPolicy:
    """数据保留策略"""
    # 保留时间（天数）
    market_data_retention_days: int = 365
    signal_retention_days: int = 180
    order_retention_days: int = 730  # 2年
    risk_metrics_retention_days: int = 365
    performance_retention_days: int = 1095  # 3年
    
    # 压缩策略
    compress_after_days: int = 30
    archive_after_days: int = 90
    
    # 自动清理配置
    enable_auto_cleanup: bool = True
    cleanup_interval_hours: int = 24
    
    # 归档配置
    enable_archiving: bool = True
    archive_to_parquet: bool = True


class HistoricalMarketData(BaseModel):
    """历史市场数据模型"""
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    # 基础字段
    timestamp: datetime
    symbol: str
    environment: str
    
    # 价格数据
    open_price: Decimal = Field(gt=0, description="开盘价")
    high_price: Decimal = Field(gt=0, description="最高价")
    low_price: Decimal = Field(gt=0, description="最低价")
    close_price: Decimal = Field(gt=0, description="收盘价")
    
    # 成交量数据
    volume: Decimal = Field(ge=0, description="成交量")
    quote_volume: Optional[Decimal] = Field(default=None, ge=0, description="报价资产成交量")
    
    # 成交统计
    trade_count: int = Field(default=0, ge=0, description="成交笔数")
    taker_buy_volume: Optional[Decimal] = Field(default=None, ge=0, description="主动买入成交量")
    taker_buy_quote_volume: Optional[Decimal] = Field(default=None, ge=0, description="主动买入报价成交量")
    
    # 时间框架
    interval: str = Field(description="时间间隔，如1m, 5m, 1h, 1d")
    
    # 订单簿数据（可选）
    bid_price: Optional[Decimal] = Field(default=None, gt=0, description="最佳买价")
    ask_price: Optional[Decimal] = Field(default=None, gt=0, description="最佳卖价")
    bid_volume: Optional[Decimal] = Field(default=None, ge=0, description="最佳买量")
    ask_volume: Optional[Decimal] = Field(default=None, ge=0, description="最佳卖量")
    
    # 衍生品特有数据
    open_interest: Optional[Decimal] = Field(default=None, ge=0, description="未平仓合约")
    funding_rate: Optional[Decimal] = Field(default=None, description="资金费率")
    mark_price: Optional[Decimal] = Field(default=None, gt=0, description="标记价格")
    index_price: Optional[Decimal] = Field(default=None, gt=0, description="指数价格")
    
    # 元数据
    data_source: str = Field(default="binance", description="数据源")
    data_quality: float = Field(default=1.0, ge=0, le=1, description="数据质量分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    # 时间戳
    ingestion_time: datetime = Field(default_factory=datetime.utcnow, description="数据摄入时间")
    
    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        valid_intervals = ['1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if v not in valid_intervals:
            raise ValueError(f'Invalid interval: {v}. Must be one of {valid_intervals}')
        return v

    def to_ohlcv_dict(self) -> Dict[str, Any]:
        """转换为OHLCV字典格式"""
        return {
            'timestamp': self.timestamp,
            'open': float(self.open_price),
            'high': float(self.high_price), 
            'low': float(self.low_price),
            'close': float(self.close_price),
            'volume': float(self.volume)
        }


class HistoricalTradingSignal(BaseModel):
    """历史交易信号模型"""
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    # 基础字段
    timestamp: datetime
    symbol: str
    environment: str
    
    # 信号信息
    signal_id: str = Field(description="信号唯一标识符")
    source: str = Field(description="信号源（Agent名称）")
    signal_type: str = Field(description="信号类型")
    
    # 交易动作
    action: str = Field(description="交易动作：BUY/SELL")
    strength: float = Field(ge=-1.0, le=1.0, description="信号强度：-1(强卖) 到 1(强买)")
    confidence: float = Field(ge=0.0, le=1.0, description="信号置信度")
    
    # 价格和数量
    target_price: Optional[Decimal] = Field(default=None, gt=0, description="目标价格")
    suggested_quantity: Optional[Decimal] = Field(default=None, gt=0, description="建议数量")
    max_position_size: Optional[Decimal] = Field(default=None, gt=0, description="最大持仓大小")
    
    # 风险控制
    stop_loss: Optional[Decimal] = Field(default=None, gt=0, description="止损价格")
    take_profit: Optional[Decimal] = Field(default=None, gt=0, description="止盈价格")
    max_drawdown_pct: Optional[float] = Field(default=None, ge=0, le=1, description="最大回撤百分比")
    
    # 时间相关
    validity_period: Optional[timedelta] = Field(default=None, description="信号有效期")
    expiry_time: Optional[datetime] = Field(default=None, description="信号过期时间")
    
    # 技术指标上下文
    technical_context: Dict[str, Any] = Field(default_factory=dict, description="技术指标上下文")
    market_condition: str = Field(default="unknown", description="市场状况：bull/bear/sideways/volatile")
    
    # 执行状态
    execution_status: str = Field(default="pending", description="执行状态：pending/executed/cancelled/expired")
    execution_price: Optional[Decimal] = Field(default=None, gt=0, description="实际执行价格")
    execution_time: Optional[datetime] = Field(default=None, description="执行时间")
    execution_quantity: Optional[Decimal] = Field(default=None, ge=0, description="实际执行数量")
    
    # 绩效数据
    realized_pnl: Optional[Decimal] = Field(default=None, description="已实现盈亏")
    unrealized_pnl: Optional[Decimal] = Field(default=None, description="未实现盈亏")
    win_rate: Optional[float] = Field(default=None, ge=0, le=1, description="胜率")
    
    # 原因和元数据
    reason: str = Field(description="信号生成原因")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    ingestion_time: datetime = Field(default_factory=datetime.utcnow, description="数据摄入时间")
    
    @field_validator('action')
    @classmethod  
    def validate_action(cls, v: str) -> str:
        if v.upper() not in ['BUY', 'SELL', 'HOLD']:
            raise ValueError(f'Invalid action: {v}. Must be BUY, SELL, or HOLD')
        return v.upper()


class HistoricalOrderRecord(BaseModel):
    """历史订单记录模型"""
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    # 基础字段
    timestamp: datetime
    symbol: str  
    environment: str
    
    # 订单标识
    order_id: str = Field(description="交易所订单ID")
    client_order_id: str = Field(description="客户端订单ID")
    strategy_id: Optional[str] = Field(default=None, description="策略ID")
    signal_id: Optional[str] = Field(default=None, description="关联信号ID")
    
    # 订单基本信息
    side: str = Field(description="订单方向：BUY/SELL")
    order_type: str = Field(description="订单类型：MARKET/LIMIT/STOP等")
    quantity: Decimal = Field(gt=0, description="订单数量")
    price: Optional[Decimal] = Field(default=None, gt=0, description="订单价格")
    
    # 执行信息
    status: str = Field(description="订单状态")
    executed_qty: Decimal = Field(ge=0, description="已执行数量")
    executed_quote_qty: Decimal = Field(ge=0, description="已执行报价数量")
    avg_price: Decimal = Field(ge=0, description="平均成交价格")
    
    # 费用信息
    commission: Decimal = Field(ge=0, description="手续费")
    commission_asset: str = Field(description="手续费资产")
    commission_rate: Optional[float] = Field(default=None, ge=0, description="手续费率")
    
    # 时间信息
    created_time: datetime = Field(description="订单创建时间")
    updated_time: datetime = Field(description="订单更新时间") 
    filled_time: Optional[datetime] = Field(default=None, description="订单完成时间")
    
    # 高级订单参数
    stop_price: Optional[Decimal] = Field(default=None, gt=0, description="停止价格")
    iceberg_qty: Optional[Decimal] = Field(default=None, gt=0, description="冰山订单数量")
    time_in_force: str = Field(default="GTC", description="有效时间类型")
    
    # 持仓相关
    position_side: str = Field(default="BOTH", description="持仓方向")
    reduce_only: bool = Field(default=False, description="是否只减仓")
    close_position: bool = Field(default=False, description="是否关闭持仓")
    
    # 绩效相关
    realized_pnl: Optional[Decimal] = Field(default=None, description="已实现盈亏")
    pnl_percentage: Optional[float] = Field(default=None, description="盈亏百分比")
    slippage: Optional[Decimal] = Field(default=None, description="滑点")
    slippage_bps: Optional[float] = Field(default=None, description="滑点（基点）")
    
    # 市场数据快照
    market_price_at_order: Optional[Decimal] = Field(default=None, gt=0, description="下单时市价")
    spread_at_order: Optional[Decimal] = Field(default=None, ge=0, description="下单时价差")
    volume_at_order: Optional[Decimal] = Field(default=None, ge=0, description="下单时成交量")
    
    # 风险指标
    leverage: Optional[float] = Field(default=None, ge=1, description="杠杆倍数")
    margin_used: Optional[Decimal] = Field(default=None, ge=0, description="使用保证金")
    liquidation_price: Optional[Decimal] = Field(default=None, gt=0, description="强平价格")
    
    # 元数据
    order_source: str = Field(default="system", description="订单来源")
    execution_algorithm: Optional[str] = Field(default=None, description="执行算法")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    ingestion_time: datetime = Field(default_factory=datetime.utcnow, description="数据摄入时间")
    
    @field_validator('side')
    @classmethod
    def validate_side(cls, v: str) -> str:
        if v.upper() not in ['BUY', 'SELL']:
            raise ValueError(f'Invalid side: {v}. Must be BUY or SELL')
        return v.upper()

    def calculate_total_cost(self) -> Decimal:
        """计算订单总成本（包含手续费）"""
        base_cost = self.executed_qty * self.avg_price
        return base_cost + self.commission

    def is_profitable(self) -> Optional[bool]:
        """判断订单是否盈利"""
        if self.realized_pnl is not None:
            return self.realized_pnl > 0
        return None


class HistoricalRiskMetrics(BaseModel):
    """历史风险指标模型"""
    model_config = ConfigDict(frozen=True, extra='forbid')
    
    # 基础字段
    timestamp: datetime
    environment: str
    calculation_period: str = Field(description="计算周期：1h/1d/1w/1m")
    
    # 账户级别风险
    account_id: Optional[str] = Field(default=None, description="账户ID")
    total_balance: Decimal = Field(ge=0, description="总余额")
    available_balance: Decimal = Field(ge=0, description="可用余额") 
    total_position_value: Decimal = Field(ge=0, description="总持仓价值")
    
    # 暴露度风险
    total_exposure: Decimal = Field(ge=0, description="总暴露度")
    long_exposure: Decimal = Field(ge=0, description="多头暴露度")
    short_exposure: Decimal = Field(ge=0, description="空头暴露度")
    net_exposure: Decimal = Field(description="净暴露度")
    gross_exposure: Decimal = Field(ge=0, description="总暴露度")
    
    # 保证金风险
    used_margin: Decimal = Field(ge=0, description="已用保证金")
    free_margin: Decimal = Field(ge=0, description="可用保证金")
    margin_ratio: float = Field(ge=0, le=1, description="保证金比率")
    maintenance_margin: Decimal = Field(ge=0, description="维持保证金")
    
    # 杠杆风险
    effective_leverage: float = Field(ge=1, description="有效杠杆")
    max_leverage_used: float = Field(ge=1, description="最大杠杆使用")
    leverage_utilization: float = Field(ge=0, le=1, description="杠杆利用率")
    
    # 流动性风险
    liquidation_buffer: Decimal = Field(ge=0, description="强平缓冲")
    time_to_liquidation_hours: Optional[float] = Field(default=None, ge=0, description="距强平时间(小时)")
    liquidity_score: float = Field(ge=0, le=1, description="流动性评分")
    
    # 回撤风险
    max_drawdown: float = Field(ge=0, description="最大回撤")
    current_drawdown: float = Field(ge=0, description="当前回撤")
    drawdown_duration_days: int = Field(ge=0, description="回撤持续天数")
    recovery_factor: Optional[float] = Field(default=None, ge=0, description="恢复因子")
    
    # VaR风险
    var_95: Decimal = Field(description="95%置信度VaR")
    var_99: Decimal = Field(description="99%置信度VaR")
    expected_shortfall_95: Optional[Decimal] = Field(default=None, description="95%ES")
    expected_shortfall_99: Optional[Decimal] = Field(default=None, description="99%ES")
    
    # 波动性风险
    portfolio_volatility: float = Field(ge=0, description="投资组合波动率")
    realized_volatility: float = Field(ge=0, description="已实现波动率")
    implied_volatility: Optional[float] = Field(default=None, ge=0, description="隐含波动率")
    volatility_percentile: float = Field(ge=0, le=100, description="波动率百分位")
    
    # 相关性风险
    max_correlation: float = Field(ge=-1, le=1, description="最大相关性")
    avg_correlation: float = Field(ge=-1, le=1, description="平均相关性")
    concentration_risk: float = Field(ge=0, le=1, description="集中度风险")
    
    # 绩效风险比率
    sharpe_ratio: Optional[float] = Field(default=None, description="夏普比率")
    sortino_ratio: Optional[float] = Field(default=None, description="索提诺比率")
    calmar_ratio: Optional[float] = Field(default=None, description="卡玛比率")
    max_drawdown_ratio: float = Field(ge=0, description="最大回撤比率")
    
    # 交易风险
    win_rate: float = Field(ge=0, le=1, description="胜率")
    profit_factor: float = Field(ge=0, description="盈利因子")
    avg_win_loss_ratio: float = Field(ge=0, description="平均盈亏比")
    consecutive_losses: int = Field(ge=0, description="连续亏损次数")
    
    # 流动性和市场风险
    market_impact: Optional[float] = Field(default=None, ge=0, description="市场影响")
    bid_ask_spread: Optional[Decimal] = Field(default=None, ge=0, description="买卖价差")
    slippage_cost: Optional[Decimal] = Field(default=None, ge=0, description="滑点成本")
    
    # 资金成本
    funding_cost: Decimal = Field(default=Decimal('0'), description="资金成本")
    borrowing_cost: Optional[Decimal] = Field(default=None, ge=0, description="借贷成本")
    opportunity_cost: Optional[Decimal] = Field(default=None, description="机会成本")
    
    # 分品种风险
    symbol_risks: Dict[str, Any] = Field(default_factory=dict, description="分品种风险")
    sector_risks: Dict[str, Any] = Field(default_factory=dict, description="分板块风险")  
    
    # 时间相关风险
    daily_pnl: Decimal = Field(description="当日盈亏")
    weekly_pnl: Optional[Decimal] = Field(default=None, description="周盈亏")
    monthly_pnl: Optional[Decimal] = Field(default=None, description="月盈亏")
    ytd_pnl: Optional[Decimal] = Field(default=None, description="年初至今盈亏")
    
    # 压力测试
    stress_test_results: Dict[str, Any] = Field(default_factory=dict, description="压力测试结果")
    scenario_analysis: Dict[str, Any] = Field(default_factory=dict, description="情景分析")
    
    # 模型风险
    model_confidence: float = Field(ge=0, le=1, description="模型置信度")
    prediction_accuracy: Optional[float] = Field(default=None, ge=0, le=1, description="预测准确度")
    backtest_performance: Dict[str, Any] = Field(default_factory=dict, description="回测表现")
    
    # 元数据
    risk_model_version: str = Field(default="v1.0", description="风控模型版本")
    calculation_method: str = Field(description="计算方法")
    data_quality_score: float = Field(ge=0, le=1, description="数据质量评分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    ingestion_time: datetime = Field(default_factory=datetime.utcnow, description="数据摄入时间")
    
    def get_risk_level(self) -> str:
        """获取风险等级"""
        # 基于多个指标综合评估风险等级
        risk_score = 0.0
        
        # 杠杆风险 (30% 权重)
        if self.effective_leverage > 10:
            risk_score += 30
        elif self.effective_leverage > 5:
            risk_score += 20
        elif self.effective_leverage > 2:
            risk_score += 10
        
        # 回撤风险 (25% 权重)  
        if self.current_drawdown > 0.2:
            risk_score += 25
        elif self.current_drawdown > 0.1:
            risk_score += 15
        elif self.current_drawdown > 0.05:
            risk_score += 8
        
        # 保证金风险 (25% 权重)
        if self.margin_ratio > 0.8:
            risk_score += 25
        elif self.margin_ratio > 0.6:
            risk_score += 15
        elif self.margin_ratio > 0.4:
            risk_score += 8
        
        # 集中度风险 (20% 权重)
        if self.concentration_risk > 0.5:
            risk_score += 20
        elif self.concentration_risk > 0.3:
            risk_score += 12
        elif self.concentration_risk > 0.2:
            risk_score += 6
        
        # 返回风险等级
        if risk_score >= 70:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"

    def is_margin_call_risk(self) -> bool:
        """判断是否有追加保证金风险"""
        return self.margin_ratio > 0.7 or self.liquidation_buffer < Decimal('1000')