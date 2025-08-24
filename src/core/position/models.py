"""
仓位管理相关数据模型

定义仓位信息、平仓策略、平仓原因等核心数据结构。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal


class ClosingReason(Enum):
    """平仓原因枚举"""
    PROFIT_TARGET = "profit_target"           # 目标盈利平仓
    STOP_LOSS = "stop_loss"                   # 止损平仓
    TRAILING_STOP = "trailing_stop"           # 跟踪止损平仓
    TIME_BASED = "time_based"                 # 时间止损平仓
    TECHNICAL_REVERSAL = "technical_reversal" # 技术指标反转信号
    SENTIMENT_CHANGE = "sentiment_change"     # 市场情绪剧烈变化
    RISK_MANAGEMENT = "risk_management"       # 风险管理平仓
    MANUAL = "manual"                         # 手动平仓
    EMERGENCY = "emergency"                   # 紧急平仓


class ClosingAction(Enum):
    """平仓动作类型"""
    FULL_CLOSE = "full_close"         # 全部平仓
    PARTIAL_CLOSE = "partial_close"   # 部分平仓
    NO_ACTION = "no_action"           # 无动作


@dataclass
class PositionInfo:
    """仓位信息数据类"""
    position_id: str                    # 仓位ID
    symbol: str                         # 交易标的
    entry_price: float                  # 入场价格
    current_price: float                # 当前价格
    quantity: float                     # 仓位数量
    side: str                          # 持仓方向 ('long' or 'short')
    entry_time: datetime               # 入场时间
    unrealized_pnl: float              # 未实现盈亏
    unrealized_pnl_pct: float          # 未实现盈亏百分比
    
    # 风险管理参数
    stop_loss: Optional[float] = None   # 止损价格
    take_profit: Optional[float] = None # 止盈价格
    trailing_stop: Optional[float] = None # 跟踪止损距离
    max_hold_time: Optional[timedelta] = None # 最大持仓时间
    
    # 跟踪数据
    highest_price: Optional[float] = None    # 最高价格（用于跟踪止损）
    lowest_price: Optional[float] = None     # 最低价格（用于跟踪止损）
    max_profit: float = 0.0                  # 最大盈利
    max_loss: float = 0.0                    # 最大亏损
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.side not in ['long', 'short']:
            raise ValueError(f"持仓方向必须是'long'或'short'，当前值: {self.side}")
        
        if self.quantity <= 0:
            raise ValueError(f"仓位数量必须大于0，当前值: {self.quantity}")
        
        if self.entry_price <= 0:
            raise ValueError(f"入场价格必须大于0，当前值: {self.entry_price}")
        
        # 初始化跟踪价格
        if self.highest_price is None:
            self.highest_price = max(self.entry_price, self.current_price)
        if self.lowest_price is None:
            self.lowest_price = min(self.entry_price, self.current_price)
    
    def update_price(self, new_price: float) -> None:
        """更新价格和相关计算"""
        self.current_price = new_price
        self.last_update = datetime.utcnow()
        
        # 更新跟踪价格
        if self.highest_price is None or new_price > self.highest_price:
            self.highest_price = new_price
        if self.lowest_price is None or new_price < self.lowest_price:
            self.lowest_price = new_price
        
        # 计算未实现盈亏
        if self.side == 'long':
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = (new_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
            self.unrealized_pnl_pct = (self.entry_price - new_price) / self.entry_price * 100
        
        # 更新最大盈利/亏损
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl
    
    @property
    def is_profitable(self) -> bool:
        """判断是否盈利"""
        return self.unrealized_pnl > 0
    
    @property
    def hold_duration(self) -> timedelta:
        """持仓时间"""
        return datetime.utcnow() - self.entry_time
    
    @property
    def is_long(self) -> bool:
        """是否多头仓位"""
        return self.side == 'long'
    
    @property
    def is_short(self) -> bool:
        """是否空头仓位"""
        return self.side == 'short'


@dataclass
class ClosingStrategy:
    """平仓策略配置"""
    strategy_name: str                      # 策略名称
    enabled: bool = True                    # 是否启用
    priority: int = 1                       # 优先级（数字越小优先级越高）
    parameters: Dict[str, Any] = field(default_factory=dict)  # 策略参数
    
    def __post_init__(self):
        """验证策略配置"""
        if not self.strategy_name:
            raise ValueError("策略名称不能为空")
        
        if self.priority < 1:
            raise ValueError("策略优先级必须大于0")


@dataclass
class PositionCloseRequest:
    """平仓请求"""
    position_id: str                        # 仓位ID
    closing_reason: ClosingReason           # 平仓原因
    action: ClosingAction                   # 平仓动作
    quantity_to_close: float               # 平仓数量
    target_price: Optional[float] = None    # 目标价格
    urgency: str = "normal"                # 紧急程度 ('low', 'normal', 'high', 'emergency')
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    request_time: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """验证请求参数"""
        if self.quantity_to_close <= 0:
            raise ValueError("平仓数量必须大于0")
        
        if self.urgency not in ['low', 'normal', 'high', 'emergency']:
            raise ValueError("紧急程度必须是: 'low', 'normal', 'high', 'emergency'")


@dataclass
class PositionCloseResult:
    """平仓结果"""
    request_id: str                         # 请求ID
    position_id: str                        # 仓位ID
    success: bool                           # 是否成功
    actual_quantity_closed: float           # 实际平仓数量
    close_price: float                      # 平仓价格
    realized_pnl: float                     # 已实现盈亏
    closing_reason: ClosingReason           # 平仓原因
    close_time: datetime                    # 平仓时间
    error_message: Optional[str] = None     # 错误信息
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外信息
    
    @property
    def is_full_close(self) -> bool:
        """是否全部平仓"""
        return self.metadata.get('is_full_close', False)
    
    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.metadata.get('remaining_quantity', 0.0)


@dataclass
class ATRInfo:
    """ATR（平均真实波幅）信息"""
    period: int = 14                       # ATR计算周期
    current_atr: float = 0.0               # 当前ATR值
    atr_multiplier: float = 2.0            # ATR倍数
    
    @property
    def dynamic_stop_distance(self) -> float:
        """基于ATR的动态止损距离"""
        return self.current_atr * self.atr_multiplier


@dataclass
class VolatilityInfo:
    """波动率信息"""
    current_volatility: float = 0.0       # 当前波动率
    avg_volatility: float = 0.0           # 平均波动率
    volatility_percentile: float = 0.5    # 波动率分位数
    
    @property
    def is_high_volatility(self) -> bool:
        """是否高波动率环境"""
        return self.volatility_percentile > 0.8
    
    @property
    def is_low_volatility(self) -> bool:
        """是否低波动率环境"""
        return self.volatility_percentile < 0.2


@dataclass
class CorrelationRisk:
    """相关性风险信息"""
    symbol: str                            # 标的符号
    correlations: Dict[str, float] = field(default_factory=dict)  # 与其他标的的相关性
    max_correlation: float = 0.0           # 最大相关性
    high_correlation_symbols: List[str] = field(default_factory=list)  # 高相关性标的
    
    def get_correlation(self, other_symbol: str) -> float:
        """获取与指定标的的相关性"""
        return self.correlations.get(other_symbol, 0.0)
    
    def add_correlation(self, other_symbol: str, correlation: float) -> None:
        """添加相关性数据"""
        self.correlations[other_symbol] = correlation
        if abs(correlation) > abs(self.max_correlation):
            self.max_correlation = correlation
        
        # 更新高相关性标的列表（绝对值大于0.7认为高相关）
        if abs(correlation) > 0.7 and other_symbol not in self.high_correlation_symbols:
            self.high_correlation_symbols.append(other_symbol)