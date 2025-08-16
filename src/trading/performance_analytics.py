import asyncio
import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from src.core.models import Order, OrderSide, OrderType, OrderStatus
from src.utils.logger import LoggerMixin


class PerformancePeriod(Enum):
    """性能分析周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Optional[Decimal] = None
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    
    # 成本和费用
    entry_commission: Decimal = Decimal("0")
    exit_commission: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    
    # 盈亏信息
    realized_pnl: Decimal = Decimal("0")
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
    
    # 交易元数据
    strategy_name: Optional[str] = None
    trade_reason: Optional[str] = None
    market_condition: Optional[str] = None
    
    # 执行信息
    execution_delay_ms: float = 0.0
    is_maker: bool = False
    
    def calculate_pnl(self):
        """计算盈亏"""
        if self.exit_price is None:
            return
            
        if self.side == "BUY":
            self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity
            
        total_commission = self.entry_commission + self.exit_commission
        self.net_pnl = self.gross_pnl - total_commission - self.slippage_cost
        self.realized_pnl = self.net_pnl
    
    def get_duration(self) -> Optional[timedelta]:
        """获取交易持续时间"""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    def get_return_rate(self) -> Decimal:
        """获取收益率"""
        if self.exit_price is None:
            return Decimal("0")
            
        cost_basis = self.entry_price * self.quantity + self.entry_commission
        if cost_basis <= 0:
            return Decimal("0")
            
        return self.net_pnl / cost_basis


@dataclass
class PositionRecord:
    """持仓记录"""
    symbol: str
    side: str
    size: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    
    # 盈亏信息
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    
    # 时间信息
    open_time: datetime = field(default_factory=datetime.utcnow)
    last_update_time: datetime = field(default_factory=datetime.utcnow)
    
    # 风险信息
    max_unrealized_loss: Decimal = Decimal("0")
    max_unrealized_profit: Decimal = Decimal("0")
    
    def update_unrealized_pnl(self, current_price: Decimal):
        """更新未实现盈亏"""
        self.current_price = current_price
        
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.size
        else:  # SHORT
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.size
            
        # 更新最大盈亏记录
        if self.unrealized_pnl > self.max_unrealized_profit:
            self.max_unrealized_profit = self.unrealized_pnl
        elif self.unrealized_pnl < self.max_unrealized_loss:
            self.max_unrealized_loss = self.unrealized_pnl
            
        self.last_update_time = datetime.utcnow()


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基础指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    
    # 盈亏指标
    total_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    net_profit: Decimal = Decimal("0")
    
    # 收益指标
    total_return: Decimal = Decimal("0")
    annualized_return: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    
    # 风险指标
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    var_95: Decimal = Decimal("0")  # 95% VaR
    
    # 交易效率指标
    profit_factor: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    avg_win_loss_ratio: Decimal = Decimal("0")
    
    # 费用指标
    total_commission: Decimal = Decimal("0")
    total_slippage: Decimal = Decimal("0")
    commission_rate: Decimal = Decimal("0")
    
    # 时间指标
    avg_trade_duration: float = 0.0  # 小时
    max_trade_duration: float = 0.0
    min_trade_duration: float = 0.0
    
    # 连续性指标
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    current_streak_type: str = "none"  # "win", "loss", "none"
    
    # 分布指标
    pnl_std: Decimal = Decimal("0")
    pnl_skewness: Decimal = Decimal("0")
    pnl_kurtosis: Decimal = Decimal("0")


class PerformanceTracker(LoggerMixin):
    """性能追踪器"""
    
    def __init__(self, initial_balance: Decimal = Decimal("100000")):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # 交易记录
        self.trade_records: List[TradeRecord] = []
        self.position_records: Dict[str, PositionRecord] = {}
        
        # 历史余额记录（用于计算回撤）
        self.balance_history: List[Tuple[datetime, Decimal]] = []
        self.daily_returns: List[Decimal] = []
        
        # 绩效缓存
        self.metrics_cache: Dict[str, PerformanceMetrics] = {}
        self.cache_timestamp: Dict[str, datetime] = {}
        
        # 基准比较
        self.benchmark_returns: List[Decimal] = []
        
        # 分组统计
        self.strategy_performance: Dict[str, List[TradeRecord]] = defaultdict(list)
        self.symbol_performance: Dict[str, List[TradeRecord]] = defaultdict(list)
        
    def record_trade(self, trade: TradeRecord):
        """记录交易"""
        self.trade_records.append(trade)
        
        # 按策略分组
        if trade.strategy_name:
            self.strategy_performance[trade.strategy_name].append(trade)
            
        # 按交易对分组
        self.symbol_performance[trade.symbol].append(trade)
        
        # 更新余额
        if trade.net_pnl != Decimal("0"):
            self.current_balance += trade.net_pnl
            self.balance_history.append((datetime.utcnow(), self.current_balance))
            
        # 清除缓存
        self._clear_cache()
        
        self.log_debug(f"Trade recorded: {trade.symbol} {trade.side} {trade.net_pnl}")
    
    def update_position(self, symbol: str, position: PositionRecord):
        """更新持仓"""
        self.position_records[symbol] = position
        self._clear_cache()
    
    def remove_position(self, symbol: str):
        """移除持仓"""
        if symbol in self.position_records:
            del self.position_records[symbol]
            self._clear_cache()
    
    def calculate_performance_metrics(self, 
                                    period: PerformancePeriod = PerformancePeriod.ALL_TIME,
                                    strategy_name: Optional[str] = None,
                                    symbol: Optional[str] = None) -> PerformanceMetrics:
        """计算性能指标"""
        
        cache_key = f"{period.value}_{strategy_name}_{symbol}"
        
        # 检查缓存
        if (cache_key in self.metrics_cache and 
            cache_key in self.cache_timestamp and 
            datetime.utcnow() - self.cache_timestamp[cache_key] < timedelta(minutes=5)):
            return self.metrics_cache[cache_key]
        
        # 过滤交易记录
        trades = self._filter_trades(period, strategy_name, symbol)
        
        if not trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # 基础统计
        metrics.total_trades = len(trades)
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.win_rate = Decimal(str(metrics.winning_trades / metrics.total_trades)) if metrics.total_trades > 0 else Decimal("0")
        
        # 盈亏统计
        metrics.total_pnl = sum(t.net_pnl for t in trades)
        metrics.gross_profit = sum(t.net_pnl for t in winning_trades)
        metrics.gross_loss = abs(sum(t.net_pnl for t in losing_trades))
        metrics.net_profit = metrics.total_pnl
        
        # 费用统计
        metrics.total_commission = sum(t.entry_commission + t.exit_commission for t in trades)
        metrics.total_slippage = sum(t.slippage_cost for t in trades)
        
        # 平均盈亏
        if winning_trades:
            metrics.avg_win = sum(t.net_pnl for t in winning_trades) / Decimal(str(len(winning_trades)))
        if losing_trades:
            metrics.avg_loss = abs(sum(t.net_pnl for t in losing_trades)) / Decimal(str(len(losing_trades)))
        
        # 盈亏比和盈利因子
        if metrics.avg_loss > 0:
            metrics.avg_win_loss_ratio = metrics.avg_win / metrics.avg_loss
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        # 时间统计
        durations = [t.get_duration().total_seconds() / 3600 for t in trades if t.get_duration()]
        if durations:
            metrics.avg_trade_duration = sum(durations) / len(durations)
            metrics.max_trade_duration = max(durations)
            metrics.min_trade_duration = min(durations)
        
        # 收益率计算
        if self.initial_balance > 0:
            metrics.total_return = metrics.total_pnl / self.initial_balance
            
        # 计算年化收益率
        if trades:
            period_days = (trades[-1].exit_time - trades[0].entry_time).days if trades[-1].exit_time else 1
            if period_days > 0:
                metrics.annualized_return = metrics.total_return * Decimal(str(365 / period_days))
        
        # 回撤计算
        metrics.max_drawdown, metrics.current_drawdown = self._calculate_drawdown(trades)
        
        # 风险指标
        returns = [float(t.get_return_rate()) for t in trades]
        if len(returns) > 1:
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.var_95 = self._calculate_var(returns, 0.95)
            
            metrics.pnl_std = Decimal(str(statistics.stdev([float(t.net_pnl) for t in trades])))
            
        # Calmar比率
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        
        # 连续性统计
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = self._calculate_streaks(trades)
        
        # 缓存结果
        self.metrics_cache[cache_key] = metrics
        self.cache_timestamp[cache_key] = datetime.utcnow()
        
        return metrics
    
    def _filter_trades(self, 
                      period: PerformancePeriod,
                      strategy_name: Optional[str] = None,
                      symbol: Optional[str] = None) -> List[TradeRecord]:
        """过滤交易记录"""
        
        trades = self.trade_records.copy()
        
        # 按策略过滤
        if strategy_name:
            trades = [t for t in trades if t.strategy_name == strategy_name]
        
        # 按交易对过滤
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        # 按时间周期过滤
        if period != PerformancePeriod.ALL_TIME:
            cutoff_date = self._get_period_start_date(period)
            trades = [t for t in trades if t.entry_time >= cutoff_date]
        
        return trades
    
    def _get_period_start_date(self, period: PerformancePeriod) -> datetime:
        """获取周期开始日期"""
        now = datetime.utcnow()
        
        if period == PerformancePeriod.DAILY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.WEEKLY:
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.MONTHLY:
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.QUARTERLY:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            return now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == PerformancePeriod.YEARLY:
            return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return datetime.min
    
    def _calculate_drawdown(self, trades: List[TradeRecord]) -> Tuple[Decimal, Decimal]:
        """计算最大回撤和当前回撤"""
        if not trades:
            return Decimal("0"), Decimal("0")
        
        # 构建权益曲线
        equity_curve = []
        running_balance = self.initial_balance
        
        for trade in sorted(trades, key=lambda x: x.entry_time):
            running_balance += trade.net_pnl
            equity_curve.append(running_balance)
        
        if not equity_curve:
            return Decimal("0"), Decimal("0")
        
        # 计算回撤
        peak = equity_curve[0]
        max_drawdown = Decimal("0")
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # 当前回撤
        current_equity = equity_curve[-1]
        current_drawdown = (peak - current_equity) / peak if peak > 0 else Decimal("0")
        
        return max_drawdown, current_drawdown
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> Decimal:
        """计算夏普比率"""
        if len(returns) < 2:
            return Decimal("0")
        
        excess_returns = [r - risk_free_rate/252 for r in returns]  # 假设252个交易日
        
        if statistics.stdev(excess_returns) == 0:
            return Decimal("0")
        
        sharpe = statistics.mean(excess_returns) / statistics.stdev(excess_returns) * (252 ** 0.5)
        return Decimal(str(round(sharpe, 4)))
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> Decimal:
        """计算索提诺比率"""
        if len(returns) < 2:
            return Decimal("0")
        
        excess_returns = [r - risk_free_rate/252 for r in returns]
        negative_returns = [r for r in excess_returns if r < 0]
        
        if not negative_returns:
            return Decimal("0")
        
        downside_deviation = (sum(r**2 for r in negative_returns) / len(negative_returns)) ** 0.5
        
        if downside_deviation == 0:
            return Decimal("0")
        
        sortino = statistics.mean(excess_returns) / downside_deviation * (252 ** 0.5)
        return Decimal(str(round(sortino, 4)))
    
    def _calculate_var(self, returns: List[float], confidence_level: float) -> Decimal:
        """计算风险价值VaR"""
        if len(returns) < 10:
            return Decimal("0")
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]
        
        return Decimal(str(round(abs(var), 4)))
    
    def _calculate_streaks(self, trades: List[TradeRecord]) -> Tuple[int, int]:
        """计算最大连续盈利和亏损次数"""
        if not trades:
            return 0, 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in sorted(trades, key=lambda x: x.entry_time):
            if trade.net_pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif trade.net_pnl < 0:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak
    
    def _clear_cache(self):
        """清除缓存"""
        self.metrics_cache.clear()
        self.cache_timestamp.clear()
    
    def compare_to_benchmark(self, benchmark_returns: List[Decimal]) -> Dict[str, Any]:
        """与基准比较"""
        if not self.trade_records or not benchmark_returns:
            return {}
        
        portfolio_metrics = self.calculate_performance_metrics()
        
        # 计算基准指标
        benchmark_total_return = sum(benchmark_returns)
        benchmark_volatility = Decimal(str(statistics.stdev([float(r) for r in benchmark_returns]))) if len(benchmark_returns) > 1 else Decimal("0")
        
        # 计算贝塔系数
        portfolio_returns = [t.get_return_rate() for t in self.trade_records]
        if len(portfolio_returns) == len(benchmark_returns) and len(portfolio_returns) > 1:
            portfolio_returns_float = [float(r) for r in portfolio_returns]
            benchmark_returns_float = [float(r) for r in benchmark_returns]
            
            covariance = np.cov(portfolio_returns_float, benchmark_returns_float)[0][1]
            benchmark_variance = np.var(benchmark_returns_float)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        else:
            beta = 0
        
        # 计算阿尔法
        expected_return = benchmark_total_return * Decimal(str(beta))
        alpha = portfolio_metrics.total_return - expected_return
        
        # 计算信息比率
        tracking_error = Decimal("0")
        if len(portfolio_returns) == len(benchmark_returns) and len(portfolio_returns) > 1:
            excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
            tracking_error = Decimal(str(statistics.stdev([float(r) for r in excess_returns])))
        
        information_ratio = alpha / tracking_error if tracking_error > 0 else Decimal("0")
        
        return {
            "portfolio_return": float(portfolio_metrics.total_return),
            "benchmark_return": float(benchmark_total_return),
            "alpha": float(alpha),
            "beta": beta,
            "information_ratio": float(information_ratio),
            "tracking_error": float(tracking_error),
            "portfolio_sharpe": float(portfolio_metrics.sharpe_ratio),
            "benchmark_volatility": float(benchmark_volatility)
        }
    
    def generate_performance_report(self, 
                                  period: PerformancePeriod = PerformancePeriod.ALL_TIME,
                                  include_trades: bool = True,
                                  include_positions: bool = True) -> Dict[str, Any]:
        """生成性能报告"""
        
        metrics = self.calculate_performance_metrics(period)
        
        report = {
            "report_timestamp": datetime.utcnow().isoformat(),
            "period": period.value,
            "initial_balance": float(self.initial_balance),
            "current_balance": float(self.current_balance),
            "metrics": asdict(metrics)
        }
        
        # 转换Decimal为float
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(v) for v in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        report = convert_decimals(report)
        
        # 添加交易记录
        if include_trades:
            trades_data = []
            for trade in self._filter_trades(period):
                trade_data = asdict(trade)
                trade_data = convert_decimals(trade_data)
                trades_data.append(trade_data)
            report["trades"] = trades_data
        
        # 添加持仓信息
        if include_positions:
            positions_data = []
            for symbol, position in self.position_records.items():
                position_data = asdict(position)
                position_data = convert_decimals(position_data)
                positions_data.append(position_data)
            report["positions"] = positions_data
        
        # 添加策略分析
        strategy_analysis = {}
        for strategy_name in self.strategy_performance.keys():
            strategy_metrics = self.calculate_performance_metrics(period, strategy_name=strategy_name)
            strategy_analysis[strategy_name] = convert_decimals(asdict(strategy_metrics))
        report["strategy_analysis"] = strategy_analysis
        
        # 添加交易对分析
        symbol_analysis = {}
        for symbol in self.symbol_performance.keys():
            symbol_metrics = self.calculate_performance_metrics(period, symbol=symbol)
            symbol_analysis[symbol] = convert_decimals(asdict(symbol_metrics))
        report["symbol_analysis"] = symbol_analysis
        
        return report
    
    def export_trades_to_csv(self, filename: str, period: PerformancePeriod = PerformancePeriod.ALL_TIME):
        """导出交易记录到CSV"""
        import csv
        
        trades = self._filter_trades(period)
        
        if not trades:
            self.log_warning("No trades to export")
            return
        
        fieldnames = [
            'trade_id', 'order_id', 'client_order_id', 'symbol', 'side', 'quantity', 
            'entry_price', 'exit_price', 'entry_time', 'exit_time', 'entry_commission', 
            'exit_commission', 'slippage_cost', 'realized_pnl', 'gross_pnl', 'net_pnl', 
            'strategy_name', 'trade_reason', 'market_condition', 'execution_delay_ms', 'is_maker'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in trades:
                row = asdict(trade)
                # 转换Decimal为字符串
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        row[key] = str(value)
                    elif isinstance(value, datetime):
                        row[key] = value.isoformat()
                writer.writerow(row)
        
        self.log_info(f"Exported {len(trades)} trades to {filename}")
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        current_equity = self.current_balance
        for position in self.position_records.values():
            current_equity += position.unrealized_pnl
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.position_records.values())
        
        return {
            "current_balance": float(self.current_balance),
            "current_equity": float(current_equity),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "total_return": float((current_equity - self.initial_balance) / self.initial_balance) if self.initial_balance > 0 else 0,
            "active_positions": len(self.position_records),
            "total_trades": len(self.trade_records),
            "last_trade_time": self.trade_records[-1].entry_time.isoformat() if self.trade_records else None
        }