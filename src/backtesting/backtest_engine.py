import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
import copy

from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData
from src.trading.enhanced_paper_trading_engine import EnhancedPaperTradingEngine
from src.trading.performance_analytics import PerformanceTracker, TradeRecord
from src.utils.logger import LoggerMixin


class BacktestMode(Enum):
    """回测模式"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    PARAMETER_OPTIMIZATION = "parameter_optimization"


@dataclass
class BacktestEvent:
    """回测事件"""
    timestamp: datetime
    event_type: str
    data: Any
    priority: int = 0  # 事件优先级，数字越小优先级越高
    
    def __lt__(self, other):
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@dataclass 
class BacktestConfig:
    """回测配置"""
    start_date: datetime
    end_date: datetime
    initial_balance: Decimal = Decimal("100000")
    
    # 策略配置
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 数据配置
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1m"])
    
    # 交易配置
    commission_rate: Decimal = Decimal("0.001")
    slippage_model: str = "fixed"  # "fixed", "linear", "market_impact"
    slippage_bps: Decimal = Decimal("1.0")  # 基点
    
    # 执行配置
    mode: BacktestMode = BacktestMode.SINGLE_STRATEGY
    parallel_workers: int = 1
    
    # 输出配置
    save_trade_details: bool = True
    save_daily_stats: bool = True
    generate_plots: bool = False


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # 绩效指标
    performance_metrics: Dict[str, Any]
    strategy_results: Dict[str, Dict[str, Any]]
    
    # 详细数据
    trade_records: List[TradeRecord]
    daily_stats: List[Dict[str, Any]]
    equity_curve: List[Tuple[datetime, Decimal]]
    
    # 执行统计
    total_events: int = 0
    processed_events: int = 0
    execution_errors: List[str] = field(default_factory=list)


class HistoricalDataProvider:
    """历史数据提供者"""
    
    def __init__(self):
        self.data_cache: Dict[str, List[MarketData]] = {}
        
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime, 
                       timeframe: str = "1m") -> List[MarketData]:
        """加载历史数据"""
        # 这里应该从DuckDB或其他数据源加载真实的历史数据
        # 为了演示，我们生成一些模拟数据
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
            
        data = self._generate_sample_data(symbol, start_date, end_date, timeframe)
        self.data_cache[cache_key] = data
        return data
    
    def _generate_sample_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime, timeframe: str) -> List[MarketData]:
        """生成样本数据（实际应从数据库读取）"""
        data = []
        current_time = start_date
        base_price = Decimal("50000.0")  # BTC基准价格
        
        # 简单的随机游走模型
        import random
        
        while current_time <= end_date:
            # 价格变动
            change_pct = (random.random() - 0.5) * 0.02  # ±1%变动
            base_price *= (1 + Decimal(str(change_pct)))
            
            volume = Decimal(str(random.uniform(100, 1000)))
            spread = base_price * Decimal("0.0002")  # 0.02%价差
            
            market_data = MarketData(
                symbol=symbol,
                price=float(base_price),
                volume=float(volume),
                bid=float(base_price - spread/2),
                ask=float(base_price + spread/2),
                bid_volume=float(volume * Decimal("0.6")),
                ask_volume=float(volume * Decimal("0.4")),
                timestamp=current_time
            )
            
            data.append(market_data)
            
            # 根据时间框架增加时间
            if timeframe == "1m":
                current_time += timedelta(minutes=1)
            elif timeframe == "5m":
                current_time += timedelta(minutes=5)
            elif timeframe == "1h":
                current_time += timedelta(hours=1)
            else:
                current_time += timedelta(minutes=1)
                
        return data


class BacktestEngine(LoggerMixin):
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_provider = HistoricalDataProvider()
        self.event_queue: List[BacktestEvent] = []
        self.current_time: Optional[datetime] = None
        
        # 交易引擎实例（每个策略一个）
        self.trading_engines: Dict[str, EnhancedPaperTradingEngine] = {}
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        
        # 策略函数注册表
        self.strategy_functions: Dict[str, Callable] = {}
        
        # 历史数据缓存
        self.market_data_cache: Dict[str, List[MarketData]] = {}
        
        # 结果收集
        self.results: Dict[str, BacktestResult] = {}
        
    def register_strategy(self, name: str, strategy_function: Callable):
        """注册策略函数"""
        self.strategy_functions[name] = strategy_function
        self.log_info(f"Strategy registered: {name}")
    
    async def initialize(self):
        """初始化回测引擎"""
        self.log_info("Initializing backtest engine...")
        
        # 为每个策略创建交易引擎
        for strategy_name in self.config.strategy_configs.keys():
            engine = EnhancedPaperTradingEngine(
                account_id=f"backtest_{strategy_name}",
                initial_balance=self.config.initial_balance
            )
            await engine.initialize()
            
            self.trading_engines[strategy_name] = engine
            self.performance_trackers[strategy_name] = PerformanceTracker(
                initial_balance=self.config.initial_balance
            )
        
        # 加载历史数据
        await self._load_historical_data()
        
        # 初始化事件队列
        await self._initialize_event_queue()
        
        self.log_info("Backtest engine initialized successfully")
    
    async def _load_historical_data(self):
        """加载历史数据"""
        self.log_info("Loading historical data...")
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                data = await self.data_provider.load_data(
                    symbol, self.config.start_date, self.config.end_date, timeframe
                )
                
                cache_key = f"{symbol}_{timeframe}"
                self.market_data_cache[cache_key] = data
                
                self.log_info(f"Loaded {len(data)} data points for {symbol} {timeframe}")
    
    async def _initialize_event_queue(self):
        """初始化事件队列"""
        self.log_info("Initializing event queue...")
        
        # 将所有市场数据转换为事件
        for cache_key, data_list in self.market_data_cache.items():
            for market_data in data_list:
                event = BacktestEvent(
                    timestamp=market_data.timestamp,
                    event_type="market_data",
                    data=market_data,
                    priority=1  # 市场数据事件优先级较低
                )
                heapq.heappush(self.event_queue, event)
        
        self.log_info(f"Event queue initialized with {len(self.event_queue)} events")
    
    async def run_backtest(self, strategy_name: Optional[str] = None) -> Dict[str, BacktestResult]:
        """运行回测"""
        start_time = datetime.utcnow()
        self.log_info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        if self.config.mode == BacktestMode.SINGLE_STRATEGY:
            result = await self._run_single_strategy(strategy_name)
            self.results[strategy_name or "default"] = result
            
        elif self.config.mode == BacktestMode.MULTI_STRATEGY:
            result = await self._run_multi_strategy()
            
        elif self.config.mode == BacktestMode.PARAMETER_OPTIMIZATION:
            result = await self._run_parameter_optimization(strategy_name)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        self.log_info(f"Backtest completed in {duration:.2f} seconds")
        return self.results
    
    async def _run_single_strategy(self, strategy_name: str) -> BacktestResult:
        """运行单策略回测"""
        if strategy_name not in self.strategy_functions:
            raise ValueError(f"Strategy '{strategy_name}' not registered")
        
        strategy_func = self.strategy_functions[strategy_name]
        engine = self.trading_engines[strategy_name]
        tracker = self.performance_trackers[strategy_name]
        
        start_time = datetime.utcnow()
        processed_events = 0
        execution_errors = []
        
        # 事件循环
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp
            
            try:
                if event.event_type == "market_data":
                    # 更新市场数据
                    market_data = event.data
                    await engine.update_market_data(market_data)
                    
                    # 调用策略函数
                    signals = await self._call_strategy_function(
                        strategy_func, market_data, engine, self.config.strategy_configs[strategy_name]
                    )
                    
                    # 处理策略信号
                    if signals:
                        await self._process_strategy_signals(signals, engine, tracker)
                
                processed_events += 1
                
                # 每1000个事件记录一次进度
                if processed_events % 1000 == 0:
                    self.log_debug(f"Processed {processed_events} events")
                    
            except Exception as e:
                error_msg = f"Error processing event at {event.timestamp}: {str(e)}"
                execution_errors.append(error_msg)
                self.log_error(error_msg)
        
        end_time = datetime.utcnow()
        
        # 收集结果
        performance_metrics = tracker.calculate_performance_metrics()
        trade_records = tracker.trade_records.copy()
        equity_curve = tracker.balance_history.copy()
        
        result = BacktestResult(
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            performance_metrics=self._serialize_metrics(performance_metrics),
            strategy_results={strategy_name: self._serialize_metrics(performance_metrics)},
            trade_records=trade_records,
            daily_stats=[],  # TODO: 实现日统计
            equity_curve=equity_curve,
            total_events=len(self.event_queue) + processed_events,
            processed_events=processed_events,
            execution_errors=execution_errors
        )
        
        return result
    
    async def _run_multi_strategy(self) -> Dict[str, BacktestResult]:
        """运行多策略回测"""
        self.log_info("Running multi-strategy backtest...")
        
        # 为了简化，我们串行运行多个策略
        # 实际生产中可以考虑并行执行
        results = {}
        
        for strategy_name in self.config.strategy_configs.keys():
            self.log_info(f"Running strategy: {strategy_name}")
            
            # 重置事件队列
            await self._initialize_event_queue()
            
            result = await self._run_single_strategy(strategy_name)
            results[strategy_name] = result
            
        return results
    
    async def _run_parameter_optimization(self, strategy_name: str) -> Dict[str, BacktestResult]:
        """运行参数优化回测"""
        self.log_info(f"Running parameter optimization for {strategy_name}...")
        
        # 这里应该实现参数网格搜索或贝叶斯优化
        # 为了演示，我们只运行一次
        result = await self._run_single_strategy(strategy_name)
        return {f"{strategy_name}_optimized": result}
    
    async def _call_strategy_function(self, strategy_func: Callable, market_data: MarketData,
                                    engine: EnhancedPaperTradingEngine, 
                                    strategy_config: Dict[str, Any]) -> List[Order]:
        """调用策略函数"""
        try:
            # 策略函数应该返回订单列表
            context = {
                "market_data": market_data,
                "engine": engine,
                "config": strategy_config,
                "current_time": self.current_time
            }
            
            # 支持同步和异步策略函数
            if asyncio.iscoroutinefunction(strategy_func):
                signals = await strategy_func(context)
            else:
                signals = strategy_func(context)
                
            return signals if signals else []
            
        except Exception as e:
            self.log_error(f"Error in strategy function: {str(e)}")
            return []
    
    async def _process_strategy_signals(self, signals: List[Order], 
                                      engine: EnhancedPaperTradingEngine,
                                      tracker: PerformanceTracker):
        """处理策略信号"""
        for order in signals:
            try:
                # 执行订单
                result = await engine.execute_order(order)
                
                # 如果订单成交，记录交易
                if result.get("status") == "FILLED":
                    trade_record = self._create_trade_record(order, result, engine)
                    if trade_record:
                        tracker.record_trade(trade_record)
                        
            except Exception as e:
                self.log_error(f"Error executing order: {str(e)}")
    
    def _create_trade_record(self, order: Order, execution_result: Dict[str, Any],
                           engine: EnhancedPaperTradingEngine) -> Optional[TradeRecord]:
        """创建交易记录"""
        try:
            # 检查是否是平仓交易
            if order.symbol in engine.positions:
                position = engine.positions[order.symbol]
                
                # 如果是完全平仓或减仓
                if ((order.side == OrderSide.SELL and position.side == "LONG") or
                    (order.side == OrderSide.BUY and position.side == "SHORT")):
                    
                    trade_record = TradeRecord(
                        trade_id=f"bt_{order.client_order_id}",
                        order_id=order.client_order_id,
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=Decimal(str(order.executed_qty)),
                        entry_price=position.entry_price,
                        exit_price=Decimal(str(order.avg_price)),
                        entry_time=position.open_time if hasattr(position, 'open_time') else self.current_time,
                        exit_time=self.current_time
                    )
                    
                    trade_record.calculate_pnl()
                    return trade_record
                    
        except Exception as e:
            self.log_error(f"Error creating trade record: {str(e)}")
            
        return None
    
    def _serialize_metrics(self, metrics) -> Dict[str, Any]:
        """序列化性能指标"""
        from dataclasses import asdict
        
        result = asdict(metrics)
        
        # 转换Decimal为float
        def convert_decimals(obj):
            if isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(v) for v in obj]
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        return convert_decimals(result)
    
    async def get_backtest_progress(self) -> Dict[str, Any]:
        """获取回测进度"""
        total_events = len(self.event_queue)
        processed_events = 0  # 需要跟踪已处理的事件数
        
        return {
            "total_events": total_events,
            "processed_events": processed_events,
            "progress_percentage": (processed_events / total_events * 100) if total_events > 0 else 0,
            "current_time": self.current_time.isoformat() if self.current_time else None,
            "estimated_completion_time": None  # 可以基于处理速度估算
        }
    
    async def stop_backtest(self):
        """停止回测"""
        self.log_info("Stopping backtest...")
        # 清空事件队列
        self.event_queue.clear()
        self.log_info("Backtest stopped")