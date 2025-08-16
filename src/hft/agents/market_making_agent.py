import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Deque
import numpy as np

from src.core.models import MarketData
from src.hft.hft_engine import HFTEngine
from src.hft.orderbook_manager import OrderBookSnapshot
from src.hft.microstructure_analyzer import MicrostructureSignal, ImbalanceMetrics
from src.hft.execution_engine import ExecutionOrder, OrderType, OrderStatus
from src.utils.logger import LoggerMixin


class QuoteStatus(Enum):
    """报价状态"""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    FILLED = "filled"
    PENDING = "pending"


@dataclass
class MarketMakingQuote:
    """做市报价"""
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    bid_quantity: Decimal
    ask_quantity: Decimal
    spread_bps: float
    mid_price: Decimal
    timestamp: float = field(default_factory=time.time)
    
    # 订单信息
    bid_order_id: Optional[str] = None
    ask_order_id: Optional[str] = None
    bid_status: QuoteStatus = QuoteStatus.PENDING
    ask_status: QuoteStatus = QuoteStatus.PENDING
    
    # 风险指标
    inventory_risk: float = 0.0
    adverse_selection_risk: float = 0.0
    expected_profit_bps: float = 0.0


@dataclass
class MarketMakingConfig:
    """做市配置"""
    # 基础参数
    base_spread_bps: float = 20.0        # 基础价差基点
    min_spread_bps: float = 5.0          # 最小价差
    max_spread_bps: float = 100.0        # 最大价差
    
    # 库存管理
    max_inventory_value: Decimal = field(default_factory=lambda: Decimal("5000"))
    inventory_target: Decimal = field(default_factory=lambda: Decimal("0"))
    skew_factor: float = 0.1             # 库存倾斜因子
    
    # 风险控制
    max_position_pct: float = 0.02       # 最大持仓百分比
    stop_loss_bps: float = 200.0         # 止损基点
    daily_loss_limit: Decimal = field(default_factory=lambda: Decimal("1000"))
    
    # 执行参数
    quote_refresh_interval: float = 0.5   # 报价刷新间隔（秒）
    min_order_value: Decimal = field(default_factory=lambda: Decimal("10"))
    max_order_value: Decimal = field(default_factory=lambda: Decimal("1000"))
    
    # 自适应参数
    volatility_lookback: int = 100        # 波动率回看期
    flow_toxicity_threshold: float = 0.5  # 流毒性阈值
    adverse_selection_decay: float = 0.95 # 逆向选择衰减
    
    # 市场微观结构
    tick_size: Decimal = field(default_factory=lambda: Decimal("0.01"))
    min_size_increment: Decimal = field(default_factory=lambda: Decimal("0.001"))


class MarketMakingAgent(LoggerMixin):
    """做市Agent
    
    实现智能做市策略：
    1. 动态价差调整
    2. 库存风险管理
    3. 逆向选择防护
    4. 自适应参数优化
    """
    
    def __init__(self, hft_engine: HFTEngine, config: Optional[MarketMakingConfig] = None):
        self.hft_engine = hft_engine
        self.config = config or MarketMakingConfig()
        
        # 市场数据历史
        self.price_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.volatility_lookback)
        )
        self.volume_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # 当前报价
        self.active_quotes: Dict[str, MarketMakingQuote] = {}
        self.quote_history: Dict[str, Deque[MarketMakingQuote]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # 库存跟踪
        self.inventory: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.avg_purchase_price: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        
        # 风险指标
        self.daily_pnl: Decimal = Decimal("0")
        self.realized_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.unrealized_pnl: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        
        # 自适应参数
        self.volatility_estimates: Dict[str, float] = defaultdict(float)
        self.adverse_selection_costs: Dict[str, float] = defaultdict(float)
        self.spread_adjustments: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # 执行状态
        self.pending_orders: Dict[str, List[str]] = defaultdict(list)
        self.fill_rates: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        
        # 统计数据
        self.stats = {
            "total_quotes": 0,
            "filled_quotes": 0,
            "fill_rate": 0.0,
            "avg_spread_bps": 0.0,
            "total_volume": Decimal("0"),
            "profit_per_trade": Decimal("0"),
            "inventory_turnover": 0.0
        }
        
        # 运行状态
        self._running = False
        self._quote_task: Optional[asyncio.Task] = None
        self._last_quote_time: Dict[str, float] = defaultdict(float)
        
        # 注册回调
        self.hft_engine.add_signal_callback(self._on_signal)
        self.hft_engine.add_order_callback(self._on_order_executed)
        
    async def start(self, symbols: List[str]):
        """启动做市Agent"""
        if self._running:
            return
            
        self._symbols = symbols
        self._running = True
        self._quote_task = asyncio.create_task(self._quote_loop())
        self.log_info(f"Market Making Agent started for symbols: {symbols}")
        
    async def stop(self):
        """停止做市Agent"""
        self._running = False
        
        # 取消所有报价
        for symbol in list(self.active_quotes.keys()):
            await self._cancel_quotes(symbol)
        
        if self._quote_task:
            self._quote_task.cancel()
            try:
                await self._quote_task
            except asyncio.CancelledError:
                pass
                
        self.log_info("Market Making Agent stopped")
        
    async def update_market_data(self, symbol: str, market_data: MarketData):
        """更新市场数据"""
        # 更新历史数据
        self.price_history[symbol].append(market_data.price)
        self.volume_history[symbol].append(market_data.volume)
        
        # 更新波动率估计
        await self._update_volatility_estimate(symbol)
        
        # 触发报价更新
        await self._update_quotes(symbol)
        
    async def _quote_loop(self):
        """报价循环"""
        while self._running:
            try:
                current_time = time.time()
                
                # 检查需要更新的报价
                for symbol in getattr(self, '_symbols', []):
                    last_quote = self._last_quote_time[symbol]
                    
                    if current_time - last_quote >= self.config.quote_refresh_interval:
                        await self._update_quotes(symbol)
                        self._last_quote_time[symbol] = current_time
                
                # 监控库存风险
                await self._monitor_inventory_risk()
                
                # 更新统计
                await self._update_statistics()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.log_error(f"Error in quote loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_quotes(self, symbol: str):
        """更新报价"""
        try:
            # 获取当前市场状态
            orderbook = self.hft_engine.get_orderbook(symbol)
            if not orderbook or not orderbook.mid_price:
                return
            
            # 获取微观结构信号
            signals = self.hft_engine.get_current_signals(symbol)
            imbalance = self.hft_engine.microstructure_analyzer.get_imbalance_metrics(symbol)
            
            # 计算最优报价
            quote = await self._calculate_optimal_quote(symbol, orderbook, signals, imbalance)
            if not quote:
                return
            
            # 取消现有报价
            await self._cancel_quotes(symbol)
            
            # 下新报价
            await self._place_quotes(quote)
            
        except Exception as e:
            self.log_error(f"Error updating quotes for {symbol}: {e}")
    
    async def _calculate_optimal_quote(self, 
                                     symbol: str,
                                     orderbook: OrderBookSnapshot,
                                     signals: List[MicrostructureSignal],
                                     imbalance: Optional[ImbalanceMetrics]) -> Optional[MarketMakingQuote]:
        """计算最优报价"""
        try:
            mid_price = orderbook.mid_price
            current_spread = orderbook.spread
            
            if not mid_price or not current_spread:
                return None
            
            # 基础价差
            base_spread = float(mid_price) * self.config.base_spread_bps / 10000
            
            # 波动率调整
            volatility = self.volatility_estimates.get(symbol, 0.02)
            volatility_adjustment = 1.0 + volatility * 10  # 波动率越高，价差越大
            
            # 库存倾斜
            inventory_value = self.inventory[symbol] * mid_price
            max_inventory = self.config.max_inventory_value
            inventory_ratio = float(inventory_value / max_inventory) if max_inventory > 0 else 0
            skew = inventory_ratio * self.config.skew_factor
            
            # 流毒性调整
            toxicity_adjustment = 1.0
            for signal in signals:
                if signal.signal_type == "toxicity" and signal.strength > self.config.flow_toxicity_threshold:
                    toxicity_adjustment += signal.strength * 0.5
            
            # 订单簿失衡调整
            imbalance_adjustment = 1.0
            if imbalance:
                # 失衡时调整价差
                imbalance_strength = abs(imbalance.bid_ask_imbalance)
                imbalance_adjustment += imbalance_strength * 0.3
            
            # 逆向选择成本
            adverse_cost = self.adverse_selection_costs.get(symbol, 0)
            adverse_adjustment = 1.0 + adverse_cost
            
            # 综合调整
            total_adjustment = (volatility_adjustment * 
                              toxicity_adjustment * 
                              imbalance_adjustment * 
                              adverse_adjustment)
            
            adjusted_spread = base_spread * total_adjustment
            
            # 确保价差在合理范围内
            min_spread = float(mid_price) * self.config.min_spread_bps / 10000
            max_spread = float(mid_price) * self.config.max_spread_bps / 10000
            adjusted_spread = max(min_spread, min(max_spread, adjusted_spread))
            
            # 计算买卖价
            half_spread = Decimal(str(adjusted_spread / 2))
            
            # 应用库存倾斜
            bid_skew = half_spread * Decimal(str(1 + skew))
            ask_skew = half_spread * Decimal(str(1 - skew))
            
            bid_price = mid_price - bid_skew
            ask_price = mid_price + ask_skew
            
            # 价格对齐到tick size
            tick_size = self.config.tick_size
            bid_price = (bid_price // tick_size) * tick_size
            ask_price = ((ask_price // tick_size) + 1) * tick_size
            
            # 计算报价数量
            bid_quantity, ask_quantity = await self._calculate_quote_sizes(
                symbol, mid_price, inventory_ratio
            )
            
            if bid_quantity <= 0 or ask_quantity <= 0:
                return None
            
            # 创建报价
            quote = MarketMakingQuote(
                symbol=symbol,
                bid_price=bid_price,
                ask_price=ask_price,
                bid_quantity=bid_quantity,
                ask_quantity=ask_quantity,
                spread_bps=float((ask_price - bid_price) / mid_price * 10000),
                mid_price=mid_price,
                inventory_risk=abs(inventory_ratio),
                adverse_selection_risk=adverse_cost,
                expected_profit_bps=float(adjusted_spread / mid_price * 10000)
            )
            
            return quote
            
        except Exception as e:
            self.log_error(f"Error calculating optimal quote: {e}")
            return None
    
    async def _calculate_quote_sizes(self, 
                                   symbol: str, 
                                   mid_price: Decimal, 
                                   inventory_ratio: float) -> Tuple[Decimal, Decimal]:
        """计算报价数量"""
        try:
            # 基础数量
            base_value = self.config.max_order_value
            base_quantity = base_value / mid_price
            
            # 库存调整
            if inventory_ratio > 0:
                # 库存过多，减少买单，增加卖单
                bid_quantity = base_quantity * Decimal(str(1 - abs(inventory_ratio)))
                ask_quantity = base_quantity * Decimal(str(1 + abs(inventory_ratio) * 0.5))
            else:
                # 库存不足，增加买单，减少卖单
                bid_quantity = base_quantity * Decimal(str(1 + abs(inventory_ratio) * 0.5))
                ask_quantity = base_quantity * Decimal(str(1 - abs(inventory_ratio)))
            
            # 确保最小数量
            min_quantity = self.config.min_order_value / mid_price
            bid_quantity = max(min_quantity, bid_quantity)
            ask_quantity = max(min_quantity, ask_quantity)
            
            # 数量对齐
            size_increment = self.config.min_size_increment
            bid_quantity = (bid_quantity // size_increment) * size_increment
            ask_quantity = (ask_quantity // size_increment) * size_increment
            
            return bid_quantity, ask_quantity
            
        except Exception as e:
            self.log_error(f"Error calculating quote sizes: {e}")
            return Decimal("0"), Decimal("0")
    
    async def _place_quotes(self, quote: MarketMakingQuote):
        """下报价单"""
        try:
            # 下买单
            bid_order_id = await self.hft_engine.place_order(
                symbol=quote.symbol,
                side="buy",
                quantity=quote.bid_quantity,
                order_type=OrderType.LIMIT,
                price=quote.bid_price,
                post_only=True  # 只做Maker
            )
            
            # 下卖单
            ask_order_id = await self.hft_engine.place_order(
                symbol=quote.symbol,
                side="sell", 
                quantity=quote.ask_quantity,
                order_type=OrderType.LIMIT,
                price=quote.ask_price,
                post_only=True
            )
            
            if bid_order_id and ask_order_id:
                quote.bid_order_id = bid_order_id
                quote.ask_order_id = ask_order_id
                quote.bid_status = QuoteStatus.ACTIVE
                quote.ask_status = QuoteStatus.ACTIVE
                
                # 记录报价
                self.active_quotes[quote.symbol] = quote
                self.quote_history[quote.symbol].append(quote)
                self.stats["total_quotes"] += 1
                
                # 记录挂单
                self.pending_orders[quote.symbol].extend([bid_order_id, ask_order_id])
                
                self.log_debug(f"Quote placed for {quote.symbol}: "
                             f"bid={quote.bid_price} ask={quote.ask_price} "
                             f"spread={quote.spread_bps:.1f}bps")
            else:
                self.log_warning(f"Failed to place complete quote for {quote.symbol}")
                
        except Exception as e:
            self.log_error(f"Error placing quotes: {e}")
    
    async def _cancel_quotes(self, symbol: str):
        """取消报价"""
        if symbol not in self.active_quotes:
            return
            
        quote = self.active_quotes[symbol]
        
        # 取消买单
        if quote.bid_order_id and quote.bid_status == QuoteStatus.ACTIVE:
            await self.hft_engine.cancel_order(quote.bid_order_id)
            quote.bid_status = QuoteStatus.CANCELLED
            
        # 取消卖单
        if quote.ask_order_id and quote.ask_status == QuoteStatus.ACTIVE:
            await self.hft_engine.cancel_order(quote.ask_order_id)
            quote.ask_status = QuoteStatus.CANCELLED
        
        # 移除活跃报价
        del self.active_quotes[symbol]
    
    async def _update_volatility_estimate(self, symbol: str):
        """更新波动率估计"""
        prices = list(self.price_history[symbol])
        if len(prices) < 20:
            return
            
        # 计算对数收益率
        log_returns = []
        for i in range(1, len(prices)):
            log_return = np.log(prices[i] / prices[i-1])
            log_returns.append(log_return)
        
        if log_returns:
            # 年化波动率
            volatility = np.std(log_returns) * np.sqrt(252 * 24 * 60)  # 假设1分钟数据
            self.volatility_estimates[symbol] = volatility
    
    async def _monitor_inventory_risk(self):
        """监控库存风险"""
        for symbol, inventory in self.inventory.items():
            if inventory == 0:
                continue
                
            # 获取当前价格
            orderbook = self.hft_engine.get_orderbook(symbol)
            if not orderbook or not orderbook.mid_price:
                continue
            
            current_price = orderbook.mid_price
            inventory_value = inventory * current_price
            
            # 检查库存限制
            if abs(inventory_value) > self.config.max_inventory_value:
                self.log_warning(f"Inventory limit exceeded for {symbol}: {inventory_value}")
                # 减少该币种的做市活动
                await self._reduce_market_making(symbol)
            
            # 计算未实现盈亏
            if symbol in self.avg_purchase_price and self.avg_purchase_price[symbol] > 0:
                cost_basis = self.avg_purchase_price[symbol]
                unrealized_pnl = inventory * (current_price - cost_basis)
                self.unrealized_pnl[symbol] = unrealized_pnl
                
                # 检查止损
                if unrealized_pnl < -Decimal(str(float(current_price) * self.config.stop_loss_bps / 10000)):
                    self.log_warning(f"Stop loss triggered for {symbol}: {unrealized_pnl}")
                    await self._emergency_liquidate(symbol)
    
    async def _reduce_market_making(self, symbol: str):
        """减少做市活动"""
        # 取消当前报价
        await self._cancel_quotes(symbol)
        
        # 暂时停止该币种做市
        # 实际实现中可以设置冷却期
        pass
    
    async def _emergency_liquidate(self, symbol: str):
        """紧急清仓"""
        inventory = self.inventory[symbol]
        if inventory == 0:
            return
        
        # 市价单清仓
        side = "sell" if inventory > 0 else "buy"
        quantity = abs(inventory)
        
        order_id = await self.hft_engine.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        if order_id:
            self.log_info(f"Emergency liquidation order placed for {symbol}: {side} {quantity}")
    
    async def _update_statistics(self):
        """更新统计数据"""
        # 成交率
        total_quotes = self.stats["total_quotes"]
        filled_quotes = self.stats["filled_quotes"]
        
        if total_quotes > 0:
            self.stats["fill_rate"] = filled_quotes / total_quotes
        
        # 平均价差
        if self.active_quotes:
            avg_spread = sum(quote.spread_bps for quote in self.active_quotes.values()) / len(self.active_quotes)
            self.stats["avg_spread_bps"] = avg_spread
        
        # 总交易量
        total_volume = sum(abs(inv) for inv in self.inventory.values())
        self.stats["total_volume"] = total_volume
        
        # 总盈亏
        total_pnl = sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values())
        self.daily_pnl = total_pnl
    
    async def _on_signal(self, signal: MicrostructureSignal):
        """处理微观结构信号"""
        # 根据信号调整做市策略
        if signal.signal_type == "toxicity" and signal.strength > 0.8:
            # 高毒性时减小报价数量
            pass
        elif signal.signal_type == "imbalance" and abs(signal.strength) > 0.7:
            # 失衡时调整价差
            symbol = signal.symbol
            if symbol in self.spread_adjustments:
                self.spread_adjustments[symbol] *= (1 + abs(signal.strength) * 0.1)
    
    async def _on_order_executed(self, order: ExecutionOrder):
        """处理订单执行"""
        if order.status != OrderStatus.FILLED:
            return
        
        symbol = order.symbol
        
        # 更新库存
        if order.side == "buy":
            self.inventory[symbol] += order.filled_quantity
            # 更新平均成本
            if symbol in self.avg_purchase_price:
                old_cost = self.avg_purchase_price[symbol] * (self.inventory[symbol] - order.filled_quantity)
                new_cost = order.average_price * order.filled_quantity
                total_quantity = self.inventory[symbol]
                if total_quantity > 0:
                    self.avg_purchase_price[symbol] = (old_cost + new_cost) / total_quantity
            else:
                self.avg_purchase_price[symbol] = order.average_price
        else:
            # 卖出时计算已实现盈亏
            if symbol in self.avg_purchase_price and self.avg_purchase_price[symbol] > 0:
                cost_basis = self.avg_purchase_price[symbol]
                pnl = order.filled_quantity * (order.average_price - cost_basis)
                self.realized_pnl[symbol] += pnl
            
            self.inventory[symbol] -= order.filled_quantity
        
        # 更新统计
        if order.order_id in [quote.bid_order_id for quote in self.active_quotes.values()] or \
           order.order_id in [quote.ask_order_id for quote in self.active_quotes.values()]:
            self.stats["filled_quotes"] += 1
        
        # 从挂单列表移除
        if order.order_id in self.pending_orders[symbol]:
            self.pending_orders[symbol].remove(order.order_id)
        
        # 更新逆向选择成本
        await self._update_adverse_selection_cost(symbol, order)
        
        self.log_debug(f"Order filled: {symbol} {order.side} {order.filled_quantity} @ {order.average_price}")
    
    async def _update_adverse_selection_cost(self, symbol: str, order: ExecutionOrder):
        """更新逆向选择成本"""
        # 检查成交后价格变动，估计逆向选择成本
        # 这里使用简化的计算方法
        orderbook = self.hft_engine.get_orderbook(symbol)
        if not orderbook or not orderbook.mid_price:
            return
        
        current_mid = orderbook.mid_price
        execution_price = order.average_price
        
        if order.side == "buy":
            # 买入后价格下跌表示逆向选择
            adverse_cost = max(0, float((execution_price - current_mid) / current_mid))
        else:
            # 卖出后价格上涨表示逆向选择
            adverse_cost = max(0, float((current_mid - execution_price) / current_mid))
        
        # 指数移动平均更新
        if symbol in self.adverse_selection_costs:
            alpha = 1 - self.config.adverse_selection_decay
            self.adverse_selection_costs[symbol] = (
                alpha * adverse_cost + 
                (1 - alpha) * self.adverse_selection_costs[symbol]
            )
        else:
            self.adverse_selection_costs[symbol] = adverse_cost
    
    def get_active_quotes(self) -> Dict[str, MarketMakingQuote]:
        """获取活跃报价"""
        return self.active_quotes.copy()
    
    def get_inventory(self) -> Dict[str, Decimal]:
        """获取当前库存"""
        return dict(self.inventory)
    
    def get_pnl(self) -> Dict[str, Decimal]:
        """获取盈亏"""
        return {
            "realized": dict(self.realized_pnl),
            "unrealized": dict(self.unrealized_pnl),
            "total": self.daily_pnl
        }
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计数据"""
        return self.stats.copy()
    
    def get_risk_metrics(self) -> Dict[str, any]:
        """获取风险指标"""
        total_inventory_value = sum(
            abs(self.inventory[symbol]) * (
                self.hft_engine.get_orderbook(symbol).mid_price 
                if self.hft_engine.get_orderbook(symbol) 
                else Decimal("0")
            )
            for symbol in self.inventory
        )
        
        return {
            "total_inventory_value": total_inventory_value,
            "max_inventory_limit": self.config.max_inventory_value,
            "inventory_utilization": float(total_inventory_value / self.config.max_inventory_value),
            "daily_pnl": self.daily_pnl,
            "volatility_estimates": dict(self.volatility_estimates),
            "adverse_selection_costs": dict(self.adverse_selection_costs)
        }
    
    def get_status(self) -> Dict[str, any]:
        """获取Agent状态"""
        return {
            "running": self._running,
            "active_quotes": len(self.active_quotes),
            "total_inventory_positions": len([inv for inv in self.inventory.values() if inv != 0]),
            "pending_orders": sum(len(orders) for orders in self.pending_orders.values()),
            "daily_pnl": float(self.daily_pnl),
            "statistics": self.get_statistics(),
            "risk_metrics": self.get_risk_metrics()
        }