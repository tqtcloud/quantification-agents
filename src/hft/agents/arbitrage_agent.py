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
from src.hft.microstructure_analyzer import MicrostructureSignal
from src.hft.execution_engine import ExecutionOrder, OrderType
from src.utils.logger import LoggerMixin


class ArbitrageType(Enum):
    """套利类型"""
    CROSS_EXCHANGE = "cross_exchange"  # 跨交易所套利
    TRIANGULAR = "triangular"          # 三角套利
    FUTURES_SPOT = "futures_spot"      # 期现套利
    CALENDAR_SPREAD = "calendar_spread" # 跨期套利


@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    arbitrage_type: ArbitrageType
    symbols: List[str]
    expected_profit_bps: float
    confidence: float
    entry_prices: Dict[str, Decimal]
    quantities: Dict[str, Decimal]
    directions: Dict[str, str]  # "buy" or "sell"
    timestamp: float = field(default_factory=time.time)
    expiry_time: float = field(default_factory=lambda: time.time() + 5.0)  # 5秒过期
    metadata: Dict = field(default_factory=dict)


@dataclass
class ArbitrageConfig:
    """套利配置"""
    # 盈利阈值
    min_profit_bps: float = 20.0  # 最小盈利基点
    min_confidence: float = 0.7   # 最小置信度
    
    # 风险控制
    max_position_value: Decimal = field(default_factory=lambda: Decimal("10000"))
    max_leverage: float = 3.0
    stop_loss_bps: float = 50.0   # 止损基点
    
    # 执行配置
    execution_timeout: float = 2.0  # 执行超时时间
    max_slippage_bps: float = 30.0  # 最大滑点
    
    # 检测参数
    price_update_window: int = 100   # 价格更新窗口
    correlation_lookback: int = 50   # 相关性回看期
    
    # 三角套利配置
    triangular_symbols: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("BTCUSDT", "ETHUSDT", "ETHBTC"),
        ("ADAUSDT", "BNBUSDT", "ADABNB")
    ])
    
    # 期现套利配置
    futures_spot_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("BTCUSDT", "BTCUSDT_PERP"),
        ("ETHUSDT", "ETHUSDT_PERP")
    ])


class ArbitrageAgent(LoggerMixin):
    """套利Agent
    
    实现多种套利策略：
    1. 跨交易所套利
    2. 三角套利
    3. 期现套利
    4. 跨期套利
    """
    
    def __init__(self, hft_engine: HFTEngine, config: Optional[ArbitrageConfig] = None):
        self.hft_engine = hft_engine
        self.config = config or ArbitrageConfig()
        
        # 价格历史
        self.price_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.price_update_window)
        )
        
        # 套利机会跟踪
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.opportunity_history: Deque[ArbitrageOpportunity] = deque(maxlen=1000)
        
        # 执行状态
        self.positions: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
        self.pending_orders: Dict[str, List[str]] = defaultdict(list)  # symbol -> order_ids
        
        # 统计数据
        self.stats = {
            "opportunities_found": 0,
            "opportunities_executed": 0,
            "total_profit_bps": 0.0,
            "win_rate": 0.0,
            "avg_holding_time": 0.0
        }
        
        # 运行状态
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 注册回调
        self.hft_engine.add_signal_callback(self._on_signal)
        self.hft_engine.add_order_callback(self._on_order_executed)
        
    async def start(self):
        """启动套利Agent"""
        if self._running:
            return
            
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.log_info("Arbitrage Agent started")
        
    async def stop(self):
        """停止套利Agent"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.log_info("Arbitrage Agent stopped")
        
    async def update_market_data(self, symbol: str, market_data: MarketData):
        """更新市场数据"""
        # 更新价格历史
        self.price_history[symbol].append(market_data.price)
        
        # 检测套利机会
        await self._detect_arbitrage_opportunities(symbol, market_data)
        
    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 清理过期机会
                await self._cleanup_expired_opportunities()
                
                # 监控执行状态
                await self._monitor_executions()
                
                # 更新统计
                await self._update_statistics()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.log_error(f"Error in arbitrage monitor loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _detect_arbitrage_opportunities(self, symbol: str, market_data: MarketData):
        """检测套利机会"""
        try:
            # 三角套利检测
            await self._detect_triangular_arbitrage(symbol, market_data)
            
            # 期现套利检测
            await self._detect_futures_spot_arbitrage(symbol, market_data)
            
            # 统计套利检测
            await self._detect_statistical_arbitrage(symbol, market_data)
            
        except Exception as e:
            self.log_error(f"Error detecting arbitrage opportunities: {e}")
    
    async def _detect_triangular_arbitrage(self, symbol: str, market_data: MarketData):
        """检测三角套利机会"""
        for base_quote, alt_quote, cross in self.config.triangular_symbols:
            if symbol not in [base_quote, alt_quote, cross]:
                continue
                
            try:
                # 获取三个交易对的价格
                base_orderbook = self.hft_engine.get_orderbook(base_quote)
                alt_orderbook = self.hft_engine.get_orderbook(alt_quote)
                cross_orderbook = self.hft_engine.get_orderbook(cross)
                
                if not all([base_orderbook, alt_orderbook, cross_orderbook]):
                    continue
                
                # 计算理论价格
                base_mid = base_orderbook.mid_price
                alt_mid = alt_orderbook.mid_price
                cross_mid = cross_orderbook.mid_price
                
                if not all([base_mid, alt_mid, cross_mid]):
                    continue
                
                # 三角套利逻辑：base/quote * cross/alt 应该等于 alt/quote
                theoretical_alt_price = base_mid * cross_mid
                actual_alt_price = alt_mid
                
                price_diff_bps = float(abs(theoretical_alt_price - actual_alt_price) / actual_alt_price * 10000)
                
                if price_diff_bps > self.config.min_profit_bps:
                    # 确定套利方向
                    if theoretical_alt_price > actual_alt_price:
                        # 买入alt，卖出base和cross
                        directions = {
                            alt_quote: "buy",
                            base_quote: "sell", 
                            cross: "sell"
                        }
                    else:
                        # 卖出alt，买入base和cross
                        directions = {
                            alt_quote: "sell",
                            base_quote: "buy",
                            cross: "buy"
                        }
                    
                    # 计算最优执行数量
                    quantities = await self._calculate_triangular_quantities(
                        base_orderbook, alt_orderbook, cross_orderbook, directions
                    )
                    
                    if quantities:
                        opportunity = ArbitrageOpportunity(
                            arbitrage_type=ArbitrageType.TRIANGULAR,
                            symbols=[base_quote, alt_quote, cross],
                            expected_profit_bps=price_diff_bps,
                            confidence=0.8,  # 基于价格稳定性计算
                            entry_prices={
                                base_quote: base_mid,
                                alt_quote: alt_mid,
                                cross: cross_mid
                            },
                            quantities=quantities,
                            directions=directions,
                            metadata={
                                "theoretical_price": float(theoretical_alt_price),
                                "actual_price": float(actual_alt_price)
                            }
                        )
                        
                        await self._evaluate_opportunity(opportunity)
                        
            except Exception as e:
                self.log_error(f"Error in triangular arbitrage detection: {e}")
    
    async def _detect_futures_spot_arbitrage(self, symbol: str, market_data: MarketData):
        """检测期现套利机会"""
        for spot_symbol, futures_symbol in self.config.futures_spot_pairs:
            if symbol not in [spot_symbol, futures_symbol]:
                continue
                
            try:
                spot_orderbook = self.hft_engine.get_orderbook(spot_symbol)
                futures_orderbook = self.hft_engine.get_orderbook(futures_symbol)
                
                if not all([spot_orderbook, futures_orderbook]):
                    continue
                
                spot_mid = spot_orderbook.mid_price
                futures_mid = futures_orderbook.mid_price
                
                if not all([spot_mid, futures_mid]):
                    continue
                
                # 计算价差
                spread_bps = float(abs(futures_mid - spot_mid) / spot_mid * 10000)
                
                if spread_bps > self.config.min_profit_bps:
                    # 确定套利方向
                    if futures_mid > spot_mid:
                        # 卖出期货，买入现货
                        directions = {
                            spot_symbol: "buy",
                            futures_symbol: "sell"
                        }
                    else:
                        # 买入期货，卖出现货
                        directions = {
                            spot_symbol: "sell", 
                            futures_symbol: "buy"
                        }
                    
                    # 计算执行数量
                    base_quantity = min(
                        self.config.max_position_value / spot_mid,
                        Decimal("1.0")  # 最大1个单位
                    )
                    
                    quantities = {
                        spot_symbol: base_quantity,
                        futures_symbol: base_quantity
                    }
                    
                    opportunity = ArbitrageOpportunity(
                        arbitrage_type=ArbitrageType.FUTURES_SPOT,
                        symbols=[spot_symbol, futures_symbol],
                        expected_profit_bps=spread_bps,
                        confidence=0.9,
                        entry_prices={
                            spot_symbol: spot_mid,
                            futures_symbol: futures_mid
                        },
                        quantities=quantities,
                        directions=directions,
                        metadata={
                            "spread": float(futures_mid - spot_mid),
                            "spot_price": float(spot_mid),
                            "futures_price": float(futures_mid)
                        }
                    )
                    
                    await self._evaluate_opportunity(opportunity)
                    
            except Exception as e:
                self.log_error(f"Error in futures-spot arbitrage detection: {e}")
    
    async def _detect_statistical_arbitrage(self, symbol: str, market_data: MarketData):
        """检测统计套利机会"""
        # 寻找相关性高的交易对
        correlations = await self._calculate_correlations(symbol)
        
        for corr_symbol, correlation in correlations.items():
            if abs(correlation) < 0.7:  # 相关性阈值
                continue
                
            try:
                # 获取价格序列
                symbol_prices = list(self.price_history[symbol])
                corr_prices = list(self.price_history[corr_symbol])
                
                if len(symbol_prices) < 20 or len(corr_prices) < 20:
                    continue
                
                # 计算价格比率
                ratios = [p1/p2 for p1, p2 in zip(symbol_prices[-20:], corr_prices[-20:])]
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                
                if std_ratio == 0:
                    continue
                
                current_ratio = symbol_prices[-1] / corr_prices[-1]
                z_score = (current_ratio - mean_ratio) / std_ratio
                
                # 检测统计异常
                if abs(z_score) > 2.0:  # 2个标准差
                    expected_profit_bps = abs(z_score) * std_ratio / mean_ratio * 10000
                    
                    if expected_profit_bps > self.config.min_profit_bps:
                        # 确定交易方向
                        if z_score > 0:
                            # symbol相对被高估，做空spread
                            directions = {symbol: "sell", corr_symbol: "buy"}
                        else:
                            # symbol相对被低估，做多spread
                            directions = {symbol: "buy", corr_symbol: "sell"}
                        
                        # 计算执行数量
                        base_value = self.config.max_position_value / 2
                        quantities = {
                            symbol: base_value / Decimal(str(symbol_prices[-1])),
                            corr_symbol: base_value / Decimal(str(corr_prices[-1]))
                        }
                        
                        opportunity = ArbitrageOpportunity(
                            arbitrage_type=ArbitrageType.CROSS_EXCHANGE,  # 统计套利归类为跨交易所
                            symbols=[symbol, corr_symbol],
                            expected_profit_bps=expected_profit_bps,
                            confidence=min(1.0, abs(z_score) / 3.0),
                            entry_prices={
                                symbol: Decimal(str(symbol_prices[-1])),
                                corr_symbol: Decimal(str(corr_prices[-1]))
                            },
                            quantities=quantities,
                            directions=directions,
                            metadata={
                                "z_score": z_score,
                                "correlation": correlation,
                                "mean_ratio": mean_ratio,
                                "std_ratio": std_ratio
                            }
                        )
                        
                        await self._evaluate_opportunity(opportunity)
                        
            except Exception as e:
                self.log_error(f"Error in statistical arbitrage detection: {e}")
    
    async def _calculate_triangular_quantities(self, 
                                             base_orderbook: OrderBookSnapshot,
                                             alt_orderbook: OrderBookSnapshot, 
                                             cross_orderbook: OrderBookSnapshot,
                                             directions: Dict[str, str]) -> Optional[Dict[str, Decimal]]:
        """计算三角套利的最优执行数量"""
        try:
            # 获取可用流动性
            base_liquidity = self._get_available_liquidity(base_orderbook, directions[base_orderbook.symbol])
            alt_liquidity = self._get_available_liquidity(alt_orderbook, directions[alt_orderbook.symbol])
            cross_liquidity = self._get_available_liquidity(cross_orderbook, directions[cross_orderbook.symbol])
            
            # 计算约束数量
            max_base_qty = min(base_liquidity, self.config.max_position_value / base_orderbook.mid_price)
            max_alt_qty = min(alt_liquidity, self.config.max_position_value / alt_orderbook.mid_price) 
            max_cross_qty = min(cross_liquidity, self.config.max_position_value / cross_orderbook.mid_price)
            
            # 三角套利数量关系约束
            # 假设 base/USDT * cross/alt = alt/USDT
            base_qty = min(max_base_qty, max_alt_qty / cross_orderbook.mid_price, max_cross_qty)
            
            return {
                base_orderbook.symbol: base_qty,
                alt_orderbook.symbol: base_qty * cross_orderbook.mid_price,
                cross_orderbook.symbol: base_qty
            }
            
        except Exception as e:
            self.log_error(f"Error calculating triangular quantities: {e}")
            return None
    
    def _get_available_liquidity(self, orderbook: OrderBookSnapshot, direction: str) -> Decimal:
        """获取可用流动性"""
        if direction == "buy":
            levels = orderbook.asks[:5]  # 前5档
        else:
            levels = orderbook.bids[:5]
            
        return sum(level.size for level in levels)
    
    async def _calculate_correlations(self, symbol: str) -> Dict[str, float]:
        """计算价格相关性"""
        correlations = {}
        
        if len(self.price_history[symbol]) < self.config.correlation_lookback:
            return correlations
        
        symbol_prices = np.array(list(self.price_history[symbol])[-self.config.correlation_lookback:])
        
        for other_symbol, other_history in self.price_history.items():
            if other_symbol == symbol or len(other_history) < self.config.correlation_lookback:
                continue
                
            other_prices = np.array(list(other_history)[-self.config.correlation_lookback:])
            
            try:
                correlation = np.corrcoef(symbol_prices, other_prices)[0, 1]
                if not np.isnan(correlation):
                    correlations[other_symbol] = correlation
            except:
                continue
                
        return correlations
    
    async def _evaluate_opportunity(self, opportunity: ArbitrageOpportunity):
        """评估套利机会"""
        # 检查是否已存在类似机会
        opp_key = f"{opportunity.arbitrage_type.value}_{'-'.join(sorted(opportunity.symbols))}"
        
        if opp_key in self.active_opportunities:
            return
        
        # 检查置信度
        if opportunity.confidence < self.config.min_confidence:
            return
        
        # 检查预期盈利
        if opportunity.expected_profit_bps < self.config.min_profit_bps:
            return
        
        # 风险检查
        if not await self._risk_check_opportunity(opportunity):
            return
        
        # 添加到活跃机会
        self.active_opportunities[opp_key] = opportunity
        self.opportunity_history.append(opportunity)
        self.stats["opportunities_found"] += 1
        
        self.log_info(f"Arbitrage opportunity found: {opportunity.arbitrage_type.value} "
                     f"profit={opportunity.expected_profit_bps:.1f}bps symbols={opportunity.symbols}")
        
        # 执行套利
        await self._execute_arbitrage(opportunity)
    
    async def _risk_check_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """套利机会风险检查"""
        try:
            # 检查总持仓价值
            total_value = sum(
                opportunity.quantities[symbol] * opportunity.entry_prices[symbol] 
                for symbol in opportunity.symbols
            )
            
            if total_value > self.config.max_position_value:
                return False
            
            # 检查单个币种持仓
            for symbol in opportunity.symbols:
                current_position = abs(self.positions[symbol])
                new_position = abs(opportunity.quantities[symbol])
                
                if current_position + new_position > self.config.max_position_value / Decimal("10"):
                    return False
            
            # 检查挂单数量
            total_pending = sum(len(orders) for orders in self.pending_orders.values())
            if total_pending > 20:  # 最多20个挂单
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Error in opportunity risk check: {e}")
            return False
    
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity):
        """执行套利"""
        try:
            order_ids = []
            
            # 同时下所有订单以减少延迟
            for symbol in opportunity.symbols:
                direction = opportunity.directions[symbol]
                quantity = opportunity.quantities[symbol]
                
                # 使用市价单快速执行
                order_id = await self.hft_engine.place_order(
                    symbol=symbol,
                    side=direction,
                    quantity=quantity,
                    order_type=OrderType.MARKET
                )
                
                if order_id:
                    order_ids.append(order_id)
                    self.pending_orders[symbol].append(order_id)
                else:
                    self.log_error(f"Failed to place arbitrage order for {symbol}")
                    # 取消已下的订单
                    for prev_order_id in order_ids:
                        await self.hft_engine.cancel_order(prev_order_id)
                    return
            
            if len(order_ids) == len(opportunity.symbols):
                self.stats["opportunities_executed"] += 1
                self.log_info(f"Arbitrage executed: {len(order_ids)} orders placed")
            
        except Exception as e:
            self.log_error(f"Error executing arbitrage: {e}")
    
    async def _cleanup_expired_opportunities(self):
        """清理过期机会"""
        current_time = time.time()
        expired_keys = []
        
        for key, opportunity in self.active_opportunities.items():
            if current_time > opportunity.expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_opportunities[key]
    
    async def _monitor_executions(self):
        """监控执行状态"""
        # 检查挂单状态，清理已完成的订单
        for symbol, order_ids in list(self.pending_orders.items()):
            completed_orders = []
            
            for order_id in order_ids:
                order = self.hft_engine.execution_engine.get_order(order_id)
                if order and order.status.value in ["filled", "cancelled", "rejected"]:
                    completed_orders.append(order_id)
                    
                    if order.status.value == "filled":
                        # 更新持仓
                        if order.side == "buy":
                            self.positions[symbol] += order.filled_quantity
                        else:
                            self.positions[symbol] -= order.filled_quantity
            
            # 移除已完成的订单
            for order_id in completed_orders:
                order_ids.remove(order_id)
            
            if not order_ids:
                del self.pending_orders[symbol]
    
    async def _update_statistics(self):
        """更新统计数据"""
        if self.stats["opportunities_executed"] > 0:
            # 计算胜率（简化计算）
            winning_trades = sum(1 for opp in self.opportunity_history 
                               if opp.expected_profit_bps > 0)
            self.stats["win_rate"] = winning_trades / len(self.opportunity_history)
    
    async def _on_signal(self, signal: MicrostructureSignal):
        """处理微观结构信号"""
        # 基于信号调整套利参数
        if signal.signal_type == "toxicity" and signal.strength > 0.8:
            # 高毒性时减少套利活动
            pass
        elif signal.signal_type == "volume_spike" and signal.strength > 0.7:
            # 成交量异常时可能有套利机会
            pass
    
    async def _on_order_executed(self, order):
        """处理订单执行"""
        # 记录执行结果
        pass
    
    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """获取活跃套利机会"""
        return list(self.active_opportunities.values())
    
    def get_positions(self) -> Dict[str, Decimal]:
        """获取当前持仓"""
        return dict(self.positions)
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计数据"""
        return self.stats.copy()
    
    def get_status(self) -> Dict[str, any]:
        """获取Agent状态"""
        return {
            "running": self._running,
            "active_opportunities": len(self.active_opportunities),
            "pending_orders": sum(len(orders) for orders in self.pending_orders.values()),
            "total_positions": len([p for p in self.positions.values() if p != 0]),
            "statistics": self.get_statistics()
        }