import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Deque
import heapq
from datetime import datetime

from src.core.models import MarketData
from src.core.ring_buffer import RingBuffer
from src.utils.logger import LoggerMixin


@dataclass
class OrderBookLevel:
    """订单簿价格层级"""
    price: Decimal
    size: Decimal
    orders: int = 1
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        return self.price < other.price


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]  # 按价格降序排列
    asks: List[OrderBookLevel]  # 按价格升序排列
    sequence: int = 0
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """最优买价"""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """最优卖价"""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """买卖价差"""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """中间价"""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / Decimal("2")
        return None
    
    def get_depth(self, side: str, levels: int = 5) -> List[OrderBookLevel]:
        """获取指定深度"""
        if side.upper() == "BUY":
            return self.bids[:levels]
        else:
            return self.asks[:levels]


class OrderBookManager(LoggerMixin):
    """低延迟订单簿管理器"""
    
    def __init__(self, max_levels: int = 50, history_size: int = 1000):
        self.max_levels = max_levels
        self.history_size = history_size
        
        # 实时订单簿数据
        self.orderbooks: Dict[str, OrderBookSnapshot] = {}
        self.last_update: Dict[str, float] = {}
        
        # 历史快照（使用环形缓冲区）
        self.history: Dict[str, RingBuffer] = {}
        
        # 增量更新队列
        self.update_queue: Dict[str, Deque] = {}
        
        # 延迟统计
        self.latency_stats: Dict[str, List[float]] = {}
        
        # 锁，用于并发控制
        self._locks: Dict[str, asyncio.Lock] = {}
        
    async def initialize(self, symbols: List[str]):
        """初始化订单簿管理器"""
        for symbol in symbols:
            self.orderbooks[symbol] = OrderBookSnapshot(
                symbol=symbol,
                timestamp=time.time(),
                bids=[],
                asks=[]
            )
            self.history[symbol] = RingBuffer(self.history_size)
            self.update_queue[symbol] = deque(maxlen=1000)
            self.latency_stats[symbol] = []
            self._locks[symbol] = asyncio.Lock()
            
        self.log_info(f"OrderBook manager initialized for {len(symbols)} symbols")
    
    async def update_orderbook(self, symbol: str, market_data: MarketData) -> bool:
        """更新订单簿数据"""
        start_time = time.perf_counter()
        
        async with self._locks.get(symbol, asyncio.Lock()):
            try:
                # 创建新的订单簿快照
                current_time = time.time()
                
                # 从市场数据构建订单簿
                bids = [OrderBookLevel(
                    price=Decimal(str(market_data.bid)),
                    size=Decimal(str(market_data.bid_volume)),
                    timestamp=current_time
                )]
                
                asks = [OrderBookLevel(
                    price=Decimal(str(market_data.ask)),
                    size=Decimal(str(market_data.ask_volume)),
                    timestamp=current_time
                )]
                
                new_snapshot = OrderBookSnapshot(
                    symbol=symbol,
                    timestamp=current_time,
                    bids=bids,
                    asks=asks,
                    sequence=self.orderbooks[symbol].sequence + 1
                )
                
                # 保存历史快照
                old_snapshot = self.orderbooks[symbol]
                self.history[symbol].append(old_snapshot)
                
                # 更新当前快照
                self.orderbooks[symbol] = new_snapshot
                self.last_update[symbol] = current_time
                
                # 记录延迟
                processing_time = (time.perf_counter() - start_time) * 1000  # 毫秒
                self.latency_stats[symbol].append(processing_time)
                
                # 保持延迟统计数量
                if len(self.latency_stats[symbol]) > 1000:
                    self.latency_stats[symbol] = self.latency_stats[symbol][-1000:]
                
                return True
                
            except Exception as e:
                self.log_error(f"Error updating orderbook for {symbol}: {e}")
                return False
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """获取当前订单簿快照"""
        return self.orderbooks.get(symbol)
    
    def get_best_prices(self, symbol: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """获取最优买卖价"""
        orderbook = self.orderbooks.get(symbol)
        if not orderbook:
            return None, None
            
        best_bid = orderbook.best_bid.price if orderbook.best_bid else None
        best_ask = orderbook.best_ask.price if orderbook.best_ask else None
        
        return best_bid, best_ask
    
    def get_spread(self, symbol: str) -> Optional[Decimal]:
        """获取买卖价差"""
        orderbook = self.orderbooks.get(symbol)
        return orderbook.spread if orderbook else None
    
    def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        """获取中间价"""
        orderbook = self.orderbooks.get(symbol)
        return orderbook.mid_price if orderbook else None
    
    def get_market_depth(self, symbol: str, side: str, depth: Decimal) -> Decimal:
        """计算指定深度的流动性"""
        orderbook = self.orderbooks.get(symbol)
        if not orderbook:
            return Decimal("0")
        
        levels = orderbook.get_depth(side, self.max_levels)
        total_volume = Decimal("0")
        
        for level in levels:
            if total_volume >= depth:
                break
            total_volume += level.size
        
        return total_volume
    
    def calculate_impact_price(self, symbol: str, side: str, quantity: Decimal) -> Optional[Decimal]:
        """计算市场冲击价格"""
        orderbook = self.orderbooks.get(symbol)
        if not orderbook:
            return None
        
        levels = orderbook.get_depth(side, self.max_levels)
        remaining_qty = quantity
        total_cost = Decimal("0")
        
        for level in levels:
            if remaining_qty <= 0:
                break
                
            fill_qty = min(remaining_qty, level.size)
            total_cost += fill_qty * level.price
            remaining_qty -= fill_qty
        
        if remaining_qty > 0:
            # 流动性不足
            return None
        
        return total_cost / quantity
    
    def get_historical_snapshot(self, symbol: str, index: int = -1) -> Optional[OrderBookSnapshot]:
        """获取历史快照"""
        history = self.history.get(symbol)
        if not history or len(history) == 0:
            return None
            
        try:
            return history[index]
        except IndexError:
            return None
    
    def get_latency_stats(self, symbol: str) -> Dict[str, float]:
        """获取延迟统计"""
        stats = self.latency_stats.get(symbol, [])
        if not stats:
            return {"avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
        
        stats_sorted = sorted(stats)
        count = len(stats_sorted)
        
        return {
            "avg": sum(stats) / count,
            "min": stats_sorted[0],
            "max": stats_sorted[-1],
            "p95": stats_sorted[int(count * 0.95)] if count > 0 else 0,
            "p99": stats_sorted[int(count * 0.99)] if count > 0 else 0,
            "count": count
        }
    
    def get_orderbook_health(self, symbol: str) -> Dict[str, any]:
        """获取订单簿健康状态"""
        orderbook = self.orderbooks.get(symbol)
        if not orderbook:
            return {"status": "no_data"}
        
        current_time = time.time()
        last_update_time = self.last_update.get(symbol, 0)
        
        # 数据新鲜度检查
        data_age = current_time - last_update_time
        is_stale = data_age > 1.0  # 1秒内的数据认为是新鲜的
        
        # 价差合理性检查
        spread = orderbook.spread
        mid_price = orderbook.mid_price
        spread_bps = None
        if spread and mid_price and mid_price > 0:
            spread_bps = float(spread / mid_price * 10000)  # 基点
        
        # 深度检查
        bid_depth = sum(level.size for level in orderbook.bids[:5])
        ask_depth = sum(level.size for level in orderbook.asks[:5])
        
        return {
            "status": "stale" if is_stale else "healthy",
            "data_age_seconds": data_age,
            "spread": float(spread) if spread else None,
            "spread_bps": spread_bps,
            "bid_depth": float(bid_depth),
            "ask_depth": float(ask_depth),
            "bid_levels": len(orderbook.bids),
            "ask_levels": len(orderbook.asks),
            "sequence": orderbook.sequence
        }
    
    async def cleanup(self):
        """清理资源"""
        self.orderbooks.clear()
        self.history.clear()
        self.update_queue.clear()
        self.latency_stats.clear()
        self.log_info("OrderBook manager cleaned up")