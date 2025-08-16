import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Deque
import heapq
from datetime import datetime

from src.core.models import Order, MarketData
from src.hft.orderbook_manager import OrderBookManager, OrderBookSnapshot
from src.utils.logger import LoggerMixin


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    POST_ONLY = "post_only"


@dataclass
class ExecutionOrder:
    """执行订单"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"  # Good Till Cancel
    reduce_only: bool = False
    post_only: bool = False
    
    # 执行相关
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    average_price: Optional[Decimal] = None
    commission: Decimal = field(default_factory=lambda: Decimal("0"))
    
    # 时间戳
    created_time: float = field(default_factory=time.time)
    submitted_time: Optional[float] = None
    filled_time: Optional[float] = None
    updated_time: float = field(default_factory=time.time)
    
    # 元数据
    client_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """执行报告"""
    order_id: str
    symbol: str
    side: str
    execution_id: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: float
    is_maker: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class SlippageConfig:
    """滑点配置"""
    max_slippage_bps: int = 100  # 最大滑点基点
    impact_threshold: Decimal = field(default_factory=lambda: Decimal("0.01"))  # 价格冲击阈值
    liquidity_buffer: Decimal = field(default_factory=lambda: Decimal("0.1"))  # 流动性缓冲
    adaptive_sizing: bool = True  # 自适应订单大小


class FastExecutionEngine(LoggerMixin):
    """快速执行引擎"""
    
    def __init__(self, 
                 orderbook_manager: OrderBookManager,
                 max_orders: int = 10000,
                 slippage_config: Optional[SlippageConfig] = None):
        self.orderbook_manager = orderbook_manager
        self.max_orders = max_orders
        self.slippage_config = slippage_config or SlippageConfig()
        
        # 订单管理
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.order_history: Dict[str, ExecutionOrder] = {}
        self.execution_reports: Dict[str, List[ExecutionReport]] = {}
        
        # 执行队列
        self.pending_orders: Deque[ExecutionOrder] = deque()
        self.priority_queue: List = []  # 优先级队列
        
        # 性能统计
        self.execution_latency: Dict[str, List[float]] = {}
        self.slippage_stats: Dict[str, List[float]] = {}
        
        # 执行回调
        self.execution_callbacks: List[Callable] = []
        
        # 异步控制
        self._locks: Dict[str, asyncio.Lock] = {}
        self._running = False
        self._execution_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """启动执行引擎"""
        if self._running:
            return
            
        self._running = True
        self._execution_task = asyncio.create_task(self._execution_loop())
        self.log_info("Fast execution engine started")
        
    async def stop(self):
        """停止执行引擎"""
        self._running = False
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        self.log_info("Fast execution engine stopped")
        
    async def submit_order(self, order: ExecutionOrder) -> bool:
        """提交订单"""
        start_time = time.perf_counter()
        
        try:
            # 验证订单
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                return False
            
            # 生成订单ID
            if not order.order_id:
                order.order_id = str(uuid.uuid4())
            
            # 获取锁
            symbol = order.symbol
            if symbol not in self._locks:
                self._locks[symbol] = asyncio.Lock()
                
            async with self._locks[symbol]:
                # 风险检查
                if not await self._risk_check(order):
                    order.status = OrderStatus.REJECTED
                    self.log_warning(f"Order {order.order_id} rejected by risk check")
                    return False
                
                # 滑点检查和价格调整
                if not await self._slippage_check(order):
                    order.status = OrderStatus.REJECTED
                    self.log_warning(f"Order {order.order_id} rejected due to excessive slippage")
                    return False
                
                # 添加到队列
                order.status = OrderStatus.SUBMITTED
                order.submitted_time = time.time()
                self.active_orders[order.order_id] = order
                
                # 根据优先级添加到队列
                if order.order_type in [OrderType.MARKET, OrderType.IOC, OrderType.FOK]:
                    # 高优先级订单
                    heapq.heappush(self.priority_queue, (0, time.time(), order))
                else:
                    # 普通订单
                    self.pending_orders.append(order)
                
                # 记录延迟
                processing_time = (time.perf_counter() - start_time) * 1000
                if symbol not in self.execution_latency:
                    self.execution_latency[symbol] = []
                self.execution_latency[symbol].append(processing_time)
                
                self.log_info(f"Order {order.order_id} submitted for {symbol}, processing time: {processing_time:.2f}ms")
                return True
                
        except Exception as e:
            self.log_error(f"Error submitting order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.active_orders:
            return False
            
        order = self.active_orders[order_id]
        symbol = order.symbol
        
        async with self._locks.get(symbol, asyncio.Lock()):
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                return False
                
            order.status = OrderStatus.CANCELLED
            order.updated_time = time.time()
            
            # 从活跃订单中移除
            self.active_orders.pop(order_id, None)
            self.order_history[order_id] = order
            
            self.log_info(f"Order {order_id} cancelled")
            await self._notify_execution(order)
            return True
    
    async def _execution_loop(self):
        """执行循环"""
        while self._running:
            try:
                # 处理高优先级订单
                while self.priority_queue and self._running:
                    _, _, order = heapq.heappop(self.priority_queue)
                    await self._execute_order(order)
                
                # 处理普通订单
                if self.pending_orders and self._running:
                    order = self.pending_orders.popleft()
                    await self._execute_order(order)
                
                # 短暂休眠避免CPU占用过高
                await asyncio.sleep(0.001)  # 1ms
                
            except Exception as e:
                self.log_error(f"Error in execution loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _execute_order(self, order: ExecutionOrder):
        """执行订单"""
        try:
            # 获取当前市场数据
            orderbook = self.orderbook_manager.get_orderbook(order.symbol)
            if not orderbook:
                order.status = OrderStatus.REJECTED
                await self._notify_execution(order)
                return
            
            # 根据订单类型执行
            if order.order_type == OrderType.MARKET:
                await self._execute_market_order(order, orderbook)
            elif order.order_type == OrderType.LIMIT:
                await self._execute_limit_order(order, orderbook)
            elif order.order_type == OrderType.IOC:
                await self._execute_ioc_order(order, orderbook)
            elif order.order_type == OrderType.FOK:
                await self._execute_fok_order(order, orderbook)
            else:
                order.status = OrderStatus.REJECTED
                self.log_warning(f"Unsupported order type: {order.order_type}")
            
            await self._notify_execution(order)
            
        except Exception as e:
            self.log_error(f"Error executing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            await self._notify_execution(order)
    
    async def _execute_market_order(self, order: ExecutionOrder, orderbook: OrderBookSnapshot):
        """执行市价单"""
        if order.side.lower() == "buy":
            best_price = orderbook.best_ask.price if orderbook.best_ask else None
        else:
            best_price = orderbook.best_bid.price if orderbook.best_bid else None
        
        if not best_price:
            order.status = OrderStatus.REJECTED
            return
        
        # 模拟执行
        execution_price = best_price
        execution_quantity = order.quantity
        
        # 创建执行报告
        execution_report = ExecutionReport(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            execution_id=str(uuid.uuid4()),
            quantity=execution_quantity,
            price=execution_price,
            commission=execution_quantity * execution_price * Decimal("0.0004"),  # 0.04% 手续费
            timestamp=time.time(),
            is_maker=False
        )
        
        # 更新订单
        order.filled_quantity = execution_quantity
        order.average_price = execution_price
        order.commission = execution_report.commission
        order.status = OrderStatus.FILLED
        order.filled_time = time.time()
        order.updated_time = time.time()
        
        # 记录执行报告
        if order.order_id not in self.execution_reports:
            self.execution_reports[order.order_id] = []
        self.execution_reports[order.order_id].append(execution_report)
        
        # 计算滑点
        await self._calculate_slippage(order, execution_price)
        
        # 从活跃订单移至历史
        self.active_orders.pop(order.order_id, None)
        self.order_history[order.order_id] = order
        
        self.log_info(f"Market order {order.order_id} executed at {execution_price}")
    
    async def _execute_limit_order(self, order: ExecutionOrder, orderbook: OrderBookSnapshot):
        """执行限价单"""
        # 检查是否能立即成交
        if order.side.lower() == "buy":
            best_ask = orderbook.best_ask.price if orderbook.best_ask else None
            can_fill = best_ask and order.price >= best_ask
        else:
            best_bid = orderbook.best_bid.price if orderbook.best_bid else None
            can_fill = best_bid and order.price <= best_bid
        
        if can_fill:
            # 立即成交
            if order.post_only:
                # Post-only订单不能立即成交
                order.status = OrderStatus.CANCELLED
                return
            
            execution_price = order.price
            execution_quantity = order.quantity
            
            # 创建执行报告
            execution_report = ExecutionReport(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                execution_id=str(uuid.uuid4()),
                quantity=execution_quantity,
                price=execution_price,
                commission=execution_quantity * execution_price * Decimal("0.0002"),  # Maker费率
                timestamp=time.time(),
                is_maker=True
            )
            
            # 更新订单
            order.filled_quantity = execution_quantity
            order.average_price = execution_price
            order.commission = execution_report.commission
            order.status = OrderStatus.FILLED
            order.filled_time = time.time()
            order.updated_time = time.time()
            
            # 记录执行报告
            if order.order_id not in self.execution_reports:
                self.execution_reports[order.order_id] = []
            self.execution_reports[order.order_id].append(execution_report)
            
            # 从活跃订单移至历史
            self.active_orders.pop(order.order_id, None)
            self.order_history[order.order_id] = order
            
            self.log_info(f"Limit order {order.order_id} executed at {execution_price}")
        else:
            # 挂单等待
            self.log_debug(f"Limit order {order.order_id} waiting for execution")
    
    async def _execute_ioc_order(self, order: ExecutionOrder, orderbook: OrderBookSnapshot):
        """执行IOC订单"""
        # IOC订单立即执行能执行的部分，其余取消
        if order.side.lower() == "buy":
            available_levels = orderbook.asks
        else:
            available_levels = orderbook.bids
        
        filled_quantity = Decimal("0")
        total_cost = Decimal("0")
        remaining_quantity = order.quantity
        
        for level in available_levels:
            if remaining_quantity <= 0:
                break
                
            if order.side.lower() == "buy" and order.price and level.price > order.price:
                break
            if order.side.lower() == "sell" and order.price and level.price < order.price:
                break
                
            fill_qty = min(remaining_quantity, level.size)
            total_cost += fill_qty * level.price
            filled_quantity += fill_qty
            remaining_quantity -= fill_qty
        
        if filled_quantity > 0:
            average_price = total_cost / filled_quantity
            
            # 创建执行报告
            execution_report = ExecutionReport(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                execution_id=str(uuid.uuid4()),
                quantity=filled_quantity,
                price=average_price,
                commission=total_cost * Decimal("0.0004"),
                timestamp=time.time(),
                is_maker=False
            )
            
            # 更新订单
            order.filled_quantity = filled_quantity
            order.average_price = average_price
            order.commission = execution_report.commission
            
            if remaining_quantity > 0:
                order.status = OrderStatus.PARTIAL_FILLED
            else:
                order.status = OrderStatus.FILLED
                order.filled_time = time.time()
            
            order.updated_time = time.time()
            
            # 记录执行报告
            if order.order_id not in self.execution_reports:
                self.execution_reports[order.order_id] = []
            self.execution_reports[order.order_id].append(execution_report)
            
            self.log_info(f"IOC order {order.order_id} partially filled: {filled_quantity}/{order.quantity}")
        
        # IOC订单未成交部分自动取消
        if order.status != OrderStatus.FILLED:
            order.status = OrderStatus.CANCELLED
        
        # 从活跃订单移至历史
        self.active_orders.pop(order.order_id, None)
        self.order_history[order.order_id] = order
    
    async def _execute_fok_order(self, order: ExecutionOrder, orderbook: OrderBookSnapshot):
        """执行FOK订单"""
        # FOK订单必须全部成交，否则全部取消
        if order.side.lower() == "buy":
            available_levels = orderbook.asks
        else:
            available_levels = orderbook.bids
        
        total_available = Decimal("0")
        total_cost = Decimal("0")
        
        for level in available_levels:
            if order.side.lower() == "buy" and order.price and level.price > order.price:
                break
            if order.side.lower() == "sell" and order.price and level.price < order.price:
                break
                
            fill_qty = min(order.quantity - total_available, level.size)
            total_cost += fill_qty * level.price
            total_available += fill_qty
            
            if total_available >= order.quantity:
                break
        
        if total_available >= order.quantity:
            # 可以全部成交
            average_price = total_cost / order.quantity
            
            # 创建执行报告
            execution_report = ExecutionReport(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                execution_id=str(uuid.uuid4()),
                quantity=order.quantity,
                price=average_price,
                commission=total_cost * Decimal("0.0004"),
                timestamp=time.time(),
                is_maker=False
            )
            
            # 更新订单
            order.filled_quantity = order.quantity
            order.average_price = average_price
            order.commission = execution_report.commission
            order.status = OrderStatus.FILLED
            order.filled_time = time.time()
            order.updated_time = time.time()
            
            # 记录执行报告
            if order.order_id not in self.execution_reports:
                self.execution_reports[order.order_id] = []
            self.execution_reports[order.order_id].append(execution_report)
            
            self.log_info(f"FOK order {order.order_id} fully executed at {average_price}")
        else:
            # 无法全部成交，取消订单
            order.status = OrderStatus.CANCELLED
            self.log_info(f"FOK order {order.order_id} cancelled - insufficient liquidity")
        
        # 从活跃订单移至历史
        self.active_orders.pop(order.order_id, None)
        self.order_history[order.order_id] = order
    
    def _validate_order(self, order: ExecutionOrder) -> bool:
        """验证订单"""
        if not order.symbol or not order.side or order.quantity <= 0:
            return False
        
        if order.side.lower() not in ["buy", "sell"]:
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not order.price:
            return False
        
        return True
    
    async def _risk_check(self, order: ExecutionOrder) -> bool:
        """风险检查"""
        # 这里可以集成更复杂的风险管理逻辑
        # 目前只做基本检查
        
        # 检查订单大小
        max_order_value = Decimal("100000")  # 最大订单价值
        if order.price and order.quantity * order.price > max_order_value:
            return False
        
        # 检查活跃订单数量
        symbol_orders = [o for o in self.active_orders.values() if o.symbol == order.symbol]
        if len(symbol_orders) > 50:  # 单个币种最多50个活跃订单
            return False
        
        return True
    
    async def _slippage_check(self, order: ExecutionOrder) -> bool:
        """滑点检查"""
        if order.order_type != OrderType.MARKET:
            return True
        
        orderbook = self.orderbook_manager.get_orderbook(order.symbol)
        if not orderbook:
            return False
        
        # 计算预期滑点
        impact_price = self.orderbook_manager.calculate_impact_price(
            order.symbol, order.side, order.quantity
        )
        
        if not impact_price:
            return False
        
        mid_price = orderbook.mid_price
        if not mid_price:
            return False
        
        # 计算滑点基点
        slippage_bps = abs(float((impact_price - mid_price) / mid_price * 10000))
        
        if slippage_bps > self.slippage_config.max_slippage_bps:
            self.log_warning(f"Order {order.order_id} slippage {slippage_bps:.2f}bps exceeds limit")
            return False
        
        return True
    
    async def _calculate_slippage(self, order: ExecutionOrder, execution_price: Decimal):
        """计算实际滑点"""
        orderbook = self.orderbook_manager.get_orderbook(order.symbol)
        if not orderbook or not orderbook.mid_price:
            return
        
        # 计算滑点
        slippage = float(abs(execution_price - orderbook.mid_price) / orderbook.mid_price * 10000)
        
        if order.symbol not in self.slippage_stats:
            self.slippage_stats[order.symbol] = []
        self.slippage_stats[order.symbol].append(slippage)
        
        # 保持统计数量
        if len(self.slippage_stats[order.symbol]) > 1000:
            self.slippage_stats[order.symbol] = self.slippage_stats[order.symbol][-1000:]
    
    async def _notify_execution(self, order: ExecutionOrder):
        """通知执行结果"""
        for callback in self.execution_callbacks:
            try:
                await callback(order)
            except Exception as e:
                self.log_error(f"Error in execution callback: {e}")
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def get_order(self, order_id: str) -> Optional[ExecutionOrder]:
        """获取订单"""
        return self.active_orders.get(order_id) or self.order_history.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[ExecutionOrder]:
        """获取活跃订单"""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_execution_reports(self, order_id: str) -> List[ExecutionReport]:
        """获取执行报告"""
        return self.execution_reports.get(order_id, [])
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict[str, any]:
        """获取性能统计"""
        if symbol:
            latency_stats = self.execution_latency.get(symbol, [])
            slippage_stats = self.slippage_stats.get(symbol, [])
        else:
            latency_stats = []
            slippage_stats = []
            for stats in self.execution_latency.values():
                latency_stats.extend(stats)
            for stats in self.slippage_stats.values():
                slippage_stats.extend(stats)
        
        def calculate_percentile(data: List[float], percentile: float) -> float:
            if not data:
                return 0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        return {
            "execution_latency": {
                "avg_ms": sum(latency_stats) / len(latency_stats) if latency_stats else 0,
                "min_ms": min(latency_stats) if latency_stats else 0,
                "max_ms": max(latency_stats) if latency_stats else 0,
                "p95_ms": calculate_percentile(latency_stats, 0.95),
                "p99_ms": calculate_percentile(latency_stats, 0.99),
                "count": len(latency_stats)
            },
            "slippage": {
                "avg_bps": sum(slippage_stats) / len(slippage_stats) if slippage_stats else 0,
                "min_bps": min(slippage_stats) if slippage_stats else 0,
                "max_bps": max(slippage_stats) if slippage_stats else 0,
                "p95_bps": calculate_percentile(slippage_stats, 0.95),
                "p99_bps": calculate_percentile(slippage_stats, 0.99),
                "count": len(slippage_stats)
            },
            "active_orders": len(self.active_orders),
            "total_executed": len(self.order_history)
        }