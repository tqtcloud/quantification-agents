import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from decimal import Decimal

from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData, Position
from src.utils.logger import LoggerMixin


@dataclass
class VirtualAccount:
    """虚拟账户"""
    account_id: str
    initial_balance: float = 100000.0  # 初始资金 $100k
    current_balance: float = 0.0
    total_pnl: float = 0.0
    available_balance: float = 0.0
    margin_used: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.current_balance == 0.0:
            self.current_balance = self.initial_balance
            self.available_balance = self.initial_balance


@dataclass
class VirtualPosition:
    """虚拟仓位"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0
    leverage: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_pnl(self, current_price: float):
        """更新盈亏"""
        self.current_price = current_price
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        self.updated_at = datetime.utcnow()


@dataclass
class VirtualTrade:
    """虚拟成交记录"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PaperTradingEngine(LoggerMixin):
    """虚拟盘交易引擎"""
    
    def __init__(self, account_id: str = "paper_account_001"):
        self.account = VirtualAccount(account_id)
        
        # 持仓管理
        self.positions: Dict[str, VirtualPosition] = {}
        
        # 订单管理
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[VirtualTrade] = []
        
        # 市场数据
        self.market_prices: Dict[str, float] = {}
        self.last_market_update: Dict[str, datetime] = {}
        
        # 交易设置
        self.commission_rate = 0.0004  # 0.04% 手续费
        self.slippage_rate = 0.0001   # 0.01% 滑点
        self.leverage_limit = 20
        
        # 统计信息
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "total_trades": 0,
            "total_commission": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": 0.0
        }
    
    async def initialize(self):
        """初始化引擎"""
        self.log_info("Initializing PaperTradingEngine", account_id=self.account.account_id)
        self.stats["peak_balance"] = self.account.initial_balance
        self.log_info("PaperTradingEngine initialized successfully")
    
    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """执行订单"""
        self.log_info(
            "Executing paper order",
            symbol=order.symbol,
            side=order.side.value,
            type=order.order_type.value,
            quantity=order.quantity,
            price=order.price
        )
        
        # 生成订单ID
        if not order.client_order_id:
            order.client_order_id = f"paper_{uuid.uuid4().hex[:8]}"
        
        order_id = f"paper_{int(time.time() * 1000)}{len(self.active_orders)}"
        order.order_id = order_id
        order.status = OrderStatus.NEW
        order.created_at = datetime.utcnow()
        
        # 添加到活跃订单
        self.active_orders[order.client_order_id] = order
        self.stats["total_orders"] += 1
        
        try:
            # 获取当前市场价格
            current_price = self._get_market_price(order.symbol)
            
            # 根据订单类型处理
            if order.order_type == OrderType.MARKET:
                # 市价单立即成交
                execution_price = self._calculate_execution_price(order, current_price)
                await self._fill_order(order, execution_price, order.quantity)
                
            elif order.order_type == OrderType.LIMIT:
                # 限价单检查是否能立即成交
                if self._can_fill_limit_order(order, current_price):
                    execution_price = order.price
                    await self._fill_order(order, execution_price, order.quantity)
                else:
                    # 挂单等待
                    order.status = OrderStatus.NEW
                    self.log_debug(f"Limit order pending: {order.client_order_id}")
            
            elif order.order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
                # 止损单转为挂单
                order.status = OrderStatus.NEW
                self.log_debug(f"Stop order placed: {order.client_order_id}")
            
            # 返回执行结果
            return {
                "orderId": order_id,
                "symbol": order.symbol,
                "status": order.status.value,
                "clientOrderId": order.client_order_id,
                "executedQty": str(order.executed_qty),
                "avgPrice": str(order.avg_price) if order.avg_price > 0 else "0",
                "transactTime": int(order.created_at.timestamp() * 1000),
                "type": order.order_type.value,
                "side": order.side.value
            }
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.log_error(f"Order execution failed: {e}")
            return {
                "orderId": order_id,
                "symbol": order.symbol,
                "status": "REJECTED",
                "clientOrderId": order.client_order_id,
                "error": str(e)
            }
    
    def _get_market_price(self, symbol: str) -> float:
        """获取市场价格"""
        # 如果有实时价格数据则使用
        if symbol in self.market_prices:
            return self.market_prices[symbol]
        
        # 否则使用模拟价格
        base_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "BNBUSDT": 400.0,
            "ADAUSDT": 1.5,
            "SOLUSDT": 100.0
        }
        
        return base_prices.get(symbol, 1000.0)
    
    def _calculate_execution_price(self, order: Order, market_price: float) -> float:
        """计算执行价格（含滑点）"""
        slippage = self.slippage_rate
        
        if order.side == OrderSide.BUY:
            # 买单向上滑点
            return market_price * (1 + slippage)
        else:
            # 卖单向下滑点
            return market_price * (1 - slippage)
    
    def _can_fill_limit_order(self, order: Order, market_price: float) -> bool:
        """判断限价单是否可以成交"""
        if not order.price:
            return False
        
        if order.side == OrderSide.BUY:
            # 买单：市价低于限价可成交
            return market_price <= order.price
        else:
            # 卖单：市价高于限价可成交
            return market_price >= order.price
    
    async def _fill_order(self, order: Order, execution_price: float, fill_quantity: float):
        """成交订单"""
        # 检查资金充足性
        if not self._check_account_balance(order, execution_price, fill_quantity):
            order.status = OrderStatus.REJECTED
            raise ValueError("Insufficient balance")
        
        # 计算手续费
        commission = fill_quantity * execution_price * self.commission_rate
        
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.executed_qty = fill_quantity
        order.avg_price = execution_price
        order.updated_at = datetime.utcnow()
        
        # 创建成交记录
        trade = VirtualTrade(
            trade_id=f"trade_{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=fill_quantity,
            price=execution_price,
            commission=commission
        )
        self.trade_history.append(trade)
        
        # 更新仓位
        await self._update_position(order, execution_price, fill_quantity)
        
        # 更新账户余额
        self._update_account_balance(order, execution_price, fill_quantity, commission)
        
        # 移动到历史记录
        self.order_history.append(order)
        if order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        # 更新统计
        self.stats["filled_orders"] += 1
        self.stats["total_trades"] += 1
        self.stats["total_commission"] += commission
        
        # 更新最大回撤
        self._update_drawdown_stats()
        
        self.log_info(
            "Order filled",
            order_id=order.order_id,
            symbol=order.symbol,
            price=execution_price,
            quantity=fill_quantity,
            commission=commission
        )
    
    def _check_account_balance(self, order: Order, price: float, quantity: float) -> bool:
        """检查账户余额"""
        required_margin = quantity * price / self.leverage_limit  # 假设最大杠杆
        commission = quantity * price * self.commission_rate
        
        total_required = required_margin + commission
        
        return self.account.available_balance >= total_required
    
    async def _update_position(self, order: Order, price: float, quantity: float):
        """更新仓位"""
        symbol = order.symbol
        side = "LONG" if order.side == OrderSide.BUY else "SHORT"
        
        if symbol not in self.positions:
            # 新建仓位
            self.positions[symbol] = VirtualPosition(
                symbol=symbol,
                side=side,
                size=quantity,
                entry_price=price,
                current_price=price,
                leverage=1  # 简化为1倍杠杆
            )
            self.log_debug(f"New position created: {symbol} {side} {quantity}@{price}")
            
        else:
            # 更新现有仓位
            position = self.positions[symbol]
            
            if position.side == side:
                # 同方向加仓
                total_value = position.size * position.entry_price + quantity * price
                total_size = position.size + quantity
                position.entry_price = total_value / total_size
                position.size = total_size
                
            else:
                # 反方向减仓或开反向仓
                if position.size > quantity:
                    # 部分平仓
                    position.size -= quantity
                    # 计算已实现盈亏
                    if position.side == "LONG":
                        pnl = (price - position.entry_price) * quantity
                    else:
                        pnl = (position.entry_price - price) * quantity
                    position.realized_pnl += pnl
                    self.account.total_pnl += pnl
                    
                elif position.size == quantity:
                    # 完全平仓
                    if position.side == "LONG":
                        pnl = (price - position.entry_price) * quantity
                    else:
                        pnl = (position.entry_price - price) * quantity
                    position.realized_pnl += pnl
                    self.account.total_pnl += pnl
                    del self.positions[symbol]
                    
                else:
                    # 平仓并开反向仓
                    close_quantity = position.size
                    remain_quantity = quantity - close_quantity
                    
                    # 平仓盈亏
                    if position.side == "LONG":
                        pnl = (price - position.entry_price) * close_quantity
                    else:
                        pnl = (position.entry_price - price) * close_quantity
                    self.account.total_pnl += pnl
                    
                    # 开新仓
                    position.side = side
                    position.size = remain_quantity
                    position.entry_price = price
                    position.realized_pnl += pnl
            
            position.current_price = price
            position.update_pnl(price)
    
    def _update_account_balance(self, order: Order, price: float, quantity: float, commission: float):
        """更新账户余额"""
        # 扣除手续费
        self.account.current_balance -= commission
        self.account.available_balance -= commission
        
        # 更新可用余额（简化处理）
        trade_value = quantity * price
        if order.side == OrderSide.BUY:
            self.account.available_balance -= trade_value / self.leverage_limit
        else:
            self.account.available_balance += trade_value / self.leverage_limit
    
    def _update_drawdown_stats(self):
        """更新回撤统计"""
        current_equity = self.account.current_balance + self.account.total_pnl
        
        if current_equity > self.stats["peak_balance"]:
            self.stats["peak_balance"] = current_equity
        
        current_drawdown = (self.stats["peak_balance"] - current_equity) / self.stats["peak_balance"]
        if current_drawdown > self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = current_drawdown
    
    async def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        """取消订单"""
        if client_order_id not in self.active_orders:
            raise ValueError(f"Order not found: {client_order_id}")
        
        order = self.active_orders[client_order_id]
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.utcnow()
        
        # 移动到历史记录
        self.order_history.append(order)
        del self.active_orders[client_order_id]
        
        self.log_info(f"Order cancelled: {client_order_id}")
        
        return {
            "orderId": order.order_id,
            "clientOrderId": client_order_id,
            "symbol": order.symbol,
            "status": "CANCELED"
        }
    
    async def get_order_status(self, client_order_id: str) -> Dict[str, Any]:
        """查询订单状态"""
        # 先查活跃订单
        if client_order_id in self.active_orders:
            order = self.active_orders[client_order_id]
        else:
            # 查历史订单
            order = None
            for hist_order in self.order_history:
                if hist_order.client_order_id == client_order_id:
                    order = hist_order
                    break
            
            if not order:
                raise ValueError(f"Order not found: {client_order_id}")
        
        return {
            "orderId": order.order_id,
            "clientOrderId": client_order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "executedQty": str(order.executed_qty),
            "avgPrice": str(order.avg_price) if order.avg_price > 0 else "0",
            "type": order.order_type.value,
            "side": order.side.value
        }
    
    def update_market_data(self, market_data: MarketData):
        """更新市场数据"""
        self.market_prices[market_data.symbol] = market_data.price
        self.last_market_update[market_data.symbol] = datetime.utcnow()
        
        # 更新持仓盈亏
        if market_data.symbol in self.positions:
            self.positions[market_data.symbol].update_pnl(market_data.price)
    
    async def process_pending_orders(self):
        """处理挂单"""
        for client_order_id, order in list(self.active_orders.items()):
            if order.status != OrderStatus.NEW:
                continue
            
            current_price = self._get_market_price(order.symbol)
            
            # 检查限价单
            if order.order_type == OrderType.LIMIT and self._can_fill_limit_order(order, current_price):
                await self._fill_order(order, order.price, order.quantity)
            
            # 检查止损单
            elif order.order_type in [OrderType.STOP, OrderType.STOP_MARKET] and order.stop_price:
                if ((order.side == OrderSide.BUY and current_price >= order.stop_price) or
                    (order.side == OrderSide.SELL and current_price <= order.stop_price)):
                    # 止损触发，转为市价单
                    execution_price = self._calculate_execution_price(order, current_price)
                    await self._fill_order(order, execution_price, order.quantity)
    
    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_equity = self.account.current_balance + total_unrealized_pnl
        
        return {
            "accountId": self.account.account_id,
            "balance": self.account.current_balance,
            "availableBalance": self.account.available_balance,
            "totalPnL": self.account.total_pnl,
            "unrealizedPnL": total_unrealized_pnl,
            "totalEquity": total_equity,
            "marginUsed": self.account.margin_used,
            "positions": len(self.positions),
            "activeOrders": len(self.active_orders)
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        return [
            {
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entryPrice": pos.entry_price,
                "currentPrice": pos.current_price,
                "unrealizedPnL": pos.unrealized_pnl,
                "realizedPnL": pos.realized_pnl,
                "leverage": pos.leverage,
                "createdAt": pos.created_at.isoformat(),
                "updatedAt": pos.updated_at.isoformat()
            }
            for pos in self.positions.values()
        ]
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """获取交易统计"""
        win_trades = sum(1 for trade in self.trade_history if self._calculate_trade_pnl(trade) > 0)
        total_trades = len(self.trade_history)
        
        stats = self.stats.copy()
        stats.update({
            "winRate": win_trades / total_trades if total_trades > 0 else 0,
            "totalTrades": total_trades,
            "avgCommission": self.stats["total_commission"] / total_trades if total_trades > 0 else 0,
            "currentEquity": self.account.current_balance + sum(pos.unrealized_pnl for pos in self.positions.values()),
            "returnRate": (self.account.current_balance - self.account.initial_balance) / self.account.initial_balance
        })
        
        return stats
    
    def _calculate_trade_pnl(self, trade: VirtualTrade) -> float:
        """计算单笔交易盈亏（简化）"""
        # 这里需要更复杂的逻辑来计算实际盈亏
        # 简化处理，返回0
        return 0.0