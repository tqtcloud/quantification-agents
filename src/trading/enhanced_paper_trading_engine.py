import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json

from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData, Position
from src.utils.logger import LoggerMixin


class PositionMode(Enum):
    """仓位模式"""
    ONE_WAY = "one_way"      # 单向持仓
    HEDGE = "hedge"          # 双向持仓


@dataclass
class EnhancedVirtualAccount:
    """增强虚拟账户"""
    account_id: str
    initial_balance: Decimal = Decimal("100000.0")
    current_balance: Decimal = Decimal("0.0")
    available_balance: Decimal = Decimal("0.0")
    margin_used: Decimal = Decimal("0.0")
    total_pnl: Decimal = Decimal("0.0")
    total_commission: Decimal = Decimal("0.0")
    
    # 账户配置
    max_leverage: int = 20
    position_mode: PositionMode = PositionMode.ONE_WAY
    margin_type: str = "ISOLATED"  # ISOLATED, CROSS
    
    # 风险控制
    max_position_size: Decimal = Decimal("10000.0")  # 单个仓位最大价值
    max_daily_loss: Decimal = Decimal("1000.0")      # 每日最大亏损
    daily_loss: Decimal = Decimal("0.0")             # 当日亏损
    last_reset_date: datetime = field(default_factory=datetime.utcnow)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if self.current_balance == 0:
            self.current_balance = self.initial_balance
            self.available_balance = self.initial_balance

    def reset_daily_stats(self):
        """重置日统计"""
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date.date():
            self.daily_loss = Decimal("0.0")
            self.last_reset_date = datetime.utcnow()


@dataclass 
class EnhancedVirtualPosition:
    """增强虚拟仓位"""
    symbol: str
    side: str  # LONG, SHORT
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    
    # 盈亏计算
    unrealized_pnl: Decimal = Decimal("0.0")
    realized_pnl: Decimal = Decimal("0.0")
    
    # 保证金和杠杆
    margin: Decimal = Decimal("0.0")
    leverage: int = 1
    margin_type: str = "ISOLATED"
    
    # 风险管理
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    liquidation_price: Optional[Decimal] = None
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_pnl(self, current_price: Decimal, contract_size: Decimal = Decimal("1.0")):
        """更新盈亏"""
        self.current_price = current_price
        price_diff = current_price - self.entry_price
        
        if self.side == "LONG":
            self.unrealized_pnl = price_diff * self.size * contract_size
        else:  # SHORT
            self.unrealized_pnl = -price_diff * self.size * contract_size
            
        self.updated_at = datetime.utcnow()
        
        # 更新强平价格
        self._calculate_liquidation_price()
    
    def _calculate_liquidation_price(self):
        """计算强平价格"""
        if self.margin <= 0:
            return
            
        maintenance_margin_rate = Decimal("0.005")  # 0.5% 维持保证金率
        
        if self.side == "LONG":
            # 多头强平价 = 开仓价 - (保证金 - 维持保证金) / 数量
            self.liquidation_price = self.entry_price - (
                self.margin * (Decimal("1.0") - maintenance_margin_rate)
            ) / self.size
        else:
            # 空头强平价 = 开仓价 + (保证金 - 维持保证金) / 数量  
            self.liquidation_price = self.entry_price + (
                self.margin * (Decimal("1.0") - maintenance_margin_rate)
            ) / self.size


@dataclass
class MarketDepth:
    """市场深度数据"""
    symbol: str
    bids: List[tuple[Decimal, Decimal]]  # [(价格, 数量)]
    asks: List[tuple[Decimal, Decimal]]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_best_bid(self) -> Optional[Decimal]:
        """获取最佳买价"""
        return self.bids[0][0] if self.bids else None
    
    def get_best_ask(self) -> Optional[Decimal]:
        """获取最佳卖价"""
        return self.asks[0][0] if self.asks else None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """获取中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / Decimal("2.0")
        return None


@dataclass
class TradingParameters:
    """交易参数配置"""
    # 滑点模拟
    base_slippage_rate: Decimal = Decimal("0.0001")  # 基础滑点 0.01%
    impact_coefficient: Decimal = Decimal("0.0001")   # 市场冲击系数
    
    # 手续费模拟
    maker_fee_rate: Decimal = Decimal("0.0002")      # Maker 0.02%
    taker_fee_rate: Decimal = Decimal("0.0004")      # Taker 0.04% 
    
    # 执行延迟模拟
    min_execution_delay_ms: int = 10    # 最小执行延迟
    max_execution_delay_ms: int = 100   # 最大执行延迟
    
    # 拒单率模拟
    rejection_rate: Decimal = Decimal("0.001")  # 0.1% 拒单率


class EnhancedPaperTradingEngine(LoggerMixin):
    """增强虚拟盘交易引擎"""
    
    def __init__(self, 
                 account_id: str = "enhanced_paper_001",
                 initial_balance: Decimal = Decimal("100000.0"),
                 trading_params: Optional[TradingParameters] = None):
        
        self.account = EnhancedVirtualAccount(
            account_id=account_id,
            initial_balance=initial_balance
        )
        self.trading_params = trading_params or TradingParameters()
        
        # 持仓管理
        self.positions: Dict[str, EnhancedVirtualPosition] = {}
        
        # 订单管理  
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # 市场数据
        self.market_prices: Dict[str, Decimal] = {}
        self.market_depths: Dict[str, MarketDepth] = {}
        self.last_market_update: Dict[str, datetime] = {}
        
        # 合约配置
        self.contract_configs: Dict[str, Dict[str, Any]] = {}
        
        # 事件订阅
        self.event_callbacks: Dict[str, List[callable]] = {}
        
        # 性能统计
        self.performance_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_trades": 0,
            "total_volume": Decimal("0.0"),
            "max_drawdown": Decimal("0.0"),
            "peak_balance": Decimal("0.0"),
            "sharpe_ratio": Decimal("0.0"),
            "win_rate": Decimal("0.0"),
            "profit_factor": Decimal("0.0")
        }
        
        # 风险控制
        self.risk_limits = {
            "max_position_count": 20,
            "max_correlation_exposure": Decimal("0.3"),  # 最大相关性暴露
            "max_leverage_per_position": 10
        }
        
        # 订单簿模拟
        self._simulate_market_depth = True
        self._depth_levels = 10
        
    async def initialize(self):
        """初始化引擎"""
        self.log_info(
            "Initializing EnhancedPaperTradingEngine",
            account_id=self.account.account_id,
            initial_balance=str(self.account.initial_balance)
        )
        
        # 初始化默认合约配置
        await self._load_contract_configs()
        
        # 设置初始统计
        self.performance_stats["peak_balance"] = self.account.initial_balance
        
        self.log_info("Enhanced paper trading engine initialized successfully")
    
    async def _load_contract_configs(self):
        """加载合约配置"""
        # 默认合约配置
        default_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
            "DOGEUSDT", "XRPUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT"
        ]
        
        for symbol in default_symbols:
            self.contract_configs[symbol] = {
                "contract_size": Decimal("1.0"),
                "tick_size": Decimal("0.01"),
                "min_quantity": Decimal("0.001"),
                "max_quantity": Decimal("1000.0"),
                "maintenance_margin_rate": Decimal("0.005"),
                "max_leverage": 20,
                "base_price": self._get_base_price(symbol)
            }
    
    def _get_base_price(self, symbol: str) -> Decimal:
        """获取基础价格"""
        base_prices = {
            "BTCUSDT": Decimal("50000.0"),
            "ETHUSDT": Decimal("3000.0"),
            "BNBUSDT": Decimal("400.0"),
            "ADAUSDT": Decimal("1.5"),
            "SOLUSDT": Decimal("100.0"),
            "DOGEUSDT": Decimal("0.1"),
            "XRPUSDT": Decimal("0.5"),
            "DOTUSDT": Decimal("30.0"),
            "AVAXUSDT": Decimal("80.0"),
            "LINKUSDT": Decimal("25.0")
        }
        return base_prices.get(symbol, Decimal("1000.0"))
    
    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """执行订单"""
        self.log_info(
            "Executing enhanced paper order",
            symbol=order.symbol,
            side=order.side.value,
            type=order.order_type.value,
            quantity=order.quantity,
            price=order.price
        )
        
        # 重置日统计
        self.account.reset_daily_stats()
        
        # 生成订单ID
        if not order.client_order_id:
            order.client_order_id = f"enhanced_paper_{uuid.uuid4().hex[:8]}"
            
        order_id = f"ep_{int(time.time() * 1000)}{len(self.active_orders)}"
        order.order_id = order_id
        order.status = OrderStatus.NEW
        order.created_at = datetime.utcnow()
        
        try:
            # 预执行风险检查
            risk_check = await self._pre_execution_risk_check(order)
            if not risk_check["passed"]:
                order.status = OrderStatus.REJECTED
                self.performance_stats["rejected_orders"] += 1
                return {
                    "orderId": order_id,
                    "status": "REJECTED",
                    "reason": risk_check["reason"]
                }
            
            # 添加到活跃订单
            self.active_orders[order.client_order_id] = order
            self.performance_stats["total_orders"] += 1
            
            # 模拟执行延迟
            await self._simulate_execution_delay()
            
            # 获取市场数据
            market_price = await self._get_current_market_price(order.symbol)
            depth = await self._get_market_depth(order.symbol)
            
            # 根据订单类型执行
            if order.order_type == OrderType.MARKET:
                return await self._execute_market_order(order, market_price, depth)
            elif order.order_type == OrderType.LIMIT:
                return await self._execute_limit_order(order, market_price, depth)
            elif order.order_type in [OrderType.STOP, OrderType.STOP_MARKET]:
                return await self._execute_stop_order(order, market_price)
            else:
                # 其他订单类型挂单处理
                order.status = OrderStatus.NEW
                return {
                    "orderId": order_id,
                    "symbol": order.symbol,
                    "status": order.status.value,
                    "clientOrderId": order.client_order_id
                }
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.log_error(f"Order execution failed: {e}")
            return {
                "orderId": order_id,
                "status": "REJECTED",
                "error": str(e)
            }
    
    async def _pre_execution_risk_check(self, order: Order) -> Dict[str, Any]:
        """预执行风险检查"""
        # 检查账户余额
        estimated_margin = self._calculate_required_margin(order)
        if self.account.available_balance < estimated_margin:
            return {
                "passed": False,
                "reason": "Insufficient available balance"
            }
        
        # 检查日亏损限制
        if self.account.daily_loss >= self.account.max_daily_loss:
            return {
                "passed": False,
                "reason": "Daily loss limit exceeded"
            }
        
        # 检查持仓数量限制
        if len(self.positions) >= self.risk_limits["max_position_count"]:
            return {
                "passed": False,
                "reason": "Maximum position count exceeded"
            }
        
        # 检查单个仓位大小限制
        if order.price:
            position_value = Decimal(str(order.quantity)) * Decimal(str(order.price))
        else:
            # 市价单使用当前市价估算
            market_price = self.market_prices.get(
                order.symbol, 
                self.contract_configs.get(order.symbol, {}).get("base_price", Decimal("1000.0"))
            )
            position_value = Decimal(str(order.quantity)) * market_price
            
        if position_value > self.account.max_position_size:
            return {
                "passed": False,
                "reason": "Position size exceeds maximum limit"
            }
        
        return {"passed": True}
    
    def _calculate_required_margin(self, order: Order) -> Decimal:
        """计算所需保证金"""
        if not order.price:
            # 市价单使用当前市价估算
            market_price = self.market_prices.get(
                order.symbol, 
                self.contract_configs.get(order.symbol, {}).get("base_price", Decimal("1000.0"))
            )
        else:
            market_price = Decimal(str(order.price))
        
        position_value = Decimal(str(order.quantity)) * market_price
        leverage = min(
            self.account.max_leverage,
            self.contract_configs.get(order.symbol, {}).get("max_leverage", 1)
        )
        
        return position_value / Decimal(str(leverage))
    
    async def _simulate_execution_delay(self):
        """模拟执行延迟"""
        import random
        delay_ms = random.randint(
            self.trading_params.min_execution_delay_ms,
            self.trading_params.max_execution_delay_ms
        )
        await asyncio.sleep(delay_ms / 1000.0)
    
    async def _get_current_market_price(self, symbol: str) -> Decimal:
        """获取当前市场价格"""
        if symbol in self.market_prices:
            return self.market_prices[symbol]
        
        # 使用配置的基础价格并加入随机波动
        base_price = self.contract_configs.get(symbol, {}).get("base_price", Decimal("1000.0"))
        
        # 添加1%的随机波动
        import random
        volatility = Decimal(str(random.uniform(-0.01, 0.01)))
        return base_price * (Decimal("1.0") + volatility)
    
    async def _get_market_depth(self, symbol: str) -> MarketDepth:
        """获取市场深度"""
        if symbol in self.market_depths:
            return self.market_depths[symbol]
        
        # 生成模拟深度数据
        current_price = await self._get_current_market_price(symbol)
        tick_size = self.contract_configs.get(symbol, {}).get("tick_size", Decimal("0.01"))
        
        bids = []
        asks = []
        
        # 生成买单深度
        for i in range(self._depth_levels):
            price = current_price - (Decimal(str(i + 1)) * tick_size)
            quantity = Decimal(str(100 - i * 5))  # 递减数量
            bids.append((price, quantity))
        
        # 生成卖单深度
        for i in range(self._depth_levels):
            price = current_price + (Decimal(str(i + 1)) * tick_size)
            quantity = Decimal(str(100 - i * 5))
            asks.append((price, quantity))
        
        depth = MarketDepth(symbol=symbol, bids=bids, asks=asks)
        self.market_depths[symbol] = depth
        return depth
    
    async def _execute_market_order(self, order: Order, market_price: Decimal, depth: MarketDepth) -> Dict[str, Any]:
        """执行市价单"""
        # 计算执行价格（包含滑点和市场冲击）
        execution_price = await self._calculate_execution_price(order, market_price, depth)
        
        # 计算手续费（市价单为taker）
        commission = self._calculate_commission(order, execution_price, is_maker=False)
        
        # 执行成交
        await self._fill_order(order, execution_price, Decimal(str(order.quantity)), commission)
        
        return {
            "orderId": order.order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "clientOrderId": order.client_order_id,
            "executedQty": str(order.executed_qty),
            "avgPrice": str(order.avg_price),
            "commission": str(commission),
            "transactTime": int(order.updated_at.timestamp() * 1000)
        }
    
    async def _execute_limit_order(self, order: Order, market_price: Decimal, depth: MarketDepth) -> Dict[str, Any]:
        """执行限价单"""
        order_price = Decimal(str(order.price))
        
        # 检查是否能立即成交
        if self._can_fill_limit_order(order, market_price):
            # 立即成交，按限价执行
            commission = self._calculate_commission(order, order_price, is_maker=True)
            await self._fill_order(order, order_price, Decimal(str(order.quantity)), commission)
        else:
            # 挂单等待
            order.status = OrderStatus.NEW
        
        return {
            "orderId": order.order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "clientOrderId": order.client_order_id,
            "executedQty": str(order.executed_qty) if order.executed_qty else "0",
            "avgPrice": str(order.avg_price) if order.avg_price else "0"
        }
    
    async def _execute_stop_order(self, order: Order, market_price: Decimal) -> Dict[str, Any]:
        """执行止损单"""
        order.status = OrderStatus.NEW  # 止损单挂单等待触发
        
        return {
            "orderId": order.order_id,
            "symbol": order.symbol,
            "status": order.status.value,
            "clientOrderId": order.client_order_id,
            "stopPrice": str(order.stop_price) if order.stop_price else None
        }
    
    async def _calculate_execution_price(self, order: Order, market_price: Decimal, depth: MarketDepth) -> Decimal:
        """计算执行价格（含滑点和市场冲击）"""
        order_quantity = Decimal(str(order.quantity))
        
        # 基础滑点
        base_slippage = self.trading_params.base_slippage_rate
        
        # 市场冲击计算（基于订单大小和深度）
        total_depth = sum(qty for _, qty in (depth.bids if order.side == OrderSide.SELL else depth.asks))
        market_impact = order_quantity / total_depth * self.trading_params.impact_coefficient
        
        # 总滑点
        total_slippage = base_slippage + market_impact
        
        if order.side == OrderSide.BUY:
            return market_price * (Decimal("1.0") + total_slippage)
        else:
            return market_price * (Decimal("1.0") - total_slippage)
    
    def _calculate_commission(self, order: Order, execution_price: Decimal, is_maker: bool) -> Decimal:
        """计算手续费"""
        trade_value = Decimal(str(order.quantity)) * execution_price
        fee_rate = self.trading_params.maker_fee_rate if is_maker else self.trading_params.taker_fee_rate
        return trade_value * fee_rate
    
    def _can_fill_limit_order(self, order: Order, market_price: Decimal) -> bool:
        """判断限价单是否能成交"""
        if not order.price:
            return False
            
        order_price = Decimal(str(order.price))
        
        if order.side == OrderSide.BUY:
            return market_price <= order_price
        else:
            return market_price >= order_price
    
    async def _fill_order(self, order: Order, execution_price: Decimal, fill_quantity: Decimal, commission: Decimal):
        """执行订单成交"""
        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.executed_qty = float(fill_quantity)
        order.avg_price = float(execution_price)
        order.updated_at = datetime.utcnow()
        
        # 创建成交记录
        trade = {
            "trade_id": f"trade_{uuid.uuid4().hex[:8]}",
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": float(fill_quantity),
            "price": float(execution_price),
            "commission": float(commission),
            "timestamp": datetime.utcnow()
        }
        self.trade_history.append(trade)
        
        # 更新仓位
        await self._update_position(order, execution_price, fill_quantity)
        
        # 更新账户
        await self._update_account(order, execution_price, fill_quantity, commission)
        
        # 移到历史记录
        self.order_history.append(order)
        if order.client_order_id in self.active_orders:
            del self.active_orders[order.client_order_id]
        
        # 更新统计
        await self._update_performance_stats(trade)
        
        # 触发成交事件
        await self._emit_event("order_filled", {
            "order": order,
            "trade": trade
        })
        
        self.log_info(
            "Order filled in enhanced engine",
            order_id=order.order_id,
            symbol=order.symbol,
            price=float(execution_price),
            quantity=float(fill_quantity),
            commission=float(commission)
        )
    
    async def _update_position(self, order: Order, execution_price: Decimal, fill_quantity: Decimal):
        """更新仓位"""
        symbol = order.symbol
        side = "LONG" if order.side == OrderSide.BUY else "SHORT"
        
        if symbol not in self.positions:
            # 新建仓位
            required_margin = self._calculate_required_margin(order)
            
            position = EnhancedVirtualPosition(
                symbol=symbol,
                side=side,
                size=fill_quantity,
                entry_price=execution_price,
                current_price=execution_price,
                margin=required_margin,
                leverage=min(self.account.max_leverage, 
                           self.contract_configs.get(symbol, {}).get("max_leverage", 1)),
                margin_type=self.account.margin_type
            )
            
            self.positions[symbol] = position
            self.log_debug(f"New enhanced position: {symbol} {side} {fill_quantity}@{execution_price}")
        
        else:
            # 更新现有仓位（简化处理，详细实现需要考虑双向持仓等复杂情况）
            position = self.positions[symbol]
            
            if position.side == side:
                # 同方向加仓
                total_value = position.size * position.entry_price + fill_quantity * execution_price
                total_size = position.size + fill_quantity
                position.entry_price = total_value / total_size
                position.size = total_size
            else:
                # 反方向操作（平仓或反向开仓）
                if position.size > fill_quantity:
                    # 部分平仓
                    position.size -= fill_quantity
                    # 计算已实现盈亏
                    if position.side == "LONG":
                        realized_pnl = (execution_price - position.entry_price) * fill_quantity
                    else:
                        realized_pnl = (position.entry_price - execution_price) * fill_quantity
                    
                    position.realized_pnl += realized_pnl
                    self.account.total_pnl += realized_pnl
                    
                elif position.size == fill_quantity:
                    # 完全平仓
                    if position.side == "LONG":
                        realized_pnl = (execution_price - position.entry_price) * fill_quantity
                    else:
                        realized_pnl = (position.entry_price - execution_price) * fill_quantity
                    
                    self.account.total_pnl += realized_pnl
                    del self.positions[symbol]
                    
                else:
                    # 平仓并反向开仓
                    close_quantity = position.size
                    remain_quantity = fill_quantity - close_quantity
                    
                    # 平仓盈亏
                    if position.side == "LONG":
                        realized_pnl = (execution_price - position.entry_price) * close_quantity
                    else:
                        realized_pnl = (position.entry_price - execution_price) * close_quantity
                    
                    self.account.total_pnl += realized_pnl
                    
                    # 开新仓
                    position.side = side
                    position.size = remain_quantity
                    position.entry_price = execution_price
                    position.realized_pnl += realized_pnl
            
            position.current_price = execution_price
            position.update_pnl(execution_price)
    
    async def _update_account(self, order: Order, execution_price: Decimal, fill_quantity: Decimal, commission: Decimal):
        """更新账户余额"""
        # 扣除手续费
        self.account.current_balance -= commission
        self.account.available_balance -= commission
        self.account.total_commission += commission
        
        # 更新已用保证金
        required_margin = self._calculate_required_margin(order)
        if order.side == OrderSide.BUY:
            self.account.margin_used += required_margin
            self.account.available_balance -= required_margin
        # 卖单释放保证金的逻辑需要基于具体仓位情况
    
    async def _update_performance_stats(self, trade: Dict[str, Any]):
        """更新性能统计"""
        self.performance_stats["filled_orders"] += 1
        self.performance_stats["total_trades"] += 1
        self.performance_stats["total_volume"] += Decimal(str(trade["quantity"] * trade["price"]))
        
        # 更新最大回撤
        current_equity = self.account.current_balance + self.account.total_pnl
        if current_equity > self.performance_stats["peak_balance"]:
            self.performance_stats["peak_balance"] = current_equity
        
        drawdown = (self.performance_stats["peak_balance"] - current_equity) / self.performance_stats["peak_balance"]
        if drawdown > self.performance_stats["max_drawdown"]:
            self.performance_stats["max_drawdown"] = drawdown
    
    async def _emit_event(self, event_type: str, data: Any):
        """触发事件"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    self.log_error(f"Event callback error: {e}")
    
    def subscribe_event(self, event_type: str, callback: callable):
        """订阅事件"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def update_market_data(self, market_data: MarketData):
        """更新市场数据"""
        symbol = market_data.symbol
        price = Decimal(str(market_data.price))
        
        self.market_prices[symbol] = price
        self.last_market_update[symbol] = datetime.utcnow()
        
        # 更新持仓盈亏
        if symbol in self.positions:
            contract_size = self.contract_configs.get(symbol, {}).get("contract_size", Decimal("1.0"))
            self.positions[symbol].update_pnl(price, contract_size)
        
        # 处理挂单
        await self.process_pending_orders()
        
        # 检查强平
        await self._check_liquidation(symbol)
    
    async def process_pending_orders(self):
        """处理挂单"""
        for client_order_id, order in list(self.active_orders.items()):
            if order.status != OrderStatus.NEW:
                continue
            
            market_price = await self._get_current_market_price(order.symbol)
            
            # 处理限价单
            if order.order_type == OrderType.LIMIT and self._can_fill_limit_order(order, market_price):
                commission = self._calculate_commission(order, Decimal(str(order.price)), is_maker=True)
                await self._fill_order(order, Decimal(str(order.price)), Decimal(str(order.quantity)), commission)
            
            # 处理止损单
            elif order.order_type in [OrderType.STOP, OrderType.STOP_MARKET] and order.stop_price:
                stop_price = Decimal(str(order.stop_price))
                if ((order.side == OrderSide.BUY and market_price >= stop_price) or
                    (order.side == OrderSide.SELL and market_price <= stop_price)):
                    
                    # 止损触发，转为市价单执行
                    depth = await self._get_market_depth(order.symbol)
                    execution_price = await self._calculate_execution_price(order, market_price, depth)
                    commission = self._calculate_commission(order, execution_price, is_maker=False)
                    await self._fill_order(order, execution_price, Decimal(str(order.quantity)), commission)
    
    async def _check_liquidation(self, symbol: str):
        """检查强平"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if not position.liquidation_price:
            return
        
        current_price = self.market_prices.get(symbol)
        if not current_price:
            return
        
        # 检查是否触发强平
        should_liquidate = False
        if position.side == "LONG" and current_price <= position.liquidation_price:
            should_liquidate = True
        elif position.side == "SHORT" and current_price >= position.liquidation_price:
            should_liquidate = True
        
        if should_liquidate:
            await self._force_liquidation(position)
    
    async def _force_liquidation(self, position: EnhancedVirtualPosition):
        """强制平仓"""
        # 创建强平订单
        liquidation_order = Order(
            symbol=position.symbol,
            side=OrderSide.SELL if position.side == "LONG" else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=float(position.size),
            client_order_id=f"liquidation_{uuid.uuid4().hex[:8]}"
        )
        
        # 执行强平
        result = await self.execute_order(liquidation_order)
        
        self.log_warning(
            "Position liquidated",
            symbol=position.symbol,
            side=position.side,
            size=float(position.size),
            liquidation_price=float(position.liquidation_price) if position.liquidation_price else None
        )
        
        # 触发强平事件
        await self._emit_event("position_liquidated", {
            "position": position,
            "liquidation_order": liquidation_order
        })
    
    def get_enhanced_account_info(self) -> Dict[str, Any]:
        """获取增强账户信息"""
        total_unrealized_pnl = sum(
            float(pos.unrealized_pnl) for pos in self.positions.values()
        )
        total_equity = float(self.account.current_balance) + total_unrealized_pnl
        
        return {
            "accountId": self.account.account_id,
            "balance": float(self.account.current_balance),
            "availableBalance": float(self.account.available_balance),
            "marginUsed": float(self.account.margin_used),
            "totalPnL": float(self.account.total_pnl),
            "unrealizedPnL": total_unrealized_pnl,
            "totalEquity": total_equity,
            "totalCommission": float(self.account.total_commission),
            "dailyLoss": float(self.account.daily_loss),
            "maxLeverage": self.account.max_leverage,
            "positionMode": self.account.position_mode.value,
            "marginType": self.account.margin_type,
            "positions": len(self.positions),
            "activeOrders": len(self.active_orders),
            "riskMetrics": self._calculate_risk_metrics()
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """计算风险指标"""
        total_position_value = sum(
            float(pos.size * pos.current_price) for pos in self.positions.values()
        )
        
        total_equity = float(self.account.current_balance + self.account.total_pnl)
        
        return {
            "portfolioValue": total_position_value,
            "leverageRatio": total_position_value / total_equity if total_equity > 0 else 0,
            "marginUtilization": float(self.account.margin_used) / float(self.account.current_balance) if self.account.current_balance > 0 else 0,
            "positionCount": len(self.positions),
            "maxDrawdown": float(self.performance_stats["max_drawdown"]),
            "dailyLossUsage": float(self.account.daily_loss) / float(self.account.max_daily_loss) if self.account.max_daily_loss > 0 else 0
        }
    
    def get_enhanced_performance_stats(self) -> Dict[str, Any]:
        """获取增强性能统计"""
        win_trades = sum(1 for trade in self.trade_history if self._is_winning_trade(trade))
        total_trades = len(self.trade_history)
        
        profit_trades = [t for t in self.trade_history if self._is_winning_trade(t)]
        loss_trades = [t for t in self.trade_history if not self._is_winning_trade(t)]
        
        avg_profit = sum(self._calculate_trade_pnl(t) for t in profit_trades) / len(profit_trades) if profit_trades else 0
        avg_loss = sum(abs(self._calculate_trade_pnl(t)) for t in loss_trades) / len(loss_trades) if loss_trades else 0
        
        profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
        
        stats = self.performance_stats.copy()
        stats.update({
            "winRate": win_trades / total_trades if total_trades > 0 else 0,
            "totalTrades": total_trades,
            "avgProfit": avg_profit,
            "avgLoss": avg_loss,
            "profitFactor": profit_factor,
            "totalVolume": float(self.performance_stats["total_volume"]),
            "avgCommission": float(self.account.total_commission) / total_trades if total_trades > 0 else 0,
            "returnRate": (float(self.account.current_balance) - float(self.account.initial_balance)) / float(self.account.initial_balance),
            "sharpeRatio": self._calculate_sharpe_ratio()
        })
        
        return stats
    
    def _is_winning_trade(self, trade: Dict[str, Any]) -> bool:
        """判断是否为盈利交易"""
        # 简化实现，实际需要跟踪每笔交易的完整盈亏
        return self._calculate_trade_pnl(trade) > 0
    
    def _calculate_trade_pnl(self, trade: Dict[str, Any]) -> float:
        """计算单笔交易盈亏"""
        # 这里需要实现更复杂的逻辑，跟踪开仓和平仓的配对
        # 简化处理：假设每笔交易都有对应的反向交易
        return 0.0  # 占位实现
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        # 简化实现，实际需要基于历史收益率计算
        if not self.trade_history:
            return 0.0
        
        # 这里应该使用日收益率序列计算
        returns = []  # 收益率序列
        if len(returns) < 2:
            return 0.0
        
        import statistics
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        return avg_return / std_return if std_return > 0 else 0.0