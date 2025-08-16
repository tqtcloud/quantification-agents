import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError
from src.core.models import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce, PositionSide
)
from src.utils.logger import LoggerMixin


class BinanceOrderType(str, Enum):
    """币安期货订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"


class WorkingType(str, Enum):
    """止损止盈触发价格类型"""
    MARK_PRICE = "MARK_PRICE"
    CONTRACT_PRICE = "CONTRACT_PRICE"


class SelfTradePreventionMode(str, Enum):
    """自成交防护模式"""
    NONE = "NONE"
    EXPIRE_TAKER = "EXPIRE_TAKER"
    EXPIRE_MAKER = "EXPIRE_MAKER"
    EXPIRE_BOTH = "EXPIRE_BOTH"


@dataclass
class OrderValidationRule:
    """订单验证规则"""
    min_quantity: float = 0.001
    max_quantity: float = 1000000.0
    min_price: float = 0.01
    max_price: float = 1000000.0
    tick_size: float = 0.01
    step_size: float = 0.001
    min_notional: float = 5.0  # 最小名义价值USDT
    max_num_orders: int = 200  # 最大挂单数量


@dataclass
class BatchOrderRequest:
    """批量订单请求"""
    orders: List[Order]
    batch_id: str = field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConditionalOrderParams:
    """条件订单参数"""
    condition_type: str  # "PRICE_ABOVE", "PRICE_BELOW", "TIME_TRIGGER"
    condition_value: float
    trigger_time: Optional[datetime] = None
    is_active: bool = True


@dataclass
class OrderRiskCheck:
    """订单风险检查结果"""
    is_valid: bool
    risk_score: float  # 0-1风险评分
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_position_value: float = 0.0
    leverage_ratio: float = 0.0


class BinanceQuantOrderManager(LoggerMixin):
    """币安量化订单管理器"""
    
    def __init__(self, binance_client: BinanceFuturesClient):
        self.client = binance_client
        
        # 交易对配置缓存
        self.symbol_configs: Dict[str, Dict[str, Any]] = {}
        
        # 订单管理
        self.active_orders: Dict[str, Order] = {}
        self.conditional_orders: Dict[str, tuple[Order, ConditionalOrderParams]] = {}
        self.batch_orders: Dict[str, BatchOrderRequest] = {}
        
        # 风险控制参数
        self.max_daily_orders = 1000
        self.max_position_value_ratio = 0.8  # 最大仓位价值占账户资金比例
        self.max_leverage = 20
        
        # 统计信息
        self.daily_order_count = 0
        self.last_reset_date = datetime.utcnow().date()
        
        self.order_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "rejected_orders": 0,
            "avg_fill_time": 0.0
        }
    
    async def initialize(self):
        """初始化管理器"""
        self.log_info("Initializing BinanceQuantOrderManager")
        
        # 获取交易规则
        await self._load_exchange_info()
        
        # 重置日统计
        self._reset_daily_stats_if_needed()
        
        self.log_info("BinanceQuantOrderManager initialized successfully")
    
    async def _load_exchange_info(self):
        """加载交易所交易规则"""
        try:
            exchange_info = await self.client.get_exchange_info()
            
            for symbol_info in exchange_info.get("symbols", []):
                symbol = symbol_info["symbol"]
                
                # 解析过滤器
                filters = {f["filterType"]: f for f in symbol_info.get("filters", [])}
                
                self.symbol_configs[symbol] = {
                    "status": symbol_info["status"],
                    "baseAsset": symbol_info["baseAsset"],
                    "quoteAsset": symbol_info["quoteAsset"],
                    "pricePrecision": symbol_info["pricePrecision"],
                    "quantityPrecision": symbol_info["quantityPrecision"],
                    "filters": filters,
                    "orderTypes": symbol_info.get("orderTypes", [])
                }
            
            self.log_info(f"Loaded exchange info for {len(self.symbol_configs)} symbols")
            
        except Exception as e:
            self.log_error(f"Failed to load exchange info: {e}")
            raise
    
    def _get_validation_rules(self, symbol: str) -> OrderValidationRule:
        """获取交易对验证规则"""
        if symbol not in self.symbol_configs:
            return OrderValidationRule()
        
        config = self.symbol_configs[symbol]
        filters = config.get("filters", {})
        
        # PRICE_FILTER
        price_filter = filters.get("PRICE_FILTER", {})
        min_price = float(price_filter.get("minPrice", 0.01))
        max_price = float(price_filter.get("maxPrice", 1000000.0))
        tick_size = float(price_filter.get("tickSize", 0.01))
        
        # LOT_SIZE
        lot_filter = filters.get("LOT_SIZE", {})
        min_qty = float(lot_filter.get("minQty", 0.001))
        max_qty = float(lot_filter.get("maxQty", 1000000.0))
        step_size = float(lot_filter.get("stepSize", 0.001))
        
        # MIN_NOTIONAL
        notional_filter = filters.get("MIN_NOTIONAL", {})
        min_notional = float(notional_filter.get("notional", 5.0))
        
        # MAX_NUM_ORDERS
        max_orders_filter = filters.get("MAX_NUM_ORDERS", {})
        max_orders = int(max_orders_filter.get("maxNumOrders", 200))
        
        return OrderValidationRule(
            min_quantity=min_qty,
            max_quantity=max_qty,
            min_price=min_price,
            max_price=max_price,
            tick_size=tick_size,
            step_size=step_size,
            min_notional=min_notional,
            max_num_orders=max_orders
        )
    
    async def validate_order(self, order: Order) -> OrderRiskCheck:
        """验证订单参数和风险"""
        violations = []
        warnings = []
        risk_score = 0.0
        
        # 基础参数验证
        if not order.symbol:
            violations.append("Missing symbol")
            
        if order.quantity <= 0:
            violations.append("Invalid quantity")
            
        if order.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT] and (not order.price or order.price <= 0):
            violations.append("Invalid price for limit/stop order")
        
        # 获取交易对规则
        rules = self._get_validation_rules(order.symbol)
        
        # 数量验证
        if order.quantity < rules.min_quantity:
            violations.append(f"Quantity {order.quantity} below minimum {rules.min_quantity}")
        
        if order.quantity > rules.max_quantity:
            violations.append(f"Quantity {order.quantity} exceeds maximum {rules.max_quantity}")
        
        # 步长验证
        qty_precision = round(order.quantity / rules.step_size) * rules.step_size
        if abs(order.quantity - qty_precision) > 1e-8:
            violations.append(f"Quantity {order.quantity} does not match step size {rules.step_size}")
        
        # 价格验证
        if order.price:
            if order.price < rules.min_price:
                violations.append(f"Price {order.price} below minimum {rules.min_price}")
            
            if order.price > rules.max_price:
                violations.append(f"Price {order.price} exceeds maximum {rules.max_price}")
            
            # 价格精度验证
            price_precision = round(order.price / rules.tick_size) * rules.tick_size
            if abs(order.price - price_precision) > 1e-8:
                violations.append(f"Price {order.price} does not match tick size {rules.tick_size}")
            
            # 名义价值验证
            notional_value = order.quantity * order.price
            if notional_value < rules.min_notional:
                violations.append(f"Notional value {notional_value} below minimum {rules.min_notional}")
        
        # 日交易限制验证
        self._reset_daily_stats_if_needed()
        if self.daily_order_count >= self.max_daily_orders:
            violations.append(f"Daily order limit {self.max_daily_orders} reached")
        
        # 活跃订单数量检查
        symbol_active_orders = sum(1 for o in self.active_orders.values() if o.symbol == order.symbol)
        if symbol_active_orders >= rules.max_num_orders:
            violations.append(f"Maximum active orders {rules.max_num_orders} for {order.symbol} reached")
        
        # 风险评分计算
        if order.price and order.quantity > 0:
            position_value = order.quantity * order.price
            
            # 基于订单大小的风险
            if position_value > 10000:  # >$10k
                risk_score += 0.3
            elif position_value > 1000:  # >$1k
                risk_score += 0.1
            
            # 基于订单类型的风险
            if order.order_type in [OrderType.MARKET, OrderType.STOP_MARKET]:
                risk_score += 0.2  # 市价单风险较高
            
        # 风险警告
        if risk_score > 0.7:
            warnings.append("High risk order detected")
        elif risk_score > 0.5:
            warnings.append("Medium risk order detected")
        
        return OrderRiskCheck(
            is_valid=len(violations) == 0,
            risk_score=risk_score,
            violations=violations,
            warnings=warnings
        )
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: BinanceOrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        position_side: PositionSide = PositionSide.BOTH,
        client_order_id: Optional[str] = None,
        working_type: WorkingType = WorkingType.CONTRACT_PRICE,
        price_protect: bool = False,
        self_trade_prevention_mode: SelfTradePreventionMode = SelfTradePreventionMode.NONE,
        callback_rate: Optional[float] = None,  # 追踪止损回调比率
        activation_price: Optional[float] = None,  # 追踪止损激活价格
        **kwargs
    ) -> Dict[str, Any]:
        """下单"""
        if not client_order_id:
            client_order_id = f"quant_{uuid.uuid4().hex[:12]}"
        
        # 创建订单对象进行验证
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType(order_type.value),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            position_side=position_side,
            client_order_id=client_order_id
        )
        
        # 验证订单
        validation = await self.validate_order(order)
        if not validation.is_valid:
            raise ValueError(f"Order validation failed: {', '.join(validation.violations)}")
        
        if validation.warnings:
            self.log_warning(f"Order warnings: {', '.join(validation.warnings)}")
        
        # 构建API参数
        params = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": str(quantity),
            "timeInForce": time_in_force.value,
            "reduceOnly": str(reduce_only).lower(),
            "positionSide": position_side.value,
            "newClientOrderId": client_order_id,
            "workingType": working_type.value,
            "priceProtect": str(price_protect).lower(),
            "selfTradePreventionMode": self_trade_prevention_mode.value
        }
        
        # 添加条件参数
        if price is not None:
            params["price"] = str(price)
        
        if stop_price is not None:
            params["stopPrice"] = str(stop_price)
        
        # 追踪止损特殊参数
        if order_type == BinanceOrderType.TRAILING_STOP_MARKET:
            if callback_rate is not None:
                params["callbackRate"] = str(callback_rate)
            if activation_price is not None:
                params["activationPrice"] = str(activation_price)
        
        # 添加其他参数
        params.update(kwargs)
        
        try:
            self.log_info(
                "Placing order",
                symbol=symbol,
                side=side.value,
                type=order_type.value,
                quantity=quantity,
                price=price,
                client_order_id=client_order_id
            )
            
            # 调用币安API
            result = await self.client.place_order(**params)
            
            # 添加到活跃订单
            order.order_id = str(result.get("orderId", ""))
            order.status = OrderStatus(result.get("status", "NEW"))
            self.active_orders[client_order_id] = order
            
            # 更新统计
            self.daily_order_count += 1
            self.order_stats["total_orders"] += 1
            if result.get("status") == "FILLED":
                self.order_stats["successful_orders"] += 1
            
            self.log_info(
                "Order placed successfully",
                order_id=result.get("orderId"),
                client_order_id=client_order_id,
                status=result.get("status")
            )
            
            return result
            
        except BinanceAPIError as e:
            self.order_stats["failed_orders"] += 1
            if e.code in [-2010, -2021]:  # 余额不足、订单被拒绝
                self.order_stats["rejected_orders"] += 1
            self.log_error(f"Failed to place order: {e}")
            raise
        except Exception as e:
            self.order_stats["failed_orders"] += 1
            self.log_error(f"Unexpected error placing order: {e}")
            raise
    
    async def place_batch_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量下单"""
        if len(orders) > 5:  # 币安批量下单限制
            raise ValueError("Batch size cannot exceed 5 orders")
        
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        
        try:
            self.log_info(f"Placing batch orders", batch_id=batch_id, count=len(orders))
            
            # 构建批量请求
            batch_orders = []
            for i, order_params in enumerate(orders):
                if "newClientOrderId" not in order_params:
                    order_params["newClientOrderId"] = f"{batch_id}_{i}"
                batch_orders.append(order_params)
            
            # TODO: 币安API目前不支持批量下单，这里逐个下单
            results = []
            for order_params in batch_orders:
                try:
                    result = await self.place_order(**order_params)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "clientOrderId": order_params.get("newClientOrderId")})
            
            return results
            
        except Exception as e:
            self.log_error(f"Batch order failed: {e}")
            raise
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """取消订单"""
        try:
            result = await self.client.cancel_order(symbol, order_id, client_order_id)
            
            # 从活跃订单中移除
            if client_order_id and client_order_id in self.active_orders:
                del self.active_orders[client_order_id]
            
            self.log_info(
                "Order cancelled",
                symbol=symbol,
                order_id=order_id,
                client_order_id=client_order_id
            )
            
            return result
            
        except Exception as e:
            self.log_error(f"Failed to cancel order: {e}")
            raise
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """取消指定交易对的所有订单"""
        try:
            result = await self.client.cancel_all_orders(symbol)
            
            # 清理活跃订单
            self.active_orders = {
                cid: order for cid, order in self.active_orders.items()
                if order.symbol != symbol
            }
            
            self.log_info(f"All orders cancelled for {symbol}")
            return result
            
        except Exception as e:
            self.log_error(f"Failed to cancel all orders for {symbol}: {e}")
            raise
    
    async def modify_order(
        self,
        symbol: str,
        order_id: int,
        side: OrderSide,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """修改订单"""
        try:
            result = await self.client.modify_order(symbol, order_id, side.value, quantity, price)
            
            self.log_info(
                "Order modified",
                symbol=symbol,
                order_id=order_id,
                new_quantity=quantity,
                new_price=price
            )
            
            return result
            
        except Exception as e:
            self.log_error(f"Failed to modify order: {e}")
            raise
    
    async def get_order_status(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """查询订单状态"""
        try:
            result = await self.client.get_order(symbol, order_id, client_order_id)
            
            # 更新本地状态
            if client_order_id and client_order_id in self.active_orders:
                order = self.active_orders[client_order_id]
                order.status = OrderStatus(result.get("status", "NEW"))
                order.executed_qty = float(result.get("executedQty", 0))
                order.avg_price = float(result.get("avgPrice", 0)) if result.get("avgPrice") else 0
                
                # 如果订单已完成，从活跃订单中移除
                if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                    del self.active_orders[client_order_id]
            
            return result
            
        except Exception as e:
            self.log_error(f"Failed to get order status: {e}")
            raise
    
    def _reset_daily_stats_if_needed(self):
        """重置日统计（如果需要）"""
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = current_date
            self.log_info(f"Daily stats reset for {current_date}")
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计"""
        success_rate = (
            self.order_stats["successful_orders"] / self.order_stats["total_orders"]
            if self.order_stats["total_orders"] > 0 else 0
        )
        
        rejection_rate = (
            self.order_stats["rejected_orders"] / self.order_stats["total_orders"]
            if self.order_stats["total_orders"] > 0 else 0
        )
        
        return {
            **self.order_stats,
            "success_rate": success_rate,
            "rejection_rate": rejection_rate,
            "daily_order_count": self.daily_order_count,
            "active_orders_count": len(self.active_orders),
            "conditional_orders_count": len(self.conditional_orders)
        }
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单列表"""
        return list(self.active_orders.values())
    
    def get_symbol_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取交易对配置"""
        return self.symbol_configs.get(symbol)
    
    async def cleanup(self):
        """清理资源"""
        self.log_info("Cleaning up BinanceQuantOrderManager")
        
        # 可以选择取消所有活跃订单
        # for order in self.active_orders.values():
        #     try:
        #         await self.cancel_order(order.symbol, client_order_id=order.client_order_id)
        #     except Exception as e:
        #         self.log_error(f"Failed to cancel order during cleanup: {e}")
        
        self.active_orders.clear()
        self.conditional_orders.clear()
        self.batch_orders.clear()
        
        self.log_info("BinanceQuantOrderManager cleanup complete")