import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.core.models import Order, OrderSide, OrderType, MarketData
from src.utils.logger import LoggerMixin


class MarketCondition(Enum):
    """市场状况"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_VOLUME = "high_volume"


@dataclass
class SlippageConfig:
    """滑点配置"""
    # 基础滑点
    base_slippage_bps: int = 1  # 基础滑点（基点，1bp = 0.01%）
    
    # 市场冲击模型参数
    impact_coefficient: Decimal = Decimal("0.1")  # 冲击系数
    impact_exponent: Decimal = Decimal("0.6")     # 冲击指数
    
    # 时间衰减参数
    temporary_impact_decay: Decimal = Decimal("0.1")  # 临时冲击衰减率
    permanent_impact_ratio: Decimal = Decimal("0.3")  # 永久冲击比例
    
    # 波动性调整
    volatility_multiplier: Decimal = Decimal("2.0")  # 波动性倍数
    
    # 订单簿深度参数
    avg_depth_usd: Decimal = Decimal("100000")  # 平均深度（美元）
    depth_concentration: Decimal = Decimal("0.3")  # 深度集中度


@dataclass
class CommissionStructure:
    """手续费结构"""
    # 币安期货手续费等级
    maker_tiers: List[Tuple[Decimal, Decimal]] = field(default_factory=lambda: [
        (Decimal("0"), Decimal("0.0002")),        # VIP 0: 0.02%
        (Decimal("250"), Decimal("0.00016")),     # VIP 1: 0.016%
        (Decimal("2500"), Decimal("0.00014")),    # VIP 2: 0.014%
        (Decimal("7500"), Decimal("0.00012")),    # VIP 3: 0.012%
        (Decimal("22500"), Decimal("0.00010")),   # VIP 4: 0.010%
    ])
    
    taker_tiers: List[Tuple[Decimal, Decimal]] = field(default_factory=lambda: [
        (Decimal("0"), Decimal("0.0004")),        # VIP 0: 0.04%
        (Decimal("250"), Decimal("0.00036")),     # VIP 1: 0.036%
        (Decimal("2500"), Decimal("0.00032")),    # VIP 2: 0.032%
        (Decimal("7500"), Decimal("0.00028")),    # VIP 3: 0.028%
        (Decimal("22500"), Decimal("0.00024")),   # VIP 4: 0.024%
    ])
    
    # BNB抵扣优惠
    bnb_discount: Decimal = Decimal("0.1")  # 10%折扣
    
    # 特殊费率
    liquidation_fee: Decimal = Decimal("0.005")  # 强平费 0.5%
    funding_fee_cap: Decimal = Decimal("0.0375")  # 资金费率上限


@dataclass
class MarketDepthLevel:
    """市场深度级别"""
    price: Decimal
    quantity: Decimal
    cumulative_quantity: Decimal


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    bids: List[MarketDepthLevel]
    asks: List[MarketDepthLevel]
    timestamp: datetime
    
    def get_spread(self) -> Decimal:
        """获取买卖价差"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return Decimal("0")
    
    def get_mid_price(self) -> Decimal:
        """获取中间价"""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / Decimal("2")
        return Decimal("0")
    
    def get_depth_at_price(self, price: Decimal, side: str) -> Decimal:
        """获取指定价格的深度"""
        levels = self.bids if side == "BUY" else self.asks
        
        cumulative = Decimal("0")
        for level in levels:
            if side == "BUY" and level.price >= price:
                cumulative += level.quantity
            elif side == "SELL" and level.price <= price:
                cumulative += level.quantity
            else:
                break
        
        return cumulative


class RealTimeSlippageCalculator(LoggerMixin):
    """实时滑点计算器"""
    
    def __init__(self, config: SlippageConfig):
        self.config = config
        self.market_conditions: Dict[str, MarketCondition] = {}
        self.recent_trades: Dict[str, List[Dict[str, Any]]] = {}
        self.volatility_cache: Dict[str, Decimal] = {}
        
    def calculate_slippage(self, 
                          order: Order, 
                          orderbook: OrderBookSnapshot,
                          market_condition: MarketCondition = MarketCondition.NORMAL) -> Tuple[Decimal, Dict[str, Any]]:
        """计算订单滑点"""
        
        order_size_usd = Decimal(str(order.quantity)) * orderbook.get_mid_price()
        
        # 1. 基础滑点
        base_slippage = Decimal(str(self.config.base_slippage_bps)) / Decimal("10000")
        
        # 2. 市场冲击
        market_impact = self._calculate_market_impact(order_size_usd, orderbook)
        
        # 3. 波动性调整
        volatility_adjustment = self._calculate_volatility_adjustment(order.symbol, market_condition)
        
        # 4. 流动性调整
        liquidity_adjustment = self._calculate_liquidity_adjustment(orderbook, market_condition)
        
        # 5. 时间成本
        timing_cost = self._calculate_timing_cost(order, market_condition)
        
        # 总滑点
        total_slippage = (base_slippage + market_impact + 
                         volatility_adjustment + liquidity_adjustment + timing_cost)
        
        # 应用方向
        if order.side == OrderSide.SELL:
            total_slippage = -total_slippage
        
        slippage_breakdown = {
            "base_slippage": float(base_slippage),
            "market_impact": float(market_impact),
            "volatility_adjustment": float(volatility_adjustment),
            "liquidity_adjustment": float(liquidity_adjustment),
            "timing_cost": float(timing_cost),
            "total_slippage": float(total_slippage),
            "market_condition": market_condition.value
        }
        
        return total_slippage, slippage_breakdown
    
    def _calculate_market_impact(self, order_size_usd: Decimal, orderbook: OrderBookSnapshot) -> Decimal:
        """计算市场冲击"""
        # 使用平方根模型：冲击 = 系数 * (订单大小 / 平均深度)^指数
        avg_depth = self.config.avg_depth_usd
        
        if avg_depth <= 0:
            return Decimal("0")
        
        size_ratio = order_size_usd / avg_depth
        impact = self.config.impact_coefficient * (size_ratio ** self.config.impact_exponent)
        
        # 考虑订单簿深度集中度
        actual_depth = self._estimate_order_book_depth(orderbook)
        depth_ratio = actual_depth / avg_depth if avg_depth > 0 else Decimal("1")
        
        # 深度越小，冲击越大
        impact_adjustment = Decimal("1") / depth_ratio if depth_ratio > 0 else Decimal("2")
        
        return impact * impact_adjustment
    
    def _estimate_order_book_depth(self, orderbook: OrderBookSnapshot) -> Decimal:
        """估算订单簿深度"""
        if not orderbook.bids or not orderbook.asks:
            return self.config.avg_depth_usd
        
        # 计算前5档的总深度价值
        bid_depth = sum(level.price * level.quantity for level in orderbook.bids[:5])
        ask_depth = sum(level.price * level.quantity for level in orderbook.asks[:5])
        
        return (bid_depth + ask_depth) / Decimal("2")
    
    def _calculate_volatility_adjustment(self, symbol: str, market_condition: MarketCondition) -> Decimal:
        """计算波动性调整"""
        base_volatility = self._get_symbol_volatility(symbol)
        
        condition_multipliers = {
            MarketCondition.NORMAL: Decimal("1.0"),
            MarketCondition.VOLATILE: Decimal("2.0"),
            MarketCondition.LOW_LIQUIDITY: Decimal("1.5"),
            MarketCondition.HIGH_VOLUME: Decimal("0.8")
        }
        
        multiplier = condition_multipliers.get(market_condition, Decimal("1.0"))
        
        return base_volatility * self.config.volatility_multiplier * multiplier
    
    def _get_symbol_volatility(self, symbol: str) -> Decimal:
        """获取品种波动性"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # 默认波动性（基于历史数据或实时计算）
        default_volatilities = {
            "BTCUSDT": Decimal("0.0002"),  # 0.02%
            "ETHUSDT": Decimal("0.0003"),  # 0.03%
            "BNBUSDT": Decimal("0.0004"),  # 0.04%
            "ADAUSDT": Decimal("0.0005"),  # 0.05%
            "SOLUSDT": Decimal("0.0006"),  # 0.06%
        }
        
        volatility = default_volatilities.get(symbol, Decimal("0.0003"))
        self.volatility_cache[symbol] = volatility
        return volatility
    
    def _calculate_liquidity_adjustment(self, orderbook: OrderBookSnapshot, market_condition: MarketCondition) -> Decimal:
        """计算流动性调整"""
        spread = orderbook.get_spread()
        mid_price = orderbook.get_mid_price()
        
        if mid_price <= 0:
            return Decimal("0")
        
        spread_ratio = spread / mid_price
        
        # 价差越大，流动性越差，滑点越大
        liquidity_impact = spread_ratio * Decimal("0.5")
        
        # 市场状况调整
        if market_condition == MarketCondition.LOW_LIQUIDITY:
            liquidity_impact *= Decimal("2.0")
        elif market_condition == MarketCondition.HIGH_VOLUME:
            liquidity_impact *= Decimal("0.5")
        
        return liquidity_impact
    
    def _calculate_timing_cost(self, order: Order, market_condition: MarketCondition) -> Decimal:
        """计算时间成本"""
        # 市价单的时间成本
        if order.order_type == OrderType.MARKET:
            base_timing_cost = Decimal("0.0001")  # 0.01%
        else:
            base_timing_cost = Decimal("0.00005")  # 0.005%
        
        # 市场状况调整
        if market_condition == MarketCondition.VOLATILE:
            base_timing_cost *= Decimal("1.5")
        
        return base_timing_cost
    
    def update_market_condition(self, symbol: str, condition: MarketCondition):
        """更新市场状况"""
        self.market_conditions[symbol] = condition
    
    def record_trade(self, symbol: str, trade_info: Dict[str, Any]):
        """记录交易信息用于滑点统计"""
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        
        self.recent_trades[symbol].append({
            **trade_info,
            "timestamp": datetime.utcnow()
        })
        
        # 保留最近100笔交易
        self.recent_trades[symbol] = self.recent_trades[symbol][-100:]


class BinanceCommissionCalculator(LoggerMixin):
    """币安手续费计算器"""
    
    def __init__(self, commission_structure: CommissionStructure):
        self.structure = commission_structure
        self.user_30d_volume: Dict[str, Decimal] = {}  # 用户30日交易量
        self.bnb_balance: Decimal = Decimal("0")       # BNB余额
        self.use_bnb_discount: bool = False            # 是否使用BNB抵扣
        
    def calculate_commission(self, 
                           order: Order, 
                           execution_price: Decimal,
                           is_maker: bool,
                           user_id: str = "default") -> Tuple[Decimal, Dict[str, Any]]:
        """计算手续费"""
        
        trade_value = Decimal(str(order.quantity)) * execution_price
        
        # 1. 确定费率档位
        user_volume = self.user_30d_volume.get(user_id, Decimal("0"))
        fee_rate = self._get_fee_rate(user_volume, is_maker)
        
        # 2. 计算基础手续费
        base_commission = trade_value * fee_rate
        
        # 3. BNB抵扣
        final_commission = base_commission
        bnb_discount_amount = Decimal("0")
        
        if self.use_bnb_discount and self.bnb_balance > 0:
            discount_amount = base_commission * self.structure.bnb_discount
            if self.bnb_balance >= discount_amount:
                bnb_discount_amount = discount_amount
                final_commission = base_commission - discount_amount
                self.bnb_balance -= discount_amount
        
        # 4. 特殊费用处理
        if hasattr(order, 'is_liquidation') and order.is_liquidation:
            final_commission += trade_value * self.structure.liquidation_fee
        
        commission_breakdown = {
            "base_commission": float(base_commission),
            "fee_rate": float(fee_rate),
            "fee_tier": self._get_fee_tier(user_volume),
            "is_maker": is_maker,
            "bnb_discount": float(bnb_discount_amount),
            "final_commission": float(final_commission),
            "trade_value": float(trade_value),
            "user_30d_volume": float(user_volume)
        }
        
        return final_commission, commission_breakdown
    
    def _get_fee_rate(self, user_volume: Decimal, is_maker: bool) -> Decimal:
        """获取费率"""
        tiers = self.structure.maker_tiers if is_maker else self.structure.taker_tiers
        
        for volume_threshold, rate in reversed(tiers):
            if user_volume >= volume_threshold:
                return rate
        
        return tiers[0][1]  # 默认最低档
    
    def _get_fee_tier(self, user_volume: Decimal) -> str:
        """获取费率档位"""
        volume_thresholds = [Decimal("0"), Decimal("250"), Decimal("2500"), 
                           Decimal("7500"), Decimal("22500")]
        
        for i, threshold in enumerate(reversed(volume_thresholds)):
            if user_volume >= threshold:
                return f"VIP {len(volume_thresholds) - 1 - i}"
        
        return "VIP 0"
    
    def update_user_volume(self, user_id: str, volume: Decimal):
        """更新用户30日交易量"""
        self.user_30d_volume[user_id] = volume
    
    def set_bnb_balance(self, balance: Decimal):
        """设置BNB余额"""
        self.bnb_balance = balance
    
    def enable_bnb_discount(self, enabled: bool):
        """启用/禁用BNB抵扣"""
        self.use_bnb_discount = enabled


class MarketDelaySimulator(LoggerMixin):
    """市场延迟模拟器"""
    
    def __init__(self):
        self.network_latency_ms = 50      # 网络延迟
        self.exchange_processing_ms = 20   # 交易所处理时间
        self.order_queue_delay_ms = 10     # 订单队列延迟
        
    async def simulate_execution_delay(self, 
                                     order: Order, 
                                     market_condition: MarketCondition = MarketCondition.NORMAL) -> float:
        """模拟执行延迟"""
        
        # 基础延迟
        base_delay = (self.network_latency_ms + 
                     self.exchange_processing_ms + 
                     self.order_queue_delay_ms)
        
        # 市场状况调整
        condition_multipliers = {
            MarketCondition.NORMAL: 1.0,
            MarketCondition.VOLATILE: 1.5,     # 波动时延迟增加
            MarketCondition.LOW_LIQUIDITY: 1.2,
            MarketCondition.HIGH_VOLUME: 2.0   # 高成交量时延迟显著增加
        }
        
        multiplier = condition_multipliers.get(market_condition, 1.0)
        
        # 订单类型调整
        if order.order_type == OrderType.MARKET:
            type_multiplier = 1.0  # 市价单优先处理
        elif order.order_type == OrderType.LIMIT:
            type_multiplier = 0.8  # 限价单处理较快
        else:
            type_multiplier = 1.2  # 其他订单类型
        
        # 添加随机波动
        random_factor = random.uniform(0.8, 1.5)
        
        total_delay_ms = base_delay * multiplier * type_multiplier * random_factor
        
        # 转换为秒并应用延迟
        delay_seconds = total_delay_ms / 1000.0
        await asyncio.sleep(delay_seconds)
        
        return delay_seconds
    
    def update_network_conditions(self, latency_ms: float, processing_ms: float):
        """更新网络状况"""
        self.network_latency_ms = latency_ms
        self.exchange_processing_ms = processing_ms


class MarketMicrostructureSimulator(LoggerMixin):
    """市场微观结构模拟器"""
    
    def __init__(self):
        self.order_arrival_rate = 10  # 每秒订单到达数
        self.market_maker_spread = Decimal("0.0001")  # 做市商价差
        self.inventory_impact = Decimal("0.00005")    # 库存影响
        
    def generate_realistic_orderbook(self, 
                                   symbol: str, 
                                   mid_price: Decimal,
                                   depth_levels: int = 20) -> OrderBookSnapshot:
        """生成真实的订单簿"""
        
        tick_size = self._get_tick_size(symbol)
        
        bids = []
        asks = []
        
        # 生成买单深度
        cumulative_bid_qty = Decimal("0")
        for i in range(depth_levels):
            price = mid_price - (Decimal(str(i + 1)) * tick_size)
            
            # 深度衰减模型：距离中间价越远，数量越少
            base_quantity = Decimal("100") * (Decimal("0.9") ** i)
            # 添加随机波动
            quantity = base_quantity * Decimal(str(random.uniform(0.5, 1.5)))
            
            cumulative_bid_qty += quantity
            bids.append(MarketDepthLevel(price, quantity, cumulative_bid_qty))
        
        # 生成卖单深度
        cumulative_ask_qty = Decimal("0")
        for i in range(depth_levels):
            price = mid_price + (Decimal(str(i + 1)) * tick_size)
            
            base_quantity = Decimal("100") * (Decimal("0.9") ** i)
            quantity = base_quantity * Decimal(str(random.uniform(0.5, 1.5)))
            
            cumulative_ask_qty += quantity
            asks.append(MarketDepthLevel(price, quantity, cumulative_ask_qty))
        
        return OrderBookSnapshot(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow()
        )
    
    def _get_tick_size(self, symbol: str) -> Decimal:
        """获取最小价格变动单位"""
        tick_sizes = {
            "BTCUSDT": Decimal("0.1"),
            "ETHUSDT": Decimal("0.01"),
            "BNBUSDT": Decimal("0.01"),
            "ADAUSDT": Decimal("0.0001"),
            "SOLUSDT": Decimal("0.001")
        }
        return tick_sizes.get(symbol, Decimal("0.01"))
    
    def simulate_market_impact(self, 
                             order: Order, 
                             orderbook: OrderBookSnapshot) -> OrderBookSnapshot:
        """模拟市场冲击对订单簿的影响"""
        
        # 计算订单对订单簿的影响
        order_size = Decimal(str(order.quantity))
        mid_price = orderbook.get_mid_price()
        
        # 简化的市场冲击模型：大订单会消耗流动性
        impact_size = order_size * Decimal("0.1")  # 10%的冲击
        
        if order.side == OrderSide.BUY:
            # 买单冲击：消耗卖单，推高价格
            new_asks = []
            remaining_impact = impact_size
            
            for level in orderbook.asks:
                if remaining_impact <= 0:
                    new_asks.append(level)
                elif level.quantity > remaining_impact:
                    # 部分消耗这一档
                    new_quantity = level.quantity - remaining_impact
                    new_asks.append(MarketDepthLevel(
                        level.price, new_quantity, level.cumulative_quantity - remaining_impact
                    ))
                    remaining_impact = Decimal("0")
                else:
                    # 完全消耗这一档
                    remaining_impact -= level.quantity
            
            return OrderBookSnapshot(
                orderbook.symbol, orderbook.bids, new_asks, datetime.utcnow()
            )
        
        else:
            # 卖单冲击：消耗买单，压低价格
            new_bids = []
            remaining_impact = impact_size
            
            for level in orderbook.bids:
                if remaining_impact <= 0:
                    new_bids.append(level)
                elif level.quantity > remaining_impact:
                    new_quantity = level.quantity - remaining_impact
                    new_bids.append(MarketDepthLevel(
                        level.price, new_quantity, level.cumulative_quantity - remaining_impact
                    ))
                    remaining_impact = Decimal("0")
                else:
                    remaining_impact -= level.quantity
            
            return OrderBookSnapshot(
                orderbook.symbol, new_bids, orderbook.asks, datetime.utcnow()
            )


class IntegratedMarketSimulator(LoggerMixin):
    """集成市场模拟器"""
    
    def __init__(self):
        self.slippage_config = SlippageConfig()
        self.commission_structure = CommissionStructure()
        
        self.slippage_calculator = RealTimeSlippageCalculator(self.slippage_config)
        self.commission_calculator = BinanceCommissionCalculator(self.commission_structure)
        self.delay_simulator = MarketDelaySimulator()
        self.microstructure_simulator = MarketMicrostructureSimulator()
        
        # 市场状况跟踪
        self.current_market_conditions: Dict[str, MarketCondition] = {}
        
    async def simulate_order_execution(self, 
                                     order: Order,
                                     base_price: Decimal,
                                     user_id: str = "default") -> Dict[str, Any]:
        """模拟订单执行的完整过程"""
        
        symbol = order.symbol
        market_condition = self.current_market_conditions.get(symbol, MarketCondition.NORMAL)
        
        # 1. 生成订单簿
        orderbook = self.microstructure_simulator.generate_realistic_orderbook(
            symbol, base_price
        )
        
        # 2. 模拟执行延迟
        execution_delay = await self.delay_simulator.simulate_execution_delay(
            order, market_condition
        )
        
        # 3. 计算滑点
        slippage_rate, slippage_breakdown = self.slippage_calculator.calculate_slippage(
            order, orderbook, market_condition
        )
        
        # 4. 计算执行价格
        mid_price = orderbook.get_mid_price()
        execution_price = mid_price * (Decimal("1") + slippage_rate)
        
        # 5. 确定是否为maker订单
        is_maker = self._determine_maker_status(order, orderbook)
        
        # 6. 计算手续费
        commission, commission_breakdown = self.commission_calculator.calculate_commission(
            order, execution_price, is_maker, user_id
        )
        
        # 7. 模拟市场冲击
        updated_orderbook = self.microstructure_simulator.simulate_market_impact(
            order, orderbook
        )
        
        # 8. 记录交易信息
        trade_info = {
            "symbol": symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "execution_price": float(execution_price),
            "slippage_rate": float(slippage_rate),
            "commission": float(commission),
            "execution_delay": execution_delay,
            "is_maker": is_maker
        }
        
        self.slippage_calculator.record_trade(symbol, trade_info)
        
        return {
            "execution_price": execution_price,
            "commission": commission,
            "execution_delay": execution_delay,
            "slippage_breakdown": slippage_breakdown,
            "commission_breakdown": commission_breakdown,
            "orderbook_before": orderbook,
            "orderbook_after": updated_orderbook,
            "trade_info": trade_info
        }
    
    def _determine_maker_status(self, order: Order, orderbook: OrderBookSnapshot) -> bool:
        """判断是否为maker订单"""
        if order.order_type == OrderType.MARKET:
            return False  # 市价单总是taker
        
        if order.order_type == OrderType.LIMIT and order.price:
            order_price = Decimal(str(order.price))
            
            if order.side == OrderSide.BUY:
                # 买单价格低于最佳卖价时为maker
                best_ask = orderbook.asks[0].price if orderbook.asks else None
                return best_ask is None or order_price < best_ask
            else:
                # 卖单价格高于最佳买价时为maker
                best_bid = orderbook.bids[0].price if orderbook.bids else None
                return best_bid is None or order_price > best_bid
        
        return True  # 其他情况默认为maker
    
    def update_market_condition(self, symbol: str, condition: MarketCondition):
        """更新市场状况"""
        self.current_market_conditions[symbol] = condition
        self.slippage_calculator.update_market_condition(symbol, condition)
    
    def update_user_trading_volume(self, user_id: str, volume_30d: Decimal):
        """更新用户交易量"""
        self.commission_calculator.update_user_volume(user_id, volume_30d)
    
    def configure_bnb_discount(self, balance: Decimal, enabled: bool = True):
        """配置BNB抵扣"""
        self.commission_calculator.set_bnb_balance(balance)
        self.commission_calculator.enable_bnb_discount(enabled)
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """获取模拟统计信息"""
        return {
            "market_conditions": {k: v.value for k, v in self.current_market_conditions.items()},
            "slippage_config": {
                "base_slippage_bps": self.slippage_config.base_slippage_bps,
                "impact_coefficient": float(self.slippage_config.impact_coefficient),
                "volatility_multiplier": float(self.slippage_config.volatility_multiplier)
            },
            "commission_config": {
                "bnb_discount_enabled": self.commission_calculator.use_bnb_discount,
                "bnb_balance": float(self.commission_calculator.bnb_balance)
            },
            "delay_config": {
                "network_latency_ms": self.delay_simulator.network_latency_ms,
                "exchange_processing_ms": self.delay_simulator.exchange_processing_ms
            }
        }