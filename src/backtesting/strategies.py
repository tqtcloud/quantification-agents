"""回测策略示例"""

from typing import List, Dict, Any
from decimal import Decimal
import asyncio

from src.core.models import Order, OrderSide, OrderType, MarketData
from src.trading.enhanced_paper_trading_engine import EnhancedPaperTradingEngine
from src.utils.logger import LoggerMixin


class BaseStrategy(LoggerMixin):
    """策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = {}  # 策略状态
        
    async def generate_signals(self, context: Dict[str, Any]) -> List[Order]:
        """生成交易信号"""
        raise NotImplementedError


class SimpleMovingAverageStrategy(BaseStrategy):
    """简单移动平均策略"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, 
                 position_size: float = 0.1):
        super().__init__("SMA_Strategy")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.price_history = []
        
    async def generate_signals(self, context: Dict[str, Any]) -> List[Order]:
        """生成移动平均交叉信号"""
        market_data: MarketData = context["market_data"]
        engine: EnhancedPaperTradingEngine = context["engine"]
        
        # 更新价格历史
        self.price_history.append(market_data.price)
        
        # 保持历史长度
        if len(self.price_history) > self.long_window:
            self.price_history = self.price_history[-self.long_window:]
            
        # 需要足够的历史数据
        if len(self.price_history) < self.long_window:
            return []
        
        # 计算移动平均
        short_ma = sum(self.price_history[-self.short_window:]) / self.short_window
        long_ma = sum(self.price_history) / self.long_window
        
        orders = []
        current_position = engine.positions.get(market_data.symbol)
        
        # 金叉：短期均线上穿长期均线，买入
        if short_ma > long_ma and not current_position:
            order = Order(
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.position_size
            )
            orders.append(order)
            self.log_debug(f"Golden cross detected: {short_ma:.2f} > {long_ma:.2f}, BUY signal")
            
        # 死叉：短期均线下穿长期均线，卖出
        elif short_ma < long_ma and current_position and current_position.side == "LONG":
            order = Order(
                symbol=market_data.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=float(current_position.size)
            )
            orders.append(order)
            self.log_debug(f"Death cross detected: {short_ma:.2f} < {long_ma:.2f}, SELL signal")
        
        return orders


class RSIStrategy(BaseStrategy):
    """RSI均值回归策略"""
    
    def __init__(self, rsi_period: int = 14, oversold_level: float = 30, 
                 overbought_level: float = 70, position_size: float = 0.1):
        super().__init__("RSI_Strategy")
        self.rsi_period = rsi_period
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
        self.position_size = position_size
        self.price_history = []
        
    def calculate_rsi(self, prices: List[float]) -> float:
        """计算RSI指标"""
        if len(prices) < self.rsi_period + 1:
            return 50.0  # 默认中性值
        
        # 计算价格变化
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # 分离上涨和下跌
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # 计算平均增益和损失
        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def generate_signals(self, context: Dict[str, Any]) -> List[Order]:
        """生成RSI策略信号"""
        market_data: MarketData = context["market_data"]
        engine: EnhancedPaperTradingEngine = context["engine"]
        
        # 更新价格历史
        self.price_history.append(market_data.price)
        
        # 保持历史长度
        if len(self.price_history) > self.rsi_period * 2:
            self.price_history = self.price_history[-(self.rsi_period * 2):]
        
        # 需要足够的历史数据
        if len(self.price_history) < self.rsi_period + 1:
            return []
        
        # 计算RSI
        rsi = self.calculate_rsi(self.price_history)
        
        orders = []
        current_position = engine.positions.get(market_data.symbol)
        
        # RSI超卖，买入信号
        if rsi < self.oversold_level and not current_position:
            order = Order(
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.position_size
            )
            orders.append(order)
            self.log_debug(f"RSI oversold: {rsi:.2f} < {self.oversold_level}, BUY signal")
        
        # RSI超买，卖出信号
        elif rsi > self.overbought_level and current_position and current_position.side == "LONG":
            order = Order(
                symbol=market_data.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=float(current_position.size)
            )
            orders.append(order)
            self.log_debug(f"RSI overbought: {rsi:.2f} > {self.overbought_level}, SELL signal")
        
        return orders


class BuyAndHoldStrategy(BaseStrategy):
    """买入持有策略"""
    
    def __init__(self, position_size: float = 0.1):
        super().__init__("Buy_And_Hold")
        self.position_size = position_size
        self.has_bought = False
        
    async def generate_signals(self, context: Dict[str, Any]) -> List[Order]:
        """生成买入持有信号"""
        market_data: MarketData = context["market_data"]
        engine: EnhancedPaperTradingEngine = context["engine"]
        
        # 只在开始时买入一次
        if not self.has_bought and not engine.positions.get(market_data.symbol):
            self.has_bought = True
            
            order = Order(
                symbol=market_data.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=self.position_size
            )
            
            self.log_info(f"Buy and hold: Initial purchase of {self.position_size} {market_data.symbol}")
            return [order]
        
        return []


# 策略工厂函数
def create_sma_strategy(context: Dict[str, Any]) -> List[Order]:
    """SMA策略工厂函数"""
    if not hasattr(create_sma_strategy, "strategy"):
        config = context.get("config", {})
        create_sma_strategy.strategy = SimpleMovingAverageStrategy(
            short_window=config.get("short_window", 10),
            long_window=config.get("long_window", 30),
            position_size=config.get("position_size", 0.1)
        )
    
    return asyncio.create_task(
        create_sma_strategy.strategy.generate_signals(context)
    )


def create_rsi_strategy(context: Dict[str, Any]) -> List[Order]:
    """RSI策略工厂函数"""
    if not hasattr(create_rsi_strategy, "strategy"):
        config = context.get("config", {})
        create_rsi_strategy.strategy = RSIStrategy(
            rsi_period=config.get("rsi_period", 14),
            oversold_level=config.get("oversold_level", 30),
            overbought_level=config.get("overbought_level", 70),
            position_size=config.get("position_size", 0.1)
        )
    
    return asyncio.create_task(
        create_rsi_strategy.strategy.generate_signals(context)
    )


def create_buy_hold_strategy(context: Dict[str, Any]) -> List[Order]:
    """买入持有策略工厂函数"""
    if not hasattr(create_buy_hold_strategy, "strategy"):
        config = context.get("config", {})
        create_buy_hold_strategy.strategy = BuyAndHoldStrategy(
            position_size=config.get("position_size", 0.1)
        )
    
    return asyncio.create_task(
        create_buy_hold_strategy.strategy.generate_signals(context)
    )