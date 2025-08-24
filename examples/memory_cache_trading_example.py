"""
内存缓存在量化交易系统中的应用示例

展示如何在实际量化交易场景中使用MemoryCachePool：
- 市场数据缓存
- 交易信号缓存
- 订单状态缓存
- 风险指标缓存
- 性能监控数据缓存
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from decimal import Decimal

from src.core.cache import MemoryCachePool, CacheConfig


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        # 转换Decimal为字符串以便序列化
        data['price'] = str(self.price)
        data['volume'] = str(self.volume)
        if self.bid:
            data['bid'] = str(self.bid)
        else:
            data['bid'] = None
        if self.ask:
            data['ask'] = str(self.ask)
        else:
            data['ask'] = None
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float   # 0.0 - 1.0
    price: Decimal
    timestamp: datetime
    strategy: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['price'] = str(self.price)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class OrderStatus:
    """订单状态"""
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    status: str
    filled_quantity: Decimal
    avg_price: Optional[Decimal]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['quantity'] = str(self.quantity)
        data['price'] = str(self.price)
        data['filled_quantity'] = str(self.filled_quantity)
        if self.avg_price:
            data['avg_price'] = str(self.avg_price)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class TradingSystemCache:
    """
    量化交易系统缓存管理器
    
    统一管理各种交易相关数据的缓存，提供高层接口
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """初始化交易系统缓存"""
        if cache_config is None:
            cache_config = CacheConfig(
                max_memory=64 * 1024 * 1024,  # 64MB
                max_keys=100000,
                default_ttl=300,  # 5分钟默认TTL
                eviction_policy="lru",
                cleanup_interval=30,  # 30秒清理一次
                enable_stats=True
            )
        
        self.cache = MemoryCachePool(cache_config)
        
        # 缓存键前缀
        self.MARKET_DATA_PREFIX = "market:"
        self.SIGNAL_PREFIX = "signal:"
        self.ORDER_PREFIX = "order:"
        self.RISK_PREFIX = "risk:"
        self.STRATEGY_PREFIX = "strategy:"
    
    async def start(self):
        """启动缓存系统"""
        await self.cache.start_cleanup_task()
        print("交易系统缓存已启动")
    
    async def stop(self):
        """停止缓存系统"""
        await self.cache.stop_cleanup_task()
        print("交易系统缓存已停止")
    
    # ==================== 市场数据缓存 ====================
    
    def cache_market_data(self, market_data: MarketData, ttl: int = 60) -> bool:
        """缓存市场数据"""
        key = f"{self.MARKET_DATA_PREFIX}{market_data.symbol}:latest"
        return self.cache.set(key, market_data.to_dict(), ex=ttl)
    
    def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """获取最新市场数据"""
        key = f"{self.MARKET_DATA_PREFIX}{symbol}:latest"
        data = self.cache.get(key)
        
        if data:
            # 重建MarketData对象
            data['price'] = Decimal(data['price'])
            data['volume'] = Decimal(data['volume'])
            if data.get('bid'):
                data['bid'] = Decimal(data['bid'])
            else:
                data['bid'] = None
            if data.get('ask'):
                data['ask'] = Decimal(data['ask'])
            else:
                data['ask'] = None
            if isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return MarketData(**data)
        
        return None
    
    def cache_market_data_history(self, symbol: str, market_data_list: List[MarketData], 
                                 ttl: int = 300) -> bool:
        """缓存市场数据历史"""
        key = f"{self.MARKET_DATA_PREFIX}{symbol}:history"
        history_data = [data.to_dict() for data in market_data_list]
        return self.cache.set(key, history_data, ex=ttl)
    
    def get_market_data_history(self, symbol: str) -> List[MarketData]:
        """获取市场数据历史"""
        key = f"{self.MARKET_DATA_PREFIX}{symbol}:history"
        data = self.cache.get(key)
        
        if data:
            return [
                MarketData(
                    symbol=item['symbol'],
                    price=Decimal(item['price']),
                    volume=Decimal(item['volume']),
                    bid=Decimal(item['bid']) if item.get('bid') else None,
                    ask=Decimal(item['ask']) if item.get('ask') else None,
                    timestamp=datetime.fromisoformat(item['timestamp']) if isinstance(item['timestamp'], str) else item['timestamp']
                ) for item in data
            ]
        
        return []
    
    # ==================== 交易信号缓存 ====================
    
    def cache_trading_signal(self, signal: TradingSignal, ttl: int = 120) -> bool:
        """缓存交易信号"""
        key = f"{self.SIGNAL_PREFIX}{signal.symbol}:{signal.strategy}"
        return self.cache.set(key, signal.to_dict(), ex=ttl)
    
    def get_trading_signal(self, symbol: str, strategy: str) -> Optional[TradingSignal]:
        """获取交易信号"""
        key = f"{self.SIGNAL_PREFIX}{symbol}:{strategy}"
        data = self.cache.get(key)
        
        if data:
            data['price'] = Decimal(data['price'])
            if isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return TradingSignal(**data)
        
        return None
    
    def get_all_signals_for_symbol(self, symbol: str) -> List[TradingSignal]:
        """获取某个符号的所有交易信号"""
        pattern = f"{self.SIGNAL_PREFIX}{symbol}:*"
        signal_keys = self.cache.keys(pattern)
        
        signals = []
        for key in signal_keys:
            data = self.cache.get(key)
            if data:
                data['price'] = Decimal(data['price'])
                if isinstance(data['timestamp'], str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                signals.append(TradingSignal(**data))
        
        return signals
    
    # ==================== 订单状态缓存 ====================
    
    def cache_order_status(self, order: OrderStatus, ttl: int = 3600) -> bool:
        """缓存订单状态"""
        key = f"{self.ORDER_PREFIX}{order.order_id}"
        return self.cache.set(key, order.to_dict(), ex=ttl)
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态"""
        key = f"{self.ORDER_PREFIX}{order_id}"
        data = self.cache.get(key)
        
        if data:
            data['quantity'] = Decimal(data['quantity'])
            data['price'] = Decimal(data['price'])
            data['filled_quantity'] = Decimal(data['filled_quantity'])
            if data.get('avg_price'):
                data['avg_price'] = Decimal(data['avg_price'])
            else:
                data['avg_price'] = None
            if isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return OrderStatus(**data)
        
        return None
    
    def get_active_orders_for_symbol(self, symbol: str) -> List[OrderStatus]:
        """获取某个符号的活跃订单"""
        all_order_keys = self.cache.keys(f"{self.ORDER_PREFIX}*")
        active_orders = []
        
        for key in all_order_keys:
            data = self.cache.get(key)
            if data and data['symbol'] == symbol and data['status'] in ['NEW', 'PARTIALLY_FILLED']:
                data['quantity'] = Decimal(data['quantity'])
                data['price'] = Decimal(data['price'])
                data['filled_quantity'] = Decimal(data['filled_quantity'])
                if data.get('avg_price'):
                    data['avg_price'] = Decimal(data['avg_price'])
                else:
                    data['avg_price'] = None
                if isinstance(data['timestamp'], str):
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                active_orders.append(OrderStatus(**data))
        
        return active_orders
    
    # ==================== 风险指标缓存 ====================
    
    def cache_risk_metrics(self, symbol: str, metrics: Dict, ttl: int = 180) -> bool:
        """缓存风险指标"""
        key = f"{self.RISK_PREFIX}{symbol}:metrics"
        return self.cache.set(key, metrics, ex=ttl)
    
    def get_risk_metrics(self, symbol: str) -> Optional[Dict]:
        """获取风险指标"""
        key = f"{self.RISK_PREFIX}{symbol}:metrics"
        return self.cache.get(key)
    
    def cache_portfolio_risk(self, portfolio_id: str, risk_data: Dict, ttl: int = 300) -> bool:
        """缓存投资组合风险数据"""
        key = f"{self.RISK_PREFIX}portfolio:{portfolio_id}"
        return self.cache.set(key, risk_data, ex=ttl)
    
    def get_portfolio_risk(self, portfolio_id: str) -> Optional[Dict]:
        """获取投资组合风险数据"""
        key = f"{self.RISK_PREFIX}portfolio:{portfolio_id}"
        return self.cache.get(key)
    
    # ==================== 策略数据缓存 ====================
    
    def cache_strategy_state(self, strategy_name: str, state_data: Dict, ttl: int = 600) -> bool:
        """缓存策略状态"""
        key = f"{self.STRATEGY_PREFIX}{strategy_name}:state"
        return self.cache.set(key, state_data, ex=ttl)
    
    def get_strategy_state(self, strategy_name: str) -> Optional[Dict]:
        """获取策略状态"""
        key = f"{self.STRATEGY_PREFIX}{strategy_name}:state"
        return self.cache.get(key)
    
    def cache_strategy_performance(self, strategy_name: str, performance_data: Dict, 
                                 ttl: int = 900) -> bool:
        """缓存策略性能数据"""
        key = f"{self.STRATEGY_PREFIX}{strategy_name}:performance"
        return self.cache.set(key, performance_data, ex=ttl)
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[Dict]:
        """获取策略性能数据"""
        key = f"{self.STRATEGY_PREFIX}{strategy_name}:performance"
        return self.cache.get(key)
    
    # ==================== 统计和监控 ====================
    
    def get_cache_statistics(self) -> Dict:
        """获取缓存统计信息"""
        stats = self.cache.get_stats()
        return {
            'hit_rate': stats.hit_rate,
            'total_operations': stats.operations,
            'memory_usage_mb': stats.memory_usage / (1024 * 1024),
            'key_count': stats.key_count,
            'evictions': stats.evictions,
            'expired_keys': stats.expired_keys
        }
    
    def get_cache_info(self) -> Dict:
        """获取详细缓存信息"""
        return self.cache.info()
    
    def clear_expired_data(self) -> int:
        """手动清理过期数据"""
        return self.cache._cleanup_expired_keys()


class MarketDataSimulator:
    """市场数据模拟器"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.base_prices = {symbol: Decimal(str(random.uniform(100, 1000))) 
                           for symbol in symbols}
    
    def generate_market_data(self, symbol: str) -> MarketData:
        """生成模拟市场数据"""
        base_price = self.base_prices[symbol]
        
        # 价格随机波动 ±2%
        price_change = Decimal(str(random.uniform(-0.02, 0.02)))
        new_price = base_price * (1 + price_change)
        self.base_prices[symbol] = new_price
        
        return MarketData(
            symbol=symbol,
            price=new_price,
            volume=Decimal(str(random.uniform(1000, 10000))),
            bid=new_price * Decimal('0.999'),
            ask=new_price * Decimal('1.001'),
            timestamp=datetime.now()
        )


async def trading_system_simulation():
    """交易系统缓存使用模拟"""
    print("启动量化交易系统缓存模拟...")
    
    # 初始化缓存系统
    cache_config = CacheConfig(
        max_memory=32 * 1024 * 1024,  # 32MB
        max_keys=50000,
        default_ttl=300,
        eviction_policy="lru",
        cleanup_interval=10
    )
    
    trading_cache = TradingSystemCache(cache_config)
    await trading_cache.start()
    
    # 交易符号
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'SOLUSDT']
    strategies = ['MA_CROSS', 'RSI_OVERSOLD', 'MACD_DIVERGENCE']
    
    # 市场数据模拟器
    market_simulator = MarketDataSimulator(symbols)
    
    try:
        # 模拟交易系统运行
        for round_num in range(10):
            print(f"\n=== 交易轮次 {round_num + 1} ===")
            
            # 1. 生成并缓存市场数据
            print("1. 更新市场数据...")
            for symbol in symbols:
                market_data = market_simulator.generate_market_data(symbol)
                trading_cache.cache_market_data(market_data, ttl=120)
                print(f"  {symbol}: ${market_data.price:.2f}")
            
            # 2. 生成并缓存交易信号
            print("2. 生成交易信号...")
            for symbol in symbols:
                for strategy in strategies:
                    if random.random() < 0.3:  # 30%概率生成信号
                        market_data = trading_cache.get_latest_market_data(symbol)
                        if market_data:
                            signal = TradingSignal(
                                symbol=symbol,
                                signal_type=random.choice(['BUY', 'SELL', 'HOLD']),
                                strength=random.uniform(0.3, 1.0),
                                price=market_data.price,
                                timestamp=datetime.now(),
                                strategy=strategy
                            )
                            trading_cache.cache_trading_signal(signal, ttl=180)
                            print(f"  {strategy} -> {symbol}: {signal.signal_type} "
                                  f"(强度: {signal.strength:.2f})")
            
            # 3. 模拟订单生成和状态更新
            print("3. 处理订单...")
            if random.random() < 0.5:  # 50%概率生成新订单
                symbol = random.choice(symbols)
                order = OrderStatus(
                    order_id=f"ORDER_{round_num}_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    side=random.choice(['BUY', 'SELL']),
                    quantity=Decimal(str(random.uniform(0.1, 10.0))),
                    price=trading_cache.get_latest_market_data(symbol).price,
                    status='NEW',
                    filled_quantity=Decimal('0'),
                    avg_price=None,
                    timestamp=datetime.now()
                )
                trading_cache.cache_order_status(order, ttl=3600)
                print(f"  新订单: {order.order_id} - {order.side} {order.quantity} {order.symbol}")
            
            # 4. 缓存风险指标
            print("4. 更新风险指标...")
            for symbol in symbols:
                risk_metrics = {
                    'var_1d': random.uniform(0.01, 0.05),
                    'volatility': random.uniform(0.1, 0.3),
                    'beta': random.uniform(0.8, 1.2),
                    'max_drawdown': random.uniform(0.02, 0.1),
                    'sharpe_ratio': random.uniform(1.0, 3.0),
                    'last_updated': datetime.now().isoformat()
                }
                trading_cache.cache_risk_metrics(symbol, risk_metrics, ttl=300)
            
            # 5. 缓存策略性能
            print("5. 更新策略性能...")
            for strategy in strategies:
                performance = {
                    'total_return': random.uniform(-0.1, 0.2),
                    'win_rate': random.uniform(0.4, 0.7),
                    'avg_trade_return': random.uniform(0.001, 0.01),
                    'max_drawdown': random.uniform(0.05, 0.15),
                    'trades_count': random.randint(50, 200),
                    'last_updated': datetime.now().isoformat()
                }
                trading_cache.cache_strategy_performance(strategy, performance, ttl=900)
            
            # 6. 显示缓存统计
            if round_num % 3 == 0:
                stats = trading_cache.get_cache_statistics()
                print(f"\n缓存统计:")
                print(f"  命中率: {stats['hit_rate']:.2%}")
                print(f"  内存使用: {stats['memory_usage_mb']:.1f}MB")
                print(f"  键数量: {stats['key_count']}")
                print(f"  总操作: {stats['total_operations']}")
                print(f"  淘汰次数: {stats['evictions']}")
            
            # 等待下一轮
            await asyncio.sleep(2)
        
        # 最终统计
        print("\n" + "="*50)
        print("模拟结束 - 最终缓存统计:")
        final_stats = trading_cache.get_cache_statistics()
        for key, value in final_stats.items():
            if key == 'hit_rate':
                print(f"  {key}: {value:.2%}")
            elif key == 'memory_usage_mb':
                print(f"  {key}: {value:.1f}MB")
            else:
                print(f"  {key}: {value}")
        
        # 演示数据检索
        print("\n演示数据检索:")
        for symbol in symbols[:2]:  # 只演示前两个符号
            market_data = trading_cache.get_latest_market_data(symbol)
            if market_data:
                print(f"  {symbol} 最新价格: ${market_data.price:.2f}")
            
            signals = trading_cache.get_all_signals_for_symbol(symbol)
            if signals:
                print(f"  {symbol} 活跃信号数量: {len(signals)}")
            
            risk_metrics = trading_cache.get_risk_metrics(symbol)
            if risk_metrics:
                print(f"  {symbol} 波动率: {risk_metrics['volatility']:.2%}")
    
    finally:
        await trading_cache.stop()


async def performance_comparison():
    """性能对比测试：缓存 vs 无缓存"""
    print("\n" + "="*60)
    print("性能对比测试：缓存 vs 无缓存")
    print("="*60)
    
    # 模拟数据库/API延迟的函数
    async def slow_data_fetch(symbol: str) -> MarketData:
        """模拟慢速数据获取（如数据库查询或API调用）"""
        await asyncio.sleep(0.01)  # 模拟10ms延迟
        return MarketData(
            symbol=symbol,
            price=Decimal(str(random.uniform(100, 1000))),
            volume=Decimal(str(random.uniform(1000, 10000))),
            timestamp=datetime.now()
        )
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    
    # 无缓存测试
    print("测试无缓存数据获取...")
    start_time = time.perf_counter()
    
    for _ in range(100):
        symbol = random.choice(symbols)
        await slow_data_fetch(symbol)
    
    no_cache_time = time.perf_counter() - start_time
    
    # 使用缓存测试
    print("测试缓存数据获取...")
    trading_cache = TradingSystemCache()
    await trading_cache.start()
    
    # 预填充缓存
    for symbol in symbols:
        market_data = await slow_data_fetch(symbol)
        trading_cache.cache_market_data(market_data, ttl=600)
    
    start_time = time.perf_counter()
    cache_hits = 0
    cache_misses = 0
    
    for _ in range(100):
        symbol = random.choice(symbols)
        market_data = trading_cache.get_latest_market_data(symbol)
        
        if market_data:
            cache_hits += 1
        else:
            cache_misses += 1
            # 缓存未命中时从"数据库"获取
            market_data = await slow_data_fetch(symbol)
            trading_cache.cache_market_data(market_data, ttl=600)
    
    cache_time = time.perf_counter() - start_time
    
    await trading_cache.stop()
    
    # 结果对比
    speedup = no_cache_time / cache_time
    
    print(f"\n性能对比结果:")
    print(f"  无缓存耗时: {no_cache_time:.3f}秒")
    print(f"  缓存耗时: {cache_time:.3f}秒")
    print(f"  性能提升: {speedup:.1f}倍")
    print(f"  缓存命中率: {cache_hits/(cache_hits+cache_misses):.1%}")
    print(f"  缓存命中: {cache_hits}, 未命中: {cache_misses}")


async def main():
    """主函数"""
    print("量化交易系统内存缓存应用示例")
    print("="*60)
    
    # 1. 交易系统模拟
    await trading_system_simulation()
    
    # 2. 性能对比
    await performance_comparison()
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    asyncio.run(main())