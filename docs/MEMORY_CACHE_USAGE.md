# 内存缓存系统使用指南

## 概述

MemoryCachePool 是专为量化交易系统设计的高性能内存缓存解决方案，用于替代Redis中间件。它提供了Redis兼容的API接口，支持多种数据类型，并针对低延迟高频交易场景进行了优化。

## 特性

### 核心功能
- **Redis兼容API**: 支持主要Redis操作（get/set/delete/exists/keys等）
- **多数据类型**: 字符串、哈希、列表、集合支持
- **TTL机制**: 自动过期和清理机制
- **线程安全**: 并发访问保护
- **内存管理**: LRU/LFU淘汰策略，内存使用限制
- **性能监控**: 命中率、内存使用等统计信息

### 性能优化
- **低延迟**: 平均操作延迟 < 50微秒
- **高吞吐**: 支持 > 20,000 ops/sec
- **并发友好**: 多线程安全访问
- **内存高效**: 智能内存估算和淘汰

## 快速开始

### 基础使用

```python
from src.core.cache import MemoryCachePool, CacheConfig

# 创建缓存实例
cache = MemoryCachePool()

# 基础操作
cache.set("key1", "value1")
value = cache.get("key1")
print(value)  # 输出: value1

# 设置过期时间
cache.set("temp_key", "temp_value", ex=60)  # 60秒后过期
```

### 配置选项

```python
config = CacheConfig(
    max_memory=64 * 1024 * 1024,  # 最大内存64MB
    max_keys=100000,              # 最大键数量
    default_ttl=300,              # 默认TTL 5分钟
    eviction_policy="lru",        # 淘汰策略
    cleanup_interval=30,          # 清理间隔30秒
    enable_stats=True,            # 启用统计
    thread_safe=True              # 线程安全
)

cache = MemoryCachePool(config)
```

## API 参考

### 基础字符串操作

```python
# 设置键值
cache.set("key", "value", ex=60)  # 60秒过期
cache.set("key", "value", px=60000)  # 60000毫秒过期
cache.set("key", "value", nx=True)   # 仅当键不存在时设置
cache.set("key", "value", xx=True)   # 仅当键存在时设置

# 获取值
value = cache.get("key")

# 删除键
count = cache.delete("key1", "key2")

# 检查存在
exists_count = cache.exists("key1", "key2")

# 获取匹配的键
keys = cache.keys("user:*")
```

### TTL操作

```python
# 设置过期时间
cache.expire("key", 60)  # 60秒后过期

# 获取剩余生存时间
ttl = cache.ttl("key")

# 移除过期时间
cache.persist("key")
```

### Hash操作

```python
# 设置Hash字段
cache.hset("user:1", name="alice", age=25)
cache.hset("user:1", {"email": "alice@test.com"})

# 获取Hash字段
name = cache.hget("user:1", "name")
user_data = cache.hgetall("user:1")

# 删除Hash字段
cache.hdel("user:1", "age")
```

### List操作

```python
# 推入元素
cache.lpush("queue", "item1", "item2")  # 左侧推入
cache.rpush("queue", "item3", "item4")  # 右侧推入

# 弹出元素
item = cache.lpop("queue")  # 左侧弹出
item = cache.rpop("queue")  # 右侧弹出

# 获取范围
items = cache.lrange("queue", 0, -1)  # 获取全部

# 获取长度
length = cache.llen("queue")
```

### Set操作

```python
# 添加元素
cache.sadd("tags", "python", "redis", "cache")

# 检查成员
is_member = cache.sismember("tags", "python")

# 获取所有成员
members = cache.smembers("tags")

# 移除元素
cache.srem("tags", "redis")

# 获取大小
size = cache.scard("tags")
```

### 数据库操作

```python
# 获取键数量
size = cache.dbsize()

# 清空所有数据
cache.flushall()
```

## 在量化交易系统中的应用

### 1. 市场数据缓存

```python
from src.core.cache import MemoryCachePool
from decimal import Decimal
from datetime import datetime

cache = MemoryCachePool()

# 缓存最新价格
def cache_market_data(symbol, price, volume):
    data = {
        'price': str(price),
        'volume': str(volume),
        'timestamp': datetime.now().isoformat()
    }
    cache.set(f"market:{symbol}:latest", data, ex=60)

# 获取最新价格
def get_latest_price(symbol):
    data = cache.get(f"market:{symbol}:latest")
    if data:
        return Decimal(data['price'])
    return None

# 使用示例
cache_market_data("BTCUSDT", Decimal("45000.00"), Decimal("100.5"))
price = get_latest_price("BTCUSDT")
```

### 2. 交易信号缓存

```python
# 缓存交易信号
def cache_trading_signal(symbol, strategy, signal_type, strength):
    signal = {
        'type': signal_type,
        'strength': strength,
        'timestamp': datetime.now().isoformat()
    }
    key = f"signal:{symbol}:{strategy}"
    cache.set(key, signal, ex=120)  # 2分钟过期

# 获取交易信号
def get_trading_signals(symbol):
    pattern = f"signal:{symbol}:*"
    signal_keys = cache.keys(pattern)
    signals = []
    
    for key in signal_keys:
        signal = cache.get(key)
        if signal:
            strategy = key.split(':')[-1]
            signal['strategy'] = strategy
            signals.append(signal)
    
    return signals
```

### 3. 风险指标缓存

```python
# 缓存风险指标
def cache_risk_metrics(symbol, var, volatility, beta):
    metrics = {
        'var_1d': var,
        'volatility': volatility,
        'beta': beta,
        'updated_at': datetime.now().isoformat()
    }
    cache.set(f"risk:{symbol}", metrics, ex=300)  # 5分钟过期

# 批量获取风险指标
def get_portfolio_risk(symbols):
    risk_data = {}
    for symbol in symbols:
        metrics = cache.get(f"risk:{symbol}")
        if metrics:
            risk_data[symbol] = metrics
    return risk_data
```

### 4. 订单状态缓存

```python
# 缓存订单状态
def cache_order_status(order_id, symbol, side, status, filled_qty):
    order = {
        'symbol': symbol,
        'side': side,
        'status': status,
        'filled_qty': str(filled_qty),
        'updated_at': datetime.now().isoformat()
    }
    cache.set(f"order:{order_id}", order, ex=3600)  # 1小时过期

# 获取活跃订单
def get_active_orders():
    order_keys = cache.keys("order:*")
    active_orders = []
    
    for key in order_keys:
        order = cache.get(key)
        if order and order['status'] in ['NEW', 'PARTIALLY_FILLED']:
            order_id = key.split(':')[1]
            order['order_id'] = order_id
            active_orders.append(order)
    
    return active_orders
```

## 高级功能

### 异步清理任务

```python
import asyncio

async def setup_cache_with_cleanup():
    cache = MemoryCachePool()
    
    # 启动自动清理任务
    await cache.start_cleanup_task()
    
    try:
        # 使用缓存
        cache.set("temp", "value", ex=1)
        await asyncio.sleep(2)
        # 过期数据自动被清理
        
    finally:
        # 停止清理任务
        await cache.stop_cleanup_task()

# 运行
asyncio.run(setup_cache_with_cleanup())
```

### 性能监控

```python
# 获取统计信息
stats = cache.get_stats()
print(f"命中率: {stats.hit_rate:.2%}")
print(f"内存使用: {stats.memory_usage / 1024 / 1024:.1f}MB")
print(f"键数量: {stats.key_count}")
print(f"总操作: {stats.operations}")

# 获取详细信息
info = cache.info()
print("缓存详细信息:", info)

# 重置统计
cache.reset_stats()
```

### 调试功能

```python
# 获取所有键的调试信息
debug_info = cache.debug_keys()
for key_info in debug_info:
    print(f"键: {key_info['key']}")
    print(f"类型: {key_info['type']}")
    print(f"TTL: {key_info['ttl']}秒")
    print(f"访问次数: {key_info['access_count']}")
    print(f"内存估算: {key_info['memory_estimate']}字节")
    print("---")
```

## 性能调优

### 内存优化

```python
# 高内存配置（适合大数据量）
high_memory_config = CacheConfig(
    max_memory=512 * 1024 * 1024,  # 512MB
    max_keys=1000000,
    eviction_policy="lru"
)

# 低延迟配置（适合高频交易）
low_latency_config = CacheConfig(
    max_memory=64 * 1024 * 1024,   # 64MB
    max_keys=50000,
    eviction_policy="lfu",
    cleanup_interval=60,
    thread_safe=True
)
```

### 并发优化

```python
from concurrent.futures import ThreadPoolExecutor

# 多线程使用示例
def worker_function(worker_id, cache, iterations):
    for i in range(iterations):
        key = f"worker_{worker_id}_key_{i}"
        value = f"worker_{worker_id}_value_{i}"
        
        cache.set(key, value)
        retrieved = cache.get(key)
        assert retrieved == value

# 并发测试
cache = MemoryCachePool(CacheConfig(thread_safe=True))

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(worker_function, i, cache, 1000)
        for i in range(4)
    ]
    
    for future in futures:
        future.result()  # 等待完成

print("并发测试完成")
```

## 最佳实践

### 1. 键命名规范

```python
# 推荐的键命名模式
"market:BTCUSDT:latest"          # 市场数据
"signal:BTCUSDT:ma_cross"        # 交易信号
"order:ORD123456"                # 订单状态
"risk:portfolio:main"            # 风险指标
"strategy:momentum:performance"   # 策略性能
```

### 2. TTL策略

```python
# 不同数据的TTL建议
cache.set("market:price", data, ex=60)      # 市场价格: 1分钟
cache.set("signal:buy", data, ex=300)       # 交易信号: 5分钟  
cache.set("risk:var", data, ex=3600)        # 风险指标: 1小时
cache.set("order:status", data, ex=86400)   # 订单状态: 24小时
```

### 3. 错误处理

```python
def safe_cache_get(key, default=None):
    """安全的缓存获取"""
    try:
        return cache.get(key) or default
    except Exception as e:
        print(f"缓存获取错误 {key}: {e}")
        return default

def safe_cache_set(key, value, ttl=None):
    """安全的缓存设置"""
    try:
        return cache.set(key, value, ex=ttl)
    except Exception as e:
        print(f"缓存设置错误 {key}: {e}")
        return False
```

### 4. 内存管理

```python
# 定期检查内存使用
def monitor_cache_memory():
    stats = cache.get_stats()
    memory_mb = stats.memory_usage / 1024 / 1024
    
    if memory_mb > 100:  # 超过100MB
        print(f"警告: 缓存内存使用过高 {memory_mb:.1f}MB")
        
        # 可以考虑清理一些数据
        expired_count = cache._cleanup_expired_keys()
        print(f"清理了 {expired_count} 个过期键")

# 定期调用监控函数
import threading
import time

def cache_monitor_thread():
    while True:
        monitor_cache_memory()
        time.sleep(300)  # 每5分钟检查一次

monitor_thread = threading.Thread(target=cache_monitor_thread, daemon=True)
monitor_thread.start()
```

## 测试和验证

### 运行测试

```bash
# 运行基础功能测试
python -m pytest tests/test_memory_cache.py -v

# 运行性能基准测试
python tests/test_memory_cache_benchmarks.py

# 运行集成示例
python examples/memory_cache_trading_example.py

# 运行完整测试套件
./scripts/test_memory_cache.sh
```

### 性能基准

在标准配置下的性能指标：

- **SET操作**: 平均延迟 < 100μs, 吞吐量 > 10,000 ops/sec
- **GET操作**: 平均延迟 < 50μs, 吞吐量 > 20,000 ops/sec
- **并发读取**: 8线程下 > 50,000 ops/sec
- **并发写入**: 4线程下 > 15,000 ops/sec
- **内存效率**: > 50% (考虑元数据开销)

## 故障排查

### 常见问题

1. **内存使用过高**
   - 检查TTL设置是否合理
   - 调整max_memory和eviction_policy
   - 增加cleanup_interval频率

2. **性能下降**
   - 检查并发配置thread_safe设置
   - 监控命中率，调整缓存策略
   - 检查是否有大对象占用过多内存

3. **数据不一致**
   - 确保在并发环境下启用thread_safe
   - 检查TTL设置，避免数据过期
   - 使用事务性操作处理相关数据

### 日志和调试

```python
import structlog

# 启用详细日志
logger = structlog.get_logger(__name__)

# 在关键操作前后添加日志
def debug_cache_operation(operation, key, **kwargs):
    logger.info("Cache operation", 
                operation=operation, 
                key=key, 
                **kwargs)
```

## 总结

MemoryCachePool为量化交易系统提供了高性能、低延迟的内存缓存解决方案。通过合理配置和使用最佳实践，可以显著提升系统性能，减少对外部依赖，满足高频交易的严格要求。

关键优势：
- 🚀 **超低延迟**: 微秒级操作响应时间
- 📈 **高吞吐量**: 支持万级ops/sec
- 🔒 **线程安全**: 并发访问无忧
- 💾 **智能内存管理**: 自动淘汰和清理
- 📊 **全面监控**: 详细性能统计
- 🔧 **Redis兼容**: 平滑迁移和集成