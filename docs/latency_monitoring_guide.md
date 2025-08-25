# 延迟监控系统使用指南

## 概述

延迟监控系统（`LatencyMonitor`）是为高频交易场景设计的核心组件，提供实时的市场数据延迟监控、自动故障切换和性能指标收集功能。

## 核心功能

### 1. 数据新鲜度检查
- 检查市场数据是否超过延迟阈值（默认100ms）
- 计算网络延迟、处理延迟和总延迟
- 标记过期数据并触发相应处理逻辑

### 2. 备用数据源切换
- 支持多数据源配置，按优先级排序
- 当主数据源延迟超标时自动切换到备用源
- 智能故障恢复和数据源健康检查

### 3. 性能监控和告警
- 实时收集延迟统计（平均值、P95、P99分位数）
- 多级别告警系统（INFO/WARNING/ERROR/CRITICAL）
- 告警冷却机制避免告警风暴

### 4. 指标收集和分析
- 延迟分布统计
- 数据源性能对比
- 系统健康状态监控

## 快速开始

### 基本配置

```python
from src.hft.hft_engine import HFTEngine, HFTConfig
from src.hft.latency_monitor import DataSourceConfig

# 1. 配置数据源
data_sources = [
    DataSourceConfig(
        name="binance_ws",
        priority=1,           # 最高优先级
        max_latency_ms=30.0,
        timeout_ms=1000.0
    ),
    DataSourceConfig(
        name="okx_ws", 
        priority=2,           # 备用数据源
        max_latency_ms=50.0,
        timeout_ms=1500.0
    )
]

# 2. 配置HFT引擎
config = HFTConfig(
    staleness_threshold_ms=100.0,     # 数据过期阈值
    latency_stats_window=1000,        # 统计窗口大小
    alert_cooldown_seconds=30.0,      # 告警冷却时间
    enable_latency_monitoring=True    # 启用延迟监控
)

# 3. 创建和初始化引擎
engine = HFTEngine(config)
await engine.initialize(["BTCUSDT", "ETHUSDT"], data_sources)
await engine.start()
```

### 市场数据更新

```python
from src.core.models import MarketData
import time

# 创建市场数据
market_data = MarketData(
    symbol="BTCUSDT",
    timestamp=int(time.time() * 1000),  # 毫秒时间戳
    price=50000.0,
    volume=1.5,
    bid=49990.0,
    ask=50010.0,
    bid_volume=10.0,
    ask_volume=8.0
)

# 更新市场数据（包含延迟检查）
success = await engine.update_market_data(
    "BTCUSDT", 
    market_data, 
    "binance_ws"  # 指定数据源
)

if not success:
    print("数据更新失败，可能是延迟超标")
```

### 监控和告警

```python
# 设置告警处理
def handle_alert(alert):
    print(f"延迟告警: {alert.message}")
    print(f"级别: {alert.level.value}")
    print(f"延迟: {alert.latency_ms}ms")

if engine.latency_monitor:
    engine.latency_monitor.add_alert_callback(handle_alert)

# 获取延迟统计
stats = engine.get_latency_stats("BTCUSDT")
print(f"平均延迟: {stats['avg_latency_ms']}ms")
print(f"P99延迟: {stats['p99_latency_ms']}ms")
```

## 配置选项

### HFTConfig延迟相关配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `staleness_threshold_ms` | float | 100.0 | 数据过期阈值（毫秒） |
| `latency_stats_window` | int | 1000 | 延迟统计窗口大小 |
| `alert_cooldown_seconds` | float | 60.0 | 告警冷却时间（秒） |
| `enable_latency_monitoring` | bool | True | 是否启用延迟监控 |

### DataSourceConfig配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 数据源名称 |
| `priority` | int | 优先级（数字越小优先级越高） |
| `max_latency_ms` | float | 最大允许延迟（毫秒） |
| `timeout_ms` | float | 超时时间（毫秒） |
| `retry_count` | int | 重试次数 |
| `health_check_interval` | float | 健康检查间隔（秒） |

## API参考

### 核心方法

#### `get_latency_stats(symbol: str) -> Dict`
获取指定交易对的延迟统计信息。

**返回字段:**
- `avg_latency_ms`: 平均延迟
- `p95_latency_ms`: 95分位数延迟
- `p99_latency_ms`: 99分位数延迟
- `max_latency_ms`: 最大延迟
- `stale_data_rate`: 过期数据率
- `error_rate`: 错误率

#### `get_data_source_status() -> Dict[str, str]`
获取所有数据源的状态。

**状态值:**
- `active`: 活跃
- `degraded`: 降级
- `failed`: 失效
- `inactive`: 不活跃

#### `get_active_data_sources() -> Dict[str, str]`
获取每个交易对当前使用的数据源。

#### `switch_data_source(symbol: str, reason: str) -> Optional[str]`
手动切换指定交易对的数据源。

### 系统监控

#### `get_system_status() -> Dict`
获取完整的系统状态，包含延迟监控相关指标。

**关键指标:**
- `data_freshness_checks`: 数据新鲜度检查次数
- `stale_data_detections`: 过期数据检测次数
- `data_source_switches`: 数据源切换次数
- `avg_data_latency_ms`: 平均数据延迟
- `p99_data_latency_ms`: P99数据延迟

## 最佳实践

### 1. 数据源配置
```python
# 推荐配置：主要 + 备用 + 本地缓存
data_sources = [
    DataSourceConfig(
        name="primary_exchange",
        priority=1,
        max_latency_ms=20.0    # 严格延迟要求
    ),
    DataSourceConfig(
        name="backup_exchange", 
        priority=2,
        max_latency_ms=50.0    # 适度放宽
    ),
    DataSourceConfig(
        name="local_cache",
        priority=3,
        max_latency_ms=100.0   # 最后备选
    )
]
```

### 2. 告警处理
```python
def smart_alert_handler(alert):
    if alert.level == AlertLevel.CRITICAL:
        # 关键告警：暂停交易
        engine.pause_trading()
        send_emergency_notification(alert)
    
    elif alert.level == AlertLevel.ERROR:
        # 错误告警：切换到保守模式
        engine.switch_to_conservative_mode()
        log_error(alert)
    
    elif alert.level == AlertLevel.WARNING:
        # 警告告警：记录并监控
        log_warning(alert)
        increase_monitoring_frequency()
```

### 3. 性能优化
```python
# 适当设置统计窗口大小
config = HFTConfig(
    latency_stats_window=500,        # 较小窗口节省内存
    staleness_threshold_ms=50.0,     # 严格的新鲜度要求
    alert_cooldown_seconds=10.0      # 较短冷却时间
)
```

### 4. 监控仪表板
```python
async def print_monitoring_dashboard():
    status = engine.get_system_status()
    
    print("=== 延迟监控仪表板 ===")
    print(f"总更新: {status['metrics']['total_updates']}")
    print(f"过期数据率: {status['metrics']['stale_data_detections'] / max(status['metrics']['data_freshness_checks'], 1):.2%}")
    print(f"平均延迟: {status['metrics']['avg_data_latency_ms']:.2f}ms")
    
    for symbol in engine._symbols:
        stats = engine.get_latency_stats(symbol)
        if stats:
            print(f"{symbol}: {stats['avg_latency_ms']:.1f}ms (P99: {stats['p99_latency_ms']:.1f}ms)")
```

## 故障排除

### 常见问题

**1. 延迟监控未启用**
```python
# 检查配置
config = HFTConfig(enable_latency_monitoring=True)
```

**2. 数据源切换过于频繁**
```python
# 调整阈值和冷却时间
config = HFTConfig(
    staleness_threshold_ms=150.0,    # 放宽阈值
    alert_cooldown_seconds=60.0      # 增加冷却时间
)
```

**3. 内存使用过高**
```python
# 减少统计窗口大小
config = HFTConfig(latency_stats_window=200)
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger('src.hft.latency_monitor').setLevel(logging.DEBUG)

# 监控系统健康
health = engine.latency_monitor.get_system_health()
print(f"监控运行状态: {health['running']}")
print(f"数据源状态: {health['data_sources']}")

# 检查告警历史
# 可以通过自定义告警处理器收集告警历史
alert_history = []
engine.latency_monitor.add_alert_callback(lambda alert: alert_history.append(alert))
```

## 示例和演示

完整的使用示例请参考：
- `examples/latency_monitoring_example.py` - 完整演示程序
- `tests/test_latency_monitor.py` - 单元测试
- `tests/test_hft_latency_integration.py` - 集成测试

运行演示：
```bash
python3 examples/latency_monitoring_example.py
```

## 性能考虑

- 延迟监控本身的开销约为 **1-3ms** 每次检查
- 内存使用与统计窗口大小成正比
- 建议在生产环境中监控延迟监控系统本身的性能
- 告警冷却机制有效防止告警风暴

## 未来扩展

延迟监控系统支持以下扩展：

1. **自适应阈值**: 基于历史数据动态调整延迟阈值
2. **机器学习预测**: 预测延迟趋势和潜在问题
3. **分布式监控**: 支持多节点部署的延迟监控
4. **更多数据源**: 扩展支持更多类型的数据源
5. **可视化界面**: Web界面展示延迟监控数据