# DuckDB历史数据存储系统

## 概述

DuckDB存储系统是量化交易平台的核心组件之一，专门用于存储和分析历史交易数据。它提供高性能的OLAP查询能力，支持复杂的时序数据分析和报表生成。

## 主要特性

### 🚀 高性能存储
- **列式存储**: 优化分析型查询性能
- **压缩优化**: 支持多种压缩算法（SNAPPY、GZIP、ZSTD等）
- **分区策略**: 基于时间的智能分区
- **并发支持**: 多线程并发读写

### 📊 数据模型
- **历史市场数据**: OHLCV数据，支持多时间框架
- **交易信号历史**: 包含执行状态和绩效跟踪
- **订单记录**: 完整订单生命周期记录
- **风险指标**: 综合风险评估和绩效统计

### 🔄 生命周期管理
- **自动清理**: 基于保留策略的过期数据清理
- **数据归档**: 历史数据压缩归档到Parquet
- **备份恢复**: 完整和增量备份机制
- **健康监控**: 存储健康状态监控和报告

## 快速开始

### 安装依赖

```bash
pip install duckdb pandas numpy pydantic
```

### 基本使用

```python
from src.core.storage import DuckDBStorage, StorageConfig
from src.exchanges.trading_interface import TradingEnvironment

# 创建存储配置
config = StorageConfig(
    max_memory_gb=4.0,
    thread_count=4,
    enable_compression=True
)

# 初始化存储系统
storage = DuckDBStorage(config=config)

# 存储市场数据
await storage.store_historical_market_data(market_data, TradingEnvironment.TESTNET)

# 查询数据
data = await storage.query_historical_market_data(
    symbol='BTCUSDT',
    interval='1m',
    start_time=start_time,
    end_time=end_time
)
```

## 数据模型详解

### 历史市场数据 (HistoricalMarketData)

```python
from decimal import Decimal
from datetime import datetime
from src.core.storage.models import HistoricalMarketData

market_data = HistoricalMarketData(
    timestamp=datetime.utcnow(),
    symbol='BTCUSDT',
    environment='testnet',
    interval='1m',
    open_price=Decimal('50000.00'),
    high_price=Decimal('50100.00'),
    low_price=Decimal('49900.00'),
    close_price=Decimal('50050.00'),
    volume=Decimal('1.5'),
    quote_volume=Decimal('75000.00'),
    trade_count=100,
    data_source='binance',
    metadata={'exchange': 'binance', 'quality': 'high'}
)
```

**字段说明:**
- `timestamp`: 数据时间戳
- `symbol`: 交易品种
- `environment`: 环境标识（testnet/mainnet）
- `interval`: 时间间隔（1m, 5m, 1h, 1d等）
- `open_price/high_price/low_price/close_price`: OHLC价格数据
- `volume/quote_volume`: 成交量数据
- `trade_count`: 成交笔数
- `data_source`: 数据来源
- `metadata`: 额外元数据

### 历史交易信号 (HistoricalTradingSignal)

```python
from src.core.storage.models import HistoricalTradingSignal

signal = HistoricalTradingSignal(
    timestamp=datetime.utcnow(),
    symbol='BTCUSDT',
    environment='testnet',
    signal_id='momentum_001',
    source='momentum_agent',
    signal_type='momentum',
    action='BUY',
    strength=0.8,
    confidence=0.9,
    target_price=Decimal('51000.00'),
    suggested_quantity=Decimal('0.1'),
    reason='强劲上涨动量信号',
    technical_context={
        'rsi': 35,
        'macd': 0.05,
        'volume_ratio': 1.5
    }
)
```

**关键字段:**
- `signal_id`: 信号唯一标识符
- `source`: 信号来源（Agent名称）
- `action`: 交易动作（BUY/SELL/HOLD）
- `strength`: 信号强度（-1到1）
- `confidence`: 信号置信度（0到1）
- `execution_status`: 执行状态
- `realized_pnl`: 已实现盈亏

### 历史订单记录 (HistoricalOrderRecord)

```python
from src.core.storage.models import HistoricalOrderRecord

order = HistoricalOrderRecord(
    timestamp=datetime.utcnow(),
    symbol='BTCUSDT',
    environment='testnet',
    order_id='order_123',
    client_order_id='client_123',
    side='BUY',
    order_type='LIMIT',
    quantity=Decimal('0.1'),
    price=Decimal('50000.00'),
    status='FILLED',
    executed_qty=Decimal('0.1'),
    avg_price=Decimal('50050.00'),
    commission=Decimal('5.00'),
    commission_asset='USDT',
    realized_pnl=Decimal('25.00')
)
```

**核心字段:**
- `order_id`: 交易所订单ID
- `client_order_id`: 客户端订单ID
- `side`: 买卖方向
- `order_type`: 订单类型
- `status`: 订单状态
- `executed_qty`: 已执行数量
- `realized_pnl`: 已实现盈亏

### 历史风险指标 (HistoricalRiskMetrics)

```python
from src.core.storage.models import HistoricalRiskMetrics

risk_metrics = HistoricalRiskMetrics(
    timestamp=datetime.utcnow(),
    environment='testnet',
    calculation_period='1h',
    total_balance=Decimal('10000.00'),
    effective_leverage=2.0,
    max_drawdown=0.05,
    current_drawdown=0.02,
    var_95=Decimal('-100.00'),
    sharpe_ratio=1.5,
    win_rate=0.65,
    profit_factor=2.1
)
```

**重要指标:**
- `effective_leverage`: 有效杠杆
- `max_drawdown`: 最大回撤
- `var_95`: 95%置信度VaR
- `sharpe_ratio`: 夏普比率
- `profit_factor`: 盈利因子

## 配置选项

### StorageConfig

```python
from src.core.storage.models import StorageConfig, CompressionType

config = StorageConfig(
    # 基本配置
    max_memory_gb=4.0,          # 最大内存使用
    thread_count=4,             # 线程数
    checkpoint_threshold_gb=2.0, # 检查点阈值
    
    # 压缩配置
    default_compression=CompressionType.SNAPPY,
    enable_compression=True,
    
    # 分区配置
    partition_strategy=PartitionStrategy.MONTHLY,
    enable_partitioning=True,
    
    # 优化配置
    auto_optimize=True,
    optimize_interval_hours=24
)
```

### DataRetentionPolicy

```python
from src.core.storage.models import DataRetentionPolicy

retention = DataRetentionPolicy(
    # 数据保留期（天数）
    market_data_retention_days=365,     # 市场数据1年
    signal_retention_days=180,          # 信号半年
    order_retention_days=730,           # 订单2年
    risk_metrics_retention_days=365,    # 风险指标1年
    
    # 压缩和归档
    compress_after_days=30,
    archive_after_days=90,
    
    # 自动清理
    enable_auto_cleanup=True,
    cleanup_interval_hours=24
)
```

## API参考

### DuckDBStorage

#### 数据存储方法

```python
# 异步存储市场数据
await storage.store_historical_market_data(
    data: List[HistoricalMarketData],
    environment: TradingEnvironment
) -> int

# 异步存储交易信号
await storage.store_historical_trading_signals(
    signals: List[HistoricalTradingSignal],
    environment: TradingEnvironment
) -> int

# 异步存储订单记录
await storage.store_historical_order_records(
    orders: List[HistoricalOrderRecord],
    environment: TradingEnvironment
) -> int

# 异步存储风险指标
await storage.store_historical_risk_metrics(
    metrics: List[HistoricalRiskMetrics],
    environment: TradingEnvironment
) -> int
```

#### 数据查询方法

```python
# 查询历史市场数据
await storage.query_historical_market_data(
    symbol: str,
    interval: str,
    start_time: datetime,
    end_time: datetime,
    environment: TradingEnvironment,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]

# 查询历史交易信号
await storage.query_historical_trading_signals(
    symbol: Optional[str] = None,
    source: Optional[str] = None,
    signal_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    environment: TradingEnvironment,
    limit: Optional[int] = 1000
) -> List[Dict[str, Any]]
```

#### 分析查询方法

```python
# 获取交易绩效分析
await storage.get_trading_performance_analysis(
    symbol: Optional[str] = None,
    strategy_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    environment: TradingEnvironment
) -> Dict[str, Any]

# 获取存储统计
await storage.get_storage_statistics(
    environment: TradingEnvironment
) -> Dict[str, Any]
```

#### 生命周期管理

```python
# 清理过期数据
await storage.cleanup_expired_data(
    environment: TradingEnvironment
) -> Dict[str, int]

# 导出数据到Parquet
await storage.export_data_to_parquet(
    table_name: str,
    output_path: Path,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    environment: TradingEnvironment
) -> bool
```

### DataMigrationManager

```python
from src.core.storage.data_migration import DataMigrationManager

migration_manager = DataMigrationManager()

# 创建备份
backup_info = await migration_manager.create_backup(
    environment=TradingEnvironment.TESTNET,
    backup_type="incremental"  # or "full"
)

# 数据归档
archive_result = await migration_manager.archive_old_data(
    environment=TradingEnvironment.TESTNET,
    archive_before_days=90
)

# 存储优化
optimization_result = await migration_manager.optimize_storage(
    environment=TradingEnvironment.TESTNET
)

# 获取健康报告
health_report = migration_manager.get_storage_health_report(
    environment=TradingEnvironment.TESTNET
)
```

## 性能优化

### 查询优化

1. **使用适当的时间范围**
```python
# 好的实践：限制查询时间范围
data = await storage.query_historical_market_data(
    symbol='BTCUSDT',
    interval='1m',
    start_time=datetime.utcnow() - timedelta(hours=1),
    end_time=datetime.utcnow(),
    limit=100
)
```

2. **利用索引**
```python
# 查询会利用以下索引：
# - (symbol, timestamp)
# - (interval, timestamp)  
# - (environment, timestamp)
```

3. **批量操作**
```python
# 批量存储而不是单条存储
await storage.store_historical_market_data(batch_data, env)
```

### 存储优化

1. **定期优化表结构**
```python
# 手动触发存储优化
await migration_manager.optimize_storage(TradingEnvironment.TESTNET)
```

2. **配置合适的内存限制**
```python
config = StorageConfig(
    max_memory_gb=8.0,  # 根据系统内存调整
    thread_count=6      # 根据CPU核心数调整
)
```

3. **使用压缩**
```python
config = StorageConfig(
    enable_compression=True,
    default_compression=CompressionType.SNAPPY  # 平衡压缩率和性能
)
```

## 监控和维护

### 存储监控

```python
# 获取存储统计
stats = await storage.get_storage_statistics(TradingEnvironment.TESTNET)
print(f"数据库大小: {stats['file_size_mb']} MB")
print(f"总记录数: {stats['total_rows']}")

# 获取健康报告
health = migration_manager.get_storage_health_report(TradingEnvironment.TESTNET)
print(f"健康评分: {health['health_score']}/100")
```

### 定期维护任务

```python
async def daily_maintenance():
    """每日维护任务"""
    # 1. 清理过期数据
    await storage.cleanup_expired_data(TradingEnvironment.TESTNET)
    
    # 2. 创建增量备份
    await migration_manager.create_backup(
        TradingEnvironment.TESTNET, 
        "incremental"
    )
    
    # 3. 存储优化
    await migration_manager.optimize_storage(TradingEnvironment.TESTNET)
    
    # 4. 健康检查
    health = migration_manager.get_storage_health_report(TradingEnvironment.TESTNET)
    if health['health_score'] < 80:
        print(f"存储健康警告: {health['recommendations']}")
```

## 故障排除

### 常见问题

1. **内存不足错误**
```python
# 解决方案：减少内存使用或增加系统内存
config = StorageConfig(max_memory_gb=2.0)  # 减少内存限制
```

2. **查询性能慢**
```python
# 解决方案：
# - 缩小查询时间范围
# - 添加LIMIT限制
# - 定期执行存储优化
```

3. **数据库文件过大**
```python
# 解决方案：
# - 减少数据保留期
# - 启用数据压缩和归档
# - 定期清理过期数据
```

### 日志记录

存储系统使用结构化日志记录，可以通过以下方式查看日志：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

关键日志消息包括：
- 数据存储操作
- 查询性能统计
- 维护任务执行
- 错误和警告信息

## 最佳实践

### 1. 数据模型设计
- 使用强类型数据模型确保数据一致性
- 合理设置字段验证规则
- 为重要数据添加元数据字段

### 2. 性能优化
- 批量操作优于单条操作
- 合理设置查询时间范围
- 定期执行存储优化

### 3. 数据管理
- 配置合适的数据保留策略
- 定期创建备份
- 监控存储健康状态

### 4. 错误处理
- 实现重试机制处理临时故障
- 记录详细的错误日志
- 监控数据质量指标

## 示例代码

完整的使用示例请参考：
- `/examples/storage/duckdb_storage_example.py`
- `/tests/core/storage/test_duckdb_storage.py`

## 扩展功能

### 自定义分析查询

```python
# 自定义SQL查询示例
async def custom_analysis(storage, environment):
    conn = storage.duckdb_manager.get_connection(environment)
    
    # 计算每日收益率
    query = """
    SELECT 
        DATE_TRUNC('day', timestamp) as date,
        symbol,
        FIRST(close_price ORDER BY timestamp) as day_open,
        LAST(close_price ORDER BY timestamp) as day_close,
        (LAST(close_price ORDER BY timestamp) - FIRST(close_price ORDER BY timestamp)) 
        / FIRST(close_price ORDER BY timestamp) as daily_return
    FROM historical_market_data
    WHERE symbol = 'BTCUSDT' AND interval = '1m'
    GROUP BY DATE_TRUNC('day', timestamp), symbol
    ORDER BY date
    """
    
    result = conn.execute(query).fetchdf()
    return result
```

### 集成外部数据源

```python
# 从外部API导入数据
async def import_external_data(storage):
    # 从API获取数据
    external_data = await fetch_from_api()
    
    # 转换为内部数据模型
    market_data = [
        HistoricalMarketData(**item) 
        for item in external_data
    ]
    
    # 存储到DuckDB
    await storage.store_historical_market_data(
        market_data, 
        TradingEnvironment.TESTNET
    )
```

## 总结

DuckDB存储系统提供了完整的历史数据管理解决方案，具有以下核心优势：

- **高性能**: 列式存储和压缩优化
- **可扩展**: 支持大数据量存储和分析
- **易用性**: 简洁的API和强类型模型
- **可靠性**: 完整的备份恢复机制
- **可维护**: 自动化生命周期管理

通过合理配置和使用，DuckDB存储系统可以满足量化交易系统对历史数据存储和分析的所有需求。