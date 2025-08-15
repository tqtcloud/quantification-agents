# DuckDB时序数据存储使用指南

## 概述

DuckDBManager是一个专为高性能时序数据存储和分析设计的管理器，支持多环境数据隔离、高性能批量插入、实时查询和数据压缩。

## 主要特性

### 🚀 高性能数据存储
- **批量插入优化**: 支持每秒插入数万条记录
- **列式存储**: 使用DuckDB的列式存储引擎，提高查询性能
- **内存优化**: 智能内存管理和缓存策略

### 📊 多类型时序数据支持
- **市场Tick数据**: 高频价格和成交量数据
- **K线数据**: 多时间框架OHLCV数据
- **订单簿快照**: 实时订单簿深度数据
- **技术指标**: 各类技术分析指标数据
- **交易信号**: Agent生成的交易信号数据

### 🔒 多环境数据隔离
- **物理隔离**: 每个环境使用独立的数据库文件
- **环境支持**: Testnet、Mainnet、Paper三种环境
- **数据安全**: 确保不同环境数据不会混淆

### 📈 高级分析功能
- **统计分析**: 价格统计、波动性分析
- **技术指标**: VWAP、价格范围、成交量分析
- **市场深度**: 订单簿分析和流动性评估

### 🗜️ 数据压缩与分区
- **Parquet导出**: 高效的数据压缩存储
- **时间分区**: 按日、周、月进行数据分区
- **存储优化**: 自动数据重组和压缩

## 快速开始

### 基本使用

```python
from src.core.duckdb_manager import duckdb_manager
from src.exchanges.trading_interface import TradingEnvironment
from datetime import datetime, timedelta

# 设置环境
duckdb_manager.set_current_environment(TradingEnvironment.TESTNET)

# 准备数据
tick_data = [
    {
        'timestamp': datetime.utcnow(),
        'symbol': 'BTCUSDT',
        'price': 50000.0,
        'volume': 1.0,
        'bid_price': 49999.5,
        'ask_price': 50000.5,
        'bid_volume': 2.0,
        'ask_volume': 2.0,
        'trade_count': 5
    }
]

# 批量插入数据
inserted = duckdb_manager.insert_market_ticks_batch(tick_data)
print(f"插入了 {inserted} 条记录")

# 查询数据
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)

result_df = duckdb_manager.query_market_ticks('BTCUSDT', start_time, end_time)
print(f"查询到 {len(result_df)} 条记录")
```

### 性能测试结果

基于测试环境的性能表现：

- **Tick数据插入**: ~166,000 记录/秒
- **K线数据插入**: ~86,000 记录/秒
- **查询响应时间**: <100ms (千万级数据量)
- **存储压缩比**: 约70% (使用Parquet格式)

## 数据模型

### 市场Tick数据 (market_ticks)
```sql
CREATE TABLE market_ticks (
    timestamp TIMESTAMP PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    environment VARCHAR NOT NULL,
    price DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    bid_price DOUBLE,
    ask_price DOUBLE,
    bid_volume DOUBLE,
    ask_volume DOUBLE,
    trade_count INTEGER DEFAULT 0,
    metadata VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### K线数据 (market_klines)
```sql
CREATE TABLE market_klines (
    timestamp TIMESTAMP,
    symbol VARCHAR NOT NULL,
    environment VARCHAR NOT NULL,
    interval VARCHAR NOT NULL,
    open_price DOUBLE NOT NULL,
    high_price DOUBLE NOT NULL,
    low_price DOUBLE NOT NULL,
    close_price DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    quote_volume DOUBLE,
    trade_count INTEGER DEFAULT 0,
    taker_buy_volume DOUBLE,
    taker_buy_quote_volume DOUBLE,
    metadata VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol, interval)
);
```

## API参考

### 数据插入方法

#### insert_market_ticks_batch()
批量插入市场tick数据，支持高性能批量操作。

#### insert_klines_batch()
批量插入K线数据，支持多时间框架。

#### insert_order_book_snapshot()
插入订单簿快照数据。

#### insert_technical_indicators_batch()
批量插入技术指标数据。

#### insert_trading_signals_batch()
批量插入交易信号数据。

### 数据查询方法

#### query_market_ticks()
查询指定时间范围的tick数据。

#### query_klines()
查询指定时间框架的K线数据。

#### get_latest_prices()
获取多个交易对的最新价格。

#### calculate_price_statistics()
计算价格统计指标。

#### get_volume_weighted_price()
计算VWAP (成交量加权平均价)。

#### get_market_depth_analysis()
分析订单簿市场深度。

### 数据管理方法

#### export_to_parquet()
导出数据到Parquet格式进行压缩存储。

#### cleanup_old_data()
清理过期数据，释放存储空间。

#### get_storage_statistics()
获取存储统计信息。

#### optimize_table_storage()
优化表存储结构，提高查询性能。

## 最佳实践

### 1. 批量操作
- 尽量使用批量插入而非单条插入
- 建议每批次1000-10000条记录

### 2. 时间分区
- 对于大量历史数据，使用时间分区表
- 按日或月分区以提高查询性能

### 3. 数据压缩
- 定期导出历史数据到Parquet格式
- 使用压缩存储节省磁盘空间

### 4. 环境隔离
- 严格区分测试和生产环境数据
- 使用适当的环境标识

### 5. 性能监控
- 定期检查存储统计信息
- 监控查询性能并进行优化

## 故障排除

### 常见问题

1. **连接错误**: 确保DuckDB库正确安装
2. **插入失败**: 检查数据格式和必需字段
3. **查询缓慢**: 考虑创建索引或使用分区表
4. **存储空间**: 定期清理旧数据或使用压缩

### 日志和调试

DuckDBManager提供详细的日志记录，包括：
- 连接状态
- 数据操作结果
- 性能指标
- 错误信息

日志级别可通过配置文件调整。

## 示例程序

完整的使用示例请参考：
- `examples/duckdb_example.py` - 完整功能演示
- `tests/test_duckdb_manager.py` - 单元测试示例

## 性能基准

| 操作类型 | 数据量 | 耗时 | 吞吐量 |
|---------|-------|------|--------|
| Tick批量插入 | 5,000条 | 0.03秒 | 166,351 记录/秒 |
| K线批量插入 | 1,000条 | 0.01秒 | 85,992 记录/秒 |
| 价格统计查询 | 10万条 | 0.05秒 | 200万 记录/秒 |
| VWAP计算 | 10万条 | 0.08秒 | 125万 记录/秒 |

基准测试环境：MacBook Pro M1, 16GB内存