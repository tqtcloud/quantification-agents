# DuckDB历史数据存储系统

## 🎯 概述

本项目实现了一个完整的DuckDB历史数据存储系统，专门为量化交易平台设计，提供高性能的OLAP查询和分析功能。

## 📁 目录结构

```
src/core/storage/
├── __init__.py              # 模块初始化和导出
├── models.py               # 数据模型定义
├── duckdb_storage.py      # 核心存储引擎
├── data_migration.py      # 数据迁移和生命周期管理
└── README.md              # 本文件
```

## 🚀 核心特性

### 1. 高性能数据存储
- **列式存储**: 优化分析型查询性能
- **压缩优化**: 支持SNAPPY、GZIP、ZSTD等压缩算法
- **智能分区**: 基于时间的数据分区策略
- **并发支持**: 多线程并发读写操作

### 2. 完整数据模型
- `HistoricalMarketData`: 历史OHLCV市场数据
- `HistoricalTradingSignal`: 交易信号历史记录
- `HistoricalOrderRecord`: 完整订单生命周期
- `HistoricalRiskMetrics`: 综合风险指标分析

### 3. 数据生命周期管理
- **自动清理**: 基于保留策略的过期数据清理
- **数据归档**: 历史数据压缩归档到Parquet
- **备份恢复**: 完整和增量备份机制
- **健康监控**: 存储健康状态实时监控

### 4. 企业级功能
- **数据验证**: 强类型模型和完整性检查
- **错误处理**: 全面的异常处理机制
- **性能优化**: 自动表结构优化
- **配置管理**: 灵活的存储配置选项

## 🛠 快速开始

### 基本使用

```python
from src.core.storage.models import StorageConfig, HistoricalMarketData
from src.core.storage.duckdb_storage import DuckDBStorage
from decimal import Decimal
from datetime import datetime

# 创建配置
config = StorageConfig(
    max_memory_gb=4.0,
    enable_compression=True,
    auto_optimize=True
)

# 初始化存储
storage = DuckDBStorage(config=config)

# 创建市场数据
market_data = HistoricalMarketData(
    timestamp=datetime.utcnow(),
    symbol='BTCUSDT',
    environment='testnet',
    interval='1m',
    open_price=Decimal('50000.00'),
    high_price=Decimal('50100.00'),
    low_price=Decimal('49900.00'),
    close_price=Decimal('50050.00'),
    volume=Decimal('1.5')
)

# 存储数据
await storage.store_historical_market_data([market_data])

# 查询数据
results = await storage.query_historical_market_data(
    symbol='BTCUSDT',
    interval='1m',
    start_time=start_time,
    end_time=end_time
)
```

### 数据迁移管理

```python
from src.core.storage.data_migration import DataMigrationManager

# 创建迁移管理器
migration = DataMigrationManager()

# 创建备份
backup_info = await migration.create_backup(
    environment=TradingEnvironment.TESTNET,
    backup_type="incremental"
)

# 数据归档
archive_result = await migration.archive_old_data(
    environment=TradingEnvironment.TESTNET
)

# 获取健康报告
health_report = migration.get_storage_health_report(
    environment=TradingEnvironment.TESTNET
)
```

## 📊 数据模型

### 历史市场数据 (HistoricalMarketData)
- **OHLCV数据**: 开盘、最高、最低、收盘价格和成交量
- **订单簿数据**: 买卖盘深度信息
- **衍生品数据**: 持仓量、资金费率、标记价格
- **数据质量**: 数据来源和质量评分

### 历史交易信号 (HistoricalTradingSignal)
- **信号信息**: 来源、类型、动作、强度、置信度
- **执行状态**: 挂起、已执行、已取消、已过期
- **绩效跟踪**: 已实现盈亏、胜率统计
- **技术上下文**: 技术指标和市场状况

### 历史订单记录 (HistoricalOrderRecord)
- **基本信息**: 订单ID、品种、方向、类型、数量
- **执行信息**: 状态、已执行数量、平均价格
- **成本信息**: 手续费、滑点成本
- **绩效信息**: 已实现盈亏、盈亏百分比

### 历史风险指标 (HistoricalRiskMetrics)
- **暴露度风险**: 总暴露、多空暴露、净暴露
- **保证金风险**: 已用保证金、保证金比率、维持保证金
- **回撤风险**: 最大回撤、当前回撤、恢复因子
- **VaR风险**: 95%和99%置信度的VaR和ES
- **绩效指标**: 夏普比率、索提诺比率、盈利因子

## ⚙️ 配置选项

### StorageConfig
```python
StorageConfig(
    max_memory_gb=4.0,              # 最大内存使用
    thread_count=4,                 # 线程池大小
    enable_compression=True,        # 启用压缩
    default_compression=SNAPPY,     # 默认压缩算法
    enable_partitioning=True,       # 启用分区
    partition_strategy=MONTHLY,     # 分区策略
    auto_optimize=True,             # 自动优化
    optimize_interval_hours=24      # 优化间隔
)
```

### DataRetentionPolicy
```python
DataRetentionPolicy(
    market_data_retention_days=365,     # 市场数据保留期
    signal_retention_days=180,          # 信号保留期
    order_retention_days=730,           # 订单保留期
    risk_metrics_retention_days=365,    # 风险指标保留期
    enable_auto_cleanup=True,           # 启用自动清理
    cleanup_interval_hours=24,          # 清理间隔
    enable_archiving=True,              # 启用归档
    archive_to_parquet=True            # 归档为Parquet格式
)
```

## 🔧 API参考

### 数据存储
- `store_historical_market_data()` - 存储市场数据
- `store_historical_trading_signals()` - 存储交易信号
- `store_historical_order_records()` - 存储订单记录
- `store_historical_risk_metrics()` - 存储风险指标

### 数据查询
- `query_historical_market_data()` - 查询市场数据
- `query_historical_trading_signals()` - 查询交易信号
- `get_trading_performance_analysis()` - 获取绩效分析
- `get_storage_statistics()` - 获取存储统计

### 生命周期管理
- `cleanup_expired_data()` - 清理过期数据
- `export_data_to_parquet()` - 导出到Parquet
- `create_backup()` - 创建备份
- `archive_old_data()` - 归档数据
- `optimize_storage()` - 优化存储

## 🎯 性能优化

### 查询优化
1. **时间范围限制**: 避免全表扫描
2. **索引利用**: 充分利用时间戳和品种索引
3. **批量操作**: 批量存储和查询
4. **结果限制**: 使用LIMIT控制返回数据量

### 存储优化
1. **压缩启用**: 减少磁盘空间占用
2. **分区策略**: 提高查询性能
3. **定期维护**: 自动表结构优化
4. **内存管理**: 合理配置内存使用

## 📈 监控和维护

### 健康监控
- **存储统计**: 表数量、记录数、文件大小
- **健康评分**: 0-100分的健康状态评估
- **建议生成**: 自动生成维护建议
- **备份状态**: 备份数量和时间跟踪

### 日常维护
- **数据清理**: 自动清理过期数据
- **存储优化**: 定期重组表结构
- **备份创建**: 增量和完整备份
- **健康检查**: 定期健康状态检查

## 🧪 测试

### 运行测试
```bash
# 数据模型测试
python3 test_models_only.py

# 基础功能测试
python3 test_duckdb_basic.py

# 完整集成测试
python3 -m pytest tests/core/storage/ -v
```

### 测试覆盖
- ✅ 数据模型验证和方法测试
- ✅ 配置系统测试
- ✅ 数据验证和错误处理
- ✅ 性能和并发测试
- ✅ 生命周期管理测试

## 📚 示例和文档

### 文件位置
- `/examples/storage/duckdb_storage_example.py` - 完整使用示例
- `/docs/duckdb_storage_guide.md` - 详细使用指南
- `/tests/core/storage/test_duckdb_storage.py` - 集成测试

### 运行示例
```bash
# 运行完整示例
python3 examples/storage/duckdb_storage_example.py
```

## 🤝 集成说明

### 与现有系统集成
1. **内存缓存**: 与Redis缓存系统协作，内存存实时数据，DuckDB存历史数据
2. **交易系统**: 接收实时交易数据并异步存储
3. **分析系统**: 为回测和分析提供历史数据支持
4. **监控系统**: 集成存储健康监控和告警

### 部署建议
1. **硬件要求**: 推荐SSD存储，充足内存
2. **网络配置**: 低延迟网络连接
3. **备份策略**: 定期异地备份
4. **监控告警**: 设置存储容量和性能告警

## 📝 变更日志

### v1.0.0 (2024-08-24)
- ✅ 完整的DuckDB存储系统实现
- ✅ 强类型数据模型定义
- ✅ 数据生命周期管理
- ✅ 备份恢复功能
- ✅ 性能优化和监控
- ✅ 完整的测试覆盖
- ✅ 详细的文档和示例

## 🔗 相关资源

- [DuckDB官方文档](https://duckdb.org/docs/)
- [Pydantic数据验证](https://pydantic-docs.helpmanual.io/)
- [量化交易系统架构](../../../docs/)

## 💡 最佳实践

1. **数据建模**: 使用强类型模型确保数据一致性
2. **性能优化**: 合理配置内存和线程数量
3. **数据管理**: 制定合适的数据保留和归档策略
4. **监控维护**: 定期检查存储健康状态
5. **备份恢复**: 建立可靠的备份恢复流程

---

**注意**: 本存储系统设计为高性能OLAP分析场景，适合历史数据存储和复杂查询分析，与Redis等内存缓存系统配合使用可获得最佳性能。