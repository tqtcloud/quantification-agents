"""
DuckDB存储系统模块
提供高性能的历史数据存储和查询功能
"""

# 延迟导入以避免初始化问题
def get_duckdb_storage(*args, **kwargs):
    from .duckdb_storage import DuckDBStorage
    return DuckDBStorage(*args, **kwargs)
from .models import (
    HistoricalMarketData,
    HistoricalTradingSignal,
    HistoricalOrderRecord,
    HistoricalRiskMetrics,
    DataRetentionPolicy,
    StorageConfig
)

__all__ = [
    "get_duckdb_storage",
    "HistoricalMarketData",
    "HistoricalTradingSignal", 
    "HistoricalOrderRecord",
    "HistoricalRiskMetrics",
    "DataRetentionPolicy",
    "StorageConfig"
]