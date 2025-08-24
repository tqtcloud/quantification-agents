"""
DuckDB数据迁移和生命周期管理
提供数据迁移、归档、备份和恢复功能
"""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    from src.config import settings
    from src.utils.logger import LoggerMixin
    from src.core.duckdb_manager import duckdb_manager
except ImportError:
    # 提供默认值以支持独立测试
    class MockSettings:
        data_directory = "./data"
    settings = MockSettings()
    
    class LoggerMixin:
        def log_info(self, msg): print(f"INFO: {msg}")
        def log_error(self, msg): print(f"ERROR: {msg}")
        def log_warning(self, msg): print(f"WARNING: {msg}")
        def log_debug(self, msg): print(f"DEBUG: {msg}")
    
    class MockDuckDBManager:
        def get_connection(self, env): pass
        def get_current_environment(self): return None
        def _get_database_path(self, env): return Path("./test.db")
    duckdb_manager = MockDuckDBManager()

try:
    from src.exchanges.trading_interface import TradingEnvironment
except ImportError:
    from enum import Enum
    class TradingEnvironment(Enum):
        TESTNET = "testnet"
        MAINNET = "mainnet"
        PAPER = "paper"
from .models import StorageConfig, DataRetentionPolicy, DataCategory, CompressionType


class DataMigrationManager(LoggerMixin):
    """数据迁移和生命周期管理器"""
    
    def __init__(
        self, 
        config: Optional[StorageConfig] = None,
        retention_policy: Optional[DataRetentionPolicy] = None
    ):
        super().__init__()
        
        self.config = config or StorageConfig()
        self.retention_policy = retention_policy or DataRetentionPolicy()
        self.duckdb_manager = duckdb_manager
        
        # 备份和归档目录
        self.backup_dir = Path(settings.data_directory) / "backups"
        self.archive_dir = Path(settings.data_directory) / "archives"
        self.migration_dir = Path(settings.data_directory) / "migrations"
        
        self._init_directories()
        
        # 线程池用于后台任务
        self.executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="DataMigration"
        )
        
        self.log_info("数据迁移管理器初始化完成")

    def _init_directories(self) -> None:
        """初始化目录结构"""
        for directory in [self.backup_dir, self.archive_dir, self.migration_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # 为每个环境创建子目录
        for env in TradingEnvironment:
            for directory in [self.backup_dir, self.archive_dir]:
                env_dir = directory / env.value
                env_dir.mkdir(exist_ok=True)

    # ==================
    # 数据迁移功能
    # ==================
    
    async def migrate_memory_cache_to_historical(
        self,
        environment: TradingEnvironment,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """从内存缓存迁移数据到历史存储"""
        def _migrate_sync():
            return self.migrate_memory_cache_to_historical_sync(environment, batch_size)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _migrate_sync
        )

    def migrate_memory_cache_to_historical_sync(
        self,
        environment: TradingEnvironment,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """同步从内存缓存迁移数据到历史存储"""
        migration_results = {
            'market_data': 0,
            'trading_signals': 0,
            'order_records': 0,
            'risk_metrics': 0
        }
        
        try:
            conn = self.duckdb_manager.get_connection(environment)
            
            # 迁移市场数据（从实时表到历史表）
            if self._table_exists(conn, 'market_ticks'):
                market_migrated = self._migrate_table_data(
                    conn, 'market_ticks', 'historical_market_data', 
                    self._transform_market_data, batch_size
                )
                migration_results['market_data'] = market_migrated
            
            # 迁移交易信号
            if self._table_exists(conn, 'trading_signals'):
                signals_migrated = self._migrate_table_data(
                    conn, 'trading_signals', 'historical_trading_signals',
                    self._transform_signal_data, batch_size
                )
                migration_results['trading_signals'] = signals_migrated
            
            # 清理已迁移的旧数据（保留最近1小时的数据）
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            self._cleanup_migrated_data(conn, cutoff_time)
            
            total_migrated = sum(migration_results.values())
            self.log_info(f"数据迁移完成: {environment.value}, 共迁移{total_migrated}条记录")
            
            return migration_results
            
        except Exception as e:
            self.log_error(f"数据迁移失败: {e}")
            raise

    def _table_exists(self, conn, table_name: str) -> bool:
        """检查表是否存在"""
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result[0] > 0
        except Exception:
            return False

    def _migrate_table_data(
        self,
        conn,
        source_table: str,
        target_table: str,
        transform_func,
        batch_size: int
    ) -> int:
        """迁移表数据"""
        try:
            # 获取需要迁移的数据总数
            count_result = conn.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()
            total_records = count_result[0] if count_result else 0
            
            if total_records == 0:
                return 0
            
            migrated_count = 0
            offset = 0
            
            while offset < total_records:
                # 分批读取数据
                query = f"""
                    SELECT * FROM {source_table}
                    ORDER BY timestamp
                    LIMIT {batch_size} OFFSET {offset}
                """
                
                batch_df = conn.execute(query).fetchdf()
                
                if batch_df.empty:
                    break
                
                # 转换数据格式
                transformed_data = transform_func(batch_df)
                
                # 插入到目标表
                if not transformed_data.empty:
                    conn.register('migration_batch', transformed_data)
                    
                    insert_sql = f"""
                        INSERT INTO {target_table}
                        SELECT * FROM migration_batch
                        ON CONFLICT DO NOTHING
                    """
                    
                    conn.execute(insert_sql)
                    migrated_count += len(transformed_data)
                
                offset += batch_size
            
            return migrated_count
            
        except Exception as e:
            self.log_error(f"表数据迁移失败 {source_table} -> {target_table}: {e}")
            return 0

    def _transform_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换市场数据格式"""
        try:
            # 重命名和转换字段以匹配历史表结构
            transformed = df.copy()
            
            # 确保必需字段存在
            required_columns = [
                'timestamp', 'symbol', 'environment', 'price', 'volume'
            ]
            
            for col in required_columns:
                if col not in transformed.columns:
                    if col == 'price':
                        # 如果没有价格字段，尝试使用其他价格字段
                        if 'close_price' in transformed.columns:
                            transformed['price'] = transformed['close_price']
                        elif 'bid_price' in transformed.columns:
                            transformed['price'] = transformed['bid_price']
                        else:
                            transformed['price'] = 0.0
            
            # 添加历史表需要的字段
            if 'interval' not in transformed.columns:
                transformed['interval'] = '1m'  # 默认1分钟间隔
            
            # 重命名字段以匹配历史表结构
            column_mapping = {
                'price': 'close_price',
                'bid': 'bid_price',
                'ask': 'ask_price',
                'bid_volume': 'bid_volume',
                'ask_volume': 'ask_volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in transformed.columns and new_col not in transformed.columns:
                    transformed[new_col] = transformed[old_col]
            
            # 为历史表添加OHLC数据（如果缺失）
            if 'open_price' not in transformed.columns:
                transformed['open_price'] = transformed['close_price']
            if 'high_price' not in transformed.columns:
                transformed['high_price'] = transformed['close_price']
            if 'low_price' not in transformed.columns:
                transformed['low_price'] = transformed['close_price']
            
            # 添加默认值
            transformed['trade_count'] = transformed.get('trade_count', 1)
            transformed['data_source'] = transformed.get('data_source', 'migration')
            transformed['data_quality'] = transformed.get('data_quality', 1.0)
            transformed['metadata'] = transformed.get('metadata', '{}')
            
            return transformed
            
        except Exception as e:
            self.log_error(f"市场数据转换失败: {e}")
            return pd.DataFrame()

    def _transform_signal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换交易信号数据格式"""
        try:
            transformed = df.copy()
            
            # 添加历史信号表需要的字段
            if 'signal_id' not in transformed.columns:
                transformed['signal_id'] = (
                    transformed['timestamp'].astype(str) + '_' + 
                    transformed['symbol'].astype(str) + '_' + 
                    transformed['source'].astype(str)
                )
            
            # 设置默认值
            transformed['execution_status'] = transformed.get('execution_status', 'pending')
            transformed['market_condition'] = transformed.get('market_condition', 'unknown')
            transformed['technical_context'] = transformed.get('technical_context', '{}')
            transformed['metadata'] = transformed.get('metadata', '{}')
            
            return transformed
            
        except Exception as e:
            self.log_error(f"信号数据转换失败: {e}")
            return pd.DataFrame()

    def _cleanup_migrated_data(self, conn, cutoff_time: datetime) -> None:
        """清理已迁移的数据"""
        try:
            tables_to_cleanup = ['market_ticks', 'trading_signals']
            
            for table in tables_to_cleanup:
                if self._table_exists(conn, table):
                    cleanup_sql = f"""
                        DELETE FROM {table}
                        WHERE timestamp < ?
                    """
                    conn.execute(cleanup_sql, [cutoff_time])
            
            self.log_info(f"清理了{cutoff_time}之前的实时数据")
            
        except Exception as e:
            self.log_error(f"清理迁移数据失败: {e}")

    # ==================
    # 数据归档功能
    # ==================
    
    async def archive_old_data(
        self,
        environment: TradingEnvironment,
        archive_before_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """归档旧数据"""
        def _archive_sync():
            return self.archive_old_data_sync(environment, archive_before_days)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _archive_sync
        )

    def archive_old_data_sync(
        self,
        environment: TradingEnvironment,
        archive_before_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """同步归档旧数据"""
        archive_days = archive_before_days or self.retention_policy.archive_after_days
        cutoff_date = datetime.utcnow() - timedelta(days=archive_days)
        
        archive_results = {
            'environment': environment.value,
            'cutoff_date': cutoff_date.isoformat(),
            'archived_tables': {},
            'total_records': 0,
            'archive_size_mb': 0
        }
        
        try:
            tables_to_archive = [
                'historical_market_data',
                'historical_trading_signals', 
                'historical_order_records',
                'historical_risk_metrics'
            ]
            
            for table_name in tables_to_archive:
                archive_info = self._archive_table_data(
                    table_name, environment, cutoff_date
                )
                archive_results['archived_tables'][table_name] = archive_info
                archive_results['total_records'] += archive_info.get('records_archived', 0)
            
            # 计算归档文件总大小
            archive_results['archive_size_mb'] = self._calculate_archive_size(environment)
            
            self.log_info(
                f"数据归档完成: {environment.value}, "
                f"归档{archive_results['total_records']}条记录"
            )
            
            return archive_results
            
        except Exception as e:
            self.log_error(f"数据归档失败: {e}")
            raise

    def _archive_table_data(
        self,
        table_name: str,
        environment: TradingEnvironment,
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """归档单个表的数据"""
        try:
            conn = self.duckdb_manager.get_connection(environment)
            
            # 查询需要归档的数据
            query = f"""
                SELECT * FROM {table_name}
                WHERE timestamp < ?
                ORDER BY timestamp
            """
            
            data_to_archive = conn.execute(query, [cutoff_date]).fetchdf()
            
            if data_to_archive.empty:
                return {
                    'records_archived': 0,
                    'archive_file': None,
                    'status': 'no_data'
                }
            
            # 生成归档文件名
            archive_filename = f"{table_name}_{environment.value}_{cutoff_date.strftime('%Y%m%d')}.parquet"
            archive_path = self.archive_dir / environment.value / archive_filename
            
            # 导出到Parquet文件
            if self.retention_policy.archive_to_parquet:
                data_to_archive.to_parquet(
                    archive_path,
                    compression=self.config.default_compression.value,
                    index=False
                )
            
            # 删除已归档的数据
            delete_sql = f"""
                DELETE FROM {table_name}
                WHERE timestamp < ?
            """
            conn.execute(delete_sql, [cutoff_date])
            
            return {
                'records_archived': len(data_to_archive),
                'archive_file': str(archive_path),
                'status': 'success',
                'date_range': {
                    'start': data_to_archive['timestamp'].min().isoformat(),
                    'end': data_to_archive['timestamp'].max().isoformat()
                }
            }
            
        except Exception as e:
            self.log_error(f"表{table_name}归档失败: {e}")
            return {
                'records_archived': 0,
                'archive_file': None,
                'status': 'error',
                'error': str(e)
            }

    def _calculate_archive_size(self, environment: TradingEnvironment) -> float:
        """计算归档文件总大小（MB）"""
        try:
            archive_env_dir = self.archive_dir / environment.value
            total_size = 0
            
            for file_path in archive_env_dir.rglob('*.parquet'):
                total_size += file_path.stat().st_size
            
            return round(total_size / (1024 * 1024), 2)
            
        except Exception as e:
            self.log_error(f"计算归档大小失败: {e}")
            return 0.0

    # ==================
    # 备份和恢复功能
    # ==================
    
    async def create_backup(
        self,
        environment: TradingEnvironment,
        backup_type: str = "full"
    ) -> Dict[str, Any]:
        """创建数据备份"""
        def _backup_sync():
            return self.create_backup_sync(environment, backup_type)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _backup_sync
        )

    def create_backup_sync(
        self,
        environment: TradingEnvironment,
        backup_type: str = "full"
    ) -> Dict[str, Any]:
        """同步创建数据备份"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{environment.value}_{backup_type}_{timestamp}"
        backup_path = self.backup_dir / environment.value / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backup_info = {
            'backup_name': backup_name,
            'backup_path': str(backup_path),
            'environment': environment.value,
            'backup_type': backup_type,
            'timestamp': timestamp,
            'tables': {},
            'total_size_mb': 0,
            'status': 'success'
        }
        
        try:
            # 获取数据库文件路径
            db_path = self.duckdb_manager._get_database_path(environment)
            
            if backup_type == "full":
                # 完整备份：复制整个数据库文件
                backup_db_path = backup_path / f"database_{environment.value}.duckdb"
                shutil.copy2(db_path, backup_db_path)
                
                backup_info['database_backup'] = str(backup_db_path)
                backup_info['total_size_mb'] = round(
                    backup_db_path.stat().st_size / (1024 * 1024), 2
                )
                
            elif backup_type == "incremental":
                # 增量备份：导出最近的数据
                conn = self.duckdb_manager.get_connection(environment)
                
                # 备份最近7天的数据
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                tables_to_backup = [
                    'historical_market_data',
                    'historical_trading_signals',
                    'historical_order_records', 
                    'historical_risk_metrics'
                ]
                
                for table_name in tables_to_backup:
                    table_backup_info = self._backup_table(
                        conn, table_name, backup_path, cutoff_date
                    )
                    backup_info['tables'][table_name] = table_backup_info
                    backup_info['total_size_mb'] += table_backup_info.get('size_mb', 0)
            
            # 创建备份元数据文件
            metadata_path = backup_path / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(backup_info, f, indent=2, default=str)
            
            self.log_info(f"备份创建成功: {backup_name}, 大小: {backup_info['total_size_mb']:.2f}MB")
            
            return backup_info
            
        except Exception as e:
            backup_info['status'] = 'error'
            backup_info['error'] = str(e)
            self.log_error(f"创建备份失败: {e}")
            return backup_info

    def _backup_table(
        self,
        conn,
        table_name: str,
        backup_path: Path,
        since_date: datetime
    ) -> Dict[str, Any]:
        """备份单个表"""
        try:
            # 查询需要备份的数据
            query = f"""
                SELECT * FROM {table_name}
                WHERE timestamp >= ?
                ORDER BY timestamp
            """
            
            df = conn.execute(query, [since_date]).fetchdf()
            
            if df.empty:
                return {
                    'records': 0,
                    'size_mb': 0,
                    'status': 'no_data'
                }
            
            # 导出到Parquet文件
            parquet_path = backup_path / f"{table_name}.parquet"
            df.to_parquet(
                parquet_path,
                compression=self.config.default_compression.value,
                index=False
            )
            
            size_mb = round(parquet_path.stat().st_size / (1024 * 1024), 2)
            
            return {
                'records': len(df),
                'size_mb': size_mb,
                'file': str(parquet_path),
                'status': 'success',
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                }
            }
            
        except Exception as e:
            self.log_error(f"备份表{table_name}失败: {e}")
            return {
                'records': 0,
                'size_mb': 0,
                'status': 'error',
                'error': str(e)
            }

    async def restore_from_backup(
        self,
        backup_path: Path,
        environment: TradingEnvironment,
        restore_type: str = "full"
    ) -> Dict[str, Any]:
        """从备份恢复数据"""
        def _restore_sync():
            return self.restore_from_backup_sync(backup_path, environment, restore_type)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _restore_sync
        )

    def restore_from_backup_sync(
        self,
        backup_path: Path,
        environment: TradingEnvironment,
        restore_type: str = "full"
    ) -> Dict[str, Any]:
        """同步从备份恢复数据"""
        restore_info = {
            'backup_path': str(backup_path),
            'environment': environment.value,
            'restore_type': restore_type,
            'restored_tables': {},
            'total_records': 0,
            'status': 'success'
        }
        
        try:
            # 读取备份元数据
            metadata_path = backup_path / "backup_metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"备份元数据文件不存在: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                backup_metadata = json.load(f)
            
            if restore_type == "full":
                # 完整恢复：替换整个数据库文件
                db_backup_path = backup_path / f"database_{environment.value}.duckdb"
                if db_backup_path.exists():
                    target_db_path = self.duckdb_manager._get_database_path(environment)
                    
                    # 关闭现有连接
                    self.duckdb_manager.close_connection(environment)
                    
                    # 替换数据库文件
                    shutil.copy2(db_backup_path, target_db_path)
                    
                    # 重新建立连接
                    self.duckdb_manager.get_connection(environment)
                    
                    restore_info['restored_database'] = True
                
            elif restore_type == "incremental":
                # 增量恢复：恢复表数据
                conn = self.duckdb_manager.get_connection(environment)
                
                for table_name in backup_metadata.get('tables', {}):
                    parquet_path = backup_path / f"{table_name}.parquet"
                    if parquet_path.exists():
                        table_restore_info = self._restore_table_data(
                            conn, table_name, parquet_path
                        )
                        restore_info['restored_tables'][table_name] = table_restore_info
                        restore_info['total_records'] += table_restore_info.get('records', 0)
            
            self.log_info(f"数据恢复完成: {environment.value}, 恢复{restore_info['total_records']}条记录")
            
            return restore_info
            
        except Exception as e:
            restore_info['status'] = 'error'
            restore_info['error'] = str(e)
            self.log_error(f"数据恢复失败: {e}")
            return restore_info

    def _restore_table_data(
        self,
        conn,
        table_name: str,
        parquet_path: Path
    ) -> Dict[str, Any]:
        """恢复表数据"""
        try:
            # 读取Parquet文件
            df = pd.read_parquet(parquet_path)
            
            if df.empty:
                return {'records': 0, 'status': 'no_data'}
            
            # 插入数据到表中
            conn.register('restore_data', df)
            
            insert_sql = f"""
                INSERT INTO {table_name}
                SELECT * FROM restore_data
                ON CONFLICT DO NOTHING
            """
            
            conn.execute(insert_sql)
            
            return {
                'records': len(df),
                'status': 'success',
                'file': str(parquet_path)
            }
            
        except Exception as e:
            self.log_error(f"恢复表{table_name}数据失败: {e}")
            return {
                'records': 0,
                'status': 'error',
                'error': str(e)
            }

    # ==================
    # 数据压缩和优化
    # ==================
    
    async def optimize_storage(
        self,
        environment: TradingEnvironment
    ) -> Dict[str, Any]:
        """优化存储"""
        def _optimize_sync():
            return self.optimize_storage_sync(environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _optimize_sync
        )

    def optimize_storage_sync(
        self,
        environment: TradingEnvironment
    ) -> Dict[str, Any]:
        """同步优化存储"""
        optimization_results = {
            'environment': environment.value,
            'optimized_tables': {},
            'total_space_saved_mb': 0,
            'status': 'success'
        }
        
        try:
            tables_to_optimize = [
                'historical_market_data',
                'historical_trading_signals',
                'historical_order_records', 
                'historical_risk_metrics'
            ]
            
            for table_name in tables_to_optimize:
                table_optimization = self.duckdb_manager.optimize_table_storage(
                    table_name, environment
                )
                
                if table_optimization:
                    optimization_results['optimized_tables'][table_name] = 'success'
                else:
                    optimization_results['optimized_tables'][table_name] = 'failed'
            
            self.log_info(f"存储优化完成: {environment.value}")
            
            return optimization_results
            
        except Exception as e:
            optimization_results['status'] = 'error'
            optimization_results['error'] = str(e)
            self.log_error(f"存储优化失败: {e}")
            return optimization_results

    # ==================
    # 清理和维护
    # ==================
    
    def list_backups(self, environment: Optional[TradingEnvironment] = None) -> List[Dict[str, Any]]:
        """列出所有备份"""
        backups = []
        
        try:
            if environment:
                environments = [environment]
            else:
                environments = list(TradingEnvironment)
            
            for env in environments:
                env_backup_dir = self.backup_dir / env.value
                if not env_backup_dir.exists():
                    continue
                
                for backup_dir in env_backup_dir.iterdir():
                    if not backup_dir.is_dir():
                        continue
                    
                    metadata_path = backup_dir / "backup_metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                backup_info = json.load(f)
                            backups.append(backup_info)
                        except Exception as e:
                            self.log_warning(f"读取备份元数据失败 {metadata_path}: {e}")
            
            # 按时间戳排序
            backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return backups
            
        except Exception as e:
            self.log_error(f"列出备份失败: {e}")
            return []

    def cleanup_old_backups(
        self, 
        environment: TradingEnvironment,
        keep_days: int = 30
    ) -> Dict[str, Any]:
        """清理旧备份"""
        cleanup_results = {
            'environment': environment.value,
            'deleted_backups': [],
            'kept_backups': [],
            'space_freed_mb': 0
        }
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
            env_backup_dir = self.backup_dir / environment.value
            
            if not env_backup_dir.exists():
                return cleanup_results
            
            for backup_dir in env_backup_dir.iterdir():
                if not backup_dir.is_dir():
                    continue
                
                # 从目录名解析时间戳
                try:
                    backup_timestamp_str = backup_dir.name.split('_')[-1]
                    backup_timestamp = datetime.strptime(backup_timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if backup_timestamp < cutoff_date:
                        # 计算目录大小
                        backup_size = sum(
                            f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()
                        ) / (1024 * 1024)
                        
                        # 删除备份
                        shutil.rmtree(backup_dir)
                        
                        cleanup_results['deleted_backups'].append({
                            'name': backup_dir.name,
                            'timestamp': backup_timestamp.isoformat(),
                            'size_mb': round(backup_size, 2)
                        })
                        cleanup_results['space_freed_mb'] += backup_size
                    else:
                        cleanup_results['kept_backups'].append(backup_dir.name)
                
                except Exception as e:
                    self.log_warning(f"处理备份目录失败 {backup_dir}: {e}")
                    continue
            
            cleanup_results['space_freed_mb'] = round(cleanup_results['space_freed_mb'], 2)
            
            self.log_info(
                f"备份清理完成: {environment.value}, "
                f"删除{len(cleanup_results['deleted_backups'])}个备份, "
                f"释放{cleanup_results['space_freed_mb']:.2f}MB空间"
            )
            
            return cleanup_results
            
        except Exception as e:
            self.log_error(f"清理备份失败: {e}")
            return cleanup_results

    def get_storage_health_report(
        self, 
        environment: TradingEnvironment
    ) -> Dict[str, Any]:
        """获取存储健康报告"""
        try:
            # 获取基本统计信息
            storage_stats = self.duckdb_manager.get_storage_statistics(environment)
            
            # 获取备份信息
            backups = self.list_backups(environment)
            
            # 获取归档信息
            archive_size = self._calculate_archive_size(environment)
            
            health_report = {
                'environment': environment.value,
                'timestamp': datetime.utcnow().isoformat(),
                'database_stats': storage_stats,
                'backup_info': {
                    'total_backups': len(backups),
                    'latest_backup': backups[0] if backups else None,
                    'oldest_backup': backups[-1] if backups else None
                },
                'archive_info': {
                    'total_size_mb': archive_size
                },
                'health_score': self._calculate_health_score(storage_stats, backups),
                'recommendations': self._get_health_recommendations(storage_stats, backups)
            }
            
            return health_report
            
        except Exception as e:
            self.log_error(f"生成存储健康报告失败: {e}")
            return {'error': str(e)}

    def _calculate_health_score(
        self, 
        storage_stats: Dict[str, Any], 
        backups: List[Dict[str, Any]]
    ) -> float:
        """计算存储健康评分（0-100）"""
        score = 100.0
        
        # 数据完整性检查
        if storage_stats.get('total_rows', 0) == 0:
            score -= 30
        
        # 备份检查
        if not backups:
            score -= 25
        else:
            # 检查最新备份时间
            latest_backup = backups[0]
            backup_timestamp = datetime.fromisoformat(latest_backup.get('timestamp', ''))
            days_since_backup = (datetime.utcnow() - backup_timestamp).days
            
            if days_since_backup > 7:
                score -= 15
            elif days_since_backup > 3:
                score -= 5
        
        # 存储大小检查
        file_size_mb = storage_stats.get('file_size_mb', 0)
        if file_size_mb > 10000:  # 超过10GB
            score -= 10
        
        return max(0.0, score)

    def _get_health_recommendations(
        self,
        storage_stats: Dict[str, Any],
        backups: List[Dict[str, Any]]
    ) -> List[str]:
        """获取健康建议"""
        recommendations = []
        
        if not backups:
            recommendations.append("建议创建数据备份")
        else:
            latest_backup = backups[0]
            backup_timestamp = datetime.fromisoformat(latest_backup.get('timestamp', ''))
            days_since_backup = (datetime.utcnow() - backup_timestamp).days
            
            if days_since_backup > 7:
                recommendations.append("建议创建新的数据备份")
        
        file_size_mb = storage_stats.get('file_size_mb', 0)
        if file_size_mb > 5000:
            recommendations.append("建议归档历史数据以减少存储大小")
        
        total_rows = storage_stats.get('total_rows', 0)
        if total_rows > 10000000:
            recommendations.append("建议优化存储以提高查询性能")
        
        if len(backups) > 10:
            recommendations.append("建议清理旧备份以释放存储空间")
        
        return recommendations

    def close(self) -> None:
        """关闭迁移管理器"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.log_info("数据迁移管理器已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()