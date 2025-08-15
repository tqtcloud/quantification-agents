"""
DuckDB时序数据存储管理器
高性能时序数据存储、查询和分析功能
"""

import asyncio
import json
import os
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Generator, AsyncGenerator
import pandas as pd
import numpy as np

import duckdb
from duckdb import DuckDBPyConnection

from src.config import settings
from src.exchanges.trading_interface import TradingEnvironment
from src.utils.logger import LoggerMixin


class DuckDBManager(LoggerMixin):
    """DuckDB时序数据管理器 - 支持多环境数据隔离和高性能时序数据操作"""
    
    def __init__(self):
        self.connections: Dict[TradingEnvironment, DuckDBPyConnection] = {}
        self._current_environment: Optional[TradingEnvironment] = None
        self._data_dir = Path(settings.data_directory)
        self._init_directories()
    
    def _init_directories(self):
        """初始化数据目录"""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        for env in TradingEnvironment:
            env_dir = self._data_dir / env.value
            env_dir.mkdir(exist_ok=True)
    
    def _get_database_path(self, environment: TradingEnvironment) -> Path:
        """获取环境特定的DuckDB数据库文件路径"""
        if environment == TradingEnvironment.TESTNET:
            db_path = self._data_dir / "testnet" / "market_data_testnet.duckdb"
        elif environment == TradingEnvironment.MAINNET:
            db_path = self._data_dir / "mainnet" / "market_data_mainnet.duckdb"
        else:  # PAPER
            db_path = self._data_dir / "paper" / "market_data_paper.duckdb"
        
        return db_path
    
    def get_connection(self, environment: TradingEnvironment) -> DuckDBPyConnection:
        """获取环境特定的DuckDB连接"""
        if environment not in self.connections:
            db_path = self._get_database_path(environment)
            
            # 创建DuckDB连接
            self.connections[environment] = duckdb.connect(str(db_path))
            
            # 配置DuckDB性能优化设置
            conn = self.connections[environment]
            
            # 内存和性能设置
            conn.execute("SET memory_limit='2GB'")
            conn.execute("SET threads=4")
            conn.execute("SET checkpoint_threshold='1GB'")
            
            # 启用时序数据优化扩展
            try:
                conn.install_extension("parquet")
                conn.load_extension("parquet")
                self.log_info(f"Loaded parquet extension for {environment.value}")
            except Exception as e:
                self.log_warning(f"Could not load parquet extension: {e}")
            
            # 初始化时序数据表结构
            self._init_timeseries_tables(conn, environment)
            
            self.log_info(f"DuckDB connection established for environment: {environment.value}")
        
        return self.connections[environment]
    
    def _init_timeseries_tables(self, conn: DuckDBPyConnection, environment: TradingEnvironment):
        """初始化时序数据表结构"""
        
        # 创建市场数据表（高频tick数据）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_ticks (
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
            )
        """)
        
        # 创建K线数据表（多时间框架）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_klines (
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
            )
        """)
        
        # 创建订单簿数据表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_book_snapshots (
                timestamp TIMESTAMP,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                bids VARCHAR NOT NULL,
                asks VARCHAR NOT NULL,
                last_update_id BIGINT,
                metadata VARCHAR,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, symbol)
            )
        """)
        
        # 创建技术指标数据表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                timestamp TIMESTAMP,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                indicator_name VARCHAR NOT NULL,
                indicator_value DOUBLE,
                indicator_data VARCHAR,
                metadata VARCHAR,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, symbol, interval, indicator_name)
            )
        """)
        
        # 创建交易信号数据表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                timestamp TIMESTAMP,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                signal_source VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                strength DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                price DOUBLE,
                quantity DOUBLE,
                reason VARCHAR,
                metadata VARCHAR,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (timestamp, symbol, signal_source, signal_type)
            )
        """)
        
        # 创建时间分区索引以提高查询性能
        self._create_time_partitioned_indexes(conn)
        
        self.log_info(f"Initialized timeseries tables for {environment.value}")
    
    def _create_time_partitioned_indexes(self, conn: DuckDBPyConnection):
        """创建时间分区索引"""
        
        # 为主要查询模式创建索引
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_market_ticks_symbol_time ON market_ticks(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_klines_symbol_interval_time ON market_klines(symbol, interval, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_order_book_symbol_time ON order_book_snapshots(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_indicator_time ON technical_indicators(symbol, indicator_name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_source_time ON trading_signals(symbol, signal_source, timestamp)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.log_warning(f"Could not create index: {e}")
    
    def set_current_environment(self, environment: TradingEnvironment):
        """设置当前活跃环境"""
        self._current_environment = environment
        self.log_debug(f"Current DuckDB environment set to: {environment.value}")
    
    def get_current_environment(self) -> Optional[TradingEnvironment]:
        """获取当前活跃环境"""
        return self._current_environment
    
    @contextmanager
    def use_environment(self, environment: TradingEnvironment):
        """临时切换环境的上下文管理器"""
        previous_env = self._current_environment
        self.set_current_environment(environment)
        try:
            yield
        finally:
            self._current_environment = previous_env
    
    def _get_environment(self) -> TradingEnvironment:
        """获取当前环境，如果未设置则使用默认值"""
        env = self.get_current_environment()
        if env is None:
            env = TradingEnvironment.TESTNET
            self.log_warning(f"No current environment set, using default: {env.value}")
        return env
    
    def close_all(self):
        """关闭所有DuckDB连接"""
        for environment, conn in self.connections.items():
            try:
                conn.close()
                self.log_info(f"Closed DuckDB connection for {environment.value}")
            except Exception as e:
                self.log_error(f"Error closing DuckDB connection for {environment.value}: {e}")
        
        self.connections.clear()
    
    def close_connection(self, environment: TradingEnvironment):
        """关闭指定环境的连接"""
        if environment in self.connections:
            try:
                self.connections[environment].close()
                del self.connections[environment]
                self.log_info(f"Closed DuckDB connection for {environment.value}")
            except Exception as e:
                self.log_error(f"Error closing DuckDB connection for {environment.value}: {e}")
    
    def get_database_info(self, environment: Optional[TradingEnvironment] = None) -> Dict[str, Any]:
        """获取数据库信息和统计"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        info = {
            "environment": env.value,
            "database_path": str(self._get_database_path(env)),
            "tables": {},
            "total_tables": 0,
            "total_size_mb": 0
        }
        
        # 获取所有表的信息
        tables_result = conn.execute("SHOW TABLES").fetchall()
        info["total_tables"] = len(tables_result)
        
        for table_row in tables_result:
            table_name = table_row[0]
            
            # 获取表行数
            try:
                count_result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = count_result[0] if count_result else 0
            except Exception:
                row_count = 0
            
            # 获取表结构信息
            try:
                schema_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
                columns = [{"name": row[0], "type": row[1]} for row in schema_result]
            except Exception:
                columns = []
            
            info["tables"][table_name] = {
                "row_count": row_count,
                "columns": columns
            }
        
        # 获取数据库文件大小
        db_path = self._get_database_path(env)
        if db_path.exists():
            info["total_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)
        
        return info
    
    # ===================
    # 高性能数据插入方法
    # ===================
    
    def insert_market_ticks_batch(self, tick_data: List[Dict[str, Any]], 
                                 environment: Optional[TradingEnvironment] = None) -> int:
        """批量插入市场tick数据 - 高性能插入"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        if not tick_data:
            return 0
        
        try:
            # 准备批量数据
            df = pd.DataFrame(tick_data)
            
            # 确保必要的列存在
            required_columns = ['timestamp', 'symbol', 'price', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # 添加环境标识
            df['environment'] = env.value
            
            # 转换时间戳
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 设置默认值
            df['bid_price'] = df.get('bid_price', None)
            df['ask_price'] = df.get('ask_price', None)
            df['bid_volume'] = df.get('bid_volume', None)
            df['ask_volume'] = df.get('ask_volume', None)
            df['trade_count'] = df.get('trade_count', 0)
            df['metadata'] = df.get('metadata', '{}')
            df['ingestion_time'] = datetime.utcnow()
            
            # 使用DuckDB的高性能插入
            conn.register('tick_batch', df)
            
            insert_sql = """
                INSERT INTO market_ticks 
                (timestamp, symbol, environment, price, volume, bid_price, ask_price, 
                 bid_volume, ask_volume, trade_count, metadata, ingestion_time)
                SELECT timestamp, symbol, environment, price, volume, bid_price, ask_price, 
                       bid_volume, ask_volume, trade_count, metadata, ingestion_time
                FROM tick_batch
                ON CONFLICT (timestamp) DO UPDATE SET
                    price = EXCLUDED.price,
                    volume = EXCLUDED.volume,
                    bid_price = EXCLUDED.bid_price,
                    ask_price = EXCLUDED.ask_price,
                    bid_volume = EXCLUDED.bid_volume,
                    ask_volume = EXCLUDED.ask_volume,
                    trade_count = EXCLUDED.trade_count,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"Inserted {len(tick_data)} tick records for {env.value}")
            return len(tick_data)
            
        except Exception as e:
            self.log_error(f"Error inserting tick data: {e}")
            raise
    
    def insert_klines_batch(self, kline_data: List[Dict[str, Any]], 
                           environment: Optional[TradingEnvironment] = None) -> int:
        """批量插入K线数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        if not kline_data:
            return 0
        
        try:
            df = pd.DataFrame(kline_data)
            
            # 必要的列检查
            required_columns = ['timestamp', 'symbol', 'interval', 'open_price', 
                              'high_price', 'low_price', 'close_price', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # 添加环境标识
            df['environment'] = env.value
            
            # 时间戳转换
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 设置默认值
            df['quote_volume'] = df.get('quote_volume', 0)
            df['trade_count'] = df.get('trade_count', 0)
            df['taker_buy_volume'] = df.get('taker_buy_volume', 0)
            df['taker_buy_quote_volume'] = df.get('taker_buy_quote_volume', 0)
            df['metadata'] = df.get('metadata', '{}')
            df['ingestion_time'] = datetime.utcnow()
            
            conn.register('kline_batch', df)
            
            insert_sql = """
                INSERT INTO market_klines 
                (timestamp, symbol, environment, interval, open_price, high_price, low_price, 
                 close_price, volume, quote_volume, trade_count, taker_buy_volume, 
                 taker_buy_quote_volume, metadata, ingestion_time)
                SELECT timestamp, symbol, environment, interval, open_price, high_price, low_price, 
                       close_price, volume, quote_volume, trade_count, taker_buy_volume, 
                       taker_buy_quote_volume, metadata, ingestion_time
                FROM kline_batch
                ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trade_count = EXCLUDED.trade_count,
                    taker_buy_volume = EXCLUDED.taker_buy_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"Inserted {len(kline_data)} kline records for {env.value}")
            return len(kline_data)
            
        except Exception as e:
            self.log_error(f"Error inserting kline data: {e}")
            raise
    
    def insert_order_book_snapshot(self, symbol: str, bids: List[List[float]], 
                                  asks: List[List[float]], last_update_id: int = None,
                                  timestamp: datetime = None, metadata: Dict = None,
                                  environment: Optional[TradingEnvironment] = None) -> bool:
        """插入订单簿快照"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # 准备数据
            data = {
                'timestamp': timestamp,
                'symbol': symbol,
                'environment': env.value,
                'bids': json.dumps(bids),
                'asks': json.dumps(asks),
                'last_update_id': last_update_id,
                'metadata': json.dumps(metadata or {}),
                'ingestion_time': datetime.utcnow()
            }
            
            # 插入数据
            conn.execute("""
                INSERT INTO order_book_snapshots 
                (timestamp, symbol, environment, bids, asks, last_update_id, metadata, ingestion_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (timestamp, symbol) DO UPDATE SET
                    bids = EXCLUDED.bids,
                    asks = EXCLUDED.asks,
                    last_update_id = EXCLUDED.last_update_id,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """, [
                data['timestamp'], data['symbol'], data['environment'],
                data['bids'], data['asks'], data['last_update_id'],
                data['metadata'], data['ingestion_time']
            ])
            
            self.log_debug(f"Inserted order book snapshot for {symbol} at {timestamp}")
            return True
            
        except Exception as e:
            self.log_error(f"Error inserting order book snapshot: {e}")
            return False
    
    def insert_technical_indicators_batch(self, indicator_data: List[Dict[str, Any]], 
                                        environment: Optional[TradingEnvironment] = None) -> int:
        """批量插入技术指标数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        if not indicator_data:
            return 0
        
        try:
            df = pd.DataFrame(indicator_data)
            
            # 必要的列检查
            required_columns = ['timestamp', 'symbol', 'interval', 'indicator_name', 'indicator_value']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # 添加环境标识
            df['environment'] = env.value
            
            # 时间戳转换
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 设置默认值
            df['indicator_data'] = df.get('indicator_data', '{}')
            df['metadata'] = df.get('metadata', '{}')
            df['ingestion_time'] = datetime.utcnow()
            
            conn.register('indicator_batch', df)
            
            insert_sql = """
                INSERT INTO technical_indicators 
                (timestamp, symbol, environment, interval, indicator_name, indicator_value, indicator_data, metadata, ingestion_time)
                SELECT timestamp, symbol, environment, interval, indicator_name, indicator_value, indicator_data, metadata, ingestion_time
                FROM indicator_batch
                ON CONFLICT (timestamp, symbol, interval, indicator_name) DO UPDATE SET
                    indicator_value = EXCLUDED.indicator_value,
                    indicator_data = EXCLUDED.indicator_data,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"Inserted {len(indicator_data)} indicator records for {env.value}")
            return len(indicator_data)
            
        except Exception as e:
            self.log_error(f"Error inserting technical indicator data: {e}")
            raise
    
    def insert_trading_signals_batch(self, signal_data: List[Dict[str, Any]], 
                                   environment: Optional[TradingEnvironment] = None) -> int:
        """批量插入交易信号数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        if not signal_data:
            return 0
        
        try:
            df = pd.DataFrame(signal_data)
            
            # 必要的列检查
            required_columns = ['timestamp', 'symbol', 'signal_source', 'signal_type', 
                              'action', 'strength', 'confidence']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # 添加环境标识
            df['environment'] = env.value
            
            # 时间戳转换
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 设置默认值
            df['price'] = df.get('price', None)
            df['quantity'] = df.get('quantity', None)
            df['reason'] = df.get('reason', '')
            df['metadata'] = df.get('metadata', '{}')
            df['ingestion_time'] = datetime.utcnow()
            
            conn.register('signal_batch', df)
            
            insert_sql = """
                INSERT INTO trading_signals 
                (timestamp, symbol, environment, signal_source, signal_type, action, strength, confidence, price, quantity, reason, metadata, ingestion_time)
                SELECT timestamp, symbol, environment, signal_source, signal_type, action, strength, confidence, price, quantity, reason, metadata, ingestion_time
                FROM signal_batch
                ON CONFLICT (timestamp, symbol, signal_source, signal_type) DO UPDATE SET
                    action = EXCLUDED.action,
                    strength = EXCLUDED.strength,
                    confidence = EXCLUDED.confidence,
                    price = EXCLUDED.price,
                    quantity = EXCLUDED.quantity,
                    reason = EXCLUDED.reason,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"Inserted {len(signal_data)} signal records for {env.value}")
            return len(signal_data)
            
        except Exception as e:
            self.log_error(f"Error inserting trading signal data: {e}")
            raise
    
    # ===================
    # 时序数据查询和分析方法
    # ===================
    
    def query_market_ticks(self, symbol: str, start_time: datetime, end_time: datetime,
                          environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """查询市场tick数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT * FROM market_ticks 
                WHERE symbol = ? 
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [symbol, env.value, start_time, end_time]).fetchdf()
            self.log_debug(f"Queried {len(result)} tick records for {symbol}")
            return result
            
        except Exception as e:
            self.log_error(f"Error querying market ticks: {e}")
            return pd.DataFrame()
    
    def query_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime,
                    environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """查询K线数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT * FROM market_klines 
                WHERE symbol = ? 
                AND interval = ?
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [symbol, interval, env.value, start_time, end_time]).fetchdf()
            self.log_debug(f"Queried {len(result)} kline records for {symbol} {interval}")
            return result
            
        except Exception as e:
            self.log_error(f"Error querying klines: {e}")
            return pd.DataFrame()
    
    def query_order_book_snapshots(self, symbol: str, start_time: datetime, end_time: datetime,
                                  environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """查询订单簿快照数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT * FROM order_book_snapshots 
                WHERE symbol = ? 
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [symbol, env.value, start_time, end_time]).fetchdf()
            self.log_debug(f"Queried {len(result)} order book snapshots for {symbol}")
            return result
            
        except Exception as e:
            self.log_error(f"Error querying order book snapshots: {e}")
            return pd.DataFrame()
    
    def query_technical_indicators(self, symbol: str, indicator_name: str, interval: str,
                                 start_time: datetime, end_time: datetime,
                                 environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """查询技术指标数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT * FROM technical_indicators 
                WHERE symbol = ? 
                AND indicator_name = ?
                AND interval = ?
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [symbol, indicator_name, interval, env.value, start_time, end_time]).fetchdf()
            self.log_debug(f"Queried {len(result)} indicator records for {symbol} {indicator_name}")
            return result
            
        except Exception as e:
            self.log_error(f"Error querying technical indicators: {e}")
            return pd.DataFrame()
    
    def query_trading_signals(self, symbol: str, signal_source: str = None, signal_type: str = None,
                            start_time: datetime = None, end_time: datetime = None,
                            environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """查询交易信号数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            conditions = ["symbol = ?", "environment = ?"]
            params = [symbol, env.value]
            
            if signal_source:
                conditions.append("signal_source = ?")
                params.append(signal_source)
            
            if signal_type:
                conditions.append("signal_type = ?")
                params.append(signal_type)
            
            if start_time and end_time:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_time, end_time])
            
            query = f"""
                SELECT * FROM trading_signals 
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            result = conn.execute(query, params).fetchdf()
            self.log_debug(f"Queried {len(result)} signal records for {symbol}")
            return result
            
        except Exception as e:
            self.log_error(f"Error querying trading signals: {e}")
            return pd.DataFrame()
    
    def get_latest_prices(self, symbols: List[str], 
                         environment: Optional[TradingEnvironment] = None) -> Dict[str, float]:
        """获取最新价格数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            if not symbols:
                return {}
            
            placeholders = ','.join(['?' for _ in symbols])
            query = f"""
                SELECT symbol, price, timestamp
                FROM (
                    SELECT symbol, price, timestamp,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM market_ticks 
                    WHERE symbol IN ({placeholders})
                    AND environment = ?
                ) ranked
                WHERE rn = 1
            """
            
            params = symbols + [env.value]
            result = conn.execute(query, params).fetchall()
            
            prices = {row[0]: float(row[1]) for row in result}
            self.log_debug(f"Retrieved latest prices for {len(prices)} symbols")
            return prices
            
        except Exception as e:
            self.log_error(f"Error getting latest prices: {e}")
            return {}
    
    def get_ohlcv_aggregation(self, symbol: str, interval: str, periods: int = 100,
                             environment: Optional[TradingEnvironment] = None) -> pd.DataFrame:
        """获取OHLCV聚合数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT * FROM market_klines 
                WHERE symbol = ? 
                AND interval = ?
                AND environment = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            result = conn.execute(query, [symbol, interval, env.value, periods]).fetchdf()
            
            if not result.empty:
                # 反转数据使其按时间升序排列
                result = result.iloc[::-1].reset_index(drop=True)
            
            self.log_debug(f"Retrieved {len(result)} OHLCV records for {symbol} {interval}")
            return result
            
        except Exception as e:
            self.log_error(f"Error getting OHLCV aggregation: {e}")
            return pd.DataFrame()
    
    def calculate_price_statistics(self, symbol: str, start_time: datetime, end_time: datetime,
                                 environment: Optional[TradingEnvironment] = None) -> Dict[str, float]:
        """计算价格统计信息"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT 
                    COUNT(*) as tick_count,
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(price) as avg_price,
                    STDDEV(price) as price_stddev,
                    SUM(volume) as total_volume,
                    AVG(volume) as avg_volume
                FROM market_ticks 
                WHERE symbol = ? 
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
            """
            
            result = conn.execute(query, [symbol, env.value, start_time, end_time]).fetchone()
            
            if result and result[0] > 0:
                stats = {
                    'tick_count': int(result[0]),
                    'min_price': float(result[1]) if result[1] else 0.0,
                    'max_price': float(result[2]) if result[2] else 0.0,
                    'avg_price': float(result[3]) if result[3] else 0.0,
                    'price_stddev': float(result[4]) if result[4] else 0.0,
                    'total_volume': float(result[5]) if result[5] else 0.0,
                    'avg_volume': float(result[6]) if result[6] else 0.0
                }
                
                # 计算价格变化幅度
                if stats['min_price'] > 0:
                    stats['price_range_percent'] = ((stats['max_price'] - stats['min_price']) / stats['min_price']) * 100
                else:
                    stats['price_range_percent'] = 0.0
                
                self.log_debug(f"Calculated price statistics for {symbol}")
                return stats
            else:
                return {}
            
        except Exception as e:
            self.log_error(f"Error calculating price statistics: {e}")
            return {}
    
    def get_volume_weighted_price(self, symbol: str, start_time: datetime, end_time: datetime,
                                environment: Optional[TradingEnvironment] = None) -> float:
        """计算成交量加权平均价格(VWAP)"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            query = """
                SELECT 
                    SUM(price * volume) / SUM(volume) as vwap
                FROM market_ticks 
                WHERE symbol = ? 
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                AND volume > 0
            """
            
            result = conn.execute(query, [symbol, env.value, start_time, end_time]).fetchone()
            
            if result and result[0]:
                vwap = float(result[0])
                self.log_debug(f"Calculated VWAP for {symbol}: {vwap}")
                return vwap
            else:
                return 0.0
            
        except Exception as e:
            self.log_error(f"Error calculating VWAP: {e}")
            return 0.0
    
    def get_market_depth_analysis(self, symbol: str, timestamp: datetime = None,
                                environment: Optional[TradingEnvironment] = None) -> Dict[str, Any]:
        """获取市场深度分析"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            if timestamp is None:
                # 获取最新的订单簿快照
                query = """
                    SELECT bids, asks FROM order_book_snapshots 
                    WHERE symbol = ? 
                    AND environment = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                params = [symbol, env.value]
            else:
                # 获取指定时间的订单簿快照
                query = """
                    SELECT bids, asks FROM order_book_snapshots 
                    WHERE symbol = ? 
                    AND environment = ?
                    AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                params = [symbol, env.value, timestamp]
            
            result = conn.execute(query, params).fetchone()
            
            if not result:
                return {}
            
            bids = json.loads(result[0])
            asks = json.loads(result[1])
            
            # 计算市场深度指标
            analysis = {
                'bid_levels': len(bids),
                'ask_levels': len(asks),
                'best_bid': float(bids[0][0]) if bids else 0.0,
                'best_ask': float(asks[0][0]) if asks else 0.0,
                'bid_volume': sum(float(level[1]) for level in bids),
                'ask_volume': sum(float(level[1]) for level in asks),
                'spread': 0.0,
                'spread_percent': 0.0,
                'imbalance': 0.0
            }
            
            if analysis['best_bid'] > 0 and analysis['best_ask'] > 0:
                analysis['spread'] = analysis['best_ask'] - analysis['best_bid']
                analysis['spread_percent'] = (analysis['spread'] / analysis['best_bid']) * 100
            
            # 计算订单簿不平衡度
            total_volume = analysis['bid_volume'] + analysis['ask_volume']
            if total_volume > 0:
                analysis['imbalance'] = (analysis['bid_volume'] - analysis['ask_volume']) / total_volume
            
            self.log_debug(f"Analyzed market depth for {symbol}")
            return analysis
            
        except Exception as e:
            self.log_error(f"Error analyzing market depth: {e}")
            return {}
    
    # ===================
    # 数据压缩和分区策略
    # ===================
    
    def export_to_parquet(self, table_name: str, output_path: Path, 
                         start_time: datetime = None, end_time: datetime = None,
                         partition_by: str = None, environment: Optional[TradingEnvironment] = None) -> bool:
        """导出数据到Parquet格式以实现压缩存储"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            # 构建查询条件
            conditions = ["environment = ?"]
            params = [env.value]
            
            if start_time and end_time:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_time, end_time])
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # 构建导出语句
            if partition_by:
                # 分区导出
                partition_path = output_path / f"{table_name}_partitioned"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                export_sql = f"""
                    COPY (
                        SELECT * FROM {table_name} 
                        {where_clause}
                        ORDER BY timestamp
                    ) TO '{partition_path}' 
                    (FORMAT PARQUET, PARTITION_BY ({partition_by}), COMPRESSION 'SNAPPY')
                """
            else:
                # 单文件导出
                output_path.parent.mkdir(parents=True, exist_ok=True)
                export_sql = f"""
                    COPY (
                        SELECT * FROM {table_name} 
                        {where_clause}
                        ORDER BY timestamp
                    ) TO '{output_path}' 
                    (FORMAT PARQUET, COMPRESSION 'SNAPPY')
                """
            
            conn.execute(export_sql, params)
            
            self.log_info(f"Exported {table_name} to Parquet: {output_path}")
            return True
            
        except Exception as e:
            self.log_error(f"Error exporting {table_name} to Parquet: {e}")
            return False
    
    def import_from_parquet(self, table_name: str, parquet_path: Path,
                          environment: Optional[TradingEnvironment] = None) -> bool:
        """从Parquet文件导入数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            if not parquet_path.exists():
                self.log_error(f"Parquet file not found: {parquet_path}")
                return False
            
            # 读取Parquet文件并插入到表中
            import_sql = f"""
                INSERT INTO {table_name} 
                SELECT * FROM read_parquet('{parquet_path}')
                ON CONFLICT DO NOTHING
            """
            
            conn.execute(import_sql)
            
            self.log_info(f"Imported data from Parquet to {table_name}: {parquet_path}")
            return True
            
        except Exception as e:
            self.log_error(f"Error importing from Parquet to {table_name}: {e}")
            return False
    
    def create_time_partitioned_table(self, base_table: str, partition_interval: str = "monthly",
                                    environment: Optional[TradingEnvironment] = None) -> bool:
        """创建时间分区表"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            # 获取表结构
            schema_result = conn.execute(f"DESCRIBE {base_table}").fetchall()
            columns = []
            
            for row in schema_result:
                col_name, col_type = row[0], row[1]
                columns.append(f"{col_name} {col_type}")
            
            columns_def = ", ".join(columns)
            
            # 创建分区表名
            partition_table = f"{base_table}_partitioned"
            
            # 创建分区表
            if partition_interval == "monthly":
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {partition_table} (
                        {columns_def}
                    ) PARTITION BY (date_trunc('month', timestamp))
                """
            elif partition_interval == "daily":
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {partition_table} (
                        {columns_def}
                    ) PARTITION BY (date_trunc('day', timestamp))
                """
            elif partition_interval == "weekly":
                create_sql = f"""
                    CREATE TABLE IF NOT EXISTS {partition_table} (
                        {columns_def}
                    ) PARTITION BY (date_trunc('week', timestamp))
                """
            else:
                self.log_error(f"Unsupported partition interval: {partition_interval}")
                return False
            
            conn.execute(create_sql)
            
            self.log_info(f"Created partitioned table {partition_table} with {partition_interval} partitions")
            return True
            
        except Exception as e:
            self.log_error(f"Error creating partitioned table: {e}")
            return False
    
    def migrate_to_partitioned_table(self, source_table: str, target_table: str,
                                   environment: Optional[TradingEnvironment] = None) -> bool:
        """将数据从普通表迁移到分区表"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            # 检查源表是否存在
            source_exists = conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{source_table}'
            """).fetchone()[0] > 0
            
            if not source_exists:
                self.log_error(f"Source table {source_table} does not exist")
                return False
            
            # 迁移数据
            migrate_sql = f"""
                INSERT INTO {target_table} 
                SELECT * FROM {source_table}
                ORDER BY timestamp
            """
            
            conn.execute(migrate_sql)
            
            # 获取迁移的行数
            count_result = conn.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()
            migrated_rows = count_result[0] if count_result else 0
            
            self.log_info(f"Migrated {migrated_rows} rows from {source_table} to {target_table}")
            return True
            
        except Exception as e:
            self.log_error(f"Error migrating data to partitioned table: {e}")
            return False
    
    def optimize_table_storage(self, table_name: str, 
                             environment: Optional[TradingEnvironment] = None) -> bool:
        """优化表存储 - 重新组织数据以提高查询性能"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            # 创建临时优化表
            temp_table = f"{table_name}_optimized_temp"
            
            # 按时间戳重新排序并重建表
            optimize_sql = f"""
                CREATE TABLE {temp_table} AS 
                SELECT * FROM {table_name} 
                ORDER BY timestamp, symbol
            """
            
            conn.execute(optimize_sql)
            
            # 删除原表
            conn.execute(f"DROP TABLE {table_name}")
            
            # 重命名优化表
            conn.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
            
            # 重新创建索引
            self._create_time_partitioned_indexes(conn)
            
            self.log_info(f"Optimized storage for table {table_name}")
            return True
            
        except Exception as e:
            self.log_error(f"Error optimizing table storage: {e}")
            return False
    
    def cleanup_old_data(self, table_name: str, retention_days: int,
                        environment: Optional[TradingEnvironment] = None) -> int:
        """清理过期数据"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # 删除过期数据
            delete_sql = f"""
                DELETE FROM {table_name} 
                WHERE timestamp < ? 
                AND environment = ?
            """
            
            # 先计算要删除的行数
            count_sql = f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE timestamp < ? 
                AND environment = ?
            """
            count_result = conn.execute(count_sql, [cutoff_date, env.value]).fetchone()
            deleted_rows = count_result[0] if count_result else 0
            
            # 执行删除
            if deleted_rows > 0:
                conn.execute(delete_sql, [cutoff_date, env.value])
            
            self.log_info(f"Cleaned up {deleted_rows} old records from {table_name} (retention: {retention_days} days)")
            return deleted_rows
            
        except Exception as e:
            self.log_error(f"Error cleaning up old data from {table_name}: {e}")
            return -1
    
    def get_storage_statistics(self, environment: Optional[TradingEnvironment] = None) -> Dict[str, Any]:
        """获取存储统计信息"""
        env = environment or self._get_environment()
        conn = self.get_connection(env)
        
        try:
            # 获取所有表的统计信息
            tables_result = conn.execute("SHOW TABLES").fetchall()
            
            stats = {
                'environment': env.value,
                'total_tables': len(tables_result),
                'tables': {},
                'total_rows': 0,
                'storage_efficiency': {}
            }
            
            for table_row in tables_result:
                table_name = table_row[0]
                
                # 获取表行数和大小估算
                try:
                    count_query = f"SELECT COUNT(*) FROM {table_name}"
                    count_result = conn.execute(count_query).fetchone()
                    row_count = count_result[0] if count_result else 0
                    
                    # 获取表的最早和最晚时间戳（如果有timestamp列）
                    try:
                        time_range_query = f"""
                            SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest 
                            FROM {table_name}
                        """
                        time_result = conn.execute(time_range_query).fetchone()
                        
                        if time_result and time_result[0]:
                            earliest = time_result[0]
                            latest = time_result[1]
                            time_span_days = (latest - earliest).days if latest and earliest else 0
                        else:
                            earliest = latest = None
                            time_span_days = 0
                    except:
                        earliest = latest = None
                        time_span_days = 0
                    
                    stats['tables'][table_name] = {
                        'row_count': row_count,
                        'earliest_record': earliest,
                        'latest_record': latest,
                        'time_span_days': time_span_days
                    }
                    
                    stats['total_rows'] += row_count
                    
                except Exception as table_error:
                    self.log_warning(f"Could not get stats for table {table_name}: {table_error}")
                    stats['tables'][table_name] = {
                        'row_count': 0,
                        'error': str(table_error)
                    }
            
            # 计算存储效率指标
            db_path = self._get_database_path(env)
            if db_path.exists():
                file_size_mb = db_path.stat().st_size / (1024 * 1024)
                stats['file_size_mb'] = round(file_size_mb, 2)
                
                if stats['total_rows'] > 0:
                    stats['storage_efficiency']['bytes_per_row'] = round(
                        (db_path.stat().st_size / stats['total_rows']), 2
                    )
                else:
                    stats['storage_efficiency']['bytes_per_row'] = 0
            
            return stats
            
        except Exception as e:
            self.log_error(f"Error getting storage statistics: {e}")
            return {'error': str(e)}


# 全局DuckDB管理器实例
duckdb_manager = DuckDBManager()