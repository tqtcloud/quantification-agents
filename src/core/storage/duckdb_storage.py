"""
DuckDB历史数据存储层
提供高性能的OLAP查询和分析型数据存储功能
"""

import asyncio
import json
import os
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Generator, AsyncGenerator, TypeVar
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import duckdb
from duckdb import DuckDBPyConnection

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
    duckdb_manager = MockDuckDBManager()

try:
    from src.exchanges.trading_interface import TradingEnvironment
except ImportError:
    from enum import Enum
    class TradingEnvironment(Enum):
        TESTNET = "testnet"
        MAINNET = "mainnet"
        PAPER = "paper"
from .models import (
    HistoricalMarketData,
    HistoricalTradingSignal, 
    HistoricalOrderRecord,
    HistoricalRiskMetrics,
    StorageConfig,
    DataRetentionPolicy,
    CompressionType,
    PartitionStrategy,
    DataCategory
)

T = TypeVar('T')


class DuckDBStorage(LoggerMixin):
    """
    DuckDB历史数据存储系统
    
    提供以下功能：
    1. 高性能历史数据存储和查询
    2. 数据生命周期管理
    3. 分区和压缩优化
    4. 数据导入导出
    5. 分析型查询支持
    6. 与内存缓存系统协作
    """
    
    def __init__(
        self, 
        config: Optional[StorageConfig] = None,
        retention_policy: Optional[DataRetentionPolicy] = None
    ):
        super().__init__()
        
        self.config = config or StorageConfig()
        self.retention_policy = retention_policy or DataRetentionPolicy()
        self.duckdb_manager = duckdb_manager
        
        # 线程池用于异步操作
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.thread_count,
            thread_name_prefix="DuckDBStorage"
        )
        
        # 初始化存储表结构
        self._init_historical_tables()
        
        # 启动后台任务
        self._start_background_tasks()
        
        self.log_info("DuckDB存储系统初始化完成")

    def _init_historical_tables(self) -> None:
        """初始化历史数据表结构"""
        for env in TradingEnvironment:
            conn = self.duckdb_manager.get_connection(env)
            
            # 历史市场数据表
            self._create_historical_market_data_table(conn, env)
            
            # 历史交易信号表
            self._create_historical_signals_table(conn, env)
            
            # 历史订单记录表
            self._create_historical_orders_table(conn, env)
            
            # 历史风险指标表
            self._create_historical_risk_metrics_table(conn, env)
            
            # 性能统计表
            self._create_performance_statistics_table(conn, env)
            
            # 创建分析视图
            self._create_analytical_views(conn, env)
            
            self.log_info(f"历史数据表结构初始化完成: {env.value}")

    def _create_historical_market_data_table(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建历史市场数据表"""
        table_name = "historical_market_data"
        
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                
                -- 价格数据
                open_price DECIMAL(18,8) NOT NULL,
                high_price DECIMAL(18,8) NOT NULL,
                low_price DECIMAL(18,8) NOT NULL,
                close_price DECIMAL(18,8) NOT NULL,
                
                -- 成交量数据
                volume DECIMAL(18,8) NOT NULL,
                quote_volume DECIMAL(18,8),
                trade_count INTEGER DEFAULT 0,
                taker_buy_volume DECIMAL(18,8),
                taker_buy_quote_volume DECIMAL(18,8),
                
                -- 订单簿数据
                bid_price DECIMAL(18,8),
                ask_price DECIMAL(18,8),
                bid_volume DECIMAL(18,8),
                ask_volume DECIMAL(18,8),
                
                -- 衍生品数据
                open_interest DECIMAL(18,8),
                funding_rate DECIMAL(12,8),
                mark_price DECIMAL(18,8),
                index_price DECIMAL(18,8),
                
                -- 数据质量
                data_source VARCHAR DEFAULT 'binance',
                data_quality DOUBLE DEFAULT 1.0,
                
                -- 元数据
                metadata JSON,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (timestamp, symbol, interval, environment)
            )
        """)
        
        # 创建索引优化查询性能
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name}(symbol, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_interval_time ON {table_name}(interval, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_env_time ON {table_name}(environment, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_composite ON {table_name}(symbol, interval, environment, timestamp DESC)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.log_warning(f"创建索引失败: {e}")

    def _create_historical_signals_table(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建历史交易信号表"""
        table_name = "historical_trading_signals"
        
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                signal_id VARCHAR NOT NULL,
                
                -- 信号基本信息
                source VARCHAR NOT NULL,
                signal_type VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                strength DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                
                -- 价格和数量
                target_price DECIMAL(18,8),
                suggested_quantity DECIMAL(18,8),
                max_position_size DECIMAL(18,8),
                
                -- 风险控制
                stop_loss DECIMAL(18,8),
                take_profit DECIMAL(18,8),
                max_drawdown_pct DOUBLE,
                
                -- 时间相关
                validity_period_minutes INTEGER,
                expiry_time TIMESTAMP,
                
                -- 技术背景
                technical_context JSON,
                market_condition VARCHAR DEFAULT 'unknown',
                
                -- 执行状态
                execution_status VARCHAR DEFAULT 'pending',
                execution_price DECIMAL(18,8),
                execution_time TIMESTAMP,
                execution_quantity DECIMAL(18,8),
                
                -- 绩效
                realized_pnl DECIMAL(18,8),
                unrealized_pnl DECIMAL(18,8),
                win_rate DOUBLE,
                
                -- 原因和元数据
                reason TEXT,
                metadata JSON,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (timestamp, symbol, signal_id, environment)
            )
        """)
        
        # 创建查询优化索引
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_time ON {table_name}(source, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_action ON {table_name}(symbol, action, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status ON {table_name}(execution_status, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signal_type ON {table_name}(signal_type, timestamp DESC)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.log_warning(f"创建信号表索引失败: {e}")

    def _create_historical_orders_table(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建历史订单记录表"""
        table_name = "historical_order_records"
        
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                environment VARCHAR NOT NULL,
                
                -- 订单标识
                order_id VARCHAR NOT NULL,
                client_order_id VARCHAR NOT NULL,
                strategy_id VARCHAR,
                signal_id VARCHAR,
                
                -- 订单基本信息
                side VARCHAR NOT NULL,
                order_type VARCHAR NOT NULL,
                quantity DECIMAL(18,8) NOT NULL,
                price DECIMAL(18,8),
                
                -- 执行信息
                status VARCHAR NOT NULL,
                executed_qty DECIMAL(18,8) DEFAULT 0,
                executed_quote_qty DECIMAL(18,8) DEFAULT 0,
                avg_price DECIMAL(18,8) DEFAULT 0,
                
                -- 费用信息
                commission DECIMAL(18,8) DEFAULT 0,
                commission_asset VARCHAR,
                commission_rate DOUBLE,
                
                -- 时间信息
                created_time TIMESTAMP NOT NULL,
                updated_time TIMESTAMP NOT NULL,
                filled_time TIMESTAMP,
                
                -- 高级订单参数
                stop_price DECIMAL(18,8),
                iceberg_qty DECIMAL(18,8),
                time_in_force VARCHAR DEFAULT 'GTC',
                position_side VARCHAR DEFAULT 'BOTH',
                reduce_only BOOLEAN DEFAULT FALSE,
                close_position BOOLEAN DEFAULT FALSE,
                
                -- 绩效相关
                realized_pnl DECIMAL(18,8),
                pnl_percentage DOUBLE,
                slippage DECIMAL(18,8),
                slippage_bps DOUBLE,
                
                -- 市场快照
                market_price_at_order DECIMAL(18,8),
                spread_at_order DECIMAL(18,8),
                volume_at_order DECIMAL(18,8),
                
                -- 风险指标
                leverage DOUBLE,
                margin_used DECIMAL(18,8),
                liquidation_price DECIMAL(18,8),
                
                -- 元数据
                order_source VARCHAR DEFAULT 'system',
                execution_algorithm VARCHAR,
                metadata JSON,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (order_id, environment)
            )
        """)
        
        # 订单表索引
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol_time ON {table_name}(symbol, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status_time ON {table_name}(status, timestamp DESC)", 
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_strategy ON {table_name}(strategy_id, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signal ON {table_name}(signal_id, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_side_type ON {table_name}(side, order_type, timestamp DESC)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.log_warning(f"创建订单表索引失败: {e}")

    def _create_historical_risk_metrics_table(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建历史风险指标表"""  
        table_name = "historical_risk_metrics"
        
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP NOT NULL,
                environment VARCHAR NOT NULL,
                calculation_period VARCHAR NOT NULL,
                account_id VARCHAR,
                
                -- 账户风险
                total_balance DECIMAL(18,8) NOT NULL,
                available_balance DECIMAL(18,8) NOT NULL,
                total_position_value DECIMAL(18,8) NOT NULL,
                
                -- 暴露度风险
                total_exposure DECIMAL(18,8) NOT NULL,
                long_exposure DECIMAL(18,8) NOT NULL,
                short_exposure DECIMAL(18,8) NOT NULL,
                net_exposure DECIMAL(18,8) NOT NULL,
                gross_exposure DECIMAL(18,8) NOT NULL,
                
                -- 保证金风险
                used_margin DECIMAL(18,8) NOT NULL,
                free_margin DECIMAL(18,8) NOT NULL,
                margin_ratio DOUBLE NOT NULL,
                maintenance_margin DECIMAL(18,8) NOT NULL,
                
                -- 杠杆风险
                effective_leverage DOUBLE NOT NULL,
                max_leverage_used DOUBLE NOT NULL,
                leverage_utilization DOUBLE NOT NULL,
                
                -- 流动性风险
                liquidation_buffer DECIMAL(18,8) NOT NULL,
                time_to_liquidation_hours DOUBLE,
                liquidity_score DOUBLE NOT NULL,
                
                -- 回撤风险
                max_drawdown DOUBLE NOT NULL,
                current_drawdown DOUBLE NOT NULL,
                drawdown_duration_days INTEGER NOT NULL,
                recovery_factor DOUBLE,
                
                -- VaR风险
                var_95 DECIMAL(18,8) NOT NULL,
                var_99 DECIMAL(18,8) NOT NULL,
                expected_shortfall_95 DECIMAL(18,8),
                expected_shortfall_99 DECIMAL(18,8),
                
                -- 波动性风险
                portfolio_volatility DOUBLE NOT NULL,
                realized_volatility DOUBLE NOT NULL,
                implied_volatility DOUBLE,
                volatility_percentile DOUBLE NOT NULL,
                
                -- 相关性风险
                max_correlation DOUBLE NOT NULL,
                avg_correlation DOUBLE NOT NULL,
                concentration_risk DOUBLE NOT NULL,
                
                -- 绩效风险比率
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                calmar_ratio DOUBLE,
                max_drawdown_ratio DOUBLE NOT NULL,
                
                -- 交易风险
                win_rate DOUBLE NOT NULL,
                profit_factor DOUBLE NOT NULL,
                avg_win_loss_ratio DOUBLE NOT NULL,
                consecutive_losses INTEGER NOT NULL,
                
                -- 流动性和市场风险
                market_impact DOUBLE,
                bid_ask_spread DECIMAL(18,8),
                slippage_cost DECIMAL(18,8),
                
                -- 资金成本
                funding_cost DECIMAL(18,8) DEFAULT 0,
                borrowing_cost DECIMAL(18,8),
                opportunity_cost DECIMAL(18,8),
                
                -- 分品种和时间风险
                symbol_risks JSON,
                sector_risks JSON,
                daily_pnl DECIMAL(18,8) NOT NULL,
                weekly_pnl DECIMAL(18,8),
                monthly_pnl DECIMAL(18,8),
                ytd_pnl DECIMAL(18,8),
                
                -- 压力测试和模型
                stress_test_results JSON,
                scenario_analysis JSON,
                model_confidence DOUBLE NOT NULL,
                prediction_accuracy DOUBLE,
                backtest_performance JSON,
                
                -- 元数据
                risk_model_version VARCHAR DEFAULT 'v1.0',
                calculation_method VARCHAR NOT NULL,
                data_quality_score DOUBLE NOT NULL,
                metadata JSON,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (timestamp, environment, calculation_period)
            )
        """)
        
        # 风险指标表索引
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_period_time ON {table_name}(calculation_period, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_account ON {table_name}(account_id, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_leverage ON {table_name}(effective_leverage, timestamp DESC)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_drawdown ON {table_name}(current_drawdown, timestamp DESC)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except Exception as e:
                self.log_warning(f"创建风险表索引失败: {e}")

    def _create_performance_statistics_table(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建性能统计表"""
        table_name = "performance_statistics"
        
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP NOT NULL,
                environment VARCHAR NOT NULL,
                symbol VARCHAR,
                strategy_id VARCHAR,
                period VARCHAR NOT NULL, -- daily, weekly, monthly, yearly
                
                -- 收益统计
                total_return DECIMAL(18,8) NOT NULL,
                annualized_return DECIMAL(18,8),
                cumulative_return DECIMAL(18,8) NOT NULL,
                
                -- 风险调整收益
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                calmar_ratio DOUBLE,
                information_ratio DOUBLE,
                
                -- 回撤统计
                max_drawdown DOUBLE NOT NULL,
                current_drawdown DOUBLE NOT NULL,
                avg_drawdown DOUBLE,
                drawdown_count INTEGER DEFAULT 0,
                
                -- 交易统计
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate DOUBLE DEFAULT 0,
                
                -- 盈亏统计
                gross_profit DECIMAL(18,8) DEFAULT 0,
                gross_loss DECIMAL(18,8) DEFAULT 0,
                net_profit DECIMAL(18,8) DEFAULT 0,
                profit_factor DOUBLE DEFAULT 0,
                
                -- 平均统计
                avg_win DECIMAL(18,8) DEFAULT 0,
                avg_loss DECIMAL(18,8) DEFAULT 0,
                avg_win_loss_ratio DOUBLE DEFAULT 0,
                avg_trade_duration_minutes INTEGER DEFAULT 0,
                
                -- 波动性统计
                volatility DOUBLE NOT NULL,
                downside_deviation DOUBLE,
                upside_capture DOUBLE,
                downside_capture DOUBLE,
                
                -- 市场相关
                beta DOUBLE,
                alpha DOUBLE,
                correlation DOUBLE,
                tracking_error DOUBLE,
                
                -- 交易成本
                total_fees DECIMAL(18,8) DEFAULT 0,
                avg_slippage_bps DOUBLE DEFAULT 0,
                turnover_rate DOUBLE DEFAULT 0,
                
                -- 元数据
                metadata JSON,
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (timestamp, environment, period, COALESCE(symbol, ''), COALESCE(strategy_id, ''))
            )
        """)

    def _create_analytical_views(self, conn: DuckDBPyConnection, env: TradingEnvironment) -> None:
        """创建分析视图"""
        
        # 市场数据分析视图
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_market_analysis AS
            SELECT 
                symbol,
                interval,
                DATE_TRUNC('day', timestamp) as trading_date,
                COUNT(*) as bar_count,
                AVG(close_price) as avg_price,
                STDDEV(close_price) as price_volatility,
                SUM(volume) as total_volume,
                MIN(low_price) as daily_low,
                MAX(high_price) as daily_high,
                FIRST(open_price ORDER BY timestamp) as daily_open,
                LAST(close_price ORDER BY timestamp) as daily_close
            FROM historical_market_data
            GROUP BY symbol, interval, DATE_TRUNC('day', timestamp)
            ORDER BY symbol, interval, trading_date DESC
        """)
        
        # 交易信号分析视图
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_signal_analysis AS
            SELECT 
                source,
                symbol,
                signal_type,
                action,
                DATE_TRUNC('day', timestamp) as signal_date,
                COUNT(*) as signal_count,
                AVG(strength) as avg_strength,
                AVG(confidence) as avg_confidence,
                COUNT(CASE WHEN execution_status = 'executed' THEN 1 END) as executed_count,
                AVG(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl END) as avg_pnl,
                COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) * 1.0 / 
                    NULLIF(COUNT(CASE WHEN realized_pnl IS NOT NULL THEN 1 END), 0) as win_rate
            FROM historical_trading_signals
            GROUP BY source, symbol, signal_type, action, DATE_TRUNC('day', timestamp)
            ORDER BY source, symbol, signal_date DESC
        """)
        
        # 风险分析视图  
        conn.execute("""
            CREATE VIEW IF NOT EXISTS v_risk_analysis AS
            SELECT 
                DATE_TRUNC('day', timestamp) as risk_date,
                calculation_period,
                AVG(effective_leverage) as avg_leverage,
                MAX(effective_leverage) as max_leverage,
                AVG(current_drawdown) as avg_drawdown,
                MAX(current_drawdown) as max_daily_drawdown,
                AVG(var_95) as avg_var_95,
                MAX(var_95) as max_var_95,
                AVG(portfolio_volatility) as avg_volatility,
                AVG(concentration_risk) as avg_concentration
            FROM historical_risk_metrics
            GROUP BY DATE_TRUNC('day', timestamp), calculation_period
            ORDER BY risk_date DESC, calculation_period
        """)

    def _start_background_tasks(self) -> None:
        """启动后台任务"""
        # 这里可以启动定期清理、优化等后台任务
        pass

    # ==================
    # 数据存储接口
    # ==================
    
    async def store_historical_market_data(
        self, 
        data: List[HistoricalMarketData],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """异步存储历史市场数据"""
        if not data:
            return 0
            
        def _store_sync():
            return self.store_historical_market_data_sync(data, environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _store_sync
        )

    def store_historical_market_data_sync(
        self, 
        data: List[HistoricalMarketData],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """同步存储历史市场数据"""
        if not data:
            return 0
        
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            # 转换数据为DataFrame格式
            records = []
            for item in data:
                record = {
                    'timestamp': item.timestamp,
                    'symbol': item.symbol,
                    'environment': item.environment,
                    'interval': item.interval,
                    'open_price': float(item.open_price),
                    'high_price': float(item.high_price),
                    'low_price': float(item.low_price),
                    'close_price': float(item.close_price),
                    'volume': float(item.volume),
                    'quote_volume': float(item.quote_volume) if item.quote_volume else None,
                    'trade_count': item.trade_count,
                    'taker_buy_volume': float(item.taker_buy_volume) if item.taker_buy_volume else None,
                    'taker_buy_quote_volume': float(item.taker_buy_quote_volume) if item.taker_buy_quote_volume else None,
                    'bid_price': float(item.bid_price) if item.bid_price else None,
                    'ask_price': float(item.ask_price) if item.ask_price else None,
                    'bid_volume': float(item.bid_volume) if item.bid_volume else None,
                    'ask_volume': float(item.ask_volume) if item.ask_volume else None,
                    'open_interest': float(item.open_interest) if item.open_interest else None,
                    'funding_rate': float(item.funding_rate) if item.funding_rate else None,
                    'mark_price': float(item.mark_price) if item.mark_price else None,
                    'index_price': float(item.index_price) if item.index_price else None,
                    'data_source': item.data_source,
                    'data_quality': item.data_quality,
                    'metadata': json.dumps(item.metadata),
                    'ingestion_time': item.ingestion_time
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            conn.register('market_data_batch', df)
            
            # 批量插入数据
            insert_sql = """
                INSERT INTO historical_market_data 
                SELECT * FROM market_data_batch
                ON CONFLICT (timestamp, symbol, interval, environment) DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trade_count = EXCLUDED.trade_count,
                    taker_buy_volume = EXCLUDED.taker_buy_volume,
                    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
                    bid_price = EXCLUDED.bid_price,
                    ask_price = EXCLUDED.ask_price,
                    bid_volume = EXCLUDED.bid_volume,
                    ask_volume = EXCLUDED.ask_volume,
                    open_interest = EXCLUDED.open_interest,
                    funding_rate = EXCLUDED.funding_rate,
                    mark_price = EXCLUDED.mark_price,
                    index_price = EXCLUDED.index_price,
                    data_source = EXCLUDED.data_source,
                    data_quality = EXCLUDED.data_quality,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"存储了{len(data)}条历史市场数据到{env.value}")
            return len(data)
            
        except Exception as e:
            self.log_error(f"存储历史市场数据失败: {e}")
            raise

    async def store_historical_trading_signals(
        self, 
        signals: List[HistoricalTradingSignal],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """异步存储历史交易信号"""
        if not signals:
            return 0
            
        def _store_sync():
            return self.store_historical_trading_signals_sync(signals, environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _store_sync
        )

    def store_historical_trading_signals_sync(
        self, 
        signals: List[HistoricalTradingSignal],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """同步存储历史交易信号"""
        if not signals:
            return 0
        
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            records = []
            for signal in signals:
                record = {
                    'timestamp': signal.timestamp,
                    'symbol': signal.symbol,
                    'environment': signal.environment,
                    'signal_id': signal.signal_id,
                    'source': signal.source,
                    'signal_type': signal.signal_type,
                    'action': signal.action,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'target_price': float(signal.target_price) if signal.target_price else None,
                    'suggested_quantity': float(signal.suggested_quantity) if signal.suggested_quantity else None,
                    'max_position_size': float(signal.max_position_size) if signal.max_position_size else None,
                    'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                    'take_profit': float(signal.take_profit) if signal.take_profit else None,
                    'max_drawdown_pct': signal.max_drawdown_pct,
                    'validity_period_minutes': signal.validity_period.total_seconds() / 60 if signal.validity_period else None,
                    'expiry_time': signal.expiry_time,
                    'technical_context': json.dumps(signal.technical_context),
                    'market_condition': signal.market_condition,
                    'execution_status': signal.execution_status,
                    'execution_price': float(signal.execution_price) if signal.execution_price else None,
                    'execution_time': signal.execution_time,
                    'execution_quantity': float(signal.execution_quantity) if signal.execution_quantity else None,
                    'realized_pnl': float(signal.realized_pnl) if signal.realized_pnl else None,
                    'unrealized_pnl': float(signal.unrealized_pnl) if signal.unrealized_pnl else None,
                    'win_rate': signal.win_rate,
                    'reason': signal.reason,
                    'metadata': json.dumps(signal.metadata),
                    'ingestion_time': signal.ingestion_time
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            conn.register('signals_batch', df)
            
            # 批量插入信号数据
            insert_sql = """
                INSERT INTO historical_trading_signals 
                SELECT * FROM signals_batch
                ON CONFLICT (timestamp, symbol, signal_id, environment) DO UPDATE SET
                    source = EXCLUDED.source,
                    signal_type = EXCLUDED.signal_type,
                    action = EXCLUDED.action,
                    strength = EXCLUDED.strength,
                    confidence = EXCLUDED.confidence,
                    target_price = EXCLUDED.target_price,
                    suggested_quantity = EXCLUDED.suggested_quantity,
                    max_position_size = EXCLUDED.max_position_size,
                    stop_loss = EXCLUDED.stop_loss,
                    take_profit = EXCLUDED.take_profit,
                    max_drawdown_pct = EXCLUDED.max_drawdown_pct,
                    validity_period_minutes = EXCLUDED.validity_period_minutes,
                    expiry_time = EXCLUDED.expiry_time,
                    technical_context = EXCLUDED.technical_context,
                    market_condition = EXCLUDED.market_condition,
                    execution_status = EXCLUDED.execution_status,
                    execution_price = EXCLUDED.execution_price,
                    execution_time = EXCLUDED.execution_time,
                    execution_quantity = EXCLUDED.execution_quantity,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    win_rate = EXCLUDED.win_rate,
                    reason = EXCLUDED.reason,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"存储了{len(signals)}条历史交易信号到{env.value}")
            return len(signals)
            
        except Exception as e:
            self.log_error(f"存储历史交易信号失败: {e}")
            raise

    async def store_historical_order_records(
        self, 
        orders: List[HistoricalOrderRecord],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """异步存储历史订单记录"""
        if not orders:
            return 0
            
        def _store_sync():
            return self.store_historical_order_records_sync(orders, environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _store_sync
        )

    def store_historical_order_records_sync(
        self, 
        orders: List[HistoricalOrderRecord],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """同步存储历史订单记录"""
        if not orders:
            return 0
        
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            records = []
            for order in orders:
                record = {
                    'timestamp': order.timestamp,
                    'symbol': order.symbol,
                    'environment': order.environment,
                    'order_id': order.order_id,
                    'client_order_id': order.client_order_id,
                    'strategy_id': order.strategy_id,
                    'signal_id': order.signal_id,
                    'side': order.side,
                    'order_type': order.order_type,
                    'quantity': float(order.quantity),
                    'price': float(order.price) if order.price else None,
                    'status': order.status,
                    'executed_qty': float(order.executed_qty),
                    'executed_quote_qty': float(order.executed_quote_qty),
                    'avg_price': float(order.avg_price),
                    'commission': float(order.commission),
                    'commission_asset': order.commission_asset,
                    'commission_rate': order.commission_rate,
                    'created_time': order.created_time,
                    'updated_time': order.updated_time,
                    'filled_time': order.filled_time,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'iceberg_qty': float(order.iceberg_qty) if order.iceberg_qty else None,
                    'time_in_force': order.time_in_force,
                    'position_side': order.position_side,
                    'reduce_only': order.reduce_only,
                    'close_position': order.close_position,
                    'realized_pnl': float(order.realized_pnl) if order.realized_pnl else None,
                    'pnl_percentage': order.pnl_percentage,
                    'slippage': float(order.slippage) if order.slippage else None,
                    'slippage_bps': order.slippage_bps,
                    'market_price_at_order': float(order.market_price_at_order) if order.market_price_at_order else None,
                    'spread_at_order': float(order.spread_at_order) if order.spread_at_order else None,
                    'volume_at_order': float(order.volume_at_order) if order.volume_at_order else None,
                    'leverage': order.leverage,
                    'margin_used': float(order.margin_used) if order.margin_used else None,
                    'liquidation_price': float(order.liquidation_price) if order.liquidation_price else None,
                    'order_source': order.order_source,
                    'execution_algorithm': order.execution_algorithm,
                    'metadata': json.dumps(order.metadata),
                    'ingestion_time': order.ingestion_time
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            conn.register('orders_batch', df)
            
            # 批量插入订单数据
            insert_sql = """
                INSERT INTO historical_order_records 
                SELECT * FROM orders_batch
                ON CONFLICT (order_id, environment) DO UPDATE SET
                    status = EXCLUDED.status,
                    executed_qty = EXCLUDED.executed_qty,
                    executed_quote_qty = EXCLUDED.executed_quote_qty,
                    avg_price = EXCLUDED.avg_price,
                    commission = EXCLUDED.commission,
                    commission_asset = EXCLUDED.commission_asset,
                    commission_rate = EXCLUDED.commission_rate,
                    updated_time = EXCLUDED.updated_time,
                    filled_time = EXCLUDED.filled_time,
                    realized_pnl = EXCLUDED.realized_pnl,
                    pnl_percentage = EXCLUDED.pnl_percentage,
                    slippage = EXCLUDED.slippage,
                    slippage_bps = EXCLUDED.slippage_bps,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"存储了{len(orders)}条历史订单记录到{env.value}")
            return len(orders)
            
        except Exception as e:
            self.log_error(f"存储历史订单记录失败: {e}")
            raise

    async def store_historical_risk_metrics(
        self, 
        metrics: List[HistoricalRiskMetrics],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """异步存储历史风险指标"""
        if not metrics:
            return 0
            
        def _store_sync():
            return self.store_historical_risk_metrics_sync(metrics, environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _store_sync
        )

    def store_historical_risk_metrics_sync(
        self, 
        metrics: List[HistoricalRiskMetrics],
        environment: Optional[TradingEnvironment] = None
    ) -> int:
        """同步存储历史风险指标"""
        if not metrics:
            return 0
        
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            records = []
            for metric in metrics:
                record = {
                    'timestamp': metric.timestamp,
                    'environment': metric.environment,
                    'calculation_period': metric.calculation_period,
                    'account_id': metric.account_id,
                    'total_balance': float(metric.total_balance),
                    'available_balance': float(metric.available_balance),
                    'total_position_value': float(metric.total_position_value),
                    'total_exposure': float(metric.total_exposure),
                    'long_exposure': float(metric.long_exposure),
                    'short_exposure': float(metric.short_exposure),
                    'net_exposure': float(metric.net_exposure),
                    'gross_exposure': float(metric.gross_exposure),
                    'used_margin': float(metric.used_margin),
                    'free_margin': float(metric.free_margin),
                    'margin_ratio': metric.margin_ratio,
                    'maintenance_margin': float(metric.maintenance_margin),
                    'effective_leverage': metric.effective_leverage,
                    'max_leverage_used': metric.max_leverage_used,
                    'leverage_utilization': metric.leverage_utilization,
                    'liquidation_buffer': float(metric.liquidation_buffer),
                    'time_to_liquidation_hours': metric.time_to_liquidation_hours,
                    'liquidity_score': metric.liquidity_score,
                    'max_drawdown': metric.max_drawdown,
                    'current_drawdown': metric.current_drawdown,
                    'drawdown_duration_days': metric.drawdown_duration_days,
                    'recovery_factor': metric.recovery_factor,
                    'var_95': float(metric.var_95),
                    'var_99': float(metric.var_99),
                    'expected_shortfall_95': float(metric.expected_shortfall_95) if metric.expected_shortfall_95 else None,
                    'expected_shortfall_99': float(metric.expected_shortfall_99) if metric.expected_shortfall_99 else None,
                    'portfolio_volatility': metric.portfolio_volatility,
                    'realized_volatility': metric.realized_volatility,
                    'implied_volatility': metric.implied_volatility,
                    'volatility_percentile': metric.volatility_percentile,
                    'max_correlation': metric.max_correlation,
                    'avg_correlation': metric.avg_correlation,
                    'concentration_risk': metric.concentration_risk,
                    'sharpe_ratio': metric.sharpe_ratio,
                    'sortino_ratio': metric.sortino_ratio,
                    'calmar_ratio': metric.calmar_ratio,
                    'max_drawdown_ratio': metric.max_drawdown_ratio,
                    'win_rate': metric.win_rate,
                    'profit_factor': metric.profit_factor,
                    'avg_win_loss_ratio': metric.avg_win_loss_ratio,
                    'consecutive_losses': metric.consecutive_losses,
                    'market_impact': metric.market_impact,
                    'bid_ask_spread': float(metric.bid_ask_spread) if metric.bid_ask_spread else None,
                    'slippage_cost': float(metric.slippage_cost) if metric.slippage_cost else None,
                    'funding_cost': float(metric.funding_cost),
                    'borrowing_cost': float(metric.borrowing_cost) if metric.borrowing_cost else None,
                    'opportunity_cost': float(metric.opportunity_cost) if metric.opportunity_cost else None,
                    'symbol_risks': json.dumps(metric.symbol_risks),
                    'sector_risks': json.dumps(metric.sector_risks),
                    'daily_pnl': float(metric.daily_pnl),
                    'weekly_pnl': float(metric.weekly_pnl) if metric.weekly_pnl else None,
                    'monthly_pnl': float(metric.monthly_pnl) if metric.monthly_pnl else None,
                    'ytd_pnl': float(metric.ytd_pnl) if metric.ytd_pnl else None,
                    'stress_test_results': json.dumps(metric.stress_test_results),
                    'scenario_analysis': json.dumps(metric.scenario_analysis),
                    'model_confidence': metric.model_confidence,
                    'prediction_accuracy': metric.prediction_accuracy,
                    'backtest_performance': json.dumps(metric.backtest_performance),
                    'risk_model_version': metric.risk_model_version,
                    'calculation_method': metric.calculation_method,
                    'data_quality_score': metric.data_quality_score,
                    'metadata': json.dumps(metric.metadata),
                    'ingestion_time': metric.ingestion_time
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            conn.register('risk_metrics_batch', df)
            
            # 批量插入风险指标数据
            insert_sql = """
                INSERT INTO historical_risk_metrics 
                SELECT * FROM risk_metrics_batch
                ON CONFLICT (timestamp, environment, calculation_period) DO UPDATE SET
                    account_id = EXCLUDED.account_id,
                    total_balance = EXCLUDED.total_balance,
                    available_balance = EXCLUDED.available_balance,
                    total_position_value = EXCLUDED.total_position_value,
                    total_exposure = EXCLUDED.total_exposure,
                    long_exposure = EXCLUDED.long_exposure,
                    short_exposure = EXCLUDED.short_exposure,
                    net_exposure = EXCLUDED.net_exposure,
                    gross_exposure = EXCLUDED.gross_exposure,
                    used_margin = EXCLUDED.used_margin,
                    free_margin = EXCLUDED.free_margin,
                    margin_ratio = EXCLUDED.margin_ratio,
                    maintenance_margin = EXCLUDED.maintenance_margin,
                    effective_leverage = EXCLUDED.effective_leverage,
                    max_leverage_used = EXCLUDED.max_leverage_used,
                    leverage_utilization = EXCLUDED.leverage_utilization,
                    liquidation_buffer = EXCLUDED.liquidation_buffer,
                    time_to_liquidation_hours = EXCLUDED.time_to_liquidation_hours,
                    liquidity_score = EXCLUDED.liquidity_score,
                    max_drawdown = EXCLUDED.max_drawdown,
                    current_drawdown = EXCLUDED.current_drawdown,
                    drawdown_duration_days = EXCLUDED.drawdown_duration_days,
                    recovery_factor = EXCLUDED.recovery_factor,
                    var_95 = EXCLUDED.var_95,
                    var_99 = EXCLUDED.var_99,
                    expected_shortfall_95 = EXCLUDED.expected_shortfall_95,
                    expected_shortfall_99 = EXCLUDED.expected_shortfall_99,
                    portfolio_volatility = EXCLUDED.portfolio_volatility,
                    realized_volatility = EXCLUDED.realized_volatility,
                    implied_volatility = EXCLUDED.implied_volatility,
                    volatility_percentile = EXCLUDED.volatility_percentile,
                    max_correlation = EXCLUDED.max_correlation,
                    avg_correlation = EXCLUDED.avg_correlation,
                    concentration_risk = EXCLUDED.concentration_risk,
                    sharpe_ratio = EXCLUDED.sharpe_ratio,
                    sortino_ratio = EXCLUDED.sortino_ratio,
                    calmar_ratio = EXCLUDED.calmar_ratio,
                    max_drawdown_ratio = EXCLUDED.max_drawdown_ratio,
                    win_rate = EXCLUDED.win_rate,
                    profit_factor = EXCLUDED.profit_factor,
                    avg_win_loss_ratio = EXCLUDED.avg_win_loss_ratio,
                    consecutive_losses = EXCLUDED.consecutive_losses,
                    market_impact = EXCLUDED.market_impact,
                    bid_ask_spread = EXCLUDED.bid_ask_spread,
                    slippage_cost = EXCLUDED.slippage_cost,
                    funding_cost = EXCLUDED.funding_cost,
                    borrowing_cost = EXCLUDED.borrowing_cost,
                    opportunity_cost = EXCLUDED.opportunity_cost,
                    symbol_risks = EXCLUDED.symbol_risks,
                    sector_risks = EXCLUDED.sector_risks,
                    daily_pnl = EXCLUDED.daily_pnl,
                    weekly_pnl = EXCLUDED.weekly_pnl,
                    monthly_pnl = EXCLUDED.monthly_pnl,
                    ytd_pnl = EXCLUDED.ytd_pnl,
                    stress_test_results = EXCLUDED.stress_test_results,
                    scenario_analysis = EXCLUDED.scenario_analysis,
                    model_confidence = EXCLUDED.model_confidence,
                    prediction_accuracy = EXCLUDED.prediction_accuracy,
                    backtest_performance = EXCLUDED.backtest_performance,
                    risk_model_version = EXCLUDED.risk_model_version,
                    calculation_method = EXCLUDED.calculation_method,
                    data_quality_score = EXCLUDED.data_quality_score,
                    metadata = EXCLUDED.metadata,
                    ingestion_time = EXCLUDED.ingestion_time
            """
            
            conn.execute(insert_sql)
            
            self.log_info(f"存储了{len(metrics)}条历史风险指标到{env.value}")
            return len(metrics)
            
        except Exception as e:
            self.log_error(f"存储历史风险指标失败: {e}")
            raise

    # ==================
    # 数据查询接口
    # ==================
    
    async def query_historical_market_data(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        environment: Optional[TradingEnvironment] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """异步查询历史市场数据"""
        def _query_sync():
            return self.query_historical_market_data_sync(
                symbol, interval, start_time, end_time, environment, limit
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _query_sync
        )

    def query_historical_market_data_sync(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        environment: Optional[TradingEnvironment] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """同步查询历史市场数据"""
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            query = """
                SELECT * FROM historical_market_data
                WHERE symbol = ? 
                AND interval = ?
                AND environment = ?
                AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            params = [symbol, interval, env.value, start_time, end_time]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            result = conn.execute(query, params).fetchdf()
            
            # 转换为字典列表
            records = result.to_dict('records')
            
            self.log_debug(f"查询到{len(records)}条历史市场数据: {symbol} {interval}")
            return records
            
        except Exception as e:
            self.log_error(f"查询历史市场数据失败: {e}")
            return []

    async def query_historical_trading_signals(
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        environment: Optional[TradingEnvironment] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """异步查询历史交易信号"""
        def _query_sync():
            return self.query_historical_trading_signals_sync(
                symbol, source, signal_type, start_time, end_time, environment, limit
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _query_sync
        )

    def query_historical_trading_signals_sync(
        self,
        symbol: Optional[str] = None,
        source: Optional[str] = None,
        signal_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        environment: Optional[TradingEnvironment] = None,
        limit: Optional[int] = 1000
    ) -> List[Dict[str, Any]]:
        """同步查询历史交易信号"""
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            conditions = ["environment = ?"]
            params = [env.value]
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if source:
                conditions.append("source = ?") 
                params.append(source)
            
            if signal_type:
                conditions.append("signal_type = ?")
                params.append(signal_type)
            
            if start_time and end_time:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_time, end_time])
            
            query = f"""
                SELECT * FROM historical_trading_signals
                WHERE {' AND '.join(conditions)}
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            result = conn.execute(query, params).fetchdf()
            records = result.to_dict('records')
            
            self.log_debug(f"查询到{len(records)}条历史交易信号")
            return records
            
        except Exception as e:
            self.log_error(f"查询历史交易信号失败: {e}")
            return []

    # ==================
    # 分析查询接口
    # ==================
    
    async def get_trading_performance_analysis(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, Any]:
        """获取交易绩效分析"""
        def _analyze_sync():
            return self.get_trading_performance_analysis_sync(
                symbol, strategy_id, start_time, end_time, environment
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _analyze_sync
        )

    def get_trading_performance_analysis_sync(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, Any]:
        """同步获取交易绩效分析"""
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        conn = self.duckdb_manager.get_connection(env)
        
        try:
            conditions = ["environment = ?", "status = 'FILLED'"]
            params = [env.value]
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)
            
            if start_time and end_time:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_time, end_time])
            
            # 获取基本统计
            stats_query = f"""
                SELECT 
                    COUNT(*) as total_orders,
                    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_orders,
                    COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_orders,
                    SUM(realized_pnl) as total_pnl,
                    AVG(realized_pnl) as avg_pnl,
                    MAX(realized_pnl) as max_win,
                    MIN(realized_pnl) as max_loss,
                    SUM(commission) as total_fees,
                    AVG(slippage_bps) as avg_slippage_bps
                FROM historical_order_records
                WHERE {' AND '.join(conditions)}
                AND realized_pnl IS NOT NULL
            """
            
            stats_result = conn.execute(stats_query, params).fetchone()
            
            if not stats_result or stats_result[0] == 0:
                return {
                    'total_orders': 0,
                    'performance_metrics': {},
                    'risk_metrics': {},
                    'summary': '无交易记录'
                }
            
            total_orders = stats_result[0]
            winning_orders = stats_result[1] or 0
            losing_orders = stats_result[2] or 0
            total_pnl = float(stats_result[3]) if stats_result[3] else 0.0
            avg_pnl = float(stats_result[4]) if stats_result[4] else 0.0
            max_win = float(stats_result[5]) if stats_result[5] else 0.0
            max_loss = float(stats_result[6]) if stats_result[6] else 0.0
            total_fees = float(stats_result[7]) if stats_result[7] else 0.0
            avg_slippage_bps = float(stats_result[8]) if stats_result[8] else 0.0
            
            # 计算绩效指标
            win_rate = winning_orders / total_orders if total_orders > 0 else 0.0
            profit_factor = abs(max_win * winning_orders / max_loss / losing_orders) if losing_orders > 0 and max_loss != 0 else float('inf')
            
            # 获取收益序列用于计算风险指标
            pnl_series_query = f"""
                SELECT realized_pnl, timestamp
                FROM historical_order_records
                WHERE {' AND '.join(conditions)}
                AND realized_pnl IS NOT NULL
                ORDER BY timestamp
            """
            
            pnl_df = conn.execute(pnl_series_query, params).fetchdf()
            
            # 计算风险指标
            risk_metrics = {}
            if len(pnl_df) > 1:
                pnl_series = pnl_df['realized_pnl'].astype(float)
                cumulative_pnl = pnl_series.cumsum()
                
                # 计算最大回撤
                peak = cumulative_pnl.expanding(min_periods=1).max()
                drawdown = (cumulative_pnl - peak) / peak.where(peak != 0, 1)
                max_drawdown = drawdown.min()
                
                # 计算夏普比率（简化版本）
                if len(pnl_series) > 1:
                    returns_std = pnl_series.std()
                    sharpe_ratio = (avg_pnl / returns_std) if returns_std > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
                
                risk_metrics = {
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility': float(returns_std) if len(pnl_series) > 1 else 0.0
                }
            
            analysis = {
                'total_orders': total_orders,
                'performance_metrics': {
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': avg_pnl,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_win': max_win,
                    'max_loss': max_loss,
                    'winning_orders': winning_orders,
                    'losing_orders': losing_orders
                },
                'cost_metrics': {
                    'total_fees': total_fees,
                    'avg_slippage_bps': avg_slippage_bps,
                    'net_pnl': total_pnl - total_fees
                },
                'risk_metrics': risk_metrics,
                'summary': f"总订单{total_orders}笔，胜率{win_rate:.2%}，总盈亏{total_pnl:.2f}"
            }
            
            self.log_debug(f"生成交易绩效分析: {total_orders}笔订单")
            return analysis
            
        except Exception as e:
            self.log_error(f"获取交易绩效分析失败: {e}")
            return {'error': str(e)}

    # ==================
    # 数据生命周期管理
    # ==================
    
    async def cleanup_expired_data(
        self, 
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, int]:
        """清理过期数据"""
        def _cleanup_sync():
            return self.cleanup_expired_data_sync(environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _cleanup_sync
        )

    def cleanup_expired_data_sync(
        self, 
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, int]:
        """同步清理过期数据"""
        env = environment or self.duckdb_manager.get_current_environment() or TradingEnvironment.TESTNET
        
        cleanup_results = {}
        
        # 市场数据清理
        market_data_deleted = self.duckdb_manager.cleanup_old_data(
            'historical_market_data', 
            self.retention_policy.market_data_retention_days,
            env
        )
        cleanup_results['market_data'] = market_data_deleted
        
        # 交易信号清理
        signals_deleted = self.duckdb_manager.cleanup_old_data(
            'historical_trading_signals',
            self.retention_policy.signal_retention_days, 
            env
        )
        cleanup_results['trading_signals'] = signals_deleted
        
        # 订单记录清理
        orders_deleted = self.duckdb_manager.cleanup_old_data(
            'historical_order_records',
            self.retention_policy.order_retention_days,
            env
        )
        cleanup_results['order_records'] = orders_deleted
        
        # 风险指标清理
        risk_deleted = self.duckdb_manager.cleanup_old_data(
            'historical_risk_metrics',
            self.retention_policy.risk_metrics_retention_days,
            env
        )
        cleanup_results['risk_metrics'] = risk_deleted
        
        # 性能统计清理
        perf_deleted = self.duckdb_manager.cleanup_old_data(
            'performance_statistics',
            self.retention_policy.performance_retention_days,
            env
        )
        cleanup_results['performance_stats'] = perf_deleted
        
        total_deleted = sum(cleanup_results.values())
        self.log_info(f"数据清理完成，共删除{total_deleted}条记录: {cleanup_results}")
        
        return cleanup_results

    # ==================
    # 导出和备份功能
    # ==================
    
    async def export_data_to_parquet(
        self,
        table_name: str,
        output_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        partition_by: Optional[str] = None,
        environment: Optional[TradingEnvironment] = None
    ) -> bool:
        """异步导出数据到Parquet格式"""
        def _export_sync():
            return self.duckdb_manager.export_to_parquet(
                table_name, output_path, start_time, end_time, partition_by, environment
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _export_sync
        )

    async def get_storage_statistics(
        self, 
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, Any]:
        """异步获取存储统计信息"""
        def _stats_sync():
            return self.duckdb_manager.get_storage_statistics(environment)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _stats_sync
        )

    def close(self) -> None:
        """关闭存储系统"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.log_info("DuckDB存储系统已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 注意：不在模块级别创建实例，避免导入时的初始化问题
# 使用时需要显式创建实例：storage = DuckDBStorage()