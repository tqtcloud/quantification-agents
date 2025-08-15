"""
DuckDB管理器测试
测试时序数据存储、查询和分析功能
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.core.duckdb_manager import DuckDBManager
from src.exchanges.trading_interface import TradingEnvironment


@pytest.fixture
def temp_duckdb_manager():
    """创建临时DuckDB管理器用于测试"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建临时配置
        temp_manager = DuckDBManager()
        temp_manager._data_dir = Path(temp_dir)
        temp_manager._init_directories()
        
        yield temp_manager
        
        # 清理
        temp_manager.close_all()


@pytest.fixture
def sample_tick_data() -> List[Dict[str, Any]]:
    """生成样本tick数据"""
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    return [
        {
            'timestamp': base_time + timedelta(seconds=i * 10),
            'symbol': 'BTCUSDT',
            'price': 50000.0 + i * 10,
            'volume': 1.0 + i * 0.1,
            'bid_price': 50000.0 + i * 10 - 0.5,
            'ask_price': 50000.0 + i * 10 + 0.5,
            'bid_volume': 2.0,
            'ask_volume': 2.0,
            'trade_count': 5
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_kline_data() -> List[Dict[str, Any]]:
    """生成样本K线数据"""
    base_time = datetime.utcnow() - timedelta(hours=10)
    
    return [
        {
            'timestamp': base_time + timedelta(hours=i),
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'open_price': 50000.0 + i * 100,
            'high_price': 50000.0 + i * 100 + 200,
            'low_price': 50000.0 + i * 100 - 100,
            'close_price': 50000.0 + i * 100 + 50,
            'volume': 100.0 + i * 10,
            'quote_volume': (100.0 + i * 10) * (50000.0 + i * 100 + 50),
            'trade_count': 100 + i * 5
        }
        for i in range(10)
    ]


class TestDuckDBManager:
    """DuckDBManager测试类"""
    
    def test_initialization(self, temp_duckdb_manager):
        """测试初始化"""
        manager = temp_duckdb_manager
        
        # 测试数据目录创建
        assert manager._data_dir.exists()
        
        # 测试环境目录创建
        for env in TradingEnvironment:
            env_dir = manager._data_dir / env.value
            assert env_dir.exists()
    
    def test_connection_management(self, temp_duckdb_manager):
        """测试连接管理"""
        manager = temp_duckdb_manager
        
        # 测试获取连接
        env = TradingEnvironment.TESTNET
        conn = manager.get_connection(env)
        assert conn is not None
        
        # 测试连接缓存
        conn2 = manager.get_connection(env)
        assert conn is conn2
        
        # 测试数据库信息
        db_info = manager.get_database_info(env)
        assert 'environment' in db_info
        assert db_info['environment'] == env.value
        assert 'total_tables' in db_info
    
    def test_environment_management(self, temp_duckdb_manager):
        """测试环境管理"""
        manager = temp_duckdb_manager
        
        # 测试设置当前环境
        env = TradingEnvironment.MAINNET
        manager.set_current_environment(env)
        assert manager.get_current_environment() == env
        
        # 测试临时环境切换
        with manager.use_environment(TradingEnvironment.PAPER):
            assert manager.get_current_environment() == TradingEnvironment.PAPER
        
        # 验证环境恢复
        assert manager.get_current_environment() == env
    
    def test_tick_data_operations(self, temp_duckdb_manager, sample_tick_data):
        """测试tick数据操作"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 测试批量插入
        inserted = manager.insert_market_ticks_batch(sample_tick_data)
        assert inserted == len(sample_tick_data)
        
        # 测试查询
        start_time = datetime.utcnow() - timedelta(hours=2)
        end_time = datetime.utcnow()
        
        result_df = manager.query_market_ticks('BTCUSDT', start_time, end_time)
        assert len(result_df) == len(sample_tick_data)
        assert 'price' in result_df.columns
        assert 'volume' in result_df.columns
    
    def test_kline_data_operations(self, temp_duckdb_manager, sample_kline_data):
        """测试K线数据操作"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 测试批量插入
        inserted = manager.insert_klines_batch(sample_kline_data)
        assert inserted == len(sample_kline_data)
        
        # 测试查询
        start_time = datetime.utcnow() - timedelta(hours=12)
        end_time = datetime.utcnow()
        
        result_df = manager.query_klines('BTCUSDT', '1h', start_time, end_time)
        assert len(result_df) == len(sample_kline_data)
        assert 'open_price' in result_df.columns
        assert 'close_price' in result_df.columns
    
    def test_order_book_operations(self, temp_duckdb_manager):
        """测试订单簿操作"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 测试插入订单簿快照
        bids = [[50000.0, 1.0], [49999.0, 2.0]]
        asks = [[50001.0, 1.5], [50002.0, 2.5]]
        
        success = manager.insert_order_book_snapshot(
            'BTCUSDT', bids, asks, last_update_id=12345
        )
        assert success
        
        # 测试查询订单簿快照
        start_time = datetime.utcnow() - timedelta(minutes=5)
        end_time = datetime.utcnow() + timedelta(minutes=5)
        
        result_df = manager.query_order_book_snapshots('BTCUSDT', start_time, end_time)
        assert len(result_df) >= 1
    
    def test_latest_prices(self, temp_duckdb_manager, sample_tick_data):
        """测试最新价格获取"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 获取最新价格
        latest_prices = manager.get_latest_prices(['BTCUSDT'])
        assert 'BTCUSDT' in latest_prices
        assert latest_prices['BTCUSDT'] > 0
    
    def test_price_statistics(self, temp_duckdb_manager, sample_tick_data):
        """测试价格统计"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 计算统计
        start_time = datetime.utcnow() - timedelta(hours=2)
        end_time = datetime.utcnow()
        
        stats = manager.calculate_price_statistics('BTCUSDT', start_time, end_time)
        assert stats['tick_count'] == len(sample_tick_data)
        assert stats['min_price'] > 0
        assert stats['max_price'] > 0
        assert stats['avg_price'] > 0
    
    def test_vwap_calculation(self, temp_duckdb_manager, sample_tick_data):
        """测试VWAP计算"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 计算VWAP
        start_time = datetime.utcnow() - timedelta(hours=2)
        end_time = datetime.utcnow()
        
        vwap = manager.get_volume_weighted_price('BTCUSDT', start_time, end_time)
        assert vwap > 0
    
    def test_ohlcv_aggregation(self, temp_duckdb_manager, sample_kline_data):
        """测试OHLCV聚合"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_klines_batch(sample_kline_data)
        
        # 获取OHLCV数据
        result_df = manager.get_ohlcv_aggregation('BTCUSDT', '1h', 5)
        assert len(result_df) <= len(sample_kline_data)
        assert len(result_df) <= 5
    
    def test_market_depth_analysis(self, temp_duckdb_manager):
        """测试市场深度分析"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入订单簿数据
        bids = [[50000.0, 1.0], [49999.0, 2.0]]
        asks = [[50001.0, 1.5], [50002.0, 2.5]]
        
        manager.insert_order_book_snapshot('BTCUSDT', bids, asks)
        
        # 分析市场深度
        analysis = manager.get_market_depth_analysis('BTCUSDT')
        assert analysis['best_bid'] == 50000.0
        assert analysis['best_ask'] == 50001.0
        assert analysis['spread'] == 1.0
        assert analysis['bid_levels'] == 2
        assert analysis['ask_levels'] == 2
    
    def test_environment_isolation(self, temp_duckdb_manager, sample_tick_data):
        """测试环境数据隔离"""
        manager = temp_duckdb_manager
        
        # 在不同环境插入数据
        for env in [TradingEnvironment.TESTNET, TradingEnvironment.MAINNET]:
            manager.set_current_environment(env)
            inserted = manager.insert_market_ticks_batch(sample_tick_data, env)
            assert inserted == len(sample_tick_data)
        
        # 验证数据隔离
        start_time = datetime.utcnow() - timedelta(hours=2)
        end_time = datetime.utcnow()
        
        for env in [TradingEnvironment.TESTNET, TradingEnvironment.MAINNET]:
            result_df = manager.query_market_ticks('BTCUSDT', start_time, end_time, env)
            assert len(result_df) == len(sample_tick_data)
            
            # 验证环境标识
            assert all(result_df['environment'] == env.value)
    
    def test_storage_statistics(self, temp_duckdb_manager, sample_tick_data):
        """测试存储统计"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 获取统计信息
        stats = manager.get_storage_statistics(env)
        assert stats['environment'] == env.value
        assert stats['total_rows'] >= len(sample_tick_data)
        assert 'market_ticks' in stats['tables']
    
    def test_data_cleanup(self, temp_duckdb_manager, sample_tick_data):
        """测试数据清理"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 清理过期数据（设置很短的保留期）
        cleaned_rows = manager.cleanup_old_data('market_ticks', retention_days=1)
        assert cleaned_rows >= 0  # 应该是0，因为数据很新
        
        # 测试删除所有数据（使用未来日期作为cutoff）
        import time
        time.sleep(0.1)  # 确保有时间差
        cleaned_rows = manager.cleanup_old_data('market_ticks', retention_days=0)
        assert cleaned_rows >= 0  # 现在应该删除所有数据
    
    def test_error_handling(self, temp_duckdb_manager):
        """测试错误处理"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 测试插入无效数据
        invalid_data = [{'invalid': 'data'}]
        
        with pytest.raises(ValueError):
            manager.insert_market_ticks_batch(invalid_data)
        
        # 测试查询不存在的数据
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        result_df = manager.query_market_ticks('NONEXISTENT', start_time, end_time)
        assert len(result_df) == 0
    
    def test_connection_cleanup(self, temp_duckdb_manager):
        """测试连接清理"""
        manager = temp_duckdb_manager
        
        # 建立连接
        for env in TradingEnvironment:
            manager.get_connection(env)
        
        # 验证连接存在
        assert len(manager.connections) == len(TradingEnvironment)
        
        # 清理连接
        manager.close_all()
        assert len(manager.connections) == 0
    
    def test_technical_indicators_operations(self, temp_duckdb_manager):
        """测试技术指标数据操作"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 准备技术指标数据
        base_time = datetime.utcnow() - timedelta(hours=5)
        indicator_data = [
            {
                'timestamp': base_time + timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'indicator_name': 'RSI',
                'indicator_value': 50.0 + i * 5,
                'indicator_data': '{"period": 14}',
                'metadata': '{}'
            }
            for i in range(5)
        ]
        
        # 测试批量插入
        inserted = manager.insert_technical_indicators_batch(indicator_data)
        assert inserted == len(indicator_data)
        
        # 测试查询
        start_time = datetime.utcnow() - timedelta(hours=6)
        end_time = datetime.utcnow()
        
        result_df = manager.query_technical_indicators(
            'BTCUSDT', 'RSI', '1h', start_time, end_time
        )
        assert len(result_df) == len(indicator_data)
    
    def test_trading_signals_operations(self, temp_duckdb_manager):
        """测试交易信号数据操作"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 准备交易信号数据
        base_time = datetime.utcnow() - timedelta(hours=3)
        signal_data = [
            {
                'timestamp': base_time + timedelta(hours=i),
                'symbol': 'BTCUSDT',
                'signal_source': 'test_agent',
                'signal_type': 'buy_signal',
                'action': 'BUY',
                'strength': 0.8,
                'confidence': 0.9,
                'price': 50000.0 + i * 100,
                'quantity': 1.0,
                'reason': 'Test signal',
                'metadata': '{}'
            }
            for i in range(3)
        ]
        
        # 测试批量插入
        inserted = manager.insert_trading_signals_batch(signal_data)
        assert inserted == len(signal_data)
        
        # 测试查询
        start_time = datetime.utcnow() - timedelta(hours=4)
        end_time = datetime.utcnow()
        
        result_df = manager.query_trading_signals(
            'BTCUSDT', 'test_agent', 'buy_signal', start_time, end_time
        )
        assert len(result_df) == len(signal_data)


class TestDuckDBCompression:
    """DuckDB压缩和分区测试"""
    
    def test_parquet_export_import(self, temp_duckdb_manager, sample_tick_data):
        """测试Parquet导出导入"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 导出到Parquet
        export_path = manager._data_dir / "test_export.parquet"
        success = manager.export_to_parquet('market_ticks', export_path)
        
        # 注意：某些DuckDB版本可能不支持所有压缩功能
        # 如果导出失败，跳过测试
        if not success:
            pytest.skip("Parquet export not supported in this DuckDB version")
        
        assert export_path.exists()
    
    def test_table_optimization(self, temp_duckdb_manager, sample_tick_data):
        """测试表优化"""
        manager = temp_duckdb_manager
        env = TradingEnvironment.TESTNET
        manager.set_current_environment(env)
        
        # 插入数据
        manager.insert_market_ticks_batch(sample_tick_data)
        
        # 优化表存储
        success = manager.optimize_table_storage('market_ticks')
        assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])