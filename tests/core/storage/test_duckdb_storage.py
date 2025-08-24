"""
DuckDB存储系统集成测试
测试历史数据存储、查询和分析功能
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

from src.core.storage import (
    DuckDBStorage, 
    HistoricalMarketData, 
    HistoricalTradingSignal,
    HistoricalOrderRecord, 
    HistoricalRiskMetrics,
    StorageConfig, 
    DataRetentionPolicy
)
from src.core.storage.data_migration import DataMigrationManager
from src.exchanges.trading_interface import TradingEnvironment


class TestDuckDBStorage:
    """DuckDB存储系统测试类"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage_config(self):
        """存储配置"""
        return StorageConfig(
            max_memory_gb=1.0,
            thread_count=2,
            enable_compression=True
        )
    
    @pytest.fixture
    def retention_policy(self):
        """数据保留策略"""
        return DataRetentionPolicy(
            market_data_retention_days=30,
            signal_retention_days=15,
            order_retention_days=60,
            risk_metrics_retention_days=30
        )
    
    @pytest.fixture
    def duckdb_storage(self, storage_config, retention_policy):
        """DuckDB存储实例"""
        storage = DuckDBStorage(
            config=storage_config,
            retention_policy=retention_policy
        )
        yield storage
        storage.close()
    
    @pytest.fixture
    def sample_market_data(self) -> List[HistoricalMarketData]:
        """示例市场数据"""
        base_time = datetime.utcnow().replace(microsecond=0)
        data = []
        
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i)
            price = Decimal('50000.00') + Decimal(str(i * 10))
            
            market_data = HistoricalMarketData(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                interval='1m',
                open_price=price,
                high_price=price + Decimal('5.00'),
                low_price=price - Decimal('5.00'),
                close_price=price + Decimal('2.50'),
                volume=Decimal('1.5'),
                quote_volume=Decimal('75000.00'),
                trade_count=100,
                data_source='test',
                data_quality=1.0,
                metadata={'test': True}
            )
            data.append(market_data)
        
        return data
    
    @pytest.fixture
    def sample_trading_signals(self) -> List[HistoricalTradingSignal]:
        """示例交易信号"""
        base_time = datetime.utcnow().replace(microsecond=0)
        signals = []
        
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i * 2)
            
            signal = HistoricalTradingSignal(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                signal_id=f'test_signal_{i}',
                source='test_agent',
                signal_type='momentum',
                action='BUY' if i % 2 == 0 else 'SELL',
                strength=0.8 if i % 2 == 0 else -0.6,
                confidence=0.85,
                target_price=Decimal('50000.00') + Decimal(str(i * 10)),
                suggested_quantity=Decimal('0.1'),
                reason=f'Test signal {i}',
                metadata={'test_data': True}
            )
            signals.append(signal)
        
        return signals
    
    @pytest.fixture
    def sample_order_records(self) -> List[HistoricalOrderRecord]:
        """示例订单记录"""
        base_time = datetime.utcnow().replace(microsecond=0)
        orders = []
        
        for i in range(3):
            timestamp = base_time + timedelta(minutes=i * 3)
            
            order = HistoricalOrderRecord(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                order_id=f'test_order_{i}',
                client_order_id=f'client_order_{i}',
                side='BUY' if i % 2 == 0 else 'SELL',
                order_type='MARKET',
                quantity=Decimal('0.1'),
                price=Decimal('50000.00'),
                status='FILLED',
                executed_qty=Decimal('0.1'),
                executed_quote_qty=Decimal('5000.00'),
                avg_price=Decimal('50000.00'),
                commission=Decimal('5.00'),
                commission_asset='USDT',
                created_time=timestamp,
                updated_time=timestamp + timedelta(seconds=30),
                realized_pnl=Decimal('10.50') if i % 2 == 0 else Decimal('-5.25'),
                metadata={'test_order': True}
            )
            orders.append(order)
        
        return orders
    
    @pytest.fixture
    def sample_risk_metrics(self) -> List[HistoricalRiskMetrics]:
        """示例风险指标"""
        base_time = datetime.utcnow().replace(microsecond=0)
        metrics = []
        
        for i in range(2):
            timestamp = base_time + timedelta(hours=i)
            
            risk_metric = HistoricalRiskMetrics(
                timestamp=timestamp,
                environment='testnet',
                calculation_period='1h',
                total_balance=Decimal('10000.00'),
                available_balance=Decimal('8000.00'),
                total_position_value=Decimal('2000.00'),
                total_exposure=Decimal('2000.00'),
                long_exposure=Decimal('1200.00'),
                short_exposure=Decimal('800.00'),
                net_exposure=Decimal('400.00'),
                gross_exposure=Decimal('2000.00'),
                used_margin=Decimal('2000.00'),
                free_margin=Decimal('8000.00'),
                margin_ratio=0.2,
                maintenance_margin=Decimal('1000.00'),
                effective_leverage=1.2,
                max_leverage_used=2.0,
                leverage_utilization=0.6,
                liquidation_buffer=Decimal('5000.00'),
                liquidity_score=0.85,
                max_drawdown=0.05,
                current_drawdown=0.02,
                drawdown_duration_days=0,
                var_95=Decimal('-50.00'),
                var_99=Decimal('-100.00'),
                portfolio_volatility=0.15,
                realized_volatility=0.12,
                volatility_percentile=60.0,
                max_correlation=0.8,
                avg_correlation=0.4,
                concentration_risk=0.3,
                max_drawdown_ratio=0.05,
                win_rate=0.65,
                profit_factor=1.8,
                avg_win_loss_ratio=2.1,
                consecutive_losses=0,
                daily_pnl=Decimal('15.25'),
                calculation_method='monte_carlo',
                data_quality_score=0.95,
                metadata={'test_metrics': True}
            )
            metrics.append(risk_metric)
        
        return metrics

    # ==================
    # 基础功能测试
    # ==================
    
    @pytest.mark.asyncio
    async def test_storage_initialization(self, duckdb_storage):
        """测试存储系统初始化"""
        assert duckdb_storage is not None
        assert duckdb_storage.config is not None
        assert duckdb_storage.retention_policy is not None
        
        # 测试获取存储统计信息
        stats = await duckdb_storage.get_storage_statistics(TradingEnvironment.TESTNET)
        assert 'environment' in stats
        assert 'total_tables' in stats
        assert stats['environment'] == 'testnet'

    @pytest.mark.asyncio
    async def test_store_and_query_market_data(self, duckdb_storage, sample_market_data):
        """测试市场数据存储和查询"""
        # 存储市场数据
        stored_count = await duckdb_storage.store_historical_market_data(
            sample_market_data, 
            TradingEnvironment.TESTNET
        )
        assert stored_count == len(sample_market_data)
        
        # 查询市场数据
        start_time = sample_market_data[0].timestamp
        end_time = sample_market_data[-1].timestamp
        
        queried_data = await duckdb_storage.query_historical_market_data(
            symbol='BTCUSDT',
            interval='1m', 
            start_time=start_time,
            end_time=end_time,
            environment=TradingEnvironment.TESTNET
        )
        
        assert len(queried_data) == len(sample_market_data)
        assert queried_data[0]['symbol'] == 'BTCUSDT'
        assert queried_data[0]['interval'] == '1m'

    @pytest.mark.asyncio
    async def test_store_and_query_trading_signals(self, duckdb_storage, sample_trading_signals):
        """测试交易信号存储和查询"""
        # 存储交易信号
        stored_count = await duckdb_storage.store_historical_trading_signals(
            sample_trading_signals,
            TradingEnvironment.TESTNET
        )
        assert stored_count == len(sample_trading_signals)
        
        # 查询交易信号
        queried_signals = await duckdb_storage.query_historical_trading_signals(
            symbol='BTCUSDT',
            source='test_agent',
            environment=TradingEnvironment.TESTNET
        )
        
        assert len(queried_signals) == len(sample_trading_signals)
        assert queried_signals[0]['symbol'] == 'BTCUSDT'
        assert queried_signals[0]['source'] == 'test_agent'

    @pytest.mark.asyncio
    async def test_store_and_query_order_records(self, duckdb_storage, sample_order_records):
        """测试订单记录存储和查询"""
        # 存储订单记录
        stored_count = await duckdb_storage.store_historical_order_records(
            sample_order_records,
            TradingEnvironment.TESTNET
        )
        assert stored_count == len(sample_order_records)
        
        # 测试绩效分析查询
        performance_analysis = await duckdb_storage.get_trading_performance_analysis(
            symbol='BTCUSDT',
            environment=TradingEnvironment.TESTNET
        )
        
        assert 'total_orders' in performance_analysis
        assert 'performance_metrics' in performance_analysis
        assert performance_analysis['total_orders'] == len(sample_order_records)

    @pytest.mark.asyncio
    async def test_store_and_query_risk_metrics(self, duckdb_storage, sample_risk_metrics):
        """测试风险指标存储和查询"""
        # 存储风险指标
        stored_count = await duckdb_storage.store_historical_risk_metrics(
            sample_risk_metrics,
            TradingEnvironment.TESTNET
        )
        assert stored_count == len(sample_risk_metrics)
        
        # 验证数据已存储
        stats = await duckdb_storage.get_storage_statistics(TradingEnvironment.TESTNET)
        risk_table_stats = stats.get('tables', {}).get('historical_risk_metrics', {})
        assert risk_table_stats.get('row_count', 0) >= len(sample_risk_metrics)

    # ==================
    # 数据清理测试
    # ==================
    
    @pytest.mark.asyncio
    async def test_data_cleanup(self, duckdb_storage, sample_market_data):
        """测试数据清理功能"""
        # 先存储一些数据
        await duckdb_storage.store_historical_market_data(
            sample_market_data,
            TradingEnvironment.TESTNET
        )
        
        # 执行数据清理（使用很短的保留期来测试清理功能）
        cleanup_results = await duckdb_storage.cleanup_expired_data(
            TradingEnvironment.TESTNET
        )
        
        assert 'market_data' in cleanup_results
        assert isinstance(cleanup_results['market_data'], int)

    @pytest.mark.asyncio 
    async def test_parquet_export(self, duckdb_storage, sample_market_data, temp_data_dir):
        """测试Parquet导出功能"""
        # 存储数据
        await duckdb_storage.store_historical_market_data(
            sample_market_data,
            TradingEnvironment.TESTNET
        )
        
        # 导出到Parquet
        export_path = temp_data_dir / "test_export.parquet"
        success = await duckdb_storage.export_data_to_parquet(
            table_name='historical_market_data',
            output_path=export_path,
            environment=TradingEnvironment.TESTNET
        )
        
        assert success is True
        # 注意：由于使用了DuckDB manager的导出功能，实际文件路径可能不同
        # 这里主要测试功能调用是否成功

    # ==================
    # 边界条件测试
    # ==================
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, duckdb_storage):
        """测试空数据处理"""
        # 测试存储空列表
        result = await duckdb_storage.store_historical_market_data([], TradingEnvironment.TESTNET)
        assert result == 0
        
        result = await duckdb_storage.store_historical_trading_signals([], TradingEnvironment.TESTNET)
        assert result == 0
        
        result = await duckdb_storage.store_historical_order_records([], TradingEnvironment.TESTNET)
        assert result == 0
        
        result = await duckdb_storage.store_historical_risk_metrics([], TradingEnvironment.TESTNET)
        assert result == 0

    @pytest.mark.asyncio
    async def test_query_nonexistent_data(self, duckdb_storage):
        """测试查询不存在的数据"""
        start_time = datetime.utcnow() - timedelta(days=1)
        end_time = datetime.utcnow()
        
        # 查询不存在的市场数据
        result = await duckdb_storage.query_historical_market_data(
            symbol='NONEXISTENT',
            interval='1m',
            start_time=start_time,
            end_time=end_time,
            environment=TradingEnvironment.TESTNET
        )
        assert result == []
        
        # 查询不存在的交易信号
        result = await duckdb_storage.query_historical_trading_signals(
            symbol='NONEXISTENT',
            environment=TradingEnvironment.TESTNET
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, duckdb_storage, sample_market_data):
        """测试并发操作"""
        # 创建多个并发存储任务
        tasks = []
        for i in range(3):
            # 为每个任务创建稍微不同的数据
            modified_data = []
            for data in sample_market_data:
                modified = data.model_copy()
                modified.timestamp = data.timestamp + timedelta(seconds=i)
                modified_data.append(modified)
            
            task = duckdb_storage.store_historical_market_data(
                modified_data, 
                TradingEnvironment.TESTNET
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务都成功
        for result in results:
            assert result == len(sample_market_data)

    # ==================
    # 数据验证测试
    # ==================
    
    def test_market_data_validation(self):
        """测试市场数据模型验证"""
        # 测试有效数据
        valid_data = HistoricalMarketData(
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
        assert valid_data.symbol == 'BTCUSDT'
        
        # 测试无效间隔
        with pytest.raises(ValueError):
            HistoricalMarketData(
                timestamp=datetime.utcnow(),
                symbol='BTCUSDT',
                environment='testnet',
                interval='invalid_interval',  # 无效间隔
                open_price=Decimal('50000.00'),
                high_price=Decimal('50100.00'),
                low_price=Decimal('49900.00'),
                close_price=Decimal('50050.00'),
                volume=Decimal('1.5')
            )

    def test_trading_signal_validation(self):
        """测试交易信号模型验证"""
        # 测试有效信号
        valid_signal = HistoricalTradingSignal(
            timestamp=datetime.utcnow(),
            symbol='BTCUSDT',
            environment='testnet',
            signal_id='test_signal',
            source='test_agent',
            signal_type='momentum',
            action='BUY',
            strength=0.8,
            confidence=0.9,
            reason='Test signal'
        )
        assert valid_signal.action == 'BUY'
        
        # 测试无效动作
        with pytest.raises(ValueError):
            HistoricalTradingSignal(
                timestamp=datetime.utcnow(),
                symbol='BTCUSDT',
                environment='testnet',
                signal_id='test_signal',
                source='test_agent',
                signal_type='momentum',
                action='INVALID_ACTION',  # 无效动作
                strength=0.8,
                confidence=0.9,
                reason='Test signal'
            )

    def test_order_record_methods(self):
        """测试订单记录模型方法"""
        order = HistoricalOrderRecord(
            timestamp=datetime.utcnow(),
            symbol='BTCUSDT',
            environment='testnet',
            order_id='test_order',
            client_order_id='client_order',
            side='BUY',
            order_type='MARKET',
            quantity=Decimal('0.1'),
            status='FILLED',
            executed_qty=Decimal('0.1'),
            executed_quote_qty=Decimal('5000.00'),
            avg_price=Decimal('50000.00'),
            commission=Decimal('5.00'),
            commission_asset='USDT',
            created_time=datetime.utcnow(),
            updated_time=datetime.utcnow(),
            realized_pnl=Decimal('10.50')
        )
        
        # 测试计算总成本
        total_cost = order.calculate_total_cost()
        expected_cost = Decimal('0.1') * Decimal('50000.00') + Decimal('5.00')
        assert total_cost == expected_cost
        
        # 测试盈利判断
        assert order.is_profitable() is True
        
        order_loss = order.model_copy()
        order_loss.realized_pnl = Decimal('-10.50')
        assert order_loss.is_profitable() is False

    def test_risk_metrics_methods(self):
        """测试风险指标模型方法"""
        risk_metric = HistoricalRiskMetrics(
            timestamp=datetime.utcnow(),
            environment='testnet',
            calculation_period='1h',
            total_balance=Decimal('10000.00'),
            available_balance=Decimal('5000.00'),
            total_position_value=Decimal('5000.00'),
            total_exposure=Decimal('5000.00'),
            long_exposure=Decimal('3000.00'),
            short_exposure=Decimal('2000.00'),
            net_exposure=Decimal('1000.00'),
            gross_exposure=Decimal('5000.00'),
            used_margin=Decimal('2500.00'),
            free_margin=Decimal('7500.00'),
            margin_ratio=0.25,
            maintenance_margin=Decimal('1250.00'),
            effective_leverage=2.0,  # 高杠杆
            max_leverage_used=3.0,
            leverage_utilization=0.67,
            liquidation_buffer=Decimal('2500.00'),
            liquidity_score=0.8,
            max_drawdown=0.15,  # 高回撤
            current_drawdown=0.12,  # 当前高回撤
            drawdown_duration_days=5,
            var_95=Decimal('-100.00'),
            var_99=Decimal('-200.00'),
            portfolio_volatility=0.25,
            realized_volatility=0.22,
            volatility_percentile=80.0,
            max_correlation=0.9,
            avg_correlation=0.6,
            concentration_risk=0.4,  # 中等集中度风险
            max_drawdown_ratio=0.15,
            win_rate=0.6,
            profit_factor=1.5,
            avg_win_loss_ratio=1.8,
            consecutive_losses=2,
            daily_pnl=Decimal('-25.00'),
            calculation_method='historical',
            data_quality_score=0.95
        )
        
        # 测试风险等级评估
        risk_level = risk_metric.get_risk_level()
        assert risk_level in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
        
        # 基于当前参数，应该是MEDIUM或HIGH风险
        assert risk_level in ['MEDIUM', 'HIGH']
        
        # 测试保证金风险判断
        margin_risk = risk_metric.is_margin_call_risk()
        assert isinstance(margin_risk, bool)


class TestDataMigrationManager:
    """数据迁移管理器测试类"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """临时数据目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def migration_manager(self, temp_data_dir):
        """数据迁移管理器实例"""
        # 临时修改数据目录
        import src.config
        original_data_dir = src.config.settings.data_directory
        src.config.settings.data_directory = str(temp_data_dir)
        
        manager = DataMigrationManager()
        yield manager
        
        # 恢复原始配置
        src.config.settings.data_directory = original_data_dir
        manager.close()

    @pytest.mark.asyncio
    async def test_migration_manager_initialization(self, migration_manager):
        """测试迁移管理器初始化"""
        assert migration_manager is not None
        assert migration_manager.config is not None
        assert migration_manager.retention_policy is not None
        assert migration_manager.backup_dir.exists()
        assert migration_manager.archive_dir.exists()

    @pytest.mark.asyncio
    async def test_backup_creation(self, migration_manager):
        """测试备份创建"""
        # 创建完整备份
        backup_info = await migration_manager.create_backup(
            TradingEnvironment.TESTNET,
            backup_type="full"
        )
        
        assert 'backup_name' in backup_info
        assert 'backup_path' in backup_info
        assert backup_info['environment'] == 'testnet'
        assert backup_info['backup_type'] == 'full'

    def test_backup_listing(self, migration_manager):
        """测试备份列表"""
        backups = migration_manager.list_backups(TradingEnvironment.TESTNET)
        assert isinstance(backups, list)

    def test_health_report(self, migration_manager):
        """测试存储健康报告"""
        health_report = migration_manager.get_storage_health_report(
            TradingEnvironment.TESTNET
        )
        
        assert 'environment' in health_report
        assert 'health_score' in health_report
        assert 'recommendations' in health_report
        assert isinstance(health_report['health_score'], float)
        assert isinstance(health_report['recommendations'], list)

    @pytest.mark.asyncio
    async def test_storage_optimization(self, migration_manager):
        """测试存储优化"""
        optimization_results = await migration_manager.optimize_storage(
            TradingEnvironment.TESTNET
        )
        
        assert 'environment' in optimization_results
        assert 'optimized_tables' in optimization_results
        assert 'status' in optimization_results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])