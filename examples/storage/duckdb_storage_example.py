#!/usr/bin/env python3
"""
DuckDB存储系统使用示例
演示历史数据存储、查询和分析功能的完整用法
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.core.storage.models import (
    HistoricalMarketData,
    HistoricalTradingSignal,
    HistoricalOrderRecord,
    HistoricalRiskMetrics,
    StorageConfig,
    DataRetentionPolicy
)
from src.core.storage.duckdb_storage import DuckDBStorage
from src.core.storage.data_migration import DataMigrationManager

# 使用枚举定义
from enum import Enum
class TradingEnvironment(Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet" 
    PAPER = "paper"


class DuckDBStorageExample:
    """DuckDB存储系统使用示例"""
    
    def __init__(self):
        # 配置存储系统
        self.storage_config = StorageConfig(
            max_memory_gb=4.0,
            thread_count=4,
            enable_compression=True,
            enable_partitioning=True,
            auto_optimize=True
        )
        
        # 配置数据保留策略
        self.retention_policy = DataRetentionPolicy(
            market_data_retention_days=365,  # 市场数据保留1年
            signal_retention_days=180,       # 信号保留半年
            order_retention_days=730,        # 订单保留2年
            risk_metrics_retention_days=365, # 风险指标保留1年
            enable_auto_cleanup=True,
            cleanup_interval_hours=24
        )
        
        # 初始化存储系统
        self.storage = DuckDBStorage(
            config=self.storage_config,
            retention_policy=self.retention_policy
        )
        
        # 初始化数据迁移管理器
        self.migration_manager = DataMigrationManager(
            config=self.storage_config,
            retention_policy=self.retention_policy
        )
        
        print("DuckDB存储系统初始化完成")

    def generate_sample_market_data(self, count: int = 100) -> List[HistoricalMarketData]:
        """生成示例市场数据"""
        print(f"生成{count}条示例市场数据...")
        
        base_time = datetime.utcnow() - timedelta(hours=count)
        base_price = Decimal('50000.00')
        data = []
        
        for i in range(count):
            timestamp = base_time + timedelta(minutes=i)
            # 模拟价格波动
            price_change = Decimal(str((i % 20 - 10) * 10))  # -100 到 +100 的变化
            current_price = base_price + price_change
            
            # 创建OHLCV数据
            open_price = current_price
            high_price = current_price + Decimal('50.00')
            low_price = current_price - Decimal('30.00')
            close_price = current_price + Decimal('10.00')
            
            market_data = HistoricalMarketData(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                interval='1m',
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=Decimal(str(1.0 + i * 0.01)),
                quote_volume=close_price * Decimal(str(1.0 + i * 0.01)),
                trade_count=50 + i,
                taker_buy_volume=Decimal(str(0.6 + i * 0.005)),
                bid_price=close_price - Decimal('5.00'),
                ask_price=close_price + Decimal('5.00'),
                bid_volume=Decimal('2.5'),
                ask_volume=Decimal('3.2'),
                data_source='binance',
                data_quality=0.95 + (i % 10) * 0.005,
                metadata={
                    'generated': True,
                    'batch': i // 10,
                    'market_session': 'asian' if i % 24 < 8 else 'european' if i % 24 < 16 else 'american'
                }
            )
            data.append(market_data)
        
        return data

    def generate_sample_trading_signals(self, count: int = 50) -> List[HistoricalTradingSignal]:
        """生成示例交易信号"""
        print(f"生成{count}条示例交易信号...")
        
        base_time = datetime.utcnow() - timedelta(hours=count * 2)
        signals = []
        
        signal_sources = ['momentum_agent', 'mean_reversion_agent', 'arbitrage_agent', 'ml_agent']
        signal_types = ['momentum', 'reversal', 'breakout', 'arbitrage']
        
        for i in range(count):
            timestamp = base_time + timedelta(minutes=i * 30)
            source = signal_sources[i % len(signal_sources)]
            signal_type = signal_types[i % len(signal_types)]
            
            # 模拟信号强度和置信度
            strength = 0.3 + (i % 7) * 0.1  # 0.3 到 0.9
            if i % 3 == 0:  # 部分信号为卖出信号
                strength = -strength
            
            confidence = 0.6 + (i % 4) * 0.1  # 0.6 到 0.9
            
            signal = HistoricalTradingSignal(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                signal_id=f'{source}_signal_{i}',
                source=source,
                signal_type=signal_type,
                action='BUY' if strength > 0 else 'SELL',
                strength=strength,
                confidence=confidence,
                target_price=Decimal('50000.00') + Decimal(str(i * 10)),
                suggested_quantity=Decimal('0.1') + Decimal(str(i * 0.001)),
                stop_loss=Decimal('49500.00') + Decimal(str(i * 8)),
                take_profit=Decimal('51000.00') + Decimal(str(i * 12)),
                max_drawdown_pct=0.02 + (i % 3) * 0.01,
                validity_period=timedelta(hours=1 + i % 4),
                technical_context={
                    'rsi': 30 + i % 40,
                    'macd': (i % 20 - 10) * 0.1,
                    'volume_ratio': 1.0 + (i % 10) * 0.1,
                    'support_level': 49000 + i * 5,
                    'resistance_level': 51000 + i * 5
                },
                market_condition='bull' if i % 3 == 0 else 'bear' if i % 3 == 1 else 'sideways',
                execution_status='executed' if i % 4 != 0 else 'pending',
                execution_price=Decimal('50050.00') + Decimal(str(i * 8)) if i % 4 != 0 else None,
                realized_pnl=Decimal(str((i % 20 - 10) * 2.5)) if i % 4 != 0 else None,
                reason=f'{signal_type} signal based on technical analysis - batch {i//10}',
                metadata={
                    'confidence_details': {
                        'technical_score': confidence * 0.7,
                        'volume_score': confidence * 0.3,
                        'market_regime_score': confidence * 0.5
                    },
                    'related_signals': [f'signal_{i-1}', f'signal_{i+1}'] if i % 5 == 0 else []
                }
            )
            signals.append(signal)
        
        return signals

    def generate_sample_order_records(self, count: int = 30) -> List[HistoricalOrderRecord]:
        """生成示例订单记录"""
        print(f"生成{count}条示例订单记录...")
        
        base_time = datetime.utcnow() - timedelta(hours=count * 3)
        orders = []
        
        order_types = ['MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT']
        
        for i in range(count):
            timestamp = base_time + timedelta(hours=i * 2)
            order_type = order_types[i % len(order_types)]
            side = 'BUY' if i % 2 == 0 else 'SELL'
            
            quantity = Decimal('0.1') + Decimal(str(i * 0.01))
            price = Decimal('50000.00') + Decimal(str(i * 50))
            
            order = HistoricalOrderRecord(
                timestamp=timestamp,
                symbol='BTCUSDT',
                environment='testnet',
                order_id=f'order_{i}',
                client_order_id=f'client_order_{i}',
                strategy_id=f'strategy_{i % 3}',
                signal_id=f'momentum_agent_signal_{i//2}' if i % 2 == 0 else None,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price if order_type != 'MARKET' else None,
                status='FILLED' if i % 5 != 0 else 'CANCELLED',
                executed_qty=quantity if i % 5 != 0 else Decimal('0'),
                executed_quote_qty=quantity * price if i % 5 != 0 else Decimal('0'),
                avg_price=price + Decimal(str((i % 10 - 5))) if i % 5 != 0 else Decimal('0'),
                commission=quantity * price * Decimal('0.001') if i % 5 != 0 else Decimal('0'),
                commission_asset='USDT',
                commission_rate=0.001,
                created_time=timestamp,
                updated_time=timestamp + timedelta(seconds=30 + i % 60),
                filled_time=timestamp + timedelta(seconds=45 + i % 90) if i % 5 != 0 else None,
                time_in_force='GTC' if order_type == 'LIMIT' else 'IOC',
                realized_pnl=Decimal(str((i % 20 - 10) * 5)) if i % 5 != 0 else None,
                pnl_percentage=(i % 20 - 10) * 0.1 if i % 5 != 0 else None,
                slippage=Decimal(str((i % 5) * 0.5)) if order_type == 'MARKET' and i % 5 != 0 else None,
                slippage_bps=(i % 5) * 0.5 if order_type == 'MARKET' and i % 5 != 0 else None,
                market_price_at_order=price + Decimal(str((i % 6 - 3))),
                spread_at_order=Decimal('10.00') + Decimal(str(i % 10)),
                leverage=2.0 + (i % 3),
                margin_used=quantity * price / Decimal(str(2.0 + (i % 3))) if i % 5 != 0 else None,
                order_source='algorithm' if i % 4 == 0 else 'manual',
                execution_algorithm='twap' if i % 6 == 0 else 'vwap' if i % 6 == 1 else None,
                metadata={
                    'order_context': {
                        'portfolio_allocation': (i % 10) * 0.1,
                        'risk_budget': (i % 5) * 0.02,
                        'market_impact_estimate': (i % 3) * 0.001
                    },
                    'execution_details': {
                        'venue': 'binance',
                        'latency_ms': 50 + i % 100,
                        'retry_count': i % 3
                    }
                }
            )
            orders.append(order)
        
        return orders

    def generate_sample_risk_metrics(self, count: int = 24) -> List[HistoricalRiskMetrics]:
        """生成示例风险指标"""
        print(f"生成{count}条示例风险指标...")
        
        base_time = datetime.utcnow() - timedelta(hours=count)
        metrics = []
        
        for i in range(count):
            timestamp = base_time + timedelta(hours=i)
            
            # 模拟账户余额变化
            balance_change = (i % 10 - 5) * 100
            total_balance = Decimal('10000.00') + Decimal(str(balance_change))
            
            # 模拟风险指标变化
            leverage = 1.5 + (i % 8) * 0.2
            drawdown = max(0.01, 0.05 + (i % 15) * 0.01)
            
            risk_metric = HistoricalRiskMetrics(
                timestamp=timestamp,
                environment='testnet',
                calculation_period='1h',
                account_id=f'account_{i % 3}',
                total_balance=total_balance,
                available_balance=total_balance * Decimal('0.8'),
                total_position_value=total_balance * Decimal('0.2'),
                total_exposure=total_balance * Decimal(str(leverage * 0.2)),
                long_exposure=total_balance * Decimal(str(leverage * 0.12)),
                short_exposure=total_balance * Decimal(str(leverage * 0.08)),
                net_exposure=total_balance * Decimal(str(leverage * 0.04)),
                gross_exposure=total_balance * Decimal(str(leverage * 0.2)),
                used_margin=total_balance * Decimal(str(0.2 / leverage)),
                free_margin=total_balance - total_balance * Decimal(str(0.2 / leverage)),
                margin_ratio=0.2 / leverage,
                maintenance_margin=total_balance * Decimal(str(0.1 / leverage)),
                effective_leverage=leverage,
                max_leverage_used=leverage * 1.2,
                leverage_utilization=leverage / 5.0,
                liquidation_buffer=total_balance * Decimal('0.3'),
                time_to_liquidation_hours=24.0 + i * 2,
                liquidity_score=0.7 + (i % 6) * 0.05,
                max_drawdown=0.1 + (i % 10) * 0.01,
                current_drawdown=drawdown,
                drawdown_duration_days=max(0, i % 7),
                recovery_factor=2.0 + (i % 5) * 0.3,
                var_95=total_balance * Decimal('-0.02') * Decimal(str(1 + i % 5 * 0.1)),
                var_99=total_balance * Decimal('-0.04') * Decimal(str(1 + i % 5 * 0.1)),
                expected_shortfall_95=total_balance * Decimal('-0.025') * Decimal(str(1 + i % 5 * 0.1)),
                expected_shortfall_99=total_balance * Decimal('-0.05') * Decimal(str(1 + i % 5 * 0.1)),
                portfolio_volatility=0.15 + (i % 8) * 0.01,
                realized_volatility=0.12 + (i % 6) * 0.01,
                implied_volatility=0.18 + (i % 7) * 0.015,
                volatility_percentile=50.0 + (i % 10 - 5) * 3,
                max_correlation=0.5 + (i % 10) * 0.04,
                avg_correlation=0.2 + (i % 8) * 0.03,
                concentration_risk=0.2 + (i % 6) * 0.05,
                sharpe_ratio=1.0 + (i % 20 - 10) * 0.1,
                sortino_ratio=1.2 + (i % 15 - 7) * 0.1,
                calmar_ratio=0.8 + (i % 12 - 6) * 0.1,
                max_drawdown_ratio=drawdown,
                win_rate=0.55 + (i % 10) * 0.03,
                profit_factor=1.2 + (i % 15) * 0.1,
                avg_win_loss_ratio=1.5 + (i % 8) * 0.2,
                consecutive_losses=max(0, i % 7 - 3),
                market_impact=0.001 + (i % 5) * 0.0005,
                bid_ask_spread=Decimal('10.00') + Decimal(str(i % 8)),
                slippage_cost=total_balance * Decimal('0.0002') * Decimal(str(1 + i % 4 * 0.5)),
                funding_cost=total_balance * Decimal('0.0001') * Decimal(str(i % 3)),
                daily_pnl=total_balance * Decimal(str((i % 20 - 10) * 0.002)),
                weekly_pnl=total_balance * Decimal(str((i % 14 - 7) * 0.01)),
                monthly_pnl=total_balance * Decimal(str((i % 8 - 4) * 0.03)),
                ytd_pnl=total_balance * Decimal(str((i % 6 - 3) * 0.1)),
                symbol_risks={
                    'BTCUSDT': {'var': float(-total_balance * Decimal('0.01')), 'correlation': 0.8},
                    'ETHUSDT': {'var': float(-total_balance * Decimal('0.008')), 'correlation': 0.6}
                },
                sector_risks={
                    'crypto': {'exposure': float(total_balance * Decimal('0.8')), 'beta': 1.2}
                },
                stress_test_results={
                    'market_crash_2008': {'pnl': float(-total_balance * Decimal('0.15'))},
                    'covid_crash_2020': {'pnl': float(-total_balance * Decimal('0.12'))},
                    'flash_crash': {'pnl': float(-total_balance * Decimal('0.08'))}
                },
                scenario_analysis={
                    'bull_market': {'expected_return': 0.25, 'max_drawdown': 0.08},
                    'bear_market': {'expected_return': -0.15, 'max_drawdown': 0.25},
                    'sideways': {'expected_return': 0.02, 'max_drawdown': 0.12}
                },
                model_confidence=0.85 + (i % 10) * 0.01,
                prediction_accuracy=0.72 + (i % 8) * 0.02,
                backtest_performance={
                    'total_return': (i % 20 - 10) * 0.02,
                    'sharpe': 1.0 + (i % 10 - 5) * 0.1,
                    'max_dd': -0.05 - (i % 8) * 0.01
                },
                calculation_method='monte_carlo' if i % 2 == 0 else 'historical',
                data_quality_score=0.9 + (i % 10) * 0.008,
                metadata={
                    'model_version': f'v{1 + i % 3}.{i % 10}',
                    'calculation_time_ms': 500 + i * 10,
                    'data_points_used': 1000 + i * 50,
                    'market_regime': 'trending' if i % 3 == 0 else 'mean_reverting'
                }
            )
            metrics.append(risk_metric)
        
        return metrics

    async def run_storage_examples(self):
        """运行存储系统示例"""
        print("\n=== DuckDB存储系统示例演示 ===\n")
        
        try:
            # 1. 生成和存储示例数据
            print("1. 生成和存储示例数据")
            await self._demo_data_storage()
            
            # 2. 数据查询示例
            print("\n2. 数据查询示例")
            await self._demo_data_queries()
            
            # 3. 分析查询示例
            print("\n3. 分析查询示例")
            await self._demo_analytical_queries()
            
            # 4. 数据生命周期管理
            print("\n4. 数据生命周期管理")
            await self._demo_lifecycle_management()
            
            # 5. 备份和恢复
            print("\n5. 备份和恢复")
            await self._demo_backup_restore()
            
            # 6. 存储统计和健康检查
            print("\n6. 存储统计和健康检查")
            await self._demo_storage_monitoring()
            
        except Exception as e:
            print(f"示例运行出错: {e}")
            raise
        finally:
            # 清理资源
            self.storage.close()
            self.migration_manager.close()
            print("\n示例演示完成，资源已清理")

    async def _demo_data_storage(self):
        """演示数据存储功能"""
        print("正在存储示例数据...")
        
        # 生成示例数据
        market_data = self.generate_sample_market_data(100)
        trading_signals = self.generate_sample_trading_signals(50)
        order_records = self.generate_sample_order_records(30)
        risk_metrics = self.generate_sample_risk_metrics(24)
        
        # 并行存储数据
        tasks = [
            self.storage.store_historical_market_data(market_data, TradingEnvironment.TESTNET),
            self.storage.store_historical_trading_signals(trading_signals, TradingEnvironment.TESTNET),
            self.storage.store_historical_order_records(order_records, TradingEnvironment.TESTNET),
            self.storage.store_historical_risk_metrics(risk_metrics, TradingEnvironment.TESTNET)
        ]
        
        results = await asyncio.gather(*tasks)
        
        print(f"✓ 存储了 {results[0]} 条市场数据")
        print(f"✓ 存储了 {results[1]} 条交易信号")
        print(f"✓ 存储了 {results[2]} 条订单记录") 
        print(f"✓ 存储了 {results[3]} 条风险指标")

    async def _demo_data_queries(self):
        """演示数据查询功能"""
        print("演示数据查询功能...")
        
        # 查询最近1小时的市场数据
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        market_data = await self.storage.query_historical_market_data(
            symbol='BTCUSDT',
            interval='1m',
            start_time=start_time,
            end_time=end_time,
            environment=TradingEnvironment.TESTNET,
            limit=10
        )
        
        print(f"✓ 查询到 {len(market_data)} 条最近1小时的市场数据")
        if market_data:
            latest = market_data[-1]
            print(f"  最新价格: {latest['close_price']}, 成交量: {latest['volume']}")
        
        # 查询特定来源的交易信号
        signals = await self.storage.query_historical_trading_signals(
            symbol='BTCUSDT',
            source='momentum_agent',
            environment=TradingEnvironment.TESTNET,
            limit=5
        )
        
        print(f"✓ 查询到 {len(signals)} 条momentum_agent的交易信号")
        if signals:
            latest_signal = signals[0]
            print(f"  最新信号: {latest_signal['action']}, 强度: {latest_signal['strength']}")

    async def _demo_analytical_queries(self):
        """演示分析查询功能"""
        print("演示分析查询功能...")
        
        # 获取交易绩效分析
        performance = await self.storage.get_trading_performance_analysis(
            symbol='BTCUSDT',
            environment=TradingEnvironment.TESTNET
        )
        
        print("✓ 交易绩效分析:")
        if 'performance_metrics' in performance:
            metrics = performance['performance_metrics']
            print(f"  总订单: {performance['total_orders']}")
            print(f"  胜率: {metrics.get('win_rate', 0):.2%}")
            print(f"  总盈亏: {metrics.get('total_pnl', 0):.2f}")
            print(f"  盈利因子: {metrics.get('profit_factor', 0):.2f}")
        
        # 获取存储统计信息
        stats = await self.storage.get_storage_statistics(TradingEnvironment.TESTNET)
        
        print("✓ 存储统计:")
        print(f"  总表数: {stats.get('total_tables', 0)}")
        print(f"  总记录数: {stats.get('total_rows', 0)}")
        print(f"  数据库大小: {stats.get('file_size_mb', 0):.2f} MB")

    async def _demo_lifecycle_management(self):
        """演示数据生命周期管理"""
        print("演示数据生命周期管理...")
        
        # 清理过期数据
        cleanup_results = await self.storage.cleanup_expired_data(
            TradingEnvironment.TESTNET
        )
        
        print("✓ 数据清理结果:")
        for table, count in cleanup_results.items():
            print(f"  {table}: {count} 条记录")
        
        # 存储优化
        optimization_results = await self.migration_manager.optimize_storage(
            TradingEnvironment.TESTNET
        )
        
        print("✓ 存储优化结果:")
        print(f"  状态: {optimization_results.get('status')}")
        optimized_tables = optimization_results.get('optimized_tables', {})
        for table, status in optimized_tables.items():
            print(f"  {table}: {status}")

    async def _demo_backup_restore(self):
        """演示备份和恢复功能"""
        print("演示备份和恢复功能...")
        
        # 创建增量备份
        backup_info = await self.migration_manager.create_backup(
            TradingEnvironment.TESTNET,
            backup_type="incremental"
        )
        
        print("✓ 备份创建结果:")
        print(f"  备份名称: {backup_info.get('backup_name')}")
        print(f"  备份大小: {backup_info.get('total_size_mb', 0):.2f} MB")
        print(f"  状态: {backup_info.get('status')}")
        
        # 列出所有备份
        backups = self.migration_manager.list_backups(TradingEnvironment.TESTNET)
        print(f"✓ 共有 {len(backups)} 个备份")
        
        # 清理旧备份
        cleanup_results = self.migration_manager.cleanup_old_backups(
            TradingEnvironment.TESTNET,
            keep_days=7  # 保留7天内的备份
        )
        
        print("✓ 备份清理结果:")
        print(f"  删除备份: {len(cleanup_results.get('deleted_backups', []))}")
        print(f"  释放空间: {cleanup_results.get('space_freed_mb', 0):.2f} MB")

    async def _demo_storage_monitoring(self):
        """演示存储监控功能"""
        print("演示存储监控功能...")
        
        # 获取存储健康报告
        health_report = self.migration_manager.get_storage_health_report(
            TradingEnvironment.TESTNET
        )
        
        print("✓ 存储健康报告:")
        print(f"  健康评分: {health_report.get('health_score', 0):.1f}/100")
        
        database_stats = health_report.get('database_stats', {})
        print(f"  总表数: {database_stats.get('total_tables', 0)}")
        print(f"  总行数: {database_stats.get('total_rows', 0)}")
        print(f"  文件大小: {database_stats.get('file_size_mb', 0):.2f} MB")
        
        backup_info = health_report.get('backup_info', {})
        print(f"  备份数量: {backup_info.get('total_backups', 0)}")
        
        recommendations = health_report.get('recommendations', [])
        if recommendations:
            print("  建议:")
            for rec in recommendations:
                print(f"    - {rec}")
        
        # 显示详细表统计
        print("\n✓ 详细表统计:")
        tables = database_stats.get('tables', {})
        for table_name, table_stats in tables.items():
            if 'row_count' in table_stats:
                print(f"  {table_name}: {table_stats['row_count']} 行")

    def create_analysis_report(self):
        """创建分析报告"""
        print("\n=== 生成DuckDB存储分析报告 ===")
        
        report = {
            "storage_system": "DuckDB",
            "features": {
                "analytical_queries": "支持复杂OLAP查询和聚合分析",
                "compression": "内置列式存储和压缩优化",
                "partitioning": "支持时间分区以提高查询性能",
                "concurrent_access": "支持多线程并发读写",
                "data_export": "支持Parquet格式导出和导入",
                "backup_restore": "完整和增量备份恢复功能"
            },
            "data_models": {
                "market_data": "历史OHLCV数据，支持多时间框架",
                "trading_signals": "交易信号历史，包含执行状态和绩效",
                "order_records": "完整订单生命周期记录",
                "risk_metrics": "综合风险指标和绩效统计"
            },
            "performance_optimization": {
                "indexing": "基于时间戳和交易品种的复合索引",
                "memory_management": "可配置内存限制和缓存策略",
                "query_optimization": "支持查询计划优化和执行统计",
                "storage_optimization": "自动表重组和存储压缩"
            },
            "lifecycle_management": {
                "data_retention": "基于策略的自动数据清理",
                "archiving": "历史数据归档到Parquet格式",
                "migration": "内存到历史数据的自动迁移",
                "monitoring": "存储健康监控和报告"
            }
        }
        
        print("报告已生成，主要特性:")
        for category, features in report.items():
            if isinstance(features, dict):
                print(f"\n{category.replace('_', ' ').title()}:")
                for key, value in features.items():
                    print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        return report


async def main():
    """主函数 - 运行所有示例"""
    example = DuckDBStorageExample()
    
    try:
        # 运行存储系统示例
        await example.run_storage_examples()
        
        # 生成分析报告
        example.create_analysis_report()
        
        print("\n=== 示例运行完成 ===")
        print("DuckDB存储系统提供了完整的历史数据管理功能:")
        print("• 高性能数据存储和查询")
        print("• 完整的数据生命周期管理")
        print("• 自动化备份和恢复")
        print("• 存储健康监控")
        print("• 与交易系统的无缝集成")
        
    except Exception as e:
        print(f"示例运行失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())