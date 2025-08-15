#!/usr/bin/env python3
"""
DuckDB时序数据管理器使用示例
演示高性能时序数据存储、查询和分析功能
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from src.core.duckdb_manager import duckdb_manager
from src.exchanges.trading_interface import TradingEnvironment


def generate_sample_tick_data(symbol: str, count: int = 1000) -> List[Dict[str, Any]]:
    """生成模拟tick数据"""
    base_price = 50000.0
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    tick_data = []
    for i in range(count):
        timestamp = base_time + timedelta(seconds=i * 3.6)  # 每3.6秒一个tick
        price = base_price + random.uniform(-1000, 1000)
        volume = random.uniform(0.1, 10.0)
        
        tick_data.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'bid_price': price - random.uniform(0.1, 1.0),
            'ask_price': price + random.uniform(0.1, 1.0),
            'bid_volume': random.uniform(1.0, 5.0),
            'ask_volume': random.uniform(1.0, 5.0),
            'trade_count': random.randint(1, 10)
        })
    
    return tick_data


def generate_sample_kline_data(symbol: str, interval: str, count: int = 100) -> List[Dict[str, Any]]:
    """生成模拟K线数据"""
    base_price = 50000.0
    base_time = datetime.utcnow() - timedelta(hours=count)
    
    kline_data = []
    for i in range(count):
        timestamp = base_time + timedelta(hours=i)
        open_price = base_price + random.uniform(-500, 500)
        close_price = open_price + random.uniform(-200, 200)
        high_price = max(open_price, close_price) + random.uniform(0, 100)
        low_price = min(open_price, close_price) - random.uniform(0, 100)
        volume = random.uniform(100, 1000)
        
        kline_data.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'interval': interval,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'volume': volume,
            'quote_volume': volume * close_price,
            'trade_count': random.randint(50, 500),
            'taker_buy_volume': volume * random.uniform(0.4, 0.6),
            'taker_buy_quote_volume': volume * close_price * random.uniform(0.4, 0.6)
        })
    
    return kline_data


def generate_sample_order_book_data(symbol: str) -> tuple:
    """生成模拟订单簿数据"""
    base_price = 50000.0
    
    # 生成买单
    bids = []
    for i in range(10):
        price = base_price - (i + 1) * random.uniform(0.5, 2.0)
        volume = random.uniform(0.1, 5.0)
        bids.append([price, volume])
    
    # 生成卖单
    asks = []
    for i in range(10):
        price = base_price + (i + 1) * random.uniform(0.5, 2.0)
        volume = random.uniform(0.1, 5.0)
        asks.append([price, volume])
    
    return bids, asks


def demo_environment_setup():
    """演示多环境数据库设置"""
    print("🏗️ 多环境数据库设置演示")
    print("=" * 50)
    
    # 初始化各环境的连接
    for env in TradingEnvironment:
        print(f"初始化 {env.value} 环境...")
        conn = duckdb_manager.get_connection(env)
        print(f"✅ {env.value} 环境连接已建立")
        
        # 获取数据库信息
        db_info = duckdb_manager.get_database_info(env)
        print(f"   数据库路径: {db_info['database_path']}")
        print(f"   表数量: {db_info['total_tables']}")
    
    print()


def demo_high_performance_insertion():
    """演示高性能数据插入"""
    print("⚡ 高性能数据插入演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    duckdb_manager.set_current_environment(env)
    
    # 生成大量测试数据
    print("生成测试数据...")
    tick_data = generate_sample_tick_data("BTCUSDT", 5000)
    kline_data = generate_sample_kline_data("BTCUSDT", "1h", 1000)
    
    # 批量插入tick数据
    print("批量插入tick数据...")
    start_time = datetime.now()
    inserted_ticks = duckdb_manager.insert_market_ticks_batch(tick_data)
    tick_duration = (datetime.now() - start_time).total_seconds()
    print(f"✅ 插入 {inserted_ticks} 条tick记录，耗时: {tick_duration:.2f}秒")
    print(f"   插入速度: {inserted_ticks/tick_duration:.0f} 记录/秒")
    
    # 批量插入K线数据
    print("批量插入K线数据...")
    start_time = datetime.now()
    inserted_klines = duckdb_manager.insert_klines_batch(kline_data)
    kline_duration = (datetime.now() - start_time).total_seconds()
    print(f"✅ 插入 {inserted_klines} 条K线记录，耗时: {kline_duration:.2f}秒")
    print(f"   插入速度: {inserted_klines/kline_duration:.0f} 记录/秒")
    
    # 插入订单簿快照
    print("插入订单簿快照...")
    bids, asks = generate_sample_order_book_data("BTCUSDT")
    snapshot_inserted = duckdb_manager.insert_order_book_snapshot(
        "BTCUSDT", bids, asks, last_update_id=12345
    )
    print(f"✅ 订单簿快照插入: {'成功' if snapshot_inserted else '失败'}")
    
    print()


def demo_time_series_queries():
    """演示时序数据查询"""
    print("📊 时序数据查询演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    duckdb_manager.set_current_environment(env)
    
    # 查询时间范围
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=30)
    
    # 查询tick数据
    print("查询tick数据...")
    tick_df = duckdb_manager.query_market_ticks("BTCUSDT", start_time, end_time)
    print(f"✅ 查询到 {len(tick_df)} 条tick记录")
    if not tick_df.empty:
        print(f"   价格范围: {tick_df['price'].min():.2f} - {tick_df['price'].max():.2f}")
        print(f"   时间范围: {tick_df['timestamp'].min()} 到 {tick_df['timestamp'].max()}")
    
    # 查询K线数据
    print("查询K线数据...")
    kline_df = duckdb_manager.query_klines("BTCUSDT", "1h", start_time, end_time)
    print(f"✅ 查询到 {len(kline_df)} 条K线记录")
    if not kline_df.empty:
        print(f"   最新收盘价: {kline_df.iloc[-1]['close_price']:.2f}")
    
    # 获取最新价格
    print("获取最新价格...")
    latest_prices = duckdb_manager.get_latest_prices(["BTCUSDT"])
    print(f"✅ 最新价格: {latest_prices}")
    
    print()


def demo_data_analysis():
    """演示数据分析功能"""
    print("📈 数据分析功能演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    duckdb_manager.set_current_environment(env)
    
    # 时间范围
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    # 计算价格统计
    print("计算价格统计...")
    price_stats = duckdb_manager.calculate_price_statistics("BTCUSDT", start_time, end_time)
    if price_stats:
        print(f"✅ 价格统计:")
        print(f"   交易次数: {price_stats['tick_count']}")
        print(f"   平均价格: {price_stats['avg_price']:.2f}")
        print(f"   价格标准差: {price_stats['price_stddev']:.2f}")
        print(f"   价格变化幅度: {price_stats['price_range_percent']:.2f}%")
        print(f"   总成交量: {price_stats['total_volume']:.2f}")
    
    # 计算VWAP
    print("计算VWAP...")
    vwap = duckdb_manager.get_volume_weighted_price("BTCUSDT", start_time, end_time)
    print(f"✅ VWAP (成交量加权平均价): {vwap:.2f}")
    
    # 获取OHLCV数据
    print("获取OHLCV聚合数据...")
    ohlcv_df = duckdb_manager.get_ohlcv_aggregation("BTCUSDT", "1h", 24)
    print(f"✅ 获取 {len(ohlcv_df)} 个时间段的OHLCV数据")
    
    # 市场深度分析
    print("市场深度分析...")
    depth_analysis = duckdb_manager.get_market_depth_analysis("BTCUSDT")
    if depth_analysis:
        print(f"✅ 市场深度分析:")
        print(f"   最佳买价: {depth_analysis['best_bid']:.2f}")
        print(f"   最佳卖价: {depth_analysis['best_ask']:.2f}")
        print(f"   买卖价差: {depth_analysis['spread']:.2f}")
        print(f"   价差百分比: {depth_analysis['spread_percent']:.3f}%")
        print(f"   订单簿不平衡度: {depth_analysis['imbalance']:.3f}")
    
    print()


def demo_data_compression_partitioning():
    """演示数据压缩和分区"""
    print("🗜️ 数据压缩和分区演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    duckdb_manager.set_current_environment(env)
    
    # 导出到Parquet格式
    export_path = Path("./data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    
    print("导出tick数据到Parquet...")
    parquet_file = export_path / "market_ticks.parquet"
    export_success = duckdb_manager.export_to_parquet(
        "market_ticks", 
        parquet_file,
        partition_by="symbol"
    )
    print(f"✅ Parquet导出: {'成功' if export_success else '失败'}")
    
    # 检查分区表创建
    print("创建时间分区表...")
    partition_success = duckdb_manager.create_time_partitioned_table(
        "market_ticks", 
        "daily"
    )
    print(f"✅ 分区表创建: {'成功' if partition_success else '失败'}")
    
    # 优化表存储
    print("优化表存储...")
    optimize_success = duckdb_manager.optimize_table_storage("market_ticks")
    print(f"✅ 存储优化: {'成功' if optimize_success else '失败'}")
    
    print()


def demo_environment_isolation():
    """演示环境数据隔离"""
    print("🔒 环境数据隔离演示")
    print("=" * 50)
    
    # 在不同环境中插入数据
    environments = [TradingEnvironment.TESTNET, TradingEnvironment.MAINNET, TradingEnvironment.PAPER]
    
    for env in environments:
        print(f"\n处理 {env.value} 环境:")
        duckdb_manager.set_current_environment(env)
        
        # 插入环境特定的测试数据
        tick_data = generate_sample_tick_data("ETHUSDT", 100)
        inserted = duckdb_manager.insert_market_ticks_batch(tick_data, env)
        print(f"  插入了 {inserted} 条记录到 {env.value}")
        
        # 查询该环境的数据
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=2)
        
        tick_df = duckdb_manager.query_market_ticks("ETHUSDT", start_time, end_time, env)
        print(f"  查询到 {len(tick_df)} 条记录来自 {env.value}")
    
    # 验证数据隔离
    print("\n验证数据隔离:")
    for env in environments:
        db_info = duckdb_manager.get_database_info(env)
        total_rows = sum(table_info.get('row_count', 0) for table_info in db_info['tables'].values())
        print(f"  {env.value}: {total_rows} 条总记录")
    
    print()


def demo_storage_statistics():
    """演示存储统计信息"""
    print("📊 存储统计信息演示")
    print("=" * 50)
    
    for env in TradingEnvironment:
        print(f"\n{env.value} 环境统计:")
        
        stats = duckdb_manager.get_storage_statistics(env)
        
        if 'error' not in stats:
            print(f"  数据库文件大小: {stats.get('file_size_mb', 0):.2f} MB")
            print(f"  总表数: {stats['total_tables']}")
            print(f"  总记录数: {stats['total_rows']}")
            
            if stats['total_rows'] > 0:
                bytes_per_row = stats['storage_efficiency'].get('bytes_per_row', 0)
                print(f"  平均每行字节数: {bytes_per_row}")
            
            print("  各表详情:")
            for table_name, table_info in stats['tables'].items():
                if 'error' not in table_info:
                    print(f"    {table_name}: {table_info['row_count']} 行")
                    if table_info.get('time_span_days'):
                        print(f"      数据时间跨度: {table_info['time_span_days']} 天")
        else:
            print(f"  错误: {stats['error']}")
    
    print()


def demo_data_cleanup():
    """演示数据清理功能"""
    print("🧹 数据清理功能演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    duckdb_manager.set_current_environment(env)
    
    # 清理7天前的旧数据
    print("清理7天前的旧数据...")
    
    # 先查看清理前的统计
    stats_before = duckdb_manager.get_storage_statistics(env)
    rows_before = stats_before.get('total_rows', 0)
    
    # 执行清理
    tables_to_clean = ['market_ticks', 'market_klines', 'order_book_snapshots']
    total_cleaned = 0
    
    for table in tables_to_clean:
        cleaned = duckdb_manager.cleanup_old_data(table, retention_days=7)
        total_cleaned += cleaned
        print(f"  {table}: 清理了 {cleaned} 条记录")
    
    # 查看清理后的统计
    stats_after = duckdb_manager.get_storage_statistics(env)
    rows_after = stats_after.get('total_rows', 0)
    
    print(f"✅ 总清理记录数: {total_cleaned}")
    print(f"   清理前总记录: {rows_before}")
    print(f"   清理后总记录: {rows_after}")
    
    print()


def main():
    """主函数"""
    print("🚀 DuckDB时序数据管理器演示程序")
    print("=" * 50)
    
    try:
        # 依次演示各项功能
        demo_environment_setup()
        demo_high_performance_insertion()
        demo_time_series_queries()
        demo_data_analysis()
        demo_data_compression_partitioning()
        demo_environment_isolation()
        demo_storage_statistics()
        demo_data_cleanup()
        
        print("✅ 所有演示完成")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        duckdb_manager.close_all()
        print("🔒 已关闭所有DuckDB连接")


if __name__ == "__main__":
    main()