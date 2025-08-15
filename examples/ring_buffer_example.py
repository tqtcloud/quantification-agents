#!/usr/bin/env python3
"""
环形缓冲区使用示例
演示高性能热数据管理、线程安全和数据过期功能
"""

import asyncio
import random
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import concurrent.futures

from src.core.ring_buffer import (
    RingBuffer,
    MultiChannelRingBuffer,
    create_market_data_buffer,
    create_tick_data_buffer,
    BufferItem
)


def demo_basic_operations():
    """演示基本操作"""
    print("🔄 基本操作演示")
    print("=" * 50)
    
    # 创建容量为5的缓冲区
    buffer = RingBuffer[str](capacity=5, auto_cleanup=False)
    
    # 添加数据
    print("添加数据到缓冲区...")
    for i in range(7):
        buffer.put(f"data_{i}")
        print(f"  添加: data_{i}, 当前大小: {len(buffer)}")
    
    print(f"\n缓冲区状态:")
    print(f"  容量: {buffer.capacity}")
    print(f"  当前大小: {len(buffer)}")
    print(f"  使用率: {buffer.usage_ratio():.1%}")
    print(f"  是否已满: {buffer.is_full()}")
    
    # 获取最新数据
    latest = buffer.get_latest(3)
    print(f"\n最新的3条数据:")
    for i, item in enumerate(latest):
        print(f"  {i+1}. {item.data} (时间戳: {item.timestamp:.3f})")
    
    # 查看最新数据（不移除）
    peek = buffer.peek_latest()
    print(f"\nPeek最新数据: {peek.data if peek else 'None'}")
    
    print()


def demo_time_based_operations():
    """演示基于时间的操作"""
    print("⏰ 基于时间的操作演示")
    print("=" * 50)
    
    buffer = RingBuffer[Dict[str, Any]](capacity=10, auto_cleanup=False)
    
    # 添加带时间戳的市场数据
    base_time = time.time()
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    
    print("添加市场数据...")
    for i in range(8):
        timestamp = base_time + i * 10  # 每10秒一条数据
        symbol = symbols[i % len(symbols)]
        price = 50000 + random.uniform(-1000, 1000)
        
        market_data = {
            "symbol": symbol,
            "price": price,
            "volume": random.uniform(0.1, 10.0),
            "timestamp": timestamp
        }
        
        buffer.put(market_data, timestamp=timestamp)
        print(f"  {symbol}: ${price:.2f} @ {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
    
    # 时间范围查询
    print(f"\n时间范围查询 (30-60秒):")
    range_data = buffer.get_range(base_time + 30, base_time + 60)
    for item in range_data:
        data = item.data
        time_str = datetime.fromtimestamp(item.timestamp).strftime('%H:%M:%S')
        print(f"  {data['symbol']}: ${data['price']:.2f} @ {time_str}")
    
    # 获取最近30秒的数据
    print(f"\n最近30秒的数据:")
    recent_data = buffer.get_since(base_time + 40)
    for item in recent_data:
        data = item.data
        time_str = datetime.fromtimestamp(item.timestamp).strftime('%H:%M:%S')
        print(f"  {data['symbol']}: ${data['price']:.2f} @ {time_str}")
    
    print()


def demo_data_expiration():
    """演示数据过期功能"""
    print("⏳ 数据过期功能演示")
    print("=" * 50)
    
    # 创建TTL为2秒的缓冲区
    buffer = RingBuffer[str](
        capacity=10, 
        ttl_seconds=2.0, 
        auto_cleanup=False
    )
    
    print("添加数据（TTL=2秒）...")
    buffer.put("data_1")
    print(f"  添加 data_1, 缓冲区大小: {len(buffer)}")
    
    time.sleep(1)
    buffer.put("data_2")
    print(f"  添加 data_2, 缓冲区大小: {len(buffer)}")
    
    time.sleep(1)
    buffer.put("data_3")
    print(f"  添加 data_3, 缓冲区大小: {len(buffer)}")
    
    print(f"\n等待1秒后检查过期情况...")
    time.sleep(1)
    
    # 手动清理过期数据
    cleaned = buffer.cleanup_expired()
    print(f"清理了 {cleaned} 条过期数据")
    print(f"清理后缓冲区大小: {len(buffer)}")
    
    # 查看剩余数据
    remaining = buffer.get_latest(10)
    print("剩余数据:")
    for item in remaining:
        age = time.time() - item.timestamp
        print(f"  {item.data} (年龄: {age:.1f}秒)")
    
    print()


def demo_auto_cleanup():
    """演示自动清理功能"""
    print("🧹 自动清理功能演示")
    print("=" * 50)
    
    # 创建自动清理的缓冲区
    buffer = RingBuffer[str](
        capacity=10,
        ttl_seconds=0.5,  # 0.5秒TTL
        auto_cleanup=True,
        cleanup_interval=0.2,  # 0.2秒清理一次
        cleanup_threshold=0.5  # 50%使用率触发清理
    )
    
    print("添加数据（自动清理开启）...")
    for i in range(8):
        buffer.put(f"data_{i}")
        print(f"  添加 data_{i}, 当前大小: {len(buffer)}")
        time.sleep(0.1)
    
    print(f"\n初始缓冲区大小: {len(buffer)}")
    
    # 等待自动清理工作
    print("等待自动清理...")
    time.sleep(1.0)
    
    print(f"清理后缓冲区大小: {len(buffer)}")
    
    # 获取统计信息
    stats = buffer.get_stats()
    print(f"\n统计信息:")
    print(f"  总插入: {stats['total_inserts']}")
    print(f"  总过期: {stats['total_expired']}")
    print(f"  总驱逐: {stats['total_evicted']}")
    
    buffer.close()
    print()


def demo_multi_channel_buffer():
    """演示多通道缓冲区"""
    print("📡 多通道缓冲区演示")
    print("=" * 50)
    
    # 创建多通道缓冲区
    buffer = MultiChannelRingBuffer(
        capacity_per_channel=5,
        ttl_seconds=60,
        max_channels=10
    )
    
    # 模拟不同交易对的数据
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
    
    print("添加多交易对数据...")
    for i in range(15):
        symbol = symbols[i % len(symbols)]
        
        tick_data = {
            "price": 50000 + random.uniform(-5000, 5000),
            "volume": random.uniform(0.1, 5.0),
            "bid": 49950 + random.uniform(-100, 100),
            "ask": 50050 + random.uniform(-100, 100),
            "sequence": i
        }
        
        buffer.put(symbol, tick_data)
        print(f"  {symbol}: ${tick_data['price']:.2f}")
    
    # 查看各通道状态
    channels = buffer.get_all_channels()
    print(f"\n活跃通道: {channels}")
    
    for channel in channels:
        latest = buffer.get_latest(channel, 3)
        print(f"\n{channel} 最新3条数据:")
        for i, item in enumerate(latest):
            data = item.data
            print(f"  {i+1}. ${data['price']:.2f} (序号: {data['sequence']})")
    
    # 获取总体统计
    total_stats = buffer.get_total_stats()
    print(f"\n总体统计:")
    print(f"  通道数: {total_stats['total_channels']}")
    print(f"  总数据量: {total_stats['total_size']}")
    print(f"  总容量: {total_stats['total_capacity']}")
    print(f"  整体使用率: {total_stats['overall_usage_ratio']:.1%}")
    
    buffer.close()
    print()


def demo_thread_safety():
    """演示线程安全"""
    print("🔒 线程安全演示")
    print("=" * 50)
    
    buffer = RingBuffer[Dict[str, Any]](capacity=1000, auto_cleanup=False)
    
    # 多线程写入数据
    num_threads = 5
    items_per_thread = 100
    
    def producer(thread_id: int):
        """生产者线程"""
        for i in range(items_per_thread):
            data = {
                "thread_id": thread_id,
                "sequence": i,
                "price": 50000 + random.uniform(-1000, 1000),
                "timestamp": time.time()
            }
            buffer.put(data)
            time.sleep(0.001)  # 模拟真实延迟
    
    def consumer():
        """消费者线程"""
        readings = []
        for _ in range(50):
            latest = buffer.get_latest(10)
            readings.append(len(latest))
            time.sleep(0.01)
        return readings
    
    print(f"启动 {num_threads} 个生产者线程和1个消费者线程...")
    
    # 启动线程
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads + 1) as executor:
        # 提交生产者任务
        producer_futures = [
            executor.submit(producer, i) for i in range(num_threads)
        ]
        
        # 提交消费者任务
        consumer_future = executor.submit(consumer)
        
        # 等待所有生产者完成
        concurrent.futures.wait(producer_futures)
        
        # 获取消费者结果
        readings = consumer_future.result()
    
    print(f"完成！")
    print(f"最终缓冲区大小: {len(buffer)}")
    print(f"总插入次数: {buffer.get_stats()['total_inserts']}")
    print(f"消费者读取样本: {readings[:10]}...")
    
    # 验证数据完整性
    all_data = buffer.get_latest(1000)
    thread_counts = {}
    for item in all_data:
        thread_id = item.data['thread_id']
        thread_counts[thread_id] = thread_counts.get(thread_id, 0) + 1
    
    print(f"各线程数据统计: {thread_counts}")
    print()


def demo_performance_benchmark():
    """演示性能基准测试"""
    print("⚡ 性能基准测试")
    print("=" * 50)
    
    # 测试高吞吐量写入
    buffer = RingBuffer[Dict[str, Any]](capacity=10000, auto_cleanup=False)
    
    print("测试高吞吐量写入性能...")
    num_items = 50000
    
    start_time = time.time()
    
    for i in range(num_items):
        market_data = {
            "symbol": "BTCUSDT",
            "price": 50000 + i * 0.01,
            "volume": 1.0 + i * 0.001,
            "bid": 49999,
            "ask": 50001,
            "sequence": i
        }
        buffer.put(market_data)
    
    end_time = time.time()
    duration = end_time - start_time
    throughput = num_items / duration
    
    print(f"✅ 写入性能:")
    print(f"  数据量: {num_items:,} 条")
    print(f"  耗时: {duration:.3f} 秒")
    print(f"  吞吐量: {throughput:,.0f} items/秒")
    print(f"  目标: >10,000 items/秒 ({'✅ 达标' if throughput > 10000 else '❌ 未达标'})")
    
    # 测试读取性能
    print(f"\n测试批量读取性能...")
    num_reads = 1000
    
    start_time = time.time()
    
    for _ in range(num_reads):
        _ = buffer.get_latest(100)
    
    end_time = time.time()
    read_duration = end_time - start_time
    read_ops_per_sec = num_reads / read_duration
    
    print(f"✅ 读取性能:")
    print(f"  操作数: {num_reads:,} 次")
    print(f"  耗时: {read_duration:.3f} 秒")
    print(f"  操作速度: {read_ops_per_sec:,.0f} ops/秒")
    
    print()


def demo_convenience_functions():
    """演示便利函数"""
    print("🛠️ 便利函数演示")
    print("=" * 50)
    
    # 创建市场数据缓冲区
    print("创建市场数据缓冲区...")
    market_buffer = create_market_data_buffer(
        capacity=1000,
        ttl_minutes=30,
        auto_cleanup=True
    )
    
    # 添加市场数据
    market_data = {
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 1.0,
        "bid": 49999.5,
        "ask": 50000.5,
        "timestamp": datetime.now().isoformat()
    }
    
    market_buffer.put(market_data)
    print(f"✅ 市场数据缓冲区: {len(market_buffer)} 条数据")
    
    # 创建tick数据缓冲区
    print("创建tick数据缓冲区...")
    tick_buffer = create_tick_data_buffer(
        capacity_per_symbol=500,
        ttl_minutes=15,
        max_symbols=50
    )
    
    # 添加多个交易对的tick数据
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    for symbol in symbols:
        tick_data = {
            "price": 50000 + random.uniform(-5000, 5000),
            "volume": random.uniform(0.1, 5.0),
            "timestamp": time.time()
        }
        tick_buffer.put(symbol, tick_data)
    
    channels = tick_buffer.get_all_channels()
    print(f"✅ Tick数据缓冲区: {len(channels)} 个交易对")
    
    for channel in channels:
        latest = tick_buffer.get_latest(channel, 1)
        if latest:
            price = latest[0].data['price']
            print(f"  {channel}: ${price:.2f}")
    
    # 清理资源
    market_buffer.close()
    tick_buffer.close()
    print()


def demo_real_world_simulation():
    """演示真实世界使用场景模拟"""
    print("🌍 真实世界使用场景模拟")
    print("=" * 50)
    
    # 创建多通道缓冲区模拟交易所数据
    exchange_buffer = MultiChannelRingBuffer(
        capacity_per_channel=1000,
        ttl_seconds=300,  # 5分钟TTL
        auto_cleanup=True,
        max_channels=20
    )
    
    # 模拟实时市场数据流
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    base_prices = {"BTCUSDT": 50000, "ETHUSDT": 3000, "ADAUSDT": 1.5, "DOTUSDT": 25, "LINKUSDT": 15}
    
    print("模拟实时市场数据流（运行10秒）...")
    
    def market_data_simulator():
        """市场数据模拟器"""
        start_time = time.time()
        sequence = 0
        
        while time.time() - start_time < 10:  # 运行10秒
            for symbol in symbols:
                # 模拟价格波动
                base_price = base_prices[symbol]
                price_change = random.uniform(-0.01, 0.01)  # ±1%波动
                new_price = base_price * (1 + price_change)
                
                tick_data = {
                    "symbol": symbol,
                    "price": new_price,
                    "volume": random.uniform(0.1, 10.0),
                    "bid": new_price * 0.9999,
                    "ask": new_price * 1.0001,
                    "timestamp": time.time(),
                    "sequence": sequence
                }
                
                exchange_buffer.put(symbol, tick_data)
                sequence += 1
            
            time.sleep(0.1)  # 100ms间隔
    
    # 运行模拟器
    simulator_thread = threading.Thread(target=market_data_simulator)
    simulator_thread.start()
    simulator_thread.join()
    
    # 分析结果
    stats = exchange_buffer.get_total_stats()
    print(f"\n模拟结果:")
    print(f"  活跃交易对: {stats['total_channels']}")
    print(f"  总数据量: {stats['total_size']:,}")
    print(f"  整体使用率: {stats['overall_usage_ratio']:.1%}")
    
    # 显示各交易对的最新价格
    print(f"\n最新价格:")
    for symbol in symbols:
        latest = exchange_buffer.get_latest(symbol, 1)
        if latest:
            price = latest[0].data['price']
            base_price = base_prices[symbol]
            change_pct = (price - base_price) / base_price * 100
            print(f"  {symbol}: ${price:.4f} ({change_pct:+.2f}%)")
    
    exchange_buffer.close()
    print()


def main():
    """主函数"""
    print("🚀 环形缓冲区功能演示程序")
    print("=" * 60)
    
    try:
        # 依次演示各项功能
        demo_basic_operations()
        demo_time_based_operations()
        demo_data_expiration()
        demo_auto_cleanup()
        demo_multi_channel_buffer()
        demo_thread_safety()
        demo_performance_benchmark()
        demo_convenience_functions()
        demo_real_world_simulation()
        
        print("✅ 所有演示完成")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()