"""
环形缓冲区测试
测试线程安全、数据过期和高性能操作
"""

import pytest
import time
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.core.ring_buffer import (
    RingBuffer, 
    MultiChannelRingBuffer, 
    BufferItem,
    create_market_data_buffer,
    create_tick_data_buffer
)


class TestBufferItem:
    """BufferItem 测试"""
    
    def test_buffer_item_creation(self):
        """测试缓冲区项创建"""
        data = {"price": 100.0, "volume": 1.0}
        timestamp = time.time()
        
        item = BufferItem(data=data, timestamp=timestamp)
        
        assert item.data == data
        assert item.timestamp == timestamp
        assert item.expires_at is None
        assert item.metadata == {}
    
    def test_buffer_item_expiration(self):
        """测试数据过期检查"""
        current_time = time.time()
        
        # 未设置过期时间
        item_no_expiry = BufferItem(data="test", timestamp=current_time)
        assert not item_no_expiry.is_expired()
        
        # 未过期的项
        item_valid = BufferItem(
            data="test", 
            timestamp=current_time,
            expires_at=current_time + 60
        )
        assert not item_valid.is_expired()
        
        # 已过期的项
        item_expired = BufferItem(
            data="test",
            timestamp=current_time - 120,
            expires_at=current_time - 60
        )
        assert item_expired.is_expired()


class TestRingBuffer:
    """RingBuffer 基础测试"""
    
    def test_ring_buffer_creation(self):
        """测试环形缓冲区创建"""
        buffer = RingBuffer[str](capacity=100)
        
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert not buffer.is_full()
        assert buffer.usage_ratio() == 0.0
    
    def test_invalid_capacity(self):
        """测试无效容量"""
        with pytest.raises(ValueError):
            RingBuffer[str](capacity=0)
        
        with pytest.raises(ValueError):
            RingBuffer[str](capacity=-1)
    
    def test_put_and_get_basic(self):
        """测试基本的放入和获取操作"""
        buffer = RingBuffer[str](capacity=5, auto_cleanup=False)
        
        # 添加数据
        assert buffer.put("data1")
        assert buffer.put("data2")
        assert buffer.put("data3")
        
        assert len(buffer) == 3
        assert buffer.usage_ratio() == 0.6
        
        # 获取最新数据
        latest = buffer.get_latest(2)
        assert len(latest) == 2
        assert latest[0].data == "data3"  # 最新的
        assert latest[1].data == "data2"
    
    def test_ring_buffer_overflow(self):
        """测试环形缓冲区溢出"""
        buffer = RingBuffer[int](capacity=3, auto_cleanup=False)
        
        # 填满缓冲区
        for i in range(5):
            buffer.put(i)
        
        # 应该只保留最新的3个
        assert len(buffer) == 3
        assert buffer.is_full()
        
        latest = buffer.get_latest(3)
        assert [item.data for item in latest] == [4, 3, 2]
    
    def test_data_expiration(self):
        """测试数据过期"""
        buffer = RingBuffer[str](capacity=10, ttl_seconds=0.2, auto_cleanup=False)
        
        # 添加数据
        buffer.put("data1")
        time.sleep(0.1)  # 等待一半TTL时间
        buffer.put("data2")
        
        # 第一个数据应该还未过期
        latest = buffer.get_latest(2)
        assert len(latest) == 2
        
        # 等待第一个数据过期但第二个数据未过期
        time.sleep(0.15)  # 第一个数据共0.25秒，第二个数据0.15秒
        
        # 手动清理过期数据
        cleaned = buffer.cleanup_expired()
        assert cleaned == 1  # 应该只清理第一个过期的数据
        
        # 现在应该只有一个有效数据
        latest = buffer.get_latest(10)
        assert len(latest) == 1
        assert latest[0].data == "data2"
    
    def test_time_range_query(self):
        """测试时间范围查询"""
        buffer = RingBuffer[str](capacity=10, auto_cleanup=False)
        
        base_time = time.time()
        
        # 添加带时间戳的数据
        buffer.put("data1", timestamp=base_time)
        buffer.put("data2", timestamp=base_time + 10)
        buffer.put("data3", timestamp=base_time + 20)
        buffer.put("data4", timestamp=base_time + 30)
        
        # 查询时间范围
        range_data = buffer.get_range(base_time + 5, base_time + 25)
        assert len(range_data) == 2
        assert range_data[0].data == "data2"
        assert range_data[1].data == "data3"
        
        # 查询自指定时间以来的数据
        since_data = buffer.get_since(base_time + 15)
        assert len(since_data) == 2
        assert since_data[0].data == "data3"
        assert since_data[1].data == "data4"
    
    def test_peek_latest(self):
        """测试查看最新数据"""
        buffer = RingBuffer[str](capacity=5, auto_cleanup=False)
        
        # 空缓冲区
        assert buffer.peek_latest() is None
        
        # 添加数据
        buffer.put("data1")
        buffer.put("data2")
        
        latest = buffer.peek_latest()
        assert latest is not None
        assert latest.data == "data2"
        
        # 数据应该还在缓冲区中
        assert len(buffer) == 2
    
    def test_iterator_support(self):
        """测试迭代器支持"""
        buffer = RingBuffer[int](capacity=5, auto_cleanup=False)
        
        # 添加数据
        for i in range(3):
            buffer.put(i)
        
        # 测试迭代
        data_list = [item.data for item in buffer]
        assert data_list == [0, 1, 2]
    
    def test_clear_buffer(self):
        """测试清空缓冲区"""
        buffer = RingBuffer[str](capacity=5, auto_cleanup=False)
        
        # 添加数据
        buffer.put("data1")
        buffer.put("data2")
        assert len(buffer) == 2
        
        # 清空
        buffer.clear()
        assert len(buffer) == 0
        assert buffer.usage_ratio() == 0.0
    
    def test_statistics(self):
        """测试统计信息"""
        buffer = RingBuffer[str](capacity=5, ttl_seconds=60, auto_cleanup=False)
        
        # 添加数据
        buffer.put("data1")
        buffer.put("data2")
        
        stats = buffer.get_stats()
        assert stats['capacity'] == 5
        assert stats['current_size'] == 2
        assert stats['usage_ratio'] == 0.4
        assert stats['total_inserts'] == 2
        assert stats['ttl_seconds'] == 60


class TestThreadSafety:
    """线程安全测试"""
    
    def test_concurrent_puts(self):
        """测试并发写入"""
        buffer = RingBuffer[int](capacity=1000, auto_cleanup=False)
        num_threads = 10
        items_per_thread = 100
        
        def worker(thread_id: int):
            for i in range(items_per_thread):
                buffer.put(thread_id * items_per_thread + i)
        
        # 启动多个线程
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证数据
        assert len(buffer) == num_threads * items_per_thread
        assert buffer.get_stats()['total_inserts'] == num_threads * items_per_thread
    
    def test_concurrent_reads_writes(self):
        """测试并发读写"""
        buffer = RingBuffer[int](capacity=500, auto_cleanup=False)
        num_writers = 5
        num_readers = 3
        items_per_writer = 50
        
        read_results = []
        read_lock = threading.Lock()
        
        def writer(thread_id: int):
            for i in range(items_per_writer):
                buffer.put(thread_id * items_per_writer + i)
                time.sleep(0.001)  # 小延迟模拟真实情况
        
        def reader():
            for _ in range(20):
                latest = buffer.get_latest(10)
                with read_lock:
                    read_results.append(len(latest))
                time.sleep(0.005)
        
        # 启动写线程
        write_threads = []
        for i in range(num_writers):
            thread = threading.Thread(target=writer, args=(i,))
            write_threads.append(thread)
            thread.start()
        
        # 启动读线程
        read_threads = []
        for _ in range(num_readers):
            thread = threading.Thread(target=reader)
            read_threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in write_threads + read_threads:
            thread.join()
        
        # 验证没有异常发生，且有数据被读取
        assert len(read_results) > 0
        assert all(count >= 0 for count in read_results)
    
    def test_auto_cleanup_thread_safety(self):
        """测试自动清理的线程安全性"""
        buffer = RingBuffer[int](
            capacity=100, 
            ttl_seconds=0.1, 
            auto_cleanup=True,
            cleanup_interval=0.05,
            cleanup_threshold=0.5
        )
        
        num_threads = 5
        items_per_thread = 20
        
        def worker(thread_id: int):
            for i in range(items_per_thread):
                buffer.put(thread_id * items_per_thread + i)
                time.sleep(0.01)
        
        # 启动多个线程
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待一段时间让清理线程工作
        time.sleep(0.5)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 等待清理完成
        time.sleep(0.2)
        
        # 应该有一些数据被清理
        stats = buffer.get_stats()
        assert stats['total_expired'] > 0 or stats['total_evicted'] > 0
        
        buffer.close()


class TestPerformance:
    """性能测试"""
    
    def test_high_throughput_writes(self):
        """测试高吞吐量写入"""
        buffer = RingBuffer[Dict[str, Any]](capacity=10000, auto_cleanup=False)
        
        num_items = 20000
        start_time = time.time()
        
        # 批量写入
        for i in range(num_items):
            data = {
                "price": 100.0 + i * 0.01,
                "volume": 1.0 + i * 0.001,
                "timestamp": start_time + i * 0.001
            }
            buffer.put(data)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_items / duration
        
        print(f"写入吞吐量: {throughput:.0f} items/second")
        
        # 应该能够达到要求的 >10,000 ticks/s
        assert throughput > 10000
        assert len(buffer) == buffer.capacity  # 应该达到容量限制
    
    def test_batch_operations_performance(self):
        """测试批量操作性能"""
        buffer = RingBuffer[int](capacity=5000, auto_cleanup=False)
        
        # 填充缓冲区
        for i in range(5000):
            buffer.put(i)
        
        # 测试批量读取性能
        start_time = time.time()
        
        for _ in range(1000):
            _ = buffer.get_latest(100)
        
        end_time = time.time()
        read_duration = end_time - start_time
        read_ops_per_sec = 1000 / read_duration
        
        print(f"批量读取性能: {read_ops_per_sec:.0f} operations/second")
        
        # 测试范围查询性能
        base_time = time.time()
        start_time = time.time()
        
        for i in range(100):
            _ = buffer.get_range(base_time - 60, base_time)
        
        end_time = time.time()
        range_duration = end_time - start_time
        range_ops_per_sec = 100 / range_duration
        
        print(f"范围查询性能: {range_ops_per_sec:.0f} operations/second")
        
        # 基本性能要求
        assert read_ops_per_sec > 1000
        assert range_ops_per_sec > 50


class TestMultiChannelRingBuffer:
    """多通道环形缓冲区测试"""
    
    def test_multi_channel_creation(self):
        """测试多通道缓冲区创建"""
        buffer = MultiChannelRingBuffer(
            capacity_per_channel=100,
            max_channels=10
        )
        
        assert buffer.capacity_per_channel == 100
        assert buffer.max_channels == 10
        assert len(buffer.get_all_channels()) == 0
    
    def test_channel_operations(self):
        """测试通道操作"""
        buffer = MultiChannelRingBuffer(capacity_per_channel=5)
        
        # 添加数据到不同通道
        assert buffer.put("BTCUSDT", "data1")
        assert buffer.put("ETHUSDT", "data2")
        assert buffer.put("BTCUSDT", "data3")
        
        # 检查通道
        channels = buffer.get_all_channels()
        assert len(channels) == 2
        assert "BTCUSDT" in channels
        assert "ETHUSDT" in channels
        
        # 获取通道数据
        btc_data = buffer.get_latest("BTCUSDT", 10)
        eth_data = buffer.get_latest("ETHUSDT", 10)
        
        assert len(btc_data) == 2
        assert len(eth_data) == 1
        assert btc_data[0].data == "data3"
        assert eth_data[0].data == "data2"
    
    def test_channel_limit(self):
        """测试通道数量限制"""
        buffer = MultiChannelRingBuffer(
            capacity_per_channel=10,
            max_channels=2
        )
        
        # 添加到限制数量的通道
        buffer.put("channel1", "data1")
        buffer.put("channel2", "data2")
        
        # 尝试添加第三个通道应该失败
        with pytest.raises(RuntimeError):
            buffer.put("channel3", "data3")
    
    def test_multi_channel_stats(self):
        """测试多通道统计"""
        buffer = MultiChannelRingBuffer(capacity_per_channel=5)
        
        # 添加数据
        buffer.put("BTCUSDT", "data1")
        buffer.put("BTCUSDT", "data2")
        buffer.put("ETHUSDT", "data3")
        
        stats = buffer.get_total_stats()
        
        assert stats['total_channels'] == 2
        assert stats['total_size'] == 3
        assert stats['total_capacity'] == 10  # 2 channels * 5 capacity
        assert 'channel_stats' in stats
        assert 'BTCUSDT' in stats['channel_stats']
        assert 'ETHUSDT' in stats['channel_stats']
    
    def test_cleanup_all_channels(self):
        """测试清理所有通道"""
        buffer = MultiChannelRingBuffer(
            capacity_per_channel=5,
            ttl_seconds=0.1
        )
        
        # 添加数据
        buffer.put("BTCUSDT", "data1")
        buffer.put("ETHUSDT", "data2")
        
        # 等待过期
        time.sleep(0.15)
        
        # 清理所有通道
        cleaned = buffer.cleanup_expired_all()
        
        # 应该有数据被清理
        assert len(cleaned) > 0
        assert 'BTCUSDT' in cleaned or 'ETHUSDT' in cleaned


class TestConvenienceFunctions:
    """便利函数测试"""
    
    def test_create_market_data_buffer(self):
        """测试创建市场数据缓冲区"""
        buffer = create_market_data_buffer(
            capacity=1000,
            ttl_minutes=30,
            auto_cleanup=True
        )
        
        assert buffer.capacity == 1000
        assert buffer.ttl_seconds == 30 * 60
        assert buffer.auto_cleanup == True
        
        # 测试添加市场数据
        market_data = {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "volume": 1.0,
            "timestamp": time.time()
        }
        
        assert buffer.put(market_data)
        latest = buffer.get_latest(1)
        assert len(latest) == 1
        assert latest[0].data["symbol"] == "BTCUSDT"
        
        buffer.close()
    
    def test_create_tick_data_buffer(self):
        """测试创建tick数据缓冲区"""
        buffer = create_tick_data_buffer(
            capacity_per_symbol=500,
            ttl_minutes=15,
            max_symbols=50
        )
        
        assert buffer.capacity_per_channel == 500
        assert buffer.ttl_seconds == 15 * 60
        assert buffer.max_channels == 50
        
        # 测试添加tick数据
        tick_data = {
            "price": 50000.0,
            "volume": 0.1,
            "bid": 49999.5,
            "ask": 50000.5
        }
        
        assert buffer.put("BTCUSDT", tick_data)
        latest = buffer.get_latest("BTCUSDT", 1)
        assert len(latest) == 1
        assert latest[0].data["price"] == 50000.0
        
        buffer.close()


class TestEdgeCases:
    """边缘情况测试"""
    
    def test_empty_buffer_operations(self):
        """测试空缓冲区操作"""
        buffer = RingBuffer[str](capacity=5, auto_cleanup=False)
        
        assert len(buffer) == 0
        assert buffer.get_latest(10) == []
        assert buffer.peek_latest() is None
        assert buffer.get_range(0, time.time()) == []
        assert buffer.cleanup_expired() == 0
    
    def test_zero_ttl(self):
        """测试零TTL"""
        buffer = RingBuffer[str](capacity=5, ttl_seconds=0, auto_cleanup=False)
        
        buffer.put("data1")
        
        # 数据应该立即过期
        time.sleep(0.001)
        cleaned = buffer.cleanup_expired()
        assert cleaned == 1
        assert len(buffer.get_latest(10)) == 0
    
    def test_very_large_capacity(self):
        """测试大容量缓冲区"""
        buffer = RingBuffer[int](capacity=100000, auto_cleanup=False)
        
        # 添加大量数据
        for i in range(1000):
            buffer.put(i)
        
        assert len(buffer) == 1000
        assert not buffer.is_full()
        
        latest = buffer.get_latest(100)
        assert len(latest) == 100
        assert latest[0].data == 999  # 最新的数据


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])