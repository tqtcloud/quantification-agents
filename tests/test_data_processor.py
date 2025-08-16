"""
数据处理器测试
测试实时数据处理管道性能和功能
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch
from collections import deque

from src.data.data_processor import (
    DataProcessor, DataWindow, ProcessingMetrics,
    price_change_processor, volume_filter, price_range_filter, spread_filter
)
from src.core.message_bus import MessageBus, Message, MessagePriority
from src.core.message_types import MarketTickMessage, MarketOrderBookMessage


class TestDataWindow:
    """数据窗口测试"""
    
    def test_data_window_creation(self):
        """测试数据窗口创建"""
        window = DataWindow(max_size=100)
        assert len(window.values) == 0
        assert window.max_size == 100
    
    def test_add_data(self):
        """测试添加数据"""
        window = DataWindow(max_size=3)
        
        window.add(1.0)
        window.add(2.0)
        window.add(3.0)
        
        assert len(window.values) == 3
        assert list(window.values) == [1.0, 2.0, 3.0]
        
        # 超过最大大小时应该移除最老的数据
        window.add(4.0)
        assert len(window.values) == 3
        assert list(window.values) == [2.0, 3.0, 4.0]
    
    def test_get_stats(self):
        """测试获取统计信息"""
        window = DataWindow()
        
        # 空窗口
        stats = window.get_stats()
        assert stats == {}
        
        # 添加数据
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            window.add(value)
        
        stats = window.get_stats()
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["latest"] == 5.0
        assert stats["std"] > 0


class TestProcessingMetrics:
    """处理性能指标测试"""
    
    def test_metrics_creation(self):
        """测试指标创建"""
        metrics = ProcessingMetrics()
        assert metrics.messages_processed == 0
        assert metrics.cache_hit_ratio == 0.0
        assert metrics.error_count == 0
    
    def test_cache_hit_ratio(self):
        """测试缓存命中率计算"""
        metrics = ProcessingMetrics()
        metrics.cache_hits = 80
        metrics.cache_misses = 20
        
        assert metrics.cache_hit_ratio == 0.8
        
        # 没有缓存操作时应该返回0
        metrics.cache_hits = 0
        metrics.cache_misses = 0
        assert metrics.cache_hit_ratio == 0.0


class TestDataProcessor:
    """数据处理器测试"""
    
    @pytest.fixture
    async def mock_message_bus(self):
        """模拟消息总线"""
        bus = MagicMock(spec=MessageBus)
        bus.create_publisher = MagicMock()
        bus.create_subscriber = MagicMock()
        bus.publish = MagicMock()
        return bus
    
    @pytest.fixture
    async def processor(self, mock_message_bus):
        """创建处理器实例"""
        processor = DataProcessor(message_bus=mock_message_bus)
        await processor.initialize()
        yield processor
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_message_bus):
        """测试初始化"""
        processor = DataProcessor(message_bus=mock_message_bus)
        await processor.initialize()
        
        # 验证初始化调用
        mock_message_bus.create_publisher.assert_called_once()
        mock_message_bus.create_subscriber.assert_called_once()
        
        await processor.shutdown()
    
    def test_add_remove_symbol(self, processor):
        """测试添加和移除交易对"""
        # 添加交易对
        processor.add_symbol("BTCUSDT")
        assert "BTCUSDT" in processor.symbols
        assert "BTCUSDT" in processor.data_cache
        
        # 移除交易对
        processor.remove_symbol("BTCUSDT")
        assert "BTCUSDT" not in processor.symbols
        assert "BTCUSDT" not in processor.data_cache
    
    def test_add_processor_and_filter(self, processor):
        """测试添加处理器和过滤器"""
        def test_processor(data):
            data["processed"] = True
            return data
        
        def test_filter(data):
            return data.get("valid", True)
        
        processor.add_processor(test_processor)
        processor.add_filter(test_filter)
        
        assert len(processor.processors) == 1
        assert len(processor.filters) == 1
    
    @pytest.mark.asyncio
    async def test_start_stop(self, processor):
        """测试启动和停止"""
        processor.add_symbol("BTCUSDT")
        
        # 启动处理器
        await processor.start()
        assert processor.running is True
        assert processor.processing_task is not None
        assert processor.metrics_task is not None
        
        # 停止处理器
        await processor.stop()
        assert processor.running is False
    
    @pytest.mark.asyncio
    async def test_process_tick_data(self, processor):
        """测试处理tick数据"""
        processor.add_symbol("BTCUSDT")
        
        # 创建模拟tick消息
        tick_message = MarketTickMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.5,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_volume=10.0,
            ask_volume=12.0,
            timestamp=time.time()
        )
        
        # 处理数据
        processed = await processor._process_tick_data(tick_message)
        
        # 验证处理结果
        assert processed is not None
        assert processed["symbol"] == "BTCUSDT"
        assert processed["price"] == 50000.0
        assert processed["spread"] == 2.0  # ask - bid
        assert processed["spread_pct"] > 0
        assert processed["mid_price"] == 50000.0  # (bid + ask) / 2
        assert "price_mean" in processed
        assert "volume_mean" in processed
    
    @pytest.mark.asyncio
    async def test_process_orderbook_data(self, processor):
        """测试处理订单簿数据"""
        # 创建模拟订单簿消息
        orderbook_message = MarketOrderBookMessage(
            symbol="BTCUSDT",
            bids=[[49999.0, 10.0], [49998.0, 5.0]],
            asks=[[50001.0, 8.0], [50002.0, 15.0]],
            last_update_id=12345,
            timestamp=time.time()
        )
        
        # 处理数据
        processed = await processor._process_orderbook_data(orderbook_message)
        
        # 验证处理结果
        assert processed is not None
        assert processed["symbol"] == "BTCUSDT"
        assert processed["best_bid"] == 49999.0
        assert processed["best_ask"] == 50001.0
        assert processed["spread"] == 2.0
        assert processed["bid_depth_5"] == 15.0  # 10 + 5
        assert processed["ask_depth_5"] == 23.0  # 8 + 15
        assert processed["imbalance"] != 0  # 应该有不平衡度
    
    def test_apply_filters(self, processor):
        """测试应用过滤器"""
        # 添加测试过滤器
        def volume_filter_test(data):
            return data.get("volume", 0) > 1.0
        
        def price_filter_test(data):
            return data.get("price", 0) > 1000.0
        
        processor.add_filter(volume_filter_test)
        processor.add_filter(price_filter_test)
        
        # 测试通过所有过滤器的数据
        valid_data = {"volume": 2.0, "price": 50000.0}
        assert processor._apply_filters(valid_data) is True
        
        # 测试不通过过滤器的数据
        invalid_data1 = {"volume": 0.5, "price": 50000.0}  # volume too low
        assert processor._apply_filters(invalid_data1) is False
        
        invalid_data2 = {"volume": 2.0, "price": 500.0}  # price too low
        assert processor._apply_filters(invalid_data2) is False
    
    @pytest.mark.asyncio
    async def test_handle_tick_message(self, processor):
        """测试处理tick消息"""
        processor.add_symbol("BTCUSDT")
        
        # 创建模拟消息
        tick_data = MarketTickMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.5,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_volume=10.0,
            ask_volume=12.0,
            timestamp=time.time()
        )
        
        message = Message(
            topic="market.btcusdt.tick",
            data=tick_data,
            timestamp=time.time()
        )
        
        # 处理消息
        await processor._handle_tick_message(message)
        
        # 验证性能指标更新
        assert processor.metrics.messages_processed > 0
        
        # 验证数据已缓存
        cached_data = processor.get_latest_data("BTCUSDT", count=1)
        assert len(cached_data) > 0
    
    @pytest.mark.asyncio
    async def test_handle_orderbook_message(self, processor):
        """测试处理订单簿消息"""
        processor.add_symbol("BTCUSDT")
        
        # 创建模拟消息
        orderbook_data = MarketOrderBookMessage(
            symbol="BTCUSDT",
            bids=[[49999.0, 10.0]],
            asks=[[50001.0, 8.0]],
            last_update_id=12345,
            timestamp=time.time()
        )
        
        message = Message(
            topic="market.btcusdt.orderbook",
            data=orderbook_data,
            timestamp=time.time()
        )
        
        # 处理消息
        await processor._handle_orderbook_message(message)
        
        # 验证性能指标更新
        assert processor.metrics.messages_processed > 0
    
    def test_update_processing_metrics(self, processor):
        """测试更新处理指标"""
        initial_count = processor.metrics.messages_processed
        
        # 更新指标
        processor._update_processing_metrics(0.005)  # 5ms
        processor._update_processing_metrics(0.010)  # 10ms
        processor._update_processing_metrics(0.003)  # 3ms
        
        # 验证指标更新
        assert processor.metrics.messages_processed == initial_count + 3
        assert processor.metrics.max_processing_time == 0.010
        assert processor.metrics.min_processing_time == 0.003
        assert processor.metrics.average_processing_time > 0
    
    def test_get_latest_data(self, processor):
        """测试获取最新数据"""
        processor.add_symbol("BTCUSDT")
        
        # 添加测试数据
        test_data = [
            {"timestamp": time.time() - 10, "price": 50000.0},
            {"timestamp": time.time() - 5, "price": 50100.0},
            {"timestamp": time.time(), "price": 50200.0}
        ]
        
        for data in test_data:
            processor.data_cache["BTCUSDT"].put(data)
        
        # 获取最新数据
        latest = processor.get_latest_data("BTCUSDT", count=2)
        assert len(latest) == 2
        assert latest[0]["price"] == 50200.0
    
    def test_get_aggregated_data(self, processor):
        """测试获取聚合数据"""
        processor.add_symbol("BTCUSDT")
        
        # 设置聚合数据
        test_data = {"symbol": "BTCUSDT", "avg_price": 50000.0}
        processor.aggregated_cache["BTCUSDT"] = test_data
        
        # 获取聚合数据
        aggregated = processor.get_aggregated_data("BTCUSDT")
        assert aggregated == test_data
        
        # 不存在的交易对应该返回None
        assert processor.get_aggregated_data("ETHUSDT") is None
    
    def test_get_statistics(self, processor):
        """测试获取统计信息"""
        processor.add_symbol("BTCUSDT")
        
        # 添加一些价格和成交量数据
        for i in range(10):
            processor.price_windows["BTCUSDT"].add(50000.0 + i * 100)
            processor.volume_windows["BTCUSDT"].add(1.0 + i * 0.1)
        
        # 获取统计信息
        stats = processor.get_statistics("BTCUSDT")
        
        assert "price_stats" in stats
        assert "volume_stats" in stats
        assert "cache_info" in stats
        
        assert stats["price_stats"]["count"] == 10
        assert stats["volume_stats"]["count"] == 10
        assert stats["price_stats"]["mean"] > 50000.0
    
    def test_get_performance_metrics(self, processor):
        """测试获取性能指标"""
        # 更新一些指标
        processor._update_processing_metrics(0.005)
        processor.metrics.error_count = 2
        processor.metrics.cache_hits = 80
        processor.metrics.cache_misses = 20
        
        # 获取性能指标
        metrics = processor.get_performance_metrics()
        
        assert metrics.messages_processed == 1
        assert metrics.error_count == 2
        assert metrics.cache_hit_ratio == 0.8


class TestDataProcessorIntegration:
    """数据处理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_processing_pipeline(self):
        """测试完整的处理管道"""
        mock_message_bus = MagicMock(spec=MessageBus)
        mock_message_bus.create_publisher = MagicMock()
        mock_message_bus.create_subscriber = MagicMock()
        mock_message_bus.publish = MagicMock()
        
        processor = DataProcessor(message_bus=mock_message_bus)
        
        try:
            await processor.initialize()
            processor.add_symbol("BTCUSDT")
            
            # 添加自定义处理器
            def custom_processor(data):
                data["custom_field"] = "processed"
                return data
            
            processor.add_processor(custom_processor)
            
            # 创建测试消息
            tick_data = MarketTickMessage(
                symbol="BTCUSDT",
                price=50000.0,
                volume=1.5,
                bid_price=49999.0,
                ask_price=50001.0,
                bid_volume=10.0,
                ask_volume=12.0
            )
            
            message = Message(topic="market.btcusdt.tick", data=tick_data)
            
            # 处理消息
            await processor._handle_tick_message(message)
            
            # 验证处理结果
            latest = processor.get_latest_data("BTCUSDT", count=1)
            assert len(latest) > 0
            assert latest[0]["custom_field"] == "processed"
            
            # 验证消息发布
            mock_message_bus.publish.assert_called()
            
        finally:
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """测试性能要求"""
        mock_message_bus = MagicMock(spec=MessageBus)
        mock_message_bus.create_publisher = MagicMock()
        mock_message_bus.create_subscriber = MagicMock()
        mock_message_bus.publish = MagicMock()
        
        processor = DataProcessor(message_bus=mock_message_bus)
        
        try:
            await processor.initialize()
            processor.add_symbol("BTCUSDT")
            
            # 测试高频处理能力
            start_time = time.time()
            message_count = 1000
            
            for i in range(message_count):
                tick_data = MarketTickMessage(
                    symbol="BTCUSDT",
                    price=50000.0 + i,
                    volume=1.5,
                    bid_price=49999.0 + i,
                    ask_price=50001.0 + i,
                    bid_volume=10.0,
                    ask_volume=12.0,
                    timestamp=time.time()
                )
                
                message = Message(topic="market.btcusdt.tick", data=tick_data)
                await processor._handle_tick_message(message)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 验证性能要求：应该能处理 >500 messages/s (realistic for complex processing)
            messages_per_second = message_count / processing_time
            assert messages_per_second > 500, f"Performance too low: {messages_per_second} msg/s"
            
            # 验证平均处理时间 < 2ms (realistic for complex processing with stats)
            avg_processing_time = processor.metrics.average_processing_time
            assert avg_processing_time < 0.002, f"Processing time too high: {avg_processing_time}s"
            
        finally:
            await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """测试错误恢复能力"""
        mock_message_bus = MagicMock(spec=MessageBus)
        mock_message_bus.create_publisher = MagicMock()
        mock_message_bus.create_subscriber = MagicMock()
        mock_message_bus.publish = MagicMock(side_effect=Exception("Publish failed"))
        
        processor = DataProcessor(message_bus=mock_message_bus)
        
        try:
            await processor.initialize()
            processor.add_symbol("BTCUSDT")
            
            # 添加会抛出异常的处理器
            def failing_processor(data):
                if data["price"] > 50000:
                    raise Exception("Processing failed")
                return data
            
            processor.add_processor(failing_processor)
            
            # 处理正常消息
            normal_tick = MarketTickMessage(
                symbol="BTCUSDT",
                price=49000.0,  # 不会触发异常
                volume=1.5,
                bid_price=48999.0,
                ask_price=49001.0,
                bid_volume=10.0,
                ask_volume=12.0
            )
            
            normal_message = Message(topic="market.btcusdt.tick", data=normal_tick)
            await processor._handle_tick_message(normal_message)
            
            # 处理会失败的消息
            failing_tick = MarketTickMessage(
                symbol="BTCUSDT",
                price=51000.0,  # 会触发异常
                volume=1.5,
                bid_price=50999.0,
                ask_price=51001.0,
                bid_volume=10.0,
                ask_volume=12.0
            )
            
            failing_message = Message(topic="market.btcusdt.tick", data=failing_tick)
            await processor._handle_tick_message(failing_message)
            
            # 验证错误计数增加但系统继续运行
            assert processor.metrics.error_count > 0
            assert processor.metrics.messages_processed > 0
            
        finally:
            await processor.shutdown()


class TestDataProcessorFilters:
    """数据处理器过滤器测试"""
    
    def test_volume_filter(self):
        """测试成交量过滤器"""
        # 高成交量数据应该通过
        high_volume_data = {"volume": 5000.0}
        assert volume_filter(high_volume_data, min_volume=1000.0) is True
        
        # 低成交量数据应该被过滤
        low_volume_data = {"volume": 500.0}
        assert volume_filter(low_volume_data, min_volume=1000.0) is False
        
        # 缺少成交量字段应该被过滤
        no_volume_data = {"price": 50000.0}
        assert volume_filter(no_volume_data, min_volume=1000.0) is False
    
    def test_price_range_filter(self):
        """测试价格范围过滤器"""
        # 正常价格应该通过
        normal_price_data = {"price": 50000.0}
        assert price_range_filter(normal_price_data, min_price=1.0, max_price=100000.0) is True
        
        # 价格过低应该被过滤
        low_price_data = {"price": 0.001}
        assert price_range_filter(low_price_data, min_price=1.0, max_price=100000.0) is False
        
        # 价格过高应该被过滤
        high_price_data = {"price": 200000.0}
        assert price_range_filter(high_price_data, min_price=1.0, max_price=100000.0) is False
    
    def test_spread_filter(self):
        """测试价差过滤器"""
        # 正常价差应该通过
        normal_spread_data = {"spread_pct": 0.1}
        assert spread_filter(normal_spread_data, max_spread_pct=1.0) is True
        
        # 价差过大应该被过滤
        high_spread_data = {"spread_pct": 2.0}
        assert spread_filter(high_spread_data, max_spread_pct=1.0) is False
        
        # 缺少价差字段应该通过（默认为0）
        no_spread_data = {"price": 50000.0}
        assert spread_filter(no_spread_data, max_spread_pct=1.0) is True


class TestDataProcessorProcessors:
    """数据处理器处理器测试"""
    
    def test_price_change_processor(self):
        """测试价格变化处理器"""
        test_data = {"price": 50000.0, "volume": 1.5}
        
        processed = price_change_processor(test_data)
        
        # 验证添加了价格变化字段
        assert "price_change" in processed
        assert "price_change_pct" in processed
        
        # 当前简化实现应该为0
        assert processed["price_change"] == 0.0
        assert processed["price_change_pct"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])