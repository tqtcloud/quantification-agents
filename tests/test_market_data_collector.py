"""
市场数据收集器测试
测试WebSocket数据接收、数据质量检查和存储功能
"""

import asyncio
import pytest
import time
import json
from unittest.mock import MagicMock, AsyncMock, patch
from collections import deque

from src.data.market_data_collector import (
    MarketDataCollector, SubscriptionConfig, DataQualityMetrics
)
from src.exchanges.binance import BinanceFuturesClient
from src.core.message_bus import MessageBus
from src.core.ring_buffer import RingBuffer


class TestDataQualityMetrics:
    """数据质量指标测试"""
    
    def test_quality_metrics_creation(self):
        """测试质量指标创建"""
        metrics = DataQualityMetrics()
        assert metrics.total_messages == 0
        assert metrics.quality_score == 0.0
        assert metrics.avg_latency == 0.0
    
    def test_quality_score_calculation(self):
        """测试质量分数计算"""
        metrics = DataQualityMetrics()
        metrics.total_messages = 100
        metrics.valid_messages = 95
        metrics.duplicate_messages = 2
        metrics.out_of_order_messages = 1
        metrics.gap_count = 1
        
        # 质量分数应该在0-1之间
        score = metrics.quality_score
        assert 0.0 <= score <= 1.0
        assert score < 1.0  # 由于有错误，分数应该小于1
    
    def test_average_latency(self):
        """测试平均延迟计算"""
        metrics = DataQualityMetrics()
        metrics.total_messages = 10
        metrics.latency_sum = 50.0
        
        assert metrics.avg_latency == 5.0


class TestSubscriptionConfig:
    """订阅配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SubscriptionConfig(symbol="BTCUSDT")
        
        assert config.symbol == "BTCUSDT"
        assert "tick" in config.data_types
        assert "depth" in config.data_types
        assert "1m" in config.kline_intervals
        assert config.enabled is True
        assert config.quality_threshold == 0.8
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SubscriptionConfig(
            symbol="ETHUSDT",
            data_types={"kline"},
            kline_intervals={"5m", "1h"},
            enabled=False,
            quality_threshold=0.9
        )
        
        assert config.symbol == "ETHUSDT"
        assert config.data_types == {"kline"}
        assert config.kline_intervals == {"5m", "1h"}
        assert config.enabled is False
        assert config.quality_threshold == 0.9


class TestMarketDataCollector:
    """市场数据收集器测试"""
    
    @pytest.fixture
    async def mock_client(self):
        """模拟币安客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        client.connect = AsyncMock()
        client.close_all_websockets = AsyncMock()
        client.disconnect = AsyncMock()
        
        # 创建模拟任务函数
        async def dummy_task():
            await asyncio.sleep(0.01)
        
        # 模拟订阅方法，每次调用时返回新的任务
        def create_task():
            return asyncio.create_task(dummy_task())
        
        client.subscribe_ticker = AsyncMock(side_effect=lambda *args, **kwargs: create_task())
        client.subscribe_depth = AsyncMock(side_effect=lambda *args, **kwargs: create_task())
        client.subscribe_kline = AsyncMock(side_effect=lambda *args, **kwargs: create_task())
        return client
    
    @pytest.fixture
    async def mock_message_bus(self):
        """模拟消息总线"""
        bus = MagicMock(spec=MessageBus)
        bus.create_publisher = MagicMock()
        bus.publish = MagicMock()
        return bus
    
    @pytest.fixture
    async def collector(self, mock_client, mock_message_bus):
        """创建收集器实例"""
        with patch('src.data.market_data_collector.DuckDBManager') as mock_duckdb:
            mock_duckdb_instance = AsyncMock()
            mock_duckdb_instance.initialize = AsyncMock()
            mock_duckdb_instance.execute = AsyncMock()
            mock_duckdb_instance.close = AsyncMock()
            mock_duckdb.return_value = mock_duckdb_instance
            
            collector = MarketDataCollector(
                client=mock_client,
                message_bus=mock_message_bus
            )
            
            await collector.initialize()
            yield collector
            await collector.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_client, mock_message_bus):
        """测试初始化"""
        with patch('src.data.market_data_collector.DuckDBManager') as mock_duckdb:
            mock_duckdb_instance = AsyncMock()
            mock_duckdb_instance.initialize = AsyncMock()
            mock_duckdb_instance.execute = AsyncMock()
            mock_duckdb.return_value = mock_duckdb_instance
            
            collector = MarketDataCollector(
                client=mock_client,
                message_bus=mock_message_bus
            )
            
            await collector.initialize()
            
            # 验证初始化调用
            mock_client.connect.assert_called_once()
            mock_message_bus.create_publisher.assert_called_once()
            mock_duckdb_instance.initialize.assert_called_once()
            
            await collector.shutdown()
    
    @pytest.mark.asyncio
    async def test_add_subscription(self, collector):
        """测试添加订阅"""
        config = SubscriptionConfig(
            symbol="BTCUSDT",
            data_types={"tick", "depth"}
        )
        
        await collector.add_subscription(config)
        
        # 验证订阅配置已添加
        assert "BTCUSDT" in collector.subscriptions
        assert collector.subscriptions["BTCUSDT"] == config
        
        # 验证环形缓冲区已创建
        assert "BTCUSDT_tick" in collector.ring_buffers
        assert "BTCUSDT_depth" in collector.ring_buffers
    
    @pytest.mark.asyncio
    async def test_remove_subscription(self, collector):
        """测试移除订阅"""
        config = SubscriptionConfig(symbol="BTCUSDT")
        await collector.add_subscription(config)
        
        # 移除订阅
        await collector.remove_subscription("BTCUSDT")
        
        # 验证订阅已移除
        assert "BTCUSDT" not in collector.subscriptions
        assert "BTCUSDT_tick" not in collector.ring_buffers
        assert "BTCUSDT_depth" not in collector.ring_buffers
    
    @pytest.mark.asyncio
    async def test_start_stop(self, collector):
        """测试启动和停止"""
        config = SubscriptionConfig(symbol="BTCUSDT")
        await collector.add_subscription(config)
        
        # 启动收集器
        await collector.start()
        assert collector.running is True
        assert collector.quality_task is not None
        
        # 停止收集器
        await collector.stop()
        assert collector.running is False
    
    def test_normalize_tick_data(self, collector):
        """测试tick数据归一化"""
        raw_data = {
            "E": 1640995200000,  # 事件时间
            "c": "50000.00",     # 最新价格
            "v": "1.5",          # 24小时成交量
            "b": "49999.00",     # 买一价
            "a": "50001.00",     # 卖一价
            "B": "10.0",         # 买一量
            "A": "12.0"          # 卖一量
        }
        
        normalized = collector._normalize_tick_data("BTCUSDT", raw_data)
        
        assert normalized is not None
        assert normalized["symbol"] == "BTCUSDT"
        assert normalized["price"] == 50000.0
        assert normalized["volume"] == 1.5
        assert normalized["bid_price"] == 49999.0
        assert normalized["ask_price"] == 50001.0
        assert normalized["timestamp"] == 1640995200.0
        assert normalized["exchange"] == "binance"
    
    def test_normalize_tick_data_invalid(self, collector):
        """测试无效tick数据归一化"""
        invalid_data = {
            "E": 1640995200000,
            "c": "invalid_price"  # 无效价格
        }
        
        normalized = collector._normalize_tick_data("BTCUSDT", invalid_data)
        assert normalized is None
    
    def test_normalize_depth_data(self, collector):
        """测试深度数据归一化"""
        raw_data = {
            "E": 1640995200000,
            "u": 12345,
            "b": [["49999.00", "10.0"], ["49998.00", "5.0"]],
            "a": [["50001.00", "8.0"], ["50002.00", "15.0"]]
        }
        
        normalized = collector._normalize_depth_data("BTCUSDT", raw_data)
        
        assert normalized is not None
        assert normalized["symbol"] == "BTCUSDT"
        assert len(normalized["bids"]) == 2
        assert len(normalized["asks"]) == 2
        assert normalized["bids"][0] == [49999.0, 10.0]
        assert normalized["asks"][0] == [50001.0, 8.0]
        assert normalized["last_update_id"] == 12345
    
    def test_normalize_kline_data(self, collector):
        """测试K线数据归一化"""
        raw_data = {
            "E": 1640995200000,
            "k": {
                "t": 1640995140000,  # 开盘时间
                "T": 1640995199999,  # 收盘时间
                "o": "49900.00",     # 开盘价
                "h": "50100.00",     # 最高价
                "l": "49800.00",     # 最低价
                "c": "50000.00",     # 收盘价
                "v": "100.5"         # 成交量
            }
        }
        
        normalized = collector._normalize_kline_data("BTCUSDT", "1m", raw_data)
        
        assert normalized is not None
        assert normalized["symbol"] == "BTCUSDT"
        assert normalized["interval"] == "1m"
        assert normalized["open_price"] == 49900.0
        assert normalized["high_price"] == 50100.0
        assert normalized["low_price"] == 49800.0
        assert normalized["close_price"] == 50000.0
        assert normalized["volume"] == 100.5
    
    def test_validate_data(self, collector):
        """测试数据验证"""
        # 有效数据
        valid_data = {
            "symbol": "BTCUSDT",
            "timestamp": time.time(),
            "price": 50000.0,
            "volume": 1.5
        }
        assert collector._validate_data(valid_data) is True
        
        # 无效数据 - 缺少必要字段
        invalid_data1 = {
            "price": 50000.0,
            "volume": 1.5
        }
        assert collector._validate_data(invalid_data1) is False
        
        # 无效数据 - 价格为负
        invalid_data2 = {
            "symbol": "BTCUSDT",
            "timestamp": time.time(),
            "price": -1000.0,
            "volume": 1.5
        }
        assert collector._validate_data(invalid_data2) is False
        
        # 无效数据 - 时间戳为0
        invalid_data3 = {
            "symbol": "BTCUSDT",
            "timestamp": 0,
            "price": 50000.0,
            "volume": 1.5
        }
        assert collector._validate_data(invalid_data3) is False
    
    def test_check_data_quality(self, collector):
        """测试数据质量检查"""
        symbol = "BTCUSDT"
        data_type = "tick"
        
        # 第一条数据
        data1 = {
            "symbol": symbol,
            "timestamp": time.time(),
            "price": 50000.0,
            "volume": 1.5
        }
        
        quality_score1 = collector._check_data_quality(symbol, data_type, data1)
        assert 0.0 <= quality_score1 <= 1.0
        
        # 第二条数据（时间稍后）
        data2 = {
            "symbol": symbol,
            "timestamp": time.time() + 1,
            "price": 50100.0,
            "volume": 2.0
        }
        
        quality_score2 = collector._check_data_quality(symbol, data_type, data2)
        assert 0.0 <= quality_score2 <= 1.0
        
        # 验证质量指标已更新
        metrics_key = f"{symbol}_{data_type}"
        metrics = collector.quality_metrics[metrics_key]
        assert metrics.total_messages == 2
        assert metrics.valid_messages == 2
    
    def test_calculate_message_hash(self, collector):
        """测试消息哈希计算"""
        data1 = {
            "symbol": "BTCUSDT",
            "timestamp": 1640995200.0,
            "price": 50000.0,
            "volume": 1.5,
            "extra_field": "ignored"
        }
        
        data2 = {
            "symbol": "BTCUSDT",
            "timestamp": 1640995200.0,
            "price": 50000.0,
            "volume": 1.5,
            "another_field": "also_ignored"
        }
        
        # 相同的关键字段应该产生相同的哈希
        hash1 = collector._calculate_message_hash(data1)
        hash2 = collector._calculate_message_hash(data2)
        assert hash1 == hash2
        
        # 不同的关键字段应该产生不同的哈希
        data3 = {
            "symbol": "BTCUSDT",
            "timestamp": 1640995200.0,
            "price": 50001.0,  # 不同的价格
            "volume": 1.5
        }
        hash3 = collector._calculate_message_hash(data3)
        assert hash1 != hash3
    
    def test_get_latest_data(self, collector):
        """测试获取最新数据"""
        symbol = "BTCUSDT"
        data_type = "tick"
        
        # 添加一些测试数据
        buffer_key = f"{symbol}_{data_type}"
        collector.ring_buffers[buffer_key] = RingBuffer(capacity=100)
        
        test_data = [
            {"timestamp": time.time() - 10, "price": 50000.0},
            {"timestamp": time.time() - 5, "price": 50100.0},
            {"timestamp": time.time(), "price": 50200.0}
        ]
        
        for data in test_data:
            collector.ring_buffers[buffer_key].put(data)
        
        # 获取最新数据
        latest = collector.get_latest_data(symbol, data_type, count=2)
        assert len(latest) == 2
        assert latest[0]["price"] == 50200.0  # 最新的
        assert latest[1]["price"] == 50100.0  # 第二新的
    
    def test_get_quality_metrics(self, collector):
        """测试获取质量指标"""
        # 创建一些测试指标
        collector.quality_metrics["BTCUSDT_tick"].total_messages = 100
        collector.quality_metrics["ETHUSDT_tick"].total_messages = 50
        
        # 获取所有指标
        all_metrics = collector.get_quality_metrics()
        assert len(all_metrics) >= 2
        
        # 获取特定交易对的指标
        btc_metrics = collector.get_quality_metrics("BTCUSDT")
        assert len(btc_metrics) == 1
        assert "BTCUSDT_tick" in btc_metrics


class TestMarketDataCollectorIntegration:
    """市场数据收集器集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_data_flow(self):
        """测试完整的数据流程"""
        # 这是一个集成测试，需要模拟完整的数据流
        # 从WebSocket接收 -> 数据处理 -> 存储 -> 发布
        
        mock_client = MagicMock(spec=BinanceFuturesClient)
        mock_client.connect = AsyncMock()
        mock_client.close_all_websockets = AsyncMock()
        mock_client.disconnect = AsyncMock()
        
        mock_message_bus = MagicMock(spec=MessageBus)
        mock_publisher = MagicMock()
        mock_message_bus.create_publisher.return_value = mock_publisher
        mock_message_bus.publish = MagicMock()
        
        with patch('src.data.market_data_collector.DuckDBManager') as mock_duckdb:
            mock_duckdb_instance = AsyncMock()
            mock_duckdb_instance.initialize = AsyncMock()
            mock_duckdb_instance.execute = AsyncMock()
            mock_duckdb.return_value = mock_duckdb_instance
            
            collector = MarketDataCollector(
                client=mock_client,
                message_bus=mock_message_bus
            )
            
            try:
                await collector.initialize()
                
                # 添加订阅
                config = SubscriptionConfig(symbol="BTCUSDT", data_types={"tick"})
                await collector.add_subscription(config)
                
                # 模拟接收tick数据
                mock_tick_data = {
                    "E": int(time.time() * 1000),
                    "c": "50000.00",
                    "v": "1.5",
                    "b": "49999.00",
                    "a": "50001.00",
                    "B": "10.0",
                    "A": "12.0"
                }
                
                await collector._handle_tick_data("BTCUSDT", mock_tick_data)
                
                # 验证数据已存储到环形缓冲区
                latest_data = collector.get_latest_data("BTCUSDT", "tick", count=1)
                assert len(latest_data) == 1
                assert latest_data[0]["price"] == 50000.0
                
                # 验证消息已发布
                mock_message_bus.publish.assert_called()
                
            finally:
                await collector.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        mock_client = MagicMock(spec=BinanceFuturesClient)
        mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
        
        collector = MarketDataCollector(client=mock_client)
        
        # 初始化应该抛出异常
        with pytest.raises(Exception):
            await collector.initialize()
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """测试性能要求"""
        # 测试高频数据处理能力
        mock_client = MagicMock(spec=BinanceFuturesClient)
        mock_client.connect = AsyncMock()
        mock_client.close_all_websockets = AsyncMock()
        mock_client.disconnect = AsyncMock()
        
        mock_message_bus = MagicMock(spec=MessageBus)
        mock_message_bus.create_publisher = MagicMock()
        mock_message_bus.publish = MagicMock()
        
        with patch('src.data.market_data_collector.DuckDBManager') as mock_duckdb:
            mock_duckdb_instance = AsyncMock()
            mock_duckdb_instance.initialize = AsyncMock()
            mock_duckdb_instance.execute = AsyncMock()
            mock_duckdb.return_value = mock_duckdb_instance
            
            collector = MarketDataCollector(
                client=mock_client,
                message_bus=mock_message_bus
            )
            
            try:
                await collector.initialize()
                
                # 模拟高频数据处理
                start_time = time.time()
                message_count = 1000
                
                for i in range(message_count):
                    mock_data = {
                        "E": int(time.time() * 1000),
                        "c": f"{50000 + i}",
                        "v": "1.5",
                        "b": f"{49999 + i}",
                        "a": f"{50001 + i}",
                        "B": "10.0",
                        "A": "12.0"
                    }
                    await collector._handle_tick_data("BTCUSDT", mock_data)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # 验证性能要求：应该能处理 >1000 ticks/s
                messages_per_second = message_count / processing_time
                assert messages_per_second > 1000, f"Performance too low: {messages_per_second} msg/s"
                
            finally:
                await collector.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])