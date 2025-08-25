"""
HFT引擎延迟监控集成测试

测试 LatencyMonitor 与 HFTEngine 的集成功能，包括：
- 延迟监控在HFT引擎中的初始化
- 市场数据更新时的延迟检查
- 数据源切换对交易执行的影响
- 延迟指标在系统状态中的反映
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from decimal import Decimal

from src.core.models import MarketData
from src.hft.hft_engine import HFTEngine, HFTConfig
from src.hft.latency_monitor import DataSourceConfig, DataSourceStatus, AlertLevel


class TestHFTLatencyIntegration:
    """HFT延迟监控集成测试类"""

    @pytest.fixture
    def hft_config(self):
        """创建HFT配置"""
        return HFTConfig(
            staleness_threshold_ms=50.0,
            latency_stats_window=100,
            alert_cooldown_seconds=1.0,
            enable_latency_monitoring=True,
            latency_target_ms=10.0
        )

    @pytest.fixture
    def hft_config_disabled(self):
        """创建禁用延迟监控的HFT配置"""
        config = HFTConfig()
        config.enable_latency_monitoring = False
        return config

    @pytest.fixture
    async def hft_engine(self, hft_config):
        """创建HFT引擎实例"""
        engine = HFTEngine(hft_config)
        yield engine
        await engine.stop()

    @pytest.fixture
    async def hft_engine_disabled(self, hft_config_disabled):
        """创建禁用延迟监控的HFT引擎实例"""
        engine = HFTEngine(hft_config_disabled)
        yield engine
        await engine.stop()

    @pytest.fixture
    def data_sources(self):
        """创建测试数据源配置"""
        return [
            DataSourceConfig(
                name="binance_ws",
                priority=1,
                max_latency_ms=30.0,
                timeout_ms=1000.0
            ),
            DataSourceConfig(
                name="okx_ws", 
                priority=2,
                max_latency_ms=50.0,
                timeout_ms=1500.0
            ),
            DataSourceConfig(
                name="bybit_ws",
                priority=3, 
                max_latency_ms=100.0,
                timeout_ms=2000.0
            )
        ]

    @pytest.fixture
    def market_data(self):
        """创建测试市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=1.5,
            bid=49990.0,
            ask=50010.0,
            bid_volume=10.0,
            ask_volume=8.0
        )

    @pytest.fixture
    def stale_market_data(self):
        """创建过期的市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int((time.time() - 0.2) * 1000),  # 200ms前的数据
            price=50000.0,
            volume=1.5,
            bid=49990.0,
            ask=50010.0,
            bid_volume=10.0,
            ask_volume=8.0
        )

    async def test_hft_engine_initialization_with_latency_monitor(self, hft_engine, data_sources):
        """测试HFT引擎与延迟监控的初始化"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        await hft_engine.initialize(symbols, data_sources)
        
        # 检查延迟监控是否正确初始化
        assert hft_engine.latency_monitor is not None
        assert len(hft_engine.latency_monitor.data_sources) == 3
        
        # 检查默认数据源设置
        for symbol in symbols:
            active_source = hft_engine.latency_monitor.get_active_data_source(symbol)
            assert active_source == "binance_ws"  # 最高优先级

    async def test_hft_engine_initialization_without_latency_monitor(self, hft_engine_disabled):
        """测试禁用延迟监控的HFT引擎初始化"""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        await hft_engine_disabled.initialize(symbols)
        
        # 检查延迟监控是否被禁用
        assert hft_engine_disabled.latency_monitor is None

    async def test_market_data_update_with_fresh_data(self, hft_engine, data_sources, market_data):
        """测试新鲜数据的市场数据更新"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # Mock执行引擎和订单簿管理器以避免依赖问题
        with patch.object(hft_engine.orderbook_manager, 'update_orderbook', return_value=True), \
             patch.object(hft_engine.orderbook_manager, 'get_orderbook', return_value=Mock()), \
             patch.object(hft_engine.microstructure_analyzer, 'update', return_value=[]):
            
            result = await hft_engine.update_market_data(
                "BTCUSDT", market_data, "binance_ws"
            )
            
            assert result is True
            assert hft_engine.metrics.data_freshness_checks == 1
            assert hft_engine.metrics.stale_data_detections == 0

    async def test_market_data_update_with_stale_data(self, hft_engine, data_sources, stale_market_data):
        """测试过期数据的市场数据更新"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        result = await hft_engine.update_market_data(
            "BTCUSDT", stale_market_data, "binance_ws"
        )
        
        # 过期数据应该被跳过处理
        assert result is False
        assert hft_engine.metrics.data_freshness_checks == 1
        assert hft_engine.metrics.stale_data_detections == 1

    async def test_data_source_switching_integration(self, hft_engine, data_sources, stale_market_data):
        """测试数据源切换集成"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # 模拟数据源问题导致切换
        with patch.object(hft_engine.orderbook_manager, 'update_orderbook', return_value=True), \
             patch.object(hft_engine.orderbook_manager, 'get_orderbook', return_value=Mock()), \
             patch.object(hft_engine.microstructure_analyzer, 'update', return_value=[]):
            
            await hft_engine.update_market_data(
                "BTCUSDT", stale_market_data, "binance_ws"
            )
            
            # 检查是否发生了数据源切换
            assert hft_engine.metrics.data_source_switches > 0
            
            # 检查活跃数据源是否已切换
            active_source = hft_engine.get_active_data_sources().get("BTCUSDT")
            assert active_source == "okx_ws"  # 应该切换到第二优先级

    async def test_latency_stats_integration(self, hft_engine, data_sources, market_data):
        """测试延迟统计集成"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # Mock组件以成功处理数据
        with patch.object(hft_engine.orderbook_manager, 'update_orderbook', return_value=True), \
             patch.object(hft_engine.orderbook_manager, 'get_orderbook', return_value=Mock()), \
             patch.object(hft_engine.microstructure_analyzer, 'update', return_value=[]):
            
            # 生成多个数据更新
            for _ in range(10):
                await hft_engine.update_market_data("BTCUSDT", market_data, "binance_ws")
                await asyncio.sleep(0.01)
            
            # 检查延迟统计
            latency_stats = hft_engine.get_latency_stats("BTCUSDT")
            assert len(latency_stats) > 0
            assert "avg_latency_ms" in latency_stats

    async def test_system_status_with_latency_monitoring(self, hft_engine, data_sources):
        """测试系统状态包含延迟监控信息"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        status = hft_engine.get_system_status()
        
        # 检查延迟监控相关字段
        assert "data_freshness_checks" in status["metrics"]
        assert "stale_data_detections" in status["metrics"]
        assert "data_source_switches" in status["metrics"]
        assert "avg_data_latency_ms" in status["metrics"]
        assert "p99_data_latency_ms" in status["metrics"]
        
        # 检查组件状态
        assert status["components"]["latency_monitor"] == "active"
        
        # 检查延迟监控健康状态
        assert "latency_monitoring" in status

    async def test_system_status_without_latency_monitoring(self, hft_engine_disabled):
        """测试禁用延迟监控时的系统状态"""
        await hft_engine_disabled.initialize(["BTCUSDT"])
        await hft_engine_disabled.start()
        
        status = hft_engine_disabled.get_system_status()
        
        # 检查延迟监控字段为默认值
        assert status["metrics"]["data_freshness_checks"] == 0
        assert status["metrics"]["stale_data_detections"] == 0
        assert status["metrics"]["data_source_switches"] == 0
        
        # 检查组件状态
        assert status["components"]["latency_monitor"] == "disabled"

    async def test_data_source_status_management(self, hft_engine, data_sources):
        """测试数据源状态管理"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # 检查初始状态
        status = hft_engine.get_data_source_status()
        assert status["binance_ws"] == "active"
        
        # 更新数据源状态
        await hft_engine.update_data_source_status("binance_ws", "degraded")
        
        # 检查状态更新
        updated_status = hft_engine.get_data_source_status()
        assert updated_status["binance_ws"] == "degraded"

    async def test_manual_data_source_switching(self, hft_engine, data_sources):
        """测试手动数据源切换"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # 检查初始数据源
        initial_sources = hft_engine.get_active_data_sources()
        assert initial_sources["BTCUSDT"] == "binance_ws"
        
        # 手动切换数据源
        new_source = await hft_engine.switch_data_source("BTCUSDT", "Manual switch for testing")
        
        assert new_source == "okx_ws"
        
        # 检查切换结果
        updated_sources = hft_engine.get_active_data_sources()
        assert updated_sources["BTCUSDT"] == "okx_ws"

    async def test_latency_alert_handling(self, hft_engine, data_sources, stale_market_data):
        """测试延迟告警处理"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # 捕获日志输出以验证告警
        with patch.object(hft_engine, 'log_warning') as mock_log_warning:
            
            await hft_engine.update_market_data(
                "BTCUSDT", stale_market_data, "binance_ws"
            )
            
            # 等待告警处理
            await asyncio.sleep(0.1)
            
            # 检查是否有告警日志
            assert mock_log_warning.called

    async def test_concurrent_symbol_processing(self, hft_engine, data_sources, market_data):
        """测试多symbol并发处理"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        await hft_engine.initialize(symbols, data_sources)
        await hft_engine.start()
        
        # Mock组件
        with patch.object(hft_engine.orderbook_manager, 'update_orderbook', return_value=True), \
             patch.object(hft_engine.orderbook_manager, 'get_orderbook', return_value=Mock()), \
             patch.object(hft_engine.microstructure_analyzer, 'update', return_value=[]):
            
            # 并发处理多个symbol
            async def process_symbol(symbol):
                for _ in range(5):
                    await hft_engine.update_market_data(symbol, market_data, "binance_ws")
                    await asyncio.sleep(0.01)
            
            tasks = [process_symbol(symbol) for symbol in symbols]
            await asyncio.gather(*tasks)
            
            # 检查所有symbol都有延迟统计
            for symbol in symbols:
                stats = hft_engine.get_latency_stats(symbol)
                assert len(stats) > 0

    async def test_latency_monitoring_lifecycle(self, hft_engine, data_sources):
        """测试延迟监控的完整生命周期"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        
        # 启动
        await hft_engine.start()
        assert hft_engine.latency_monitor._running is True
        
        # 运行一段时间
        await asyncio.sleep(0.5)
        
        # 停止
        await hft_engine.stop()
        assert hft_engine.latency_monitor._running is False

    async def test_performance_impact(self, hft_engine, data_sources, market_data):
        """测试延迟监控对性能的影响"""
        await hft_engine.initialize(["BTCUSDT"], data_sources)
        await hft_engine.start()
        
        # Mock组件以专注于延迟监控性能
        with patch.object(hft_engine.orderbook_manager, 'update_orderbook', return_value=True), \
             patch.object(hft_engine.orderbook_manager, 'get_orderbook', return_value=Mock()), \
             patch.object(hft_engine.microstructure_analyzer, 'update', return_value=[]):
            
            # 测量处理时间
            start_time = time.perf_counter()
            
            # 批量处理数据
            for _ in range(100):
                await hft_engine.update_market_data("BTCUSDT", market_data, "binance_ws")
            
            end_time = time.perf_counter()
            processing_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            # 检查平均处理时间（应该保持在合理范围内）
            avg_time_per_update = processing_time / 100
            assert avg_time_per_update < 5.0  # 每次更新应该少于5ms

    @pytest.mark.parametrize("staleness_threshold", [50.0, 100.0, 200.0])
    async def test_different_staleness_thresholds(self, staleness_threshold, data_sources):
        """测试不同的数据过期阈值"""
        config = HFTConfig(
            staleness_threshold_ms=staleness_threshold,
            enable_latency_monitoring=True
        )
        
        engine = HFTEngine(config)
        try:
            await engine.initialize(["BTCUSDT"], data_sources)
            await engine.start()
            
            # 创建特定延迟的数据
            delayed_data = MarketData(
                symbol="BTCUSDT",
                timestamp=int((time.time() - staleness_threshold/1000 - 0.01) * 1000),
                price=50000.0,
                volume=1.5,
                bid=49990.0,
                ask=50010.0,
                bid_volume=10.0,
                ask_volume=8.0
            )
            
            result = await engine.update_market_data("BTCUSDT", delayed_data, "binance_ws")
            
            # 数据应该被认为是过期的
            assert result is False
            assert engine.metrics.stale_data_detections > 0
            
        finally:
            await engine.stop()