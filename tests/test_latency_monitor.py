"""
延迟监控系统测试

测试 LatencyMonitor 类的各项功能，包括：
- 数据新鲜度检查
- 备用数据源切换
- 延迟性能监控
- 告警系统
- 性能指标收集
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from decimal import Decimal

from src.core.models import MarketData
from src.hft.latency_monitor import (
    LatencyMonitor, 
    DataSourceConfig, 
    DataSourceStatus,
    AlertEvent,
    AlertLevel,
    LatencyMetrics,
    PerformanceStats
)


class TestLatencyMonitor:
    """延迟监控系统测试类"""

    @pytest.fixture
    async def latency_monitor(self):
        """创建延迟监控实例"""
        monitor = LatencyMonitor(
            staleness_threshold_ms=100.0,
            stats_window_size=50,
            alert_cooldown_seconds=1.0
        )
        yield monitor
        await monitor.stop()

    @pytest.fixture
    def data_sources(self):
        """创建测试数据源配置"""
        return [
            DataSourceConfig(
                name="primary_feed",
                priority=1,
                max_latency_ms=50.0,
                timeout_ms=1000.0
            ),
            DataSourceConfig(
                name="backup_feed", 
                priority=2,
                max_latency_ms=100.0,
                timeout_ms=2000.0
            ),
            DataSourceConfig(
                name="fallback_feed",
                priority=3, 
                max_latency_ms=200.0,
                timeout_ms=3000.0
            )
        ]

    @pytest.fixture
    def market_data(self):
        """创建测试市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),  # 当前时间戳（毫秒）
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
            timestamp=int((time.time() - 1) * 1000),  # 1秒前的数据
            price=50000.0,
            volume=1.5,
            bid=49990.0,
            ask=50010.0,
            bid_volume=10.0,
            ask_volume=8.0
        )

    async def test_initialization(self, latency_monitor, data_sources):
        """测试初始化"""
        await latency_monitor.initialize(data_sources)
        
        assert len(latency_monitor.data_sources) == 3
        assert "primary_feed" in latency_monitor.data_sources
        assert latency_monitor.data_source_status["primary_feed"] == DataSourceStatus.ACTIVE
        
    async def test_start_stop(self, latency_monitor, data_sources):
        """测试启动和停止"""
        await latency_monitor.initialize(data_sources)
        
        # 启动
        await latency_monitor.start()
        assert latency_monitor._running is True
        
        # 停止
        await latency_monitor.stop()
        assert latency_monitor._running is False

    async def test_data_freshness_check_fresh(self, latency_monitor, data_sources, market_data):
        """测试新鲜数据检查"""
        await latency_monitor.initialize(data_sources)
        
        is_fresh, metrics = await latency_monitor.check_data_freshness(
            "BTCUSDT", market_data, "primary_feed"
        )
        
        assert is_fresh is True
        assert metrics.is_stale is False
        assert metrics.symbol == "BTCUSDT"
        assert metrics.data_source == "primary_feed"
        assert metrics.total_latency_ms < latency_monitor.staleness_threshold_ms

    async def test_data_freshness_check_stale(self, latency_monitor, data_sources, stale_market_data):
        """测试过期数据检查"""
        await latency_monitor.initialize(data_sources)
        
        is_fresh, metrics = await latency_monitor.check_data_freshness(
            "BTCUSDT", stale_market_data, "primary_feed"
        )
        
        assert is_fresh is False
        assert metrics.is_stale is True
        assert metrics.total_latency_ms > latency_monitor.staleness_threshold_ms

    async def test_data_source_switching(self, latency_monitor, data_sources):
        """测试数据源切换"""
        await latency_monitor.initialize(data_sources)
        
        # 设置初始活跃数据源
        latency_monitor.set_active_data_source("BTCUSDT", "primary_feed")
        assert latency_monitor.get_active_data_source("BTCUSDT") == "primary_feed"
        
        # 执行切换
        new_source = await latency_monitor.switch_data_source("BTCUSDT", "Testing switch")
        
        assert new_source == "backup_feed"  # 应该切换到优先级第二的数据源
        assert latency_monitor.get_active_data_source("BTCUSDT") == "backup_feed"

    async def test_data_source_switching_no_backup(self, latency_monitor):
        """测试没有备用数据源时的切换"""
        # 只有一个数据源
        single_source = [DataSourceConfig(name="only_feed", priority=1)]
        await latency_monitor.initialize(single_source)
        
        latency_monitor.set_active_data_source("BTCUSDT", "only_feed")
        
        # 模拟数据源失效
        await latency_monitor.update_data_source_status("only_feed", DataSourceStatus.FAILED)
        
        new_source = await latency_monitor.switch_data_source("BTCUSDT", "Testing no backup")
        assert new_source is None

    async def test_performance_statistics(self, latency_monitor, data_sources, market_data):
        """测试性能统计"""
        await latency_monitor.initialize(data_sources)
        
        # 生成多个延迟指标
        for i in range(10):
            await latency_monitor.check_data_freshness(
                "BTCUSDT", market_data, "primary_feed"
            )
            await asyncio.sleep(0.01)  # 小延迟
        
        stats = latency_monitor.get_performance_stats(symbol="BTCUSDT", data_source="primary_feed")
        assert len(stats) > 0
        
        key = list(stats.keys())[0]
        stat = stats[key]
        assert isinstance(stat, PerformanceStats)
        assert stat.total_samples == 10
        assert stat.avg_latency_ms > 0

    async def test_latency_summary(self, latency_monitor, data_sources, market_data):
        """测试延迟摘要"""
        await latency_monitor.initialize(data_sources)
        latency_monitor.set_active_data_source("BTCUSDT", "primary_feed")
        
        # 生成一些数据
        for _ in range(30):  # 需要足够的数据来计算分位数
            await latency_monitor.check_data_freshness(
                "BTCUSDT", market_data, "primary_feed"
            )
        
        summary = latency_monitor.get_latency_summary("BTCUSDT")
        
        assert "avg_latency_ms" in summary
        assert "p95_latency_ms" in summary
        assert "p99_latency_ms" in summary
        assert "max_latency_ms" in summary
        assert "stale_data_rate" in summary

    async def test_alert_system(self, latency_monitor, data_sources, stale_market_data):
        """测试告警系统"""
        await latency_monitor.initialize(data_sources)
        
        # 设置告警回调
        alerts_received = []
        
        def alert_callback(alert: AlertEvent):
            alerts_received.append(alert)
        
        latency_monitor.add_alert_callback(alert_callback)
        
        # 触发过期数据告警
        await latency_monitor.check_data_freshness(
            "BTCUSDT", stale_market_data, "primary_feed"
        )
        
        # 等待告警处理
        await asyncio.sleep(0.1)
        
        assert len(alerts_received) > 0
        alert = alerts_received[0]
        assert alert.level == AlertLevel.WARNING
        assert "Stale data detected" in alert.message
        assert alert.symbol == "BTCUSDT"

    async def test_alert_cooldown(self, latency_monitor, data_sources, stale_market_data):
        """测试告警冷却机制"""
        latency_monitor.alert_cooldown_seconds = 0.5  # 设置较短的冷却时间
        await latency_monitor.initialize(data_sources)
        
        alerts_received = []
        latency_monitor.add_alert_callback(lambda alert: alerts_received.append(alert))
        
        # 快速触发多次告警
        for _ in range(3):
            await latency_monitor.check_data_freshness(
                "BTCUSDT", stale_market_data, "primary_feed"
            )
            await asyncio.sleep(0.1)
        
        # 应该只收到一次告警（由于冷却机制）
        assert len(alerts_received) <= 1

    async def test_system_health(self, latency_monitor, data_sources):
        """测试系统健康状态"""
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        
        health = latency_monitor.get_system_health()
        
        assert health["running"] is True
        assert "data_sources" in health
        assert len(health["data_sources"]) == 3
        assert health["data_sources"]["primary_feed"] == "active"

    async def test_data_source_status_update(self, latency_monitor, data_sources):
        """测试数据源状态更新"""
        await latency_monitor.initialize(data_sources)
        
        # 更新数据源状态
        await latency_monitor.update_data_source_status("primary_feed", DataSourceStatus.DEGRADED)
        
        assert latency_monitor.data_source_status["primary_feed"] == DataSourceStatus.DEGRADED

    async def test_source_failure_handling(self, latency_monitor, data_sources):
        """测试数据源失效处理"""
        await latency_monitor.initialize(data_sources)
        
        # 设置活跃数据源
        latency_monitor.set_active_data_source("BTCUSDT", "primary_feed")
        latency_monitor.set_active_data_source("ETHUSDT", "primary_feed")
        
        # 模拟数据源失效
        await latency_monitor.update_data_source_status("primary_feed", DataSourceStatus.FAILED)
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 检查是否已切换到备用数据源
        btc_source = latency_monitor.get_active_data_source("BTCUSDT")
        eth_source = latency_monitor.get_active_data_source("ETHUSDT")
        
        assert btc_source == "backup_feed"
        assert eth_source == "backup_feed"

    async def test_health_check_monitoring(self, latency_monitor, data_sources):
        """测试健康检查监控"""
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        
        # 让监控循环运行一段时间
        await asyncio.sleep(1.1)  # 略长于健康检查间隔
        
        # 确认监控任务正在运行
        assert latency_monitor._running is True
        assert latency_monitor._monitor_task is not None

    async def test_network_latency_estimation(self, latency_monitor, data_sources):
        """测试网络延迟估算"""
        await latency_monitor.initialize(data_sources)
        
        # 测试网络延迟估算
        network_latency = latency_monitor._estimate_network_latency("primary_feed")
        
        assert isinstance(network_latency, float)
        assert network_latency > 0
        assert network_latency <= data_sources[0].max_latency_ms

    def test_configuration_validation(self):
        """测试配置验证"""
        # 测试有效配置
        monitor = LatencyMonitor(
            staleness_threshold_ms=100.0,
            stats_window_size=1000,
            alert_cooldown_seconds=30.0
        )
        
        assert monitor.staleness_threshold_ms == 100.0
        assert monitor.stats_window_size == 1000
        assert monitor.alert_cooldown_seconds == 30.0

    async def test_concurrent_operations(self, latency_monitor, data_sources, market_data):
        """测试并发操作"""
        await latency_monitor.initialize(data_sources)
        await latency_monitor.start()
        
        # 并发检查多个symbol的数据新鲜度
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        async def check_symbol_freshness(symbol):
            for _ in range(5):
                await latency_monitor.check_data_freshness(
                    symbol, market_data, "primary_feed"
                )
                await asyncio.sleep(0.01)
        
        # 启动并发任务
        tasks = [check_symbol_freshness(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        # 检查统计信息
        for symbol in symbols:
            latency_monitor.set_active_data_source(symbol, "primary_feed")
            summary = latency_monitor.get_latency_summary(symbol)
            assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, latency_monitor, data_sources, market_data):
        """测试内存效率（统计窗口限制）"""
        # 设置小的统计窗口
        monitor = LatencyMonitor(
            staleness_threshold_ms=100.0,
            stats_window_size=10,  # 小窗口
            alert_cooldown_seconds=1.0
        )
        
        try:
            await monitor.initialize(data_sources)
            
            # 生成大量数据点
            for i in range(50):
                await monitor.check_data_freshness(
                    "BTCUSDT", market_data, "primary_feed"
                )
            
            # 检查历史记录是否被限制在窗口大小内
            key = ("BTCUSDT", "primary_feed")
            history = monitor.latency_history[key]
            
            assert len(history) <= 10  # 不应超过窗口大小
            
        finally:
            await monitor.stop()