"""
测试仓位管理器

测试PositionManager的完整功能，包括开平仓、风险控制、监控等。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.core.position.models import (
    PositionInfo, ClosingReason, PositionCloseResult, ATRInfo, VolatilityInfo, CorrelationRisk
)
from src.core.position.position_manager import PositionManager, MarketDataProvider, RiskMetrics
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength


@pytest.fixture
def mock_market_data_provider():
    """创建模拟市场数据提供器"""
    provider = Mock(spec=MarketDataProvider)
    
    provider.get_current_price = Mock(return_value=50000.0)
    provider.get_atr_info = Mock(return_value=ATRInfo(
        period=14,
        current_atr=500.0,
        atr_multiplier=2.0
    ))
    provider.get_volatility_info = Mock(return_value=VolatilityInfo(
        current_volatility=0.04,
        avg_volatility=0.03,
        volatility_percentile=0.6
    ))
    provider.get_correlation_matrix = Mock(return_value={
        'BTCUSDT': {'ETHUSDT': 0.7, 'LTCUSDT': 0.5},
        'ETHUSDT': {'BTCUSDT': 0.7, 'LTCUSDT': 0.6}
    })
    
    return provider


@pytest.fixture
def sample_signal():
    """创建示例信号"""
    trading_signal = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.8,
        entry_price=50000.0,
        target_price=52000.0,
        stop_loss=48000.0,
        reasoning=["Technical breakout"],
        indicators_consensus={"ma": 0.7}
    )
    
    return MultiDimensionalSignal(
        primary_signal=trading_signal,
        momentum_score=0.6,
        mean_reversion_score=-0.2,
        volatility_score=0.4,
        volume_score=0.8,
        sentiment_score=0.3,
        overall_confidence=0.7,
        risk_reward_ratio=2.5,
        max_position_size=0.8
    )


@pytest.fixture
def position_manager(mock_market_data_provider):
    """创建仓位管理器实例"""
    config = {
        'max_positions': 10,
        'max_exposure_per_symbol': 0.2,
        'risk_check_interval_seconds': 1,
        'performance_update_interval_seconds': 2,
        'correlation_update_interval_minutes': 1,
        'enable_risk_monitoring': True,
        'enable_performance_tracking': True,
        'auto_closer': {
            'monitoring_interval_seconds': 1,
            'enable_emergency_stop': True,
            'emergency_loss_threshold': -8.0
        }
    }
    return PositionManager(config, mock_market_data_provider)


class TestPositionManagerInitialization:
    """测试仓位管理器初始化"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        manager = PositionManager()
        
        assert not manager.is_running
        assert manager.auto_closer is not None
        assert manager.max_positions == 20
        assert manager.max_exposure_per_symbol == 0.1
        assert isinstance(manager.risk_metrics, RiskMetrics)
    
    def test_custom_config_initialization(self, mock_market_data_provider):
        """测试自定义配置初始化"""
        config = {
            'max_positions': 5,
            'max_exposure_per_symbol': 0.15,
            'enable_risk_monitoring': False
        }
        
        manager = PositionManager(config, mock_market_data_provider)
        
        assert manager.max_positions == 5
        assert manager.max_exposure_per_symbol == 0.15
        assert not manager.config['enable_risk_monitoring']
    
    def test_market_data_provider_assignment(self, mock_market_data_provider):
        """测试市场数据提供器分配"""
        manager = PositionManager(market_data_provider=mock_market_data_provider)
        
        assert manager.market_data_provider == mock_market_data_provider


class TestPositionOpenClose:
    """测试开平仓功能"""
    
    @pytest.mark.asyncio
    async def test_open_position_success(self, position_manager, sample_signal):
        """测试成功开仓"""
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long",
            signal=sample_signal
        )
        
        assert position_id is not None
        assert position_id.startswith("BTCUSDT_long_")
        
        # 检查仓位是否添加到自动平仓器
        position = position_manager.auto_closer.get_position(position_id)
        assert position is not None
        assert position.symbol == "BTCUSDT"
        assert position.side == "long"
        assert position.quantity == 0.5
        assert position.stop_loss == sample_signal.primary_signal.stop_loss
        assert position.take_profit == sample_signal.primary_signal.target_price
    
    @pytest.mark.asyncio
    async def test_open_position_without_signal(self, position_manager):
        """测试无信号开仓"""
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="short"
        )
        
        assert position_id is not None
        
        position = position_manager.auto_closer.get_position(position_id)
        assert position.stop_loss is None
        assert position.take_profit is None
    
    @pytest.mark.asyncio
    async def test_open_position_risk_check_failed(self, position_manager):
        """测试开仓风险检查失败"""
        # 模拟超过最大仓位数
        position_manager.max_positions = 0
        
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        assert position_id is None
    
    @pytest.mark.asyncio
    async def test_close_position_success(self, position_manager, sample_signal):
        """测试成功平仓"""
        # 先开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long",
            signal=sample_signal
        )
        
        # 平仓
        result = await position_manager.close_position(position_id, reason="test_close")
        
        assert result is not None
        assert result.success
        assert result.position_id == position_id
        assert result.closing_reason == ClosingReason.MANUAL
        assert result.metadata['manual_reason'] == "test_close"
        assert position_id in position_manager.closed_positions
    
    @pytest.mark.asyncio
    async def test_close_position_partial(self, position_manager, sample_signal):
        """测试部分平仓"""
        # 先开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=1.0,
            side="long"
        )
        
        # 部分平仓
        result = await position_manager.close_position(position_id, quantity=0.3)
        
        assert result is not None
        assert result.success
        assert result.actual_quantity_closed == 0.3
        assert not result.is_full_close
        
        # 检查剩余数量
        position = position_manager.auto_closer.get_position(position_id)
        assert position is not None
        assert position.quantity == 0.7  # 1.0 - 0.3
    
    @pytest.mark.asyncio
    async def test_close_nonexistent_position(self, position_manager):
        """测试平仓不存在的仓位"""
        result = await position_manager.close_position("NONEXISTENT")
        assert result is None


class TestPriceUpdates:
    """测试价格更新"""
    
    @pytest.mark.asyncio
    async def test_update_position_prices(self, position_manager, sample_signal):
        """测试批量更新价格"""
        # 开多个仓位
        position_ids = []
        for i, symbol in enumerate(['BTCUSDT', 'ETHUSDT', 'LTCUSDT']):
            position_id = await position_manager.open_position(
                symbol=symbol,
                entry_price=50000.0 - i * 1000,
                quantity=0.5,
                side="long"
            )
            position_ids.append(position_id)
        
        # 批量更新价格
        price_data = {
            'BTCUSDT': 51000.0,
            'ETHUSDT': 48500.0,
            'LTCUSDT': 48000.0
        }
        
        await position_manager.update_position_prices(price_data)
        
        # 检查价格是否更新
        for position_id in position_ids:
            position = position_manager.auto_closer.get_position(position_id)
            assert position is not None
            expected_price = price_data[position.symbol]
            assert position.current_price == expected_price


class TestPositionMonitoring:
    """测试仓位监控"""
    
    @pytest.mark.asyncio
    async def test_run_position_monitoring(self, position_manager, sample_signal):
        """测试运行仓位监控"""
        # 开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        # 更新价格到盈利目标
        position_manager.auto_closer.update_position_price(position_id, 52500.0)  # 5% profit
        
        # 运行监控
        signal_data = {"BTCUSDT": sample_signal}
        close_requests = await position_manager.run_position_monitoring(signal_data)
        
        assert len(close_requests) == 1
        assert close_requests[0].position_id == position_id
        assert close_requests[0].closing_reason == ClosingReason.PROFIT_TARGET
    
    @pytest.mark.asyncio
    async def test_run_position_monitoring_no_triggers(self, position_manager):
        """测试监控无触发"""
        # 开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        # 价格变化不大，不触发任何策略
        position_manager.auto_closer.update_position_price(position_id, 50500.0)  # 1% profit
        
        close_requests = await position_manager.run_position_monitoring()
        
        assert len(close_requests) == 0


class TestRiskManagement:
    """测试风险管理"""
    
    @pytest.mark.asyncio
    async def test_pre_position_risk_check_max_positions(self, position_manager):
        """测试最大仓位数限制"""
        # 设置最大仓位为1
        position_manager.max_positions = 1
        
        # 第一个仓位应该成功
        position_id1 = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        assert position_id1 is not None
        
        # 第二个仓位应该失败
        position_id2 = await position_manager.open_position(
            symbol="ETHUSDT",
            entry_price=3000.0,
            quantity=1.0,
            side="long"
        )
        assert position_id2 is None
    
    @pytest.mark.asyncio
    async def test_risk_metrics_update(self, position_manager, sample_signal):
        """测试风险指标更新"""
        # 开多个仓位
        for i in range(3):
            await position_manager.open_position(
                symbol=f"TEST{i}USDT",
                entry_price=50000.0,
                quantity=0.5,
                side="long"
            )
        
        # 更新价格
        positions = position_manager.auto_closer.get_all_positions()
        for position in positions.values():
            position.update_price(51000.0)  # 2% profit
        
        # 手动触发风险指标更新
        await position_manager._update_risk_metrics()
        
        risk_metrics = position_manager.get_risk_metrics()
        assert risk_metrics.portfolio_value > 0
        assert risk_metrics.total_exposure > 0
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, position_manager):
        """测试风险告警生成"""
        # 模拟高回撤情况
        position_manager.risk_metrics.current_drawdown = 0.12  # 12% drawdown
        position_manager.config['max_portfolio_drawdown_pct'] = -10.0
        
        # 触发风险检查
        await position_manager._check_risk_alerts()
        
        alerts = position_manager.get_risk_alerts()
        assert len(alerts) > 0
        assert any('drawdown' in alert['alert_type'] for alert in alerts)


class TestCallbacks:
    """测试回调函数"""
    
    @pytest.mark.asyncio
    async def test_position_opened_callback(self, position_manager):
        """测试开仓回调"""
        callback_called = False
        callback_position = None
        
        async def opened_callback(position):
            nonlocal callback_called, callback_position
            callback_called = True
            callback_position = position
        
        position_manager.add_position_opened_callback(opened_callback)
        
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        assert callback_called
        assert callback_position is not None
        assert callback_position.position_id == position_id
    
    @pytest.mark.asyncio
    async def test_position_closed_callback(self, position_manager):
        """测试平仓回调"""
        callback_called = False
        callback_result = None
        
        async def closed_callback(result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result
        
        position_manager.add_position_closed_callback(closed_callback)
        
        # 开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        # 平仓
        result = await position_manager.close_position(position_id)
        
        assert callback_called
        assert callback_result is not None
        assert callback_result.position_id == position_id
    
    @pytest.mark.asyncio
    async def test_risk_alert_callback(self, position_manager):
        """测试风险告警回调"""
        callback_called = False
        callback_alert = None
        
        async def alert_callback(alert):
            nonlocal callback_called, callback_alert
            callback_called = True
            callback_alert = alert
        
        position_manager.add_risk_alert_callback(alert_callback)
        
        # 手动触发告警
        await position_manager._trigger_risk_alert('test_alert', 'Test alert message')
        
        assert callback_called
        assert callback_alert is not None
        assert callback_alert['alert_type'] == 'test_alert'


class TestLifecycleManagement:
    """测试生命周期管理"""
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, position_manager):
        """测试启动停止生命周期"""
        assert not position_manager.is_running
        assert not position_manager.auto_closer.is_running
        
        # 启动
        await position_manager.start()
        assert position_manager.is_running
        assert position_manager.auto_closer.is_running
        assert len(position_manager.tasks) > 0
        
        # 等待一小段时间让任务运行
        await asyncio.sleep(0.1)
        
        # 停止
        await position_manager.stop()
        assert not position_manager.is_running
        assert not position_manager.auto_closer.is_running
        assert len(position_manager.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_loops(self, position_manager):
        """测试监控循环"""
        await position_manager.start()
        
        # 验证监控任务正在运行
        assert any('risk' in task.get_name() for task in position_manager.tasks if hasattr(task, 'get_name'))
        
        # 等待几个监控周期
        await asyncio.sleep(0.2)
        
        await position_manager.stop()


class TestDataQueries:
    """测试数据查询"""
    
    @pytest.mark.asyncio
    async def test_get_position_history(self, position_manager):
        """测试获取仓位历史"""
        # 开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        # 平仓
        await position_manager.close_position(position_id)
        
        # 获取历史
        history = position_manager.get_position_history(position_id)
        
        assert len(history) >= 2  # 至少有开仓和平仓记录
        assert any(event['event_type'] == 'opened' for event in history)
        assert any(event['event_type'] == 'closed' for event in history)
    
    def test_get_risk_metrics(self, position_manager):
        """测试获取风险指标"""
        risk_metrics = position_manager.get_risk_metrics()
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert hasattr(risk_metrics, 'portfolio_value')
        assert hasattr(risk_metrics, 'total_exposure')
    
    def test_get_detailed_statistics(self, position_manager):
        """测试获取详细统计"""
        stats = position_manager.get_detailed_statistics()
        
        assert 'auto_closer_stats' in stats
        assert 'risk_metrics' in stats
        assert 'symbol_stats' in stats
        assert 'recent_alerts' in stats
        assert 'is_running' in stats
        assert 'config' in stats
        assert 'last_update' in stats
    
    @pytest.mark.asyncio
    async def test_symbol_statistics(self, position_manager):
        """测试按标的统计"""
        # 开多个仓位
        await position_manager.open_position("BTCUSDT", 50000.0, 0.5, "long")
        await position_manager.open_position("BTCUSDT", 49000.0, 0.3, "short")
        await position_manager.open_position("ETHUSDT", 3000.0, 2.0, "long")
        
        stats = position_manager.get_detailed_statistics()
        symbol_stats = stats['symbol_stats']
        
        # 检查BTCUSDT统计
        btc_stats = symbol_stats['BTCUSDT']
        assert btc_stats['count'] == 2
        assert btc_stats['long_count'] == 1
        assert btc_stats['short_count'] == 1
        
        # 检查ETHUSDT统计
        eth_stats = symbol_stats['ETHUSDT']
        assert eth_stats['count'] == 1
        assert eth_stats['long_count'] == 1
        assert eth_stats['short_count'] == 0


class TestMarketContextIntegration:
    """测试市场环境上下文集成"""
    
    @pytest.mark.asyncio
    async def test_prepare_market_context(self, position_manager):
        """测试准备市场环境数据"""
        # 开仓
        position_id = await position_manager.open_position(
            symbol="BTCUSDT",
            entry_price=50000.0,
            quantity=0.5,
            side="long"
        )
        
        # 准备市场环境
        context = await position_manager._prepare_market_context([position_id])
        
        assert 'BTCUSDT' in context
        btc_context = context['BTCUSDT']
        assert 'atr_info' in btc_context
        assert 'volatility_info' in btc_context
        assert isinstance(btc_context['atr_info'], ATRInfo)
        assert isinstance(btc_context['volatility_info'], VolatilityInfo)
    
    @pytest.mark.asyncio
    async def test_correlation_context(self, position_manager):
        """测试相关性上下文"""
        # 开多个仓位
        await position_manager.open_position("BTCUSDT", 50000.0, 0.5, "long")
        await position_manager.open_position("ETHUSDT", 3000.0, 1.0, "long")
        
        positions = position_manager.auto_closer.get_all_positions()
        position_ids = list(positions.keys())
        
        context = await position_manager._prepare_market_context(position_ids)
        
        # 检查相关性风险
        btc_context = context.get('BTCUSDT', {})
        if 'correlation_risk' in btc_context:
            corr_risk = btc_context['correlation_risk']
            assert isinstance(corr_risk, CorrelationRisk)
            assert 'ETHUSDT' in corr_risk.correlations


if __name__ == "__main__":
    pytest.main([__file__])