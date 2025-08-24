"""
测试自动平仓器

测试AutoPositionCloser的核心功能，包括仓位管理、策略执行、监控循环等。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.core.position.models import (
    PositionInfo, ClosingReason, ClosingAction, PositionCloseRequest, PositionCloseResult
)
from src.core.position.auto_position_closer import AutoPositionCloser
from src.core.position.closing_strategies import ProfitTargetStrategy, StopLossStrategy
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength


@pytest.fixture
def sample_position():
    """创建示例仓位"""
    return PositionInfo(
        position_id="TEST_001",
        symbol="BTCUSDT",
        entry_price=50000.0,
        current_price=51000.0,
        quantity=0.5,
        side="long",
        entry_time=datetime.utcnow(),
        unrealized_pnl=500.0,
        unrealized_pnl_pct=2.0
    )


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
def auto_closer():
    """创建自动平仓器实例"""
    config = {
        'monitoring_interval_seconds': 1,
        'enable_emergency_stop': True,
        'emergency_loss_threshold': -10.0,
        'strategies': {
            'profit_target': {
                'strategy_class': ProfitTargetStrategy,
                'parameters': {'target_profit_pct': 5.0, 'priority': 2},
                'enabled': True
            },
            'stop_loss': {
                'strategy_class': StopLossStrategy,
                'parameters': {'stop_loss_pct': -2.0, 'priority': 1},
                'enabled': True
            }
        }
    }
    return AutoPositionCloser(config)


class TestAutoPositionCloserInitialization:
    """测试自动平仓器初始化"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        closer = AutoPositionCloser()
        
        assert not closer.is_running
        assert len(closer.strategies) >= 2  # 至少有默认策略
        assert 'profit_target' in closer.strategies
        assert 'stop_loss' in closer.strategies
        assert closer.total_positions_managed == 0
    
    def test_custom_config_initialization(self):
        """测试自定义配置初始化"""
        config = {
            'monitoring_interval_seconds': 10,
            'enable_emergency_stop': False,
            'strategies': {
                'profit_target': {
                    'strategy_class': ProfitTargetStrategy,
                    'parameters': {'target_profit_pct': 8.0},
                    'enabled': True
                }
            }
        }
        
        closer = AutoPositionCloser(config)
        
        assert closer.config['monitoring_interval_seconds'] == 10
        assert not closer.config['enable_emergency_stop']
        assert len(closer.strategies) == 1
        assert 'profit_target' in closer.strategies
    
    def test_custom_strategies(self):
        """测试自定义策略添加"""
        custom_strategy = ProfitTargetStrategy({'target_profit_pct': 3.0})
        custom_strategies = {'custom_profit': custom_strategy}
        
        closer = AutoPositionCloser(custom_strategies=custom_strategies)
        
        assert 'custom_profit' in closer.strategies
        assert closer.strategies['custom_profit'] == custom_strategy


class TestPositionManagement:
    """测试仓位管理"""
    
    def test_add_position(self, auto_closer, sample_position):
        """测试添加仓位"""
        auto_closer.add_position(sample_position)
        
        assert sample_position.position_id in auto_closer.active_positions
        assert sample_position.position_id in auto_closer.monitoring_states
        assert auto_closer.total_positions_managed == 1
    
    def test_remove_position(self, auto_closer, sample_position):
        """测试移除仓位"""
        auto_closer.add_position(sample_position)
        
        removed = auto_closer.remove_position(sample_position.position_id)
        
        assert removed == sample_position
        assert sample_position.position_id not in auto_closer.active_positions
        assert sample_position.position_id not in auto_closer.monitoring_states
    
    def test_update_position_price(self, auto_closer, sample_position):
        """测试更新仓位价格"""
        auto_closer.add_position(sample_position)
        
        success = auto_closer.update_position_price(sample_position.position_id, 52000.0)
        
        assert success
        position = auto_closer.get_position(sample_position.position_id)
        assert position.current_price == 52000.0
        assert position.unrealized_pnl == 1000.0  # (52000 - 50000) * 0.5
    
    def test_update_nonexistent_position_price(self, auto_closer):
        """测试更新不存在仓位的价格"""
        success = auto_closer.update_position_price("NONEXISTENT", 50000.0)
        assert not success
    
    def test_get_all_positions(self, auto_closer, sample_position):
        """测试获取所有仓位"""
        auto_closer.add_position(sample_position)
        
        positions = auto_closer.get_all_positions()
        
        assert len(positions) == 1
        assert sample_position.position_id in positions
        assert positions[sample_position.position_id] == sample_position


class TestPositionClosingLogic:
    """测试仓位平仓逻辑"""
    
    @pytest.mark.asyncio
    async def test_manage_position_profit_target(self, auto_closer, sample_position, sample_signal):
        """测试仓位管理 - 盈利目标触发"""
        auto_closer.add_position(sample_position)
        
        # 设置价格达到盈利目标（5%）
        close_request = await auto_closer.manage_position(
            sample_position.position_id,
            52500.0,  # 5% profit
            sample_signal
        )
        
        assert close_request is not None
        assert close_request.closing_reason == ClosingReason.PROFIT_TARGET
        assert 'triggered_strategy' in close_request.metadata
    
    @pytest.mark.asyncio
    async def test_manage_position_stop_loss(self, auto_closer, sample_position, sample_signal):
        """测试仓位管理 - 止损触发"""
        auto_closer.add_position(sample_position)
        
        # 设置价格达到止损（-2%）
        close_request = await auto_closer.manage_position(
            sample_position.position_id,
            49000.0,  # -2% loss
            sample_signal
        )
        
        assert close_request is not None
        assert close_request.closing_reason == ClosingReason.STOP_LOSS
    
    @pytest.mark.asyncio
    async def test_manage_position_emergency_stop(self, auto_closer, sample_position):
        """测试紧急止损"""
        auto_closer.add_position(sample_position)
        
        # 设置价格达到紧急止损（-10%）
        close_request = await auto_closer.manage_position(
            sample_position.position_id,
            45000.0  # -10% loss
        )
        
        assert close_request is not None
        assert close_request.closing_reason == ClosingReason.EMERGENCY
        assert close_request.urgency == "emergency"
    
    @pytest.mark.asyncio
    async def test_manage_position_no_trigger(self, auto_closer, sample_position, sample_signal):
        """测试仓位管理 - 无触发条件"""
        auto_closer.add_position(sample_position)
        
        # 价格变化但未触发任何条件
        close_request = await auto_closer.manage_position(
            sample_position.position_id,
            50500.0,  # 1% profit，未达到5%目标
            sample_signal
        )
        
        assert close_request is None
    
    @pytest.mark.asyncio
    async def test_manage_nonexistent_position(self, auto_closer):
        """测试管理不存在的仓位"""
        close_request = await auto_closer.manage_position(
            "NONEXISTENT",
            50000.0
        )
        
        assert close_request is None


class TestCloseRequestExecution:
    """测试平仓请求执行"""
    
    @pytest.mark.asyncio
    async def test_execute_close_request_success(self, auto_closer, sample_position):
        """测试执行平仓请求成功"""
        auto_closer.add_position(sample_position)
        
        close_request = PositionCloseRequest(
            position_id=sample_position.position_id,
            closing_reason=ClosingReason.PROFIT_TARGET,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=sample_position.quantity
        )
        
        result = await auto_closer.execute_close_request(close_request)
        
        assert result.success
        assert result.position_id == sample_position.position_id
        assert result.actual_quantity_closed == sample_position.quantity
        assert result.is_full_close
        assert auto_closer.total_positions_closed == 1
        
        # 全仓平仓后应该移除仓位
        assert sample_position.position_id not in auto_closer.active_positions
    
    @pytest.mark.asyncio
    async def test_execute_partial_close_request(self, auto_closer, sample_position):
        """测试执行部分平仓请求"""
        auto_closer.add_position(sample_position)
        
        close_request = PositionCloseRequest(
            position_id=sample_position.position_id,
            closing_reason=ClosingReason.PROFIT_TARGET,
            action=ClosingAction.PARTIAL_CLOSE,
            quantity_to_close=0.25
        )
        
        result = await auto_closer.execute_close_request(close_request)
        
        assert result.success
        assert result.actual_quantity_closed == 0.25
        assert not result.is_full_close
        
        # 部分平仓后仓位仍存在，但数量减少
        position = auto_closer.get_position(sample_position.position_id)
        assert position is not None
        assert position.quantity == 0.25  # 原来0.5 - 平仓0.25
    
    @pytest.mark.asyncio
    async def test_execute_close_request_with_callback(self, auto_closer, sample_position):
        """测试使用回调函数执行平仓"""
        auto_closer.add_position(sample_position)
        
        close_request = PositionCloseRequest(
            position_id=sample_position.position_id,
            closing_reason=ClosingReason.MANUAL,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=sample_position.quantity
        )
        
        # 自定义执行回调
        async def custom_execution(request):
            return PositionCloseResult(
                request_id="",
                position_id=request.position_id,
                success=True,
                actual_quantity_closed=request.quantity_to_close,
                close_price=51500.0,
                realized_pnl=750.0,  # Custom profit
                closing_reason=request.closing_reason,
                close_time=datetime.utcnow(),
                metadata={'execution_type': 'custom'}
            )
        
        result = await auto_closer.execute_close_request(close_request, custom_execution)
        
        assert result.success
        assert result.realized_pnl == 750.0
        assert result.metadata['execution_type'] == 'custom'


class TestForceClosing:
    """测试强制平仓"""
    
    @pytest.mark.asyncio
    async def test_force_close_position(self, auto_closer, sample_position):
        """测试强制平仓单个仓位"""
        auto_closer.add_position(sample_position)
        
        result = await auto_closer.force_close_position(sample_position.position_id, "test_reason")
        
        assert result is not None
        assert result.success
        assert result.closing_reason == ClosingReason.MANUAL
        assert result.metadata['force_close'] == True
        assert result.metadata['reason'] == "test_reason"
    
    @pytest.mark.asyncio
    async def test_force_close_nonexistent_position(self, auto_closer):
        """测试强制平仓不存在的仓位"""
        result = await auto_closer.force_close_position("NONEXISTENT")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_force_close_all_positions(self, auto_closer):
        """测试强制平仓所有仓位"""
        # 添加多个仓位
        positions = []
        for i in range(3):
            position = PositionInfo(
                position_id=f"TEST_{i:03d}",
                symbol="BTCUSDT",
                entry_price=50000.0,
                current_price=51000.0,
                quantity=0.5,
                side="long",
                entry_time=datetime.utcnow(),
                unrealized_pnl=500.0,
                unrealized_pnl_pct=2.0
            )
            positions.append(position)
            auto_closer.add_position(position)
        
        results = await auto_closer.force_close_all_positions()
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert len(auto_closer.active_positions) == 0


class TestStrategyManagement:
    """测试策略管理"""
    
    def test_enable_disable_strategy(self, auto_closer):
        """测试启用/禁用策略"""
        strategy_name = 'profit_target'
        
        # 禁用策略
        success = auto_closer.disable_strategy(strategy_name)
        assert success
        assert not auto_closer.strategies[strategy_name].enabled
        
        # 启用策略
        success = auto_closer.enable_strategy(strategy_name)
        assert success
        assert auto_closer.strategies[strategy_name].enabled
    
    def test_enable_disable_nonexistent_strategy(self, auto_closer):
        """测试启用/禁用不存在的策略"""
        assert not auto_closer.enable_strategy('nonexistent')
        assert not auto_closer.disable_strategy('nonexistent')
    
    def test_update_strategy_parameters(self, auto_closer):
        """测试更新策略参数"""
        strategy_name = 'profit_target'
        new_params = {'target_profit_pct': 8.0}
        
        success = auto_closer.update_strategy_parameters(strategy_name, new_params)
        
        assert success
        strategy = auto_closer.strategies[strategy_name]
        assert strategy.parameters['target_profit_pct'] == 8.0
    
    def test_get_strategy_statistics(self, auto_closer):
        """测试获取策略统计信息"""
        strategy_name = 'profit_target'
        
        stats = auto_closer.get_strategy_statistics(strategy_name)
        
        assert stats is not None
        assert 'strategy_name' in stats
        assert 'trigger_count' in stats
        assert 'success_count' in stats
        assert stats['strategy_name'] == strategy_name


class TestStatistics:
    """测试统计信息"""
    
    def test_get_statistics(self, auto_closer, sample_position):
        """测试获取统计信息"""
        auto_closer.add_position(sample_position)
        
        stats = auto_closer.get_statistics()
        
        assert 'active_positions' in stats
        assert 'total_managed' in stats
        assert 'total_closed' in stats
        assert 'strategy_stats' in stats
        assert 'position_stats' in stats
        assert stats['active_positions'] == 1
        assert stats['total_managed'] == 1
    
    @pytest.mark.asyncio
    async def test_statistics_after_closing(self, auto_closer, sample_position):
        """测试平仓后的统计信息"""
        auto_closer.add_position(sample_position)
        
        # 执行平仓
        close_request = PositionCloseRequest(
            position_id=sample_position.position_id,
            closing_reason=ClosingReason.MANUAL,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=sample_position.quantity
        )
        
        result = await auto_closer.execute_close_request(close_request)
        
        stats = auto_closer.get_statistics()
        
        assert stats['active_positions'] == 0
        assert stats['total_closed'] == 1
        assert stats['total_profit'] > 0 if result.realized_pnl > 0 else True


class TestLifecycleManagement:
    """测试生命周期管理"""
    
    @pytest.mark.asyncio
    async def test_start_stop(self, auto_closer):
        """测试启动和停止"""
        assert not auto_closer.is_running
        
        # 启动
        await auto_closer.start()
        assert auto_closer.is_running
        assert auto_closer.monitoring_task is not None
        
        # 等待一小段时间让监控任务运行
        await asyncio.sleep(0.1)
        
        # 停止
        await auto_closer.stop()
        assert not auto_closer.is_running
        assert auto_closer.monitoring_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_double_start(self, auto_closer):
        """测试重复启动"""
        await auto_closer.start()
        
        # 重复启动应该无效果
        await auto_closer.start()
        assert auto_closer.is_running
        
        await auto_closer.stop()


class TestErrorHandling:
    """测试错误处理"""
    
    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, auto_closer, sample_position):
        """测试策略错误处理"""
        auto_closer.add_position(sample_position)
        
        # 模拟策略抛出异常
        strategy = auto_closer.strategies['profit_target']
        original_method = strategy.should_close_position
        
        async def error_method(*args, **kwargs):
            raise ValueError("Simulated strategy error")
        
        strategy.should_close_position = error_method
        
        # 应该能正常处理错误，返回None而不是抛出异常
        close_request = await auto_closer.manage_position(
            sample_position.position_id,
            52500.0
        )
        
        assert close_request is None
        
        # 检查监控状态记录了错误
        monitoring_state = auto_closer.monitoring_states[sample_position.position_id]
        assert monitoring_state.error_count > 0
        assert "Strategy profit_target error" in monitoring_state.last_error
        
        # 恢复原方法
        strategy.should_close_position = original_method
    
    @pytest.mark.asyncio
    async def test_execution_error_handling(self, auto_closer, sample_position):
        """测试执行错误处理"""
        auto_closer.add_position(sample_position)
        
        close_request = PositionCloseRequest(
            position_id=sample_position.position_id,
            closing_reason=ClosingReason.MANUAL,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=sample_position.quantity
        )
        
        # 使用会抛出异常的回调
        async def error_callback(request):
            raise ValueError("Execution failed")
        
        result = await auto_closer.execute_close_request(close_request, error_callback)
        
        assert not result.success
        assert "Execution failed" in result.error_message
        assert result.actual_quantity_closed == 0.0


if __name__ == "__main__":
    pytest.main([__file__])