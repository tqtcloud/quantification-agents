"""
测试仓位模型

测试所有数据模型的创建、验证和操作逻辑。
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.core.position.models import (
    PositionInfo, ClosingReason, ClosingAction, PositionCloseRequest, PositionCloseResult,
    ATRInfo, VolatilityInfo, CorrelationRisk, ClosingStrategy
)


class TestPositionInfo:
    """测试PositionInfo类"""
    
    def test_position_creation_valid(self):
        """测试有效仓位创建"""
        position = PositionInfo(
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
        
        assert position.position_id == "TEST_001"
        assert position.symbol == "BTCUSDT"
        assert position.is_long
        assert not position.is_short
        assert position.is_profitable
        assert position.highest_price == 51000.0
        assert position.lowest_price == 50000.0
    
    def test_position_creation_invalid_side(self):
        """测试无效持仓方向"""
        with pytest.raises(ValueError, match="持仓方向必须是'long'或'short'"):
            PositionInfo(
                position_id="TEST_001",
                symbol="BTCUSDT",
                entry_price=50000.0,
                current_price=50000.0,
                quantity=0.5,
                side="invalid",
                entry_time=datetime.utcnow(),
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )
    
    def test_position_creation_invalid_quantity(self):
        """测试无效数量"""
        with pytest.raises(ValueError, match="仓位数量必须大于0"):
            PositionInfo(
                position_id="TEST_001",
                symbol="BTCUSDT",
                entry_price=50000.0,
                current_price=50000.0,
                quantity=-0.5,
                side="long",
                entry_time=datetime.utcnow(),
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )
    
    def test_position_price_update_long(self):
        """测试多头仓位价格更新"""
        position = PositionInfo(
            position_id="TEST_001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            current_price=50000.0,
            quantity=0.5,
            side="long",
            entry_time=datetime.utcnow(),
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        )
        
        # 价格上涨
        position.update_price(52000.0)
        assert position.current_price == 52000.0
        assert position.unrealized_pnl == 1000.0  # (52000 - 50000) * 0.5
        assert position.unrealized_pnl_pct == 4.0  # (52000 - 50000) / 50000 * 100
        assert position.highest_price == 52000.0
        assert position.max_profit == 1000.0
        
        # 价格下跌
        position.update_price(49000.0)
        assert position.current_price == 49000.0
        assert position.unrealized_pnl == -500.0  # (49000 - 50000) * 0.5
        assert position.unrealized_pnl_pct == -2.0
        assert position.lowest_price == 49000.0
        assert position.max_loss == -500.0
    
    def test_position_price_update_short(self):
        """测试空头仓位价格更新"""
        position = PositionInfo(
            position_id="TEST_001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            current_price=50000.0,
            quantity=0.5,
            side="short",
            entry_time=datetime.utcnow(),
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        )
        
        # 价格下跌（空头盈利）
        position.update_price(48000.0)
        assert position.current_price == 48000.0
        assert position.unrealized_pnl == 1000.0  # (50000 - 48000) * 0.5
        assert position.unrealized_pnl_pct == 4.0  # (50000 - 48000) / 50000 * 100
        assert position.lowest_price == 48000.0
        
        # 价格上涨（空头亏损）
        position.update_price(52000.0)
        assert position.current_price == 52000.0
        assert position.unrealized_pnl == -1000.0  # (50000 - 52000) * 0.5
        assert position.unrealized_pnl_pct == -4.0
        assert position.highest_price == 52000.0
    
    def test_hold_duration(self):
        """测试持仓时间计算"""
        entry_time = datetime.utcnow() - timedelta(hours=2)
        
        position = PositionInfo(
            position_id="TEST_001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            current_price=50000.0,
            quantity=0.5,
            side="long",
            entry_time=entry_time,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0
        )
        
        duration = position.hold_duration
        assert duration.total_seconds() >= 7200  # 至少2小时


class TestClosingStrategy:
    """测试平仓策略配置"""
    
    def test_valid_strategy_config(self):
        """测试有效策略配置"""
        strategy = ClosingStrategy(
            strategy_name="profit_target",
            enabled=True,
            priority=1,
            parameters={"target_pct": 5.0}
        )
        
        assert strategy.strategy_name == "profit_target"
        assert strategy.enabled
        assert strategy.priority == 1
        assert strategy.parameters["target_pct"] == 5.0
    
    def test_invalid_strategy_name(self):
        """测试无效策略名称"""
        with pytest.raises(ValueError, match="策略名称不能为空"):
            ClosingStrategy(strategy_name="")
    
    def test_invalid_priority(self):
        """测试无效优先级"""
        with pytest.raises(ValueError, match="策略优先级必须大于0"):
            ClosingStrategy(strategy_name="test", priority=0)


class TestPositionCloseRequest:
    """测试平仓请求"""
    
    def test_valid_close_request(self):
        """测试有效平仓请求"""
        request = PositionCloseRequest(
            position_id="TEST_001",
            closing_reason=ClosingReason.PROFIT_TARGET,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=0.5,
            urgency="normal"
        )
        
        assert request.position_id == "TEST_001"
        assert request.closing_reason == ClosingReason.PROFIT_TARGET
        assert request.action == ClosingAction.FULL_CLOSE
        assert request.quantity_to_close == 0.5
        assert request.urgency == "normal"
    
    def test_invalid_quantity(self):
        """测试无效平仓数量"""
        with pytest.raises(ValueError, match="平仓数量必须大于0"):
            PositionCloseRequest(
                position_id="TEST_001",
                closing_reason=ClosingReason.PROFIT_TARGET,
                action=ClosingAction.FULL_CLOSE,
                quantity_to_close=0.0
            )
    
    def test_invalid_urgency(self):
        """测试无效紧急程度"""
        with pytest.raises(ValueError, match="紧急程度必须是"):
            PositionCloseRequest(
                position_id="TEST_001",
                closing_reason=ClosingReason.PROFIT_TARGET,
                action=ClosingAction.FULL_CLOSE,
                quantity_to_close=0.5,
                urgency="invalid"
            )


class TestATRInfo:
    """测试ATR信息"""
    
    def test_atr_calculation(self):
        """测试ATR动态止损距离计算"""
        atr_info = ATRInfo(
            period=14,
            current_atr=100.0,
            atr_multiplier=2.0
        )
        
        assert atr_info.dynamic_stop_distance == 200.0  # 100 * 2.0


class TestVolatilityInfo:
    """测试波动率信息"""
    
    def test_volatility_classification(self):
        """测试波动率分类"""
        # 高波动率
        high_vol = VolatilityInfo(
            current_volatility=0.05,
            avg_volatility=0.03,
            volatility_percentile=0.9
        )
        assert high_vol.is_high_volatility
        assert not high_vol.is_low_volatility
        
        # 低波动率
        low_vol = VolatilityInfo(
            current_volatility=0.01,
            avg_volatility=0.03,
            volatility_percentile=0.1
        )
        assert not low_vol.is_high_volatility
        assert low_vol.is_low_volatility


class TestCorrelationRisk:
    """测试相关性风险"""
    
    def test_correlation_management(self):
        """测试相关性管理"""
        corr_risk = CorrelationRisk(symbol="BTCUSDT")
        
        # 添加相关性数据
        corr_risk.add_correlation("ETHUSDT", 0.8)
        corr_risk.add_correlation("LTCUSDT", 0.6)
        corr_risk.add_correlation("ADAUSDT", 0.75)
        
        assert corr_risk.get_correlation("ETHUSDT") == 0.8
        assert corr_risk.max_correlation == 0.8
        assert "ETHUSDT" in corr_risk.high_correlation_symbols
        assert "ADAUSDT" in corr_risk.high_correlation_symbols
        assert "LTCUSDT" not in corr_risk.high_correlation_symbols


class TestPositionCloseResult:
    """测试平仓结果"""
    
    def test_close_result_properties(self):
        """测试平仓结果属性"""
        # 全仓平仓
        full_result = PositionCloseResult(
            request_id="REQ_001",
            position_id="POS_001",
            success=True,
            actual_quantity_closed=1.0,
            close_price=51000.0,
            realized_pnl=1000.0,
            closing_reason=ClosingReason.PROFIT_TARGET,
            close_time=datetime.utcnow(),
            metadata={'is_full_close': True, 'remaining_quantity': 0.0}
        )
        
        assert full_result.is_full_close
        assert full_result.remaining_quantity == 0.0
        
        # 部分平仓
        partial_result = PositionCloseResult(
            request_id="REQ_002",
            position_id="POS_001",
            success=True,
            actual_quantity_closed=0.5,
            close_price=51000.0,
            realized_pnl=500.0,
            closing_reason=ClosingReason.PROFIT_TARGET,
            close_time=datetime.utcnow(),
            metadata={'is_full_close': False, 'remaining_quantity': 0.5}
        )
        
        assert not partial_result.is_full_close
        assert partial_result.remaining_quantity == 0.5


if __name__ == "__main__":
    pytest.main([__file__])