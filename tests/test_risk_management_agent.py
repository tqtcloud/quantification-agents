"""
风险管理Agent测试
测试风险评估、仓位大小计算、限制检查和动态风险指标监控功能
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from src.agents.risk_management_agent import (
    RiskManagementAgent, RiskLevel, PositionLimits, RiskLimits,
    RiskAlert, create_risk_management_agent
)
from src.agents.base import AgentConfig
from src.core.models import (
    TradingState, Order, Position, Signal, RiskMetrics, MarketData,
    OrderSide, PositionSide, OrderType
)
from src.core.message_bus import MessageBus


class TestRiskManagementAgent:
    """风险管理Agent测试"""
    
    @pytest.fixture
    def agent_config(self):
        """创建Agent配置"""
        return AgentConfig(
            name="test_risk_manager",
            parameters={"total_capital": 100000.0}
        )
    
    @pytest.fixture
    def position_limits(self):
        """创建仓位限制"""
        return PositionLimits(
            max_position_size=0.05,
            max_total_exposure=0.8,
            max_leverage=10,
            max_positions_count=5
        )
    
    @pytest.fixture
    def risk_limits(self):
        """创建风险限制"""
        return RiskLimits(
            max_daily_loss=0.05,
            max_drawdown=0.15,
            min_margin_ratio=0.2
        )
    
    @pytest.fixture
    def risk_agent(self, agent_config, position_limits, risk_limits):
        """创建风险管理Agent"""
        return RiskManagementAgent(
            config=agent_config,
            position_limits=position_limits,
            risk_limits=risk_limits
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """创建示例市场数据"""
        return {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                timestamp=int(time.time()),
                price=50000.0,
                volume=1000.0,
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
        }
    
    @pytest.fixture
    def sample_position(self):
        """创建示例仓位"""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=48000.0,
            mark_price=50000.0,
            unrealized_pnl=200.0,
            margin=1000.0,
            leverage=5
        )
    
    @pytest.fixture
    def sample_order(self):
        """创建示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
    
    @pytest.fixture
    def trading_state(self, sample_market_data, sample_position):
        """创建交易状态"""
        return TradingState(
            market_data=sample_market_data,
            positions={"BTCUSDT": sample_position},
            risk_metrics=RiskMetrics(
                total_exposure=5000.0,
                max_drawdown=0.05,
                current_drawdown=0.02,
                sharpe_ratio=1.5,
                win_rate=0.6,
                profit_factor=1.2,
                var_95=-500.0,
                margin_usage=0.1,
                leverage_ratio=2.0,
                daily_pnl=100.0,
                total_pnl=1000.0
            )
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self, risk_agent):
        """测试Agent初始化"""
        await risk_agent.initialize()
        
        assert risk_agent.name == "test_risk_manager"
        assert risk_agent.total_capital == 100000.0
        assert risk_agent.current_risk_level == RiskLevel.LOW
        assert len(risk_agent.active_alerts) == 0

    @pytest.mark.asyncio
    async def test_check_basic_limits(self, risk_agent, sample_order, trading_state):
        """测试基本限制检查"""
        await risk_agent.initialize()
        
        # 正常订单应该通过
        result = await risk_agent.check_risk(sample_order, trading_state)
        assert result is True
        
        # 无效数量的订单应该失败
        invalid_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.0  # 无效数量
        )
        result = await risk_agent.check_risk(invalid_order, trading_state)
        assert result is False

    @pytest.mark.asyncio
    async def test_position_size_limits(self, risk_agent, trading_state):
        """测试仓位大小限制"""
        await risk_agent.initialize()
        
        # 超过单个仓位限制的订单
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=2.0  # 100,000 USDT 价值，超过5%限制
        )
        
        result = await risk_agent.check_risk(large_order, trading_state)
        assert result is False

    @pytest.mark.asyncio
    async def test_leverage_limits(self, risk_agent, trading_state):
        """测试杠杆限制"""
        await risk_agent.initialize()
        
        # 创建高杠杆仓位
        high_leverage_position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=3000.0,
            mark_price=3000.0,
            unrealized_pnl=0.0,
            margin=100.0,
            leverage=15  # 超过10倍限制
        )
        
        high_leverage_state = TradingState(
            market_data=trading_state.market_data,
            positions={"ETHUSDT": high_leverage_position}
        )
        
        order = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        result = await risk_agent.check_risk(order, high_leverage_state)
        assert result is False

    @pytest.mark.asyncio
    async def test_adjust_position_size(self, risk_agent, sample_order, trading_state):
        """测试仓位大小调整"""
        await risk_agent.initialize()
        
        original_quantity = sample_order.quantity
        adjusted_order = await risk_agent.adjust_position_size(sample_order, trading_state)
        
        # 调整后的订单应该有效
        assert adjusted_order.quantity > 0
        assert adjusted_order.symbol == sample_order.symbol
        assert adjusted_order.side == sample_order.side

    @pytest.mark.asyncio
    async def test_risk_level_updates(self, risk_agent, trading_state):
        """测试风险等级更新"""
        await risk_agent.initialize()
        
        # 初始风险等级应该是LOW
        assert risk_agent.current_risk_level == RiskLevel.LOW
        
        # 创建高风险状态
        high_risk_metrics = RiskMetrics(
            total_exposure=80000.0,  # 80%敞口
            max_drawdown=0.12,
            current_drawdown=0.12,   # 12%回撤
            sharpe_ratio=0.5,
            win_rate=0.4,
            profit_factor=0.8,
            var_95=-8000.0,
            margin_usage=0.9,        # 90%保证金使用
            leverage_ratio=8.0,
            daily_pnl=-4000.0,       # 每日亏损
            total_pnl=-5000.0
        )
        
        high_risk_state = TradingState(
            market_data=trading_state.market_data,
            positions=trading_state.positions,
            risk_metrics=high_risk_metrics
        )
        
        # 分析高风险状态应该生成风险信号
        signals = await risk_agent.analyze(high_risk_state)
        assert len(signals) > 0
        
        # 风险等级应该提升
        assert risk_agent.current_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_correlation_risk_check(self, risk_agent, trading_state):
        """测试相关性风险检查"""
        await risk_agent.initialize()
        
        # 添加高相关性仓位
        eth_position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=3000.0,
            mark_price=3000.0,
            unrealized_pnl=0.0,
            margin=500.0,
            leverage=5
        )
        
        # 模拟ETH市场数据
        eth_market_data = MarketData(
            symbol="ETHUSDT",
            timestamp=int(time.time()),
            price=3000.0,
            volume=1000.0,
            bid=2999.0,
            ask=3001.0,
            bid_volume=100.0,
            ask_volume=100.0
        )
        
        correlated_state = TradingState(
            market_data={
                "BTCUSDT": trading_state.market_data["BTCUSDT"],
                "ETHUSDT": eth_market_data
            },
            positions={
                "BTCUSDT": trading_state.positions["BTCUSDT"],
                "ETHUSDT": eth_position
            }
        )
        
        # 尝试增加更多BTC仓位（与ETH高度相关）
        large_btc_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        result = await risk_agent.check_risk(large_btc_order, correlated_state)
        # 应该因为相关性风险被限制
        assert result is False

    @pytest.mark.asyncio
    async def test_margin_risk_signals(self, risk_agent):
        """测试保证金风险信号"""
        await risk_agent.initialize()
        
        # 创建高保证金使用率状态
        high_margin_metrics = RiskMetrics(
            total_exposure=50000.0,
            max_drawdown=0.05,
            current_drawdown=0.02,
            sharpe_ratio=1.5,
            win_rate=0.6,
            profit_factor=1.2,
            var_95=-500.0,
            margin_usage=0.95,  # 95%保证金使用率
            leverage_ratio=5.0,
            daily_pnl=100.0,
            total_pnl=1000.0
        )
        
        high_margin_state = TradingState(
            market_data={},
            risk_metrics=high_margin_metrics
        )
        
        signals = await risk_agent.analyze(high_margin_state)
        
        # 应该生成保证金风险信号
        margin_signals = [s for s in signals if "margin" in s.reason.lower()]
        assert len(margin_signals) > 0
        assert margin_signals[0].action == OrderSide.SELL
        assert margin_signals[0].strength < 0

    @pytest.mark.asyncio
    async def test_drawdown_risk_signals(self, risk_agent):
        """测试回撤风险信号"""
        await risk_agent.initialize()
        
        # 创建高回撤状态
        high_drawdown_metrics = RiskMetrics(
            total_exposure=50000.0,
            max_drawdown=0.18,     # 超过15%限制
            current_drawdown=0.18,
            sharpe_ratio=0.5,
            win_rate=0.4,
            profit_factor=0.8,
            var_95=-500.0,
            margin_usage=0.5,
            leverage_ratio=3.0,
            daily_pnl=-1000.0,
            total_pnl=-15000.0
        )
        
        high_drawdown_state = TradingState(
            market_data={},
            risk_metrics=high_drawdown_metrics
        )
        
        signals = await risk_agent.analyze(high_drawdown_state)
        
        # 应该生成回撤风险信号
        drawdown_signals = [s for s in signals if "drawdown" in s.reason.lower()]
        assert len(drawdown_signals) > 0
        assert drawdown_signals[0].strength <= -0.9
        assert drawdown_signals[0].confidence >= 0.9

    @pytest.mark.asyncio
    async def test_daily_loss_signals(self, risk_agent):
        """测试每日亏损信号"""
        await risk_agent.initialize()
        
        # 创建每日亏损超限状态
        daily_loss_metrics = RiskMetrics(
            total_exposure=50000.0,
            max_drawdown=0.08,
            current_drawdown=0.05,
            sharpe_ratio=1.0,
            win_rate=0.5,
            profit_factor=1.0,
            var_95=-500.0,
            margin_usage=0.4,
            leverage_ratio=2.0,
            daily_pnl=-6000.0,  # 超过5%限制
            total_pnl=5000.0
        )
        
        daily_loss_state = TradingState(
            market_data={},
            risk_metrics=daily_loss_metrics
        )
        
        signals = await risk_agent.analyze(daily_loss_state)
        
        # 应该生成每日亏损信号
        loss_signals = [s for s in signals if "daily" in s.reason.lower()]
        assert len(loss_signals) > 0

    @pytest.mark.asyncio
    async def test_position_concentration_check(self, risk_agent, sample_market_data):
        """测试仓位集中度检查"""
        await risk_agent.initialize()
        
        # 创建高集中度仓位
        large_position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=10.0,  # 500,000 USDT价值
            entry_price=50000.0,
            mark_price=50000.0,
            unrealized_pnl=0.0,
            margin=50000.0,
            leverage=10
        )
        
        small_position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=1.0,    # 3,000 USDT价值
            entry_price=3000.0,
            mark_price=3000.0,
            unrealized_pnl=0.0,
            margin=300.0,
            leverage=10
        )
        
        concentrated_state = TradingState(
            market_data=sample_market_data,
            positions={
                "BTCUSDT": large_position,
                "ETHUSDT": small_position
            }
        )
        
        signals = await risk_agent.analyze(concentrated_state)
        
        # 应该检测到集中度风险
        concentration_signals = [s for s in signals if "concentration" in s.reason.lower()]
        assert len(concentration_signals) > 0

    def test_risk_status_summary(self, risk_agent):
        """测试风险状态摘要"""
        status = risk_agent.get_risk_status()
        
        assert "current_risk_level" in status
        assert "active_alerts_count" in status
        assert "position_limits" in status
        assert "risk_limits" in status
        assert "total_capital" in status
        assert status["total_capital"] == 100000.0

    def test_create_risk_management_agent_function(self):
        """测试创建风险管理Agent的便捷函数"""
        agent = create_risk_management_agent(
            name="test_agent",
            total_capital=50000.0,
            max_daily_loss=0.03,
            max_position_size=0.08
        )
        
        assert agent.name == "test_agent"
        assert agent.total_capital == 50000.0
        assert agent.risk_limits.max_daily_loss == 0.03
        assert agent.position_limits.max_position_size == 0.08

    @pytest.mark.asyncio
    async def test_message_handling(self, risk_agent):
        """测试消息处理"""
        await risk_agent.initialize()
        
        # 测试仓位更新消息处理
        await risk_agent._handle_position_update("test_agent", {"symbol": "BTCUSDT"})
        
        # 测试订单执行消息处理
        await risk_agent._handle_order_execution("test_agent", {"order_id": "123"})
        
        # 测试市场数据更新消息处理
        await risk_agent._handle_market_data_update("test_agent", {"symbol": "BTCUSDT"})

    @pytest.mark.asyncio
    async def test_risk_monitoring_performance(self, risk_agent, trading_state):
        """测试风险监控性能"""
        await risk_agent.initialize()
        
        # 测试大量风险检查的性能
        start_time = time.time()
        
        for i in range(100):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.01
            )
            await risk_agent.check_risk(order, trading_state)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 100次风险检查应该在1秒内完成
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_risk_agent_shutdown(self, risk_agent):
        """测试风险Agent关闭"""
        await risk_agent.initialize()
        assert risk_agent._initialized is True
        
        await risk_agent.shutdown()
        assert risk_agent._initialized is False


if __name__ == "__main__":
    pytest.main([__file__])