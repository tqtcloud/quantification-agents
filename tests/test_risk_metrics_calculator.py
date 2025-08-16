"""
风险指标计算引擎测试
测试VAR、夏普比率等风险指标计算准确性，最大回撤监控和止损止盈调整
"""

import pytest
import numpy as np
import time
from datetime import datetime
from unittest.mock import MagicMock

from src.risk.risk_metrics_calculator import (
    RiskMetricsCalculator, RiskConfig, PortfolioSnapshot, RiskCalculationMethod
)
from src.core.models import (
    TradingState, Position, RiskMetrics, MarketData, PositionSide
)


class TestRiskMetricsCalculator:
    """风险指标计算器测试"""
    
    @pytest.fixture
    def risk_config(self):
        """创建风险配置"""
        return RiskConfig(
            var_confidence_level=0.95,
            lookback_days=100,
            min_data_points=20
        )
    
    @pytest.fixture
    def calculator(self, risk_config):
        """创建风险计算器"""
        return RiskMetricsCalculator(risk_config)
    
    @pytest.fixture
    def sample_returns(self):
        """创建示例收益率数据"""
        # 模拟正态分布的日收益率
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1%均值，2%标准差
        return returns.tolist()
    
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
            ),
            "ETHUSDT": MarketData(
                symbol="ETHUSDT",
                timestamp=int(time.time()),
                price=3000.0,
                volume=500.0,
                bid=2999.0,
                ask=3001.0,
                bid_volume=50.0,
                ask_volume=50.0
            )
        }
    
    @pytest.fixture
    def sample_positions(self):
        """创建示例仓位"""
        return {
            "BTCUSDT": Position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                quantity=0.1,
                entry_price=48000.0,
                mark_price=50000.0,
                unrealized_pnl=200.0,
                margin=1000.0,
                leverage=5
            ),
            "ETHUSDT": Position(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                quantity=1.0,
                entry_price=2800.0,
                mark_price=3000.0,
                unrealized_pnl=200.0,
                margin=600.0,
                leverage=5
            )
        }
    
    @pytest.fixture
    def trading_state(self, sample_market_data, sample_positions):
        """创建交易状态"""
        return TradingState(
            market_data=sample_market_data,
            positions=sample_positions
        )

    def test_calculator_initialization(self, calculator):
        """测试计算器初始化"""
        assert calculator.config.var_confidence_level == 0.95
        assert calculator.config.lookback_days == 100
        assert len(calculator.price_history) == 0
        assert len(calculator.portfolio_history) == 0
        assert calculator.current_metrics is None

    def test_calculate_historical_var(self, calculator, sample_returns):
        """测试历史VaR计算"""
        var_95 = calculator.calculate_var(sample_returns, 0.95)
        var_99 = calculator.calculate_var(sample_returns, 0.99)
        
        # VaR应该是负数（损失）
        assert var_95 < 0
        assert var_99 < 0
        # 99% VaR应该比95% VaR更极端（更负）
        assert var_99 < var_95

    def test_calculate_parametric_var(self, calculator, sample_returns):
        """测试参数法VaR计算"""
        var = calculator.calculate_var(
            sample_returns, 
            0.95, 
            RiskCalculationMethod.PARAMETRIC
        )
        
        assert var < 0
        assert isinstance(var, float)

    def test_calculate_monte_carlo_var(self, calculator, sample_returns):
        """测试蒙特卡洛VaR计算"""
        var = calculator.calculate_var(
            sample_returns, 
            0.95, 
            RiskCalculationMethod.MONTE_CARLO
        )
        
        assert var < 0
        assert isinstance(var, float)

    def test_calculate_conditional_var(self, calculator, sample_returns):
        """测试条件VaR计算"""
        var = calculator.calculate_var(sample_returns, 0.95)
        cvar = calculator.calculate_conditional_var(sample_returns, 0.95)
        
        # CVaR应该比VaR更极端
        assert cvar <= var
        assert cvar < 0

    def test_calculate_beta(self, calculator):
        """测试Beta系数计算"""
        # 创建相关的资产和市场收益率
        market_returns = [0.01, -0.02, 0.015, -0.01, 0.005] * 10
        asset_returns = [0.015, -0.03, 0.02, -0.015, 0.008] * 10  # 更高波动率
        
        beta = calculator.calculate_beta(asset_returns, market_returns)
        
        # Beta应该大于1（高波动率资产）
        assert beta > 1.0
        assert isinstance(beta, float)

    def test_calculate_correlation_matrix(self, calculator):
        """测试相关性矩阵计算"""
        # 模拟价格历史
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # 添加模拟价格数据
        for symbol in symbols:
            calculator.price_history[symbol] = [
                50000 + i * 100 + np.random.normal(0, 500) 
                for i in range(50)
            ]
        
        correlation_matrix = calculator.calculate_correlation_matrix(symbols)
        
        # 检查矩阵结构
        assert len(correlation_matrix) == len(symbols)
        for symbol in symbols:
            assert symbol in correlation_matrix
            assert len(correlation_matrix[symbol]) == len(symbols)
            # 自相关应该是1
            assert correlation_matrix[symbol][symbol] == 1.0

    def test_calculate_portfolio_volatility(self, calculator):
        """测试投资组合波动率计算"""
        positions = {"BTCUSDT": 50000, "ETHUSDT": 30000}
        correlation_matrix = {
            "BTCUSDT": {"BTCUSDT": 1.0, "ETHUSDT": 0.8},
            "ETHUSDT": {"BTCUSDT": 0.8, "ETHUSDT": 1.0}
        }
        volatilities = {"BTCUSDT": 0.04, "ETHUSDT": 0.05}
        
        portfolio_vol = calculator.calculate_portfolio_volatility(
            positions, correlation_matrix, volatilities
        )
        
        assert portfolio_vol > 0
        assert portfolio_vol < max(volatilities.values())  # 分散化效应

    def test_calculate_stop_loss_level(self, calculator):
        """测试止损水平计算"""
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=50000.0,
            unrealized_pnl=0.0,
            margin=1000.0,
            leverage=5
        )
        
        stop_loss = calculator.calculate_stop_loss_level(position, 0.02)
        
        # 多头止损应该低于当前价格
        assert stop_loss < position.mark_price
        # 止损距离应该合理
        loss_pct = (position.mark_price - stop_loss) / position.mark_price
        assert 0.01 < loss_pct < 0.05  # 1-5%范围

    def test_calculate_take_profit_level(self, calculator):
        """测试止盈水平计算"""
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.1,
            entry_price=50000.0,
            mark_price=50000.0,
            unrealized_pnl=0.0,
            margin=1000.0,
            leverage=5
        )
        
        stop_loss = 49000.0  # 2%止损
        take_profit = calculator.calculate_take_profit_level(
            position, 2.0, stop_loss
        )
        
        # 多头止盈应该高于当前价格
        assert take_profit > position.mark_price
        # 风险回报比应该合理
        risk = position.mark_price - stop_loss
        reward = take_profit - position.mark_price
        risk_reward_ratio = reward / risk
        assert 1.8 < risk_reward_ratio < 2.2  # 约2:1

    def test_calculate_risk_metrics(self, calculator, trading_state):
        """测试完整风险指标计算"""
        total_capital = 100000.0
        
        # 添加一些历史数据
        for i in range(30):
            calculator.portfolio_history.append(total_capital + i * 100)
            if i > 0:
                prev_value = calculator.portfolio_history[-2]
                current_value = calculator.portfolio_history[-1]
                return_rate = (current_value - prev_value) / prev_value
                calculator.return_history.append(return_rate)
        
        metrics = calculator.calculate_risk_metrics(trading_state, total_capital)
        
        # 验证指标结构
        assert isinstance(metrics, RiskMetrics)
        assert metrics.total_exposure > 0
        assert metrics.max_drawdown >= 0
        assert metrics.current_drawdown >= 0
        assert isinstance(metrics.sharpe_ratio, float)
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0
        assert isinstance(metrics.var_95, float)
        assert 0 <= metrics.margin_usage <= 1
        assert metrics.leverage_ratio >= 0

    def test_portfolio_value_calculation(self, calculator, trading_state):
        """测试投资组合价值计算"""
        total_capital = 100000.0
        
        portfolio_value = calculator._calculate_portfolio_value(trading_state, total_capital)
        
        # 投资组合价值应该包括资本和盈亏
        expected_value = total_capital + sum(
            pos.unrealized_pnl + pos.realized_pnl 
            for pos in trading_state.positions.values()
        )
        
        assert portfolio_value == expected_value

    def test_sharpe_ratio_calculation(self, calculator):
        """测试夏普比率计算"""
        # 添加正收益的历史数据
        positive_returns = [0.01, 0.005, 0.015, 0.008, 0.012] * 10
        calculator.return_history.extend(positive_returns)
        
        sharpe = calculator._calculate_sharpe_ratio()
        
        # 正收益应该产生正的夏普比率
        assert sharpe > 0
        assert isinstance(sharpe, float)

    def test_drawdown_calculation(self, calculator):
        """测试回撤计算"""
        # 模拟投资组合价值变化，包含回撤
        values = [100000, 105000, 110000, 108000, 95000, 98000, 103000]
        
        for value in values:
            calculator.portfolio_history.append(value)
        
        max_drawdown = calculator._calculate_max_drawdown()
        current_drawdown = calculator._calculate_current_drawdown(values[-1])
        
        # 最大回撤应该捕获从110000到95000的下跌
        expected_max_dd = (110000 - 95000) / 110000
        assert abs(max_drawdown - expected_max_dd) < 0.01
        
        # 当前回撤应该反映从峰值到当前的下跌
        assert current_drawdown >= 0

    def test_win_rate_calculation(self, calculator):
        """测试胜率计算"""
        # 60%胜率的收益率序列
        returns = [0.01, 0.005, 0.015, -0.008, -0.012, 0.008, 0.012, -0.005, 0.018, 0.002]
        calculator.return_history.extend(returns)
        
        win_rate = calculator._calculate_win_rate(None)
        
        # 胜率应该在合理范围内
        assert 0 <= win_rate <= 1
        # 这个序列有6个正收益，4个负收益，胜率应该是0.6
        assert abs(win_rate - 0.6) < 0.1

    def test_profit_factor_calculation(self, calculator):
        """测试盈利因子计算"""
        # 总盈利2倍于总亏损的收益率序列
        returns = [0.02, 0.01, 0.01, -0.005, -0.005, 0.01, -0.003, 0.015]
        calculator.return_history.extend(returns)
        
        profit_factor = calculator._calculate_profit_factor(None)
        
        # 盈利因子应该大于1（有盈利）
        assert profit_factor >= 1.0

    def test_risk_summary(self, calculator, trading_state):
        """测试风险摘要"""
        total_capital = 100000.0
        
        # 计算风险指标以填充数据
        calculator.calculate_risk_metrics(trading_state, total_capital)
        
        summary = calculator.get_risk_summary()
        
        # 验证摘要结构
        expected_keys = [
            "total_exposure", "current_drawdown", "max_drawdown",
            "sharpe_ratio", "var_95", "margin_usage", "leverage_ratio",
            "last_calculation", "data_points"
        ]
        
        for key in expected_keys:
            assert key in summary

    def test_performance_with_large_dataset(self, calculator):
        """测试大数据集性能"""
        # 创建大量历史数据
        large_returns = np.random.normal(0.001, 0.02, 1000).tolist()
        
        start_time = time.time()
        var = calculator.calculate_var(large_returns, 0.95)
        end_time = time.time()
        
        # VaR计算应该在合理时间内完成
        assert end_time - start_time < 1.0  # 1秒内
        assert var < 0

    def test_edge_cases(self, calculator):
        """测试边缘情况"""
        # 空收益率列表
        var_empty = calculator.calculate_var([], 0.95)
        assert var_empty == 0.0
        
        # 不足数据点
        few_returns = [0.01, -0.02]
        var_few = calculator.calculate_var(few_returns, 0.95)
        assert var_few == 0.0
        
        # 所有相同的收益率
        same_returns = [0.01] * 50
        var_same = calculator.calculate_var(same_returns, 0.95)
        # 无波动性情况下VaR应该接近均值
        assert abs(var_same - 0.01) < 0.001


if __name__ == "__main__":
    pytest.main([__file__])