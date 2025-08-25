#!/usr/bin/env python3
"""
管理Agent系统演示和测试
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # 基础导入
    from src.agents.management.risk_management import (
        RiskManagementAgent, RiskLevel, RiskModel, RiskMetrics
    )
    from src.agents.management.portfolio_management import (
        PortfolioManagementAgent, PortfolioOptimizer
    )
    from src.agents.management.optimization import (
        OptimizationEngine, OptimizationType, OptimizationConstraints
    )
    from src.agents.base import AgentConfig
    from src.core.models import TradingState, MarketData, Position
    
    print("✅ 所有模块导入成功")
    
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("这可能是由于缺少scipy或cvxpy依赖")
    print("请运行: pip install scipy cvxpy")
    sys.exit(1)


class ManagementAgentDemo:
    """管理Agent系统演示"""
    
    def __init__(self):
        """初始化演示"""
        print("🚀 初始化管理Agent系统演示")
        
        # 风险管理Agent配置
        self.risk_config = AgentConfig(
            name="risk_manager",
            parameters={
                "lookback_period": 50,
                "max_var_95": 0.05,
                "max_drawdown": 0.20,
                "max_concentration": 0.30
            }
        )
        
        # 投资组合管理Agent配置
        self.portfolio_config = AgentConfig(
            name="portfolio_manager",
            parameters={
                "risk_free_rate": 0.02,
                "transaction_cost": 0.001,
                "rebalance_threshold": 0.05,
                "risk_aversion": 1.0
            }
        )
        
        # 创建Agent
        self.risk_agent = RiskManagementAgent(self.risk_config)
        self.portfolio_agent = PortfolioManagementAgent(self.portfolio_config)
        
        # 创建优化引擎
        self.optimization_engine = OptimizationEngine(risk_free_rate=0.02)
        
        print("✅ 管理Agent系统初始化完成")
    
    def test_risk_model(self):
        """测试风险模型"""
        print("\n📊 测试风险模型")
        
        risk_model = RiskModel(lookback_period=100)
        
        # 生成测试数据
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        prices = np.cumprod(1 + returns)
        
        # 测试VaR计算
        var_95 = risk_model.calculate_var(returns, 0.95)
        var_99 = risk_model.calculate_var(returns, 0.99)
        print(f"  VaR 95%: {var_95:.4f}")
        print(f"  VaR 99%: {var_99:.4f}")
        
        # 测试CVaR计算
        cvar_95 = risk_model.calculate_cvar(returns, 0.95)
        print(f"  CVaR 95%: {cvar_95:.4f}")
        
        # 测试回撤计算
        max_dd, current_dd = risk_model.calculate_max_drawdown(returns)
        print(f"  最大回撤: {max_dd:.4f}")
        print(f"  当前回撤: {current_dd:.4f}")
        
        # 测试夏普比率
        sharpe = risk_model.calculate_sharpe_ratio(returns)
        print(f"  夏普比率: {sharpe:.4f}")
        
        # 测试集中度风险
        positions = {"BTC": 0.4, "ETH": 0.3, "ADA": 0.2, "DOT": 0.1}
        concentration = risk_model.assess_concentration_risk(positions)
        print(f"  集中度风险: {concentration:.4f}")
        
        print("✅ 风险模型测试完成")
    
    def test_portfolio_optimizer(self):
        """测试投资组合优化器"""
        print("\n📈 测试投资组合优化器")
        
        optimizer = PortfolioOptimizer(
            risk_free_rate=0.02,
            transaction_cost=0.001,
            max_position_size=0.3,
            min_position_size=0.01
        )
        
        # 创建测试数据
        np.random.seed(42)
        n_assets = 4
        
        # 期望收益
        expected_returns = np.array([0.08, 0.12, 0.10, 0.06])
        print(f"  期望收益: {expected_returns}")
        
        # 协方差矩阵
        A = np.random.randn(n_assets, n_assets)
        covariance_matrix = np.dot(A, A.T) / 100
        # 确保正定性
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        eigenvalues = np.maximum(eigenvalues, 0.001)
        covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        print(f"  协方差矩阵对角线: {np.diag(covariance_matrix)}")
        
        # 测试马科维茨优化
        try:
            weights = optimizer.optimize_markowitz(expected_returns, covariance_matrix, risk_aversion=1.0)
            print(f"  马科维茨权重: {weights}")
            print(f"  权重和: {np.sum(weights):.6f}")
        except Exception as e:
            print(f"  马科维茨优化错误: {e}")
        
        # 测试风险平价优化
        try:
            rp_weights = optimizer.optimize_risk_parity(covariance_matrix)
            print(f"  风险平价权重: {rp_weights}")
            print(f"  权重和: {np.sum(rp_weights):.6f}")
        except Exception as e:
            print(f"  风险平价优化错误: {e}")
        
        print("✅ 投资组合优化器测试完成")
    
    async def test_optimization_engine(self):
        """测试优化引擎"""
        print("\n🔧 测试优化引擎")
        
        # 创建测试收益率数据
        np.random.seed(42)
        returns_data = {
            "BTCUSDT": np.random.normal(0.001, 0.02, 100).tolist(),
            "ETHUSDT": np.random.normal(0.0008, 0.025, 100).tolist(),
            "ADAUSDT": np.random.normal(0.0005, 0.03, 100).tolist(),
            "DOTUSDT": np.random.normal(0.0003, 0.028, 100).tolist()
        }
        
        print(f"  测试数据长度: {len(returns_data)}")
        
        # 测试不同优化类型
        optimization_types = [
            OptimizationType.MEAN_VARIANCE,
            OptimizationType.RISK_PARITY,
            OptimizationType.MINIMUM_VARIANCE,
            OptimizationType.EQUAL_WEIGHT
        ]
        
        for opt_type in optimization_types:
            try:
                result = await self.optimization_engine.optimize(
                    returns_data,
                    opt_type
                )
                
                if result.success:
                    print(f"  {opt_type.value}:")
                    print(f"    配置: {result.allocations}")
                    print(f"    预期收益: {result.expected_return:.4f}")
                    print(f"    风险: {result.risk:.4f}")
                    print(f"    夏普比率: {result.sharpe_ratio:.4f}")
                    if hasattr(result, 'diversification_ratio') and result.diversification_ratio:
                        print(f"    分散化比率: {result.diversification_ratio:.4f}")
                else:
                    print(f"  {opt_type.value}: 优化失败 - {result.message}")
            
            except Exception as e:
                print(f"  {opt_type.value}: 错误 - {e}")
        
        print("✅ 优化引擎测试完成")
    
    def create_sample_trading_state(self):
        """创建示例交易状态"""
        return TradingState(
            timestamp=datetime.now(),
            active_symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            market_data={
                "BTCUSDT": MarketData(
                    symbol="BTCUSDT",
                    timestamp=datetime.now(),
                    open=50000,
                    high=51000,
                    low=49000,
                    close=50500,
                    volume=1000
                ),
                "ETHUSDT": MarketData(
                    symbol="ETHUSDT",
                    timestamp=datetime.now(),
                    open=3000,
                    high=3100,
                    low=2900,
                    close=3050,
                    volume=2000
                ),
                "ADAUSDT": MarketData(
                    symbol="ADAUSDT",
                    timestamp=datetime.now(),
                    open=1.0,
                    high=1.1,
                    low=0.9,
                    close=1.05,
                    volume=5000
                )
            },
            positions={
                "BTCUSDT": Position(symbol="BTCUSDT", quantity=0.5, average_price=50000, current_price=50500),
                "ETHUSDT": Position(symbol="ETHUSDT", quantity=2.0, average_price=3000, current_price=3050),
                "ADAUSDT": Position(symbol="ADAUSDT", quantity=1000, average_price=1.0, current_price=1.05)
            }
        )
    
    async def test_risk_agent(self):
        """测试风险管理Agent"""
        print("\n🛡️ 测试风险管理Agent")
        
        # 创建交易状态
        trading_state = self.create_sample_trading_state()
        
        # 为TradingState添加session_id属性以兼容Risk Agent
        trading_state.session_id = "demo_session"
        
        # 添加历史数据
        self.risk_agent.price_history = {
            "BTCUSDT": [48000, 49000, 50000, 51000, 50500],
            "ETHUSDT": [2800, 2900, 3000, 3100, 3050],
            "ADAUSDT": [0.9, 0.95, 1.0, 1.1, 1.05]
        }
        
        try:
            # 执行风险评估
            assessment = await self.risk_agent.assess_risk(trading_state)
            
            print(f"  风险等级: {assessment.risk_level.value}")
            print(f"  风险分数: {assessment.risk_score:.2f}")
            print(f"  VaR 95%: {assessment.risk_metrics.var_95:.4f}")
            print(f"  最大回撤: {assessment.risk_metrics.max_drawdown:.4f}")
            print(f"  夏普比率: {assessment.risk_metrics.sharpe_ratio:.4f}")
            print(f"  集中度风险: {assessment.risk_metrics.concentration_risk:.4f}")
            
            print(f"  风险因子数量: {len(assessment.risk_factors)}")
            for factor in assessment.risk_factors[:3]:  # 显示前3个
                print(f"    - {factor.factor_name}: {factor.description}")
            
            print(f"  仓位建议: {assessment.position_recommendations}")
            print(f"  风险警告数量: {len(assessment.risk_warnings)}")
            
        except Exception as e:
            print(f"  风险评估错误: {e}")
            import traceback
            traceback.print_exc()
        
        print("✅ 风险管理Agent测试完成")
    
    async def test_portfolio_agent(self):
        """测试投资组合管理Agent"""
        print("\n💼 测试投资组合管理Agent")
        
        # 创建交易状态
        trading_state = self.create_sample_trading_state()
        
        # 添加历史收益率数据
        self.portfolio_agent.returns_history = {
            "BTCUSDT": [0.01, 0.02, -0.01, 0.015, 0.005],
            "ETHUSDT": [0.015, -0.005, 0.02, 0.01, 0.008],
            "ADAUSDT": [0.005, 0.01, 0.008, -0.002, 0.003]
        }
        
        try:
            # 计算当前权重
            current_weights = self.portfolio_agent._calculate_current_weights(trading_state)
            print(f"  当前权重: {current_weights}")
            
            # 测试决策融合
            mock_opinions = [
                {"symbol": "BTCUSDT", "recommendation": "BUY", "confidence": 0.8},
                {"symbol": "ETHUSDT", "recommendation": "HOLD", "confidence": 0.6},
                {"symbol": "ADAUSDT", "recommendation": "SELL", "confidence": 0.7}
            ]
            
            consensus = self.portfolio_agent._fuse_decisions(mock_opinions)
            print(f"  决策共识: {consensus}")
            
            # 计算投资组合指标
            metrics = self.portfolio_agent._calculate_portfolio_metrics(trading_state)
            print(f"  投资组合总价值: {metrics.total_value:.2f}")
            print(f"  预期收益: {metrics.expected_return:.4f}")
            print(f"  投资组合波动率: {metrics.portfolio_volatility:.4f}")
            print(f"  夏普比率: {metrics.sharpe_ratio:.4f}")
            print(f"  有效资产数量: {metrics.effective_assets}")
            
        except Exception as e:
            print(f"  投资组合管理错误: {e}")
        
        print("✅ 投资组合管理Agent测试完成")
    
    async def run_integration_test(self):
        """运行集成测试"""
        print("\n🔗 运行集成测试")
        
        trading_state = self.create_sample_trading_state()
        trading_state.session_id = "integration_test"
        
        # 为两个Agent添加历史数据
        historical_data = {
            "BTCUSDT": [48000, 49000, 50000, 51000, 50500],
            "ETHUSDT": [2800, 2900, 3000, 3100, 3050],
            "ADAUSDT": [0.9, 0.95, 1.0, 1.1, 1.05]
        }
        
        returns_data = {
            "BTCUSDT": [0.02, 0.02, 0.02, -0.01],
            "ETHUSDT": [0.035, 0.034, 0.033, -0.016],
            "ADAUSDT": [0.055, 0.05, 0.095, -0.045]
        }
        
        self.risk_agent.price_history = historical_data
        self.portfolio_agent.returns_history = returns_data
        
        try:
            # 1. 风险评估
            print("  执行风险评估...")
            risk_signals = await self.risk_agent.analyze(trading_state)
            print(f"    生成 {len(risk_signals)} 个风险信号")
            
            # 2. 投资组合优化
            print("  执行投资组合分析...")
            portfolio_signals = await self.portfolio_agent.analyze(trading_state)
            print(f"    生成 {len(portfolio_signals)} 个投资组合信号")
            
            # 3. 信号汇总
            all_signals = risk_signals + portfolio_signals
            print(f"  总信号数: {len(all_signals)}")
            
            for signal in all_signals[:5]:  # 显示前5个信号
                print(f"    {signal.source}: {signal.symbol} -> {signal.action} (强度: {signal.strength:.3f})")
            
        except Exception as e:
            print(f"  集成测试错误: {e}")
        
        print("✅ 集成测试完成")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("="*60)
        print("🧪 管理Agent系统测试开始")
        print("="*60)
        
        # 基础组件测试
        self.test_risk_model()
        self.test_portfolio_optimizer()
        await self.test_optimization_engine()
        
        # Agent测试
        await self.risk_agent._initialize()
        await self.portfolio_agent._initialize()
        
        await self.test_risk_agent()
        await self.test_portfolio_agent()
        
        # 集成测试
        await self.run_integration_test()
        
        print("\n" + "="*60)
        print("✅ 所有测试完成!")
        print("="*60)


async def main():
    """主函数"""
    try:
        demo = ManagementAgentDemo()
        await demo.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())