#!/usr/bin/env python3
"""
信号模型使用示例

演示如何使用SignalStrength、TradingSignal和MultiDimensionalSignal类
"""

from datetime import datetime
from src.core.models.signals import (
    SignalStrength,
    TradingSignal,
    MultiDimensionalSignal,
    SignalAggregator,
)


def demo_basic_trading_signal():
    """演示基础交易信号的创建和使用"""
    print("=== 基础交易信号示例 ===")
    
    # 创建一个买入信号
    buy_signal = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.85,
        entry_price=50000.0,
        target_price=55000.0,
        stop_loss=48000.0,
        reasoning=[
            "RSI从超卖区域反弹",
            "MACD金叉形成",
            "成交量放大确认突破"
        ],
        indicators_consensus={
            "RSI": 0.7,
            "MACD": 0.8,
            "Volume": 0.9,
            "Moving_Average": 0.75
        }
    )
    
    print(f"信号标的: {buy_signal.symbol}")
    print(f"信号类型: {buy_signal.signal_type.name}")
    print(f"置信度: {buy_signal.confidence}")
    print(f"入场价格: ${buy_signal.entry_price:,}")
    print(f"目标价格: ${buy_signal.target_price:,}")
    print(f"止损价格: ${buy_signal.stop_loss:,}")
    print(f"风险收益比: {buy_signal.risk_reward_ratio:.2f}")
    print(f"信号有效性: {buy_signal.is_valid}")
    print(f"推理逻辑: {', '.join(buy_signal.reasoning)}")
    print()


def demo_multidimensional_signal():
    """演示多维度信号的创建和分析"""
    print("=== 多维度信号示例 ===")
    
    # 先创建基础信号
    primary_signal = TradingSignal(
        symbol="ETHUSDT",
        signal_type=SignalStrength.STRONG_BUY,
        confidence=0.9,
        entry_price=3000.0,
        target_price=3300.0,
        stop_loss=2850.0,
        reasoning=[
            "突破关键阻力位",
            "多重技术指标确认",
            "市场情绪转向乐观"
        ],
        indicators_consensus={
            "RSI": 0.75,
            "MACD": 0.85,
            "Bollinger": 0.8,
            "Volume_Profile": 0.9
        }
    )
    
    # 创建多维度信号
    multi_signal = MultiDimensionalSignal(
        primary_signal=primary_signal,
        momentum_score=0.8,           # 强劲上涨动量
        mean_reversion_score=-0.3,    # 不是均值回归机会
        volatility_score=0.4,         # 中等波动率
        volume_score=0.9,             # 高成交量确认
        sentiment_score=0.7,          # 积极市场情绪
        overall_confidence=0.88,      # 综合置信度
        risk_reward_ratio=2.0,        # 1:2风险收益比
        max_position_size=0.4,        # 最大40%仓位
        market_regime="上涨趋势",
        technical_levels={
            "support": 2950.0,
            "resistance": 3350.0,
            "pivot": 3150.0
        }
    )
    
    print(f"主要信号: {multi_signal.primary_signal.symbol} - {multi_signal.primary_signal.signal_type.name}")
    print(f"动量评分: {multi_signal.momentum_score:+.2f}")
    print(f"均值回归评分: {multi_signal.mean_reversion_score:+.2f}")
    print(f"波动率评分: {multi_signal.volatility_score:.2f}")
    print(f"成交量评分: {multi_signal.volume_score:.2f}")
    print(f"情绪评分: {multi_signal.sentiment_score:+.2f}")
    print(f"综合置信度: {multi_signal.overall_confidence:.2f}")
    print(f"信号质量评分: {multi_signal.signal_quality_score:.2f}")
    print(f"方向一致性: {multi_signal.signal_direction_consensus:+.2f}")
    print(f"市场状态: {multi_signal.market_regime}")
    print()
    
    # 仓位建议
    base_position = 1.0
    risk_tolerance = 0.8
    recommended_size = multi_signal.get_position_sizing_recommendation(
        base_position, risk_tolerance
    )
    print(f"仓位建议: {recommended_size:.2%} (基于{base_position:.0%}基础仓位和{risk_tolerance:.0%}风险容忍度)")
    print()


def demo_signal_aggregation():
    """演示信号聚合功能"""
    print("=== 信号聚合示例 ===")
    
    # 创建多个信号源的信号
    signals = []
    
    # 技术分析信号
    tech_signal = MultiDimensionalSignal(
        primary_signal=TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.75,
            entry_price=50000.0,
            target_price=52000.0,
            stop_loss=48500.0,
            reasoning=["技术分析看多"],
            indicators_consensus={"RSI": 0.7, "MACD": 0.6}
        ),
        momentum_score=0.6,
        mean_reversion_score=0.1,
        volatility_score=0.3,
        volume_score=0.7,
        sentiment_score=0.5,
        overall_confidence=0.75,
        risk_reward_ratio=1.33,
        max_position_size=0.3
    )
    
    # 基本面分析信号
    fundamental_signal = MultiDimensionalSignal(
        primary_signal=TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.STRONG_BUY,
            confidence=0.9,
            entry_price=50000.0,
            target_price=54000.0,
            stop_loss=47000.0,
            reasoning=["基本面强劲"],
            indicators_consensus={"Adoption": 0.9, "Institutional": 0.8}
        ),
        momentum_score=0.8,
        mean_reversion_score=-0.1,
        volatility_score=0.2,
        volume_score=0.8,
        sentiment_score=0.9,
        overall_confidence=0.9,
        risk_reward_ratio=1.33,
        max_position_size=0.5
    )
    
    # 量化模型信号
    quant_signal = MultiDimensionalSignal(
        primary_signal=TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.WEAK_BUY,
            confidence=0.65,
            entry_price=50000.0,
            target_price=51500.0,
            stop_loss=49000.0,
            reasoning=["量化模型预测"],
            indicators_consensus={"ML_Model": 0.65, "Statistical": 0.6}
        ),
        momentum_score=0.4,
        mean_reversion_score=0.3,
        volatility_score=0.4,
        volume_score=0.6,
        sentiment_score=0.3,
        overall_confidence=0.65,
        risk_reward_ratio=1.5,
        max_position_size=0.25
    )
    
    signals.extend([tech_signal, fundamental_signal, quant_signal])
    
    # 过滤低质量信号
    high_quality_signals = SignalAggregator.filter_signals_by_quality(
        signals,
        min_quality_score=0.5,
        min_confidence=0.7
    )
    
    print(f"原始信号数量: {len(signals)}")
    print(f"高质量信号数量: {len(high_quality_signals)}")
    
    # 组合信号
    combined_signal = SignalAggregator.combine_signals(
        high_quality_signals,
        weights={"signal_0": 0.3, "signal_1": 0.5, "signal_2": 0.2}  # 基本面权重最高
    )
    
    if combined_signal:
        print(f"\n组合信号结果:")
        print(f"- 主信号来源: 置信度最高的信号 ({combined_signal.primary_signal.confidence:.2f})")
        print(f"- 综合置信度: {combined_signal.overall_confidence:.2f}")
        print(f"- 综合质量评分: {combined_signal.signal_quality_score:.2f}")
        print(f"- 建议最大仓位: {combined_signal.max_position_size:.1%}")
        print(f"- 方向一致性: {combined_signal.signal_direction_consensus:+.2f}")


def demo_error_handling():
    """演示错误处理和数据验证"""
    print("=== 错误处理示例 ===")
    
    try:
        # 尝试创建无效的信号
        invalid_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=1.5,  # 超出范围
            entry_price=50000.0,
            target_price=55000.0,
            stop_loss=48000.0,
            reasoning=["测试"],
            indicators_consensus={}
        )
    except ValueError as e:
        print(f"捕获到预期的错误: {e}")
    
    try:
        # 尝试创建逻辑错误的买入信号
        TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=45000.0,  # 目标价低于入场价
            stop_loss=48000.0,
            reasoning=["测试"],
            indicators_consensus={}
        )
    except ValueError as e:
        print(f"捕获到逻辑错误: {e}")
    
    print("错误处理测试完成")
    print()


def main():
    """主函数：运行所有示例"""
    print("🚀 信号模型示例演示")
    print("=" * 50)
    print()
    
    demo_basic_trading_signal()
    demo_multidimensional_signal()
    demo_signal_aggregation()
    demo_error_handling()
    
    print("✅ 所有示例演示完成！")
    print("\n📝 关键特性总结:")
    print("1. ✅ 强类型定义的信号数据结构")
    print("2. ✅ 自动数据验证和错误处理")
    print("3. ✅ 多维度市场分析支持")
    print("4. ✅ 智能信号聚合和过滤")
    print("5. ✅ 仓位管理建议算法")
    print("6. ✅ 风险收益比自动计算")


if __name__ == "__main__":
    main()