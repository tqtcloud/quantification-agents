#!/usr/bin/env python3
"""
快速测试信号聚合器系统
"""

import sys
import asyncio
from datetime import datetime
sys.path.append('.')

from src.strategy.signal_aggregator import (
    SignalAggregator, AggregationStrategy, SignalSource,
    SignalInput, AggregationConfig
)
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength

async def quick_test():
    print("🧪 快速测试信号聚合器")
    
    # 创建配置
    config = AggregationConfig(
        strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        min_signal_count=2,
        min_confidence_threshold=0.5,
        min_quality_threshold=0.4,
        time_window_seconds=3600  # 1小时窗口
    )
    
    # 创建聚合器
    aggregator = SignalAggregator(config=config)
    
    # 创建测试信号，手动设置时间戳
    current_time = datetime.now()
    
    signal1_data = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.85,
        entry_price=50000.0,
        target_price=52000.0,
        stop_loss=48000.0,
        reasoning=["HFT信号"],
        indicators_consensus={"rsi": 0.7}
    )
    signal1_data.timestamp = current_time  # 手动设置时间戳
    
    signal2_data = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=0.75,
        entry_price=50000.0,
        target_price=51500.0,
        stop_loss=48500.0,
        reasoning=["AI信号"],
        indicators_consensus={"macd": 0.6}
    )
    signal2_data.timestamp = current_time  # 手动设置时间戳
    
    signal1 = MultiDimensionalSignal(
        primary_signal=signal1_data,
        momentum_score=0.8,
        mean_reversion_score=0.2,
        volatility_score=0.3,
        volume_score=0.7,
        sentiment_score=0.5,
        overall_confidence=0.85,
        risk_reward_ratio=2.0,
        max_position_size=0.1
    )
    
    signal2 = MultiDimensionalSignal(
        primary_signal=signal2_data,
        momentum_score=0.6,
        mean_reversion_score=0.3,
        volatility_score=0.35,
        volume_score=0.6,
        sentiment_score=0.4,
        overall_confidence=0.75,
        risk_reward_ratio=1.8,
        max_position_size=0.12
    )
    
    # 创建信号输入
    signal_inputs = [
        SignalInput(
            signal_id="hft_signal_1",
            signal=signal1,
            source_type=SignalSource.HFT_ENGINE,
            source_id="hft_001",
            priority=0.8,
            received_at=current_time  # 设置接收时间
        ),
        SignalInput(
            signal_id="ai_signal_1",
            signal=signal2,
            source_type=SignalSource.AI_AGENT,
            source_id="ai_001",
            priority=0.6,
            received_at=current_time  # 设置接收时间
        )
    ]
    
    print(f"✅ 创建了 {len(signal_inputs)} 个信号输入")
    
    # 验证输入
    validation_result = aggregator._validate_inputs(signal_inputs)
    print(f"📝 输入验证结果: {validation_result}")
    
    if not validation_result:
        print("❌ 验证失败，退出测试")
        return False
    
    # 启动聚合器
    await aggregator.start()
    
    try:
        # 执行聚合
        result = await aggregator.aggregate_signals(signal_inputs)
        
        print(f"✅ 信号聚合成功!")
        print(f"   聚合ID: {result.aggregation_id}")
        print(f"   聚合策略: {result.strategy_used.value}")
        print(f"   质量分数: {result.quality_score:.3f}")
        print(f"   置信度调整: {result.confidence_adjustment:.3f}")
        print(f"   处理时间: {result.processing_time_ms:.2f}ms")
        print(f"   冲突检测: {len(result.conflicts_detected)} 个")
        
        if result.aggregated_signal:
            aggregated = result.aggregated_signal
            print(f"   聚合信号置信度: {aggregated.overall_confidence:.3f}")
            print(f"   聚合信号质量: {aggregated.signal_quality_score:.3f}")
        
        return True
        
    finally:
        await aggregator.stop()

if __name__ == "__main__":
    result = asyncio.run(quick_test())
    print(f"\n🎯 测试结果: {'通过' if result else '失败'}")