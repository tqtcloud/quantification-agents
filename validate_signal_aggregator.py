#!/usr/bin/env python3
"""
信号聚合器系统集成验证脚本

验证所有组件是否能正确导入和初始化
"""

import sys
import asyncio
import traceback
from datetime import datetime

# 添加项目路径
sys.path.append('.')

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_info(message):
    print(f"ℹ️  {message}")

def test_imports():
    """测试模块导入"""
    print_header("测试模块导入")
    
    try:
        # 测试核心模型导入
        from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
        print_success("核心信号模型导入成功")
        
        # 测试冲突解决器导入
        from src.strategy.conflict_resolver import (
            ConflictResolver, ConflictType, ConflictSeverity, 
            ConflictResolutionStrategy
        )
        print_success("冲突解决器导入成功")
        
        # 测试优先级管理器导入
        from src.strategy.priority_manager import (
            PriorityManager, PriorityCategory, MarketCondition
        )
        print_success("优先级管理器导入成功")
        
        # 测试信号聚合器导入
        from src.strategy.signal_aggregator import (
            SignalAggregator, AggregationStrategy, SignalSource,
            SignalInput, AggregationResult, UnifiedSignalInterface
        )
        print_success("信号聚合器导入成功")
        
        return True
        
    except Exception as e:
        print_error(f"模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_signal_creation():
    """测试信号创建"""
    print_header("测试信号创建")
    
    try:
        from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
        
        # 创建基础交易信号
        trading_signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=52000.0,
            stop_loss=48000.0,
            reasoning=["技术指标显示买入信号"],
            indicators_consensus={"rsi": 0.7, "macd": 0.6}
        )
        print_success(f"创建交易信号成功: {trading_signal.symbol}")
        
        # 创建多维度信号
        multidimensional_signal = MultiDimensionalSignal(
            primary_signal=trading_signal,
            momentum_score=0.7,
            mean_reversion_score=0.2,
            volatility_score=0.3,
            volume_score=0.6,
            sentiment_score=0.4,
            overall_confidence=0.8,
            risk_reward_ratio=2.5,
            max_position_size=0.1
        )
        print_success(f"创建多维度信号成功，质量分数: {multidimensional_signal.signal_quality_score:.2f}")
        
        return True
        
    except Exception as e:
        print_error(f"信号创建失败: {e}")
        traceback.print_exc()
        return False

def test_conflict_resolver():
    """测试冲突解决器"""
    print_header("测试冲突解决器")
    
    try:
        from src.strategy.conflict_resolver import ConflictResolver
        from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
        
        # 创建冲突解决器
        resolver = ConflictResolver()
        print_success("冲突解决器初始化成功")
        
        # 创建冲突信号
        buy_signal_data = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.STRONG_BUY,
            confidence=0.9,
            entry_price=50000.0,
            target_price=55000.0,
            stop_loss=45000.0,
            reasoning=["强烈买入信号"],
            indicators_consensus={"rsi": 0.8}
        )
        
        sell_signal_data = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.STRONG_SELL,
            confidence=0.85,
            entry_price=50000.0,
            target_price=45000.0,
            stop_loss=55000.0,
            reasoning=["强烈卖出信号"],
            indicators_consensus={"rsi": 0.2}
        )
        
        buy_signal = MultiDimensionalSignal(
            primary_signal=buy_signal_data,
            momentum_score=0.8,
            mean_reversion_score=0.1,
            volatility_score=0.4,
            volume_score=0.7,
            sentiment_score=0.6,
            overall_confidence=0.9,
            risk_reward_ratio=2.0,
            max_position_size=0.15
        )
        
        sell_signal = MultiDimensionalSignal(
            primary_signal=sell_signal_data,
            momentum_score=-0.8,
            mean_reversion_score=0.1,
            volatility_score=0.4,
            volume_score=0.7,
            sentiment_score=-0.6,
            overall_confidence=0.85,
            risk_reward_ratio=2.0,
            max_position_size=0.15
        )
        
        # 检测冲突
        signals = [buy_signal, sell_signal]
        conflicts = resolver.detect_conflicts(signals)
        print_success(f"检测到 {len(conflicts)} 个冲突")
        
        if conflicts:
            # 解决冲突
            resolutions = resolver.resolve_conflicts(signals, conflicts)
            print_success(f"解决了 {len(resolutions)} 个冲突")
            
            for resolution in resolutions:
                print_info(f"冲突解决: {resolution.strategy_used}")
        
        # 获取统计信息
        stats = resolver.get_conflict_statistics()
        print_success(f"获取统计信息成功: 总冲突数 {stats['total_conflicts']}")
        
        return True
        
    except Exception as e:
        print_error(f"冲突解决器测试失败: {e}")
        traceback.print_exc()
        return False

def test_priority_manager():
    """测试优先级管理器"""
    print_header("测试优先级管理器")
    
    try:
        from src.strategy.priority_manager import PriorityManager, MarketCondition
        from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength
        
        # 创建优先级管理器
        manager = PriorityManager()
        print_success("优先级管理器初始化成功")
        
        # 注册信号源
        success = manager.register_signal_source(
            source_id="hft_001",
            source_name="HFT引擎1",
            source_type="HFT",
            base_priority=0.8
        )
        print_success(f"注册HFT信号源: {success}")
        
        success = manager.register_signal_source(
            source_id="ai_001",
            source_name="AI代理1",
            source_type="AI_AGENT",
            base_priority=0.6
        )
        print_success(f"注册AI信号源: {success}")
        
        # 创建测试信号
        signal_data = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=52000.0,
            stop_loss=48000.0,
            reasoning=["测试信号"],
            indicators_consensus={"rsi": 0.7}
        )
        
        test_signal = MultiDimensionalSignal(
            primary_signal=signal_data,
            momentum_score=0.7,
            mean_reversion_score=0.2,
            volatility_score=0.3,
            volume_score=0.6,
            sentiment_score=0.4,
            overall_confidence=0.8,
            risk_reward_ratio=2.5,
            max_position_size=0.1
        )
        
        # 获取信号优先级
        priority = manager.get_signal_priority(test_signal, "hft_001")
        print_success(f"获取HFT信号优先级: {priority:.3f}")
        
        priority = manager.get_signal_priority(test_signal, "ai_001")
        print_success(f"获取AI信号优先级: {priority:.3f}")
        
        # 更新性能数据
        performance_data = {
            'return': 0.05,
            'accuracy': 0.8,
            'trade_count': 10
        }
        success = manager.update_performance("hft_001", performance_data)
        print_success(f"更新HFT性能数据: {success}")
        
        # 更新市场条件
        market_condition = MarketCondition(
            volatility_level="high",
            trend_strength=0.8,
            volume_profile="high",
            market_stress=0.3,
            sentiment_score=0.6
        )
        manager.update_market_condition(market_condition)
        print_success("更新市场条件成功")
        
        # 获取优先级排名
        rankings = manager.get_priority_rankings()
        print_success(f"获取优先级排名: {len(rankings)} 个信号源")
        
        for source_id, priority in rankings:
            print_info(f"  {source_id}: {priority:.3f}")
        
        # 获取统计信息
        stats = manager.get_priority_statistics()
        print_success(f"获取统计信息: {stats['total_sources']} 个信号源")
        
        return True
        
    except Exception as e:
        print_error(f"优先级管理器测试失败: {e}")
        traceback.print_exc()
        return False

async def test_signal_aggregator():
    """测试信号聚合器"""
    print_header("测试信号聚合器")
    
    try:
        from src.strategy.signal_aggregator import (
            SignalAggregator, AggregationStrategy, SignalSource,
            SignalInput, AggregationConfig
        )
        from src.strategy.conflict_resolver import ConflictResolver
        from src.strategy.priority_manager import PriorityManager
        from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength
        
        # 创建配置
        config = AggregationConfig(
            strategy=AggregationStrategy.HYBRID_FUSION,
            min_signal_count=2,
            min_confidence_threshold=0.5,
            min_quality_threshold=0.4,
            time_window_seconds=600  # 10分钟窗口
        )
        
        # 创建聚合器
        aggregator = SignalAggregator(config=config)
        print_success("信号聚合器初始化成功")
        
        # 创建测试信号
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
                priority=0.8
            ),
            SignalInput(
                signal_id="ai_signal_1",
                signal=signal2,
                source_type=SignalSource.AI_AGENT,
                source_id="ai_001",
                priority=0.6
            )
        ]
        
        # 启动聚合器
        await aggregator.start()
        print_success("信号聚合器启动成功")
        
        try:
            # 调试信号信息
            print_info("调试信号信息:")
            for i, si in enumerate(signal_inputs):
                print_info(f"  信号{i+1}: 置信度={si.signal.overall_confidence:.3f}, 质量={si.signal.signal_quality_score:.3f}")
                print_info(f"    时间戳: {si.signal.primary_signal.timestamp}")
                print_info(f"    接收时间: {si.received_at}")
            
            # 手动验证输入
            validation_result = aggregator._validate_inputs(signal_inputs)
            print_info(f"手动验证结果: {validation_result}")
            
            # 执行信号聚合
            result = await aggregator.aggregate_signals(signal_inputs)
            print_success(f"信号聚合成功: {result.aggregation_id}")
            print_info(f"  聚合策略: {result.strategy_used}")
            print_info(f"  质量分数: {result.quality_score:.3f}")
            print_info(f"  置信度调整: {result.confidence_adjustment:.3f}")
            print_info(f"  处理时间: {result.processing_time_ms:.2f}ms")
            print_info(f"  冲突检测: {len(result.conflicts_detected)} 个")
            print_info(f"  冲突解决: {len(result.conflicts_resolved)} 个")
            
            if result.aggregated_signal:
                aggregated = result.aggregated_signal
                print_info(f"  聚合信号置信度: {aggregated.overall_confidence:.3f}")
                print_info(f"  聚合信号质量: {aggregated.signal_quality_score:.3f}")
                print_info(f"  聚合信号方向: {aggregated.primary_signal.signal_type}")
            
            # 获取统计信息
            stats = aggregator.get_aggregation_statistics()
            print_success(f"获取统计信息: 成功率 {stats.get('success_rate', 0):.2f}")
            
        finally:
            # 停止聚合器
            await aggregator.stop()
            print_success("信号聚合器停止成功")
        
        return True
        
    except Exception as e:
        print_error(f"信号聚合器测试失败: {e}")
        traceback.print_exc()
        return False

async def test_unified_interface():
    """测试统一信号接口"""
    print_header("测试统一信号接口")
    
    try:
        from src.strategy.signal_aggregator import UnifiedSignalInterface, AggregationResult, AggregationStrategy
        
        # 创建接口
        interface = UnifiedSignalInterface()
        print_success("统一信号接口创建成功")
        
        # 回调结果收集
        callback_results = []
        error_results = []
        
        # 定义回调函数
        def signal_callback(result):
            callback_results.append(result)
            print_info(f"接收到信号: {result.aggregation_id}")
        
        def error_callback(error, context):
            error_results.append((error, context))
            print_info(f"接收到错误: {error}")
        
        # 注册回调
        interface.register_signal_callback(signal_callback)
        interface.register_error_callback(error_callback)
        print_success("回调函数注册成功")
        
        # 创建测试结果
        test_result = AggregationResult(
            aggregation_id="test_aggregation",
            aggregated_signal=None,
            input_signals=[],
            conflicts_detected=[],
            conflicts_resolved=[],
            strategy_used=AggregationStrategy.WEIGHTED_AVERAGE,
            confidence_adjustment=0.0,
            quality_score=0.8,
            reasoning=["测试聚合"],
            processing_time_ms=5.0
        )
        
        # 发送信号
        await interface.emit_signal(test_result)
        print_success("信号发送成功")
        
        # 发送错误
        test_error = ValueError("测试错误")
        test_context = {"test": "context"}
        await interface.emit_error(test_error, test_context)
        print_success("错误发送成功")
        
        # 验证回调
        assert len(callback_results) == 1, "信号回调应被调用一次"
        assert len(error_results) == 1, "错误回调应被调用一次"
        assert callback_results[0] == test_result, "回调结果应匹配"
        
        print_success("统一信号接口测试通过")
        return True
        
    except Exception as e:
        print_error(f"统一信号接口测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """集成测试"""
    print_header("系统集成验证")
    
    try:
        from src.strategy.signal_aggregator import SignalAggregator
        from src.strategy.conflict_resolver import ConflictResolver
        from src.strategy.priority_manager import PriorityManager
        
        # 创建完整系统
        conflict_resolver = ConflictResolver()
        priority_manager = PriorityManager()
        
        # 注册信号源
        priority_manager.register_signal_source("hft_001", "HFT引擎1", "HFT", 0.8)
        priority_manager.register_signal_source("ai_001", "AI代理1", "AI_AGENT", 0.6)
        
        aggregator = SignalAggregator(
            conflict_resolver=conflict_resolver,
            priority_manager=priority_manager
        )
        
        print_success("完整系统创建成功")
        
        # 验证组件连接
        assert aggregator.conflict_resolver == conflict_resolver
        assert aggregator.priority_manager == priority_manager
        print_success("组件连接验证成功")
        
        # 验证配置
        assert aggregator.config is not None
        print_success("配置验证成功")
        
        # 验证统计接口
        stats = aggregator.get_aggregation_statistics()
        assert isinstance(stats, dict)
        print_success("统计接口验证成功")
        
        return True
        
    except Exception as e:
        print_error(f"集成测试失败: {e}")
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print_header("信号聚合器系统验证")
    print_info(f"验证时间: {datetime.now().isoformat()}")
    
    # 测试结果跟踪
    test_results = {}
    
    # 执行各项测试
    test_results['imports'] = test_imports()
    test_results['signal_creation'] = test_signal_creation()
    test_results['conflict_resolver'] = test_conflict_resolver()
    test_results['priority_manager'] = test_priority_manager()
    test_results['signal_aggregator'] = await test_signal_aggregator()
    test_results['unified_interface'] = await test_unified_interface()
    test_results['integration'] = test_integration()
    
    # 总结结果
    print_header("验证结果总结")
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if passed:
            passed_count += 1
    
    print(f"\n总体结果: {passed_count}/{total_count} 测试通过")
    
    if passed_count == total_count:
        print_success("🎉 所有测试通过！信号聚合器系统验证成功！")
        return True
    else:
        print_error(f"❌ 有 {total_count - passed_count} 个测试失败")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n验证被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n验证过程中发生异常: {e}")
        traceback.print_exc()
        sys.exit(1)