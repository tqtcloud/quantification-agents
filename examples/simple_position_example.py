#!/usr/bin/env python3
"""
简单的仓位管理示例

展示如何使用AutoPositionCloser进行基本的仓位管理。
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.position.models import PositionInfo, ClosingReason
from src.core.position.auto_position_closer import AutoPositionCloser
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength


async def basic_position_management_example():
    """基本仓位管理示例"""
    print("🚀 启动基本仓位管理示例")
    
    # 1. 创建自动平仓器
    config = {
        'monitoring_interval_seconds': 2,
        'enable_emergency_stop': True,
        'emergency_loss_threshold': -5.0,
        'strategies': {
            'profit_target': {
                'strategy_class': 'ProfitTargetStrategy',
                'parameters': {'target_profit_pct': 3.0, 'priority': 2},
                'enabled': True
            },
            'stop_loss': {
                'strategy_class': 'StopLossStrategy', 
                'parameters': {'stop_loss_pct': -2.0, 'priority': 1},
                'enabled': True
            }
        }
    }
    
    auto_closer = AutoPositionCloser(config)
    
    # 2. 创建示例仓位
    position = PositionInfo(
        position_id="DEMO_BTC_001",
        symbol="BTCUSDT",
        entry_price=50000.0,
        current_price=50000.0,
        quantity=0.5,
        side="long",
        entry_time=datetime.utcnow(),
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0
    )
    
    print(f"📍 创建仓位: {position.position_id}")
    print(f"   标的: {position.symbol}")
    print(f"   方向: {position.side}")
    print(f"   数量: {position.quantity}")
    print(f"   入场价: {position.entry_price}")
    
    # 3. 添加到自动平仓监控
    auto_closer.add_position(position)
    
    # 4. 模拟价格变化和监控
    print("\n📈 开始价格模拟和监控...")
    
    price_sequence = [50000, 50500, 51000, 51500, 50800, 49500, 49000, 48500]
    
    for i, new_price in enumerate(price_sequence):
        print(f"\n--- 第 {i+1} 轮监控 (价格: {new_price}) ---")
        
        # 检查平仓条件
        close_request = await auto_closer.manage_position(
            position_id=position.position_id,
            current_price=new_price
        )
        
        # 显示仓位状态
        current_pos = auto_closer.get_position(position.position_id)
        if current_pos:
            pnl_symbol = "💰" if current_pos.unrealized_pnl > 0 else "💸"
            print(f"   {pnl_symbol} 当前价格: {current_pos.current_price}")
            print(f"   💵 未实现盈亏: {current_pos.unrealized_pnl:.2f} ({current_pos.unrealized_pnl_pct:.2f}%)")
            print(f"   📊 最高价: {current_pos.highest_price:.2f}, 最低价: {current_pos.lowest_price:.2f}")
        
        # 处理平仓请求
        if close_request:
            print(f"   🎯 触发平仓条件: {close_request.closing_reason.value}")
            print(f"   ⚡ 紧急程度: {close_request.urgency}")
            print(f"   📋 元数据: {close_request.metadata}")
            
            # 执行平仓
            result = await auto_closer.execute_close_request(close_request)
            
            if result.success:
                profit_symbol = "📈" if result.realized_pnl > 0 else "📉"
                print(f"   ✅ 平仓成功 {profit_symbol}")
                print(f"   💰 已实现盈亏: {result.realized_pnl:.2f}")
                print(f"   🏷️ 平仓价格: {result.close_price:.2f}")
                print(f"   📅 平仓时间: {result.close_time}")
                break
            else:
                print(f"   ❌ 平仓失败: {result.error_message}")
        else:
            print("   ⏳ 无平仓触发条件")
        
        await asyncio.sleep(1)  # 等待1秒
    
    # 5. 显示最终统计
    print("\n📊 最终统计信息:")
    stats = auto_closer.get_statistics()
    print(f"   管理的仓位总数: {stats['total_managed']}")
    print(f"   已平仓数量: {stats['total_closed']}")
    print(f"   总利润: {stats['total_profit']:.2f}")
    print(f"   总亏损: {stats['total_loss']:.2f}")
    print(f"   净盈亏: {stats['net_pnl']:.2f}")
    print(f"   平仓成功率: {stats['close_success_rate']*100:.1f}%")
    
    # 显示策略统计
    print("\n🎯 策略触发统计:")
    for strategy_name, strategy_stat in stats['strategy_stats'].items():
        if strategy_stat['trigger_count'] > 0:
            print(f"   {strategy_name}: 触发 {strategy_stat['trigger_count']} 次, "
                  f"成功 {strategy_stat['success_count']} 次")


async def advanced_strategy_example():
    """高级策略示例"""
    print("\n\n🔬 高级策略配置示例")
    
    # 创建包含所有策略的配置
    advanced_config = {
        'monitoring_interval_seconds': 1,
        'enable_emergency_stop': True,
        'emergency_loss_threshold': -8.0,
    }
    
    auto_closer = AutoPositionCloser(advanced_config)
    
    # 显示可用策略
    print("📋 可用的平仓策略:")
    for strategy_name, strategy in auto_closer.strategies.items():
        status = "✅" if strategy.enabled else "❌"
        print(f"   {status} {strategy_name} (优先级: {strategy.priority})")
    
    # 动态调整策略参数
    print("\n⚙️ 动态调整策略参数:")
    
    # 更新盈利目标策略
    success = auto_closer.update_strategy_parameters('profit_target', {
        'target_profit_pct': 8.0,
        'partial_close_enabled': True,
        'first_partial_target': 4.0,
        'first_partial_pct': 60.0
    })
    if success:
        print("   ✅ 盈利目标策略参数已更新")
    
    # 更新止损策略
    success = auto_closer.update_strategy_parameters('stop_loss', {
        'stop_loss_pct': -1.5,
        'emergency_stop_pct': -3.0,
        'use_atr_stop': True
    })
    if success:
        print("   ✅ 止损策略参数已更新")
    
    # 禁用某个策略
    if auto_closer.disable_strategy('time_based'):
        print("   ⏸️ 时间止损策略已禁用")
    
    # 显示策略统计
    print("\n📊 策略详细信息:")
    for strategy_name in auto_closer.strategies:
        stats = auto_closer.get_strategy_statistics(strategy_name)
        if stats:
            enabled_status = "🟢" if stats['enabled'] else "🔴"
            print(f"   {enabled_status} {strategy_name}:")
            print(f"      优先级: {stats.get('priority', 'N/A')}")
            print(f"      触发次数: {stats['trigger_count']}")
            print(f"      成功次数: {stats['success_count']}")
            if stats['trigger_count'] > 0:
                success_rate = stats['success_count'] / stats['trigger_count'] * 100
                print(f"      成功率: {success_rate:.1f}%")


async def signal_integration_example():
    """信号集成示例"""
    print("\n\n📡 多维度信号集成示例")
    
    auto_closer = AutoPositionCloser()
    
    # 创建仓位
    position = PositionInfo(
        position_id="DEMO_ETH_001",
        symbol="ETHUSDT", 
        entry_price=3000.0,
        current_price=3000.0,
        quantity=1.0,
        side="long",
        entry_time=datetime.utcnow(),
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0
    )
    
    auto_closer.add_position(position)
    
    # 创建多维度信号
    trading_signal = TradingSignal(
        symbol="ETHUSDT",
        signal_type=SignalStrength.WEAK_SELL,  # 弱卖出信号
        confidence=0.7,
        entry_price=3000.0,
        target_price=2850.0,  # 5%下跌目标
        stop_loss=3150.0,     # 5%止损
        reasoning=["RSI超买", "均线死叉", "成交量萎缩"],
        indicators_consensus={"rsi": -0.8, "ma": -0.6, "volume": -0.4}
    )
    
    multi_signal = MultiDimensionalSignal(
        primary_signal=trading_signal,
        momentum_score=-0.6,        # 负动量（不利于多头）
        mean_reversion_score=0.3,   # 有一定回归倾向
        volatility_score=0.7,       # 高波动率
        volume_score=0.4,           # 成交量一般
        sentiment_score=-0.5,       # 负面情绪
        overall_confidence=0.65,    # 中等置信度
        risk_reward_ratio=1.8,      # 风险收益比
        max_position_size=0.6       # 建议最大仓位
    )
    
    print("📊 多维度信号信息:")
    print(f"   主信号类型: {multi_signal.primary_signal.signal_type.value}")
    print(f"   动量分数: {multi_signal.momentum_score}")
    print(f"   波动率分数: {multi_signal.volatility_score}")
    print(f"   情绪分数: {multi_signal.sentiment_score}")
    print(f"   综合置信度: {multi_signal.overall_confidence}")
    print(f"   信号质量分数: {multi_signal.signal_quality_score:.2f}")
    print(f"   方向一致性: {multi_signal.signal_direction_consensus:.2f}")
    
    # 使用信号检查平仓条件
    print("\n🔍 基于多维度信号检查平仓条件...")
    
    # 模拟价格上涨（与信号方向相反）
    position.update_price(3100.0)  # 3.33%涨幅
    
    close_request = await auto_closer.manage_position(
        position_id=position.position_id,
        current_price=position.current_price,
        signal=multi_signal
    )
    
    if close_request:
        print(f"   🎯 信号触发平仓: {close_request.closing_reason.value}")
        print(f"   📋 触发策略: {close_request.metadata.get('triggered_strategy', 'N/A')}")
        
        # 执行平仓
        result = await auto_closer.execute_close_request(close_request)
        if result.success:
            print(f"   ✅ 基于信号的平仓执行成功")
            print(f"   💰 盈亏: {result.realized_pnl:.2f}")
    else:
        print("   ⏳ 信号暂未触发平仓条件")


async def main():
    """主函数"""
    print("🎯 智能平仓系统示例集合")
    print("=" * 60)
    
    try:
        # 运行基本示例
        await basic_position_management_example()
        
        # 运行高级策略示例
        await advanced_strategy_example()
        
        # 运行信号集成示例
        await signal_integration_example()
        
        print("\n✅ 所有示例运行完成!")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())