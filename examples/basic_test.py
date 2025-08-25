#!/usr/bin/env python3
"""
最基础的策略管理系统验证
不启动HFT引擎，只验证管理功能
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.strategy.strategy_manager import (
    StrategyManager, 
    StrategyConfig, 
    StrategyType, 
    StrategyStatus
)

async def basic_test():
    """基础测试"""
    print("🧪 开始基础策略管理测试...")
    
    try:
        # 创建策略配置
        config = StrategyConfig(
            strategy_id="basic_test",
            strategy_type=StrategyType.HFT,
            name="基础测试策略",
            description="用于基础功能测试",
            max_memory_mb=512,
            max_cpu_percent=20.0
        )
        
        print("✅ 策略配置创建成功")
        
        # 创建管理器（不启动）
        manager = StrategyManager()
        print("✅ 策略管理器创建成功")
        
        # 不调用initialize，避免启动复杂组件
        
        # 验证基本功能
        print(f"📊 系统状态验证...")
        
        # 验证空策略列表
        strategies = manager.list_strategies()
        print(f"📋 初始策略列表: {len(strategies)} 个策略")
        
        # 验证系统状态
        status = manager.get_system_status()
        print(f"📊 系统状态: {status['total_strategies']} 个策略")
        
        print("🎉 基础功能验证通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(basic_test())
    if success:
        print("\n✅ 策略管理系统基础功能正常!")
        exit(0)
    else:
        print("\n❌ 测试失败!")
        exit(1)