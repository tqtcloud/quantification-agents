#!/bin/bash

# 快速策略管理系统验证脚本

set -e

echo "=== 快速策略管理系统验证 ==="

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📁 项目目录: $PROJECT_DIR"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export LOG_LEVEL="ERROR"  # 只显示错误信息

# 快速导入测试
echo "🔍 测试模块导入..."

python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')

try:
    # 测试策略管理器导入
    from src.strategy.strategy_manager import StrategyManager, StrategyConfig, StrategyType
    print('✅ StrategyManager 导入成功')
    
    # 测试资源分配器导入
    from src.strategy.resource_allocator import ResourceAllocator, ResourceLimit
    print('✅ ResourceAllocator 导入成功')
    
    # 测试监控器导入
    from src.strategy.strategy_monitor import StrategyMonitor, MonitoringLevel
    print('✅ StrategyMonitor 导入成功')
    
    # 测试配置管理器导入
    from src.strategy.config_manager import StrategyConfigManager
    print('✅ StrategyConfigManager 导入成功')
    
    # 测试整个模块导入
    from src.strategy import *
    print('✅ 策略模块完整导入成功')
    
    print('🎉 所有核心组件导入测试通过!')
    
except Exception as e:
    print(f'❌ 导入失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# 快速功能测试
echo "🧪 测试基本功能..."

python3 -c "
import sys, asyncio
sys.path.insert(0, '$PROJECT_DIR')

async def quick_test():
    try:
        from src.strategy.strategy_manager import StrategyManager, StrategyConfig, StrategyType
        from src.hft.hft_engine import HFTConfig
        
        # 创建基本配置
        config = StrategyConfig(
            strategy_id='test_quick',
            strategy_type=StrategyType.HFT,
            name='快速测试策略',
            hft_config=HFTConfig()
        )
        
        print('✅ 策略配置创建成功')
        
        # 测试状态枚举
        from src.strategy.strategy_manager import StrategyStatus
        print(f'✅ 策略状态枚举: {[s.value for s in StrategyStatus]}')
        
        print('🎉 基本功能测试通过!')
        
    except Exception as e:
        print(f'❌ 功能测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

# 运行测试
if asyncio.run(quick_test()):
    print('✅ 快速验证通过')
else:
    print('❌ 快速验证失败')
    exit(1)
"

echo "✅ 策略管理系统快速验证完成"
echo "💡 运行完整测试: ./scripts/test_strategy_manager.sh"
echo "💡 运行演示程序: ./scripts/run_strategy_demo.sh"