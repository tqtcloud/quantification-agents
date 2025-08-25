#!/usr/bin/env python3
"""
简化策略管理系统演示
只演示HFT策略，避免依赖问题
"""

import asyncio
import logging
from datetime import datetime

from src.strategy.strategy_manager import (
    StrategyManager, 
    StrategyConfig, 
    StrategyType, 
    StrategyStatus
)
from src.hft.hft_engine import HFTConfig
from src.core.message_bus import MessageBus

# 配置日志 - 减少详细输出
logging.basicConfig(
    level=logging.WARNING,  # 只显示WARNING及以上级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 为演示专门设置INFO级别
demo_logger = logging.getLogger(__name__)
demo_logger.setLevel(logging.INFO)
logger = demo_logger


async def main():
    """简化演示主函数"""
    logger.info("🚀 启动简化策略管理系统演示")
    
    # 创建组件
    message_bus = MessageBus()
    strategy_manager = StrategyManager(message_bus)
    
    try:
        # 初始化组件
        logger.info("\n⚙️ 初始化系统组件...")
        await strategy_manager.initialize()
        
        logger.info("✅ 系统初始化完成")
        
        # 创建HFT策略配置
        hft_config = StrategyConfig(
            strategy_id="demo_hft_simple",
            strategy_type=StrategyType.HFT,
            name="简化演示HFT策略",
            description="用于简化演示的高频交易策略",
            max_memory_mb=1024,
            max_cpu_percent=30.0,
            max_network_connections=100,
            priority=1,
            hft_config=HFTConfig(
                max_orderbook_levels=50,
                update_interval_ms=1.0,
                latency_target_ms=10.0
            )
        )
        
        logger.info(f"📝 创建策略配置: {hft_config.name}")
        
        # 注册策略
        strategy_id = await strategy_manager.register_strategy(hft_config)
        logger.info(f"✅ 策略注册成功: {strategy_id}")
        
        # 查看系统状态
        system_status = strategy_manager.get_system_status()
        logger.info(f"📊 系统状态: {system_status['total_strategies']}个策略")
        
        # 启动策略
        logger.info("\n🚀 启动策略...")
        success = await strategy_manager.start_strategy(strategy_id)
        
        if success:
            logger.info("✅ 策略启动成功")
            
            # 等待一段时间观察运行
            logger.info("⏳ 策略运行中...")
            await asyncio.sleep(3)
            
            # 查看策略状态
            status = strategy_manager.get_strategy_status(strategy_id)
            logger.info(f"📋 策略状态: {status['name']} - {status['status']}")
            
            # 查看资源状态
            resource_allocator = strategy_manager._resource_allocator
            resource_status = await resource_allocator.get_resource_status(strategy_id)
            logger.info(f"🏗️ 资源状态: 内存限制={resource_status['resource_limit']['memory_mb']}MB")
            
            # 查看监控指标
            strategy_monitor = strategy_manager._strategy_monitor
            metrics = strategy_monitor.get_strategy_metrics(strategy_id)
            if metrics:
                logger.info(f"📊 监控指标: CPU={metrics['current_cpu_usage']:.1f}%")
            
        else:
            logger.error("❌ 策略启动失败")
            return
        
        # 演示暂停和恢复
        logger.info("\n⏸️ 暂停策略...")
        await strategy_manager.pause_strategy(strategy_id)
        await asyncio.sleep(1)
        
        logger.info("▶️ 恢复策略...")
        await strategy_manager.resume_strategy(strategy_id)
        await asyncio.sleep(1)
        
        # 停止策略
        logger.info("\n🛑 停止策略...")
        await strategy_manager.stop_strategy(strategy_id)
        logger.info("✅ 策略已停止")
        
        # 注销策略
        logger.info("🧹 注销策略...")
        await strategy_manager.unregister_strategy(strategy_id)
        logger.info("✅ 策略已注销")
        
        # 显示最终系统状态
        final_status = strategy_manager.get_system_status()
        logger.info(f"\n📊 最终系统状态:")
        logger.info(f"   策略总数: {final_status['total_strategies']}")
        logger.info(f"   创建总数: {final_status['total_created']}")
        logger.info(f"   终止总数: {final_status['total_terminated']}")
        
    except Exception as e:
        logger.error(f"❌ 演示过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # 关闭组件
        logger.info("\n🔧 关闭系统组件...")
        try:
            await strategy_manager.shutdown()
            message_bus.close()
            logger.info("✅ 系统关闭完成")
        except Exception as e:
            logger.error(f"❌ 关闭系统时发生错误: {e}")
    
    logger.info("\n🎉 简化策略管理系统演示完成!")


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行演示
    asyncio.run(main())