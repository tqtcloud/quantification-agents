#!/usr/bin/env python3
"""
策略管理系统演示
展示双策略管理和隔离系统的完整功能
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

from src.strategy.strategy_manager import (
    StrategyManager, 
    StrategyConfig, 
    StrategyType, 
    StrategyStatus
)
from src.strategy.config_manager import StrategyConfigManager
from src.hft.hft_engine import HFTConfig
from src.agents.orchestrator import WorkflowConfig
from src.core.message_bus import MessageBus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_sample_configs(config_manager: StrategyConfigManager):
    """创建示例配置"""
    logger.info("=== 创建示例配置 ===")
    
    # 创建高频交易策略配置
    hft_config = StrategyConfig(
        strategy_id="demo_hft_strategy",
        strategy_type=StrategyType.HFT,
        name="演示高频交易策略",
        description="用于演示的高频交易策略，专注于市场微结构分析",
        max_memory_mb=2048,
        max_cpu_percent=50.0,
        max_network_connections=200,
        priority=2,
        auto_restart=True,
        max_restarts=5,
        hft_config=HFTConfig(
            max_orderbook_levels=100,
            orderbook_history_size=2000,
            microstructure_lookback=200,
            min_signal_strength=0.3,
            max_orders=20000,
            update_interval_ms=0.5,  # 更高频率更新
            latency_target_ms=5.0    # 更严格的延迟要求
        )
    )
    
    # 创建AI智能策略配置
    ai_config = StrategyConfig(
        strategy_id="demo_ai_strategy",
        strategy_type=StrategyType.AI_AGENT,
        name="演示AI智能策略",
        description="用于演示的AI智能策略，基于多Agent协作决策",
        max_memory_mb=1024,
        max_cpu_percent=30.0,
        max_network_connections=100,
        priority=1,
        auto_restart=True,
        max_restarts=3,
        workflow_config=WorkflowConfig(
            max_parallel_agents=8,
            enable_checkpointing=True,
            checkpoint_interval=3,
            timeout_seconds=600,
            retry_failed_nodes=True,
            max_retries=2,
            aggregation_method="weighted_voting",
            consensus_threshold=0.7
        )
    )
    
    # 保存配置
    await config_manager.save_config(hft_config)
    await config_manager.save_config(ai_config)
    
    logger.info(f"✅ 创建HFT策略配置: {hft_config.name}")
    logger.info(f"✅ 创建AI策略配置: {ai_config.name}")
    
    return hft_config, ai_config


async def demonstrate_strategy_lifecycle(strategy_manager: StrategyManager, configs):
    """演示策略生命周期管理"""
    logger.info("\n=== 策略生命周期管理演示 ===")
    
    hft_config, ai_config = configs
    strategy_ids = []
    
    try:
        # 1. 注册策略
        logger.info("📝 注册策略...")
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        strategy_ids = [hft_id, ai_id]
        
        logger.info(f"✅ HFT策略注册成功: {hft_id}")
        logger.info(f"✅ AI策略注册成功: {ai_id}")
        
        # 2. 查看系统状态
        system_status = strategy_manager.get_system_status()
        logger.info(f"📊 系统状态: {system_status['total_strategies']}个策略, "
                   f"{system_status['running_strategies']}个运行中")
        
        # 3. 启动策略
        logger.info("\n🚀 启动策略...")
        hft_success = await strategy_manager.start_strategy(hft_id)
        ai_success = await strategy_manager.start_strategy(ai_id)
        
        if hft_success and ai_success:
            logger.info("✅ 所有策略启动成功")
            
            # 等待策略运行一段时间
            logger.info("⏳ 等待策略运行...")
            await asyncio.sleep(5)
            
            # 4. 检查策略状态
            for strategy_id in strategy_ids:
                status = strategy_manager.get_strategy_status(strategy_id)
                logger.info(f"📋 {status['name']}: {status['status']}")
            
            # 5. 演示暂停和恢复
            logger.info(f"\n⏸️ 暂停HFT策略...")
            await strategy_manager.pause_strategy(hft_id)
            await asyncio.sleep(2)
            
            logger.info(f"▶️ 恢复HFT策略...")
            await strategy_manager.resume_strategy(hft_id)
            await asyncio.sleep(2)
            
            # 6. 演示重启
            logger.info(f"\n🔄 重启AI策略...")
            await strategy_manager.restart_strategy(ai_id)
            await asyncio.sleep(3)
            
        else:
            logger.error("❌ 策略启动失败")
            return
        
        # 7. 停止策略
        logger.info("\n🛑 停止所有策略...")
        for strategy_id in strategy_ids:
            await strategy_manager.stop_strategy(strategy_id)
        
        logger.info("✅ 所有策略已停止")
        
    finally:
        # 清理：注销策略
        logger.info("\n🧹 清理策略...")
        for strategy_id in strategy_ids:
            try:
                await strategy_manager.unregister_strategy(strategy_id)
                logger.info(f"✅ 策略 {strategy_id} 已注销")
            except Exception as e:
                logger.error(f"❌ 注销策略 {strategy_id} 失败: {e}")


async def demonstrate_resource_isolation(strategy_manager: StrategyManager):
    """演示资源隔离"""
    logger.info("\n=== 资源隔离演示 ===")
    
    resource_allocator = strategy_manager._resource_allocator
    
    # 获取系统资源状态
    system_status = await resource_allocator.get_resource_status()
    logger.info("🖥️ 系统资源状态:")
    logger.info(f"   总内存: {system_status['system_resources']['total_memory_mb']:.0f}MB")
    logger.info(f"   可用内存: {system_status['system_resources']['available_memory_mb']:.0f}MB")
    logger.info(f"   CPU核心: {system_status['system_resources']['total_cpu_cores']}")
    logger.info(f"   CPU使用率: {system_status['system_resources']['cpu_usage_percent']:.1f}%")
    
    # 获取隔离组信息
    isolation_groups = resource_allocator.get_isolation_groups()
    logger.info("\n🔒 资源隔离组:")
    for group, strategies in isolation_groups.items():
        logger.info(f"   {group}组: {strategies if strategies else '空'}")
    
    # 获取分配详情
    logger.info("\n📋 资源分配详情:")
    for strategy_id, allocation_data in system_status.get('allocations', {}).items():
        logger.info(f"   {strategy_id}: {allocation_data['type']}, "
                   f"优先级={allocation_data['priority']}, 状态={allocation_data['status']}")


async def demonstrate_monitoring(strategy_manager: StrategyManager):
    """演示监控功能"""
    logger.info("\n=== 监控功能演示 ===")
    
    strategy_monitor = strategy_manager._strategy_monitor
    
    # 获取系统监控指标
    system_metrics = strategy_monitor.get_system_metrics()
    logger.info("📊 系统监控指标:")
    logger.info(f"   总策略数: {system_metrics['total_strategies']}")
    logger.info(f"   活跃策略数: {system_metrics['active_strategies']}")
    logger.info(f"   总交易数: {system_metrics['total_trades']}")
    logger.info(f"   总错误数: {system_metrics['total_errors']}")
    logger.info(f"   总PnL: {system_metrics['total_pnl']:.2f}")
    logger.info(f"   活跃告警数: {system_metrics['active_alerts']}")
    
    # 获取告警规则
    alert_rules = strategy_monitor.alert_rules
    logger.info(f"\n⚠️ 配置的告警规则数: {len(alert_rules)}")
    for rule_id, rule in list(alert_rules.items())[:3]:  # 显示前3个规则
        logger.info(f"   {rule.name}: {rule.metric_name} {rule.condition} {rule.threshold}")
    
    # 获取活跃告警
    active_alerts = strategy_monitor.get_active_alerts()
    if active_alerts:
        logger.info(f"\n🚨 当前活跃告警: {len(active_alerts)}")
        for alert in active_alerts[:3]:  # 显示前3个告警
            logger.info(f"   [{alert['level'].upper()}] {alert['title']}: {alert['message']}")
    else:
        logger.info("\n✅ 当前无活跃告警")


async def demonstrate_config_management(config_manager: StrategyConfigManager):
    """演示配置管理"""
    logger.info("\n=== 配置管理演示 ===")
    
    # 列出所有配置
    all_configs = config_manager.list_configs()
    logger.info(f"📂 总配置数: {len(all_configs)}")
    
    for config in all_configs:
        logger.info(f"   {config['config_id']}: {config['name']} ({config['strategy_type']})")
    
    # 演示配置更新
    if all_configs:
        config_id = all_configs[0]['config_id']
        logger.info(f"\n✏️ 更新配置: {config_id}")
        
        updates = {
            'description': f'已更新 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }
        
        success = await config_manager.update_config(config_id, updates)
        if success:
            logger.info("✅ 配置更新成功")
            
            # 获取更新后的配置
            updated_config = config_manager.get_config(config_id)
            logger.info(f"   新描述: {updated_config.description}")
        else:
            logger.error("❌ 配置更新失败")


async def simulate_trading_activity(strategy_manager: StrategyManager, strategy_ids: list):
    """模拟交易活动"""
    logger.info("\n=== 模拟交易活动 ===")
    
    strategy_monitor = strategy_manager._strategy_monitor
    
    # 模拟不同的指标更新
    import random
    
    for i in range(10):
        for strategy_id in strategy_ids:
            if strategy_id not in strategy_manager.strategies:
                continue
            
            # 模拟CPU和内存使用
            cpu_usage = random.uniform(20, 80)
            memory_usage = random.uniform(500, 2000)
            
            # 模拟交易统计
            total_trades = random.randint(50, 200)
            successful_trades = int(total_trades * random.uniform(0.85, 0.98))
            
            # 模拟PnL
            daily_pnl = random.uniform(-500, 1000)
            
            # 更新指标
            strategy_monitor.update_metric(strategy_id, 'cpu_usage', cpu_usage)
            strategy_monitor.update_metric(strategy_id, 'memory_usage', memory_usage)
            strategy_monitor.update_metric(strategy_id, 'total_trades', total_trades)
            strategy_monitor.update_metric(strategy_id, 'successful_trades', successful_trades)
            strategy_monitor.update_metric(strategy_id, 'daily_pnl', daily_pnl)
            
            # 模拟延迟指标
            avg_latency = random.uniform(1, 50)
            strategy_monitor.update_metric(strategy_id, 'avg_latency_ms', avg_latency)
        
        await asyncio.sleep(1)
    
    logger.info("✅ 交易活动模拟完成")
    
    # 显示最终指标
    logger.info("\n📊 最终策略指标:")
    for strategy_id in strategy_ids:
        metrics = strategy_monitor.get_strategy_metrics(strategy_id)
        if metrics:
            logger.info(f"   {strategy_id}:")
            logger.info(f"     CPU: {metrics['current_cpu_usage']:.1f}%")
            logger.info(f"     内存: {metrics['current_memory_usage']:.1f}MB")
            logger.info(f"     交易: {metrics['total_trades']} (成功率: {metrics['success_rate']:.2%})")
            logger.info(f"     PnL: {metrics['daily_pnl']:.2f}")
            logger.info(f"     延迟: {metrics['avg_latency_ms']:.2f}ms")


async def run_callback_demo(strategy_manager: StrategyManager, strategy_ids: list):
    """演示回调功能"""
    logger.info("\n=== 回调功能演示 ===")
    
    # 定义回调函数
    def on_strategy_start(instance, **kwargs):
        logger.info(f"🟢 回调: 策略 {instance.config.strategy_id} 已启动")
    
    def on_strategy_stop(instance, **kwargs):
        logger.info(f"🔴 回调: 策略 {instance.config.strategy_id} 已停止")
    
    def on_strategy_error(instance, **kwargs):
        logger.error(f"❌ 回调: 策略 {instance.config.strategy_id} 发生错误")
    
    def on_metrics_update(instance, **kwargs):
        logger.debug(f"📊 回调: 策略 {instance.config.strategy_id} 指标已更新")
    
    # 注册回调
    for strategy_id in strategy_ids:
        strategy_manager.register_callback(strategy_id, 'on_start', on_strategy_start)
        strategy_manager.register_callback(strategy_id, 'on_stop', on_strategy_stop)
        strategy_manager.register_callback(strategy_id, 'on_error', on_strategy_error)
        strategy_manager.register_callback(strategy_id, 'on_metrics_update', on_metrics_update)
    
    logger.info("✅ 回调函数已注册")


async def main():
    """主函数"""
    logger.info("🚀 启动策略管理系统演示")
    
    # 创建组件
    message_bus = MessageBus()
    strategy_manager = StrategyManager(message_bus)
    config_manager = StrategyConfigManager("config/demo_strategies")
    
    try:
        # 初始化组件
        logger.info("\n⚙️ 初始化系统组件...")
        await message_bus.initialize()
        await strategy_manager.initialize()
        await config_manager.initialize()
        
        logger.info("✅ 系统初始化完成")
        
        # 创建示例配置
        configs = await create_sample_configs(config_manager)
        
        # 演示配置管理
        await demonstrate_config_management(config_manager)
        
        # 演示策略生命周期
        await demonstrate_strategy_lifecycle(strategy_manager, configs)
        
        # 重新注册策略用于其他演示
        hft_config, ai_config = configs
        hft_id = await strategy_manager.register_strategy(hft_config)
        ai_id = await strategy_manager.register_strategy(ai_config)
        strategy_ids = [hft_id, ai_id]
        
        # 注册回调
        await run_callback_demo(strategy_manager, strategy_ids)
        
        # 启动策略
        await strategy_manager.start_strategy(hft_id)
        await strategy_manager.start_strategy(ai_id)
        
        # 演示资源隔离
        await demonstrate_resource_isolation(strategy_manager)
        
        # 模拟交易活动
        await simulate_trading_activity(strategy_manager, strategy_ids)
        
        # 演示监控功能
        await demonstrate_monitoring(strategy_manager)
        
        # 显示最终系统状态
        logger.info("\n📊 最终系统状态:")
        final_status = strategy_manager.get_system_status()
        logger.info(f"   运行时间: {final_status['uptime_seconds']:.1f}秒")
        logger.info(f"   策略总数: {final_status['total_strategies']}")
        logger.info(f"   运行策略: {final_status['running_strategies']}")
        logger.info(f"   创建总数: {final_status['total_created']}")
        logger.info(f"   终止总数: {final_status['total_terminated']}")
        
        # 清理
        logger.info("\n🧹 清理系统...")
        for strategy_id in strategy_ids:
            await strategy_manager.stop_strategy(strategy_id)
            await strategy_manager.unregister_strategy(strategy_id)
        
    except Exception as e:
        logger.error(f"❌ 演示过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    finally:
        # 关闭组件
        logger.info("\n🔧 关闭系统组件...")
        try:
            await strategy_manager.shutdown()
            await config_manager.shutdown()
            await message_bus.shutdown()
            logger.info("✅ 系统关闭完成")
        except Exception as e:
            logger.error(f"❌ 关闭系统时发生错误: {e}")
    
    logger.info("\n🎉 策略管理系统演示完成!")


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行演示
    asyncio.run(main())