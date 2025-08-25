"""
延迟监控系统使用示例

演示如何在高频交易环境中使用延迟监控系统：
1. 配置多个数据源
2. 监控数据新鲜度
3. 自动切换数据源
4. 收集和分析延迟指标
5. 处理延迟告警
"""

import asyncio
import time
import random
from decimal import Decimal
from typing import List

from src.core.models import MarketData
from src.hft.hft_engine import HFTEngine, HFTConfig
from src.hft.latency_monitor import DataSourceConfig, DataSourceStatus, AlertEvent


class LatencyMonitoringExample:
    """延迟监控示例类"""
    
    def __init__(self):
        self.hft_engine: HFTEngine = None
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        self.data_sources = [
            DataSourceConfig(
                name="binance_websocket",
                priority=1,
                max_latency_ms=30.0,
                timeout_ms=1000.0,
                health_check_interval=30.0
            ),
            DataSourceConfig(
                name="okx_websocket", 
                priority=2,
                max_latency_ms=50.0,
                timeout_ms=1500.0,
                health_check_interval=30.0
            ),
            DataSourceConfig(
                name="bybit_websocket",
                priority=3, 
                max_latency_ms=100.0,
                timeout_ms=2000.0,
                health_check_interval=45.0
            ),
            DataSourceConfig(
                name="local_cache",
                priority=4,
                max_latency_ms=200.0,
                timeout_ms=500.0,
                health_check_interval=60.0
            )
        ]
        
    async def setup_hft_engine(self):
        """设置HFT引擎"""
        config = HFTConfig(
            # 延迟监控配置
            staleness_threshold_ms=100.0,  # 100ms过期阈值
            latency_stats_window=1000,     # 1000个样本的统计窗口
            alert_cooldown_seconds=30.0,   # 30秒告警冷却
            enable_latency_monitoring=True,
            
            # 性能配置
            latency_target_ms=10.0,        # 目标延迟10ms
            update_interval_ms=1.0,        # 1ms更新间隔
        )
        
        self.hft_engine = HFTEngine(config)
        
        # 初始化引擎
        await self.hft_engine.initialize(self.symbols, self.data_sources)
        
        # 设置告警处理
        self.setup_alert_handling()
        
        print("✅ HFT引擎初始化完成")
        print(f"📊 监控symbols: {self.symbols}")
        print(f"🔌 数据源: {[ds.name for ds in self.data_sources]}")
        
    def setup_alert_handling(self):
        """设置告警处理"""
        if self.hft_engine.latency_monitor:
            self.hft_engine.latency_monitor.add_alert_callback(self.handle_latency_alert)
    
    def handle_latency_alert(self, alert: AlertEvent):
        """处理延迟告警"""
        level_emoji = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "critical": "🚨"
        }
        
        emoji = level_emoji.get(alert.level.value, "📢")
        
        print(f"\n{emoji} 延迟告警 [{alert.level.value.upper()}]")
        print(f"   消息: {alert.message}")
        print(f"   交易对: {alert.symbol}")
        print(f"   数据源: {alert.data_source}")
        print(f"   延迟: {alert.latency_ms:.2f}ms")
        print(f"   时间: {time.strftime('%H:%M:%S', time.localtime(alert.timestamp))}")
        
        # 根据告警级别采取不同行动
        if alert.level.value == "critical":
            print("🚨 关键告警：考虑暂停交易或切换到紧急模式")
        elif alert.level.value == "error":
            print("❌ 错误告警：检查数据源连接和网络状况")
        elif alert.level.value == "warning":
            print("⚠️  警告告警：监控情况并准备备用方案")
    
    def generate_market_data(self, symbol: str, latency_ms: float = 0) -> MarketData:
        """生成模拟市场数据"""
        current_time = time.time()
        
        # 模拟延迟
        data_timestamp = int((current_time - latency_ms / 1000) * 1000)
        
        # 随机价格波动
        base_prices = {"BTCUSDT": 50000, "ETHUSDT": 3000, "ADAUSDT": 0.5}
        base_price = base_prices.get(symbol, 100)
        
        price = base_price * (1 + random.uniform(-0.01, 0.01))
        bid = price * 0.9995
        ask = price * 1.0005
        
        return MarketData(
            symbol=symbol,
            timestamp=data_timestamp,
            price=price,
            volume=random.uniform(0.1, 10.0),
            bid=bid,
            ask=ask,
            bid_volume=random.uniform(1.0, 100.0),
            ask_volume=random.uniform(1.0, 100.0)
        )
    
    async def simulate_normal_trading(self, duration_seconds: int = 30):
        """模拟正常交易场景"""
        print(f"\n📈 开始模拟正常交易 ({duration_seconds}秒)")
        
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < duration_seconds:
            # 为每个symbol生成新鲜数据
            for symbol in self.symbols:
                # 随机选择数据源
                data_source = random.choice([ds.name for ds in self.data_sources[:2]])  # 优先使用前两个
                
                # 生成低延迟数据（0-20ms）
                market_data = self.generate_market_data(symbol, random.uniform(0, 20))
                
                # 更新市场数据
                await self.hft_engine.update_market_data(symbol, market_data, data_source)
                update_count += 1
            
            await asyncio.sleep(0.1)  # 100ms间隔
            
            # 每10次更新打印一次状态
            if update_count % 30 == 0:
                await self.print_latency_summary()
        
        print(f"✅ 正常交易模拟完成，共处理 {update_count} 次更新")
    
    async def simulate_latency_issues(self, duration_seconds: int = 20):
        """模拟延迟问题场景"""
        print(f"\n⚠️  开始模拟延迟问题 ({duration_seconds}秒)")
        
        start_time = time.time()
        issue_count = 0
        
        while time.time() - start_time < duration_seconds:
            for symbol in self.symbols:
                # 随机决定是否有延迟问题
                if random.random() < 0.3:  # 30%概率有延迟
                    # 生成高延迟数据（100-500ms）
                    latency = random.uniform(100, 500)
                    data_source = "binance_websocket"
                    issue_count += 1
                    
                    print(f"🐌 模拟{symbol}延迟: {latency:.1f}ms")
                else:
                    # 正常延迟
                    latency = random.uniform(10, 50)
                    data_source = random.choice([ds.name for ds in self.data_sources[:2]])
                
                market_data = self.generate_market_data(symbol, latency)
                await self.hft_engine.update_market_data(symbol, market_data, data_source)
            
            await asyncio.sleep(0.2)  # 200ms间隔
        
        print(f"✅ 延迟问题模拟完成，触发 {issue_count} 次延迟")
    
    async def simulate_data_source_failure(self):
        """模拟数据源失效场景"""
        print(f"\n🚫 模拟数据源失效场景")
        
        # 模拟主数据源失效
        primary_source = "binance_websocket"
        print(f"❌ 主数据源 {primary_source} 失效")
        
        await self.hft_engine.update_data_source_status(primary_source, "failed")
        
        # 等待系统响应
        await asyncio.sleep(2)
        
        # 检查数据源切换结果
        active_sources = self.hft_engine.get_active_data_sources()
        print(f"🔄 当前活跃数据源: {active_sources}")
        
        # 恢复数据源
        await asyncio.sleep(5)
        print(f"✅ 数据源 {primary_source} 恢复")
        await self.hft_engine.update_data_source_status(primary_source, "active")
    
    async def print_system_status(self):
        """打印系统状态"""
        status = self.hft_engine.get_system_status()
        
        print(f"\n📊 系统状态概览")
        print(f"   运行状态: {'🟢 运行中' if status['running'] else '🔴 已停止'}")
        print(f"   监控symbols: {len(status['symbols'])}")
        
        metrics = status['metrics']
        print(f"\n📈 性能指标:")
        print(f"   数据更新次数: {metrics['total_updates']}")
        print(f"   平均更新延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"   最大更新延迟: {metrics['max_latency_ms']:.2f}ms")
        print(f"   数据新鲜度检查: {metrics['data_freshness_checks']}")
        print(f"   过期数据检测: {metrics['stale_data_detections']}")
        print(f"   数据源切换次数: {metrics['data_source_switches']}")
        print(f"   平均数据延迟: {metrics['avg_data_latency_ms']:.2f}ms")
        print(f"   P99数据延迟: {metrics['p99_data_latency_ms']:.2f}ms")
        
        if 'latency_monitoring' in status:
            latency_health = status['latency_monitoring']
            print(f"\n🔍 延迟监控健康:")
            print(f"   总检查次数: {latency_health['total_checks']}")
            print(f"   过期数据率: {latency_health['stale_detection_rate']:.2%}")
            print(f"   数据源切换次数: {latency_health['source_switches']}")
    
    async def print_latency_summary(self):
        """打印延迟摘要"""
        print(f"\n⏱️  延迟统计摘要:")
        
        for symbol in self.symbols:
            stats = self.hft_engine.get_latency_stats(symbol)
            if stats:
                print(f"   {symbol}:")
                print(f"     平均延迟: {stats['avg_latency_ms']:.2f}ms")
                print(f"     P95延迟: {stats['p95_latency_ms']:.2f}ms")
                print(f"     P99延迟: {stats['p99_latency_ms']:.2f}ms")
                print(f"     最大延迟: {stats['max_latency_ms']:.2f}ms")
                print(f"     过期数据率: {stats['stale_data_rate']:.2%}")
    
    async def print_data_source_status(self):
        """打印数据源状态"""
        source_status = self.hft_engine.get_data_source_status()
        active_sources = self.hft_engine.get_active_data_sources()
        
        print(f"\n🔌 数据源状态:")
        for source_name, status in source_status.items():
            status_emoji = {"active": "🟢", "degraded": "🟡", "failed": "🔴", "inactive": "⚪"}
            emoji = status_emoji.get(status, "❓")
            print(f"   {emoji} {source_name}: {status}")
        
        print(f"\n🎯 当前活跃数据源:")
        for symbol, source in active_sources.items():
            print(f"   {symbol} → {source}")
    
    async def run_comprehensive_demo(self):
        """运行完整演示"""
        try:
            # 初始化系统
            await self.setup_hft_engine()
            await self.hft_engine.start()
            
            # 打印初始状态
            await self.print_system_status()
            await self.print_data_source_status()
            
            # 1. 正常交易场景
            await self.simulate_normal_trading(15)
            await self.print_latency_summary()
            
            # 2. 延迟问题场景
            await self.simulate_latency_issues(10)
            await self.print_latency_summary()
            
            # 3. 数据源失效场景
            await self.simulate_data_source_failure()
            await self.print_data_source_status()
            
            # 4. 最终状态报告
            await self.print_system_status()
            await self.print_latency_summary()
            
        finally:
            if self.hft_engine:
                await self.hft_engine.stop()
                print(f"\n🛑 HFT引擎已停止")


async def main():
    """主函数"""
    print("🚀 延迟监控系统演示开始")
    print("=" * 60)
    
    demo = LatencyMonitoringExample()
    await demo.run_comprehensive_demo()
    
    print("\n" + "=" * 60)
    print("✅ 延迟监控系统演示完成")


if __name__ == "__main__":
    asyncio.run(main())