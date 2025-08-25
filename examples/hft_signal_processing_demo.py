"""
高频交易信号处理集成系统演示

此演示展示了完整的HFT信号处理流程：
1. 多维度技术指标计算
2. 信号过滤和验证
3. 智能订单生成
4. 容错和异常处理
5. 延迟监控和故障转移
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from src.hft.integrated_signal_processor import IntegratedHFTSignalProcessor, OrderRequest
from src.hft.smart_order_router import SmartOrderRouter, ExecutionReport
from src.hft.fault_tolerance_manager import FaultToleranceManager, ErrorEvent, ComponentStatus
from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.hft.latency_monitor import LatencyMonitor, DataSourceConfig, DataSourceStatus
from src.hft.signal_processor import LatencySensitiveSignalProcessor
from src.core.models.trading import MarketData
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength


class MockDataFeed:
    """模拟数据源，生成实时市场数据"""
    
    def __init__(self, symbol: str = "BTCUSDT", base_price: float = 50000.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        self.tick = 0
        
    def generate_market_data(self) -> MarketData:
        """生成模拟市场数据"""
        self.tick += 1
        
        # 添加随机波动
        price_change = np.random.normal(0, 0.001) * self.current_price  # 0.1%的标准差
        self.current_price += price_change
        
        # 生成OHLC数据
        high = self.current_price + abs(np.random.normal(0, 0.0005)) * self.current_price
        low = self.current_price - abs(np.random.normal(0, 0.0005)) * self.current_price
        volume = np.random.exponential(1000)  # 指数分布的成交量
        
        # 生成买卖盘数据
        spread = np.random.uniform(0.0001, 0.001) * self.current_price  # 0.01%-0.1%的价差
        bid = self.current_price - spread / 2
        ask = self.current_price + spread / 2
        
        return MarketData(
            symbol=self.symbol,
            timestamp=datetime.now(),
            open=self.current_price,  # 简化处理
            high=high,
            low=low,
            close=self.current_price,
            volume=volume,
            bid=bid,
            ask=ask,
            bid_volume=np.random.uniform(5, 50),
            ask_volume=np.random.uniform(5, 50)
        )


class HFTSystemDemo:
    """高频交易系统演示"""
    
    def __init__(self):
        self.setup_logging()
        
        # 初始化组件
        self.multidimensional_engine = MultiDimensionalIndicatorEngine(max_workers=4)
        self.latency_monitor = LatencyMonitor(
            staleness_threshold_ms=50.0,
            alert_cooldown_seconds=30.0
        )
        self.signal_processor = LatencySensitiveSignalProcessor(
            max_queue_size=1000,
            latency_target_ms=1.0,
            processing_workers=2
        )
        
        # 集成处理器
        self.integrated_processor = IntegratedHFTSignalProcessor(
            multidimensional_engine=self.multidimensional_engine,
            latency_monitor=self.latency_monitor,
            signal_processor=self.signal_processor,
            max_latency_ms=5.0,
            min_confidence_threshold=0.65,
            min_signal_strength=0.5
        )
        
        # 智能订单路由器
        self.order_router = SmartOrderRouter(
            max_child_orders=15,
            max_order_value=100000.0,
            default_slice_size=0.15
        )
        
        # 容错管理器
        self.fault_manager = FaultToleranceManager(
            error_window_size=500,
            health_check_interval=5.0
        )
        
        # 数据源
        self.data_feed = MockDataFeed("BTCUSDT", 50000.0)
        
        # 统计信息
        self.demo_stats = {
            "processed_data_points": 0,
            "generated_signals": 0,
            "routed_orders": 0,
            "execution_reports": 0,
            "errors_handled": 0,
            "start_time": None
        }
        
        # 设置回调
        self._setup_callbacks()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_callbacks(self):
        """设置回调函数"""
        # 订单生成回调
        self.integrated_processor.add_order_callback(self._handle_order_generated)
        
        # 执行完成回调
        self.order_router.add_execution_callback(self._handle_execution_complete)
        
        # 错误处理回调
        self.fault_manager.add_error_callback(self._handle_error_event)
        self.fault_manager.add_recovery_callback(self._handle_component_recovery)
        
        # 处理错误回调
        self.integrated_processor.add_error_callback(self._handle_processing_error)
        self.order_router.add_error_callback(self._handle_routing_error)
    
    async def initialize_system(self):
        """初始化系统"""
        self.logger.info("正在初始化HFT信号处理系统...")
        
        try:
            # 初始化延迟监控器数据源
            data_sources = [
                DataSourceConfig(
                    name="binance",
                    priority=1,
                    max_latency_ms=20.0,
                    description="Binance主数据源"
                ),
                DataSourceConfig(
                    name="okex",
                    priority=2,
                    max_latency_ms=30.0,
                    description="OKEx备用数据源"
                )
            ]
            await self.latency_monitor.initialize(data_sources)
            
            # 启动所有组件
            await self.fault_manager.start()
            await self.latency_monitor.start()
            await self.signal_processor.start()
            await self.integrated_processor.start()
            await self.order_router.start()
            
            self.logger.info("系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    async def run_demo(self, duration_seconds: int = 60):
        """运行演示"""
        self.logger.info(f"开始运行HFT系统演示，持续时间: {duration_seconds}秒")
        
        self.demo_stats["start_time"] = time.time()
        
        try:
            await self.initialize_system()
            
            # 主处理循环
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                try:
                    # 生成市场数据
                    market_data = self.data_feed.generate_market_data()
                    self.demo_stats["processed_data_points"] += 1
                    
                    # 使用容错管理器保护操作
                    async with self.fault_manager.protected_operation("data_processing"):
                        # 处理市场数据
                        orders = await self.integrated_processor.process_market_data(market_data)
                        
                        if orders:
                            self.demo_stats["generated_signals"] += 1
                            self.logger.info(f"生成 {len(orders)} 个订单: {market_data.symbol} @ {market_data.close:.2f}")
                            
                            # 路由订单
                            for order in orders:
                                execution_id = await self.order_router.route_order(order, market_data)
                                if execution_id:
                                    self.demo_stats["routed_orders"] += 1
                    
                    # 随机注入错误以测试容错机制
                    if np.random.random() < 0.05:  # 5%概率
                        await self._inject_test_error()
                    
                    # 定期打印状态
                    if self.demo_stats["processed_data_points"] % 20 == 0:
                        await self._print_status()
                    
                    # 控制处理频率
                    await asyncio.sleep(0.1)  # 10 Hz频率
                    
                except Exception as e:
                    await self.fault_manager.handle_error("demo_loop", e)
                    await asyncio.sleep(0.5)  # 错误后短暂暂停
            
        except KeyboardInterrupt:
            self.logger.info("收到中断信号，正在停止演示...")
        except Exception as e:
            self.logger.error(f"演示运行过程中出错: {e}")
        finally:
            await self._cleanup()
            await self._print_final_summary()
    
    async def _inject_test_error(self):
        """注入测试错误"""
        error_types = [
            (ConnectionError("模拟网络连接失败"), "network_component"),
            (TimeoutError("模拟超时错误"), "timeout_component"),
            (ValueError("模拟数据解析错误"), "data_component")
        ]
        
        error, component = np.random.choice(error_types, size=1)[0]
        await self.fault_manager.handle_error(component, error)
        self.demo_stats["errors_handled"] += 1
    
    def _handle_order_generated(self, order: OrderRequest):
        """处理订单生成事件"""
        self.logger.info(f"订单生成: {order.symbol} {order.side} {order.quantity} @ {order.price or 'MARKET'}")
    
    def _handle_execution_complete(self, report: ExecutionReport):
        """处理执行完成事件"""
        self.demo_stats["execution_reports"] += 1
        self.logger.info(
            f"执行完成: {report.symbol} 成交率: {report.filled_quantity/report.total_quantity:.2%} "
            f"平均价格: {report.avg_fill_price:.2f} 用时: {report.execution_time_ms:.1f}ms"
        )
    
    def _handle_error_event(self, error_event: ErrorEvent):
        """处理错误事件"""
        self.logger.warning(
            f"错误事件: {error_event.component} - {error_event.category.value} - {error_event.error_message}"
        )
    
    def _handle_component_recovery(self, component: str, status: ComponentStatus):
        """处理组件恢复事件"""
        self.logger.info(f"组件状态变更: {component} -> {status.value}")
    
    def _handle_processing_error(self, error: Exception, context: Dict[str, Any]):
        """处理处理错误"""
        self.logger.error(f"处理错误: {error} 上下文: {context}")
    
    def _handle_routing_error(self, execution_id: str, error: Exception):
        """处理路由错误"""
        self.logger.error(f"路由错误 {execution_id}: {error}")
    
    async def _print_status(self):
        """打印当前状态"""
        # 获取系统状态
        processor_status = self.integrated_processor.get_system_status()
        router_stats = self.order_router.get_routing_stats()
        fault_summary = self.fault_manager.get_system_health_summary()
        
        elapsed_time = time.time() - self.demo_stats["start_time"]
        
        print("\n" + "="*80)
        print("HFT系统状态概览")
        print("="*80)
        print(f"运行时间: {elapsed_time:.1f}s")
        print(f"数据点处理: {self.demo_stats['processed_data_points']}")
        print(f"生成信号: {self.demo_stats['generated_signals']}")
        print(f"路由订单: {self.demo_stats['routed_orders']}")
        print(f"执行报告: {self.demo_stats['execution_reports']}")
        print(f"处理错误: {self.demo_stats['errors_handled']}")
        
        print(f"\n信号处理器统计:")
        print(f"  处理成功率: {processor_status['stats']['processed']}/{processor_status['stats']['total_received']}")
        print(f"  平均延迟: {processor_status['stats']['avg_latency_ms']:.2f}ms")
        print(f"  过滤掉的信号: {processor_status['stats']['filtered_out']}")
        
        print(f"\n订单路由统计:")
        print(f"  成功路由: {router_stats['successful_routes']}")
        print(f"  失败路由: {router_stats['failed_routes']}")
        print(f"  平均执行时间: {router_stats.get('avg_execution_time_ms', 0):.2f}ms")
        
        print(f"\n系统健康状态:")
        print(f"  整体状态: {fault_summary['overall_status']}")
        print(f"  健康组件: {fault_summary['healthy_count']}/{fault_summary['total_components']}")
        print(f"  错误总数: {fault_summary['total_errors']}")
        
        print("="*80)
    
    async def _cleanup(self):
        """清理资源"""
        self.logger.info("正在清理系统资源...")
        
        try:
            await self.integrated_processor.stop()
            await self.order_router.stop()
            await self.signal_processor.stop()
            await self.latency_monitor.stop()
            await self.fault_manager.stop()
            
        except Exception as e:
            self.logger.error(f"清理过程中出错: {e}")
    
    async def _print_final_summary(self):
        """打印最终摘要"""
        elapsed_time = time.time() - self.demo_stats["start_time"]
        
        print("\n" + "="*80)
        print("HFT系统演示最终摘要")
        print("="*80)
        print(f"总运行时间: {elapsed_time:.1f}秒")
        print(f"处理数据点: {self.demo_stats['processed_data_points']} ({self.demo_stats['processed_data_points']/elapsed_time:.1f}/s)")
        print(f"生成信号: {self.demo_stats['generated_signals']} ({self.demo_stats['generated_signals']/elapsed_time:.2f}/s)")
        print(f"信号生成率: {self.demo_stats['generated_signals']/max(1, self.demo_stats['processed_data_points']):.1%}")
        print(f"路由订单: {self.demo_stats['routed_orders']}")
        print(f"执行报告: {self.demo_stats['execution_reports']}")
        print(f"处理错误: {self.demo_stats['errors_handled']}")
        
        # 最终系统状态
        processor_stats = self.integrated_processor.get_processing_stats()
        router_stats = self.order_router.get_routing_stats()
        fault_summary = self.fault_manager.get_system_health_summary()
        
        print(f"\n最终性能指标:")
        print(f"  信号处理平均延迟: {processor_stats.avg_processing_latency_ms:.2f}ms")
        print(f"  信号处理最大延迟: {processor_stats.max_processing_latency_ms:.2f}ms")
        print(f"  订单路由成功率: {router_stats['successful_routes']/(router_stats['successful_routes']+router_stats['failed_routes']):.1%}")
        print(f"  系统整体健康状态: {fault_summary['overall_status']}")
        
        print("="*80)


async def main():
    """主函数"""
    demo = HFTSystemDemo()
    
    try:
        # 运行60秒演示
        await demo.run_demo(duration_seconds=60)
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("高频交易信号处理集成系统演示")
    print("="*50)
    print("本演示将展示：")
    print("1. 实时市场数据处理")
    print("2. 多维度技术指标计算")
    print("3. 信号过滤和验证")
    print("4. 智能订单生成和路由")
    print("5. 容错和异常处理")
    print("6. 系统性能监控")
    print("\n按 Ctrl+C 可以随时停止演示")
    print("="*50)
    
    asyncio.run(main())