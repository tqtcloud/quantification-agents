#!/usr/bin/env python3
"""
智能平仓管理系统演示

展示完整的自动平仓管理系统功能，包括：
- 仓位开平仓管理
- 多策略自动平仓
- 风险监控和告警
- 实时数据更新和分析
- 性能统计和报告
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import random

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.position import PositionManager, MarketDataProvider, ATRInfo, VolatilityInfo
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength


class MockMarketDataProvider:
    """模拟市场数据提供器
    
    提供模拟的价格、ATR、波动率和相关性数据用于演示。
    """
    
    def __init__(self):
        self.base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'LTCUSDT': 100.0,
            'ADAUSDT': 1.5,
            'DOTUSDT': 20.0
        }
        
        self.price_trends = {}
        self.volatility_cache = {}
        self.correlation_cache = None
        self.last_update = datetime.utcnow()
        
        # 初始化价格趋势
        for symbol in self.base_prices:
            self.price_trends[symbol] = random.uniform(-0.002, 0.002)  # ±0.2%的趋势
        
        self._update_correlation_matrix()
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格（模拟价格波动）"""
        if symbol not in self.base_prices:
            return 0.0
        
        base_price = self.base_prices[symbol]
        trend = self.price_trends.get(symbol, 0.0)
        
        # 添加随机波动
        volatility = random.uniform(0.001, 0.01)  # 0.1% - 1%波动
        change = trend + random.normalvariate(0, volatility)
        
        # 更新基础价格
        new_price = base_price * (1 + change)
        self.base_prices[symbol] = max(new_price, base_price * 0.5)  # 防止价格过低
        
        # 偶尔改变趋势
        if random.random() < 0.05:  # 5%概率改变趋势
            self.price_trends[symbol] = random.uniform(-0.002, 0.002)
        
        return self.base_prices[symbol]
    
    def get_atr_info(self, symbol: str) -> ATRInfo:
        """获取ATR信息"""
        base_price = self.base_prices.get(symbol, 50000.0)
        
        # ATR通常是价格的1-3%
        atr_value = base_price * random.uniform(0.01, 0.03)
        
        return ATRInfo(
            period=14,
            current_atr=atr_value,
            atr_multiplier=2.0
        )
    
    def get_volatility_info(self, symbol: str) -> VolatilityInfo:
        """获取波动率信息"""
        # 缓存波动率信息以保持一致性
        if symbol not in self.volatility_cache:
            current_vol = random.uniform(0.02, 0.08)
            avg_vol = current_vol * random.uniform(0.8, 1.2)
            percentile = random.uniform(0.1, 0.9)
            
            self.volatility_cache[symbol] = VolatilityInfo(
                current_volatility=current_vol,
                avg_volatility=avg_vol,
                volatility_percentile=percentile
            )
        
        return self.volatility_cache[symbol]
    
    def get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """获取相关性矩阵"""
        if self.correlation_cache is None or len(symbols) > len(self.correlation_cache):
            self._update_correlation_matrix(symbols)
        
        # 返回请求符号的子矩阵
        result = {}
        for symbol1 in symbols:
            if symbol1 in self.correlation_cache:
                result[symbol1] = {}
                for symbol2 in symbols:
                    if symbol2 in self.correlation_cache[symbol1]:
                        result[symbol1][symbol2] = self.correlation_cache[symbol1][symbol2]
                    elif symbol1 == symbol2:
                        result[symbol1][symbol2] = 1.0
                    else:
                        # 生成随机相关性
                        corr = random.uniform(-0.3, 0.8)
                        result[symbol1][symbol2] = corr
        
        return result
    
    def _update_correlation_matrix(self, symbols: Optional[List[str]] = None):
        """更新相关性矩阵"""
        if symbols is None:
            symbols = list(self.base_prices.keys())
        
        self.correlation_cache = {}
        
        for i, symbol1 in enumerate(symbols):
            self.correlation_cache[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    self.correlation_cache[symbol1][symbol2] = 1.0
                elif symbol2 in self.correlation_cache:
                    # 使用对称性
                    self.correlation_cache[symbol1][symbol2] = self.correlation_cache[symbol2][symbol1]
                else:
                    # 生成相关性（加密货币通常正相关）
                    if 'BTC' in symbol1 or 'BTC' in symbol2:
                        corr = random.uniform(0.4, 0.8)  # BTC与其他币种高相关
                    else:
                        corr = random.uniform(0.2, 0.7)  # 其他币种中等相关
                    
                    self.correlation_cache[symbol1][symbol2] = corr


class PositionManagerDemo:
    """仓位管理器演示类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.market_provider = MockMarketDataProvider()
        self.position_manager = PositionManager(
            config=self.config['position_manager'],
            market_data_provider=self.market_provider
        )
        
        self.running = False
        self.demo_positions: Dict[str, Dict] = {}
        self.statistics = {
            'start_time': datetime.utcnow(),
            'positions_opened': 0,
            'positions_closed': 0,
            'total_pnl': 0.0,
            'max_positions': 0,
            'alerts_triggered': 0
        }
        
        self._setup_logging()
        self._setup_callbacks()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = project_root / "config" / "position_manager.json"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logging.error(f"配置文件解析失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'position_manager': {
                'max_positions': 10,
                'max_exposure_per_symbol': 0.2,
                'risk_check_interval_seconds': 30,
                'enable_risk_monitoring': True,
                'enable_performance_tracking': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': True
            }
        }
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        
        # 配置根日志器
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    project_root / "logs" / f"position_demo_{datetime.now().strftime('%Y%m%d')}.log"
                )
            ]
        )
        
        self.logger = logging.getLogger("PositionDemo")
        self.logger.info("日志系统初始化完成")
    
    def _setup_callbacks(self):
        """设置回调函数"""
        self.position_manager.add_position_opened_callback(self._on_position_opened)
        self.position_manager.add_position_closed_callback(self._on_position_closed)
        self.position_manager.add_risk_alert_callback(self._on_risk_alert)
    
    async def _on_position_opened(self, position):
        """仓位开启回调"""
        self.statistics['positions_opened'] += 1
        self.statistics['max_positions'] = max(
            self.statistics['max_positions'],
            len(self.position_manager.auto_closer.get_all_positions())
        )
        
        self.demo_positions[position.position_id] = {
            'open_time': datetime.utcnow(),
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'quantity': position.quantity
        }
        
        self.logger.info(f"🔓 开仓: {position.position_id} ({position.symbol} {position.side} "
                        f"{position.quantity}@{position.entry_price})")
    
    async def _on_position_closed(self, result):
        """仓位关闭回调"""
        self.statistics['positions_closed'] += 1
        self.statistics['total_pnl'] += result.realized_pnl
        
        demo_pos = self.demo_positions.pop(result.position_id, {})
        hold_time = (datetime.utcnow() - demo_pos.get('open_time', datetime.utcnow())).total_seconds()
        
        pnl_symbol = "💰" if result.realized_pnl > 0 else "💸"
        self.logger.info(f"🔒 平仓: {result.position_id} "
                        f"{pnl_symbol} PnL: {result.realized_pnl:.2f} "
                        f"({result.closing_reason.value}) 持仓: {hold_time:.0f}s")
    
    async def _on_risk_alert(self, alert):
        """风险告警回调"""
        self.statistics['alerts_triggered'] += 1
        
        alert_emoji = {
            'low': '⚠️',
            'medium': '🔶',
            'high': '🔴',
            'critical': '🚨'
        }
        
        level = 'medium'  # 默认级别
        emoji = alert_emoji.get(level, '⚠️')
        
        self.logger.warning(f"{emoji} 风险告警 [{alert['alert_type']}]: {alert['message']}")
    
    def create_sample_signal(self, symbol: str, signal_type: str = "BUY") -> MultiDimensionalSignal:
        """创建示例信号"""
        current_price = self.market_provider.get_current_price(symbol)
        
        if signal_type.upper() == "BUY":
            signal_strength = SignalStrength.BUY
            target_price = current_price * 1.05  # 5% 目标
            stop_loss = current_price * 0.98     # 2% 止损
        else:
            signal_strength = SignalStrength.SELL
            target_price = current_price * 0.95  # 5% 目标
            stop_loss = current_price * 1.02     # 2% 止损
        
        trading_signal = TradingSignal(
            symbol=symbol,
            signal_type=signal_strength,
            confidence=random.uniform(0.6, 0.9),
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=[f"Demo {signal_type} signal", "Technical analysis"],
            indicators_consensus={"ma": 0.7, "rsi": 0.6}
        )
        
        return MultiDimensionalSignal(
            primary_signal=trading_signal,
            momentum_score=random.uniform(-0.5, 0.8),
            mean_reversion_score=random.uniform(-0.3, 0.3),
            volatility_score=random.uniform(0.2, 0.8),
            volume_score=random.uniform(0.4, 1.0),
            sentiment_score=random.uniform(-0.4, 0.6),
            overall_confidence=random.uniform(0.6, 0.85),
            risk_reward_ratio=random.uniform(1.5, 3.0),
            max_position_size=random.uniform(0.5, 0.9)
        )
    
    async def simulate_market_activity(self):
        """模拟市场活动"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'DOTUSDT']
        
        while self.running:
            try:
                # 更新所有符号的价格
                price_data = {}
                for symbol in symbols:
                    price_data[symbol] = self.market_provider.get_current_price(symbol)
                
                # 批量更新仓位价格
                await self.position_manager.update_position_prices(price_data)
                
                # 创建信号数据
                signal_data = {}
                for symbol in symbols:
                    if random.random() < 0.3:  # 30%概率生成新信号
                        signal_data[symbol] = self.create_sample_signal(symbol)
                
                # 运行仓位监控
                close_requests = await self.position_manager.run_position_monitoring(signal_data)
                
                if close_requests:
                    self.logger.info(f"📈 监控发现 {len(close_requests)} 个平仓触发条件")
                
                # 随机开仓（模拟交易策略）
                if (random.random() < 0.1 and  # 10%概率开仓
                    len(self.position_manager.auto_closer.get_all_positions()) < 8):
                    
                    symbol = random.choice(symbols)
                    side = random.choice(['long', 'short'])
                    quantity = random.uniform(0.1, 1.0)
                    signal = self.create_sample_signal(symbol, "BUY" if side == "long" else "SELL")
                    
                    position_id = await self.position_manager.open_position(
                        symbol=symbol,
                        entry_price=price_data[symbol],
                        quantity=quantity,
                        side=side,
                        signal=signal
                    )
                    
                    if not position_id:
                        self.logger.warning(f"❌ 开仓失败: {symbol} {side} {quantity}")
                
                await asyncio.sleep(2)  # 2秒间隔
                
            except Exception as e:
                self.logger.error(f"市场模拟错误: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def print_statistics(self):
        """打印统计信息"""
        while self.running:
            try:
                await asyncio.sleep(30)  # 30秒间隔
                
                # 获取详细统计
                stats = self.position_manager.get_detailed_statistics()
                positions = self.position_manager.auto_closer.get_all_positions()
                
                print("\n" + "="*80)
                print("📊 智能平仓管理系统状态报告")
                print("="*80)
                
                # 基本统计
                runtime = datetime.utcnow() - self.statistics['start_time']
                print(f"⏱️  运行时间: {runtime}")
                print(f"📈 活跃仓位: {len(positions)}")
                print(f"🔓 已开仓位: {self.statistics['positions_opened']}")
                print(f"🔒 已平仓位: {self.statistics['positions_closed']}")
                print(f"💰 总盈亏: {self.statistics['total_pnl']:.2f}")
                print(f"⚠️  风险告警: {self.statistics['alerts_triggered']}")
                print(f"📊 最大仓位数: {self.statistics['max_positions']}")
                
                # 当前仓位详情
                if positions:
                    print("\n📋 当前活跃仓位:")
                    for pos_id, pos in positions.items():
                        pnl_symbol = "📈" if pos.unrealized_pnl > 0 else "📉"
                        print(f"   {pnl_symbol} {pos.symbol:<10} {pos.side:<5} "
                              f"数量: {pos.quantity:<8.4f} "
                              f"价格: {pos.current_price:<10.2f} "
                              f"盈亏: {pos.unrealized_pnl:>8.2f} ({pos.unrealized_pnl_pct:>6.2f}%)")
                
                # 策略统计
                print("\n🎯 平仓策略统计:")
                strategy_stats = stats['auto_closer_stats']['strategy_stats']
                for name, stat in strategy_stats.items():
                    if stat['enabled']:
                        success_rate = stat['success_rate'] * 100
                        print(f"   {name:<20} 触发: {stat['trigger_count']:<3} "
                              f"成功: {stat['success_count']:<3} "
                              f"成功率: {success_rate:>5.1f}%")
                
                # 风险指标
                risk_metrics = stats['risk_metrics']
                print(f"\n⚡ 风险指标:")
                print(f"   组合价值: {risk_metrics['portfolio_value']:.2f}")
                print(f"   总敞口: {risk_metrics['total_exposure']:.2f}")
                print(f"   当前回撤: {risk_metrics['current_drawdown']*100:.2f}%")
                print(f"   最大回撤: {risk_metrics['max_drawdown']*100:.2f}%")
                
                print("="*80)
                
            except Exception as e:
                self.logger.error(f"统计打印错误: {e}", exc_info=True)
    
    async def start(self):
        """启动演示"""
        self.logger.info("🚀 启动智能平仓管理系统演示...")
        
        try:
            # 启动仓位管理器
            await self.position_manager.start()
            self.running = True
            
            # 启动后台任务
            tasks = [
                asyncio.create_task(self.simulate_market_activity()),
                asyncio.create_task(self.print_statistics())
            ]
            
            self.logger.info("✅ 演示系统启动成功")
            self.logger.info("按 Ctrl+C 停止演示")
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 收到停止信号...")
        except Exception as e:
            self.logger.error(f"演示运行错误: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止演示"""
        if not self.running:
            return
        
        self.logger.info("⏹️ 停止智能平仓管理系统...")
        self.running = False
        
        try:
            # 强制平仓所有仓位
            positions = self.position_manager.auto_closer.get_all_positions()
            if positions:
                self.logger.info(f"💼 强制平仓 {len(positions)} 个仓位...")
                results = await self.position_manager.auto_closer.force_close_all_positions()
                success_count = sum(1 for r in results if r.success)
                self.logger.info(f"✅ 成功平仓 {success_count}/{len(results)} 个仓位")
            
            # 停止仓位管理器
            await self.position_manager.stop()
            
            # 打印最终统计
            print("\n" + "="*80)
            print("📊 演示结束 - 最终统计")
            print("="*80)
            runtime = datetime.utcnow() - self.statistics['start_time']
            print(f"⏱️  总运行时间: {runtime}")
            print(f"🔓 总开仓数: {self.statistics['positions_opened']}")
            print(f"🔒 总平仓数: {self.statistics['positions_closed']}")
            print(f"💰 总盈亏: {self.statistics['total_pnl']:.2f}")
            print(f"⚠️  总告警数: {self.statistics['alerts_triggered']}")
            print("="*80)
            
            self.logger.info("🏁 演示系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止过程出错: {e}", exc_info=True)


def setup_signal_handlers(demo: PositionManagerDemo):
    """设置信号处理器"""
    def signal_handler(signum, frame):
        print(f"\n收到信号 {signum}，正在优雅关闭...")
        asyncio.create_task(demo.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="智能平仓管理系统演示")
    parser.add_argument('--config-file', type=str, help="配置文件路径")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="日志级别")
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(project_root / "logs", exist_ok=True)
    
    # 创建演示实例
    demo = PositionManagerDemo(args.config_file)
    
    # 设置信号处理
    setup_signal_handlers(demo)
    
    # 运行演示
    try:
        asyncio.run(demo.start())
    except KeyboardInterrupt:
        print("\n演示被中断")
    except Exception as e:
        print(f"演示运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()