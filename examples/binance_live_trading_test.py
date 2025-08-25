#!/usr/bin/env python3
"""
币安模拟盘实时交易测试
集成技术指标引擎、自动平仓系统和币安API，进行完整的信号测试和订单测试

运行前请确保：
1. 复制 .env.example 为 .env 并配置币安API密钥
2. 确保 BINANCE_TESTNET=true 使用测试网
3. 安装必要依赖：uv add python-binance ccxt
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.core.position.position_manager import PositionManager
from src.exchanges.binance import BinanceFuturesClient
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength
from src.utils.logger import setup_logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logger = setup_logging("binance_live_test", log_level="INFO")


class BinanceLiveTradingTest:
    """币安实时交易测试类"""
    
    def __init__(self):
        """初始化测试环境"""
        self.binance_client: Optional[BinanceFuturesClient] = None
        self.indicator_engine: Optional[MultiDimensionalIndicatorEngine] = None
        self.position_manager: Optional[PositionManager] = None
        
        # 测试配置
        self.test_symbols = ['BTCUSDT', 'ETHUSDT']
        self.test_balance = 10000.0  # 模拟资金 10000 USDT
        self.position_size_pct = 0.1  # 每个仓位占总资金的 10%
        
        # 交易统计
        self.trade_stats = {
            'total_signals': 0,
            'strong_signals': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'total_pnl': 0.0
        }
        
        logger.info("🚀 币安实时交易测试初始化完成")

    async def initialize(self):
        """初始化所有组件"""
        try:
            logger.info("🔧 初始化系统组件...")
            
            # 初始化币安客户端
            self.binance_client = BinanceFuturesClient(testnet=True)
            await self.binance_client.connect()
            
            # 验证连接
            ping_result = await self.binance_client.ping()
            logger.info("✅ 币安API连接成功", extra={'ping': ping_result})
            
            # 获取账户信息
            account_info = await self.binance_client.get_account_info()
            logger.info("💰 账户信息获取成功", extra={
                'balance': sum(float(asset['balance']) for asset in account_info.get('assets', []) if asset['asset'] == 'USDT')
            })
            
            # 初始化技术指标引擎
            self.indicator_engine = MultiDimensionalIndicatorEngine()
            logger.info("📊 技术指标引擎初始化完成")
            
            # 初始化仓位管理器
            position_config = {
                'max_positions': 5,
                'max_exposure_per_symbol': 0.2,
                'enable_risk_monitoring': True,
                'auto_closer': {
                    'enable_emergency_stop': True,
                    'emergency_loss_threshold': -5.0
                }
            }
            self.position_manager = PositionManager(position_config)
            await self.position_manager.start()
            logger.info("🏦 仓位管理器初始化完成")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            return False

    async def fetch_historical_data(self, symbol: str, interval: str = '1h', limit: int = 200) -> list:
        """获取历史K线数据"""
        try:
            klines = await self.binance_client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # 转换为标准格式
            formatted_data = []
            for kline in klines:
                formatted_data.append({
                    'timestamp': kline[0] / 1000,  # 转换为秒
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
            
            logger.info(f"📈 获取 {symbol} 历史数据成功", extra={'count': len(formatted_data)})
            return formatted_data
            
        except Exception as e:
            logger.error(f"❌ 获取历史数据失败 {symbol}: {e}")
            return []

    async def generate_trading_signal(self, symbol: str) -> Optional[MultiDimensionalSignal]:
        """生成交易信号"""
        try:
            # 获取历史数据
            historical_data = await self.fetch_historical_data(symbol)
            if not historical_data:
                return None
            
            # 更新指标引擎数据
            for data_point in historical_data[-100:]:  # 使用最近100个数据点
                self.indicator_engine.update_market_data(symbol, data_point)
            
            # 生成多维度信号
            signal = await self.indicator_engine.generate_multidimensional_signal(
                symbol=symbol,
                timeframe='1h'
            )
            
            if signal:
                self.trade_stats['total_signals'] += 1
                if signal.overall_confidence > 0.7:
                    self.trade_stats['strong_signals'] += 1
                
                logger.info(f"🎯 {symbol} 信号生成", extra={
                    'confidence': signal.overall_confidence,
                    'signal_type': signal.primary_signal.signal_type.value if signal.primary_signal else 'None',
                    'risk_reward': signal.risk_reward_ratio
                })
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ 信号生成失败 {symbol}: {e}")
            return None

    async def execute_signal_if_valid(self, symbol: str, signal: MultiDimensionalSignal) -> bool:
        """如果信号有效则执行交易"""
        try:
            # 信号过滤条件
            if (signal.overall_confidence < 0.6 or  # 信心度太低
                signal.risk_reward_ratio < 1.5 or   # 风险回报比太低
                not signal.primary_signal):          # 没有主信号
                logger.info(f"🚫 {symbol} 信号不满足执行条件")
                return False
            
            # 获取当前价格
            ticker = await self.binance_client.get_ticker_price(symbol)
            current_price = float(ticker['price'])
            
            # 计算仓位大小
            position_value = self.test_balance * self.position_size_pct
            quantity = round(position_value / current_price, 3)
            
            # 确定交易方向
            if signal.primary_signal.signal_type in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
                side = 'long'
            elif signal.primary_signal.signal_type in [SignalStrength.SELL, SignalStrength.STRONG_SELL]:
                side = 'short'
            else:
                logger.info(f"🤔 {symbol} 信号方向不明确，跳过")
                return False
            
            # 开仓
            position_id = await self.position_manager.open_position(
                symbol=symbol,
                entry_price=current_price,
                quantity=quantity,
                side=side,
                signal=signal
            )
            
            if position_id:
                self.trade_stats['positions_opened'] += 1
                logger.info(f"✅ {symbol} 开仓成功", extra={
                    'position_id': position_id,
                    'side': side,
                    'quantity': quantity,
                    'entry_price': current_price
                })
                return True
            else:
                logger.warning(f"⚠️ {symbol} 开仓失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 信号执行失败 {symbol}: {e}")
            return False

    async def monitor_positions(self):
        """监控持仓"""
        try:
            # 更新所有持仓的市场价格
            price_updates = {}
            for symbol in self.test_symbols:
                try:
                    ticker = await self.binance_client.get_ticker_price(symbol)
                    price_updates[symbol] = float(ticker['price'])
                except Exception as e:
                    logger.warning(f"⚠️ 获取 {symbol} 价格失败: {e}")
            
            if price_updates:
                await self.position_manager.update_position_prices(price_updates)
            
            # 检查平仓信号
            close_requests = await self.position_manager.run_position_monitoring()
            
            for request in close_requests:
                logger.info(f"📤 平仓请求", extra={
                    'position_id': request.position_id,
                    'reason': request.closing_reason.value,
                    'quantity': request.quantity_to_close
                })
                
                # 执行平仓
                result = await self.position_manager.close_position(
                    request.position_id,
                    quantity=request.quantity_to_close,
                    reason=f"auto_close_{request.closing_reason.value}"
                )
                
                if result and result.success:
                    self.trade_stats['positions_closed'] += 1
                    self.trade_stats['total_pnl'] += result.realized_pnl or 0.0
                    
                    logger.info(f"✅ 平仓完成", extra={
                        'position_id': result.position_id,
                        'pnl': result.realized_pnl,
                        'close_price': result.close_price
                    })
                
        except Exception as e:
            logger.error(f"❌ 持仓监控失败: {e}")

    async def run_trading_loop(self, duration_minutes: int = 60):
        """运行主交易循环"""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        logger.info(f"🚀 开始交易循环", extra={
            'duration': f"{duration_minutes}分钟",
            'symbols': self.test_symbols
        })
        
        loop_count = 0
        
        try:
            while datetime.now() < end_time:
                loop_count += 1
                loop_start = datetime.now()
                
                logger.info(f"🔄 交易循环 #{loop_count}")
                
                # 为每个交易对生成信号
                for symbol in self.test_symbols:
                    signal = await self.generate_trading_signal(symbol)
                    if signal:
                        await self.execute_signal_if_valid(symbol, signal)
                    
                    # 避免API限流
                    await asyncio.sleep(1)
                
                # 监控现有持仓
                await self.monitor_positions()
                
                # 打印统计信息
                if loop_count % 5 == 0:  # 每5轮打印一次
                    await self.print_statistics()
                
                # 控制循环频率（每2分钟一轮）
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, 120 - loop_duration)  # 2分钟间隔
                if sleep_time > 0:
                    logger.info(f"😴 等待 {sleep_time:.1f} 秒进入下一轮")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 用户手动停止交易")
        except Exception as e:
            logger.error(f"❌ 交易循环异常: {e}")
        finally:
            await self.cleanup()

    async def print_statistics(self):
        """打印交易统计信息"""
        # 获取仓位管理器统计
        position_stats = self.position_manager.get_detailed_statistics()
        
        stats_info = {
            '📊 信号统计': {
                '总信号数': self.trade_stats['total_signals'],
                '强信号数': self.trade_stats['strong_signals'],
                '信号质量': f"{self.trade_stats['strong_signals']}/{self.trade_stats['total_signals']}" if self.trade_stats['total_signals'] > 0 else "0/0"
            },
            '💼 交易统计': {
                '开仓数': self.trade_stats['positions_opened'],
                '平仓数': self.trade_stats['positions_closed'],
                '当前持仓': len(position_stats.get('active_positions', [])),
                '累计盈亏': f"{self.trade_stats['total_pnl']:.2f} USDT"
            },
            '🎯 性能指标': {
                '胜率': f"{position_stats.get('win_rate', 0) * 100:.1f}%" if position_stats.get('win_rate') else "N/A",
                '平均盈利': f"{position_stats.get('avg_profit', 0):.2f} USDT" if position_stats.get('avg_profit') else "N/A",
                '最大回撤': f"{position_stats.get('max_drawdown', 0) * 100:.1f}%" if position_stats.get('max_drawdown') else "N/A"
            }
        }
        
        logger.info("📈 交易统计报告", extra=stats_info)

    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 开始清理资源...")
        
        try:
            # 关闭所有持仓（可选）
            if self.position_manager:
                positions = self.position_manager.auto_closer.get_all_positions()
                if positions:
                    logger.info(f"⚠️ 发现 {len(positions)} 个未平仓位，建议手动处理")
                
                await self.position_manager.stop()
            
            # 断开币安连接
            if self.binance_client:
                await self.binance_client.disconnect()
            
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 清理过程出错: {e}")

    async def run_test(self, duration_minutes: int = 60):
        """运行完整测试"""
        logger.info("🎯 开始币安模拟盘实时交易测试")
        
        # 初始化
        if not await self.initialize():
            logger.error("❌ 初始化失败，测试终止")
            return False
        
        try:
            # 运行交易循环
            await self.run_trading_loop(duration_minutes)
            
            # 最终统计
            logger.info("📊 最终统计报告")
            await self.print_statistics()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试执行失败: {e}")
            return False
        finally:
            await self.cleanup()


async def main():
    """主函数"""
    # 检查环境配置
    required_env = ['BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_env = [env for env in required_env if not os.getenv(env)]
    
    if missing_env:
        print(f"❌ 缺少必要的环境变量: {missing_env}")
        print("请复制 .env.example 为 .env 并配置币安API密钥")
        return
    
    if not os.getenv('BINANCE_TESTNET', 'false').lower() == 'true':
        print("⚠️ 警告: 建议设置 BINANCE_TESTNET=true 使用测试网进行测试")
        response = input("确定要在主网进行测试吗? (y/N): ")
        if response.lower() != 'y':
            return
    
    print("🎯 币安模拟盘实时交易测试")
    print("=" * 50)
    print("该测试将：")
    print("1. 连接币安API获取实时市场数据")
    print("2. 使用技术指标引擎生成交易信号")
    print("3. 通过仓位管理器执行模拟交易")
    print("4. 自动监控持仓并执行平仓策略")
    print()
    
    duration = input("请输入测试时长(分钟，默认60): ") or "60"
    try:
        duration = int(duration)
    except ValueError:
        duration = 60
    
    # 创建并运行测试
    test = BinanceLiveTradingTest()
    success = await test.run_test(duration)
    
    if success:
        print("✅ 测试完成！")
    else:
        print("❌ 测试失败！")


if __name__ == "__main__":
    # 设置事件循环策略 (Windows兼容)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())