#!/usr/bin/env python3
"""
快速信号测试脚本
不需要币安API，使用模拟数据快速验证技术指标引擎和信号生成功能

适用场景：
1. 快速验证系统功能
2. 开发调试
3. 无API密钥的情况下测试
"""

import asyncio
import logging
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.core.position.auto_position_closer import AutoPositionCloser
from src.core.models.signals import MultiDimensionalSignal, TradingSignal, SignalStrength
from src.utils.logger import setup_logging

# 设置日志
logger = setup_logging("quick_signal_test", log_level="INFO")


class QuickSignalTest:
    """快速信号测试类"""
    
    def __init__(self):
        """初始化测试"""
        self.indicator_engine = MultiDimensionalIndicatorEngine()
        self.auto_closer = AutoPositionCloser()
        self.test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        logger.info("🚀 快速信号测试初始化完成")

    def generate_mock_data(self, 
                          symbol: str, 
                          length: int = 200,
                          base_price: float = 50000.0,
                          trend: float = 0.001,
                          volatility: float = 0.02) -> List[Dict[str, Any]]:
        """生成模拟市场数据"""
        np.random.seed(42)  # 确保结果可重复
        
        data = []
        current_time = datetime.now() - timedelta(hours=length)
        current_price = base_price
        
        for i in range(length):
            # 生成价格变化
            random_change = np.random.normal(trend, volatility)
            current_price *= (1 + random_change)
            
            # 添加一些趋势和周期性
            trend_factor = np.sin(i * 0.05) * 0.005  # 周期性趋势
            current_price *= (1 + trend_factor)
            
            # 生成OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            volume = np.random.uniform(1000, 10000)
            
            # 确保OHLC逻辑正确
            prices = [open_price, current_price, high, low]
            high = max(prices)
            low = min(prices)
            
            data.append({
                'timestamp': current_time.timestamp() + i * 3600,
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })
        
        logger.info(f"📊 生成 {symbol} 模拟数据", extra={
            'length': length,
            'price_range': f"{min(d['close'] for d in data):.2f} - {max(d['close'] for d in data):.2f}"
        })
        
        return data

    async def test_signal_generation(self, symbol: str) -> List[MultiDimensionalSignal]:
        """测试信号生成"""
        logger.info(f"🎯 开始测试 {symbol} 信号生成")
        
        # 生成模拟数据
        mock_data = self.generate_mock_data(symbol)
        
        # 喂入历史数据
        for data_point in mock_data[:-20]:  # 留下最后20个点用于实时模拟
            self.indicator_engine.update_market_data(symbol, data_point)
        
        signals = []
        
        # 模拟实时数据流
        for i, data_point in enumerate(mock_data[-20:]):
            # 更新数据
            self.indicator_engine.update_market_data(symbol, data_point)
            
            # 生成信号
            signal = await self.indicator_engine.generate_multidimensional_signal(
                symbol=symbol,
                timeframe='1h'
            )
            
            if signal:
                signals.append(signal)
                logger.info(f"📈 {symbol} 第{i+1}个信号", extra={
                    'timestamp': datetime.fromtimestamp(data_point['timestamp']).strftime('%H:%M:%S'),
                    'price': f"{data_point['close']:.2f}",
                    'signal_type': signal.primary_signal.signal_type.value if signal.primary_signal else 'None',
                    'confidence': f"{signal.overall_confidence:.3f}",
                    'momentum': f"{signal.momentum_score:.3f}",
                    'volatility': f"{signal.volatility_score:.3f}",
                    'volume': f"{signal.volume_score:.3f}",
                    'sentiment': f"{signal.sentiment_score:.3f}",
                    'risk_reward': f"{signal.risk_reward_ratio:.2f}"
                })
            
            # 模拟延迟
            await asyncio.sleep(0.1)
        
        return signals

    async def test_position_management(self, signals: List[MultiDimensionalSignal]) -> Dict[str, Any]:
        """测试仓位管理"""
        logger.info("🏦 开始测试仓位管理")
        
        position_stats = {
            'positions_opened': 0,
            'positions_closed': 0,
            'total_signals': len(signals),
            'strong_signals': 0,
            'successful_closes': 0
        }
        
        current_price = 50000.0
        
        for i, signal in enumerate(signals):
            # 统计强信号
            if signal.overall_confidence > 0.7:
                position_stats['strong_signals'] += 1
            
            # 只对高质量信号开仓
            if (signal.overall_confidence > 0.6 and 
                signal.risk_reward_ratio > 1.5 and
                signal.primary_signal):
                
                # 模拟价格变化
                price_change = np.random.normal(0.001, 0.02)
                current_price *= (1 + price_change)
                
                # 确定交易方向
                if signal.primary_signal.signal_type in [SignalStrength.BUY, SignalStrength.STRONG_BUY]:
                    side = 'long'
                elif signal.primary_signal.signal_type in [SignalStrength.SELL, SignalStrength.STRONG_SELL]:
                    side = 'short'
                else:
                    continue
                
                # 添加仓位到自动平仓器
                position_id = f"TEST_{side}_{i}_{int(datetime.now().timestamp())}"
                
                # 创建模拟仓位
                from src.core.position.models import PositionInfo
                position = PositionInfo(
                    position_id=position_id,
                    symbol="TESTUSDT",
                    entry_price=current_price,
                    current_price=current_price,
                    quantity=0.1,
                    side=side,
                    entry_time=datetime.now(),
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    stop_loss=signal.primary_signal.stop_loss,
                    take_profit=signal.primary_signal.target_price
                )
                
                self.auto_closer.add_position(position)
                position_stats['positions_opened'] += 1
                
                logger.info(f"📈 开仓 {position_id}", extra={
                    'side': side,
                    'entry_price': f"{current_price:.2f}",
                    'confidence': f"{signal.overall_confidence:.3f}"
                })
                
                # 模拟价格变动并检查平仓条件
                for _ in range(10):  # 模拟10次价格变动
                    price_change = np.random.normal(0, 0.01)
                    current_price *= (1 + price_change)
                    
                    # 更新仓位价格
                    self.auto_closer.update_position_price(position_id, current_price)
                    
                    # 检查平仓条件
                    close_requests = await self.auto_closer.check_closing_conditions([position_id])
                    
                    if close_requests:
                        position_stats['positions_closed'] += 1
                        position_stats['successful_closes'] += 1
                        
                        logger.info(f"📉 平仓 {position_id}", extra={
                            'reason': close_requests[0].closing_reason.value,
                            'close_price': f"{current_price:.2f}",
                            'pnl_pct': f"{position.unrealized_pnl_pct:.2f}%"
                        })
                        break
                    
                    await asyncio.sleep(0.05)
        
        return position_stats

    async def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("🎯 开始综合信号和仓位管理测试")
        
        all_signals = []
        
        # 为每个测试标的生成信号
        for symbol in self.test_symbols:
            signals = await self.test_signal_generation(symbol)
            all_signals.extend(signals)
        
        # 测试仓位管理
        position_stats = await self.test_position_management(all_signals)
        
        # 生成测试报告
        await self.generate_test_report(all_signals, position_stats)

    async def generate_test_report(self, signals: List[MultiDimensionalSignal], position_stats: Dict[str, Any]):
        """生成测试报告"""
        logger.info("📊 生成测试报告")
        
        if not signals:
            logger.warning("⚠️ 没有生成任何信号")
            return
        
        # 信号质量分析
        signal_quality = {
            'total_signals': len(signals),
            'high_confidence': sum(1 for s in signals if s.overall_confidence > 0.8),
            'medium_confidence': sum(1 for s in signals if 0.6 <= s.overall_confidence <= 0.8),
            'low_confidence': sum(1 for s in signals if s.overall_confidence < 0.6),
            'avg_confidence': np.mean([s.overall_confidence for s in signals]),
            'avg_risk_reward': np.mean([s.risk_reward_ratio for s in signals])
        }
        
        # 信号类型分布
        signal_types = {}
        for signal in signals:
            if signal.primary_signal:
                signal_type = signal.primary_signal.signal_type.value
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        # 维度得分统计
        dimension_stats = {
            'avg_momentum': np.mean([s.momentum_score for s in signals]),
            'avg_volatility': np.mean([s.volatility_score for s in signals]),
            'avg_volume': np.mean([s.volume_score for s in signals]),
            'avg_sentiment': np.mean([s.sentiment_score for s in signals])
        }
        
        # 打印详细报告
        report = {
            '📈 信号质量分析': {
                '总信号数': signal_quality['total_signals'],
                '高信心信号': f"{signal_quality['high_confidence']} ({signal_quality['high_confidence']/signal_quality['total_signals']*100:.1f}%)",
                '中信心信号': f"{signal_quality['medium_confidence']} ({signal_quality['medium_confidence']/signal_quality['total_signals']*100:.1f}%)",
                '低信心信号': f"{signal_quality['low_confidence']} ({signal_quality['low_confidence']/signal_quality['total_signals']*100:.1f}%)",
                '平均信心度': f"{signal_quality['avg_confidence']:.3f}",
                '平均风险回报比': f"{signal_quality['avg_risk_reward']:.2f}"
            },
            '📊 信号类型分布': signal_types,
            '🎯 维度得分统计': {
                '动量得分': f"{dimension_stats['avg_momentum']:.3f}",
                '波动率得分': f"{dimension_stats['avg_volatility']:.3f}",
                '成交量得分': f"{dimension_stats['avg_volume']:.3f}",
                '情绪得分': f"{dimension_stats['avg_sentiment']:.3f}"
            },
            '💼 仓位管理统计': {
                '开仓数量': position_stats['positions_opened'],
                '平仓数量': position_stats['positions_closed'],
                '强信号数': position_stats['strong_signals'],
                '成功平仓': position_stats['successful_closes'],
                '开仓成功率': f"{position_stats['positions_opened']/position_stats['strong_signals']*100:.1f}%" if position_stats['strong_signals'] > 0 else "N/A",
                '平仓成功率': f"{position_stats['successful_closes']/position_stats['positions_opened']*100:.1f}%" if position_stats['positions_opened'] > 0 else "N/A"
            }
        }
        
        # 使用结构化日志输出报告
        logger.info("🎉 测试完成 - 详细报告", extra=report)
        
        # 评估测试结果
        self.evaluate_test_results(signal_quality, position_stats)

    def evaluate_test_results(self, signal_quality: Dict, position_stats: Dict):
        """评估测试结果"""
        score = 0
        max_score = 100
        
        # 信号质量评分 (40分)
        if signal_quality['avg_confidence'] > 0.7:
            score += 20
        elif signal_quality['avg_confidence'] > 0.5:
            score += 10
        
        if signal_quality['avg_risk_reward'] > 2.0:
            score += 20
        elif signal_quality['avg_risk_reward'] > 1.5:
            score += 10
        
        # 信号数量评分 (20分)
        if signal_quality['total_signals'] > 10:
            score += 20
        elif signal_quality['total_signals'] > 5:
            score += 10
        
        # 仓位管理评分 (40分)
        if position_stats['positions_opened'] > 0:
            score += 20
            
            close_rate = position_stats['successful_closes'] / position_stats['positions_opened']
            if close_rate > 0.8:
                score += 20
            elif close_rate > 0.5:
                score += 10
        
        # 评级
        if score >= 80:
            grade = "优秀 ⭐⭐⭐⭐⭐"
        elif score >= 60:
            grade = "良好 ⭐⭐⭐⭐"
        elif score >= 40:
            grade = "及格 ⭐⭐⭐"
        else:
            grade = "需改进 ⭐⭐"
        
        logger.info("🏆 测试评估结果", extra={
            'total_score': f"{score}/{max_score}",
            'grade': grade,
            'system_status': '✅ 系统功能正常' if score >= 60 else '⚠️ 系统需要优化'
        })


async def main():
    """主函数"""
    print("🎯 快速信号测试")
    print("=" * 50)
    print("该测试将使用模拟数据验证：")
    print("1. 技术指标引擎信号生成")
    print("2. 多维度信号评分")
    print("3. 自动平仓策略")
    print("4. 综合系统性能")
    print()
    
    # 创建并运行测试
    test = QuickSignalTest()
    await test.run_comprehensive_test()
    
    print("✅ 快速测试完成！")
    print("如果测试评分良好，可以继续进行币安API实盘测试")


if __name__ == "__main__":
    # 设置事件循环策略 (Windows兼容)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())