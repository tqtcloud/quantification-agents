"""
多维度技术指标引擎使用示例

展示如何使用MultiDimensionalIndicatorEngine生成综合交易信号
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine
from src.core.indicators.timeframe import TimeFrame
from src.core.models.signals import SignalStrength


class MarketDataSimulator:
    """市场数据模拟器"""
    
    @staticmethod
    def generate_trending_data(n_points: int = 200, trend_strength: float = 0.1) -> dict:
        """生成趋势性数据"""
        np.random.seed(42)
        
        # 生成基础价格序列
        base_price = 100.0
        trend = np.linspace(0, trend_strength * n_points, n_points)
        noise = np.random.normal(0, 1, n_points)
        
        closes = base_price + trend + noise
        
        # 生成OHLC数据
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        # 生成高低价
        daily_range = np.random.uniform(0.5, 3.0, n_points)
        highs = closes + daily_range * np.random.uniform(0.3, 1.0, n_points)
        lows = closes - daily_range * np.random.uniform(0.3, 1.0, n_points)
        
        # 确保价格逻辑正确
        for i in range(n_points):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        # 生成成交量（趋势期间成交量可能增加）
        base_volume = 10000
        volume_trend = np.linspace(0, 5000, n_points) if trend_strength > 0 else np.zeros(n_points)
        volume_noise = np.random.uniform(-2000, 2000, n_points)
        volumes = base_volume + volume_trend + volume_noise
        volumes = np.maximum(volumes, 1000)  # 确保成交量为正
        
        return {
            'open': opens.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }
    
    @staticmethod
    def generate_sideways_data(n_points: int = 200, volatility: float = 1.0) -> dict:
        """生成横盘整理数据"""
        np.random.seed(123)
        
        base_price = 100.0
        # 横盘数据：无明显趋势，但有波动
        noise = np.random.normal(0, volatility, n_points)
        # 添加一些周期性波动
        cycle = 2 * volatility * np.sin(np.linspace(0, 4*np.pi, n_points))
        
        closes = base_price + noise + cycle
        
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        daily_range = np.random.uniform(0.5, 2.5, n_points)
        highs = closes + daily_range * np.random.uniform(0.2, 0.8, n_points)
        lows = closes - daily_range * np.random.uniform(0.2, 0.8, n_points)
        
        # 确保价格逻辑
        for i in range(n_points):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        # 横盘期间成交量通常较低且稳定
        volumes = np.random.uniform(5000, 12000, n_points)
        
        return {
            'open': opens.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }
    
    @staticmethod
    def generate_volatile_data(n_points: int = 200, volatility: float = 3.0) -> dict:
        """生成高波动率数据"""
        np.random.seed(789)
        
        base_price = 100.0
        # 高波动率：大幅随机波动
        large_moves = np.random.choice([-1, 1], n_points) * np.random.exponential(volatility, n_points)
        noise = np.random.normal(0, volatility*0.5, n_points)
        
        closes = base_price + np.cumsum(large_moves + noise)
        
        opens = np.roll(closes, 1)
        opens[0] = base_price
        
        # 高波动期间日内幅度也大
        daily_range = np.random.uniform(2.0, 8.0, n_points)
        highs = closes + daily_range * np.random.uniform(0.3, 1.0, n_points)
        lows = closes - daily_range * np.random.uniform(0.3, 1.0, n_points)
        
        for i in range(n_points):
            highs[i] = max(opens[i], closes[i], highs[i])
            lows[i] = min(opens[i], closes[i], lows[i])
        
        # 高波动期间成交量通常激增
        volumes = np.random.uniform(15000, 35000, n_points)
        
        return {
            'open': opens.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist(),
            'close': closes.tolist(),
            'volume': volumes.tolist()
        }


async def analyze_market_scenario(
    engine: MultiDimensionalIndicatorEngine,
    scenario_name: str,
    market_data: dict,
    enable_multiframe: bool = True
) -> None:
    """分析市场场景"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"分析场景: {scenario_name}")
    logger.info(f"{'='*60}")
    
    try:
        # 生成多维度信号
        signal = await engine.generate_multidimensional_signal(
            symbol="DEMO/USDT",
            market_data=market_data,
            enable_multiframe_analysis=enable_multiframe
        )
        
        if signal is None:
            logger.info("❌ 未生成有效信号（可能是中性市场）")
            return
        
        # 显示主要信号信息
        logger.info(f"📊 主要信号:")
        logger.info(f"  • 信号类型: {signal.primary_signal.signal_type.name}")
        logger.info(f"  • 置信度: {signal.primary_signal.confidence:.3f}")
        logger.info(f"  • 入场价格: {signal.primary_signal.entry_price:.2f}")
        logger.info(f"  • 目标价格: {signal.primary_signal.target_price:.2f}")
        logger.info(f"  • 止损价格: {signal.primary_signal.stop_loss:.2f}")
        logger.info(f"  • 风险收益比: {signal.primary_signal.risk_reward_ratio:.2f}")
        
        # 显示多维度分析
        logger.info(f"\n🔍 多维度分析:")
        logger.info(f"  • 动量评分: {signal.momentum_score:+.3f}")
        logger.info(f"  • 均值回归评分: {signal.mean_reversion_score:+.3f}")
        logger.info(f"  • 波动率评分: {signal.volatility_score:.3f}")
        logger.info(f"  • 成交量评分: {signal.volume_score:.3f}")
        logger.info(f"  • 情绪评分: {signal.sentiment_score:+.3f}")
        logger.info(f"  • 综合置信度: {signal.overall_confidence:.3f}")
        
        # 显示质量评估
        logger.info(f"\n📈 信号质量评估:")
        logger.info(f"  • 信号质量评分: {signal.signal_quality_score:.3f}")
        logger.info(f"  • 方向一致性: {signal.signal_direction_consensus:+.3f}")
        logger.info(f"  • 市场状态: {signal.market_regime}")
        logger.info(f"  • 建议最大仓位: {signal.max_position_size:.1%}")
        
        # 显示仓位建议
        conservative_position = signal.get_position_sizing_recommendation(
            base_position_size=1.0, risk_tolerance=0.5
        )
        aggressive_position = signal.get_position_sizing_recommendation(
            base_position_size=1.0, risk_tolerance=1.0
        )
        
        logger.info(f"\n💼 仓位建议:")
        logger.info(f"  • 保守策略: {conservative_position:.1%}")
        logger.info(f"  • 积极策略: {aggressive_position:.1%}")
        
        # 显示推理过程
        logger.info(f"\n🧠 决策推理:")
        for i, reason in enumerate(signal.primary_signal.reasoning, 1):
            logger.info(f"  {i}. {reason}")
        
        # 显示技术位信息
        if signal.technical_levels:
            logger.info(f"\n📏 关键技术位:")
            for level_name, level_price in signal.technical_levels.items():
                logger.info(f"  • {level_name}: {level_price:.2f}")
        
    except Exception as e:
        logger.error(f"❌ 分析场景时发生错误: {str(e)}")


async def performance_comparison():
    """性能对比测试"""
    logger.info(f"\n{'='*60}")
    logger.info("性能对比测试")
    logger.info(f"{'='*60}")
    
    # 测试不同工作线程数的性能
    import time
    
    test_data = MarketDataSimulator.generate_trending_data(500, 0.2)
    
    for workers in [2, 4, 8]:
        engine = MultiDimensionalIndicatorEngine(max_workers=workers)
        
        try:
            start_time = time.time()
            
            # 并发生成多个信号
            tasks = []
            for i in range(10):
                task = engine.generate_multidimensional_signal(
                    f"TEST{i}/USDT",
                    test_data,
                    enable_multiframe_analysis=True
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 统计结果
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            logger.info(f"工作线程数: {workers}")
            logger.info(f"  • 处理时间: {processing_time:.3f}s")
            logger.info(f"  • 成功信号: {successful}/10")
            logger.info(f"  • 失败信号: {failed}/10")
            logger.info(f"  • 平均每信号: {processing_time/10:.3f}s")
            
            # 显示引擎统计
            stats = engine.get_performance_stats()
            logger.info(f"  • 引擎统计: {stats}")
            
        finally:
            engine.cleanup()


async def main():
    """主函数"""
    logger.info("🚀 多维度技术指标引擎示例")
    
    # 创建引擎实例
    engine = MultiDimensionalIndicatorEngine(max_workers=4)
    
    try:
        # 1. 上涨趋势场景
        trending_up_data = MarketDataSimulator.generate_trending_data(200, 0.15)
        await analyze_market_scenario(
            engine, "强势上涨趋势", trending_up_data, enable_multiframe=True
        )
        
        # 2. 下跌趋势场景
        trending_down_data = MarketDataSimulator.generate_trending_data(200, -0.12)
        await analyze_market_scenario(
            engine, "明显下跌趋势", trending_down_data, enable_multiframe=True
        )
        
        # 3. 横盘整理场景
        sideways_data = MarketDataSimulator.generate_sideways_data(200, 1.5)
        await analyze_market_scenario(
            engine, "横盘整理", sideways_data, enable_multiframe=False
        )
        
        # 4. 高波动率场景
        volatile_data = MarketDataSimulator.generate_volatile_data(200, 2.5)
        await analyze_market_scenario(
            engine, "高波动率市场", volatile_data, enable_multiframe=True
        )
        
        # 5. 性能对比测试
        await performance_comparison()
        
        # 显示最终统计
        logger.info(f"\n{'='*60}")
        logger.info("最终引擎统计")
        logger.info(f"{'='*60}")
        
        final_stats = engine.get_performance_stats()
        for stat_name, stat_value in final_stats.items():
            logger.info(f"{stat_name}: {stat_value}")
        
    except Exception as e:
        logger.error(f"❌ 运行示例时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理资源
        engine.cleanup()
        logger.info("✅ 示例运行完成，资源已清理")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())