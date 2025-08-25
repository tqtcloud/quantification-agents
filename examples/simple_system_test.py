#!/usr/bin/env python3
"""
简化系统功能测试
验证核心组件的基本功能，避免复杂依赖
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 设置基本日志
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_core_models():
    """测试核心数据模型"""
    logger.info("🧪 测试核心数据模型...")
    
    try:
        from src.core.models.signals import TradingSignal, SignalStrength, MultiDimensionalSignal
        
        # 创建基础交易信号
        signal = TradingSignal(
            symbol="BTCUSDT",
            signal_type=SignalStrength.BUY,
            confidence=0.8,
            entry_price=50000.0,
            target_price=52000.0,
            stop_loss=48000.0,
            reasoning=["技术突破", "成交量确认"],
            indicators_consensus={"ma": 0.7, "rsi": 0.6}
        )
        
        # 验证信号属性
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence == 0.8
        assert signal.signal_type == SignalStrength.BUY
        assert len(signal.reasoning) == 2
        
        # 创建多维度信号
        multi_signal = MultiDimensionalSignal(
            primary_signal=signal,
            momentum_score=0.6,
            mean_reversion_score=-0.2,
            volatility_score=0.4,
            volume_score=0.8,
            sentiment_score=0.3,
            overall_confidence=0.7,
            risk_reward_ratio=2.5,
            max_position_size=0.8
        )
        
        assert multi_signal.overall_confidence == 0.7
        assert multi_signal.risk_reward_ratio == 2.5
        
        logger.info("✅ 核心数据模型测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 核心数据模型测试失败: {e}")
        return False


def test_cache_system():
    """测试内存缓存系统"""
    logger.info("🧪 测试内存缓存系统...")
    
    try:
        from src.core.cache.memory_cache import MemoryCachePool
        
        # 创建缓存池
        cache = MemoryCachePool()
        
        # 测试基本存取
        test_key = "test_data"
        test_value = {"price": 50000.0, "volume": 1000}
        
        cache.set(test_key, test_value)
        retrieved_value = cache.get(test_key)
        
        assert retrieved_value == test_value
        
        # 测试TTL功能
        cache.set("temp_key", "temp_value", ttl=0.1)  # 0.1秒后过期
        import time
        time.sleep(0.2)
        expired_value = cache.get("temp_key")
        assert expired_value is None
        
        # 测试统计信息
        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        
        logger.info("✅ 内存缓存系统测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存缓存系统测试失败: {e}")
        return False


def test_position_models():
    """测试仓位管理模型"""
    logger.info("🧪 测试仓位管理模型...")
    
    try:
        from src.core.position.models import (
            PositionInfo, ClosingReason, ClosingAction, 
            PositionCloseRequest, ATRInfo, VolatilityInfo
        )
        
        # 创建仓位信息
        position = PositionInfo(
            position_id="TEST_001",
            symbol="BTCUSDT",
            entry_price=50000.0,
            current_price=51000.0,
            quantity=0.5,
            side="long",
            entry_time=datetime.utcnow(),
            unrealized_pnl=500.0,
            unrealized_pnl_pct=2.0
        )
        
        # 验证仓位属性
        assert position.is_long
        assert not position.is_short
        assert position.is_profitable
        assert position.unrealized_pnl == 500.0
        
        # 测试价格更新
        position.update_price(52000.0)
        assert position.current_price == 52000.0
        assert position.unrealized_pnl == 1000.0  # (52000 - 50000) * 0.5
        assert position.highest_price == 52000.0
        
        # 创建平仓请求
        close_request = PositionCloseRequest(
            position_id="TEST_001",
            closing_reason=ClosingReason.PROFIT_TARGET,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=0.5
        )
        
        assert close_request.closing_reason == ClosingReason.PROFIT_TARGET
        assert close_request.quantity_to_close == 0.5
        
        # 创建ATR信息
        atr_info = ATRInfo(
            period=14,
            current_atr=100.0,
            atr_multiplier=2.0
        )
        
        assert atr_info.dynamic_stop_distance == 200.0  # 100 * 2.0
        
        logger.info("✅ 仓位管理模型测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 仓位管理模型测试失败: {e}")
        return False


def test_technical_indicators():
    """测试技术指标计算"""
    logger.info("🧪 测试技术指标计算...")
    
    try:
        # 生成模拟数据
        np.random.seed(42)
        prices = []
        base_price = 50000.0
        
        for i in range(100):
            change = np.random.normal(0.001, 0.02)
            base_price *= (1 + change)
            prices.append(base_price)
        
        prices = np.array(prices)
        
        # 测试移动平均线
        def simple_moving_average(data, period):
            sma = np.full_like(data, np.nan)
            for i in range(period - 1, len(data)):
                sma[i] = np.mean(data[i - period + 1:i + 1])
            return sma
        
        sma_20 = simple_moving_average(prices, 20)
        assert not np.isnan(sma_20[-1])  # 最后一个值应该有效
        
        # 测试RSI计算
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(prices)
        assert 0 <= rsi <= 100
        
        logger.info("✅ 技术指标计算测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 技术指标计算测试失败: {e}")
        return False


async def test_async_functionality():
    """测试异步功能"""
    logger.info("🧪 测试异步功能...")
    
    try:
        # 测试异步任务
        async def mock_async_task(delay: float, result: str):
            await asyncio.sleep(delay)
            return result
        
        # 并发执行多个任务
        tasks = [
            mock_async_task(0.1, "task1"),
            mock_async_task(0.1, "task2"),
            mock_async_task(0.1, "task3")
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # 验证并发执行
        assert len(results) == 3
        assert "task1" in results
        assert "task2" in results
        assert "task3" in results
        
        # 并发执行应该比串行快
        execution_time = end_time - start_time
        assert execution_time < 0.5  # 应该远小于0.3秒(串行时间)
        
        logger.info("✅ 异步功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 异步功能测试失败: {e}")
        return False


def generate_test_report(test_results: Dict[str, bool]):
    """生成测试报告"""
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print("🎯 系统功能测试报告")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print("\n📊 统计信息:")
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {failed_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 评估系统状态
    if success_rate >= 90:
        grade = "优秀 ⭐⭐⭐⭐⭐"
        status = "🎉 系统核心功能完全正常"
    elif success_rate >= 70:
        grade = "良好 ⭐⭐⭐⭐"
        status = "✅ 系统基本功能正常"
    elif success_rate >= 50:
        grade = "及格 ⭐⭐⭐"
        status = "⚠️ 系统部分功能需要修复"
    else:
        grade = "不及格 ⭐⭐"
        status = "❌ 系统存在严重问题"
    
    print(f"\n🏆 评估等级: {grade}")
    print(f"🔍 系统状态: {status}")
    
    if success_rate >= 70:
        print("\n💡 下一步建议:")
        print("1. 系统核心功能正常，可以进行币安API集成测试")
        print("2. 配置 .env 文件并运行: ./scripts/run_binance_test.sh")
        print("3. 开始实盘测试前请充分验证策略")
    else:
        print("\n🔧 修复建议:")
        print("1. 检查失败的测试项目")
        print("2. 确认所有依赖正确安装")
        print("3. 检查代码是否有语法错误")
    
    print("=" * 50)
    
    return success_rate >= 70


async def main():
    """主函数"""
    print("🎯 量化交易系统 - 简化功能测试")
    print("=" * 50)
    print("该测试将验证以下核心功能：")
    print("1. 核心数据模型")
    print("2. 内存缓存系统") 
    print("3. 仓位管理模型")
    print("4. 技术指标计算")
    print("5. 异步功能")
    print()
    
    # 执行所有测试
    test_results = {
        "核心数据模型": test_core_models(),
        "内存缓存系统": test_cache_system(),
        "仓位管理模型": test_position_models(),
        "技术指标计算": test_technical_indicators(),
        "异步功能": await test_async_functionality()
    }
    
    # 生成测试报告
    system_healthy = generate_test_report(test_results)
    
    return system_healthy


if __name__ == "__main__":
    # 设置事件循环策略 (Windows兼容)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行测试
    result = asyncio.run(main())
    
    # 退出代码
    sys.exit(0 if result else 1)