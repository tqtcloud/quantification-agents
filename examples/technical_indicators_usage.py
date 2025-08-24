"""
技术指标计算引擎使用示例
展示如何使用新的技术指标系统进行各种计算和分析
"""

import asyncio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, List

# 导入技术指标模块
from src.core.indicators import (
    TechnicalIndicators,
    RelativeStrengthIndex,
    MACD,
    SimpleMovingAverage,
    ExponentialMovingAverage,
    BollingerBands,
    StochasticOscillator,
    CCI,
    WilliamsR,
    ADX,
    ParabolicSAR,
    IchimokuCloud,
    ATR,
    StandardDeviation,
    KeltnerChannels,
    DonchianChannels,
    VIXProxy,
    IndicatorNormalizer,
    NormalizationMethod,
    NormalizationConfig,
    TimeFrameManager,
    TimeFrame,
    TimeFrameConfig,
    OHLCVData,
    CustomIndicator,
    calculate_sma,
    calculate_ema,
    calculate_rsi
)


class TechnicalIndicatorDemo:
    """技术指标演示类"""
    
    def __init__(self):
        self.indicators_manager = TechnicalIndicators()
        self.timeframe_manager = TimeFrameManager()
        self.normalizer = IndicatorNormalizer()
        
    def generate_sample_data(self, length: int = 200, symbol: str = "BTCUSDT") -> Dict[str, np.ndarray]:
        """生成样本价格数据"""
        np.random.seed(42)
        
        # 模拟比特币价格走势
        base_price = 50000
        returns = np.random.normal(0.001, 0.02, length)  # 平均上涨趋势
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices[1:])
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price
        
        # 生成高低价
        volatility = 0.015
        high_low_spread = np.random.uniform(0.005, 0.025, length)
        high_prices = close_prices * (1 + high_low_spread)
        low_prices = close_prices * (1 - high_low_spread)
        
        # 确保OHLC逻辑正确
        for i in range(length):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        volumes = np.random.uniform(100000, 1000000, length)
        timestamps = np.arange(length) * 3600  # 每小时数据
        
        return {
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }
    
    def demo_basic_indicators(self):
        """基础指标使用演示"""
        print("=== 基础技术指标演示 ===")
        
        # 生成测试数据
        sample_data = self.generate_sample_data(100)
        symbol = "DEMO_BASIC"
        
        # 批量更新数据
        df = pd.DataFrame(sample_data)
        self.indicators_manager.update_data_batch(symbol, df)
        
        # 计算所有默认指标
        results = self.indicators_manager.calculate_all_indicators(symbol)
        
        print(f"计算了 {len(results)} 个指标:")
        for name, result in results.items():
            if isinstance(result.value, dict):
                print(f"{name}: {result.value}")
            else:
                print(f"{name}: {result.value:.4f}")
            print(f"  - 信号: {result.metadata.get('signal', 'N/A')}")
            print()
    
    def demo_custom_indicators(self):
        """自定义指标演示"""
        print("=== 自定义指标演示 ===")
        
        # 定义自定义动量指标
        def custom_momentum(data: Dict[str, np.ndarray], params: Dict) -> float:
            close_prices = data.get('close', np.array([]))
            volume = data.get('volume', np.array([]))
            
            if len(close_prices) < 2:
                return np.nan
            
            # 价格变化率
            price_change = (close_prices[-1] / close_prices[-2] - 1) * 100
            
            # 成交量相对强度
            volume_strength = 1
            if len(volume) >= 10:
                recent_vol = np.mean(volume[-5:])
                long_vol = np.mean(volume[-10:])
                volume_strength = recent_vol / long_vol if long_vol > 0 else 1
            
            return price_change * volume_strength
        
        # 注册自定义指标
        self.indicators_manager.add_custom_indicator(
            name="CustomMomentum",
            calculation_func=custom_momentum,
            parameters={}
        )
        
        # 测试自定义指标
        sample_data = self.generate_sample_data(50)
        symbol = "DEMO_CUSTOM"
        
        df = pd.DataFrame(sample_data)
        self.indicators_manager.update_data_batch(symbol, df)
        
        result = self.indicators_manager.calculate_indicator("CustomMomentum", symbol)
        print(f"自定义动量指标结果: {result.value:.4f}")
        print(f"元数据: {result.metadata}")
        print()
    
    async def demo_async_calculations(self):
        """异步计算演示"""
        print("=== 异步计算演示 ===")
        
        symbols = ["BTC", "ETH", "ADA", "DOT", "SOL"]
        tasks = []
        
        # 为每个symbol准备数据并创建异步任务
        for symbol in symbols:
            sample_data = self.generate_sample_data(80, symbol=symbol)
            df = pd.DataFrame(sample_data)
            self.indicators_manager.update_data_batch(symbol, df)
            
            # 创建异步计算任务
            task = self.indicators_manager.calculate_all_indicators_async(symbol)
            tasks.append((symbol, task))
        
        # 并行执行所有计算
        start_time = time.time()
        results_list = await asyncio.gather(*[task for _, task in tasks])
        async_time = time.time() - start_time
        
        print(f"异步计算 {len(symbols)} 个symbol完成，用时: {async_time:.2f}秒")
        
        for i, (symbol, _) in enumerate(tasks):
            results = results_list[i]
            print(f"{symbol}: 计算了 {len(results)} 个指标")
        
        # 比较同步计算时间
        start_time = time.time()
        for symbol in symbols:
            self.indicators_manager.calculate_all_indicators(symbol)
        sync_time = time.time() - start_time
        
        print(f"同步计算用时: {sync_time:.2f}秒")
        print(f"异步加速比: {sync_time/async_time:.2f}x")
        print()
    
    def demo_normalization(self):
        """指标标准化演示"""
        print("=== 指标标准化演示 ===")
        
        # 生成RSI数据
        sample_data = self.generate_sample_data(100)
        rsi_values = calculate_rsi(sample_data['close'], 14)
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        
        print(f"原始RSI范围: [{np.min(valid_rsi):.2f}, {np.max(valid_rsi):.2f}]")
        
        # 不同标准化方法演示
        methods = [
            NormalizationMethod.MIN_MAX,
            NormalizationMethod.Z_SCORE,
            NormalizationMethod.PERCENTILE,
            NormalizationMethod.TANH
        ]
        
        for method in methods:
            normalized = self.normalizer.normalize(
                valid_rsi, 
                "RSI_14", 
                method=method,
                target_range=(-1, 1)
            )
            
            print(f"{method.value}标准化后范围: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        
        print()
    
    def demo_multi_timeframe(self):
        """多时间框架演示"""
        print("=== 多时间框架演示 ===")
        
        symbol = "BTCUSDT"
        
        # 注册不同时间框架的RSI指标
        timeframes = [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.MINUTE_15, TimeFrame.HOUR_1]
        
        for tf in timeframes:
            rsi_indicator = RelativeStrengthIndex(period=14)
            self.timeframe_manager.register_indicator_for_timeframe(tf, rsi_indicator)
        
        # 模拟tick数据输入
        base_time = time.time()
        base_price = 50000
        
        for i in range(300):  # 5分钟的数据，每秒一个tick
            timestamp = base_time + i
            price = base_price + np.sin(i/10) * 100 + np.random.normal(0, 50)
            volume = np.random.uniform(10, 100)
            
            self.timeframe_manager.update_tick_data(symbol, timestamp, price, volume)
        
        # 获取多时间框架结果
        multi_tf_results = self.timeframe_manager.get_multi_timeframe_results(symbol, "RSI_14")
        
        print(f"多时间框架RSI结果:")
        for tf, result in multi_tf_results.items():
            if result and result.is_valid:
                print(f"{tf.value}: {result.value:.2f} (信号: {result.metadata.get('signal', 'N/A')})")
        
        print()
    
    def demo_advanced_indicators(self):
        """高级指标演示"""
        print("=== 高级技术指标演示 ===")
        
        sample_data = self.generate_sample_data(100)
        
        # 测试ADX
        adx_indicator = ADX(period=14)
        adx_result = adx_indicator.calculate(sample_data, "DEMO")
        print(f"ADX: {adx_result.value}")
        print(f"趋势强度: {adx_result.metadata['trend_strength']}")
        print(f"趋势方向: {adx_result.metadata['trend_direction']}")
        print()
        
        # 测试一目均衡表
        ichimoku_indicator = IchimokuCloud()
        ichimoku_result = ichimoku_indicator.calculate(sample_data, "DEMO")
        print(f"一目均衡表: {ichimoku_result.value}")
        print(f"信号: {ichimoku_result.metadata['signal']}")
        print(f"云颜色: {ichimoku_result.metadata['cloud_color']}")
        print(f"价格vs云: {ichimoku_result.metadata['price_vs_cloud']}")
        print()
        
        # 测试VIX代理指标
        vix_indicator = VIXProxy(period=30)
        vix_result = vix_indicator.calculate(sample_data, "DEMO")
        if vix_result.is_valid:
            print(f"VIX代理: {vix_result.value:.2f}")
            print(f"恐慌水平: {vix_result.metadata['fear_level']}")
            print(f"市场压力: {vix_result.metadata['market_stress']}")
        print()
    
    def demo_performance_monitoring(self):
        """性能监控演示"""
        print("=== 性能监控演示 ===")
        
        # 大量计算测试
        symbols = [f"SYMBOL_{i}" for i in range(10)]
        
        for symbol in symbols:
            sample_data = self.generate_sample_data(500)
            df = pd.DataFrame(sample_data)
            self.indicators_manager.update_data_batch(symbol, df)
            
            # 强制重新计算多次
            for _ in range(5):
                self.indicators_manager.calculate_all_indicators(symbol, force_recalculate=True)
        
        # 获取性能统计
        perf_stats = self.indicators_manager.get_global_performance_stats()
        
        print(f"总计算次数: {perf_stats['total_calculations']}")
        print(f"总计算时间: {perf_stats['total_time']:.3f}秒")
        print(f"平均计算时间: {perf_stats['avg_calculation_time']:.4f}秒")
        print(f"缓存命中率: {perf_stats['cache_hit_rate']:.2%}")
        
        print("\n各指标性能统计:")
        for indicator_name, stats in perf_stats['indicator_stats'].items():
            if stats['calculation_count'] > 0:
                print(f"{indicator_name}: {stats['average_time']:.4f}秒 "
                     f"(计算{stats['calculation_count']}次, 错误{stats['error_count']}次)")
        
        print()
    
    def demo_data_export(self):
        """数据导出演示"""
        print("=== 数据导出演示 ===")
        
        symbol = "EXPORT_DEMO"
        sample_data = self.generate_sample_data(50)
        df = pd.DataFrame(sample_data)
        
        self.indicators_manager.update_data_batch(symbol, df)
        self.indicators_manager.calculate_all_indicators(symbol)
        
        # 导出为pandas DataFrame
        exported_df = self.indicators_manager.export_data(symbol, format='pandas')
        print(f"导出DataFrame形状: {exported_df.shape}")
        print(f"列名: {list(exported_df.columns)}")
        print()
        
        # 保存到CSV文件
        exported_df.to_csv(f'/tmp/technical_indicators_{symbol}.csv', index=False)
        print(f"数据已保存到 /tmp/technical_indicators_{symbol}.csv")
        print()
    
    def demo_visualization(self):
        """可视化演示"""
        print("=== 技术指标可视化演示 ===")
        
        # 生成较长的数据序列用于可视化
        sample_data = self.generate_sample_data(200)
        symbol = "VIZ_DEMO"
        
        df = pd.DataFrame(sample_data)
        self.indicators_manager.update_data_batch(symbol, df)
        
        # 计算一些关键指标
        sma20 = SimpleMovingAverage(20)
        sma50 = SimpleMovingAverage(50)
        rsi = RelativeStrengthIndex(14)
        macd_indicator = MACD()
        bb = BollingerBands()
        
        # 手动计算指标用于绘图
        sma20_result = sma20.calculate(sample_data, symbol)
        sma50_result = sma50.calculate(sample_data, symbol)
        rsi_result = rsi.calculate(sample_data, symbol)
        macd_result = macd_indicator.calculate(sample_data, symbol)
        bb_result = bb.calculate(sample_data, symbol)
        
        # 计算完整的指标序列用于绘图
        sma20_series = calculate_sma(sample_data['close'], 20)
        sma50_series = calculate_sma(sample_data['close'], 50)
        rsi_series = calculate_rsi(sample_data['close'], 14)
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 价格和移动平均线
        axes[0].plot(sample_data['close'], label='Price', linewidth=1)
        axes[0].plot(sma20_series, label='SMA20', alpha=0.7)
        axes[0].plot(sma50_series, label='SMA50', alpha=0.7)
        axes[0].set_title('Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(rsi_series, label='RSI', color='purple')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[1].set_title('RSI (Relative Strength Index)')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 成交量
        axes[2].bar(range(len(sample_data['volume'])), sample_data['volume'], alpha=0.6, color='gray')
        axes[2].set_title('Volume')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('/tmp/technical_indicators_demo.png', dpi=150, bbox_inches='tight')
        print("图表已保存到 /tmp/technical_indicators_demo.png")
        
        # 显示最新指标值
        print(f"\n最新指标值:")
        print(f"价格: ${sample_data['close'][-1]:.2f}")
        print(f"SMA20: ${sma20_result.value:.2f} ({sma20_result.metadata['trend']})")
        print(f"SMA50: ${sma50_result.value:.2f} ({sma50_result.metadata['trend']})")
        print(f"RSI: {rsi_result.value:.2f} ({rsi_result.metadata['signal']})")
        
        if isinstance(macd_result.value, dict):
            print(f"MACD: {macd_result.value['macd']:.4f} ({macd_result.metadata['signal']})")
        
        plt.show()
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("========================================")
        print("技术指标计算引擎完整演示")
        print("========================================\n")
        
        # 基础功能演示
        self.demo_basic_indicators()
        
        # 自定义指标演示
        self.demo_custom_indicators()
        
        # 异步计算演示
        print("运行异步计算演示...")
        asyncio.run(self.demo_async_calculations())
        
        # 标准化演示
        self.demo_normalization()
        
        # 多时间框架演示
        self.demo_multi_timeframe()
        
        # 高级指标演示
        self.demo_advanced_indicators()
        
        # 性能监控演示
        self.demo_performance_monitoring()
        
        # 数据导出演示
        self.demo_data_export()
        
        # 可视化演示
        self.demo_visualization()
        
        print("\n========================================")
        print("演示完成！")
        print("========================================")


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        self.indicators_manager = TechnicalIndicators()
    
    def benchmark_calculation_speed(self):
        """计算速度基准测试"""
        print("=== 计算速度基准测试 ===")
        
        data_sizes = [100, 500, 1000, 5000, 10000]
        results = {}
        
        for size in data_sizes:
            print(f"测试数据大小: {size}")
            
            # 生成测试数据
            sample_data = self._generate_benchmark_data(size)
            symbol = f"BENCHMARK_{size}"
            
            # 测试数据更新速度
            start_time = time.time()
            df = pd.DataFrame(sample_data)
            self.indicators_manager.update_data_batch(symbol, df)
            update_time = time.time() - start_time
            
            # 测试指标计算速度
            start_time = time.time()
            indicators_results = self.indicators_manager.calculate_all_indicators(symbol)
            calc_time = time.time() - start_time
            
            results[size] = {
                'update_time': update_time,
                'calc_time': calc_time,
                'indicators_count': len(indicators_results),
                'throughput': len(indicators_results) / calc_time if calc_time > 0 else 0
            }
            
            print(f"  数据更新: {update_time:.3f}秒")
            print(f"  指标计算: {calc_time:.3f}秒")
            print(f"  计算指标数: {len(indicators_results)}")
            print(f"  吞吐量: {results[size]['throughput']:.1f} 指标/秒")
            print()
        
        return results
    
    def benchmark_memory_usage(self):
        """内存使用基准测试"""
        print("=== 内存使用基准测试 ===")
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"初始内存使用: {initial_memory:.1f} MB")
        
        # 逐步增加数据量测试内存使用
        memory_usage = {}
        
        for i in range(1, 11):
            symbol = f"MEM_TEST_{i}"
            sample_data = self._generate_benchmark_data(1000)
            df = pd.DataFrame(sample_data)
            
            self.indicators_manager.update_data_batch(symbol, df)
            self.indicators_manager.calculate_all_indicators(symbol)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_usage[i] = current_memory - initial_memory
            
            print(f"处理 {i} 个symbol后，内存增长: {memory_usage[i]:.1f} MB")
        
        # 测试缓存清理效果
        print("\n清理缓存...")
        self.indicators_manager.clear_cache()
        gc.collect()
        
        after_clear_memory = process.memory_info().rss / 1024 / 1024
        print(f"清理后内存使用: {after_clear_memory:.1f} MB")
        print(f"内存回收: {current_memory - after_clear_memory:.1f} MB")
        
        return memory_usage
    
    async def benchmark_async_performance(self):
        """异步性能基准测试"""
        print("=== 异步性能基准测试 ===")
        
        symbol_counts = [1, 5, 10, 20, 50]
        
        for count in symbol_counts:
            symbols = [f"ASYNC_TEST_{i}" for i in range(count)]
            
            # 准备数据
            for symbol in symbols:
                sample_data = self._generate_benchmark_data(500)
                df = pd.DataFrame(sample_data)
                self.indicators_manager.update_data_batch(symbol, df)
            
            # 测试异步计算
            start_time = time.time()
            tasks = [self.indicators_manager.calculate_all_indicators_async(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks)
            async_time = time.time() - start_time
            
            # 测试同步计算
            start_time = time.time()
            for symbol in symbols:
                self.indicators_manager.calculate_all_indicators(symbol, force_recalculate=True)
            sync_time = time.time() - start_time
            
            speedup = sync_time / async_time if async_time > 0 else 0
            
            print(f"{count} 个symbol:")
            print(f"  异步时间: {async_time:.3f}秒")
            print(f"  同步时间: {sync_time:.3f}秒")
            print(f"  加速比: {speedup:.2f}x")
            print()
    
    def benchmark_normalization_performance(self):
        """标准化性能基准测试"""
        print("=== 标准化性能基准测试 ===")
        
        normalizer = IndicatorNormalizer()
        data_sizes = [100, 1000, 10000, 100000]
        methods = [
            NormalizationMethod.MIN_MAX,
            NormalizationMethod.Z_SCORE,
            NormalizationMethod.PERCENTILE,
            NormalizationMethod.TANH
        ]
        
        for size in data_sizes:
            print(f"数据大小: {size}")
            test_data = np.random.normal(100, 20, size)
            
            for method in methods:
                start_time = time.time()
                normalized = normalizer.normalize(
                    test_data, 
                    f"test_{method.value}", 
                    method=method
                )
                norm_time = time.time() - start_time
                
                print(f"  {method.value}: {norm_time:.4f}秒")
            print()
    
    def _generate_benchmark_data(self, length: int) -> Dict[str, np.ndarray]:
        """生成基准测试数据"""
        np.random.seed(42)
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, length)
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices[1:])
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price
        
        volatility = 0.01
        high_prices = close_prices * (1 + np.random.uniform(0, volatility, length))
        low_prices = close_prices * (1 - np.random.uniform(0, volatility, length))
        
        for i in range(length):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        volumes = np.random.uniform(100000, 1000000, length)
        timestamps = np.arange(length) * 3600
        
        return {
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }
    
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("========================================")
        print("技术指标性能基准测试")
        print("========================================\n")
        
        # 计算速度测试
        speed_results = self.benchmark_calculation_speed()
        
        # 内存使用测试
        memory_results = self.benchmark_memory_usage()
        
        # 异步性能测试
        print("运行异步性能测试...")
        asyncio.run(self.benchmark_async_performance())
        
        # 标准化性能测试
        self.benchmark_normalization_performance()
        
        print("========================================")
        print("基准测试完成！")
        print("========================================")
        
        return {
            'speed': speed_results,
            'memory': memory_results
        }


def main():
    """主函数"""
    print("技术指标计算引擎使用示例和性能测试")
    print("=====================================\n")
    
    # 运行演示
    demo = TechnicalIndicatorDemo()
    demo.run_complete_demo()
    
    print("\n" + "="*50 + "\n")
    
    # 运行基准测试
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()