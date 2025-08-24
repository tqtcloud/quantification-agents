# 多维度技术指标引擎

## 概述

`MultiDimensionalIndicatorEngine` 是量化交易系统的核心组件，负责整合多种技术指标的并行计算，实现多时间框架一致性检查，并提供综合信号强度评估。

## 核心功能

### 1. 多维度技术指标计算

引擎整合了五个主要维度的技术指标：

#### 动量维度 (Momentum)
- **RSI (相对强弱指数)**: 14期和21期
- **MACD**: 标准(12,26,9)和快速(8,21,5)配置
- **随机指标**: 标准(14,3)和快速(5,3)配置
- **CCI (顾比通道指数)**: 20期
- **威廉指标**: 14期

#### 趋势维度 (Trend)
- **移动平均线**: SMA(10,20,50,200)和EMA(12,26)
- **ADX (平均趋向指数)**: 14期，包含+DI和-DI
- **抛物线SAR**: 动态止损指标
- **一目均衡表**: 完整的Ichimoku系统

#### 波动率维度 (Volatility)
- **布林带**: 标准(20,2)和紧缩(10,1.5)配置
- **ATR (平均真实范围)**: 14期和7期
- **标准差**: 20期价格标准差
- **肯特纳通道**: 20期
- **唐奇安通道**: 20期

#### 成交量维度 (Volume)
- 成交量相对强度分析
- 成交量趋势检测
- 异常成交量识别
- 成交量-价格背离分析

#### 情绪维度 (Sentiment)
- 基于价格行为的情绪分析
- 涨跌频率统计
- 上影线/下影线分析
- 短期vs长期动量对比

### 2. 多时间框架一致性检查

支持以下时间框架的一致性分析：
- 1分钟 (1m)
- 5分钟 (5m)
- 15分钟 (15m)
- 30分钟 (30m)
- 1小时 (1h)
- 4小时 (4h)
- 1日 (1d)

每个时间框架分析包括：
- **趋势方向**: -1到1的数值，表示看跌到看涨程度
- **趋势强度**: 0到1的数值，表示趋势的可靠性
- **波动率水平**: 该时间框架的波动率特征
- **成交量特征**: 成交量分布和变化模式
- **权重系数**: 该时间框架在综合分析中的重要性

### 3. 信号强度综合评估

引擎使用智能权重分配算法，综合评估各维度指标：

#### 权重配置
```python
dimension_weights = {
    'momentum': 0.25,    # 动量维度
    'trend': 0.25,       # 趋势维度
    'volatility': 0.15,  # 波动率维度
    'volume': 0.20,      # 成交量维度
    'sentiment': 0.15    # 情绪维度
}
```

#### 信号分类
- **强烈买入** (STRONG_BUY): 综合评分 > 0.7
- **买入** (BUY): 综合评分 > 0.3
- **弱买入** (WEAK_BUY): 综合评分 > 0.1
- **中性** (NEUTRAL): -0.1 ≤ 综合评分 ≤ 0.1
- **弱卖出** (WEAK_SELL): 综合评分 < -0.1
- **卖出** (SELL): 综合评分 < -0.3
- **强烈卖出** (STRONG_SELL): 综合评分 < -0.7

## 使用方法

### 基本用法

```python
import asyncio
from src.core.engine.multidimensional_engine import MultiDimensionalIndicatorEngine

async def main():
    # 创建引擎实例
    engine = MultiDimensionalIndicatorEngine(max_workers=4)
    
    # 准备市场数据
    market_data = {
        'open': [100.0, 101.0, 102.0, ...],
        'high': [101.5, 102.5, 103.0, ...],
        'low': [99.5, 100.5, 101.5, ...],
        'close': [101.0, 102.0, 102.5, ...],
        'volume': [10000, 12000, 11000, ...]
    }
    
    try:
        # 生成多维度信号
        signal = await engine.generate_multidimensional_signal(
            symbol="BTCUSDT",
            market_data=market_data,
            enable_multiframe_analysis=True
        )
        
        if signal:
            print(f"信号类型: {signal.primary_signal.signal_type.name}")
            print(f"置信度: {signal.overall_confidence:.3f}")
            print(f"风险收益比: {signal.risk_reward_ratio:.2f}")
            print(f"建议仓位: {signal.max_position_size:.1%}")
    
    finally:
        engine.cleanup()

# 运行示例
asyncio.run(main())
```

### 高级配置

```python
# 自定义时间框架
from src.core.indicators.timeframe import TimeFrame

custom_timeframes = [
    TimeFrame.MINUTE_5,
    TimeFrame.MINUTE_15,
    TimeFrame.HOUR_1,
    TimeFrame.HOUR_4
]

signal = await engine.generate_multidimensional_signal(
    symbol="ETHUSDT",
    market_data=market_data,
    timeframes=custom_timeframes,
    enable_multiframe_analysis=True
)
```

### 批量信号生成

```python
async def generate_multiple_signals():
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    tasks = []
    
    for symbol in symbols:
        task = engine.generate_multidimensional_signal(
            symbol=symbol,
            market_data=get_market_data(symbol),
            enable_multiframe_analysis=True
        )
        tasks.append(task)
    
    # 并行生成所有信号
    signals = await asyncio.gather(*tasks, return_exceptions=True)
    
    return signals
```

## 输出结果

### MultiDimensionalSignal 结构

```python
@dataclass
class MultiDimensionalSignal:
    primary_signal: TradingSignal          # 主要交易信号
    momentum_score: float                  # 动量维度评分 [-1, 1]
    mean_reversion_score: float           # 均值回归评分 [-1, 1]
    volatility_score: float               # 波动率评分 [0, 1]
    volume_score: float                   # 成交量评分 [0, 1]
    sentiment_score: float                # 情绪评分 [-1, 1]
    overall_confidence: float             # 综合置信度 [0, 1]
    risk_reward_ratio: float              # 风险收益比
    max_position_size: float              # 建议最大仓位 [0, 1]
    market_regime: str                    # 市场状态
    technical_levels: Dict[str, float]    # 关键技术位
```

### 信号质量评估

```python
# 信号质量综合评分
quality_score = signal.signal_quality_score  # [0, 1]

# 信号方向一致性
direction_consensus = signal.signal_direction_consensus  # [-1, 1]

# 仓位建议
conservative_size = signal.get_position_sizing_recommendation(
    base_position_size=1.0,
    risk_tolerance=0.5  # 保守策略
)

aggressive_size = signal.get_position_sizing_recommendation(
    base_position_size=1.0,
    risk_tolerance=1.0  # 积极策略
)
```

## 性能特性

### 并行处理
- 使用 `ThreadPoolExecutor` 实现指标并行计算
- 支持异步信号生成，提高系统吞吐量
- 可配置工作线程数量，优化资源使用

### 性能指标
- **平均延迟**: < 2秒/信号 (1000个数据点)
- **并发吞吐量**: > 5信号/秒 (多线程)
- **内存使用**: < 100MB (大数据集)
- **错误处理开销**: < 100% (相对正常处理)

### 缓存机制
- 指标计算结果缓存
- 5分钟TTL缓存策略
- 内存高效的数据结构

## 配置选项

### 引擎配置

```python
engine = MultiDimensionalIndicatorEngine(
    max_workers=8,              # 最大工作线程数
)
```

### 时间框架权重配置

```python
timeframe_weights = {
    TimeFrame.MINUTE_1: 0.1,    # 短期噪音，较低权重
    TimeFrame.MINUTE_5: 0.15,   # 短期趋势
    TimeFrame.MINUTE_15: 0.2,   # 中短期趋势
    TimeFrame.MINUTE_30: 0.2,   # 中期趋势
    TimeFrame.HOUR_1: 0.15,     # 中长期趋势
    TimeFrame.HOUR_4: 0.15,     # 长期趋势
    TimeFrame.DAY_1: 0.05       # 超长期，较低权重
}
```

## 市场状态识别

引擎能够识别以下市场状态：

- **上升趋势** (TRENDING_UP): 明确的上涨趋势
- **下降趋势** (TRENDING_DOWN): 明确的下跌趋势
- **横盘整理** (SIDEWAYS): 无明显方向性
- **高波动率** (HIGH_VOLATILITY): 剧烈价格波动
- **低波动率** (LOW_VOLATILITY): 价格稳定
- **盘整状态** (CONSOLIDATION): 价格区间整理

## 风险管理集成

### 动态仓位管理

```python
# 基于信号质量调整仓位
position_adjustment = signal.signal_quality_score

# 基于波动率调整仓位
volatility_adjustment = 1.0 - (signal.volatility_score * 0.5)

# 基于风险收益比调整仓位
rr_adjustment = min(signal.risk_reward_ratio / 2.0, 1.0)

# 综合建议仓位
recommended_position = (
    base_position * 
    position_adjustment * 
    volatility_adjustment * 
    rr_adjustment
)
```

### 止损止盈设置

```python
# 基于ATR的动态止损
atr_multiplier = 1.0  # 保守
# atr_multiplier = 1.5  # 中性
# atr_multiplier = 2.0  # 激进

stop_loss = entry_price - (atr * atr_multiplier)  # 买入
stop_loss = entry_price + (atr * atr_multiplier)  # 卖出

# 基于风险收益比的目标价格
target_price = entry_price + (risk_amount * target_rr_ratio)
```

## 最佳实践

### 1. 数据质量要求
- 至少需要200个数据点进行可靠分析
- 确保OHLCV数据完整性和一致性
- 处理异常值和缺失数据

### 2. 性能优化
- 合理配置工作线程数(推荐4-8个)
- 启用多时间框架分析时考虑性能影响
- 定期清理引擎资源

### 3. 信号过滤
- 设置最低置信度阈值(推荐0.6)
- 过滤低质量信号(quality_score < 0.5)
- 考虑市场状态和波动率环境

### 4. 风险控制
- 根据信号强度动态调整仓位
- 严格执行止损策略
- 分散化投资，避免单一信号依赖

## 故障排除

### 常见问题

1. **内存使用过高**
   - 减少工作线程数量
   - 限制数据集大小
   - 定期调用garbage collection

2. **性能问题**
   - 关闭不必要的多时间框架分析
   - 减少并发信号生成数量
   - 检查数据质量和大小

3. **信号质量不佳**
   - 增加历史数据长度
   - 检查市场数据完整性
   - 调整信号过滤阈值

### 调试工具

```python
# 获取引擎性能统计
stats = engine.get_performance_stats()
print(f"生成信号数: {stats['signals_generated']}")
print(f"平均处理时间: {stats['avg_processing_time']:.3f}s")
print(f"错误数量: {stats['error_count']}")

# 重置统计数据
engine.reset_stats()
```

## 扩展开发

### 添加新的技术指标

```python
# 1. 实现新指标类
class CustomIndicator(AsyncTechnicalIndicator):
    def calculate(self, data, symbol):
        # 实现计算逻辑
        pass

# 2. 添加到相应维度组
engine.momentum_indicators['custom'] = CustomIndicator()

# 3. 更新评分计算逻辑
async def _calculate_custom_dimension(self, symbol, market_data):
    # 实现自定义维度计算
    pass
```

### 自定义市场状态识别

```python
def _determine_market_regime(self, dimension_scores, timeframe_consensus):
    # 实现自定义市场状态逻辑
    if custom_condition:
        return MarketRegime.CUSTOM_STATE
    else:
        return super()._determine_market_regime(dimension_scores, timeframe_consensus)
```

## 版本历史

- **v1.0**: 初始版本，基本多维度信号生成
- **v1.1**: 添加多时间框架一致性检查
- **v1.2**: 优化性能和并行处理
- **v2.0**: 完整重构，增强信号质量评估

## 相关文档

- [技术指标计算引擎](./indicators.md)
- [信号数据模型](./signals.md) 
- [时间框架管理](./timeframe.md)
- [性能测试指南](./performance_testing.md)