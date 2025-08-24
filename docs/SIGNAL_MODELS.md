# 信号模型使用指南

本文档介绍量化交易系统中新增的信号数据模型的使用方法。

## 概述

新的信号模型提供了强类型的数据结构来表示和处理交易信号，包括：

1. **SignalStrength** - 信号强度枚举
2. **TradingSignal** - 基础交易信号数据类
3. **MultiDimensionalSignal** - 多维度信号数据类
4. **SignalAggregator** - 信号聚合器

## 核心组件

### 1. SignalStrength 枚举

定义了7个信号强度等级：

```python
from src.core.models.signals import SignalStrength

# 可用的信号强度
SignalStrength.STRONG_SELL   # -1
SignalStrength.SELL          #  0  
SignalStrength.WEAK_SELL     #  1
SignalStrength.NEUTRAL       #  2
SignalStrength.WEAK_BUY      #  3
SignalStrength.BUY           #  4
SignalStrength.STRONG_BUY    #  5
```

### 2. TradingSignal 基础信号

包含交易信号的核心信息：

```python
from src.core.models.signals import TradingSignal, SignalStrength

signal = TradingSignal(
    symbol="BTCUSDT",                    # 交易标的
    signal_type=SignalStrength.BUY,      # 信号类型
    confidence=0.85,                     # 置信度 [0-1]
    entry_price=50000.0,                 # 入场价格
    target_price=55000.0,                # 目标价格
    stop_loss=48000.0,                   # 止损价格
    reasoning=["RSI反弹", "MACD金叉"],    # 推理逻辑
    indicators_consensus={               # 技术指标共识
        "RSI": 0.7,
        "MACD": 0.8
    }
)

# 自动计算的属性
print(f"风险收益比: {signal.risk_reward_ratio}")
print(f"信号有效性: {signal.is_valid}")
```

### 3. MultiDimensionalSignal 多维度信号

扩展基础信号，提供多维度市场分析：

```python
from src.core.models.signals import MultiDimensionalSignal

multi_signal = MultiDimensionalSignal(
    primary_signal=signal,           # 主要交易信号
    momentum_score=0.8,              # 动量维度 [-1, 1]
    mean_reversion_score=-0.2,       # 均值回归维度 [-1, 1]
    volatility_score=0.3,            # 波动率维度 [0, 1]
    volume_score=0.9,                # 成交量维度 [0, 1]
    sentiment_score=0.7,             # 市场情绪维度 [-1, 1]
    overall_confidence=0.88,         # 综合置信度 [0, 1]
    risk_reward_ratio=2.5,           # 风险收益比
    max_position_size=0.4            # 建议最大仓位 [0, 1]
)

# 计算衍生指标
quality = multi_signal.signal_quality_score          # 信号质量评分
consensus = multi_signal.signal_direction_consensus  # 方向一致性

# 仓位建议
recommended_size = multi_signal.get_position_sizing_recommendation(
    base_position_size=1.0,
    risk_tolerance=0.8
)
```

### 4. SignalAggregator 信号聚合

处理多个信号的聚合和过滤：

```python
from src.core.models.signals import SignalAggregator

# 过滤低质量信号
high_quality_signals = SignalAggregator.filter_signals_by_quality(
    signals,
    min_quality_score=0.6,
    min_confidence=0.7
)

# 组合多个信号
combined_signal = SignalAggregator.combine_signals(
    signals,
    weights={"signal_0": 0.3, "signal_1": 0.5, "signal_2": 0.2}
)
```

## 数据验证

所有模型都包含自动数据验证：

### 价格逻辑验证
- 买入信号：目标价格 > 入场价格 > 止损价格
- 卖出信号：止损价格 > 入场价格 > 目标价格

### 范围验证
- 置信度：[0, 1]
- 动量/均值回归/情绪评分：[-1, 1]
- 波动率/成交量评分：[0, 1]
- 最大仓位：[0, 1]

### 错误处理示例

```python
try:
    signal = TradingSignal(
        symbol="BTCUSDT",
        signal_type=SignalStrength.BUY,
        confidence=1.5,  # 超出范围！
        entry_price=50000.0,
        target_price=45000.0,  # 逻辑错误！
        stop_loss=48000.0,
        reasoning=["测试"],
        indicators_consensus={}
    )
except ValueError as e:
    print(f"验证错误: {e}")
```

## 使用模式

### 1. 信号生成Agent

```python
class TechnicalAnalysisAgent:
    def generate_signal(self, symbol: str, market_data) -> MultiDimensionalSignal:
        # 技术分析逻辑...
        
        primary_signal = TradingSignal(
            symbol=symbol,
            signal_type=self._determine_signal_type(analysis),
            confidence=analysis.confidence,
            entry_price=market_data.price,
            target_price=self._calculate_target(analysis),
            stop_loss=self._calculate_stop_loss(analysis),
            reasoning=analysis.reasons,
            indicators_consensus=analysis.indicator_scores
        )
        
        return MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=self._calculate_momentum(market_data),
            mean_reversion_score=self._calculate_mean_reversion(market_data),
            volatility_score=self._calculate_volatility(market_data),
            volume_score=self._calculate_volume_score(market_data),
            sentiment_score=self._calculate_sentiment(market_data),
            overall_confidence=self._calculate_overall_confidence(),
            risk_reward_ratio=primary_signal.risk_reward_ratio,
            max_position_size=self._calculate_max_position(analysis)
        )
```

### 2. 信号决策引擎

```python
class SignalDecisionEngine:
    def process_signals(self, signals: List[MultiDimensionalSignal]) -> Optional[MultiDimensionalSignal]:
        # 过滤低质量信号
        quality_signals = SignalAggregator.filter_signals_by_quality(
            signals,
            min_quality_score=self.min_quality_threshold,
            min_confidence=self.min_confidence_threshold
        )
        
        if not quality_signals:
            return None
        
        # 按信号源设置权重
        weights = self._calculate_signal_weights(quality_signals)
        
        # 组合信号
        return SignalAggregator.combine_signals(quality_signals, weights)
```

### 3. 风险管理集成

```python
class RiskManager:
    def validate_signal(self, signal: MultiDimensionalSignal) -> bool:
        # 检查信号质量
        if signal.signal_quality_score < self.min_quality:
            return False
        
        # 检查风险收益比
        if signal.risk_reward_ratio < self.min_risk_reward:
            return False
        
        # 检查仓位限制
        if signal.max_position_size > self.max_single_position:
            return False
        
        return True
    
    def adjust_position_size(self, signal: MultiDimensionalSignal) -> float:
        return signal.get_position_sizing_recommendation(
            base_position_size=self.base_position,
            risk_tolerance=self.risk_tolerance
        )
```

## 最佳实践

### 1. 信号创建
- 始终提供详细的推理逻辑
- 确保技术指标共识字典的完整性
- 合理设置置信度，避免过度自信

### 2. 多维度评分
- 动量评分：基于趋势强度，正值表示上涨动量
- 均值回归评分：基于价格偏离程度，正值表示回归买入机会
- 波动率评分：基于历史波动率，高值表示高风险
- 成交量评分：基于成交量确认，高值表示强确认
- 情绪评分：基于市场情绪指标，正值表示乐观情绪

### 3. 信号聚合
- 使用权重区分不同信号源的重要性
- 定期调整质量过滤阈值
- 监控组合信号的表现并优化参数

### 4. 错误处理
- 始终使用try-catch捕获验证错误
- 记录无效信号用于模型调优
- 实现降级策略处理信号生成失败

## 示例代码

完整的使用示例请参考：`examples/signal_model_example.py`

该文件包含了所有核心功能的详细演示，包括：
- 基础信号创建
- 多维度信号分析
- 信号聚合处理
- 错误处理机制

## 注意事项

1. **性能考虑**：多维度信号计算相对复杂，在高频场景下需要考虑性能优化
2. **参数调优**：各维度评分的权重需要根据实际交易表现进行调优  
3. **数据依赖**：确保输入的市场数据质量，避免基于错误数据生成信号
4. **版本兼容**：新版本可能会调整数据结构，注意向后兼容性

## 集成指南

要在现有系统中使用新的信号模型：

1. 导入所需的类：
```python
from src.core.models.signals import (
    SignalStrength,
    TradingSignal, 
    MultiDimensionalSignal,
    SignalAggregator
)
```

2. 更新Agent接口以返回新的信号类型
3. 修改决策引擎以处理多维度信号
4. 更新风险管理模块以利用新的信号属性
5. 调整回测系统以支持新的信号格式

通过这些改进，系统现在具备了更强大和更灵活的信号处理能力！