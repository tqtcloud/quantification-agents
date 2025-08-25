"""
策略信号聚合器 (SignalAggregator)

实现不同策略信号的融合逻辑，集成HFT和AI策略的交易信号，
提供统一的交易信号输出接口。
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from collections import defaultdict, deque
import numpy as np
import statistics

from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength
from src.strategy.conflict_resolver import ConflictResolver, ConflictDetail, ConflictResolution
from src.strategy.priority_manager import PriorityManager, MarketCondition
from src.utils.logger import LoggerMixin


class AggregationStrategy(Enum):
    """信号聚合策略"""
    WEIGHTED_AVERAGE = "weighted_average"           # 加权平均
    PRIORITY_SELECTION = "priority_selection"       # 优先级选择
    CONSENSUS_VOTING = "consensus_voting"           # 共识投票
    CONFIDENCE_THRESHOLD = "confidence_threshold"   # 置信度阈值
    QUALITY_FILTERING = "quality_filtering"         # 质量过滤
    HYBRID_FUSION = "hybrid_fusion"                # 混合融合
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"           # 动态自适应


class SignalSource(Enum):
    """信号源类型"""
    HFT_ENGINE = "hft_engine"           # HFT引擎
    AI_AGENT = "ai_agent"               # AI Agent
    TECHNICAL_ANALYSIS = "technical"     # 技术分析
    FUNDAMENTAL_ANALYSIS = "fundamental" # 基本面分析
    SENTIMENT_ANALYSIS = "sentiment"     # 情绪分析
    EXTERNAL_PROVIDER = "external"       # 外部提供商


@dataclass
class SignalInput:
    """信号输入"""
    signal_id: str
    signal: MultiDimensionalSignal
    source_type: SignalSource
    source_id: str
    priority: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.now)


@dataclass
class AggregationResult:
    """聚合结果"""
    aggregation_id: str
    aggregated_signal: Optional[MultiDimensionalSignal]
    input_signals: List[SignalInput]
    conflicts_detected: List[ConflictDetail]
    conflicts_resolved: List[ConflictResolution]
    strategy_used: AggregationStrategy
    confidence_adjustment: float
    quality_score: float
    reasoning: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AggregationConfig:
    """聚合配置"""
    strategy: AggregationStrategy = AggregationStrategy.HYBRID_FUSION
    min_signal_count: int = 2
    max_signal_count: int = 10
    min_confidence_threshold: float = 0.6
    min_quality_threshold: float = 0.5
    consensus_threshold: float = 0.7
    conflict_resolution_enabled: bool = True
    priority_weighting_enabled: bool = True
    time_window_seconds: int = 300  # 5分钟
    enable_quality_boost: bool = True
    enable_consistency_check: bool = True


@dataclass
class AggregationStatistics:
    """聚合统计"""
    total_aggregations: int = 0
    successful_aggregations: int = 0
    failed_aggregations: int = 0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    signal_sources_stats: Dict[str, int] = field(default_factory=dict)
    strategy_usage_stats: Dict[str, int] = field(default_factory=dict)


class UnifiedSignalInterface:
    """统一信号输出接口"""
    
    def __init__(self):
        self.signal_callbacks: List[Callable[[AggregationResult], None]] = []
        self.error_callbacks: List[Callable[[Exception, Dict[str, Any]], None]] = []
    
    def register_signal_callback(self, callback: Callable[[AggregationResult], None]) -> None:
        """注册信号回调"""
        self.signal_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception, Dict[str, Any]], None]) -> None:
        """注册错误回调"""
        self.error_callbacks.append(callback)
    
    async def emit_signal(self, result: AggregationResult) -> None:
        """发送聚合信号"""
        for callback in self.signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                await self.emit_error(e, {'result': result})
    
    async def emit_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """发送错误"""
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, context)
                else:
                    callback(error, context)
            except Exception as e:
                logging.error(f"错误回调失败: {e}")


class SignalAggregator(LoggerMixin):
    """
    策略信号聚合器
    
    功能：
    1. 接收来自不同策略的信号
    2. 检测和解决信号冲突
    3. 应用优先级加权
    4. 执行信号融合算法
    5. 输出统一的交易信号
    6. 提供完整的审计轨迹
    """
    
    def __init__(self, 
                 config: Optional[AggregationConfig] = None,
                 conflict_resolver: Optional[ConflictResolver] = None,
                 priority_manager: Optional[PriorityManager] = None):
        """
        初始化信号聚合器
        
        Args:
            config: 聚合配置
            conflict_resolver: 冲突解决器
            priority_manager: 优先级管理器
        """
        super().__init__()
        
        self.config = config or AggregationConfig()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.priority_manager = priority_manager or PriorityManager()
        
        # 信号接口
        self.unified_interface = UnifiedSignalInterface()
        
        # 信号缓存和历史
        self.signal_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.aggregation_history: deque = deque(maxlen=1000)
        
        # 统计信息
        self.statistics = AggregationStatistics()
        
        # 融合权重配置
        self.fusion_weights = {
            'signal_quality': 0.3,
            'confidence_score': 0.25,
            'priority_score': 0.2,
            'consistency_score': 0.15,
            'recency_factor': 0.1
        }
        
        # 策略特定配置
        self.strategy_configs = {
            AggregationStrategy.WEIGHTED_AVERAGE: {
                'use_priority_weights': True,
                'normalize_weights': True,
                'confidence_boost': True
            },
            AggregationStrategy.CONSENSUS_VOTING: {
                'min_consensus_ratio': 0.6,
                'direction_weight': 0.4,
                'strength_weight': 0.6
            },
            AggregationStrategy.QUALITY_FILTERING: {
                'min_quality_score': 0.7,
                'quality_decay_factor': 0.9
            }
        }
        
        # 运行状态
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        
        self.log_info("SignalAggregator初始化完成")
    
    async def start(self) -> None:
        """启动信号聚合器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动处理任务
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        self.log_info("SignalAggregator已启动")
    
    async def stop(self) -> None:
        """停止信号聚合器"""
        self._running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.log_info("SignalAggregator已停止")
    
    async def aggregate_signals(self, 
                               signal_inputs: List[SignalInput],
                               strategy: Optional[AggregationStrategy] = None) -> AggregationResult:
        """
        聚合信号
        
        Args:
            signal_inputs: 输入信号列表
            strategy: 聚合策略
            
        Returns:
            聚合结果
        """
        start_time = time.perf_counter()
        aggregation_id = f"agg_{uuid.uuid4().hex[:8]}"
        
        try:
            self.statistics.total_aggregations += 1
            
            # 验证输入
            if not self._validate_inputs(signal_inputs):
                raise ValueError("输入信号验证失败")
            
            # 选择聚合策略
            selected_strategy = strategy or self.config.strategy
            
            # 预处理信号
            processed_signals = await self._preprocess_signals(signal_inputs)
            
            # 检测冲突
            conflicts_detected = []
            conflicts_resolved = []
            
            if self.config.conflict_resolution_enabled:
                signals = [si.signal for si in processed_signals]
                priorities = {str(id(si.signal)): si.priority for si in processed_signals}
                
                conflicts_detected = self.conflict_resolver.detect_conflicts(signals, priorities)
                
                if conflicts_detected:
                    conflicts_resolved = self.conflict_resolver.resolve_conflicts(
                        signals, conflicts_detected, priorities
                    )
            
            # 执行聚合
            aggregated_signal = await self._execute_aggregation(
                processed_signals, selected_strategy, conflicts_resolved
            )
            
            # 计算调整和质量评分
            confidence_adjustment = self._calculate_confidence_adjustment(conflicts_detected, conflicts_resolved)
            quality_score = self._calculate_aggregation_quality(processed_signals, aggregated_signal)
            
            # 生成推理
            reasoning = self._generate_aggregation_reasoning(
                processed_signals, selected_strategy, conflicts_detected, conflicts_resolved
            )
            
            # 计算处理时间
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # 创建结果
            result = AggregationResult(
                aggregation_id=aggregation_id,
                aggregated_signal=aggregated_signal,
                input_signals=signal_inputs,
                conflicts_detected=conflicts_detected,
                conflicts_resolved=conflicts_resolved,
                strategy_used=selected_strategy,
                confidence_adjustment=confidence_adjustment,
                quality_score=quality_score,
                reasoning=reasoning,
                processing_time_ms=processing_time,
                metadata={
                    'input_count': len(signal_inputs),
                    'conflicts_count': len(conflicts_detected),
                    'resolutions_count': len(conflicts_resolved)
                }
            )
            
            # 更新统计
            self._update_statistics(result)
            
            # 保存历史
            self.aggregation_history.append(result)
            
            # 发送信号
            await self.unified_interface.emit_signal(result)
            
            self.statistics.successful_aggregations += 1
            self.log_info(f"信号聚合完成: {aggregation_id}, 耗时: {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.statistics.failed_aggregations += 1
            
            # 创建失败结果
            processing_time = (time.perf_counter() - start_time) * 1000
            
            result = AggregationResult(
                aggregation_id=aggregation_id,
                aggregated_signal=None,
                input_signals=signal_inputs,
                conflicts_detected=[],
                conflicts_resolved=[],
                strategy_used=strategy or self.config.strategy,
                confidence_adjustment=0.0,
                quality_score=0.0,
                reasoning=[f"聚合失败: {str(e)}"],
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
            
            await self.unified_interface.emit_error(e, {'result': result})
            
            self.log_error(f"信号聚合失败: {e}")
            raise
    
    def _validate_inputs(self, signal_inputs: List[SignalInput]) -> bool:
        """验证输入信号"""
        try:
            if not signal_inputs:
                return False
            
            if len(signal_inputs) < self.config.min_signal_count:
                return False
            
            if len(signal_inputs) > self.config.max_signal_count:
                return False
            
            # 检查信号质量
            for signal_input in signal_inputs:
                signal = signal_input.signal
                if signal.overall_confidence < self.config.min_confidence_threshold:
                    return False
                
                if signal.signal_quality_score < self.config.min_quality_threshold:
                    return False
            
            # 检查时间窗口
            current_time = datetime.now()
            time_window = timedelta(seconds=self.config.time_window_seconds)
            
            for signal_input in signal_inputs:
                signal_time = signal_input.signal.primary_signal.timestamp
                if current_time - signal_time > time_window:
                    return False
            
            return True
            
        except Exception as e:
            self.log_error(f"输入验证失败: {e}")
            return False
    
    async def _preprocess_signals(self, signal_inputs: List[SignalInput]) -> List[SignalInput]:
        """预处理信号"""
        processed_signals = []
        
        try:
            for signal_input in signal_inputs:
                # 更新优先级
                if self.config.priority_weighting_enabled:
                    priority = self.priority_manager.get_signal_priority(
                        signal_input.signal, signal_input.source_id
                    )
                    signal_input.priority = priority
                
                # 计算权重
                weight = self._calculate_signal_weight(signal_input)
                signal_input.weight = weight
                
                processed_signals.append(signal_input)
            
            # 按优先级排序
            processed_signals.sort(key=lambda si: si.priority, reverse=True)
            
            return processed_signals
            
        except Exception as e:
            self.log_error(f"信号预处理失败: {e}")
            return signal_inputs
    
    def _calculate_signal_weight(self, signal_input: SignalInput) -> float:
        """计算信号权重"""
        try:
            signal = signal_input.signal
            
            # 基础权重组件
            quality_weight = signal.signal_quality_score * self.fusion_weights['signal_quality']
            confidence_weight = signal.overall_confidence * self.fusion_weights['confidence_score']
            priority_weight = signal_input.priority * self.fusion_weights['priority_score']
            
            # 一致性权重
            consistency_score = abs(signal.signal_direction_consensus)
            consistency_weight = consistency_score * self.fusion_weights['consistency_score']
            
            # 时效性权重
            time_diff = (datetime.now() - signal_input.received_at).total_seconds()
            recency_factor = max(0, 1.0 - (time_diff / self.config.time_window_seconds))
            recency_weight = recency_factor * self.fusion_weights['recency_factor']
            
            # 计算总权重
            total_weight = (
                quality_weight + confidence_weight + priority_weight + 
                consistency_weight + recency_weight
            )
            
            return max(0.1, min(2.0, total_weight))
            
        except Exception as e:
            self.log_error(f"计算信号权重失败: {e}")
            return 1.0
    
    async def _execute_aggregation(self, 
                                  signal_inputs: List[SignalInput],
                                  strategy: AggregationStrategy,
                                  conflicts_resolved: List[ConflictResolution]) -> Optional[MultiDimensionalSignal]:
        """执行信号聚合"""
        try:
            if strategy == AggregationStrategy.WEIGHTED_AVERAGE:
                return await self._weighted_average_aggregation(signal_inputs)
            elif strategy == AggregationStrategy.PRIORITY_SELECTION:
                return await self._priority_selection_aggregation(signal_inputs)
            elif strategy == AggregationStrategy.CONSENSUS_VOTING:
                return await self._consensus_voting_aggregation(signal_inputs)
            elif strategy == AggregationStrategy.CONFIDENCE_THRESHOLD:
                return await self._confidence_threshold_aggregation(signal_inputs)
            elif strategy == AggregationStrategy.QUALITY_FILTERING:
                return await self._quality_filtering_aggregation(signal_inputs)
            elif strategy == AggregationStrategy.HYBRID_FUSION:
                return await self._hybrid_fusion_aggregation(signal_inputs, conflicts_resolved)
            elif strategy == AggregationStrategy.DYNAMIC_ADAPTIVE:
                return await self._dynamic_adaptive_aggregation(signal_inputs)
            else:
                return await self._weighted_average_aggregation(signal_inputs)
                
        except Exception as e:
            self.log_error(f"执行聚合失败: {e}")
            return None
    
    async def _weighted_average_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """加权平均聚合"""
        try:
            if not signal_inputs:
                return None
            
            # 计算权重
            total_weight = sum(si.weight for si in signal_inputs)
            if total_weight <= 0:
                return None
            
            # 标准化权重
            weights = [si.weight / total_weight for si in signal_inputs]
            
            # 加权平均各维度
            momentum_scores = [si.signal.momentum_score for si in signal_inputs]
            mean_reversion_scores = [si.signal.mean_reversion_score for si in signal_inputs]
            volatility_scores = [si.signal.volatility_score for si in signal_inputs]
            volume_scores = [si.signal.volume_score for si in signal_inputs]
            sentiment_scores = [si.signal.sentiment_score for si in signal_inputs]
            confidence_scores = [si.signal.overall_confidence for si in signal_inputs]
            
            # 计算加权平均
            weighted_momentum = np.average(momentum_scores, weights=weights)
            weighted_mean_reversion = np.average(mean_reversion_scores, weights=weights)
            weighted_volatility = np.average(volatility_scores, weights=weights)
            weighted_volume = np.average(volume_scores, weights=weights)
            weighted_sentiment = np.average(sentiment_scores, weights=weights)
            weighted_confidence = np.average(confidence_scores, weights=weights)
            
            # 选择主信号（最高权重的信号）
            best_signal_input = max(signal_inputs, key=lambda si: si.weight)
            primary_signal = best_signal_input.signal.primary_signal
            
            # 计算风险收益比
            risk_reward_ratios = [si.signal.risk_reward_ratio for si in signal_inputs]
            avg_risk_reward = np.average(risk_reward_ratios, weights=weights)
            
            # 计算最大仓位（取最小值，更保守）
            max_position_sizes = [si.signal.max_position_size for si in signal_inputs]
            min_position_size = min(max_position_sizes)
            
            # 创建聚合信号
            aggregated_signal = MultiDimensionalSignal(
                primary_signal=primary_signal,
                momentum_score=float(weighted_momentum),
                mean_reversion_score=float(weighted_mean_reversion),
                volatility_score=float(weighted_volatility),
                volume_score=float(weighted_volume),
                sentiment_score=float(weighted_sentiment),
                overall_confidence=float(weighted_confidence),
                risk_reward_ratio=float(avg_risk_reward),
                max_position_size=float(min_position_size)
            )
            
            return aggregated_signal
            
        except Exception as e:
            self.log_error(f"加权平均聚合失败: {e}")
            return None
    
    async def _priority_selection_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """优先级选择聚合"""
        try:
            if not signal_inputs:
                return None
            
            # 选择优先级最高的信号
            best_signal_input = max(signal_inputs, key=lambda si: si.priority)
            
            # 但是调整置信度和质量分数
            other_signals = [si for si in signal_inputs if si != best_signal_input]
            if other_signals:
                # 计算支持度
                support_confidence = np.mean([si.signal.overall_confidence for si in other_signals])
                support_quality = np.mean([si.signal.signal_quality_score for si in other_signals])
                
                # 调整置信度
                adjustment_factor = 1.0 + (support_confidence - 0.5) * 0.2
                adjustment_factor = max(0.8, min(1.2, adjustment_factor))
                
                adjusted_confidence = best_signal_input.signal.overall_confidence * adjustment_factor
                adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
                
                # 创建调整后的信号
                adjusted_signal = MultiDimensionalSignal(
                    primary_signal=best_signal_input.signal.primary_signal,
                    momentum_score=best_signal_input.signal.momentum_score,
                    mean_reversion_score=best_signal_input.signal.mean_reversion_score,
                    volatility_score=best_signal_input.signal.volatility_score,
                    volume_score=best_signal_input.signal.volume_score,
                    sentiment_score=best_signal_input.signal.sentiment_score,
                    overall_confidence=adjusted_confidence,
                    risk_reward_ratio=best_signal_input.signal.risk_reward_ratio,
                    max_position_size=best_signal_input.signal.max_position_size
                )
                
                return adjusted_signal
            
            return best_signal_input.signal
            
        except Exception as e:
            self.log_error(f"优先级选择聚合失败: {e}")
            return None
    
    async def _consensus_voting_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """共识投票聚合"""
        try:
            if not signal_inputs:
                return None
            
            # 按方向分组
            buy_signals = []
            sell_signals = []
            neutral_signals = []
            
            for si in signal_inputs:
                if si.signal.primary_signal._is_buy_signal():
                    buy_signals.append(si)
                elif si.signal.primary_signal._is_sell_signal():
                    sell_signals.append(si)
                else:
                    neutral_signals.append(si)
            
            # 计算加权投票
            buy_weight = sum(si.weight for si in buy_signals)
            sell_weight = sum(si.weight for si in sell_signals)
            neutral_weight = sum(si.weight for si in neutral_signals)
            
            total_weight = buy_weight + sell_weight + neutral_weight
            if total_weight <= 0:
                return None
            
            # 确定共识方向
            buy_ratio = buy_weight / total_weight
            sell_ratio = sell_weight / total_weight
            neutral_ratio = neutral_weight / total_weight
            
            # 检查是否达到共识阈值
            if buy_ratio >= self.config.consensus_threshold:
                consensus_signals = buy_signals
            elif sell_ratio >= self.config.consensus_threshold:
                consensus_signals = sell_signals
            elif neutral_ratio >= self.config.consensus_threshold:
                consensus_signals = neutral_signals
            else:
                # 无共识，选择权重最大的组
                if buy_weight >= sell_weight and buy_weight >= neutral_weight:
                    consensus_signals = buy_signals
                elif sell_weight >= neutral_weight:
                    consensus_signals = sell_signals
                else:
                    consensus_signals = neutral_signals
            
            # 在共识组内进行加权平均
            if consensus_signals:
                return await self._weighted_average_aggregation(consensus_signals)
            
            return None
            
        except Exception as e:
            self.log_error(f"共识投票聚合失败: {e}")
            return None
    
    async def _confidence_threshold_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """置信度阈值聚合"""
        try:
            # 过滤低置信度信号
            high_confidence_signals = [
                si for si in signal_inputs 
                if si.signal.overall_confidence >= self.config.min_confidence_threshold
            ]
            
            if not high_confidence_signals:
                return None
            
            # 对高置信度信号进行加权平均
            return await self._weighted_average_aggregation(high_confidence_signals)
            
        except Exception as e:
            self.log_error(f"置信度阈值聚合失败: {e}")
            return None
    
    async def _quality_filtering_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """质量过滤聚合"""
        try:
            # 过滤低质量信号
            high_quality_signals = [
                si for si in signal_inputs 
                if si.signal.signal_quality_score >= self.config.min_quality_threshold
            ]
            
            if not high_quality_signals:
                return None
            
            # 按质量分数重新分配权重
            for si in high_quality_signals:
                quality_bonus = si.signal.signal_quality_score - self.config.min_quality_threshold
                si.weight *= (1.0 + quality_bonus)
            
            # 对高质量信号进行加权平均
            return await self._weighted_average_aggregation(high_quality_signals)
            
        except Exception as e:
            self.log_error(f"质量过滤聚合失败: {e}")
            return None
    
    async def _hybrid_fusion_aggregation(self, 
                                       signal_inputs: List[SignalInput],
                                       conflicts_resolved: List[ConflictResolution]) -> Optional[MultiDimensionalSignal]:
        """混合融合聚合"""
        try:
            # 步骤1：质量过滤
            quality_filtered = [
                si for si in signal_inputs 
                if si.signal.signal_quality_score >= self.config.min_quality_threshold
            ]
            
            if not quality_filtered:
                return None
            
            # 步骤2：置信度检查
            confidence_filtered = [
                si for si in quality_filtered 
                if si.signal.overall_confidence >= self.config.min_confidence_threshold
            ]
            
            if not confidence_filtered:
                confidence_filtered = quality_filtered
            
            # 步骤3：共识检查
            consensus_result = await self._consensus_voting_aggregation(confidence_filtered)
            if consensus_result and consensus_result.overall_confidence >= 0.7:
                return consensus_result
            
            # 步骤4：优先级选择（如果没有强共识）
            if len(confidence_filtered) >= 2:
                weighted_result = await self._weighted_average_aggregation(confidence_filtered)
                priority_result = await self._priority_selection_aggregation(confidence_filtered)
                
                # 选择质量更高的结果
                if (weighted_result and priority_result and
                    weighted_result.signal_quality_score > priority_result.signal_quality_score):
                    return weighted_result
                else:
                    return priority_result
            
            # 步骤5：单信号处理
            if confidence_filtered:
                return confidence_filtered[0].signal
            
            return None
            
        except Exception as e:
            self.log_error(f"混合融合聚合失败: {e}")
            return None
    
    async def _dynamic_adaptive_aggregation(self, signal_inputs: List[SignalInput]) -> Optional[MultiDimensionalSignal]:
        """动态自适应聚合"""
        try:
            # 根据当前市场条件选择最佳策略
            if hasattr(self.priority_manager, 'current_market_condition') and self.priority_manager.current_market_condition:
                market_condition = self.priority_manager.current_market_condition
                
                # 高波动率时使用优先级选择
                if market_condition.volatility_level == "high":
                    return await self._priority_selection_aggregation(signal_inputs)
                # 低波动率时使用加权平均
                elif market_condition.volatility_level == "low":
                    return await self._weighted_average_aggregation(signal_inputs)
                # 中等波动率时使用共识投票
                else:
                    return await self._consensus_voting_aggregation(signal_inputs)
            
            # 默认使用混合融合
            return await self._hybrid_fusion_aggregation(signal_inputs, [])
            
        except Exception as e:
            self.log_error(f"动态自适应聚合失败: {e}")
            return None
    
    def _calculate_confidence_adjustment(self, 
                                       conflicts_detected: List[ConflictDetail],
                                       conflicts_resolved: List[ConflictResolution]) -> float:
        """计算置信度调整"""
        try:
            if not conflicts_detected:
                return 0.0
            
            # 基于冲突严重性的调整
            total_impact = sum(conflict.confidence_impact for conflict in conflicts_detected)
            
            # 基于解决情况的补偿
            resolution_bonus = len(conflicts_resolved) * 0.05
            
            # 最终调整
            adjustment = total_impact + resolution_bonus
            return max(-0.5, min(0.2, adjustment))
            
        except Exception as e:
            self.log_error(f"计算置信度调整失败: {e}")
            return 0.0
    
    def _calculate_aggregation_quality(self, 
                                     signal_inputs: List[SignalInput],
                                     aggregated_signal: Optional[MultiDimensionalSignal]) -> float:
        """计算聚合质量分数"""
        try:
            if not aggregated_signal or not signal_inputs:
                return 0.0
            
            # 输入信号质量平均
            input_quality = np.mean([si.signal.signal_quality_score for si in signal_inputs])
            
            # 聚合信号质量
            output_quality = aggregated_signal.signal_quality_score
            
            # 一致性评分
            consistency_scores = []
            for si in signal_inputs:
                direction_consistency = self._calculate_direction_consistency(
                    si.signal, aggregated_signal
                )
                consistency_scores.append(direction_consistency)
            
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.5
            
            # 综合质量分数
            quality_score = (
                input_quality * 0.4 +
                output_quality * 0.4 +
                avg_consistency * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.log_error(f"计算聚合质量失败: {e}")
            return 0.0
    
    def _calculate_direction_consistency(self, 
                                       signal1: MultiDimensionalSignal,
                                       signal2: MultiDimensionalSignal) -> float:
        """计算方向一致性"""
        try:
            # 简化的方向一致性计算
            sig1_buy = signal1.primary_signal._is_buy_signal()
            sig1_sell = signal1.primary_signal._is_sell_signal()
            sig2_buy = signal2.primary_signal._is_buy_signal()
            sig2_sell = signal2.primary_signal._is_sell_signal()
            
            if (sig1_buy and sig2_buy) or (sig1_sell and sig2_sell):
                return 1.0  # 完全一致
            elif (sig1_buy and sig2_sell) or (sig1_sell and sig2_buy):
                return 0.0  # 完全冲突
            else:
                return 0.5  # 部分一致
                
        except Exception as e:
            self.log_error(f"计算方向一致性失败: {e}")
            return 0.5
    
    def _generate_aggregation_reasoning(self, 
                                      signal_inputs: List[SignalInput],
                                      strategy: AggregationStrategy,
                                      conflicts_detected: List[ConflictDetail],
                                      conflicts_resolved: List[ConflictResolution]) -> List[str]:
        """生成聚合推理"""
        reasoning = []
        
        reasoning.append(f"聚合策略: {strategy.value}")
        reasoning.append(f"输入信号数量: {len(signal_inputs)}")
        
        if signal_inputs:
            avg_confidence = np.mean([si.signal.overall_confidence for si in signal_inputs])
            avg_quality = np.mean([si.signal.signal_quality_score for si in signal_inputs])
            reasoning.append(f"平均置信度: {avg_confidence:.2f}")
            reasoning.append(f"平均质量分数: {avg_quality:.2f}")
        
        if conflicts_detected:
            reasoning.append(f"检测到冲突: {len(conflicts_detected)}个")
            conflict_types = set(c.conflict_type.value for c in conflicts_detected)
            reasoning.append(f"冲突类型: {', '.join(conflict_types)}")
        
        if conflicts_resolved:
            reasoning.append(f"解决冲突: {len(conflicts_resolved)}个")
        
        # 信号源统计
        source_counts = defaultdict(int)
        for si in signal_inputs:
            source_counts[si.source_type.value] += 1
        
        if source_counts:
            source_info = ", ".join([f"{k}: {v}" for k, v in source_counts.items()])
            reasoning.append(f"信号源分布: {source_info}")
        
        return reasoning
    
    def _update_statistics(self, result: AggregationResult) -> None:
        """更新统计信息"""
        try:
            # 更新处理时间统计
            if self.statistics.total_aggregations == 1:
                self.statistics.avg_processing_time_ms = result.processing_time_ms
            else:
                # 移动平均
                self.statistics.avg_processing_time_ms = (
                    self.statistics.avg_processing_time_ms * 0.9 + 
                    result.processing_time_ms * 0.1
                )
            
            self.statistics.max_processing_time_ms = max(
                self.statistics.max_processing_time_ms,
                result.processing_time_ms
            )
            
            # 更新冲突统计
            self.statistics.conflicts_detected += len(result.conflicts_detected)
            self.statistics.conflicts_resolved += len(result.conflicts_resolved)
            
            # 更新信号源统计
            for signal_input in result.input_signals:
                source_type = signal_input.source_type.value
                self.statistics.signal_sources_stats[source_type] = (
                    self.statistics.signal_sources_stats.get(source_type, 0) + 1
                )
            
            # 更新策略使用统计
            strategy = result.strategy_used.value
            self.statistics.strategy_usage_stats[strategy] = (
                self.statistics.strategy_usage_stats.get(strategy, 0) + 1
            )
            
        except Exception as e:
            self.log_error(f"更新统计失败: {e}")
    
    async def _processing_loop(self) -> None:
        """处理循环"""
        while self._running:
            try:
                # 执行定期维护
                await self._periodic_maintenance()
                await asyncio.sleep(60)  # 每分钟执行一次
                
            except Exception as e:
                self.log_error(f"处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _periodic_maintenance(self) -> None:
        """定期维护"""
        try:
            # 清理过期缓存
            current_time = datetime.now()
            time_threshold = current_time - timedelta(hours=1)
            
            for symbol in list(self.signal_cache.keys()):
                cache = self.signal_cache[symbol]
                # 保留最近1小时内的数据
                filtered_cache = deque([
                    item for item in cache 
                    if hasattr(item, 'created_at') and item.created_at > time_threshold
                ], maxlen=100)
                self.signal_cache[symbol] = filtered_cache
            
            # 清理历史记录
            if len(self.aggregation_history) > 500:
                # 只保留最近500个聚合结果
                recent_history = deque(list(self.aggregation_history)[-500:], maxlen=1000)
                self.aggregation_history = recent_history
            
        except Exception as e:
            self.log_error(f"定期维护失败: {e}")
    
    # 公共接口方法
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        try:
            success_rate = 0.0
            if self.statistics.total_aggregations > 0:
                success_rate = self.statistics.successful_aggregations / self.statistics.total_aggregations
            
            return {
                'total_aggregations': self.statistics.total_aggregations,
                'successful_aggregations': self.statistics.successful_aggregations,
                'failed_aggregations': self.statistics.failed_aggregations,
                'success_rate': success_rate,
                'avg_processing_time_ms': self.statistics.avg_processing_time_ms,
                'max_processing_time_ms': self.statistics.max_processing_time_ms,
                'conflicts_detected': self.statistics.conflicts_detected,
                'conflicts_resolved': self.statistics.conflicts_resolved,
                'signal_sources_stats': dict(self.statistics.signal_sources_stats),
                'strategy_usage_stats': dict(self.statistics.strategy_usage_stats),
                'recent_aggregations': len(self.aggregation_history)
            }
            
        except Exception as e:
            self.log_error(f"获取统计信息失败: {e}")
            return {}
    
    def get_recent_aggregations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的聚合结果"""
        try:
            recent_results = list(self.aggregation_history)[-limit:]
            
            return [{
                'aggregation_id': result.aggregation_id,
                'strategy_used': result.strategy_used.value,
                'input_count': len(result.input_signals),
                'quality_score': result.quality_score,
                'confidence_adjustment': result.confidence_adjustment,
                'processing_time_ms': result.processing_time_ms,
                'conflicts_detected': len(result.conflicts_detected),
                'created_at': result.created_at.isoformat(),
                'success': result.aggregated_signal is not None
            } for result in recent_results]
            
        except Exception as e:
            self.log_error(f"获取最近聚合结果失败: {e}")
            return []
    
    def update_config(self, config: AggregationConfig) -> None:
        """更新配置"""
        try:
            self.config = config
            self.log_info("SignalAggregator配置已更新")
            
        except Exception as e:
            self.log_error(f"更新配置失败: {e}")
    
    def get_unified_interface(self) -> UnifiedSignalInterface:
        """获取统一信号接口"""
        return self.unified_interface