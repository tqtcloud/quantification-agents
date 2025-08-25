"""
信号聚合器系统完整测试套件

测试覆盖：
1. ConflictResolver 冲突解决器
2. PriorityManager 优先级管理器  
3. SignalAggregator 核心信号聚合器
4. UnifiedSignalInterface 统一信号接口
5. 集成测试和边界条件测试
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, List, Any

# 导入待测试的模块
from src.strategy.conflict_resolver import (
    ConflictResolver, ConflictType, ConflictSeverity, 
    ConflictResolutionStrategy, ConflictDetail, ConflictResolution
)
from src.strategy.priority_manager import (
    PriorityManager, PriorityCategory, SignalSource as PrioritySignalSource,
    PerformanceMetrics, MarketCondition
)
from src.strategy.signal_aggregator import (
    SignalAggregator, AggregationStrategy, SignalSource,
    SignalInput, AggregationResult, AggregationConfig,
    UnifiedSignalInterface
)
from src.core.models.signals import TradingSignal, MultiDimensionalSignal, SignalStrength


class TestSignalCreationHelpers:
    """测试信号创建辅助类"""
    
    @staticmethod
    def create_trading_signal(
        symbol: str = "BTCUSDT",
        signal_type: SignalStrength = SignalStrength.BUY,
        confidence: float = 0.8,
        entry_price: float = 50000.0,
        target_price: float = 52000.0,
        stop_loss: float = 48000.0
    ) -> TradingSignal:
        """创建测试用交易信号"""
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=["测试信号"],
            indicators_consensus={"rsi": 0.7, "macd": 0.6}
        )
    
    @staticmethod
    def create_multidimensional_signal(
        symbol: str = "BTCUSDT",
        signal_type: SignalStrength = SignalStrength.BUY,
        overall_confidence: float = 0.8,
        momentum_score: float = 0.7,
        volatility_score: float = 0.3
    ) -> MultiDimensionalSignal:
        """创建测试用多维度信号"""
        primary_signal = TestSignalCreationHelpers.create_trading_signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=overall_confidence
        )
        
        return MultiDimensionalSignal(
            primary_signal=primary_signal,
            momentum_score=momentum_score,
            mean_reversion_score=0.2,
            volatility_score=volatility_score,
            volume_score=0.6,
            sentiment_score=0.4,
            overall_confidence=overall_confidence,
            risk_reward_ratio=2.5,
            max_position_size=0.1
        )


class TestConflictResolver:
    """冲突解决器测试类"""
    
    @pytest.fixture
    def conflict_resolver(self):
        """创建冲突解决器实例"""
        return ConflictResolver(
            conflict_threshold=0.3,
            time_window_minutes=5
        )
    
    def test_init(self, conflict_resolver):
        """测试初始化"""
        assert conflict_resolver.conflict_threshold == 0.3
        assert conflict_resolver.time_window.total_seconds() == 300
        assert len(conflict_resolver.conflict_history) == 0
        assert len(conflict_resolver.resolution_history) == 0
    
    def test_detect_no_conflicts_single_signal(self, conflict_resolver):
        """测试单个信号无冲突情况"""
        signal = TestSignalCreationHelpers.create_multidimensional_signal()
        conflicts = conflict_resolver.detect_conflicts([signal])
        assert len(conflicts) == 0
    
    def test_detect_direction_conflict(self, conflict_resolver):
        """测试方向冲突检测"""
        buy_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.STRONG_BUY
        )
        sell_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.STRONG_SELL
        )
        
        conflicts = conflict_resolver.detect_conflicts([buy_signal, sell_signal])
        
        assert len(conflicts) >= 1
        direction_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DIRECTION_CONFLICT]
        assert len(direction_conflicts) >= 1
        assert direction_conflicts[0].severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]
    
    def test_detect_strength_conflict(self, conflict_resolver):
        """测试强度冲突检测"""
        high_conf_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.9
        )
        low_conf_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.4
        )
        
        conflicts = conflict_resolver.detect_conflicts([high_conf_signal, low_conf_signal])
        
        strength_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.STRENGTH_CONFLICT]
        assert len(strength_conflicts) >= 1
    
    def test_resolve_by_confidence(self, conflict_resolver):
        """测试基于置信度的冲突解决"""
        high_conf_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.9
        )
        low_conf_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.5
        )
        
        signals = [high_conf_signal, low_conf_signal]
        conflicts = conflict_resolver.detect_conflicts(signals)
        
        if conflicts:
            resolutions = conflict_resolver.resolve_conflicts(
                signals, conflicts, strategy=ConflictResolutionStrategy.CONFIDENCE_WEIGHTED
            )
            
            assert len(resolutions) > 0
            for resolution in resolutions:
                assert resolution.resolved_signal is not None
                assert resolution.strategy_used == ConflictResolutionStrategy.CONFIDENCE_WEIGHTED
    
    def test_resolve_by_priority(self, conflict_resolver):
        """测试基于优先级的冲突解决"""
        signal1 = TestSignalCreationHelpers.create_multidimensional_signal()
        signal2 = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.SELL
        )
        
        signals = [signal1, signal2]
        priorities = {str(id(signal1)): 0.8, str(id(signal2)): 0.6}
        conflicts = conflict_resolver.detect_conflicts(signals, priorities)
        
        if conflicts:
            resolutions = conflict_resolver.resolve_conflicts(
                signals, conflicts, priorities, ConflictResolutionStrategy.PRIORITY_WEIGHTED
            )
            
            assert len(resolutions) > 0
            for resolution in resolutions:
                assert resolution.resolved_signal is not None
    
    def test_get_statistics(self, conflict_resolver):
        """测试统计信息获取"""
        # 创建一些冲突来产生统计数据
        buy_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.BUY
        )
        sell_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.SELL
        )
        
        signals = [buy_signal, sell_signal]
        conflicts = conflict_resolver.detect_conflicts(signals)
        
        stats = conflict_resolver.get_conflict_statistics()
        
        assert 'total_conflicts' in stats
        assert 'conflicts_by_type' in stats
        assert 'resolution_count' in stats
        assert isinstance(stats['total_conflicts'], int)


class TestPriorityManager:
    """优先级管理器测试类"""
    
    @pytest.fixture
    def priority_manager(self):
        """创建优先级管理器实例"""
        return PriorityManager(
            default_priority=0.5,
            performance_window_hours=24,
            min_trades_for_adjustment=5
        )
    
    def test_init(self, priority_manager):
        """测试初始化"""
        assert priority_manager.default_priority == 0.5
        assert priority_manager.performance_window.total_seconds() == 86400  # 24 hours
        assert priority_manager.min_trades_for_adjustment == 5
        assert len(priority_manager.signal_sources) == 0
    
    def test_register_signal_source(self, priority_manager):
        """测试信号源注册"""
        success = priority_manager.register_signal_source(
            source_id="hft_001",
            source_name="HFT Engine 1",
            source_type="HFT",
            base_priority=0.8
        )
        
        assert success is True
        assert "hft_001" in priority_manager.signal_sources
        assert priority_manager.signal_sources["hft_001"].base_priority == 0.8
        assert priority_manager.signal_sources["hft_001"].current_priority == 0.8
    
    def test_register_duplicate_source(self, priority_manager):
        """测试重复注册信号源"""
        priority_manager.register_signal_source("test_001", "Test Source", "TEST")
        
        # 重复注册应该失败
        success = priority_manager.register_signal_source("test_001", "Test Source 2", "TEST")
        assert success is False
    
    def test_unregister_signal_source(self, priority_manager):
        """测试信号源注销"""
        priority_manager.register_signal_source("test_001", "Test Source", "TEST")
        
        success = priority_manager.unregister_signal_source("test_001")
        assert success is True
        assert "test_001" not in priority_manager.signal_sources
    
    def test_get_signal_priority(self, priority_manager):
        """测试获取信号优先级"""
        priority_manager.register_signal_source("test_001", "Test Source", "HFT", 0.7)
        
        signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.8
        )
        
        priority = priority_manager.get_signal_priority(signal, "test_001")
        
        assert isinstance(priority, float)
        assert 0.0 <= priority <= 1.0
    
    def test_update_performance(self, priority_manager):
        """测试性能数据更新"""
        priority_manager.register_signal_source("test_001", "Test Source", "HFT")
        
        performance_data = {
            'return': 0.05,
            'accuracy': 0.75,
            'trade_count': 10
        }
        
        success = priority_manager.update_performance("test_001", performance_data)
        assert success is True
        
        # 检查历史记录
        assert len(priority_manager.performance_history["test_001"]) == 1
    
    def test_update_market_condition(self, priority_manager):
        """测试市场条件更新"""
        # 先注册一些信号源
        priority_manager.register_signal_source("hft_001", "HFT Engine", "HFT", 0.7)
        priority_manager.register_signal_source("ai_001", "AI Agent", "AI_AGENT", 0.6)
        
        market_condition = MarketCondition(
            volatility_level="high",
            trend_strength=0.8,
            volume_profile="high",
            market_stress=0.6,
            sentiment_score=0.3
        )
        
        priority_manager.update_market_condition(market_condition)
        
        assert priority_manager.current_market_condition == market_condition
        assert len(priority_manager.market_condition_history) == 1
    
    def test_set_manual_priority(self, priority_manager):
        """测试手动设置优先级"""
        priority_manager.register_signal_source("test_001", "Test Source", "TEST", 0.5)
        
        success = priority_manager.set_manual_priority("test_001", 0.9, "测试手动设置")
        assert success is True
        
        source = priority_manager.signal_sources["test_001"]
        assert source.current_priority == 0.9
        assert len(priority_manager.adjustment_history) == 1
    
    def test_get_priority_rankings(self, priority_manager):
        """测试优先级排名获取"""
        priority_manager.register_signal_source("source1", "Source 1", "TEST", 0.8)
        priority_manager.register_signal_source("source2", "Source 2", "TEST", 0.6)
        priority_manager.register_signal_source("source3", "Source 3", "TEST", 0.9)
        
        rankings = priority_manager.get_priority_rankings()
        
        assert len(rankings) == 3
        assert rankings[0][1] >= rankings[1][1] >= rankings[2][1]  # 降序排列
        assert rankings[0][0] == "source3"  # 最高优先级
    
    def test_get_priority_statistics(self, priority_manager):
        """测试优先级统计信息获取"""
        priority_manager.register_signal_source("hft1", "HFT 1", "HFT", 0.8)
        priority_manager.register_signal_source("ai1", "AI 1", "AI_AGENT", 0.6)
        priority_manager.register_signal_source("hft2", "HFT 2", "HFT", 0.7)
        
        stats = priority_manager.get_priority_statistics()
        
        assert 'total_sources' in stats
        assert 'avg_priority' in stats
        assert 'by_category' in stats
        assert stats['total_sources'] == 3
        assert 'HFT' in stats['by_category']
        assert 'AI_AGENT' in stats['by_category']
        assert stats['by_category']['HFT']['count'] == 2


class TestSignalAggregator:
    """信号聚合器测试类"""
    
    @pytest.fixture
    def config(self):
        """创建聚合配置"""
        return AggregationConfig(
            strategy=AggregationStrategy.WEIGHTED_AVERAGE,
            min_signal_count=2,
            max_signal_count=5,
            min_confidence_threshold=0.5,
            min_quality_threshold=0.4
        )
    
    @pytest.fixture
    def signal_aggregator(self, config):
        """创建信号聚合器实例"""
        return SignalAggregator(config=config)
    
    @pytest.fixture
    def sample_signal_inputs(self):
        """创建样本信号输入"""
        signal1 = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.BUY,
            overall_confidence=0.8
        )
        signal2 = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.BUY,
            overall_confidence=0.7
        )
        
        return [
            SignalInput(
                signal_id="sig1",
                signal=signal1,
                source_type=SignalSource.HFT_ENGINE,
                source_id="hft_001",
                priority=0.8
            ),
            SignalInput(
                signal_id="sig2",
                signal=signal2,
                source_type=SignalSource.AI_AGENT,
                source_id="ai_001",
                priority=0.6
            )
        ]
    
    def test_init(self, signal_aggregator, config):
        """测试初始化"""
        assert signal_aggregator.config == config
        assert signal_aggregator.conflict_resolver is not None
        assert signal_aggregator.priority_manager is not None
        assert signal_aggregator.unified_interface is not None
    
    @pytest.mark.asyncio
    async def test_start_stop(self, signal_aggregator):
        """测试启动和停止"""
        assert not signal_aggregator._running
        
        await signal_aggregator.start()
        assert signal_aggregator._running
        assert signal_aggregator._processing_task is not None
        
        await signal_aggregator.stop()
        assert not signal_aggregator._running
    
    def test_validate_inputs_valid(self, signal_aggregator, sample_signal_inputs):
        """测试有效输入验证"""
        result = signal_aggregator._validate_inputs(sample_signal_inputs)
        assert result is True
    
    def test_validate_inputs_insufficient_count(self, signal_aggregator):
        """测试信号数量不足的验证"""
        single_signal = [SignalInput(
            signal_id="sig1",
            signal=TestSignalCreationHelpers.create_multidimensional_signal(),
            source_type=SignalSource.HFT_ENGINE,
            source_id="hft_001",
            priority=0.8
        )]
        
        result = signal_aggregator._validate_inputs(single_signal)
        assert result is False
    
    def test_validate_inputs_low_confidence(self, signal_aggregator, config):
        """测试低置信度信号验证"""
        low_conf_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            overall_confidence=0.3  # 低于阈值0.5
        )
        
        signal_inputs = [
            SignalInput(
                signal_id="sig1",
                signal=low_conf_signal,
                source_type=SignalSource.HFT_ENGINE,
                source_id="hft_001",
                priority=0.8
            ),
            SignalInput(
                signal_id="sig2",
                signal=TestSignalCreationHelpers.create_multidimensional_signal(),
                source_type=SignalSource.AI_AGENT,
                source_id="ai_001",
                priority=0.6
            )
        ]
        
        result = signal_aggregator._validate_inputs(signal_inputs)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_weighted_average_aggregation(self, signal_aggregator, sample_signal_inputs):
        """测试加权平均聚合"""
        result = await signal_aggregator._weighted_average_aggregation(sample_signal_inputs)
        
        assert result is not None
        assert isinstance(result, MultiDimensionalSignal)
        assert 0.0 <= result.overall_confidence <= 1.0
        assert 0.0 <= result.signal_quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_priority_selection_aggregation(self, signal_aggregator, sample_signal_inputs):
        """测试优先级选择聚合"""
        result = await signal_aggregator._priority_selection_aggregation(sample_signal_inputs)
        
        assert result is not None
        assert isinstance(result, MultiDimensionalSignal)
        
        # 应该选择优先级最高的信号（第一个信号，优先级0.8）
        highest_priority_signal = max(sample_signal_inputs, key=lambda x: x.priority)
        assert result.primary_signal.symbol == highest_priority_signal.signal.primary_signal.symbol
    
    @pytest.mark.asyncio
    async def test_consensus_voting_aggregation(self, signal_aggregator, sample_signal_inputs):
        """测试共识投票聚合"""
        result = await signal_aggregator._consensus_voting_aggregation(sample_signal_inputs)
        
        assert result is not None
        assert isinstance(result, MultiDimensionalSignal)
    
    @pytest.mark.asyncio
    async def test_hybrid_fusion_aggregation(self, signal_aggregator, sample_signal_inputs):
        """测试混合融合聚合"""
        result = await signal_aggregator._hybrid_fusion_aggregation(sample_signal_inputs, [])
        
        assert result is not None
        assert isinstance(result, MultiDimensionalSignal)
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_success(self, signal_aggregator, sample_signal_inputs):
        """测试成功的信号聚合"""
        # 设置回调来捕获结果
        callback_results = []
        
        async def signal_callback(result):
            callback_results.append(result)
        
        signal_aggregator.unified_interface.register_signal_callback(signal_callback)
        
        result = await signal_aggregator.aggregate_signals(sample_signal_inputs)
        
        assert result is not None
        assert isinstance(result, AggregationResult)
        assert result.aggregated_signal is not None
        assert result.strategy_used == AggregationStrategy.WEIGHTED_AVERAGE
        assert result.processing_time_ms > 0
        assert len(result.reasoning) > 0
        
        # 检查回调是否被调用
        assert len(callback_results) == 1
        assert callback_results[0] == result
    
    @pytest.mark.asyncio
    async def test_aggregate_signals_with_conflicts(self, signal_aggregator):
        """测试带冲突的信号聚合"""
        # 创建冲突信号（买入vs卖出）
        buy_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.STRONG_BUY
        )
        sell_signal = TestSignalCreationHelpers.create_multidimensional_signal(
            signal_type=SignalStrength.STRONG_SELL
        )
        
        signal_inputs = [
            SignalInput(
                signal_id="buy_sig",
                signal=buy_signal,
                source_type=SignalSource.HFT_ENGINE,
                source_id="hft_001",
                priority=0.8
            ),
            SignalInput(
                signal_id="sell_sig",
                signal=sell_signal,
                source_type=SignalSource.AI_AGENT,
                source_id="ai_001",
                priority=0.7
            )
        ]
        
        result = await signal_aggregator.aggregate_signals(signal_inputs)
        
        assert result is not None
        assert len(result.conflicts_detected) > 0
        # 可能会有冲突解决结果
        assert len(result.conflicts_resolved) >= 0
    
    def test_calculate_signal_weight(self, signal_aggregator):
        """测试信号权重计算"""
        signal_input = SignalInput(
            signal_id="test_sig",
            signal=TestSignalCreationHelpers.create_multidimensional_signal(
                overall_confidence=0.8
            ),
            source_type=SignalSource.HFT_ENGINE,
            source_id="hft_001",
            priority=0.7
        )
        
        weight = signal_aggregator._calculate_signal_weight(signal_input)
        
        assert isinstance(weight, float)
        assert 0.1 <= weight <= 2.0
    
    def test_calculate_aggregation_quality(self, signal_aggregator, sample_signal_inputs):
        """测试聚合质量计算"""
        aggregated_signal = TestSignalCreationHelpers.create_multidimensional_signal()
        
        quality = signal_aggregator._calculate_aggregation_quality(
            sample_signal_inputs, aggregated_signal
        )
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
    
    def test_get_aggregation_statistics(self, signal_aggregator):
        """测试聚合统计信息获取"""
        # 先更新一些统计数据
        signal_aggregator.statistics.total_aggregations = 10
        signal_aggregator.statistics.successful_aggregations = 8
        signal_aggregator.statistics.failed_aggregations = 2
        
        stats = signal_aggregator.get_aggregation_statistics()
        
        assert 'total_aggregations' in stats
        assert 'successful_aggregations' in stats
        assert 'success_rate' in stats
        assert stats['total_aggregations'] == 10
        assert stats['success_rate'] == 0.8
    
    def test_update_config(self, signal_aggregator):
        """测试配置更新"""
        new_config = AggregationConfig(
            strategy=AggregationStrategy.CONSENSUS_VOTING,
            min_signal_count=3
        )
        
        signal_aggregator.update_config(new_config)
        
        assert signal_aggregator.config == new_config
        assert signal_aggregator.config.strategy == AggregationStrategy.CONSENSUS_VOTING
        assert signal_aggregator.config.min_signal_count == 3


class TestUnifiedSignalInterface:
    """统一信号接口测试类"""
    
    @pytest.fixture
    def interface(self):
        """创建统一信号接口实例"""
        return UnifiedSignalInterface()
    
    def test_init(self, interface):
        """测试初始化"""
        assert len(interface.signal_callbacks) == 0
        assert len(interface.error_callbacks) == 0
    
    def test_register_callbacks(self, interface):
        """测试回调注册"""
        def signal_callback(result):
            pass
        
        def error_callback(error, context):
            pass
        
        interface.register_signal_callback(signal_callback)
        interface.register_error_callback(error_callback)
        
        assert len(interface.signal_callbacks) == 1
        assert len(interface.error_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_emit_signal(self, interface):
        """测试信号发送"""
        callback_results = []
        
        def signal_callback(result):
            callback_results.append(result)
        
        interface.register_signal_callback(signal_callback)
        
        # 创建测试结果
        test_result = AggregationResult(
            aggregation_id="test_agg",
            aggregated_signal=None,
            input_signals=[],
            conflicts_detected=[],
            conflicts_resolved=[],
            strategy_used=AggregationStrategy.WEIGHTED_AVERAGE,
            confidence_adjustment=0.0,
            quality_score=0.5,
            reasoning=["测试"],
            processing_time_ms=10.0
        )
        
        await interface.emit_signal(test_result)
        
        assert len(callback_results) == 1
        assert callback_results[0] == test_result
    
    @pytest.mark.asyncio
    async def test_emit_error(self, interface):
        """测试错误发送"""
        error_results = []
        
        def error_callback(error, context):
            error_results.append((error, context))
        
        interface.register_error_callback(error_callback)
        
        test_error = ValueError("测试错误")
        test_context = {"test": "context"}
        
        await interface.emit_error(test_error, test_context)
        
        assert len(error_results) == 1
        assert error_results[0][0] == test_error
        assert error_results[0][1] == test_context


class TestIntegrationScenarios:
    """集成测试场景"""
    
    @pytest.fixture
    def full_system(self):
        """创建完整系统"""
        conflict_resolver = ConflictResolver()
        priority_manager = PriorityManager()
        config = AggregationConfig()
        
        # 注册一些信号源
        priority_manager.register_signal_source("hft_001", "HFT Engine 1", "HFT", 0.8)
        priority_manager.register_signal_source("ai_001", "AI Agent 1", "AI_AGENT", 0.6)
        
        aggregator = SignalAggregator(
            config=config,
            conflict_resolver=conflict_resolver,
            priority_manager=priority_manager
        )
        
        return {
            'aggregator': aggregator,
            'conflict_resolver': conflict_resolver,
            'priority_manager': priority_manager
        }
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, full_system):
        """测试完整工作流"""
        aggregator = full_system['aggregator']
        priority_manager = full_system['priority_manager']
        
        await aggregator.start()
        
        try:
            # 创建多个不同类型的信号
            hft_signal = TestSignalCreationHelpers.create_multidimensional_signal(
                signal_type=SignalStrength.BUY,
                overall_confidence=0.85
            )
            
            ai_signal = TestSignalCreationHelpers.create_multidimensional_signal(
                signal_type=SignalStrength.BUY,
                overall_confidence=0.75,
                momentum_score=0.6
            )
            
            signal_inputs = [
                SignalInput(
                    signal_id="hft_sig1",
                    signal=hft_signal,
                    source_type=SignalSource.HFT_ENGINE,
                    source_id="hft_001",
                    priority=0.8
                ),
                SignalInput(
                    signal_id="ai_sig1",
                    signal=ai_signal,
                    source_type=SignalSource.AI_AGENT,
                    source_id="ai_001",
                    priority=0.6
                )
            ]
            
            # 执行聚合
            result = await aggregator.aggregate_signals(signal_inputs)
            
            # 验证结果
            assert result is not None
            assert result.aggregated_signal is not None
            assert result.quality_score > 0
            assert len(result.reasoning) > 0
            
            # 验证统计信息
            stats = aggregator.get_aggregation_statistics()
            assert stats['total_aggregations'] >= 1
            assert stats['successful_aggregations'] >= 1
            
        finally:
            await aggregator.stop()
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_workflow(self, full_system):
        """测试冲突解决工作流"""
        aggregator = full_system['aggregator']
        
        await aggregator.start()
        
        try:
            # 创建冲突信号
            buy_signal = TestSignalCreationHelpers.create_multidimensional_signal(
                signal_type=SignalStrength.STRONG_BUY,
                overall_confidence=0.8
            )
            
            sell_signal = TestSignalCreationHelpers.create_multidimensional_signal(
                signal_type=SignalStrength.STRONG_SELL,
                overall_confidence=0.75
            )
            
            signal_inputs = [
                SignalInput(
                    signal_id="buy_sig",
                    signal=buy_signal,
                    source_type=SignalSource.HFT_ENGINE,
                    source_id="hft_001",
                    priority=0.8
                ),
                SignalInput(
                    signal_id="sell_sig",
                    signal=sell_signal,
                    source_type=SignalSource.AI_AGENT,
                    source_id="ai_001",
                    priority=0.7
                )
            ]
            
            # 执行聚合（应该检测到冲突）
            result = await aggregator.aggregate_signals(signal_inputs)
            
            # 验证冲突处理
            assert result is not None
            assert len(result.conflicts_detected) > 0
            
            # 验证冲突解决器统计
            conflict_stats = aggregator.conflict_resolver.get_conflict_statistics()
            assert conflict_stats['total_conflicts'] > 0
            
        finally:
            await aggregator.stop()
    
    @pytest.mark.asyncio
    async def test_priority_adjustment_workflow(self, full_system):
        """测试优先级调整工作流"""
        priority_manager = full_system['priority_manager']
        
        # 模拟性能数据更新
        performance_data = [
            {'return': 0.05, 'accuracy': 0.8},
            {'return': 0.03, 'accuracy': 0.75},
            {'return': 0.02, 'accuracy': 0.85},
            {'return': 0.04, 'accuracy': 0.9},
            {'return': 0.01, 'accuracy': 0.7}
        ]
        
        for data in performance_data:
            priority_manager.update_performance("hft_001", data)
        
        # 检查是否有调整记录
        assert len(priority_manager.performance_history["hft_001"]) == 5
        
        # 测试市场条件更新
        market_condition = MarketCondition(
            volatility_level="high",
            trend_strength=0.9,
            volume_profile="high",
            market_stress=0.4,
            sentiment_score=0.6
        )
        
        priority_manager.update_market_condition(market_condition)
        
        # 验证市场条件影响
        assert priority_manager.current_market_condition == market_condition


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_aggregation_with_invalid_signals(self):
        """测试无效信号的聚合"""
        aggregator = SignalAggregator()
        
        # 创建无效信号输入（空列表）
        with pytest.raises(ValueError):
            await aggregator.aggregate_signals([])
    
    @pytest.mark.asyncio
    async def test_conflict_resolver_with_none_signals(self):
        """测试空信号的冲突检测"""
        resolver = ConflictResolver()
        
        conflicts = resolver.detect_conflicts([])
        assert len(conflicts) == 0
        
        # 测试None信号
        conflicts = resolver.detect_conflicts([None])
        assert len(conflicts) == 0
    
    def test_priority_manager_invalid_operations(self):
        """测试优先级管理器的无效操作"""
        manager = PriorityManager()
        
        # 测试获取不存在信号源的优先级
        signal = TestSignalCreationHelpers.create_multidimensional_signal()
        priority = manager.get_signal_priority(signal, "nonexistent")
        assert priority == manager.default_priority
        
        # 测试为不存在的信号源更新性能
        success = manager.update_performance("nonexistent", {'return': 0.05})
        assert success is False
        
        # 测试设置无效优先级
        manager.register_signal_source("test", "Test", "TEST")
        success = manager.set_manual_priority("test", 1.5)  # 超出范围
        assert success is False


class TestPerformanceAndScaling:
    """性能和扩展性测试"""
    
    @pytest.mark.asyncio
    async def test_large_signal_aggregation(self):
        """测试大量信号聚合性能"""
        config = AggregationConfig(max_signal_count=50)
        aggregator = SignalAggregator(config=config)
        
        # 创建大量信号
        signal_inputs = []
        for i in range(20):
            signal = TestSignalCreationHelpers.create_multidimensional_signal()
            signal_inputs.append(SignalInput(
                signal_id=f"sig_{i}",
                signal=signal,
                source_type=SignalSource.HFT_ENGINE,
                source_id=f"hft_{i}",
                priority=0.5 + (i % 5) * 0.1
            ))
        
        start_time = time.perf_counter()
        result = await aggregator.aggregate_signals(signal_inputs)
        processing_time = time.perf_counter() - start_time
        
        assert result is not None
        assert processing_time < 1.0  # 应该在1秒内完成
        assert result.processing_time_ms > 0
    
    def test_memory_usage_with_history(self):
        """测试历史记录的内存使用"""
        aggregator = SignalAggregator()
        
        # 模拟添加大量历史记录
        for i in range(2000):
            result = AggregationResult(
                aggregation_id=f"agg_{i}",
                aggregated_signal=None,
                input_signals=[],
                conflicts_detected=[],
                conflicts_resolved=[],
                strategy_used=AggregationStrategy.WEIGHTED_AVERAGE,
                confidence_adjustment=0.0,
                quality_score=0.5,
                reasoning=[],
                processing_time_ms=1.0
            )
            aggregator.aggregation_history.append(result)
        
        # 检查历史记录限制是否工作
        assert len(aggregator.aggregation_history) <= 1000  # maxlen限制


if __name__ == "__main__":
    pytest.main([__file__, "-v"])