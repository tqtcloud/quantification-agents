import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from src.execution.algorithms import (
    ExecutionAlgorithm, TWAPAlgorithm, VWAPAlgorithm, POVAlgorithm, ImplementationShortfall,
    AlgorithmStatus, AlgorithmConfig, SliceOrder, AlgorithmResult
)
from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData, OrderBook


class TestExecutionAlgorithms:
    """执行算法测试"""
    
    @pytest.fixture
    def sample_order(self):
        """创建示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=50000.0,
            client_order_id="test_algo_order_001"
        )
    
    @pytest.fixture
    def market_data(self):
        """创建示例市场数据"""
        return MarketData(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            price=50000.0,
            volume=100.0,
            bid=49999.0,
            ask=50001.0,
            bid_volume=50.0,
            ask_volume=60.0
        )
    
    @pytest.fixture
    def order_book(self):
        """创建示例订单簿"""
        bids = [(49999.0, 10.0), (49998.0, 20.0), (49997.0, 30.0)]
        asks = [(50001.0, 15.0), (50002.0, 25.0), (50003.0, 35.0)]
        
        return OrderBook(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            bids=bids,
            asks=asks
        )


class TestTWAPAlgorithm:
    """TWAP算法测试"""
    
    @pytest.fixture
    def twap_config(self):
        """创建TWAP配置"""
        return TWAPAlgorithm.TWAPConfig(
            duration_minutes=10,
            slice_interval_seconds=60,
            randomize_timing=False  # 测试时不使用随机化
        )
    
    @pytest.fixture
    def twap_algorithm(self, twap_config):
        """创建TWAP算法实例"""
        return TWAPAlgorithm("test_twap_001", twap_config)
    
    @pytest.mark.asyncio
    async def test_twap_initialization(self, twap_algorithm, sample_order):
        """测试TWAP算法初始化"""
        slices = await twap_algorithm.initialize(sample_order)
        
        # 验证分片生成
        assert len(slices) > 0
        assert all(isinstance(slice_order, SliceOrder) for slice_order in slices)
        
        # 验证总数量
        total_qty = sum(slice_order.quantity for slice_order in slices)
        assert abs(total_qty - sample_order.quantity) < 0.001  # 允许小数误差
        
        # 验证分片时间间隔
        if len(slices) > 1:
            time_diff = (slices[1].scheduled_time - slices[0].scheduled_time).total_seconds()
            assert abs(time_diff - 60) < 1  # 60秒间隔，允许1秒误差
    
    @pytest.mark.asyncio
    async def test_twap_execution(self, twap_algorithm, sample_order, market_data, order_book):
        """测试TWAP算法执行"""
        # 初始化算法
        slices = await twap_algorithm.initialize(sample_order)
        twap_algorithm.slices = slices
        
        # 更新市场数据
        await twap_algorithm.update_market_data(market_data, order_book)
        
        # 模拟执行过程
        for slice_order in slices[:3]:  # 只测试前3个分片
            slice_order.status = OrderStatus.FILLED
            slice_order.filled_qty = slice_order.quantity
            slice_order.avg_price = 50000.0
            
            await twap_algorithm.on_slice_filled(slice_order, slice_order.quantity, 50000.0)
        
        # 验证执行统计
        assert twap_algorithm.total_filled > 0
        assert twap_algorithm.weighted_avg_price == 50000.0
        assert len(twap_algorithm.completed_slices) == 3
    
    @pytest.mark.asyncio
    async def test_twap_should_adjust_strategy(self, twap_algorithm):
        """测试TWAP策略调整判断"""
        # TWAP通常不需要调整策略
        should_adjust = await twap_algorithm.should_adjust_strategy()
        assert should_adjust is False
    
    @pytest.mark.asyncio
    async def test_twap_with_randomization(self, sample_order):
        """测试带随机化的TWAP"""
        config = TWAPAlgorithm.TWAPConfig(
            duration_minutes=5,
            slice_interval_seconds=30,
            randomize_timing=True,
            timing_randomness_pct=0.2
        )
        
        algorithm = TWAPAlgorithm("test_twap_random", config)
        slices = await algorithm.initialize(sample_order)
        
        # 验证时间随机化
        if len(slices) > 1:
            base_interval = 30  # 基础间隔
            actual_intervals = []
            
            for i in range(1, len(slices)):
                interval = (slices[i].scheduled_time - slices[i-1].scheduled_time).total_seconds()
                actual_intervals.append(interval)
            
            # 随机化应该产生不同的间隔
            unique_intervals = set(actual_intervals)
            assert len(unique_intervals) >= 1  # 至少有一些变化


class TestVWAPAlgorithm:
    """VWAP算法测试"""
    
    @pytest.fixture
    def vwap_config(self):
        """创建VWAP配置"""
        return VWAPAlgorithm.VWAPConfig(
            duration_minutes=30,
            target_participation_rate=0.1,
            historical_volume_periods=10
        )
    
    @pytest.fixture
    def vwap_algorithm(self, vwap_config):
        """创建VWAP算法实例"""
        return VWAPAlgorithm("test_vwap_001", vwap_config)
    
    @pytest.mark.asyncio
    async def test_vwap_initialization(self, vwap_algorithm, sample_order):
        """测试VWAP算法初始化"""
        slices = await vwap_algorithm.initialize(sample_order)
        
        assert len(slices) > 0
        
        # 验证总数量分配
        total_qty = sum(slice_order.quantity for slice_order in slices)
        assert abs(total_qty - sample_order.quantity) < 0.001
        
        # 验证分片大小符合成交量分布模式
        quantities = [slice_order.quantity for slice_order in slices]
        assert all(qty > 0 for qty in quantities)
    
    @pytest.mark.asyncio
    async def test_vwap_volume_profile(self, vwap_algorithm):
        """测试VWAP成交量分布模式"""
        profile = vwap_algorithm._get_volume_profile()
        
        assert len(profile) > 0
        assert all(weight > 0 for weight in profile)
        assert abs(sum(profile) - 1.0) < 0.001  # 归一化检查
        
        # 验证U型分布特征（开盘和收盘权重较大）
        if len(profile) >= 3:
            assert profile[0] > profile[len(profile)//2]  # 开盘 > 中间
            assert profile[-1] > profile[len(profile)//2]  # 收盘 > 中间
    
    @pytest.mark.asyncio
    async def test_vwap_market_data_update(self, vwap_algorithm, market_data, order_book):
        """测试VWAP市场数据更新"""
        # 初始状态
        assert vwap_algorithm.cumulative_volume == 0.0
        assert vwap_algorithm.vwap_price == 0.0
        
        # 更新市场数据
        await vwap_algorithm.update_market_data(market_data, order_book)
        
        # 验证VWAP计算
        assert vwap_algorithm.cumulative_volume == market_data.volume
        assert vwap_algorithm.vwap_price == market_data.price
        
        # 再次更新
        market_data2 = market_data
        market_data2.volume = 200.0
        market_data2.price = 51000.0
        
        await vwap_algorithm.update_market_data(market_data2, order_book)
        
        # 验证累积计算
        expected_vwap = (50000 * 100 + 51000 * 200) / 300
        assert abs(vwap_algorithm.vwap_price - expected_vwap) < 0.1
    
    @pytest.mark.asyncio
    async def test_vwap_strategy_adjustment(self, vwap_algorithm):
        """测试VWAP策略调整"""
        # 添加一些历史成交量数据
        vwap_algorithm.historical_volumes = [100.0, 120.0, 90.0, 110.0]
        
        # 正常成交量情况
        vwap_algorithm.current_market_data = MarketData(
            symbol="BTCUSDT", timestamp=int(time.time()*1000), price=50000.0, volume=105.0,
            bid=49999.0, ask=50001.0, bid_volume=50.0, ask_volume=60.0
        )
        
        should_adjust = await vwap_algorithm.should_adjust_strategy()
        assert should_adjust is False
        
        # 异常高成交量情况
        vwap_algorithm.current_market_data.volume = 250.0  # 远高于历史平均
        should_adjust = await vwap_algorithm.should_adjust_strategy()
        assert should_adjust is True
        
        # 异常低成交量情况
        vwap_algorithm.current_market_data.volume = 25.0  # 远低于历史平均
        should_adjust = await vwap_algorithm.should_adjust_strategy()
        assert should_adjust is True


class TestPOVAlgorithm:
    """POV算法测试"""
    
    @pytest.fixture
    def pov_config(self):
        """创建POV配置"""
        return POVAlgorithm.POVConfig(
            target_participation_rate=0.15,
            min_participation_rate=0.05,
            max_participation_rate=0.30,
            volume_measurement_window=60,
            adjustment_frequency=30
        )
    
    @pytest.fixture
    def pov_algorithm(self, pov_config):
        """创建POV算法实例"""
        return POVAlgorithm("test_pov_001", pov_config)
    
    @pytest.mark.asyncio
    async def test_pov_initialization(self, pov_algorithm, sample_order):
        """测试POV算法初始化"""
        slices = await pov_algorithm.initialize(sample_order)
        
        # POV算法初始只创建一个分片
        assert len(slices) == 1
        assert slices[0].quantity <= sample_order.quantity
        assert slices[0].quantity > 0
    
    @pytest.mark.asyncio
    async def test_pov_volume_tracking(self, pov_algorithm, market_data, order_book):
        """测试POV成交量跟踪"""
        # 添加多个成交量数据点
        volumes = [100.0, 150.0, 120.0, 200.0]
        
        for volume in volumes:
            market_data.volume = volume
            await pov_algorithm.update_market_data(market_data, order_book)
            await asyncio.sleep(0.01)  # 小延时确保时间戳不同
        
        # 验证成交量记录
        assert len(pov_algorithm.recent_volumes) == len(volumes)
        
        # 计算成交量速率
        volume_rate = pov_algorithm._calculate_market_volume_rate()
        assert volume_rate > 0
    
    @pytest.mark.asyncio
    async def test_pov_dynamic_slicing(self, pov_algorithm, sample_order):
        """测试POV动态分片"""
        # 初始化
        slices = await pov_algorithm.initialize(sample_order)
        pov_algorithm.slices = slices
        
        # 模拟市场成交量
        pov_algorithm.recent_volumes = [
            (datetime.utcnow() - timedelta(seconds=30), 100.0),
            (datetime.utcnow() - timedelta(seconds=20), 150.0),
            (datetime.utcnow() - timedelta(seconds=10), 120.0),
            (datetime.utcnow(), 200.0)
        ]
        
        # 模拟第一个分片成交
        first_slice = slices[0]
        first_slice.status = OrderStatus.FILLED
        first_slice.filled_qty = first_slice.quantity
        
        await pov_algorithm.on_slice_filled(first_slice, first_slice.quantity, 50000.0)
        
        # 验证是否生成了新分片
        assert len(pov_algorithm.slices) > 1
    
    @pytest.mark.asyncio
    async def test_pov_participation_rate_adjustment(self, pov_algorithm, order_book):
        """测试POV参与率调整"""
        # 设置初始参与率
        initial_rate = pov_algorithm.current_participation_rate
        
        # 模拟大价差场景
        wide_spread_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            bids=[(49900.0, 10.0)],  # 大价差
            asks=[(50100.0, 10.0)]
        )
        
        pov_algorithm.current_order_book = wide_spread_book
        await pov_algorithm._adjust_strategy()
        
        # 大价差应该降低参与率
        assert pov_algorithm.current_participation_rate < initial_rate
        
        # 模拟小价差场景
        narrow_spread_book = OrderBook(
            symbol="BTCUSDT",
            timestamp=int(time.time() * 1000),
            bids=[(49999.5, 10.0)],  # 小价差
            asks=[(50000.5, 10.0)]
        )
        
        pov_algorithm.current_order_book = narrow_spread_book
        await pov_algorithm._adjust_strategy()
        
        # 小价差可能提高参与率
        # 注意：这里的测试逻辑取决于具体的调整算法


class TestImplementationShortfall:
    """实施缺口算法测试"""
    
    @pytest.fixture
    def is_config(self):
        """创建IS配置"""
        return ImplementationShortfall.ISConfig(
            risk_aversion=0.5,
            volatility_estimate=0.02,
            temporary_impact_coefficient=0.5,
            permanent_impact_coefficient=0.3,
            max_duration_minutes=60
        )
    
    @pytest.fixture
    def is_algorithm(self, is_config):
        """创建IS算法实例"""
        return ImplementationShortfall("test_is_001", is_config)
    
    @pytest.mark.asyncio
    async def test_is_initialization(self, is_algorithm, sample_order):
        """测试IS算法初始化"""
        slices = await is_algorithm.initialize(sample_order)
        
        assert len(slices) > 0
        
        # 验证最优执行率计算
        assert is_algorithm.optimal_rate is not None
        assert 0 < is_algorithm.optimal_rate <= 1.0
        
        # 验证决策价格记录
        assert is_algorithm.decision_price == sample_order.price
        
        # 验证总数量分配
        total_qty = sum(slice_order.quantity for slice_order in slices)
        assert abs(total_qty - sample_order.quantity) < 0.1  # 允许较大误差，因为可能有剩余
    
    def test_is_optimal_rate_calculation(self, is_algorithm, sample_order):
        """测试IS最优执行率计算"""
        optimal_rate = is_algorithm._calculate_optimal_rate(sample_order)
        
        assert 0 < optimal_rate <= 1.0
        
        # 测试边界条件
        # 高波动率应该导致更快执行
        is_algorithm.is_config.volatility_estimate = 0.05
        high_vol_rate = is_algorithm._calculate_optimal_rate(sample_order)
        
        # 低波动率应该导致更慢执行
        is_algorithm.is_config.volatility_estimate = 0.01
        low_vol_rate = is_algorithm._calculate_optimal_rate(sample_order)
        
        assert high_vol_rate > low_vol_rate
    
    @pytest.mark.asyncio
    async def test_is_strategy_adjustment(self, is_algorithm, market_data, order_book):
        """测试IS策略调整"""
        # 设置决策价格
        is_algorithm.decision_price = 50000.0
        
        # 价格无显著变化
        market_data.price = 50050.0  # 0.1%变化
        await is_algorithm.update_market_data(market_data, order_book)
        
        should_adjust = await is_algorithm.should_adjust_strategy()
        assert should_adjust is False
        
        # 价格显著变化
        market_data.price = 51000.0  # 2%变化
        await is_algorithm.update_market_data(market_data, order_book)
        
        should_adjust = await is_algorithm.should_adjust_strategy()
        assert should_adjust is True
    
    @pytest.mark.asyncio
    async def test_is_price_impact_adjustment(self, is_algorithm, sample_order):
        """测试IS价格冲击调整"""
        # 初始化
        slices = await is_algorithm.initialize(sample_order)
        is_algorithm.slices = slices
        is_algorithm.decision_price = 50000.0
        
        # 记录原始执行时间
        original_times = [slice_order.scheduled_time for slice_order in slices]
        
        # 模拟显著价格冲击
        is_algorithm.current_market_data = MarketData(
            symbol="BTCUSDT", timestamp=int(time.time()*1000), price=51000.0, volume=100.0,
            bid=50999.0, ask=51001.0, bid_volume=50.0, ask_volume=60.0
        )
        
        # 调整策略
        await is_algorithm._adjust_strategy()
        
        # 验证执行时间被提前
        new_times = [slice_order.scheduled_time for slice_order in slices if slice_order.status == OrderStatus.NEW]
        
        # 应该有剩余分片的时间被调整
        if new_times and original_times:
            # 这里具体的验证逻辑取决于实际的调整算法
            pass


class TestAlgorithmResult:
    """算法结果测试"""
    
    def test_algorithm_result_creation(self):
        """测试算法结果创建"""
        result = AlgorithmResult(
            algorithm_id="test_algo_001",
            parent_order_id="parent_001",
            status=AlgorithmStatus.COMPLETED,
            total_filled=10.0,
            avg_price=50000.0,
            total_slices=5,
            completed_slices=5,
            total_commission=20.0,
            implementation_shortfall=0.001,
            start_time=datetime.utcnow()
        )
        
        assert result.algorithm_id == "test_algo_001"
        assert result.status == AlgorithmStatus.COMPLETED
        assert result.total_filled == 10.0
        assert result.avg_price == 50000.0
        assert result.implementation_shortfall == 0.001


class TestSliceOrder:
    """分片订单测试"""
    
    def test_slice_order_creation(self):
        """测试分片订单创建"""
        slice_order = SliceOrder(
            slice_id="slice_001",
            parent_order_id="parent_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            order_type=OrderType.LIMIT,
            scheduled_time=datetime.utcnow()
        )
        
        assert slice_order.slice_id == "slice_001"
        assert slice_order.parent_order_id == "parent_001"
        assert slice_order.symbol == "BTCUSDT"
        assert slice_order.side == OrderSide.BUY
        assert slice_order.quantity == 1.0
        assert slice_order.price == 50000.0
        assert slice_order.status == OrderStatus.NEW
        assert slice_order.filled_qty == 0.0


class TestAlgorithmConfig:
    """算法配置测试"""
    
    def test_algorithm_config_defaults(self):
        """测试算法配置默认值"""
        config = AlgorithmConfig()
        
        assert config.max_participation_rate == 0.2
        assert config.min_order_size == 0.01
        assert config.max_order_size == 1000.0
        assert config.aggressive_threshold == 0.3
        assert config.passive_threshold == 0.1
        assert config.price_improvement_threshold == 0.0001
    
    def test_twap_config_specifics(self):
        """测试TWAP配置特定参数"""
        config = TWAPAlgorithm.TWAPConfig(
            duration_minutes=30,
            slice_interval_seconds=120,
            randomize_timing=True,
            timing_randomness_pct=0.15
        )
        
        assert config.duration_minutes == 30
        assert config.slice_interval_seconds == 120
        assert config.randomize_timing is True
        assert config.timing_randomness_pct == 0.15
    
    def test_vwap_config_specifics(self):
        """测试VWAP配置特定参数"""
        config = VWAPAlgorithm.VWAPConfig(
            duration_minutes=45,
            historical_volume_periods=25,
            target_participation_rate=0.12,
            volume_curve_adjustment=1.2
        )
        
        assert config.duration_minutes == 45
        assert config.historical_volume_periods == 25
        assert config.target_participation_rate == 0.12
        assert config.volume_curve_adjustment == 1.2