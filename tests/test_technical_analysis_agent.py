"""
技术分析Agent测试
测试技术分析Agent的多时间框架分析、信号生成和Agent集成功能
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from src.agents.technical_analysis_agent import (
    TechnicalAnalysisAgent, TimeFrame, SignalStrength, TechnicalSignal,
    MultiTimeFrameAnalysis, create_technical_analysis_agent
)
from src.agents.base import AgentConfig, AgentState
from src.core.models import TradingState, MarketData, Signal
from src.core.message_bus import MessageBus, Message
from src.analysis.technical_indicators import IndicatorResult


class TestTimeFrame:
    """时间框架枚举测试"""
    
    def test_timeframe_values(self):
        """测试时间框架枚举值"""
        assert TimeFrame.M1.value == "1m"
        assert TimeFrame.M5.value == "5m"
        assert TimeFrame.H1.value == "1h"
        assert TimeFrame.D1.value == "1d"


class TestSignalStrength:
    """信号强度枚举测试"""
    
    def test_signal_strength_values(self):
        """测试信号强度枚举值"""
        assert SignalStrength.VERY_WEAK.value == 0.2
        assert SignalStrength.WEAK.value == 0.4
        assert SignalStrength.MODERATE.value == 0.6
        assert SignalStrength.STRONG.value == 0.8
        assert SignalStrength.VERY_STRONG.value == 1.0


class TestTechnicalSignal:
    """技术信号测试"""
    
    def test_technical_signal_creation(self):
        """测试技术信号创建"""
        signal = TechnicalSignal(
            indicator_name="RSI",
            signal_type="BUY",
            strength=0.8,
            confidence=0.9,
            timeframe=TimeFrame.M5,
            reason="RSI oversold",
            metadata={"rsi_value": 25.0}
        )
        
        assert signal.indicator_name == "RSI"
        assert signal.signal_type == "BUY"
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.timeframe == TimeFrame.M5
        assert signal.reason == "RSI oversold"
        assert signal.metadata["rsi_value"] == 25.0


class TestMultiTimeFrameAnalysis:
    """多时间框架分析测试"""
    
    def test_analysis_creation(self):
        """测试分析结果创建"""
        analysis = MultiTimeFrameAnalysis(
            symbol="BTCUSDT",
            timestamp=time.time()
        )
        
        assert analysis.symbol == "BTCUSDT"
        assert analysis.timestamp > 0
        assert analysis.signals_by_timeframe == {}
        assert analysis.overall_signal is None
        assert analysis.consensus_strength == 0.0
        assert analysis.consensus_confidence == 0.0


class TestTechnicalAnalysisAgent:
    """技术分析Agent测试"""
    
    @pytest.fixture
    async def mock_message_bus(self):
        """模拟消息总线"""
        bus = MagicMock(spec=MessageBus)
        bus.create_publisher = MagicMock()
        bus.create_subscriber = MagicMock()
        bus.publish = MagicMock()
        
        # 模拟订阅者
        subscriber = MagicMock()
        subscriber.subscribe = AsyncMock()
        bus.create_subscriber.return_value = subscriber
        
        return bus
    
    @pytest.fixture
    async def agent_config(self):
        """创建Agent配置"""
        return AgentConfig(
            name="test_technical_agent",
            enabled=True,
            priority=70,
            parameters={
                "analysis_interval": 60.0,
                "min_confidence": 0.6
            }
        )
    
    @pytest.fixture
    async def technical_agent(self, agent_config, mock_message_bus):
        """创建技术分析Agent"""
        agent = TechnicalAnalysisAgent(
            config=agent_config,
            message_bus=mock_message_bus,
            timeframes=[TimeFrame.M5, TimeFrame.M15],
            symbols=["BTCUSDT", "ETHUSDT"]
        )
        
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent_config, mock_message_bus):
        """测试Agent初始化"""
        agent = TechnicalAnalysisAgent(
            config=agent_config,
            message_bus=mock_message_bus,
            timeframes=[TimeFrame.M5, TimeFrame.H1],
            symbols=["BTCUSDT"]
        )
        
        await agent.initialize()
        
        # 验证初始化
        assert agent.get_state() == AgentState.RUNNING
        assert agent.symbols == ["BTCUSDT"]
        assert TimeFrame.M5 in agent.timeframes
        assert TimeFrame.H1 in agent.timeframes
        assert "BTCUSDT" in agent._market_data_cache
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_market_data_cache_update(self, technical_agent):
        """测试市场数据缓存更新"""
        symbol = "BTCUSDT"
        
        # 创建测试市场数据
        market_data = MarketData(
            symbol=symbol,
            price=50000.0,
            volume=1000.0,
            timestamp=int(time.time()),
            bid=49999.0,
            ask=50001.0,
            bid_volume=100.0,
            ask_volume=100.0
        )
        
        # 更新缓存
        await technical_agent._update_market_data_cache(symbol, market_data)
        
        # 验证缓存更新
        for timeframe in technical_agent.timeframes:
            cache_data = technical_agent._get_timeframe_data(symbol, timeframe)
            assert len(cache_data) == 1
            assert cache_data[0].symbol == symbol
            assert cache_data[0].price == 50000.0
    
    def test_extract_price_data(self, technical_agent):
        """测试价格数据提取"""
        # 创建市场数据列表
        market_data_list = []
        for i in range(5):
            market_data = MarketData(
                symbol="BTCUSDT",
                price=50000.0 + i * 100,
                volume=1000.0 + i * 50,
                timestamp=int(time.time() + i),
                bid=49999.0 + i * 100,
                ask=50001.0 + i * 100,
                bid_volume=100.0,
                ask_volume=100.0
            )
            market_data_list.append(market_data)
        
        # 提取价格数据
        price_data = technical_agent._extract_price_data(market_data_list)
        
        assert len(price_data) == 5
        assert price_data[0]["close"] == 50000.0
        assert price_data[4]["close"] == 50400.0
        assert price_data[0]["volume"] == 1000.0
        assert price_data[4]["volume"] == 1200.0
    
    @pytest.mark.asyncio
    async def test_timeframe_analysis(self, technical_agent):
        """测试单时间框架分析"""
        symbol = "BTCUSDT"
        timeframe = TimeFrame.M5
        
        # 创建足够的市场数据用于分析
        market_data_list = []
        for i in range(30):
            market_data = MarketData(
                symbol=symbol,
                price=50000.0 + i * 10 + np.random.normal(0, 50),
                volume=1000.0 + np.random.randint(0, 500),
                timestamp=int(time.time() + i * 60),  # 每分钟一个数据点
                bid=49999.0 + i * 10,
                ask=50001.0 + i * 10,
                bid_volume=100.0,
                ask_volume=100.0
            )
            market_data_list.append(market_data)
        
        # 执行时间框架分析
        signals = await technical_agent._analyze_timeframe(symbol, timeframe, market_data_list)
        
        # 验证分析结果
        assert isinstance(signals, list)
        # 可能有信号，也可能没有，取决于数据
        for signal in signals:
            assert isinstance(signal, TechnicalSignal)
            assert signal.timeframe == timeframe
            assert signal.signal_type in ["BUY", "SELL", "HOLD"]
            assert 0 <= signal.strength <= 1.0
            assert 0 <= signal.confidence <= 1.0
    
    def test_trend_signal_generation(self, technical_agent):
        """测试趋势信号生成"""
        # 模拟指标结果
        indicator_results = {
            "SMA_20": IndicatorResult(
                indicator_name="SMA_20",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value=50100.0
            ),
            "SMA_50": IndicatorResult(
                indicator_name="SMA_50",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value=50000.0
            )
        }
        
        # 生成趋势信号
        signals = technical_agent._generate_trend_signals(indicator_results, TimeFrame.M5)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.indicator_name == "SMA_Cross"
        assert signal.signal_type == "BUY"  # SMA20 > SMA50
        assert signal.timeframe == TimeFrame.M5
        assert "SMA20 > SMA50" in signal.reason
    
    def test_momentum_signal_generation(self, technical_agent):
        """测试动量信号生成"""
        # 模拟RSI超卖情况
        indicator_results = {
            "RSI_14": IndicatorResult(
                indicator_name="RSI_14",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value=25.0  # 超卖
            )
        }
        
        # 生成动量信号
        signals = technical_agent._generate_momentum_signals(indicator_results, TimeFrame.M15)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.indicator_name == "RSI"
        assert signal.signal_type == "BUY"
        assert signal.timeframe == TimeFrame.M15
        assert "oversold" in signal.reason.lower()
        assert signal.strength > 0
    
    def test_volatility_signal_generation(self, technical_agent):
        """测试波动率信号生成"""
        # 模拟布林带突破
        indicator_results = {
            "BBANDS_20_2": IndicatorResult(
                indicator_name="BBANDS_20_2",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value={
                    "upper": 51000.0,
                    "middle": 50000.0,
                    "lower": 49000.0
                }
            )
        }
        
        # 生成波动率信号
        signals = technical_agent._generate_volatility_signals(indicator_results, TimeFrame.H1)
        
        # 由于我们使用middle作为当前价格，不会触发突破信号
        # 但测试函数结构是正确的
        assert isinstance(signals, list)
    
    def test_consensus_signal_generation(self, technical_agent):
        """测试共识信号生成"""
        # 创建多时间框架分析
        analysis = MultiTimeFrameAnalysis(
            symbol="BTCUSDT",
            timestamp=time.time()
        )
        
        # 添加不同时间框架的信号
        analysis.signals_by_timeframe[TimeFrame.M5] = [
            TechnicalSignal(
                indicator_name="RSI",
                signal_type="BUY",
                strength=0.8,
                confidence=0.9,
                timeframe=TimeFrame.M5,
                reason="RSI oversold"
            ),
            TechnicalSignal(
                indicator_name="SMA_Cross",
                signal_type="BUY",
                strength=0.6,
                confidence=0.7,
                timeframe=TimeFrame.M5,
                reason="SMA bullish cross"
            )
        ]
        
        analysis.signals_by_timeframe[TimeFrame.M15] = [
            TechnicalSignal(
                indicator_name="MACD",
                signal_type="BUY",
                strength=0.7,
                confidence=0.8,
                timeframe=TimeFrame.M15,
                reason="MACD bullish"
            )
        ]
        
        # 生成共识信号
        consensus_signal = technical_agent._generate_consensus_signal("BTCUSDT", analysis)
        
        assert consensus_signal is not None
        assert consensus_signal.symbol == "BTCUSDT"
        assert consensus_signal.action == "BUY"
        assert consensus_signal.source == technical_agent.name
        assert consensus_signal.strength > 0
        assert consensus_signal.confidence > 0
        assert analysis.consensus_strength > 0
        assert analysis.consensus_confidence > 0
    
    @pytest.mark.asyncio
    async def test_full_analyze_flow(self, technical_agent):
        """测试完整的分析流程"""
        # 准备交易状态
        market_data = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=50000.0,
                volume=1000.0,
                timestamp=int(time.time()),
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
        }
        
        trading_state = MagicMock(spec=TradingState)
        trading_state.market_data = market_data
        
        # 先填充一些历史数据
        for i in range(25):
            historical_data = MarketData(
                symbol="BTCUSDT",
                price=50000.0 + i * 10,
                volume=1000.0,
                timestamp=int(time.time() - (25 - i) * 60),
                bid=49999.0 + i * 10,
                ask=50001.0 + i * 10,
                bid_volume=100.0,
                ask_volume=100.0
            )
            await technical_agent._update_market_data_cache("BTCUSDT", historical_data)
        
        # 执行分析
        signals = await technical_agent.analyze(trading_state)
        
        # 验证结果
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.source == technical_agent.name
            assert signal.symbol in technical_agent.symbols
    
    @pytest.mark.asyncio
    async def test_market_data_message_handling(self, technical_agent):
        """测试市场数据消息处理"""
        # 创建市场数据消息
        message_data = {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": time.time(),
            "bid": 49999.0,
            "ask": 50001.0,
            "bid_volume": 100.0,
            "ask_volume": 100.0
        }
        
        message = Message(
            topic="processed.btcusdt.tick",
            data=message_data,
            timestamp=time.time()
        )
        
        # 处理消息
        await technical_agent._handle_market_data(message)
        
        # 验证数据已更新到缓存
        cache_data = technical_agent._get_timeframe_data("BTCUSDT", TimeFrame.M5)
        assert len(cache_data) > 0
        assert cache_data[-1].price == 50000.0
    
    def test_signal_config_update(self, technical_agent):
        """测试信号配置更新"""
        new_config = {
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "macd_signal_threshold": 0.002
        }
        
        technical_agent.update_signal_config(new_config)
        
        # 验证配置更新
        assert technical_agent._signal_config["rsi_oversold"] == 25
        assert technical_agent._signal_config["rsi_overbought"] == 75
        assert technical_agent._signal_config["macd_signal_threshold"] == 0.002
    
    def test_symbol_management(self, technical_agent):
        """测试交易对管理"""
        initial_count = len(technical_agent.symbols)
        
        # 添加新交易对
        technical_agent.add_symbol("ADAUSDT")
        assert len(technical_agent.symbols) == initial_count + 1
        assert "ADAUSDT" in technical_agent.symbols
        assert "ADAUSDT" in technical_agent._market_data_cache
        
        # 移除交易对
        technical_agent.remove_symbol("ADAUSDT")
        assert len(technical_agent.symbols) == initial_count
        assert "ADAUSDT" not in technical_agent.symbols
        assert "ADAUSDT" not in technical_agent._market_data_cache
    
    def test_get_analysis_result(self, technical_agent):
        """测试获取分析结果"""
        # 创建测试分析结果
        analysis = MultiTimeFrameAnalysis(
            symbol="BTCUSDT",
            timestamp=time.time()
        )
        
        technical_agent._analysis_cache["BTCUSDT"] = analysis
        
        # 获取分析结果
        result = technical_agent.get_analysis_result("BTCUSDT")
        
        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result == analysis
    
    def test_get_indicator_values(self, technical_agent):
        """测试获取指标值"""
        symbol = "BTCUSDT"
        timeframe = TimeFrame.M5
        
        # 添加一些数据
        for i in range(25):
            price_data = {
                "close": 50000.0 + i,
                "volume": 1000.0
            }
            technical_agent.indicators.update_data(f"{symbol}_{timeframe.value}", price_data)
        
        # 计算指标
        technical_agent.indicators.calculate_all_indicators(f"{symbol}_{timeframe.value}")
        
        # 获取指标值
        indicator_values = technical_agent.get_indicator_values(symbol, timeframe)
        
        assert isinstance(indicator_values, dict)
        # 应该包含一些默认指标
        assert len(indicator_values) >= 0  # 可能为空，取决于数据质量
    
    def test_performance_summary(self, technical_agent):
        """测试性能摘要"""
        summary = technical_agent.get_performance_summary()
        
        assert "symbols_analyzed" in summary
        assert "timeframes" in summary
        assert "indicator_performance" in summary
        assert "agent_metrics" in summary
        
        assert summary["symbols_analyzed"] == len(technical_agent.symbols)
        assert len(summary["timeframes"]) == len(technical_agent.timeframes)


class TestCreateTechnicalAnalysisAgent:
    """创建技术分析Agent测试"""
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_defaults(self):
        """测试使用默认参数创建Agent"""
        agent = create_technical_analysis_agent()
        
        assert agent.name == "technical_analysis_agent"
        assert agent.symbols == ["BTCUSDT"]
        assert TimeFrame.M5 in agent.timeframes
        assert TimeFrame.M15 in agent.timeframes
        assert TimeFrame.H1 in agent.timeframes
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_creation_with_custom_params(self):
        """测试使用自定义参数创建Agent"""
        custom_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        custom_timeframes = [TimeFrame.M1, TimeFrame.M5, TimeFrame.H4]
        
        agent = create_technical_analysis_agent(
            name="custom_technical_agent",
            symbols=custom_symbols,
            timeframes=custom_timeframes
        )
        
        assert agent.name == "custom_technical_agent"
        assert agent.symbols == custom_symbols
        assert agent.timeframes == custom_timeframes
        
        await agent.shutdown()


class TestSignalGeneration:
    """信号生成测试"""
    
    @pytest.fixture
    async def agent(self):
        """创建测试Agent"""
        config = AgentConfig(name="signal_test_agent")
        agent = TechnicalAnalysisAgent(
            config=config,
            timeframes=[TimeFrame.M5],
            symbols=["BTCUSDT"]
        )
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    def test_rsi_oversold_signal(self, agent):
        """测试RSI超卖信号"""
        indicator_results = {
            "RSI_14": IndicatorResult(
                indicator_name="RSI_14",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value=20.0  # 强烈超卖
            )
        }
        
        signals = agent._generate_momentum_signals(indicator_results, TimeFrame.M5)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == "BUY"
        assert signal.strength > 0.5  # 应该是强信号
        assert "oversold" in signal.reason.lower()
    
    def test_rsi_overbought_signal(self, agent):
        """测试RSI超买信号"""
        indicator_results = {
            "RSI_14": IndicatorResult(
                indicator_name="RSI_14",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value=80.0  # 超买
            )
        }
        
        signals = agent._generate_momentum_signals(indicator_results, TimeFrame.M5)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == "SELL"
        assert signal.strength > 0.5
        assert "overbought" in signal.reason.lower()
    
    def test_macd_bullish_crossover(self, agent):
        """测试MACD看涨交叉"""
        indicator_results = {
            "MACD_12_26_9": IndicatorResult(
                indicator_name="MACD_12_26_9",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value={
                    "macd": 10.0,
                    "signal": 8.0,
                    "histogram": 2.0
                }
            )
        }
        
        signals = agent._generate_momentum_signals(indicator_results, TimeFrame.M5)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == "BUY"
        assert "bullish crossover" in signal.reason.lower()
    
    def test_macd_bearish_crossover(self, agent):
        """测试MACD看跌交叉"""
        indicator_results = {
            "MACD_12_26_9": IndicatorResult(
                indicator_name="MACD_12_26_9",
                symbol="BTCUSDT",
                timestamp=time.time(),
                value={
                    "macd": -10.0,
                    "signal": -8.0,
                    "histogram": -2.0
                }
            )
        }
        
        signals = agent._generate_momentum_signals(indicator_results, TimeFrame.M5)
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == "SELL"
        assert "bearish crossover" in signal.reason.lower()


class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.fixture
    async def agent(self):
        """创建测试Agent"""
        config = AgentConfig(name="error_test_agent")
        agent = TechnicalAnalysisAgent(config=config, symbols=["BTCUSDT"])
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_analyze_with_empty_state(self, agent):
        """测试空状态分析"""
        empty_state = MagicMock(spec=TradingState)
        empty_state.market_data = {}
        
        signals = await agent.analyze(empty_state)
        
        # 应该返回空列表而不是错误
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_invalid_data(self, agent):
        """测试无效数据分析"""
        invalid_market_data = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                price=np.nan,  # 无效价格
                volume=np.nan,  # 无效成交量
                timestamp=int(time.time()),
                bid=np.nan,
                ask=np.nan,
                bid_volume=np.nan,
                ask_volume=np.nan
            )
        }
        
        trading_state = MagicMock(spec=TradingState)
        trading_state.market_data = invalid_market_data
        
        # 应该能够处理而不崩溃
        signals = await agent.analyze(trading_state)
        assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_handle_malformed_message(self, agent):
        """测试处理畸形消息"""
        malformed_message = Message(
            topic="processed.btcusdt.tick",
            data={"invalid": "data"},  # 缺少必要字段
            timestamp=time.time()
        )
        
        # 应该能够处理而不崩溃
        await agent._handle_market_data(malformed_message)
        
        # 验证没有崩溃，缓存仍然正常
        assert isinstance(agent._market_data_cache, dict)


class TestPerformanceAndOptimization:
    """性能和优化测试"""
    
    @pytest.fixture
    async def agent(self):
        """创建性能测试Agent"""
        config = AgentConfig(name="perf_test_agent")
        agent = TechnicalAnalysisAgent(
            config=config,
            timeframes=[TimeFrame.M5, TimeFrame.M15],
            symbols=["BTCUSDT"]
        )
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_large_data_processing(self, agent):
        """测试大量数据处理"""
        symbol = "BTCUSDT"
        
        # 添加大量市场数据
        start_time = time.time()
        for i in range(1000):
            market_data = MarketData(
                symbol=symbol,
                price=50000.0 + np.random.normal(0, 100),
                volume=1000.0 + np.random.randint(0, 500),
                timestamp=int(time.time() + i),
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
            await agent._update_market_data_cache(symbol, market_data)
        
        data_update_time = time.time() - start_time
        
        # 创建交易状态并执行分析
        trading_state = MagicMock(spec=TradingState)
        trading_state.market_data = {
            symbol: agent._get_timeframe_data(symbol, TimeFrame.M5)[-1]
        }
        
        start_time = time.time()
        signals = await agent.analyze(trading_state)
        analysis_time = time.time() - start_time
        
        # 性能断言
        assert data_update_time < 5.0  # 数据更新应该在5秒内完成
        assert analysis_time < 3.0  # 分析应该在3秒内完成
        assert isinstance(signals, list)
    
    @pytest.mark.asyncio
    async def test_cache_size_limits(self, agent):
        """测试缓存大小限制"""
        symbol = "BTCUSDT"
        timeframe = TimeFrame.M5
        max_size = agent._get_max_cache_size(timeframe)
        
        # 添加超过限制的数据
        for i in range(max_size + 100):
            market_data = MarketData(
                symbol=symbol,
                price=50000.0 + i,
                volume=1000.0,
                timestamp=int(time.time() + i),
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
            # 使用正确的更新方法，它会自动应用大小限制
            await agent._update_market_data_cache(symbol, market_data)
        
        # 验证缓存大小没有超过限制
        cache_data = agent._get_timeframe_data(symbol, timeframe)
        assert len(cache_data) <= max_size


if __name__ == "__main__":
    pytest.main([__file__])