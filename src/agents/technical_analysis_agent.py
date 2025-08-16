"""
技术分析Agent实现
基于Agent基类实现技术分析功能，生成技术信号和强度评估，支持多时间框架分析
"""

import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from src.agents.base import BaseAgent, AgentConfig, AnalysisAgent
from src.analysis.technical_indicators import TechnicalIndicators, IndicatorResult, IndicatorType
from src.core.models import Signal, TradingState, MarketData
from src.core.message_bus import MessageBus, Message, MessagePriority
from src.utils.logger import LoggerMixin


class TimeFrame(Enum):
    """时间框架枚举"""
    M1 = "1m"      # 1分钟
    M5 = "5m"      # 5分钟
    M15 = "15m"    # 15分钟
    M30 = "30m"    # 30分钟
    H1 = "1h"      # 1小时
    H4 = "4h"      # 4小时
    D1 = "1d"      # 1天


class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


@dataclass
class TechnicalSignal:
    """技术分析信号"""
    indicator_name: str
    signal_type: str  # "BUY", "SELL", "HOLD"
    strength: float   # 0-1之间
    confidence: float # 0-1之间
    timeframe: TimeFrame
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiTimeFrameAnalysis:
    """多时间框架分析结果"""
    symbol: str
    timestamp: float
    signals_by_timeframe: Dict[TimeFrame, List[TechnicalSignal]] = field(default_factory=dict)
    overall_signal: Optional[Signal] = None
    consensus_strength: float = 0.0
    consensus_confidence: float = 0.0


class TechnicalAnalysisAgent(AnalysisAgent):
    """技术分析Agent"""
    
    def __init__(
        self,
        config: AgentConfig,
        message_bus: Optional[MessageBus] = None,
        timeframes: List[TimeFrame] = None,
        symbols: List[str] = None
    ):
        super().__init__(config, message_bus)
        
        # 时间框架配置
        self.timeframes = timeframes or [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1]
        self.symbols = symbols or ["BTCUSDT"]
        
        # 技术指标计算器
        self.indicators = TechnicalIndicators()
        
        # 多时间框架数据缓存
        self._market_data_cache: Dict[str, Dict[TimeFrame, List[MarketData]]] = {}
        
        # 分析结果缓存
        self._analysis_cache: Dict[str, MultiTimeFrameAnalysis] = {}
        
        # 信号生成配置
        self._signal_config = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_signal_threshold": 0.001,
            "bb_squeeze_threshold": 0.02,
            "volume_spike_threshold": 2.0,
            "trend_confirmation_periods": 3
        }
        
        # 权重配置（不同时间框架的权重）
        self._timeframe_weights = {
            TimeFrame.M1: 0.1,
            TimeFrame.M5: 0.2,
            TimeFrame.M15: 0.3,
            TimeFrame.M30: 0.4,
            TimeFrame.H1: 0.6,
            TimeFrame.H4: 0.8,
            TimeFrame.D1: 1.0
        }
    
    async def _initialize(self):
        """Agent特定初始化"""
        # 初始化数据缓存
        for symbol in self.symbols:
            self._market_data_cache[symbol] = {tf: [] for tf in self.timeframes}
        
        # 订阅市场数据
        if self.message_bus:
            for symbol in self.symbols:
                await self.subscriber.subscribe(
                    f"processed.{symbol.lower()}.tick",
                    self._handle_market_data
                )
        
        self.log_info(f"TechnicalAnalysisAgent initialized for symbols: {self.symbols}")
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """执行技术分析并生成信号"""
        signals = []
        
        for symbol in self.symbols:
            try:
                # 执行多时间框架分析
                analysis = await self._perform_multi_timeframe_analysis(symbol, state)
                
                if analysis and analysis.overall_signal:
                    signals.append(analysis.overall_signal)
                    
                    # 缓存分析结果
                    self._analysis_cache[symbol] = analysis
                
            except Exception as e:
                self.log_error(f"Error analyzing {symbol}: {e}")
                self.metrics.error_count += 1
        
        return signals
    
    async def _perform_multi_timeframe_analysis(
        self,
        symbol: str,
        state: TradingState
    ) -> Optional[MultiTimeFrameAnalysis]:
        """执行多时间框架技术分析"""
        analysis = MultiTimeFrameAnalysis(
            symbol=symbol,
            timestamp=time.time()
        )
        
        # 获取不同时间框架的市场数据
        timeframe_data = await self._prepare_timeframe_data(symbol, state)
        
        if not timeframe_data:
            return None
        
        # 对每个时间框架进行分析
        for timeframe, market_data in timeframe_data.items():
            signals = await self._analyze_timeframe(symbol, timeframe, market_data)
            if signals:
                analysis.signals_by_timeframe[timeframe] = signals
        
        # 生成综合信号
        overall_signal = self._generate_consensus_signal(symbol, analysis)
        analysis.overall_signal = overall_signal
        
        return analysis
    
    async def _prepare_timeframe_data(
        self,
        symbol: str,
        state: TradingState
    ) -> Dict[TimeFrame, List[MarketData]]:
        """准备不同时间框架的市场数据"""
        timeframe_data = {}
        
        # 获取当前市场数据
        current_market_data = state.market_data.get(symbol)
        if not current_market_data:
            return timeframe_data
        
        # 更新缓存数据
        await self._update_market_data_cache(symbol, current_market_data)
        
        # 为每个时间框架准备数据
        for timeframe in self.timeframes:
            data = self._get_timeframe_data(symbol, timeframe)
            if data and len(data) >= 20:  # 至少需要20个数据点
                timeframe_data[timeframe] = data
        
        return timeframe_data
    
    async def _update_market_data_cache(self, symbol: str, market_data: MarketData):
        """更新市场数据缓存"""
        if symbol not in self._market_data_cache:
            self._market_data_cache[symbol] = {tf: [] for tf in self.timeframes}
        
        # 简化实现：将当前数据添加到所有时间框架
        # 实际应用中需要根据时间框架进行数据聚合
        for timeframe in self.timeframes:
            cache = self._market_data_cache[symbol][timeframe]
            cache.append(market_data)
            
            # 保持缓存大小限制
            max_size = self._get_max_cache_size(timeframe)
            if len(cache) > max_size:
                cache.pop(0)
    
    def _get_timeframe_data(self, symbol: str, timeframe: TimeFrame) -> List[MarketData]:
        """获取指定时间框架的数据"""
        return self._market_data_cache.get(symbol, {}).get(timeframe, [])
    
    def _get_max_cache_size(self, timeframe: TimeFrame) -> int:
        """获取不同时间框架的最大缓存大小"""
        size_map = {
            TimeFrame.M1: 1000,
            TimeFrame.M5: 500,
            TimeFrame.M15: 200,
            TimeFrame.M30: 100,
            TimeFrame.H1: 100,
            TimeFrame.H4: 50,
            TimeFrame.D1: 30
        }
        return size_map.get(timeframe, 100)
    
    async def _analyze_timeframe(
        self,
        symbol: str,
        timeframe: TimeFrame,
        market_data: List[MarketData]
    ) -> List[TechnicalSignal]:
        """分析单个时间框架"""
        signals = []
        
        # 准备技术指标计算数据
        price_data = self._extract_price_data(market_data)
        
        # 更新指标计算器的数据
        for i, data_point in enumerate(price_data):
            self.indicators.update_data(f"{symbol}_{timeframe.value}", data_point)
        
        # 计算所有技术指标
        indicator_results = self.indicators.calculate_all_indicators(
            f"{symbol}_{timeframe.value}"
        )
        
        # 基于指标结果生成信号
        signals.extend(self._generate_trend_signals(indicator_results, timeframe))
        signals.extend(self._generate_momentum_signals(indicator_results, timeframe))
        signals.extend(self._generate_volatility_signals(indicator_results, timeframe))
        
        return signals
    
    def _extract_price_data(self, market_data: List[MarketData]) -> List[Dict[str, float]]:
        """从市场数据提取价格数据"""
        price_data = []
        
        for data in market_data:
            price_point = {
                "open": data.price,  # 简化处理，实际应有OHLCV数据
                "high": data.price,
                "low": data.price,
                "close": data.price,
                "volume": data.volume,
                "timestamp": data.timestamp
            }
            price_data.append(price_point)
        
        return price_data
    
    def _generate_trend_signals(
        self,
        indicator_results: Dict[str, IndicatorResult],
        timeframe: TimeFrame
    ) -> List[TechnicalSignal]:
        """生成趋势信号"""
        signals = []
        
        # SMA交叉信号
        sma20 = indicator_results.get("SMA_20")
        sma50 = indicator_results.get("SMA_50")
        
        if sma20 and sma50 and sma20.is_valid and sma50.is_valid:
            if sma20.value > sma50.value:
                signals.append(TechnicalSignal(
                    indicator_name="SMA_Cross",
                    signal_type="BUY",
                    strength=0.6,
                    confidence=0.7,
                    timeframe=timeframe,
                    reason="SMA20 > SMA50 indicating uptrend",
                    metadata={"sma20": sma20.value, "sma50": sma50.value}
                ))
            else:
                signals.append(TechnicalSignal(
                    indicator_name="SMA_Cross",
                    signal_type="SELL",
                    strength=0.6,
                    confidence=0.7,
                    timeframe=timeframe,
                    reason="SMA20 < SMA50 indicating downtrend",
                    metadata={"sma20": sma20.value, "sma50": sma50.value}
                ))
        
        return signals
    
    def _generate_momentum_signals(
        self,
        indicator_results: Dict[str, IndicatorResult],
        timeframe: TimeFrame
    ) -> List[TechnicalSignal]:
        """生成动量信号"""
        signals = []
        
        # RSI信号
        rsi = indicator_results.get("RSI_14")
        if rsi and rsi.is_valid:
            rsi_value = rsi.value
            
            if rsi_value < self._signal_config["rsi_oversold"]:
                signals.append(TechnicalSignal(
                    indicator_name="RSI",
                    signal_type="BUY",
                    strength=min((self._signal_config["rsi_oversold"] - rsi_value) / 10, 1.0),
                    confidence=0.8,
                    timeframe=timeframe,
                    reason=f"RSI oversold at {rsi_value:.2f}",
                    metadata={"rsi": rsi_value}
                ))
            
            elif rsi_value > self._signal_config["rsi_overbought"]:
                signals.append(TechnicalSignal(
                    indicator_name="RSI",
                    signal_type="SELL",
                    strength=min((rsi_value - self._signal_config["rsi_overbought"]) / 10, 1.0),
                    confidence=0.8,
                    timeframe=timeframe,
                    reason=f"RSI overbought at {rsi_value:.2f}",
                    metadata={"rsi": rsi_value}
                ))
        
        # MACD信号
        macd = indicator_results.get("MACD_12_26_9")
        if macd and macd.is_valid:
            macd_value = macd.value
            if isinstance(macd_value, dict):
                macd_line = macd_value.get("macd", 0)
                signal_line = macd_value.get("signal", 0)
                histogram = macd_value.get("histogram", 0)
                
                # MACD线上穿信号线
                if macd_line > signal_line and histogram > self._signal_config["macd_signal_threshold"]:
                    signals.append(TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="BUY",
                        strength=min(abs(histogram) * 100, 1.0),
                        confidence=0.75,
                        timeframe=timeframe,
                        reason="MACD bullish crossover",
                        metadata=macd_value
                    ))
                
                # MACD线下穿信号线
                elif macd_line < signal_line and histogram < -self._signal_config["macd_signal_threshold"]:
                    signals.append(TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="SELL",
                        strength=min(abs(histogram) * 100, 1.0),
                        confidence=0.75,
                        timeframe=timeframe,
                        reason="MACD bearish crossover",
                        metadata=macd_value
                    ))
        
        return signals
    
    def _generate_volatility_signals(
        self,
        indicator_results: Dict[str, IndicatorResult],
        timeframe: TimeFrame
    ) -> List[TechnicalSignal]:
        """生成波动率信号"""
        signals = []
        
        # 布林带信号
        bbands = indicator_results.get("BBANDS_20_2")
        if bbands and bbands.is_valid:
            bb_value = bbands.value
            if isinstance(bb_value, dict):
                upper = bb_value.get("upper", 0)
                lower = bb_value.get("lower", 0)
                middle = bb_value.get("middle", 0)
                
                # 假设当前价格为middle（简化处理）
                current_price = middle
                
                # 布林带上轨突破
                if current_price > upper:
                    signals.append(TechnicalSignal(
                        indicator_name="BollingerBands",
                        signal_type="SELL",
                        strength=0.7,
                        confidence=0.6,
                        timeframe=timeframe,
                        reason="Price above Bollinger upper band",
                        metadata=bb_value
                    ))
                
                # 布林带下轨突破
                elif current_price < lower:
                    signals.append(TechnicalSignal(
                        indicator_name="BollingerBands",
                        signal_type="BUY",
                        strength=0.7,
                        confidence=0.6,
                        timeframe=timeframe,
                        reason="Price below Bollinger lower band",
                        metadata=bb_value
                    ))
        
        return signals
    
    def _generate_consensus_signal(
        self,
        symbol: str,
        analysis: MultiTimeFrameAnalysis
    ) -> Optional[Signal]:
        """生成综合共识信号"""
        if not analysis.signals_by_timeframe:
            return None
        
        # 统计不同类型信号的权重得分
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        confidence_scores = []
        reasons = []
        
        # 计算各时间框架信号的加权得分
        for timeframe, signals in analysis.signals_by_timeframe.items():
            timeframe_weight = self._timeframe_weights.get(timeframe, 0.5)
            
            for signal in signals:
                weight = timeframe_weight * signal.strength
                total_weight += weight
                confidence_scores.append(signal.confidence)
                
                if signal.signal_type == "BUY":
                    buy_score += weight
                    reasons.append(f"{timeframe.value}: {signal.reason}")
                elif signal.signal_type == "SELL":
                    sell_score += weight
                    reasons.append(f"{timeframe.value}: {signal.reason}")
        
        if total_weight == 0:
            return None
        
        # 确定最终信号方向和强度
        if buy_score > sell_score:
            action = "BUY"
            strength = min(buy_score / total_weight, 1.0)
        elif sell_score > buy_score:
            action = "SELL"
            strength = min(sell_score / total_weight, 1.0)
        else:
            action = "HOLD"
            strength = 0.0
        
        # 计算综合置信度
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # 时间框架一致性加分
        consistency_bonus = min(len(analysis.signals_by_timeframe) / len(self.timeframes), 1.0) * 0.2
        final_confidence = min(avg_confidence + consistency_bonus, 1.0)
        
        # 更新分析结果
        analysis.consensus_strength = strength
        analysis.consensus_confidence = final_confidence
        
        # 创建最终信号
        signal = Signal(
            source=self.name,
            symbol=symbol,
            action=action,
            strength=strength,
            confidence=final_confidence,
            reason=f"Multi-timeframe technical analysis: {'; '.join(reasons[:3])}",
            metadata={
                "analysis_type": "technical_multi_timeframe",
                "timeframes_analyzed": [tf.value for tf in analysis.signals_by_timeframe.keys()],
                "buy_score": buy_score,
                "sell_score": sell_score,
                "total_signals": sum(len(signals) for signals in analysis.signals_by_timeframe.values()),
                "timestamp": analysis.timestamp
            }
        )
        
        return signal
    
    async def _handle_market_data(self, message: Message):
        """处理市场数据消息"""
        try:
            data = message.data
            symbol = data.get("symbol", "").upper()
            
            if symbol in self.symbols:
                # 创建MarketData对象
                market_data = MarketData(
                    symbol=symbol,
                    price=data.get("price", 0.0),
                    volume=data.get("volume", 0.0),
                    timestamp=int(data.get("timestamp", time.time())),
                    bid=data.get("bid", data.get("bid_price", 0.0)),
                    ask=data.get("ask", data.get("ask_price", 0.0)),
                    bid_volume=data.get("bid_volume", 0.0),
                    ask_volume=data.get("ask_volume", 0.0)
                )
                
                # 更新缓存
                await self._update_market_data_cache(symbol, market_data)
        
        except Exception as e:
            self.log_error(f"Error handling market data: {e}")
    
    def get_analysis_result(self, symbol: str) -> Optional[MultiTimeFrameAnalysis]:
        """获取分析结果"""
        return self._analysis_cache.get(symbol)
    
    def get_indicator_values(self, symbol: str, timeframe: TimeFrame) -> Dict[str, IndicatorResult]:
        """获取指定时间框架的指标值"""
        return self.indicators.get_all_results(f"{symbol}_{timeframe.value}")
    
    def update_signal_config(self, config_updates: Dict[str, Any]):
        """更新信号生成配置"""
        self._signal_config.update(config_updates)
        self.log_info(f"Signal config updated: {list(config_updates.keys())}")
    
    def add_symbol(self, symbol: str):
        """添加新的交易对"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self._market_data_cache[symbol] = {tf: [] for tf in self.timeframes}
            self.log_info(f"Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """移除交易对"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self._market_data_cache:
                del self._market_data_cache[symbol]
            if symbol in self._analysis_cache:
                del self._analysis_cache[symbol]
            self.log_info(f"Removed symbol: {symbol}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        indicator_stats = self.indicators.get_performance_stats()
        
        return {
            "symbols_analyzed": len(self.symbols),
            "timeframes": [tf.value for tf in self.timeframes],
            "indicator_performance": indicator_stats,
            "agent_metrics": {
                "signals_generated": self.metrics.signals_generated,
                "error_count": self.metrics.error_count,
                "avg_processing_time": self.metrics.avg_processing_time,
                "uptime_seconds": self.metrics.uptime_seconds
            }
        }


# 便捷的Agent配置创建函数
def create_technical_analysis_agent(
    name: str = "technical_analysis_agent",
    symbols: List[str] = None,
    timeframes: List[TimeFrame] = None,
    message_bus: Optional[MessageBus] = None
) -> TechnicalAnalysisAgent:
    """创建技术分析Agent"""
    config = AgentConfig(
        name=name,
        enabled=True,
        priority=70,
        parameters={
            "analysis_interval": 60.0,  # 分析间隔（秒）
            "min_confidence": 0.6,      # 最小置信度
            "signal_cooldown": 300.0     # 信号冷却时间（秒）
        }
    )
    
    return TechnicalAnalysisAgent(
        config=config,
        message_bus=message_bus,
        timeframes=timeframes or [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1],
        symbols=symbols or ["BTCUSDT"]
    )