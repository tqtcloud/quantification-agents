"""
多维度技术指标引擎

整合各类技术指标的并行计算，实现多时间框架一致性检查，
提供综合信号强度评估和智能权重分配算法。
"""

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from ..indicators import (
    # 动量指标
    RelativeStrengthIndex, MACD, StochasticOscillator, CCI, WilliamsR,
    # 趋势指标  
    SimpleMovingAverage, ExponentialMovingAverage, ADX, ParabolicSAR, IchimokuCloud,
    # 波动率指标
    BollingerBands, ATR, StandardDeviation, KeltnerChannels, DonchianChannels,
    # 基础类和工具
    IndicatorResult, IndicatorConfig, IndicatorType, AsyncTechnicalIndicator,
    TimeFrameManager, TimeFrame, TimeFrameConfig, OHLCVData
)
from ..models.signals import (
    TradingSignal, MultiDimensionalSignal, SignalStrength, SignalAggregator
)
from ...utils.logger import LoggerMixin


class MarketRegime(Enum):
    """市场状态枚举"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CONSOLIDATION = "consolidation"


@dataclass
class TimeFrameConsensus:
    """时间框架共识数据"""
    timeframe: TimeFrame
    trend_direction: float  # -1到1，负值看跌，正值看涨
    trend_strength: float   # 0到1，趋势强度
    volatility: float      # 波动率
    volume_profile: float  # 成交量特征
    weight: float         # 该时间框架的权重
    
    @property
    def bullish_consensus(self) -> bool:
        """是否看涨共识"""
        return self.trend_direction > 0.3 and self.trend_strength > 0.5
    
    @property  
    def bearish_consensus(self) -> bool:
        """是否看跌共识"""
        return self.trend_direction < -0.3 and self.trend_strength > 0.5


@dataclass
class DimensionScore:
    """维度评分数据"""
    momentum_score: float           # 动量维度评分
    trend_score: float             # 趋势维度评分  
    volatility_score: float        # 波动率维度评分
    volume_score: float            # 成交量维度评分
    sentiment_score: float         # 情绪维度评分
    
    # 各维度的置信度
    momentum_confidence: float = 0.0
    trend_confidence: float = 0.0
    volatility_confidence: float = 0.0
    volume_confidence: float = 0.0
    sentiment_confidence: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """综合评分"""
        weights = {
            'momentum': 0.25,
            'trend': 0.25,
            'volatility': 0.15,
            'volume': 0.20,
            'sentiment': 0.15
        }
        
        return (
            self.momentum_score * weights['momentum'] +
            self.trend_score * weights['trend'] +
            self.volatility_score * weights['volatility'] +
            self.volume_score * weights['volume'] +
            self.sentiment_score * weights['sentiment']
        )
    
    @property
    def overall_confidence(self) -> float:
        """综合置信度"""
        confidences = [
            self.momentum_confidence,
            self.trend_confidence, 
            self.volatility_confidence,
            self.volume_confidence,
            self.sentiment_confidence
        ]
        return np.mean([c for c in confidences if c > 0])


class MultiDimensionalIndicatorEngine(LoggerMixin):
    """多维度技术指标引擎
    
    核心功能：
    1. 整合各类技术指标的并行计算
    2. 多时间框架一致性检查
    3. 信号强度综合评估
    4. 智能权重分配算法
    """
    
    def __init__(self, max_workers: int = 8):
        super().__init__()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 时间框架配置
        self.timeframes = [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5, 
            TimeFrame.MINUTE_15,
            TimeFrame.MINUTE_30,
            TimeFrame.HOUR_1,
            TimeFrame.HOUR_4,
            TimeFrame.DAY_1
        ]
        
        # 初始化各维度指标组
        self._init_indicators()
        
        # 时间框架管理器
        self.timeframe_manager = TimeFrameManager()
        
        # 性能统计
        self.stats = {
            'signals_generated': 0,
            'avg_processing_time': 0.0,
            'error_count': 0,
            'cache_hits': 0
        }
        
        # 缓存
        self._cache = {}
        self._cache_ttl = 300  # 5分钟TTL
    
    def _init_indicators(self):
        """初始化各维度指标组"""
        
        # 动量指标组
        self.momentum_indicators = {
            'rsi_14': RelativeStrengthIndex(period=14),
            'rsi_21': RelativeStrengthIndex(period=21),
            'macd': MACD(fast_period=12, slow_period=26, signal_period=9),
            'macd_fast': MACD(fast_period=8, slow_period=21, signal_period=5),
            'stoch': StochasticOscillator(k_period=14, d_period=3),
            'stoch_fast': StochasticOscillator(k_period=5, d_period=3),
            'cci': CCI(period=20),
            'williams_r': WilliamsR(period=14)
        }
        
        # 趋势指标组
        self.trend_indicators = {
            'sma_10': SimpleMovingAverage(period=10),
            'sma_20': SimpleMovingAverage(period=20),
            'sma_50': SimpleMovingAverage(period=50),
            'sma_200': SimpleMovingAverage(period=200),
            'ema_12': ExponentialMovingAverage(period=12),
            'ema_26': ExponentialMovingAverage(period=26),
            'adx': ADX(period=14),
            'psar': ParabolicSAR(),
            'ichimoku': IchimokuCloud()
        }
        
        # 波动率指标组
        self.volatility_indicators = {
            'bb': BollingerBands(period=20, std_dev=2),
            'bb_tight': BollingerBands(period=10, std_dev=1.5),
            'atr': ATR(period=14),
            'atr_short': ATR(period=7),
            'std_dev': StandardDeviation(period=20),
            'keltner': KeltnerChannels(ema_period=20, atr_period=10),
            'donchian': DonchianChannels(period=20)
        }
        
        # 成交量指标组（暂时使用简单的成交量分析）
        self.volume_indicators = {}  # 将在后续版本中扩展
        
        # 所有指标的集合
        self.all_indicators = {
            **self.momentum_indicators,
            **self.trend_indicators, 
            **self.volatility_indicators,
            **self.volume_indicators
        }
    
    async def generate_multidimensional_signal(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        timeframes: Optional[List[TimeFrame]] = None,
        enable_multiframe_analysis: bool = True
    ) -> Optional[MultiDimensionalSignal]:
        """生成多维度综合交易信号
        
        Args:
            symbol: 交易标的符号
            market_data: 市场数据，包含OHLCV数据
            timeframes: 分析的时间框架列表
            enable_multiframe_analysis: 是否启用多时间框架分析
            
        Returns:
            MultiDimensionalSignal: 多维度信号，失败时返回None
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始生成 {symbol} 的多维度信号")
            
            # 验证输入数据
            if not self._validate_market_data(market_data):
                self.logger.error(f"市场数据验证失败: {symbol}")
                return None
            
            # 使用默认时间框架
            if timeframes is None:
                timeframes = self.timeframes
            
            # 步骤1: 并行计算各维度指标
            dimension_scores = await self._calculate_dimension_scores(
                symbol, market_data, timeframes[0]  # 使用主要时间框架
            )
            
            if dimension_scores is None:
                self.logger.error(f"维度评分计算失败: {symbol}")
                return None
            
            # 步骤2: 多时间框架一致性检查
            timeframe_consensus = []
            if enable_multiframe_analysis and len(timeframes) > 1:
                timeframe_consensus = await self._check_timeframe_consistency(
                    symbol, market_data, timeframes
                )
            
            # 步骤3: 生成主要交易信号
            primary_signal = await self._generate_primary_signal(
                symbol, market_data, dimension_scores, timeframe_consensus
            )
            
            if primary_signal is None:
                self.logger.error(f"主信号生成失败: {symbol}")
                return None
            
            # 步骤4: 计算综合评估指标
            multidimensional_signal = await self._create_multidimensional_signal(
                primary_signal, dimension_scores, timeframe_consensus
            )
            
            # 更新统计信息
            processing_time = time.time() - start_time
            self.stats['signals_generated'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['signals_generated'] - 1) +
                 processing_time) / self.stats['signals_generated']
            )
            
            self.logger.info(
                f"成功生成 {symbol} 多维度信号，处理时间: {processing_time:.3f}s，"
                f"信号强度: {multidimensional_signal.primary_signal.signal_type.name}，"
                f"综合置信度: {multidimensional_signal.overall_confidence:.3f}"
            )
            
            return multidimensional_signal
            
        except Exception as e:
            self.stats['error_count'] += 1
            self.logger.error(f"生成多维度信号时发生错误: {symbol}, 错误: {str(e)}")
            return None
    
    async def _calculate_dimension_scores(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        timeframe: TimeFrame
    ) -> Optional[DimensionScore]:
        """并行计算各维度指标评分"""
        
        try:
            # 准备计算任务
            calculation_tasks = []
            
            # 动量维度任务
            momentum_task = asyncio.create_task(
                self._calculate_momentum_dimension(symbol, market_data)
            )
            calculation_tasks.append(('momentum', momentum_task))
            
            # 趋势维度任务
            trend_task = asyncio.create_task(
                self._calculate_trend_dimension(symbol, market_data)
            )
            calculation_tasks.append(('trend', trend_task))
            
            # 波动率维度任务
            volatility_task = asyncio.create_task(
                self._calculate_volatility_dimension(symbol, market_data)
            )
            calculation_tasks.append(('volatility', volatility_task))
            
            # 成交量维度任务
            volume_task = asyncio.create_task(
                self._calculate_volume_dimension(symbol, market_data)
            )
            calculation_tasks.append(('volume', volume_task))
            
            # 情绪维度任务
            sentiment_task = asyncio.create_task(
                self._calculate_sentiment_dimension(symbol, market_data)
            )
            calculation_tasks.append(('sentiment', sentiment_task))
            
            # 等待所有任务完成
            results = {}
            confidences = {}
            
            for dimension, task in calculation_tasks:
                try:
                    score, confidence = await task
                    results[dimension] = score
                    confidences[dimension] = confidence
                except Exception as e:
                    self.logger.warning(f"计算 {dimension} 维度时出错: {str(e)}")
                    results[dimension] = 0.0
                    confidences[dimension] = 0.0
            
            return DimensionScore(
                momentum_score=results.get('momentum', 0.0),
                trend_score=results.get('trend', 0.0),
                volatility_score=results.get('volatility', 0.0),
                volume_score=results.get('volume', 0.0),
                sentiment_score=results.get('sentiment', 0.0),
                momentum_confidence=confidences.get('momentum', 0.0),
                trend_confidence=confidences.get('trend', 0.0),
                volatility_confidence=confidences.get('volatility', 0.0),
                volume_confidence=confidences.get('volume', 0.0),
                sentiment_confidence=confidences.get('sentiment', 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"计算维度评分时发生错误: {str(e)}")
            return None
    
    async def _calculate_momentum_dimension(
        self, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算动量维度评分"""
        
        try:
            scores = []
            confidences = []
            
            for name, indicator in self.momentum_indicators.items():
                result = indicator.calculate(market_data, symbol)
                
                if result.is_valid:
                    # 根据不同指标类型计算标准化评分
                    if 'rsi' in name:
                        # RSI: 0-100范围，转换为-1到1
                        score = (result.value - 50) / 50
                        confidence = 0.9 if 30 <= result.value <= 70 else 0.7
                    elif 'macd' in name:
                        # MACD: 根据MACD线和信号线关系
                        if isinstance(result.value, dict):
                            macd_line = result.value.get('macd', 0)
                            signal_line = result.value.get('signal', 0)
                            score = np.tanh((macd_line - signal_line) * 10)
                            confidence = 0.8
                        else:
                            score = np.tanh(result.value * 10)
                            confidence = 0.8
                    elif 'stoch' in name:
                        # 随机指标: 0-100范围
                        if isinstance(result.value, dict):
                            k_value = result.value.get('k', 50)
                            score = (k_value - 50) / 50
                        else:
                            score = (result.value - 50) / 50
                        confidence = 0.7
                    elif 'cci' in name:
                        # CCI: 通常在-200到200之间
                        score = np.tanh(result.value / 100)
                        confidence = 0.75
                    elif 'williams' in name:
                        # Williams %R: -100到0之间
                        score = (result.value + 50) / 50
                        confidence = 0.75
                    else:
                        score = 0.0
                        confidence = 0.0
                    
                    # 限制评分范围
                    score = max(-1.0, min(1.0, score))
                    scores.append(score)
                    confidences.append(confidence)
            
            if scores:
                # 加权平均
                weights = np.array(confidences)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                final_score = np.average(scores, weights=weights)
                final_confidence = np.mean(confidences)
                
                return final_score, final_confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            self.logger.error(f"计算动量维度评分时出错: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_trend_dimension(
        self, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算趋势维度评分"""
        
        try:
            scores = []
            confidences = []
            
            current_price = market_data['close'][-1]
            
            for name, indicator in self.trend_indicators.items():
                result = indicator.calculate(market_data, symbol)
                
                if result.is_valid:
                    if 'sma' in name or 'ema' in name:
                        # 移动平均线: 价格与均线的关系
                        ma_value = result.value
                        score = (current_price - ma_value) / ma_value
                        # 限制评分范围
                        score = np.tanh(score * 20)  # 约5%的价格偏离对应1.0评分
                        confidence = 0.8
                    elif 'adx' in name:
                        # ADX: 趋势强度指标，需要结合+DI和-DI
                        if isinstance(result.value, dict):
                            adx_value = result.value.get('adx', 0)
                            plus_di = result.value.get('plus_di', 0)
                            minus_di = result.value.get('minus_di', 0)
                            
                            # 趋势方向
                            if plus_di > minus_di:
                                score = adx_value / 100  # 看涨趋势
                            else:
                                score = -adx_value / 100  # 看跌趋势
                            confidence = min(adx_value / 25, 1.0)  # ADX越高置信度越高
                        else:
                            score = 0.0
                            confidence = 0.0
                    elif 'psar' in name:
                        # 抛物线SAR
                        psar_value = result.value
                        if current_price > psar_value:
                            score = 0.7  # 看涨
                        else:
                            score = -0.7  # 看跌
                        confidence = 0.7
                    elif 'ichimoku' in name:
                        # 一目均衡表
                        if isinstance(result.value, dict):
                            conversion_line = result.value.get('conversion', current_price)
                            base_line = result.value.get('base', current_price)
                            
                            if conversion_line > base_line:
                                score = 0.8
                            else:
                                score = -0.8
                            confidence = 0.8
                        else:
                            score = 0.0
                            confidence = 0.0
                    else:
                        score = 0.0
                        confidence = 0.0
                    
                    scores.append(score)
                    confidences.append(confidence)
            
            if scores:
                weights = np.array(confidences)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                final_score = np.average(scores, weights=weights)
                final_confidence = np.mean(confidences)
                
                return final_score, final_confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            self.logger.error(f"计算趋势维度评分时出错: {str(e)}")
            return 0.0, 0.0
    
    async def _calculate_volatility_dimension(
        self, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算波动率维度评分"""
        
        try:
            scores = []
            confidences = []
            
            current_price = market_data['close'][-1]
            
            for name, indicator in self.volatility_indicators.items():
                result = indicator.calculate(market_data, symbol)
                
                if result.is_valid:
                    if 'bb' in name:
                        # 布林带
                        if isinstance(result.value, dict):
                            upper = result.value.get('upper', current_price)
                            lower = result.value.get('lower', current_price)
                            middle = result.value.get('middle', current_price)
                            
                            # 计算价格在布林带中的位置
                            if upper != lower:
                                bb_position = (current_price - lower) / (upper - lower)
                            else:
                                bb_position = 0.5
                            
                            # 转换为波动率评分 (0到1)
                            score = abs(bb_position - 0.5) * 2  # 离中线越远波动率越高
                            confidence = 0.8
                        else:
                            score = 0.5
                            confidence = 0.5
                    elif 'atr' in name:
                        # 平均真实范围
                        atr_value = result.value
                        # 将ATR标准化为相对于价格的百分比
                        atr_percentage = atr_value / current_price
                        score = min(atr_percentage * 50, 1.0)  # 2%的ATR对应1.0评分
                        confidence = 0.9
                    elif 'std_dev' in name:
                        # 标准差
                        std_value = result.value
                        std_percentage = std_value / current_price
                        score = min(std_percentage * 100, 1.0)
                        confidence = 0.8
                    elif 'keltner' in name or 'donchian' in name:
                        # 肯特纳通道或唐奇安通道
                        if isinstance(result.value, dict):
                            upper = result.value.get('upper', current_price)
                            lower = result.value.get('lower', current_price)
                            
                            if upper != lower:
                                channel_width = (upper - lower) / current_price
                                score = min(channel_width * 25, 1.0)
                            else:
                                score = 0.0
                            confidence = 0.7
                        else:
                            score = 0.0
                            confidence = 0.0
                    else:
                        score = 0.0
                        confidence = 0.0
                    
                    scores.append(score)
                    confidences.append(confidence)
            
            if scores:
                weights = np.array(confidences)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                
                final_score = np.average(scores, weights=weights)
                final_confidence = np.mean(confidences)
                
                return final_score, final_confidence
            else:
                return 0.5, 0.0  # 默认中等波动率
                
        except Exception as e:
            self.logger.error(f"计算波动率维度评分时出错: {str(e)}")
            return 0.5, 0.0
    
    async def _calculate_volume_dimension(
        self, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算成交量维度评分"""
        
        try:
            if 'volume' not in market_data or len(market_data['volume']) < 20:
                return 0.5, 0.0
            
            volumes = np.array(market_data['volume'])
            current_volume = volumes[-1]
            
            # 计算成交量的统计特征
            recent_avg_volume = np.mean(volumes[-20:])  # 20期平均成交量
            volume_std = np.std(volumes[-20:])
            
            # 成交量相对强度
            if recent_avg_volume > 0:
                volume_ratio = current_volume / recent_avg_volume
                volume_z_score = (current_volume - recent_avg_volume) / (volume_std + 1e-8)
            else:
                volume_ratio = 1.0
                volume_z_score = 0.0
            
            # 成交量趋势 (最近10期 vs 前10期)
            if len(volumes) >= 20:
                recent_volume = np.mean(volumes[-10:])
                previous_volume = np.mean(volumes[-20:-10])
                volume_trend = (recent_volume - previous_volume) / (previous_volume + 1e-8)
            else:
                volume_trend = 0.0
            
            # 综合评分
            # 1. 相对强度评分
            strength_score = min(volume_ratio / 2, 1.0)  # 2倍平均成交量对应1.0
            
            # 2. Z-score评分（异常成交量检测）
            zscore_score = min(abs(volume_z_score) / 3, 1.0)  # 3个标准差对应1.0
            
            # 3. 趋势评分
            trend_score = np.tanh(volume_trend * 5)  # 趋势评分转换为-1到1
            
            # 加权组合
            final_score = (
                strength_score * 0.5 +  
                zscore_score * 0.3 +
                abs(trend_score) * 0.2
            )
            
            # 置信度基于数据质量
            confidence = min(len(volumes) / 50, 0.8)  # 有足够历史数据时置信度更高
            
            return final_score, confidence
            
        except Exception as e:
            self.logger.error(f"计算成交量维度评分时出错: {str(e)}")
            return 0.5, 0.0
    
    async def _calculate_sentiment_dimension(
        self, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """计算情绪维度评分"""
        
        try:
            # 基于价格行为的简单情绪分析
            closes = np.array(market_data['close'])
            highs = np.array(market_data['high'])
            lows = np.array(market_data['low'])
            
            if len(closes) < 10:
                return 0.0, 0.0
            
            # 1. 价格动量情绪
            short_ma = np.mean(closes[-5:])
            long_ma = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            momentum_sentiment = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            # 2. 涨跌频率情绪
            price_changes = np.diff(closes[-20:]) if len(closes) >= 20 else np.diff(closes)
            bullish_days = np.sum(price_changes > 0)
            total_days = len(price_changes)
            frequency_sentiment = (bullish_days / total_days - 0.5) * 2 if total_days > 0 else 0
            
            # 3. 上影线/下影线情绪分析
            if len(closes) >= 10:
                recent_closes = closes[-10:]
                recent_highs = highs[-10:]
                recent_lows = lows[-10:]
                recent_opens = market_data['open'][-10:] if 'open' in market_data else recent_closes
                
                upper_shadows = recent_highs - np.maximum(recent_closes, recent_opens)
                lower_shadows = np.minimum(recent_closes, recent_opens) - recent_lows
                
                avg_upper_shadow = np.mean(upper_shadows)
                avg_lower_shadow = np.mean(lower_shadows)
                
                if avg_upper_shadow + avg_lower_shadow > 0:
                    shadow_sentiment = (avg_lower_shadow - avg_upper_shadow) / (avg_upper_shadow + avg_lower_shadow)
                else:
                    shadow_sentiment = 0
            else:
                shadow_sentiment = 0
            
            # 综合情绪评分
            sentiment_score = np.tanh(
                momentum_sentiment * 0.5 +
                frequency_sentiment * 0.3 +
                shadow_sentiment * 0.2
            )
            
            # 置信度基于数据充分性
            confidence = min(len(closes) / 50, 0.7)
            
            return sentiment_score, confidence
            
        except Exception as e:
            self.logger.error(f"计算情绪维度评分时出错: {str(e)}")
            return 0.0, 0.0
    
    async def _check_timeframe_consistency(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        timeframes: List[TimeFrame]
    ) -> List[TimeFrameConsensus]:
        """检查多时间框架一致性"""
        
        consensus_results = []
        
        try:
            # 为每个时间框架计算共识
            for tf in timeframes:
                try:
                    # 获取该时间框架的数据
                    tf_data = self._aggregate_to_timeframe(market_data, tf)
                    
                    if tf_data is None:
                        continue
                    
                    # 计算趋势方向和强度
                    trend_direction, trend_strength = await self._calculate_trend_metrics(tf_data)
                    
                    # 计算波动率
                    volatility = self._calculate_volatility_metric(tf_data)
                    
                    # 计算成交量特征
                    volume_profile = self._calculate_volume_profile(tf_data)
                    
                    # 计算该时间框架的权重
                    weight = self._calculate_timeframe_weight(tf)
                    
                    consensus = TimeFrameConsensus(
                        timeframe=tf,
                        trend_direction=trend_direction,
                        trend_strength=trend_strength,
                        volatility=volatility,
                        volume_profile=volume_profile,
                        weight=weight
                    )
                    
                    consensus_results.append(consensus)
                    
                except Exception as e:
                    self.logger.warning(f"处理时间框架 {tf.value} 时出错: {str(e)}")
                    continue
            
            return consensus_results
            
        except Exception as e:
            self.logger.error(f"检查时间框架一致性时发生错误: {str(e)}")
            return []
    
    def _aggregate_to_timeframe(
        self, 
        market_data: Dict[str, Any], 
        timeframe: TimeFrame
    ) -> Optional[Dict[str, Any]]:
        """将数据聚合到指定时间框架"""
        
        try:
            # 简化实现：对于演示，我们假设输入数据已经是合适的时间框架
            # 在实际应用中，这里需要实现真正的时间聚合逻辑
            return market_data
            
        except Exception as e:
            self.logger.error(f"数据聚合到时间框架 {timeframe.value} 时出错: {str(e)}")
            return None
    
    async def _calculate_trend_metrics(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """计算趋势方向和强度"""
        
        try:
            closes = np.array(data['close'])
            
            if len(closes) < 20:
                return 0.0, 0.0
            
            # 使用简单的移动平均系统判断趋势
            sma_short = np.mean(closes[-10:])
            sma_long = np.mean(closes[-20:])
            
            # 趋势方向
            trend_direction = (sma_short - sma_long) / sma_long if sma_long != 0 else 0
            trend_direction = np.tanh(trend_direction * 10)  # 标准化到-1到1
            
            # 趋势强度：基于价格与移动平均线的一致性
            price_above_sma = np.sum(closes[-10:] > sma_short)
            trend_strength = price_above_sma / 10 if trend_direction > 0 else (10 - price_above_sma) / 10
            
            return trend_direction, trend_strength
            
        except Exception as e:
            self.logger.error(f"计算趋势指标时出错: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_volatility_metric(self, data: Dict[str, Any]) -> float:
        """计算波动率指标"""
        
        try:
            closes = np.array(data['close'])
            
            if len(closes) < 10:
                return 0.5
            
            # 计算价格变化的标准差
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(len(returns))
            
            # 标准化到0-1范围
            return min(volatility * 20, 1.0)  # 5%的波动率对应1.0
            
        except Exception as e:
            self.logger.error(f"计算波动率指标时出错: {str(e)}")
            return 0.5
    
    def _calculate_volume_profile(self, data: Dict[str, Any]) -> float:
        """计算成交量特征"""
        
        try:
            if 'volume' not in data:
                return 0.5
            
            volumes = np.array(data['volume'])
            
            if len(volumes) < 10:
                return 0.5
            
            # 计算成交量的变异系数
            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            if avg_volume > 0:
                cv = std_volume / avg_volume
                return min(cv, 1.0)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"计算成交量特征时出错: {str(e)}")
            return 0.5
    
    def _calculate_timeframe_weight(self, timeframe: TimeFrame) -> float:
        """计算时间框架权重"""
        
        # 时间框架权重配置
        weights = {
            TimeFrame.MINUTE_1: 0.1,
            TimeFrame.MINUTE_5: 0.15,
            TimeFrame.MINUTE_15: 0.2,
            TimeFrame.MINUTE_30: 0.2,
            TimeFrame.HOUR_1: 0.15,
            TimeFrame.HOUR_4: 0.15,
            TimeFrame.DAY_1: 0.05
        }
        
        return weights.get(timeframe, 0.1)
    
    async def _generate_primary_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        dimension_scores: DimensionScore,
        timeframe_consensus: List[TimeFrameConsensus]
    ) -> Optional[TradingSignal]:
        """生成主要交易信号"""
        
        try:
            current_price = market_data['close'][-1]
            
            # 基于维度评分判断信号强度
            overall_score = dimension_scores.overall_score
            overall_confidence = dimension_scores.overall_confidence
            
            # 多时间框架一致性调整
            if timeframe_consensus:
                bullish_consensus = sum(1 for c in timeframe_consensus if c.bullish_consensus)
                bearish_consensus = sum(1 for c in timeframe_consensus if c.bearish_consensus)
                total_frames = len(timeframe_consensus)
                
                if total_frames > 0:
                    consensus_adjustment = (bullish_consensus - bearish_consensus) / total_frames
                    overall_score = (overall_score + consensus_adjustment) / 2
            
            # 确定信号类型
            if overall_score > 0.7:
                signal_type = SignalStrength.STRONG_BUY
            elif overall_score > 0.3:
                signal_type = SignalStrength.BUY
            elif overall_score > 0.1:
                signal_type = SignalStrength.WEAK_BUY
            elif overall_score < -0.7:
                signal_type = SignalStrength.STRONG_SELL
            elif overall_score < -0.3:
                signal_type = SignalStrength.SELL
            elif overall_score < -0.1:
                signal_type = SignalStrength.WEAK_SELL
            else:
                signal_type = SignalStrength.NEUTRAL
            
            # 如果是中性信号且置信度不高，直接返回None
            if signal_type == SignalStrength.NEUTRAL and overall_confidence < 0.5:
                return None
            
            # 计算价格目标和止损
            atr = self._estimate_atr(market_data)
            
            if signal_type in [SignalStrength.WEAK_BUY, SignalStrength.BUY, SignalStrength.STRONG_BUY]:
                # 买入信号
                target_price = current_price + (atr * 2.0)  # 2倍ATR作为目标
                stop_loss = current_price - (atr * 1.0)     # 1倍ATR作为止损
            elif signal_type in [SignalStrength.WEAK_SELL, SignalStrength.SELL, SignalStrength.STRONG_SELL]:
                # 卖出信号
                target_price = current_price - (atr * 2.0)  # 2倍ATR作为目标
                stop_loss = current_price + (atr * 1.0)     # 1倍ATR作为止损
            else:
                # 中性信号
                target_price = current_price
                stop_loss = current_price - (atr * 0.5)
            
            # 生成推理过程
            reasoning = self._generate_reasoning(dimension_scores, timeframe_consensus, overall_score)
            
            # 生成指标共识
            indicators_consensus = self._generate_indicators_consensus(dimension_scores)
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=overall_confidence,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                indicators_consensus=indicators_consensus
            )
            
        except Exception as e:
            self.logger.error(f"生成主要交易信号时出错: {str(e)}")
            return None
    
    def _estimate_atr(self, market_data: Dict[str, Any]) -> float:
        """估算平均真实范围"""
        
        try:
            highs = np.array(market_data['high'])
            lows = np.array(market_data['low'])
            closes = np.array(market_data['close'])
            
            if len(highs) < 14:
                # 数据不足，使用简单估算
                return (np.max(highs[-5:]) - np.min(lows[-5:])) * 0.5
            
            # 简单ATR计算
            true_ranges = []
            for i in range(1, min(len(highs), 14)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            return np.mean(true_ranges) if true_ranges else (highs[-1] - lows[-1]) * 0.5
            
        except Exception as e:
            self.logger.error(f"估算ATR时出错: {str(e)}")
            return market_data['close'][-1] * 0.02  # 默认使用2%作为ATR
    
    def _generate_reasoning(
        self, 
        dimension_scores: DimensionScore,
        timeframe_consensus: List[TimeFrameConsensus],
        overall_score: float
    ) -> List[str]:
        """生成信号推理过程"""
        
        reasoning = []
        
        # 维度分析
        if abs(dimension_scores.momentum_score) > 0.3:
            direction = "看涨" if dimension_scores.momentum_score > 0 else "看跌"
            reasoning.append(f"动量指标显示{direction}信号，评分: {dimension_scores.momentum_score:.2f}")
        
        if abs(dimension_scores.trend_score) > 0.3:
            direction = "上升" if dimension_scores.trend_score > 0 else "下降"
            reasoning.append(f"趋势指标显示{direction}趋势，评分: {dimension_scores.trend_score:.2f}")
        
        if dimension_scores.volatility_score > 0.7:
            reasoning.append(f"高波动率环境，评分: {dimension_scores.volatility_score:.2f}")
        elif dimension_scores.volatility_score < 0.3:
            reasoning.append(f"低波动率环境，评分: {dimension_scores.volatility_score:.2f}")
        
        if dimension_scores.volume_score > 0.7:
            reasoning.append(f"成交量活跃，评分: {dimension_scores.volume_score:.2f}")
        
        # 时间框架一致性
        if timeframe_consensus:
            bullish_frames = sum(1 for c in timeframe_consensus if c.bullish_consensus)
            bearish_frames = sum(1 for c in timeframe_consensus if c.bearish_consensus)
            
            if bullish_frames > bearish_frames:
                reasoning.append(f"多时间框架看涨共识: {bullish_frames}/{len(timeframe_consensus)}")
            elif bearish_frames > bullish_frames:
                reasoning.append(f"多时间框架看跌共识: {bearish_frames}/{len(timeframe_consensus)}")
            else:
                reasoning.append("多时间框架信号分歧")
        
        # 综合评估
        if abs(overall_score) > 0.5:
            direction = "强烈看涨" if overall_score > 0.5 else "强烈看跌"
            reasoning.append(f"综合评分显示{direction}，评分: {overall_score:.2f}")
        
        return reasoning if reasoning else ["综合技术指标分析"]
    
    def _generate_indicators_consensus(self, dimension_scores: DimensionScore) -> Dict[str, float]:
        """生成指标共识字典"""
        
        return {
            'momentum': dimension_scores.momentum_score,
            'trend': dimension_scores.trend_score,
            'volatility': dimension_scores.volatility_score,
            'volume': dimension_scores.volume_score,
            'sentiment': dimension_scores.sentiment_score,
            'overall': dimension_scores.overall_score
        }
    
    async def _create_multidimensional_signal(
        self,
        primary_signal: TradingSignal,
        dimension_scores: DimensionScore,
        timeframe_consensus: List[TimeFrameConsensus]
    ) -> MultiDimensionalSignal:
        """创建多维度信号"""
        
        try:
            # 计算风险收益比
            risk_reward_ratio = primary_signal.risk_reward_ratio
            
            # 基于波动率和信号强度计算建议最大仓位
            volatility_adjustment = 1.0 - (dimension_scores.volatility_score * 0.5)
            confidence_adjustment = dimension_scores.overall_confidence
            signal_strength = abs(dimension_scores.overall_score)
            
            max_position_size = min(
                volatility_adjustment * confidence_adjustment * signal_strength,
                1.0
            )
            
            # 确定市场状态
            market_regime = self._determine_market_regime(dimension_scores, timeframe_consensus)
            
            # 构建技术位字典
            technical_levels = self._calculate_technical_levels(primary_signal)
            
            return MultiDimensionalSignal(
                primary_signal=primary_signal,
                momentum_score=dimension_scores.momentum_score,
                mean_reversion_score=-dimension_scores.trend_score,  # 均值回归与趋势相反
                volatility_score=dimension_scores.volatility_score,
                volume_score=dimension_scores.volume_score,
                sentiment_score=dimension_scores.sentiment_score,
                overall_confidence=dimension_scores.overall_confidence,
                risk_reward_ratio=risk_reward_ratio,
                max_position_size=max_position_size,
                market_regime=market_regime.value,
                technical_levels=technical_levels
            )
            
        except Exception as e:
            self.logger.error(f"创建多维度信号时出错: {str(e)}")
            raise
    
    def _determine_market_regime(
        self,
        dimension_scores: DimensionScore,
        timeframe_consensus: List[TimeFrameConsensus]
    ) -> MarketRegime:
        """确定市场状态"""
        
        try:
            # 基于趋势强度和波动率确定市场状态
            trend_strength = abs(dimension_scores.trend_score)
            volatility = dimension_scores.volatility_score
            
            if trend_strength > 0.6 and dimension_scores.trend_score > 0:
                return MarketRegime.TRENDING_UP
            elif trend_strength > 0.6 and dimension_scores.trend_score < 0:
                return MarketRegime.TRENDING_DOWN
            elif volatility > 0.8:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.2:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength < 0.3:
                return MarketRegime.SIDEWAYS
            else:
                return MarketRegime.CONSOLIDATION
                
        except Exception as e:
            self.logger.error(f"确定市场状态时出错: {str(e)}")
            return MarketRegime.CONSOLIDATION
    
    def _calculate_technical_levels(self, primary_signal: TradingSignal) -> Dict[str, float]:
        """计算技术位"""
        
        entry_price = primary_signal.entry_price
        target_price = primary_signal.target_price
        stop_loss = primary_signal.stop_loss
        
        return {
            'support': min(entry_price, stop_loss),
            'resistance': max(entry_price, target_price),
            'entry': entry_price,
            'target': target_price,
            'stop_loss': stop_loss
        }
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """验证市场数据"""
        
        required_fields = ['open', 'high', 'low', 'close']
        
        try:
            for field in required_fields:
                if field not in market_data:
                    self.logger.error(f"缺少必需字段: {field}")
                    return False
                
                if not isinstance(market_data[field], (list, np.ndarray)):
                    self.logger.error(f"字段 {field} 必须是列表或数组")
                    return False
                
                if len(market_data[field]) < 20:
                    self.logger.warning(f"字段 {field} 数据不足，至少需要20个数据点")
                    return False
            
            # 验证数据长度一致性
            length = len(market_data['close'])
            for field in required_fields:
                if len(market_data[field]) != length:
                    self.logger.error(f"数据长度不一致: {field}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证市场数据时出错: {str(e)}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计数据"""
        self.stats = {
            'signals_generated': 0,
            'avg_processing_time': 0.0,
            'error_count': 0,
            'cache_hits': 0
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("多维度指标引擎资源已清理")
        except Exception as e:
            self.logger.error(f"清理资源时出错: {str(e)}")