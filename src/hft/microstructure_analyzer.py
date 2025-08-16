import asyncio
import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Deque
from datetime import datetime, timedelta
import numpy as np

from src.hft.orderbook_manager import OrderBookSnapshot, OrderBookLevel
from src.core.models import MarketData
from src.utils.logger import LoggerMixin


@dataclass
class MicrostructureSignal:
    """微观结构信号"""
    signal_type: str
    symbol: str
    timestamp: float
    strength: float  # 信号强度 -1.0 到 1.0
    confidence: float  # 置信度 0.0 到 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class ImbalanceMetrics:
    """订单簿失衡指标"""
    bid_ask_imbalance: float  # 买卖盘失衡
    volume_imbalance: float   # 成交量失衡
    depth_imbalance: float    # 深度失衡
    price_impact: float       # 价格冲击
    timestamp: float = field(default_factory=time.time)


@dataclass
class FlowToxicity:
    """订单流毒性指标"""
    vpin: float  # Volume-synchronized Probability of Informed Trading
    trade_intensity: float  # 交易强度
    adverse_selection: float  # 逆向选择成本
    timestamp: float = field(default_factory=time.time)


class MicrostructureAnalyzer(LoggerMixin):
    """微观结构分析器"""
    
    def __init__(self, lookback_window: int = 100, min_signal_strength: float = 0.3):
        self.lookback_window = lookback_window
        self.min_signal_strength = min_signal_strength
        
        # 历史数据缓存
        self.price_history: Dict[str, Deque[float]] = {}
        self.volume_history: Dict[str, Deque[float]] = {}
        self.spread_history: Dict[str, Deque[float]] = {}
        self.imbalance_history: Dict[str, Deque[ImbalanceMetrics]] = {}
        self.toxicity_history: Dict[str, Deque[FlowToxicity]] = {}
        
        # 实时指标
        self.current_signals: Dict[str, List[MicrostructureSignal]] = {}
        
        # 计算参数
        self.ema_alpha = 0.1  # EMA平滑因子
        self.vpin_window = 50  # VPIN计算窗口
        
    async def initialize(self, symbols: List[str]):
        """初始化分析器"""
        for symbol in symbols:
            self.price_history[symbol] = deque(maxlen=self.lookback_window)
            self.volume_history[symbol] = deque(maxlen=self.lookback_window)
            self.spread_history[symbol] = deque(maxlen=self.lookback_window)
            self.imbalance_history[symbol] = deque(maxlen=self.lookback_window)
            self.toxicity_history[symbol] = deque(maxlen=self.lookback_window)
            self.current_signals[symbol] = []
            
        self.log_info(f"Microstructure analyzer initialized for {len(symbols)} symbols")
    
    async def update(self, symbol: str, orderbook: OrderBookSnapshot, 
                    market_data: MarketData) -> List[MicrostructureSignal]:
        """更新微观结构分析"""
        signals = []
        
        try:
            # 更新历史数据
            self._update_history(symbol, orderbook, market_data)
            
            # 计算订单簿失衡
            imbalance = self._calculate_imbalance(orderbook)
            if imbalance:
                self.imbalance_history[symbol].append(imbalance)
                
                # 生成失衡信号
                imbalance_signal = self._generate_imbalance_signal(symbol, imbalance)
                if imbalance_signal:
                    signals.append(imbalance_signal)
            
            # 计算流动性毒性
            toxicity = self._calculate_flow_toxicity(symbol, orderbook, market_data)
            if toxicity:
                self.toxicity_history[symbol].append(toxicity)
                
                # 生成毒性信号
                toxicity_signal = self._generate_toxicity_signal(symbol, toxicity)
                if toxicity_signal:
                    signals.append(toxicity_signal)
            
            # 检测价格异常
            price_signal = self._detect_price_anomaly(symbol, market_data.price)
            if price_signal:
                signals.append(price_signal)
            
            # 检测成交量异常
            volume_signal = self._detect_volume_spike(symbol, market_data.volume)
            if volume_signal:
                signals.append(volume_signal)
            
            # 检测价差异常
            spread_signal = self._detect_spread_anomaly(symbol, orderbook)
            if spread_signal:
                signals.append(spread_signal)
            
            # 更新当前信号
            self.current_signals[symbol] = signals
            
            return signals
            
        except Exception as e:
            self.log_error(f"Error updating microstructure analysis for {symbol}: {e}")
            return []
    
    def _update_history(self, symbol: str, orderbook: OrderBookSnapshot, market_data: MarketData):
        """更新历史数据"""
        self.price_history[symbol].append(market_data.price)
        self.volume_history[symbol].append(market_data.volume)
        
        if orderbook.spread:
            self.spread_history[symbol].append(float(orderbook.spread))
    
    def _calculate_imbalance(self, orderbook: OrderBookSnapshot) -> Optional[ImbalanceMetrics]:
        """计算订单簿失衡指标"""
        if not orderbook.best_bid or not orderbook.best_ask:
            return None
        
        # 买卖盘失衡 (OIR - Order Imbalance Ratio)
        bid_size = orderbook.best_bid.size
        ask_size = orderbook.best_ask.size
        total_size = bid_size + ask_size
        
        if total_size == 0:
            return None
            
        bid_ask_imbalance = float((bid_size - ask_size) / total_size)
        
        # 深度失衡（前5档）
        bid_depth = sum(level.size for level in orderbook.bids[:5])
        ask_depth = sum(level.size for level in orderbook.asks[:5])
        total_depth = bid_depth + ask_depth
        
        depth_imbalance = float((bid_depth - ask_depth) / total_depth) if total_depth > 0 else 0
        
        # 成交量加权失衡
        bid_volume_weighted = sum(level.size * level.price for level in orderbook.bids[:5])
        ask_volume_weighted = sum(level.size * level.price for level in orderbook.asks[:5])
        total_volume_weighted = bid_volume_weighted + ask_volume_weighted
        
        volume_imbalance = float((bid_volume_weighted - ask_volume_weighted) / total_volume_weighted) if total_volume_weighted > 0 else 0
        
        # 价格冲击估计
        mid_price = orderbook.mid_price
        if mid_price:
            bid_impact = float(abs(orderbook.best_bid.price - mid_price) / mid_price)
            ask_impact = float(abs(orderbook.best_ask.price - mid_price) / mid_price)
            price_impact = (bid_impact + ask_impact) / 2
        else:
            price_impact = 0
        
        return ImbalanceMetrics(
            bid_ask_imbalance=bid_ask_imbalance,
            volume_imbalance=volume_imbalance,
            depth_imbalance=depth_imbalance,
            price_impact=price_impact
        )
    
    def _calculate_flow_toxicity(self, symbol: str, orderbook: OrderBookSnapshot, 
                                market_data: MarketData) -> Optional[FlowToxicity]:
        """计算订单流毒性"""
        if len(self.volume_history[symbol]) < self.vpin_window:
            return None
        
        try:
            # 计算VPIN (Volume-synchronized Probability of Informed Trading)
            volumes = list(self.volume_history[symbol])[-self.vpin_window:]
            prices = list(self.price_history[symbol])[-self.vpin_window:]
            
            if len(volumes) != len(prices):
                return None
            
            # 计算买卖订单流
            buy_volumes = []
            sell_volumes = []
            
            for i in range(1, len(prices)):
                price_change = prices[i] - prices[i-1]
                volume = volumes[i]
                
                if price_change > 0:
                    buy_volumes.append(volume)
                    sell_volumes.append(0)
                elif price_change < 0:
                    buy_volumes.append(0)
                    sell_volumes.append(volume)
                else:
                    # 价格未变，假设各占一半
                    buy_volumes.append(volume / 2)
                    sell_volumes.append(volume / 2)
            
            if not buy_volumes:
                return None
            
            # VPIN计算
            total_volume = sum(buy_volumes) + sum(sell_volumes)
            if total_volume == 0:
                vpin = 0
            else:
                volume_imbalance = abs(sum(buy_volumes) - sum(sell_volumes))
                vpin = volume_imbalance / total_volume
            
            # 交易强度
            trade_intensity = np.std(volumes) / np.mean(volumes) if volumes else 0
            
            # 逆向选择成本（基于价差和成交量）
            spreads = list(self.spread_history[symbol])[-min(len(self.spread_history[symbol]), 20):]
            if spreads and volumes:
                avg_spread = np.mean(spreads)
                avg_volume = np.mean(volumes[-20:])
                adverse_selection = avg_spread * avg_volume / 10000  # 标准化
            else:
                adverse_selection = 0
            
            return FlowToxicity(
                vpin=vpin,
                trade_intensity=trade_intensity,
                adverse_selection=adverse_selection
            )
            
        except Exception as e:
            self.log_error(f"Error calculating flow toxicity for {symbol}: {e}")
            return None
    
    def _generate_imbalance_signal(self, symbol: str, imbalance: ImbalanceMetrics) -> Optional[MicrostructureSignal]:
        """生成失衡信号"""
        # 综合失衡指标
        combined_imbalance = (
            imbalance.bid_ask_imbalance * 0.4 + 
            imbalance.depth_imbalance * 0.4 + 
            imbalance.volume_imbalance * 0.2
        )
        
        # 信号强度和方向
        strength = abs(combined_imbalance)
        if strength < self.min_signal_strength:
            return None
        
        # 置信度基于历史一致性
        recent_imbalances = list(self.imbalance_history[symbol])[-10:]
        if len(recent_imbalances) > 1:
            recent_values = [im.bid_ask_imbalance for im in recent_imbalances]
            consistency = 1 - (np.std(recent_values) / (np.mean(np.abs(recent_values)) + 1e-6))
            confidence = max(0, min(1, consistency))
        else:
            confidence = 0.5
        
        return MicrostructureSignal(
            signal_type="imbalance",
            symbol=symbol,
            timestamp=time.time(),
            strength=combined_imbalance,
            confidence=confidence,
            metadata={
                "bid_ask_imbalance": imbalance.bid_ask_imbalance,
                "depth_imbalance": imbalance.depth_imbalance,
                "volume_imbalance": imbalance.volume_imbalance,
                "price_impact": imbalance.price_impact
            }
        )
    
    def _generate_toxicity_signal(self, symbol: str, toxicity: FlowToxicity) -> Optional[MicrostructureSignal]:
        """生成毒性信号"""
        # VPIN大于阈值时生成信号
        vpin_threshold = 0.3
        if toxicity.vpin < vpin_threshold:
            return None
        
        # 信号强度基于VPIN和交易强度
        strength = min(1.0, toxicity.vpin + toxicity.trade_intensity / 10)
        
        # 置信度基于历史稳定性
        recent_toxicity = list(self.toxicity_history[symbol])[-5:]
        if len(recent_toxicity) > 1:
            vpin_values = [t.vpin for t in recent_toxicity]
            stability = 1 - (np.std(vpin_values) / (np.mean(vpin_values) + 1e-6))
            confidence = max(0.3, min(1, stability))
        else:
            confidence = 0.5
        
        return MicrostructureSignal(
            signal_type="toxicity",
            symbol=symbol,
            timestamp=time.time(),
            strength=strength,
            confidence=confidence,
            metadata={
                "vpin": toxicity.vpin,
                "trade_intensity": toxicity.trade_intensity,
                "adverse_selection": toxicity.adverse_selection
            }
        )
    
    def _detect_price_anomaly(self, symbol: str, current_price: float) -> Optional[MicrostructureSignal]:
        """检测价格异常"""
        prices = list(self.price_history[symbol])
        if len(prices) < 20:
            return None
        
        # 计算Z-score
        mean_price = np.mean(prices[-20:])
        std_price = np.std(prices[-20:])
        
        if std_price == 0:
            return None
        
        z_score = (current_price - mean_price) / std_price
        
        # 异常阈值
        if abs(z_score) < 2.0:
            return None
        
        strength = min(1.0, abs(z_score) / 5.0)  # 标准化到0-1
        confidence = min(1.0, abs(z_score) / 3.0)
        
        return MicrostructureSignal(
            signal_type="price_anomaly",
            symbol=symbol,
            timestamp=time.time(),
            strength=strength * (1 if z_score > 0 else -1),
            confidence=confidence,
            metadata={
                "z_score": z_score,
                "current_price": current_price,
                "mean_price": mean_price,
                "std_price": std_price
            }
        )
    
    def _detect_volume_spike(self, symbol: str, current_volume: float) -> Optional[MicrostructureSignal]:
        """检测成交量异常"""
        volumes = list(self.volume_history[symbol])
        if len(volumes) < 20:
            return None
        
        # 计算移动平均和标准差
        mean_volume = np.mean(volumes[-20:])
        std_volume = np.std(volumes[-20:])
        
        if std_volume == 0 or mean_volume == 0:
            return None
        
        # 成交量倍数
        volume_ratio = current_volume / mean_volume
        
        # 只关注显著增加的成交量
        if volume_ratio < 2.0:
            return None
        
        strength = min(1.0, volume_ratio / 5.0)
        confidence = min(1.0, (volume_ratio - 1) / 3.0)
        
        return MicrostructureSignal(
            signal_type="volume_spike",
            symbol=symbol,
            timestamp=time.time(),
            strength=strength,
            confidence=confidence,
            metadata={
                "volume_ratio": volume_ratio,
                "current_volume": current_volume,
                "mean_volume": mean_volume
            }
        )
    
    def _detect_spread_anomaly(self, symbol: str, orderbook: OrderBookSnapshot) -> Optional[MicrostructureSignal]:
        """检测价差异常"""
        if not orderbook.spread:
            return None
        
        spreads = list(self.spread_history[symbol])
        if len(spreads) < 10:
            return None
        
        current_spread = float(orderbook.spread)
        mean_spread = np.mean(spreads[-20:])
        std_spread = np.std(spreads[-20:])
        
        if std_spread == 0 or mean_spread == 0:
            return None
        
        # 价差比率
        spread_ratio = current_spread / mean_spread
        
        # 异常阈值（价差显著增加）
        if spread_ratio < 1.5:
            return None
        
        strength = min(1.0, spread_ratio / 3.0)
        confidence = min(1.0, (spread_ratio - 1) / 2.0)
        
        return MicrostructureSignal(
            signal_type="spread_anomaly",
            symbol=symbol,
            timestamp=time.time(),
            strength=strength,
            confidence=confidence,
            metadata={
                "spread_ratio": spread_ratio,
                "current_spread": current_spread,
                "mean_spread": mean_spread
            }
        )
    
    def get_current_signals(self, symbol: str) -> List[MicrostructureSignal]:
        """获取当前信号"""
        return self.current_signals.get(symbol, [])
    
    def get_imbalance_metrics(self, symbol: str) -> Optional[ImbalanceMetrics]:
        """获取最新失衡指标"""
        history = self.imbalance_history.get(symbol)
        return history[-1] if history else None
    
    def get_toxicity_metrics(self, symbol: str) -> Optional[FlowToxicity]:
        """获取最新毒性指标"""
        history = self.toxicity_history.get(symbol)
        return history[-1] if history else None
    
    def get_analytics_summary(self, symbol: str) -> Dict[str, any]:
        """获取分析摘要"""
        signals = self.current_signals.get(symbol, [])
        imbalance = self.get_imbalance_metrics(symbol)
        toxicity = self.get_toxicity_metrics(symbol)
        
        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "active_signals": len(signals),
            "signal_types": [s.signal_type for s in signals],
            "max_signal_strength": max([abs(s.strength) for s in signals]) if signals else 0,
            "avg_confidence": np.mean([s.confidence for s in signals]) if signals else 0,
            "imbalance": {
                "bid_ask_imbalance": imbalance.bid_ask_imbalance if imbalance else None,
                "depth_imbalance": imbalance.depth_imbalance if imbalance else None,
                "volume_imbalance": imbalance.volume_imbalance if imbalance else None
            },
            "toxicity": {
                "vpin": toxicity.vpin if toxicity else None,
                "trade_intensity": toxicity.trade_intensity if toxicity else None,
                "adverse_selection": toxicity.adverse_selection if toxicity else None
            }
        }