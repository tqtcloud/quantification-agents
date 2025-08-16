import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Deque
import numpy as np

from src.hft.microstructure_analyzer import MicrostructureSignal
from src.utils.logger import LoggerMixin


class SignalPriority(Enum):
    """信号优先级"""
    CRITICAL = 1    # 关键信号，需要立即处理
    HIGH = 2        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级


class ActionType(Enum):
    """动作类型"""
    BUY = "buy"
    SELL = "sell"
    CANCEL = "cancel"
    ADJUST = "adjust"
    HOLD = "hold"


@dataclass
class SignalAction:
    """信号动作"""
    action_type: ActionType
    symbol: str
    priority: SignalPriority
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None
    urgency_score: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessingMetrics:
    """处理指标"""
    total_signals: int = 0
    processed_signals: int = 0
    avg_processing_latency: float = 0.0
    max_processing_latency: float = 0.0
    queue_size: int = 0
    dropped_signals: int = 0
    actions_generated: int = 0


class LatencySensitiveSignalProcessor(LoggerMixin):
    """延迟敏感的信号处理器
    
    特性：
    1. 优先级队列处理
    2. 实时延迟监控
    3. 信号融合和过滤
    4. 自适应阈值调整
    5. 异步并行处理
    """
    
    def __init__(self, 
                 max_queue_size: int = 10000,
                 latency_target_ms: float = 1.0,
                 processing_workers: int = 4):
        self.max_queue_size = max_queue_size
        self.latency_target_ms = latency_target_ms
        self.processing_workers = processing_workers
        
        # 信号队列 - 按优先级分层
        self.signal_queues: Dict[SignalPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size // 4)
            for priority in SignalPriority
        }
        
        # 信号历史和缓存
        self.signal_history: Dict[str, Deque[MicrostructureSignal]] = {}
        self.last_signals: Dict[str, MicrostructureSignal] = {}
        
        # 融合窗口
        self.fusion_window_ms: float = 10.0  # 10ms融合窗口
        self.pending_fusion: Dict[str, List[MicrostructureSignal]] = {}
        
        # 阈值和过滤器
        self.signal_thresholds: Dict[str, float] = {
            "imbalance": 0.3,
            "toxicity": 0.5,
            "price_anomaly": 0.4,
            "volume_spike": 0.6,
            "spread_anomaly": 0.5
        }
        
        # 信号权重
        self.signal_weights: Dict[str, float] = {
            "imbalance": 1.0,
            "toxicity": 1.5,    # 毒性信号权重更高
            "price_anomaly": 1.2,
            "volume_spike": 0.8,
            "spread_anomaly": 0.9
        }
        
        # 动作生成器
        self.action_generators: Dict[str, Callable] = {}
        self.action_callbacks: List[Callable] = []
        
        # 性能监控
        self.metrics = ProcessingMetrics()
        self.processing_times: Deque[float] = deque(maxlen=1000)
        
        # 运行状态
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._fusion_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """启动信号处理器"""
        if self._running:
            return
            
        self._running = True
        
        # 启动处理工作线程
        for i in range(self.processing_workers):
            worker = asyncio.create_task(self._processing_worker(i))
            self._workers.append(worker)
        
        # 启动信号融合任务
        self._fusion_task = asyncio.create_task(self._fusion_loop())
        
        self.log_info(f"Signal processor started with {self.processing_workers} workers")
        
    async def stop(self):
        """停止信号处理器"""
        self._running = False
        
        # 停止融合任务
        if self._fusion_task:
            self._fusion_task.cancel()
            try:
                await self._fusion_task
            except asyncio.CancelledError:
                pass
        
        # 停止工作线程
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        self.log_info("Signal processor stopped")
        
    async def process_signal(self, signal: MicrostructureSignal) -> bool:
        """处理单个信号"""
        start_time = time.perf_counter()
        
        try:
            self.metrics.total_signals += 1
            
            # 信号过滤
            if not self._filter_signal(signal):
                return False
            
            # 确定优先级
            priority = self._determine_priority(signal)
            
            # 添加到相应队列
            queue = self.signal_queues[priority]
            
            try:
                # 非阻塞添加
                queue.put_nowait(signal)
                
                # 更新历史
                self._update_signal_history(signal)
                
                # 记录处理时间
                processing_time = (time.perf_counter() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                return True
                
            except asyncio.QueueFull:
                self.metrics.dropped_signals += 1
                self.log_warning(f"Signal queue full, dropping signal: {signal.signal_type}")
                return False
                
        except Exception as e:
            self.log_error(f"Error processing signal: {e}")
            return False
    
    async def process_signals_batch(self, signals: List[MicrostructureSignal]) -> int:
        """批量处理信号"""
        processed = 0
        
        # 按优先级分组
        grouped_signals: Dict[SignalPriority, List[MicrostructureSignal]] = {
            priority: [] for priority in SignalPriority
        }
        
        for signal in signals:
            if self._filter_signal(signal):
                priority = self._determine_priority(signal)
                grouped_signals[priority].append(signal)
        
        # 按优先级处理
        for priority in SignalPriority:
            for signal in grouped_signals[priority]:
                if await self.process_signal(signal):
                    processed += 1
        
        return processed
    
    def _filter_signal(self, signal: MicrostructureSignal) -> bool:
        """信号过滤"""
        # 强度阈值过滤
        threshold = self.signal_thresholds.get(signal.signal_type, 0.5)
        if abs(signal.strength) < threshold:
            return False
        
        # 置信度过滤
        if signal.confidence < 0.3:
            return False
        
        # 时间戳检查（避免过期信号）
        current_time = time.time()
        if current_time - signal.timestamp > 5.0:  # 5秒过期
            return False
        
        # 重复信号过滤
        last_signal = self.last_signals.get(signal.symbol)
        if last_signal and self._is_duplicate_signal(signal, last_signal):
            return False
        
        return True
    
    def _is_duplicate_signal(self, signal: MicrostructureSignal, last_signal: MicrostructureSignal) -> bool:
        """检查是否为重复信号"""
        if signal.signal_type != last_signal.signal_type:
            return False
        
        # 时间间隔检查
        time_diff = signal.timestamp - last_signal.timestamp
        if time_diff < 0.1:  # 100ms内的类似信号视为重复
            strength_diff = abs(signal.strength - last_signal.strength)
            if strength_diff < 0.1:  # 强度差异小于0.1
                return True
        
        return False
    
    def _determine_priority(self, signal: MicrostructureSignal) -> SignalPriority:
        """确定信号优先级"""
        # 基于信号类型和强度确定优先级
        if signal.signal_type == "toxicity" and signal.strength > 0.8:
            return SignalPriority.CRITICAL
        
        if signal.signal_type == "price_anomaly" and abs(signal.strength) > 0.8:
            return SignalPriority.CRITICAL
        
        if signal.signal_type == "imbalance" and abs(signal.strength) > 0.7:
            return SignalPriority.HIGH
        
        if signal.signal_type == "volume_spike" and signal.strength > 0.8:
            return SignalPriority.HIGH
        
        if signal.confidence > 0.8:
            return SignalPriority.HIGH
        
        if signal.confidence > 0.6:
            return SignalPriority.MEDIUM
        
        return SignalPriority.LOW
    
    def _update_signal_history(self, signal: MicrostructureSignal):
        """更新信号历史"""
        symbol = signal.symbol
        
        if symbol not in self.signal_history:
            self.signal_history[symbol] = deque(maxlen=100)
        
        self.signal_history[symbol].append(signal)
        self.last_signals[symbol] = signal
    
    async def _processing_worker(self, worker_id: int):
        """信号处理工作线程"""
        while self._running:
            try:
                signal = None
                
                # 按优先级从队列获取信号
                for priority in SignalPriority:
                    queue = self.signal_queues[priority]
                    try:
                        signal = queue.get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue
                
                if signal is None:
                    await asyncio.sleep(0.001)  # 1ms休眠
                    continue
                
                # 处理信号
                await self._process_individual_signal(signal, worker_id)
                
            except Exception as e:
                self.log_error(f"Error in processing worker {worker_id}: {e}")
                await asyncio.sleep(0.01)
    
    async def _process_individual_signal(self, signal: MicrostructureSignal, worker_id: int):
        """处理单个信号"""
        start_time = time.perf_counter()
        
        try:
            # 生成交易动作
            actions = await self._generate_actions(signal)
            
            if actions:
                # 执行回调
                for action in actions:
                    for callback in self.action_callbacks:
                        try:
                            await callback(action)
                        except Exception as e:
                            self.log_error(f"Error in action callback: {e}")
                
                self.metrics.actions_generated += len(actions)
            
            self.metrics.processed_signals += 1
            
            # 更新延迟统计
            processing_latency = (time.perf_counter() - start_time) * 1000
            self.metrics.avg_processing_latency = (
                (self.metrics.avg_processing_latency * (self.metrics.processed_signals - 1) + processing_latency) 
                / self.metrics.processed_signals
            )
            self.metrics.max_processing_latency = max(
                self.metrics.max_processing_latency, processing_latency
            )
            
            # 延迟警告
            if processing_latency > self.latency_target_ms:
                self.log_debug(f"High processing latency: {processing_latency:.2f}ms for {signal.signal_type}")
            
        except Exception as e:
            self.log_error(f"Error processing signal {signal.signal_type}: {e}")
    
    async def _generate_actions(self, signal: MicrostructureSignal) -> List[SignalAction]:
        """生成交易动作"""
        actions = []
        
        # 检查是否有自定义动作生成器
        if signal.signal_type in self.action_generators:
            custom_actions = await self.action_generators[signal.signal_type](signal)
            if custom_actions:
                actions.extend(custom_actions)
                return actions
        
        # 默认动作生成逻辑
        action = await self._default_action_generation(signal)
        if action:
            actions.append(action)
        
        return actions
    
    async def _default_action_generation(self, signal: MicrostructureSignal) -> Optional[SignalAction]:
        """默认动作生成"""
        if signal.signal_type == "imbalance":
            # 订单簿失衡信号
            if signal.strength > 0.5:  # 买盘失衡
                return SignalAction(
                    action_type=ActionType.BUY,
                    symbol=signal.symbol,
                    priority=SignalPriority.HIGH,
                    urgency_score=abs(signal.strength),
                    confidence=signal.confidence,
                    metadata={"signal_type": signal.signal_type, "strength": signal.strength}
                )
            elif signal.strength < -0.5:  # 卖盘失衡
                return SignalAction(
                    action_type=ActionType.SELL,
                    symbol=signal.symbol,
                    priority=SignalPriority.HIGH,
                    urgency_score=abs(signal.strength),
                    confidence=signal.confidence,
                    metadata={"signal_type": signal.signal_type, "strength": signal.strength}
                )
        
        elif signal.signal_type == "toxicity":
            # 流毒性信号 - 通常建议减少交易
            if signal.strength > 0.7:
                return SignalAction(
                    action_type=ActionType.CANCEL,
                    symbol=signal.symbol,
                    priority=SignalPriority.CRITICAL,
                    urgency_score=signal.strength,
                    confidence=signal.confidence,
                    metadata={"signal_type": signal.signal_type, "reason": "high_toxicity"}
                )
        
        elif signal.signal_type == "price_anomaly":
            # 价格异常信号
            if abs(signal.strength) > 0.6:
                action_type = ActionType.BUY if signal.strength < 0 else ActionType.SELL
                return SignalAction(
                    action_type=action_type,
                    symbol=signal.symbol,
                    priority=SignalPriority.HIGH,
                    urgency_score=abs(signal.strength),
                    confidence=signal.confidence,
                    metadata={"signal_type": signal.signal_type, "anomaly_strength": signal.strength}
                )
        
        elif signal.signal_type == "volume_spike":
            # 成交量异常信号
            if signal.strength > 0.7:
                return SignalAction(
                    action_type=ActionType.ADJUST,
                    symbol=signal.symbol,
                    priority=SignalPriority.MEDIUM,
                    urgency_score=signal.strength,
                    confidence=signal.confidence,
                    metadata={"signal_type": signal.signal_type, "adjust_type": "increase_spread"}
                )
        
        return None
    
    async def _fusion_loop(self):
        """信号融合循环"""
        while self._running:
            try:
                current_time = time.time()
                
                # 处理待融合信号
                for symbol, signals in list(self.pending_fusion.items()):
                    if not signals:
                        continue
                    
                    # 检查融合窗口
                    oldest_signal = signals[0]
                    if (current_time - oldest_signal.timestamp) * 1000 > self.fusion_window_ms:
                        # 执行信号融合
                        fused_signal = await self._fuse_signals(symbol, signals)
                        if fused_signal:
                            await self.process_signal(fused_signal)
                        
                        # 清空待融合信号
                        self.pending_fusion[symbol] = []
                
                await asyncio.sleep(self.fusion_window_ms / 1000)
                
            except Exception as e:
                self.log_error(f"Error in fusion loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _fuse_signals(self, symbol: str, signals: List[MicrostructureSignal]) -> Optional[MicrostructureSignal]:
        """融合多个信号"""
        if not signals:
            return None
        
        # 按信号类型分组
        signal_groups: Dict[str, List[MicrostructureSignal]] = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)
        
        # 选择最强信号类型
        max_weight = 0
        dominant_type = None
        combined_strength = 0
        combined_confidence = 0
        
        for signal_type, type_signals in signal_groups.items():
            weight = self.signal_weights.get(signal_type, 1.0)
            avg_strength = sum(s.strength for s in type_signals) / len(type_signals)
            avg_confidence = sum(s.confidence for s in type_signals) / len(type_signals)
            
            weighted_score = weight * abs(avg_strength) * avg_confidence
            
            if weighted_score > max_weight:
                max_weight = weighted_score
                dominant_type = signal_type
                combined_strength = avg_strength
                combined_confidence = avg_confidence
        
        if dominant_type:
            return MicrostructureSignal(
                signal_type=f"fused_{dominant_type}",
                symbol=symbol,
                timestamp=time.time(),
                strength=combined_strength,
                confidence=min(1.0, combined_confidence * 1.1),  # 融合信号置信度略高
                metadata={
                    "fused_from": list(signal_groups.keys()),
                    "signal_count": len(signals),
                    "fusion_window_ms": self.fusion_window_ms
                }
            )
        
        return None
    
    def register_action_generator(self, signal_type: str, generator: Callable):
        """注册自定义动作生成器"""
        self.action_generators[signal_type] = generator
    
    def add_action_callback(self, callback: Callable):
        """添加动作回调"""
        self.action_callbacks.append(callback)
    
    def update_threshold(self, signal_type: str, threshold: float):
        """更新信号阈值"""
        self.signal_thresholds[signal_type] = threshold
    
    def update_weight(self, signal_type: str, weight: float):
        """更新信号权重"""
        self.signal_weights[signal_type] = weight
    
    def get_metrics(self) -> ProcessingMetrics:
        """获取处理指标"""
        # 更新队列大小
        self.metrics.queue_size = sum(queue.qsize() for queue in self.signal_queues.values())
        return self.metrics
    
    def get_signal_stats(self) -> Dict[str, any]:
        """获取信号统计"""
        if not self.processing_times:
            return {}
        
        sorted_times = sorted(self.processing_times)
        
        return {
            "total_signals": self.metrics.total_signals,
            "processed_signals": self.metrics.processed_signals,
            "dropped_signals": self.metrics.dropped_signals,
            "processing_rate": self.metrics.processed_signals / max(1, self.metrics.total_signals),
            "latency_ms": {
                "avg": self.metrics.avg_processing_latency,
                "max": self.metrics.max_processing_latency,
                "p95": sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0,
                "p99": sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
            },
            "queue_sizes": {
                priority.name: queue.qsize() 
                for priority, queue in self.signal_queues.items()
            },
            "actions_generated": self.metrics.actions_generated
        }
    
    def get_status(self) -> Dict[str, any]:
        """获取处理器状态"""
        return {
            "running": self._running,
            "workers": len(self._workers),
            "fusion_enabled": self._fusion_task is not None,
            "latency_target_ms": self.latency_target_ms,
            "metrics": self.get_metrics().__dict__,
            "signal_stats": self.get_signal_stats(),
            "thresholds": dict(self.signal_thresholds),
            "weights": dict(self.signal_weights)
        }