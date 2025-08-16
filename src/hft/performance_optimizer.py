import asyncio
import os
import sys
import time
import psutil
import mmap
import struct
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from decimal import Decimal
import numpy as np

from src.utils.logger import LoggerMixin

# 尝试导入uvloop（如果可用）
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False


@dataclass
class PerformanceConfig:
    """性能优化配置"""
    # 事件循环优化
    use_uvloop: bool = True
    
    # CPU亲和性设置
    cpu_affinity_enabled: bool = True
    dedicated_cores: List[int] = field(default_factory=list)  # 专用CPU核心
    
    # 内存优化
    memory_pool_size: int = 1024 * 1024 * 100  # 100MB内存池
    zero_copy_enabled: bool = True
    prealloc_buffers: bool = True
    
    # 线程优化
    thread_priority_boost: bool = True
    max_worker_threads: int = 4
    
    # 网络优化
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True
    socket_buffer_size: int = 65536
    
    # 调度优化
    scheduler_policy: str = "FIFO"  # FIFO, RR, OTHER
    scheduler_priority: int = 99    # 最高优先级
    
    # GC优化
    gc_optimization: bool = True
    gc_disable_during_trading: bool = True


@dataclass
class PerformanceMetrics:
    """性能指标"""
    event_loop_latency_us: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_latency_us: float = 0.0
    gc_collections: int = 0
    gc_time_ms: float = 0.0
    context_switches: int = 0
    page_faults: int = 0
    thread_count: int = 0


class ZeroCopyBuffer:
    """零拷贝缓冲区"""
    
    def __init__(self, size: int):
        self.size = size
        self.data = bytearray(size)
        self.position = 0
        self.limit = size
        
    def clear(self):
        """清空缓冲区"""
        self.position = 0
        self.limit = self.size
        
    def flip(self):
        """翻转缓冲区（准备读取）"""
        self.limit = self.position
        self.position = 0
        
    def compact(self):
        """压缩缓冲区"""
        remaining = self.limit - self.position
        if remaining > 0:
            self.data[0:remaining] = self.data[self.position:self.limit]
        self.position = remaining
        self.limit = self.size
        
    def put_bytes(self, data: bytes) -> bool:
        """写入字节数据"""
        if self.position + len(data) > self.limit:
            return False
        self.data[self.position:self.position + len(data)] = data
        self.position += len(data)
        return True
        
    def get_bytes(self, length: int) -> Optional[bytes]:
        """读取字节数据"""
        if self.position + length > self.limit:
            return None
        data = bytes(self.data[self.position:self.position + length])
        self.position += length
        return data
        
    def put_double(self, value: float):
        """写入双精度浮点数"""
        data = struct.pack('d', value)
        return self.put_bytes(data)
        
    def get_double(self) -> Optional[float]:
        """读取双精度浮点数"""
        data = self.get_bytes(8)
        if data:
            return struct.unpack('d', data)[0]
        return None
        
    def put_int64(self, value: int):
        """写入64位整数"""
        data = struct.pack('q', value)
        return self.put_bytes(data)
        
    def get_int64(self) -> Optional[int]:
        """读取64位整数"""
        data = self.get_bytes(8)
        if data:
            return struct.unpack('q', data)[0]
        return None
        
    def remaining(self) -> int:
        """剩余可读字节数"""
        return self.limit - self.position
        
    def has_remaining(self) -> bool:
        """是否还有可读数据"""
        return self.position < self.limit


class MemoryPool:
    """内存池"""
    
    def __init__(self, pool_size: int, buffer_size: int = 4096):
        self.pool_size = pool_size
        self.buffer_size = buffer_size
        self.num_buffers = pool_size // buffer_size
        
        # 创建内存映射
        self.mmap_fd = -1
        try:
            self.memory = mmap.mmap(-1, pool_size)
        except Exception:
            # 回退到普通内存分配
            self.memory = bytearray(pool_size)
        
        # 可用缓冲区队列
        self.available_buffers: deque = deque()
        self.used_buffers: Dict[int, ZeroCopyBuffer] = {}
        
        # 初始化缓冲区
        for i in range(self.num_buffers):
            offset = i * buffer_size
            buffer = ZeroCopyBuffer(buffer_size)
            buffer.data = memoryview(self.memory)[offset:offset + buffer_size]
            self.available_buffers.append(buffer)
        
    def acquire_buffer(self) -> Optional[ZeroCopyBuffer]:
        """获取缓冲区"""
        if self.available_buffers:
            buffer = self.available_buffers.popleft()
            buffer.clear()
            buffer_id = id(buffer)
            self.used_buffers[buffer_id] = buffer
            return buffer
        return None
        
    def release_buffer(self, buffer: ZeroCopyBuffer):
        """释放缓冲区"""
        buffer_id = id(buffer)
        if buffer_id in self.used_buffers:
            del self.used_buffers[buffer_id]
            buffer.clear()
            self.available_buffers.append(buffer)
            
    def get_stats(self) -> Dict[str, int]:
        """获取内存池统计"""
        return {
            "total_buffers": self.num_buffers,
            "available_buffers": len(self.available_buffers),
            "used_buffers": len(self.used_buffers),
            "utilization_percent": int(len(self.used_buffers) / self.num_buffers * 100)
        }


class HFTPerformanceOptimizer(LoggerMixin):
    """高频交易性能优化器"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=1000)
        
        # 内存池
        self.memory_pool: Optional[MemoryPool] = None
        
        # 原始事件循环
        self.original_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 性能监控任务
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = False
        
        # 进程信息
        self.process = psutil.Process()
        
    async def initialize(self):
        """初始化性能优化器"""
        # 1. 优化事件循环
        await self._optimize_event_loop()
        
        # 2. 设置CPU亲和性
        await self._setup_cpu_affinity()
        
        # 3. 初始化内存池
        await self._setup_memory_pool()
        
        # 4. 优化进程优先级
        await self._optimize_process_priority()
        
        # 5. 优化垃圾回收
        await self._optimize_garbage_collection()
        
        # 6. 启动性能监控
        await self._start_performance_monitoring()
        
        self.log_info("HFT Performance Optimizer initialized")
        
    async def _optimize_event_loop(self):
        """优化事件循环"""
        if not self.config.use_uvloop or not UVLOOP_AVAILABLE:
            self.log_info("Using default asyncio event loop")
            return
            
        try:
            # 保存原始事件循环
            self.original_loop = asyncio.get_event_loop()
            
            # 安装uvloop
            uvloop.install()
            
            # 创建新的事件循环
            loop = uvloop.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.log_info("Successfully switched to uvloop for better performance")
            
        except Exception as e:
            self.log_error(f"Failed to setup uvloop: {e}")
            
    async def _setup_cpu_affinity(self):
        """设置CPU亲和性"""
        if not self.config.cpu_affinity_enabled:
            return
            
        try:
            # 获取可用CPU核心
            available_cores = list(range(psutil.cpu_count()))
            
            if self.config.dedicated_cores:
                # 使用指定的CPU核心
                target_cores = [core for core in self.config.dedicated_cores if core in available_cores]
            else:
                # 自动选择高性能核心（通常是最后几个）
                target_cores = available_cores[-min(4, len(available_cores)):]
            
            if target_cores:
                self.process.cpu_affinity(target_cores)
                self.log_info(f"Set CPU affinity to cores: {target_cores}")
            
        except Exception as e:
            self.log_error(f"Failed to set CPU affinity: {e}")
            
    async def _setup_memory_pool(self):
        """设置内存池"""
        if not self.config.prealloc_buffers:
            return
            
        try:
            self.memory_pool = MemoryPool(
                pool_size=self.config.memory_pool_size,
                buffer_size=4096
            )
            self.log_info(f"Memory pool initialized: {self.config.memory_pool_size // (1024*1024)}MB")
            
        except Exception as e:
            self.log_error(f"Failed to setup memory pool: {e}")
            
    async def _optimize_process_priority(self):
        """优化进程优先级"""
        if not self.config.thread_priority_boost:
            return
            
        try:
            # 设置进程优先级
            if sys.platform.startswith('linux'):
                # Linux系统
                import os
                try:
                    # 尝试设置实时调度
                    if self.config.scheduler_policy == "FIFO":
                        policy = os.SCHED_FIFO
                    elif self.config.scheduler_policy == "RR":
                        policy = os.SCHED_RR
                    else:
                        policy = os.SCHED_OTHER
                    
                    # 需要root权限
                    os.sched_setscheduler(0, policy, os.sched_param(self.config.scheduler_priority))
                    self.log_info(f"Set scheduler policy to {self.config.scheduler_policy}")
                    
                except PermissionError:
                    # 降级到nice值设置
                    os.nice(-10)  # 提高优先级
                    self.log_info("Set process nice priority to -10")
                    
            elif sys.platform == 'win32':
                # Windows系统
                import win32api
                import win32process
                win32process.SetPriorityClass(win32api.GetCurrentProcess(), win32process.HIGH_PRIORITY_CLASS)
                self.log_info("Set Windows process priority to HIGH")
                
        except Exception as e:
            self.log_warning(f"Could not set process priority: {e}")
            
    async def _optimize_garbage_collection(self):
        """优化垃圾回收"""
        if not self.config.gc_optimization:
            return
            
        try:
            import gc
            
            # 调整GC阈值
            gc.set_threshold(1000, 15, 15)  # 减少GC频率
            
            # 禁用自动GC（在交易时手动控制）
            if self.config.gc_disable_during_trading:
                gc.disable()
                self.log_info("Disabled automatic garbage collection")
            
            self.log_info("Optimized garbage collection settings")
            
        except Exception as e:
            self.log_error(f"Failed to optimize GC: {e}")
            
    async def _start_performance_monitoring(self):
        """启动性能监控"""
        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def _monitoring_loop(self):
        """性能监控循环"""
        while self._monitoring_enabled:
            try:
                await self._collect_metrics()
                await asyncio.sleep(1.0)  # 每秒收集一次指标
                
            except Exception as e:
                self.log_error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5.0)
                
    async def _collect_metrics(self):
        """收集性能指标"""
        try:
            # CPU使用率
            self.metrics.cpu_usage_percent = self.process.cpu_percent()
            
            # 内存使用
            memory_info = self.process.memory_info()
            self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            # 线程数
            self.metrics.thread_count = self.process.num_threads()
            
            # 上下文切换
            ctx_switches = self.process.num_ctx_switches()
            self.metrics.context_switches = ctx_switches.voluntary + ctx_switches.involuntary
            
            # 事件循环延迟
            start_time = time.perf_counter()
            await asyncio.sleep(0)  # 让出控制权
            loop_latency = (time.perf_counter() - start_time) * 1_000_000  # 微秒
            self.metrics.event_loop_latency_us = loop_latency
            
            # GC统计
            import gc
            self.metrics.gc_collections = len(gc.get_stats())
            
            # 保存历史
            self.metrics_history.append({
                "timestamp": time.time(),
                "cpu_percent": self.metrics.cpu_usage_percent,
                "memory_mb": self.metrics.memory_usage_mb,
                "loop_latency_us": self.metrics.event_loop_latency_us,
                "threads": self.metrics.thread_count
            })
            
        except Exception as e:
            self.log_error(f"Error collecting metrics: {e}")
            
    def get_buffer(self) -> Optional[ZeroCopyBuffer]:
        """获取零拷贝缓冲区"""
        if self.memory_pool:
            return self.memory_pool.acquire_buffer()
        return None
        
    def release_buffer(self, buffer: ZeroCopyBuffer):
        """释放缓冲区"""
        if self.memory_pool:
            self.memory_pool.release_buffer(buffer)
            
    def create_zero_copy_message(self, symbol: str, price: float, volume: float, timestamp: int) -> Optional[ZeroCopyBuffer]:
        """创建零拷贝消息"""
        buffer = self.get_buffer()
        if not buffer:
            return None
            
        try:
            # 消息格式：symbol(8字节) + price(8字节) + volume(8字节) + timestamp(8字节)
            symbol_bytes = symbol.encode('utf-8')[:8].ljust(8, b'\x00')
            
            buffer.put_bytes(symbol_bytes)
            buffer.put_double(price)
            buffer.put_double(volume)
            buffer.put_int64(timestamp)
            
            buffer.flip()  # 准备读取
            return buffer
            
        except Exception as e:
            self.log_error(f"Error creating zero-copy message: {e}")
            self.release_buffer(buffer)
            return None
            
    def parse_zero_copy_message(self, buffer: ZeroCopyBuffer) -> Optional[Dict[str, Any]]:
        """解析零拷贝消息"""
        try:
            symbol_bytes = buffer.get_bytes(8)
            price = buffer.get_double()
            volume = buffer.get_double()
            timestamp = buffer.get_int64()
            
            if all(x is not None for x in [symbol_bytes, price, volume, timestamp]):
                symbol = symbol_bytes.rstrip(b'\x00').decode('utf-8')
                return {
                    "symbol": symbol,
                    "price": price,
                    "volume": volume,
                    "timestamp": timestamp
                }
            return None
            
        except Exception as e:
            self.log_error(f"Error parsing zero-copy message: {e}")
            return None
            
    def optimize_socket(self, sock):
        """优化socket设置"""
        try:
            import socket
            
            if self.config.tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
            if self.config.tcp_keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
            # 设置缓冲区大小
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_buffer_size)
            
        except Exception as e:
            self.log_error(f"Error optimizing socket: {e}")
            
    async def manual_gc(self):
        """手动垃圾回收"""
        if self.config.gc_disable_during_trading:
            import gc
            start_time = time.perf_counter()
            collected = gc.collect()
            gc_time = (time.perf_counter() - start_time) * 1000
            self.metrics.gc_time_ms = gc_time
            
            if collected > 0:
                self.log_debug(f"Manual GC collected {collected} objects in {gc_time:.2f}ms")
                
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = list(self.metrics_history)[-60:]  # 最近60秒
        
        cpu_values = [m["cpu_percent"] for m in recent_metrics]
        memory_values = [m["memory_mb"] for m in recent_metrics]
        latency_values = [m["loop_latency_us"] for m in recent_metrics]
        
        report = {
            "current_metrics": {
                "cpu_percent": self.metrics.cpu_usage_percent,
                "memory_mb": self.metrics.memory_usage_mb,
                "loop_latency_us": self.metrics.event_loop_latency_us,
                "thread_count": self.metrics.thread_count,
                "context_switches": self.metrics.context_switches
            },
            "averages_1min": {
                "cpu_percent": np.mean(cpu_values) if cpu_values else 0,
                "memory_mb": np.mean(memory_values) if memory_values else 0,
                "loop_latency_us": np.mean(latency_values) if latency_values else 0
            },
            "maximums_1min": {
                "cpu_percent": np.max(cpu_values) if cpu_values else 0,
                "memory_mb": np.max(memory_values) if memory_values else 0,
                "loop_latency_us": np.max(latency_values) if latency_values else 0
            },
            "optimizations": {
                "uvloop_enabled": UVLOOP_AVAILABLE and self.config.use_uvloop,
                "cpu_affinity_set": self.config.cpu_affinity_enabled,
                "memory_pool_active": self.memory_pool is not None,
                "gc_optimized": self.config.gc_optimization
            }
        }
        
        if self.memory_pool:
            report["memory_pool"] = self.memory_pool.get_stats()
            
        return report
        
    async def shutdown(self):
        """关闭性能优化器"""
        self._monitoring_enabled = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 恢复GC
        if self.config.gc_disable_during_trading:
            import gc
            gc.enable()
        
        # 恢复原始事件循环
        if self.original_loop:
            asyncio.set_event_loop(self.original_loop)
            
        self.log_info("Performance optimizer shutdown complete")
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            "uvloop": {
                "available": UVLOOP_AVAILABLE,
                "enabled": self.config.use_uvloop and UVLOOP_AVAILABLE,
                "current_loop": str(type(asyncio.get_event_loop()))
            },
            "cpu_affinity": {
                "enabled": self.config.cpu_affinity_enabled,
                "current_affinity": self.process.cpu_affinity() if hasattr(self.process, 'cpu_affinity') else None
            },
            "memory_pool": {
                "enabled": self.config.prealloc_buffers,
                "size_mb": self.config.memory_pool_size // (1024*1024),
                "stats": self.memory_pool.get_stats() if self.memory_pool else None
            },
            "gc_optimization": {
                "enabled": self.config.gc_optimization,
                "disabled_during_trading": self.config.gc_disable_during_trading
            },
            "monitoring": {
                "active": self._monitoring_enabled,
                "metrics_collected": len(self.metrics_history)
            }
        }