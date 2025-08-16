import asyncio
import socket
import time
import struct
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import ssl

from src.hft.performance_optimizer import ZeroCopyBuffer, HFTPerformanceOptimizer
from src.utils.logger import LoggerMixin


@dataclass
class NetworkConfig:
    """网络优化配置"""
    # TCP优化
    tcp_nodelay: bool = True
    tcp_quickack: bool = True  # Linux specific
    tcp_cork: bool = False     # Linux specific
    
    # 缓冲区设置
    send_buffer_size: int = 1024 * 1024      # 1MB
    recv_buffer_size: int = 1024 * 1024      # 1MB
    
    # 连接优化
    keepalive_enabled: bool = True
    keepalive_idle: int = 60
    keepalive_interval: int = 10
    keepalive_probes: int = 3
    
    # 超时设置
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    
    # 高级优化
    zero_copy_send: bool = True
    message_batching: bool = True
    batch_size: int = 100
    batch_timeout_ms: float = 1.0
    
    # 连接池
    connection_pool_size: int = 10
    connection_reuse: bool = True


@dataclass
class ConnectionMetrics:
    """连接指标"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0


class OptimizedConnection:
    """优化的网络连接"""
    
    def __init__(self, 
                 host: str,
                 port: int,
                 config: NetworkConfig,
                 performance_optimizer: Optional[HFTPerformanceOptimizer] = None):
        self.host = host
        self.port = port
        self.config = config
        self.performance_optimizer = performance_optimizer
        
        # 连接状态
        self.socket: Optional[socket.socket] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connected = False
        
        # 缓冲区管理
        self.send_buffer = deque()
        self.pending_messages = deque()
        
        # 性能指标
        self.metrics = ConnectionMetrics()
        self.latency_samples = deque(maxlen=1000)
        
        # 异步任务
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._batch_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """建立连接"""
        try:
            # 创建socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 应用优化设置
            self._optimize_socket()
            
            # 建立连接
            await asyncio.wait_for(
                asyncio.get_event_loop().sock_connect(self.socket, (self.host, self.port)),
                timeout=self.config.connect_timeout
            )
            
            # 创建流
            self.reader, self.writer = await asyncio.open_connection(
                sock=self.socket
            )
            
            self.connected = True
            self.metrics.total_connections += 1
            self.metrics.active_connections += 1
            
            # 启动后台任务
            if self.config.message_batching:
                self._batch_task = asyncio.create_task(self._batch_sender())
            
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            return True
            
        except Exception as e:
            self.metrics.failed_connections += 1
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
            raise e
    
    def _optimize_socket(self):
        """优化socket设置"""
        if not self.socket:
            return
            
        try:
            # TCP_NODELAY - 禁用Nagle算法
            if self.config.tcp_nodelay:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 设置缓冲区大小
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.send_buffer_size)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.recv_buffer_size)
            
            # Keepalive设置
            if self.config.keepalive_enabled:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Linux特定设置
                if hasattr(socket, 'TCP_KEEPIDLE'):
                    self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.keepalive_idle)
                if hasattr(socket, 'TCP_KEEPINTVL'):
                    self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.keepalive_interval)
                if hasattr(socket, 'TCP_KEEPCNT'):
                    self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.keepalive_probes)
            
            # Linux特定TCP优化
            if hasattr(socket, 'TCP_QUICKACK') and self.config.tcp_quickack:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
            if hasattr(socket, 'TCP_CORK') and self.config.tcp_cork:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_CORK, 1)
            
            # 应用性能优化器的socket设置
            if self.performance_optimizer:
                self.performance_optimizer.optimize_socket(self.socket)
                
        except Exception as e:
            # 优化失败不影响连接
            pass
    
    async def send_message(self, data: bytes, priority: bool = False) -> bool:
        """发送消息"""
        if not self.connected or not self.writer:
            return False
            
        try:
            if self.config.message_batching and not priority:
                # 批量发送
                self.pending_messages.append(data)
            else:
                # 立即发送
                await self._send_immediate(data)
            
            return True
            
        except Exception as e:
            self.metrics.errors += 1
            return False
    
    async def send_zero_copy_message(self, buffer: ZeroCopyBuffer, priority: bool = False) -> bool:
        """发送零拷贝消息"""
        if not self.connected or not self.writer:
            return False
            
        try:
            # 从缓冲区读取数据
            data = bytes(buffer.data[:buffer.limit])
            return await self.send_message(data, priority)
            
        except Exception as e:
            self.metrics.errors += 1
            return False
    
    async def _send_immediate(self, data: bytes):
        """立即发送数据"""
        start_time = time.perf_counter()
        
        # 添加消息长度前缀
        length_prefix = struct.pack('I', len(data))
        full_message = length_prefix + data
        
        self.writer.write(full_message)
        await self.writer.drain()
        
        # 更新指标
        self.metrics.bytes_sent += len(full_message)
        self.metrics.messages_sent += 1
        
        # 记录延迟
        latency = (time.perf_counter() - start_time) * 1000
        self.latency_samples.append(latency)
        self._update_latency_stats()
    
    async def _batch_sender(self):
        """批量发送任务"""
        while self.connected:
            try:
                # 等待批处理超时或收集足够消息
                await asyncio.sleep(self.config.batch_timeout_ms / 1000)
                
                if self.pending_messages:
                    batch = []
                    
                    # 收集待发送消息
                    for _ in range(min(self.config.batch_size, len(self.pending_messages))):
                        if self.pending_messages:
                            batch.append(self.pending_messages.popleft())
                    
                    if batch:
                        await self._send_batch(batch)
                
            except Exception as e:
                self.metrics.errors += 1
                await asyncio.sleep(0.1)
    
    async def _send_batch(self, messages: List[bytes]):
        """发送批量消息"""
        if not self.writer:
            return
            
        start_time = time.perf_counter()
        
        try:
            # 组装批量消息
            batch_data = bytearray()
            
            for message in messages:
                length_prefix = struct.pack('I', len(message))
                batch_data.extend(length_prefix)
                batch_data.extend(message)
            
            # 一次性发送
            self.writer.write(batch_data)
            await self.writer.drain()
            
            # 更新指标
            self.metrics.bytes_sent += len(batch_data)
            self.metrics.messages_sent += len(messages)
            
            # 记录延迟
            latency = (time.perf_counter() - start_time) * 1000
            self.latency_samples.append(latency)
            self._update_latency_stats()
            
        except Exception as e:
            self.metrics.errors += 1
            # 重新加入队列
            for message in reversed(messages):
                self.pending_messages.appendleft(message)
    
    async def _receive_loop(self):
        """接收循环"""
        while self.connected and self.reader:
            try:
                # 读取消息长度
                length_data = await asyncio.wait_for(
                    self.reader.readexactly(4),
                    timeout=self.config.read_timeout
                )
                
                message_length = struct.unpack('I', length_data)[0]
                
                # 读取消息内容
                message_data = await asyncio.wait_for(
                    self.reader.readexactly(message_length),
                    timeout=self.config.read_timeout
                )
                
                # 更新指标
                self.metrics.bytes_received += len(length_data) + len(message_data)
                self.metrics.messages_received += 1
                
                # 处理消息（由子类实现）
                await self._handle_received_message(message_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.errors += 1
                break
    
    async def _handle_received_message(self, data: bytes):
        """处理接收到的消息（子类可重写）"""
        pass
    
    def _update_latency_stats(self):
        """更新延迟统计"""
        if self.latency_samples:
            self.metrics.avg_latency_ms = sum(self.latency_samples) / len(self.latency_samples)
            self.metrics.max_latency_ms = max(self.latency_samples)
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
        
        # 取消后台任务
        for task in [self._send_task, self._receive_task, self._batch_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 关闭连接
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        
        if self.socket:
            self.socket.close()
        
        self.metrics.active_connections -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "metrics": {
                "total_connections": self.metrics.total_connections,
                "active_connections": self.metrics.active_connections,
                "failed_connections": self.metrics.failed_connections,
                "bytes_sent": self.metrics.bytes_sent,
                "bytes_received": self.metrics.bytes_received,
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "max_latency_ms": self.metrics.max_latency_ms,
                "errors": self.metrics.errors
            },
            "pending_messages": len(self.pending_messages),
            "latency_samples": len(self.latency_samples)
        }


class ConnectionPool:
    """连接池"""
    
    def __init__(self, config: NetworkConfig, performance_optimizer: Optional[HFTPerformanceOptimizer] = None):
        self.config = config
        self.performance_optimizer = performance_optimizer
        
        # 连接池
        self.pools: Dict[str, List[OptimizedConnection]] = {}
        self.active_connections: Dict[str, List[OptimizedConnection]] = {}
        
        # 全局指标
        self.total_metrics = ConnectionMetrics()
        
    async def get_connection(self, host: str, port: int) -> Optional[OptimizedConnection]:
        """获取连接"""
        pool_key = f"{host}:{port}"
        
        # 检查可用连接
        if pool_key in self.pools and self.pools[pool_key]:
            connection = self.pools[pool_key].pop()
            if connection.connected:
                if pool_key not in self.active_connections:
                    self.active_connections[pool_key] = []
                self.active_connections[pool_key].append(connection)
                return connection
        
        # 创建新连接
        try:
            connection = OptimizedConnection(host, port, self.config, self.performance_optimizer)
            await connection.connect()
            
            if pool_key not in self.active_connections:
                self.active_connections[pool_key] = []
            self.active_connections[pool_key].append(connection)
            
            return connection
            
        except Exception as e:
            return None
    
    async def return_connection(self, connection: OptimizedConnection):
        """归还连接"""
        pool_key = f"{connection.host}:{connection.port}"
        
        # 从活跃连接移除
        if pool_key in self.active_connections:
            try:
                self.active_connections[pool_key].remove(connection)
            except ValueError:
                pass
        
        # 如果连接正常且启用了连接复用，归还到池中
        if connection.connected and self.config.connection_reuse:
            if pool_key not in self.pools:
                self.pools[pool_key] = []
            
            if len(self.pools[pool_key]) < self.config.connection_pool_size:
                self.pools[pool_key].append(connection)
            else:
                await connection.disconnect()
        else:
            await connection.disconnect()
    
    async def close_all(self):
        """关闭所有连接"""
        # 关闭活跃连接
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.disconnect()
        
        # 关闭池中连接
        for connections in self.pools.values():
            for connection in connections:
                await connection.disconnect()
        
        self.pools.clear()
        self.active_connections.clear()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        pool_stats = {}
        
        for pool_key, connections in self.pools.items():
            pool_stats[pool_key] = {
                "available": len(connections),
                "active": len(self.active_connections.get(pool_key, [])),
                "total": len(connections) + len(self.active_connections.get(pool_key, []))
            }
        
        return {
            "pools": pool_stats,
            "total_pools": len(self.pools),
            "config": {
                "pool_size": self.config.connection_pool_size,
                "reuse_enabled": self.config.connection_reuse
            }
        }


class NetworkLatencyOptimizer(LoggerMixin):
    """网络延迟优化器"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.performance_optimizer: Optional[HFTPerformanceOptimizer] = None
        
        # 连接池
        self.connection_pool = ConnectionPool(self.config)
        
        # 延迟监控
        self.latency_history = deque(maxlen=10000)
        
    def set_performance_optimizer(self, optimizer: HFTPerformanceOptimizer):
        """设置性能优化器"""
        self.performance_optimizer = optimizer
        self.connection_pool.performance_optimizer = optimizer
    
    async def create_optimized_connection(self, host: str, port: int) -> Optional[OptimizedConnection]:
        """创建优化的连接"""
        return await self.connection_pool.get_connection(host, port)
    
    async def measure_latency(self, host: str, port: int, samples: int = 10) -> Dict[str, float]:
        """测量网络延迟"""
        latencies = []
        
        for _ in range(samples):
            try:
                start_time = time.perf_counter()
                
                # 创建测试连接
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(5.0)
                
                await asyncio.get_event_loop().sock_connect(test_socket, (host, port))
                
                latency = (time.perf_counter() - start_time) * 1000  # 毫秒
                latencies.append(latency)
                
                test_socket.close()
                
            except Exception:
                continue
        
        if latencies:
            return {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "avg_ms": sum(latencies) / len(latencies),
                "samples": len(latencies)
            }
        
        return {"error": "No successful measurements"}
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            "config": {
                "tcp_nodelay": self.config.tcp_nodelay,
                "tcp_quickack": self.config.tcp_quickack,
                "send_buffer_size": self.config.send_buffer_size,
                "recv_buffer_size": self.config.recv_buffer_size,
                "keepalive_enabled": self.config.keepalive_enabled,
                "zero_copy_send": self.config.zero_copy_send,
                "message_batching": self.config.message_batching,
                "batch_size": self.config.batch_size
            },
            "connection_pool": self.connection_pool.get_pool_stats(),
            "performance_optimizer": self.performance_optimizer is not None
        }
    
    async def shutdown(self):
        """关闭网络优化器"""
        await self.connection_pool.close_all()
        self.log_info("Network optimizer shutdown complete")