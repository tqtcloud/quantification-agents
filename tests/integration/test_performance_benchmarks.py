"""
性能基准测试套件
测试API响应时间和WebSocket延迟
"""

import asyncio
import json
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets
from httpx import AsyncClient, ASGITransport
from websockets.exceptions import ConnectionClosed

from src.api.trading_api import TradingAPI
from src.websocket.websocket_manager import WebSocketManager
from src.core.database import DatabaseManager


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.response_times = []
        self.throughput_data = []
        self.error_count = 0
        self.success_count = 0
        self.start_time = None
        self.end_time = None
    
    def start_measurement(self):
        """开始性能测量"""
        self.start_time = time.time()
        self.response_times.clear()
        self.throughput_data.clear()
        self.error_count = 0
        self.success_count = 0
    
    def end_measurement(self):
        """结束性能测量"""
        self.end_time = time.time()
    
    def record_response_time(self, duration: float):
        """记录响应时间"""
        self.response_times.append(duration)
    
    def record_success(self):
        """记录成功请求"""
        self.success_count += 1
    
    def record_error(self):
        """记录错误请求"""
        self.error_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.response_times:
            return {}
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        total_requests = len(self.response_times)
        
        return {
            'total_requests': total_requests,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times),
            'median_response_time': statistics.median(self.response_times),
            'p95_response_time': self._percentile(self.response_times, 95),
            'p99_response_time': self._percentile(self.response_times, 99),
            'min_response_time': min(self.response_times),
            'max_response_time': max(self.response_times),
            'throughput_rps': total_requests / total_time if total_time > 0 else 0,
            'total_duration': total_time
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestAPIPerformance:
    """API性能测试"""
    
    @pytest_asyncio.fixture
    async def performance_api(self):
        """创建用于性能测试的API"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {
                'secret_key': 'perf_test_key_123456',
                'access_token_expire_minutes': 60
            },
            'rate_limiting': {
                'enabled': False  # 禁用速率限制用于性能测试
            },
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 创建测试用户
        user_data = {
            'username': 'perf_user',
            'email': 'perf@example.com',
            'password': 'perf_password_123'
        }
        
        user = await trading_api.auth_manager.create_user(**user_data)
        token_data = await trading_api.auth_manager.create_tokens(user)
        access_token = token_data['access_token']
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test",
            timeout=30.0  # 增加超时时间
        ) as client:
            yield client, access_token, trading_api
        
        await trading_api.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_api_response_time_benchmarks(self, performance_api):
        """测试API响应时间基准"""
        client, token, _ = performance_api
        headers = {'Authorization': f'Bearer {token}'}
        
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        # 测试不同端点的响应时间
        endpoints = [
            ('/auth/user-info', 'GET'),
            ('/strategies', 'GET'),
            ('/system/metrics', 'GET'),
        ]
        
        test_iterations = 100
        
        for _ in range(test_iterations):
            for endpoint, method in endpoints:
                start_time = time.time()
                
                try:
                    if method == 'GET':
                        response = await client.get(endpoint, headers=headers)
                    elif method == 'POST':
                        response = await client.post(endpoint, headers=headers, json={})
                    
                    duration = time.time() - start_time
                    metrics.record_response_time(duration)
                    
                    if response.status_code < 500:  # 非服务器错误都算成功
                        metrics.record_success()
                    else:
                        metrics.record_error()
                        
                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_response_time(duration)
                    metrics.record_error()
                    print(f"请求失败: {endpoint} - {e}")
        
        metrics.end_measurement()
        stats = metrics.get_statistics()
        
        # 打印性能统计
        print("\n=== API性能基准测试结果 ===")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"平均响应时间: {stats['avg_response_time']*1000:.2f}ms")
        print(f"中位数响应时间: {stats['median_response_time']*1000:.2f}ms")
        print(f"P95响应时间: {stats['p95_response_time']*1000:.2f}ms")
        print(f"P99响应时间: {stats['p99_response_time']*1000:.2f}ms")
        print(f"最小响应时间: {stats['min_response_time']*1000:.2f}ms")
        print(f"最大响应时间: {stats['max_response_time']*1000:.2f}ms")
        print(f"吞吐量: {stats['throughput_rps']:.2f} RPS")
        
        # 性能断言 (目标: < 100ms平均响应时间)
        assert stats['avg_response_time'] < 0.1, f"平均响应时间过长: {stats['avg_response_time']*1000:.2f}ms"
        assert stats['p95_response_time'] < 0.2, f"P95响应时间过长: {stats['p95_response_time']*1000:.2f}ms"
        assert stats['success_rate'] > 0.95, f"成功率过低: {stats['success_rate']:.2%}"
    
    @pytest.mark.asyncio
    async def test_concurrent_api_load(self, performance_api):
        """测试API并发负载"""
        client, token, _ = performance_api
        headers = {'Authorization': f'Bearer {token}'}
        
        concurrent_users = 50
        requests_per_user = 10
        
        metrics = PerformanceMetrics()
        metrics.start_measurement()
        
        async def user_load_test(user_id: int):
            """单个用户的负载测试"""
            user_metrics = PerformanceMetrics()
            
            for i in range(requests_per_user):
                start_time = time.time()
                
                try:
                    response = await client.get('/auth/user-info', headers=headers)
                    duration = time.time() - start_time
                    
                    metrics.record_response_time(duration)
                    user_metrics.record_response_time(duration)
                    
                    if response.status_code < 500:
                        metrics.record_success()
                        user_metrics.record_success()
                    else:
                        metrics.record_error()
                        user_metrics.record_error()
                        
                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_response_time(duration)
                    metrics.record_error()
                    user_metrics.record_response_time(duration)
                    user_metrics.record_error()
                    print(f"用户{user_id}请求失败: {e}")
                
                # 小延迟模拟真实用户行为
                await asyncio.sleep(0.01)
        
        # 创建并发任务
        tasks = [user_load_test(i) for i in range(concurrent_users)]
        await asyncio.gather(*tasks)
        
        metrics.end_measurement()
        stats = metrics.get_statistics()
        
        print("\n=== API并发负载测试结果 ===")
        print(f"并发用户数: {concurrent_users}")
        print(f"每用户请求数: {requests_per_user}")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"平均响应时间: {stats['avg_response_time']*1000:.2f}ms")
        print(f"P95响应时间: {stats['p95_response_time']*1000:.2f}ms")
        print(f"吞吐量: {stats['throughput_rps']:.2f} RPS")
        print(f"测试时长: {stats['total_duration']:.2f}s")
        
        # 并发性能断言
        assert stats['success_rate'] > 0.90, f"并发测试成功率过低: {stats['success_rate']:.2%}"
        assert stats['throughput_rps'] > 100, f"吞吐量过低: {stats['throughput_rps']:.2f} RPS"


class WebSocketPerformanceClient:
    """WebSocket性能测试客户端"""
    
    def __init__(self, uri: str, client_id: str):
        self.uri = uri
        self.client_id = client_id
        self.websocket = None
        self.connected = False
        self.message_times = {}
        self.received_messages = 0
        self.metrics = PerformanceMetrics()
    
    async def connect(self):
        """连接WebSocket"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            asyncio.create_task(self._receive_messages())
        except Exception as e:
            print(f"WebSocket连接失败: {e}")
            self.connected = False
    
    async def _receive_messages(self):
        """接收消息并计算延迟"""
        try:
            while self.connected and self.websocket:
                message = await self.websocket.recv()
                receive_time = time.time()
                
                try:
                    data = json.loads(message)
                    
                    # 计算延迟
                    if 'timestamp' in data and 'message_id' in data:
                        send_time = data['timestamp']
                        if isinstance(send_time, str):
                            send_time = datetime.fromisoformat(send_time).timestamp()
                        
                        latency = receive_time - send_time
                        self.metrics.record_response_time(latency)
                    
                    self.received_messages += 1
                    
                except json.JSONDecodeError:
                    pass
                    
        except ConnectionClosed:
            self.connected = False
        except Exception as e:
            print(f"接收消息错误: {e}")
            self.connected = False
    
    async def send_message(self, message: dict):
        """发送消息"""
        if self.websocket and self.connected:
            message['timestamp'] = time.time()
            message['message_id'] = f"{self.client_id}_{time.time()}"
            await self.websocket.send(json.dumps(message))
    
    async def close(self):
        """关闭连接"""
        self.connected = False
        if self.websocket:
            await self.websocket.close()


class TestWebSocketPerformance:
    """WebSocket性能测试"""
    
    @pytest_asyncio.fixture
    async def websocket_performance_system(self):
        """创建WebSocket性能测试系统"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'websocket': {
                'host': '127.0.0.1',
                'port': 8768,
                'max_connections': 1000,
                'heartbeat_interval': 60
            }
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        websocket_manager = WebSocketManager(config['websocket'], database)
        await websocket_manager.initialize()
        
        # 启动WebSocket服务器
        server_task = asyncio.create_task(websocket_manager.start_server())
        
        # 等待服务器启动
        await asyncio.sleep(0.5)
        
        yield websocket_manager, config['websocket']['host'], config['websocket']['port']
        
        # 清理
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        await websocket_manager.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_websocket_latency_benchmark(self, websocket_performance_system):
        """测试WebSocket延迟基准"""
        websocket_manager, host, port = websocket_performance_system
        uri = f"ws://{host}:{port}"
        
        client = WebSocketPerformanceClient(uri, "latency_test_client")
        await client.connect()
        
        if not client.connected:
            pytest.skip("无法连接到WebSocket服务器")
        
        client.metrics.start_measurement()
        
        # 发送测试消息
        test_messages = 100
        
        for i in range(test_messages):
            test_message = {
                'type': 'ping',
                'sequence': i,
                'payload': f'test_data_{i}'
            }
            
            await client.send_message(test_message)
            await asyncio.sleep(0.01)  # 小延迟避免过快发送
        
        # 等待所有消息处理完成
        await asyncio.sleep(2.0)
        
        client.metrics.end_measurement()
        await client.close()
        
        stats = client.metrics.get_statistics()
        
        print("\n=== WebSocket延迟基准测试结果 ===")
        if stats:
            print(f"测试消息数: {test_messages}")
            print(f"接收消息数: {client.received_messages}")
            print(f"平均延迟: {stats['avg_response_time']*1000:.2f}ms")
            print(f"中位数延迟: {stats['median_response_time']*1000:.2f}ms")
            print(f"P95延迟: {stats['p95_response_time']*1000:.2f}ms")
            print(f"P99延迟: {stats['p99_response_time']*1000:.2f}ms")
            print(f"最小延迟: {stats['min_response_time']*1000:.2f}ms")
            print(f"最大延迟: {stats['max_response_time']*1000:.2f}ms")
            
            # WebSocket延迟断言 (目标: < 50ms平均延迟)
            assert stats['avg_response_time'] < 0.05, f"WebSocket平均延迟过高: {stats['avg_response_time']*1000:.2f}ms"
        else:
            print("没有收到延迟数据")
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_connections(self, websocket_performance_system):
        """测试WebSocket并发连接性能"""
        websocket_manager, host, port = websocket_performance_system
        uri = f"ws://{host}:{port}"
        
        concurrent_clients = 100
        messages_per_client = 10
        
        clients = []
        connection_metrics = PerformanceMetrics()
        connection_metrics.start_measurement()
        
        # 创建并发连接
        for i in range(concurrent_clients):
            client = WebSocketPerformanceClient(uri, f"concurrent_client_{i}")
            
            connect_start = time.time()
            await client.connect()
            connect_duration = time.time() - connect_start
            
            connection_metrics.record_response_time(connect_duration)
            
            if client.connected:
                clients.append(client)
                connection_metrics.record_success()
            else:
                connection_metrics.record_error()
        
        connection_metrics.end_measurement()
        
        print(f"\n=== WebSocket并发连接测试结果 ===")
        print(f"目标连接数: {concurrent_clients}")
        print(f"成功连接数: {len(clients)}")
        print(f"连接成功率: {len(clients)/concurrent_clients:.2%}")
        
        connection_stats = connection_metrics.get_statistics()
        if connection_stats:
            print(f"平均连接时间: {connection_stats['avg_response_time']*1000:.2f}ms")
            print(f"最大连接时间: {connection_stats['max_response_time']*1000:.2f}ms")
        
        # 测试消息传输性能
        if clients:
            message_metrics = PerformanceMetrics()
            message_metrics.start_measurement()
            
            async def client_message_test(client):
                """单个客户端的消息测试"""
                for i in range(messages_per_client):
                    message = {
                        'type': 'test_message',
                        'sequence': i,
                        'client_id': client.client_id
                    }
                    
                    start_time = time.time()
                    await client.send_message(message)
                    duration = time.time() - start_time
                    
                    message_metrics.record_response_time(duration)
                    message_metrics.record_success()
                    
                    await asyncio.sleep(0.001)  # 微小延迟
            
            # 并发发送消息
            tasks = [client_message_test(client) for client in clients]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # 等待消息处理
            await asyncio.sleep(2.0)
            
            message_metrics.end_measurement()
            msg_stats = message_metrics.get_statistics()
            
            if msg_stats:
                print(f"总发送消息数: {msg_stats['total_requests']}")
                print(f"平均发送时间: {msg_stats['avg_response_time']*1000:.2f}ms")
                print(f"消息发送吞吐量: {msg_stats['throughput_rps']:.2f} MPS")
            
            # 统计接收到的消息
            total_received = sum(client.received_messages for client in clients)
            print(f"总接收消息数: {total_received}")
        
        # 清理连接
        for client in clients:
            await client.close()
        
        # 性能断言
        assert len(clients) >= concurrent_clients * 0.8, f"连接成功率过低: {len(clients)}/{concurrent_clients}"


class TestSystemResourceUsage:
    """系统资源使用测试"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\n=== 内存使用测试 ===")
        print(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 创建系统负载
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'memory_test_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 8769,
                'max_connections': 100
            }
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        websocket_manager = WebSocketManager(config['websocket'], database)
        await websocket_manager.initialize()
        
        # 创建负载
        load_tasks = []
        
        # 模拟API负载
        async def api_load():
            async with AsyncClient(
                transport=ASGITransport(app=trading_api.app),
                base_url="http://test"
            ) as client:
                for _ in range(50):
                    try:
                        await client.get('/health')
                    except:
                        pass
                    await asyncio.sleep(0.01)
        
        # 启动负载任务
        for _ in range(10):
            load_tasks.append(asyncio.create_task(api_load()))
        
        # 监控内存使用
        max_memory = initial_memory
        
        # 运行负载测试
        await asyncio.gather(*load_tasks)
        
        # 检查峰值内存使用
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(max_memory, peak_memory)
        
        print(f"峰值内存使用: {peak_memory:.2f} MB")
        print(f"内存增长: {peak_memory - initial_memory:.2f} MB")
        
        # 清理
        await trading_api.shutdown()
        await websocket_manager.shutdown()
        await database.close()
        
        # 检查内存清理
        await asyncio.sleep(1.0)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"清理后内存: {final_memory:.2f} MB")
        
        # 内存使用断言 (不应该增长过多)
        memory_growth = peak_memory - initial_memory
        assert memory_growth < 100, f"内存增长过多: {memory_growth:.2f} MB"
    
    @pytest.mark.asyncio
    async def test_cpu_usage_efficiency(self):
        """测试CPU使用效率"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 记录CPU使用情况
        cpu_percent_before = process.cpu_percent()
        
        print(f"\n=== CPU使用效率测试 ===")
        print(f"测试前CPU使用: {cpu_percent_before:.2f}%")
        
        # 创建CPU密集型负载
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'cpu_test_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 创建大量并发请求
        async def cpu_intensive_load():
            async with AsyncClient(
                transport=ASGITransport(app=trading_api.app),
                base_url="http://test"
            ) as client:
                tasks = []
                for _ in range(100):
                    tasks.append(client.get('/health'))
                
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # 执行负载测试
        start_time = time.time()
        await cpu_intensive_load()
        end_time = time.time()
        
        # 检查CPU使用
        cpu_percent_after = process.cpu_percent()
        
        print(f"负载测试耗时: {end_time - start_time:.2f}s")
        print(f"测试后CPU使用: {cpu_percent_after:.2f}%")
        
        # 清理
        await trading_api.shutdown()
        await database.close()
        
        print("CPU使用效率测试完成")


if __name__ == "__main__":
    # 运行性能基准测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s"  # 显示详细输出
    ])