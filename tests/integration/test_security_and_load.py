"""
安全和负载测试套件
测试并发连接、攻击防护和系统安全性
"""

import asyncio
import json
import random
import string
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import websockets
from httpx import AsyncClient, ASGITransport
from websockets.exceptions import ConnectionClosed

from src.api.trading_api import TradingAPI
from src.websocket.websocket_manager import WebSocketManager
from src.core.database import DatabaseManager


class AttackSimulator:
    """攻击模拟器"""
    
    def __init__(self):
        self.attack_results = []
    
    def generate_malicious_payload(self) -> str:
        """生成恶意负载"""
        payloads = [
            # SQL注入尝试
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'/*",
            
            # XSS尝试
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            
            # 命令注入尝试
            "; ls -la",
            "&& whoami",
            "| cat /etc/passwd",
            
            # JSON注入
            '{"malicious": "$(whoami)"}',
            '{"eval": "require(\'child_process\').exec(\'ls\')"}',
            
            # 缓冲区溢出尝试
            "A" * 10000,
            "B" * 100000,
        ]
        
        return random.choice(payloads)
    
    def generate_invalid_json(self) -> str:
        """生成无效JSON"""
        invalid_jsons = [
            '{"incomplete": ',
            '{"unclosed": "string}',
            '{invalid_key: "value"}',
            '{"trailing_comma": "value",}',
            'not_json_at_all',
            '{' + 'A' * 1000 + '}',
        ]
        
        return random.choice(invalid_jsons)
    
    def generate_large_payload(self, size_mb: int = 10) -> str:
        """生成大型负载"""
        size_bytes = size_mb * 1024 * 1024
        return 'X' * size_bytes
    
    def record_attack_result(self, attack_type: str, payload: str, response_code: int, blocked: bool):
        """记录攻击结果"""
        self.attack_results.append({
            'attack_type': attack_type,
            'payload_length': len(payload),
            'response_code': response_code,
            'blocked': blocked,
            'timestamp': datetime.now()
        })
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """获取攻击总结"""
        if not self.attack_results:
            return {}
        
        total_attacks = len(self.attack_results)
        blocked_attacks = sum(1 for result in self.attack_results if result['blocked'])
        
        return {
            'total_attacks': total_attacks,
            'blocked_attacks': blocked_attacks,
            'success_rate': (total_attacks - blocked_attacks) / total_attacks,
            'block_rate': blocked_attacks / total_attacks,
            'attack_types': list(set(result['attack_type'] for result in self.attack_results))
        }


class TestSecurityAttackPrevention:
    """安全攻击防护测试"""
    
    @pytest_asyncio.fixture
    async def security_test_system(self):
        """创建安全测试系统"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {
                'secret_key': 'security_test_key_extremely_secure_123456789',
                'access_token_expire_minutes': 30,
                'failed_login_max_attempts': 5,
                'failed_login_lockout_minutes': 15
            },
            'rate_limiting': {
                'enabled': True,
                'default_rate': '100/minute',
                'burst_rate': '10/second',
                'attack_detection': True
            },
            'websocket': {
                'enabled': True,
                'host': '127.0.0.1',
                'port': 8770,
                'max_connections': 100,
                'max_message_size': 1024 * 1024,  # 1MB限制
                'connection_timeout': 30
            },
            'security': {
                'enable_request_validation': True,
                'max_request_size': 10 * 1024 * 1024,  # 10MB
                'enable_sql_injection_detection': True,
                'enable_xss_protection': True
            }
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        websocket_manager = WebSocketManager(config['websocket'], database)
        await websocket_manager.initialize()
        
        # 启动WebSocket服务器
        server_task = asyncio.create_task(websocket_manager.start_server())
        await asyncio.sleep(0.5)
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test",
            timeout=30.0
        ) as http_client:
            yield {
                'http_client': http_client,
                'websocket_manager': websocket_manager,
                'trading_api': trading_api,
                'config': config
            }
        
        # 清理
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        await trading_api.shutdown()
        await websocket_manager.shutdown()
        await database.close()
    
    @pytest.mark.asyncio
    async def test_sql_injection_attacks(self, security_test_system):
        """测试SQL注入攻击防护"""
        system = security_test_system
        http_client = system['http_client']
        
        simulator = AttackSimulator()
        
        # SQL注入攻击载荷
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1' --",
            "' UNION SELECT * FROM users --",
            "admin'/**/OR/**/1=1#",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        print("\n=== SQL注入攻击测试 ===")
        
        for payload in sql_injection_payloads:
            # 尝试在用户注册中使用SQL注入
            malicious_data = {
                'username': payload,
                'email': f'test_{random.randint(1000, 9999)}@example.com',
                'password': 'test_password'
            }
            
            response = await http_client.post('/auth/register', json=malicious_data)
            
            # SQL注入应该被阻止或返回安全错误
            blocked = response.status_code in [400, 422, 403, 500]
            
            simulator.record_attack_result('sql_injection', payload, response.status_code, blocked)
            
            if not blocked:
                print(f"警告: SQL注入可能成功: {payload[:50]}...")
        
        # 尝试在登录中使用SQL注入
        for payload in sql_injection_payloads:
            malicious_login = {
                'username': payload,
                'password': 'any_password'
            }
            
            response = await http_client.post('/auth/login', json=malicious_login)
            blocked = response.status_code in [400, 401, 422, 403]
            
            simulator.record_attack_result('sql_injection_login', payload, response.status_code, blocked)
        
        # 获取攻击总结
        summary = simulator.get_attack_summary()
        print(f"SQL注入攻击总数: {summary['total_attacks']}")
        print(f"被阻止的攻击: {summary['blocked_attacks']}")
        print(f"阻止率: {summary['block_rate']:.2%}")
        
        # 安全断言 - 大部分攻击应该被阻止
        assert summary['block_rate'] > 0.8, f"SQL注入阻止率过低: {summary['block_rate']:.2%}"
    
    @pytest.mark.asyncio
    async def test_xss_attack_prevention(self, security_test_system):
        """测试XSS攻击防护"""
        system = security_test_system
        http_client = system['http_client']
        
        simulator = AttackSimulator()
        
        # XSS攻击载荷
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src=javascript:alert('xss')></iframe>",
            "';alert('xss');//",
            "<body onload=alert('xss')>",
        ]
        
        print("\n=== XSS攻击测试 ===")
        
        for payload in xss_payloads:
            # 在用户数据中尝试XSS
            malicious_data = {
                'username': f'user_{random.randint(1000, 9999)}',
                'email': f'test_{random.randint(1000, 9999)}@example.com',
                'password': 'test_password',
                'full_name': payload  # XSS在全名字段中
            }
            
            response = await http_client.post('/auth/register', json=malicious_data)
            blocked = response.status_code in [400, 422, 403]
            
            simulator.record_attack_result('xss', payload, response.status_code, blocked)
            
            # 如果注册成功，检查返回数据是否被转义
            if response.status_code == 201:
                response_data = response.json()
                if 'full_name' in response_data:
                    contains_script = '<script>' in response_data['full_name']
                    if contains_script:
                        print(f"警告: XSS载荷未被转义: {payload[:50]}...")
                        simulator.record_attack_result('xss_unescaped', payload, 200, False)
                    else:
                        simulator.record_attack_result('xss_escaped', payload, 200, True)
        
        summary = simulator.get_attack_summary()
        print(f"XSS攻击总数: {summary['total_attacks']}")
        print(f"被阻止的攻击: {summary['blocked_attacks']}")
        print(f"阻止率: {summary['block_rate']:.2%}")
        
        # XSS防护断言
        assert summary['block_rate'] > 0.7, f"XSS阻止率过低: {summary['block_rate']:.2%}"
    
    @pytest.mark.asyncio
    async def test_brute_force_attack_protection(self, security_test_system):
        """测试暴力破解攻击防护"""
        system = security_test_system
        http_client = system['http_client']
        
        # 先创建一个测试用户
        user_data = {
            'username': 'brute_force_target',
            'email': 'bruteforce@example.com',
            'password': 'correct_password_123'
        }
        
        register_response = await http_client.post('/auth/register', json=user_data)
        assert register_response.status_code == 201
        
        print("\n=== 暴力破解攻击测试 ===")
        
        # 模拟暴力破解攻击
        failed_attempts = 0
        locked_out = False
        
        # 尝试多次错误登录
        for attempt in range(10):
            malicious_login = {
                'username': 'brute_force_target',
                'password': f'wrong_password_{attempt}'
            }
            
            response = await http_client.post('/auth/login', json=malicious_login)
            
            if response.status_code == 401:
                failed_attempts += 1
            elif response.status_code == 429:  # Too Many Requests
                locked_out = True
                break
            elif response.status_code == 403:  # Forbidden (account locked)
                locked_out = True
                break
            
            await asyncio.sleep(0.1)  # 小延迟避免过快请求
        
        print(f"失败尝试次数: {failed_attempts}")
        print(f"账户被锁定: {locked_out}")
        
        # 验证正确密码在锁定期间也被拒绝
        if locked_out:
            correct_login = {
                'username': 'brute_force_target',
                'password': 'correct_password_123'
            }
            
            response = await http_client.post('/auth/login', json=correct_login)
            still_locked = response.status_code in [403, 429]
            print(f"正确密码仍被拒绝: {still_locked}")
            
            assert still_locked, "账户锁定机制未正常工作"
        
        # 暴力破解防护断言
        assert locked_out or failed_attempts >= 5, "暴力破解防护未触发"
    
    @pytest.mark.asyncio
    async def test_ddos_attack_simulation(self, security_test_system):
        """测试DDoS攻击模拟"""
        system = security_test_system
        http_client = system['http_client']
        
        print("\n=== DDoS攻击模拟测试 ===")
        
        # 模拟大量并发请求
        concurrent_requests = 100
        requests_per_client = 50
        
        async def ddos_client(client_id: int):
            """单个DDoS客户端"""
            success_count = 0
            blocked_count = 0
            
            for i in range(requests_per_client):
                try:
                    response = await http_client.get('/health')
                    
                    if response.status_code == 200:
                        success_count += 1
                    elif response.status_code == 429:  # Rate limited
                        blocked_count += 1
                    
                except Exception:
                    blocked_count += 1
                
                # 极小延迟模拟真实攻击
                await asyncio.sleep(0.001)
            
            return success_count, blocked_count
        
        start_time = time.time()
        
        # 启动并发攻击
        tasks = [ddos_client(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 统计结果
        total_success = 0
        total_blocked = 0
        
        for result in results:
            if isinstance(result, tuple):
                success, blocked = result
                total_success += success
                total_blocked += blocked
        
        total_requests = total_success + total_blocked
        block_rate = total_blocked / total_requests if total_requests > 0 else 0
        
        print(f"测试时长: {duration:.2f}s")
        print(f"总请求数: {total_requests}")
        print(f"成功请求: {total_success}")
        print(f"被阻止请求: {total_blocked}")
        print(f"阻止率: {block_rate:.2%}")
        print(f"请求速率: {total_requests/duration:.2f} RPS")
        
        # DDoS防护断言 - 应该有大量请求被阻止
        assert block_rate > 0.5, f"DDoS防护不足，阻止率: {block_rate:.2%}"


class TestWebSocketSecurityAndLoad:
    """WebSocket安全和负载测试"""
    
    @pytest_asyncio.fixture
    async def websocket_security_system(self):
        """创建WebSocket安全测试系统"""
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'websocket': {
                'host': '127.0.0.1',
                'port': 8771,
                'max_connections': 1000,
                'max_message_size': 64 * 1024,  # 64KB限制
                'connection_timeout': 60,
                'heartbeat_interval': 30,
                'rate_limit_messages': 100,  # 每分钟100条消息
                'enable_message_validation': True
            }
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        websocket_manager = WebSocketManager(config['websocket'], database)
        await websocket_manager.initialize()
        
        # 启动服务器
        server_task = asyncio.create_task(websocket_manager.start_server())
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
    async def test_websocket_connection_flooding(self, websocket_security_system):
        """测试WebSocket连接洪水攻击"""
        websocket_manager, host, port = websocket_security_system
        uri = f"ws://{host}:{port}"
        
        print("\n=== WebSocket连接洪水攻击测试 ===")
        
        # 尝试建立大量连接
        flood_connections = 200
        successful_connections = 0
        rejected_connections = 0
        connections = []
        
        async def create_connection(client_id: int):
            try:
                websocket = await websockets.connect(uri, timeout=5)
                connections.append(websocket)
                return True
            except Exception as e:
                print(f"连接{client_id}失败: {e}")
                return False
        
        # 并发建立连接
        start_time = time.time()
        tasks = [create_connection(i) for i in range(flood_connections)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 统计结果
        for result in results:
            if result is True:
                successful_connections += 1
            else:
                rejected_connections += 1
        
        print(f"尝试连接数: {flood_connections}")
        print(f"成功连接数: {successful_connections}")
        print(f"被拒绝连接数: {rejected_connections}")
        print(f"连接成功率: {successful_connections/flood_connections:.2%}")
        print(f"连接建立时间: {end_time - start_time:.2f}s")
        
        # 清理连接
        for websocket in connections:
            try:
                await websocket.close()
            except:
                pass
        
        # 连接洪水防护断言
        assert successful_connections < flood_connections, "连接洪水攻击未被有效防护"
        assert successful_connections <= 1000, "超过了最大连接数限制"
    
    @pytest.mark.asyncio
    async def test_websocket_message_bombing(self, websocket_security_system):
        """测试WebSocket消息轰炸攻击"""
        websocket_manager, host, port = websocket_security_system
        uri = f"ws://{host}:{port}"
        
        print("\n=== WebSocket消息轰炸攻击测试 ===")
        
        # 建立单个连接
        try:
            websocket = await websockets.connect(uri, timeout=5)
        except Exception as e:
            pytest.skip(f"无法建立WebSocket连接: {e}")
        
        # 发送大量消息
        message_bomb_count = 1000
        messages_sent = 0
        messages_blocked = 0
        
        test_message = {
            'type': 'spam_message',
            'data': 'This is a spam message' * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        for i in range(message_bomb_count):
            try:
                await websocket.send(json.dumps(test_message))
                messages_sent += 1
                
                # 极小延迟
                await asyncio.sleep(0.001)
                
            except Exception as e:
                messages_blocked += 1
                if "connection closed" in str(e).lower():
                    print(f"连接在第{i}条消息后被关闭")
                    break
        
        end_time = time.time()
        
        print(f"尝试发送消息数: {message_bomb_count}")
        print(f"成功发送消息数: {messages_sent}")
        print(f"被阻止消息数: {messages_blocked}")
        print(f"消息发送率: {messages_sent/(end_time - start_time):.2f} MPS")
        
        # 清理连接
        try:
            await websocket.close()
        except:
            pass
        
        # 消息轰炸防护断言
        assert messages_blocked > 0 or messages_sent < message_bomb_count, "消息轰炸攻击未被防护"
    
    @pytest.mark.asyncio
    async def test_large_message_attack(self, websocket_security_system):
        """测试大消息攻击"""
        websocket_manager, host, port = websocket_security_system
        uri = f"ws://{host}:{port}"
        
        print("\n=== 大消息攻击测试 ===")
        
        try:
            websocket = await websockets.connect(uri, timeout=5)
        except Exception as e:
            pytest.skip(f"无法建立WebSocket连接: {e}")
        
        # 测试不同大小的消息
        message_sizes = [
            1 * 1024,      # 1KB
            64 * 1024,     # 64KB (限制)
            128 * 1024,    # 128KB (超过限制)
            1 * 1024 * 1024,  # 1MB (远超限制)
        ]
        
        results = {}
        
        for size in message_sizes:
            large_payload = 'X' * size
            large_message = {
                'type': 'large_message',
                'payload': large_payload,
                'size': size
            }
            
            try:
                await websocket.send(json.dumps(large_message))
                results[size] = 'sent'
                await asyncio.sleep(0.1)  # 等待服务器处理
                
            except Exception as e:
                results[size] = f'blocked: {str(e)}'
                print(f"{size//1024}KB消息被阻止: {e}")
        
        # 清理连接
        try:
            await websocket.close()
        except:
            pass
        
        print("大消息测试结果:")
        for size, result in results.items():
            print(f"  {size//1024}KB: {result}")
        
        # 大消息攻击防护断言
        # 超过64KB的消息应该被阻止
        large_messages_blocked = sum(
            1 for size, result in results.items() 
            if size > 64*1024 and 'blocked' in result
        )
        
        assert large_messages_blocked > 0, "大消息攻击未被有效防护"


class TestLoadBalancingAndScalability:
    """负载均衡和可扩展性测试"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_load(self):
        """测试高并发负载"""
        print("\n=== 高并发负载测试 ===")
        
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'load_test_key'},
            'rate_limiting': {'enabled': False},
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        # 高并发测试
        concurrent_users = 200
        requests_per_user = 25
        
        async def concurrent_user_simulation(user_id: int):
            """并发用户模拟"""
            async with AsyncClient(
                transport=ASGITransport(app=trading_api.app),
                base_url="http://test",
                timeout=30.0
            ) as client:
                
                success_count = 0
                error_count = 0
                response_times = []
                
                for i in range(requests_per_user):
                    start_time = time.time()
                    
                    try:
                        response = await client.get('/health')
                        duration = time.time() - start_time
                        response_times.append(duration)
                        
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        error_count += 1
                        duration = time.time() - start_time
                        response_times.append(duration)
                    
                    # 模拟用户思考时间
                    await asyncio.sleep(random.uniform(0.001, 0.01))
                
                return {
                    'user_id': user_id,
                    'success_count': success_count,
                    'error_count': error_count,
                    'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                    'total_requests': success_count + error_count
                }
        
        # 执行并发测试
        start_time = time.time()
        
        tasks = [concurrent_user_simulation(i) for i in range(concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 统计结果
        total_requests = 0
        total_success = 0
        total_errors = 0
        total_response_time = 0
        valid_results = 0
        
        for result in results:
            if isinstance(result, dict):
                total_requests += result['total_requests']
                total_success += result['success_count']
                total_errors += result['error_count']
                total_response_time += result['avg_response_time']
                valid_results += 1
        
        avg_response_time = total_response_time / valid_results if valid_results > 0 else 0
        success_rate = total_success / total_requests if total_requests > 0 else 0
        throughput = total_requests / total_duration
        
        print(f"并发用户数: {concurrent_users}")
        print(f"每用户请求数: {requests_per_user}")
        print(f"总请求数: {total_requests}")
        print(f"成功请求数: {total_success}")
        print(f"错误请求数: {total_errors}")
        print(f"成功率: {success_rate:.2%}")
        print(f"平均响应时间: {avg_response_time*1000:.2f}ms")
        print(f"总测试时长: {total_duration:.2f}s")
        print(f"系统吞吐量: {throughput:.2f} RPS")
        
        # 清理
        await trading_api.shutdown()
        await database.close()
        
        # 高并发性能断言
        assert success_rate > 0.95, f"高并发成功率过低: {success_rate:.2%}"
        assert avg_response_time < 0.5, f"高并发平均响应时间过长: {avg_response_time*1000:.2f}ms"
        assert throughput > 100, f"系统吞吐量过低: {throughput:.2f} RPS"


class TestResilienceAndRecovery:
    """系统弹性和恢复测试"""
    
    @pytest.mark.asyncio
    async def test_service_recovery_after_overload(self):
        """测试过载后的服务恢复"""
        print("\n=== 服务过载恢复测试 ===")
        
        config = {
            'database': {'url': 'sqlite:///:memory:'},
            'auth': {'secret_key': 'recovery_test_key'},
            'rate_limiting': {
                'enabled': True,
                'default_rate': '50/minute',  # 较低的限制
                'burst_rate': '10/second'
            },
            'websocket': {'enabled': False}
        }
        
        database = DatabaseManager(config['database']['url'])
        await database.initialize()
        
        trading_api = TradingAPI(config=config, database=database)
        await trading_api.initialize()
        
        async with AsyncClient(
            transport=ASGITransport(app=trading_api.app),
            base_url="http://test"
        ) as client:
            
            # 1. 创建过载条件
            print("阶段1: 创建系统过载...")
            
            overload_requests = 100
            overload_results = []
            
            for i in range(overload_requests):
                try:
                    response = await client.get('/health')
                    overload_results.append(response.status_code)
                except:
                    overload_results.append(500)
            
            rate_limited_count = sum(1 for status in overload_results if status == 429)
            print(f"过载期间被限制的请求: {rate_limited_count}/{overload_requests}")
            
            # 2. 等待系统恢复
            print("阶段2: 等待系统恢复...")
            await asyncio.sleep(65)  # 等待速率限制窗口重置
            
            # 3. 测试恢复后的服务质量
            print("阶段3: 测试恢复后服务...")
            
            recovery_requests = 20
            recovery_success = 0
            
            for i in range(recovery_requests):
                try:
                    response = await client.get('/health')
                    if response.status_code == 200:
                        recovery_success += 1
                except:
                    pass
                
                await asyncio.sleep(0.1)  # 适当间隔
            
            recovery_rate = recovery_success / recovery_requests
            print(f"恢复期间成功率: {recovery_rate:.2%}")
            
            # 清理
            await trading_api.shutdown()
            await database.close()
            
            # 恢复能力断言
            assert recovery_rate > 0.8, f"服务恢复能力不足: {recovery_rate:.2%}"
            assert rate_limited_count > 0, "系统未正确识别过载状态"


if __name__ == "__main__":
    # 运行安全和负载测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-s"
    ])