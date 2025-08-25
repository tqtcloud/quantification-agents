"""
集成测试配置和通用fixtures
提供测试所需的公共配置和工具函数
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# 添加项目根目录到Python路径
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.api.trading_api import TradingAPI
from src.websocket.websocket_manager import WebSocketManager
from src.core.database import DatabaseManager
from src.core.config_manager import ConfigManager


# 测试配置常量
TEST_CONFIG = {
    'database': {
        'url': 'sqlite:///:memory:',
        'echo': False,
        'pool_pre_ping': True
    },
    'auth': {
        'secret_key': 'integration_test_secret_key_12345_very_secure',
        'access_token_expire_minutes': 30,
        'refresh_token_expire_days': 7,
        'failed_login_max_attempts': 5,
        'failed_login_lockout_minutes': 15
    },
    'rate_limiting': {
        'enabled': True,
        'default_rate': '1000/minute',
        'burst_rate': '100/second',
        'attack_detection': True
    },
    'websocket': {
        'enabled': True,
        'host': '127.0.0.1',
        'port': 8800,  # 基础端口，测试时会动态分配
        'max_connections': 1000,
        'max_message_size': 1024 * 1024,  # 1MB
        'connection_timeout': 30,
        'heartbeat_interval': 30,
        'cleanup_interval': 60
    },
    'api': {
        'host': '127.0.0.1',
        'port': 8000,
        'debug': False,
        'enable_docs': False
    },
    'security': {
        'enable_request_validation': True,
        'max_request_size': 10 * 1024 * 1024,  # 10MB
        'enable_sql_injection_detection': True,
        'enable_xss_protection': True,
        'cors_origins': ["*"],
        'cors_credentials': True,
        'cors_methods': ["*"],
        'cors_headers': ["*"]
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

# 测试用户数据
TEST_USERS = [
    {
        'username': 'test_user_001',
        'email': 'test001@example.com',
        'password': 'secure_password_123',
        'full_name': 'Test User 001',
        'permissions': ['trading', 'monitoring']
    },
    {
        'username': 'test_admin_001',
        'email': 'admin001@example.com',
        'password': 'admin_secure_password_456',
        'full_name': 'Test Admin 001',
        'permissions': ['admin', 'trading', 'monitoring']
    },
    {
        'username': 'test_readonly_001',
        'email': 'readonly001@example.com',
        'password': 'readonly_password_789',
        'full_name': 'Test ReadOnly 001',
        'permissions': ['monitoring']
    }
]

# 端口管理器，避免端口冲突
class PortManager:
    """端口管理器"""
    
    def __init__(self, start_port: int = 8800):
        self.current_port = start_port
        self.used_ports = set()
    
    def get_next_port(self) -> int:
        """获取下一个可用端口"""
        while self.current_port in self.used_ports:
            self.current_port += 1
        
        port = self.current_port
        self.used_ports.add(port)
        self.current_port += 1
        return port
    
    def release_port(self, port: int):
        """释放端口"""
        self.used_ports.discard(port)

# 全局端口管理器
port_manager = PortManager()


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环，用于整个测试会话"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """测试配置"""
    # 为每个测试分配唯一端口
    config = TEST_CONFIG.copy()
    config['websocket']['port'] = port_manager.get_next_port()
    config['api']['port'] = port_manager.get_next_port()
    
    yield config
    
    # 清理端口
    port_manager.release_port(config['websocket']['port'])
    port_manager.release_port(config['api']['port'])


@pytest_asyncio.fixture
async def test_database() -> DatabaseManager:
    """测试数据库"""
    database = DatabaseManager(TEST_CONFIG['database']['url'])
    await database.initialize()
    
    yield database
    
    await database.close()


@pytest_asyncio.fixture
async def clean_database() -> DatabaseManager:
    """每次测试使用全新的数据库"""
    # 使用临时数据库文件
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    db_url = f"sqlite:///{temp_db.name}"
    database = DatabaseManager(db_url)
    await database.initialize()
    
    yield database
    
    await database.close()
    
    # 清理临时文件
    try:
        os.unlink(temp_db.name)
    except:
        pass


@pytest_asyncio.fixture
async def trading_api(test_config: Dict[str, Any]) -> TradingAPI:
    """交易API实例"""
    # 使用内存数据库
    database = DatabaseManager(test_config['database']['url'])
    await database.initialize()
    
    api = TradingAPI(config=test_config, database=database)
    await api.initialize()
    
    yield api
    
    await api.shutdown()
    await database.close()


@pytest_asyncio.fixture
async def websocket_manager(test_config: Dict[str, Any]) -> WebSocketManager:
    """WebSocket管理器实例"""
    database = DatabaseManager(test_config['database']['url'])
    await database.initialize()
    
    ws_manager = WebSocketManager(test_config['websocket'], database)
    await ws_manager.initialize()
    
    yield ws_manager
    
    await ws_manager.shutdown()
    await database.close()


@pytest_asyncio.fixture
async def http_client(trading_api: TradingAPI) -> AsyncClient:
    """HTTP客户端"""
    async with AsyncClient(
        transport=ASGITransport(app=trading_api.app),
        base_url="http://test",
        timeout=30.0
    ) as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_client(trading_api: TradingAPI) -> tuple[AsyncClient, str]:
    """已认证的HTTP客户端"""
    # 创建测试用户
    user_data = TEST_USERS[0].copy()
    user = await trading_api.auth_manager.create_user(**user_data)
    
    # 获取访问令牌
    token_data = await trading_api.auth_manager.create_tokens(user)
    access_token = token_data['access_token']
    
    async with AsyncClient(
        transport=ASGITransport(app=trading_api.app),
        base_url="http://test",
        timeout=30.0,
        headers={'Authorization': f'Bearer {access_token}'}
    ) as client:
        yield client, access_token


@pytest_asyncio.fixture
async def admin_client(trading_api: TradingAPI) -> tuple[AsyncClient, str]:
    """管理员HTTP客户端"""
    # 创建管理员用户
    admin_data = TEST_USERS[1].copy()
    admin_user = await trading_api.auth_manager.create_user(**admin_data)
    
    # 获取访问令牌
    token_data = await trading_api.auth_manager.create_tokens(admin_user)
    access_token = token_data['access_token']
    
    async with AsyncClient(
        transport=ASGITransport(app=trading_api.app),
        base_url="http://test",
        timeout=30.0,
        headers={'Authorization': f'Bearer {access_token}'}
    ) as client:
        yield client, access_token


@pytest_asyncio.fixture
async def multiple_users(trading_api: TradingAPI) -> List[Dict[str, Any]]:
    """创建多个测试用户"""
    created_users = []
    
    for user_data in TEST_USERS:
        user = await trading_api.auth_manager.create_user(**user_data)
        token_data = await trading_api.auth_manager.create_tokens(user)
        
        created_users.append({
            'user': user,
            'access_token': token_data['access_token'],
            'refresh_token': token_data['refresh_token'],
            'user_data': user_data
        })
    
    yield created_users


class MockWebSocketConnection:
    """模拟WebSocket连接"""
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or f"mock_client_{int(time.time() * 1000)}"
        self.messages = []
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.remote_address = ('127.0.0.1', 12345)
        self.request_headers = {}
    
    async def send(self, message: str):
        """发送消息"""
        if self.closed:
            from websockets.exceptions import ConnectionClosed
            raise ConnectionClosed(None, None)
        self.messages.append(message)
    
    async def recv(self) -> str:
        """接收消息（模拟）"""
        await asyncio.sleep(0.01)  # 模拟网络延迟
        return '{"type": "pong"}'
    
    async def close(self, code: int = 1000, reason: str = ""):
        """关闭连接"""
        self.closed = True
        self.close_code = code
        self.close_reason = reason
    
    def get_messages(self) -> List[dict]:
        """获取发送的消息"""
        import json
        result = []
        for msg in self.messages:
            try:
                result.append(json.loads(msg))
            except json.JSONDecodeError:
                result.append({'raw': msg})
        return result
    
    def clear_messages(self):
        """清空消息"""
        self.messages.clear()


@pytest.fixture
def mock_websocket() -> MockWebSocketConnection:
    """模拟WebSocket连接"""
    return MockWebSocketConnection()


@pytest.fixture
def multiple_mock_websockets() -> List[MockWebSocketConnection]:
    """多个模拟WebSocket连接"""
    return [MockWebSocketConnection(f"client_{i:03d}") for i in range(5)]


class TestDataFactory:
    """测试数据工厂"""
    
    @staticmethod
    def create_user_data(username: str = None) -> Dict[str, Any]:
        """创建用户数据"""
        if username is None:
            username = f"test_user_{int(time.time())}"
        
        return {
            'username': username,
            'email': f'{username}@example.com',
            'password': f'{username}_password_123',
            'full_name': f'Test User {username}'
        }
    
    @staticmethod
    def create_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """创建市场数据"""
        import random
        
        return {
            'type': 'market_data',
            'symbol': symbol,
            'price': round(random.uniform(40000, 60000), 2),
            'volume': round(random.uniform(0.1, 10.0), 4),
            'high': round(random.uniform(45000, 65000), 2),
            'low': round(random.uniform(35000, 55000), 2),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def create_websocket_message(msg_type: str, **kwargs) -> Dict[str, Any]:
        """创建WebSocket消息"""
        message = {
            'type': msg_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        return message
    
    @staticmethod
    def create_attack_payload(attack_type: str) -> str:
        """创建攻击载荷"""
        payloads = {
            'sql_injection': "'; DROP TABLE users; --",
            'xss': "<script>alert('xss')</script>",
            'command_injection': "; ls -la",
            'large_payload': 'X' * (10 * 1024 * 1024),  # 10MB
            'malformed_json': '{"incomplete": '
        }
        return payloads.get(attack_type, "generic_attack_payload")


@pytest.fixture
def test_data_factory() -> TestDataFactory:
    """测试数据工厂"""
    return TestDataFactory()


class PerformanceTimer:
    """性能计时器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.durations = []
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self):
        """结束计时"""
        self.end_time = time.time()
        if self.start_time:
            duration = self.end_time - self.start_time
            self.durations.append(duration)
            return duration
        return 0
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.durations:
            return {}
        
        import statistics
        return {
            'count': len(self.durations),
            'total': sum(self.durations),
            'average': statistics.mean(self.durations),
            'median': statistics.median(self.durations),
            'min': min(self.durations),
            'max': max(self.durations),
            'stddev': statistics.stdev(self.durations) if len(self.durations) > 1 else 0
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@pytest.fixture
def performance_timer() -> PerformanceTimer:
    """性能计时器"""
    return PerformanceTimer()


# 标记定义
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


# 测试会话钩子
def pytest_sessionstart(session):
    """测试会话开始"""
    print(f"\n=== 集成测试会话开始 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目根目录: {PROJECT_ROOT}")


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束"""
    print(f"\n=== 集成测试会话结束 ===")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"退出状态: {exitstatus}")


# 测试失败时的处理
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """创建测试报告"""
    outcome = yield
    rep = outcome.get_result()
    
    # 为失败的测试添加额外信息
    if rep.when == "call" and rep.failed:
        # 添加环境信息
        if not hasattr(rep, 'extra_info'):
            rep.extra_info = []
        
        rep.extra_info.extend([
            f"测试时间: {datetime.now().isoformat()}",
            f"Python版本: {sys.version}",
            f"工作目录: {os.getcwd()}"
        ])


# 清理函数
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """测试后自动清理"""
    yield
    
    # 清理可能残留的asyncio任务
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    for task in tasks:
        task.cancel()
    
    # 等待任务取消完成
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# 环境变量设置
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT / "src"))


# 日志配置
@pytest.fixture(autouse=True)
def setup_logging():
    """设置测试日志"""
    import logging
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 降低一些第三方库的日志级别
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


# 超时处理
@pytest.fixture(autouse=True)
def test_timeout():
    """设置测试超时"""
    import signal
    
    def timeout_handler(signum, frame):
        pytest.fail("测试超时")
    
    # 设置测试超时时间（5分钟）
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    
    yield
    
    # 取消超时
    signal.alarm(0)


# 资源使用监控
@pytest.fixture
def resource_monitor():
    """资源使用监控"""
    import psutil
    
    process = psutil.Process()
    
    # 记录初始状态
    initial_memory = process.memory_info().rss
    initial_cpu_percent = process.cpu_percent()
    
    yield {
        'get_memory_usage': lambda: process.memory_info().rss - initial_memory,
        'get_cpu_percent': lambda: process.cpu_percent(),
        'initial_memory': initial_memory,
        'initial_cpu_percent': initial_cpu_percent
    }


# 错误收集器
class ErrorCollector:
    """错误收集器"""
    
    def __init__(self):
        self.errors = []
    
    def add_error(self, error: Exception, context: str = ""):
        """添加错误"""
        self.errors.append({
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'type': type(error).__name__
        })
    
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """获取错误摘要"""
        if not self.errors:
            return "无错误"
        
        summary = f"共发现 {len(self.errors)} 个错误:\n"
        for i, error_info in enumerate(self.errors, 1):
            summary += f"{i}. {error_info['type']}: {error_info['error']}"
            if error_info['context']:
                summary += f" (上下文: {error_info['context']})"
            summary += "\n"
        
        return summary


@pytest.fixture
def error_collector() -> ErrorCollector:
    """错误收集器"""
    return ErrorCollector()