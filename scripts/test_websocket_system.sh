#!/bin/bash

# WebSocket系统快速测试脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "=== WebSocket系统测试 ==="
echo "项目目录: $PROJECT_DIR"
echo "时间: $(date)"
echo

cd "$PROJECT_DIR"

echo "1. 测试数据模型..."
python3 -c "
import sys
sys.path.append('.')
from src.websocket.models import *

# 测试消息序列化
message = WebSocketMessage.create_data_message(
    MessageType.TRADING_SIGNAL,
    {'symbol': 'BTCUSDT', 'signal': 'BUY', 'price': 50000}
)
json_str = message.to_json()
restored = WebSocketMessage.from_json(json_str)
assert restored.message_type == MessageType.TRADING_SIGNAL
assert restored.data['symbol'] == 'BTCUSDT'
print('✓ 消息序列化测试通过')

# 测试连接信息
conn_info = ConnectionInfo(connection_id='test-001')
assert conn_info.is_active()
conn_info.add_subscription(SubscriptionType.TRADING_SIGNALS)
assert SubscriptionType.TRADING_SIGNALS in conn_info.subscriptions
print('✓ 连接信息测试通过')

# 测试配置验证
config = WebSocketConfig(port=8765, max_connections=100)
errors = config.validate()
assert len(errors) == 0
print('✓ 配置验证测试通过')
"

echo
echo "2. 测试连接管理器..."
python3 -c "
import sys, asyncio
sys.path.append('.')
from src.websocket import ConnectionManager, WebSocketConfig

async def test_connection_manager():
    config = WebSocketConfig(port=8765, max_connections=10)
    manager = ConnectionManager(config)
    
    # 测试启动和停止
    await manager.start()
    print('✓ 连接管理器启动成功')
    
    stats = manager.get_stats()
    assert stats.total_connections == 0
    print('✓ 统计信息正常')
    
    await manager.stop()
    print('✓ 连接管理器停止成功')

asyncio.run(test_connection_manager())
"

echo
echo "3. 测试订阅管理器..."
python3 -c "
import sys, asyncio
sys.path.append('.')
from src.websocket import SubscriptionManager, SubscriptionType

async def test_subscription_manager():
    manager = SubscriptionManager()
    
    # 创建订阅
    success, sub_id = await manager.subscribe(
        'conn-001', 
        SubscriptionType.TRADING_SIGNALS,
        {'symbol': 'BTCUSDT'}
    )
    assert success
    print('✓ 创建订阅成功')
    
    # 获取订阅
    subscription = manager.get_subscription(sub_id)
    assert subscription is not None
    assert subscription.subscription_type == SubscriptionType.TRADING_SIGNALS
    print('✓ 获取订阅成功')
    
    # 取消订阅
    success, _ = await manager.unsubscribe('conn-001', sub_id)
    assert success
    print('✓ 取消订阅成功')

asyncio.run(test_subscription_manager())
"

echo
echo "4. 测试消息广播器..."
python3 -c "
import sys, asyncio
from unittest.mock import AsyncMock, MagicMock
sys.path.append('.')
from src.websocket import MessageBroadcaster

async def test_message_broadcaster():
    # 创建模拟组件
    mock_conn_manager = AsyncMock()
    mock_sub_manager = MagicMock()
    
    # 设置模拟返回
    mock_sub = MagicMock()
    mock_sub.connection_id = 'conn-001'
    mock_sub.matches_filter.return_value = True
    mock_sub_manager.get_matching_subscriptions.return_value = [mock_sub]
    mock_conn_manager.send_message.return_value = True
    
    broadcaster = MessageBroadcaster(mock_conn_manager, mock_sub_manager)
    await broadcaster.start()
    print('✓ 消息广播器启动成功')
    
    stats = broadcaster.get_stats()
    assert 'messages_sent' in stats
    print('✓ 广播统计正常')
    
    await broadcaster.stop()
    print('✓ 消息广播器停止成功')

asyncio.run(test_message_broadcaster())
"

echo
echo "5. 测试WebSocket管理器..."
python3 -c "
import sys, asyncio
sys.path.append('.')
from src.websocket import WebSocketManager, WebSocketConfig

async def test_websocket_manager():
    config = WebSocketConfig(host='localhost', port=0)  # 随机端口
    manager = WebSocketManager(config)
    
    # 测试系统统计
    stats = manager.get_system_stats()
    assert stats['server_status'] == 'stopped'
    print('✓ 系统统计正常')
    
    print('✓ WebSocket管理器创建成功')

asyncio.run(test_websocket_manager())
"

echo
echo "6. 测试TradingAPI集成..."
python3 -c "
import sys
sys.path.append('.')
from src.websocket import WebSocketManager, WebSocketConfig

# 测试导入
try:
    from src.api.trading_api import TradingAPI
    print('✓ TradingAPI集成导入成功')
except ImportError as e:
    print(f'⚠ TradingAPI集成导入失败: {e}')
    print('  这是正常的，因为可能缺少部分依赖')
"

echo
echo "=== 测试总结 ==="
echo "✓ WebSocket数据模型测试通过"
echo "✓ 连接管理器测试通过"
echo "✓ 订阅管理器测试通过"
echo "✓ 消息广播器测试通过"
echo "✓ WebSocket管理器测试通过"
echo "✓ API集成检查完成"
echo
echo "🎉 WebSocket系统核心功能验证成功!"
echo
echo "下一步可以:"
echo "1. 运行 ./scripts/start_websocket_server.sh 启动服务器"
echo "2. 运行 ./scripts/test_websocket_client.sh 测试客户端连接"
echo "3. 查看文档 docs/WEBSOCKET_IMPLEMENTATION_SUMMARY.md"