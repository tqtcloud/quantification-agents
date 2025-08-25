#!/bin/bash

# WebSocket客户端测试脚本
# 用于测试WebSocket服务器的连接和消息接收

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHON_ENV=${PYTHON_ENV:-development}

echo "=== WebSocket客户端测试 ==="
echo "项目目录: $PROJECT_DIR"
echo "环境: $PYTHON_ENV"
echo "时间: $(date)"
echo

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未找到"
    exit 1
fi

# 检查依赖
if ! python -c "import websockets, asyncio" 2>/dev/null; then
    echo "错误: 缺少WebSocket依赖，请先安装："
    echo "pip install websockets"
    exit 1
fi

# 创建日志目录
mkdir -p "$PROJECT_DIR/logs"

# 设置日志文件
LOG_FILE="$PROJECT_DIR/logs/websocket_client.log"

echo "启动WebSocket客户端测试..."
echo "日志输出到: $LOG_FILE"
echo "连接地址: ws://localhost:8765"
echo "测试时长: 90秒"
echo

# 启动WebSocket客户端
cd "$PROJECT_DIR"
python -c "
import asyncio
import sys
import os
import logging

# 添加项目路径
sys.path.insert(0, os.getcwd())

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/websocket_client.log'),
        logging.StreamHandler()
    ]
)

# 导入和运行WebSocket客户端
from examples.websocket_realtime_demo import WebSocketClient

async def main():
    client = WebSocketClient('ws://localhost:8765')
    
    try:
        print('连接到WebSocket服务器...')
        receive_task = await client.connect()
        
        print('订阅数据类型...')
        await client.subscribe('trading_signals', {'symbol': 'BTCUSDT'})
        await client.subscribe('strategy_status')
        await client.subscribe('market_data', {'symbol': 'BTCUSDT'})
        await client.subscribe('risk_alerts')
        await client.subscribe('system_monitor')
        
        print('开始接收消息（60秒）...')
        await asyncio.sleep(60)
        
        print('取消部分订阅...')
        await client.unsubscribe('market_data')
        await client.unsubscribe('system_monitor')
        
        print('继续接收消息（30秒）...')
        await asyncio.sleep(30)
        
        print('测试完成')
        
    except KeyboardInterrupt:
        print('\\n收到中断信号，断开连接...')
    except Exception as e:
        print(f'客户端错误: {e}')
    finally:
        await client.disconnect()
        print('客户端已断开连接')

if __name__ == '__main__':
    print('启动WebSocket客户端测试...')
    print('按Ctrl+C提前停止测试')
    asyncio.run(main())
" 2>&1 | tee -a "$LOG_FILE"

echo
echo "WebSocket客户端测试完成"