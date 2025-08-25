#!/bin/bash

# WebSocket实时推送服务器启动脚本
# 用于启动完整的WebSocket服务器，包括演示数据推送

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHON_ENV=${PYTHON_ENV:-development}

echo "=== WebSocket服务器启动 ==="
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
LOG_FILE="$PROJECT_DIR/logs/websocket_server.log"

echo "启动WebSocket服务器..."
echo "日志输出到: $LOG_FILE"
echo "服务地址: ws://localhost:8765"
echo

# 启动WebSocket服务器
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
        logging.FileHandler('logs/websocket_server.log'),
        logging.StreamHandler()
    ]
)

# 导入和运行WebSocket演示
from examples.websocket_realtime_demo import WebSocketDemo

async def main():
    demo = WebSocketDemo()
    try:
        await demo.run_forever()
    except KeyboardInterrupt:
        print('\n收到中断信号，正在停止服务器...')
        await demo.stop_server()
        print('WebSocket服务器已停止')

if __name__ == '__main__':
    print('启动WebSocket实时推送服务器...')
    print('按Ctrl+C停止服务器')
    asyncio.run(main())
" 2>&1 | tee -a "$LOG_FILE"

echo
echo "WebSocket服务器已停止"