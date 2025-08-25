#!/bin/bash

# 量化交易API开发模式启动脚本

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 进入项目目录
cd "$PROJECT_ROOT"

# 检查Python环境
if [ ! -d ".venv" ]; then
    echo "❌ Python virtual environment not found. Please run setup.sh first."
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 创建日志目录
mkdir -p logs/api

# 设置开发环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export LOG_LEVEL="DEBUG"
export API_HOST="127.0.0.1"
export API_PORT="8000"
export API_DEBUG="true"
export API_RELOAD="true"

echo "🔧 Starting Trading API in Development Mode..."
echo "📍 Host: $API_HOST"
echo "🔌 Port: $API_PORT"
echo "🐛 Debug: ON"
echo "🔄 Auto-reload: ON"
echo "📚 Docs: http://$API_HOST:$API_PORT/docs"
echo "🔍 ReDoc: http://$API_HOST:$API_PORT/redoc"
echo "=================================="

# 使用uvicorn直接启动（开发模式）
uvicorn \
    "src.api.trading_api:app" \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --reload \
    --reload-dir src \
    --log-level debug \
    --access-log \
    --loop uvloop \
    --factory