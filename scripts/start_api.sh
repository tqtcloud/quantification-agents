#!/bin/bash

# 量化交易API服务启动脚本

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

# 检查依赖
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ Missing dependencies. Installing..."
    uv pip install -r requirements.txt
fi

# 创建日志目录
mkdir -p logs/api

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export API_WORKERS="${API_WORKERS:-1}"
export API_RELOAD="${API_RELOAD:-false}"

# 日志文件
LOG_FILE="logs/api/api_$(date +%Y%m%d_%H%M%S).log"

echo "🚀 Starting Trading API Server..."
echo "📍 Host: $API_HOST"
echo "🔌 Port: $API_PORT"
echo "👥 Workers: $API_WORKERS"
echo "🔄 Reload: $API_RELOAD"
echo "📝 Log file: $LOG_FILE"
echo "=================================="

# 启动API服务器
python -c "
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from src.api.trading_api import TradingAPI
from src.config import Config

def main():
    try:
        config = Config.load_from_env()
        api = TradingAPI(config)
        
        # 运行服务器
        api.run(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', 8000)),
            workers=int(os.getenv('API_WORKERS', 1)),
            reload=os.getenv('API_RELOAD', 'false').lower() == 'true'
        )
    except KeyboardInterrupt:
        print('\\n🛑 API server stopped by user')
        sys.exit(0)
    except Exception as e:
        print(f'❌ Failed to start API server: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
" 2>&1 | tee "$LOG_FILE"