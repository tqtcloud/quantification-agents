#!/bin/bash

# 快速信号测试启动脚本
# 不需要API密钥，快速验证系统功能

set -e

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}⚡ 快速信号测试启动脚本${NC}"
echo "============================================"

# 检查 Python 虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}📦 创建 Python 虚拟环境...${NC}"
    uv venv .venv
fi

# 激活虚拟环境
echo -e "${GREEN}🔧 激活虚拟环境...${NC}"
source .venv/bin/activate

# 安装依赖
echo -e "${GREEN}📚 安装项目依赖...${NC}"
uv pip install -e .

# 创建必要的目录
echo -e "${GREEN}📁 创建必要目录...${NC}"
mkdir -p logs
mkdir -p data

# 运行快速测试
echo -e "${GREEN}🚀 启动快速信号测试...${NC}"
echo "============================================"

python examples/quick_signal_test.py

echo ""
echo -e "${GREEN}✅ 快速测试完成！${NC}"
echo ""
echo -e "${YELLOW}💡 下一步建议:${NC}"
echo "1. 如果测试评分良好，可以配置 .env 文件并运行币安API测试"
echo "2. 使用命令: ./scripts/run_binance_test.sh"
echo "3. 查看日志文件了解详细执行情况: logs/"