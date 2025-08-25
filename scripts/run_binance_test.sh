#!/bin/bash

# 币安模拟盘交易测试启动脚本
# 使用方法: ./scripts/run_binance_test.sh [测试时长分钟]

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

echo -e "${BLUE}🎯 币安模拟盘交易测试启动脚本${NC}"
echo "=================================================="

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ 未找到 .env 文件${NC}"
    echo -e "${YELLOW}正在从 .env.example 创建 .env 文件...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  请编辑 .env 文件，配置您的币安API密钥！${NC}"
    echo "必须设置的参数："
    echo "  - BINANCE_API_KEY=your_api_key_here"
    echo "  - BINANCE_API_SECRET=your_api_secret_here"  
    echo "  - BINANCE_TESTNET=true (建议使用测试网)"
    echo ""
    read -p "配置完成后按 Enter 继续..."
fi

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
uv add python-dotenv aiohttp websockets tenacity ccxt

# 检查必要的环境变量
echo -e "${GREEN}🔍 检查环境配置...${NC}"
source .env

if [ -z "$BINANCE_API_KEY" ] || [ "$BINANCE_API_KEY" = "your_binance_api_key_here" ]; then
    echo -e "${RED}❌ 请在 .env 文件中配置有效的 BINANCE_API_KEY${NC}"
    exit 1
fi

if [ -z "$BINANCE_API_SECRET" ] || [ "$BINANCE_API_SECRET" = "your_binance_api_secret_here" ]; then
    echo -e "${RED}❌ 请在 .env 文件中配置有效的 BINANCE_API_SECRET${NC}"
    exit 1
fi

# 显示配置信息
echo -e "${BLUE}📋 当前配置:${NC}"
echo "  - 币安API密钥: ${BINANCE_API_KEY:0:10}***"
echo "  - 测试网模式: ${BINANCE_TESTNET:-false}"
echo "  - 交易模式: ${TRADING_MODE:-paper}"
echo "  - 日志级别: ${LOG_LEVEL:-INFO}"
echo ""

# 创建必要的目录
echo -e "${GREEN}📁 创建必要目录...${NC}"
mkdir -p logs
mkdir -p data

# 获取测试时长参数
DURATION=${1:-60}
echo -e "${YELLOW}⏰ 测试时长: ${DURATION} 分钟${NC}"

# 运行测试
echo -e "${GREEN}🚀 启动币安模拟盘交易测试...${NC}"
echo "=================================================="

python examples/binance_live_trading_test.py <<EOF
${DURATION}
EOF

echo ""
echo -e "${GREEN}✅ 测试完成！${NC}"
echo "日志文件位置: logs/"
echo "数据文件位置: data/"