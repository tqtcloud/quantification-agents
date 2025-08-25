#!/bin/bash

# Agent状态管理测试脚本
# 测试Agent状态管理系统的功能

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

echo -e "${BLUE}🧪 Agent状态管理测试${NC}"
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
uv pip install -e . >/dev/null 2>&1
uv pip install pytest pytest-asyncio >/dev/null 2>&1

# 创建必要的目录
echo -e "${GREEN}📁 创建必要目录...${NC}"
mkdir -p logs
mkdir -p data/agent_states

# 运行测试
echo -e "${GREEN}🚀 运行Agent状态管理测试...${NC}"
echo "============================================"

# 运行测试并捕获输出
if python -m pytest tests/agents/test_state_manager.py -v --tb=short; then
    echo ""
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    exit_code=0
else
    echo ""
    echo -e "${RED}❌ 部分测试失败，请检查错误信息${NC}"
    exit_code=1
fi

echo ""
echo -e "${YELLOW}💡 测试完成${NC}"
echo "查看详细日志: logs/"

exit $exit_code