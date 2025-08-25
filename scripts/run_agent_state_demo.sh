#!/bin/bash

# Agent状态管理演示脚本
# 展示Agent间的状态共享、消息传递、共识机制等功能

set -e

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}🤖 Agent状态管理系统演示${NC}"
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

# 创建必要的目录
echo -e "${GREEN}📁 创建必要目录...${NC}"
mkdir -p logs
mkdir -p data/demo_states

# 清理旧的演示数据
echo -e "${YELLOW}🧹 清理旧数据...${NC}"
rm -rf data/demo_states/*

# 运行演示
echo -e "${BLUE}🚀 启动Agent状态管理演示...${NC}"
echo "============================================"
echo ""

python examples/agent_state_demo.py

echo ""
echo "============================================"
echo -e "${GREEN}✅ 演示完成！${NC}"
echo ""
echo -e "${YELLOW}💡 功能展示:${NC}"
echo "1. ✅ Agent状态管理 - 创建、更新、合并状态"
echo "2. ✅ 共享内存 - Agent间数据共享与锁定机制"
echo "3. ✅ 消息传递 - 点对点和广播消息"
echo "4. ✅ 共识机制 - 多Agent投票决策"
echo "5. ✅ 序列化/反序列化 - 状态持久化"
echo "6. ✅ 检查点管理 - 状态恢复"
echo "7. ✅ 性能追踪 - Agent性能统计"
echo ""
echo -e "${CYAN}📊 查看保存的状态文件:${NC}"
echo "ls -la data/demo_states/"
ls -la data/demo_states/ 2>/dev/null || echo "  (无状态文件)"
echo ""
echo -e "${BLUE}📝 相关文件:${NC}"
echo "- 状态管理器: src/agents/state_manager.py"
echo "- 数据模型: src/agents/models.py"
echo "- 单元测试: tests/agents/test_state_manager.py"
echo "- 演示代码: examples/agent_state_demo.py"