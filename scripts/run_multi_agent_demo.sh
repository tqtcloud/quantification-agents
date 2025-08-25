#!/bin/bash

# 多Agent智能分析系统综合演示启动脚本
# 展示完整的17-Agent协作投资决策流程

set -e

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎯 多Agent智能分析系统综合演示${NC}"
echo "================================================================="
echo -e "${PURPLE}该演示将展示：${NC}"
echo "1. 🏛️  Agent状态管理系统 (任务3.1)"
echo "2. 🤖 投资大师Agent基类 (任务3.2)" 
echo "3. 👥 15个专业分析师Agent集群 (任务3.3)"
echo "4. 🛡️  风险管理和投资组合管理Agent (任务3.4)"
echo "5. 🔄 LangGraph工作流编排系统 (任务3.5)"
echo ""

# 检查 Python 虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}📦 创建 Python 虚拟环境...${NC}"
    python3 -m venv .venv
fi

# 激活虚拟环境
echo -e "${GREEN}🔧 激活虚拟环境...${NC}"
source .venv/bin/activate

# 检查并安装依赖
echo -e "${GREEN}📚 检查项目依赖...${NC}"
pip install --quiet -e . 2>/dev/null || true

# 确保必要的包安装
python3 -c "import langchain, langgraph" 2>/dev/null || {
    echo -e "${YELLOW}📦 安装LangChain依赖...${NC}"
    pip install --quiet langchain langgraph langchain-openai
}

python3 -c "import numpy, pandas, scipy, cvxpy" 2>/dev/null || {
    echo -e "${YELLOW}📦 安装数据科学依赖...${NC}"
    pip install --quiet numpy pandas scipy cvxpy
}

# 创建必要的目录
echo -e "${GREEN}📁 创建必要目录...${NC}"
mkdir -p logs
mkdir -p data
mkdir -p temp

# 检查核心模块
echo -e "${GREEN}🔍 检查核心模块...${NC}"
python3 -c "
import sys
sys.path.append('.')

try:
    from src.agents.state_manager import AgentStateManager
    from src.agents.base_agent import InvestmentMasterAgent
    from src.agents.investment_masters import INVESTMENT_MASTERS
    from src.agents.management.risk_management import RiskManagementAgent
    from src.agents.orchestrator import MultiAgentOrchestrator
    print('✅ 所有核心模块检查通过')
except ImportError as e:
    print(f'❌ 模块检查失败: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 核心模块检查失败，请检查系统完整性${NC}"
    exit 1
fi

# 运行综合演示
echo -e "${GREEN}🚀 启动多Agent智能分析系统演示...${NC}"
echo "================================================================="

PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python examples/multi_agent_system_demo.py

DEMO_EXIT_CODE=$?

echo ""
echo "================================================================="
if [ $DEMO_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 多Agent智能分析系统演示成功完成！${NC}"
    echo ""
    echo -e "${YELLOW}📊 演示成果总结：${NC}"
    echo "✅ Agent状态管理系统 - 17个Agent状态同步"
    echo "✅ 投资大师Agent基类 - LLM调用和分析框架"
    echo "✅ 专业分析师集群 - 15个投资大师并行分析"
    echo "✅ 管理Agent系统 - 风险评估和组合优化"
    echo "✅ LangGraph工作流编排 - 完整决策流水线"
    echo ""
    echo -e "${BLUE}💡 下一步建议：${NC}"
    echo "1. 查看日志文件了解详细执行过程: logs/"
    echo "2. 运行单独的Agent测试了解各组件: ./scripts/"
    echo "3. 集成到完整的量化交易系统中"
    echo "4. 配置真实的LLM API密钥进行实际分析"
else
    echo -e "${RED}❌ 演示过程中出现问题${NC}"
    echo ""
    echo -e "${YELLOW}🔧 问题排查建议：${NC}"
    echo "1. 检查日志文件: logs/"
    echo "2. 确认所有依赖正确安装"
    echo "3. 检查API配置和网络连接"
    echo "4. 运行单独的组件测试定位问题"
fi

echo ""
echo -e "${BLUE}📁 相关文件位置：${NC}"
echo "- 演示程序: examples/multi_agent_system_demo.py"
echo "- Agent代码: src/agents/"
echo "- 测试用例: tests/agents/"
echo "- 日志文件: logs/"
echo "- 任务文档: .kiro/specs/core-trading-logic/tasks.md"