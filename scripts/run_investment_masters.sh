#!/bin/bash

# 投资大师Agent集群运行脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}     投资大师Agent集群测试系统        ${NC}"
echo -e "${GREEN}========================================${NC}"

# 切换到项目根目录
cd /Users/mrtang/Documents/project-ai/quantification-agents

# 激活虚拟环境
echo -e "${YELLOW}激活Python虚拟环境...${NC}"
source .venv/bin/activate 2>/dev/null || {
    echo -e "${RED}虚拟环境未找到，请先运行: uv venv${NC}"
    exit 1
}

# 设置Python路径
export PYTHONPATH=/Users/mrtang/Documents/project-ai/quantification-agents:$PYTHONPATH

# 创建日志目录
mkdir -p logs

# 运行测试
echo -e "${YELLOW}运行投资大师Agent测试...${NC}"
echo ""

# 测试单个Agent
echo -e "${GREEN}1. 测试Warren Buffett Agent${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_warren_buffett_agent -v

echo ""
echo -e "${GREEN}2. 测试Cathie Wood Agent${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_cathie_wood_agent -v

echo ""
echo -e "${GREEN}3. 测试Ray Dalio Agent${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_ray_dalio_agent -v

echo ""
echo -e "${GREEN}4. 测试技术分析专家Agent${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_technical_analyst_agent -v

echo ""
echo -e "${GREEN}5. 测试量化分析师Agent${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_quantitative_analyst_agent -v

echo ""
echo -e "${GREEN}6. 测试Agent注册器${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_agent_registry -v

echo ""
echo -e "${GREEN}7. 测试Agent编排器（共识决策）${NC}"
python -m pytest tests/agents/test_investment_masters.py::TestInvestmentMasters::test_agent_orchestrator -v

echo ""
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}运行完整测试套件...${NC}"
echo -e "${YELLOW}========================================${NC}"

# 运行完整测试
python -m pytest tests/agents/test_investment_masters.py -v --tb=short 2>&1 | tee logs/investment_masters_test_$(date +%Y%m%d_%H%M%S).log

# 检查测试结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 所有测试通过！${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ 部分测试失败，请检查日志${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}测试日志已保存到 logs/ 目录${NC}"