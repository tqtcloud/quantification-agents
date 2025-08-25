#!/bin/bash

# 投资大师Agent集群演示脚本

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   投资大师Agent集群演示系统           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"

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

# 运行演示
echo -e "${GREEN}启动投资大师Agent集群演示...${NC}"
echo ""

# 运行演示程序
python examples/investment_masters_demo.py 2>&1 | tee logs/masters_demo_$(date +%Y%m%d_%H%M%S).log

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         ✅ 演示成功完成！              ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════╗${NC}"
    echo -e "${RED}║         ❌ 演示出现错误                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════╝${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}日志已保存到: logs/ 目录${NC}"
echo ""