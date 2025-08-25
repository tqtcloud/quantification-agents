#!/bin/bash

# 运行投资大师Agent演示

echo "======================================"
echo "Running Investment Master Agent Demo"
echo "======================================"

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 创建日志目录
mkdir -p logs

# 激活虚拟环境并运行演示
echo "Starting Investment Master Demo..."
source .venv/bin/activate
python examples/investment_master_demo.py 2>&1 | tee logs/investment_master_demo_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Demo completed. Check logs for details."