#!/bin/bash

# 测试投资大师Agent基类

echo "======================================"
echo "Testing Investment Master Agent Base Class"
echo "======================================"

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 创建日志目录
mkdir -p logs

# 激活虚拟环境并运行测试
echo "Running Investment Master Agent tests..."
source .venv/bin/activate
python -m pytest tests/agents/test_base_agent.py -v --tb=short 2>&1 | tee logs/test_base_agent_$(date +%Y%m%d_%H%M%S).log

# 检查测试结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ All tests passed successfully!"
else
    echo ""
    echo "❌ Some tests failed. Check the logs for details."
    exit 1
fi