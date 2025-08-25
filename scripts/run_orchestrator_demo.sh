#!/bin/bash

# 运行MultiAgentOrchestrator演示脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 创建日志目录
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/orchestrator_demo_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "MultiAgentOrchestrator 演示"
echo "=========================================="
echo "项目目录: $PROJECT_ROOT"
echo "日志文件: $LOG_FILE"
echo ""

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "错误: 虚拟环境不存在，请先运行: uv venv"
    exit 1
fi

# 激活虚拟环境并运行
echo "启动演示..."
echo ""

# 运行演示，同时输出到控制台和日志文件
uv run python examples/orchestrator_demo.py 2>&1 | tee "$LOG_FILE"

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "演示执行成功!"
    echo "日志已保存到: $LOG_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "演示执行失败!"
    echo "请查看日志: $LOG_FILE"
    echo "=========================================="
    exit 1
fi