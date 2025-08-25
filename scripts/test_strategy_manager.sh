#!/bin/bash

# 策略管理器测试脚本

set -e

echo "=== 策略管理器测试 ==="

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📁 项目目录: $PROJECT_DIR"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请先安装Python3"
    exit 1
fi

echo "🐍 Python版本: $(python3 --version)"

# 检查pytest
if ! python3 -c "import pytest" &> /dev/null; then
    echo "📦 安装pytest..."
    python3 -m pip install pytest pytest-asyncio
fi

# 创建日志目录
mkdir -p logs

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export LOG_LEVEL="WARNING"  # 减少测试时的日志输出

echo "🧪 运行策略管理器测试..."

# 运行特定的策略管理器测试
python3 -m pytest tests/strategy/test_strategy_manager.py -v -s \
    --tb=short \
    --log-cli-level=WARNING \
    2>&1 | tee logs/strategy_test_$(date +%Y%m%d_%H%M%S).log

echo "✅ 测试完成，结果已保存到 logs/ 目录"

# 检查测试结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "🎉 所有测试通过!"
else
    echo "❌ 部分测试失败，请检查日志文件"
    exit 1
fi