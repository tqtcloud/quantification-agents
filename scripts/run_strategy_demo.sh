#!/bin/bash

# 策略管理系统演示脚本

set -e

echo "=== 策略管理系统演示 ==="

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

# 创建日志目录
mkdir -p logs

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export LOG_LEVEL="INFO"

echo "🚀 启动策略管理系统演示..."

# 运行演示
python3 examples/strategy_management_demo.py 2>&1 | tee logs/strategy_demo_$(date +%Y%m%d_%H%M%S).log

echo "✅ 演示完成，日志已保存到 logs/ 目录"