#!/bin/bash

# 简化策略管理系统演示脚本

set -e

echo "=== 简化策略管理系统演示 ==="

# 获取脚本所在目录的父目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "📁 项目目录: $PROJECT_DIR"

# 设置环境变量
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export LOG_LEVEL="INFO"

echo "🚀 启动简化策略管理系统演示..."

# 运行简化演示
python3 examples/simple_strategy_demo.py

echo "✅ 简化演示完成"