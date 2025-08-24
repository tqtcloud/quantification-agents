#!/bin/bash

# 多维度技术指标引擎示例启动脚本

set -e

echo "🚀 启动多维度技术指标引擎示例"
echo "================================"

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 未找到虚拟环境，请先运行 scripts/setup.sh"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import numpy, pandas" || {
    echo "❌ 缺少必要依赖，请运行 pip install numpy pandas"
    exit 1
}

# 设置Python路径
export PYTHONPATH=.

echo "📊 运行多维度信号生成示例..."
python3 examples/multidimensional_signal_example.py

echo ""
echo "✅ 示例运行完成"
echo ""
echo "💡 提示："
echo "- 查看 docs/multidimensional_engine.md 了解详细文档"
echo "- 运行 python3 tests/core/engine/test_multidimensional_engine.py 进行测试"
echo "- 查看 src/core/engine/multidimensional_engine.py 了解实现细节"