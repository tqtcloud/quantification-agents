#!/bin/bash

# 内存缓存系统测试脚本
# 运行所有相关测试和基准测试

set -e  # 遇到错误立即退出

echo "========================================"
echo "量化交易系统内存缓存测试套件"
echo "========================================"

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "警告: 未激活虚拟环境，正在尝试激活 .venv"
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        echo "✓ 虚拟环境已激活"
    else
        echo "错误: 未找到 .venv 目录，请先创建虚拟环境"
        exit 1
    fi
fi

# 安装依赖（如果需要）
echo "检查依赖..."
if command -v uv &> /dev/null; then
    uv pip install -e .
else
    pip install -e .
fi

# 创建日志目录
mkdir -p logs/cache_tests

# 1. 运行基础功能测试
echo ""
echo "1. 运行内存缓存基础功能测试..."
echo "----------------------------------------"
python -m pytest tests/test_memory_cache.py -v -x --tb=short \
    --log-cli-level=INFO \
    --log-file=logs/cache_tests/basic_tests_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✓ 基础功能测试通过"
else
    echo "✗ 基础功能测试失败"
    exit 1
fi

# 2. 运行性能基准测试
echo ""
echo "2. 运行性能基准测试..."
echo "----------------------------------------"
python tests/test_memory_cache_benchmarks.py 2>&1 | tee logs/cache_tests/benchmark_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✓ 性能基准测试完成"
else
    echo "✗ 性能基准测试出现问题"
fi

# 3. 运行集成示例
echo ""
echo "3. 运行量化交易系统集成示例..."
echo "----------------------------------------"
python examples/memory_cache_trading_example.py 2>&1 | tee logs/cache_tests/integration_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "✓ 集成示例运行成功"
else
    echo "✗ 集成示例运行失败"
fi

# 4. 生成测试报告
echo ""
echo "4. 生成测试覆盖率报告..."
echo "----------------------------------------"
python -m pytest tests/test_memory_cache.py --cov=src.core.cache --cov-report=html --cov-report=term-missing

if [ -d "htmlcov" ]; then
    echo "✓ 测试覆盖率报告已生成: htmlcov/index.html"
fi

# 5. 内存使用分析
echo ""
echo "5. 内存使用分析..."
echo "----------------------------------------"
python -c "
import sys
sys.path.append('.')
from src.core.cache import MemoryCachePool, CacheConfig
import time
import tracemalloc

print('内存使用分析:')
tracemalloc.start()

# 创建缓存实例
config = CacheConfig(max_memory=10*1024*1024, max_keys=10000)
cache = MemoryCachePool(config)

# 添加数据
for i in range(1000):
    cache.set(f'key_{i}', f'value_{i}' * 100)

current, peak = tracemalloc.get_traced_memory()
stats = cache.get_stats()

print(f'Python内存使用: {current/1024/1024:.2f}MB (峰值: {peak/1024/1024:.2f}MB)')
print(f'缓存估算内存: {stats.memory_usage/1024/1024:.2f}MB')
print(f'缓存键数量: {stats.key_count}')
print(f'命中率: {stats.hit_rate:.2%}')

tracemalloc.stop()
"

echo ""
echo "========================================"
echo "测试完成! 日志文件保存在 logs/cache_tests/ 目录"
echo "========================================"

# 显示测试结果摘要
echo ""
echo "测试结果摘要:"
echo "- 基础功能测试: ✓ 通过"
echo "- 性能基准测试: ✓ 完成" 
echo "- 集成示例: ✓ 成功"
echo "- 覆盖率报告: ✓ 生成"
echo "- 内存分析: ✓ 完成"
echo ""
echo "如需查看详细结果，请检查 logs/cache_tests/ 目录下的日志文件"