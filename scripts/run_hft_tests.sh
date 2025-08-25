#!/bin/bash

# 高频交易系统完整测试执行脚本
# 包括性能测试、功能测试、集成测试和可靠性测试

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 创建日志目录
LOGS_DIR="logs/tests"
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$LOGS_DIR/hft_tests_$TIMESTAMP.log"

log_info "高频交易系统完整测试开始..."
log_info "测试日志将保存到: $TEST_LOG"

# 检查Python环境
log_info "检查Python环境..."
if ! command -v python &> /dev/null; then
    log_error "Python未安装或不在PATH中"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
log_info "Python版本: $PYTHON_VERSION"

# 检查虚拟环境
if [[ -d ".venv" ]]; then
    log_info "激活虚拟环境..."
    source .venv/bin/activate
else
    log_warning "未找到虚拟环境，将使用系统Python"
fi

# 安装测试依赖
log_info "安装测试依赖..."
if [[ -f "pyproject.toml" ]]; then
    uv pip install -e ".[test]" >> "$TEST_LOG" 2>&1 || {
        log_warning "使用uv安装失败，尝试pip..."
        pip install -e ".[test]" >> "$TEST_LOG" 2>&1
    }
else
    pip install pytest pytest-asyncio pytest-cov psutil >> "$TEST_LOG" 2>&1
fi

# 检查必要的模块是否可导入
log_info "检查模块导入..."
python -c "
try:
    import pytest
    import asyncio 
    import psutil
    print('✅ 所有必要模块都可以导入')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    exit(1)
" || {
    log_error "模块导入检查失败"
    exit 1
}

# 测试函数
run_test_suite() {
    local test_name="$1"
    local test_path="$2" 
    local extra_args="$3"
    
    log_info "运行 $test_name..."
    
    local test_output_file="$LOGS_DIR/${test_name}_$TIMESTAMP.log"
    
    if pytest "$test_path" $extra_args -v --tb=short > "$test_output_file" 2>&1; then
        log_success "$test_name 通过"
        # 显示简要结果
        tail -n 10 "$test_output_file" | grep -E "(PASSED|FAILED|ERROR|===)"
    else
        log_error "$test_name 失败"
        log_error "详细错误信息请查看: $test_output_file"
        # 显示错误摘要
        tail -n 20 "$test_output_file" | grep -E "(FAILED|ERROR|AssertionError)"
        return 1
    fi
}

# 测试执行计划
declare -a test_suites=(
    "性能测试:tests/test_hft_system_validation.py::TestSystemPerformance:-x"
    "功能测试:tests/test_hft_system_validation.py::TestFunctionalValidation:-x"
    "集成测试:tests/test_hft_system_validation.py::TestIntegrationValidation:-x"
    "可靠性测试:tests/test_hft_system_validation.py::TestReliabilityValidation:-x"
    "基准测试:tests/test_hft_system_validation.py::TestSystemBenchmarks:-s --tb=short"
)

# 执行所有测试套件
failed_tests=0
total_tests=${#test_suites[@]}

log_info "开始执行 $total_tests 个测试套件..."
echo "=============================================="

for test_suite in "${test_suites[@]}"; do
    IFS=':' read -r test_name test_path extra_args <<< "$test_suite"
    
    if ! run_test_suite "$test_name" "$test_path" "$extra_args"; then
        ((failed_tests++))
    fi
    
    echo "----------------------------------------------"
done

# 运行覆盖率测试（可选）
if [[ "$1" == "--coverage" ]]; then
    log_info "运行测试覆盖率分析..."
    
    coverage_file="$LOGS_DIR/coverage_$TIMESTAMP.html"
    
    pytest tests/test_hft_system_validation.py \
        --cov=src/hft \
        --cov-report=html:"$coverage_file" \
        --cov-report=term \
        >> "$TEST_LOG" 2>&1
    
    if [[ $? -eq 0 ]]; then
        log_success "覆盖率报告生成成功: $coverage_file"
    else
        log_warning "覆盖率分析失败"
    fi
fi

# 生成测试报告
log_info "生成测试报告..."

cat << EOF > "$LOGS_DIR/test_report_$TIMESTAMP.md"
# 高频交易系统测试报告

## 测试概况
- 测试时间: $(date)
- Python版本: $PYTHON_VERSION
- 测试套件总数: $total_tests
- 通过数量: $((total_tests - failed_tests))
- 失败数量: $failed_tests

## 测试结果
EOF

for test_suite in "${test_suites[@]}"; do
    IFS=':' read -r test_name test_path extra_args <<< "$test_suite"
    test_result_file="$LOGS_DIR/${test_name}_$TIMESTAMP.log"
    
    if [[ -f "$test_result_file" ]]; then
        echo "### $test_name" >> "$LOGS_DIR/test_report_$TIMESTAMP.md"
        
        # 提取关键信息
        if grep -q "FAILED" "$test_result_file"; then
            echo "❌ 状态: 失败" >> "$LOGS_DIR/test_report_$TIMESTAMP.md"
        else
            echo "✅ 状态: 通过" >> "$LOGS_DIR/test_report_$TIMESTAMP.md"
        fi
        
        # 提取性能数据
        grep -E "(延迟|吞吐|内存|成功率)" "$test_result_file" | head -5 >> "$LOGS_DIR/test_report_$TIMESTAMP.md"
        echo "" >> "$LOGS_DIR/test_report_$TIMESTAMP.md"
    fi
done

# 测试结果汇总
echo "=============================================="
log_info "测试执行完毕"
log_info "测试报告: $LOGS_DIR/test_report_$TIMESTAMP.md"

if [[ $failed_tests -eq 0 ]]; then
    log_success "所有测试套件通过! 🎉"
    
    # 运行系统基准测试
    log_info "运行系统基准评估..."
    python -c "
import asyncio
import sys
sys.path.append('.')
from tests.test_hft_system_validation import TestSystemBenchmarks

async def run_benchmark():
    benchmark = TestSystemBenchmarks()
    await benchmark.test_system_benchmark_suite()

if __name__ == '__main__':
    asyncio.run(run_benchmark())
" >> "$TEST_LOG" 2>&1
    
    echo ""
    log_success "系统已通过所有测试，可以部署到生产环境! 🚀"
    exit 0
else
    log_error "有 $failed_tests 个测试套件失败"
    log_error "请检查测试日志并修复问题后重试"
    exit 1
fi