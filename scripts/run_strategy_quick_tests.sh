#!/bin/bash
# 快速策略测试脚本 - 适用于CI/CD和快速验证

set -e  # 遇到错误立即退出

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
QUICK_TEST_LOG="$LOG_DIR/quick_strategy_tests_$TIMESTAMP.log"

echo -e "${BLUE}⚡ 开始快速策略测试验证${NC}"
echo "测试日志: $QUICK_TEST_LOG"
echo "=============================================" | tee -a "$QUICK_TEST_LOG"

# 函数：记录日志消息
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$QUICK_TEST_LOG"
}

# 函数：运行快速测试
run_quick_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "${YELLOW}🔍 运行: $test_name${NC}"
    log_message "开始: $test_name"
    
    if eval "$test_command" >> "$QUICK_TEST_LOG" 2>&1; then
        echo -e "${GREEN}✅ 通过: $test_name${NC}"
        log_message "通过: $test_name"
        return 0
    else
        echo -e "${RED}❌ 失败: $test_name${NC}"
        log_message "失败: $test_name"
        return 1
    fi
}

# 主执行函数
main() {
    local exit_code=0
    local failed_tests=()
    
    log_message "开始快速策略测试验证"
    
    # 检查环境
    echo -e "${BLUE}🔍 检查测试环境${NC}"
    if ! command -v uv >/dev/null; then
        echo -e "${RED}❌ 未找到uv命令${NC}"
        exit 1
    fi
    
    # 安装依赖（只安装必要的）
    echo "快速安装测试依赖..."
    uv sync --dev --no-progress
    
    # 1. 导入测试 - 验证所有模块可以正确导入
    if ! run_quick_test "模块导入验证" "uv run python -c 'from src.strategy import *; print(\"所有策略模块导入成功\"))'"; then
        failed_tests+=("import_test")
        exit_code=1
    fi
    
    # 2. 基础单元测试 - 只运行快速单元测试
    if ! run_quick_test "基础单元测试" "uv run pytest tests/strategy/ -m 'unit or not slow' --tb=line -q --disable-warnings --maxfail=3"; then
        failed_tests+=("unit_tests")
        exit_code=1
    fi
    
    # 3. 关键组件快速测试
    if ! run_quick_test "策略管理器快速测试" "uv run pytest tests/strategy/test_strategy_manager.py::TestStrategyManager::test_create_strategy --tb=line -q --disable-warnings"; then
        failed_tests+=("strategy_manager_basic")
        exit_code=1
    fi
    
    # 4. 信号聚合器快速测试  
    if ! run_quick_test "信号聚合器快速测试" "uv run pytest tests/test_signal_aggregator.py -k 'test_basic' --tb=line -q --disable-warnings --maxfail=1"; then
        failed_tests+=("signal_aggregator_basic")
        exit_code=1
    fi
    
    # 5. 性能基准快速检查
    if ! run_quick_test "性能基准快速检查" "uv run pytest tests/strategy/test_performance_benchmarks.py::TestBenchmarkSuite::test_quick_performance_check --tb=line -q --disable-warnings"; then
        failed_tests+=("performance_quick_check")
        exit_code=1
    fi
    
    # 6. 代码质量检查（如果有工具）
    if command -v ruff >/dev/null; then
        if ! run_quick_test "代码质量检查" "ruff check src/strategy/ --select E,W,F"; then
            failed_tests+=("code_quality")
            exit_code=1
        fi
    fi
    
    # 测试结果汇总
    echo "=============================================" | tee -a "$QUICK_TEST_LOG"
    log_message "快速测试验证完成"
    
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}🎉 快速验证通过！所有关键测试正常${NC}"
        log_message "快速验证通过"
        
        # 显示测试摘要
        echo -e "${BLUE}📊 测试摘要:${NC}"
        echo "  ✅ 模块导入: 正常"
        echo "  ✅ 单元测试: 通过"
        echo "  ✅ 核心组件: 正常"
        echo "  ✅ 性能基准: 达标"
        
    else
        echo -e "${RED}❌ 快速验证失败${NC}"
        log_message "失败的测试: ${failed_tests[*]}"
        
        echo -e "${YELLOW}失败的测试项:${NC}"
        for test in "${failed_tests[@]}"; do
            echo "  - $test"
        done
        
        echo ""
        echo -e "${YELLOW}💡 建议操作:${NC}"
        echo "  1. 查看详细日志: cat $QUICK_TEST_LOG"
        echo "  2. 运行完整测试: ./scripts/run_strategy_integration_tests.sh"
        echo "  3. 检查失败的具体测试项"
    fi
    
    echo -e "${BLUE}📁 测试日志: $QUICK_TEST_LOG${NC}"
    echo "============================================="
    
    return $exit_code
}

# 处理命令行参数
case "${1:-all}" in
    "import")
        run_quick_test "模块导入验证" "uv run python -c 'from src.strategy import *; print(\"导入成功\")'"
        ;;
    "unit")
        run_quick_test "基础单元测试" "uv run pytest tests/strategy/ -m 'unit' --tb=line -q --disable-warnings"
        ;;
    "performance")
        run_quick_test "性能快速检查" "uv run pytest tests/strategy/test_performance_benchmarks.py::TestBenchmarkSuite::test_quick_performance_check --tb=line -q"
        ;;
    "quality")
        if command -v ruff >/dev/null; then
            run_quick_test "代码质量检查" "ruff check src/strategy/"
        else
            echo -e "${YELLOW}⚠️  ruff未安装，跳过代码质量检查${NC}"
        fi
        ;;
    "all"|*)
        main
        exit $?
        ;;
esac