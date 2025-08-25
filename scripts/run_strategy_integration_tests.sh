#!/bin/bash
# 运行双策略管理系统集成测试套件

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

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_LOG="$LOG_DIR/strategy_integration_tests_$TIMESTAMP.log"

echo -e "${BLUE}🚀 开始运行双策略管理系统集成测试套件${NC}"
echo "测试日志将保存到: $TEST_LOG"
echo "=============================================" | tee -a "$TEST_LOG"

# 函数：打印带时间戳的消息
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$TEST_LOG"
}

# 函数：运行测试并记录结果
run_test_suite() {
    local test_name="$1"
    local test_path="$2"
    local test_args="$3"
    
    echo -e "${YELLOW}📋 运行测试套件: $test_name${NC}"
    log_message "开始测试套件: $test_name"
    
    # 创建测试专用日志文件
    local suite_log="$LOG_DIR/${test_name}_$TIMESTAMP.log"
    
    # 运行测试
    if uv run pytest "$test_path" $test_args \
        --tb=short \
        --disable-warnings \
        -v \
        --log-cli-level=INFO \
        --log-cli-format="%(asctime)s [%(levelname)s] %(message)s" \
        --log-file="$suite_log" 2>&1 | tee -a "$TEST_LOG"; then
        
        echo -e "${GREEN}✅ 测试套件通过: $test_name${NC}"
        log_message "测试套件通过: $test_name"
        return 0
    else
        echo -e "${RED}❌ 测试套件失败: $test_name${NC}"
        log_message "测试套件失败: $test_name"
        return 1
    fi
}

# 函数：运行性能测试
run_performance_tests() {
    echo -e "${YELLOW}🚄 运行性能基准测试${NC}"
    log_message "开始性能基准测试"
    
    local perf_log="$LOG_DIR/performance_results_$TIMESTAMP.json"
    
    # 运行性能测试并生成JSON报告
    if uv run pytest tests/strategy/test_performance_benchmarks.py \
        --tb=short \
        -v \
        --json-report \
        --json-report-file="$perf_log" \
        --disable-warnings 2>&1 | tee -a "$TEST_LOG"; then
        
        echo -e "${GREEN}✅ 性能测试完成${NC}"
        log_message "性能测试完成，结果保存到: $perf_log"
        
        # 如果有性能报告，显示关键指标
        if [[ -f "$perf_log" ]]; then
            echo -e "${BLUE}📊 性能测试摘要:${NC}"
            python3 -c "
import json
try:
    with open('$perf_log', 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    print(f'  总测试数: {summary.get(\"total\", 0)}')
    print(f'  通过数: {summary.get(\"passed\", 0)}')
    print(f'  失败数: {summary.get(\"failed\", 0)}')
    print(f'  耗时: {summary.get(\"duration\", 0):.2f}秒')
    
    # 如果有性能数据，显示关键指标
    if 'tests' in data:
        for test in data['tests']:
            if 'performance' in test.get('nodeid', ''):
                print(f'  {test[\"nodeid\"].split(\"::\")[-1]}: {test[\"outcome\"]}')
except Exception as e:
    print(f'  无法解析性能报告: {e}')
"
        fi
        return 0
    else
        echo -e "${RED}❌ 性能测试失败${NC}"
        log_message "性能测试失败"
        return 1
    fi
}

# 函数：生成测试报告
generate_test_report() {
    local report_file="$LOG_DIR/strategy_test_report_$TIMESTAMP.html"
    
    echo -e "${YELLOW}📑 生成综合测试报告${NC}"
    log_message "生成综合测试报告: $report_file"
    
    # 运行所有测试并生成HTML报告
    if uv run pytest tests/strategy/ \
        --html="$report_file" \
        --self-contained-html \
        --tb=short \
        --disable-warnings \
        -v 2>&1 | tee -a "$TEST_LOG"; then
        
        echo -e "${GREEN}✅ 测试报告已生成: $report_file${NC}"
        log_message "测试报告已生成: $report_file"
        
        # 如果在macOS上，尝试打开报告
        if [[ "$OSTYPE" == "darwin"* ]] && command -v open >/dev/null; then
            echo "正在打开测试报告..."
            open "$report_file"
        fi
        
        return 0
    else
        echo -e "${RED}❌ 测试报告生成失败${NC}"
        log_message "测试报告生成失败"
        return 1
    fi
}

# 函数：运行覆盖率测试
run_coverage_tests() {
    echo -e "${YELLOW}📈 运行测试覆盖率分析${NC}"
    log_message "开始测试覆盖率分析"
    
    local coverage_dir="$LOG_DIR/coverage_$TIMESTAMP"
    mkdir -p "$coverage_dir"
    
    # 安装pytest-cov插件（如果未安装）
    uv add pytest-cov --dev 2>/dev/null || true
    
    # 运行覆盖率测试
    if uv run pytest tests/strategy/ \
        --cov=src/strategy \
        --cov-report=html:"$coverage_dir/html" \
        --cov-report=xml:"$coverage_dir/coverage.xml" \
        --cov-report=term \
        --cov-branch \
        --tb=short \
        --disable-warnings 2>&1 | tee -a "$TEST_LOG"; then
        
        echo -e "${GREEN}✅ 覆盖率分析完成${NC}"
        log_message "覆盖率分析完成，报告保存到: $coverage_dir"
        
        # 显示覆盖率摘要
        echo -e "${BLUE}📊 覆盖率摘要:${NC}"
        if [[ -f "$coverage_dir/coverage.xml" ]]; then
            python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('$coverage_dir/coverage.xml')
    root = tree.getroot()
    
    line_rate = float(root.get('line-rate', 0))
    branch_rate = float(root.get('branch-rate', 0))
    
    print(f'  行覆盖率: {line_rate*100:.1f}%')
    print(f'  分支覆盖率: {branch_rate*100:.1f}%')
    
    if line_rate < 0.9:
        print('  ⚠️  行覆盖率低于90%')
    if branch_rate < 0.8:
        print('  ⚠️  分支覆盖率低于80%')
        
except Exception as e:
    print(f'  无法解析覆盖率报告: {e}')
"
        fi
        
        # 在macOS上打开覆盖率报告
        if [[ "$OSTYPE" == "darwin"* ]] && command -v open >/dev/null; then
            if [[ -f "$coverage_dir/html/index.html" ]]; then
                echo "正在打开覆盖率报告..."
                open "$coverage_dir/html/index.html"
            fi
        fi
        
        return 0
    else
        echo -e "${RED}❌ 覆盖率分析失败${NC}"
        log_message "覆盖率分析失败"
        return 1
    fi
}

# 主执行流程
main() {
    local exit_code=0
    local failed_tests=()
    
    log_message "开始双策略管理系统集成测试"
    
    # 检查Python环境
    echo -e "${BLUE}🔍 检查测试环境${NC}"
    if ! command -v uv >/dev/null; then
        echo -e "${RED}❌ 未找到uv命令，请先安装uv${NC}"
        exit 1
    fi
    
    # 安装测试依赖
    echo "安装测试依赖..."
    uv sync --dev
    
    # 1. 运行集成测试
    if ! run_test_suite "双策略集成测试" "tests/strategy/test_integration_dual_strategy.py" "--durations=10"; then
        failed_tests+=("dual_strategy_integration")
        exit_code=1
    fi
    
    # 2. 运行性能基准测试
    if ! run_performance_tests; then
        failed_tests+=("performance_benchmarks")
        exit_code=1
    fi
    
    # 3. 运行可靠性测试
    if ! run_test_suite "可靠性和恢复测试" "tests/strategy/test_reliability_and_recovery.py" "--durations=10"; then
        failed_tests+=("reliability_recovery")
        exit_code=1
    fi
    
    # 4. 运行覆盖率测试
    if ! run_coverage_tests; then
        failed_tests+=("coverage_analysis")
        exit_code=1
    fi
    
    # 5. 生成综合报告
    if ! generate_test_report; then
        failed_tests+=("test_report")
        exit_code=1
    fi
    
    # 测试总结
    echo "=============================================" | tee -a "$TEST_LOG"
    log_message "测试套件执行完成"
    
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}🎉 所有测试套件执行成功！${NC}"
        log_message "所有测试套件执行成功"
    else
        echo -e "${RED}❌ 部分测试套件执行失败${NC}"
        log_message "失败的测试套件: ${failed_tests[*]}"
        
        echo -e "${YELLOW}失败的测试套件:${NC}"
        for test in "${failed_tests[@]}"; do
            echo "  - $test"
        done
    fi
    
    echo -e "${BLUE}📁 测试日志和报告保存在: $LOG_DIR${NC}"
    echo "============================================="
    
    return $exit_code
}

# 处理命令行参数
case "${1:-all}" in
    "integration")
        run_test_suite "双策略集成测试" "tests/strategy/test_integration_dual_strategy.py" "--durations=10"
        ;;
    "performance") 
        run_performance_tests
        ;;
    "reliability")
        run_test_suite "可靠性和恢复测试" "tests/strategy/test_reliability_and_recovery.py" "--durations=10"
        ;;
    "coverage")
        run_coverage_tests
        ;;
    "report")
        generate_test_report
        ;;
    "all"|*)
        main
        exit $?
        ;;
esac