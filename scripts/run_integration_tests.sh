#!/bin/bash

# REST API和WebSocket集成测试执行脚本
# 用法: ./run_integration_tests.sh [test_suite] [options]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 测试报告目录
REPORT_DIR="$PROJECT_ROOT/test_reports"
mkdir -p "$REPORT_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 日志文件
LOG_FILE="$LOG_DIR/integration_tests_$TIMESTAMP.log"
COVERAGE_FILE="$REPORT_DIR/coverage_$TIMESTAMP.xml"
JUNIT_FILE="$REPORT_DIR/junit_$TIMESTAMP.xml"
HTML_REPORT="$REPORT_DIR/integration_report_$TIMESTAMP.html"

# 函数定义
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}    REST API & WebSocket 集成测试套件    ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "测试时间: $(date)"
    echo -e "项目路径: $PROJECT_ROOT"
    echo -e "日志文件: $LOG_FILE"
    echo -e "${BLUE}============================================${NC}"
}

print_section() {
    echo -e "\n${YELLOW}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 检查依赖
check_dependencies() {
    print_section "检查测试依赖"
    
    # 检查Python虚拟环境
    if [[ -z "$VIRTUAL_ENV" && ! -d ".venv" ]]; then
        print_error "未找到Python虚拟环境"
        echo "请运行: python -m venv .venv && source .venv/bin/activate"
        exit 1
    fi
    
    # 激活虚拟环境
    if [[ -d ".venv" && -z "$VIRTUAL_ENV" ]]; then
        source .venv/bin/activate
        print_success "激活虚拟环境: .venv"
    fi
    
    # 检查pytest
    if ! command -v pytest &> /dev/null; then
        print_error "pytest未安装"
        echo "请运行: pip install pytest pytest-asyncio pytest-cov"
        exit 1
    fi
    
    # 检查必需的包
    required_packages=("pytest-asyncio" "pytest-cov" "httpx" "websockets" "psutil")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            print_warning "$package 未安装，尝试安装..."
            pip install "$package" || print_error "安装 $package 失败"
        fi
    done
    
    print_success "依赖检查完成"
}

# 环境设置
setup_environment() {
    print_section "设置测试环境"
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    export TESTING=1
    export LOG_LEVEL=INFO
    
    # 清理旧的测试数据
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    
    # 确保测试目录存在
    mkdir -p "$PROJECT_ROOT/tests/integration"
    
    print_success "测试环境设置完成"
}

# 运行特定测试套件
run_test_suite() {
    local test_name="$1"
    local test_file="$2"
    local description="$3"
    
    print_section "运行 $test_name"
    echo "描述: $description"
    echo "文件: $test_file"
    
    if [[ ! -f "$test_file" ]]; then
        print_error "测试文件不存在: $test_file"
        return 1
    fi
    
    local suite_log="$LOG_DIR/${test_name}_$TIMESTAMP.log"
    local suite_junit="$REPORT_DIR/${test_name}_$TIMESTAMP.xml"
    
    # 运行测试
    echo "开始时间: $(date)" | tee "$suite_log"
    
    pytest "$test_file" \
        -v \
        --tb=short \
        --asyncio-mode=auto \
        --durations=10 \
        --junitxml="$suite_junit" \
        --cov=src \
        --cov-report=term-missing \
        --cov-append \
        -s \
        2>&1 | tee -a "$suite_log" "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    echo "结束时间: $(date)" | tee -a "$suite_log"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "$test_name 测试通过"
        return 0
    else
        print_error "$test_name 测试失败 (退出码: $exit_code)"
        return $exit_code
    fi
}

# 运行性能基准测试
run_performance_tests() {
    print_section "运行性能基准测试"
    
    local perf_log="$LOG_DIR/performance_$TIMESTAMP.log"
    local perf_junit="$REPORT_DIR/performance_$TIMESTAMP.xml"
    
    # 设置性能测试专用参数
    export PERFORMANCE_TEST=1
    export MAX_WORKERS=4
    
    pytest tests/integration/test_performance_benchmarks.py \
        -v \
        --tb=short \
        --asyncio-mode=auto \
        --durations=0 \
        --junitxml="$perf_junit" \
        --benchmark-only \
        --benchmark-sort=mean \
        --benchmark-json="$REPORT_DIR/benchmark_$TIMESTAMP.json" \
        -s \
        2>&1 | tee "$perf_log" "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "性能测试完成"
    else
        print_warning "性能测试出现问题 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 运行安全测试
run_security_tests() {
    print_section "运行安全和负载测试"
    
    local security_log="$LOG_DIR/security_$TIMESTAMP.log"
    local security_junit="$REPORT_DIR/security_$TIMESTAMP.xml"
    
    # 安全测试可能需要更长的超时时间
    export TEST_TIMEOUT=300
    
    pytest tests/integration/test_security_and_load.py \
        -v \
        --tb=short \
        --asyncio-mode=auto \
        --timeout=300 \
        --durations=10 \
        --junitxml="$security_junit" \
        -s \
        2>&1 | tee "$security_log" "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "安全测试完成"
    else
        print_warning "安全测试出现问题 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 生成覆盖率报告
generate_coverage_report() {
    print_section "生成测试覆盖率报告"
    
    # 生成XML报告
    coverage xml -o "$COVERAGE_FILE" 2>/dev/null || print_warning "无法生成XML覆盖率报告"
    
    # 生成HTML报告
    local html_cov_dir="$REPORT_DIR/htmlcov_$TIMESTAMP"
    coverage html -d "$html_cov_dir" 2>/dev/null || print_warning "无法生成HTML覆盖率报告"
    
    # 显示覆盖率总结
    echo -e "\n${BLUE}=== 测试覆盖率总结 ===${NC}"
    coverage report --show-missing 2>/dev/null || print_warning "无法显示覆盖率报告"
    
    if [[ -f "$COVERAGE_FILE" ]]; then
        print_success "覆盖率报告已生成: $COVERAGE_FILE"
    fi
    
    if [[ -d "$html_cov_dir" ]]; then
        print_success "HTML覆盖率报告: $html_cov_dir/index.html"
    fi
}

# 生成集成测试报告
generate_integration_report() {
    print_section "生成集成测试报告"
    
    local report_file="$REPORT_DIR/integration_summary_$TIMESTAMP.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>REST API & WebSocket 集成测试报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        .info { color: #17a2b8; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metrics { display: flex; justify-content: space-between; }
        .metric-box { background: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>REST API & WebSocket 集成测试报告</h1>
        <p><strong>生成时间:</strong> $(date)</p>
        <p><strong>项目路径:</strong> $PROJECT_ROOT</p>
    </div>
    
    <div class="section">
        <h2>测试概览</h2>
        <div class="metrics">
            <div class="metric-box">
                <h3>API集成测试</h3>
                <p class="info">认证、权限、速率限制</p>
            </div>
            <div class="metric-box">
                <h3>WebSocket测试</h3>
                <p class="info">连接管理、消息推送</p>
            </div>
            <div class="metric-box">
                <h3>端到端测试</h3>
                <p class="info">完整用户场景</p>
            </div>
            <div class="metric-box">
                <h3>性能测试</h3>
                <p class="info">响应时间、吞吐量</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>测试文件列表</h2>
        <table>
            <tr>
                <th>测试套件</th>
                <th>描述</th>
                <th>状态</th>
            </tr>
EOF

    # 检查各个测试文件的状态
    local test_files=(
        "tests/integration/test_api_integration.py:API集成测试"
        "tests/integration/test_websocket_integration.py:WebSocket集成测试"
        "tests/integration/test_end_to_end_integration.py:端到端集成测试"
        "tests/integration/test_performance_benchmarks.py:性能基准测试"
        "tests/integration/test_security_and_load.py:安全和负载测试"
    )
    
    for test_info in "${test_files[@]}"; do
        IFS=':' read -r test_file test_desc <<< "$test_info"
        if [[ -f "$test_file" ]]; then
            echo "            <tr><td>$(basename "$test_file")</td><td>$test_desc</td><td class=\"success\">✓ 存在</td></tr>" >> "$report_file"
        else
            echo "            <tr><td>$(basename "$test_file")</td><td>$test_desc</td><td class=\"error\">✗ 缺失</td></tr>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>日志文件</h2>
        <ul>
            <li><strong>主日志:</strong> $LOG_FILE</li>
            <li><strong>报告目录:</strong> $REPORT_DIR</li>
EOF

    # 列出生成的日志文件
    for log_file in "$LOG_DIR"/*_"$TIMESTAMP".log; do
        if [[ -f "$log_file" ]]; then
            echo "            <li>$(basename "$log_file")</li>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF
        </ul>
    </div>
    
    <div class="section">
        <h2>测试指标</h2>
        <p>详细的性能指标和覆盖率数据请查看相应的报告文件。</p>
        <ul>
            <li><strong>目标API响应时间:</strong> &lt; 100ms</li>
            <li><strong>目标WebSocket延迟:</strong> &lt; 50ms</li>
            <li><strong>目标并发连接:</strong> 1000+</li>
            <li><strong>目标测试覆盖率:</strong> &gt; 85%</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>建议和下一步</h2>
        <ol>
            <li>检查所有测试是否通过</li>
            <li>分析性能指标是否满足要求</li>
            <li>审查安全测试结果</li>
            <li>优化发现的性能瓶颈</li>
            <li>定期运行集成测试确保代码质量</li>
        </ol>
    </div>
    
    <div class="section">
        <h2>联系信息</h2>
        <p>如有测试相关问题，请查看日志文件或联系开发团队。</p>
    </div>
    
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>© $(date +%Y) 量化交易系统 - 集成测试报告</p>
    </footer>
</body>
</html>
EOF
    
    print_success "集成测试报告已生成: $report_file"
    
    # 如果是macOS，尝试打开报告
    if command -v open &> /dev/null; then
        open "$report_file" 2>/dev/null || true
    fi
}

# 清理函数
cleanup() {
    print_section "清理测试环境"
    
    # 停止可能的后台进程
    pkill -f "pytest.*integration" 2>/dev/null || true
    
    # 清理临时文件
    find "$PROJECT_ROOT" -name ".coverage*" -delete 2>/dev/null || true
    
    print_success "清理完成"
}

# 主执行函数
main() {
    local test_suite="$1"
    local run_all=false
    local exit_code=0
    
    # 解析参数
    case "$test_suite" in
        "api")
            test_suite="api"
            ;;
        "websocket")
            test_suite="websocket"
            ;;
        "e2e")
            test_suite="e2e"
            ;;
        "performance")
            test_suite="performance"
            ;;
        "security")
            test_suite="security"
            ;;
        "all"|"")
            run_all=true
            ;;
        "-h"|"--help")
            echo "用法: $0 [test_suite] [options]"
            echo ""
            echo "测试套件:"
            echo "  api         - API集成测试"
            echo "  websocket   - WebSocket集成测试"
            echo "  e2e         - 端到端集成测试"
            echo "  performance - 性能基准测试"
            echo "  security    - 安全和负载测试"
            echo "  all         - 运行所有测试 (默认)"
            echo ""
            echo "选项:"
            echo "  -h, --help  - 显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 api      - 仅运行API集成测试"
            echo "  $0 all      - 运行所有集成测试"
            exit 0
            ;;
        *)
            print_error "未知的测试套件: $test_suite"
            echo "使用 $0 --help 查看可用选项"
            exit 1
            ;;
    esac
    
    # 设置陷阱函数
    trap cleanup EXIT
    
    # 开始测试
    print_header
    
    # 检查依赖和设置环境
    check_dependencies
    setup_environment
    
    # 运行测试
    if [[ "$run_all" == true ]]; then
        print_section "运行完整集成测试套件"
        
        # 运行所有测试套件
        local test_suites=(
            "API集成测试:tests/integration/test_api_integration.py:测试认证、权限、速率限制和参数验证"
            "WebSocket集成测试:tests/integration/test_websocket_integration.py:测试连接管理、订阅和消息推送"
            "端到端集成测试:tests/integration/test_end_to_end_integration.py:测试API和WebSocket协同工作"
            "性能基准测试:tests/integration/test_performance_benchmarks.py:测试API响应时间和WebSocket延迟"
            "安全和负载测试:tests/integration/test_security_and_load.py:测试并发连接和攻击防护"
        )
        
        for suite_info in "${test_suites[@]}"; do
            IFS=':' read -r suite_name suite_file suite_desc <<< "$suite_info"
            
            if ! run_test_suite "$suite_name" "$suite_file" "$suite_desc"; then
                exit_code=1
            fi
            
            # 短暂休息
            sleep 2
        done
        
    else
        # 运行特定测试套件
        case "$test_suite" in
            "api")
                run_test_suite "API集成测试" "tests/integration/test_api_integration.py" "测试认证、权限、速率限制和参数验证"
                exit_code=$?
                ;;
            "websocket")
                run_test_suite "WebSocket集成测试" "tests/integration/test_websocket_integration.py" "测试连接管理、订阅和消息推送"
                exit_code=$?
                ;;
            "e2e")
                run_test_suite "端到端集成测试" "tests/integration/test_end_to_end_integration.py" "测试API和WebSocket协同工作"
                exit_code=$?
                ;;
            "performance")
                run_performance_tests
                exit_code=$?
                ;;
            "security")
                run_security_tests
                exit_code=$?
                ;;
        esac
    fi
    
    # 生成报告
    if [[ "$run_all" == true ]] || [[ "$test_suite" == "performance" ]]; then
        generate_coverage_report
    fi
    
    generate_integration_report
    
    # 总结
    print_section "测试执行总结"
    echo -e "测试时间: $(date)"
    echo -e "日志文件: $LOG_FILE"
    echo -e "报告目录: $REPORT_DIR"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "所有测试执行完成"
    else
        print_warning "部分测试未通过，请查看日志文件获取详细信息"
    fi
    
    return $exit_code
}

# 执行主函数
main "$@"