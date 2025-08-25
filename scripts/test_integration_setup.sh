#!/bin/bash

# 集成测试环境验证脚本
# 用于验证集成测试环境是否正确设置

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

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}      集成测试环境验证脚本               ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "验证时间: $(date)"
    echo -e "项目路径: $PROJECT_ROOT"
    echo -e "${BLUE}============================================${NC}\n"
}

print_section() {
    echo -e "${YELLOW}>>> $1${NC}"
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

# 检查Python环境
check_python_environment() {
    print_section "检查Python环境"
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python版本: $PYTHON_VERSION"
        
        # 检查版本是否满足要求
        if python3 -c "import sys; assert sys.version_info >= (3, 8)"; then
            print_success "Python版本满足要求 (>= 3.8)"
        else
            print_error "Python版本不满足要求，需要 >= 3.8"
            return 1
        fi
    else
        print_error "未找到Python3"
        return 1
    fi
    
    # 检查虚拟环境
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "已激活虚拟环境: $VIRTUAL_ENV"
    elif [[ -d ".venv" ]]; then
        print_warning "检测到.venv目录但未激活，尝试激活..."
        source .venv/bin/activate
        if [[ -n "$VIRTUAL_ENV" ]]; then
            print_success "成功激活虚拟环境: $VIRTUAL_ENV"
        else
            print_error "无法激活虚拟环境"
            return 1
        fi
    else
        print_warning "未检测到虚拟环境"
    fi
}

# 检查项目结构
check_project_structure() {
    print_section "检查项目结构"
    
    # 必需的目录
    required_dirs=(
        "src"
        "src/api" 
        "src/websocket"
        "tests"
        "tests/integration"
        "scripts"
        "config"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            print_success "目录存在: $dir"
        else
            print_error "目录缺失: $dir"
            return 1
        fi
    done
    
    # 检查关键文件
    key_files=(
        "src/api/trading_api.py"
        "src/websocket/websocket_manager.py"
        "tests/integration/conftest.py"
        "tests/integration/pytest.ini"
        "scripts/run_integration_tests.sh"
    )
    
    for file in "${key_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_success "文件存在: $file"
        else
            print_error "文件缺失: $file"
            return 1
        fi
    done
}

# 检查依赖包
check_dependencies() {
    print_section "检查Python依赖包"
    
    # 核心依赖
    core_deps=(
        "pytest"
        "pytest_asyncio" 
        "httpx"
        "websockets"
        "fastapi"
        "sqlalchemy"
        "pydantic"
    )
    
    missing_deps=()
    
    for dep in "${core_deps[@]}"; do
        if python3 -c "import $dep" 2>/dev/null; then
            # 获取版本信息
            version=$(python3 -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null)
            print_success "$dep (版本: $version)"
        else
            print_error "缺失依赖: $dep"
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "发现缺失的依赖包，尝试安装..."
        
        # 检查是否存在requirements文件
        if [[ -f "tests/integration/requirements.txt" ]]; then
            pip install -r tests/integration/requirements.txt
            print_success "依赖安装完成"
        else
            print_error "未找到requirements.txt文件"
            return 1
        fi
    fi
}

# 检查端口可用性
check_port_availability() {
    print_section "检查端口可用性"
    
    test_ports=(8000 8765 8800 8801 8802)
    
    for port in "${test_ports[@]}"; do
        if lsof -i:$port >/dev/null 2>&1; then
            print_warning "端口 $port 已被占用"
            # 显示占用进程
            occupying_process=$(lsof -ti:$port | xargs ps -p | tail -n +2)
            echo "占用进程: $occupying_process"
        else
            print_success "端口 $port 可用"
        fi
    done
}

# 检查系统资源
check_system_resources() {
    print_section "检查系统资源"
    
    # 检查内存
    if command -v free >/dev/null 2>&1; then
        # Linux
        available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
        print_success "可用内存: ${available_memory}GB"
        
        if (( $(echo "$available_memory > 1.0" | bc -l) )); then
            print_success "内存充足"
        else
            print_warning "可用内存不足，建议至少2GB"
        fi
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        page_size=4096
        available_memory_mb=$((free_pages * page_size / 1024 / 1024))
        available_memory_gb=$(echo "scale=1; $available_memory_mb / 1024" | bc)
        
        print_success "可用内存: ${available_memory_gb}GB"
        
        if (( $(echo "$available_memory_gb > 1.0" | bc -l) )); then
            print_success "内存充足"
        else
            print_warning "可用内存不足，建议至少2GB"
        fi
    else
        print_warning "无法检查内存使用情况"
    fi
    
    # 检查磁盘空间
    available_space=$(df -h . | awk 'NR==2 {print $4}')
    print_success "可用磁盘空间: $available_space"
}

# 运行简单测试
run_simple_test() {
    print_section "运行简单测试验证"
    
    # 创建临时测试文件
    cat > /tmp/simple_test.py << 'EOF'
import asyncio
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """简单的异步测试"""
    await asyncio.sleep(0.01)
    assert True

def test_simple_assertion():
    """简单断言测试"""
    assert 1 + 1 == 2
EOF
    
    # 运行测试
    if pytest /tmp/simple_test.py -v --tb=short; then
        print_success "简单测试验证通过"
        rm /tmp/simple_test.py
    else
        print_error "简单测试验证失败"
        rm /tmp/simple_test.py
        return 1
    fi
}

# 测试导入
test_imports() {
    print_section "测试关键模块导入"
    
    # 添加src到Python路径
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
    
    # 测试导入的模块
    test_modules=(
        "src.core.database"
        "src.api.trading_api"
        "src.websocket.websocket_manager"
        "src.api.auth_manager"
        "src.api.rate_limiter"
    )
    
    for module in "${test_modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            print_success "成功导入: $module"
        else
            print_warning "导入失败: $module"
            # 显示详细错误信息
            python3 -c "import $module" 2>&1 | head -5
        fi
    done
}

# 检查配置文件
check_config_files() {
    print_section "检查配置文件"
    
    config_files=(
        "config/development.yaml"
        "config/testing.yaml"
        "tests/integration/pytest.ini"
        "tests/integration/conftest.py"
    )
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            print_success "配置文件存在: $config_file"
            
            # 检查文件是否可读
            if [[ -r "$config_file" ]]; then
                print_success "配置文件可读: $config_file"
            else
                print_warning "配置文件不可读: $config_file"
            fi
        else
            print_warning "配置文件缺失: $config_file"
        fi
    done
}

# 创建测试目录
create_test_directories() {
    print_section "创建测试相关目录"
    
    test_dirs=(
        "logs"
        "test_reports"
        "test_reports/htmlcov"
    )
    
    for dir in "${test_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_success "创建目录: $dir"
        else
            print_success "目录已存在: $dir"
        fi
    done
}

# 生成环境报告
generate_environment_report() {
    print_section "生成环境报告"
    
    report_file="test_reports/environment_check_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "集成测试环境检查报告"
        echo "生成时间: $(date)"
        echo "项目路径: $PROJECT_ROOT"
        echo ""
        
        echo "=== 系统信息 ==="
        uname -a
        echo ""
        
        echo "=== Python信息 ==="
        python3 --version
        echo "虚拟环境: $VIRTUAL_ENV"
        echo ""
        
        echo "=== 已安装的包 ==="
        pip list 2>/dev/null | head -20
        echo ""
        
        echo "=== 环境变量 ==="
        env | grep -E "(PYTHON|PATH|VIRTUAL)" | sort
        echo ""
        
        echo "=== 项目结构 ==="
        ls -la
        echo ""
        
        echo "=== 集成测试文件 ==="
        find tests/integration -name "*.py" -type f | head -10
        
    } > "$report_file"
    
    print_success "环境报告已生成: $report_file"
}

# 主函数
main() {
    print_header
    
    local exit_code=0
    
    # 依次运行检查
    checks=(
        "check_python_environment"
        "check_project_structure"
        "create_test_directories"
        "check_dependencies"
        "check_config_files"
        "test_imports"
        "check_port_availability"
        "check_system_resources"
        "run_simple_test"
        "generate_environment_report"
    )
    
    for check in "${checks[@]}"; do
        if ! $check; then
            exit_code=1
            print_error "检查失败: $check"
        fi
        echo ""
    done
    
    # 总结
    print_section "环境验证总结"
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "所有检查都通过了！"
        print_success "集成测试环境配置正确"
        echo ""
        echo -e "${GREEN}可以运行以下命令开始集成测试:${NC}"
        echo -e "${BLUE}./scripts/run_integration_tests.sh${NC}"
    else
        print_error "部分检查失败"
        print_warning "请根据上述错误信息修复问题后重试"
        echo ""
        echo -e "${YELLOW}常见解决方案:${NC}"
        echo "1. 安装缺失的依赖: pip install -r tests/integration/requirements.txt"
        echo "2. 激活虚拟环境: source .venv/bin/activate"
        echo "3. 检查Python版本: python3 --version (需要 >= 3.8)"
        echo "4. 检查端口占用: lsof -i:8000"
    fi
    
    return $exit_code
}

# 执行主函数
main "$@"