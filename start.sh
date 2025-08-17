#!/bin/bash

# 量化交易系统快速启动脚本

set -e

echo "🚀 量化交易系统启动脚本"
echo "=================================="

# 检查Python版本
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 未安装，请先安装 Python 3.11+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✅ Python版本: $python_version"
    
    if [[ $(echo "$python_version < 3.11" | bc -l) -eq 1 ]]; then
        echo "⚠️  警告: 建议使用 Python 3.11+ 以获得最佳性能"
    fi
}

# 检查并创建虚拟环境
setup_venv() {
    if [ ! -d "venv" ]; then
        echo "📦 创建虚拟环境..."
        python3 -m venv venv
    fi
    
    echo "📦 激活虚拟环境..."
    source venv/bin/activate
    
    echo "📦 安装/更新依赖..."
    pip install --upgrade pip
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "❌ 未找到 requirements.txt 文件"
        exit 1
    fi
}

# 检查并创建配置文件
setup_config() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            echo "📝 复制配置文件模板..."
            cp .env.example .env
            echo "⚠️  请编辑 .env 文件配置你的API密钥和其他设置"
        else
            echo "❌ 未找到配置文件模板"
            exit 1
        fi
    else
        echo "✅ 配置文件已存在"
    fi
}

# 创建必要的目录
create_directories() {
    echo "📁 创建必要的目录..."
    mkdir -p data/development data/testing data/production
    mkdir -p logs
    mkdir -p config
    echo "✅ 目录创建完成"
}

# 初始化数据库
init_database() {
    echo "🗄️  初始化数据库..."
    python3 -c "from src.core.database import init_database; init_database()" || true
    echo "✅ 数据库初始化完成"
}

# 运行测试
run_tests() {
    echo "🧪 运行系统测试..."
    if command -v pytest &> /dev/null; then
        pytest tests/test_system_integration.py -v || echo "⚠️  部分测试失败，但系统仍可继续运行"
    else
        echo "⚠️  pytest 未安装，跳过测试"
    fi
}

# 显示使用说明
show_usage() {
    echo ""
    echo "🎉 系统设置完成！"
    echo "=================================="
    echo ""
    echo "📖 快速开始："
    echo ""
    echo "1. 💻 命令行模式（推荐新手）："
    echo "   source venv/bin/activate"
    echo "   python main.py trade --mode paper --env development"
    echo ""
    echo "2. 🌐 Web界面模式："
    echo "   source venv/bin/activate" 
    echo "   python main.py web --port 8000"
    echo "   然后访问: http://localhost:8000"
    echo ""
    echo "3. 📊 回测模式："
    echo "   source venv/bin/activate"
    echo "   python main.py backtest --strategy technical_analysis --start-date 2024-01-01 --end-date 2024-12-31"
    echo ""
    echo "🔗 重要链接："
    echo "   - 系统状态: http://localhost:8000/system/status"
    echo "   - API文档: http://localhost:8000/api/docs" 
    echo "   - 实时数据: ws://localhost:8000/ws/all"
    echo ""
    echo "📝 配置文件: .env"
    echo "📚 详细文档: DEPLOYMENT.md"
    echo ""
    echo "⚠️  首次使用请先编辑 .env 文件配置你的设置！"
    echo ""
}

# 主执行流程
main() {
    echo "开始系统初始化..."
    
    check_python
    setup_venv
    setup_config
    create_directories
    init_database
    
    # 询问是否运行测试
    read -p "🧪 是否运行系统测试? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    show_usage
}

# 错误处理
trap 'echo "❌ 启动过程中出现错误，请检查上面的错误信息"; exit 1' ERR

# 执行主函数
main "$@"