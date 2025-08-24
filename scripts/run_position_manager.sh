#!/bin/bash

# 自动平仓管理系统启动脚本
# 用于启动和监控智能平仓系统

set -e

# 脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/position_manager.pid"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "$LOGS_DIR"

# 日志函数
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# 检查Python虚拟环境
check_venv() {
    if [ ! -d "$PROJECT_ROOT/.venv" ]; then
        error "Python虚拟环境不存在，请先创建: uv venv"
        exit 1
    fi
    
    if [ -z "$VIRTUAL_ENV" ]; then
        log "激活Python虚拟环境..."
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi
}

# 检查依赖
check_dependencies() {
    log "检查Python依赖..."
    
    if ! command -v uv &> /dev/null; then
        error "uv命令未找到，请安装: pip install uv"
        exit 1
    fi
    
    # 检查关键模块是否可导入
    python -c "import asyncio, datetime, logging" 2>/dev/null || {
        error "Python标准库模块导入失败"
        exit 1
    }
    
    python -c "import numpy" 2>/dev/null || {
        warn "numpy未安装，某些功能可能无法使用"
    }
    
    info "依赖检查完成"
}

# 检查进程状态
check_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0  # 进程存在
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1  # 进程不存在
}

# 启动服务
start_service() {
    log "启动智能平仓管理系统..."
    
    if check_process; then
        warn "服务已在运行 (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    check_venv
    check_dependencies
    
    # 设置日志文件
    LOG_FILE="$LOGS_DIR/position_manager_$(date +%Y%m%d_%H%M%S).log"
    
    # 启动Python程序
    cd "$PROJECT_ROOT"
    nohup python -m examples.position_manager_demo \
        --config-file="config/position_manager.json" \
        --log-level="INFO" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    # 等待启动
    sleep 2
    
    if check_process; then
        log "服务启动成功 (PID: $PID)"
        log "日志文件: $LOG_FILE"
        info "查看实时日志: tail -f $LOG_FILE"
    else
        error "服务启动失败，请检查日志: $LOG_FILE"
        exit 1
    fi
}

# 停止服务
stop_service() {
    log "停止智能平仓管理系统..."
    
    if ! check_process; then
        warn "服务未运行"
        return 0
    fi
    
    PID=$(cat "$PID_FILE")
    
    # 优雅停止
    kill -TERM "$PID" 2>/dev/null || true
    
    # 等待进程结束
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    # 强制终止
    if ps -p "$PID" > /dev/null 2>&1; then
        warn "进程未响应，强制终止..."
        kill -KILL "$PID" 2>/dev/null || true
        sleep 1
    fi
    
    rm -f "$PID_FILE"
    log "服务已停止"
}

# 重启服务
restart_service() {
    log "重启智能平仓管理系统..."
    stop_service
    sleep 2
    start_service
}

# 查看状态
status_service() {
    if check_process; then
        PID=$(cat "$PID_FILE")
        log "服务正在运行 (PID: $PID)"
        
        # 显示进程信息
        ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem 2>/dev/null || true
        
        # 显示最新日志
        LATEST_LOG=$(ls -t "$LOGS_DIR"/position_manager_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            info "最新日志文件: $LATEST_LOG"
            info "最近10行日志:"
            tail -10 "$LATEST_LOG" 2>/dev/null || true
        fi
    else
        warn "服务未运行"
        return 1
    fi
}

# 查看日志
view_logs() {
    LATEST_LOG=$(ls -t "$LOGS_DIR"/position_manager_*.log 2>/dev/null | head -1)
    
    if [ -z "$LATEST_LOG" ]; then
        warn "未找到日志文件"
        return 1
    fi
    
    info "查看日志文件: $LATEST_LOG"
    
    if [ "$1" = "follow" ] || [ "$1" = "-f" ]; then
        tail -f "$LATEST_LOG"
    else
        tail -50 "$LATEST_LOG"
    fi
}

# 运行测试
run_tests() {
    log "运行智能平仓系统测试..."
    
    check_venv
    cd "$PROJECT_ROOT"
    
    # 运行position相关测试
    python -m pytest tests/position/ -v --tb=short
    
    if [ $? -eq 0 ]; then
        log "测试通过"
    else
        error "测试失败"
        exit 1
    fi
}

# 清理日志
cleanup_logs() {
    log "清理旧日志文件..."
    
    # 保留最近7天的日志
    find "$LOGS_DIR" -name "position_manager_*.log" -mtime +7 -delete 2>/dev/null || true
    
    # 压缩1天前的日志
    find "$LOGS_DIR" -name "position_manager_*.log" -mtime +1 -exec gzip {} \; 2>/dev/null || true
    
    info "日志清理完成"
}

# 显示帮助信息
show_help() {
    echo "智能平仓管理系统控制脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start       启动服务"
    echo "  stop        停止服务"
    echo "  restart     重启服务"
    echo "  status      查看服务状态"
    echo "  logs        查看最近日志"
    echo "  logs -f     实时查看日志"
    echo "  test        运行测试"
    echo "  cleanup     清理旧日志"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start              # 启动服务"
    echo "  $0 status             # 查看状态"
    echo "  $0 logs -f            # 实时查看日志"
    echo "  $0 restart            # 重启服务"
}

# 主函数
main() {
    case "${1:-help}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            status_service
            ;;
        logs)
            view_logs "$2"
            ;;
        test)
            run_tests
            ;;
        cleanup)
            cleanup_logs
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 信号处理
trap 'error "脚本被中断"; exit 1' INT TERM

# 执行主函数
main "$@"