# 量化交易系统部署和运行指南

## 🚀 快速开始

### 1. 环境准备

#### Python环境
```bash
# 确保Python 3.11+
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

#### 安装依赖
```bash
# 安装项目依赖
pip install -r requirements.txt

# 或者使用poetry (推荐)
pip install poetry
poetry install
```

### 2. 配置设置

#### 创建环境配置文件
```bash
# 复制环境配置模板
cp .env.example .env

# 编辑配置文件
vim .env
```

#### 必要的API配置
在 `.env` 文件中配置：
```env
# Binance API配置 (可选，用于实盘交易)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true

# 交易模式
TRADING_MODE=paper  # paper, backtest, live

# 日志级别
LOG_LEVEL=INFO

# 数据库配置
DATABASE_URL=sqlite:///./data/trading.db

# Web界面配置
WEB_HOST=0.0.0.0
WEB_PORT=8000
```

### 3. 运行方式

#### 方式一：命令行模式 (推荐)
```bash
# 启动模拟交易模式 (默认)
python main.py trade --mode paper --env development

# 启动回测模式
python main.py trade --mode backtest --env testing

# 启动实盘交易 (谨慎使用)
python main.py trade --mode live --env production
```

#### 方式二：Web界面模式
```bash
# 启动Web监控界面
python main.py web --host 0.0.0.0 --port 8000

# 启用自动重载 (开发模式)
python main.py web --reload
```

#### 方式三：回测模式
```bash
# 运行策略回测
python main.py backtest \
  --strategy "technical_analysis" \
  --start-date "2024-01-01" \
  --end-date "2024-12-31" \
  --initial-capital 10000
```

## 📊 监控和管理

### Web监控界面
访问以下地址：
- **主界面**: http://localhost:8000
- **API文档**: http://localhost:8000/api/docs
- **系统状态**: http://localhost:8000/system/status
- **系统指标**: http://localhost:8000/system/metrics
- **告警信息**: http://localhost:8000/system/alerts

### 实时数据流
WebSocket连接端点：
- **市场数据**: ws://localhost:8000/ws/market
- **订单更新**: ws://localhost:8000/ws/orders
- **系统事件**: ws://localhost:8000/ws/system
- **性能指标**: ws://localhost:8000/ws/performance
- **全部数据**: ws://localhost:8000/ws/all

## 🛠️ 开发和调试

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试模块
pytest tests/test_system_integration.py

# 运行Web API测试
pytest tests/test_web_api.py -v

# 生成测试覆盖率报告
pytest --cov=src --cov-report=html
```

### 日志查看
```bash
# 查看实时日志
tail -f logs/development.log

# 查看错误日志
grep ERROR logs/development.log

# 查看交易日志
grep "trading" logs/development.log
```

### 数据库管理
```bash
# 查看数据库
sqlite3 data/development/trading.db

# 查看市场数据
python -c "from src.core.duckdb_manager import DuckDBManager; dm = DuckDBManager(); print(dm.query('SELECT * FROM market_data LIMIT 10'))"
```

## 🔧 配置管理

### 环境切换
系统支持三种环境：
- **development**: 开发环境，调试模式，详细日志
- **testing**: 测试环境，内存数据库，快速测试
- **production**: 生产环境，性能优化，错误告警

```bash
# 切换到测试环境
python main.py trade --env testing

# 切换到生产环境
python main.py trade --env production
```

### 配置热更新
系统支持配置文件热更新，修改 `config/` 目录下的YAML文件会自动生效：

```bash
# 修改开发环境配置
vim config/development.yaml

# 修改生产环境配置
vim config/production.yaml
```

## 📈 性能监控

### 系统指标
系统会自动收集以下指标：
- CPU使用率
- 内存使用率
- 磁盘使用率
- 网络IO
- 交易指标
- API请求统计

### 告警设置
系统内置以下告警规则：
- CPU使用率 > 90%
- 内存使用率 > 85%
- 磁盘使用率 > 90%
- 错误率 > 5个/分钟
- 交易系统状态异常

## 🚨 故障排除

### 常见问题

#### 1. 启动失败
```bash
# 检查Python版本
python --version  # 需要3.11+

# 检查依赖安装
pip list | grep fastapi

# 查看详细错误
python main.py trade --mode paper --env development
```

#### 2. 数据库连接失败
```bash
# 检查数据目录权限
ls -la data/

# 手动创建数据库
python -c "from src.core.database import init_database; init_database()"
```

#### 3. API无法访问
```bash
# 检查端口占用
netstat -tulpn | grep 8000

# 检查防火墙设置
sudo ufw status
```

#### 4. WebSocket连接失败
```bash
# 检查WebSocket路由
curl -I http://localhost:8000/ws/market

# 查看WebSocket日志
grep "websocket" logs/development.log
```

### 日志级别
调整日志级别获取更多信息：
```env
# .env文件中设置
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 🔒 安全注意事项

### 实盘交易警告
⚠️ **使用实盘模式前请务必注意**：
1. 确保API密钥安全存储
2. 设置合理的仓位和风险限制
3. 在测试环境充分验证策略
4. 建议先小资金测试

### API密钥管理
```bash
# 使用环境变量 (推荐)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# 或使用.env文件，确保不提交到版本控制
echo ".env" >> .gitignore
```

### 生产环境配置
```yaml
# config/production.yaml
trading:
  mode: live
  max_position_size: 1000.0
  max_daily_loss_percent: 2.0

logging:
  level: WARNING
  file_enabled: true

web:
  cors_enabled: false
  debug: false
```

## 📱 移动端监控

### 通过浏览器访问
移动设备可以直接访问Web界面：
- http://your-server-ip:8000

### 告警通知
可以配置邮件或Webhook告警：
```python
# 在代码中添加邮件告警
from src.monitoring.alert_manager import alert_manager, EmailAlertChannel

email_channel = EmailAlertChannel(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-app-password",
    from_email="your-email@gmail.com",
    to_emails=["alert@yourcompany.com"]
)
alert_manager.add_channel(email_channel)
```

## 🔄 系统维护

### 定期维护任务
```bash
# 清理旧日志 (保留30天)
find logs/ -name "*.log" -mtime +30 -delete

# 备份数据库
cp data/production/trading.db backups/trading_$(date +%Y%m%d).db

# 导出系统指标
curl http://localhost:8000/system/metrics > metrics_$(date +%Y%m%d).json
```

### 系统升级
```bash
# 停止系统
pkill -f "python main.py"

# 更新代码
git pull origin main

# 安装新依赖
pip install -r requirements.txt

# 重启系统
python main.py trade --mode paper --env production
```

祝你交易顺利！ 🚀📈