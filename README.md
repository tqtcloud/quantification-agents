# 加密货币量化交易系统

一个轻量级的个人使用的加密货币量化交易系统，支持高频和低频交易策略，具备多Agent架构设计。

## 特性

- 🚀 **多种交易模式**：支持实盘、模拟盘和回测
- 🤖 **多Agent架构**：基于LangGraph的智能决策系统
- 📊 **双频策略**：同时支持高频交易(HFT)和低频交易
- 🔄 **币安集成**：完整的Binance Futures API对接
- 💡 **轻量设计**：无需Kafka/Redis等重型中间件
- 📈 **实时监控**：Web界面实时查看交易状态

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Web监控界面 (FastAPI + React)              │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      Agent编排层 (LangGraph)                   │
├─────────────────────────────────────────────────────────────┤
│  市场分析Agent │ 风险管理Agent │ 策略执行Agent │ 监控报警Agent  │
│  技术指标Agent │ 情绪分析Agent │ 套利检测Agent │ 做市策略Agent  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                         核心服务层                             │
│     交易引擎 │ 数据处理 │ 回测引擎 │ 虚拟盘引擎                │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                       数据与通信层                             │
│   Binance API │ SQLite/DuckDB │ ZeroMQ │ 文件系统            │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.11+
- Poetry (包管理)
- TA-Lib (技术指标库)

### 安装

1. 克隆仓库
```bash
git clone <repository-url>
cd quantification-agents
```

2. 安装依赖
```bash
# 安装TA-Lib (macOS)
brew install ta-lib

# 安装TA-Lib (Ubuntu)
sudo apt-get install ta-lib

# 安装Python依赖
poetry install
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的API密钥
```

### 运行

#### 模拟盘交易
```bash
python main.py trade --mode paper
```

#### 回测
```bash
python main.py backtest \
  --strategy trend_following \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --initial-capital 10000
```

#### Web界面
```bash
python main.py web --reload
# 访问 http://localhost:8000
```

## 项目结构

```
quantification-agents/
├── src/
│   ├── agents/          # 智能Agent实现
│   │   ├── base.py      # Agent基类
│   │   ├── technical.py # 技术分析Agent
│   │   ├── risk.py      # 风险管理Agent
│   │   └── execution.py # 执行Agent
│   ├── core/            # 核心模块
│   │   ├── models.py    # 数据模型
│   │   ├── database.py  # 数据库模型
│   │   └── engine.py    # 交易引擎
│   ├── exchanges/       # 交易所接口
│   │   └── binance.py   # 币安API客户端
│   ├── strategies/      # 交易策略
│   │   ├── base.py      # 策略基类
│   │   ├── hft.py       # 高频策略
│   │   └── trend.py     # 趋势策略
│   ├── utils/           # 工具模块
│   │   └── logger.py    # 日志配置
│   └── web/             # Web界面
│       └── api.py       # FastAPI路由
├── tests/               # 测试文件
├── config/              # 配置文件
├── data/                # 数据存储
├── logs/                # 日志文件
├── main.py              # 程序入口
├── pyproject.toml       # 项目配置
└── README.md            # 项目文档
```

## 配置说明

### 交易配置
- `TRADING_MODE`: 交易模式 (paper/backtest/live)
- `DEFAULT_LEVERAGE`: 默认杠杆倍数
- `MAX_POSITION_SIZE`: 最大仓位大小

### 风险管理
- `MAX_DAILY_LOSS_PERCENT`: 每日最大亏损百分比
- `MAX_POSITION_PERCENT`: 单个仓位最大占比
- `STOP_LOSS_PERCENT`: 止损百分比

### Agent配置
每个Agent都可以通过配置文件自定义参数，例如：
```python
{
    "name": "TechnicalAnalysisAgent",
    "enabled": true,
    "priority": 10,
    "parameters": {
        "indicators": ["RSI", "MACD", "MA"],
        "timeframes": ["5m", "15m", "1h"]
    }
}
```

## 开发计划

- [x] Phase 1: 基础架构设计
- [ ] Phase 2: 核心功能开发
  - [ ] Binance API客户端
  - [ ] 数据存储层
  - [ ] 基础Agent框架
- [ ] Phase 3: 高级功能
  - [ ] 高频交易模块
  - [ ] 多Agent协同
  - [ ] 回测系统
- [ ] Phase 4: 优化与测试

## 注意事项

⚠️ **风险警告**：
- 加密货币交易具有高风险，可能导致资金损失
- 在使用实盘模式前，请充分测试策略
- 建议先使用模拟盘熟悉系统

## 技术栈

- **Python 3.11+**: 主要开发语言
- **FastAPI**: Web框架
- **LangGraph**: Agent编排
- **SQLite/DuckDB**: 数据存储
- **ZeroMQ**: 轻量级消息队列
- **TA-Lib**: 技术指标计算

## 参考资源

- [Binance API文档](https://binance-docs.github.io/apidocs/)
- [LangGraph文档](https://python.langchain.com/docs/langgraph)
- [ai-hedge-fund项目](https://github.com/virattt/ai-hedge-fund)

## License

MIT License

## 贡献

欢迎提交Issue和Pull Request！