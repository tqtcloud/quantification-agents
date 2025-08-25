# 量化交易系统API文档

## 概览

量化交易系统提供完整的REST API接口，支持策略控制、系统监控、信号查询等功能。API采用RESTful设计，支持JWT认证、速率限制、参数验证等安全特性。

## 基础信息

- **基础URL**: `http://localhost:8000`
- **API版本**: `v1.0.0`
- **认证方式**: JWT Bearer Token / API Key
- **数据格式**: JSON
- **字符编码**: UTF-8

## 认证

### JWT认证流程

1. **登录获取Token**
   ```http
   POST /auth/login
   Content-Type: application/json
   
   {
     "username": "admin",
     "password": "admin123"
   }
   ```

2. **使用Token访问API**
   ```http
   GET /strategies
   Authorization: Bearer <access_token>
   ```

3. **刷新Token**
   ```http
   POST /auth/refresh
   Content-Type: application/json
   
   {
     "refresh_token": "<refresh_token>"
   }
   ```

### API Key认证

```http
GET /strategies
X-API-Key: <api_key>
```

## API端点

### 认证相关

#### POST /auth/login
用户登录获取访问令牌。

**请求体:**
```json
{
  "username": "string",
  "password": "string",
  "remember_me": false
}
```

**响应:**
```json
{
  "status": "success",
  "message": "Authentication successful",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800,
    "expires_at": "2024-01-01T12:30:00Z",
    "permissions": ["strategy:start", "strategy:stop", "strategy:view"]
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### POST /auth/refresh
刷新访问令牌。

#### POST /auth/logout
用户登出。

#### GET /auth/me
获取当前用户信息。

#### POST /auth/api-keys
创建API密钥。

### 策略控制

#### POST /strategies/{strategy_id}/start
启动指定策略。

**路径参数:**
- `strategy_id` (string): 策略ID

**请求体:**
```json
{
  "strategy_id": "my_hft_strategy",
  "strategy_type": "hft",
  "config": {
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "max_position": 0.1
  },
  "force": false,
  "dry_run": false
}
```

**响应:**
```json
{
  "status": "success",
  "message": "Strategy started successfully",
  "data": {
    "strategy_id": "my_hft_strategy",
    "action": "start",
    "success": true,
    "message": "Strategy started successfully",
    "new_status": "running",
    "execution_time_ms": 150.5
  }
}
```

#### POST /strategies/{strategy_id}/stop
停止指定策略。

**请求体:**
```json
{
  "strategy_id": "my_hft_strategy",
  "force": false,
  "save_state": true,
  "reason": "Manual stop"
}
```

#### POST /strategies/{strategy_id}/restart
重启指定策略。

#### GET /strategies/{strategy_id}/status
获取策略状态。

**响应:**
```json
{
  "status": "success",
  "message": "Strategy status retrieved successfully",
  "data": {
    "strategy_id": "my_hft_strategy",
    "name": "My HFT Strategy",
    "strategy_type": "hft",
    "status": "running",
    "description": "High frequency trading strategy for BTCUSDT",
    "config": {
      "symbol": "BTCUSDT",
      "timeframe": "1m"
    },
    "metrics": {
      "total_signals": 1250,
      "successful_trades": 847,
      "failed_trades": 12,
      "total_profit_loss": 125.67,
      "win_rate": 0.758,
      "average_return": 0.0015
    },
    "created_at": "2024-01-01T10:00:00Z",
    "updated_at": "2024-01-01T12:00:00Z",
    "started_at": "2024-01-01T10:30:00Z",
    "health_status": "healthy",
    "resource_usage": {
      "cpu_percent": 5.2,
      "memory_usage_mb": 128.5
    }
  }
}
```

#### GET /strategies
获取策略列表（支持分页和过滤）。

**查询参数:**
- `page` (int): 页码，默认1
- `page_size` (int): 每页数量，默认20，最大100
- `status_filter` (string): 状态过滤
- `strategy_type` (string): 策略类型过滤

**响应:**
```json
{
  "status": "success",
  "message": "Strategies retrieved successfully",
  "data": {
    "items": [
      {
        "strategy_id": "strategy_1",
        "name": "Strategy 1",
        "strategy_type": "hft",
        "status": "running"
      }
    ],
    "total": 25,
    "page": 1,
    "page_size": 20,
    "total_pages": 2,
    "has_next": true,
    "has_prev": false
  }
}
```

#### PUT /strategies/{strategy_id}/config
更新策略配置。

### 信号查询

#### GET /signals/history
获取信号历史记录。

**查询参数:**
- `page` (int): 页码
- `page_size` (int): 每页数量
- `strategy_ids` (string): 策略ID列表，逗号分隔
- `start_time` (datetime): 开始时间
- `end_time` (datetime): 结束时间
- `signal_types` (string): 信号类型过滤
- `min_confidence` (float): 最小置信度

**响应:**
```json
{
  "status": "success",
  "message": "Signal history retrieved successfully",
  "data": {
    "items": [
      {
        "signal_id": "signal_123",
        "strategy_id": "my_hft_strategy",
        "signal_type": "trade",
        "action": "buy",
        "confidence": 0.85,
        "strength": 0.75,
        "symbol": "BTCUSDT",
        "price": 50000.0,
        "volume": 0.1,
        "metadata": {
          "indicator_values": {
            "rsi": 65.2,
            "macd": 0.15
          }
        },
        "timestamp": "2024-01-01T12:00:00Z",
        "created_at": "2024-01-01T12:00:01Z"
      }
    ],
    "total": 500,
    "page": 1,
    "page_size": 50
  }
}
```

#### GET /signals/aggregation/statistics
获取信号聚合统计信息。

### 系统监控

#### GET /health
系统健康检查（无需认证）。

#### GET /system/status
获取详细系统状态。

**响应:**
```json
{
  "status": "success",
  "data": {
    "overall_status": "healthy",
    "uptime_seconds": 3600,
    "version": "1.0.0",
    "environment": "production",
    "components": [
      {
        "component_name": "database",
        "status": "healthy",
        "uptime_seconds": 3600,
        "last_error": null,
        "metrics": {
          "connection_count": 5,
          "query_avg_time_ms": 15.2
        }
      },
      {
        "component_name": "strategy_manager",
        "status": "healthy",
        "uptime_seconds": 3500,
        "metrics": {
          "active_strategies": 3,
          "total_signals": 1250
        }
      }
    ],
    "resource_usage": {
      "cpu_percent": 15.2,
      "memory_usage_mb": 512.5,
      "memory_percent": 12.8,
      "disk_usage_gb": 2.5,
      "disk_percent": 5.0
    },
    "active_strategies": 3,
    "total_requests": 15420,
    "error_rate": 0.002,
    "average_response_time_ms": 85.5
  }
}
```

#### GET /system/metrics
获取系统指标数据。

## WebSocket API

### 连接
```
ws://localhost:8000/ws/{connection_id}
```

### 订阅消息
```json
{
  "type": "subscribe",
  "channels": ["signals", "strategy_status", "system_metrics"],
  "filters": {
    "strategy_id": "my_strategy",
    "min_confidence": 0.8
  }
}
```

### 实时数据推送
```json
{
  "channel": "signals",
  "data": {
    "signal_id": "signal_456",
    "strategy_id": "my_hft_strategy",
    "action": "buy",
    "confidence": 0.92,
    "symbol": "BTCUSDT",
    "price": 50100.0
  },
  "timestamp": "2024-01-01T12:05:00Z"
}
```

## 错误处理

### 错误响应格式
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": "Field 'strategy_id' is required",
    "timestamp": "2024-01-01T12:00:00Z",
    "trace_id": "abc123",
    "context": {
      "field": "strategy_id",
      "value": null
    }
  },
  "status": "error",
  "timestamp": "2024-01-01T12:00:00Z",
  "path": "/strategies/start",
  "method": "POST"
}
```

### HTTP状态码

| 状态码 | 描述 | 示例场景 |
|--------|------|----------|
| 200 | 成功 | 正常请求成功 |
| 201 | 创建成功 | 创建资源成功 |
| 400 | 请求错误 | 参数验证失败 |
| 401 | 未认证 | Token无效或过期 |
| 403 | 权限不足 | 无操作权限 |
| 404 | 资源未找到 | 策略不存在 |
| 409 | 冲突 | 策略已在运行 |
| 422 | 数据错误 | 业务逻辑验证失败 |
| 429 | 请求过多 | 触发速率限制 |
| 500 | 服务器错误 | 内部系统错误 |
| 503 | 服务不可用 | 系统维护中 |

### 常见错误代码

| 错误代码 | 描述 |
|----------|------|
| AUTHENTICATION_FAILED | 认证失败 |
| TOKEN_EXPIRED | Token已过期 |
| INSUFFICIENT_PERMISSIONS | 权限不足 |
| VALIDATION_ERROR | 参数验证失败 |
| STRATEGY_NOT_FOUND | 策略不存在 |
| STRATEGY_ALREADY_RUNNING | 策略已在运行 |
| RATE_LIMIT_EXCEEDED | 速率限制超出 |
| INTERNAL_ERROR | 内部服务器错误 |

## 速率限制

API实施多层速率限制：

| 限制类型 | 默认限制 | 说明 |
|----------|----------|------|
| 全局限制 | 1000请求/分钟 | 整个系统的总限制 |
| IP限制 | 100请求/分钟 | 每个IP地址的限制 |
| 用户限制 | 200请求/分钟 | 每个认证用户的限制 |
| API Key限制 | 500请求/分钟 | 每个API密钥的限制 |
| 认证端点 | 10请求/分钟 | 登录端点的特殊限制 |

速率限制响应头：
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704110400
Retry-After: 60
```

## 数据模型

### 策略实例
```json
{
  "strategy_id": "string",
  "name": "string",
  "strategy_type": "hft | ai_agent",
  "status": "stopped | starting | running | paused | stopping | error",
  "description": "string",
  "config": {},
  "metrics": {
    "total_signals": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "total_profit_loss": 0.0,
    "win_rate": 0.0,
    "average_return": 0.0
  },
  "created_at": "datetime",
  "updated_at": "datetime",
  "started_at": "datetime",
  "stopped_at": "datetime"
}
```

### 信号数据
```json
{
  "signal_id": "string",
  "strategy_id": "string",
  "signal_type": "string",
  "action": "buy | sell | hold",
  "confidence": 0.0,
  "strength": 0.0,
  "symbol": "string",
  "price": 0.0,
  "volume": 0.0,
  "metadata": {},
  "timestamp": "datetime"
}
```

## SDK和示例

### Python客户端示例
```python
import requests
import json

class TradingAPIClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.token = None
        self.login(username, password)
    
    def login(self, username, password):
        response = requests.post(f"{self.base_url}/auth/login", json={
            "username": username,
            "password": password
        })
        if response.status_code == 200:
            self.token = response.json()["data"]["access_token"]
        else:
            raise Exception("Login failed")
    
    def get_strategies(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.base_url}/strategies", headers=headers)
        return response.json()
    
    def start_strategy(self, strategy_id, config=None):
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {
            "strategy_id": strategy_id,
            "strategy_type": "hft",
            "config": config or {}
        }
        response = requests.post(
            f"{self.base_url}/strategies/{strategy_id}/start",
            headers=headers,
            json=data
        )
        return response.json()

# 使用示例
client = TradingAPIClient("http://localhost:8000", "admin", "admin123")
strategies = client.get_strategies()
result = client.start_strategy("my_strategy", {"symbol": "BTCUSDT"})
```

### JavaScript客户端示例
```javascript
class TradingAPIClient {
    constructor(baseUrl, username, password) {
        this.baseUrl = baseUrl;
        this.token = null;
        this.login(username, password);
    }
    
    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/auth/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password})
        });
        
        if (response.ok) {
            const data = await response.json();
            this.token = data.data.access_token;
        } else {
            throw new Error('Login failed');
        }
    }
    
    async getStrategies() {
        const response = await fetch(`${this.baseUrl}/strategies`, {
            headers: {'Authorization': `Bearer ${this.token}`}
        });
        return await response.json();
    }
    
    async startStrategy(strategyId, config = {}) {
        const response = await fetch(`${this.baseUrl}/strategies/${strategyId}/start`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                strategy_id: strategyId,
                strategy_type: 'hft',
                config: config
            })
        });
        return await response.json();
    }
}

// 使用示例
const client = new TradingAPIClient('http://localhost:8000', 'admin', 'admin123');
const strategies = await client.getStrategies();
const result = await client.startStrategy('my_strategy', {symbol: 'BTCUSDT'});
```

## 部署和配置

### 环境变量
```bash
# API服务配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# 认证配置
AUTH_JWT_SECRET_KEY=your-secret-key
AUTH_ACCESS_TOKEN_EXPIRE_MINUTES=30
AUTH_REFRESH_TOKEN_EXPIRE_DAYS=7

# 速率限制配置
RATE_LIMIT_GLOBAL_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_PER_IP_REQUESTS_PER_MINUTE=100
RATE_LIMIT_PER_USER_REQUESTS_PER_MINUTE=200

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db

# Redis配置（用于缓存和会话）
REDIS_URL=redis://localhost:6379/0
```

### Docker部署
```yaml
version: '3.8'
services:
  trading-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DATABASE_URL=postgresql://user:password@db:5432/trading_db
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    
volumes:
  postgres_data:
```

## 监控和日志

### 日志格式
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "trading_api",
  "message": "Strategy started successfully",
  "context": {
    "strategy_id": "my_hft_strategy",
    "user_id": "admin-001",
    "request_id": "abc123",
    "execution_time_ms": 150.5
  }
}
```

### 健康检查端点
- `GET /health` - 基础健康检查
- `GET /health/ready` - 就绪检查
- `GET /health/live` - 存活检查

### 指标监控
API暴露Prometheus格式的指标：
- `http_requests_total` - 总请求数
- `http_request_duration_seconds` - 请求持续时间
- `active_strategies_total` - 活跃策略数
- `websocket_connections_total` - WebSocket连接数

## 常见问题

### Q: 如何处理Token过期？
A: 当收到401错误且错误代码为TOKEN_EXPIRED时，使用refresh_token调用/auth/refresh端点获取新的访问令牌。

### Q: API支持哪些策略类型？
A: 目前支持HFT（高频交易）和AI_AGENT（AI代理）两种策略类型。

### Q: 如何监控API性能？
A: 可以通过/system/metrics端点获取性能指标，或使用Prometheus抓取指标数据。

### Q: WebSocket连接如何认证？
A: 在建立WebSocket连接时，可以通过查询参数传递token：`ws://localhost:8000/ws/connection_id?token=<access_token>`

### Q: 速率限制达到后如何处理？
A: 检查响应头中的Retry-After值，等待指定时间后重试。考虑实施指数退避策略。

---

更多详细信息请参考：
- [在线API文档](http://localhost:8000/docs)
- [ReDoc文档](http://localhost:8000/redoc)
- [OpenAPI规范](http://localhost:8000/openapi.json)