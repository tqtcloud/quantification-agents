# 量化交易API快速启动指南

## 🚀 快速开始

### 1. 启动API服务

#### 开发模式（推荐用于测试）
```bash
# 启动开发服务器（自动重载）
./scripts/dev_api.sh
```

#### 生产模式
```bash
# 启动生产服务器
./scripts/start_api.sh
```

### 2. 访问API文档
启动后访问以下地址：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/health

## 🔐 认证快速测试

### 1. 登录获取Token
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

响应示例：
```json
{
  "status": "success",
  "message": "Authentication successful", 
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

### 2. 使用Token访问受保护的API
```bash
# 使用获取的token
TOKEN="your_access_token_here"

curl -X GET "http://localhost:8000/strategies" \
  -H "Authorization: Bearer $TOKEN"
```

## 🎯 主要API端点测试

### 策略控制
```bash
# 获取策略列表
curl -X GET "http://localhost:8000/strategies" \
  -H "Authorization: Bearer $TOKEN"

# 启动策略
curl -X POST "http://localhost:8000/strategies/test_strategy/start" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "test_strategy",
    "strategy_type": "hft",
    "config": {"symbol": "BTCUSDT"}
  }'

# 获取策略状态  
curl -X GET "http://localhost:8000/strategies/test_strategy/status" \
  -H "Authorization: Bearer $TOKEN"
```

### 系统监控
```bash
# 健康检查（无需认证）
curl -X GET "http://localhost:8000/health"

# 系统状态（需要认证）
curl -X GET "http://localhost:8000/system/status" \
  -H "Authorization: Bearer $TOKEN"
```

### 信号查询
```bash
# 获取信号历史
curl -X GET "http://localhost:8000/signals/history?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN"

# 获取聚合统计
curl -X GET "http://localhost:8000/signals/aggregation/statistics" \
  -H "Authorization: Bearer $TOKEN"
```

## 🔧 WebSocket连接测试

### JavaScript示例
```javascript
// 建立WebSocket连接
const ws = new WebSocket('ws://localhost:8000/ws/test_connection');

ws.onopen = function(event) {
    console.log('WebSocket连接已建立');
    
    // 订阅信号频道
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['signals', 'strategy_status'],
        filters: {
            strategy_id: 'test_strategy',
            min_confidence: 0.8
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('收到消息:', data);
};

ws.onerror = function(error) {
    console.error('WebSocket错误:', error);
};
```

### Python示例
```python
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"收到消息: {data}")

def on_open(ws):
    print("WebSocket连接已建立")
    # 订阅信号
    subscribe_msg = {
        "type": "subscribe",
        "channels": ["signals", "strategy_status"],
        "filters": {
            "strategy_id": "test_strategy"
        }
    }
    ws.send(json.dumps(subscribe_msg))

# 建立连接
ws = websocket.WebSocketApp("ws://localhost:8000/ws/test_connection",
                          on_open=on_open,
                          on_message=on_message)
ws.run_forever()
```

## 📊 Python客户端示例

```python
import requests
import json

class TradingAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
    
    def login(self, username="admin", password="admin123"):
        """登录获取Token"""
        response = requests.post(f"{self.base_url}/auth/login", json={
            "username": username,
            "password": password
        })
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["data"]["access_token"]
            print("登录成功！")
            return True
        else:
            print(f"登录失败: {response.text}")
            return False
    
    def get_strategies(self):
        """获取策略列表"""
        if not self.token:
            raise Exception("请先登录")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(f"{self.base_url}/strategies", headers=headers)
        
        if response.status_code == 200:
            return response.json()["data"]["items"]
        else:
            raise Exception(f"获取策略失败: {response.text}")
    
    def start_strategy(self, strategy_id, config=None):
        """启动策略"""
        if not self.token:
            raise Exception("请先登录")
        
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
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"启动策略失败: {response.text}")

# 使用示例
client = TradingAPIClient()

# 登录
if client.login():
    # 获取策略列表
    strategies = client.get_strategies()
    print(f"找到 {len(strategies)} 个策略")
    
    # 启动策略
    if strategies:
        result = client.start_strategy(strategies[0]["strategy_id"])
        print(f"策略启动结果: {result}")
```

## 🧪 运行测试

```bash
# 运行基础API测试
source .venv/bin/activate
python -m pytest tests/test_api_basic.py -v

# 运行所有API测试（需要服务器运行）
python -m pytest tests/test_trading_api.py -v

# 运行特定测试
python -m pytest tests/test_api_basic.py::test_import_api_components -v
```

## 📈 性能测试

### 并发测试示例
```python
import asyncio
import aiohttp
import time

async def test_concurrent_requests(num_requests=100):
    """并发请求测试"""
    
    # 先登录获取token
    async with aiohttp.ClientSession() as session:
        # 登录
        login_data = {"username": "admin", "password": "admin123"}
        async with session.post("http://localhost:8000/auth/login", json=login_data) as resp:
            result = await resp.json()
            token = result["data"]["access_token"]
        
        # 并发请求
        headers = {"Authorization": f"Bearer {token}"}
        
        async def make_request():
            async with session.get("http://localhost:8000/health", headers=headers) as resp:
                return resp.status
        
        start_time = time.time()
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        success_count = sum(1 for status in results if status == 200)
        total_time = end_time - start_time
        
        print(f"并发请求测试结果:")
        print(f"总请求数: {num_requests}")
        print(f"成功请求数: {success_count}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均QPS: {num_requests/total_time:.2f}")

# 运行并发测试
asyncio.run(test_concurrent_requests())
```

## 🐛 常见问题解决

### 1. 端口占用
```bash
# 检查端口占用
lsof -i :8000

# 杀死占用进程
kill -9 <PID>
```

### 2. 权限问题
```bash
# 确保脚本可执行
chmod +x scripts/dev_api.sh
chmod +x scripts/start_api.sh
```

### 3. 依赖问题
```bash
# 重新安装依赖
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. 数据库连接问题
```bash
# 检查数据库配置
export DATABASE_URL="sqlite:///./data/trading.db"
```

## 📞 获取帮助

- **API文档**: http://localhost:8000/docs
- **系统状态**: http://localhost:8000/system/status
- **健康检查**: http://localhost:8000/health
- **日志文件**: `logs/api/`

## 🎉 恭喜！

您现在已经成功启动了量化交易API服务！可以开始构建您的交易应用了。

建议下一步：
1. 熟悉API文档中的所有端点
2. 尝试不同的策略控制操作
3. 设置WebSocket实时数据订阅
4. 集成到您的交易应用中