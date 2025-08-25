# REST API & WebSocket 集成测试套件

这是量化交易系统的完整集成测试套件，用于验证REST API和WebSocket服务的功能、性能和安全性。

## 📋 测试覆盖范围

### 1. API集成测试 (`test_api_integration.py`)
- **用户认证和授权**
  - 用户注册、登录、登出
  - 令牌刷新和过期处理
  - 权限控制和角色管理
- **速率限制机制**
  - 请求频率限制
  - 突发请求处理
  - 限制重置和恢复
- **请求验证**
  - 参数验证和错误处理
  - 输入数据清理和转换
  - 恶意输入防护
- **错误处理**
  - HTTP状态码正确性
  - 错误消息格式化
  - 异常情况处理

### 2. WebSocket集成测试 (`test_websocket_integration.py`)
- **连接管理**
  - 连接建立和维护
  - 心跳机制
  - 连接数量限制
  - 异常断开处理
- **订阅管理**
  - 频道订阅和取消
  - 多重订阅支持
  - 订阅权限控制
- **消息广播**
  - 实时消息推送
  - 消息路由和分发
  - 消息序列化和反序列化

### 3. 端到端集成测试 (`test_end_to_end_integration.py`)
- **完整用户场景**
  - 用户注册 → 登录 → WebSocket连接 → 数据订阅 → 消息接收
  - 多用户并发场景
  - 服务间协同工作验证
- **实时数据同步**
  - API触发的WebSocket推送
  - 数据一致性验证
  - 跨服务状态同步
- **错误传播和恢复**
  - 错误隔离机制
  - 服务自愈能力
  - 故障转移处理

### 4. 性能基准测试 (`test_performance_benchmarks.py`)
- **API性能指标**
  - 响应时间 (目标: <100ms)
  - 吞吐量 (目标: >1000 RPS)
  - 并发处理能力
- **WebSocket性能**
  - 连接延迟 (目标: <50ms)
  - 消息传输延迟
  - 并发连接数 (目标: 1000+)
- **系统资源使用**
  - 内存使用监控
  - CPU使用率分析
  - 网络带宽测试

### 5. 安全和负载测试 (`test_security_and_load.py`)
- **安全攻击防护**
  - SQL注入攻击防护
  - XSS攻击防护
  - 暴力破解防护
- **负载和压力测试**
  - DDoS攻击模拟
  - 连接洪水攻击
  - 消息轰炸攻击
- **系统稳定性**
  - 高并发处理
  - 过载恢复机制
  - 资源耗尽处理

## 🛠️ 环境要求

### Python版本
- Python 3.8+

### 必需依赖
```bash
# 安装测试依赖
pip install -r tests/integration/requirements.txt
```

### 系统要求
- 内存: 至少 2GB 可用内存
- 磁盘: 至少 1GB 可用空间
- 网络: 本地网络访问权限
- 端口: 8000-9000范围内的端口访问权限

## 🚀 快速开始

### 1. 环境准备
```bash
# 进入项目根目录
cd /path/to/quantification-agents

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r tests/integration/requirements.txt
```

### 2. 运行测试

#### 运行所有集成测试
```bash
./scripts/run_integration_tests.sh
```

#### 运行特定测试套件
```bash
# API集成测试
./scripts/run_integration_tests.sh api

# WebSocket集成测试  
./scripts/run_integration_tests.sh websocket

# 端到端测试
./scripts/run_integration_tests.sh e2e

# 性能测试
./scripts/run_integration_tests.sh performance

# 安全测试
./scripts/run_integration_tests.sh security
```

#### 直接使用pytest运行
```bash
# 运行所有集成测试
pytest tests/integration/ -v

# 运行特定测试文件
pytest tests/integration/test_api_integration.py -v

# 运行特定测试类
pytest tests/integration/test_api_integration.py::TestAPIAuthentication -v

# 运行特定测试方法
pytest tests/integration/test_api_integration.py::TestAPIAuthentication::test_user_registration_and_authentication -v
```

### 3. 测试标记使用

#### 按标记运行测试
```bash
# 只运行快速测试
pytest -m "fast" tests/integration/ -v

# 跳过慢速测试
pytest -m "not slow" tests/integration/ -v

# 只运行API相关测试
pytest -m "api" tests/integration/ -v

# 只运行性能测试
pytest -m "performance" tests/integration/ -v

# 只运行安全测试
pytest -m "security" tests/integration/ -v
```

#### 组合标记
```bash
# 运行API和WebSocket测试，但跳过慢速测试
pytest -m "api or websocket and not slow" tests/integration/ -v
```

## 📊 测试报告

### 报告类型

#### 1. HTML报告
- **位置**: `test_reports/integration_report_YYYYMMDD_HHMMSS.html`
- **内容**: 完整的可视化测试报告，包含图表和统计信息

#### 2. JSON报告
- **位置**: `test_reports/integration_report_YYYYMMDD_HHMMSS.json`
- **内容**: 机器可读的详细测试数据

#### 3. 覆盖率报告
- **HTML**: `test_reports/htmlcov/index.html`
- **XML**: `test_reports/coverage.xml`
- **终端**: 测试运行时实时显示

#### 4. JUnit XML报告
- **位置**: `test_reports/junit_YYYYMMDD_HHMMSS.xml`
- **用途**: CI/CD系统集成

### 报告查看
```bash
# 在浏览器中打开HTML报告
open test_reports/integration_report_latest.html

# 查看覆盖率报告
open test_reports/htmlcov/index.html
```

## ⚙️ 配置说明

### pytest配置 (`pytest.ini`)
- 测试发现规则
- 标记定义
- 覆盖率配置
- 日志设置
- 超时配置

### 测试配置 (`conftest.py`)
- 通用fixtures
- 测试数据工厂
- Mock对象
- 环境设置

### 主要配置参数
```python
# 数据库配置
DATABASE_URL = "sqlite:///:memory:"

# API配置
API_HOST = "127.0.0.1"
API_PORT = 8000

# WebSocket配置
WEBSOCKET_HOST = "127.0.0.1"  
WEBSOCKET_PORT = 8765

# 性能目标
API_RESPONSE_TIME_TARGET = 100  # ms
WEBSOCKET_LATENCY_TARGET = 50   # ms
CONCURRENT_CONNECTIONS_TARGET = 1000
```

## 🧪 编写新测试

### 测试类结构
```python
import pytest
import pytest_asyncio

class TestMyFeature:
    """我的功能测试"""
    
    @pytest_asyncio.fixture
    async def setup_feature(self):
        """测试前设置"""
        # 设置代码
        yield
        # 清理代码
    
    @pytest.mark.asyncio
    async def test_feature_works(self, setup_feature):
        """测试功能正常工作"""
        # 测试代码
        assert True
```

### 使用fixtures
```python
@pytest.mark.asyncio
async def test_with_authenticated_client(self, authenticated_client):
    """使用已认证客户端的测试"""
    client, token = authenticated_client
    
    response = await client.get("/protected-endpoint")
    assert response.status_code == 200
```

### 性能测试示例
```python
@pytest.mark.performance
@pytest.mark.asyncio
async def test_api_response_time(self, http_client, performance_timer):
    """测试API响应时间"""
    with performance_timer:
        response = await http_client.get("/api/endpoint")
    
    duration = performance_timer.stop()
    assert duration < 0.1  # 100ms
    assert response.status_code == 200
```

### 安全测试示例
```python
@pytest.mark.security
@pytest.mark.asyncio
async def test_sql_injection_protection(self, http_client, test_data_factory):
    """测试SQL注入防护"""
    malicious_payload = test_data_factory.create_attack_payload('sql_injection')
    
    response = await http_client.post('/api/search', json={
        'query': malicious_payload
    })
    
    # 应该被阻止或返回安全错误
    assert response.status_code in [400, 403, 422]
```

## 📝 测试最佳实践

### 1. 测试命名
- 使用描述性名称: `test_user_can_register_with_valid_email`
- 遵循模式: `test_<what>_<when>_<expected_result>`

### 2. 测试结构
- **Arrange**: 设置测试数据和环境
- **Act**: 执行被测试的操作
- **Assert**: 验证结果

### 3. 异步测试
```python
@pytest.mark.asyncio
async def test_async_operation(self):
    """异步操作测试"""
    result = await some_async_function()
    assert result is not None
```

### 4. 错误处理测试
```python
@pytest.mark.asyncio
async def test_handles_invalid_input(self, http_client):
    """测试无效输入处理"""
    response = await http_client.post('/api/endpoint', json={})
    
    assert response.status_code == 422
    error_data = response.json()
    assert 'detail' in error_data
```

### 5. 性能断言
```python
def test_performance_requirement(self, performance_timer):
    """性能要求测试"""
    with performance_timer:
        # 执行操作
        result = expensive_operation()
    
    stats = performance_timer.get_stats()
    assert stats['average'] < 0.1  # 100ms以内
    assert result is not None
```

## 🔧 故障排除

### 常见问题

#### 1. 端口冲突
```bash
# 错误: Address already in use
# 解决: 修改配置中的端口号，或杀死占用进程
lsof -ti:8765 | xargs kill -9
```

#### 2. 数据库连接问题
```bash
# 错误: Database connection failed
# 解决: 检查数据库配置，确保SQLite可写权限
```

#### 3. 内存不足
```bash
# 错误: MemoryError
# 解决: 减少并发测试数量，或增加系统内存
export PYTEST_WORKERS=2  # 减少并行度
```

#### 4. 测试超时
```bash
# 错误: Test timeout
# 解决: 增加超时时间或优化测试代码
pytest --timeout=600  # 10分钟超时
```

### 调试技巧

#### 1. 详细日志
```bash
pytest tests/integration/ -v -s --log-cli-level=DEBUG
```

#### 2. 停在第一个失败
```bash
pytest tests/integration/ -x
```

#### 3. 调试特定测试
```bash
pytest tests/integration/test_api_integration.py::TestAPIAuthentication::test_user_registration_and_authentication -v -s
```

#### 4. 检查覆盖率
```bash
pytest tests/integration/ --cov=src --cov-report=html
open htmlcov/index.html
```

## 📈 持续集成

### GitHub Actions 配置示例
```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/integration/requirements.txt
        
    - name: Run integration tests
      run: |
        ./scripts/run_integration_tests.sh
        
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-reports
        path: test_reports/
```

## 🔍 性能监控

### 性能指标目标
- **API响应时间**: < 100ms (平均)
- **WebSocket延迟**: < 50ms (平均)  
- **并发连接**: > 1000
- **吞吐量**: > 1000 RPS
- **测试覆盖率**: > 85%
- **成功率**: > 99%

### 监控命令
```bash
# 性能基准测试
pytest tests/integration/test_performance_benchmarks.py -v --benchmark-only

# 内存使用监控
pytest tests/integration/ --profile

# 详细性能报告
pytest tests/integration/ --durations=0
```

## 📚 扩展阅读

- [pytest 官方文档](https://docs.pytest.org/)
- [pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)
- [FastAPI 测试指南](https://fastapi.tiangolo.com/tutorial/testing/)
- [WebSocket 测试最佳实践](https://websockets.readthedocs.io/en/stable/topics/testing.html)

## 🤝 贡献指南

### 添加新测试
1. 在合适的测试文件中添加测试类/方法
2. 使用适当的标记标注测试
3. 编写清晰的文档字符串
4. 确保测试可重复和独立运行
5. 更新相关文档

### 测试审查清单
- [ ] 测试名称清晰描述测试内容
- [ ] 测试覆盖正常和异常情况
- [ ] 使用适当的断言
- [ ] 测试执行时间合理
- [ ] 测试之间无依赖关系
- [ ] 适当的错误处理
- [ ] 文档和注释完善

---

**联系方式**: 如有问题或建议，请通过项目Issue或邮件联系开发团队。