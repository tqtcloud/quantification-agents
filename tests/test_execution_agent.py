import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.execution_agent import (
    ExecutionAgent, ExecutionAgentConfig, ExecutionMode, ExecutionResult, 
    ExecutionContext, ExecutionReport
)
from src.core.models import Order, OrderSide, OrderType, OrderStatus, TimeInForce, PositionSide
from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError
from src.core.message_bus import MessageBus


class TestExecutionAgent:
    """执行Agent测试"""
    
    @pytest.fixture
    def execution_config(self):
        """创建执行Agent配置"""
        return ExecutionAgentConfig(
            name="test_execution_agent",
            execution_mode=ExecutionMode.PAPER,
            max_order_value=10000.0,
            max_daily_trades=100,
            order_timeout=30.0,
            enable_order_tracking=True,
            commission_rate=0.0004
        )
    
    @pytest.fixture
    def mock_binance_client(self):
        """创建模拟的币安客户端"""
        client = MagicMock(spec=BinanceFuturesClient)
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.place_order = AsyncMock()
        client.cancel_order = AsyncMock()
        client.get_order = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_message_bus(self):
        """创建模拟的消息总线"""
        return MagicMock(spec=MessageBus)
    
    @pytest.fixture
    async def execution_agent(self, execution_config, mock_binance_client, mock_message_bus):
        """创建执行Agent实例"""
        agent = ExecutionAgent(execution_config, mock_binance_client, mock_message_bus)
        await agent.initialize()
        yield agent
        await agent.shutdown()
    
    @pytest.fixture
    def sample_order(self):
        """创建示例订单"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            client_order_id="test_order_001"
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, execution_config, mock_binance_client, mock_message_bus):
        """测试Agent初始化"""
        agent = ExecutionAgent(execution_config, mock_binance_client, mock_message_bus)
        
        assert agent.execution_config.execution_mode == ExecutionMode.PAPER
        assert agent.execution_config.max_order_value == 10000.0
        assert len(agent._active_orders) == 0
        assert agent._daily_trade_count == 0
        
        await agent.initialize()
        
        # 验证币安客户端连接
        mock_binance_client.connect.assert_called_once()
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_paper_order_execution(self, execution_agent, sample_order):
        """测试模拟盘订单执行"""
        # 执行订单
        report = await execution_agent.execute_order(sample_order)
        
        # 验证执行结果
        assert report.result == ExecutionResult.SUCCESS
        assert report.executed_qty == sample_order.quantity
        assert report.avg_price > 0
        assert report.commission > 0
        assert report.execution_time > 0
        assert report.client_order_id == sample_order.client_order_id
        
        # 验证统计更新
        stats = execution_agent.get_execution_stats()
        assert stats["total_orders"] == 1
        assert stats["successful_orders"] == 1
        assert stats["total_volume"] > 0
    
    @pytest.mark.asyncio
    async def test_live_order_execution_success(self, mock_binance_client):
        """测试实盘订单执行成功"""
        # 配置实盘模式
        config = ExecutionAgentConfig(
            name="test_live_agent",
            execution_mode=ExecutionMode.LIVE,
            order_timeout=10.0
        )
        
        agent = ExecutionAgent(config, mock_binance_client)
        await agent.initialize()
        
        # 模拟币安API响应
        mock_binance_client.place_order.return_value = {
            "orderId": 12345,
            "status": "FILLED",
            "executedQty": "1.0",
            "avgPrice": "50000.0"
        }
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            client_order_id="test_live_001"
        )
        
        # 执行订单
        report = await agent.execute_order(order)
        
        # 验证结果
        assert report.result == ExecutionResult.SUCCESS
        assert report.order_id == "12345"
        assert report.executed_qty == 1.0
        assert report.avg_price == 50000.0
        
        # 验证币安API调用
        mock_binance_client.place_order.assert_called_once()
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_live_order_execution_pending(self, mock_binance_client):
        """测试实盘订单执行待成交"""
        config = ExecutionAgentConfig(
            name="test_pending_agent",
            execution_mode=ExecutionMode.LIVE,
            order_timeout=2.0  # 短超时时间用于测试
        )
        
        agent = ExecutionAgent(config, mock_binance_client)
        await agent.initialize()
        
        # 模拟币安API响应 - 订单提交成功但未立即成交
        mock_binance_client.place_order.return_value = {
            "orderId": 12346,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        # 模拟订单状态查询 - 最终成交
        mock_binance_client.get_order.return_value = {
            "orderId": 12346,
            "status": "FILLED",
            "executedQty": "1.0",
            "avgPrice": "50000.0"
        }
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            client_order_id="test_pending_001"
        )
        
        # 执行订单
        report = await agent.execute_order(order)
        
        # 验证结果
        assert report.result == ExecutionResult.SUCCESS
        assert report.executed_qty == 1.0
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_order_execution_failure(self, mock_binance_client):
        """测试订单执行失败"""
        config = ExecutionAgentConfig(
            name="test_failure_agent",
            execution_mode=ExecutionMode.LIVE
        )
        
        agent = ExecutionAgent(config, mock_binance_client)
        await agent.initialize()
        
        # 模拟币安API错误
        mock_binance_client.place_order.side_effect = BinanceAPIError(-1013, "Invalid quantity")
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,  # 无效数量
            price=50000.0,
            client_order_id="test_failure_001"
        )
        
        # 执行订单应该抛出异常
        with pytest.raises(BinanceAPIError):
            await agent.execute_order(order)
        
        # 验证失败统计
        assert len(agent._order_history) == 1
        assert agent._order_history[0].result == ExecutionResult.FAILED
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, execution_agent, sample_order, mock_binance_client):
        """测试订单取消"""
        # 修改为实盘模式以测试真实取消
        execution_agent.execution_config.execution_mode = ExecutionMode.LIVE
        
        # 模拟未成交的订单
        mock_binance_client.place_order.return_value = {
            "orderId": 12347,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        # 添加到活跃订单（模拟订单提交）
        from src.agents.execution_agent import ExecutionContext
        context = ExecutionContext(order=sample_order, execution_mode=ExecutionMode.LIVE)
        execution_agent._active_orders[sample_order.client_order_id] = context
        
        # 取消订单
        success = await execution_agent.cancel_order(sample_order.client_order_id)
        
        assert success
        assert sample_order.client_order_id not in execution_agent._active_orders
        mock_binance_client.cancel_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_order_status_query(self, execution_agent, sample_order, mock_binance_client):
        """测试订单状态查询"""
        # 添加到活跃订单
        from src.agents.execution_agent import ExecutionContext
        context = ExecutionContext(order=sample_order, execution_mode=ExecutionMode.LIVE)
        execution_agent._active_orders[sample_order.client_order_id] = context
        execution_agent.execution_config.execution_mode = ExecutionMode.LIVE
        
        # 模拟状态查询响应
        mock_binance_client.get_order.return_value = {
            "orderId": 12348,
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.5",
            "avgPrice": "50000.0"
        }
        
        # 查询订单状态
        status = await execution_agent.get_order_status(sample_order.client_order_id)
        
        assert status is not None
        assert status["status"] == "PARTIALLY_FILLED"
        assert status["executedQty"] == "0.5"
        mock_binance_client.get_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daily_trade_limit(self, execution_agent, sample_order):
        """测试每日交易限制"""
        # 设置较低的每日限制
        execution_agent.execution_config.max_daily_trades = 2
        
        # 执行第一笔订单
        order1 = sample_order.copy()
        order1.client_order_id = "test_limit_001"
        await execution_agent.execute_order(order1)
        
        # 执行第二笔订单
        order2 = sample_order.copy()
        order2.client_order_id = "test_limit_002"
        await execution_agent.execute_order(order2)
        
        # 第三笔订单应该被拒绝
        order3 = sample_order.copy()
        order3.client_order_id = "test_limit_003"
        
        with pytest.raises(ValueError, match="Daily trade limit reached"):
            await execution_agent.execute_order(order3)
    
    @pytest.mark.asyncio
    async def test_order_value_limit(self, execution_agent):
        """测试订单金额限制"""
        # 创建超过限制的大订单
        large_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=20000.0,  # 超过max_order_value(10000)
            client_order_id="test_large_001"
        )
        
        with pytest.raises(ValueError, match="Order value .* exceeds limit"):
            await execution_agent.execute_order(large_order)
    
    @pytest.mark.asyncio
    async def test_order_timeout_handling(self, mock_binance_client):
        """测试订单超时处理"""
        config = ExecutionAgentConfig(
            name="test_timeout_agent",
            execution_mode=ExecutionMode.LIVE,
            order_timeout=1.0,  # 1秒超时
            enable_order_tracking=True
        )
        
        agent = ExecutionAgent(config, mock_binance_client)
        await agent.initialize()
        
        # 模拟订单提交成功但长时间未成交
        mock_binance_client.place_order.return_value = {
            "orderId": 12349,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        # 模拟订单状态查询 - 始终未成交
        mock_binance_client.get_order.return_value = {
            "orderId": 12349,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=45000.0,  # 远离市价的限价单
            client_order_id="test_timeout_001"
        )
        
        # 执行订单，应该超时
        report = await agent.execute_order(order)
        
        # 验证超时结果
        assert report.result == ExecutionResult.FAILED
        assert "timeout" in report.error_message.lower()
        
        # 验证取消订单被调用
        mock_binance_client.cancel_order.assert_called()
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_execution_statistics(self, execution_agent, sample_order):
        """测试执行统计"""
        # 执行多个订单
        orders = []
        for i in range(3):
            order = sample_order.copy()
            order.client_order_id = f"test_stats_{i:03d}"
            order.quantity = 1.0 + i * 0.5  # 不同数量
            orders.append(order)
        
        # 执行所有订单
        for order in orders:
            await execution_agent.execute_order(order)
        
        # 检查统计
        stats = execution_agent.get_execution_stats()
        assert stats["total_orders"] == 3
        assert stats["successful_orders"] == 3
        assert stats["failed_orders"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["total_volume"] > 0
        assert stats["avg_execution_time"] > 0
        assert stats["daily_trade_count"] == 3
    
    @pytest.mark.asyncio
    async def test_active_orders_tracking(self, execution_agent, mock_binance_client):
        """测试活跃订单跟踪"""
        # 切换到实盘模式以测试真实跟踪
        execution_agent.execution_config.execution_mode = ExecutionMode.LIVE
        
        # 模拟未立即成交的订单
        mock_binance_client.place_order.return_value = {
            "orderId": 12350,
            "status": "NEW",
            "executedQty": "0.0"
        }
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=45000.0,
            client_order_id="test_tracking_001"
        )
        
        # 手动添加到活跃订单（模拟部分执行流程）
        from src.agents.execution_agent import ExecutionContext
        context = ExecutionContext(order=order, execution_mode=ExecutionMode.LIVE)
        execution_agent._active_orders[order.client_order_id] = context
        
        # 检查活跃订单
        active_orders = execution_agent.get_active_orders()
        assert len(active_orders) == 1
        assert active_orders[0]["client_order_id"] == order.client_order_id
        assert active_orders[0]["symbol"] == order.symbol
        assert active_orders[0]["quantity"] == order.quantity
    
    @pytest.mark.asyncio
    async def test_order_history(self, execution_agent, sample_order):
        """测试订单历史记录"""
        # 执行几个订单
        for i in range(5):
            order = sample_order.copy()
            order.client_order_id = f"test_history_{i:03d}"
            await execution_agent.execute_order(order)
        
        # 检查历史记录
        history = execution_agent.get_order_history()
        assert len(history) == 5
        
        # 检查限制数量
        limited_history = execution_agent.get_order_history(limit=3)
        assert len(limited_history) == 3
        
        # 验证历史记录内容
        for report in history:
            assert isinstance(report, ExecutionReport)
            assert report.result == ExecutionResult.SUCCESS
            assert report.executed_qty > 0
    
    @pytest.mark.asyncio
    async def test_message_handlers(self, execution_agent, sample_order):
        """测试消息处理器"""
        # 测试执行订单消息
        order_data = {
            "symbol": sample_order.symbol,
            "side": sample_order.side.value,
            "order_type": sample_order.order_type.value,
            "quantity": sample_order.quantity,
            "price": sample_order.price,
            "client_order_id": sample_order.client_order_id
        }
        
        # 模拟消息处理
        await execution_agent._handle_execute_order_message(
            "test_agent", 
            {"order": order_data}
        )
        
        # 验证订单被执行
        assert len(execution_agent._order_history) == 1
        assert execution_agent._order_history[0].client_order_id == sample_order.client_order_id
    
    def test_execution_context_creation(self):
        """测试执行上下文创建"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0
        )
        
        context = ExecutionContext(
            order=order,
            execution_mode=ExecutionMode.PAPER
        )
        
        assert context.order == order
        assert context.execution_mode == ExecutionMode.PAPER
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert context.max_slippage == 0.005
    
    def test_execution_report_creation(self):
        """测试执行报告创建"""
        report = ExecutionReport(
            order_id="12345",
            client_order_id="test_001",
            result=ExecutionResult.SUCCESS,
            executed_qty=1.0,
            avg_price=50000.0,
            commission=20.0,
            slippage=0.001,
            execution_time=1.5
        )
        
        assert report.order_id == "12345"
        assert report.client_order_id == "test_001"
        assert report.result == ExecutionResult.SUCCESS
        assert report.executed_qty == 1.0
        assert report.avg_price == 50000.0
        assert report.commission == 20.0
        assert report.slippage == 0.001
        assert report.execution_time == 1.5
        assert isinstance(report.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_shutdown_with_active_orders(self, mock_binance_client):
        """测试关闭时处理活跃订单"""
        config = ExecutionAgentConfig(
            name="test_shutdown_agent",
            execution_mode=ExecutionMode.LIVE
        )
        
        agent = ExecutionAgent(config, mock_binance_client)
        await agent.initialize()
        
        # 添加一些活跃订单
        for i in range(3):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                price=50000.0,
                client_order_id=f"shutdown_test_{i}"
            )
            
            from src.agents.execution_agent import ExecutionContext
            context = ExecutionContext(order=order, execution_mode=ExecutionMode.LIVE)
            agent._active_orders[order.client_order_id] = context
        
        # 关闭Agent
        await agent.shutdown()
        
        # 验证所有活跃订单被取消
        assert mock_binance_client.cancel_order.call_count == 3
        assert len(agent._active_orders) == 0