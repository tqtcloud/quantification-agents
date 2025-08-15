import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.exchanges.trading_interface import (
    TradingInterface,
    BinanceTradingInterface,
    TradingEnvironment,
    TradingContext
)
from src.exchanges.trading_manager import TradingManager
from src.core.models import Order, OrderSide, OrderType, PositionSide, TimeInForce


class TestTradingContext:
    """交易上下文测试"""
    
    def test_trading_context_creation(self):
        """测试交易上下文创建"""
        context = TradingContext(
            environment=TradingEnvironment.TESTNET,
            session_id="test_session",
            user_id="test_user"
        )
        
        assert context.environment == TradingEnvironment.TESTNET
        assert context.session_id == "test_session"
        assert context.user_id == "test_user"
        assert context.metadata == {}
    
    def test_trading_context_metadata_init(self):
        """测试元数据自动初始化"""
        context = TradingContext(
            environment=TradingEnvironment.MAINNET,
            session_id="test"
        )
        
        assert context.metadata is not None
        assert isinstance(context.metadata, dict)


class TestBinanceTradingInterface:
    """币安交易接口测试"""
    
    @pytest.fixture
    def mock_client(self):
        """创建模拟客户端"""
        client = AsyncMock()
        client.testnet = True
        return client
    
    @pytest.fixture
    def testnet_interface(self, mock_client):
        """创建testnet接口"""
        return BinanceTradingInterface(mock_client, TradingEnvironment.TESTNET)
    
    def test_interface_initialization_testnet(self, mock_client):
        """测试testnet接口初始化"""
        interface = BinanceTradingInterface(mock_client, TradingEnvironment.TESTNET)
        
        assert interface.environment == TradingEnvironment.TESTNET
        assert interface.client == mock_client
        assert interface.is_testnet
        assert not interface.is_mainnet
        assert not interface.is_paper
    
    def test_interface_initialization_mainnet(self):
        """测试mainnet接口初始化"""
        mock_client = AsyncMock()
        mock_client.testnet = False
        
        interface = BinanceTradingInterface(mock_client, TradingEnvironment.MAINNET)
        
        assert interface.environment == TradingEnvironment.MAINNET
        assert not interface.is_testnet
        assert interface.is_mainnet
        assert not interface.is_paper
    
    def test_interface_client_environment_mismatch(self):
        """测试客户端环境不匹配"""
        mock_client = AsyncMock()
        mock_client.testnet = False
        
        # testnet环境但客户端配置为mainnet
        with pytest.raises(ValueError, match="Client must be configured for testnet"):
            BinanceTradingInterface(mock_client, TradingEnvironment.TESTNET)
    
    def test_data_prefix(self, testnet_interface):
        """测试数据前缀"""
        prefix = testnet_interface.get_data_prefix()
        assert prefix == "testnet_"
    
    def test_tag_order(self, testnet_interface):
        """测试订单标签"""
        order_data = {"orderId": "123", "symbol": "BTCUSDT"}
        tagged = testnet_interface.tag_order(order_data)
        
        assert tagged["trading_environment"] == "testnet"
        assert tagged["session_id"] == testnet_interface.context.session_id
        assert tagged["orderId"] == "123"
        assert tagged["symbol"] == "BTCUSDT"
    
    def test_tag_position(self, testnet_interface):
        """测试持仓标签"""
        position_data = {"symbol": "ETHUSDT", "positionAmt": "1.5"}
        tagged = testnet_interface.tag_position(position_data)
        
        assert tagged["trading_environment"] == "testnet"
        assert tagged["session_id"] == testnet_interface.context.session_id
        assert tagged["symbol"] == "ETHUSDT"
        assert tagged["positionAmt"] == "1.5"
    
    @pytest.mark.asyncio
    async def test_connect_success(self, testnet_interface, mock_client):
        """测试连接成功"""
        mock_client.connect.return_value = None
        mock_client.ping.return_value = {}
        
        result = await testnet_interface.connect()
        
        assert result is True
        mock_client.connect.assert_called_once()
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, testnet_interface, mock_client):
        """测试连接失败"""
        mock_client.connect.side_effect = Exception("Connection failed")
        
        result = await testnet_interface.connect()
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_place_order(self, testnet_interface, mock_client):
        """测试下单"""
        mock_client.place_order.return_value = {
            "orderId": "12345",
            "symbol": "BTCUSDT",
            "status": "NEW"
        }
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            time_in_force=TimeInForce.GTC,
            position_side=PositionSide.BOTH
        )
        
        result = await testnet_interface.place_order(order)
        
        assert result["orderId"] == "12345"
        assert result["trading_environment"] == "testnet"
        assert result["session_id"] == testnet_interface.context.session_id
        
        mock_client.place_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=1.0,
            price=50000.0,
            stop_price=None,
            time_in_force="GTC",
            reduce_only=False,
            position_side="BOTH",
            client_order_id=None
        )
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, testnet_interface, mock_client):
        """测试撤单"""
        mock_client.cancel_order.return_value = {
            "orderId": "12345",
            "status": "CANCELED"
        }
        
        result = await testnet_interface.cancel_order("BTCUSDT", "12345")
        
        assert result["orderId"] == "12345"
        assert result["trading_environment"] == "testnet"
        
        mock_client.cancel_order.assert_called_once_with("BTCUSDT", order_id=12345)
    
    @pytest.mark.asyncio
    async def test_get_open_orders_single(self, testnet_interface, mock_client):
        """测试获取挂单 - 单个订单"""
        mock_client.get_open_orders.return_value = {
            "orderId": "12345",
            "symbol": "BTCUSDT"
        }
        
        result = await testnet_interface.get_open_orders()
        
        assert len(result) == 1
        assert result[0]["orderId"] == "12345"
        assert result[0]["trading_environment"] == "testnet"
    
    @pytest.mark.asyncio
    async def test_get_open_orders_multiple(self, testnet_interface, mock_client):
        """测试获取挂单 - 多个订单"""
        mock_client.get_open_orders.return_value = [
            {"orderId": "12345", "symbol": "BTCUSDT"},
            {"orderId": "67890", "symbol": "ETHUSDT"}
        ]
        
        result = await testnet_interface.get_open_orders()
        
        assert len(result) == 2
        for order in result:
            assert order["trading_environment"] == "testnet"


class TestTradingManager:
    """交易管理器测试"""
    
    @pytest.fixture
    def manager(self):
        """创建交易管理器"""
        return TradingManager()
    
    @pytest.fixture
    def mock_interface(self):
        """创建模拟交易接口"""
        interface = AsyncMock(spec=TradingInterface)
        interface.environment = TradingEnvironment.TESTNET
        return interface
    
    def test_manager_initialization(self, manager):
        """测试管理器初始化"""
        assert manager._current_environment is None
        assert len(manager._interfaces) == 0
        assert manager._data_isolation_enabled is True
        assert manager.available_environments == []
    
    def test_register_interface(self, manager, mock_interface):
        """测试注册交易接口"""
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        
        assert TradingEnvironment.TESTNET in manager._interfaces
        assert manager._interfaces[TradingEnvironment.TESTNET] == mock_interface
        assert TradingEnvironment.TESTNET in manager.available_environments
    
    def test_set_environment(self, manager, mock_interface):
        """测试设置环境"""
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        manager.set_environment(TradingEnvironment.TESTNET)
        
        assert manager.get_current_environment() == TradingEnvironment.TESTNET
    
    def test_set_invalid_environment(self, manager):
        """测试设置无效环境"""
        with pytest.raises(ValueError, match="No interface registered"):
            manager.set_environment(TradingEnvironment.TESTNET)
    
    def test_get_interface_current(self, manager, mock_interface):
        """测试获取当前环境接口"""
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        manager.set_environment(TradingEnvironment.TESTNET)
        
        interface = manager.get_interface()
        assert interface == mock_interface
    
    def test_get_interface_specific(self, manager, mock_interface):
        """测试获取特定环境接口"""
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        
        interface = manager.get_interface(TradingEnvironment.TESTNET)
        assert interface == mock_interface
    
    def test_get_interface_no_current(self, manager):
        """测试无当前环境时获取接口"""
        with pytest.raises(ValueError, match="No current environment set"):
            manager.get_interface()
    
    def test_get_isolated_key(self, manager):
        """测试获取隔离键"""
        manager.set_environment = MagicMock()
        manager._current_environment = TradingEnvironment.TESTNET
        
        key = manager.get_isolated_key("test_key")
        assert key == "testnet:test_key"
    
    def test_get_isolated_key_disabled(self, manager):
        """测试禁用隔离时获取键"""
        manager.enable_data_isolation(False)
        key = manager.get_isolated_key("test_key")
        assert key == "test_key"
    
    def test_extract_base_key(self, manager):
        """测试提取基础键"""
        base_key, env = manager.extract_base_key("testnet:orders")
        assert base_key == "orders"
        assert env == TradingEnvironment.TESTNET
    
    def test_extract_base_key_no_prefix(self, manager):
        """测试提取无前缀键"""
        base_key, env = manager.extract_base_key("orders")
        assert base_key == "orders"
        assert env is None
    
    def test_filter_data_by_environment(self, manager):
        """测试按环境过滤数据"""
        data = [
            {"id": "1", "trading_environment": "testnet"},
            {"id": "2", "trading_environment": "mainnet"},
            {"id": "3", "trading_environment": "testnet"},
        ]
        
        testnet_data = manager.filter_data_by_environment(data, TradingEnvironment.TESTNET)
        assert len(testnet_data) == 2
        assert all(item["trading_environment"] == "testnet" for item in testnet_data)
        
        mainnet_data = manager.filter_data_by_environment(data, TradingEnvironment.MAINNET)
        assert len(mainnet_data) == 1
        assert mainnet_data[0]["trading_environment"] == "mainnet"
    
    def test_enable_disable_isolation(self, manager):
        """测试启用/禁用数据隔离"""
        assert manager._data_isolation_enabled is True
        
        manager.enable_data_isolation(False)
        assert manager._data_isolation_enabled is False
        
        manager.enable_data_isolation(True)
        assert manager._data_isolation_enabled is True
    
    @pytest.mark.asyncio
    async def test_use_environment_context(self, manager, mock_interface):
        """测试环境上下文管理器"""
        mock_interface.connect.return_value = True
        mock_interface.disconnect.return_value = None
        
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        
        async with manager.use_environment(TradingEnvironment.TESTNET) as interface:
            assert interface == mock_interface
            assert manager.get_current_environment() == TradingEnvironment.TESTNET
        
        # 上下文结束后应该清除当前环境
        assert manager.get_current_environment() is None
        
        mock_interface.connect.assert_called_once()
        mock_interface.disconnect.assert_called_once()
    
    def test_repr(self, manager, mock_interface):
        """测试字符串表示"""
        manager.register_interface(TradingEnvironment.TESTNET, mock_interface)
        manager.set_environment(TradingEnvironment.TESTNET)
        
        repr_str = repr(manager)
        assert "TradingManager" in repr_str
        assert "TESTNET" in repr_str
        assert "isolation_enabled=True" in repr_str