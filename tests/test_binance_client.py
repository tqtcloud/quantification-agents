import asyncio
import json
import pytest
import websockets
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch

from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError, BinanceConnectionError, BinanceWebSocketError


class TestBinanceFuturesClient:
    """币安期货客户端测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    def test_client_initialization(self, client):
        """测试客户端初始化"""
        assert client.api_key == "test_key"
        assert client.api_secret == "test_secret"
        assert client.testnet is True
        assert client.base_url == "https://testnet.binancefuture.com"
        assert client.ws_url == "wss://stream.binancefuture.com"
        assert client.environment == "testnet"
    
    def test_signature_generation(self, client):
        """测试签名生成"""
        params = {"symbol": "BTCUSDT", "side": "BUY", "timestamp": 1640995200000}
        signature = client._generate_signature(params)
        
        # 签名应该是64字符的十六进制字符串
        assert isinstance(signature, str)
        assert len(signature) == 64
        assert all(c in '0123456789abcdef' for c in signature.lower())
    
    def test_prepare_params_unsigned(self, client):
        """测试未签名参数准备"""
        params = {"symbol": "BTCUSDT", "limit": 500, "empty": None}
        prepared = client._prepare_params(params, signed=False)
        
        assert "empty" not in prepared
        assert prepared["symbol"] == "BTCUSDT"
        assert prepared["limit"] == 500
    
    def test_prepare_params_signed(self, client):
        """测试已签名参数准备"""
        params = {"symbol": "BTCUSDT"}
        prepared = client._prepare_params(params, signed=True)
        
        assert "timestamp" in prepared
        assert "signature" in prepared
        assert "recvWindow" in prepared
        assert prepared["symbol"] == "BTCUSDT"
    
    def test_get_headers_unsigned(self, client):
        """测试未签名请求头"""
        headers = client._get_headers(signed=False)
        assert "X-MBX-APIKEY" not in headers
    
    def test_get_headers_signed(self, client):
        """测试已签名请求头"""
        headers = client._get_headers(signed=True)
        assert headers["X-MBX-APIKEY"] == "test_key"
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """测试异步上下文管理器"""
        with patch.object(client, 'connect', new_callable=AsyncMock) as mock_connect:
            with patch.object(client, 'disconnect', new_callable=AsyncMock) as mock_disconnect:
                async with client:
                    pass
                
                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, client):
        """测试连接和断开"""
        assert not client.is_connected
        
        await client.connect()
        assert client.is_connected
        
        await client.disconnect()
        assert not client.is_connected
    
    def test_mainnet_urls(self):
        """测试主网URL配置"""
        client = BinanceFuturesClient(
            api_key="test",
            api_secret="test", 
            testnet=False
        )
        
        assert client.base_url == "https://fapi.binance.com"
        assert client.ws_url == "wss://fstream.binance.com"
        assert client.environment == "mainnet"
    
    def test_repr(self, client):
        """测试字符串表示"""
        repr_str = repr(client)
        assert "BinanceFuturesClient" in repr_str
        assert "testnet=True" in repr_str
        assert "connected=False" in repr_str
    
    # WebSocket 相关测试
    
    def test_websocket_initialization(self, client):
        """测试WebSocket相关属性初始化"""
        assert isinstance(client._ws_connections, dict)
        assert isinstance(client._ws_tasks, set)
        assert isinstance(client._message_handlers, dict)
        assert client._reconnect_delay == 5.0
        assert client._max_reconnect_attempts == 10
    
    def test_get_stream_url(self, client):
        """测试数据流URL生成"""
        stream_name = "btcusdt@ticker"
        expected_url = f"{client.ws_url}/ws/{stream_name}"
        assert client._get_stream_url(stream_name) == expected_url
    
    def test_active_streams_property(self, client):
        """测试活跃数据流属性"""
        assert client.active_streams == []
        
        # 模拟添加连接
        client._ws_connections["btcusdt@ticker"] = MagicMock()
        assert client.active_streams == ["btcusdt@ticker"]
    
    @pytest.mark.asyncio
    async def test_handle_ws_message_with_handler(self, client):
        """测试WebSocket消息处理 - 有处理器"""
        test_data = {"test": "data"}
        handler_called = False
        
        async def test_handler(data):
            nonlocal handler_called
            handler_called = True
            assert data == test_data
        
        client._message_handlers["test_stream"] = test_handler
        await client._handle_ws_message("test_stream", test_data)
        
        assert handler_called
    
    @pytest.mark.asyncio
    async def test_handle_ws_message_without_handler(self, client):
        """测试WebSocket消息处理 - 无处理器"""
        test_data = {"test": "data"}
        
        # 应该不抛出异常
        await client._handle_ws_message("non_exist_stream", test_data)
    
    @pytest.mark.asyncio
    async def test_handle_ws_message_handler_exception(self, client):
        """测试WebSocket消息处理器异常"""
        async def failing_handler(data):
            raise ValueError("Handler error")
        
        client._message_handlers["test_stream"] = failing_handler
        
        # 应该捕获异常而不抛出
        await client._handle_ws_message("test_stream", {"test": "data"})
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, client):
        """测试取消订阅"""
        # 设置模拟连接和处理器
        mock_websocket = AsyncMock()
        stream_name = "btcusdt@ticker"
        
        client._ws_connections[stream_name] = mock_websocket
        client._message_handlers[stream_name] = AsyncMock()
        
        await client.unsubscribe(stream_name)
        
        # 验证连接被关闭
        mock_websocket.close.assert_called_once()
        
        # 验证清理
        assert stream_name not in client._ws_connections
        assert stream_name not in client._message_handlers
    
    @pytest.mark.asyncio
    async def test_close_all_websockets(self, client):
        """测试关闭所有WebSocket连接"""
        # 设置模拟连接和任务
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        mock_task = AsyncMock()
        mock_task.done.return_value = False
        
        client._ws_connections["stream1"] = mock_websocket1
        client._ws_connections["stream2"] = mock_websocket2
        client._ws_tasks.add(mock_task)
        client._message_handlers["stream1"] = AsyncMock()
        
        await client.close_all_websockets()
        
        # 验证任务被取消
        mock_task.cancel.assert_called_once()
        
        # 验证连接被关闭
        mock_websocket1.close.assert_called_once()
        mock_websocket2.close.assert_called_once()
        
        # 验证清理
        assert len(client._ws_connections) == 0
        assert len(client._ws_tasks) == 0
        assert len(client._message_handlers) == 0


class TestBinanceAPIRequests:
    """币安API请求测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    @pytest.mark.asyncio
    async def test_get_server_time(self, client):
        """测试获取服务器时间"""
        mock_response = {"serverTime": 1640995200000}
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_server_time()
            
            assert result == mock_response
            mock_request.assert_called_once_with('GET', '/fapi/v1/time')
    
    @pytest.mark.asyncio
    async def test_get_exchange_info(self, client):
        """测试获取交易规则信息"""
        mock_response = {
            "symbols": [{
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT"
            }]
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_exchange_info()
            
            assert result == mock_response
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_ticker_24hr(self, client):
        """测试24小时价格变动情况"""
        mock_response = {
            "symbol": "BTCUSDT",
            "priceChange": "1000.00",
            "priceChangePercent": "2.50",
            "lastPrice": "41000.00"
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_ticker_24hr("BTCUSDT")
            
            assert result == mock_response
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_depth(self, client):
        """测试获取深度信息"""
        mock_response = {
            "lastUpdateId": 1027024,
            "bids": [["40000.00", "1.5"], ["39999.00", "2.0"]],
            "asks": [["40001.00", "1.0"], ["40002.00", "1.5"]]
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_order_book("BTCUSDT", limit=100)
            
            assert result == mock_response
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_klines(self, client):
        """测试获取K线数据"""
        mock_response = [
            [1640995200000, "40000.00", "41000.00", "39500.00", "40500.00", "1000.50", 1640998800000, "40525000.00", 500, "600.25", "24315000.00", "0"]
        ]
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_klines("BTCUSDT", "1h", limit=1)
            
            assert result == mock_response
            mock_request.assert_called_once()


class TestBinanceTradingAPI:
    """币安交易API测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, client):
        """测试获取账户信息"""
        mock_response = {
            "totalWalletBalance": "10000.00",
            "totalUnrealizedProfit": "0.00",
            "totalMarginBalance": "10000.00",
            "totalPositionInitialMargin": "0.00",
            "totalOpenOrderInitialMargin": "0.00",
            "assets": [{
                "asset": "USDT",
                "walletBalance": "10000.00",
                "unrealizedProfit": "0.00"
            }]
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_account_info()
            
            assert result == mock_response
            mock_request.assert_called_once_with('GET', '/fapi/v2/account', signed=True)
    
    @pytest.mark.asyncio
    async def test_get_balance(self, client):
        """测试获取余额信息"""
        mock_response = [
            {
                "accountAlias": "SgsR",
                "asset": "USDT",
                "balance": "10000.00000000",
                "crossWalletBalance": "10000.00000000",
                "crossUnPnl": "0.00000000",
                "availableBalance": "10000.00000000"
            }
        ]
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_balance()
            
            assert result == mock_response
            mock_request.assert_called_once_with('GET', '/fapi/v2/balance', signed=True)
    
    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """测试获取持仓信息"""
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.001",
                "entryPrice": "40000.00",
                "markPrice": "40100.00",
                "unRealizedProfit": "0.10",
                "liquidationPrice": "0",
                "leverage": "20",
                "maxNotionalValue": "250000",
                "marginType": "cross",
                "isolatedMargin": "0.00000000",
                "isAutoAddMargin": "false",
                "positionSide": "BOTH"
            }
        ]
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_position_info()
            
            assert result == mock_response
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_limit(self, client):
        """测试限价单下单"""
        mock_response = {
            "orderId": 28,
            "symbol": "BTCUSDT",
            "status": "NEW",
            "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
            "price": "40000.00",
            "avgPrice": "0.00",
            "origQty": "0.001",
            "executedQty": "0",
            "cumQuote": "0",
            "timeInForce": "GTC",
            "type": "LIMIT",
            "reduceOnly": False,
            "closePosition": False,
            "side": "BUY",
            "positionSide": "BOTH",
            "stopPrice": "0",
            "workingType": "CONTRACT_PRICE",
            "priceProtect": False,
            "origType": "LIMIT",
            "updateTime": 1640995200000
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.place_order(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                quantity=0.001,
                price=40000.0,
                time_in_force="GTC"
            )
            
            assert result == mock_response
            mock_request.assert_called_once()
            args, kwargs = mock_request.call_args
            assert args[0] == 'POST'
            assert args[1] == '/fapi/v1/order'
            assert kwargs['signed'] is True
    
    @pytest.mark.asyncio
    async def test_place_order_market(self, client):
        """测试市价单下单"""
        mock_response = {
            "orderId": 29,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "price": "0",
            "avgPrice": "40000.00",
            "origQty": "0.001",
            "executedQty": "0.001",
            "type": "MARKET",
            "side": "BUY"
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.place_order(
                symbol="BTCUSDT",
                side="BUY",
                order_type="MARKET",
                quantity=0.001
            )
            
            assert result == mock_response
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """测试撤销订单"""
        mock_response = {
            "clientOrderId": "myOrder1",
            "cumQty": "0",
            "cumQuote": "0",
            "executedQty": "0",
            "orderId": 283194212,
            "origQty": "11",
            "price": "0",
            "reduceOnly": False,
            "side": "BUY",
            "status": "CANCELED",
            "stopPrice": "9300",
            "symbol": "BTCUSDT",
            "timeInForce": "GTC",
            "type": "STOP_MARKET",
            "updateTime": 1571110484038
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.cancel_order("BTCUSDT", order_id=283194212)
            
            assert result == mock_response
            mock_request.assert_called_once_with(
                'DELETE', '/fapi/v1/order',
                params={'symbol': 'BTCUSDT', 'orderId': 283194212},
                signed=True
            )
    
    @pytest.mark.asyncio
    async def test_get_open_orders(self, client):
        """测试获取当前挂单"""
        mock_response = [
            {
                "avgPrice": "0.00000",
                "clientOrderId": "abc",
                "cumQuote": "0",
                "executedQty": "0",
                "orderId": 1917641,
                "origQty": "0.40",
                "price": "4000",
                "reduceOnly": False,
                "side": "BUY",
                "status": "NEW",
                "stopPrice": "0",
                "symbol": "BTCUSDT",
                "time": 1499827319559,
                "timeInForce": "GTC",
                "type": "LIMIT",
                "updateTime": 1499827319559
            }
        ]
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_open_orders("BTCUSDT")
            
            assert result == mock_response
            mock_request.assert_called_once_with(
                'GET', '/fapi/v1/openOrders',
                params={'symbol': 'BTCUSDT'},
                signed=True
            )
    
    @pytest.mark.asyncio
    async def test_get_order(self, client):
        """测试查询订单"""
        mock_response = {
            "avgPrice": "0.00000",
            "clientOrderId": "abc",
            "cumQuote": "0",
            "executedQty": "0",
            "orderId": 1917641,
            "origQty": "0.40",
            "origType": "TRAILING_STOP_MARKET",
            "price": "0",
            "reduceOnly": False,
            "side": "BUY",
            "status": "NEW",
            "stopPrice": "9300",
            "closePosition": False,
            "symbol": "BTCUSDT",
            "time": 1499827319559,
            "timeInForce": "GTC",
            "type": "TRAILING_STOP_MARKET",
            "activatePrice": "9020",
            "priceRate": "0.3",
            "updateTime": 1499827319559,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": False
        }
        
        with patch.object(client, '_request') as mock_request:
            mock_request.return_value = mock_response
            
            result = await client.get_order("BTCUSDT", order_id=1917641)
            
            assert result == mock_response
            mock_request.assert_called_once_with(
                'GET', '/fapi/v1/order',
                params={'symbol': 'BTCUSDT', 'orderId': 1917641},
                signed=True
            )


class TestBinanceWebSocketStability:
    """币安WebSocket稳定性测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    @pytest.mark.asyncio
    async def test_websocket_connection_success(self, client):
        """测试WebSocket连接成功"""
        mock_websocket = AsyncMock()
        stream_name = "btcusdt@ticker"
        
        with patch.object(client, '_ws_connect') as mock_ws_connect:
            mock_ws_connect.return_value = mock_websocket
            
            async def mock_handler(data):
                pass
            
            await client.subscribe_ticker("BTCUSDT", mock_handler)
            
            assert stream_name in client._ws_connections
            assert stream_name in client._message_handlers
            mock_ws_connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_failure(self, client):
        """测试WebSocket连接失败"""
        with patch.object(client, '_ws_connect') as mock_ws_connect:
            mock_ws_connect.side_effect = BinanceWebSocketError("Failed to connect")
            
            async def mock_handler(data):
                pass
            
            with pytest.raises(BinanceWebSocketError, match="Failed to connect"):
                await client.subscribe_ticker("BTCUSDT", mock_handler)
    
    @pytest.mark.asyncio
    async def test_websocket_auto_reconnect(self, client):
        """测试WebSocket自动重连"""
        mock_websocket = AsyncMock()
        stream_name = "btcusdt@ticker"
        reconnect_count = 0
        
        async def mock_recv():
            nonlocal reconnect_count
            if reconnect_count == 0:
                reconnect_count += 1
                raise websockets.ConnectionClosed(None, None)
            else:
                return json.dumps({"stream": stream_name, "data": {"s": "BTCUSDT", "c": "40000"}})
        
        mock_websocket.recv = mock_recv
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            with patch('asyncio.sleep'):
                async def mock_handler(data):
                    pass
                
                await client.subscribe_ticker("BTCUSDT", mock_handler)
                
                # 模拟运行一段时间
                try:
                    await client._listen_to_stream(stream_name)
                except websockets.ConnectionClosed:
                    pass
                
                assert reconnect_count == 1
    
    @pytest.mark.asyncio
    async def test_websocket_max_reconnect_attempts(self, client):
        """测试WebSocket最大重连次数"""
        client._max_reconnect_attempts = 3
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = ConnectionRefusedError("Connection refused")
            
            with patch('asyncio.sleep'):
                async def mock_handler(data):
                    pass
                
                # 模拟连续失败
                with pytest.raises(BinanceWebSocketError, match="Max reconnection attempts"):
                    await client.subscribe_ticker("BTCUSDT", mock_handler)
                
                # 验证连接尝试次数
                assert mock_connect.call_count <= client._max_reconnect_attempts + 1
    
    @pytest.mark.asyncio
    async def test_websocket_message_parsing(self, client):
        """测试WebSocket消息解析"""
        stream_name = "btcusdt@ticker"
        test_message = {
            "stream": stream_name,
            "data": {
                "s": "BTCUSDT",
                "c": "40000.00",
                "o": "39500.00",
                "h": "41000.00",
                "l": "39000.00"
            }
        }
        
        received_data = None
        
        async def test_handler(data):
            nonlocal received_data
            received_data = data
        
        client._message_handlers[stream_name] = test_handler
        
        # 模拟消息处理
        await client._handle_ws_message(stream_name, test_message["data"])
        
        assert received_data == test_message["data"]
    
    @pytest.mark.asyncio
    async def test_websocket_invalid_message(self, client):
        """测试WebSocket无效消息处理"""
        stream_name = "btcusdt@ticker"
        
        async def test_handler(data):
            pass
        
        client._message_handlers[stream_name] = test_handler
        
        # 模拟处理无效消息，不应该抛出异常
        await client._handle_ws_message(stream_name, None)
        await client._handle_ws_message(stream_name, "invalid_json")
    
    @pytest.mark.asyncio
    async def test_multiple_websocket_subscriptions(self, client):
        """测试多个WebSocket订阅"""
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.side_effect = [mock_websocket1.__aenter__(), mock_websocket2.__aenter__()]
            
            async def handler1(data):
                pass
            
            async def handler2(data):
                pass
            
            # 订阅多个数据流
            await client.subscribe_ticker("BTCUSDT", handler1)
            await client.subscribe_depth("ETHUSDT", handler2)
            
            assert "btcusdt@ticker" in client._ws_connections
            assert "ethusdt@depth5@100ms" in client._ws_connections
            assert len(client._ws_connections) == 2
            assert mock_connect.call_count == 2
    
    @pytest.mark.asyncio
    async def test_websocket_graceful_shutdown(self, client):
        """测试WebSocket优雅关闭"""
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        mock_task1 = AsyncMock()
        mock_task2 = AsyncMock()
        
        mock_task1.done.return_value = False
        mock_task2.done.return_value = False
        
        client._ws_connections["stream1"] = mock_websocket1
        client._ws_connections["stream2"] = mock_websocket2
        client._ws_tasks.add(mock_task1)
        client._ws_tasks.add(mock_task2)
        
        # 测试关闭所有连接
        await client.close_all_websockets()
        
        # 验证所有连接都被关闭
        mock_websocket1.close.assert_called_once()
        mock_websocket2.close.assert_called_once()
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()
        
        # 验证数据结构清理
        assert len(client._ws_connections) == 0
        assert len(client._ws_tasks) == 0


class TestBinanceErrorHandling:
    """币安错误处理测试"""
    
    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        return BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
    
    @pytest.mark.asyncio
    async def test_http_400_error(self, client):
        """测试HTTP 400错误处理"""
        mock_response = {
            "code": -1100,
            "msg": "Illegal characters found in parameter 'symbol'; legal range is '^[A-Z0-9-_.]{1,20}$'."
        }
        
        with patch.object(client, '_request') as mock_request:
            api_error = BinanceAPIError(-1100, "Illegal characters found in parameter 'symbol'; legal range is '^[A-Z0-9-_.]{1,20}$'.")
            mock_request.side_effect = api_error
            
            with pytest.raises(BinanceAPIError) as exc_info:
                await client.place_order(
                    symbol="INVALID@SYMBOL",
                    side="BUY",
                    order_type="MARKET",
                    quantity=0.001
                )
            
            assert exc_info.value.code == -1100
            assert "Illegal characters" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_http_403_error(self, client):
        """测试HTTP 403权限错误"""
        mock_response = {
            "code": -2014,
            "msg": "API-key format invalid."
        }
        
        with patch.object(client, '_request') as mock_request:
            api_error = BinanceAPIError(-2014, "API-key format invalid.")
            mock_request.side_effect = api_error
            
            with pytest.raises(BinanceAPIError) as exc_info:
                await client.get_account_info()
            
            assert exc_info.value.code == -2014
            assert "API-key format invalid" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_http_429_rate_limit(self, client):
        """测试HTTP 429限流错误"""
        mock_response = {
            "code": -1003,
            "msg": "Too much request weight used; current limit is 1200 request weight per 1 MINUTE."
        }
        
        with patch.object(client, '_request') as mock_request:
            api_error = BinanceAPIError(-1003, "Too much request weight used; current limit is 1200 request weight per 1 MINUTE.")
            mock_request.side_effect = api_error
            
            with pytest.raises(BinanceAPIError) as exc_info:
                await client.get_ticker_24hr("BTCUSDT")
            
            assert exc_info.value.code == -1003
            assert "Too much request weight" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_http_500_server_error(self, client):
        """测试HTTP 500服务器错误"""
        with patch.object(client, '_request') as mock_request:
            api_error = BinanceAPIError(None, "HTTP 500: Internal Server Error")
            mock_request.side_effect = api_error
            
            with pytest.raises(BinanceAPIError) as exc_info:
                await client.get_server_time()
            
            assert exc_info.value.code is None
            assert "HTTP 500" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_connection_error(self, client):
        """测试连接错误"""
        with patch.object(client, '_request') as mock_request:
            connection_error = BinanceConnectionError("Connection refused")
            mock_request.side_effect = connection_error
            
            with pytest.raises(BinanceConnectionError):
                await client.get_server_time()
    
    @pytest.mark.asyncio
    async def test_timeout_error(self, client):
        """测试超时错误"""
        with patch.object(client, '_request') as mock_request:
            timeout_error = BinanceConnectionError("Request timeout")
            mock_request.side_effect = timeout_error
            
            with pytest.raises(BinanceConnectionError):
                await client.get_server_time()
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_success(self, client):
        """测试重试机制成功"""
        mock_success_response = {"serverTime": 1640995200000}
        call_count = 0
        
        async def mock_get_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # 前两次失败
                raise aiohttp.ClientConnectorError(
                    connection_key="connection", 
                    os_error=OSError("Connection refused")
                )
            else:  # 第三次成功
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json.return_value = mock_success_response
                return mock_response
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = mock_get_side_effect
            
            result = await client.get_server_time()
            
            assert result == mock_success_response
            assert call_count == 3  # 确保重试了2次
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_failure(self, client):
        """测试重试机制最终失败"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientConnectorError(
                connection_key="connection", 
                os_error=OSError("Connection refused")
            )
            
            with pytest.raises(BinanceConnectionError):
                await client.get_server_time()
            
            # 验证重试次数（初始请求 + 重试次数）
            assert mock_get.call_count > 1
    
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, client):
        """测试无效JSON响应"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.json.side_effect = json.JSONDecodeError(
                "Expecting value", "invalid json", 0
            )
            mock_get.return_value.__aenter__.return_value.text.return_value = "invalid json"
            
            with pytest.raises(BinanceAPIError, match="Invalid JSON response"):
                await client.get_server_time()


class TestBinanceMockEnvironment:
    """币安Mock测试环境"""
    
    def test_mock_client_creation(self):
        """测试Mock客户端创建"""
        client = BinanceFuturesClient(
            api_key="mock_key",
            api_secret="mock_secret",
            testnet=True
        )
        
        assert client.api_key == "mock_key"
        assert client.api_secret == "mock_secret"
        assert client.testnet is True
        assert "testnet" in client.base_url
    
    @pytest.mark.asyncio
    async def test_mock_trading_flow(self):
        """测试完整的模拟交易流程"""
        client = BinanceFuturesClient(
            api_key="mock_key",
            api_secret="mock_secret",
            testnet=True
        )
        
        # Mock所有需要的API调用
        with patch.object(client, 'get_account_info') as mock_account, \
             patch.object(client, 'get_positions') as mock_positions, \
             patch.object(client, 'place_order') as mock_place_order, \
             patch.object(client, 'get_open_orders') as mock_open_orders, \
             patch.object(client, 'cancel_order') as mock_cancel_order:
            
            # 设置Mock返回值
            mock_account.return_value = {
                "totalWalletBalance": "10000.00",
                "totalUnrealizedProfit": "0.00"
            }
            
            mock_positions.return_value = [
                {
                    "symbol": "BTCUSDT",
                    "positionAmt": "0.000",
                    "entryPrice": "0.00000",
                    "markPrice": "40000.00000",
                    "unRealizedProfit": "0.00000000",
                    "positionSide": "BOTH"
                }
            ]
            
            mock_place_order.return_value = {
                "orderId": 123456,
                "symbol": "BTCUSDT",
                "status": "NEW",
                "clientOrderId": "test_order",
                "price": "40000.00",
                "origQty": "0.001",
                "type": "LIMIT",
                "side": "BUY"
            }
            
            mock_open_orders.return_value = [
                {
                    "orderId": 123456,
                    "symbol": "BTCUSDT",
                    "status": "NEW",
                    "price": "40000.00",
                    "origQty": "0.001"
                }
            ]
            
            mock_cancel_order.return_value = {
                "orderId": 123456,
                "symbol": "BTCUSDT",
                "status": "CANCELED"
            }
            
            # 执行完整的交易流程测试
            # 1. 获取账户信息
            account = await client.get_account_info()
            assert account["totalWalletBalance"] == "10000.00"
            
            # 2. 获取持仓信息
            positions = await client.get_positions()
            assert len(positions) == 1
            assert positions[0]["symbol"] == "BTCUSDT"
            
            # 3. 下单
            order = await client.place_order(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                quantity=0.001,
                price=40000.0
            )
            assert order["orderId"] == 123456
            assert order["status"] == "NEW"
            
            # 4. 查询挂单
            open_orders = await client.get_open_orders("BTCUSDT")
            assert len(open_orders) == 1
            assert open_orders[0]["orderId"] == 123456
            
            # 5. 撤单
            cancel_result = await client.cancel_order("BTCUSDT", order_id=123456)
            assert cancel_result["orderId"] == 123456
            assert cancel_result["status"] == "CANCELED"
            
            # 验证所有方法都被调用
            mock_account.assert_called_once()
            mock_positions.assert_called_once()
            mock_place_order.assert_called_once()
            mock_open_orders.assert_called_once()
            mock_cancel_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mock_websocket_data_flow(self):
        """测试WebSocket数据流Mock"""
        client = BinanceFuturesClient(
            api_key="mock_key",
            api_secret="mock_secret",
            testnet=True
        )
        
        received_data = []
        
        async def mock_handler(data):
            received_data.append(data)
        
        # Mock WebSocket连接和消息
        mock_websocket = AsyncMock()
        mock_messages = [
            json.dumps({
                "stream": "btcusdt@ticker",
                "data": {
                    "s": "BTCUSDT",
                    "c": "40000.00",
                    "o": "39500.00",
                    "h": "41000.00",
                    "l": "39000.00",
                    "v": "1000.50"
                }
            }),
            json.dumps({
                "stream": "btcusdt@ticker",
                "data": {
                    "s": "BTCUSDT",
                    "c": "40100.00",
                    "o": "40000.00",
                    "h": "41200.00",
                    "l": "39800.00",
                    "v": "1200.75"
                }
            })
        ]
        
        message_iter = iter(mock_messages)
        
        async def mock_recv():
            return next(message_iter)
        
        mock_websocket.recv = mock_recv
        
        with patch('websockets.connect') as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            # 订阅ticker数据
            await client.subscribe_ticker("BTCUSDT", mock_handler)
            
            # 手动处理消息来验证数据流
            for mock_message in mock_messages:
                message_data = json.loads(mock_message)
                await client._handle_ws_message(
                    message_data["stream"], 
                    message_data["data"]
                )
            
            # 验证接收到的数据
            assert len(received_data) == 2
            assert received_data[0]["s"] == "BTCUSDT"
            assert received_data[0]["c"] == "40000.00"
            assert received_data[1]["c"] == "40100.00"