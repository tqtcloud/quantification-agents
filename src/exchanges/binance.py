import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlencode

import aiohttp
import websockets
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.logger import LoggerMixin


class BinanceAPIError(Exception):
    """币安API错误"""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error {code}: {message}")


class BinanceConnectionError(Exception):
    """币安连接错误"""
    pass


class BinanceWebSocketError(Exception):
    """币安WebSocket错误"""
    pass


class BinanceFuturesClient(LoggerMixin):
    """币安期货客户端 - 支持testnet和mainnet"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: Optional[bool] = None,
    ):
        self.api_key = api_key or settings.binance_api_key
        self.api_secret = api_secret or settings.binance_api_secret
        self.testnet = testnet if testnet is not None else settings.binance_testnet
        
        # 根据模式选择基础URL
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"
        
        self._session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket相关属性
        self._ws_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._ws_tasks: Set[asyncio.Task] = set()
        self._message_handlers: Dict[str, Callable] = {}
        self._reconnect_delay: float = 5.0
        self._max_reconnect_attempts: int = 10
        
        # 验证API凭证
        if not self.api_key or not self.api_secret:
            self.log_warning("Missing API credentials - only public endpoints will be available")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.disconnect()
        await self.close_all_websockets()
    
    async def connect(self):
        """创建HTTP会话"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "crypto-quant-trading/0.1.0",
                }
            )
            self.log_info("Connected to Binance API", testnet=self.testnet)
    
    async def disconnect(self):
        """关闭HTTP会话"""
        if self._session:
            await self._session.close()
            self._session = None
            self.log_info("Disconnected from Binance API")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """生成API请求签名"""
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _prepare_params(self, params: Dict[str, Any], signed: bool = False) -> Dict[str, Any]:
        """准备请求参数"""
        # 移除None值
        params = {k: v for k, v in params.items() if v is not None}
        
        if signed:
            # 添加时间戳
            params['timestamp'] = int(time.time() * 1000)
            
            # 添加接收窗口（可选）
            if 'recvWindow' not in params:
                params['recvWindow'] = 60000
            
            # 生成签名
            params['signature'] = self._generate_signature(params)
        
        return params
    
    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """获取请求头"""
        headers = {}
        
        if signed and self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        return headers
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
        retry_count: int = 3,
    ) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self._session:
            await self.connect()
        
        params = params or {}
        params = self._prepare_params(params, signed)
        headers = self._get_headers(signed)
        
        url = f"{self.base_url}{endpoint}"
        
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(retry_count),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((aiohttp.ClientError, BinanceConnectionError)),
            reraise=True,
        ):
            with attempt:
                try:
                    self.log_debug(f"Making {method} request", url=url, params=params)
                    
                    if method.upper() == 'GET':
                        async with self._session.get(url, params=params, headers=headers) as response:
                            return await self._handle_response(response)
                    elif method.upper() == 'POST':
                        async with self._session.post(url, data=params, headers=headers) as response:
                            return await self._handle_response(response)
                    elif method.upper() == 'PUT':
                        async with self._session.put(url, data=params, headers=headers) as response:
                            return await self._handle_response(response)
                    elif method.upper() == 'DELETE':
                        async with self._session.delete(url, params=params, headers=headers) as response:
                            return await self._handle_response(response)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                
                except aiohttp.ClientError as e:
                    self.log_error(f"HTTP request failed", error=str(e), attempt=attempt.retry_state.attempt_number)
                    raise BinanceConnectionError(f"Connection error: {e}") from e
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """处理API响应"""
        try:
            data = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            text = await response.text()
            self.log_error("Failed to decode JSON response", status=response.status, text=text)
            raise BinanceAPIError(-1, f"Invalid JSON response: {text}")
        
        if response.status == 200:
            return data
        
        # 处理错误响应
        error_code = data.get('code', response.status)
        error_msg = data.get('msg', f'HTTP {response.status}')
        
        self.log_error("API request failed", code=error_code, message=error_msg)
        
        # 根据错误代码进行分类处理
        if response.status == 429:
            raise BinanceAPIError(429, "Rate limit exceeded")
        elif response.status in (401, 403):
            raise BinanceAPIError(error_code, f"Authentication failed: {error_msg}")
        elif response.status >= 500:
            raise BinanceConnectionError(f"Server error: {error_msg}")
        else:
            raise BinanceAPIError(error_code, error_msg)
    
    # Public API endpoints
    
    async def ping(self) -> Dict[str, Any]:
        """测试连接"""
        return await self._request('GET', '/fapi/v1/ping')
    
    async def get_server_time(self) -> Dict[str, Any]:
        """获取服务器时间"""
        return await self._request('GET', '/fapi/v1/time')
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """获取交易规则和交易对信息"""
        return await self._request('GET', '/fapi/v1/exchangeInfo')
    
    async def get_ticker_24hr(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取24小时价格变动统计"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v1/ticker/24hr', params)
    
    async def get_ticker_price(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取最新价格"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v1/ticker/price', params)
    
    async def get_order_book(self, symbol: str, limit: int = 500) -> Dict[str, Any]:
        """获取订单簿"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._request('GET', '/fapi/v1/depth', params)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> Dict[str, Any]:
        """获取K线数据"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return await self._request('GET', '/fapi/v1/klines', params)
    
    # Private API endpoints (需要API密钥)
    
    # 交易接口
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        position_side: str = "BOTH",
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """下单"""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity),
            'timeInForce': time_in_force,
            'reduceOnly': str(reduce_only).lower(),
            'positionSide': position_side
        }
        
        if price is not None:
            params['price'] = str(price)
        if stop_price is not None:
            params['stopPrice'] = str(stop_price)
        if client_order_id:
            params['newClientOrderId'] = client_order_id
        
        return await self._request('POST', '/fapi/v1/order', params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """撤单"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = str(order_id)
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Must provide either order_id or client_order_id")
        
        return await self._request('DELETE', '/fapi/v1/order', params, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """撤销所有订单"""
        params = {'symbol': symbol}
        return await self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)
    
    async def get_order(self, symbol: str, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """查询订单"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = str(order_id)
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Must provide either order_id or client_order_id")
        
        return await self._request('GET', '/fapi/v1/order', params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取当前挂单"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v1/openOrders', params, signed=True)
    
    async def get_all_orders(self, symbol: str, limit: int = 500) -> Dict[str, Any]:
        """获取所有订单"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._request('GET', '/fapi/v1/allOrders', params, signed=True)
    
    async def modify_order(
        self,
        symbol: str,
        order_id: int,
        side: str,
        quantity: float,
        price: float
    ) -> Dict[str, Any]:
        """修改订单"""
        params = {
            'symbol': symbol,
            'orderId': str(order_id),
            'side': side.upper(),
            'quantity': str(quantity),
            'price': str(price)
        }
        return await self._request('PUT', '/fapi/v1/order', params, signed=True)
    
    # 账户和仓位管理
    
    async def change_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """调整杠杆倍数"""
        params = {'symbol': symbol, 'leverage': str(leverage)}
        return await self._request('POST', '/fapi/v1/leverage', params, signed=True)
    
    async def change_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """调整保证金模式"""
        params = {'symbol': symbol, 'marginType': margin_type.upper()}
        return await self._request('POST', '/fapi/v1/marginType', params, signed=True)
    
    async def modify_position_margin(self, symbol: str, amount: float, position_side: str = "BOTH") -> Dict[str, Any]:
        """调整逐仓保证金"""
        params = {
            'symbol': symbol,
            'amount': str(amount),
            'type': 1,  # 1: 增加保证金, 2: 减少保证金
            'positionSide': position_side
        }
        return await self._request('POST', '/fapi/v1/positionMargin', params, signed=True)
    
    async def get_position_margin_history(self, symbol: str, limit: int = 500) -> Dict[str, Any]:
        """获取调整保证金历史"""
        params = {'symbol': symbol, 'limit': limit}
        return await self._request('GET', '/fapi/v1/positionMargin/history', params, signed=True)
    
    async def get_income_history(self, symbol: Optional[str] = None, income_type: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """获取收益历史"""
        params = {'limit': limit}
        if symbol:
            params['symbol'] = symbol
        if income_type:
            params['incomeType'] = income_type
        return await self._request('GET', '/fapi/v1/income', params, signed=True)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        return await self._request('GET', '/fapi/v2/account', signed=True)
    
    async def get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        return await self._request('GET', '/fapi/v2/balance', signed=True)
    
    async def get_position_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """获取持仓信息"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v2/positionRisk', params, signed=True)
    
    # WebSocket 相关方法
    
    async def _ws_connect(self, stream_name: str, url: str) -> websockets.WebSocketServerProtocol:
        """创建WebSocket连接"""
        try:
            self.log_debug(f"Connecting to WebSocket", stream=stream_name, url=url)
            websocket = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self._ws_connections[stream_name] = websocket
            self.log_info(f"WebSocket connected", stream=stream_name)
            return websocket
        except Exception as e:
            self.log_error(f"Failed to connect WebSocket", stream=stream_name, error=str(e))
            raise BinanceWebSocketError(f"WebSocket connection failed: {e}") from e
    
    async def _ws_listen(self, stream_name: str, websocket: websockets.WebSocketServerProtocol):
        """监听WebSocket消息"""
        reconnect_attempts = 0
        
        while reconnect_attempts < self._max_reconnect_attempts:
            try:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_ws_message(stream_name, data)
                    except json.JSONDecodeError as e:
                        self.log_warning(f"Failed to parse WebSocket message", stream=stream_name, error=str(e))
                    except Exception as e:
                        self.log_error(f"Error handling WebSocket message", stream=stream_name, error=str(e))
                
                # 连接正常关闭，重置重连计数
                break
                
            except websockets.exceptions.ConnectionClosed as e:
                reconnect_attempts += 1
                self.log_warning(
                    f"WebSocket connection closed, attempting reconnect",
                    stream=stream_name,
                    attempt=reconnect_attempts,
                    reason=str(e)
                )
                
                if reconnect_attempts < self._max_reconnect_attempts:
                    await asyncio.sleep(self._reconnect_delay)
                    try:
                        # 重新连接
                        url = self._get_stream_url(stream_name)
                        websocket = await self._ws_connect(stream_name, url)
                    except Exception as reconnect_error:
                        self.log_error(
                            f"Failed to reconnect WebSocket",
                            stream=stream_name,
                            error=str(reconnect_error)
                        )
                else:
                    self.log_error(
                        f"Max reconnection attempts reached for WebSocket",
                        stream=stream_name
                    )
                    break
                    
            except Exception as e:
                self.log_error(f"Unexpected WebSocket error", stream=stream_name, error=str(e))
                break
        
        # 清理连接
        if stream_name in self._ws_connections:
            del self._ws_connections[stream_name]
    
    async def _handle_ws_message(self, stream_name: str, data: Dict[str, Any]):
        """处理WebSocket消息"""
        if stream_name in self._message_handlers:
            try:
                await self._message_handlers[stream_name](data)
            except Exception as e:
                self.log_error(f"Error in message handler", stream=stream_name, error=str(e))
        else:
            self.log_debug(f"No handler for stream", stream=stream_name, data=data)
    
    def _get_stream_url(self, stream_name: str) -> str:
        """获取数据流URL"""
        return f"{self.ws_url}/ws/{stream_name}"
    
    async def subscribe_ticker(self, symbol: str, callback: Callable[[Dict[str, Any]], None]):
        """订阅ticker数据流"""
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@ticker"
        url = self._get_stream_url(stream_name)
        
        self._message_handlers[stream_name] = callback
        
        try:
            websocket = await self._ws_connect(stream_name, url)
            task = asyncio.create_task(self._ws_listen(stream_name, websocket))
            self._ws_tasks.add(task)
            task.add_done_callback(self._ws_tasks.discard)
            
            self.log_info(f"Subscribed to ticker stream", symbol=symbol)
            return task
            
        except Exception as e:
            self.log_error(f"Failed to subscribe to ticker", symbol=symbol, error=str(e))
            raise
    
    async def subscribe_depth(
        self, 
        symbol: str, 
        callback: Callable[[Dict[str, Any]], None],
        speed: str = "100ms"
    ):
        """订阅深度/订单簿数据流"""
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@depth@{speed}"
        url = self._get_stream_url(stream_name)
        
        self._message_handlers[stream_name] = callback
        
        try:
            websocket = await self._ws_connect(stream_name, url)
            task = asyncio.create_task(self._ws_listen(stream_name, websocket))
            self._ws_tasks.add(task)
            task.add_done_callback(self._ws_tasks.discard)
            
            self.log_info(f"Subscribed to depth stream", symbol=symbol, speed=speed)
            return task
            
        except Exception as e:
            self.log_error(f"Failed to subscribe to depth", symbol=symbol, error=str(e))
            raise
    
    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """订阅K线数据流"""
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@kline_{interval}"
        url = self._get_stream_url(stream_name)
        
        self._message_handlers[stream_name] = callback
        
        try:
            websocket = await self._ws_connect(stream_name, url)
            task = asyncio.create_task(self._ws_listen(stream_name, websocket))
            self._ws_tasks.add(task)
            task.add_done_callback(self._ws_tasks.discard)
            
            self.log_info(f"Subscribed to kline stream", symbol=symbol, interval=interval)
            return task
            
        except Exception as e:
            self.log_error(f"Failed to subscribe to kline", symbol=symbol, error=str(e))
            raise
    
    async def unsubscribe(self, stream_name: str):
        """取消订阅数据流"""
        if stream_name in self._ws_connections:
            websocket = self._ws_connections[stream_name]
            await websocket.close()
            del self._ws_connections[stream_name]
            
        if stream_name in self._message_handlers:
            del self._message_handlers[stream_name]
            
        self.log_info(f"Unsubscribed from stream", stream=stream_name)
    
    async def close_all_websockets(self):
        """关闭所有WebSocket连接"""
        # 取消所有任务
        for task in self._ws_tasks:
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        if self._ws_tasks:
            await asyncio.gather(*self._ws_tasks, return_exceptions=True)
            self._ws_tasks.clear()
        
        # 关闭所有WebSocket连接
        for stream_name, websocket in list(self._ws_connections.items()):
            try:
                await websocket.close()
            except Exception as e:
                self.log_warning(f"Error closing WebSocket", stream=stream_name, error=str(e))
        
        self._ws_connections.clear()
        self._message_handlers.clear()
        self.log_info("All WebSocket connections closed")
    
    @property
    def active_streams(self) -> List[str]:
        """获取活跃的数据流列表"""
        return list(self._ws_connections.keys())
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._session is not None and not self._session.closed
    
    @property 
    def environment(self) -> str:
        """获取当前环境"""
        return "testnet" if self.testnet else "mainnet"
    
    def __repr__(self) -> str:
        return f"BinanceFuturesClient(testnet={self.testnet}, connected={self.is_connected})"