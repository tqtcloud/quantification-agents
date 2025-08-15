from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.models import Order, Position, OrderSide, OrderType, OrderStatus, PositionSide
from src.utils.logger import LoggerMixin


class TradingEnvironment(str, Enum):
    """交易环境枚举"""
    TESTNET = "testnet"  # 模拟盘
    MAINNET = "mainnet"  # 实盘
    PAPER = "paper"      # 虚拟盘


@dataclass
class TradingContext:
    """交易上下文"""
    environment: TradingEnvironment
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TradingInterface(ABC, LoggerMixin):
    """统一交易接口抽象基类"""
    
    def __init__(self, environment: TradingEnvironment):
        self.environment = environment
        self.context = TradingContext(
            environment=environment,
            session_id=f"{environment.value}_{datetime.utcnow().isoformat()}"
        )
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接交易接口"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开交易接口"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """下单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """撤单"""
        pass
    
    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """查询订单"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取当前挂单"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        pass
    
    @property
    def is_testnet(self) -> bool:
        """是否为测试环境"""
        return self.environment == TradingEnvironment.TESTNET
    
    @property
    def is_mainnet(self) -> bool:
        """是否为主网环境"""
        return self.environment == TradingEnvironment.MAINNET
    
    @property
    def is_paper(self) -> bool:
        """是否为虚拟盘"""
        return self.environment == TradingEnvironment.PAPER
    
    def get_data_prefix(self) -> str:
        """获取数据前缀，用于数据隔离"""
        return f"{self.environment.value}_"
    
    def tag_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """为订单数据添加环境标签"""
        order_data['trading_environment'] = self.environment.value
        order_data['session_id'] = self.context.session_id
        return order_data
    
    def tag_position(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """为持仓数据添加环境标签"""
        position_data['trading_environment'] = self.environment.value
        position_data['session_id'] = self.context.session_id
        return position_data


class BinanceTradingInterface(TradingInterface):
    """币安交易接口实现"""
    
    def __init__(self, client, environment: TradingEnvironment):
        super().__init__(environment)
        self.client = client
        
        # 确保客户端环境与接口环境一致
        if environment == TradingEnvironment.TESTNET:
            if not client.testnet:
                raise ValueError("Client must be configured for testnet when using TESTNET environment")
        elif environment == TradingEnvironment.MAINNET:
            if client.testnet:
                raise ValueError("Client must be configured for mainnet when using MAINNET environment")
    
    async def connect(self) -> bool:
        """连接交易接口"""
        try:
            await self.client.connect()
            # 测试连接
            await self.client.ping()
            self.log_info(f"Connected to Binance {self.environment.value}")
            return True
        except Exception as e:
            self.log_error(f"Failed to connect to Binance {self.environment.value}", error=str(e))
            return False
    
    async def disconnect(self):
        """断开交易接口"""
        await self.client.disconnect()
        await self.client.close_all_websockets()
        self.log_info(f"Disconnected from Binance {self.environment.value}")
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """下单"""
        try:
            result = await self.client.place_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force.value,
                reduce_only=order.reduce_only,
                position_side=order.position_side.value,
                client_order_id=order.client_order_id
            )
            
            # 添加环境标签
            result = self.tag_order(result)
            
            self.log_info(
                f"Order placed in {self.environment.value}",
                symbol=order.symbol,
                side=order.side.value,
                order_id=result.get('orderId')
            )
            
            return result
            
        except Exception as e:
            self.log_error(
                f"Failed to place order in {self.environment.value}",
                symbol=order.symbol,
                error=str(e)
            )
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """撤单"""
        try:
            result = await self.client.cancel_order(symbol, order_id=int(order_id))
            result = self.tag_order(result)
            
            self.log_info(
                f"Order cancelled in {self.environment.value}",
                symbol=symbol,
                order_id=order_id
            )
            
            return result
            
        except Exception as e:
            self.log_error(
                f"Failed to cancel order in {self.environment.value}",
                symbol=symbol,
                order_id=order_id,
                error=str(e)
            )
            raise
    
    async def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """查询订单"""
        try:
            result = await self.client.get_order(symbol, order_id=int(order_id))
            return self.tag_order(result)
        except Exception as e:
            self.log_error(
                f"Failed to get order in {self.environment.value}",
                symbol=symbol,
                order_id=order_id,
                error=str(e)
            )
            raise
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取当前挂单"""
        try:
            result = await self.client.get_open_orders(symbol)
            # 如果result是单个订单字典，转换为列表
            if isinstance(result, dict) and 'orderId' in result:
                result = [result]
            # 为每个订单添加环境标签
            return [self.tag_order(order) for order in result]
        except Exception as e:
            self.log_error(
                f"Failed to get open orders in {self.environment.value}",
                symbol=symbol,
                error=str(e)
            )
            raise
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        try:
            result = await self.client.get_position_info(symbol)
            # 为每个持仓添加环境标签
            return [self.tag_position(pos) for pos in result]
        except Exception as e:
            self.log_error(
                f"Failed to get positions in {self.environment.value}",
                symbol=symbol,
                error=str(e)
            )
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        try:
            result = await self.client.get_account_info()
            result['trading_environment'] = self.environment.value
            result['session_id'] = self.context.session_id
            return result
        except Exception as e:
            self.log_error(f"Failed to get account info in {self.environment.value}", error=str(e))
            raise
    
    async def get_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        try:
            result = await self.client.get_balance()
            # 为余额数据添加环境标签
            for balance in result:
                balance['trading_environment'] = self.environment.value
                balance['session_id'] = self.context.session_id
            return result
        except Exception as e:
            self.log_error(f"Failed to get balance in {self.environment.value}", error=str(e))
            raise