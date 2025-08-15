from typing import Dict, Optional, List, Any
import asyncio
from contextlib import asynccontextmanager

from src.config import settings
from src.exchanges.binance import BinanceFuturesClient
from src.exchanges.trading_interface import (
    TradingInterface, 
    BinanceTradingInterface, 
    TradingEnvironment
)
from src.core.models import Order
from src.utils.logger import LoggerMixin


class TradingManager(LoggerMixin):
    """交易管理器 - 管理多环境交易接口和数据隔离"""
    
    def __init__(self):
        self._interfaces: Dict[TradingEnvironment, TradingInterface] = {}
        self._current_environment: Optional[TradingEnvironment] = None
        self._data_isolation_enabled = True
    
    def register_interface(self, environment: TradingEnvironment, interface: TradingInterface):
        """注册交易接口"""
        self._interfaces[environment] = interface
        self.log_info(f"Registered trading interface for {environment.value}")
    
    async def initialize_binance_interfaces(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None
    ):
        """初始化币安交易接口"""
        api_key = api_key or settings.binance_api_key
        api_secret = api_secret or settings.binance_api_secret
        
        # 初始化testnet接口
        testnet_client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True
        )
        testnet_interface = BinanceTradingInterface(testnet_client, TradingEnvironment.TESTNET)
        self.register_interface(TradingEnvironment.TESTNET, testnet_interface)
        
        # 初始化mainnet接口
        mainnet_client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=False
        )
        mainnet_interface = BinanceTradingInterface(mainnet_client, TradingEnvironment.MAINNET)
        self.register_interface(TradingEnvironment.MAINNET, mainnet_interface)
        
        self.log_info("Initialized Binance trading interfaces")
    
    def set_environment(self, environment: TradingEnvironment):
        """设置当前交易环境"""
        if environment not in self._interfaces:
            raise ValueError(f"No interface registered for environment: {environment}")
        
        old_env = self._current_environment
        self._current_environment = environment
        
        self.log_info(
            f"Switched trading environment",
            from_env=old_env.value if old_env else None,
            to_env=environment.value
        )
    
    def get_current_environment(self) -> Optional[TradingEnvironment]:
        """获取当前交易环境"""
        return self._current_environment
    
    def get_interface(self, environment: Optional[TradingEnvironment] = None) -> TradingInterface:
        """获取交易接口"""
        target_env = environment or self._current_environment
        
        if not target_env:
            raise ValueError("No current environment set and no environment specified")
        
        if target_env not in self._interfaces:
            raise ValueError(f"No interface registered for environment: {target_env}")
        
        return self._interfaces[target_env]
    
    @asynccontextmanager
    async def use_environment(self, environment: TradingEnvironment):
        """临时切换环境的上下文管理器"""
        old_env = self._current_environment
        self.set_environment(environment)
        
        interface = self.get_interface()
        connected = await interface.connect()
        
        if not connected:
            raise RuntimeError(f"Failed to connect to {environment.value}")
        
        try:
            yield interface
        finally:
            await interface.disconnect()
            if old_env:
                self.set_environment(old_env)
            else:
                self._current_environment = None
    
    async def connect_environment(self, environment: TradingEnvironment) -> bool:
        """连接指定环境"""
        interface = self.get_interface(environment)
        return await interface.connect()
    
    async def disconnect_environment(self, environment: TradingEnvironment):
        """断开指定环境连接"""
        interface = self.get_interface(environment)
        await interface.disconnect()
    
    async def connect_all(self) -> Dict[TradingEnvironment, bool]:
        """连接所有环境"""
        results = {}
        for env in self._interfaces.keys():
            try:
                results[env] = await self.connect_environment(env)
            except Exception as e:
                self.log_error(f"Failed to connect {env.value}", error=str(e))
                results[env] = False
        return results
    
    async def disconnect_all(self):
        """断开所有环境连接"""
        for env in self._interfaces.keys():
            try:
                await self.disconnect_environment(env)
            except Exception as e:
                self.log_warning(f"Error disconnecting {env.value}", error=str(e))
    
    # 数据隔离相关方法
    
    def get_isolated_key(self, base_key: str, environment: Optional[TradingEnvironment] = None) -> str:
        """获取环境隔离的数据键"""
        if not self._data_isolation_enabled:
            return base_key
        
        env = environment or self._current_environment
        if not env:
            return base_key
        
        return f"{env.value}:{base_key}"
    
    def extract_base_key(self, isolated_key: str) -> tuple[str, Optional[TradingEnvironment]]:
        """从隔离键中提取基础键和环境"""
        if ":" not in isolated_key:
            return isolated_key, None
        
        env_str, base_key = isolated_key.split(":", 1)
        try:
            environment = TradingEnvironment(env_str)
            return base_key, environment
        except ValueError:
            return isolated_key, None
    
    def filter_data_by_environment(
        self, 
        data: List[Dict[str, Any]], 
        environment: Optional[TradingEnvironment] = None
    ) -> List[Dict[str, Any]]:
        """按环境过滤数据"""
        target_env = environment or self._current_environment
        if not target_env or not self._data_isolation_enabled:
            return data
        
        return [
            item for item in data 
            if item.get('trading_environment') == target_env.value
        ]
    
    # 交易接口代理方法
    
    async def place_order(self, order: Order, environment: Optional[TradingEnvironment] = None) -> Dict[str, Any]:
        """下单"""
        interface = self.get_interface(environment)
        return await interface.place_order(order)
    
    async def cancel_order(
        self, 
        symbol: str, 
        order_id: str, 
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, Any]:
        """撤单"""
        interface = self.get_interface(environment)
        return await interface.cancel_order(symbol, order_id)
    
    async def get_order(
        self, 
        symbol: str, 
        order_id: str, 
        environment: Optional[TradingEnvironment] = None
    ) -> Dict[str, Any]:
        """查询订单"""
        interface = self.get_interface(environment)
        return await interface.get_order(symbol, order_id)
    
    async def get_open_orders(
        self, 
        symbol: Optional[str] = None, 
        environment: Optional[TradingEnvironment] = None
    ) -> List[Dict[str, Any]]:
        """获取当前挂单"""
        interface = self.get_interface(environment)
        orders = await interface.get_open_orders(symbol)
        return self.filter_data_by_environment(orders, environment)
    
    async def get_positions(
        self, 
        symbol: Optional[str] = None, 
        environment: Optional[TradingEnvironment] = None
    ) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        interface = self.get_interface(environment)
        positions = await interface.get_positions(symbol)
        return self.filter_data_by_environment(positions, environment)
    
    async def get_account_info(self, environment: Optional[TradingEnvironment] = None) -> Dict[str, Any]:
        """获取账户信息"""
        interface = self.get_interface(environment)
        return await interface.get_account_info()
    
    async def get_balance(self, environment: Optional[TradingEnvironment] = None) -> List[Dict[str, Any]]:
        """获取账户余额"""
        interface = self.get_interface(environment)
        balances = await interface.get_balance()
        return self.filter_data_by_environment(balances, environment)
    
    # 跨环境操作
    
    async def compare_environments(
        self, 
        operation: str, 
        *args, 
        environments: Optional[List[TradingEnvironment]] = None,
        **kwargs
    ) -> Dict[TradingEnvironment, Any]:
        """比较不同环境的操作结果"""
        if not environments:
            environments = list(self._interfaces.keys())
        
        results = {}
        
        for env in environments:
            try:
                interface = self.get_interface(env)
                method = getattr(interface, operation)
                results[env] = await method(*args, **kwargs)
            except Exception as e:
                self.log_error(f"Failed to execute {operation} in {env.value}", error=str(e))
                results[env] = {"error": str(e)}
        
        return results
    
    def enable_data_isolation(self, enabled: bool = True):
        """启用/禁用数据隔离"""
        self._data_isolation_enabled = enabled
        self.log_info(f"Data isolation {'enabled' if enabled else 'disabled'}")
    
    @property
    def available_environments(self) -> List[TradingEnvironment]:
        """获取可用环境列表"""
        return list(self._interfaces.keys())
    
    def __repr__(self) -> str:
        return (
            f"TradingManager(current_env={self._current_environment}, "
            f"environments={list(self._interfaces.keys())}, "
            f"isolation_enabled={self._data_isolation_enabled})"
        )