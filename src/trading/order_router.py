import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.core.models import Order, OrderSide, OrderType, OrderStatus, MarketData
from src.exchanges.binance import BinanceFuturesClient
from src.exchanges.binance_quant_order_manager import BinanceQuantOrderManager
from src.utils.logger import LoggerMixin


class TradingMode(Enum):
    """交易模式"""
    PAPER = "paper"  # 模拟盘
    LIVE = "live"    # 实盘


class RouteDecision(Enum):
    """路由决策"""
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    REJECTED = "rejected"


@dataclass
class RoutingContext:
    """路由上下文"""
    order: Order
    trading_mode: TradingMode
    market_data: Optional[MarketData] = None
    account_balance: float = 0.0
    risk_level: str = "LOW"
    environment_confirmed: bool = False


@dataclass
class RoutingResult:
    """路由结果"""
    decision: RouteDecision
    target_engine: str  # "paper" or "live"
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    routing_time: float = 0.0
    confirmation_required: bool = False


class EnvironmentConfirmation:
    """环境确认机制"""
    
    def __init__(self):
        self.confirmation_required_conditions = {
            "large_order_value": 10000.0,  # >$10k需要确认
            "high_risk_symbols": ["BTCUSDT", "ETHUSDT"],  # 高风险交易对
            "leverage_threshold": 10,  # 高杠杆需要确认
        }
        self.auto_confirm_whitelist = set()  # 自动确认白名单
    
    def requires_confirmation(self, context: RoutingContext) -> bool:
        """判断是否需要环境确认"""
        order = context.order
        
        # 实盘模式且是重要订单需要确认
        if context.trading_mode == TradingMode.LIVE:
            # 大订单
            if order.price and order.quantity * order.price > self.confirmation_required_conditions["large_order_value"]:
                return True
            
            # 高风险交易对
            if order.symbol in self.confirmation_required_conditions["high_risk_symbols"]:
                return True
            
            # 市价单需要确认
            if order.order_type == OrderType.MARKET:
                return True
        
        return False
    
    async def confirm_environment(self, context: RoutingContext) -> bool:
        """确认交易环境"""
        if not self.requires_confirmation(context):
            return True
        
        # 检查白名单
        order_signature = f"{context.order.symbol}_{context.trading_mode.value}"
        if order_signature in self.auto_confirm_whitelist:
            return True
        
        # 这里可以实现人工确认机制
        # 简化实现：基于风险级别自动决策
        if context.risk_level == "LOW":
            return True
        elif context.risk_level == "MEDIUM":
            # 中风险：部分确认
            return context.order.quantity * (context.order.price or 0) < 5000
        else:
            # 高风险：需要人工确认
            return False


class TradingModeRouter(LoggerMixin):
    """交易模式路由器"""
    
    def __init__(
        self,
        live_client: Optional[BinanceFuturesClient] = None,
        paper_client: Optional[BinanceFuturesClient] = None
    ):
        # 实盘客户端和管理器
        self.live_client = live_client
        self.live_order_manager: Optional[BinanceQuantOrderManager] = None
        
        # 模拟盘客户端和管理器  
        self.paper_client = paper_client
        self.paper_order_manager: Optional[BinanceQuantOrderManager] = None
        
        # 虚拟盘引擎（稍后实现）
        self.paper_trading_engine = None
        
        # 环境确认机制
        self.env_confirmation = EnvironmentConfirmation()
        
        # 路由统计
        self.routing_stats = {
            "total_routes": 0,
            "paper_routes": 0,
            "live_routes": 0,
            "rejected_routes": 0,
            "confirmation_required": 0,
            "avg_routing_time": 0.0
        }
        
        # 路由历史
        self.routing_history: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """初始化路由器"""
        self.log_info("Initializing TradingModeRouter")
        
        # 初始化实盘管理器
        if self.live_client:
            self.live_order_manager = BinanceQuantOrderManager(self.live_client)
            await self.live_order_manager.initialize()
            self.log_info("Live trading manager initialized")
        
        # 初始化模拟盘管理器
        if self.paper_client:
            self.paper_order_manager = BinanceQuantOrderManager(self.paper_client)
            await self.paper_order_manager.initialize()
            self.log_info("Paper trading manager initialized")
        
        self.log_info("TradingModeRouter initialized successfully")
    
    async def route_order(
        self,
        order: Order,
        trading_mode: TradingMode,
        market_data: Optional[MarketData] = None,
        account_balance: float = 0.0,
        risk_level: str = "LOW",
        force_confirmation: bool = False
    ) -> RoutingResult:
        """路由订单到对应的交易环境"""
        start_time = asyncio.get_event_loop().time()
        
        # 创建路由上下文
        context = RoutingContext(
            order=order,
            trading_mode=trading_mode,
            market_data=market_data,
            account_balance=account_balance,
            risk_level=risk_level
        )
        
        self.log_info(
            "Routing order",
            symbol=order.symbol,
            side=order.side.value,
            trading_mode=trading_mode.value,
            risk_level=risk_level
        )
        
        try:
            # 预路由检查
            pre_check_result = await self._pre_route_checks(context)
            if pre_check_result.decision == RouteDecision.REJECTED:
                return pre_check_result
            
            # 环境确认
            confirmation_required = force_confirmation or self.env_confirmation.requires_confirmation(context)
            if confirmation_required:
                self.routing_stats["confirmation_required"] += 1
                confirmed = await self.env_confirmation.confirm_environment(context)
                if not confirmed:
                    return RoutingResult(
                        decision=RouteDecision.REJECTED,
                        target_engine="none",
                        error_message="Environment confirmation failed",
                        confirmation_required=True
                    )
                context.environment_confirmed = True
            
            # 执行路由
            if trading_mode == TradingMode.LIVE:
                result = await self._route_to_live(context)
            else:
                result = await self._route_to_paper(context)
            
            # 记录统计
            routing_time = asyncio.get_event_loop().time() - start_time
            result.routing_time = routing_time
            await self._update_routing_stats(result, routing_time)
            
            # 记录历史
            self._record_routing_history(context, result)
            
            return result
            
        except Exception as e:
            self.log_error(f"Order routing failed: {e}")
            routing_time = asyncio.get_event_loop().time() - start_time
            
            error_result = RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="error",
                error_message=str(e),
                routing_time=routing_time
            )
            
            await self._update_routing_stats(error_result, routing_time)
            return error_result
    
    async def _pre_route_checks(self, context: RoutingContext) -> RoutingResult:
        """预路由检查"""
        order = context.order
        
        # 基础订单验证
        if not order.symbol:
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="none",
                error_message="Missing order symbol"
            )
        
        if order.quantity <= 0:
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="none",
                error_message="Invalid order quantity"
            )
        
        # 检查对应的管理器是否可用
        if context.trading_mode == TradingMode.LIVE and not self.live_order_manager:
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="none",
                error_message="Live trading manager not available"
            )
        
        if context.trading_mode == TradingMode.PAPER and not self.paper_order_manager:
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="none", 
                error_message="Paper trading manager not available"
            )
        
        # 资金检查
        if context.trading_mode == TradingMode.LIVE:
            estimated_cost = order.quantity * (order.price or 0)
            if estimated_cost > context.account_balance * 0.95:  # 保留5%缓冲
                return RoutingResult(
                    decision=RouteDecision.REJECTED,
                    target_engine="none",
                    error_message="Insufficient account balance"
                )
        
        # 通过所有检查
        return RoutingResult(decision=RouteDecision.PAPER_TRADING, target_engine="pre_check_passed")
    
    async def _route_to_live(self, context: RoutingContext) -> RoutingResult:
        """路由到实盘交易"""
        if not self.live_order_manager:
            raise ValueError("Live trading manager not initialized")
        
        order = context.order
        
        try:
            self.log_info(
                "Routing to live trading",
                symbol=order.symbol,
                client_order_id=order.client_order_id
            )
            
            # 使用实盘订单管理器执行
            result = await self.live_order_manager.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                reduce_only=order.reduce_only,
                position_side=order.position_side,
                client_order_id=order.client_order_id
            )
            
            return RoutingResult(
                decision=RouteDecision.LIVE_TRADING,
                target_engine="live",
                execution_result=result
            )
            
        except Exception as e:
            self.log_error(f"Live trading execution failed: {e}")
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="live",
                error_message=str(e)
            )
    
    async def _route_to_paper(self, context: RoutingContext) -> RoutingResult:
        """路由到模拟盘交易"""
        order = context.order
        
        try:
            self.log_info(
                "Routing to paper trading",
                symbol=order.symbol,
                client_order_id=order.client_order_id
            )
            
            # 优先使用虚拟盘引擎
            if self.paper_trading_engine:
                result = await self.paper_trading_engine.execute_order(order)
                return RoutingResult(
                    decision=RouteDecision.PAPER_TRADING,
                    target_engine="paper_engine",
                    execution_result=result
                )
            
            # 备用：使用模拟盘订单管理器
            elif self.paper_order_manager:
                result = await self.paper_order_manager.place_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type.value,
                    quantity=order.quantity,
                    price=order.price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    reduce_only=order.reduce_only,
                    position_side=order.position_side,
                    client_order_id=order.client_order_id
                )
                
                return RoutingResult(
                    decision=RouteDecision.PAPER_TRADING,
                    target_engine="paper",
                    execution_result=result
                )
            
            else:
                # 最简单的模拟执行
                simulated_result = {
                    "orderId": f"paper_{order.client_order_id}",
                    "symbol": order.symbol,
                    "status": "FILLED",
                    "executedQty": str(order.quantity),
                    "avgPrice": str(order.price or 50000.0),
                    "transactTime": int(datetime.utcnow().timestamp() * 1000)
                }
                
                return RoutingResult(
                    decision=RouteDecision.PAPER_TRADING,
                    target_engine="simple_paper",
                    execution_result=simulated_result
                )
                
        except Exception as e:
            self.log_error(f"Paper trading execution failed: {e}")
            return RoutingResult(
                decision=RouteDecision.REJECTED,
                target_engine="paper",
                error_message=str(e)
            )
    
    async def cancel_order(
        self,
        symbol: str,
        trading_mode: TradingMode,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """取消订单"""
        try:
            if trading_mode == TradingMode.LIVE and self.live_order_manager:
                return await self.live_order_manager.cancel_order(symbol, order_id, client_order_id)
            
            elif trading_mode == TradingMode.PAPER:
                if self.paper_trading_engine:
                    return await self.paper_trading_engine.cancel_order(client_order_id)
                elif self.paper_order_manager:
                    return await self.paper_order_manager.cancel_order(symbol, order_id, client_order_id)
                else:
                    # 简单模拟取消
                    return {
                        "symbol": symbol,
                        "orderId": order_id,
                        "clientOrderId": client_order_id,
                        "status": "CANCELED"
                    }
            
            else:
                raise ValueError(f"No manager available for {trading_mode.value} mode")
                
        except Exception as e:
            self.log_error(f"Failed to cancel order: {e}")
            raise
    
    async def get_order_status(
        self,
        symbol: str,
        trading_mode: TradingMode,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """查询订单状态"""
        try:
            if trading_mode == TradingMode.LIVE and self.live_order_manager:
                return await self.live_order_manager.get_order_status(symbol, order_id, client_order_id)
            
            elif trading_mode == TradingMode.PAPER:
                if self.paper_trading_engine:
                    return await self.paper_trading_engine.get_order_status(client_order_id)
                elif self.paper_order_manager:
                    return await self.paper_order_manager.get_order_status(symbol, order_id, client_order_id)
                else:
                    # 简单模拟状态
                    return {
                        "symbol": symbol,
                        "orderId": order_id,
                        "clientOrderId": client_order_id,
                        "status": "FILLED",
                        "executedQty": "1.0",
                        "avgPrice": "50000.0"
                    }
            
            else:
                raise ValueError(f"No manager available for {trading_mode.value} mode")
                
        except Exception as e:
            self.log_error(f"Failed to get order status: {e}")
            raise
    
    async def _update_routing_stats(self, result: RoutingResult, routing_time: float):
        """更新路由统计"""
        self.routing_stats["total_routes"] += 1
        
        if result.decision == RouteDecision.LIVE_TRADING:
            self.routing_stats["live_routes"] += 1
        elif result.decision == RouteDecision.PAPER_TRADING:
            self.routing_stats["paper_routes"] += 1
        else:
            self.routing_stats["rejected_routes"] += 1
        
        # 更新平均路由时间
        total_time = (self.routing_stats["avg_routing_time"] * 
                     (self.routing_stats["total_routes"] - 1) + routing_time)
        self.routing_stats["avg_routing_time"] = total_time / self.routing_stats["total_routes"]
    
    def _record_routing_history(self, context: RoutingContext, result: RoutingResult):
        """记录路由历史"""
        history_entry = {
            "timestamp": datetime.utcnow(),
            "symbol": context.order.symbol,
            "side": context.order.side.value,
            "quantity": context.order.quantity,
            "trading_mode": context.trading_mode.value,
            "decision": result.decision.value,
            "target_engine": result.target_engine,
            "routing_time": result.routing_time,
            "risk_level": context.risk_level,
            "environment_confirmed": context.environment_confirmed,
            "success": result.error_message is None
        }
        
        self.routing_history.append(history_entry)
        
        # 保留最近1000条记录
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计"""
        total = self.routing_stats["total_routes"]
        
        stats = self.routing_stats.copy()
        if total > 0:
            stats["live_route_ratio"] = self.routing_stats["live_routes"] / total
            stats["paper_route_ratio"] = self.routing_stats["paper_routes"] / total
            stats["rejection_rate"] = self.routing_stats["rejected_routes"] / total
            stats["confirmation_rate"] = self.routing_stats["confirmation_required"] / total
        
        return stats
    
    def get_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取路由历史"""
        return self.routing_history[-limit:] if limit > 0 else self.routing_history
    
    def set_paper_trading_engine(self, engine):
        """设置虚拟盘引擎"""
        self.paper_trading_engine = engine
        self.log_info("Paper trading engine configured")
    
    def add_auto_confirm_order(self, symbol: str, trading_mode: TradingMode):
        """添加到自动确认白名单"""
        order_signature = f"{symbol}_{trading_mode.value}"
        self.env_confirmation.auto_confirm_whitelist.add(order_signature)
        self.log_info(f"Added to auto-confirm whitelist: {order_signature}")
    
    async def cleanup(self):
        """清理资源"""
        self.log_info("Cleaning up TradingModeRouter")
        
        if self.live_order_manager:
            await self.live_order_manager.cleanup()
        
        if self.paper_order_manager:
            await self.paper_order_manager.cleanup()
        
        self.routing_history.clear()
        self.log_info("TradingModeRouter cleanup complete")