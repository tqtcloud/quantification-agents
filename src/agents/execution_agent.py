import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from decimal import Decimal

from src.agents.base import BaseAgent, AgentConfig
from src.core.models import (
    Order, OrderSide, OrderType, OrderStatus, Position, Signal, TradingState,
    TimeInForce, PositionSide
)
from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError
from src.core.message_bus import MessageBus, MessagePriority
from src.utils.logger import LoggerMixin


class ExecutionMode(Enum):
    """执行模式"""
    PAPER = "paper"  # 模拟盘
    LIVE = "live"    # 实盘


class ExecutionResult(Enum):
    """执行结果"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class ExecutionContext:
    """执行上下文"""
    order: Order
    execution_mode: ExecutionMode
    retry_count: int = 0
    max_retries: int = 3
    execution_start_time: float = field(default_factory=time.time)
    expected_slippage: float = 0.0
    max_slippage: float = 0.005  # 最大滑点 0.5%
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ExecutionReport:
    """执行报告"""
    order_id: str
    client_order_id: str
    result: ExecutionResult
    executed_qty: float
    avg_price: float
    commission: float
    slippage: float
    execution_time: float  # 执行耗时(秒)
    error_message: Optional[str] = None
    fills: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ExecutionAgentConfig(AgentConfig):
    """执行Agent配置"""
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    max_order_value: float = 10000.0  # 单笔订单最大金额
    max_daily_trades: int = 1000  # 每日最大交易次数
    order_timeout: float = 30.0  # 订单超时时间(秒)
    retry_delays: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0])
    enable_partial_fills: bool = True
    enable_order_tracking: bool = True
    commission_rate: float = 0.0004  # 手续费率 0.04%


class ExecutionAgent(BaseAgent):
    """执行Agent - 负责订单执行和交易管理"""
    
    def __init__(
        self, 
        config: ExecutionAgentConfig,
        binance_client: Optional[BinanceFuturesClient] = None,
        message_bus: Optional[MessageBus] = None
    ):
        super().__init__(config, message_bus)
        self.execution_config = config
        self.binance_client = binance_client
        
        # 订单跟踪
        self._active_orders: Dict[str, ExecutionContext] = {}
        self._order_history: List[ExecutionReport] = []
        self._daily_trade_count = 0
        self._last_reset_date = datetime.utcnow().date()
        
        # 性能统计
        self._execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "total_volume": 0.0,
            "total_commission": 0.0,
            "avg_execution_time": 0.0,
            "avg_slippage": 0.0
        }
        
        # 异步任务管理
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def _initialize(self) -> None:
        """Agent特定初始化"""
        self.log_info("Initializing ExecutionAgent", mode=self.execution_config.execution_mode.value)
        
        # 初始化币安客户端
        if self.binance_client is None:
            self.binance_client = BinanceFuturesClient(
                testnet=(self.execution_config.execution_mode == ExecutionMode.PAPER)
            )
        
        # 连接币安客户端
        await self.binance_client.connect()
        
        # 启动订单监控任务
        if self.execution_config.enable_order_tracking:
            self._monitoring_task = asyncio.create_task(self._monitor_orders())
        
        # 注册消息处理器
        if self.message_bus:
            self.register_message_handler("execute_order", self._handle_execute_order_message)
            self.register_message_handler("cancel_order", self._handle_cancel_order_message)
            self.register_message_handler("get_order_status", self._handle_get_order_status_message)
        
        self.log_info("ExecutionAgent initialized successfully")
    
    async def _shutdown(self) -> None:
        """Agent特定关闭逻辑"""
        self.log_info("Shutting down ExecutionAgent")
        
        # 停止监控任务
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活跃订单
        if self._active_orders:
            self.log_warning(f"Cancelling {len(self._active_orders)} active orders")
            for order_id in list(self._active_orders.keys()):
                try:
                    await self.cancel_order(order_id)
                except Exception as e:
                    self.log_error(f"Failed to cancel order {order_id}: {e}")
        
        # 关闭币安客户端
        if self.binance_client:
            await self.binance_client.disconnect()
        
        self.log_info("ExecutionAgent shutdown complete")
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """执行Agent不生成信号，而是执行其他Agent的信号"""
        return []
    
    async def execute_order(self, order: Order) -> ExecutionReport:
        """执行订单"""
        self._reset_daily_stats_if_needed()
        
        # 检查每日交易限制
        if self._daily_trade_count >= self.execution_config.max_daily_trades:
            raise ValueError(f"Daily trade limit reached: {self._daily_trade_count}")
        
        # 检查订单金额限制
        order_value = order.quantity * (order.price or 0)
        if order_value > self.execution_config.max_order_value:
            raise ValueError(f"Order value {order_value} exceeds limit {self.execution_config.max_order_value}")
        
        # 创建执行上下文
        context = ExecutionContext(
            order=order,
            execution_mode=self.execution_config.execution_mode
        )
        
        # 生成client_order_id
        if not order.client_order_id:
            order.client_order_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        try:
            self.log_info(
                "Executing order",
                symbol=order.symbol,
                side=order.side.value,
                type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                mode=self.execution_config.execution_mode.value
            )
            
            # 根据执行模式选择执行方法
            if self.execution_config.execution_mode == ExecutionMode.PAPER:
                report = await self._execute_paper_order(context)
            else:
                report = await self._execute_live_order(context)
            
            # 更新统计
            self._update_execution_stats(report)
            self._order_history.append(report)
            self._daily_trade_count += 1
            
            # 发布执行完成事件
            if self.publisher:
                await self._publish_message(
                    f"execution.{order.symbol}.completed",
                    {
                        "order_id": report.order_id,
                        "result": report.result.value,
                        "executed_qty": report.executed_qty,
                        "avg_price": report.avg_price,
                        "slippage": report.slippage
                    },
                    priority=MessagePriority.HIGH
                )
            
            return report
            
        except Exception as e:
            self.log_error(f"Order execution failed: {e}", order_id=order.client_order_id)
            
            # 创建失败报告
            report = ExecutionReport(
                order_id="",
                client_order_id=order.client_order_id or "",
                result=ExecutionResult.FAILED,
                executed_qty=0.0,
                avg_price=0.0,
                commission=0.0,
                slippage=0.0,
                execution_time=time.time() - context.execution_start_time,
                error_message=str(e)
            )
            
            self._order_history.append(report)
            raise
    
    async def _execute_paper_order(self, context: ExecutionContext) -> ExecutionReport:
        """执行模拟订单"""
        order = context.order
        
        # 模拟执行延迟
        await asyncio.sleep(0.1)
        
        # 模拟市场价格（这里简化处理，实际应该使用实时市场数据）
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交，模拟少量滑点
            executed_price = order.price or 50000.0  # 假设价格
            slippage = 0.0005 if order.side == OrderSide.BUY else -0.0005  # 0.05%滑点
            executed_price *= (1 + slippage)
        else:
            # 限价单按限价成交
            executed_price = order.price or 50000.0
            slippage = 0.0
        
        # 计算手续费
        commission = order.quantity * executed_price * self.execution_config.commission_rate
        
        execution_time = time.time() - context.execution_start_time
        
        return ExecutionReport(
            order_id=f"paper_{int(time.time())}{order.client_order_id[-4:]}",
            client_order_id=order.client_order_id or "",
            result=ExecutionResult.SUCCESS,
            executed_qty=order.quantity,
            avg_price=executed_price,
            commission=commission,
            slippage=abs(slippage),
            execution_time=execution_time
        )
    
    async def _execute_live_order(self, context: ExecutionContext) -> ExecutionReport:
        """执行实盘订单"""
        order = context.order
        
        # 添加到活跃订单跟踪
        self._active_orders[order.client_order_id] = context
        
        try:
            # 调用币安API下单
            result = await self.binance_client.place_order(
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
            
            # 解析响应
            order_id = str(result.get('orderId', ''))
            status = result.get('status', 'NEW')
            executed_qty = float(result.get('executedQty', 0))
            
            # 如果是市价单或者立即成交的限价单
            if status == 'FILLED':
                avg_price = float(result.get('avgPrice', 0)) or float(result.get('price', 0))
                commission = 0.0  # 实际应该从交易详情中获取
                
                # 计算滑点
                expected_price = order.price or avg_price
                slippage = abs(avg_price - expected_price) / expected_price if expected_price > 0 else 0
                
                execution_time = time.time() - context.execution_start_time
                
                # 从活跃订单中移除
                if order.client_order_id in self._active_orders:
                    del self._active_orders[order.client_order_id]
                
                return ExecutionReport(
                    order_id=order_id,
                    client_order_id=order.client_order_id or "",
                    result=ExecutionResult.SUCCESS,
                    executed_qty=executed_qty,
                    avg_price=avg_price,
                    commission=commission,
                    slippage=slippage,
                    execution_time=execution_time
                )
            
            elif status in ['NEW', 'PARTIALLY_FILLED']:
                # 订单已提交但未完全成交，需要等待或监控
                self.log_info(f"Order submitted successfully", order_id=order_id, status=status)
                
                # 如果是限价单，可能需要等待成交
                if order.order_type in [OrderType.LIMIT, OrderType.STOP, OrderType.TAKE_PROFIT]:
                    return await self._wait_for_order_completion(order_id, order.client_order_id, context)
                
            else:
                # 订单被拒绝
                raise BinanceAPIError(-1, f"Order rejected with status: {status}")
                
        except BinanceAPIError as e:
            # 从活跃订单中移除
            if order.client_order_id in self._active_orders:
                del self._active_orders[order.client_order_id]
            
            self.log_error(f"Binance API error: {e}")
            raise
        
        except Exception as e:
            # 从活跃订单中移除
            if order.client_order_id in self._active_orders:
                del self._active_orders[order.client_order_id]
            
            self.log_error(f"Unexpected error during order execution: {e}")
            raise
    
    async def _wait_for_order_completion(
        self, 
        order_id: str, 
        client_order_id: str, 
        context: ExecutionContext
    ) -> ExecutionReport:
        """等待订单完成"""
        order = context.order
        start_time = time.time()
        timeout = self.execution_config.order_timeout
        
        while time.time() - start_time < timeout:
            try:
                # 查询订单状态
                result = await self.binance_client.get_order(
                    symbol=order.symbol,
                    client_order_id=client_order_id
                )
                
                status = result.get('status', 'NEW')
                executed_qty = float(result.get('executedQty', 0))
                
                if status == 'FILLED':
                    avg_price = float(result.get('avgPrice', 0))
                    commission = 0.0  # 实际应该计算实际手续费
                    
                    # 计算滑点
                    expected_price = order.price or avg_price
                    slippage = abs(avg_price - expected_price) / expected_price if expected_price > 0 else 0
                    
                    execution_time = time.time() - context.execution_start_time
                    
                    # 从活跃订单中移除
                    if client_order_id in self._active_orders:
                        del self._active_orders[client_order_id]
                    
                    return ExecutionReport(
                        order_id=order_id,
                        client_order_id=client_order_id,
                        result=ExecutionResult.SUCCESS,
                        executed_qty=executed_qty,
                        avg_price=avg_price,
                        commission=commission,
                        slippage=slippage,
                        execution_time=execution_time
                    )
                
                elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                    # 从活跃订单中移除
                    if client_order_id in self._active_orders:
                        del self._active_orders[client_order_id]
                    
                    return ExecutionReport(
                        order_id=order_id,
                        client_order_id=client_order_id,
                        result=ExecutionResult.FAILED,
                        executed_qty=executed_qty,
                        avg_price=0.0,
                        commission=0.0,
                        slippage=0.0,
                        execution_time=time.time() - context.execution_start_time,
                        error_message=f"Order {status.lower()}"
                    )
                
                elif status == 'PARTIALLY_FILLED' and self.execution_config.enable_partial_fills:
                    # 部分成交，继续等待
                    self.log_debug(f"Order partially filled", order_id=order_id, executed_qty=executed_qty)
                
                # 等待一小段时间再查询
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.log_warning(f"Error checking order status: {e}")
                await asyncio.sleep(1.0)
        
        # 超时处理
        self.log_warning(f"Order timeout, attempting to cancel", order_id=order_id)
        try:
            await self.binance_client.cancel_order(symbol=order.symbol, client_order_id=client_order_id)
        except Exception as e:
            self.log_error(f"Failed to cancel timeout order: {e}")
        
        # 从活跃订单中移除
        if client_order_id in self._active_orders:
            del self._active_orders[client_order_id]
        
        return ExecutionReport(
            order_id=order_id,
            client_order_id=client_order_id,
            result=ExecutionResult.FAILED,
            executed_qty=0.0,
            avg_price=0.0,
            commission=0.0,
            slippage=0.0,
            execution_time=time.time() - context.execution_start_time,
            error_message="Order timeout"
        )
    
    async def cancel_order(self, client_order_id: str) -> bool:
        """取消订单"""
        if client_order_id not in self._active_orders:
            self.log_warning(f"Order not found in active orders: {client_order_id}")
            return False
        
        context = self._active_orders[client_order_id]
        
        try:
            if self.execution_config.execution_mode == ExecutionMode.LIVE:
                await self.binance_client.cancel_order(
                    symbol=context.order.symbol,
                    client_order_id=client_order_id
                )
            
            # 从活跃订单中移除
            del self._active_orders[client_order_id]
            
            self.log_info(f"Order cancelled successfully", client_order_id=client_order_id)
            return True
            
        except Exception as e:
            self.log_error(f"Failed to cancel order: {e}", client_order_id=client_order_id)
            return False
    
    async def get_order_status(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        if client_order_id not in self._active_orders:
            return None
        
        context = self._active_orders[client_order_id]
        
        if self.execution_config.execution_mode == ExecutionMode.LIVE:
            try:
                return await self.binance_client.get_order(
                    symbol=context.order.symbol,
                    client_order_id=client_order_id
                )
            except Exception as e:
                self.log_error(f"Failed to get order status: {e}", client_order_id=client_order_id)
                return None
        else:
            # 模拟盘返回模拟状态
            return {
                "symbol": context.order.symbol,
                "orderId": f"paper_{client_order_id}",
                "clientOrderId": client_order_id,
                "status": "FILLED",
                "executedQty": str(context.order.quantity),
                "avgPrice": str(context.order.price or 50000.0)
            }
    
    async def _monitor_orders(self):
        """监控活跃订单"""
        while not self._shutdown_event.is_set():
            try:
                # 检查超时订单
                current_time = time.time()
                timeout_orders = []
                
                for client_order_id, context in self._active_orders.items():
                    if current_time - context.execution_start_time > self.execution_config.order_timeout:
                        timeout_orders.append(client_order_id)
                
                # 处理超时订单
                for client_order_id in timeout_orders:
                    self.log_warning(f"Order timeout detected", client_order_id=client_order_id)
                    await self.cancel_order(client_order_id)
                
                await asyncio.sleep(5.0)  # 每5秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Error in order monitoring: {e}")
                await asyncio.sleep(5.0)
    
    def _reset_daily_stats_if_needed(self):
        """重置每日统计（如果需要）"""
        current_date = datetime.utcnow().date()
        if current_date != self._last_reset_date:
            self._daily_trade_count = 0
            self._last_reset_date = current_date
            self.log_info(f"Daily stats reset for {current_date}")
    
    def _update_execution_stats(self, report: ExecutionReport):
        """更新执行统计"""
        self._execution_stats["total_orders"] += 1
        
        if report.result == ExecutionResult.SUCCESS:
            self._execution_stats["successful_orders"] += 1
            self._execution_stats["total_volume"] += report.executed_qty * report.avg_price
            self._execution_stats["total_commission"] += report.commission
            
            # 更新平均执行时间
            total_time = (self._execution_stats["avg_execution_time"] * 
                         (self._execution_stats["successful_orders"] - 1) + report.execution_time)
            self._execution_stats["avg_execution_time"] = total_time / self._execution_stats["successful_orders"]
            
            # 更新平均滑点
            total_slippage = (self._execution_stats["avg_slippage"] * 
                            (self._execution_stats["successful_orders"] - 1) + report.slippage)
            self._execution_stats["avg_slippage"] = total_slippage / self._execution_stats["successful_orders"]
        else:
            self._execution_stats["failed_orders"] += 1
    
    # 消息处理器
    async def _handle_execute_order_message(self, from_agent: str, data: Dict[str, Any]):
        """处理执行订单消息"""
        try:
            order_data = data.get("order")
            if not order_data:
                self.log_error("Missing order data in execute_order message")
                return
            
            # 构造Order对象
            order = Order(**order_data)
            report = await self.execute_order(order)
            
            # 发送执行结果
            await self.send_message_to_agent(
                from_agent,
                "order_executed",
                {
                    "report": {
                        "order_id": report.order_id,
                        "client_order_id": report.client_order_id,
                        "result": report.result.value,
                        "executed_qty": report.executed_qty,
                        "avg_price": report.avg_price,
                        "commission": report.commission,
                        "slippage": report.slippage,
                        "execution_time": report.execution_time,
                        "error_message": report.error_message
                    }
                }
            )
            
        except Exception as e:
            self.log_error(f"Error handling execute_order message: {e}")
    
    async def _handle_cancel_order_message(self, from_agent: str, data: Dict[str, Any]):
        """处理取消订单消息"""
        try:
            client_order_id = data.get("client_order_id")
            if not client_order_id:
                self.log_error("Missing client_order_id in cancel_order message")
                return
            
            success = await self.cancel_order(client_order_id)
            
            # 发送取消结果
            await self.send_message_to_agent(
                from_agent,
                "order_cancelled",
                {
                    "client_order_id": client_order_id,
                    "success": success
                }
            )
            
        except Exception as e:
            self.log_error(f"Error handling cancel_order message: {e}")
    
    async def _handle_get_order_status_message(self, from_agent: str, data: Dict[str, Any]):
        """处理获取订单状态消息"""
        try:
            client_order_id = data.get("client_order_id")
            if not client_order_id:
                self.log_error("Missing client_order_id in get_order_status message")
                return
            
            status = await self.get_order_status(client_order_id)
            
            # 发送订单状态
            await self.send_message_to_agent(
                from_agent,
                "order_status",
                {
                    "client_order_id": client_order_id,
                    "status": status
                }
            )
            
        except Exception as e:
            self.log_error(f"Error handling get_order_status message: {e}")
    
    # 公共接口
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = self._execution_stats.copy()
        stats["active_orders_count"] = len(self._active_orders)
        stats["daily_trade_count"] = self._daily_trade_count
        stats["success_rate"] = (stats["successful_orders"] / stats["total_orders"] 
                               if stats["total_orders"] > 0 else 0)
        return stats
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取活跃订单列表"""
        return [
            {
                "client_order_id": client_order_id,
                "symbol": context.order.symbol,
                "side": context.order.side.value,
                "order_type": context.order.order_type.value,
                "quantity": context.order.quantity,
                "price": context.order.price,
                "execution_start_time": context.execution_start_time,
                "retry_count": context.retry_count
            }
            for client_order_id, context in self._active_orders.items()
        ]
    
    def get_order_history(self, limit: int = 100) -> List[ExecutionReport]:
        """获取订单历史"""
        return self._order_history[-limit:] if limit > 0 else self._order_history