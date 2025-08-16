import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from src.core.models import Order, OrderSide, OrderType, Position, MarketData, OrderBook
from src.agents.execution_agent import ExecutionAgent, ExecutionMode, ExecutionReport, ExecutionResult
from src.execution.algorithms import (
    ExecutionAlgorithm, TWAPAlgorithm, VWAPAlgorithm, POVAlgorithm, ImplementationShortfall, AlgorithmStatus
)
from src.utils.logger import LoggerMixin


class RoutingStrategy(Enum):
    """路由策略"""
    DIRECT = "direct"  # 直接执行
    SMART = "smart"    # 智能路由
    TWAP = "twap"      # 时间加权平均价格
    VWAP = "vwap"      # 成交量加权平均价格
    POV = "pov"        # 市场参与率
    IS = "is"          # 实施缺口


class OrderPriority(Enum):
    """订单优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class RoutingConfig:
    """路由配置"""
    strategy: RoutingStrategy = RoutingStrategy.SMART
    priority: OrderPriority = OrderPriority.NORMAL
    max_slice_size: float = 1000.0  # 最大分片大小
    min_slice_size: float = 0.01    # 最小分片大小
    slippage_tolerance: float = 0.005  # 滑点容忍度 0.5%
    timeout_seconds: int = 300      # 超时时间
    enable_dark_pool: bool = False  # 启用暗池
    enable_iceberg: bool = False    # 启用冰山订单
    iceberg_visible_ratio: float = 0.1  # 冰山订单可见比例


@dataclass
class RoutingContext:
    """路由上下文"""
    order: Order
    config: RoutingConfig
    market_data: Optional[MarketData] = None
    order_book: Optional[OrderBook] = None
    position: Optional[Position] = None
    urgency_score: float = 0.5  # 紧急程度 0-1
    market_impact_estimate: float = 0.0  # 市场冲击估计
    liquidity_score: float = 0.5  # 流动性评分 0-1


class OrderRouter(LoggerMixin):
    """订单路由器 - 负责智能订单路由和执行算法选择"""
    
    def __init__(self):
        self.execution_agents: Dict[ExecutionMode, ExecutionAgent] = {}
        self.algorithms: Dict[str, ExecutionAlgorithm] = {}
        self.routing_history: List[Dict[str, Any]] = []
        
        # 路由统计
        self.routing_stats = {
            "total_routed": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "avg_execution_time": 0.0,
            "avg_slippage": 0.0,
            "strategy_usage": {strategy.value: 0 for strategy in RoutingStrategy}
        }
    
    def register_execution_agent(self, mode: ExecutionMode, agent: ExecutionAgent):
        """注册执行Agent"""
        self.execution_agents[mode] = agent
        self.log_info(f"Registered execution agent", mode=mode.value)
    
    async def route_order(
        self, 
        order: Order, 
        config: Optional[RoutingConfig] = None,
        context: Optional[RoutingContext] = None
    ) -> ExecutionReport:
        """路由订单"""
        if config is None:
            config = RoutingConfig()
        
        if context is None:
            context = RoutingContext(order=order, config=config)
        else:
            context.config = config
        
        self.log_info(
            "Routing order",
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            strategy=config.strategy.value,
            priority=config.priority.value
        )
        
        try:
            # 分析订单和市场状况
            await self._analyze_routing_context(context)
            
            # 选择执行策略
            execution_strategy = await self._select_execution_strategy(context)
            
            # 执行订单
            report = await self._execute_with_strategy(context, execution_strategy)
            
            # 更新统计
            self._update_routing_stats(context, report, True)
            
            # 记录路由历史
            self._record_routing_history(context, execution_strategy, report)
            
            return report
            
        except Exception as e:
            self.log_error(f"Order routing failed: {e}", symbol=order.symbol)
            self._update_routing_stats(context, None, False)
            raise
    
    async def _analyze_routing_context(self, context: RoutingContext):
        """分析路由上下文"""
        order = context.order
        
        # 计算订单价值
        order_value = order.quantity * (order.price or 50000.0)  # 假设默认价格
        
        # 评估紧急程度
        if context.config.priority == OrderPriority.URGENT:
            context.urgency_score = 1.0
        elif context.config.priority == OrderPriority.HIGH:
            context.urgency_score = 0.8
        elif context.config.priority == OrderPriority.NORMAL:
            context.urgency_score = 0.5
        else:
            context.urgency_score = 0.2
        
        # 估算市场冲击
        if context.market_data and context.order_book:
            context.market_impact_estimate = self._estimate_market_impact(context)
            context.liquidity_score = self._evaluate_liquidity(context)
        
        self.log_debug(
            "Analyzed routing context",
            urgency_score=context.urgency_score,
            market_impact=context.market_impact_estimate,
            liquidity_score=context.liquidity_score,
            order_value=order_value
        )
    
    def _estimate_market_impact(self, context: RoutingContext) -> float:
        """估算市场冲击"""
        if not context.order_book:
            return 0.001  # 默认0.1%
        
        order = context.order
        side = order.side
        
        # 获取相应方向的订单簿
        if side == OrderSide.BUY:
            levels = context.order_book.asks
        else:
            levels = context.order_book.bids
        
        if not levels:
            return 0.01  # 1%
        
        # 计算需要消耗多少层级的流动性
        remaining_qty = order.quantity
        weighted_price = 0.0
        total_consumed = 0.0
        
        for price, size in levels:
            if remaining_qty <= 0:
                break
            
            consumed = min(remaining_qty, size)
            weighted_price += price * consumed
            total_consumed += consumed
            remaining_qty -= consumed
        
        if total_consumed > 0:
            avg_fill_price = weighted_price / total_consumed
            mid_price = context.order_book.mid_price
            
            if mid_price > 0:
                impact = abs(avg_fill_price - mid_price) / mid_price
                return impact
        
        return 0.005  # 默认0.5%
    
    def _evaluate_liquidity(self, context: RoutingContext) -> float:
        """评估流动性"""
        if not context.order_book:
            return 0.5
        
        # 基于订单簿深度评估流动性
        bid_depth = sum(size for _, size in context.order_book.bids[:10])  # 前10档
        ask_depth = sum(size for _, size in context.order_book.asks[:10])
        total_depth = bid_depth + ask_depth
        
        # 基于价差评估流动性
        spread_score = 1.0 / (1.0 + context.order_book.spread / context.order_book.mid_price * 100)
        
        # 基于深度评估流动性
        depth_score = min(total_depth / (context.order.quantity * 10), 1.0)
        
        # 综合评分
        liquidity_score = (spread_score + depth_score) / 2
        return liquidity_score
    
    async def _select_execution_strategy(self, context: RoutingContext) -> RoutingStrategy:
        """选择执行策略"""
        config = context.config
        
        # 如果指定了策略，直接使用
        if config.strategy != RoutingStrategy.SMART:
            return config.strategy
        
        # 智能策略选择
        order = context.order
        order_value = order.quantity * (order.price or 50000.0)
        
        # 小订单：直接执行
        if order_value < 1000:  # 小于$1000
            return RoutingStrategy.DIRECT
        
        # 紧急订单：直接执行
        if context.urgency_score > 0.8:
            return RoutingStrategy.DIRECT
        
        # 大订单且时间充裕：使用算法执行
        if order_value > 10000:  # 大于$10000
            # 根据市场状况选择算法
            if context.liquidity_score > 0.7:
                # 流动性好，使用VWAP
                return RoutingStrategy.VWAP
            elif context.market_impact_estimate > 0.01:
                # 市场冲击较大，使用TWAP
                return RoutingStrategy.TWAP
            else:
                # 使用实施缺口优化
                return RoutingStrategy.IS
        
        # 中等订单：使用POV
        return RoutingStrategy.POV
    
    async def _execute_with_strategy(
        self, 
        context: RoutingContext, 
        strategy: RoutingStrategy
    ) -> ExecutionReport:
        """使用指定策略执行订单"""
        order = context.order
        
        self.log_info(
            f"Executing order with {strategy.value} strategy",
            symbol=order.symbol,
            quantity=order.quantity
        )
        
        # 更新策略使用统计
        self.routing_stats["strategy_usage"][strategy.value] += 1
        
        if strategy == RoutingStrategy.DIRECT:
            # 直接执行
            return await self._execute_direct(context)
        
        elif strategy in [RoutingStrategy.TWAP, RoutingStrategy.VWAP, RoutingStrategy.POV, RoutingStrategy.IS]:
            # 使用算法执行
            return await self._execute_with_algorithm(context, strategy)
        
        else:
            # 默认直接执行
            return await self._execute_direct(context)
    
    async def _execute_direct(self, context: RoutingContext) -> ExecutionReport:
        """直接执行订单"""
        order = context.order
        config = context.config
        
        # 选择执行环境
        if order.symbol.endswith("USDT"):  # 实盘
            execution_mode = ExecutionMode.LIVE
        else:
            execution_mode = ExecutionMode.PAPER
        
        # 获取执行Agent
        execution_agent = self.execution_agents.get(execution_mode)
        if not execution_agent:
            raise ValueError(f"No execution agent available for mode: {execution_mode}")
        
        # 应用滑点保护
        if order.order_type == OrderType.LIMIT and order.price:
            # 根据滑点容忍度调整限价
            if order.side == OrderSide.BUY:
                # 买单：适当提高限价
                order.price *= (1 + config.slippage_tolerance / 2)
            else:
                # 卖单：适当降低限价
                order.price *= (1 - config.slippage_tolerance / 2)
        
        # 执行订单
        return await execution_agent.execute_order(order)
    
    async def _execute_with_algorithm(
        self, 
        context: RoutingContext, 
        strategy: RoutingStrategy
    ) -> ExecutionReport:
        """使用算法执行订单"""
        order = context.order
        
        # 创建算法实例
        algorithm = self._create_algorithm(strategy, context)
        
        # 执行算法
        algorithm_result = await algorithm.start(order)
        
        # 转换为ExecutionReport
        return ExecutionReport(
            order_id=algorithm_result.algorithm_id,
            client_order_id=order.client_order_id or "",
            result=(ExecutionResult.SUCCESS if algorithm_result.status == AlgorithmStatus.COMPLETED 
                   else ExecutionResult.FAILED),
            executed_qty=algorithm_result.total_filled,
            avg_price=algorithm_result.avg_price,
            commission=algorithm_result.total_commission,
            slippage=abs(algorithm_result.implementation_shortfall),
            execution_time=(algorithm_result.end_time - algorithm_result.start_time).total_seconds() 
                          if algorithm_result.end_time else 0.0,
            error_message=algorithm_result.error_message
        )
    
    def _create_algorithm(self, strategy: RoutingStrategy, context: RoutingContext) -> ExecutionAlgorithm:
        """创建执行算法"""
        algorithm_id = f"{strategy.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        if strategy == RoutingStrategy.TWAP:
            from src.execution.algorithms import TWAPAlgorithm
            config = TWAPAlgorithm.TWAPConfig(
                duration_minutes=max(30, context.order.quantity * 5),  # 根据订单大小调整时间
                slice_interval_seconds=60,
                max_participation_rate=context.config.slippage_tolerance * 10  # 基于滑点容忍度
            )
            return TWAPAlgorithm(algorithm_id, config)
        
        elif strategy == RoutingStrategy.VWAP:
            from src.execution.algorithms import VWAPAlgorithm
            config = VWAPAlgorithm.VWAPConfig(
                duration_minutes=max(30, context.order.quantity * 3),
                target_participation_rate=min(0.15, context.liquidity_score * 0.3)
            )
            return VWAPAlgorithm(algorithm_id, config)
        
        elif strategy == RoutingStrategy.POV:
            from src.execution.algorithms import POVAlgorithm
            config = POVAlgorithm.POVConfig(
                target_participation_rate=min(0.2, context.liquidity_score * 0.4),
                max_participation_rate=context.config.slippage_tolerance * 20
            )
            return POVAlgorithm(algorithm_id, config)
        
        elif strategy == RoutingStrategy.IS:
            from src.execution.algorithms import ImplementationShortfall
            config = ImplementationShortfall.ISConfig(
                risk_aversion=1.0 - context.urgency_score,  # 紧急程度越高，风险厌恶越低
                volatility_estimate=context.market_impact_estimate * 5
            )
            return ImplementationShortfall(algorithm_id, config)
        
        else:
            raise ValueError(f"Unsupported algorithm strategy: {strategy}")
    
    def _update_routing_stats(self, context: RoutingContext, report: Optional[ExecutionReport], success: bool):
        """更新路由统计"""
        self.routing_stats["total_routed"] += 1
        
        if success and report:
            self.routing_stats["successful_routes"] += 1
            
            # 更新平均执行时间
            total_time = (self.routing_stats["avg_execution_time"] * 
                         (self.routing_stats["successful_routes"] - 1) + report.execution_time)
            self.routing_stats["avg_execution_time"] = total_time / self.routing_stats["successful_routes"]
            
            # 更新平均滑点
            total_slippage = (self.routing_stats["avg_slippage"] * 
                            (self.routing_stats["successful_routes"] - 1) + report.slippage)
            self.routing_stats["avg_slippage"] = total_slippage / self.routing_stats["successful_routes"]
        else:
            self.routing_stats["failed_routes"] += 1
    
    def _record_routing_history(
        self, 
        context: RoutingContext, 
        strategy: RoutingStrategy, 
        report: ExecutionReport
    ):
        """记录路由历史"""
        history_entry = {
            "timestamp": datetime.utcnow(),
            "symbol": context.order.symbol,
            "side": context.order.side.value,
            "quantity": context.order.quantity,
            "strategy": strategy.value,
            "priority": context.config.priority.value,
            "urgency_score": context.urgency_score,
            "market_impact": context.market_impact_estimate,
            "liquidity_score": context.liquidity_score,
            "execution_result": {
                "executed_qty": report.executed_qty,
                "avg_price": report.avg_price,
                "slippage": report.slippage,
                "execution_time": report.execution_time,
                "success": report.result.value == "success"
            }
        }
        
        self.routing_history.append(history_entry)
        
        # 保留最近1000条记录
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        stats = self.routing_stats.copy()
        stats["success_rate"] = (
            stats["successful_routes"] / stats["total_routed"] 
            if stats["total_routed"] > 0 else 0
        )
        return stats
    
    def get_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取路由历史"""
        return self.routing_history[-limit:] if limit > 0 else self.routing_history
    
    async def cancel_algorithm(self, algorithm_id: str) -> bool:
        """取消算法执行"""
        if algorithm_id in self.algorithms:
            algorithm = self.algorithms[algorithm_id]
            await algorithm.stop()
            del self.algorithms[algorithm_id]
            self.log_info(f"Algorithm cancelled", algorithm_id=algorithm_id)
            return True
        
        self.log_warning(f"Algorithm not found for cancellation", algorithm_id=algorithm_id)
        return False
    
    async def pause_algorithm(self, algorithm_id: str) -> bool:
        """暂停算法执行"""
        if algorithm_id in self.algorithms:
            algorithm = self.algorithms[algorithm_id]
            await algorithm.pause()
            self.log_info(f"Algorithm paused", algorithm_id=algorithm_id)
            return True
        
        return False
    
    async def resume_algorithm(self, algorithm_id: str) -> bool:
        """恢复算法执行"""
        if algorithm_id in self.algorithms:
            algorithm = self.algorithms[algorithm_id]
            await algorithm.resume()
            self.log_info(f"Algorithm resumed", algorithm_id=algorithm_id)
            return True
        
        return False