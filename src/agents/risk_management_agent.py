"""
风险管理Agent实现
基于Agent基类实现风险评估、仓位大小计算和限制检查，添加动态风险指标监控
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base import RiskAgent, AgentConfig
from src.core.models import (
    TradingState, Order, Position, Signal, RiskMetrics, 
    OrderSide, PositionSide, MarketData
)
from src.core.message_bus import MessageBus


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionLimits:
    """仓位限制配置"""
    max_position_size: float = 0.05  # 单个交易对最大仓位（占总资金比例）
    max_total_exposure: float = 0.8   # 总仓位敞口限制
    max_leverage: int = 10             # 最大杠杆倍数
    max_positions_count: int = 10      # 最大持仓数量
    max_correlation_exposure: float = 0.3  # 相关性仓位限制


@dataclass
class RiskLimits:
    """风险限制配置"""
    max_daily_loss: float = 0.05      # 每日最大亏损比例
    max_drawdown: float = 0.15         # 最大回撤比例
    min_margin_ratio: float = 0.2      # 最小保证金比例
    stop_loss_ratio: float = 0.02     # 默认止损比例
    take_profit_ratio: float = 0.04   # 默认止盈比例
    var_threshold: float = 0.1         # VaR阈值


@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    level: RiskLevel
    message: str
    symbol: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskManagementAgent(RiskAgent):
    """风险管理Agent"""
    
    def __init__(self, 
                 config: AgentConfig,
                 message_bus: Optional[MessageBus] = None,
                 position_limits: Optional[PositionLimits] = None,
                 risk_limits: Optional[RiskLimits] = None):
        super().__init__(config, message_bus)
        
        self.position_limits = position_limits or PositionLimits()
        self.risk_limits = risk_limits or RiskLimits()
        
        # 风险监控状态
        self.current_risk_level = RiskLevel.LOW
        self.active_alerts: List[RiskAlert] = []
        self.risk_history: List[Tuple[datetime, RiskLevel, str]] = []
        
        # 统计数据
        self.total_capital = config.parameters.get("total_capital", 100000.0)
        self.daily_pnl_start = 0.0
        self.day_start_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 仓位跟踪
        self.position_history: Dict[str, List[float]] = {}  # symbol -> [position_sizes]
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        self.log_info(f"RiskManagementAgent initialized with capital: {self.total_capital}")

    async def _initialize(self) -> None:
        """初始化风险管理Agent"""
        # 注册消息处理器
        self.register_message_handler("position_update", self._handle_position_update)
        self.register_message_handler("order_execution", self._handle_order_execution)
        self.register_message_handler("market_data", self._handle_market_data_update)
        
        # 启动风险监控任务
        self._running_tasks["risk_monitoring"] = asyncio.create_task(self._risk_monitoring_loop())
        
        self.log_info("RiskManagementAgent initialized")

    async def check_risk(self, order: Order, state: TradingState) -> bool:
        """检查订单是否通过风险管理规则"""
        try:
            # 1. 检查基本限制
            if not await self._check_basic_limits(order, state):
                return False
            
            # 2. 检查仓位限制
            if not await self._check_position_limits(order, state):
                return False
            
            # 3. 检查杠杆限制
            if not await self._check_leverage_limits(order, state):
                return False
            
            # 4. 检查资金充足性
            if not await self._check_capital_sufficiency(order, state):
                return False
            
            # 5. 检查相关性风险
            if not await self._check_correlation_risk(order, state):
                return False
            
            self.log_debug(f"Risk check passed for order: {order.symbol} {order.side} {order.quantity}")
            return True
            
        except Exception as e:
            self.log_error(f"Error in risk check: {e}")
            return False

    async def adjust_position_size(self, order: Order, state: TradingState) -> Order:
        """根据风险管理规则调整订单大小"""
        try:
            original_quantity = order.quantity
            
            # 1. 基于资金管理调整
            adjusted_order = await self._adjust_by_capital_management(order, state)
            
            # 2. 基于波动率调整
            adjusted_order = await self._adjust_by_volatility(adjusted_order, state)
            
            # 3. 基于当前风险水平调整
            adjusted_order = await self._adjust_by_risk_level(adjusted_order, state)
            
            # 4. 确保满足最小交易单位
            adjusted_order = await self._apply_min_trade_size(adjusted_order, state)
            
            if adjusted_order.quantity != original_quantity:
                self.log_info(
                    f"Position size adjusted: {order.symbol} "
                    f"{original_quantity} -> {adjusted_order.quantity}"
                )
            
            return adjusted_order
            
        except Exception as e:
            self.log_error(f"Error adjusting position size: {e}")
            return order

    async def analyze(self, state: TradingState) -> List[Signal]:
        """分析风险状况并生成风险信号"""
        signals = []
        
        try:
            # 更新风险指标
            await self._update_risk_metrics(state)
            
            # 检查各种风险条件
            risk_signals = []
            
            # 1. 检查保证金使用率
            if state.risk_metrics and state.risk_metrics.margin_usage > 0.8:
                risk_signals.extend(await self._generate_margin_risk_signals(state))
            
            # 2. 检查回撤风险
            if state.risk_metrics and state.risk_metrics.current_drawdown > self.risk_limits.max_drawdown * 0.8:
                risk_signals.extend(await self._generate_drawdown_risk_signals(state))
            
            # 3. 检查每日亏损
            daily_pnl = await self._calculate_daily_pnl(state)
            if daily_pnl < -self.risk_limits.max_daily_loss * self.total_capital * 0.8:
                risk_signals.extend(await self._generate_daily_loss_signals(state, daily_pnl))
            
            # 4. 检查仓位集中度
            concentration_signals = await self._check_position_concentration(state)
            risk_signals.extend(concentration_signals)
            
            # 5. 更新风险等级
            await self._update_risk_level(state)
            
            # 6. 生成风险调整信号
            if self.current_risk_level != RiskLevel.LOW:
                risk_signals.extend(await self._generate_risk_adjustment_signals(state))
            
            return risk_signals
            
        except Exception as e:
            self.log_error(f"Error in risk analysis: {e}")
            return []

    async def _check_basic_limits(self, order: Order, state: TradingState) -> bool:
        """检查基本限制"""
        # 检查订单数量
        if order.quantity <= 0:
            self.log_warning(f"Invalid order quantity: {order.quantity}")
            return False
        
        # 检查最大持仓数量
        current_positions = len([p for p in state.positions.values() if p.quantity != 0])
        if current_positions >= self.position_limits.max_positions_count:
            self.log_warning(f"Maximum positions limit reached: {current_positions}")
            return False
        
        return True

    async def _check_position_limits(self, order: Order, state: TradingState) -> bool:
        """检查仓位限制"""
        # 计算订单价值
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            self.log_warning(f"No market data for {order.symbol}")
            return False
        
        estimated_price = market_data.price
        order_value = order.quantity * estimated_price
        
        # 检查单个交易对仓位限制
        max_position_value = self.total_capital * self.position_limits.max_position_size
        
        current_position = state.positions.get(order.symbol)
        current_value = 0
        if current_position:
            current_value = abs(current_position.quantity) * current_position.mark_price
        
        new_total_value = current_value + order_value
        if new_total_value > max_position_value:
            self.log_warning(
                f"Position limit exceeded for {order.symbol}: "
                f"{new_total_value} > {max_position_value}"
            )
            return False
        
        # 检查总敞口限制
        total_exposure = sum(
            abs(pos.quantity) * pos.mark_price 
            for pos in state.positions.values()
        ) + order_value
        
        max_total_exposure = self.total_capital * self.position_limits.max_total_exposure
        if total_exposure > max_total_exposure:
            self.log_warning(
                f"Total exposure limit exceeded: {total_exposure} > {max_total_exposure}"
            )
            return False
        
        return True

    async def _check_leverage_limits(self, order: Order, state: TradingState) -> bool:
        """检查杠杆限制"""
        current_position = state.positions.get(order.symbol)
        if current_position and current_position.leverage > self.position_limits.max_leverage:
            self.log_warning(
                f"Leverage limit exceeded for {order.symbol}: "
                f"{current_position.leverage} > {self.position_limits.max_leverage}"
            )
            return False
        
        return True

    async def _check_capital_sufficiency(self, order: Order, state: TradingState) -> bool:
        """检查资金充足性"""
        # 简化的资金检查，实际应该考虑保证金要求
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return False
        
        estimated_price = market_data.price
        required_margin = (order.quantity * estimated_price) / self.position_limits.max_leverage
        
        # 计算可用资金（简化计算）
        total_margin_used = sum(pos.margin for pos in state.positions.values())
        available_capital = self.total_capital - total_margin_used
        
        if required_margin > available_capital:
            self.log_warning(
                f"Insufficient capital for {order.symbol}: "
                f"required {required_margin}, available {available_capital}"
            )
            return False
        
        return True

    async def _check_correlation_risk(self, order: Order, state: TradingState) -> bool:
        """检查相关性风险"""
        # 简化的相关性检查，实际应该使用历史价格数据计算相关性
        symbol = order.symbol
        
        # 检查是否与现有仓位高度相关
        for pos_symbol, position in state.positions.items():
            if position.quantity == 0 or pos_symbol == symbol:
                continue
            
            # 简化的相关性假设（实际应该从历史数据计算）
            correlation = self._get_correlation(symbol, pos_symbol)
            
            if abs(correlation) > 0.7:  # 高相关性阈值
                combined_exposure = abs(position.quantity * position.mark_price)
                
                market_data = state.market_data.get(symbol)
                if market_data:
                    combined_exposure += order.quantity * market_data.price
                
                max_correlated_exposure = self.total_capital * self.position_limits.max_correlation_exposure
                
                if combined_exposure > max_correlated_exposure:
                    self.log_warning(
                        f"Correlation risk exceeded: {symbol} and {pos_symbol} "
                        f"correlation: {correlation}, exposure: {combined_exposure}"
                    )
                    return False
        
        return True

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """获取两个交易对的相关性（简化实现）"""
        # 简化的相关性假设，实际应该从历史价格数据计算
        correlation_map = {
            ("BTCUSDT", "ETHUSDT"): 0.8,
            ("ETHUSDT", "ADAUSDT"): 0.7,
            ("BTCUSDT", "LTCUSDT"): 0.75,
        }
        
        key = tuple(sorted([symbol1, symbol2]))
        return correlation_map.get(key, 0.3)  # 默认中等相关性

    async def _adjust_by_capital_management(self, order: Order, state: TradingState) -> Order:
        """基于资金管理调整仓位大小"""
        # Kelly公式或固定比例资金管理
        max_risk_per_trade = self.total_capital * 0.02  # 每笔交易最大风险2%
        
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return order
        
        estimated_price = market_data.price
        stop_loss_price = estimated_price * (1 - self.risk_limits.stop_loss_ratio)
        risk_per_unit = abs(estimated_price - stop_loss_price)
        
        if risk_per_unit > 0:
            max_quantity = max_risk_per_trade / risk_per_unit
            if order.quantity > max_quantity:
                order.quantity = max_quantity
        
        return order

    async def _adjust_by_volatility(self, order: Order, state: TradingState) -> Order:
        """基于波动率调整仓位大小"""
        # 简化的波动率调整，实际应该使用ATR或历史波动率
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return order
        
        # 假设的波动率（实际应该计算）
        volatility = 0.02  # 2%日波动率
        
        # 高波动率时减少仓位
        if volatility > 0.05:  # 5%
            order.quantity *= 0.5
        elif volatility > 0.03:  # 3%
            order.quantity *= 0.7
        
        return order

    async def _adjust_by_risk_level(self, order: Order, state: TradingState) -> Order:
        """基于当前风险水平调整仓位大小"""
        risk_multiplier = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.5,
            RiskLevel.CRITICAL: 0.2
        }
        
        multiplier = risk_multiplier.get(self.current_risk_level, 0.5)
        order.quantity *= multiplier
        
        return order

    async def _apply_min_trade_size(self, order: Order, state: TradingState) -> Order:
        """应用最小交易单位"""
        # 简化实现，实际应该从交易所获取最小交易单位
        min_qty = 0.001  # 假设最小交易量
        
        if order.quantity < min_qty:
            order.quantity = min_qty
        
        # 精度调整（保留3位小数）
        order.quantity = round(order.quantity, 3)
        
        return order

    async def _update_risk_metrics(self, state: TradingState):
        """更新风险指标"""
        # 这里会调用RiskMetricsCalculator来计算风险指标
        # 现在先简化实现
        pass

    async def _generate_margin_risk_signals(self, state: TradingState) -> List[Signal]:
        """生成保证金风险信号"""
        signals = []
        
        if state.risk_metrics and state.risk_metrics.margin_usage > 0.9:
            signals.append(Signal(
                source=self.name,
                symbol="ALL",
                action=OrderSide.SELL,
                strength=-0.8,
                confidence=0.9,
                reason="Critical margin usage level",
                metadata={
                    "margin_usage": state.risk_metrics.margin_usage,
                    "action_required": "reduce_positions"
                }
            ))
        
        return signals

    async def _generate_drawdown_risk_signals(self, state: TradingState) -> List[Signal]:
        """生成回撤风险信号"""
        signals = []
        
        if state.risk_metrics and state.risk_metrics.current_drawdown > self.risk_limits.max_drawdown:
            signals.append(Signal(
                source=self.name,
                symbol="ALL",
                action=OrderSide.SELL,
                strength=-1.0,
                confidence=0.95,
                reason="Maximum drawdown exceeded",
                metadata={
                    "current_drawdown": state.risk_metrics.current_drawdown,
                    "max_drawdown": self.risk_limits.max_drawdown,
                    "action_required": "emergency_stop"
                }
            ))
        
        return signals

    async def _generate_daily_loss_signals(self, state: TradingState, daily_pnl: float) -> List[Signal]:
        """生成每日亏损风险信号"""
        signals = []
        
        max_daily_loss = self.risk_limits.max_daily_loss * self.total_capital
        
        if daily_pnl < -max_daily_loss:
            signals.append(Signal(
                source=self.name,
                symbol="ALL",
                action=OrderSide.SELL,
                strength=-0.9,
                confidence=0.9,
                reason="Daily loss limit exceeded",
                metadata={
                    "daily_pnl": daily_pnl,
                    "max_daily_loss": max_daily_loss,
                    "action_required": "stop_new_trades"
                }
            ))
        
        return signals

    async def _check_position_concentration(self, state: TradingState) -> List[Signal]:
        """检查仓位集中度"""
        signals = []
        
        if not state.positions:
            return signals
        
        # 计算仓位集中度
        total_value = sum(abs(pos.quantity) * pos.mark_price for pos in state.positions.values())
        
        for symbol, position in state.positions.items():
            if position.quantity == 0:
                continue
            
            position_value = abs(position.quantity) * position.mark_price
            concentration = position_value / total_value if total_value > 0 else 0
            
            if concentration > 0.5:  # 单个仓位超过50%
                signals.append(Signal(
                    source=self.name,
                    symbol=symbol,
                    action=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                    strength=-0.6,
                    confidence=0.8,
                    reason="Position concentration too high",
                    metadata={
                        "concentration": concentration,
                        "action_required": "reduce_position"
                    }
                ))
        
        return signals

    async def _generate_risk_adjustment_signals(self, state: TradingState) -> List[Signal]:
        """生成风险调整信号"""
        signals = []
        
        if self.current_risk_level == RiskLevel.HIGH:
            signals.append(Signal(
                source=self.name,
                symbol="ALL",
                action=OrderSide.SELL,
                strength=-0.5,
                confidence=0.7,
                reason="High risk level detected",
                metadata={
                    "risk_level": self.current_risk_level.value,
                    "action_required": "reduce_exposure"
                }
            ))
        elif self.current_risk_level == RiskLevel.CRITICAL:
            signals.append(Signal(
                source=self.name,
                symbol="ALL",
                action=OrderSide.SELL,
                strength=-0.9,
                confidence=0.95,
                reason="Critical risk level",
                metadata={
                    "risk_level": self.current_risk_level.value,
                    "action_required": "emergency_action"
                }
            ))
        
        return signals

    async def _update_risk_level(self, state: TradingState):
        """更新风险等级"""
        risk_score = 0
        
        # 基于保证金使用率
        if state.risk_metrics:
            if state.risk_metrics.margin_usage > 0.8:
                risk_score += 3
            elif state.risk_metrics.margin_usage > 0.6:
                risk_score += 2
            elif state.risk_metrics.margin_usage > 0.4:
                risk_score += 1
            
            # 基于回撤
            if state.risk_metrics.current_drawdown > self.risk_limits.max_drawdown * 0.8:
                risk_score += 3
            elif state.risk_metrics.current_drawdown > self.risk_limits.max_drawdown * 0.5:
                risk_score += 2
        
        # 基于每日亏损
        daily_pnl = await self._calculate_daily_pnl(state)
        daily_loss_ratio = abs(daily_pnl) / self.total_capital
        
        if daily_loss_ratio > self.risk_limits.max_daily_loss * 0.8:
            risk_score += 3
        elif daily_loss_ratio > self.risk_limits.max_daily_loss * 0.5:
            risk_score += 2
        
        # 确定风险等级
        old_level = self.current_risk_level
        
        if risk_score >= 6:
            self.current_risk_level = RiskLevel.CRITICAL
        elif risk_score >= 4:
            self.current_risk_level = RiskLevel.HIGH
        elif risk_score >= 2:
            self.current_risk_level = RiskLevel.MEDIUM
        else:
            self.current_risk_level = RiskLevel.LOW
        
        # 记录风险等级变化
        if old_level != self.current_risk_level:
            self.risk_history.append((
                datetime.utcnow(),
                self.current_risk_level,
                f"Risk level changed from {old_level.value} to {self.current_risk_level.value}"
            ))
            
            self.log_warning(
                f"Risk level changed: {old_level.value} -> {self.current_risk_level.value}"
            )
            
            # 发送风险等级变化通知
            await self.broadcast_message("risk_level_change", {
                "old_level": old_level.value,
                "new_level": self.current_risk_level.value,
                "risk_score": risk_score
            })

    async def _calculate_daily_pnl(self, state: TradingState) -> float:
        """计算当日盈亏"""
        if not state.risk_metrics:
            return 0.0
        
        # 检查是否需要重置日期
        current_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if current_day > self.day_start_time:
            self.day_start_time = current_day
            self.daily_pnl_start = state.risk_metrics.total_pnl
        
        return state.risk_metrics.total_pnl - self.daily_pnl_start

    async def _risk_monitoring_loop(self):
        """风险监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)  # 每5秒检查一次
                
                # 检查活动警报
                self._check_alert_expiry()
                
                # 发送风险监控心跳
                if self.publisher:
                    await self._publish_message(
                        f"risk.{self.name}.status",
                        {
                            "risk_level": self.current_risk_level.value,
                            "active_alerts": len(self.active_alerts),
                            "timestamp": time.time()
                        }
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Error in risk monitoring loop: {e}")

    def _check_alert_expiry(self):
        """检查警报过期"""
        current_time = datetime.utcnow()
        expired_alerts = [
            alert for alert in self.active_alerts
            if (current_time - alert.timestamp).total_seconds() > 3600  # 1小时过期
        ]
        
        for alert in expired_alerts:
            self.active_alerts.remove(alert)
            self.log_debug(f"Risk alert expired: {alert.alert_id}")

    async def _handle_position_update(self, from_agent: str, data: Any):
        """处理仓位更新消息"""
        self.log_debug(f"Received position update from {from_agent}")
        # 处理仓位更新逻辑

    async def _handle_order_execution(self, from_agent: str, data: Any):
        """处理订单执行消息"""
        self.log_debug(f"Received order execution from {from_agent}")
        # 处理订单执行逻辑

    async def _handle_market_data_update(self, from_agent: str, data: Any):
        """处理市场数据更新消息"""
        self.log_debug(f"Received market data update from {from_agent}")
        # 处理市场数据更新逻辑

    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态摘要"""
        return {
            "current_risk_level": self.current_risk_level.value,
            "active_alerts_count": len(self.active_alerts),
            "position_limits": {
                "max_position_size": self.position_limits.max_position_size,
                "max_total_exposure": self.position_limits.max_total_exposure,
                "max_leverage": self.position_limits.max_leverage,
                "max_positions_count": self.position_limits.max_positions_count
            },
            "risk_limits": {
                "max_daily_loss": self.risk_limits.max_daily_loss,
                "max_drawdown": self.risk_limits.max_drawdown,
                "min_margin_ratio": self.risk_limits.min_margin_ratio
            },
            "total_capital": self.total_capital,
            "last_update": datetime.utcnow().isoformat()
        }


def create_risk_management_agent(
    name: str = "risk_manager",
    total_capital: float = 100000.0,
    max_daily_loss: float = 0.05,
    max_drawdown: float = 0.15,
    max_position_size: float = 0.05,
    message_bus: Optional[MessageBus] = None
) -> RiskManagementAgent:
    """创建风险管理Agent的便捷函数"""
    
    config = AgentConfig(
        name=name,
        parameters={
            "total_capital": total_capital,
            "max_processing_time": 2.0,
            "heartbeat_interval": 30.0
        }
    )
    
    position_limits = PositionLimits(
        max_position_size=max_position_size,
        max_total_exposure=0.8,
        max_leverage=10,
        max_positions_count=10
    )
    
    risk_limits = RiskLimits(
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
        min_margin_ratio=0.2,
        stop_loss_ratio=0.02,
        take_profit_ratio=0.04
    )
    
    return RiskManagementAgent(
        config=config,
        message_bus=message_bus,
        position_limits=position_limits,
        risk_limits=risk_limits
    )