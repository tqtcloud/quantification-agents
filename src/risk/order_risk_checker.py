"""
订单风险检查系统
创建订单预执行风险评估、仓位限制和杠杆控制、相关性风险检查
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.models import (
    Order, Position, TradingState, RiskMetrics, MarketData,
    OrderSide, PositionSide, OrderType
)
from src.risk.risk_metrics_calculator import RiskMetricsCalculator, RiskConfig
from src.utils.logger import LoggerMixin


class RiskViolationType(Enum):
    """风险违规类型"""
    POSITION_SIZE_LIMIT = "position_size_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    MARGIN_REQUIREMENT = "margin_requirement"
    CORRELATION_RISK = "correlation_risk"
    CONCENTRATION_RISK = "concentration_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    VOLATILITY_RISK = "volatility_risk"
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    VAR_LIMIT = "var_limit"
    TIME_RESTRICTION = "time_restriction"
    SYMBOL_BLACKLIST = "symbol_blacklist"


@dataclass
class RiskViolation:
    """风险违规记录"""
    violation_type: RiskViolationType
    description: str
    current_value: float
    limit_value: float
    severity: str  # "low", "medium", "high", "critical"
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RiskCheckResult:
    """风险检查结果"""
    approved: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_order: Optional[Order] = None
    risk_score: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """风险限制配置"""
    # 仓位限制
    max_position_size_pct: float = 0.05     # 单个仓位最大占比
    max_total_exposure_pct: float = 0.8     # 总敞口最大占比
    max_leverage: float = 10.0              # 最大杠杆
    max_positions_count: int = 10           # 最大持仓数量
    
    # 风险限制
    max_daily_loss_pct: float = 0.05        # 每日最大亏损比例
    max_drawdown_pct: float = 0.15          # 最大回撤比例
    min_margin_ratio: float = 0.2           # 最小保证金比例
    max_var_pct: float = 0.08               # 最大VaR比例
    
    # 相关性和集中度
    max_correlation_exposure_pct: float = 0.3  # 相关性敞口限制
    max_sector_exposure_pct: float = 0.4       # 行业敞口限制
    min_diversification_ratio: float = 0.6     # 最小分散化比例
    
    # 流动性和波动率
    min_daily_volume: float = 1000000.0     # 最小日交易量
    max_volatility: float = 0.1             # 最大波动率
    max_spread_bps: float = 50.0            # 最大价差（基点）
    
    # 时间限制
    trading_hours_start: str = "00:00"      # 交易时间开始
    trading_hours_end: str = "23:59"        # 交易时间结束
    blocked_symbols: List[str] = field(default_factory=list)  # 禁止交易的符号


class OrderRiskChecker(LoggerMixin):
    """订单风险检查器"""
    
    def __init__(self, 
                 risk_limits: RiskLimits = None,
                 risk_calculator: RiskMetricsCalculator = None,
                 total_capital: float = 100000.0):
        
        self.risk_limits = risk_limits or RiskLimits()
        self.risk_calculator = risk_calculator or RiskMetricsCalculator()
        self.total_capital = total_capital
        
        # 历史数据
        self.order_history: List[Order] = []
        self.violation_history: List[RiskViolation] = []
        
        # 缓存
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
        self.volatility_cache: Dict[str, float] = {}
        self.liquidity_cache: Dict[str, Dict[str, float]] = {}
        
        self.log_info("OrderRiskChecker initialized")

    async def check_order_risk(self, 
                              order: Order, 
                              state: TradingState) -> RiskCheckResult:
        """检查订单风险"""
        
        try:
            violations = []
            warnings = []
            risk_score = 0.0
            
            # 1. 基本验证
            basic_violations = await self._check_basic_validations(order, state)
            violations.extend(basic_violations)
            
            # 2. 仓位大小检查
            position_violations = await self._check_position_limits(order, state)
            violations.extend(position_violations)
            
            # 3. 杠杆检查
            leverage_violations = await self._check_leverage_limits(order, state)
            violations.extend(leverage_violations)
            
            # 4. 保证金检查
            margin_violations = await self._check_margin_requirements(order, state)
            violations.extend(margin_violations)
            
            # 5. 相关性风险检查
            correlation_violations = await self._check_correlation_risk(order, state)
            violations.extend(correlation_violations)
            
            # 6. 集中度风险检查
            concentration_violations = await self._check_concentration_risk(order, state)
            violations.extend(concentration_violations)
            
            # 7. 流动性风险检查
            liquidity_violations = await self._check_liquidity_risk(order, state)
            violations.extend(liquidity_violations)
            
            # 8. 波动率风险检查
            volatility_violations = await self._check_volatility_risk(order, state)
            violations.extend(volatility_violations)
            
            # 9. 投资组合风险检查
            portfolio_violations = await self._check_portfolio_risk(order, state)
            violations.extend(portfolio_violations)
            
            # 10. 时间和符号限制检查
            restriction_violations = await self._check_restrictions(order, state)
            violations.extend(restriction_violations)
            
            # 计算风险分数
            risk_score = self._calculate_risk_score(violations)
            
            # 生成调整后的订单（如果需要）
            adjusted_order = await self._generate_adjusted_order(order, violations, state)
            
            # 确定是否批准
            critical_violations = [v for v in violations if v.severity == "critical"]
            high_violations = [v for v in violations if v.severity == "high"]
            
            approved = len(critical_violations) == 0 and len(high_violations) <= 1
            
            # 生成解释
            explanation = self._generate_explanation(violations, approved, risk_score)
            
            result = RiskCheckResult(
                approved=approved,
                violations=violations,
                warnings=warnings,
                adjusted_order=adjusted_order,
                risk_score=risk_score,
                explanation=explanation,
                metadata={
                    "check_timestamp": datetime.utcnow().isoformat(),
                    "total_capital": self.total_capital,
                    "violation_count": len(violations)
                }
            )
            
            # 记录结果
            if violations:
                self.violation_history.extend(violations)
                self.log_warning(f"Risk violations found for order {order.symbol}: {len(violations)} violations")
            
            self.log_debug(f"Risk check completed: approved={approved}, score={risk_score:.2f}")
            
            return result
            
        except Exception as e:
            self.log_error(f"Error in risk check: {e}")
            return RiskCheckResult(
                approved=False,
                violations=[RiskViolation(
                    violation_type=RiskViolationType.VAR_LIMIT,
                    description=f"Risk check failed: {e}",
                    current_value=0.0,
                    limit_value=0.0,
                    severity="critical",
                    recommendation="Manual review required"
                )],
                explanation="Risk check system error"
            )

    async def _check_basic_validations(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """基本验证检查"""
        violations = []
        
        # 订单数量检查
        if order.quantity <= 0:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE_LIMIT,
                description="Invalid order quantity",
                current_value=order.quantity,
                limit_value=0.0,
                severity="critical",
                recommendation="Use positive quantity"
            ))
        
        # 市场数据检查
        if order.symbol not in state.market_data:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.LIQUIDITY_RISK,
                description="No market data available",
                current_value=0.0,
                limit_value=1.0,
                severity="critical",
                recommendation="Wait for market data"
            ))
        
        return violations

    async def _check_position_limits(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查仓位限制"""
        violations = []
        
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return violations
        
        # 计算订单价值
        estimated_price = market_data.price
        order_value = order.quantity * estimated_price
        
        # 单个仓位大小检查
        max_position_value = self.total_capital * self.risk_limits.max_position_size_pct
        
        current_position = state.positions.get(order.symbol)
        current_value = 0
        if current_position:
            current_value = abs(current_position.quantity) * current_position.mark_price
        
        new_total_value = current_value + order_value
        
        if new_total_value > max_position_value:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE_LIMIT,
                description=f"Position size limit exceeded for {order.symbol}",
                current_value=new_total_value,
                limit_value=max_position_value,
                severity="high",
                recommendation=f"Reduce order size to max {max_position_value - current_value:.2f}"
            ))
        
        # 总敞口检查
        total_exposure = sum(
            abs(pos.quantity) * pos.mark_price 
            for pos in state.positions.values()
        ) + order_value
        
        max_total_exposure = self.total_capital * self.risk_limits.max_total_exposure_pct
        
        if total_exposure > max_total_exposure:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE_LIMIT,
                description="Total exposure limit exceeded",
                current_value=total_exposure,
                limit_value=max_total_exposure,
                severity="high",
                recommendation="Reduce total exposure"
            ))
        
        # 持仓数量检查
        current_positions = len([p for p in state.positions.values() if p.quantity != 0])
        if order.symbol not in state.positions and current_positions >= self.risk_limits.max_positions_count:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE_LIMIT,
                description="Maximum positions count exceeded",
                current_value=current_positions + 1,
                limit_value=self.risk_limits.max_positions_count,
                severity="medium",
                recommendation="Close existing positions first"
            ))
        
        return violations

    async def _check_leverage_limits(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查杠杆限制"""
        violations = []
        
        current_position = state.positions.get(order.symbol)
        if current_position and current_position.leverage > self.risk_limits.max_leverage:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.LEVERAGE_LIMIT,
                description=f"Leverage limit exceeded for {order.symbol}",
                current_value=current_position.leverage,
                limit_value=self.risk_limits.max_leverage,
                severity="high",
                recommendation=f"Reduce leverage to max {self.risk_limits.max_leverage}"
            ))
        
        return violations

    async def _check_margin_requirements(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查保证金要求"""
        violations = []
        
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return violations
        
        # 计算所需保证金
        estimated_price = market_data.price
        required_margin = (order.quantity * estimated_price) / self.risk_limits.max_leverage
        
        # 计算可用资金
        total_margin_used = sum(pos.margin for pos in state.positions.values())
        available_capital = self.total_capital - total_margin_used
        
        if required_margin > available_capital:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MARGIN_REQUIREMENT,
                description="Insufficient margin",
                current_value=available_capital,
                limit_value=required_margin,
                severity="critical",
                recommendation="Reduce order size or close existing positions"
            ))
        
        # 检查保证金比例
        total_value = sum(abs(pos.quantity) * pos.mark_price for pos in state.positions.values())
        if total_value > 0:
            margin_ratio = total_margin_used / total_value
            if margin_ratio < self.risk_limits.min_margin_ratio:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.MARGIN_REQUIREMENT,
                    description="Margin ratio too low",
                    current_value=margin_ratio,
                    limit_value=self.risk_limits.min_margin_ratio,
                    severity="medium",
                    recommendation="Increase margin or reduce positions"
                ))
        
        return violations

    async def _check_correlation_risk(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查相关性风险"""
        violations = []
        
        symbol = order.symbol
        market_data = state.market_data.get(symbol)
        
        if not market_data:
            return violations
        
        order_value = order.quantity * market_data.price
        max_correlated_exposure = self.total_capital * self.risk_limits.max_correlation_exposure_pct
        
        for pos_symbol, position in state.positions.items():
            if position.quantity == 0 or pos_symbol == symbol:
                continue
            
            # 获取相关性（这里使用简化的相关性假设）
            correlation = self._get_correlation(symbol, pos_symbol)
            
            if abs(correlation) > 0.7:  # 高相关性阈值
                combined_exposure = abs(position.quantity * position.mark_price) + order_value
                
                if combined_exposure > max_correlated_exposure:
                    violations.append(RiskViolation(
                        violation_type=RiskViolationType.CORRELATION_RISK,
                        description=f"High correlation exposure: {symbol} and {pos_symbol}",
                        current_value=combined_exposure,
                        limit_value=max_correlated_exposure,
                        severity="medium",
                        recommendation="Diversify holdings",
                        metadata={"correlation": correlation, "correlated_symbol": pos_symbol}
                    ))
        
        return violations

    async def _check_concentration_risk(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查集中度风险"""
        violations = []
        
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return violations
        
        # 计算新的投资组合集中度
        order_value = order.quantity * market_data.price
        total_portfolio_value = sum(
            abs(pos.quantity) * pos.mark_price 
            for pos in state.positions.values()
        ) + order_value
        
        if total_portfolio_value == 0:
            return violations
        
        current_position = state.positions.get(order.symbol)
        current_value = 0
        if current_position:
            current_value = abs(current_position.quantity) * current_position.mark_price
        
        new_position_value = current_value + order_value
        concentration = new_position_value / total_portfolio_value
        
        if concentration > 0.5:  # 单个仓位超过50%
            violations.append(RiskViolation(
                violation_type=RiskViolationType.CONCENTRATION_RISK,
                description=f"High concentration in {order.symbol}",
                current_value=concentration,
                limit_value=0.5,
                severity="medium",
                recommendation="Diversify holdings"
            ))
        
        return violations

    async def _check_liquidity_risk(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查流动性风险"""
        violations = []
        
        market_data = state.market_data.get(order.symbol)
        if not market_data:
            return violations
        
        # 检查日交易量
        if hasattr(market_data, 'daily_volume') and market_data.daily_volume < self.risk_limits.min_daily_volume:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.LIQUIDITY_RISK,
                description=f"Low liquidity for {order.symbol}",
                current_value=getattr(market_data, 'daily_volume', 0),
                limit_value=self.risk_limits.min_daily_volume,
                severity="medium",
                recommendation="Consider alternative symbols or smaller size"
            ))
        
        # 检查价差
        if market_data.bid > 0 and market_data.ask > 0:
            spread = market_data.ask - market_data.bid
            spread_bps = (spread / market_data.price) * 10000  # 基点
            
            if spread_bps > self.risk_limits.max_spread_bps:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.LIQUIDITY_RISK,
                    description=f"Wide spread for {order.symbol}",
                    current_value=spread_bps,
                    limit_value=self.risk_limits.max_spread_bps,
                    severity="low",
                    recommendation="Use limit orders to control execution price"
                ))
        
        return violations

    async def _check_volatility_risk(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查波动率风险"""
        violations = []
        
        # 简化的波动率检查（实际应该从历史数据计算）
        symbol_volatility = self._get_volatility(order.symbol)
        
        if symbol_volatility > self.risk_limits.max_volatility:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.VOLATILITY_RISK,
                description=f"High volatility for {order.symbol}",
                current_value=symbol_volatility,
                limit_value=self.risk_limits.max_volatility,
                severity="low",
                recommendation="Consider smaller position size or wait for lower volatility"
            ))
        
        return violations

    async def _check_portfolio_risk(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查投资组合风险"""
        violations = []
        
        if not state.risk_metrics:
            return violations
        
        # 检查回撤限制
        if state.risk_metrics.current_drawdown > self.risk_limits.max_drawdown_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.DRAWDOWN_LIMIT,
                description="Portfolio drawdown limit exceeded",
                current_value=state.risk_metrics.current_drawdown,
                limit_value=self.risk_limits.max_drawdown_pct,
                severity="high",
                recommendation="Stop new trades until recovery"
            ))
        
        # 检查每日亏损限制
        daily_loss_pct = abs(state.risk_metrics.daily_pnl) / self.total_capital
        if state.risk_metrics.daily_pnl < 0 and daily_loss_pct > self.risk_limits.max_daily_loss_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.DAILY_LOSS_LIMIT,
                description="Daily loss limit exceeded",
                current_value=daily_loss_pct,
                limit_value=self.risk_limits.max_daily_loss_pct,
                severity="high",
                recommendation="Stop trading for today"
            ))
        
        # 检查VaR限制
        var_pct = abs(state.risk_metrics.var_95) / self.total_capital
        if var_pct > self.risk_limits.max_var_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.VAR_LIMIT,
                description="Portfolio VaR limit exceeded",
                current_value=var_pct,
                limit_value=self.risk_limits.max_var_pct,
                severity="medium",
                recommendation="Reduce overall risk exposure"
            ))
        
        return violations

    async def _check_restrictions(self, order: Order, state: TradingState) -> List[RiskViolation]:
        """检查时间和符号限制"""
        violations = []
        
        # 检查禁止交易的符号
        if order.symbol in self.risk_limits.blocked_symbols:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.SYMBOL_BLACKLIST,
                description=f"Symbol {order.symbol} is blacklisted",
                current_value=1.0,
                limit_value=0.0,
                severity="critical",
                recommendation="Use alternative symbols"
            ))
        
        # 检查交易时间（简化实现）
        current_time = datetime.utcnow().time()
        start_time = datetime.strptime(self.risk_limits.trading_hours_start, "%H:%M").time()
        end_time = datetime.strptime(self.risk_limits.trading_hours_end, "%H:%M").time()
        
        if not (start_time <= current_time <= end_time):
            violations.append(RiskViolation(
                violation_type=RiskViolationType.TIME_RESTRICTION,
                description="Outside trading hours",
                current_value=current_time.hour * 60 + current_time.minute,
                limit_value=start_time.hour * 60 + start_time.minute,
                severity="medium",
                recommendation="Wait for trading hours"
            ))
        
        return violations

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """获取两个交易对的相关性"""
        # 简化实现，实际应该从历史数据计算
        correlation_map = {
            ("BTCUSDT", "ETHUSDT"): 0.8,
            ("ETHUSDT", "ADAUSDT"): 0.7,
            ("BTCUSDT", "LTCUSDT"): 0.75,
        }
        
        key = tuple(sorted([symbol1, symbol2]))
        return correlation_map.get(key, 0.3)

    def _get_volatility(self, symbol: str) -> float:
        """获取交易对的波动率"""
        # 简化实现，实际应该从历史价格计算
        volatility_map = {
            "BTCUSDT": 0.04,
            "ETHUSDT": 0.05,
            "ADAUSDT": 0.08,
            "DOGEUSDT": 0.12,
        }
        
        return volatility_map.get(symbol, 0.06)

    def _calculate_risk_score(self, violations: List[RiskViolation]) -> float:
        """计算风险分数"""
        severity_weights = {
            "low": 1.0,
            "medium": 2.5,
            "high": 5.0,
            "critical": 10.0
        }
        
        total_score = sum(severity_weights.get(v.severity, 1.0) for v in violations)
        return min(total_score, 100.0)  # 限制在0-100之间

    async def _generate_adjusted_order(self, 
                                     order: Order, 
                                     violations: List[RiskViolation],
                                     state: TradingState) -> Optional[Order]:
        """生成调整后的订单"""
        
        if not violations:
            return None
        
        adjusted_order = Order(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            reduce_only=order.reduce_only,
            close_position=order.close_position,
            position_side=order.position_side
        )
        
        # 根据违规类型调整订单
        for violation in violations:
            if violation.violation_type == RiskViolationType.POSITION_SIZE_LIMIT:
                # 调整数量
                if violation.limit_value > 0:
                    market_data = state.market_data.get(order.symbol)
                    if market_data:
                        max_quantity = violation.limit_value / market_data.price
                        adjusted_order.quantity = min(adjusted_order.quantity, max_quantity)
            
            elif violation.violation_type == RiskViolationType.MARGIN_REQUIREMENT:
                # 减少数量以满足保证金要求
                adjusted_order.quantity *= 0.8
        
        # 确保调整后的数量有效
        if adjusted_order.quantity <= 0:
            return None
        
        return adjusted_order

    def _generate_explanation(self, 
                            violations: List[RiskViolation], 
                            approved: bool, 
                            risk_score: float) -> str:
        """生成风险检查解释"""
        
        if not violations:
            return f"Order approved with low risk (score: {risk_score:.1f})"
        
        if approved:
            return f"Order approved with warnings (score: {risk_score:.1f}): {len(violations)} minor issues detected"
        else:
            critical_count = len([v for v in violations if v.severity == "critical"])
            high_count = len([v for v in violations if v.severity == "high"])
            
            return (f"Order rejected due to risk violations (score: {risk_score:.1f}): "
                   f"{critical_count} critical, {high_count} high severity issues")

    def get_risk_statistics(self) -> Dict[str, Any]:
        """获取风险统计信息"""
        return {
            "total_checks": len(self.order_history),
            "total_violations": len(self.violation_history),
            "violation_types": {
                vtype.value: len([v for v in self.violation_history if v.violation_type == vtype])
                for vtype in RiskViolationType
            },
            "current_limits": {
                "max_position_size_pct": self.risk_limits.max_position_size_pct,
                "max_leverage": self.risk_limits.max_leverage,
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_drawdown_pct": self.risk_limits.max_drawdown_pct
            }
        }