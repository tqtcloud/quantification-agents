"""
仓位管理器

协调仓位监控、自动平仓、风险控制和数据收集的核心管理器。
整合多个模块提供统一的仓位管理接口。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .models import (
    PositionInfo, ClosingReason, ClosingAction, PositionCloseRequest, PositionCloseResult,
    ATRInfo, VolatilityInfo, CorrelationRisk
)
from .auto_position_closer import AutoPositionCloser
from ..models.signals import MultiDimensionalSignal


logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """风险指标"""
    portfolio_value: float = 0.0
    total_exposure: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0
    sharpe_ratio: float = 0.0
    correlation_risk_score: float = 0.0
    concentration_risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'portfolio_value': self.portfolio_value,
            'total_exposure': self.total_exposure,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'var_95': self.var_95,
            'sharpe_ratio': self.sharpe_ratio,
            'correlation_risk_score': self.correlation_risk_score,
            'concentration_risk_score': self.concentration_risk_score
        }


@dataclass
class MarketDataProvider:
    """市场数据提供器接口"""
    get_current_price: Callable[[str], float]
    get_atr_info: Callable[[str], ATRInfo]
    get_volatility_info: Callable[[str], VolatilityInfo]
    get_correlation_matrix: Callable[[List[str]], Dict[str, Dict[str, float]]]


class PositionManager:
    """仓位管理器"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 market_data_provider: Optional[MarketDataProvider] = None):
        """
        初始化仓位管理器
        
        Args:
            config: 配置参数
            market_data_provider: 市场数据提供器
        """
        self.config = config or self._get_default_config()
        self.market_data_provider = market_data_provider
        
        # 初始化自动平仓器
        closer_config = self.config.get('auto_closer', {})
        self.auto_closer = AutoPositionCloser(closer_config)
        
        # 仓位数据跟踪
        self.position_history: Dict[str, List[Dict[str, Any]]] = {}
        self.closed_positions: Dict[str, PositionCloseResult] = {}
        
        # 风险管理
        self.risk_metrics = RiskMetrics()
        self.risk_alerts: List[Dict[str, Any]] = []
        self.max_positions = self.config.get('max_positions', 20)
        self.max_exposure_per_symbol = self.config.get('max_exposure_per_symbol', 0.1)
        
        # 数据收集和分析
        self.performance_data: List[Dict[str, Any]] = []
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="PositionManager")
        
        # 任务管理
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # 回调函数
        self.position_opened_callbacks: List[Callable] = []
        self.position_closed_callbacks: List[Callable] = []
        self.risk_alert_callbacks: List[Callable] = []
        
        logger.info("PositionManager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_positions': 20,
            'max_exposure_per_symbol': 0.1,
            'risk_check_interval_seconds': 30,
            'performance_update_interval_seconds': 60,
            'correlation_update_interval_minutes': 15,
            'enable_risk_monitoring': True,
            'enable_performance_tracking': True,
            'auto_rebalance_enabled': False,
            'emergency_stop_enabled': True,
            'max_daily_loss_pct': -5.0,
            'max_portfolio_drawdown_pct': -10.0,
            'correlation_threshold': 0.7,
            'concentration_threshold': 0.3,
            'auto_closer': {
                'monitoring_interval_seconds': 5,
                'enable_emergency_stop': True,
                'emergency_loss_threshold': -8.0
            }
        }
    
    async def start(self) -> None:
        """启动仓位管理器"""
        if self.is_running:
            logger.warning("PositionManager is already running")
            return
        
        self.is_running = True
        
        # 启动自动平仓器
        await self.auto_closer.start()
        
        # 启动监控任务
        if self.config['enable_risk_monitoring']:
            risk_task = asyncio.create_task(self._risk_monitoring_loop())
            self.tasks.append(risk_task)
        
        if self.config['enable_performance_tracking']:
            perf_task = asyncio.create_task(self._performance_tracking_loop())
            self.tasks.append(perf_task)
        
        correlation_task = asyncio.create_task(self._correlation_monitoring_loop())
        self.tasks.append(correlation_task)
        
        logger.info("PositionManager started with all monitoring tasks")
    
    async def stop(self) -> None:
        """停止仓位管理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止自动平仓器
        await self.auto_closer.stop()
        
        # 取消所有任务
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
        self._executor.shutdown(wait=True)
        
        logger.info("PositionManager stopped")
    
    async def open_position(
        self, 
        symbol: str,
        entry_price: float,
        quantity: float,
        side: str,
        signal: Optional[MultiDimensionalSignal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        开仓
        
        Args:
            symbol: 交易标的
            entry_price: 入场价格
            quantity: 数量
            side: 方向 ('long' or 'short')
            signal: 信号数据
            metadata: 额外元数据
            
        Returns:
            str: 仓位ID，如果开仓失败则返回None
        """
        # 风险检查
        if not await self._pre_position_risk_check(symbol, quantity, side):
            logger.warning(f"Risk check failed for opening position: {symbol} {side} {quantity}")
            return None
        
        # 创建仓位
        position_id = f"{symbol}_{side}_{uuid.uuid4().hex[:8]}"
        
        position = PositionInfo(
            position_id=position_id,
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            side=side,
            entry_time=datetime.utcnow(),
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            metadata=metadata or {}
        )
        
        # 如果有信号数据，设置止损止盈
        if signal:
            if side == 'long':
                position.stop_loss = signal.primary_signal.stop_loss
                position.take_profit = signal.primary_signal.target_price
            else:
                position.stop_loss = signal.primary_signal.stop_loss
                position.take_profit = signal.primary_signal.target_price
            
            # 添加信号相关元数据
            position.metadata.update({
                'signal_confidence': signal.overall_confidence,
                'signal_quality': signal.signal_quality_score,
                'risk_reward_ratio': signal.risk_reward_ratio
            })
        
        # 添加到自动平仓监控
        self.auto_closer.add_position(position)
        
        # 记录历史
        self._record_position_event(position_id, 'opened', {
            'entry_price': entry_price,
            'quantity': quantity,
            'side': side,
            'signal_data': signal.primary_signal.__dict__ if signal else None
        })
        
        # 触发回调
        for callback in self.position_opened_callbacks:
            try:
                await callback(position)
            except Exception as e:
                logger.error(f"Position opened callback error: {e}")
        
        logger.info(f"Opened position: {position_id} ({symbol} {side} {quantity}@{entry_price})")
        return position_id
    
    async def close_position(
        self, 
        position_id: str,
        quantity: Optional[float] = None,
        reason: str = "manual",
        execution_callback: Optional[Callable] = None
    ) -> Optional[PositionCloseResult]:
        """
        手动平仓
        
        Args:
            position_id: 仓位ID
            quantity: 平仓数量，None表示全仓
            reason: 平仓原因
            execution_callback: 执行回调
            
        Returns:
            PositionCloseResult: 平仓结果
        """
        position = self.auto_closer.get_position(position_id)
        if not position:
            logger.error(f"Position {position_id} not found for closing")
            return None
        
        close_quantity = quantity or position.quantity
        action = ClosingAction.FULL_CLOSE if close_quantity >= position.quantity else ClosingAction.PARTIAL_CLOSE
        
        close_request = PositionCloseRequest(
            position_id=position_id,
            closing_reason=ClosingReason.MANUAL,
            action=action,
            quantity_to_close=close_quantity,
            urgency="normal",
            metadata={'manual_reason': reason}
        )
        
        result = await self.auto_closer.execute_close_request(close_request, execution_callback)
        
        if result.success:
            # 记录到历史
            self.closed_positions[position_id] = result
            
            # 记录事件
            self._record_position_event(position_id, 'closed', {
                'close_price': result.close_price,
                'quantity_closed': result.actual_quantity_closed,
                'realized_pnl': result.realized_pnl,
                'reason': reason
            })
            
            # 触发回调
            for callback in self.position_closed_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    logger.error(f"Position closed callback error: {e}")
        
        return result
    
    async def update_position_prices(self, price_data: Dict[str, float]) -> None:
        """批量更新仓位价格"""
        positions = self.auto_closer.get_all_positions()
        
        updated_count = 0
        for position_id, position in positions.items():
            symbol = position.symbol
            if symbol in price_data:
                new_price = price_data[symbol]
                if self.auto_closer.update_position_price(position_id, new_price):
                    updated_count += 1
        
        if updated_count > 0:
            logger.debug(f"Updated prices for {updated_count} positions")
    
    async def run_position_monitoring(
        self, 
        signal_data: Optional[Dict[str, MultiDimensionalSignal]] = None
    ) -> List[PositionCloseRequest]:
        """
        运行仓位监控检查
        
        Args:
            signal_data: 各标的的信号数据
            
        Returns:
            List[PositionCloseRequest]: 触发的平仓请求列表
        """
        positions = self.auto_closer.get_all_positions()
        close_requests = []
        
        # 准备市场环境数据
        market_context = await self._prepare_market_context(list(positions.keys()))
        
        # 检查每个仓位
        for position_id, position in positions.items():
            symbol = position.symbol
            current_signal = signal_data.get(symbol) if signal_data else None
            
            try:
                close_request = await self.auto_closer.manage_position(
                    position_id=position_id,
                    current_price=position.current_price,
                    signal=current_signal,
                    market_context=market_context.get(symbol)
                )
                
                if close_request:
                    close_requests.append(close_request)
                    
            except Exception as e:
                logger.error(f"Error monitoring position {position_id}: {e}")
        
        return close_requests
    
    async def _prepare_market_context(self, position_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """准备市场环境数据"""
        if not self.market_data_provider:
            return {}
        
        positions = self.auto_closer.get_all_positions()
        symbols = list(set(positions[pid].symbol for pid in position_ids if pid in positions))
        
        context = {}
        
        try:
            # 获取ATR信息
            for symbol in symbols:
                try:
                    atr_info = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self.market_data_provider.get_atr_info, symbol
                    )
                    
                    volatility_info = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self.market_data_provider.get_volatility_info, symbol
                    )
                    
                    context[symbol] = {
                        'atr_info': atr_info,
                        'volatility_info': volatility_info
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get market context for {symbol}: {e}")
            
            # 获取相关性数据
            if len(symbols) > 1:
                try:
                    correlation_matrix = await asyncio.get_event_loop().run_in_executor(
                        self._executor, self.market_data_provider.get_correlation_matrix, symbols
                    )
                    
                    for symbol in symbols:
                        if symbol in context and symbol in correlation_matrix:
                            correlations = correlation_matrix[symbol]
                            
                            correlation_risk = CorrelationRisk(symbol=symbol, correlations=correlations)
                            for other_symbol, corr in correlations.items():
                                correlation_risk.add_correlation(other_symbol, corr)
                            
                            context[symbol]['correlation_risk'] = correlation_risk
                            
                except Exception as e:
                    logger.warning(f"Failed to get correlation data: {e}")
        
        except Exception as e:
            logger.error(f"Error preparing market context: {e}")
        
        return context
    
    async def _pre_position_risk_check(self, symbol: str, quantity: float, side: str) -> bool:
        """开仓前风险检查"""
        positions = self.auto_closer.get_all_positions()
        
        # 检查最大仓位数限制
        if len(positions) >= self.max_positions:
            logger.warning(f"Maximum positions limit reached: {len(positions)}/{self.max_positions}")
            return False
        
        # 检查单标的敞口限制
        symbol_exposure = 0.0
        for position in positions.values():
            if position.symbol == symbol:
                symbol_exposure += abs(position.quantity * position.current_price)
        
        # 模拟新仓位价格（这里需要实际价格数据）
        estimated_price = 100.0  # 应该从市场数据获取
        new_exposure = quantity * estimated_price
        
        total_symbol_exposure = symbol_exposure + new_exposure
        portfolio_value = self.risk_metrics.portfolio_value or 100000.0  # 默认组合价值
        
        exposure_ratio = total_symbol_exposure / portfolio_value
        if exposure_ratio > self.max_exposure_per_symbol:
            logger.warning(f"Symbol exposure limit exceeded: {exposure_ratio:.2%} > {self.max_exposure_per_symbol:.2%}")
            return False
        
        return True
    
    async def _risk_monitoring_loop(self) -> None:
        """风险监控循环"""
        logger.info("Started risk monitoring loop")
        
        while self.is_running:
            try:
                await self._update_risk_metrics()
                await self._check_risk_alerts()
                
                interval = self.config['risk_check_interval_seconds']
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}", exc_info=True)
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self) -> None:
        """业绩跟踪循环"""
        logger.info("Started performance tracking loop")
        
        while self.is_running:
            try:
                await self._update_performance_data()
                
                interval = self.config['performance_update_interval_seconds']
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _correlation_monitoring_loop(self) -> None:
        """相关性监控循环"""
        logger.info("Started correlation monitoring loop")
        
        while self.is_running:
            try:
                await self._update_correlation_risk()
                
                interval = self.config['correlation_update_interval_minutes'] * 60
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Correlation monitoring error: {e}", exc_info=True)
                await asyncio.sleep(300)
    
    async def _update_risk_metrics(self) -> None:
        """更新风险指标"""
        positions = self.auto_closer.get_all_positions()
        
        if not positions:
            return
        
        # 计算组合价值和敞口
        total_value = 0.0
        total_exposure = 0.0
        unrealized_pnls = []
        
        for position in positions.values():
            position_value = position.quantity * position.current_price
            total_value += position_value
            total_exposure += abs(position_value)
            unrealized_pnls.append(position.unrealized_pnl)
        
        self.risk_metrics.portfolio_value = total_value
        self.risk_metrics.total_exposure = total_exposure
        
        if unrealized_pnls:
            # 计算VaR (简化版)
            pnl_array = np.array(unrealized_pnls)
            self.risk_metrics.var_95 = float(np.percentile(pnl_array, 5))
            
            # 更新回撤
            total_unrealized = float(np.sum(pnl_array))
            if total_unrealized < 0:
                self.risk_metrics.current_drawdown = abs(total_unrealized) / max(total_value, 1)
                self.risk_metrics.max_drawdown = max(
                    self.risk_metrics.max_drawdown, 
                    self.risk_metrics.current_drawdown
                )
    
    async def _check_risk_alerts(self) -> None:
        """检查风险告警"""
        # 检查最大回撤
        max_drawdown_pct = self.config['max_portfolio_drawdown_pct']
        if self.risk_metrics.current_drawdown * 100 <= max_drawdown_pct:
            await self._trigger_risk_alert(
                'max_drawdown_exceeded',
                f"Portfolio drawdown {self.risk_metrics.current_drawdown:.2%} exceeds limit {max_drawdown_pct}%"
            )
        
        # 检查日亏损限制
        # TODO: 实现日亏损计算逻辑
        
        # 检查集中度风险
        positions = self.auto_closer.get_all_positions()
        if positions:
            symbol_exposures = {}
            total_exposure = self.risk_metrics.total_exposure
            
            for position in positions.values():
                symbol = position.symbol
                exposure = abs(position.quantity * position.current_price)
                symbol_exposures[symbol] = symbol_exposures.get(symbol, 0) + exposure
            
            max_symbol_exposure = max(symbol_exposures.values()) if symbol_exposures else 0
            concentration_ratio = max_symbol_exposure / max(total_exposure, 1)
            
            self.risk_metrics.concentration_risk_score = concentration_ratio
            
            concentration_threshold = self.config['concentration_threshold']
            if concentration_ratio > concentration_threshold:
                await self._trigger_risk_alert(
                    'concentration_risk',
                    f"Concentration risk {concentration_ratio:.2%} exceeds threshold {concentration_threshold:.2%}"
                )
    
    async def _update_correlation_risk(self) -> None:
        """更新相关性风险"""
        positions = self.auto_closer.get_all_positions()
        symbols = list(set(p.symbol for p in positions.values()))
        
        if len(symbols) < 2 or not self.market_data_provider:
            return
        
        try:
            correlation_matrix = await asyncio.get_event_loop().run_in_executor(
                self._executor, self.market_data_provider.get_correlation_matrix, symbols
            )
            
            # 计算相关性风险分数
            correlations = []
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j and symbol1 in correlation_matrix and symbol2 in correlation_matrix[symbol1]:
                        correlations.append(abs(correlation_matrix[symbol1][symbol2]))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                max_correlation = max(correlations)
                
                self.risk_metrics.correlation_risk_score = max_correlation
                
                correlation_threshold = self.config['correlation_threshold']
                if max_correlation > correlation_threshold:
                    await self._trigger_risk_alert(
                        'high_correlation',
                        f"Maximum correlation {max_correlation:.3f} exceeds threshold {correlation_threshold:.3f}"
                    )
        
        except Exception as e:
            logger.error(f"Correlation risk update failed: {e}")
    
    async def _update_performance_data(self) -> None:
        """更新业绩数据"""
        stats = self.auto_closer.get_statistics()
        risk_metrics = self.risk_metrics.to_dict()
        
        performance_record = {
            'timestamp': datetime.utcnow(),
            'positions_stats': stats,
            'risk_metrics': risk_metrics,
            'active_positions': len(self.auto_closer.get_all_positions())
        }
        
        self.performance_data.append(performance_record)
        
        # 保持最近的1000条记录
        if len(self.performance_data) > 1000:
            self.performance_data = self.performance_data[-1000:]
    
    async def _trigger_risk_alert(self, alert_type: str, message: str) -> None:
        """触发风险告警"""
        alert = {
            'alert_id': str(uuid.uuid4()),
            'alert_type': alert_type,
            'message': message,
            'timestamp': datetime.utcnow(),
            'risk_metrics': self.risk_metrics.to_dict()
        }
        
        self.risk_alerts.append(alert)
        
        # 保持最近100个告警
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]
        
        # 触发回调
        for callback in self.risk_alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Risk alert callback error: {e}")
        
        logger.warning(f"Risk Alert [{alert_type}]: {message}")
    
    def _record_position_event(self, position_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """记录仓位事件"""
        if position_id not in self.position_history:
            self.position_history[position_id] = []
        
        event = {
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'data': data
        }
        
        self.position_history[position_id].append(event)
    
    # 回调管理方法
    def add_position_opened_callback(self, callback: Callable) -> None:
        """添加开仓回调"""
        self.position_opened_callbacks.append(callback)
    
    def add_position_closed_callback(self, callback: Callable) -> None:
        """添加平仓回调"""
        self.position_closed_callbacks.append(callback)
    
    def add_risk_alert_callback(self, callback: Callable) -> None:
        """添加风险告警回调"""
        self.risk_alert_callbacks.append(callback)
    
    # 查询方法
    def get_position_history(self, position_id: str) -> List[Dict[str, Any]]:
        """获取仓位历史"""
        return self.position_history.get(position_id, [])
    
    def get_risk_metrics(self) -> RiskMetrics:
        """获取风险指标"""
        return self.risk_metrics
    
    def get_risk_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取风险告警"""
        return self.risk_alerts[-limit:]
    
    def get_performance_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取业绩数据"""
        return self.performance_data[-limit:]
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        closer_stats = self.auto_closer.get_statistics()
        positions = self.auto_closer.get_all_positions()
        
        # 按标的分组统计
        symbol_stats = {}
        for position in positions.values():
            symbol = position.symbol
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'count': 0,
                    'total_quantity': 0.0,
                    'total_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'long_count': 0,
                    'short_count': 0
                }
            
            stats = symbol_stats[symbol]
            stats['count'] += 1
            stats['total_quantity'] += position.quantity
            stats['total_value'] += position.quantity * position.current_price
            stats['unrealized_pnl'] += position.unrealized_pnl
            
            if position.is_long:
                stats['long_count'] += 1
            else:
                stats['short_count'] += 1
        
        return {
            'auto_closer_stats': closer_stats,
            'risk_metrics': self.risk_metrics.to_dict(),
            'symbol_stats': symbol_stats,
            'recent_alerts': self.get_risk_alerts(5),
            'closed_positions_count': len(self.closed_positions),
            'is_running': self.is_running,
            'config': self.config,
            'last_update': datetime.utcnow()
        }