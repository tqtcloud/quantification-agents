"""
智能自动平仓管理系统

实现多策略自动平仓逻辑，包括目标盈利、止损、跟踪止损、时间止损等7种策略。
提供完整的仓位监控、风险控制和平仓决策功能。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import uuid
from concurrent.futures import ThreadPoolExecutor
import json

from .models import (
    PositionInfo, ClosingReason, ClosingAction, PositionCloseRequest, PositionCloseResult,
    ATRInfo, VolatilityInfo, CorrelationRisk
)
from .closing_strategies import (
    BaseClosingStrategy, ProfitTargetStrategy, StopLossStrategy, TrailingStopStrategy,
    TimeBasedStrategy, TechnicalReversalStrategy, SentimentStrategy, DynamicTrailingStrategy
)
from ..models.signals import MultiDimensionalSignal


logger = logging.getLogger(__name__)


@dataclass
class ClosingStrategyConfig:
    """平仓策略配置"""
    strategy_class: type
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 5


@dataclass 
class PositionMonitoringState:
    """仓位监控状态"""
    position_id: str
    last_check_time: datetime
    check_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    alert_count: int = 0
    
    def increment_check(self) -> None:
        """增加检查计数"""
        self.check_count += 1
        self.last_check_time = datetime.utcnow()
    
    def increment_error(self, error_msg: str) -> None:
        """增加错误计数"""
        self.error_count += 1
        self.last_error = error_msg
        logger.error(f"Position {self.position_id} error: {error_msg}")
    
    def increment_alert(self) -> None:
        """增加告警计数"""
        self.alert_count += 1


class AutoPositionCloser:
    """智能自动平仓管理器"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 custom_strategies: Optional[Dict[str, BaseClosingStrategy]] = None):
        """
        初始化自动平仓管理器
        
        Args:
            config: 配置参数
            custom_strategies: 自定义策略字典
        """
        self.config = config or self._get_default_config()
        
        # 仓位跟踪
        self.active_positions: Dict[str, PositionInfo] = {}
        self.monitoring_states: Dict[str, PositionMonitoringState] = {}
        
        # 策略管理
        self.strategies: Dict[str, BaseClosingStrategy] = {}
        self._initialize_strategies(custom_strategies)
        
        # 请求跟踪
        self.pending_requests: Dict[str, PositionCloseRequest] = {}
        self.completed_requests: Dict[str, PositionCloseResult] = {}
        
        # 运行状态
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="PositionCloser")
        
        # 统计数据
        self.total_positions_managed = 0
        self.total_positions_closed = 0
        self.total_profit_realized = 0.0
        self.total_loss_realized = 0.0
        
        # 市场数据缓存
        self._market_context_cache: Dict[str, Any] = {}
        self._last_context_update: Optional[datetime] = None
        
        logger.info(f"AutoPositionCloser initialized with {len(self.strategies)} strategies")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'monitoring_interval_seconds': 5,    # 监控间隔
            'max_concurrent_closes': 10,         # 最大并发平仓数
            'enable_emergency_stop': True,      # 启用紧急停止
            'emergency_loss_threshold': -10.0,  # 紧急止损阈值
            'position_timeout_hours': 48,       # 仓位超时时间
            'enable_statistics_logging': True,  # 启用统计日志
            'log_interval_minutes': 30,         # 日志间隔
            'enable_risk_alerts': True,         # 启用风险告警
            'max_daily_closes': 100,            # 每日最大平仓次数
            'correlation_check_enabled': True,   # 启用相关性检查
            'volatility_adjustment_enabled': True, # 启用波动率调整
        }
    
    def _initialize_strategies(self, custom_strategies: Optional[Dict[str, BaseClosingStrategy]]) -> None:
        """初始化平仓策略"""
        # 默认策略配置
        default_strategy_configs = {
            'profit_target': ClosingStrategyConfig(
                strategy_class=ProfitTargetStrategy,
                parameters={'target_profit_pct': 5.0, 'priority': 2},
                enabled=True
            ),
            'stop_loss': ClosingStrategyConfig(
                strategy_class=StopLossStrategy,
                parameters={'stop_loss_pct': -2.0, 'priority': 1},
                enabled=True
            ),
            'trailing_stop': ClosingStrategyConfig(
                strategy_class=TrailingStopStrategy,
                parameters={'trailing_distance_pct': 1.5, 'priority': 2},
                enabled=True
            ),
            'time_based': ClosingStrategyConfig(
                strategy_class=TimeBasedStrategy,
                parameters={'max_hold_hours': 24, 'priority': 4},
                enabled=True
            ),
            'technical_reversal': ClosingStrategyConfig(
                strategy_class=TechnicalReversalStrategy,
                parameters={'reversal_threshold': -0.5, 'priority': 3},
                enabled=True
            ),
            'sentiment_change': ClosingStrategyConfig(
                strategy_class=SentimentStrategy,
                parameters={'sentiment_change_threshold': 0.4, 'priority': 5},
                enabled=True
            ),
            'dynamic_trailing': ClosingStrategyConfig(
                strategy_class=DynamicTrailingStrategy,
                parameters={'base_trailing_pct': 2.0, 'priority': 3},
                enabled=True
            )
        }
        
        # 从配置中获取策略设置
        strategy_configs = self.config.get('strategies', default_strategy_configs)
        
        # 初始化策略实例
        for strategy_name, strategy_config in strategy_configs.items():
            # 处理不同格式的配置
            if isinstance(strategy_config, ClosingStrategyConfig):
                config_obj = strategy_config
            elif isinstance(strategy_config, dict):
                # 兼容字典配置格式
                config_obj = ClosingStrategyConfig(
                    strategy_class=strategy_config.get('strategy_class'),
                    parameters=strategy_config.get('parameters', {}),
                    enabled=strategy_config.get('enabled', True),
                    priority=strategy_config.get('priority', 5)
                )
            else:
                logger.error(f"Invalid strategy config format for {strategy_name}")
                continue
            
            if config_obj.enabled:
                try:
                    if isinstance(config_obj.strategy_class, str):
                        # 如果是字符串，需要从模块中导入
                        strategy_class = globals().get(config_obj.strategy_class)
                        if not strategy_class:
                            logger.error(f"Strategy class {config_obj.strategy_class} not found")
                            continue
                    else:
                        strategy_class = config_obj.strategy_class
                    
                    strategy_instance = strategy_class(config_obj.parameters)
                    self.strategies[strategy_name] = strategy_instance
                    logger.info(f"Initialized strategy: {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize strategy {strategy_name}: {e}")
        
        # 添加自定义策略
        if custom_strategies:
            for name, strategy in custom_strategies.items():
                self.strategies[name] = strategy
                logger.info(f"Added custom strategy: {name}")
        
        # 按优先级排序策略
        self.strategies = dict(sorted(
            self.strategies.items(), 
            key=lambda item: item[1].priority
        ))
    
    async def start(self) -> None:
        """启动自动平仓监控"""
        if self.is_running:
            logger.warning("AutoPositionCloser is already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("AutoPositionCloser started")
    
    async def stop(self) -> None:
        """停止自动平仓监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 等待所有挂起的平仓请求完成
        if self.pending_requests:
            logger.info(f"Waiting for {len(self.pending_requests)} pending close requests to complete")
            await asyncio.sleep(2)  # 给一些时间完成
        
        self._executor.shutdown(wait=True)
        logger.info("AutoPositionCloser stopped")
    
    def add_position(self, position: PositionInfo) -> None:
        """添加仓位进行监控"""
        self.active_positions[position.position_id] = position
        self.monitoring_states[position.position_id] = PositionMonitoringState(
            position_id=position.position_id,
            last_check_time=datetime.utcnow()
        )
        
        self.total_positions_managed += 1
        logger.info(f"Added position {position.position_id} for monitoring ({position.symbol}, {position.side})")
    
    def remove_position(self, position_id: str) -> Optional[PositionInfo]:
        """移除仓位监控"""
        position = self.active_positions.pop(position_id, None)
        self.monitoring_states.pop(position_id, None)
        
        if position:
            logger.info(f"Removed position {position_id} from monitoring")
        
        return position
    
    def update_position_price(self, position_id: str, new_price: float) -> bool:
        """更新仓位价格"""
        position = self.active_positions.get(position_id)
        if position:
            position.update_price(new_price)
            return True
        return False
    
    def get_position(self, position_id: str) -> Optional[PositionInfo]:
        """获取仓位信息"""
        return self.active_positions.get(position_id)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """获取所有活跃仓位"""
        return self.active_positions.copy()
    
    async def manage_position(
        self, 
        position_id: str, 
        current_price: float,
        signal: Optional[MultiDimensionalSignal] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[PositionCloseRequest]:
        """
        管理单个仓位，检查是否需要平仓
        
        Args:
            position_id: 仓位ID
            current_price: 当前价格
            signal: 多维度信号
            market_context: 市场环境上下文
            
        Returns:
            PositionCloseRequest: 平仓请求，如果不需要平仓则返回None
        """
        position = self.active_positions.get(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found")
            return None
        
        # 更新价格
        position.update_price(current_price)
        
        # 更新监控状态
        monitoring_state = self.monitoring_states.get(position_id)
        if monitoring_state:
            monitoring_state.increment_check()
        
        try:
            # 检查紧急止损
            emergency_request = await self._check_emergency_stop(position)
            if emergency_request:
                return emergency_request
            
            # 按优先级检查各策略
            for strategy_name, strategy in self.strategies.items():
                if not strategy.enabled:
                    continue
                
                try:
                    close_request = await strategy.should_close_position(
                        position=position,
                        current_signal=signal,
                        market_context=market_context
                    )
                    
                    if close_request:
                        logger.info(f"Strategy {strategy_name} triggered close for position {position_id}")
                        
                        # 添加策略信息到元数据
                        close_request.metadata['triggered_strategy'] = strategy_name
                        close_request.metadata['strategy_priority'] = strategy.priority
                        
                        return close_request
                
                except Exception as e:
                    error_msg = f"Strategy {strategy_name} error: {e}"
                    if monitoring_state:
                        monitoring_state.increment_error(error_msg)
                    logger.error(error_msg, exc_info=True)
            
            return None
            
        except Exception as e:
            error_msg = f"Position management error for {position_id}: {e}"
            if monitoring_state:
                monitoring_state.increment_error(error_msg)
            logger.error(error_msg, exc_info=True)
            return None
    
    async def _check_emergency_stop(self, position: PositionInfo) -> Optional[PositionCloseRequest]:
        """检查紧急止损条件"""
        if not self.config['enable_emergency_stop']:
            return None
        
        emergency_threshold = self.config['emergency_loss_threshold']
        
        if position.unrealized_pnl_pct <= emergency_threshold:
            logger.critical(f"Emergency stop triggered for position {position.position_id}: "
                          f"{position.unrealized_pnl_pct:.2f}% <= {emergency_threshold}%")
            
            return PositionCloseRequest(
                position_id=position.position_id,
                closing_reason=ClosingReason.EMERGENCY,
                action=ClosingAction.FULL_CLOSE,
                quantity_to_close=position.quantity,
                urgency="emergency",
                metadata={
                    'emergency_loss_pct': position.unrealized_pnl_pct,
                    'emergency_threshold': emergency_threshold,
                    'trigger_strategy': 'emergency_stop'
                }
            )
        
        return None
    
    async def execute_close_request(
        self, 
        request: PositionCloseRequest,
        execution_callback: Optional[callable] = None
    ) -> PositionCloseResult:
        """
        执行平仓请求
        
        Args:
            request: 平仓请求
            execution_callback: 执行回调函数，签名为 async (request) -> PositionCloseResult
            
        Returns:
            PositionCloseResult: 平仓结果
        """
        request_id = str(uuid.uuid4())
        self.pending_requests[request_id] = request
        
        try:
            logger.info(f"Executing close request {request_id} for position {request.position_id}")
            
            # 如果提供了自定义执行回调
            if execution_callback:
                result = await execution_callback(request)
            else:
                # 使用默认的模拟执行
                result = await self._simulate_close_execution(request)
            
            # 设置请求ID
            result.request_id = request_id
            
            # 更新统计数据
            if result.success:
                self.total_positions_closed += 1
                if result.realized_pnl > 0:
                    self.total_profit_realized += result.realized_pnl
                else:
                    self.total_loss_realized += abs(result.realized_pnl)
            
            # 如果是全仓平仓且成功，移除仓位监控
            if result.success and result.is_full_close:
                self.remove_position(request.position_id)
            elif result.success:
                # 部分平仓，更新仓位数量
                position = self.active_positions.get(request.position_id)
                if position:
                    position.quantity -= result.actual_quantity_closed
            
            self.completed_requests[request_id] = result
            return result
            
        except Exception as e:
            error_result = PositionCloseResult(
                request_id=request_id,
                position_id=request.position_id,
                success=False,
                actual_quantity_closed=0.0,
                close_price=0.0,
                realized_pnl=0.0,
                closing_reason=request.closing_reason,
                close_time=datetime.utcnow(),
                error_message=str(e)
            )
            
            self.completed_requests[request_id] = error_result
            logger.error(f"Failed to execute close request {request_id}: {e}")
            return error_result
            
        finally:
            self.pending_requests.pop(request_id, None)
    
    async def _simulate_close_execution(self, request: PositionCloseRequest) -> PositionCloseResult:
        """模拟平仓执行（用于测试和演示）"""
        position = self.active_positions.get(request.position_id)
        if not position:
            raise ValueError(f"Position {request.position_id} not found")
        
        # 模拟执行延迟
        await asyncio.sleep(0.1)
        
        # 模拟滑点
        slippage = 0.001  # 0.1%
        close_price = position.current_price * (1 - slippage if position.is_long else 1 + slippage)
        
        # 计算已实现盈亏
        if position.is_long:
            realized_pnl = (close_price - position.entry_price) * request.quantity_to_close
        else:
            realized_pnl = (position.entry_price - close_price) * request.quantity_to_close
        
        # 判断是否全仓平仓
        is_full_close = request.quantity_to_close >= position.quantity
        remaining_quantity = max(0, position.quantity - request.quantity_to_close)
        
        return PositionCloseResult(
            request_id="",  # 将在外部设置
            position_id=request.position_id,
            success=True,
            actual_quantity_closed=request.quantity_to_close,
            close_price=close_price,
            realized_pnl=realized_pnl,
            closing_reason=request.closing_reason,
            close_time=datetime.utcnow(),
            metadata={
                'is_full_close': is_full_close,
                'remaining_quantity': remaining_quantity,
                'slippage': slippage,
                'execution_type': 'simulated'
            }
        )
    
    async def _monitoring_loop(self) -> None:
        """主监控循环"""
        logger.info("Started position monitoring loop")
        
        while self.is_running:
            try:
                await self._monitor_all_positions()
                
                # 记录统计信息
                if self.config['enable_statistics_logging']:
                    await self._log_statistics()
                
                # 等待下一次检查
                interval = self.config['monitoring_interval_seconds']
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(5)  # 出错后短暂等待
    
    async def _monitor_all_positions(self) -> None:
        """监控所有活跃仓位"""
        if not self.active_positions:
            return
        
        logger.debug(f"Monitoring {len(self.active_positions)} positions")
        
        # 批量处理仓位
        tasks = []
        for position_id in list(self.active_positions.keys()):
            task = asyncio.create_task(self._monitor_single_position(position_id))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _monitor_single_position(self, position_id: str) -> None:
        """监控单个仓位"""
        try:
            position = self.active_positions.get(position_id)
            if not position:
                return
            
            # 检查仓位超时
            timeout_hours = self.config['position_timeout_hours']
            if position.hold_duration > timedelta(hours=timeout_hours):
                logger.warning(f"Position {position_id} has exceeded timeout ({timeout_hours} hours)")
                
                # 创建超时平仓请求
                timeout_request = PositionCloseRequest(
                    position_id=position_id,
                    closing_reason=ClosingReason.TIME_BASED,
                    action=ClosingAction.FULL_CLOSE,
                    quantity_to_close=position.quantity,
                    urgency="high",
                    metadata={'timeout_trigger': True, 'timeout_hours': timeout_hours}
                )
                
                # 执行平仓
                await self.execute_close_request(timeout_request)
            
        except Exception as e:
            logger.error(f"Error monitoring position {position_id}: {e}", exc_info=True)
    
    async def _log_statistics(self) -> None:
        """记录统计信息"""
        stats = self.get_statistics()
        logger.info(f"Position Statistics: {json.dumps(stats, indent=2, default=str)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_count = len(self.active_positions)
        pending_count = len(self.pending_requests)
        
        # 按策略统计
        strategy_stats = {}
        for name, strategy in self.strategies.items():
            strategy_stats[name] = strategy.get_statistics()
        
        # 仓位统计
        position_stats = {
            'long_positions': 0,
            'short_positions': 0,
            'total_unrealized_pnl': 0.0,
            'profitable_positions': 0
        }
        
        for position in self.active_positions.values():
            if position.is_long:
                position_stats['long_positions'] += 1
            else:
                position_stats['short_positions'] += 1
                
            position_stats['total_unrealized_pnl'] += position.unrealized_pnl
            
            if position.is_profitable:
                position_stats['profitable_positions'] += 1
        
        return {
            'active_positions': active_count,
            'pending_requests': pending_count,
            'total_managed': self.total_positions_managed,
            'total_closed': self.total_positions_closed,
            'total_profit': self.total_profit_realized,
            'total_loss': self.total_loss_realized,
            'net_pnl': self.total_profit_realized - self.total_loss_realized,
            'close_success_rate': (self.total_positions_closed / max(self.total_positions_managed, 1)),
            'position_stats': position_stats,
            'strategy_stats': strategy_stats,
            'enabled_strategies': [name for name, strategy in self.strategies.items() if strategy.enabled],
            'is_running': self.is_running,
            'last_update': datetime.utcnow()
        }
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """启用策略"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            strategy.enable()
            logger.info(f"Enabled strategy: {strategy_name}")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """禁用策略"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            strategy.disable()
            logger.info(f"Disabled strategy: {strategy_name}")
            return True
        return False
    
    def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> bool:
        """更新策略参数"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            strategy.update_parameters(parameters)
            logger.info(f"Updated parameters for strategy {strategy_name}: {parameters}")
            return True
        return False
    
    def get_strategy_statistics(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """获取特定策略的统计信息"""
        strategy = self.strategies.get(strategy_name)
        if strategy:
            return strategy.get_statistics()
        return None
    
    async def force_close_position(
        self, 
        position_id: str, 
        reason: str = "manual",
        execution_callback: Optional[callable] = None
    ) -> Optional[PositionCloseResult]:
        """强制平仓指定仓位"""
        position = self.active_positions.get(position_id)
        if not position:
            logger.error(f"Cannot force close: position {position_id} not found")
            return None
        
        force_request = PositionCloseRequest(
            position_id=position_id,
            closing_reason=ClosingReason.MANUAL,
            action=ClosingAction.FULL_CLOSE,
            quantity_to_close=position.quantity,
            urgency="high",
            metadata={'force_close': True, 'reason': reason}
        )
        
        result = await self.execute_close_request(force_request, execution_callback)
        logger.info(f"Force closed position {position_id} with result: {result.success}")
        return result
    
    async def force_close_all_positions(self, execution_callback: Optional[callable] = None) -> List[PositionCloseResult]:
        """强制平仓所有仓位"""
        results = []
        
        # 创建所有仓位的平仓任务
        tasks = []
        for position_id in list(self.active_positions.keys()):
            task = asyncio.create_task(
                self.force_close_position(position_id, "force_close_all", execution_callback)
            )
            tasks.append(task)
        
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, PositionCloseResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Force close error: {result}")
        
        logger.info(f"Force closed {len(results)} positions")
        return results