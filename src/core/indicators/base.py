"""
技术指标基础类和核心管理器
提供异步计算、缓存管理和性能监控等功能
"""

import asyncio
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from collections import deque, defaultdict

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    from ..mock_talib import MockTalib
    talib = MockTalib()

from src.utils.logger import LoggerMixin


class IndicatorType(Enum):
    """技术指标类型"""
    TREND = "trend"           # 趋势指标
    MOMENTUM = "momentum"     # 动量指标
    VOLATILITY = "volatility" # 波动率指标
    VOLUME = "volume"         # 成交量指标
    CUSTOM = "custom"         # 自定义指标


@dataclass
class IndicatorConfig:
    """指标配置"""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # 计算设置
    min_periods: int = 1
    lookback_window: int = 1000  # 保留的历史数据窗口大小
    
    # 缓存设置
    cache_enabled: bool = True
    cache_ttl: float = 300.0  # 缓存TTL（秒）
    
    # 异步设置
    async_enabled: bool = True
    batch_size: int = 1000  # 批量计算大小
    
    # 归一化设置
    normalize: bool = False
    normalize_range: Tuple[float, float] = (-1.0, 1.0)


@dataclass
class IndicatorResult:
    """指标计算结果"""
    indicator_name: str
    symbol: str
    timestamp: float
    value: Union[float, List[float], Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        if isinstance(self.value, float):
            return not np.isnan(self.value) and np.isfinite(self.value)
        elif isinstance(self.value, list):
            return len(self.value) > 0 and not all(np.isnan(v) or not np.isfinite(v) for v in self.value)
        elif isinstance(self.value, dict):
            return len(self.value) > 0 and not all(np.isnan(v) or not np.isfinite(v) for v in self.value.values())
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'indicator_name': self.indicator_name,
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'value': self.value,
            'metadata': self.metadata
        }


class BaseIndicator(ABC):
    """技术指标基类"""
    
    def __init__(self, config: IndicatorConfig):
        self.config = config
        self.name = config.name
        self.parameters = config.parameters
        self.min_periods = config.min_periods
        
        # 数据缓存
        self._data_cache: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.lookback_window)
        )
        self._result_cache: Dict[str, IndicatorResult] = {}
        self._last_calculation_time: Dict[str, float] = {}
        
        # 性能监控
        self._calculation_count = 0
        self._total_calculation_time = 0.0
        self._error_count = 0
    
    @abstractmethod
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """计算技术指标"""
        pass
    
    def can_calculate(self, data: Dict[str, np.ndarray]) -> bool:
        """检查是否有足够数据进行计算"""
        required_keys = self._get_required_data_keys()
        for key in required_keys:
            if key not in data or len(data[key]) < self.min_periods:
                return False
        return True
    
    def _get_required_data_keys(self) -> List[str]:
        """获取所需的数据字段"""
        return ["close"]  # 默认需要收盘价
    
    def _validate_data(self, data: Dict[str, np.ndarray]) -> bool:
        """验证输入数据"""
        for key, values in data.items():
            if not isinstance(values, np.ndarray):
                return False
            if len(values) == 0:
                return False
            # 检查是否有过多的NaN值
            nan_ratio = np.sum(np.isnan(values)) / len(values)
            if nan_ratio > 0.5:  # 超过50%的NaN值
                return False
        return True
    
    def _should_recalculate(self, symbol: str) -> bool:
        """检查是否需要重新计算"""
        if not self.config.cache_enabled:
            return True
        
        last_calc_time = self._last_calculation_time.get(symbol, 0)
        return (time.time() - last_calc_time) > self.config.cache_ttl
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        avg_time = self._total_calculation_time / max(self._calculation_count, 1)
        return {
            'calculation_count': self._calculation_count,
            'total_time': self._total_calculation_time,
            'average_time': avg_time,
            'error_count': self._error_count,
            'error_rate': self._error_count / max(self._calculation_count, 1)
        }


class AsyncTechnicalIndicator(BaseIndicator):
    """支持异步计算的技术指标基类"""
    
    async def calculate_async(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """异步计算技术指标"""
        if not self.config.async_enabled:
            return self.calculate(data, symbol)
        
        # 在线程池中执行计算
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calculate, data, symbol)
    
    async def calculate_batch_async(
        self, 
        data_batch: List[Tuple[Dict[str, np.ndarray], str]]
    ) -> List[IndicatorResult]:
        """批量异步计算"""
        tasks = []
        for data, symbol in data_batch:
            task = self.calculate_async(data, symbol)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                self._error_count += 1
                continue
            valid_results.append(result)
        
        return valid_results


class CustomIndicator(BaseIndicator):
    """自定义指标基类"""
    
    def __init__(self, name: str, calculation_func: Callable, parameters: Dict[str, Any] = None):
        config = IndicatorConfig(
            name=name,
            indicator_type=IndicatorType.CUSTOM,
            parameters=parameters or {}
        )
        super().__init__(config)
        self.calculation_func = calculation_func
    
    def calculate(self, data: Dict[str, np.ndarray], symbol: str) -> IndicatorResult:
        """执行自定义计算函数"""
        start_time = time.time()
        try:
            if not self._validate_data(data):
                raise ValueError("Invalid input data")
            
            result_value = self.calculation_func(data, self.parameters)
            
            result = IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=result_value,
                metadata=self.parameters
            )
            
            # 更新统计
            self._calculation_count += 1
            self._total_calculation_time += time.time() - start_time
            
            return result
            
        except Exception as e:
            self._error_count += 1
            return IndicatorResult(
                indicator_name=self.name,
                symbol=symbol,
                timestamp=time.time(),
                value=np.nan,
                metadata={"error": str(e)}
            )


class TechnicalIndicators(LoggerMixin):
    """技术指标计算管理器 - 增强版"""
    
    def __init__(self):
        # 注册的指标
        self._indicators: Dict[str, BaseIndicator] = {}
        
        # 数据缓存 - 使用更高效的结构
        self._data_buffer: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # 计算结果缓存
        self._results_cache: Dict[str, Dict[str, IndicatorResult]] = defaultdict(dict)
        
        # 性能统计
        self._global_stats = {
            'total_calculations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 异步任务队列
        self._task_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._processing_tasks = False
        
    def register_indicator(self, indicator: BaseIndicator):
        """注册技术指标"""
        self._indicators[indicator.name] = indicator
        self.log_debug(f"Registered indicator: {indicator.name}")
    
    def unregister_indicator(self, indicator_name: str):
        """注销技术指标"""
        if indicator_name in self._indicators:
            del self._indicators[indicator_name]
            self.log_debug(f"Unregistered indicator: {indicator_name}")
    
    def add_custom_indicator(
        self, 
        name: str, 
        calculation_func: Callable, 
        parameters: Dict[str, Any] = None
    ):
        """添加自定义指标"""
        custom_indicator = CustomIndicator(name, calculation_func, parameters)
        self.register_indicator(custom_indicator)
    
    def update_data(self, symbol: str, price_data: Dict[str, float]):
        """更新价格数据"""
        buffer = self._data_buffer[symbol]
        
        # 添加时间戳
        if "timestamp" not in price_data:
            price_data = price_data.copy()
            price_data["timestamp"] = time.time()
        
        for key, value in price_data.items():
            buffer[key].append(value)
    
    def update_data_batch(self, symbol: str, price_data: pd.DataFrame):
        """批量更新价格数据"""
        buffer = self._data_buffer[symbol]
        
        for column in price_data.columns:
            values = price_data[column].values
            buffer[column].extend(values)
    
    async def calculate_indicator_async(
        self, 
        indicator_name: str, 
        symbol: str, 
        force_recalculate: bool = False
    ) -> Optional[IndicatorResult]:
        """异步计算单个指标"""
        if indicator_name not in self._indicators:
            self.log_warning(f"Indicator {indicator_name} not found")
            return None
        
        indicator = self._indicators[indicator_name]
        
        # 检查缓存
        if not force_recalculate and not indicator._should_recalculate(symbol):
            cached_result = self._results_cache[symbol].get(indicator_name)
            if cached_result:
                self._global_stats['cache_hits'] += 1
                return cached_result
        
        self._global_stats['cache_misses'] += 1
        
        # 准备数据
        data = self._prepare_data_for_calculation(symbol)
        if not data:
            return None
        
        # 异步计算
        if isinstance(indicator, AsyncTechnicalIndicator):
            result = await indicator.calculate_async(data, symbol)
        else:
            # 在线程池中执行同步计算
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, indicator.calculate, data, symbol)
        
        # 缓存结果
        if result and result.is_valid:
            self._results_cache[symbol][indicator_name] = result
        
        self._global_stats['total_calculations'] += 1
        
        return result
    
    def calculate_indicator(
        self, 
        indicator_name: str, 
        symbol: str, 
        force_recalculate: bool = False
    ) -> Optional[IndicatorResult]:
        """同步计算单个指标（向后兼容）"""
        if indicator_name not in self._indicators:
            self.log_warning(f"Indicator {indicator_name} not found")
            return None
        
        indicator = self._indicators[indicator_name]
        
        # 检查缓存
        if not force_recalculate and not indicator._should_recalculate(symbol):
            cached_result = self._results_cache[symbol].get(indicator_name)
            if cached_result:
                self._global_stats['cache_hits'] += 1
                return cached_result
        
        self._global_stats['cache_misses'] += 1
        
        # 准备数据
        data = self._prepare_data_for_calculation(symbol)
        if not data:
            return None
        
        # 计算指标
        start_time = time.time()
        try:
            result = indicator.calculate(data, symbol)
            
            # 记录性能
            calculation_time = time.time() - start_time
            self._global_stats['total_time'] += calculation_time
            
            # 缓存结果
            if result and result.is_valid:
                self._results_cache[symbol][indicator_name] = result
            
            self._global_stats['total_calculations'] += 1
            
            return result
            
        except Exception as e:
            self.log_error(f"Error calculating indicator {indicator_name} for {symbol}: {e}")
            return None
    
    async def calculate_all_indicators_async(
        self, 
        symbol: str, 
        force_recalculate: bool = False
    ) -> Dict[str, IndicatorResult]:
        """异步计算所有指标"""
        results = {}
        tasks = []
        
        for indicator_name, indicator in self._indicators.items():
            if not indicator.config.enabled:
                continue
            
            task = self.calculate_indicator_async(indicator_name, symbol, force_recalculate)
            tasks.append((indicator_name, task))
        
        # 并行执行
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # 处理结果
        for i, ((indicator_name, _), result) in enumerate(zip(tasks, completed_tasks)):
            if isinstance(result, Exception):
                self.log_error(f"Error calculating {indicator_name}: {result}")
                continue
            if result:
                results[indicator_name] = result
        
        return results
    
    def calculate_all_indicators(
        self, 
        symbol: str, 
        force_recalculate: bool = False
    ) -> Dict[str, IndicatorResult]:
        """同步计算所有指标（向后兼容）"""
        results = {}
        
        for indicator_name in self._indicators.keys():
            if not self._indicators[indicator_name].config.enabled:
                continue
            
            result = self.calculate_indicator(indicator_name, symbol, force_recalculate)
            if result:
                results[indicator_name] = result
        
        return results
    
    def _prepare_data_for_calculation(self, symbol: str) -> Dict[str, np.ndarray]:
        """准备计算用的数据"""
        if symbol not in self._data_buffer:
            return {}
        
        buffer = self._data_buffer[symbol]
        data = {}
        
        for key, values in buffer.items():
            if len(values) > 0:
                data[key] = np.array(list(values))
        
        return data
    
    def get_indicator_result(self, symbol: str, indicator_name: str) -> Optional[IndicatorResult]:
        """获取指标计算结果"""
        return self._results_cache[symbol].get(indicator_name)
    
    def get_all_results(self, symbol: str) -> Dict[str, IndicatorResult]:
        """获取所有指标结果"""
        return self._results_cache[symbol].copy()
    
    def get_global_performance_stats(self) -> Dict[str, Any]:
        """获取全局性能统计"""
        stats = self._global_stats.copy()
        if stats['total_calculations'] > 0:
            stats['avg_calculation_time'] = stats['total_time'] / stats['total_calculations']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        
        # 添加各个指标的性能统计
        indicator_stats = {}
        for name, indicator in self._indicators.items():
            indicator_stats[name] = indicator.get_performance_stats()
        stats['indicator_stats'] = indicator_stats
        
        return stats
    
    def clear_cache(self, symbol: str = None):
        """清理缓存"""
        if symbol:
            if symbol in self._results_cache:
                del self._results_cache[symbol]
            if symbol in self._data_buffer:
                del self._data_buffer[symbol]
        else:
            self._results_cache.clear()
            self._data_buffer.clear()
        
        self.log_debug(f"Cache cleared for symbol: {symbol or 'all'}")
    
    def get_registered_indicators(self) -> List[str]:
        """获取已注册的指标列表"""
        return list(self._indicators.keys())
    
    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """获取指标配置"""
        indicator = self._indicators.get(indicator_name)
        return indicator.config if indicator else None
    
    def export_data(self, symbol: str, format: str = 'pandas') -> Union[pd.DataFrame, Dict]:
        """导出数据和计算结果"""
        if format == 'pandas':
            # 准备数据
            data = self._prepare_data_for_calculation(symbol)
            results = self.get_all_results(symbol)
            
            # 创建DataFrame
            df_data = {}
            for key, values in data.items():
                df_data[key] = values
            
            for name, result in results.items():
                if isinstance(result.value, dict):
                    for sub_key, sub_value in result.value.items():
                        df_data[f"{name}_{sub_key}"] = sub_value
                else:
                    df_data[name] = result.value
            
            return pd.DataFrame(df_data)
        
        elif format == 'dict':
            return {
                'data': self._prepare_data_for_calculation(symbol),
                'results': {name: result.to_dict() for name, result in self.get_all_results(symbol).items()}
            }
        
        else:
            raise ValueError(f"Unsupported format: {format}")