"""
多时间框架管理器
支持不同时间周期的技术指标计算和数据聚合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time

from .base import BaseIndicator, IndicatorResult


class TimeFrame(Enum):
    """时间框架枚举"""
    TICK = "tick"           # 逐笔数据
    SECOND_1 = "1s"         # 1秒
    SECOND_5 = "5s"         # 5秒
    SECOND_15 = "15s"       # 15秒
    SECOND_30 = "30s"       # 30秒
    MINUTE_1 = "1m"         # 1分钟
    MINUTE_3 = "3m"         # 3分钟
    MINUTE_5 = "5m"         # 5分钟
    MINUTE_15 = "15m"       # 15分钟
    MINUTE_30 = "30m"       # 30分钟
    HOUR_1 = "1h"           # 1小时
    HOUR_2 = "2h"           # 2小时
    HOUR_4 = "4h"           # 4小时
    HOUR_6 = "6h"           # 6小时
    HOUR_8 = "8h"           # 8小时
    HOUR_12 = "12h"         # 12小时
    DAY_1 = "1d"            # 1天
    DAY_3 = "3d"            # 3天
    WEEK_1 = "1w"           # 1周
    MONTH_1 = "1M"          # 1月


@dataclass
class TimeFrameConfig:
    """时间框架配置"""
    timeframe: TimeFrame
    enabled: bool = True
    aggregation_method: str = "standard"  # standard, weighted, vwap
    min_data_points: int = 100  # 最小数据点数量
    
    # 数据缓存设置
    max_cache_size: int = 10000
    cache_compression: bool = True
    
    # 实时更新设置
    real_time_updates: bool = True
    update_threshold_ms: int = 100  # 更新阈值（毫秒）


@dataclass 
class OHLCVData:
    """OHLCV数据结构"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }


class TimeFrameManager:
    """多时间框架管理器"""
    
    # 时间框架到秒的映射
    TIMEFRAME_TO_SECONDS = {
        TimeFrame.SECOND_1: 1,
        TimeFrame.SECOND_5: 5,
        TimeFrame.SECOND_15: 15,
        TimeFrame.SECOND_30: 30,
        TimeFrame.MINUTE_1: 60,
        TimeFrame.MINUTE_3: 180,
        TimeFrame.MINUTE_5: 300,
        TimeFrame.MINUTE_15: 900,
        TimeFrame.MINUTE_30: 1800,
        TimeFrame.HOUR_1: 3600,
        TimeFrame.HOUR_2: 7200,
        TimeFrame.HOUR_4: 14400,
        TimeFrame.HOUR_6: 21600,
        TimeFrame.HOUR_8: 28800,
        TimeFrame.HOUR_12: 43200,
        TimeFrame.DAY_1: 86400,
        TimeFrame.DAY_3: 259200,
        TimeFrame.WEEK_1: 604800,
        TimeFrame.MONTH_1: 2592000,  # 30天近似
    }
    
    def __init__(self):
        # 时间框架配置
        self._timeframe_configs: Dict[TimeFrame, TimeFrameConfig] = {}
        
        # 数据缓存：symbol -> timeframe -> data
        self._data_cache: Dict[str, Dict[TimeFrame, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=10000))
        )
        
        # 当前聚合状态：symbol -> timeframe -> current_bar
        self._current_bars: Dict[str, Dict[TimeFrame, OHLCVData]] = defaultdict(
            lambda: defaultdict(lambda: None)
        )
        
        # 指标实例：timeframe -> indicator_name -> indicator
        self._indicators: Dict[TimeFrame, Dict[str, BaseIndicator]] = defaultdict(dict)
        
        # 结果缓存：symbol -> timeframe -> indicator_name -> result
        self._results_cache: Dict[str, Dict[TimeFrame, Dict[str, IndicatorResult]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        
        # 最后更新时间
        self._last_update: Dict[str, Dict[TimeFrame, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # 注册默认时间框架
        self._register_default_timeframes()
    
    def _register_default_timeframes(self):
        """注册默认时间框架"""
        default_timeframes = [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5,
            TimeFrame.MINUTE_15,
            TimeFrame.HOUR_1,
            TimeFrame.HOUR_4,
            TimeFrame.DAY_1
        ]
        
        for tf in default_timeframes:
            self.register_timeframe(TimeFrameConfig(timeframe=tf))
    
    def register_timeframe(self, config: TimeFrameConfig):
        """注册时间框架"""
        self._timeframe_configs[config.timeframe] = config
    
    def register_indicator_for_timeframe(
        self, 
        timeframe: TimeFrame, 
        indicator: BaseIndicator
    ):
        """为特定时间框架注册指标"""
        if timeframe not in self._timeframe_configs:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        self._indicators[timeframe][indicator.name] = indicator
    
    def update_tick_data(
        self, 
        symbol: str, 
        timestamp: float, 
        price: float, 
        volume: float = 0.0
    ):
        """更新逐笔数据"""
        # 更新所有已注册的时间框架
        for timeframe in self._timeframe_configs:
            if not self._timeframe_configs[timeframe].enabled:
                continue
            
            self._aggregate_tick_to_timeframe(
                symbol, timeframe, timestamp, price, volume
            )
    
    def update_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        ohlcv: OHLCVData
    ):
        """直接更新OHLCV数据"""
        if timeframe not in self._timeframe_configs:
            raise ValueError(f"Timeframe {timeframe} not registered")
        
        cache = self._data_cache[symbol][timeframe]
        cache.append(ohlcv)
        
        # 更新最后更新时间
        self._last_update[symbol][timeframe] = time.time()
        
        # 计算指标
        self._calculate_indicators_for_timeframe(symbol, timeframe)
    
    def _aggregate_tick_to_timeframe(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        timestamp: float, 
        price: float, 
        volume: float
    ):
        """将逐笔数据聚合到特定时间框架"""
        if timeframe == TimeFrame.TICK:
            # 对于逐笔数据，直接创建OHLCV
            ohlcv = OHLCVData(timestamp, price, price, price, price, volume)
            self._data_cache[symbol][timeframe].append(ohlcv)
            return
        
        # 计算时间框架的开始时间
        interval_seconds = self.TIMEFRAME_TO_SECONDS[timeframe]
        bar_start_time = int(timestamp // interval_seconds) * interval_seconds
        
        current_bar = self._current_bars[symbol][timeframe]
        
        # 检查是否需要创建新的K线
        if (current_bar is None or 
            int(current_bar.timestamp // interval_seconds) * interval_seconds != bar_start_time):
            
            # 保存前一个K线（如果存在）
            if current_bar is not None:
                self._data_cache[symbol][timeframe].append(current_bar)
                self._calculate_indicators_for_timeframe(symbol, timeframe)
            
            # 创建新的K线
            current_bar = OHLCVData(
                timestamp=bar_start_time,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
            self._current_bars[symbol][timeframe] = current_bar
        else:
            # 更新当前K线
            current_bar.high = max(current_bar.high, price)
            current_bar.low = min(current_bar.low, price)
            current_bar.close = price
            current_bar.volume += volume
            
            # 实时更新检查
            config = self._timeframe_configs[timeframe]
            if config.real_time_updates:
                last_update = self._last_update[symbol][timeframe]
                if (time.time() - last_update) * 1000 >= config.update_threshold_ms:
                    self._calculate_indicators_for_timeframe(symbol, timeframe, real_time=True)
                    self._last_update[symbol][timeframe] = time.time()
    
    def _calculate_indicators_for_timeframe(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        real_time: bool = False
    ):
        """计算特定时间框架的指标"""
        if timeframe not in self._indicators:
            return
        
        # 准备数据
        data = self._prepare_timeframe_data(symbol, timeframe, include_current=real_time)
        if not data:
            return
        
        # 计算所有指标
        for indicator_name, indicator in self._indicators[timeframe].items():
            try:
                result = indicator.calculate(data, symbol)
                if result and result.is_valid:
                    # 添加时间框架信息到元数据
                    result.metadata = result.metadata or {}
                    result.metadata['timeframe'] = timeframe.value
                    result.metadata['real_time'] = real_time
                    
                    self._results_cache[symbol][timeframe][indicator_name] = result
            except Exception as e:
                # 记录错误但不中断其他指标的计算
                print(f"Error calculating {indicator_name} for {timeframe.value}: {e}")
    
    def _prepare_timeframe_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        include_current: bool = False
    ) -> Dict[str, np.ndarray]:
        """准备时间框架数据"""
        cache = self._data_cache[symbol][timeframe]
        
        if len(cache) == 0:
            return {}
        
        # 获取数据列表
        data_list = list(cache)
        
        # 如果需要包含当前未完成的K线
        if include_current and symbol in self._current_bars:
            current_bar = self._current_bars[symbol].get(timeframe)
            if current_bar:
                data_list.append(current_bar)
        
        if len(data_list) == 0:
            return {}
        
        # 转换为numpy数组
        timestamps = np.array([bar.timestamp for bar in data_list])
        opens = np.array([bar.open for bar in data_list])
        highs = np.array([bar.high for bar in data_list])
        lows = np.array([bar.low for bar in data_list])
        closes = np.array([bar.close for bar in data_list])
        volumes = np.array([bar.volume for bar in data_list])
        
        return {
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
    
    def get_indicator_result(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        indicator_name: str
    ) -> Optional[IndicatorResult]:
        """获取特定时间框架的指标结果"""
        return self._results_cache[symbol][timeframe].get(indicator_name)
    
    def get_all_results_for_timeframe(
        self, 
        symbol: str, 
        timeframe: TimeFrame
    ) -> Dict[str, IndicatorResult]:
        """获取特定时间框架的所有指标结果"""
        return self._results_cache[symbol][timeframe].copy()
    
    def get_multi_timeframe_results(
        self, 
        symbol: str, 
        indicator_name: str
    ) -> Dict[TimeFrame, IndicatorResult]:
        """获取多时间框架的同一指标结果"""
        results = {}
        for timeframe in self._timeframe_configs:
            result = self.get_indicator_result(symbol, timeframe, indicator_name)
            if result:
                results[timeframe] = result
        return results
    
    def get_timeframe_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        limit: Optional[int] = None
    ) -> List[OHLCVData]:
        """获取时间框架数据"""
        cache = self._data_cache[symbol][timeframe]
        data = list(cache)
        
        if limit:
            data = data[-limit:]
        
        return data
    
    def convert_timeframe(
        self, 
        symbol: str, 
        from_timeframe: TimeFrame, 
        to_timeframe: TimeFrame, 
        limit: Optional[int] = None
    ) -> List[OHLCVData]:
        """时间框架转换"""
        source_data = self.get_timeframe_data(symbol, from_timeframe, limit)
        
        if not source_data:
            return []
        
        # 检查转换可行性
        from_seconds = self.TIMEFRAME_TO_SECONDS.get(from_timeframe)
        to_seconds = self.TIMEFRAME_TO_SECONDS.get(to_timeframe)
        
        if not from_seconds or not to_seconds:
            raise ValueError("Unsupported timeframe conversion")
        
        if to_seconds <= from_seconds:
            raise ValueError("Cannot convert to smaller timeframe")
        
        # 执行转换
        converted_data = []
        interval_ratio = to_seconds // from_seconds
        
        for i in range(0, len(source_data), interval_ratio):
            chunk = source_data[i:i + interval_ratio]
            if len(chunk) == 0:
                continue
            
            # 聚合数据
            aggregated = OHLCVData(
                timestamp=chunk[0].timestamp,
                open=chunk[0].open,
                high=max(bar.high for bar in chunk),
                low=min(bar.low for bar in chunk),
                close=chunk[-1].close,
                volume=sum(bar.volume for bar in chunk)
            )
            converted_data.append(aggregated)
        
        return converted_data
    
    def export_multi_timeframe_data(
        self, 
        symbol: str, 
        format: str = 'pandas'
    ) -> Dict[TimeFrame, Union[pd.DataFrame, Dict]]:
        """导出多时间框架数据"""
        exported_data = {}
        
        for timeframe in self._timeframe_configs:
            data = self.get_timeframe_data(symbol, timeframe)
            
            if format == 'pandas':
                if data:
                    df_data = {
                        'timestamp': [bar.timestamp for bar in data],
                        'open': [bar.open for bar in data],
                        'high': [bar.high for bar in data],
                        'low': [bar.low for bar in data],
                        'close': [bar.close for bar in data],
                        'volume': [bar.volume for bar in data]
                    }
                    
                    # 添加指标数据
                    results = self.get_all_results_for_timeframe(symbol, timeframe)
                    for indicator_name, result in results.items():
                        if isinstance(result.value, dict):
                            for key, value in result.value.items():
                                df_data[f"{indicator_name}_{key}"] = value
                        else:
                            df_data[indicator_name] = result.value
                    
                    exported_data[timeframe] = pd.DataFrame(df_data)
                else:
                    exported_data[timeframe] = pd.DataFrame()
            
            elif format == 'dict':
                exported_data[timeframe] = {
                    'ohlcv': [bar.to_dict() for bar in data],
                    'indicators': {
                        name: result.to_dict() 
                        for name, result in self.get_all_results_for_timeframe(symbol, timeframe).items()
                    }
                }
        
        return exported_data
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[TimeFrame] = None):
        """清理缓存"""
        if symbol and timeframe:
            # 清理特定symbol和timeframe
            if symbol in self._data_cache:
                self._data_cache[symbol].pop(timeframe, None)
            if symbol in self._results_cache:
                self._results_cache[symbol].pop(timeframe, None)
            if symbol in self._current_bars:
                self._current_bars[symbol].pop(timeframe, None)
        elif symbol:
            # 清理特定symbol的所有数据
            self._data_cache.pop(symbol, None)
            self._results_cache.pop(symbol, None)
            self._current_bars.pop(symbol, None)
        elif timeframe:
            # 清理特定timeframe的所有数据
            for sym in self._data_cache:
                self._data_cache[sym].pop(timeframe, None)
            for sym in self._results_cache:
                self._results_cache[sym].pop(timeframe, None)
            for sym in self._current_bars:
                self._current_bars[sym].pop(timeframe, None)
        else:
            # 清理所有数据
            self._data_cache.clear()
            self._results_cache.clear()
            self._current_bars.clear()
    
    def get_registered_timeframes(self) -> List[TimeFrame]:
        """获取已注册的时间框架"""
        return list(self._timeframe_configs.keys())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'total_symbols': len(self._data_cache),
            'total_timeframes': len(self._timeframe_configs),
            'cache_sizes': {},
            'indicator_counts': {}
        }
        
        for timeframe, config in self._timeframe_configs.items():
            indicator_count = len(self._indicators.get(timeframe, {}))
            stats['indicator_counts'][timeframe.value] = indicator_count
        
        for symbol, tf_data in self._data_cache.items():
            for tf, cache in tf_data.items():
                key = f"{symbol}_{tf.value}"
                stats['cache_sizes'][key] = len(cache)
        
        return stats