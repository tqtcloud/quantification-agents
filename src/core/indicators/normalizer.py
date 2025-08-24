"""
指标标准化和归一化模块
提供不同的标准化方法，将指标值映射到统一范围
"""

import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class NormalizationMethod(Enum):
    """标准化方法枚举"""
    MIN_MAX = "min_max"              # 最小-最大标准化
    Z_SCORE = "z_score"              # Z分数标准化
    ROBUST = "robust"                # 鲁棒标准化（使用中位数和IQR）
    PERCENTILE = "percentile"        # 百分位数标准化
    TANH = "tanh"                    # tanh标准化
    SIGMOID = "sigmoid"              # sigmoid标准化
    UNIT_VECTOR = "unit_vector"      # 单位向量标准化


@dataclass
class NormalizationConfig:
    """标准化配置"""
    method: NormalizationMethod = NormalizationMethod.MIN_MAX
    target_range: Tuple[float, float] = (-1.0, 1.0)
    clip_outliers: bool = True
    outlier_threshold: float = 3.0  # 离群值阈值（标准差倍数）
    window_size: Optional[int] = None  # 滚动窗口大小，None表示使用全部历史数据
    
    # 方法特定参数
    percentile_range: Tuple[float, float] = (5.0, 95.0)  # 百分位数范围
    robust_center: bool = True  # 鲁棒标准化是否中心化
    

class IndicatorNormalizer:
    """技术指标标准化器"""
    
    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
        
        # 存储历史统计信息用于增量标准化
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
    
    def normalize(
        self, 
        values: Union[float, np.ndarray], 
        indicator_name: str,
        method: Optional[NormalizationMethod] = None,
        target_range: Optional[Tuple[float, float]] = None
    ) -> Union[float, np.ndarray]:
        """
        标准化指标值
        
        Args:
            values: 待标准化的值
            indicator_name: 指标名称，用于缓存统计信息
            method: 标准化方法，None则使用配置中的方法
            target_range: 目标范围，None则使用配置中的范围
        
        Returns:
            标准化后的值
        """
        method = method or self.config.method
        target_range = target_range or self.config.target_range
        
        # 转换为numpy数组
        is_scalar = np.isscalar(values)
        values_array = np.array([values] if is_scalar else values, dtype=float)
        
        if len(values_array) == 0:
            return values
        
        # 预处理：处理异常值
        if self.config.clip_outliers:
            values_array = self._clip_outliers(values_array, indicator_name)
        
        # 应用标准化方法
        normalized = self._apply_normalization_method(
            values_array, indicator_name, method, target_range
        )
        
        return normalized[0] if is_scalar else normalized
    
    def _apply_normalization_method(
        self,
        values: np.ndarray,
        indicator_name: str,
        method: NormalizationMethod,
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """应用具体的标准化方法"""
        
        if method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.ROBUST:
            return self._robust_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.PERCENTILE:
            return self._percentile_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.TANH:
            return self._tanh_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.SIGMOID:
            return self._sigmoid_normalize(values, indicator_name, target_range)
        
        elif method == NormalizationMethod.UNIT_VECTOR:
            return self._unit_vector_normalize(values, target_range)
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    def _min_max_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """最小-最大标准化"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        min_val = stats.get('min', np.nanmin(values))
        max_val = stats.get('max', np.nanmax(values))
        
        if max_val == min_val:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 标准化到[0,1]
        normalized = (values - min_val) / (max_val - min_val)
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    def _z_score_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """Z分数标准化"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        mean = stats.get('mean', np.nanmean(values))
        std = stats.get('std', np.nanstd(values, ddof=1))
        
        if std == 0:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # Z分数
        z_scores = (values - mean) / std
        
        # 使用tanh函数将Z分数映射到[-1,1]，然后缩放到目标范围
        normalized = np.tanh(z_scores * 0.5)  # 缩放因子0.5使得±2σ大约映射到±0.96
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) / 2 + (target_range[0] + target_range[1]) / 2
    
    def _robust_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """鲁棒标准化（使用中位数和IQR）"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        median = stats.get('median', np.nanmedian(values))
        q75 = stats.get('q75', np.nanpercentile(values, 75))
        q25 = stats.get('q25', np.nanpercentile(values, 25))
        iqr = q75 - q25
        
        if iqr == 0:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 鲁棒标准化
        if self.config.robust_center:
            normalized = (values - median) / iqr
        else:
            normalized = values / iqr
        
        # 使用tanh函数限制范围
        normalized = np.tanh(normalized)
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) / 2 + (target_range[0] + target_range[1]) / 2
    
    def _percentile_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """百分位数标准化"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        p_low, p_high = self.config.percentile_range
        low_val = stats.get(f'p{p_low}', np.nanpercentile(values, p_low))
        high_val = stats.get(f'p{p_high}', np.nanpercentile(values, p_high))
        
        if high_val == low_val:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 标准化到[0,1]
        normalized = np.clip((values - low_val) / (high_val - low_val), 0, 1)
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    def _tanh_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """tanh标准化"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        mean = stats.get('mean', np.nanmean(values))
        std = stats.get('std', np.nanstd(values, ddof=1))
        
        if std == 0:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 标准化后应用tanh
        z_scores = (values - mean) / std
        normalized = np.tanh(z_scores)
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) / 2 + (target_range[0] + target_range[1]) / 2
    
    def _sigmoid_normalize(
        self, 
        values: np.ndarray, 
        indicator_name: str, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """sigmoid标准化"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        mean = stats.get('mean', np.nanmean(values))
        std = stats.get('std', np.nanstd(values, ddof=1))
        
        if std == 0:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 标准化后应用sigmoid
        z_scores = (values - mean) / std
        normalized = 1 / (1 + np.exp(-z_scores))
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) + target_range[0]
    
    def _unit_vector_normalize(
        self, 
        values: np.ndarray, 
        target_range: Tuple[float, float]
    ) -> np.ndarray:
        """单位向量标准化"""
        # 计算L2范数
        norm = np.linalg.norm(values)
        
        if norm == 0:
            return np.full_like(values, (target_range[0] + target_range[1]) / 2)
        
        # 单位向量
        normalized = values / norm
        
        # 缩放到目标范围
        return normalized * (target_range[1] - target_range[0]) / 2 + (target_range[0] + target_range[1]) / 2
    
    def _clip_outliers(self, values: np.ndarray, indicator_name: str) -> np.ndarray:
        """处理离群值"""
        stats = self._get_or_update_stats(values, indicator_name)
        
        mean = stats.get('mean', np.nanmean(values))
        std = stats.get('std', np.nanstd(values, ddof=1))
        
        if std == 0:
            return values
        
        # 计算离群值阈值
        threshold = self.config.outlier_threshold * std
        lower_bound = mean - threshold
        upper_bound = mean + threshold
        
        # 限制离群值
        return np.clip(values, lower_bound, upper_bound)
    
    def _get_or_update_stats(self, values: np.ndarray, indicator_name: str) -> Dict[str, Any]:
        """获取或更新统计信息"""
        # 如果使用滚动窗口，只使用最近的数据
        if self.config.window_size and len(values) > self.config.window_size:
            values = values[-self.config.window_size:]
        
        # 过滤掉NaN值
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return {}
        
        stats = {
            'mean': np.mean(valid_values),
            'std': np.std(valid_values, ddof=1),
            'min': np.min(valid_values),
            'max': np.max(valid_values),
            'median': np.median(valid_values),
            'q25': np.percentile(valid_values, 25),
            'q75': np.percentile(valid_values, 75),
        }
        
        # 计算百分位数
        p_low, p_high = self.config.percentile_range
        stats[f'p{p_low}'] = np.percentile(valid_values, p_low)
        stats[f'p{p_high}'] = np.percentile(valid_values, p_high)
        
        # 缓存统计信息
        self._stats_cache[indicator_name] = stats
        
        return stats
    
    def denormalize(
        self, 
        normalized_values: Union[float, np.ndarray], 
        indicator_name: str,
        method: Optional[NormalizationMethod] = None,
        target_range: Optional[Tuple[float, float]] = None
    ) -> Union[float, np.ndarray]:
        """
        反标准化
        
        Args:
            normalized_values: 标准化后的值
            indicator_name: 指标名称
            method: 标准化方法
            target_range: 目标范围
        
        Returns:
            原始值
        """
        method = method or self.config.method
        target_range = target_range or self.config.target_range
        
        if indicator_name not in self._stats_cache:
            raise ValueError(f"No cached statistics for indicator: {indicator_name}")
        
        stats = self._stats_cache[indicator_name]
        
        # 转换为numpy数组
        is_scalar = np.isscalar(normalized_values)
        norm_array = np.array([normalized_values] if is_scalar else normalized_values, dtype=float)
        
        # 反标准化
        if method == NormalizationMethod.MIN_MAX:
            # 从目标范围还原到[0,1]
            unit_values = (norm_array - target_range[0]) / (target_range[1] - target_range[0])
            # 还原到原始范围
            original = unit_values * (stats['max'] - stats['min']) + stats['min']
        
        elif method == NormalizationMethod.PERCENTILE:
            # 从目标范围还原到[0,1]
            unit_values = (norm_array - target_range[0]) / (target_range[1] - target_range[0])
            # 还原到百分位数范围
            p_low, p_high = self.config.percentile_range
            original = unit_values * (stats[f'p{p_high}'] - stats[f'p{p_low}']) + stats[f'p{p_low}']
        
        else:
            # 对于其他方法，反标准化比较复杂，这里提供简化实现
            # 实际应用中可能需要更精确的反标准化
            original = norm_array  # 保持标准化值
        
        return original[0] if is_scalar else original
    
    def get_normalization_info(self, indicator_name: str) -> Dict[str, Any]:
        """获取指标的标准化信息"""
        return {
            'method': self.config.method.value,
            'target_range': self.config.target_range,
            'stats': self._stats_cache.get(indicator_name, {}),
            'config': {
                'clip_outliers': self.config.clip_outliers,
                'outlier_threshold': self.config.outlier_threshold,
                'window_size': self.config.window_size,
                'percentile_range': self.config.percentile_range,
                'robust_center': self.config.robust_center,
            }
        }
    
    def clear_cache(self, indicator_name: Optional[str] = None):
        """清理统计信息缓存"""
        if indicator_name:
            self._stats_cache.pop(indicator_name, None)
        else:
            self._stats_cache.clear()


# 便捷函数
def normalize_indicator_values(
    values: Union[float, np.ndarray],
    method: NormalizationMethod = NormalizationMethod.MIN_MAX,
    target_range: Tuple[float, float] = (-1.0, 1.0),
    clip_outliers: bool = True
) -> Union[float, np.ndarray]:
    """
    快速标准化指标值的便捷函数
    
    Args:
        values: 待标准化的值
        method: 标准化方法
        target_range: 目标范围
        clip_outliers: 是否处理离群值
    
    Returns:
        标准化后的值
    """
    config = NormalizationConfig(
        method=method,
        target_range=target_range,
        clip_outliers=clip_outliers
    )
    
    normalizer = IndicatorNormalizer(config)
    return normalizer.normalize(values, "temp_indicator", method, target_range)