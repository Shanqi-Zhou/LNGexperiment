"""
增量特征计算器 - 避免重复计算的高效特征工程
整合优化方案的增量计算策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import signal
import hashlib
from collections import defaultdict, deque
from threading import Lock
import psutil
import gc


class FeatureCache:
    """特征缓存系统 - 避免重复计算"""
    
    def __init__(self, max_size_mb: int = 512, logger: Optional[logging.Logger] = None):
        self.max_size_mb = max_size_mb
        self.logger = logger or logging.getLogger(__name__)
        self.cache = {}
        self.access_times = {}
        self.cache_lock = Lock()
        self.hit_count = 0
        self.miss_count = 0
        
    def _compute_hash(self, data: np.ndarray) -> str:
        """计算数据哈希值"""
        return hashlib.md5(data.tobytes()).hexdigest()
    
    def _get_cache_size_mb(self) -> float:
        """估计缓存大小(MB)"""
        total_bytes = 0
        for key, value in self.cache.items():
            if isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, np.ndarray):
                        total_bytes += v.nbytes
        return total_bytes / (1024 * 1024)
    
    def _evict_lru(self):
        """LRU缓存清理"""
        if not self.access_times:
            return
            
        # 按访问时间排序，移除最久未使用的
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        keys_to_remove = [k for k, _ in sorted_keys[:len(sorted_keys)//4]]  # 移除25%
        
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                
        gc.collect()  # 强制垃圾回收
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self.cache_lock:
            if key in self.cache:
                self.access_times[key] = pd.Timestamp.now()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        with self.cache_lock:
            # 检查内存使用
            current_size = self._get_cache_size_mb()
            if current_size > self.max_size_mb:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = pd.Timestamp.now()
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size_mb': self._get_cache_size_mb(),
            'num_entries': len(self.cache)
        }


class IncrementalStatsCalculator:
    """增量统计计算器 - 滑动窗口统计的高效更新"""
    
    def __init__(self, window_size: int = 180, logger: Optional[logging.Logger] = None):
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        
        # 滑动窗口数据
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        
        # 增量统计缓存
        self.running_sums = defaultdict(float)
        self.running_sum_squares = defaultdict(float)
        self.running_sum_cubes = defaultdict(float)
        self.running_sum_quads = defaultdict(float)
        self.n_samples = defaultdict(int)
        
        # 变化率缓存
        self.last_values = {}
        self.change_rate_windows = defaultdict(lambda: deque(maxlen=window_size-1))
    
    def update(self, column: str, new_value: float) -> Dict[str, float]:
        """
        增量更新统计信息
        Args:
            column: 列名
            new_value: 新数据点
        Returns:
            stats: 当前窗口的统计特征
        """
        window = self.data_windows[column]
        
        # 如果窗口满了，需要减去要移除的值的贡献
        if len(window) == self.window_size:
            old_value = window[0]  # 将被移除的值
            
            # 更新累积统计
            self.running_sums[column] -= old_value
            self.running_sum_squares[column] -= old_value ** 2
            self.running_sum_cubes[column] -= old_value ** 3
            self.running_sum_quads[column] -= old_value ** 4
            self.n_samples[column] -= 1
        
        # 添加新值
        window.append(new_value)
        
        # 更新累积统计
        self.running_sums[column] += new_value
        self.running_sum_squares[column] += new_value ** 2
        self.running_sum_cubes[column] += new_value ** 3
        self.running_sum_quads[column] += new_value ** 4
        self.n_samples[column] += 1
        
        # 计算变化率
        if column in self.last_values:
            change_rate = abs(new_value - self.last_values[column])
            
            change_window = self.change_rate_windows[column]
            if len(change_window) == self.window_size - 1:
                change_window.append(change_rate)
            else:
                change_window.append(change_rate)
        
        self.last_values[column] = new_value
        
        # 计算当前统计特征
        return self._compute_stats(column)
    
    def _compute_stats(self, column: str) -> Dict[str, float]:
        """基于增量统计计算特征"""
        n = self.n_samples[column]
        if n == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0,
                'rms': 0.0, 'mean_change_rate': 0.0, 'std_change_rate': 0.0
            }
        
        # 基础统计量
        mean = self.running_sums[column] / n
        
        # 方差和标准差
        if n > 1:
            variance = (self.running_sum_squares[column] / n) - (mean ** 2)
            std = np.sqrt(max(0, variance))
        else:
            std = 0.0
        
        # RMS
        rms = np.sqrt(self.running_sum_squares[column] / n)
        
        # 偏度和峰度（需要标准化）
        if std > 1e-8 and n > 2:
            # 中心化的三次矩
            m3 = (self.running_sum_cubes[column] / n) - \
                 3 * mean * (self.running_sum_squares[column] / n) + \
                 2 * (mean ** 3)
            skewness = m3 / (std ** 3)
            
            if n > 3:
                # 中心化的四次矩
                m4 = (self.running_sum_quads[column] / n) - \
                     4 * mean * (self.running_sum_cubes[column] / n) + \
                     6 * (mean ** 2) * (self.running_sum_squares[column] / n) - \
                     3 * (mean ** 4)
                kurtosis = (m4 / (std ** 4)) - 3
            else:
                kurtosis = 0.0
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # 变化率统计
        change_rates = list(self.change_rate_windows[column])
        if change_rates:
            mean_change_rate = np.mean(change_rates)
            std_change_rate = np.std(change_rates)
        else:
            mean_change_rate = 0.0
            std_change_rate = 0.0
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'rms': rms,
            'mean_change_rate': mean_change_rate,
            'std_change_rate': std_change_rate
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有列的统计信息"""
        all_stats = {}
        for column in self.data_windows.keys():
            all_stats[column] = self._compute_stats(column)
        return all_stats


class IncrementalFeatureExtractor:
    """增量特征提取器 - 整合缓存和增量计算"""
    
    def __init__(self, 
                 window_size: int = 180,
                 cache_size_mb: int = 512,
                 logger: Optional[logging.Logger] = None):
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        
        # 缓存系统
        self.feature_cache = FeatureCache(cache_size_mb, logger)
        
        # 增量统计计算器
        self.stats_calculator = IncrementalStatsCalculator(window_size, logger)
        
        # EMD缓存（EMD计算昂贵，重点缓存）
        self.emd_cache = {}
        
        # 特征计算历史
        self.computation_history = deque(maxlen=1000)
        
    def extract_features_incremental(self, 
                                   data_stream: pd.DataFrame,
                                   is_streaming: bool = True) -> Dict[str, np.ndarray]:
        """
        增量特征提取
        Args:
            data_stream: 数据流（新数据或批次数据）
            is_streaming: 是否为流式处理
        Returns:
            features: 提取的特征字典
        """
        features = {}
        
        if is_streaming:
            # 流式处理：逐点更新
            for idx, row in data_stream.iterrows():
                timestamp = row.get('timestamp', idx)
                features[timestamp] = self._process_single_point(row)
        else:
            # 批处理：利用缓存优化
            features = self._process_batch(data_stream)
        
        # 记录计算统计
        self._log_computation_stats()
        
        return features
    
    def _process_single_point(self, data_point: pd.Series) -> np.ndarray:
        """处理单个数据点（流式模式）"""
        features = []
        
        # 为每个数值列更新增量统计
        for col, value in data_point.items():
            if pd.api.types.is_numeric_dtype(type(value)) and pd.notna(value):
                stats = self.stats_calculator.update(col, float(value))
                
                # 提取基础统计特征
                features.extend([
                    stats['mean'], stats['std'], stats['skewness'], 
                    stats['kurtosis'], stats['rms'], 
                    stats['mean_change_rate'], stats['std_change_rate']
                ])
        
        return np.array(features)
    
    def _process_batch(self, data_batch: pd.DataFrame) -> Dict[str, np.ndarray]:
        """批处理模式 - 利用缓存优化"""
        features = {}
        
        # 为每个可能的窗口计算特征
        n_samples = len(data_batch)
        numeric_cols = data_batch.select_dtypes(include=[np.number]).columns
        
        for i in range(n_samples - self.window_size + 1):
            start_idx = i
            end_idx = i + self.window_size
            
            # 生成窗口数据的哈希值
            window_data = data_batch[numeric_cols].iloc[start_idx:end_idx].values
            data_hash = self.feature_cache._compute_hash(window_data)
            
            # 检查缓存
            cached_features = self.feature_cache.get(data_hash)
            if cached_features is not None:
                features[f'window_{i}'] = cached_features
                continue
            
            # 计算新特征
            computed_features = self._compute_window_features(window_data, numeric_cols)
            
            # 缓存结果
            self.feature_cache.set(data_hash, computed_features)
            features[f'window_{i}'] = computed_features
        
        return features
    
    def _compute_window_features(self, 
                                window_data: np.ndarray, 
                                column_names: List[str]) -> np.ndarray:
        """计算窗口特征（未缓存的情况）"""
        features = []
        
        for ch_idx, col in enumerate(column_names):
            channel_data = window_data[:, ch_idx]
            
            # 基础统计特征（向量化计算）
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # 偏度和峰度
            if std_val > 1e-8:
                normalized_data = (channel_data - mean_val) / std_val
                skewness = np.mean(normalized_data ** 3)
                kurtosis = np.mean(normalized_data ** 4) - 3
            else:
                skewness = 0.0
                kurtosis = 0.0
            
            rms = np.sqrt(np.mean(channel_data ** 2))
            
            # 变化率特征
            diff_data = np.diff(channel_data)
            mean_change_rate = np.mean(np.abs(diff_data))
            std_change_rate = np.std(diff_data)
            
            # EMD特征（昂贵计算，优先缓存）
            emd_features = self._compute_emd_features_cached(channel_data, col)
            
            channel_features = [
                mean_val, std_val, skewness, kurtosis, rms,
                mean_change_rate, std_change_rate
            ] + list(emd_features)
            
            features.extend(channel_features)
        
        return np.array(features)
    
    def _compute_emd_features_cached(self, 
                                   data: np.ndarray, 
                                   column_name: str) -> Tuple[float, float]:
        """缓存的EMD特征计算"""
        # 生成数据哈希
        data_hash = f"{column_name}_{self.feature_cache._compute_hash(data)}"
        
        # 检查EMD缓存
        if data_hash in self.emd_cache:
            return self.emd_cache[data_hash]
        
        try:
            from PyEMD import EMD
            emd = EMD()
            
            # EMD分解
            imfs = emd(data)
            if len(imfs) < 2:
                result = (0.0, 0.0)
            else:
                imf1 = imfs[0]
                
                # IMF1主频
                peaks, _ = signal.find_peaks(imf1)
                if len(peaks) > 1:
                    avg_period = np.mean(np.diff(peaks))
                    imf1_freq = 1.0 / avg_period if avg_period > 0 else 0.0
                else:
                    imf1_freq = 0.0
                
                # IMF1能量比
                total_energy = np.sum([np.sum(imf**2) for imf in imfs])
                imf1_energy = np.sum(imf1**2)
                imf1_energy_ratio = imf1_energy / total_energy if total_energy > 0 else 0.0
                
                result = (imf1_freq, imf1_energy_ratio)
            
            # 缓存结果（EMD缓存独立管理，因为计算昂贵）
            self.emd_cache[data_hash] = result
            
            # EMD缓存管理（防止过大）
            if len(self.emd_cache) > 1000:
                # 移除最老的一半
                keys_to_remove = list(self.emd_cache.keys())[:500]
                for key in keys_to_remove:
                    del self.emd_cache[key]
            
            return result
            
        except Exception as e:
            self.logger.warning(f"EMD计算失败 {column_name}: {e}")
            result = (0.0, 0.0)
            self.emd_cache[data_hash] = result
            return result
    
    def _log_computation_stats(self):
        """记录计算统计信息"""
        cache_stats = self.feature_cache.get_stats()
        memory_usage = psutil.virtual_memory().percent
        
        stats_info = {
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size_mb': cache_stats['cache_size_mb'],
            'memory_usage_percent': memory_usage,
            'emd_cache_size': len(self.emd_cache)
        }
        
        self.computation_history.append({
            'timestamp': pd.Timestamp.now(),
            'stats': stats_info
        })
        
        if len(self.computation_history) % 100 == 0:
            self.logger.info(f"特征计算统计: {stats_info}")
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        cache_stats = self.feature_cache.get_stats()
        
        if self.computation_history:
            recent_stats = [h['stats'] for h in list(self.computation_history)[-10:]]
            avg_memory = np.mean([s['memory_usage_percent'] for s in recent_stats])
            avg_cache_size = np.mean([s['cache_size_mb'] for s in recent_stats])
        else:
            avg_memory = 0.0
            avg_cache_size = 0.0
        
        return {
            'feature_cache': cache_stats,
            'emd_cache_size': len(self.emd_cache),
            'avg_memory_usage_percent': avg_memory,
            'avg_cache_size_mb': avg_cache_size,
            'computation_history_length': len(self.computation_history)
        }
    
    def clear_caches(self):
        """清理所有缓存"""
        self.feature_cache.cache.clear()
        self.feature_cache.access_times.clear()
        self.emd_cache.clear()
        self.computation_history.clear()
        gc.collect()
        
        self.logger.info("所有缓存已清理")


# 便捷接口函数
def create_incremental_extractor(window_size: int = 180, 
                                cache_size_mb: int = 512,
                                logger: Optional[logging.Logger] = None) -> IncrementalFeatureExtractor:
    """创建增量特征提取器"""
    return IncrementalFeatureExtractor(window_size, cache_size_mb, logger)