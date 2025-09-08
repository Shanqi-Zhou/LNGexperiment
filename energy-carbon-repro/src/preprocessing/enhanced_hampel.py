"""
增强Hampel异常检测器
基于综合优化方案的稳健异常检测实现
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
import logging
from scipy import stats
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')


class EnhancedHampelFilter:
    """
    增强Hampel滤波器 - 整合多种策略
    
    特性:
    1. 多窗口尺寸自适应
    2. 工况感知的分块处理  
    3. 多线程并行处理
    4. 统计稳健性增强
    5. 异常值分级处理
    """
    
    def __init__(self, 
                 window_sizes: List[int] = [7, 11, 15],
                 n_sigma: float = 3.0,
                 min_periods: int = 3,
                 regime_aware: bool = True,
                 n_clusters: int = 5,
                 parallel: bool = True,
                 max_workers: int = 4,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            window_sizes: 多个窗口尺寸列表，自适应选择
            n_sigma: MAD倍数阈值
            min_periods: 窗口内最少有效点数
            regime_aware: 是否开启工况感知
            n_clusters: 工况聚类数量
            parallel: 是否并行处理
            max_workers: 最大工作线程数
            logger: 日志器
        """
        self.window_sizes = window_sizes
        self.n_sigma = n_sigma
        self.min_periods = min_periods
        self.regime_aware = regime_aware
        self.n_clusters = n_clusters
        self.parallel = parallel
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        
        # 运行时状态
        self.regime_labels = None
        self.outlier_stats = {}
        self.fitted_params = {}
        
    def _compute_mad_threshold(self, window_data: np.ndarray, n_sigma: float) -> Tuple[float, float]:
        """计算MAD阈值"""
        if len(window_data) < self.min_periods:
            return np.nan, np.nan
            
        median = np.nanmedian(window_data)
        mad = np.nanmedian(np.abs(window_data - median))
        
        # 稳健的MAD计算，避免mad=0的情况
        if mad < 1e-8:
            # 使用IQR作为备选
            q75, q25 = np.nanpercentile(window_data, [75, 25])
            mad = (q75 - q25) / 1.35  # IQR转换为MAD等价值
            
        threshold = n_sigma * mad
        return median, threshold
        
    def _adaptive_window_size(self, data: np.ndarray, base_idx: int) -> int:
        """自适应选择窗口尺寸"""
        # 基于局部方差选择窗口
        local_variances = []
        
        for window_size in self.window_sizes:
            start = max(0, base_idx - window_size // 2)
            end = min(len(data), base_idx + window_size // 2 + 1)
            window = data[start:end]
            
            if len(window) >= self.min_periods:
                var = np.nanvar(window)
                local_variances.append(var)
            else:
                local_variances.append(np.inf)
                
        # 选择方差适中的窗口（既不过于平滑也不过于敏感）
        if local_variances:
            median_var = np.median([v for v in local_variances if v < np.inf])
            best_idx = np.argmin([abs(v - median_var) for v in local_variances])
            return self.window_sizes[best_idx]
        else:
            return self.window_sizes[1]  # 默认中间窗口
            
    def _detect_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """工况检测和聚类"""
        if not self.regime_aware:
            return np.zeros(len(data))
            
        self.logger.info("执行工况聚类...")
        
        # 选择关键特征用于聚类
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # 计算特征的统计特性
        features = []
        for col in numeric_cols:
            if col in data.columns:
                values = data[col].values
                rolling_mean = pd.Series(values).rolling(20, min_periods=5).mean().values
                rolling_std = pd.Series(values).rolling(20, min_periods=5).std().values
                
                features.append(rolling_mean)
                features.append(rolling_std)
        
        if not features:
            return np.zeros(len(data))
            
        feature_matrix = np.column_stack(features)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # KMeans聚类
        try:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(feature_matrix)
            
            self.logger.info(f"检测到{self.n_clusters}种工况")
            return regime_labels
        except Exception as e:
            self.logger.warning(f"工况聚类失败: {e}，使用默认单一工况")
            return np.zeros(len(data))
            
    def _filter_column(self, data: np.ndarray, regime_labels: np.ndarray, col_name: str) -> Tuple[np.ndarray, Dict]:
        """对单列数据执行增强Hampel滤波"""
        filtered_data = data.copy()
        outlier_info = {
            'total_outliers': 0,
            'regime_outliers': {},
            'outlier_indices': [],
            'replacement_values': []
        }
        
        n = len(data)
        
        for i in range(n):
            current_regime = regime_labels[i]
            
            # 自适应窗口尺寸
            window_size = self._adaptive_window_size(data, i)
            
            # 工况感知的窗口选择
            if self.regime_aware:
                # 优先选择同工况的点
                regime_mask = (regime_labels == current_regime)
                regime_indices = np.where(regime_mask)[0]
                
                # 在同工况点中找最近的邻居
                distances = np.abs(regime_indices - i)
                closest_indices = regime_indices[np.argsort(distances)[:window_size]]
                
                if len(closest_indices) >= self.min_periods:
                    window_data = data[closest_indices]
                else:
                    # 回退到常规窗口
                    start = max(0, i - window_size // 2)
                    end = min(n, i + window_size // 2 + 1)
                    window_data = data[start:end]
            else:
                # 常规窗口
                start = max(0, i - window_size // 2)
                end = min(n, i + window_size // 2 + 1)
                window_data = data[start:end]
                
            # 计算MAD阈值
            median, threshold = self._compute_mad_threshold(window_data, self.n_sigma)
            
            if not np.isnan(threshold):
                deviation = abs(data[i] - median)
                
                if deviation > threshold:
                    # 检测到异常值
                    outlier_info['total_outliers'] += 1
                    outlier_info['outlier_indices'].append(i)
                    
                    # 记录按工况的异常值
                    if current_regime not in outlier_info['regime_outliers']:
                        outlier_info['regime_outliers'][current_regime] = 0
                    outlier_info['regime_outliers'][current_regime] += 1
                    
                    # 智能替换策略
                    replacement_value = self._smart_replacement(
                        data, i, window_data, median, deviation, threshold
                    )
                    
                    filtered_data[i] = replacement_value
                    outlier_info['replacement_values'].append(replacement_value)
                    
        return filtered_data, outlier_info
        
    def _smart_replacement(self, data: np.ndarray, idx: int, 
                          window_data: np.ndarray, median: float,
                          deviation: float, threshold: float) -> float:
        """智能替换策略"""
        
        # 异常程度分级
        severity = deviation / threshold
        
        if severity > 5.0:
            # 严重异常：使用窗口中位数
            return median
        elif severity > 2.0:
            # 中等异常：使用加权平均
            weights = np.exp(-0.5 * ((window_data - median) / np.std(window_data)) ** 2)
            weights = weights / np.sum(weights)
            return np.sum(window_data * weights)
        else:
            # 轻微异常：朝中位数方向缩放
            shrinkage = 0.7  # 缩放因子
            return median + shrinkage * (data[idx] - median)
            
    def _process_parallel(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
        """并行处理多列数据"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if self.parallel and len(numeric_cols) > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for col in numeric_cols:
                    future = executor.submit(
                        self._filter_column, 
                        data[col].values, 
                        regime_labels, 
                        col
                    )
                    futures[col] = future
                
                # 收集结果
                filtered_data = data.copy()
                all_stats = {}
                
                for col, future in futures.items():
                    filtered_values, col_stats = future.result()
                    filtered_data[col] = filtered_values
                    all_stats[col] = col_stats
                    
        else:
            # 串行处理
            filtered_data = data.copy()
            all_stats = {}
            
            for col in numeric_cols:
                filtered_values, col_stats = self._filter_column(
                    data[col].values, regime_labels, col
                )
                filtered_data[col] = filtered_values
                all_stats[col] = col_stats
                
        return filtered_data, all_stats
        
    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """拟合并转换数据"""
        self.logger.info("执行增强Hampel滤波...")
        
        # 工况检测
        regime_labels = self._detect_regimes(data)
        self.regime_labels = regime_labels
        
        # 执行滤波
        filtered_data, outlier_stats = self._process_parallel(data, regime_labels)
        
        # 保存统计信息
        self.outlier_stats = outlier_stats
        
        # 汇总统计
        total_outliers = sum(stats['total_outliers'] for stats in outlier_stats.values())
        total_samples = len(data) * len(data.select_dtypes(include=[np.number]).columns)
        outlier_rate = total_outliers / total_samples * 100
        
        self.logger.info(f"Hampel滤波完成: {total_outliers}/{total_samples} ({outlier_rate:.2f}%) 异常值")
        
        summary_stats = {
            'total_outliers': total_outliers,
            'outlier_rate_percent': outlier_rate,
            'regime_counts': dict(zip(*np.unique(regime_labels, return_counts=True))),
            'column_stats': outlier_stats
        }
        
        return filtered_data, summary_stats
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """仅转换（假设已经fit）"""
        if self.regime_labels is None:
            self.logger.warning("未进行fit操作，执行fit_transform")
            filtered_data, _ = self.fit_transform(data)
            return filtered_data
        else:
            filtered_data, _ = self._process_parallel(data, self.regime_labels)
            return filtered_data


class RobustPreprocessor:
    """
    稳健预处理器 - 整合增强Hampel和增量计算
    来自综合优化方案的实现
    """
    
    def __init__(self, 
                 hampel_config: Optional[Dict] = None,
                 incremental_stats: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            hampel_config: Hampel滤波器配置
            incremental_stats: 是否使用增量统计
            logger: 日志器
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化增强Hampel滤波器
        hampel_config = hampel_config or {}
        self.hampel_filter = EnhancedHampelFilter(
            window_sizes=hampel_config.get('window_sizes', [7, 11, 15]),
            n_sigma=hampel_config.get('n_sigma', 3.0),
            regime_aware=hampel_config.get('regime_aware', True),
            parallel=hampel_config.get('parallel', True),
            logger=self.logger
        )
        
        # 增量统计计算器
        self.incremental_stats = incremental_stats
        self.stats_cache = {}
        self.training = True
        
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """拟合并变换数据"""
        # 1. 增强Hampel异常检测
        clean_data, outlier_stats = self.hampel_filter.fit_transform(data)
        
        # 2. 增量特征计算（如果启用）
        if self.incremental_stats:
            processed_data = self._compute_incremental_features(clean_data)
        else:
            processed_data = clean_data
            
        self.logger.info(f"预处理完成: {data.shape} -> {processed_data.shape}")
        return processed_data
        
    def _compute_incremental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """增量特征计算 - 避免重复计算"""
        # TODO: 在下一个任务中实现
        return data
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """变换数据（推理时使用）"""
        self.training = False
        clean_data = self.hampel_filter.transform(data)
        
        if self.incremental_stats:
            processed_data = self._compute_incremental_features(clean_data)
        else:
            processed_data = clean_data
            
        return processed_data