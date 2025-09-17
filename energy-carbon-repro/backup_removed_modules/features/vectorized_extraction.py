"""
向量化特征提取模块
Day 2 优化：批量特征计算，预期40-50%性能提升
"""
import numpy as np
import pandas as pd
import time
import warnings
from tqdm import tqdm
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

class VectorizedFeatureExtractor:
    """向量化特征提取器，批量处理窗口"""

    def __init__(self, window_size=180, stride=30):
        self.window_size = window_size
        self.stride = stride
        self.feature_names = []

        # 性能统计
        self.performance_stats = {
            'total_windows': 0,
            'extraction_time': 0.0,
            'vectorization_efficiency': 0.0,
            'memory_usage_mb': 0.0
        }

    def extract_all_windows_vectorized(self, data):
        """向量化提取所有窗口特征"""
        start_time = time.time()

        n_samples, n_features = data.shape
        n_windows = (n_samples - self.window_size) // self.stride + 1
        self.performance_stats['total_windows'] = n_windows

        # 生成特征名称
        self.feature_names = self._generate_feature_names(n_features)

        print(f"  向量化处理 {n_windows} 个窗口，特征数: {len(self.feature_names)}")
        print(f"    数据形状: {data.shape}")
        print(f"    窗口大小: {self.window_size}, 步长: {self.stride}")

        # 批量创建窗口视图（内存高效）
        windows = self._create_sliding_window_view(data)
        print(f"    窗口视图形状: {windows.shape}")

        # 并行计算所有窗口的特征
        features = self._compute_batch_features(windows)

        # 记录性能统计
        self.performance_stats['extraction_time'] = time.time() - start_time
        self.performance_stats['vectorization_efficiency'] = n_windows / self.performance_stats['extraction_time']

        print(f"  向量化特征提取完成，耗时: {self.performance_stats['extraction_time']:.2f}秒")
        print(f"    处理效率: {self.performance_stats['vectorization_efficiency']:.1f} 窗口/秒")
        print(f"    生成特征形状: {features.shape}")

        return features, self.feature_names

    def _generate_feature_names(self, n_features):
        """生成特征名称"""
        feature_names = []

        for col_idx in range(n_features):
            col_name = f'feature_{col_idx}'

            # 为每列生成所有特征名（与传统方法保持一致）
            feature_names.extend([
                f'{col_name}_mean',
                f'{col_name}_std',
                f'{col_name}_skew',
                f'{col_name}_kurtosis',
                f'{col_name}_rms',
                f'{col_name}_mean_change_rate',
                f'{col_name}_change_rate_std',
                f'{col_name}_imf1_freq',  # EMD placeholder (保持兼容性)
                f'{col_name}_imf1_energy_ratio'  # EMD placeholder (保持兼容性)
            ])

        return feature_names

    def _create_sliding_window_view(self, data):
        """创建滑动窗口视图（避免数据复制）"""
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            # 创建二维滑动窗口
            window_shape = (self.window_size, data.shape[1])
            windows = sliding_window_view(data, window_shape)[::self.stride, 0]

            return windows
        except ImportError:
            # 兼容较老的numpy版本
            return self._create_sliding_window_manual(data)

    def _create_sliding_window_manual(self, data):
        """手动创建滑动窗口（兼容性实现）"""
        n_samples, n_features = data.shape
        n_windows = (n_samples - self.window_size) // self.stride + 1

        # 预分配窗口数组
        windows = np.zeros((n_windows, self.window_size, n_features))

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            windows[i] = data[start_idx:end_idx]

        return windows

    def _compute_batch_features(self, windows):
        """批量计算特征"""
        n_windows, window_size, n_cols = windows.shape
        n_features_per_col = 9  # mean, std, skew, kurt, rms, mcr, crs, imf1_freq, imf1_energy_ratio

        features = np.zeros((n_windows, n_cols * n_features_per_col))

        print("    开始批量特征计算...")

        for col_idx in tqdm(range(n_cols), desc="      处理特征列"):
            col_data = windows[:, :, col_idx]  # (n_windows, window_size)

            start_idx = col_idx * n_features_per_col
            end_idx = start_idx + n_features_per_col

            # 批量计算所有窗口的统计量
            features[:, start_idx:end_idx] = self._batch_compute_stats(col_data)

        return features

    def _batch_compute_stats(self, col_data):
        """批量计算单列所有窗口的统计特征"""
        # col_data shape: (n_windows, window_size)

        # 基础统计量（向量化）
        means = np.mean(col_data, axis=1)
        stds = np.std(col_data, axis=1)

        # 鲁棒的高阶矩（批量处理）
        skews, kurts = self._batch_safe_moments(col_data)

        # RMS（向量化）
        rms = np.sqrt(np.mean(col_data**2, axis=1))

        # 变化率（向量化）
        diffs = np.diff(col_data, axis=1)
        mean_change_rates = np.mean(diffs, axis=1)
        change_rate_stds = np.std(diffs, axis=1)

        # EMD占位符特征（保持与传统方法兼容）
        n_windows = col_data.shape[0]
        imf1_freq = np.zeros(n_windows)  # EMD placeholder
        imf1_energy_ratio = np.zeros(n_windows)  # EMD placeholder

        # 组合特征（与传统方法保持一致）
        return np.column_stack([means, stds, skews, kurts, rms, mean_change_rates, change_rate_stds, imf1_freq, imf1_energy_ratio])

    def _batch_safe_moments(self, data):
        """批量安全计算偏度和峰度"""
        n_windows, window_size = data.shape

        skews = np.zeros(n_windows)
        kurts = np.zeros(n_windows)

        # 检查每个窗口的变异性
        stds = np.std(data, axis=1)
        valid_mask = stds > 1e-10

        if np.any(valid_mask):
            # 对有效窗口计算矩
            valid_data = data[valid_mask]

            # 标准化数据（提高数值稳定性）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                means = np.mean(valid_data, axis=1, keepdims=True)
                stds_valid = np.std(valid_data, axis=1, keepdims=True)

                # 避免除零
                stds_valid = np.where(stds_valid > 1e-15, stds_valid, 1.0)
                standardized = (valid_data - means) / stds_valid

                # 计算三阶和四阶中心矩
                m3 = np.mean(standardized**3, axis=1)
                m4 = np.mean(standardized**4, axis=1)

                skews[valid_mask] = m3
                kurts[valid_mask] = m4 - 3  # 超额峰度

        return skews, kurts

    def get_performance_stats(self):
        """获取性能统计信息"""
        return self.performance_stats.copy()

    def validate_feature_array(self, features):
        """验证特征数组质量"""
        if features is None or len(features) == 0:
            return False

        # 检查NaN和Inf
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False

        # 检查特征方差（避免常数特征）
        feature_vars = np.var(features, axis=0)
        if np.any(feature_vars < 1e-15):
            warnings.warn("发现低方差特征，可能影响模型性能")

        return True


class BatchFeatureProcessor:
    """批量特征处理器，集成缓存和优化"""

    def __init__(self, window_size=180, stride=30, use_cache=True):
        self.extractor = VectorizedFeatureExtractor(window_size, stride)
        self.use_cache = use_cache
        self.cache = {}

    def process_with_caching(self, data, cache_key=None):
        """带缓存的特征处理"""
        if cache_key and self.use_cache and cache_key in self.cache:
            print("  从缓存加载特征...")
            return self.cache[cache_key]

        # 提取特征
        features, feature_names = self.extractor.extract_all_windows_vectorized(data)

        # 验证特征质量
        if not self.extractor.validate_feature_array(features):
            raise ValueError("特征提取验证失败")

        # 缓存结果
        if cache_key and self.use_cache:
            self.cache[cache_key] = (features, feature_names)
            print(f"  特征已缓存，键: {cache_key}")

        return features, feature_names

    def get_cache_stats(self):
        """获取缓存统计"""
        return {
            'cache_entries': len(self.cache),
            'cache_enabled': self.use_cache,
            'total_cache_size_mb': sum(arr[0].nbytes for arr in self.cache.values()) / 1024 / 1024
        }


# 使用示例和测试
if __name__ == '__main__':
    print("=== 向量化特征提取器测试 ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 10000
    n_features = 4
    test_data = np.random.randn(n_samples, n_features)

    # 添加一些模式
    t = np.linspace(0, 10*np.pi, n_samples)
    test_data[:, 0] += np.sin(t)  # 正弦模式
    test_data[:, 1] += 0.5 * np.cos(2*t)  # 余弦模式
    test_data[:, 2] += np.cumsum(np.random.randn(n_samples)) * 0.1  # 随机游走
    test_data[:, 3] += np.where(np.random.rand(n_samples) > 0.95, 5, 0)  # 稀疏脉冲

    # 测试向量化提取器
    extractor = VectorizedFeatureExtractor(window_size=180, stride=30)

    start_time = time.time()
    features, feature_names = extractor.extract_all_windows_vectorized(test_data)
    extraction_time = time.time() - start_time

    print(f"\n测试结果:")
    print(f"  输入数据形状: {test_data.shape}")
    print(f"  输出特征形状: {features.shape}")
    print(f"  特征名称数量: {len(feature_names)}")
    print(f"  提取时间: {extraction_time:.3f}秒")
    print(f"  特征质量检查: {'✅ 通过' if extractor.validate_feature_array(features) else '❌ 失败'}")

    # 性能统计
    stats = extractor.get_performance_stats()
    print(f"\n性能统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n=== 向量化特征提取器测试完成 ===")