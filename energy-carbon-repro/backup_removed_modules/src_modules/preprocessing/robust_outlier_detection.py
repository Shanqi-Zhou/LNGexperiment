"""
鲁棒异常检测模块 - Week 2 优化
实现优化的Hampel滤波器，提升数据质量和模型性能
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from typing import Union, Tuple, Optional
import time


class OptimizedHampelFilter:
    """优化的Hampel滤波器，用于时序数据异常检测和修复"""

    def __init__(self, window: int = 11, n_sigma: float = 3.0, min_periods: int = 5):
        """
        初始化Hampel滤波器

        Args:
            window: 滑动窗口大小
            n_sigma: 异常阈值倍数
            min_periods: 计算所需的最小有效数据点数
        """
        self.window = window
        self.n_sigma = n_sigma
        self.min_periods = min_periods

        # 性能统计
        self.stats = {
            'total_points': 0,
            'outliers_detected': 0,
            'outliers_replaced': 0,
            'processing_time': 0.0
        }

    def filter_vectorized(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        向量化Hampel滤波，高效处理多列数据

        Args:
            data: 输入数据（numpy数组或pandas DataFrame）

        Returns:
            滤波后的数据，格式与输入保持一致
        """
        start_time = time.time()

        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
            return_numpy = True
        else:
            df = data.copy()
            return_numpy = False

        self.stats['total_points'] = df.size

        # 向量化滚动计算 - 核心优化
        print(f"    执行Hampel滤波，窗口大小: {self.window}, 阈值: {self.n_sigma}σ")

        # 滚动中位数 - 鲁棒的位置估计
        rolling_median = df.rolling(
            window=self.window,
            center=True,
            min_periods=self.min_periods
        ).median()

        # 滚动MAD (Median Absolute Deviation) - 鲁棒的尺度估计
        rolling_mad = df.rolling(
            window=self.window,
            center=True,
            min_periods=self.min_periods
        ).apply(lambda x: self._robust_mad(x), raw=False)

        # 计算异常阈值
        # 1.4826是MAD到标准差的转换因子（假设正态分布）
        threshold = self.n_sigma * 1.4826 * rolling_mad

        # 识别异常值
        outlier_mask = np.abs(df - rolling_median) > threshold

        # 统计异常值
        outliers_detected = outlier_mask.sum().sum()
        self.stats['outliers_detected'] = outliers_detected

        # 替换异常值
        filtered_data = df.copy()
        filtered_data[outlier_mask] = rolling_median[outlier_mask]

        # 统计替换的异常值
        replaced_mask = ~pd.isna(rolling_median[outlier_mask])
        self.stats['outliers_replaced'] = replaced_mask.sum() if hasattr(replaced_mask, 'sum') else 0

        # 记录性能
        self.stats['processing_time'] = time.time() - start_time

        print(f"    异常检测完成: 发现 {outliers_detected} 个异常值, 替换 {self.stats['outliers_replaced']} 个")
        print(f"    处理时间: {self.stats['processing_time']:.3f}秒")

        return filtered_data.values if return_numpy else filtered_data

    def _robust_mad(self, x: pd.Series) -> float:
        """
        计算鲁棒的中位数绝对偏差 (MAD)

        Args:
            x: 数据序列

        Returns:
            MAD值
        """
        if len(x) < 3:  # 太少数据点无法计算可靠的MAD
            return np.nan

        # 移除NaN值
        valid_x = x.dropna()
        if len(valid_x) < 3:
            return np.nan

        median_val = np.median(valid_x)
        mad_val = np.median(np.abs(valid_x - median_val))

        # 避免MAD为零的情况
        return max(mad_val, 1e-10)

    def batch_filter_multiple_series(self, data_dict: dict) -> dict:
        """
        批量处理多个时序数据

        Args:
            data_dict: {series_name: data} 的字典

        Returns:
            滤波后的数据字典
        """
        print(f"  批量Hampel滤波: {len(data_dict)} 个序列")

        results = {}
        total_outliers = 0

        for name, data in data_dict.items():
            print(f"    处理序列: {name}")

            filtered_data = self.filter_vectorized(data)
            results[name] = filtered_data
            total_outliers += self.stats['outliers_detected']

        print(f"  批量处理完成，总计发现 {total_outliers} 个异常值")
        return results

    def get_outlier_statistics(self) -> dict:
        """获取异常检测统计信息"""
        if self.stats['total_points'] == 0:
            return {"message": "尚未处理任何数据"}

        outlier_rate = (self.stats['outliers_detected'] / self.stats['total_points']) * 100
        replacement_rate = (self.stats['outliers_replaced'] / max(self.stats['outliers_detected'], 1)) * 100

        return {
            'total_data_points': self.stats['total_points'],
            'outliers_detected': self.stats['outliers_detected'],
            'outliers_replaced': self.stats['outliers_replaced'],
            'outlier_rate_percent': outlier_rate,
            'replacement_success_rate_percent': replacement_rate,
            'processing_time_seconds': self.stats['processing_time'],
            'throughput_points_per_second': self.stats['total_points'] / max(self.stats['processing_time'], 1e-6)
        }


class AdaptiveHampelFilter(OptimizedHampelFilter):
    """
    自适应Hampel滤波器
    根据数据特征自动调整参数
    """

    def __init__(self, base_window: int = 11, adaptive_threshold: bool = True):
        super().__init__(window=base_window)
        self.base_window = base_window
        self.adaptive_threshold = adaptive_threshold

    def filter_adaptive(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        自适应滤波，根据数据特征调整参数

        Args:
            data: 输入数据

        Returns:
            滤波后的数据
        """
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
            return_numpy = True
        else:
            df = data.copy()
            return_numpy = False

        print("    执行自适应Hampel滤波...")

        # 分析数据特征
        data_volatility = self._analyze_volatility(df)

        # 根据数据波动性调整参数
        if data_volatility > 2.0:  # 高波动性数据
            self.window = int(self.base_window * 1.5)  # 增大窗口
            self.n_sigma = 2.5  # 降低阈值敏感性
            print(f"    检测到高波动性数据，调整参数: window={self.window}, n_sigma={self.n_sigma}")
        elif data_volatility < 0.5:  # 低波动性数据
            self.window = max(7, int(self.base_window * 0.8))  # 减小窗口
            self.n_sigma = 3.5  # 提高阈值敏感性
            print(f"    检测到低波动性数据，调整参数: window={self.window}, n_sigma={self.n_sigma}")
        else:  # 中等波动性数据
            self.window = self.base_window
            self.n_sigma = 3.0
            print(f"    使用标准参数: window={self.window}, n_sigma={self.n_sigma}")

        # 执行滤波
        result = self.filter_vectorized(df if not return_numpy else df.values)

        return result

    def _analyze_volatility(self, df: pd.DataFrame) -> float:
        """
        分析数据波动性

        Args:
            df: 输入数据框

        Returns:
            波动性指标（变异系数的平均值）
        """
        # 计算每列的变异系数
        cv_values = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 10:  # 足够的数据点
                mean_val = np.mean(series)
                std_val = np.std(series)
                if abs(mean_val) > 1e-10:
                    cv = std_val / abs(mean_val)
                    cv_values.append(cv)

        return np.mean(cv_values) if cv_values else 1.0


# 便捷函数
def hampel_filter_data(data: Union[np.ndarray, pd.DataFrame],
                      window: int = 11,
                      n_sigma: float = 3.0,
                      adaptive: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """
    便捷的Hampel滤波函数

    Args:
        data: 输入数据
        window: 滑动窗口大小
        n_sigma: 异常阈值倍数
        adaptive: 是否使用自适应参数

    Returns:
        滤波后的数据
    """
    if adaptive:
        filter_obj = AdaptiveHampelFilter(base_window=window)
        return filter_obj.filter_adaptive(data)
    else:
        filter_obj = OptimizedHampelFilter(window=window, n_sigma=n_sigma)
        return filter_obj.filter_vectorized(data)


# 使用示例和测试
if __name__ == '__main__':
    print("=== Hampel滤波器测试 ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000

    # 生成带异常值的时序数据
    t = np.linspace(0, 10, n_samples)
    signal = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(n_samples)

    # 人工添加异常值
    outlier_indices = np.random.choice(n_samples, size=50, replace=False)
    signal[outlier_indices] += np.random.randn(50) * 5  # 强异常值

    # 创建多列测试数据
    test_data = pd.DataFrame({
        'signal_1': signal,
        'signal_2': signal * 0.5 + np.random.randn(n_samples) * 0.2,
        'signal_3': np.cumsum(np.random.randn(n_samples)) * 0.1 + signal * 0.3
    })

    print(f"测试数据: {test_data.shape}, 人工异常值: {len(outlier_indices)}")

    # 测试标准Hampel滤波
    print("\n--- 标准Hampel滤波测试 ---")
    hampel_filter = OptimizedHampelFilter(window=15, n_sigma=3.0)

    start_time = time.time()
    filtered_data = hampel_filter.filter_vectorized(test_data)
    processing_time = time.time() - start_time

    print(f"处理时间: {processing_time:.3f}秒")
    print("异常检测统计:", hampel_filter.get_outlier_statistics())

    # 测试自适应Hampel滤波
    print("\n--- 自适应Hampel滤波测试 ---")
    adaptive_filter = AdaptiveHampelFilter(base_window=15)

    start_time = time.time()
    adaptive_filtered = adaptive_filter.filter_adaptive(test_data)
    adaptive_time = time.time() - start_time

    print(f"处理时间: {adaptive_time:.3f}秒")
    print("异常检测统计:", adaptive_filter.get_outlier_statistics())

    # 测试批量处理
    print("\n--- 批量处理测试 ---")
    data_dict = {
        'series_A': test_data['signal_1'].values,
        'series_B': test_data['signal_2'].values,
        'series_C': test_data['signal_3'].values
    }

    batch_results = hampel_filter.batch_filter_multiple_series(data_dict)
    print(f"批量处理完成，处理了 {len(batch_results)} 个序列")

    print("\n=== Hampel滤波器测试完成 ===")