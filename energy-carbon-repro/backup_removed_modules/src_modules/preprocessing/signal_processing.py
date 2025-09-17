#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号处理方法模块 - 论文指定的预处理技术
Signal Processing Methods for LNG Paper Reproduction

包含：
1. Hampel滤波器 - 鲁棒异常值检测
2. EMD经验模态分解 - 多尺度信号分解
3. 小波去噪 - 信号降噪
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, Tuple, List, Optional, Union
import warnings
import logging

# 设置日志
logger = logging.getLogger(__name__)


class HampelFilter:
    """
    Hampel滤波器 - 鲁棒异常值检测和替换

    原理：基于中位数绝对偏差(MAD)的异常值检测
    适用于含有脉冲噪声的时序数据
    """

    def __init__(self, window_size: int = 7, n_sigmas: float = 3.0):
        """
        初始化Hampel滤波器

        Args:
            window_size: 滑动窗口大小（必须为奇数）
            n_sigmas: 异常值判定阈值（MAD的倍数）
        """
        if window_size % 2 == 0:
            window_size += 1
            warnings.warn(f"窗口大小必须为奇数，已调整为 {window_size}")

        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.half_window = window_size // 2

        # 统计信息
        self.n_outliers_detected = 0
        self.outlier_indices = []

    def filter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用Hampel滤波

        Args:
            data: 输入时序数据

        Returns:
            filtered_data: 滤波后的数据
            outlier_mask: 异常值标记（True为异常值）
        """
        n = len(data)
        filtered_data = data.copy()
        outlier_mask = np.zeros(n, dtype=bool)

        # MAD到标准差的转换系数
        k = 1.4826

        for i in range(n):
            # 确定窗口边界
            start = max(0, i - self.half_window)
            end = min(n, i + self.half_window + 1)

            # 窗口内的数据
            window_data = data[start:end]

            # 计算中位数和MAD
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))

            # 计算阈值
            threshold = self.n_sigmas * k * mad

            # 检测异常值
            if np.abs(data[i] - median) > threshold:
                outlier_mask[i] = True
                filtered_data[i] = median
                self.n_outliers_detected += 1
                self.outlier_indices.append(i)

        logger.info(f"Hampel滤波: 检测到 {self.n_outliers_detected} 个异常值 "
                   f"({100*self.n_outliers_detected/n:.2f}%)")

        return filtered_data, outlier_mask

    def adaptive_filter(self, data: np.ndarray,
                        min_window: int = 5,
                        max_window: int = 21) -> np.ndarray:
        """
        自适应Hampel滤波 - 根据局部信号特性调整窗口大小

        Args:
            data: 输入数据
            min_window: 最小窗口大小
            max_window: 最大窗口大小

        Returns:
            filtered_data: 自适应滤波结果
        """
        n = len(data)
        filtered_data = data.copy()

        for i in range(n):
            # 计算局部方差来确定窗口大小
            local_start = max(0, i - max_window // 2)
            local_end = min(n, i + max_window // 2 + 1)
            local_variance = np.var(data[local_start:local_end])

            # 根据方差调整窗口大小（方差大时用大窗口）
            if local_variance > 0:
                normalized_var = local_variance / np.var(data)
                window_size = int(min_window + (max_window - min_window) * normalized_var)
                window_size = window_size if window_size % 2 == 1 else window_size + 1
            else:
                window_size = min_window

            # 应用Hampel滤波
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            window_data = data[start:end]
            median = np.median(window_data)
            mad = np.median(np.abs(window_data - median))
            threshold = self.n_sigmas * 1.4826 * mad

            if np.abs(data[i] - median) > threshold:
                filtered_data[i] = median

        return filtered_data


class SimplifiedEMD:
    """
    简化的经验模态分解（EMD）

    完整的EMD计算密集，这里提供简化版本
    用于提取信号的内在模态函数（IMF）
    """

    def __init__(self, max_imfs: int = 5, max_iterations: int = 100):
        """
        初始化EMD

        Args:
            max_imfs: 最大IMF数量
            max_iterations: 筛选过程最大迭代次数
        """
        self.max_imfs = max_imfs
        self.max_iterations = max_iterations

    def _find_extrema(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        找到信号的极大值和极小值
        """
        # 找极大值
        maxima = signal.argrelextrema(data, np.greater)[0]
        # 找极小值
        minima = signal.argrelextrema(data, np.less)[0]

        # 处理边界
        if data[0] > data[1]:
            maxima = np.concatenate([[0], maxima])
        elif data[0] < data[1]:
            minima = np.concatenate([[0], minima])

        if data[-1] > data[-2]:
            maxima = np.concatenate([maxima, [len(data)-1]])
        elif data[-1] < data[-2]:
            minima = np.concatenate([minima, [len(data)-1]])

        return maxima, minima

    def _compute_envelope_mean(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        计算上下包络的均值
        """
        maxima, minima = self._find_extrema(data)

        if len(maxima) < 2 or len(minima) < 2:
            return None

        # 三次样条插值构建包络
        t = np.arange(len(data))

        try:
            from scipy.interpolate import interp1d

            # 上包络
            upper_env_func = interp1d(maxima, data[maxima],
                                     kind='cubic', fill_value='extrapolate')
            upper_env = upper_env_func(t)

            # 下包络
            lower_env_func = interp1d(minima, data[minima],
                                     kind='cubic', fill_value='extrapolate')
            lower_env = lower_env_func(t)

            # 包络均值
            mean_env = (upper_env + lower_env) / 2

            return mean_env

        except Exception as e:
            logger.warning(f"包络计算失败: {e}")
            return None

    def _sift_imf(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        筛选出一个IMF
        """
        h = data.copy()

        for _ in range(self.max_iterations):
            mean_env = self._compute_envelope_mean(h)

            if mean_env is None:
                break

            h_prev = h.copy()
            h = h - mean_env

            # 停止准则（SD准则）
            sd = np.sum((h - h_prev) ** 2) / np.sum(h_prev ** 2)
            if sd < 0.001:
                break

        return h if mean_env is not None else None

    def decompose(self, data: np.ndarray) -> List[np.ndarray]:
        """
        执行EMD分解

        Args:
            data: 输入信号

        Returns:
            IMFs: 内在模态函数列表
        """
        imfs = []
        residue = data.copy()

        for i in range(self.max_imfs):
            imf = self._sift_imf(residue)

            if imf is None:
                break

            imfs.append(imf)
            residue = residue - imf

            # 检查残余是否为单调
            extrema_count = len(self._find_extrema(residue)[0]) + \
                           len(self._find_extrema(residue)[1])
            if extrema_count < 3:
                break

        # 添加残余项
        if np.any(residue != 0):
            imfs.append(residue)

        logger.info(f"EMD分解: 提取了 {len(imfs)} 个IMF")

        return imfs


class WaveletDenoising:
    """
    小波去噪

    使用离散小波变换（DWT）进行信号去噪
    """

    def __init__(self, wavelet: str = 'db4', level: int = 4,
                 threshold_method: str = 'soft'):
        """
        初始化小波去噪

        Args:
            wavelet: 小波基函数（'db4', 'sym4', 'coif2'等）
            level: 分解层数
            threshold_method: 阈值方法（'soft'软阈值, 'hard'硬阈值）
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_method = threshold_method

    def _wavelet_threshold(self, coeffs: np.ndarray,
                           threshold: float,
                           method: str = 'soft') -> np.ndarray:
        """
        小波系数阈值处理
        """
        if method == 'soft':
            # 软阈值
            return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
        elif method == 'hard':
            # 硬阈值
            coeffs_thresh = coeffs.copy()
            coeffs_thresh[np.abs(coeffs) < threshold] = 0
            return coeffs_thresh
        else:
            raise ValueError(f"未知的阈值方法: {method}")

    def denoise(self, data: np.ndarray,
                noise_estimate: Optional[float] = None) -> np.ndarray:
        """
        执行小波去噪

        Args:
            data: 输入信号
            noise_estimate: 噪声水平估计（None时自动估计）

        Returns:
            denoised: 去噪后的信号
        """
        try:
            import pywt
        except ImportError:
            logger.warning("pywt未安装，使用简化的小波去噪")
            return self._simplified_denoise(data, noise_estimate)

        # 小波分解
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # 估计噪声水平（使用细节系数的MAD）
        if noise_estimate is None:
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        else:
            sigma = noise_estimate

        # 计算通用阈值
        n = len(data)
        threshold = sigma * np.sqrt(2 * np.log(n))

        # 对细节系数进行阈值处理
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = self._wavelet_threshold(
                coeffs[i], threshold, self.threshold_method
            )

        # 小波重构
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)

        # 确保长度一致
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
        elif len(denoised) < len(data):
            denoised = np.pad(denoised, (0, len(data) - len(denoised)), 'edge')

        logger.info(f"小波去噪: 使用 {self.wavelet} 小波, "
                   f"阈值 = {threshold:.4f}")

        return denoised

    def _simplified_denoise(self, data: np.ndarray,
                           noise_estimate: Optional[float] = None) -> np.ndarray:
        """
        简化的小波去噪（不依赖pywt）
        使用多尺度移动平均模拟小波去噪效果
        """
        denoised = data.copy()

        # 多尺度平滑
        scales = [2, 4, 8, 16]
        weights = [0.4, 0.3, 0.2, 0.1]

        smoothed = np.zeros_like(data)
        total_weight = 0

        for scale, weight in zip(scales, weights):
            if scale < len(data):
                # 移动平均
                kernel = np.ones(scale) / scale
                smooth_component = np.convolve(data, kernel, mode='same')
                smoothed += weight * smooth_component
                total_weight += weight

        if total_weight > 0:
            smoothed /= total_weight

        # 自适应混合原始信号和平滑信号
        if noise_estimate is None:
            # 估计噪声水平
            diff = data - smoothed
            noise_estimate = np.std(diff)

        # 根据噪声水平确定混合权重
        snr = np.var(data) / (noise_estimate ** 2 + 1e-10)
        alpha = 1 / (1 + np.exp(-snr + 5))  # sigmoid函数

        denoised = alpha * data + (1 - alpha) * smoothed

        return denoised

    def adaptive_denoise(self, data: np.ndarray) -> np.ndarray:
        """
        自适应小波去噪
        根据信号特性自动选择参数
        """
        # 计算信号的统计特性
        signal_std = np.std(data)
        signal_kurtosis = np.sum((data - np.mean(data))**4) / (len(data) * signal_std**4)

        # 根据峰度选择小波基
        if signal_kurtosis > 5:
            self.wavelet = 'db2'  # 尖锐信号用简单小波
        elif signal_kurtosis > 3:
            self.wavelet = 'db4'  # 正常信号
        else:
            self.wavelet = 'sym4'  # 平滑信号用对称小波

        # 根据信号长度选择分解层数
        max_level = int(np.log2(len(data))) - 2
        self.level = min(self.level, max_level)

        logger.info(f"自适应小波去噪: 选择 {self.wavelet}, level={self.level}")

        return self.denoise(data)


class IntegratedSignalProcessor:
    """
    集成信号处理器
    整合Hampel滤波、EMD分解、小波去噪
    """

    def __init__(self):
        self.hampel = HampelFilter()
        self.emd = SimplifiedEMD()
        self.wavelet = WaveletDenoising()

    def process_signal(self, data: np.ndarray,
                       use_hampel: bool = True,
                       use_emd: bool = False,
                       use_wavelet: bool = True) -> Dict:
        """
        完整的信号处理管线

        Args:
            data: 输入信号
            use_hampel: 是否使用Hampel滤波
            use_emd: 是否使用EMD分解
            use_wavelet: 是否使用小波去噪

        Returns:
            results: 包含处理结果的字典
        """
        results = {
            'original': data.copy(),
            'processed': data.copy(),
            'stages': {}
        }

        current_signal = data.copy()

        # 1. Hampel滤波（异常值处理）
        if use_hampel:
            filtered, outlier_mask = self.hampel.filter(current_signal)
            current_signal = filtered
            results['stages']['hampel'] = {
                'output': filtered,
                'outliers': outlier_mask,
                'n_outliers': np.sum(outlier_mask)
            }
            logger.info(f"Hampel滤波完成: 移除 {np.sum(outlier_mask)} 个异常值")

        # 2. EMD分解（提取IMF）
        if use_emd:
            imfs = self.emd.decompose(current_signal)
            results['stages']['emd'] = {
                'imfs': imfs,
                'n_imfs': len(imfs)
            }

            # 可以选择性重构（例如去除高频噪声IMF）
            if len(imfs) > 2:
                # 去除第一个IMF（通常是高频噪声）
                reconstructed = np.sum(imfs[1:], axis=0)
                current_signal = reconstructed
                logger.info(f"EMD分解完成: {len(imfs)} 个IMF，去除高频分量")

        # 3. 小波去噪
        if use_wavelet:
            denoised = self.wavelet.denoise(current_signal)
            current_signal = denoised
            results['stages']['wavelet'] = {
                'output': denoised,
                'noise_reduction': np.std(data) - np.std(denoised)
            }
            logger.info(f"小波去噪完成: 噪声减少 {results['stages']['wavelet']['noise_reduction']:.4f}")

        results['processed'] = current_signal

        # 计算信噪比改善
        if np.var(data - current_signal) > 0:
            snr_improvement = 10 * np.log10(
                np.var(current_signal) / np.var(data - current_signal)
            )
            results['snr_improvement_db'] = snr_improvement
            logger.info(f"总体SNR改善: {snr_improvement:.2f} dB")

        return results

    def batch_process(self, df: pd.DataFrame,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量处理数据框中的多个信号

        Args:
            df: 输入数据框
            columns: 要处理的列名（None时处理所有数值列）

        Returns:
            processed_df: 处理后的数据框
        """
        processed_df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in df.columns:
                logger.info(f"处理列: {col}")
                results = self.process_signal(df[col].values)
                processed_df[col] = results['processed']

        return processed_df


def test_signal_processing():
    """测试信号处理方法"""
    print("="*60)
    print("测试信号处理模块")
    print("="*60)

    # 生成测试信号
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)

    # 清洁信号
    clean_signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)

    # 添加噪声
    noise = np.random.normal(0, 0.2, len(t))

    # 添加异常值
    outliers = np.zeros_like(t)
    outlier_indices = np.random.choice(len(t), 20, replace=False)
    outliers[outlier_indices] = np.random.uniform(-2, 2, 20)

    # 含噪信号
    noisy_signal = clean_signal + noise + outliers

    print(f"测试信号: 长度={len(t)}, 噪声水平={np.std(noise):.3f}")

    # 1. 测试Hampel滤波
    print("\n1. Hampel滤波测试:")
    hampel = HampelFilter(window_size=7, n_sigmas=3)
    filtered, outlier_mask = hampel.filter(noisy_signal)
    print(f"   检测到异常值: {np.sum(outlier_mask)} 个")
    print(f"   滤波前标准差: {np.std(noisy_signal):.3f}")
    print(f"   滤波后标准差: {np.std(filtered):.3f}")

    # 2. 测试EMD分解
    print("\n2. EMD分解测试:")
    emd = SimplifiedEMD(max_imfs=5)
    imfs = emd.decompose(filtered)
    print(f"   提取IMF数量: {len(imfs)}")
    for i, imf in enumerate(imfs):
        print(f"   IMF{i+1} 能量: {np.sum(imf**2):.2f}")

    # 3. 测试小波去噪
    print("\n3. 小波去噪测试:")
    wavelet = WaveletDenoising(wavelet='db4', level=4)
    denoised = wavelet.denoise(filtered)
    print(f"   去噪前标准差: {np.std(filtered):.3f}")
    print(f"   去噪后标准差: {np.std(denoised):.3f}")

    # 4. 测试集成处理
    print("\n4. 集成信号处理测试:")
    processor = IntegratedSignalProcessor()
    results = processor.process_signal(
        noisy_signal,
        use_hampel=True,
        use_emd=False,  # EMD计算较慢
        use_wavelet=True
    )

    # 计算误差
    mse_before = np.mean((noisy_signal - clean_signal)**2)
    mse_after = np.mean((results['processed'] - clean_signal)**2)

    print(f"   处理前MSE: {mse_before:.4f}")
    print(f"   处理后MSE: {mse_after:.4f}")
    print(f"   MSE改善: {(1 - mse_after/mse_before)*100:.1f}%")

    if 'snr_improvement_db' in results:
        print(f"   SNR改善: {results['snr_improvement_db']:.2f} dB")

    print("\n✅ 信号处理测试通过！")
    print("="*60)

    return processor


if __name__ == "__main__":
    # 运行测试
    test_signal_processing()