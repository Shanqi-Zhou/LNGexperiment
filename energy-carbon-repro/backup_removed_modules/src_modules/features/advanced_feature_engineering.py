#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级特征工程模块 - 论文规格实现
Advanced Feature Engineering for LNG Paper Reproduction

特征体系：
- 动态特征：9类 × K维（时域、频域、统计特征）
- 静态特征：32维（设备参数、工况、时间编码）
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
import logging

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特征配置类"""
    # 时序窗口参数
    window_size: int = 180  # 30分钟，10秒分辨率
    stride: int = 30  # 5分钟步长

    # 动态特征配置
    n_statistical: int = 5  # 统计特征数量
    n_frequency: int = 10  # 频域特征数量
    n_temporal: int = 6  # 时序特征数量

    # 静态特征配置
    n_device_params: int = 12  # 设备参数特征
    n_operating_mode: int = 8  # 工况特征
    n_temporal_encoding: int = 12  # 时间编码特征

    # 信号处理参数
    use_hampel: bool = True  # 使用Hampel滤波
    use_wavelet: bool = True  # 使用小波去噪
    use_emd: bool = False  # EMD计算密集，可选


class DynamicFeatureExtractor:
    """动态特征提取器 - 9类特征"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def extract_statistical_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        1. 统计特征提取
        包括：均值、方差、偏度、峰度、变异系数
        """
        features = {}

        # 基础统计量
        features['mean'] = np.mean(window)
        features['std'] = np.std(window)
        features['variance'] = np.var(window)

        # 高阶统计量
        features['skewness'] = stats.skew(window)
        features['kurtosis'] = stats.kurtosis(window)

        # 变异系数（相对标准差）
        if features['mean'] != 0:
            features['cv'] = features['std'] / abs(features['mean'])
        else:
            features['cv'] = 0

        # 分位数特征
        features['q25'] = np.percentile(window, 25)
        features['q50'] = np.percentile(window, 50)  # 中位数
        features['q75'] = np.percentile(window, 75)
        features['iqr'] = features['q75'] - features['q25']  # 四分位距

        return features

    def extract_frequency_features(self, window: np.ndarray,
                                  sampling_rate: float = 0.1) -> Dict[str, float]:
        """
        2. 频域特征提取
        使用FFT分析频域特性
        """
        features = {}

        # 执行FFT
        fft_vals = rfft(window)
        fft_freqs = rfftfreq(len(window), d=1/sampling_rate)

        # 功率谱密度
        psd = np.abs(fft_vals) ** 2

        # 主频率
        dominant_freq_idx = np.argmax(psd[1:]) + 1  # 排除DC分量
        features['dominant_freq'] = fft_freqs[dominant_freq_idx]
        features['dominant_power'] = psd[dominant_freq_idx]

        # 频谱能量分布
        total_power = np.sum(psd)
        if total_power > 0:
            features['low_freq_energy'] = np.sum(psd[:len(psd)//3]) / total_power
            features['mid_freq_energy'] = np.sum(psd[len(psd)//3:2*len(psd)//3]) / total_power
            features['high_freq_energy'] = np.sum(psd[2*len(psd)//3:]) / total_power
        else:
            features['low_freq_energy'] = 0
            features['mid_freq_energy'] = 0
            features['high_freq_energy'] = 0

        # 频谱熵（衡量信号复杂度）
        if total_power > 0:
            psd_norm = psd / total_power
            psd_norm = psd_norm[psd_norm > 0]  # 移除零值
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        else:
            features['spectral_entropy'] = 0

        # 频谱质心（频率重心）
        if total_power > 0:
            features['spectral_centroid'] = np.sum(fft_freqs * psd) / total_power
        else:
            features['spectral_centroid'] = 0

        return features

    def extract_temporal_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        3. 时序特征提取
        趋势、周期性、自相关等
        """
        features = {}

        # 趋势特征（线性拟合）
        x = np.arange(len(window))
        if len(window) > 1:
            slope, intercept = np.polyfit(x, window, 1)
            features['trend_slope'] = slope
            features['trend_intercept'] = intercept

            # 去趋势后的残差标准差
            detrended = window - (slope * x + intercept)
            features['detrended_std'] = np.std(detrended)
        else:
            features['trend_slope'] = 0
            features['trend_intercept'] = window[0] if len(window) > 0 else 0
            features['detrended_std'] = 0

        # 自相关特征
        if len(window) > 10:
            # 计算lag=1的自相关
            acf_1 = np.corrcoef(window[:-1], window[1:])[0, 1]
            features['autocorr_lag1'] = acf_1 if not np.isnan(acf_1) else 0

            # 计算lag=10的自相关（周期性检测）
            acf_10 = np.corrcoef(window[:-10], window[10:])[0, 1]
            features['autocorr_lag10'] = acf_10 if not np.isnan(acf_10) else 0
        else:
            features['autocorr_lag1'] = 0
            features['autocorr_lag10'] = 0

        # 变化率特征
        if len(window) > 1:
            diff = np.diff(window)
            features['mean_abs_change'] = np.mean(np.abs(diff))
            features['max_abs_change'] = np.max(np.abs(diff))
        else:
            features['mean_abs_change'] = 0
            features['max_abs_change'] = 0

        # 复杂度特征（近似熵）
        features['complexity'] = self._approximate_entropy(window, 2, 0.2 * np.std(window))

        return features

    def extract_shape_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        4. 形态特征提取
        峰值、谷值、过零率等
        """
        features = {}

        # 峰值和谷值
        features['max_value'] = np.max(window)
        features['min_value'] = np.min(window)
        features['range'] = features['max_value'] - features['min_value']

        # 峰值位置
        features['argmax'] = np.argmax(window) / len(window)  # 归一化位置
        features['argmin'] = np.argmin(window) / len(window)

        # 过零率
        mean_centered = window - np.mean(window)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(window)

        # 峰值计数
        peaks, _ = signal.find_peaks(window, height=np.mean(window))
        features['n_peaks'] = len(peaks)

        return features

    def extract_energy_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        5. 能量特征提取
        信号能量、功率等
        """
        features = {}

        # 总能量
        features['total_energy'] = np.sum(window ** 2)

        # 平均功率
        features['mean_power'] = features['total_energy'] / len(window)

        # 瞬时功率变化
        power = window ** 2
        features['power_std'] = np.std(power)

        # 能量比率（前半段vs后半段）
        mid_point = len(window) // 2
        energy_first_half = np.sum(window[:mid_point] ** 2)
        energy_second_half = np.sum(window[mid_point:] ** 2)

        if energy_second_half > 0:
            features['energy_ratio'] = energy_first_half / energy_second_half
        else:
            features['energy_ratio'] = 1.0

        return features

    def extract_nonlinear_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        6. 非线性特征提取
        Hurst指数、Lyapunov指数等
        """
        features = {}

        # Hurst指数（长程相关性）
        features['hurst_exponent'] = self._calculate_hurst_exponent(window)

        # 样本熵
        features['sample_entropy'] = self._sample_entropy(window, 2, 0.2 * np.std(window))

        # 分形维数（盒计数法简化版）
        features['fractal_dimension'] = self._box_counting_dimension(window)

        return features

    def extract_wavelet_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        7. 小波特征提取
        多尺度分解特征
        """
        features = {}

        # 使用简单的小波分解（这里用简化的多尺度分析）
        # 实际应用中可以使用pywt库

        # 低频分量（移动平均）
        window_size = min(10, len(window) // 4)
        if window_size > 0:
            low_freq = pd.Series(window).rolling(window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
            high_freq = window - low_freq

            features['wavelet_low_energy'] = np.sum(low_freq ** 2)
            features['wavelet_high_energy'] = np.sum(high_freq ** 2)
            features['wavelet_energy_ratio'] = features['wavelet_low_energy'] / (features['wavelet_high_energy'] + 1e-10)
        else:
            features['wavelet_low_energy'] = 0
            features['wavelet_high_energy'] = 0
            features['wavelet_energy_ratio'] = 0

        return features

    def extract_correlation_features(self, window: np.ndarray,
                                    reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        8. 相关性特征提取
        与参考信号的相关性
        """
        features = {}

        if reference is not None and len(reference) == len(window):
            # 皮尔逊相关系数
            corr = np.corrcoef(window, reference)[0, 1]
            features['pearson_correlation'] = corr if not np.isnan(corr) else 0

            # 互相关最大值
            cross_corr = np.correlate(window, reference, mode='same')
            features['max_cross_correlation'] = np.max(np.abs(cross_corr))

            # 相位差（通过互相关峰值位置）
            lag = np.argmax(np.abs(cross_corr)) - len(window) // 2
            features['phase_lag'] = lag / len(window)
        else:
            features['pearson_correlation'] = 0
            features['max_cross_correlation'] = 0
            features['phase_lag'] = 0

        return features

    def extract_information_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        9. 信息论特征提取
        信息熵、互信息等
        """
        features = {}

        # 香农熵
        if len(window) > 0:
            # 离散化数据
            n_bins = min(10, len(np.unique(window)))
            if n_bins > 1:
                hist, _ = np.histogram(window, bins=n_bins)
                hist = hist[hist > 0]  # 移除零计数
                prob = hist / np.sum(hist)
                features['shannon_entropy'] = -np.sum(prob * np.log2(prob))
            else:
                features['shannon_entropy'] = 0
        else:
            features['shannon_entropy'] = 0

        # 排列熵（衡量时序复杂度）
        features['permutation_entropy'] = self._permutation_entropy(window, order=3)

        return features

    # 辅助方法
    def _approximate_entropy(self, U, m, r):
        """计算近似熵"""
        def _maxdist(x_i, x_j, m):
            return max([abs(ua - va) for ua, va in zip(x_i[0:m], x_j[0:m])])

        def _phi(m):
            patterns = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                for j in range(N - m + 1):
                    if i != j and _maxdist(U[i:i+m], U[j:j+m], m) <= r:
                        patterns[i] += 1
            return patterns

        N = len(U)
        if N < m + 1:
            return 0

        patterns_m = _phi(m)
        patterns_m1 = _phi(m + 1) if N >= m + 2 else np.zeros(N - m)

        phi_m = np.mean(np.log(patterns_m + 1))
        phi_m1 = np.mean(np.log(patterns_m1[:len(patterns_m)] + 1))

        return phi_m - phi_m1

    def _sample_entropy(self, U, m, r):
        """计算样本熵"""
        N = len(U)
        if N < m + 1:
            return 0

        def _maxdist(x_i, x_j, m):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        # 计算模板匹配
        B = 0
        A = 0

        for i in range(N - m):
            for j in range(i + 1, N - m):
                if _maxdist(U[i:i+m], U[j:j+m], m) <= r:
                    B += 1
                    if N > m and i < N - m - 1 and j < N - m - 1:
                        if abs(U[i+m] - U[j+m]) <= r:
                            A += 1

        if B == 0:
            return 0

        return -np.log(A / B) if A > 0 else -np.log(1 / B)

    def _calculate_hurst_exponent(self, ts):
        """计算Hurst指数"""
        lags = range(2, min(100, len(ts) // 2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        if len(tau) > 0 and all(t > 0 for t in tau):
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        return 0.5

    def _box_counting_dimension(self, data):
        """简化的盒计数分形维数"""
        # 这是一个简化实现
        scales = np.logspace(0, 1, num=10, base=2)
        counts = []

        for scale in scales:
            bins = int(len(data) / scale)
            if bins > 0:
                hist, _ = np.histogram(data, bins=bins)
                counts.append(np.sum(hist > 0))

        if len(counts) > 1:
            coeffs = np.polyfit(np.log(scales[:len(counts)]), np.log(counts), 1)
            return -coeffs[0]
        return 1.0

    def _permutation_entropy(self, time_series, order=3):
        """计算排列熵"""
        if len(time_series) < order:
            return 0

        # 创建排列
        permutations = []
        for i in range(len(time_series) - order + 1):
            permutations.append(tuple(np.argsort(time_series[i:i+order])))

        # 计算概率分布
        from collections import Counter
        counts = Counter(permutations)
        probabilities = np.array(list(counts.values())) / len(permutations)

        # 计算熵
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))


class StaticFeatureExtractor:
    """静态特征提取器 - 32维特征"""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def extract_device_parameters(self, metadata: Dict) -> np.ndarray:
        """
        设备参数特征（12维）
        包括：泵参数、压缩机参数、储罐参数等
        """
        features = np.zeros(12)

        # 示例设备参数
        features[0] = metadata.get('booster_pump_rated_power', 220.0) / 1000  # 归一化
        features[1] = metadata.get('booster_pump_efficiency', 0.85)
        features[2] = metadata.get('hp_pump_rated_power', 1200.0) / 1000
        features[3] = metadata.get('hp_pump_efficiency', 0.82)
        features[4] = metadata.get('tank_capacity_m3', 50000.0) / 100000
        features[5] = metadata.get('tank_pressure_max_kPa', 3500.0) / 5000
        features[6] = metadata.get('compressor_stages', 3) / 5
        features[7] = metadata.get('compressor_efficiency', 0.78)
        features[8] = metadata.get('orv_capacity_tph', 1000.0) / 2000
        features[9] = metadata.get('recondenser_efficiency', 0.90)
        features[10] = metadata.get('system_age_years', 5.0) / 20
        features[11] = metadata.get('maintenance_score', 0.95)

        return features

    def extract_operating_mode(self, mode_info: Dict) -> np.ndarray:
        """
        工况特征（8维）
        包括：loading/unloading状态、运行模式等
        """
        features = np.zeros(8)

        # One-hot编码主要运行模式
        mode = mode_info.get('operation_mode', 'normal')
        mode_mapping = {
            'loading': 0,
            'unloading': 1,
            'holding': 2,
            'normal': 3,
            'emergency': 4
        }

        if mode in mode_mapping:
            features[mode_mapping[mode]] = 1.0

        # 其他工况参数
        features[5] = mode_info.get('load_factor', 0.8)
        features[6] = mode_info.get('ambient_temp_C', 20.0) / 50  # 归一化
        features[7] = mode_info.get('sea_state', 2) / 5  # 海况等级

        return features

    def extract_temporal_encoding(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        时间编码特征（12维）
        包括：小时、星期、月份的循环编码
        """
        features = np.zeros(12)

        # 小时的循环编码（sin/cos）
        hour = timestamp.hour
        features[0] = np.sin(2 * np.pi * hour / 24)
        features[1] = np.cos(2 * np.pi * hour / 24)

        # 星期的循环编码
        day_of_week = timestamp.dayofweek
        features[2] = np.sin(2 * np.pi * day_of_week / 7)
        features[3] = np.cos(2 * np.pi * day_of_week / 7)

        # 月份的循环编码
        month = timestamp.month
        features[4] = np.sin(2 * np.pi * month / 12)
        features[5] = np.cos(2 * np.pi * month / 12)

        # 年内进度
        day_of_year = timestamp.dayofyear
        features[6] = np.sin(2 * np.pi * day_of_year / 365)
        features[7] = np.cos(2 * np.pi * day_of_year / 365)

        # 工作日/周末标志
        features[8] = 1.0 if timestamp.weekday() < 5 else 0.0

        # 季节编码（北半球）
        season = (timestamp.month % 12 + 3) // 3
        features[9] = season / 4

        # 班次编码（假设三班倒）
        if 6 <= hour < 14:
            features[10] = 1.0  # 早班
        elif 14 <= hour < 22:
            features[11] = 1.0  # 中班
        # 夜班默认为0

        return features

    def extract_all_static_features(self,
                                   timestamp: pd.Timestamp,
                                   metadata: Dict = None,
                                   mode_info: Dict = None) -> np.ndarray:
        """
        提取所有静态特征（32维）
        """
        # 使用默认值如果未提供
        if metadata is None:
            metadata = {}
        if mode_info is None:
            mode_info = {}

        # 提取各类静态特征
        device_features = self.extract_device_parameters(metadata)
        mode_features = self.extract_operating_mode(mode_info)
        temporal_features = self.extract_temporal_encoding(timestamp)

        # 合并为32维特征向量
        static_features = np.concatenate([
            device_features,  # 12维
            mode_features,    # 8维
            temporal_features # 12维
        ])

        assert len(static_features) == 32, f"静态特征维度错误: {len(static_features)} != 32"

        return static_features


class AdvancedFeatureEngineering:
    """
    完整的特征工程管线
    整合9类动态特征和32维静态特征
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.dynamic_extractor = DynamicFeatureExtractor(config)
        self.static_extractor = StaticFeatureExtractor(config)

        logger.info(f"初始化高级特征工程模块")
        logger.info(f"  动态特征: 9类")
        logger.info(f"  静态特征: 32维")

    def extract_features_from_window(self,
                                    window_data: pd.DataFrame,
                                    timestamp: pd.Timestamp,
                                    metadata: Dict = None,
                                    mode_info: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从时间窗口提取完整特征集

        Returns:
            dynamic_features: [n_features, window_size] 动态特征
            static_features: [32] 静态特征
        """
        dynamic_features_list = []

        # 对每个变量提取9类动态特征
        for col in window_data.columns:
            if col == 'ts':
                continue

            values = window_data[col].values

            # 1. 统计特征
            stat_features = self.dynamic_extractor.extract_statistical_features(values)

            # 2. 频域特征
            freq_features = self.dynamic_extractor.extract_frequency_features(values)

            # 3. 时序特征
            temporal_features = self.dynamic_extractor.extract_temporal_features(values)

            # 4. 形态特征
            shape_features = self.dynamic_extractor.extract_shape_features(values)

            # 5. 能量特征
            energy_features = self.dynamic_extractor.extract_energy_features(values)

            # 6. 非线性特征
            nonlinear_features = self.dynamic_extractor.extract_nonlinear_features(values)

            # 7. 小波特征
            wavelet_features = self.dynamic_extractor.extract_wavelet_features(values)

            # 8. 相关性特征（如果有参考信号）
            corr_features = self.dynamic_extractor.extract_correlation_features(values)

            # 9. 信息论特征
            info_features = self.dynamic_extractor.extract_information_features(values)

            # 合并所有特征
            all_features = {
                **stat_features,
                **freq_features,
                **temporal_features,
                **shape_features,
                **energy_features,
                **nonlinear_features,
                **wavelet_features,
                **corr_features,
                **info_features
            }

            dynamic_features_list.append(list(all_features.values()))

        # 转换为numpy数组
        dynamic_features = np.array(dynamic_features_list)

        # 提取静态特征
        static_features = self.static_extractor.extract_all_static_features(
            timestamp, metadata, mode_info
        )

        return dynamic_features, static_features

    def process_dataset(self,
                       df: pd.DataFrame,
                       metadata: Dict = None,
                       mode_info: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理整个数据集

        Returns:
            X_dynamic: [n_windows, n_features, window_size]
            X_static: [n_windows, 32]
        """
        window_size = self.config.window_size
        stride = self.config.stride

        n_windows = (len(df) - window_size) // stride + 1

        logger.info(f"处理数据集: {len(df)} 行 -> {n_windows} 个窗口")

        X_dynamic_list = []
        X_static_list = []

        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            window = df.iloc[start_idx:end_idx]
            timestamp = window.iloc[-1]['ts'] if 'ts' in window.columns else pd.Timestamp.now()

            dynamic_feat, static_feat = self.extract_features_from_window(
                window, timestamp, metadata, mode_info
            )

            X_dynamic_list.append(dynamic_feat)
            X_static_list.append(static_feat)

            if (i + 1) % 100 == 0:
                logger.info(f"  处理进度: {i+1}/{n_windows} 窗口")

        X_dynamic = np.array(X_dynamic_list)
        X_static = np.array(X_static_list)

        logger.info(f"特征提取完成:")
        logger.info(f"  动态特征形状: {X_dynamic.shape}")
        logger.info(f"  静态特征形状: {X_static.shape}")

        return X_dynamic, X_static


def test_advanced_features():
    """测试高级特征工程"""
    print("="*60)
    print("测试高级特征工程模块")
    print("="*60)

    # 创建模拟数据
    n_samples = 1000
    n_features = 19  # LNG系统的19个监测变量

    # 生成模拟时序数据
    np.random.seed(42)
    data = {}
    data['ts'] = pd.date_range('2024-01-01', periods=n_samples, freq='10S')

    for i in range(n_features):
        # 生成具有不同特性的信号
        t = np.arange(n_samples)
        signal_base = np.sin(2 * np.pi * 0.01 * t) + 0.5 * np.sin(2 * np.pi * 0.05 * t)
        noise = np.random.normal(0, 0.1, n_samples)
        data[f'feature_{i}'] = signal_base + noise

    df = pd.DataFrame(data)

    print(f"测试数据: {len(df)} 行, {len(df.columns)} 列")

    # 初始化特征工程
    config = FeatureConfig(window_size=180, stride=30)
    feature_engine = AdvancedFeatureEngineering(config)

    # 测试单个窗口
    window = df.iloc[:180]
    timestamp = window.iloc[-1]['ts']

    print("\n测试单窗口特征提取...")
    dynamic_feat, static_feat = feature_engine.extract_features_from_window(
        window, timestamp
    )

    print(f"  动态特征形状: {dynamic_feat.shape}")
    print(f"  静态特征形状: {static_feat.shape}")
    print(f"  静态特征示例: {static_feat[:5]}...")

    # 测试批处理
    print("\n测试批处理...")
    X_dynamic, X_static = feature_engine.process_dataset(df[:500])

    print(f"  批处理结果:")
    print(f"    动态特征: {X_dynamic.shape}")
    print(f"    静态特征: {X_static.shape}")

    print("\n✅ 高级特征工程测试通过！")
    print("="*60)

    # 特征统计
    print("\n特征体系总结:")
    print("  动态特征（9类）:")
    print("    1. 统计特征: 均值、方差、偏度、峰度等")
    print("    2. 频域特征: 主频、功率谱、频谱熵等")
    print("    3. 时序特征: 趋势、自相关、变化率等")
    print("    4. 形态特征: 峰值、过零率、峰值计数等")
    print("    5. 能量特征: 总能量、功率、能量比等")
    print("    6. 非线性特征: Hurst指数、样本熵、分形维数等")
    print("    7. 小波特征: 多尺度能量分布")
    print("    8. 相关性特征: 互相关、相位差等")
    print("    9. 信息论特征: 香农熵、排列熵等")

    print("\n  静态特征（32维）:")
    print("    - 设备参数: 12维")
    print("    - 工况特征: 8维")
    print("    - 时间编码: 12维")

    return feature_engine


if __name__ == "__main__":
    # 运行测试
    test_advanced_features()