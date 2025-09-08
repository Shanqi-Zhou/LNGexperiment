"""
特征工程模块 - V_t/V_s生成
基于技术路线固定特征集 + 增量计算优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from scipy import signal
from PyEMD import EMD
import pywt
from .incremental_calculator import IncrementalFeatureExtractor


class DynamicFeatureExtractor:
    """动态特征提取器 V_t"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.emd = EMD()
    
    def extract_features(self, window_data: np.ndarray) -> np.ndarray:
        """
        对窗口内每条通道提取9个统计特征
        Args:
            window_data: shape (window_size, n_channels)
        Returns:
            features: shape (9 * n_channels,)
        """
        window_size, n_channels = window_data.shape
        features = []
        
        for ch in range(n_channels):
            channel_data = window_data[:, ch]
            
            # 基础统计特征
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            skewness = self._compute_skewness(channel_data)
            kurtosis = self._compute_kurtosis(channel_data)
            rms = np.sqrt(np.mean(channel_data**2))
            
            # 变化率特征
            diff_data = np.diff(channel_data)
            mean_change_rate = np.mean(np.abs(diff_data))
            std_change_rate = np.std(diff_data)
            
            # EMD频域特征
            try:
                imf1_freq, imf1_energy_ratio = self._extract_emd_features(channel_data)
            except:
                imf1_freq = 0.0
                imf1_energy_ratio = 0.0
            
            channel_features = [
                mean_val, std_val, skewness, kurtosis, rms,
                mean_change_rate, std_change_rate, imf1_freq, imf1_energy_ratio
            ]
            
            features.extend(channel_features)
        
        return np.array(features)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _extract_emd_features(self, data: np.ndarray) -> Tuple[float, float]:
        """提取EMD特征：IMF1主频和能量比"""
        try:
            # EMD分解
            imfs = self.emd(data)
            if len(imfs) < 2:
                return 0.0, 0.0
            
            imf1 = imfs[0]
            
            # IMF1主频（通过峰值检测）
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
            
            return imf1_freq, imf1_energy_ratio
            
        except Exception as e:
            self.logger.warning(f"EMD特征提取失败: {e}")
            return 0.0, 0.0


class StaticFeatureExtractor:
    """静态/上下文特征提取器 V_s"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 设备额定功率（基于技术路线）
        self.equipment_ratings = {
            'pumps_booster_power_kw': 220,
            'pumps_hp_power_kw': 1200,
            'bog_compressor_total_power_kw': 1000,  # 估算值
        }
    
    def extract_features(self, data: pd.DataFrame, window_indices: np.ndarray) -> np.ndarray:
        """
        提取静态特征 32维
        Args:
            data: 完整数据集
            window_indices: 当前窗口的索引范围
        Returns:
            static_features: shape (32,)
        """
        window_data = data.iloc[window_indices]
        
        features = []
        
        # 1. 设备额定功率编码 (3维)
        for equipment, rating in self.equipment_ratings.items():
            if equipment in data.columns:
                # 归一化的额定功率
                features.append(rating / 1000.0)  # 转换为MW
            else:
                features.append(0.0)
        
        # 2. 工况模式编码 (3维: 卸载/外输/停运)
        operating_mode = self._detect_operating_mode(window_data)
        mode_encoding = [0.0, 0.0, 0.0]
        mode_encoding[operating_mode] = 1.0
        features.extend(mode_encoding)
        
        # 3. 时间特征 (4维: 小时sin/cos, 日期sin/cos)
        if 'timestamp' in window_data.columns:
            timestamps = pd.to_datetime(window_data['timestamp'])
            hour_of_day = timestamps.dt.hour.mean()
            day_of_year = timestamps.dt.dayofyear.mean()
            
            features.extend([
                np.sin(2 * np.pi * hour_of_day / 24),
                np.cos(2 * np.pi * hour_of_day / 24),
                np.sin(2 * np.pi * day_of_year / 365),
                np.cos(2 * np.pi * day_of_year / 365)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 4. 设定点统计 (22维)
        setpoint_features = self._extract_setpoint_statistics(window_data)
        features.extend(setpoint_features)
        
        # 确保特征维度为32
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32])
    
    def _detect_operating_mode(self, window_data: pd.DataFrame) -> int:
        """检测运行模式: 0=卸载, 1=外输, 2=停运"""
        # 基于流量判断运行模式
        if 'export_meter' in window_data.columns:
            export_flow = window_data['export_meter'].mean()
        elif 'q_Nm3h' in window_data.columns:
            export_flow = window_data['q_Nm3h'].mean()
        else:
            export_flow = 0
        
        if 'pumps_booster_flow_m3h' in window_data.columns:
            booster_flow = window_data['pumps_booster_flow_m3h'].mean()
        else:
            booster_flow = 0
        
        # 判断逻辑
        if export_flow > 100:  # 有外输
            return 1  # 外输模式
        elif booster_flow > 50:  # 有内部流动但无外输
            return 0  # 卸载模式
        else:
            return 2  # 停运模式
    
    def _extract_setpoint_statistics(self, window_data: pd.DataFrame) -> List[float]:
        """提取设定点统计特征"""
        features = []
        
        # 关键变量的均值和标准差
        key_variables = [
            'tank_p_top_kPa',      # 罐顶压力
            'orv_T_out_C',         # ORV出口温度  
            'T_amb_C',             # 环境温度
            'seawater_T_C',        # 海水温度
            'export_meter',        # 外输流量（如果有的话）
        ]
        
        # 为每个关键变量计算均值和标准差 (5*2=10维)
        for var in key_variables:
            if var in window_data.columns:
                values = window_data[var].dropna()
                if len(values) > 0:
                    features.extend([values.mean(), values.std()])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        
        # 阀位统计（模拟，实际需要根据数据调整）
        # 泵效率统计 (2*2=4维)
        efficiency_vars = ['pumps_booster_efficiency', 'pumps_hp_efficiency']
        for var in efficiency_vars:
            if var in window_data.columns:
                values = window_data[var].dropna()
                if len(values) > 0:
                    features.extend([values.mean(), values.std()])
                else:
                    features.extend([0.5, 0.1])  # 默认效率
            else:
                features.extend([0.5, 0.1])
        
        # 功率利用率统计 (4*2=8维)
        power_vars = [
            ('pumps_booster_power_kw', 220),
            ('pumps_hp_power_kw', 1200),
            ('bog_compressor_total_power_kw', 1000),
            ('orv_Q_MW', 100)  # ORV热负荷，转换为类似功率的概念
        ]
        
        for var, rating in power_vars:
            if var in window_data.columns:
                values = window_data[var].dropna()
                if len(values) > 0:
                    utilization = values.mean() / rating
                    utilization_std = values.std() / rating
                    features.extend([utilization, utilization_std])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        
        return features


class LNGFeatureEngine:
    """LNG特征工程主引擎 - 整合增量计算优化"""
    
    def __init__(self, window_size: int = 180, stride: int = 30, 
                 enable_incremental: bool = True, cache_size_mb: int = 512,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            window_size: 窗口大小（时间步数），默认180（30分钟×10s采样）
            stride: 滑动步长，默认30（5分钟）
            enable_incremental: 是否启用增量计算
            cache_size_mb: 特征缓存大小(MB)
        """
        self.window_size = window_size
        self.stride = stride
        self.enable_incremental = enable_incremental
        self.logger = logger or logging.getLogger(__name__)
        
        self.dynamic_extractor = DynamicFeatureExtractor(logger)
        self.static_extractor = StaticFeatureExtractor(logger)
        
        # 增量计算器（可选）
        if enable_incremental:
            self.incremental_extractor = IncrementalFeatureExtractor(
                window_size=window_size,
                cache_size_mb=cache_size_mb,
                logger=logger
            )
            self.logger.info("启用增量特征计算优化")
        else:
            self.incremental_extractor = None
    
    def create_windowed_features(self, data: pd.DataFrame, 
                                target_column: str, 
                                use_incremental: bool = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建窗口化特征 - 自动选择最优计算模式
        Args:
            use_incremental: 是否使用增量模式（None=自动选择）
        Returns:
            dynamic_features: (n_windows, n_dynamic_features)
            static_features: (n_windows, 32)
            targets: (n_windows,)
        """
        self.logger.info(f"创建窗口化特征，窗口大小={self.window_size}, 步长={self.stride}")
        
        # 自动选择计算模式
        if use_incremental is None:
            use_incremental = self.enable_incremental and len(data) > 5000
            
        if use_incremental and self.incremental_extractor is not None:
            return self._create_features_incremental(data, target_column)
        else:
            return self._create_features_standard(data, target_column)
    
    def _create_features_incremental(self, data: pd.DataFrame, 
                                   target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """增量模式特征创建"""
        self.logger.info("使用增量计算模式")
        
        # 选择数值列作为动态特征输入
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        n_samples = len(data)
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        # 使用增量提取器处理批次数据
        incremental_features = self.incremental_extractor.extract_features_incremental(
            data[feature_cols], is_streaming=False
        )
        
        # 转换增量特征格式
        dynamic_features_list = []
        static_features_list = []
        targets_list = []
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            if end_idx > n_samples:
                break
                
            # 从增量结果获取动态特征
            window_key = f'window_{i}'
            if window_key in incremental_features:
                dynamic_feat = incremental_features[window_key]
            else:
                # 回退到标准计算
                window_indices = np.arange(start_idx, end_idx)
                window_data = data[feature_cols].iloc[window_indices].values
                dynamic_feat = self.dynamic_extractor.extract_features(window_data)
                
            dynamic_features_list.append(dynamic_feat)
            
            # 静态特征（仍使用标准方法）
            window_indices = np.arange(start_idx, end_idx)
            static_feat = self.static_extractor.extract_features(data, window_indices)
            static_features_list.append(static_feat)
            
            # 目标值
            target_val = data[target_column].iloc[end_idx - 1]
            targets_list.append(target_val)
        
        # 记录增量计算性能
        performance_report = self.incremental_extractor.get_performance_report()
        cache_hit_rate = performance_report['feature_cache']['hit_rate']
        self.logger.info(f"增量计算完成，缓存命中率: {cache_hit_rate:.2%}")
        
        dynamic_features = np.array(dynamic_features_list)
        static_features = np.array(static_features_list)
        targets = np.array(targets_list)
        
        self.logger.info(f"增量特征创建完成:")
        self.logger.info(f"  动态特征: {dynamic_features.shape}")
        self.logger.info(f"  静态特征: {static_features.shape}")  
        self.logger.info(f"  目标变量: {targets.shape}")
        
        return dynamic_features, static_features, targets
    
    def _create_features_standard(self, data: pd.DataFrame, 
                                target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """标准模式特征创建（原方法）"""
        self.logger.info("使用标准计算模式")
        
        # 选择数值列作为动态特征输入
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        n_samples = len(data)
        n_windows = (n_samples - self.window_size) // self.stride + 1
        
        dynamic_features_list = []
        static_features_list = []
        targets_list = []
        
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            
            # 检查边界
            if end_idx > n_samples:
                break
            
            window_indices = np.arange(start_idx, end_idx)
            window_data = data[feature_cols].iloc[window_indices].values
            
            # 提取动态特征
            dynamic_feat = self.dynamic_extractor.extract_features(window_data)
            dynamic_features_list.append(dynamic_feat)
            
            # 提取静态特征
            static_feat = self.static_extractor.extract_features(data, window_indices)
            static_features_list.append(static_feat)
            
            # 目标值（窗口末尾的能耗）
            target_val = data[target_column].iloc[end_idx - 1]
            targets_list.append(target_val)
        
        dynamic_features = np.array(dynamic_features_list)
        static_features = np.array(static_features_list)
        targets = np.array(targets_list)
        
        self.logger.info(f"标准特征创建完成:")
        self.logger.info(f"  动态特征: {dynamic_features.shape}")
        self.logger.info(f"  静态特征: {static_features.shape}")  
        self.logger.info(f"  目标变量: {targets.shape}")
        
        return dynamic_features, static_features, targets
    
    def get_performance_stats(self) -> Optional[Dict]:
        """获取性能统计（仅增量模式）"""
        if self.incremental_extractor is not None:
            return self.incremental_extractor.get_performance_report()
        return None
    
    def clear_caches(self):
        """清理缓存（仅增量模式）"""
        if self.incremental_extractor is not None:
            self.incremental_extractor.clear_caches()
            self.logger.info("特征工程缓存已清理")


def create_lng_features(data: pd.DataFrame, target_column: str,
                       window_size: int = 180, stride: int = 30,
                       enable_incremental: bool = True, cache_size_mb: int = 512,
                       logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便捷函数：创建LNG特征 - 支持增量计算"""
    engine = LNGFeatureEngine(window_size, stride, enable_incremental, cache_size_mb, logger)
    return engine.create_windowed_features(data, target_column)