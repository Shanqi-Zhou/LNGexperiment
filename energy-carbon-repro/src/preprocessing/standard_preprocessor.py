"""
预处理模块 - 对齐/插值/去噪/窗口
基于技术路线固定流程
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging
from scipy import signal
from PyEMD import EMD
import pywt


class LNGPreprocessor:
    """LNG数据预处理器 - 技术路线标准流程"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fitted = False
        self.normalization_params = {}
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(df).transform(df)
    
    def fit(self, df: pd.DataFrame) -> 'LNGPreprocessor':
        """拟合预处理参数"""
        self.logger.info("拟合预处理参数...")
        
        # 计算归一化参数（仅基于训练数据）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                values = df[col].dropna()
                self.normalization_params[col] = {
                    'mean': values.mean(),
                    'std': values.std()
                }
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用预处理流程"""
        if not self.fitted:
            raise ValueError("必须先调用 fit() 方法")
        
        self.logger.info("执行预处理流程...")
        processed_df = df.copy()
        
        # 1. 降采样（1s → 10s）
        processed_df = self._downsample(processed_df)
        
        # 2. 缺失值修复
        processed_df = self._repair_missing_values(processed_df)
        
        # 3. 离群值处理（Hampel滤波）
        processed_df = self._hampel_filter(processed_df)
        
        # 4. 多尺度去噪
        processed_df = self._multiscale_denoising(processed_df)
        
        # 5. 归一化（Z-score）
        processed_df = self._normalize(processed_df)
        
        self.logger.info(f"预处理完成: {df.shape} -> {processed_df.shape}")
        return processed_df
    
    def _downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        """降采样：1s → 10s"""
        if 'timestamp' not in df.columns:
            self.logger.warning("缺少timestamp列，跳过降采样")
            return df
        
        self.logger.info("执行降采样 1s -> 10s...")
        
        # 设置timestamp为索引
        df_resampled = df.set_index('timestamp')
        
        # 聚合到10s边界
        df_resampled = df_resampled.resample('10S').agg({
            col: 'mean' for col in df_resampled.select_dtypes(include=[np.number]).columns
        })
        
        # 重置索引
        df_resampled = df_resampled.reset_index()
        
        self.logger.info(f"降采样: {len(df)} -> {len(df_resampled)} 样本")
        return df_resampled
    
    def _repair_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值修复：≤60s线性插值，>60s剔除"""
        self.logger.info("修复缺失值...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 线性插值（限制最大间隔为6个点，即60s）
            df[col] = df[col].interpolate(method='linear', limit=6)
        
        # 移除仍有大量缺失值的行（>60s窗口）
        initial_len = len(df)
        df = df.dropna(thresh=len(df.columns) * 0.8)  # 保留至少80%非空的行
        
        if len(df) < initial_len:
            self.logger.info(f"移除缺失值过多的行: {initial_len} -> {len(df)}")
        
        return df
    
    def _hampel_filter(self, df: pd.DataFrame, window_size: int = 11, n_sigma: float = 3.0) -> pd.DataFrame:
        """增强Hampel滤波处理离群值"""
        from .enhanced_hampel import EnhancedHampelFilter
        
        self.logger.info("执行增强Hampel离群值滤波...")
        
        # 使用增强版Hampel滤波器
        enhanced_filter = EnhancedHampelFilter(
            window_sizes=[7, window_size, window_size + 4],
            n_sigma=n_sigma,
            regime_aware=True,
            parallel=True,
            logger=self.logger
        )
        
        filtered_df, outlier_stats = enhanced_filter.fit_transform(df)
        
        # 记录滤波统计信息
        self.logger.info(f"Hampel滤波统计: {outlier_stats['outlier_rate_percent']:.2f}% 异常值被处理")
        
        return filtered_df
    
    def _multiscale_denoising(self, df: pd.DataFrame) -> pd.DataFrame:
        """多尺度去噪：EMD + 小波软阈值"""
        self.logger.info("执行多尺度去噪...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        emd = EMD()
        
        for col in numeric_cols:
            try:
                data = df[col].values
                
                # EMD分解
                imfs = emd(data)
                
                if len(imfs) >= 2:
                    # 对IMF1和IMF2应用小波软阈值去噪
                    denoised_imfs = []
                    
                    for i, imf in enumerate(imfs):
                        if i < 2:  # IMF1, IMF2
                            # Daubechies-4小波软阈值
                            coeffs = pywt.wavedec(imf, 'db4', level=3)
                            threshold = np.std(imf) * 0.1  # 软阈值
                            coeffs_soft = [pywt.threshold(c, threshold, 'soft') for c in coeffs]
                            denoised_imf = pywt.waverec(coeffs_soft, 'db4')
                            
                            # 确保长度一致
                            if len(denoised_imf) > len(imf):
                                denoised_imf = denoised_imf[:len(imf)]
                            elif len(denoised_imf) < len(imf):
                                denoised_imf = np.pad(denoised_imf, (0, len(imf) - len(denoised_imf)))
                                
                            denoised_imfs.append(denoised_imf)
                        else:
                            denoised_imfs.append(imf)
                    
                    # 重构信号
                    denoised_signal = np.sum(denoised_imfs, axis=0)
                    df[col] = denoised_signal
                    
            except Exception as e:
                self.logger.warning(f"列 {col} 去噪失败: {e}")
                continue
        
        return df
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score归一化"""
        self.logger.info("执行Z-score归一化...")
        
        for col, params in self.normalization_params.items():
            if col in df.columns:
                mean = params['mean']
                std = params['std']
                
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = df[col] - mean  # 标准差为0时只减去均值
        
        return df
    
    def inverse_transform_target(self, target_name: str, scaled_values: np.ndarray) -> np.ndarray:
        """反归一化目标变量"""
        if target_name in self.normalization_params:
            params = self.normalization_params[target_name]
            return scaled_values * params['std'] + params['mean']
        return scaled_values