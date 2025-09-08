"""
高斯过程回归 (GPR) - 残差修正
基于技术路线的标准实现
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Union
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class GaussianProcessRegression:
    """高斯过程回归模型 - 用于MLR残差修正"""
    
    def __init__(self, 
                 kernel_type: str = 'rbf',
                 length_scale: float = 1.0,
                 length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
                 nu: float = 1.5,  # Matern核参数
                 alpha: float = 1e-10,
                 n_restarts_optimizer: int = 5,
                 normalize_y: bool = True,
                 random_state: int = 42,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            kernel_type: 核函数类型 ('rbf', 'matern', 'rbf_white')
            length_scale: 长度尺度参数
            length_scale_bounds: 长度尺度边界
            nu: Matern核的光滑性参数
            alpha: 对角线正则化项
            n_restarts_optimizer: 优化器重启次数
            normalize_y: 是否标准化目标值
            random_state: 随机种子
        """
        self.kernel_type = kernel_type
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.nu = nu
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化核函数
        self.kernel = self._create_kernel()
        
        # 初始化GP模型
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state
        )
        
        # 特征缩放器
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.training_score = None
        
    def _create_kernel(self):
        """创建核函数"""
        if self.kernel_type == 'rbf':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=self.length_scale, 
                length_scale_bounds=self.length_scale_bounds
            )
        elif self.kernel_type == 'matern':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds, 
                nu=self.nu
            )
        elif self.kernel_type == 'rbf_white':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel_type}")
            
        return kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegression':
        """训练GPR模型"""
        self.logger.info(f"训练GPR模型，核函数: {self.kernel_type}, 输入维度: {X.shape}")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练GP模型
        self.gp.fit(X_scaled, y)
        
        # 记录训练分数
        self.training_score = self.gp.score(X_scaled, y)
        
        self.is_fitted = True
        self.logger.info(f"GPR训练完成，训练R²: {self.training_score:.4f}")
        self.logger.info(f"优化后核函数: {self.gp.kernel_}")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        X_scaled = self.scaler.transform(X)
        
        if return_std:
            mean, std = self.gp.predict(X_scaled, return_std=True)
            return mean, std
        else:
            return self.gp.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """预测并返回不确定性区间"""
        mean, std = self.predict(X, return_std=True)
        
        # 计算置信区间
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        lower_bound = mean - z_score * std
        upper_bound = mean + z_score * std
        
        return mean, lower_bound, upper_bound
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def log_marginal_likelihood(self) -> float:
        """返回对数边际似然"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.gp.log_marginal_likelihood()
    
    def sample_y(self, X: np.ndarray, n_samples: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """从后验分布中采样"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        X_scaled = self.scaler.transform(X)
        return self.gp.sample_y(X_scaled, n_samples=n_samples, random_state=random_state)
    
    def get_kernel_parameters(self) -> Dict[str, float]:
        """获取核函数参数"""
        if not self.is_fitted:
            return {}
        
        kernel_params = {}
        kernel_str = str(self.gp.kernel_)
        
        # 解析核函数参数
        if 'RBF' in kernel_str or 'Matern' in kernel_str:
            kernel_params['length_scale'] = self.gp.kernel_.k2.length_scale if hasattr(self.gp.kernel_, 'k2') else self.gp.kernel_.length_scale
        
        if 'ConstantKernel' in kernel_str:
            kernel_params['constant_value'] = self.gp.kernel_.k1.constant_value if hasattr(self.gp.kernel_, 'k1') else getattr(self.gp.kernel_, 'constant_value', 1.0)
        
        if 'WhiteKernel' in kernel_str:
            if hasattr(self.gp.kernel_, 'k3'):
                kernel_params['noise_level'] = self.gp.kernel_.k3.noise_level
            elif hasattr(self.gp.kernel_, 'noise_level'):
                kernel_params['noise_level'] = self.gp.kernel_.noise_level
        
        return kernel_params
    
    def summary(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """模型摘要信息"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        predictions = self.predict(X)
        residuals = y - predictions
        
        summary_info = {
            'model_type': f'GPR_{self.kernel_type}',
            'kernel_function': str(self.gp.kernel_),
            'n_features': X.shape[1],
            'log_marginal_likelihood': self.log_marginal_likelihood(),
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'kernel_parameters': self.get_kernel_parameters()
        }
        
        return summary_info


class ResidualGPR:
    """残差修正用的GPR包装器"""
    
    def __init__(self, 
                 base_model,
                 gpr_kwargs: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            base_model: 基础模型 (如MLR)
            gpr_kwargs: GPR模型参数
        """
        self.base_model = base_model
        self.gpr_kwargs = gpr_kwargs or {}
        self.logger = logger or logging.getLogger(__name__)
        
        self.gpr = GaussianProcessRegression(logger=logger, **self.gpr_kwargs)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ResidualGPR':
        """训练基础模型和残差GPR"""
        self.logger.info("训练残差修正GPR模型...")
        
        # 训练基础模型
        self.base_model.fit(X, y)
        
        # 计算基础模型残差
        base_predictions = self.base_model.predict(X)
        residuals = y - base_predictions
        
        # 用残差训练GPR
        self.gpr.fit(X, residuals)
        
        self.is_fitted = True
        self.logger.info("残差修正GPR训练完成")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测（基础预测 + 残差修正）"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        # 基础模型预测
        base_predictions = self.base_model.predict(X)
        
        # GPR残差预测
        if return_std:
            residual_mean, residual_std = self.gpr.predict(X, return_std=True)
            final_predictions = base_predictions + residual_mean
            return final_predictions, residual_std
        else:
            residual_predictions = self.gpr.predict(X)
            final_predictions = base_predictions + residual_predictions
            return final_predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def summary(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """模型摘要"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        base_predictions = self.base_model.predict(X)
        final_predictions = self.predict(X)
        
        base_r2 = r2_score(y, base_predictions)
        final_r2 = r2_score(y, final_predictions)
        
        summary_info = {
            'model_type': 'MLR_GPR_Hybrid',
            'base_model_r2': base_r2,
            'final_r2': final_r2,
            'improvement': final_r2 - base_r2,
            'base_model_summary': self.base_model.summary(X, y) if hasattr(self.base_model, 'summary') else {},
            'gpr_summary': self.gpr.summary(X, y - base_predictions)
        }
        
        return summary_info