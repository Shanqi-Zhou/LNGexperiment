"""
多元线性回归模型 (MLR)
基于技术路线的标准实现
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MultipleLinearRegression:
    """多元线性回归模型"""
    
    def __init__(self, 
                 regularization: str = 'ridge',
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 normalize_features: bool = True,
                 polynomial_degree: int = 1,
                 interaction_only: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            regularization: 正则化类型 ('ridge', 'none')
            alpha: 正则化强度
            fit_intercept: 是否拟合截距
            normalize_features: 是否标准化特征
            polynomial_degree: 多项式特征阶数
            interaction_only: 是否只包含交互项
        """
        self.regularization = regularization
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化模型
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize_features)
        else:
            self.model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize_features)
        
        # 多项式特征生成器
        if polynomial_degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=polynomial_degree, 
                interaction_only=interaction_only,
                include_bias=False
            )
        else:
            self.poly_features = None
            
        self.feature_names = None
        self.is_fitted = False
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultipleLinearRegression':
        """训练MLR模型"""
        self.logger.info(f"训练MLR模型，输入特征维度: {X.shape}")
        
        # 特征工程
        X_processed = self._prepare_features(X, fit=True)
        
        # 训练模型
        self.model.fit(X_processed, y)
        
        # 计算特征重要性
        self._compute_feature_importance(X_processed, y)
        
        self.is_fitted = True
        self.logger.info(f"MLR训练完成，处理后特征维度: {X_processed.shape}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()方法")
        
        X_processed = self._prepare_features(X, fit=False)
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def _prepare_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """特征预处理"""
        X_processed = X.copy()
        
        # 多项式特征
        if self.poly_features is not None:
            if fit:
                X_processed = self.poly_features.fit_transform(X_processed)
                # 保存特征名称
                if hasattr(self.poly_features, 'get_feature_names_out'):
                    try:
                        self.feature_names = self.poly_features.get_feature_names_out()
                    except:
                        self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
                else:
                    self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
            else:
                X_processed = self.poly_features.transform(X_processed)
        else:
            if fit:
                self.feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
        
        return X_processed
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
        """计算特征重要性"""
        try:
            # 获取回归系数
            coefficients = np.abs(self.model.coef_)
            
            # 标准化系数作为特征重要性
            if len(coefficients) > 0:
                self.feature_importance = coefficients / np.sum(coefficients)
            else:
                self.feature_importance = np.array([])
                
        except Exception as e:
            self.logger.warning(f"特征重要性计算失败: {e}")
            self.feature_importance = np.zeros(X.shape[1])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性字典"""
        if not self.is_fitted or self.feature_importance is None:
            return {}
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        else:
            feature_names = self.feature_names
        
        return dict(zip(feature_names, self.feature_importance))
    
    def get_coefficients(self) -> Dict[str, float]:
        """获取回归系数"""
        if not self.is_fitted:
            return {}
        
        coefficients = self.model.coef_
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(coefficients))]
        else:
            feature_names = self.feature_names
            
        coef_dict = dict(zip(feature_names, coefficients))
        
        if self.fit_intercept:
            coef_dict['intercept'] = self.model.intercept_
            
        return coef_dict
    
    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算残差"""
        predictions = self.predict(X)
        return y - predictions
    
    def summary(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """模型摘要信息"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        predictions = self.predict(X)
        residuals = y - predictions
        
        summary_info = {
            'model_type': f'MLR_{self.regularization}',
            'n_features': X.shape[1],
            'polynomial_degree': self.polynomial_degree,
            'regularization_alpha': self.alpha,
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'feature_importance': self.get_feature_importance(),
            'coefficients': self.get_coefficients()
        }
        
        return summary_info


class PolynomialMLR(MultipleLinearRegression):
    """多项式多元线性回归"""
    
    def __init__(self, degree: int = 2, interaction_only: bool = False, **kwargs):
        """
        Args:
            degree: 多项式阶数
            interaction_only: 是否只包含交互项
        """
        super().__init__(polynomial_degree=degree, interaction_only=interaction_only, **kwargs)


class RidgeMLR(MultipleLinearRegression):
    """Ridge多元线性回归"""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Args:
            alpha: Ridge正则化强度
        """
        super().__init__(regularization='ridge', alpha=alpha, **kwargs)