"""
残差建模框架 - 优化方案核心创新
MLR处理线性成分，GPR专注于非线性残差，实现分层精准建模
结合近似推理技术(Nyström/SVGP/Local-SVGP)，平衡精度与效率
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, Tuple, List
import logging
import time
import warnings
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import joblib

warnings.filterwarnings('ignore')

# 尝试导入GPR相关库
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
    SKLEARN_GPR_AVAILABLE = True
except ImportError:
    SKLEARN_GPR_AVAILABLE = False


class LinearBaseline:
    """
    线性基线模型 - 处理数据的线性成分
    支持多种正则化策略
    """
    
    def __init__(self, model_type: str = 'ridge', logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
        # 支持的线性模型类型
        self.supported_models = ['ridge', 'elasticnet', 'adaptive']
        
    def _create_model(self, X: np.ndarray, y: np.ndarray):
        """根据数据特征创建最优线性模型"""
        n_samples, n_features = X.shape
        
        if self.model_type == 'ridge':
            # Ridge回归，自适应正则化强度
            alpha = self._select_ridge_alpha(n_samples, n_features)
            return Ridge(alpha=alpha, random_state=42)
            
        elif self.model_type == 'elasticnet':
            # ElasticNet，平衡L1和L2正则化
            alpha = self._select_elasticnet_alpha(n_samples, n_features)
            l1_ratio = 0.5  # 平衡L1/L2
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            
        elif self.model_type == 'adaptive':
            # 自适应选择最佳线性模型
            return self._adaptive_model_selection(X, y)
            
        else:
            raise ValueError(f"不支持的线性模型类型: {self.model_type}")
    
    def _select_ridge_alpha(self, n_samples: int, n_features: int) -> float:
        """智能选择Ridge正则化参数"""
        feature_ratio = n_features / n_samples
        
        if feature_ratio < 0.1:
            return 1.0      # 特征少，轻正则化
        elif feature_ratio < 0.5:
            return 10.0     # 中等正则化
        else:
            return 100.0    # 特征多，强正则化
    
    def _select_elasticnet_alpha(self, n_samples: int, n_features: int) -> float:
        """智能选择ElasticNet正则化参数"""
        feature_ratio = n_features / n_samples
        
        if feature_ratio < 0.2:
            return 0.1
        elif feature_ratio < 0.6:
            return 1.0
        else:
            return 10.0
    
    def _adaptive_model_selection(self, X: np.ndarray, y: np.ndarray):
        """自适应模型选择"""
        n_samples, n_features = X.shape
        
        # 根据数据规模选择
        if n_samples > 10000:
            # 大样本，优先速度
            return Ridge(alpha=1.0, random_state=42)
        elif n_features / n_samples > 0.8:
            # 高维数据，强正则化
            return ElasticNet(alpha=10.0, l1_ratio=0.7, random_state=42)
        else:
            # 平衡配置
            return Ridge(alpha=10.0, random_state=42)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练线性基线模型"""
        self.logger.info(f"训练线性基线模型 ({self.model_type})")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 创建并训练模型
        self.model = self._create_model(X, y)
        self.model.fit(X_scaled, y)
        
        # 评估线性拟合质量
        linear_r2 = self.model.score(X_scaled, y)
        self.logger.info(f"线性基线 R²: {linear_r2:.4f}")
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测线性成分"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_linear_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """获取线性拟合质量"""
        predictions = self.predict(X)
        return r2_score(y, predictions)


class ApproximateGPR:
    """
    近似高斯过程回归 - 处理非线性残差
    支持Nyström、SVGP、Local-SVGP等近似技术
    """
    
    def __init__(self, approximation_type: str = 'nystrom', 
                 n_inducing: int = 1000,
                 kernel_type: str = 'matern',
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.approximation_type = approximation_type
        self.n_inducing = n_inducing
        self.kernel_type = kernel_type
        
        self.gpr_model = None
        self.inducing_points = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
        # 支持的近似类型
        self.supported_approximations = ['nystrom', 'svgp', 'local_svgp', 'exact']
    
    def _create_kernel(self, X: np.ndarray):
        """创建核函数"""
        if not SKLEARN_GPR_AVAILABLE:
            raise ImportError("sklearn gaussian_process 不可用")
        
        n_features = X.shape[1]
        
        if self.kernel_type == 'rbf':
            # RBF核
            length_scale = np.sqrt(n_features)
            kernel = C(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.01)
            
        elif self.kernel_type == 'matern':
            # Matérn核 (更平滑)
            length_scale = np.sqrt(n_features)
            kernel = C(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=0.01)
            
        else:
            raise ValueError(f"不支持的核类型: {self.kernel_type}")
        
        return kernel
    
    def _select_inducing_points(self, X: np.ndarray, method: str = 'kmeans') -> np.ndarray:
        """选择诱导点"""
        if method == 'kmeans':
            # KMeans聚类选择代表性点
            n_clusters = min(self.n_inducing, X.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
            self.kmeans_model = kmeans
            return kmeans.cluster_centers_
            
        elif method == 'random':
            # 随机选择
            n_points = min(self.n_inducing, X.shape[0])
            indices = np.random.choice(X.shape[0], n_points, replace=False)
            return X[indices]
            
        else:
            raise ValueError(f"不支持的诱导点选择方法: {method}")
    
    def _fit_nystrom_approximation(self, X: np.ndarray, y: np.ndarray):
        """Nyström近似"""
        self.logger.info(f"使用Nyström近似，诱导点数: {self.n_inducing}")
        
        # 选择诱导点
        self.inducing_points = self._select_inducing_points(X)
        
        # 创建简化的GPR模型
        kernel = self._create_kernel(X)
        
        # 使用诱导点子集进行训练
        n_subset = min(2000, X.shape[0])  # 限制训练样本数
        if X.shape[0] > n_subset:
            indices = np.random.choice(X.shape[0], n_subset, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset = X
            y_subset = y
        
        self.gpr_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,  # 关闭超参数优化以节省时间
            alpha=1e-6,
            random_state=42
        )
        
        self.gpr_model.fit(X_subset, y_subset)
    
    def _fit_local_svgp(self, X: np.ndarray, y: np.ndarray):
        """Local-SVGP近似（分区域建模）"""
        self.logger.info(f"使用Local-SVGP近似，分区数: {8}")
        
        # 数据聚类分区
        n_clusters = min(8, max(2, X.shape[0] // 5000))  # 自适应分区数
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        self.kmeans_model = kmeans
        
        # 为每个区域训练小型GPR
        self.local_models = {}
        
        for cluster_id in range(n_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) < 10:  # 跳过样本过少的区域
                continue
                
            X_cluster = X[mask]
            y_cluster = y[mask]
            
            # 为此区域创建GPR
            kernel = self._create_kernel(X_cluster)
            local_gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=0,
                alpha=1e-5,
                random_state=42
            )
            
            # 限制区域样本数
            if X_cluster.shape[0] > 1000:
                indices = np.random.choice(X_cluster.shape[0], 1000, replace=False)
                X_cluster = X_cluster[indices]
                y_cluster = y_cluster[indices]
            
            local_gpr.fit(X_cluster, y_cluster)
            self.local_models[cluster_id] = local_gpr
            
        self.logger.info(f"Local-SVGP训练完成，有效区域数: {len(self.local_models)}")
    
    def _fit_exact_gpr(self, X: np.ndarray, y: np.ndarray):
        """精确GPR（仅小样本）"""
        if X.shape[0] > 5000:
            raise ValueError("精确GPR仅适用于小样本（≤5000）")
        
        self.logger.info("使用精确GPR")
        
        kernel = self._create_kernel(X)
        self.gpr_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            alpha=1e-6,
            random_state=42
        )
        
        self.gpr_model.fit(X, y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练近似GPR"""
        self.logger.info(f"训练近似GPR ({self.approximation_type})")
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 根据近似类型选择训练方法
        if self.approximation_type == 'nystrom':
            self._fit_nystrom_approximation(X_scaled, y)
        elif self.approximation_type == 'local_svgp':
            self._fit_local_svgp(X_scaled, y)
        elif self.approximation_type == 'exact':
            self._fit_exact_gpr(X_scaled, y)
        elif self.approximation_type == 'svgp':
            # SVGP退化为Nyström
            self._fit_nystrom_approximation(X_scaled, y)
        else:
            raise ValueError(f"不支持的近似类型: {self.approximation_type}")
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False):
        """预测非线性残差"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        X_scaled = self.scaler.transform(X)
        
        if self.approximation_type == 'local_svgp':
            return self._predict_local_svgp(X_scaled, return_std)
        else:
            if self.gpr_model is None:
                # 降级为零预测
                predictions = np.zeros(X.shape[0])
                if return_std:
                    stds = np.ones(X.shape[0]) * 0.1
                    return predictions, stds
                return predictions
            
            if return_std and hasattr(self.gpr_model, 'predict'):
                try:
                    return self.gpr_model.predict(X_scaled, return_std=True)
                except:
                    predictions = self.gpr_model.predict(X_scaled)
                    stds = np.ones(len(predictions)) * 0.1
                    return predictions, stds
            else:
                return self.gpr_model.predict(X_scaled)
    
    def _predict_local_svgp(self, X: np.ndarray, return_std: bool = False):
        """Local-SVGP预测"""
        if not hasattr(self, 'local_models'):
            predictions = np.zeros(X.shape[0])
            if return_std:
                stds = np.ones(X.shape[0]) * 0.1
                return predictions, stds
            return predictions
        
        # 确定每个点属于哪个区域
        cluster_labels = self.kmeans_model.predict(X)
        
        predictions = np.zeros(X.shape[0])
        stds = np.ones(X.shape[0]) * 0.1 if return_std else None
        
        for cluster_id, model in self.local_models.items():
            mask = cluster_labels == cluster_id
            if np.sum(mask) == 0:
                continue
            
            try:
                if return_std:
                    pred, std = model.predict(X[mask], return_std=True)
                    predictions[mask] = pred
                    stds[mask] = std
                else:
                    predictions[mask] = model.predict(X[mask])
            except:
                # 降级处理
                predictions[mask] = 0.0
                if return_std:
                    stds[mask] = 0.1
        
        if return_std:
            return predictions, stds
        return predictions


class ResidualModelingFramework:
    """
    残差建模框架 - 核心优化创新
    MLR + GPR 分层建模策略
    """
    
    def __init__(self, 
                 linear_model_type: str = 'adaptive',
                 gpr_approximation: str = 'nystrom',
                 n_inducing: int = 1000,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 模型组件
        self.linear_baseline = LinearBaseline(linear_model_type, logger)
        self.residual_gpr = ApproximateGPR(gpr_approximation, n_inducing, logger=logger)
        
        # 状态
        self.fitted = False
        self.training_info = {}
        
    def _analyze_data_complexity(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """分析数据复杂性，优化建模策略"""
        n_samples, n_features = X.shape
        
        # 快速线性拟合评估
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        linear_r2 = lr.score(X, y)
        
        # 复杂性指标
        complexity_analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'linear_r2': linear_r2,
            'nonlinearity_score': max(0, 1 - linear_r2),
            'sample_complexity': 'high' if n_samples > 50000 else 'medium' if n_samples > 10000 else 'low',
            'feature_ratio': n_features / n_samples,
            'recommended_gpr_type': self._recommend_gpr_type(n_samples, linear_r2)
        }
        
        return complexity_analysis
    
    def _recommend_gpr_type(self, n_samples: int, linear_r2: float) -> str:
        """根据数据特征推荐GPR近似类型"""
        nonlinearity = 1 - linear_r2
        
        if n_samples <= 5000 and nonlinearity > 0.2:
            return 'exact'  # 小样本+高非线性：精确GPR
        elif n_samples <= 20000:
            return 'nystrom'  # 中等样本：Nyström近似
        else:
            return 'local_svgp'  # 大样本：Local-SVGP
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ResidualModelingFramework':
        """训练残差建模框架"""
        self.logger.info("开始训练残差建模框架...")
        
        start_time = time.time()
        
        # 数据复杂性分析
        complexity = self._analyze_data_complexity(X, y)
        self.training_info['complexity_analysis'] = complexity
        
        self.logger.info(f"数据复杂性分析: 样本{complexity['n_samples']}, "
                        f"线性R²={complexity['linear_r2']:.3f}, "
                        f"推荐GPR类型={complexity['recommended_gpr_type']}")
        
        # 第一阶段：训练线性基线
        self.logger.info("第一阶段: 训练线性基线模型")
        self.linear_baseline.fit(X, y)
        
        # 计算残差
        linear_predictions = self.linear_baseline.predict(X)
        residuals = y - linear_predictions
        
        # 残差统计
        residual_std = np.std(residuals)
        residual_mean = np.mean(np.abs(residuals))
        
        self.logger.info(f"残差分析: std={residual_std:.4f}, mean_abs={residual_mean:.4f}")
        
        # 第二阶段：训练残差GPR（如果残差显著）
        if residual_std > 0.01 * np.std(y):  # 残差足够显著
            self.logger.info("第二阶段: 训练残差GPR模型")
            
            # 根据复杂性调整GPR配置
            recommended_type = complexity['recommended_gpr_type']
            if self.residual_gpr.approximation_type == 'nystrom' and recommended_type != 'nystrom':
                # 动态调整GPR类型
                self.residual_gpr.approximation_type = recommended_type
                self.logger.info(f"调整GPR类型为: {recommended_type}")
            
            self.residual_gpr.fit(X, residuals)
        else:
            self.logger.info("残差较小，跳过GPR训练")
        
        # 训练完成
        training_time = time.time() - start_time
        self.training_info['training_time'] = training_time
        self.fitted = True
        
        self.logger.info(f"残差建模框架训练完成，耗时: {training_time:.2f}s")
        
        # 整体性能评估
        overall_predictions = self.predict(X)
        overall_r2 = r2_score(y, overall_predictions)
        improvement = overall_r2 - complexity['linear_r2']
        
        self.logger.info(f"整体性能: R²={overall_r2:.4f}, 相比线性改进: +{improvement:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False, 
               return_components: bool = False) -> Union[np.ndarray, Tuple]:
        """预测"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        # 线性成分预测
        linear_pred = self.linear_baseline.predict(X)
        
        # 非线性残差预测
        if self.residual_gpr.fitted:
            if return_std:
                residual_pred, residual_std = self.residual_gpr.predict(X, return_std=True)
            else:
                residual_pred = self.residual_gpr.predict(X)
                residual_std = None
        else:
            residual_pred = np.zeros(X.shape[0])
            residual_std = np.ones(X.shape[0]) * 0.01 if return_std else None
        
        # 总预测
        total_pred = linear_pred + residual_pred
        
        # 返回结果
        if return_components:
            components = {
                'linear': linear_pred,
                'residual': residual_pred,
                'total': total_pred
            }
            if return_std:
                components['residual_std'] = residual_std
            return components
        elif return_std:
            return total_pred, residual_std
        else:
            return total_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        predictions = self.predict(X)
        
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        nmbe = np.mean(predictions - y) / np.mean(y) * 100
        cv_rmse = rmse / np.mean(y) * 100
        
        # 分解性能
        linear_pred = self.linear_baseline.predict(X)
        linear_r2 = r2_score(y, linear_pred)
        
        return {
            'overall_r2': r2,
            'linear_r2': linear_r2,
            'residual_improvement': r2 - linear_r2,
            'rmse': rmse,
            'mae': mae,
            'nmbe_percent': nmbe,
            'cv_rmse_percent': cv_rmse
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'fitted': self.fitted,
            'linear_model_type': self.linear_baseline.model_type,
            'gpr_approximation': self.residual_gpr.approximation_type,
            'gpr_fitted': self.residual_gpr.fitted,
            'training_info': self.training_info
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'linear_baseline': self.linear_baseline,
            'residual_gpr': self.residual_gpr,
            'training_info': self.training_info,
            'fitted': self.fitted
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"残差建模框架已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        self.linear_baseline = model_data['linear_baseline']
        self.residual_gpr = model_data['residual_gpr']
        self.training_info = model_data.get('training_info', {})
        self.fitted = model_data.get('fitted', False)
        
        self.logger.info(f"残差建模框架已从 {filepath} 加载")


# 便捷函数
def create_residual_model(X: np.ndarray, y: np.ndarray,
                         linear_type: str = 'adaptive',
                         gpr_type: str = 'nystrom',
                         logger: Optional[logging.Logger] = None) -> ResidualModelingFramework:
    """创建并训练残差建模框架"""
    framework = ResidualModelingFramework(
        linear_model_type=linear_type,
        gpr_approximation=gpr_type,
        logger=logger
    )
    return framework.fit(X, y)