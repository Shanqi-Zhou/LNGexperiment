"""
样本量自适应模型选择策略 - 优化方案核心创新
根据数据规模自动选择最优算法组合（小/中/大样本分层策略）
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, Tuple, List
import logging
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class SampleSizeAdaptiveModelFactory:
    """
    样本量自适应模型工厂
    实现优化方案的分层自适应策略：
    - 小样本(≤1万): MLR基线 + 精确GPR残差 + 轻量交叉注意力
    - 中样本(1-10万): HGBR基线 + Nyström/SVGP + TCN+单层注意力
    - 大样本(>10万): HGBR基线 + Local-SVGP + TCN+单层注意力+混合精度
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.strategy_cache = {}
        self.performance_history = []
        
    def analyze_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        数据集分析 - 确定最优策略
        """
        n_samples, n_features = X.shape
        
        # 基本统计
        analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_size_category': self._categorize_sample_size(n_samples),
            'feature_to_sample_ratio': n_features / n_samples,
            'target_variance': np.var(y),
            'target_range': np.ptp(y),  # peak-to-peak range
        }
        
        # 数据复杂性评估
        analysis.update(self._assess_data_complexity(X, y))
        
        # 计算资源评估
        analysis.update(self._assess_computational_resources(n_samples, n_features))
        
        # 推荐策略
        analysis['recommended_strategy'] = self._recommend_strategy(analysis)
        
        self.logger.info(f"数据集分析完成: {n_samples}样本, {n_features}特征")
        self.logger.info(f"类别: {analysis['sample_size_category']}, 推荐策略: {analysis['recommended_strategy']}")
        
        return analysis
    
    def _categorize_sample_size(self, n_samples: int) -> str:
        """样本量分类"""
        if n_samples <= 10000:
            return "small"      # ≤1万: 精度优先
        elif n_samples <= 100000:
            return "medium"     # 1-10万: 平衡效率
        else:
            return "large"      # >10万: 速度优先
    
    def _assess_data_complexity(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """数据复杂性评估"""
        try:
            # 线性相关性（使用简单线性回归的R²）
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            linear_r2 = lr.score(X, y)
            
            # 特征相关性
            correlation_matrix = np.corrcoef(X.T)
            high_corr_count = np.sum(np.abs(correlation_matrix) > 0.8) - X.shape[1]  # 减去对角线
            
            # 噪声水平估计
            residuals = y - lr.predict(X)
            noise_level = np.std(residuals) / np.std(y)
            
            # 非线性指标（基于残差的方差结构）
            nonlinearity_score = 1.0 - linear_r2
            
            return {
                'linear_r2': linear_r2,
                'high_correlation_pairs': high_corr_count,
                'estimated_noise_level': noise_level,
                'nonlinearity_score': nonlinearity_score,
                'complexity_level': 'high' if nonlinearity_score > 0.3 else 'medium' if nonlinearity_score > 0.1 else 'low'
            }
            
        except Exception as e:
            self.logger.warning(f"数据复杂性评估失败: {e}")
            return {
                'linear_r2': 0.5,
                'high_correlation_pairs': 0,
                'estimated_noise_level': 0.5,
                'nonlinearity_score': 0.3,
                'complexity_level': 'medium'
            }
    
    def _assess_computational_resources(self, n_samples: int, n_features: int) -> Dict[str, Any]:
        """计算资源评估"""
        # 内存需求估计（基于经验公式）
        base_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)  # 基础数据内存
        
        # GPR内存需求（O(n²)对于精确GPR，O(mn)对于近似）
        exact_gpr_memory_mb = (n_samples ** 2 * 8) / (1024 * 1024)
        approximate_gpr_memory_mb = min(1500 * n_samples * 8 / (1024 * 1024), exact_gpr_memory_mb)
        
        # 训练时间估计（基于经验）
        estimated_training_time = {
            'linear_models': max(1, n_samples / 50000),  # 秒
            'tree_models': max(5, n_samples / 10000),
            'exact_gpr': max(10, (n_samples ** 1.5) / 100000),
            'approximate_gpr': max(5, n_samples / 5000),
            'deep_models': max(30, n_samples / 1000)
        }
        
        return {
            'base_memory_mb': base_memory_mb,
            'exact_gpr_memory_mb': exact_gpr_memory_mb,
            'approximate_gpr_memory_mb': approximate_gpr_memory_mb,
            'estimated_training_time': estimated_training_time,
            'memory_constraint': exact_gpr_memory_mb > 6000,  # RTX 4060 8GB约6GB可用
            'time_constraint': estimated_training_time['exact_gpr'] > 300  # 5分钟限制
        }
    
    def _recommend_strategy(self, analysis: Dict[str, Any]) -> str:
        """推荐最优策略"""
        sample_category = analysis['sample_size_category']
        complexity = analysis['complexity_level']
        memory_constrained = analysis['memory_constraint']
        time_constrained = analysis['time_constraint']
        
        if sample_category == 'small':
            if complexity == 'low':
                return 'small_linear'      # MLR主导
            elif not memory_constrained:
                return 'small_exact_gpr'   # 精确GPR
            else:
                return 'small_approximate' # 近似GPR
                
        elif sample_category == 'medium':
            if time_constrained or memory_constrained:
                return 'medium_fast'       # HGBR + 轻量GPR
            elif complexity == 'high':
                return 'medium_complex'    # HGBR + SVGP + 深度模型
            else:
                return 'medium_balanced'   # HGBR + Nyström GPR
                
        else:  # large
            if complexity == 'low':
                return 'large_simple'      # HGBR主导
            elif complexity == 'high':
                return 'large_complex'     # Local-SVGP + 深度模型
            else:
                return 'large_balanced'    # 平衡配置
    
    def create_model_ensemble(self, X: np.ndarray, y: np.ndarray, 
                            strategy: str = None) -> 'AdaptiveModelEnsemble':
        """
        创建自适应模型集成
        """
        # 数据分析
        analysis = self.analyze_dataset(X, y)
        
        # 确定策略
        if strategy is None:
            strategy = analysis['recommended_strategy']
        
        self.logger.info(f"使用策略: {strategy}")
        
        # 创建模型集成
        ensemble = AdaptiveModelEnsemble(strategy, analysis, self.logger)
        
        return ensemble
    
    def get_strategy_description(self, strategy: str) -> Dict[str, Any]:
        """获取策略详细描述"""
        strategy_configs = {
            'small_linear': {
                'description': '小样本线性策略',
                'models': ['Ridge', 'ElasticNet'],
                'focus': '精度优先，简单快速',
                'memory_usage': 'Low',
                'training_time': 'Fast'
            },
            'small_exact_gpr': {
                'description': '小样本精确GPR策略',
                'models': ['MLR基线', '精确GPR残差'],
                'focus': '最高精度，不确定性量化',
                'memory_usage': 'Medium',
                'training_time': 'Medium'
            },
            'small_approximate': {
                'description': '小样本近似GPR策略',
                'models': ['MLR基线', 'Nyström GPR'],
                'focus': '平衡精度和效率',
                'memory_usage': 'Low',
                'training_time': 'Fast'
            },
            'medium_fast': {
                'description': '中样本快速策略',
                'models': ['HGBR', '轻量GPR'],
                'focus': '速度优先',
                'memory_usage': 'Medium',
                'training_time': 'Fast'
            },
            'medium_balanced': {
                'description': '中样本平衡策略',
                'models': ['HGBR基线', 'Nyström GPR', 'TCN'],
                'focus': '精度与效率平衡',
                'memory_usage': 'Medium',
                'training_time': 'Medium'
            },
            'medium_complex': {
                'description': '中样本复杂策略',
                'models': ['HGBR', 'SVGP', 'TCN+Attention'],
                'focus': '处理复杂非线性',
                'memory_usage': 'High',
                'training_time': 'Slow'
            },
            'large_simple': {
                'description': '大样本简单策略',
                'models': ['HGBR高速配置'],
                'focus': '极致速度',
                'memory_usage': 'Medium',
                'training_time': 'Fast'
            },
            'large_balanced': {
                'description': '大样本平衡策略',
                'models': ['HGBR', 'Local-SVGP'],
                'focus': '大规模处理',
                'memory_usage': 'High',
                'training_time': 'Medium'
            },
            'large_complex': {
                'description': '大样本复杂策略',
                'models': ['HGBR', 'Local-SVGP', 'TCN+Linear Attention+AMP'],
                'focus': '复杂大规模建模',
                'memory_usage': 'High',
                'training_time': 'Slow'
            }
        }
        
        return strategy_configs.get(strategy, {'description': '未知策略'})


class AdaptiveModelEnsemble:
    """
    自适应模型集成 - 执行具体的建模策略
    """
    
    def __init__(self, strategy: str, analysis: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        self.strategy = strategy
        self.analysis = analysis
        self.logger = logger or logging.getLogger(__name__)
        
        self.models = {}
        self.fitted = False
        self.best_model_name = None
        self.training_time = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveModelEnsemble':
        """训练自适应模型集成"""
        self.logger.info(f"开始训练自适应模型集成，策略: {self.strategy}")
        
        start_time = time.time()
        
        # 根据策略创建和训练模型
        if self.strategy.startswith('small_'):
            self._fit_small_sample_models(X, y)
        elif self.strategy.startswith('medium_'):
            self._fit_medium_sample_models(X, y)
        elif self.strategy.startswith('large_'):
            self._fit_large_sample_models(X, y)
        else:
            raise ValueError(f"未知策略: {self.strategy}")
        
        # 选择最佳模型
        self._select_best_model(X, y)
        
        self.training_time = time.time() - start_time
        self.fitted = True
        
        self.logger.info(f"自适应模型训练完成，耗时: {self.training_time:.2f}s")
        self.logger.info(f"最佳模型: {self.best_model_name}")
        
        return self
    
    def _fit_small_sample_models(self, X: np.ndarray, y: np.ndarray):
        """小样本模型训练"""
        from ..models.hgbr_baseline import AdaptiveHGBRBaseline
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.preprocessing import StandardScaler
        
        if self.strategy == 'small_linear':
            # 线性模型为主
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Ridge回归
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_scaled, y)
            self.models['ridge'] = {'model': ridge, 'scaler': scaler}
            
            # ElasticNet
            elastic = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
            elastic.fit(X_scaled, y)
            self.models['elasticnet'] = {'model': elastic, 'scaler': scaler}
            
        elif self.strategy == 'small_exact_gpr':
            # MLR + 精确GPR残差
            self._create_residual_gpr_model(X, y, gpr_type='exact')
            
        elif self.strategy == 'small_approximate':
            # MLR + 近似GPR
            self._create_residual_gpr_model(X, y, gpr_type='nystrom')
    
    def _fit_medium_sample_models(self, X: np.ndarray, y: np.ndarray):
        """中样本模型训练"""
        from ..models.hgbr_baseline import AdaptiveHGBRBaseline
        
        # HGBR基线
        hgbr_baseline = AdaptiveHGBRBaseline(self.logger)
        hgbr_baseline.fit(X, y)
        self.models['hgbr_baseline'] = hgbr_baseline
        
        if self.strategy == 'medium_fast':
            # 快速策略：只用HGBR
            pass
            
        elif self.strategy == 'medium_balanced':
            # 平衡策略：HGBR + Nyström GPR
            self._create_residual_gpr_model(X, y, gpr_type='nystrom')
            
        elif self.strategy == 'medium_complex':
            # 复杂策略：HGBR + SVGP + 深度模型
            self._create_residual_gpr_model(X, y, gpr_type='svgp')
            # TODO: 添加深度模型（下一个任务）
    
    def _fit_large_sample_models(self, X: np.ndarray, y: np.ndarray):
        """大样本模型训练"""
        from ..models.hgbr_baseline import AdaptiveHGBRBaseline
        
        # HGBR高速配置
        hgbr_baseline = AdaptiveHGBRBaseline(self.logger)
        hgbr_baseline.fit(X, y)
        self.models['hgbr_fast'] = hgbr_baseline
        
        if self.strategy == 'large_simple':
            # 简单策略：只用HGBR
            pass
            
        elif self.strategy in ['large_balanced', 'large_complex']:
            # Local-SVGP
            self._create_residual_gpr_model(X, y, gpr_type='local_svgp')
            # TODO: 大样本深度模型优化
    
    def _create_residual_gpr_model(self, X: np.ndarray, y: np.ndarray, gpr_type: str):
        """创建残差GPR模型"""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from ..models.gpr import GaussianProcessRegression
        
        # MLR基线
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        mlr = Ridge(alpha=1.0, random_state=42)
        mlr.fit(X_scaled, y)
        
        # 计算残差
        y_linear = mlr.predict(X_scaled)
        residuals = y - y_linear
        
        # GPR配置
        if gpr_type == 'exact':
            gpr_config = {'kernel': 'matern', 'approximation': None}
        elif gpr_type == 'nystrom':
            gpr_config = {'kernel': 'matern', 'approximation': 'nystrom', 'n_inducing': 1500}
        elif gpr_type == 'svgp':
            gpr_config = {'kernel': 'matern', 'approximation': 'svgp', 'n_inducing': 1000}
        elif gpr_type == 'local_svgp':
            gpr_config = {'kernel': 'matern', 'approximation': 'local_svgp', 'n_clusters': 8}
        else:
            raise ValueError(f"未知GPR类型: {gpr_type}")
        
        # 训练GPR
        try:
            gpr = GaussianProcessRegression(logger=self.logger, **gpr_config)
            gpr.fit(X_scaled, residuals)
            
            # 保存残差模型
            residual_model = {
                'mlr': mlr,
                'gpr': gpr,
                'scaler': scaler,
                'type': 'residual'
            }
            
            self.models[f'residual_gpr_{gpr_type}'] = residual_model
            
        except Exception as e:
            self.logger.warning(f"GPR训练失败 ({gpr_type}): {e}")
    
    def _select_best_model(self, X: np.ndarray, y: np.ndarray):
        """选择最佳模型"""
        best_score = -np.inf
        best_name = None
        
        for name, model_data in self.models.items():
            try:
                predictions = self._predict_single(name, X)
                score = r2_score(y, predictions)
                
                if score > best_score:
                    best_score = score
                    best_name = name
                    
                self.logger.debug(f"模型 {name}: R² = {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"模型 {name} 评估失败: {e}")
        
        self.best_model_name = best_name
    
    def predict(self, X: np.ndarray, return_std: bool = False, 
               use_best: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测"""
        if not self.fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        if use_best:
            return self._predict_single(self.best_model_name, X, return_std)
        else:
            # 集成预测
            predictions = []
            for name in self.models.keys():
                pred = self._predict_single(name, X)
                predictions.append(pred)
            
            ensemble_pred = np.mean(predictions, axis=0)
            
            if return_std:
                ensemble_std = np.std(predictions, axis=0)
                return ensemble_pred, ensemble_std
            return ensemble_pred
    
    def _predict_single(self, model_name: str, X: np.ndarray, 
                       return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """单模型预测"""
        model_data = self.models[model_name]
        
        if hasattr(model_data, 'predict'):  # HGBR baseline
            return model_data.predict(X, use_best=True)
            
        elif isinstance(model_data, dict):
            if model_data.get('type') == 'residual':
                # 残差模型预测
                scaler = model_data['scaler']
                mlr = model_data['mlr']
                gpr = model_data['gpr']
                
                X_scaled = scaler.transform(X)
                y_linear = mlr.predict(X_scaled)
                
                if return_std and hasattr(gpr, 'predict'):
                    y_residual, residual_std = gpr.predict(X_scaled, return_std=True)
                    return y_linear + y_residual, residual_std
                else:
                    y_residual = gpr.predict(X_scaled)
                    return y_linear + y_residual
            else:
                # 普通模型
                model = model_data['model']
                scaler = model_data.get('scaler')
                
                if scaler is not None:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
                    
                return model.predict(X_scaled)
        
        raise ValueError(f"无法处理模型: {model_name}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'strategy': self.strategy,
            'analysis': self.analysis,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'best_model': self.best_model_name,
            'training_time': self.training_time,
            'fitted': self.fitted
        }


# 便捷函数
def create_adaptive_model(X: np.ndarray, y: np.ndarray, 
                         strategy: str = None,
                         logger: Optional[logging.Logger] = None) -> AdaptiveModelEnsemble:
    """创建并训练自适应模型"""
    factory = SampleSizeAdaptiveModelFactory(logger)
    ensemble = factory.create_model_ensemble(X, y, strategy)
    return ensemble.fit(X, y)