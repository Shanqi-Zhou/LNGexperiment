"""
HGBR基线模型集成 - 基于优化方案的自适应基线策略
集成Histogram-based Gradient Boosting作为中大样本的核心基线
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, Tuple
import logging
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import time

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AdaptiveHGBRBaseline:
    """
    自适应HGBR基线模型 - 根据数据规模自动选择最优配置
    整合优化方案的分层自适应策略
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.best_model_name = None
        self.data_profile = {}
        
    def _profile_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """数据集剖析 - 确定最优策略"""
        n_samples, n_features = X.shape
        
        # 计算数据特征
        feature_variance = np.var(X, axis=0)
        target_variance = np.var(y)
        correlation_matrix = np.corrcoef(X.T)
        
        # 检测多重共线性
        high_corr_pairs = 0
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                if abs(correlation_matrix[i,j]) > 0.8:
                    high_corr_pairs += 1
        
        # 检测噪声水平
        noise_level = self._estimate_noise_level(X, y)
        
        profile = {
            'n_samples': n_samples,
            'n_features': n_features,
            'sample_size_category': self._categorize_sample_size(n_samples),
            'feature_variance_mean': np.mean(feature_variance),
            'target_variance': target_variance,
            'high_correlation_pairs': high_corr_pairs,
            'estimated_noise_level': noise_level,
            'features_per_sample_ratio': n_features / n_samples
        }
        
        self.data_profile = profile
        self.logger.info(f"数据集剖析完成: {n_samples}样本, {n_features}特征, 类别={profile['sample_size_category']}")
        
        return profile
    
    def _categorize_sample_size(self, n_samples: int) -> str:
        """样本量分类 - 对应优化方案的分层策略"""
        if n_samples <= 10000:
            return "small"      # ≤1万: 精度优先
        elif n_samples <= 100000:
            return "medium"     # 1-10万: 平衡效率
        else:
            return "large"      # >10万: 速度优先
    
    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """估计数据噪声水平"""
        try:
            # 使用简单的Ridge回归估计噪声
            ridge = Ridge(alpha=1.0)
            scores = cross_val_score(ridge, X, y, cv=3, scoring='r2')
            noise_level = 1.0 - np.mean(scores)
            return max(0.0, min(1.0, noise_level))
        except:
            return 0.5  # 默认中等噪声
    
    def _create_small_sample_models(self, X: np.ndarray, y: np.ndarray):
        """小样本策略: 精度优先，MLR基线"""
        self.logger.info("使用小样本策略: Ridge+ElasticNet基线")
        
        # Ridge回归（多个正则化强度）
        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            model_name = f'ridge_alpha_{alpha}'
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
        # ElasticNet（平衡L1/L2）
        for l1_ratio in [0.3, 0.5, 0.7]:
            model = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, random_state=42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            model_name = f'elasticnet_l1_{l1_ratio}'
            self.models[model_name] = model
            self.scalers[model_name] = scaler
        
        # 轻量级随机森林作为非线性补充
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        self.models['rf_light'] = model
        self.scalers['rf_light'] = None
        
    def _create_medium_sample_models(self, X: np.ndarray, y: np.ndarray):
        """中样本策略: HGBR + LightGBM"""
        self.logger.info("使用中样本策略: HGBR主导 + LightGBM辅助")
        
        # HGBR配置（平衡精度和速度）
        hgbr_configs = [
            {
                'name': 'hgbr_balanced',
                'max_iter': 200,
                'max_depth': 8,
                'learning_rate': 0.1,
                'l2_regularization': 0.1,
                'min_samples_leaf': 20
            },
            {
                'name': 'hgbr_deep',
                'max_iter': 150,
                'max_depth': 12,
                'learning_rate': 0.08,
                'l2_regularization': 0.05,
                'min_samples_leaf': 10
            },
            {
                'name': 'hgbr_regularized',
                'max_iter': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'l2_regularization': 0.2,
                'min_samples_leaf': 50
            }
        ]
        
        for config in hgbr_configs:
            model = HistGradientBoostingRegressor(
                max_iter=config['max_iter'],
                max_depth=config['max_depth'],
                learning_rate=config['learning_rate'],
                l2_regularization=config['l2_regularization'],
                min_samples_leaf=config['min_samples_leaf'],
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            )
            
            model.fit(X, y)
            self.models[config['name']] = model
            self.scalers[config['name']] = None
            
        # LightGBM作为辅助模型（如果可用）
        if LIGHTGBM_AVAILABLE:
            lgb_configs = [
                {
                    'name': 'lgb_fast',
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'num_leaves': 31,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8
                },
                {
                    'name': 'lgb_accurate',
                    'n_estimators': 400,
                    'learning_rate': 0.05,
                    'num_leaves': 63,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.9
                }
            ]
            
            for config in lgb_configs:
                model = lgb.LGBMRegressor(
                    n_estimators=config['n_estimators'],
                    learning_rate=config['learning_rate'],
                    num_leaves=config['num_leaves'],
                    feature_fraction=config['feature_fraction'],
                    bagging_fraction=config['bagging_fraction'],
                    bagging_freq=5,
                    objective='regression',
                    metric='rmse',
                    verbosity=-1,
                    random_state=42
                )
                
                model.fit(X, y)
                self.models[config['name']] = model
                self.scalers[config['name']] = None
        
        # Ridge基线（正则化基准）
        model = Ridge(alpha=1.0, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        self.models['ridge_baseline'] = model
        self.scalers['ridge_baseline'] = scaler
        
    def _create_large_sample_models(self, X: np.ndarray, y: np.ndarray):
        """大样本策略: HGBR高速配置"""
        self.logger.info("使用大样本策略: HGBR高速优化")
        
        # HGBR高速配置（速度优先）
        hgbr_configs = [
            {
                'name': 'hgbr_fast',
                'max_iter': 100,
                'max_depth': 6,
                'learning_rate': 0.2,
                'l2_regularization': 0.1,
                'min_samples_leaf': 100
            },
            {
                'name': 'hgbr_turbo',
                'max_iter': 50,
                'max_depth': 4,
                'learning_rate': 0.3,
                'l2_regularization': 0.05,
                'min_samples_leaf': 200
            },
            {
                'name': 'hgbr_balanced_large',
                'max_iter': 150,
                'max_depth': 8,
                'learning_rate': 0.15,
                'l2_regularization': 0.15,
                'min_samples_leaf': 50
            }
        ]
        
        for config in hgbr_configs:
            model = HistGradientBoostingRegressor(
                max_iter=config['max_iter'],
                max_depth=config['max_depth'],
                learning_rate=config['learning_rate'],
                l2_regularization=config['l2_regularization'],
                min_samples_leaf=config['min_samples_leaf'],
                early_stopping=True,
                validation_fraction=0.05,  # 更小的验证集
                n_iter_no_change=10,       # 更早停止
                random_state=42
            )
            
            model.fit(X, y)
            self.models[config['name']] = model
            self.scalers[config['name']] = None
        
        # 轻量级LightGBM（如果可用）
        if LIGHTGBM_AVAILABLE:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.2,
                num_leaves=31,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=3,
                objective='regression',
                metric='rmse',
                verbosity=-1,
                random_state=42
            )
            model.fit(X, y)
            self.models['lgb_fast_large'] = model
            self.scalers['lgb_fast_large'] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveHGBRBaseline':
        """自适应训练基线模型"""
        self.logger.info("开始自适应HGBR基线模型训练...")
        
        start_time = time.time()
        
        # 数据集剖析
        profile = self._profile_dataset(X, y)
        
        # 根据样本量选择策略
        if profile['sample_size_category'] == 'small':
            self._create_small_sample_models(X, y)
        elif profile['sample_size_category'] == 'medium':
            self._create_medium_sample_models(X, y)
        else:  # large
            self._create_large_sample_models(X, y)
        
        # 选择最佳模型
        self._select_best_model(X, y)
        
        training_time = time.time() - start_time
        self.logger.info(f"HGBR基线训练完成，耗时: {training_time:.2f}s，最佳模型: {self.best_model_name}")
        
        return self
    
    def _select_best_model(self, X: np.ndarray, y: np.ndarray):
        """选择最佳基线模型"""
        if not self.models:
            return
        
        best_score = -np.inf
        best_name = None
        
        # 评估所有模型
        for name, model in self.models.items():
            try:
                if name in self.scalers and self.scalers[name] is not None:
                    X_eval = self.scalers[name].transform(X)
                else:
                    X_eval = X
                
                # 使用交叉验证评分
                scores = cross_val_score(model, X_eval, y, cv=3, scoring='r2')
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_name = name
                    
                self.logger.debug(f"模型 {name}: CV R² = {avg_score:.4f}")
                    
            except Exception as e:
                self.logger.warning(f"模型 {name} 评估失败: {e}")
        
        self.best_model_name = best_name
        self.logger.info(f"最佳基线模型: {best_name}, CV R² = {best_score:.4f}")
    
    def predict(self, X: np.ndarray, use_best: bool = True) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """预测"""
        if use_best:
            if self.best_model_name is None:
                raise ValueError("未选择最佳模型，请先训练")
            return self._predict_single(self.best_model_name, X)
        else:
            # 返回所有模型预测
            predictions = {}
            for name in self.models.keys():
                predictions[name] = self._predict_single(name, X)
            return predictions
    
    def _predict_single(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """单模型预测"""
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        return model.predict(X_scaled)
    
    def get_feature_importance(self, model_name: str = None) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        else:
            return None
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """评估所有模型性能"""
        results = []
        
        for name, model in self.models.items():
            try:
                predictions = self._predict_single(name, X)
                
                r2 = r2_score(y, predictions)
                mse = mean_squared_error(y, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, predictions)
                nmbe = np.mean(predictions - y) / np.mean(y) * 100
                cv_rmse = rmse / np.mean(y) * 100
                
                results.append({
                    'model_name': name,
                    'is_best': name == self.best_model_name,
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'NMBE(%)': nmbe,
                    'CV(RMSE)(%)': cv_rmse
                })
                
            except Exception as e:
                self.logger.warning(f"模型 {name} 评估失败: {e}")
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('R²', ascending=False)
        
        return df
    
    def save_model(self, filepath: str, model_name: str = None):
        """保存模型"""
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers[model_name],
            'model_name': model_name,
            'data_profile': self.data_profile
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"模型 {model_name} 已保存到 {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        
        model_name = model_data['model_name']
        self.models[model_name] = model_data['model']
        self.scalers[model_name] = model_data['scaler']
        self.best_model_name = model_name
        self.data_profile = model_data.get('data_profile', {})
        
        self.logger.info(f"模型 {model_name} 已从 {filepath} 加载")
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要"""
        return {
            'data_profile': self.data_profile,
            'num_models_trained': len(self.models),
            'best_model': self.best_model_name,
            'available_models': list(self.models.keys()),
            'strategy_used': self.data_profile.get('sample_size_category', 'unknown')
        }


# 便捷函数
def create_adaptive_hgbr_baseline(X: np.ndarray, y: np.ndarray, 
                                 logger: Optional[logging.Logger] = None) -> AdaptiveHGBRBaseline:
    """创建自适应HGBR基线模型"""
    baseline = AdaptiveHGBRBaseline(logger)
    return baseline.fit(X, y)