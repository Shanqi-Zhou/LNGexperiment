"""
双通道特征选择系统 - 优化方案智能特征工程
专门处理V_t(动态时序)和V_s(静态上下文)特征的分离选择与智能融合
结合统计方法、机器学习方法和领域知识进行最优特征组合
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any, Tuple, List, Set
import logging
import warnings
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats
from itertools import combinations
import time

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器 - 多角度评估特征重要性
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.importance_scores = {}
        
    def analyze_statistical_importance(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> Dict[str, float]:
        """统计重要性分析"""
        self.logger.info("计算统计重要性...")
        
        importance_dict = {}
        
        # 皮尔逊相关系数
        for i, feature_name in enumerate(feature_names):
            corr, p_value = stats.pearsonr(X[:, i], y)
            importance_dict[f"{feature_name}_pearson"] = abs(corr)
            
        # F-统计量
        try:
            f_scores = f_regression(X, y)[0]
            for i, feature_name in enumerate(feature_names):
                importance_dict[f"{feature_name}_f_score"] = f_scores[i]
        except:
            self.logger.warning("F-统计量计算失败")
        
        # 互信息
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42)
            for i, feature_name in enumerate(feature_names):
                importance_dict[f"{feature_name}_mutual_info"] = mi_scores[i]
        except:
            self.logger.warning("互信息计算失败")
            
        return importance_dict
    
    def analyze_model_based_importance(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> Dict[str, float]:
        """基于模型的重要性分析"""
        self.logger.info("计算模型重要性...")
        
        importance_dict = {}
        
        # 随机森林重要性
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            for i, feature_name in enumerate(feature_names):
                importance_dict[f"{feature_name}_rf_importance"] = rf.feature_importances_[i]
        except Exception as e:
            self.logger.warning(f"随机森林重要性计算失败: {e}")
        
        # Lasso重要性 (L1正则化)
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(X_scaled, y)
            
            for i, feature_name in enumerate(feature_names):
                importance_dict[f"{feature_name}_lasso_coef"] = abs(lasso.coef_[i])
        except Exception as e:
            self.logger.warning(f"Lasso重要性计算失败: {e}")
            
        return importance_dict
    
    def compute_composite_importance(self, importance_dict: Dict[str, float], 
                                   feature_names: List[str]) -> Dict[str, float]:
        """计算复合重要性分数"""
        composite_scores = {}
        
        for feature_name in feature_names:
            scores = []
            
            # 收集该特征的所有重要性分数
            for key, value in importance_dict.items():
                if key.startswith(feature_name + '_'):
                    scores.append(value)
            
            if scores:
                # 标准化后平均
                scores = np.array(scores)
                if np.std(scores) > 0:
                    scores = (scores - np.mean(scores)) / np.std(scores)
                
                composite_scores[feature_name] = np.mean(scores)
            else:
                composite_scores[feature_name] = 0.0
                
        return composite_scores


class VtFeatureSelector:
    """
    V_t动态特征选择器 - 专门处理时序动态特征
    """
    
    def __init__(self, window_sizes: List[int] = [30, 60, 120, 180], 
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.window_sizes = window_sizes
        self.selected_features = []
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(logger)
        
    def extract_temporal_features(self, data: pd.DataFrame, 
                                target_column: str) -> Dict[str, np.ndarray]:
        """提取时序特征"""
        self.logger.info("提取V_t动态特征...")
        
        features = {}
        
        for window in self.window_sizes:
            window_name = f"w{window}"
            
            # 滚动统计特征
            rolling = data.rolling(window=window, min_periods=max(1, window//2))
            
            # 基础统计
            features[f"{window_name}_mean"] = rolling.mean().iloc[:, :-1].values  # 排除目标列
            features[f"{window_name}_std"] = rolling.std().iloc[:, :-1].values
            features[f"{window_name}_min"] = rolling.min().iloc[:, :-1].values
            features[f"{window_name}_max"] = rolling.max().iloc[:, :-1].values
            
            # 变化率特征
            features[f"{window_name}_change"] = data.iloc[:, :-1].pct_change(window).values
            features[f"{window_name}_momentum"] = (data.iloc[:, :-1] / rolling.mean().iloc[:, :-1] - 1).values
            
            # 波动性特征
            features[f"{window_name}_volatility"] = rolling.std().iloc[:, :-1].values / rolling.mean().iloc[:, :-1].values
            
            # 趋势特征
            features[f"{window_name}_trend"] = data.iloc[:, :-1].diff(window).values
            
        return features
    
    def select_optimal_vt_features(self, X_vt: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str], 
                                 selection_ratio: float = 0.3) -> Tuple[List[str], List[int]]:
        """选择最优V_t特征"""
        self.logger.info(f"选择最优V_t特征，目标比例: {selection_ratio}")
        
        # 移除NaN值
        mask = ~np.isnan(X_vt).any(axis=1) & ~np.isnan(y)
        X_clean = X_vt[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            self.logger.warning("有效样本过少，使用所有特征")
            return feature_names, list(range(len(feature_names)))
        
        # 计算重要性
        stat_importance = self.feature_importance_analyzer.analyze_statistical_importance(
            X_clean, y_clean, feature_names
        )
        model_importance = self.feature_importance_analyzer.analyze_model_based_importance(
            X_clean, y_clean, feature_names
        )
        
        # 合并重要性分数
        all_importance = {**stat_importance, **model_importance}
        composite_importance = self.feature_importance_analyzer.compute_composite_importance(
            all_importance, feature_names
        )
        
        # 选择top特征
        n_select = max(1, int(len(feature_names) * selection_ratio))
        sorted_features = sorted(composite_importance.items(), key=lambda x: x[1], reverse=True)
        
        selected_names = [name for name, _ in sorted_features[:n_select]]
        selected_indices = [feature_names.index(name) for name in selected_names]
        
        self.logger.info(f"V_t特征选择完成，从{len(feature_names)}中选择了{len(selected_names)}个")
        
        return selected_names, selected_indices


class VsFeatureSelector:
    """
    V_s静态特征选择器 - 专门处理静态上下文特征
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.selected_features = []
        self.feature_importance_analyzer = FeatureImportanceAnalyzer(logger)
        
    def extract_static_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取静态上下文特征"""
        self.logger.info("提取V_s静态特征...")
        
        features = {}
        
        # 设备额定特征 (如果存在相关列)
        equipment_columns = [col for col in data.columns if any(keyword in col.lower() 
                           for keyword in ['rating', 'capacity', 'power', 'efficiency'])]
        
        if equipment_columns:
            for col in equipment_columns:
                # 设备额定值统计
                features[f"equipment_{col}_mean"] = np.full(len(data), data[col].mean())
                features[f"equipment_{col}_std"] = np.full(len(data), data[col].std())
        
        # 时间编码特征
        if 'timestamp' in data.columns or data.index.name == 'timestamp':
            if hasattr(data.index, 'hour'):
                features['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
                features['day_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
                features['day_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
                features['weekday'] = data.index.weekday.values
        
        # 环境条件特征
        env_columns = [col for col in data.columns if any(keyword in col.lower() 
                      for keyword in ['temp', 'pressure', 'humidity', 'wind'])]
        
        for col in env_columns:
            if col in data.columns:
                # 环境条件分位数特征
                features[f"env_{col}_q25"] = np.full(len(data), data[col].quantile(0.25))
                features[f"env_{col}_q75"] = np.full(len(data), data[col].quantile(0.75))
                features[f"env_{col}_range"] = np.full(len(data), data[col].max() - data[col].min())
        
        # 操作模式特征 (基于数据分布)
        for col in data.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                # 创建操作模式指标
                values = data[col].values
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                
                try:
                    mode_labels = kmeans.fit_predict(values.reshape(-1, 1))
                    features[f"mode_{col}"] = mode_labels.astype(float)
                except:
                    features[f"mode_{col}"] = np.zeros(len(data))
        
        return features
    
    def select_optimal_vs_features(self, X_vs: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str],
                                 selection_ratio: float = 0.4) -> Tuple[List[str], List[int]]:
        """选择最优V_s特征"""
        self.logger.info(f"选择最优V_s特征，目标比例: {selection_ratio}")
        
        # 移除常数特征和NaN特征
        valid_features = []
        valid_indices = []
        
        for i, feature_name in enumerate(feature_names):
            feature_values = X_vs[:, i]
            if not np.isnan(feature_values).all() and np.std(feature_values) > 1e-8:
                valid_features.append(feature_name)
                valid_indices.append(i)
        
        if not valid_features:
            self.logger.warning("无有效V_s特征")
            return [], []
        
        X_valid = X_vs[:, valid_indices]
        
        # 移除NaN样本
        mask = ~np.isnan(X_valid).any(axis=1) & ~np.isnan(y)
        X_clean = X_valid[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 50:
            self.logger.warning("V_s有效样本过少，使用所有有效特征")
            return valid_features, valid_indices
        
        # 计算重要性
        stat_importance = self.feature_importance_analyzer.analyze_statistical_importance(
            X_clean, y_clean, valid_features
        )
        model_importance = self.feature_importance_analyzer.analyze_model_based_importance(
            X_clean, y_clean, valid_features
        )
        
        # 合并重要性分数
        all_importance = {**stat_importance, **model_importance}
        composite_importance = self.feature_importance_analyzer.compute_composite_importance(
            all_importance, valid_features
        )
        
        # 选择top特征
        n_select = max(1, int(len(valid_features) * selection_ratio))
        sorted_features = sorted(composite_importance.items(), key=lambda x: x[1], reverse=True)
        
        selected_names = [name for name, _ in sorted_features[:n_select]]
        selected_indices = [valid_indices[valid_features.index(name)] for name in selected_names]
        
        self.logger.info(f"V_s特征选择完成，从{len(valid_features)}个有效特征中选择了{len(selected_names)}个")
        
        return selected_names, selected_indices


class DualChannelFeatureSelector:
    """
    双通道特征选择系统 - V_t和V_s特征的统一管理
    """
    
    def __init__(self, 
                 vt_selection_ratio: float = 0.3,
                 vs_selection_ratio: float = 0.4,
                 fusion_method: str = 'concatenate',
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.vt_selector = VtFeatureSelector(logger=logger)
        self.vs_selector = VsFeatureSelector(logger=logger)
        
        self.vt_selection_ratio = vt_selection_ratio
        self.vs_selection_ratio = vs_selection_ratio
        self.fusion_method = fusion_method
        
        # 选择结果
        self.vt_selected_features = []
        self.vs_selected_features = []
        self.vt_selected_indices = []
        self.vs_selected_indices = []
        
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, target_column: str) -> 'DualChannelFeatureSelector':
        """训练双通道特征选择器"""
        self.logger.info("开始双通道特征选择...")
        
        start_time = time.time()
        
        # 分离目标变量
        y = data[target_column].values
        feature_data = data.drop(columns=[target_column])
        
        # 第一阶段：V_t动态特征提取和选择
        self.logger.info("第一阶段: V_t动态特征处理")
        vt_features = self.vt_selector.extract_temporal_features(data, target_column)
        
        # 组装V_t特征矩阵
        vt_feature_names = []
        vt_matrices = []
        
        for feature_name, feature_matrix in vt_features.items():
            if feature_matrix.size > 0:
                # 处理多列特征
                if len(feature_matrix.shape) == 2:
                    for col_idx in range(feature_matrix.shape[1]):
                        col_name = f"{feature_name}_col{col_idx}"
                        vt_feature_names.append(col_name)
                        vt_matrices.append(feature_matrix[:, col_idx])
                else:
                    vt_feature_names.append(feature_name)
                    vt_matrices.append(feature_matrix)
        
        if vt_matrices:
            X_vt = np.column_stack(vt_matrices)
            self.vt_selected_features, self.vt_selected_indices = self.vt_selector.select_optimal_vt_features(
                X_vt, y, vt_feature_names, self.vt_selection_ratio
            )
        
        # 第二阶段：V_s静态特征提取和选择
        self.logger.info("第二阶段: V_s静态特征处理")
        vs_features = self.vs_selector.extract_static_features(data)
        
        # 组装V_s特征矩阵
        vs_feature_names = []
        vs_matrices = []
        
        for feature_name, feature_matrix in vs_features.items():
            if feature_matrix.size > 0:
                vs_feature_names.append(feature_name)
                vs_matrices.append(feature_matrix)
        
        if vs_matrices:
            X_vs = np.column_stack(vs_matrices)
            self.vs_selected_features, self.vs_selected_indices = self.vs_selector.select_optimal_vs_features(
                X_vs, y, vs_feature_names, self.vs_selection_ratio
            )
        
        training_time = time.time() - start_time
        self.fitted = True
        
        self.logger.info(f"双通道特征选择完成，耗时: {training_time:.2f}s")
        self.logger.info(f"V_t选择: {len(self.vt_selected_features)}, V_s选择: {len(self.vs_selected_features)}")
        
        return self
    
    def transform(self, data: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """转换数据到选择的特征空间"""
        if not self.fitted:
            raise ValueError("特征选择器未训练，请先调用fit方法")
        
        # 重新提取特征
        if target_column:
            data_for_vt = data
        else:
            data_for_vt = data
        
        # V_t特征提取
        vt_features = self.vt_selector.extract_temporal_features(data_for_vt, target_column or 'dummy')
        
        # 组装V_t特征
        vt_matrices = []
        vt_feature_names = []
        
        for feature_name, feature_matrix in vt_features.items():
            if feature_matrix.size > 0:
                if len(feature_matrix.shape) == 2:
                    for col_idx in range(feature_matrix.shape[1]):
                        col_name = f"{feature_name}_col{col_idx}"
                        vt_feature_names.append(col_name)
                        vt_matrices.append(feature_matrix[:, col_idx])
                else:
                    vt_feature_names.append(feature_name)
                    vt_matrices.append(feature_matrix)
        
        selected_vt_data = []
        if vt_matrices and self.vt_selected_indices:
            X_vt = np.column_stack(vt_matrices)
            selected_vt_data = X_vt[:, self.vt_selected_indices]
        
        # V_s特征提取
        vs_features = self.vs_selector.extract_static_features(data)
        
        # 组装V_s特征
        vs_matrices = []
        for feature_name, feature_matrix in vs_features.items():
            if feature_matrix.size > 0:
                vs_matrices.append(feature_matrix)
        
        selected_vs_data = []
        if vs_matrices and self.vs_selected_indices:
            X_vs = np.column_stack(vs_matrices)
            selected_vs_data = X_vs[:, self.vs_selected_indices]
        
        # 特征融合
        if len(selected_vt_data) > 0 and len(selected_vs_data) > 0:
            if self.fusion_method == 'concatenate':
                return np.column_stack([selected_vt_data, selected_vs_data])
            else:
                # 其他融合方法可以在这里添加
                return np.column_stack([selected_vt_data, selected_vs_data])
        elif len(selected_vt_data) > 0:
            return selected_vt_data
        elif len(selected_vs_data) > 0:
            return selected_vs_data
        else:
            raise ValueError("没有选择到有效特征")
    
    def get_selected_feature_names(self) -> Dict[str, List[str]]:
        """获取选择的特征名称"""
        return {
            'vt_features': self.vt_selected_features,
            'vs_features': self.vs_selected_features,
            'total_features': len(self.vt_selected_features) + len(self.vs_selected_features)
        }
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """获取特征重要性报告"""
        return {
            'vt_selection_ratio': self.vt_selection_ratio,
            'vs_selection_ratio': self.vs_selection_ratio,
            'vt_selected_count': len(self.vt_selected_features),
            'vs_selected_count': len(self.vs_selected_features),
            'fusion_method': self.fusion_method,
            'vt_features': self.vt_selected_features,
            'vs_features': self.vs_selected_features
        }


# 便捷函数
def create_dual_channel_selector(vt_ratio: float = 0.3, 
                                vs_ratio: float = 0.4,
                                logger: Optional[logging.Logger] = None) -> DualChannelFeatureSelector:
    """创建双通道特征选择器"""
    return DualChannelFeatureSelector(
        vt_selection_ratio=vt_ratio,
        vs_selection_ratio=vs_ratio,
        logger=logger
    )