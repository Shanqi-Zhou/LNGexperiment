"""
Purged Walk-Forward验证系统 - 优化方案时序验证核心
专门为时序数据设计的严格验证框架，防止数据泄漏
实现滚动窗口、纯化间隙、禁用期等高级时序验证技术
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, Tuple, List, Generator, Callable
import logging
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import BaseCrossValidator
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


@dataclass
class TimeSeriesValidationConfig:
    """时序验证配置"""
    train_size: int = 1000          # 训练窗口大小
    test_size: int = 200            # 测试窗口大小
    purge_size: int = 50           # 纯化间隙大小
    embargo_size: int = 10         # 禁用期大小
    step_size: int = 100           # 滚动步长
    min_train_size: int = 500      # 最小训练集大小
    max_splits: int = 10           # 最大分割数
    require_full_test: bool = True  # 是否要求完整测试集


class PurgedTimeSeriesCV(BaseCrossValidator):
    """
    纯化时序交叉验证器
    实现Marcos López de Prado的纯化Walk-Forward验证
    """
    
    def __init__(self, config: TimeSeriesValidationConfig, 
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.splits_generated = []
        
    def _purge_overlaps(self, train_indices: np.ndarray, 
                       test_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """纯化重叠数据"""
        if len(train_indices) == 0 or len(test_indices) == 0:
            return train_indices, test_indices
        
        # 计算需要从训练集中移除的索引
        test_start = test_indices.min()
        purge_start = test_start - self.config.purge_size
        
        # 从训练集中移除与测试集过近的数据点
        valid_train_mask = train_indices < purge_start
        purged_train_indices = train_indices[valid_train_mask]
        
        return purged_train_indices, test_indices
    
    def _apply_embargo(self, test_indices: np.ndarray, 
                      total_length: int) -> np.ndarray:
        """应用禁用期"""
        if len(test_indices) == 0:
            return test_indices
        
        # 从测试集开始移除embargo_size个点
        test_start = test_indices.min()
        embargo_end = min(test_start + self.config.embargo_size, total_length - 1)
        
        # 保留禁用期之后的测试数据
        valid_test_mask = test_indices >= embargo_end
        embargoed_test_indices = test_indices[valid_test_mask]
        
        return embargoed_test_indices
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """生成训练/测试分割"""
        n_samples = len(X)
        
        self.logger.info(f"开始Purged Walk-Forward验证，总样本数: {n_samples}")
        
        splits_count = 0
        current_start = 0
        
        while (current_start + self.config.min_train_size + self.config.test_size < n_samples 
               and splits_count < self.config.max_splits):
            
            # 定义训练集
            train_end = current_start + self.config.train_size
            train_end = min(train_end, n_samples - self.config.test_size - self.config.purge_size)
            
            if train_end - current_start < self.config.min_train_size:
                break
            
            train_indices = np.arange(current_start, train_end)
            
            # 定义测试集
            test_start = train_end + self.config.purge_size
            test_end = test_start + self.config.test_size
            test_end = min(test_end, n_samples)
            
            if test_end - test_start < self.config.test_size and self.config.require_full_test:
                break
            
            test_indices = np.arange(test_start, test_end)
            
            # 纯化重叠数据
            purged_train_indices, purged_test_indices = self._purge_overlaps(
                train_indices, test_indices
            )
            
            # 应用禁用期
            final_test_indices = self._apply_embargo(purged_test_indices, n_samples)
            
            # 检查最终大小
            if (len(purged_train_indices) >= self.config.min_train_size and 
                len(final_test_indices) > 0):
                
                split_info = {
                    'split_id': splits_count,
                    'train_start': purged_train_indices.min(),
                    'train_end': purged_train_indices.max(),
                    'train_size': len(purged_train_indices),
                    'test_start': final_test_indices.min(),
                    'test_end': final_test_indices.max(),
                    'test_size': len(final_test_indices),
                    'purge_applied': len(train_indices) - len(purged_train_indices),
                    'embargo_applied': len(purged_test_indices) - len(final_test_indices)
                }
                
                self.splits_generated.append(split_info)
                
                self.logger.debug(f"分割 {splits_count}: 训练集 {len(purged_train_indices)}, "
                                f"测试集 {len(final_test_indices)}")
                
                yield purged_train_indices, final_test_indices
                
                splits_count += 1
            
            # 滚动到下一个窗口
            current_start += self.config.step_size
        
        self.logger.info(f"Purged Walk-Forward验证完成，生成 {splits_count} 个分割")
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """获取分割数量"""
        if X is None:
            return self.config.max_splits
        
        n_samples = len(X)
        max_possible_splits = max(0, (n_samples - self.config.min_train_size - self.config.test_size - self.config.purge_size) 
                                 // self.config.step_size)
        return min(max_possible_splits, self.config.max_splits)
    
    def get_splits_info(self) -> List[Dict[str, Any]]:
        """获取分割信息"""
        return self.splits_generated


class TimeSeriesModelValidator:
    """
    时序模型验证器 - 统一的验证框架
    """
    
    def __init__(self, config: TimeSeriesValidationConfig,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.cv = PurgedTimeSeriesCV(config, logger)
        self.validation_results = []
        
    def validate_model(self, model, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        验证模型性能
        
        Args:
            model: 要验证的模型（需要fit和predict方法）
            X: 特征数据
            y: 目标变量
            fit_params: 模型训练参数
        """
        self.logger.info("开始时序模型验证...")
        
        start_time = time.time()
        fit_params = fit_params or {}
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            fold_start_time = time.time()
            
            # 划分数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # 训练模型
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train, **fit_params)
                
                # 预测
                y_pred_train = model_clone.predict(X_train)
                y_pred_test = model_clone.predict(X_test)
                
                # 计算指标
                fold_metrics = self._calculate_metrics(
                    y_train, y_pred_train, y_test, y_pred_test
                )
                
                fold_metrics.update({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_start_idx': train_idx.min(),
                    'train_end_idx': train_idx.max(),
                    'test_start_idx': test_idx.min(),
                    'test_end_idx': test_idx.max(),
                    'fold_time': time.time() - fold_start_time
                })
                
                fold_results.append(fold_metrics)
                
                self.logger.info(f"折 {fold}: 测试R²={fold_metrics['test_r2']:.4f}, "
                               f"RMSE={fold_metrics['test_rmse']:.4f}")
                
            except Exception as e:
                self.logger.error(f"折 {fold} 验证失败: {e}")
                continue
        
        # 汇总结果
        if fold_results:
            summary_results = self._summarize_results(fold_results)
            summary_results['total_time'] = time.time() - start_time
            summary_results['splits_info'] = self.cv.get_splits_info()
            
            self.validation_results.append(summary_results)
            
            self.logger.info(f"验证完成: 平均测试R²={summary_results['mean_test_r2']:.4f} "
                           f"(±{summary_results['std_test_r2']:.4f})")
            
            return summary_results
        else:
            self.logger.error("所有验证折都失败了")
            return {}
    
    def _clone_model(self, model):
        """克隆模型"""
        try:
            # 尝试使用sklearn的clone
            from sklearn.base import clone
            return clone(model)
        except:
            # 如果不是sklearn模型，尝试深拷贝
            import copy
            try:
                return copy.deepcopy(model)
            except:
                # 最后返回原模型（风险较高）
                self.logger.warning("无法克隆模型，使用原模型")
                return model
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray) -> Dict[str, float]:
        """计算验证指标"""
        metrics = {}
        
        # 训练集指标
        metrics['train_r2'] = r2_score(y_train, y_pred_train)
        metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        
        # 测试集指标
        metrics['test_r2'] = r2_score(y_test, y_pred_test)
        metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
        
        # NMBE (归一化平均偏差误差)
        metrics['train_nmbe'] = np.mean(y_pred_train - y_train) / np.mean(y_train) * 100
        metrics['test_nmbe'] = np.mean(y_pred_test - y_test) / np.mean(y_test) * 100
        
        # CV(RMSE) (变异系数)
        metrics['train_cv_rmse'] = metrics['train_rmse'] / np.mean(y_train) * 100
        metrics['test_cv_rmse'] = metrics['test_rmse'] / np.mean(y_test) * 100
        
        return metrics
    
    def _summarize_results(self, fold_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """汇总验证结果"""
        if not fold_results:
            return {}
        
        # 提取数值指标
        metrics_to_summarize = [
            'train_r2', 'test_r2', 'train_rmse', 'test_rmse',
            'train_mae', 'test_mae', 'train_nmbe', 'test_nmbe',
            'train_cv_rmse', 'test_cv_rmse'
        ]
        
        summary = {'fold_results': fold_results}
        
        for metric in metrics_to_summarize:
            values = [fold[metric] for fold in fold_results if metric in fold]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)
        
        # 稳定性指标
        test_r2_values = [fold['test_r2'] for fold in fold_results]
        if len(test_r2_values) > 1:
            summary['stability_score'] = 1 - (np.std(test_r2_values) / np.mean(test_r2_values))
        else:
            summary['stability_score'] = 1.0
        
        return summary
    
    def compare_models(self, models_dict: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                      fit_params_dict: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """比较多个模型"""
        self.logger.info(f"开始比较 {len(models_dict)} 个模型...")
        
        results = []
        fit_params_dict = fit_params_dict or {}
        
        for model_name, model in models_dict.items():
            self.logger.info(f"验证模型: {model_name}")
            
            fit_params = fit_params_dict.get(model_name, {})
            model_results = self.validate_model(model, X, y, fit_params)
            
            if model_results:
                model_summary = {
                    'model_name': model_name,
                    'mean_test_r2': model_results.get('mean_test_r2', 0),
                    'std_test_r2': model_results.get('std_test_r2', 0),
                    'mean_test_rmse': model_results.get('mean_test_rmse', float('inf')),
                    'std_test_rmse': model_results.get('std_test_rmse', 0),
                    'mean_test_nmbe': model_results.get('mean_test_nmbe', 0),
                    'mean_test_cv_rmse': model_results.get('mean_test_cv_rmse', 0),
                    'stability_score': model_results.get('stability_score', 0),
                    'n_folds': len(model_results.get('fold_results', []))
                }
                results.append(model_summary)
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('mean_test_r2', ascending=False)
            
            self.logger.info("模型比较完成")
            self.logger.info(f"最佳模型: {df_results.iloc[0]['model_name']} "
                           f"(R²={df_results.iloc[0]['mean_test_r2']:.4f})")
            
            return df_results
        else:
            self.logger.error("所有模型验证都失败了")
            return pd.DataFrame()
    
    def plot_validation_results(self, results_df: pd.DataFrame, 
                               save_path: Optional[str] = None):
        """绘制验证结果"""
        if results_df.empty:
            self.logger.warning("没有结果可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² 分数
        ax1 = axes[0, 0]
        ax1.bar(results_df['model_name'], results_df['mean_test_r2'], 
                yerr=results_df['std_test_r2'], capsize=5)
        ax1.set_title('Test R² Score')
        ax1.set_ylabel('R²')
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE
        ax2 = axes[0, 1]
        ax2.bar(results_df['model_name'], results_df['mean_test_rmse'],
                yerr=results_df['std_test_rmse'], capsize=5)
        ax2.set_title('Test RMSE')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # NMBE
        ax3 = axes[1, 0]
        ax3.bar(results_df['model_name'], results_df['mean_test_nmbe'])
        ax3.set_title('Test NMBE (%)')
        ax3.set_ylabel('NMBE (%)')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.tick_params(axis='x', rotation=45)
        
        # 稳定性分数
        ax4 = axes[1, 1]
        ax4.bar(results_df['model_name'], results_df['stability_score'])
        ax4.set_title('Stability Score')
        ax4.set_ylabel('Stability')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"验证结果图已保存: {save_path}")
        
        plt.show()


class TimeSeriesBacktester:
    """
    时序回测器 - 专门用于时序预测的回测
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.backtest_results = []
    
    def backtest_model(self, model, X: np.ndarray, y: np.ndarray, 
                      timestamps: Optional[pd.DatetimeIndex] = None,
                      initial_train_size: int = 1000,
                      refit_frequency: int = 100) -> Dict[str, Any]:
        """
        模型回测
        
        Args:
            model: 预测模型
            X: 特征数据
            y: 目标变量
            timestamps: 时间戳（可选）
            initial_train_size: 初始训练集大小
            refit_frequency: 重新训练频率
        """
        self.logger.info("开始时序模型回测...")
        
        n_samples = len(X)
        predictions = []
        actuals = []
        prediction_timestamps = []
        refit_points = []
        
        # 初始训练
        model.fit(X[:initial_train_size], y[:initial_train_size])
        last_refit = initial_train_size
        
        # 滚动预测
        for i in range(initial_train_size, n_samples):
            # 预测
            pred = model.predict(X[i:i+1])[0]
            predictions.append(pred)
            actuals.append(y[i])
            
            if timestamps is not None:
                prediction_timestamps.append(timestamps[i])
            
            # 检查是否需要重新训练
            if i - last_refit >= refit_frequency:
                # 重新训练模型
                train_end = i
                train_start = max(0, train_end - initial_train_size)
                
                model.fit(X[train_start:train_end], y[train_start:train_end])
                last_refit = i
                refit_points.append(i)
                
                self.logger.debug(f"在步骤 {i} 重新训练模型")
        
        # 计算回测指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        backtest_metrics = self._calculate_backtest_metrics(actuals, predictions)
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': prediction_timestamps,
            'refit_points': refit_points,
            'metrics': backtest_metrics,
            'n_predictions': len(predictions),
            'refit_frequency': refit_frequency
        }
        
        self.backtest_results.append(results)
        
        self.logger.info(f"回测完成: R²={backtest_metrics['r2']:.4f}, "
                       f"RMSE={backtest_metrics['rmse']:.4f}")
        
        return results
    
    def _calculate_backtest_metrics(self, actuals: np.ndarray, 
                                   predictions: np.ndarray) -> Dict[str, float]:
        """计算回测指标"""
        return {
            'r2': r2_score(actuals, predictions),
            'mse': mean_squared_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mae': mean_absolute_error(actuals, predictions),
            'nmbe_percent': np.mean(predictions - actuals) / np.mean(actuals) * 100,
            'cv_rmse_percent': np.sqrt(mean_squared_error(actuals, predictions)) / np.mean(actuals) * 100
        }


# 便捷函数
def create_purged_cv(train_size: int = 1000, test_size: int = 200, 
                    purge_size: int = 50, step_size: int = 100,
                    logger: Optional[logging.Logger] = None) -> PurgedTimeSeriesCV:
    """创建纯化时序交叉验证器"""
    config = TimeSeriesValidationConfig(
        train_size=train_size,
        test_size=test_size,
        purge_size=purge_size,
        step_size=step_size
    )
    return PurgedTimeSeriesCV(config, logger)


def validate_timeseries_model(model, X: np.ndarray, y: np.ndarray,
                             config: Optional[TimeSeriesValidationConfig] = None,
                             logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """时序模型验证便捷函数"""
    config = config or TimeSeriesValidationConfig()
    validator = TimeSeriesModelValidator(config, logger)
    return validator.validate_model(model, X, y)