"""
LNG评估器 - 模型性能评估
基于技术路线的标准评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import yaml
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .metrics import calculate_lng_metrics, MetricsReport


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 技术路线要求的指标阈值
    r2_threshold: float = 0.75
    cv_rmse_threshold: float = 6.0  # %
    nmbe_range: Tuple[float, float] = (-0.6, 0.6)  # %
    picp_90_range: Tuple[float, float] = (0.85, 0.95)
    
    # 评估设置
    confidence_level: float = 0.90
    bootstrap_samples: int = 1000
    save_predictions: bool = True
    save_plots: bool = True
    
    def meets_requirements(self, metrics: Dict[str, float]) -> bool:
        """检查指标是否满足技术路线要求"""
        checks = [
            metrics.get('r2_score', 0) >= self.r2_threshold,
            metrics.get('cv_rmse_percent', 100) <= self.cv_rmse_threshold,
            self.nmbe_range[0] <= metrics.get('nmbe_percent', 0) <= self.nmbe_range[1],
        ]
        
        if 'picp_90' in metrics:
            checks.append(
                self.picp_90_range[0] <= metrics['picp_90'] <= self.picp_90_range[1]
            )
        
        return all(checks)


class LNGEvaluator:
    """LNG模型评估器"""
    
    def __init__(self, config: EvaluationConfig = None, logger: Optional[logging.Logger] = None):
        self.config = config or EvaluationConfig()
        self.logger = logger or logging.getLogger(__name__)
        
    def evaluate_model(self, 
                      model: Any,
                      X_test: np.ndarray,
                      y_test: np.ndarray, 
                      model_type: str = 'transformer',
                      preprocessor: Any = None,
                      target_column: str = None) -> MetricsReport:
        """
        完整模型评估
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            model_type: 模型类型
            preprocessor: 预处理器（用于反归一化）
            target_column: 目标列名
            
        Returns:
            评估结果报告
        """
        
        self.logger.info("开始模型评估...")
        
        # 1. 获取预测
        predictions = self._get_predictions(model, X_test, model_type)
        
        # 2. 反归一化（如果需要）
        pred_original, y_original = self._inverse_transform(
            predictions, y_test, preprocessor, target_column
        )
        
        # 3. 计算基础指标
        metrics = calculate_lng_metrics(y_original, pred_original, self.config)
        
        # 4. 检查是否满足要求
        meets_requirements = self.config.meets_requirements(metrics)
        
        # 5. 创建报告
        report = MetricsReport(
            metrics=metrics,
            predictions=pred_original,
            targets=y_original,
            meets_requirements=meets_requirements,
            config=self.config
        )
        
        self.logger.info(f"评估完成 - R²: {metrics['r2_score']:.4f}, "
                        f"满足要求: {'是' if meets_requirements else '否'}")
        
        return report
    
    def _get_predictions(self, model: Any, X_test: np.ndarray, model_type: str) -> np.ndarray:
        """获取模型预测"""
        
        if model_type == 'transformer':
            # PyTorch模型
            model.eval()
            with torch.no_grad():
                if isinstance(X_test, tuple):
                    # 动态+静态特征
                    X_dynamic, X_static = X_test
                    X_dynamic_tensor = torch.FloatTensor(X_dynamic)
                    X_static_tensor = torch.FloatTensor(X_static)
                    
                    device = next(model.parameters()).device
                    X_dynamic_tensor = X_dynamic_tensor.to(device)
                    X_static_tensor = X_static_tensor.to(device)
                    
                    predictions = model(X_dynamic_tensor, X_static_tensor).cpu().numpy()
                else:
                    # 单一特征
                    X_tensor = torch.FloatTensor(X_test)
                    device = next(model.parameters()).device
                    X_tensor = X_tensor.to(device)
                    
                    predictions = model(X_tensor).cpu().numpy()
                
                predictions = predictions.squeeze()
                
        else:
            # Sklearn模型
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            elif hasattr(model, 'predict_all'):  # BaselineModels
                predictions_dict = model.predict_all(X_test)
                # 使用第一个模型的预测（可以改进为最佳模型）
                predictions = list(predictions_dict.values())[0]
            else:
                raise ValueError("模型没有predict方法")
        
        return predictions
    
    def _inverse_transform(self, predictions: np.ndarray, targets: np.ndarray,
                          preprocessor: Any, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """反归一化"""
        
        if preprocessor is not None and target_column is not None:
            try:
                pred_original = preprocessor.inverse_transform_target(target_column, predictions)
                y_original = preprocessor.inverse_transform_target(target_column, targets)
                return pred_original, y_original
            except Exception as e:
                self.logger.warning(f"反归一化失败: {e}")
        
        return predictions, targets
    
    def compare_models(self, 
                      models_dict: Dict[str, Any],
                      X_test: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
                      y_test: np.ndarray,
                      model_types: Dict[str, str] = None) -> pd.DataFrame:
        """
        比较多个模型性能
        
        Args:
            models_dict: 模型字典 {name: model}
            X_test: 测试特征
            y_test: 测试目标
            model_types: 模型类型字典 {name: type}
            
        Returns:
            比较结果DataFrame
        """
        
        self.logger.info(f"比较 {len(models_dict)} 个模型...")
        
        results = []
        
        for name, model in models_dict.items():
            model_type = model_types.get(name, 'transformer') if model_types else 'transformer'
            
            try:
                predictions = self._get_predictions(model, X_test, model_type)
                metrics = calculate_lng_metrics(y_test, predictions, self.config)
                meets_req = self.config.meets_requirements(metrics)
                
                result = {
                    'model_name': name,
                    'model_type': model_type,
                    'meets_requirements': meets_req,
                    **metrics
                }
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"评估模型 {name} 失败: {e}")
                
        df_results = pd.DataFrame(results)
        
        # 按R²降序排列
        if not df_results.empty:
            df_results = df_results.sort_values('r2_score', ascending=False)
        
        return df_results
    
    def save_report(self, report: MetricsReport, output_dir: str, filename: str = None):
        """保存评估报告"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = 'evaluation_report.yaml'
        
        # 转换为可序列化的格式
        report_dict = {
            'metrics': report.metrics,
            'meets_requirements': report.meets_requirements,
            'config': {
                'r2_threshold': self.config.r2_threshold,
                'cv_rmse_threshold': self.config.cv_rmse_threshold,
                'nmbe_range': self.config.nmbe_range,
                'picp_90_range': self.config.picp_90_range
            },
            'summary': {
                'n_samples': len(report.targets),
                'predictions_mean': float(np.mean(report.predictions)),
                'targets_mean': float(np.mean(report.targets)),
                'status': 'PASS' if report.meets_requirements else 'FAIL'
            }
        }
        
        # 保存YAML报告
        with open(output_path / filename, 'w', encoding='utf-8') as f:
            yaml.dump(report_dict, f, default_flow_style=False, allow_unicode=True)
        
        # 如果配置允许，保存预测结果
        if self.config.save_predictions:
            predictions_df = pd.DataFrame({
                'targets': report.targets,
                'predictions': report.predictions,
                'residuals': report.targets - report.predictions
            })
            predictions_df.to_csv(output_path / 'predictions.csv', index=False)
        
        self.logger.info(f"评估报告已保存到: {output_path / filename}")


def quick_evaluate(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                  model_type: str = 'transformer') -> Dict[str, float]:
    """快速评估接口"""
    
    evaluator = LNGEvaluator()
    report = evaluator.evaluate_model(model, X_test, y_test, model_type)
    
    return report.metrics