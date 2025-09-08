"""
评估指标计算
基于技术路线的标准指标
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@dataclass
class MetricsReport:
    """指标报告"""
    metrics: Dict[str, float]
    predictions: np.ndarray
    targets: np.ndarray
    meets_requirements: bool
    config: Any


def calculate_lng_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         config: Any = None) -> Dict[str, float]:
    """
    计算LNG技术路线要求的评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        config: 评估配置
        
    Returns:
        指标字典
    """
    
    # 基础回归指标
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 技术路线要求的指标
    # 1. NMBE (Normalized Mean Bias Error)
    nmbe = np.mean(y_pred - y_true) / np.mean(y_true) * 100
    
    # 2. CV(RMSE) (Coefficient of Variation of RMSE)  
    cv_rmse = rmse / np.mean(y_true) * 100
    
    # 3. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 4. 相对误差统计
    relative_errors = (y_pred - y_true) / y_true * 100
    re_mean = np.mean(relative_errors)
    re_std = np.std(relative_errors)
    
    # 5. 预测区间覆盖概率 (PICP) - 需要不确定性信息
    # 这里计算简化版本，基于残差分布
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    
    # 90%置信区间
    z_90 = 1.645  # 90% 单侧
    lower_bound = y_pred - z_90 * residual_std
    upper_bound = y_pred + z_90 * residual_std
    picp_90 = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    
    # 95%置信区间
    z_95 = 1.96  # 95% 单侧
    lower_bound_95 = y_pred - z_95 * residual_std
    upper_bound_95 = y_pred + z_95 * residual_std
    picp_95 = np.mean((y_true >= lower_bound_95) & (y_true <= upper_bound_95))
    
    metrics = {
        # 基础指标
        'r2_score': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        
        # 技术路线指标
        'nmbe_percent': float(nmbe),
        'cv_rmse_percent': float(cv_rmse),
        'mape_percent': float(mape),
        
        # 相对误差统计
        'relative_error_mean': float(re_mean),
        'relative_error_std': float(re_std),
        
        # 预测区间覆盖
        'picp_90': float(picp_90),
        'picp_95': float(picp_95),
        
        # 附加统计
        'predictions_mean': float(np.mean(y_pred)),
        'predictions_std': float(np.std(y_pred)),
        'targets_mean': float(np.mean(y_true)),
        'targets_std': float(np.std(y_true)),
        'residuals_mean': float(np.mean(residuals)),
        'residuals_std': float(residual_std),
        
        # 分布统计
        'predictions_min': float(np.min(y_pred)),
        'predictions_max': float(np.max(y_pred)),
        'targets_min': float(np.min(y_true)),
        'targets_max': float(np.max(y_true))
    }
    
    return metrics


def calculate_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算高级评估指标（包含不确定性）
    
    Args:
        y_true: 真实值
        y_pred: 预测均值
        y_pred_std: 预测标准差（不确定性）
        
    Returns:
        高级指标字典
    """
    
    metrics = {}
    
    if y_pred_std is not None:
        # 准确的PICP计算
        z_90 = 1.645
        z_95 = 1.96
        
        lower_90 = y_pred - z_90 * y_pred_std
        upper_90 = y_pred + z_90 * y_pred_std
        picp_90 = np.mean((y_true >= lower_90) & (y_true <= upper_90))
        
        lower_95 = y_pred - z_95 * y_pred_std
        upper_95 = y_pred + z_95 * y_pred_std
        picp_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))
        
        # 平均区间宽度
        interval_width_90 = np.mean(upper_90 - lower_90)
        interval_width_95 = np.mean(upper_95 - lower_95)
        
        # 归一化平均区间宽度
        normalized_width_90 = interval_width_90 / np.mean(y_true)
        normalized_width_95 = interval_width_95 / np.mean(y_true)
        
        # 区间评分 (Interval Score)
        alpha_90 = 0.1
        alpha_95 = 0.05
        
        interval_score_90 = _calculate_interval_score(y_true, lower_90, upper_90, alpha_90)
        interval_score_95 = _calculate_interval_score(y_true, lower_95, upper_95, alpha_95)
        
        metrics.update({
            'picp_90_accurate': float(picp_90),
            'picp_95_accurate': float(picp_95),
            'interval_width_90': float(interval_width_90),
            'interval_width_95': float(interval_width_95),
            'normalized_width_90': float(normalized_width_90),
            'normalized_width_95': float(normalized_width_95),
            'interval_score_90': float(np.mean(interval_score_90)),
            'interval_score_95': float(np.mean(interval_score_95)),
            'uncertainty_mean': float(np.mean(y_pred_std)),
            'uncertainty_std': float(np.std(y_pred_std))
        })
    
    return metrics


def _calculate_interval_score(y_true: np.ndarray, lower: np.ndarray, 
                             upper: np.ndarray, alpha: float) -> np.ndarray:
    """计算区间评分"""
    
    width = upper - lower
    lower_penalty = 2 * alpha * np.maximum(0, lower - y_true)
    upper_penalty = 2 * alpha * np.maximum(0, y_true - upper)
    
    return width + lower_penalty + upper_penalty


def calculate_technical_compliance(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    检查技术路线合规性
    
    技术路线要求:
    - R² ≥ 0.75
    - CV(RMSE) ≤ 6%
    - NMBE ∈ [-0.6%, 0.6%]
    - PICP@90% ∈ [0.85, 0.95]
    """
    
    requirements = {
        'r2_score': {'threshold': 0.75, 'operator': '>=', 'unit': ''},
        'cv_rmse_percent': {'threshold': 6.0, 'operator': '<=', 'unit': '%'},
        'nmbe_percent': {'range': (-0.6, 0.6), 'operator': 'in_range', 'unit': '%'},
        'picp_90': {'range': (0.85, 0.95), 'operator': 'in_range', 'unit': ''}
    }
    
    compliance_results = {}
    
    for metric_name, requirement in requirements.items():
        if metric_name not in metrics:
            compliance_results[metric_name] = {
                'status': 'MISSING',
                'value': None,
                'requirement': requirement,
                'compliant': False
            }
            continue
            
        value = metrics[metric_name]
        
        if requirement['operator'] == '>=':
            compliant = value >= requirement['threshold']
            req_text = f">= {requirement['threshold']}"
        elif requirement['operator'] == '<=':
            compliant = value <= requirement['threshold']
            req_text = f"<= {requirement['threshold']}"
        elif requirement['operator'] == 'in_range':
            min_val, max_val = requirement['range']
            compliant = min_val <= value <= max_val
            req_text = f"∈ [{min_val}, {max_val}]"
        else:
            compliant = False
            req_text = "Unknown requirement"
        
        compliance_results[metric_name] = {
            'status': 'PASS' if compliant else 'FAIL',
            'value': value,
            'requirement_text': req_text + requirement['unit'],
            'compliant': compliant
        }
    
    # 总体合规性
    overall_compliant = all(result['compliant'] for result in compliance_results.values())
    
    compliance_results['overall'] = {
        'status': 'PASS' if overall_compliant else 'FAIL',
        'compliant': overall_compliant,
        'passed_checks': sum(1 for r in compliance_results.values() if r.get('compliant', False)),
        'total_checks': len(requirements)
    }
    
    return compliance_results


def format_metrics_report(metrics: Dict[str, float], 
                         compliance: Optional[Dict[str, Any]] = None) -> str:
    """格式化指标报告为字符串"""
    
    report = []
    report.append("=" * 60)
    report.append("LNG 能耗预测模型评估报告")
    report.append("=" * 60)
    
    # 基础指标
    report.append("\n基础回归指标:")
    report.append(f"  R² Score:        {metrics.get('r2_score', 0):.4f}")
    report.append(f"  RMSE:            {metrics.get('rmse', 0):.4f}")
    report.append(f"  MAE:             {metrics.get('mae', 0):.4f}")
    
    # 技术路线指标
    report.append("\n技术路线指标:")
    report.append(f"  NMBE:            {metrics.get('nmbe_percent', 0):.3f}%")
    report.append(f"  CV(RMSE):        {metrics.get('cv_rmse_percent', 0):.3f}%")
    report.append(f"  MAPE:            {metrics.get('mape_percent', 0):.3f}%")
    report.append(f"  PICP@90%:        {metrics.get('picp_90', 0):.3f}")
    
    # 合规性检查
    if compliance:
        report.append("\n合规性检查:")
        for metric_name, result in compliance.items():
            if metric_name == 'overall':
                continue
            status_symbol = "✓" if result['compliant'] else "✗"
            report.append(f"  {status_symbol} {metric_name}: {result['value']:.4f} ({result['requirement_text']})")
        
        overall = compliance['overall']
        report.append(f"\n总体状态: {overall['status']} ({overall['passed_checks']}/{overall['total_checks']})")
    
    report.append("=" * 60)
    
    return "\n".join(report)