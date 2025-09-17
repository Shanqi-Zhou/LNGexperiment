"""
自动化验证框架 - Week 2 优化
提供全面的模型验证、性能指标检查和质量保证功能
"""

import numpy as np
import pandas as pd
import warnings
import time
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, KFold
import psutil
from pathlib import Path
import json
from datetime import datetime


class AutomatedValidation:
    """
    自动化验证框架
    提供模型性能、数值稳定性、资源使用的全面验证
    """

    def __init__(self, target_metrics: Dict[str, float] = None,
                 output_dir: str = "validation_results"):
        """
        初始化自动化验证框架

        Args:
            target_metrics: 目标性能指标
            output_dir: 输出目录
        """
        # 默认目标指标（基于LNG项目要求）
        self.target_metrics = target_metrics or {
            'r2_min': 0.75,
            'cv_rmse_max': 0.06,  # 6%
            'nmbe_max': 0.006,    # 0.6%
            'mae_max_percent': 5.0,  # 5%
            'inference_time_per_sample_ms': 10.0  # 10ms per sample
        }

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 验证结果存储
        self.validation_results = {}
        self.validation_history = []

        print(f"  自动化验证框架初始化")
        print(f"    目标指标: R²≥{self.target_metrics['r2_min']}, "
              f"CV(RMSE)≤{self.target_metrics['cv_rmse_max']:.1%}, "
              f"|NMBE|≤{self.target_metrics['nmbe_max']:.1%}")

    def run_full_validation(self, model, X_test: np.ndarray, y_test: np.ndarray,
                          X_train: np.ndarray = None, y_train: np.ndarray = None,
                          validation_name: str = "default") -> Dict[str, Any]:
        """
        运行完整验证流程

        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            X_train: 训练特征（用于交叉验证）
            y_train: 训练目标（用于交叉验证）
            validation_name: 验证名称

        Returns:
            完整的验证结果
        """
        print(f"=== 开始自动化验证: {validation_name} ===")
        validation_start = time.time()

        # 1. 性能指标验证
        print("\n--- 性能指标验证 ---")
        performance_results = self._validate_performance_metrics(model, X_test, y_test)

        # 2. 数值稳定性验证
        print("\n--- 数值稳定性验证 ---")
        stability_results = self._validate_numerical_stability(model, X_test)

        # 3. 资源使用验证
        print("\n--- 资源使用验证 ---")
        resource_results = self._validate_resource_usage(model, X_test)

        # 4. 交叉验证（如果有训练数据）
        cross_validation_results = {}
        if X_train is not None and y_train is not None:
            print("\n--- 交叉验证 ---")
            cross_validation_results = self._perform_cross_validation(model, X_train, y_train)

        # 5. 预测一致性验证
        print("\n--- 预测一致性验证 ---")
        consistency_results = self._validate_prediction_consistency(model, X_test)

        # 6. 边界情况验证
        print("\n--- 边界情况验证 ---")
        edge_case_results = self._validate_edge_cases(model, X_test, y_test)

        # 7. 生成综合验证结果
        total_validation_time = time.time() - validation_start

        comprehensive_results = {
            'validation_name': validation_name,
            'timestamp': datetime.now().isoformat(),
            'total_validation_time': total_validation_time,
            'performance_metrics': performance_results,
            'numerical_stability': stability_results,
            'resource_usage': resource_results,
            'cross_validation': cross_validation_results,
            'prediction_consistency': consistency_results,
            'edge_cases': edge_case_results,
            'overall_assessment': self._generate_overall_assessment({
                'performance_metrics': performance_results,
                'numerical_stability': stability_results,
                'resource_usage': resource_results,
                'cross_validation': cross_validation_results,
                'prediction_consistency': consistency_results,
                'edge_cases': edge_case_results
            })
        }

        # 8. 保存验证结果
        self._save_validation_results(comprehensive_results, validation_name)

        # 9. 生成验证报告
        report = self._generate_validation_report(comprehensive_results)
        print(report)

        # 10. 更新验证历史
        self.validation_results[validation_name] = comprehensive_results
        self.validation_history.append(comprehensive_results)

        print(f"\n=== 验证完成，总用时: {total_validation_time:.2f}秒 ===")

        return comprehensive_results

    def _validate_performance_metrics(self, model, X_test: np.ndarray,
                                    y_test: np.ndarray) -> Dict[str, Any]:
        """验证性能指标"""
        try:
            # 进行预测
            y_pred = model.predict(X_test)

            # 基础指标
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # 相对指标
            y_mean = np.mean(y_test)
            cv_rmse = rmse / abs(y_mean) if abs(y_mean) > 1e-10 else float('inf')
            nmbe = np.mean(y_pred - y_test) / y_mean if abs(y_mean) > 1e-10 else 0
            mape = mean_absolute_percentage_error(y_test, y_pred)

            # 目标一致性指标
            residuals = y_pred - y_test
            residuals_std = np.std(residuals)
            residuals_normalized = residuals / (abs(y_mean) + 1e-10)

            results = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'cv_rmse': cv_rmse,
                'nmbe': nmbe,
                'mape': mape,
                'residuals_std': residuals_std,
                'residuals_mean': np.mean(residuals),
                'residuals_normalized_std': np.std(residuals_normalized),

                # 目标达成情况
                'r2_passed': r2 >= self.target_metrics['r2_min'],
                'cv_rmse_passed': cv_rmse <= self.target_metrics['cv_rmse_max'],
                'nmbe_passed': abs(nmbe) <= self.target_metrics['nmbe_max'],
                'mape_passed': mape <= self.target_metrics['mae_max_percent'] / 100,

                # 预测质量分析
                'prediction_range': (float(np.min(y_pred)), float(np.max(y_pred))),
                'actual_range': (float(np.min(y_test)), float(np.max(y_test))),
                'prediction_std': float(np.std(y_pred)),
                'actual_std': float(np.std(y_test))
            }

            # 计算性能等级
            performance_score = self._calculate_performance_score(results)
            results['performance_score'] = performance_score
            results['performance_grade'] = self._get_performance_grade(performance_score)

            print(f"    R²: {r2:.4f} {'✅' if results['r2_passed'] else '❌'}")
            print(f"    CV(RMSE): {cv_rmse:.4f} ({cv_rmse:.1%}) {'✅' if results['cv_rmse_passed'] else '❌'}")
            print(f"    NMBE: {nmbe:.4f} ({nmbe:.1%}) {'✅' if results['nmbe_passed'] else '❌'}")
            print(f"    MAPE: {mape:.4f} ({mape:.1%}) {'✅' if results['mape_passed'] else '❌'}")
            print(f"    性能等级: {results['performance_grade']}")

            return results

        except Exception as e:
            print(f"    ❌ 性能指标验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _validate_numerical_stability(self, model, X_test: np.ndarray) -> Dict[str, Any]:
        """验证数值稳定性"""
        try:
            results = {
                'has_nan_predictions': False,
                'has_inf_predictions': False,
                'prediction_variance_stability': 0.0,
                'repeated_prediction_consistency': True,
                'numerical_precision_loss': False
            }

            # 多次预测检查稳定性
            predictions_list = []
            n_repeats = 5

            for i in range(n_repeats):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = model.predict(X_test)
                    predictions_list.append(pred)

                # 检查NaN和Inf
                if np.any(np.isnan(pred)):
                    results['has_nan_predictions'] = True
                if np.any(np.isinf(pred)):
                    results['has_inf_predictions'] = True

            # 计算多次预测的一致性
            if len(predictions_list) > 1:
                pred_array = np.array(predictions_list)
                prediction_variance = np.var(pred_array, axis=0)
                max_variance = np.max(prediction_variance)
                mean_variance = np.mean(prediction_variance)

                results['prediction_variance_stability'] = float(mean_variance)
                results['max_prediction_variance'] = float(max_variance)
                results['repeated_prediction_consistency'] = max_variance < 1e-10

            # 检查数值精度
            large_input = X_test * 1e6
            small_input = X_test * 1e-6

            try:
                pred_large = model.predict(large_input)
                pred_small = model.predict(small_input)

                if np.any(np.isnan(pred_large)) or np.any(np.isnan(pred_small)):
                    results['numerical_precision_loss'] = True

            except Exception:
                results['numerical_precision_loss'] = True

            # 整体数值稳定性评估
            stability_issues = sum([
                results['has_nan_predictions'],
                results['has_inf_predictions'],
                not results['repeated_prediction_consistency'],
                results['numerical_precision_loss']
            ])

            results['numerical_stability_score'] = max(0, 100 - stability_issues * 25)
            results['numerical_stability_passed'] = stability_issues == 0

            print(f"    NaN预测: {'❌ 发现' if results['has_nan_predictions'] else '✅ 无'}")
            print(f"    Inf预测: {'❌ 发现' if results['has_inf_predictions'] else '✅ 无'}")
            print(f"    预测一致性: {'❌ 不稳定' if not results['repeated_prediction_consistency'] else '✅ 稳定'}")
            print(f"    数值精度: {'❌ 损失' if results['numerical_precision_loss'] else '✅ 正常'}")
            print(f"    稳定性评分: {results['numerical_stability_score']}/100")

            return results

        except Exception as e:
            print(f"    ❌ 数值稳定性验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _validate_resource_usage(self, model, X_test: np.ndarray) -> Dict[str, Any]:
        """验证资源使用情况"""
        try:
            # 内存使用测试
            memory_before = psutil.Process().memory_info().rss / 1024**2

            # 推理时间测试
            inference_times = []
            batch_sizes = [1, 10, 100, min(1000, len(X_test))]

            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    continue

                batch_data = X_test[:batch_size]

                # 预热
                _ = model.predict(batch_data)

                # 测试推理时间
                start_time = time.time()
                _ = model.predict(batch_data)
                end_time = time.time()

                batch_time = end_time - start_time
                per_sample_time = batch_time / batch_size * 1000  # ms

                inference_times.append({
                    'batch_size': batch_size,
                    'total_time_ms': batch_time * 1000,
                    'per_sample_time_ms': per_sample_time
                })

            memory_after = psutil.Process().memory_info().rss / 1024**2
            memory_usage = memory_after - memory_before

            # 大批量推理测试
            large_batch_size = min(len(X_test), 5000)
            start_time = time.time()
            cpu_before = psutil.cpu_percent()

            _ = model.predict(X_test[:large_batch_size])

            end_time = time.time()
            cpu_after = psutil.cpu_percent()

            large_batch_time = end_time - start_time
            large_batch_per_sample_ms = (large_batch_time / large_batch_size) * 1000

            results = {
                'memory_usage_mb': memory_usage,
                'inference_times': inference_times,
                'large_batch_inference_time_ms': large_batch_time * 1000,
                'large_batch_per_sample_ms': large_batch_per_sample_ms,
                'cpu_usage_change': cpu_after - cpu_before,

                # 性能要求验证
                'inference_time_passed': large_batch_per_sample_ms <= self.target_metrics['inference_time_per_sample_ms'],
                'memory_efficient': memory_usage < 1000,  # < 1GB
                'cpu_efficient': (cpu_after - cpu_before) < 90,  # CPU增长 < 90%

                # 可扩展性评估
                'scalability_score': self._calculate_scalability_score(inference_times)
            }

            print(f"    内存使用: {memory_usage:.1f}MB")
            print(f"    大批量推理: {large_batch_per_sample_ms:.2f}ms/样本 "
                  f"{'✅' if results['inference_time_passed'] else '❌'}")
            print(f"    CPU效率: {'✅ 高效' if results['cpu_efficient'] else '❌ 低效'}")
            print(f"    可扩展性评分: {results['scalability_score']}/100")

            return results

        except Exception as e:
            print(f"    ❌ 资源使用验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _perform_cross_validation(self, model, X_train: np.ndarray,
                                y_train: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """执行交叉验证"""
        try:
            # K折交叉验证
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # 执行交叉验证
            cv_scores = cross_val_score(model, X_train, y_train,
                                      cv=kfold, scoring='r2')

            # 手动计算更详细的交叉验证指标
            cv_detailed_results = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                # 训练模型（注意：这里假设模型可以重新拟合）
                try:
                    fold_model = type(model)()  # 创建同类型的新模型
                    if hasattr(fold_model, 'fit'):
                        fold_model.fit(X_fold_train, y_fold_train)
                        y_pred = fold_model.predict(X_fold_val)
                    else:
                        # 如果模型不支持重新训练，使用原模型
                        y_pred = model.predict(X_fold_val)

                    fold_r2 = r2_score(y_fold_val, y_pred)
                    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))

                    cv_detailed_results.append({
                        'fold': fold_idx + 1,
                        'r2': fold_r2,
                        'rmse': fold_rmse,
                        'train_size': len(train_idx),
                        'val_size': len(val_idx)
                    })

                except Exception:
                    # 如果无法重新训练，跳过详细分析
                    pass

            results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'cv_min': float(np.min(cv_scores)),
                'cv_max': float(np.max(cv_scores)),
                'detailed_results': cv_detailed_results,

                # 交叉验证质量评估
                'cv_consistency': float(np.std(cv_scores)) < 0.05,  # 标准差小于0.05
                'cv_performance_passed': float(np.mean(cv_scores)) >= self.target_metrics['r2_min'],
                'cv_stability_score': max(0, 100 - float(np.std(cv_scores)) * 1000)
            }

            print(f"    交叉验证R²: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            print(f"    性能一致性: {'✅ 稳定' if results['cv_consistency'] else '❌ 不稳定'}")
            print(f"    CV性能要求: {'✅ 达标' if results['cv_performance_passed'] else '❌ 不达标'}")

            return results

        except Exception as e:
            print(f"    ❌ 交叉验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _validate_prediction_consistency(self, model, X_test: np.ndarray) -> Dict[str, Any]:
        """验证预测一致性"""
        try:
            # 测试输入顺序一致性
            indices = np.arange(len(X_test))
            shuffled_indices = np.random.permutation(indices)

            pred_original = model.predict(X_test)
            pred_shuffled = model.predict(X_test[shuffled_indices])
            pred_reordered = pred_shuffled[np.argsort(shuffled_indices)]

            order_consistency = np.allclose(pred_original, pred_reordered, rtol=1e-10)

            # 测试数据副本一致性
            X_copy = X_test.copy()
            pred_copy = model.predict(X_copy)
            copy_consistency = np.allclose(pred_original, pred_copy, rtol=1e-10)

            # 测试子集预测一致性
            subset_size = min(100, len(X_test))
            subset_indices = np.random.choice(len(X_test), subset_size, replace=False)

            pred_subset_standalone = model.predict(X_test[subset_indices])
            pred_subset_from_full = pred_original[subset_indices]

            subset_consistency = np.allclose(pred_subset_standalone, pred_subset_from_full, rtol=1e-10)

            results = {
                'order_consistency': order_consistency,
                'copy_consistency': copy_consistency,
                'subset_consistency': subset_consistency,
                'max_order_diff': float(np.max(np.abs(pred_original - pred_reordered))) if not order_consistency else 0.0,
                'max_copy_diff': float(np.max(np.abs(pred_original - pred_copy))) if not copy_consistency else 0.0,
                'consistency_score': sum([order_consistency, copy_consistency, subset_consistency]) / 3 * 100,
                'overall_consistency_passed': all([order_consistency, copy_consistency, subset_consistency])
            }

            print(f"    顺序一致性: {'✅' if order_consistency else '❌'}")
            print(f"    副本一致性: {'✅' if copy_consistency else '❌'}")
            print(f"    子集一致性: {'✅' if subset_consistency else '❌'}")
            print(f"    一致性评分: {results['consistency_score']:.1f}/100")

            return results

        except Exception as e:
            print(f"    ❌ 预测一致性验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _validate_edge_cases(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """验证边界情况处理"""
        try:
            results = {
                'zero_input_handling': True,
                'extreme_value_handling': True,
                'small_batch_handling': True,
                'single_sample_handling': True,
                'edge_case_errors': []
            }

            # 测试零输入
            try:
                zero_input = np.zeros_like(X_test[:1])
                zero_pred = model.predict(zero_input)
                if np.any(np.isnan(zero_pred)) or np.any(np.isinf(zero_pred)):
                    results['zero_input_handling'] = False
                    results['edge_case_errors'].append("Zero input produces NaN/Inf")
            except Exception as e:
                results['zero_input_handling'] = False
                results['edge_case_errors'].append(f"Zero input error: {str(e)}")

            # 测试极值输入
            try:
                extreme_input = X_test[:1] * 1000
                extreme_pred = model.predict(extreme_input)
                if np.any(np.isnan(extreme_pred)) or np.any(np.isinf(extreme_pred)):
                    results['extreme_value_handling'] = False
                    results['edge_case_errors'].append("Extreme values produce NaN/Inf")
            except Exception as e:
                results['extreme_value_handling'] = False
                results['edge_case_errors'].append(f"Extreme value error: {str(e)}")

            # 测试单样本预测
            try:
                single_pred = model.predict(X_test[:1])
                if len(single_pred) != 1:
                    results['single_sample_handling'] = False
                    results['edge_case_errors'].append("Single sample prediction shape error")
            except Exception as e:
                results['single_sample_handling'] = False
                results['edge_case_errors'].append(f"Single sample error: {str(e)}")

            # 测试小批量处理
            try:
                small_batch_pred = model.predict(X_test[:3])
                if len(small_batch_pred) != 3:
                    results['small_batch_handling'] = False
                    results['edge_case_errors'].append("Small batch prediction shape error")
            except Exception as e:
                results['small_batch_handling'] = False
                results['edge_case_errors'].append(f"Small batch error: {str(e)}")

            # 计算边界情况处理评分
            edge_cases_passed = sum([
                results['zero_input_handling'],
                results['extreme_value_handling'],
                results['small_batch_handling'],
                results['single_sample_handling']
            ])

            results['edge_case_score'] = (edge_cases_passed / 4) * 100
            results['edge_case_robustness_passed'] = edge_cases_passed >= 3

            print(f"    零值输入: {'✅' if results['zero_input_handling'] else '❌'}")
            print(f"    极值处理: {'✅' if results['extreme_value_handling'] else '❌'}")
            print(f"    单样本预测: {'✅' if results['single_sample_handling'] else '❌'}")
            print(f"    小批量处理: {'✅' if results['small_batch_handling'] else '❌'}")
            print(f"    鲁棒性评分: {results['edge_case_score']:.1f}/100")

            if results['edge_case_errors']:
                print(f"    边界情况错误: {len(results['edge_case_errors'])}个")

            return results

        except Exception as e:
            print(f"    ❌ 边界情况验证失败: {e}")
            return {'error': str(e), 'failed': True}

    def _calculate_performance_score(self, performance_results: Dict) -> float:
        """计算性能综合评分"""
        score = 0.0

        # R²权重40%
        if 'r2' in performance_results:
            r2_score = min(100, max(0, performance_results['r2'] * 100))
            score += r2_score * 0.4

        # CV(RMSE)权重30%
        if 'cv_rmse' in performance_results:
            cv_rmse = performance_results['cv_rmse']
            cv_rmse_score = max(0, 100 - cv_rmse * 1000)  # 转换为0-100分
            score += cv_rmse_score * 0.3

        # NMBE权重20%
        if 'nmbe' in performance_results:
            nmbe = abs(performance_results['nmbe'])
            nmbe_score = max(0, 100 - nmbe * 10000)  # 转换为0-100分
            score += nmbe_score * 0.2

        # MAPE权重10%
        if 'mape' in performance_results:
            mape = performance_results['mape']
            mape_score = max(0, 100 - mape * 100)
            score += mape_score * 0.1

        return min(100, max(0, score))

    def _get_performance_grade(self, score: float) -> str:
        """根据评分获取性能等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (满意)"
        elif score >= 60:
            return "C (一般)"
        else:
            return "D (不达标)"

    def _calculate_scalability_score(self, inference_times: List[Dict]) -> float:
        """计算可扩展性评分"""
        if len(inference_times) < 2:
            return 50.0

        # 分析推理时间与批量大小的关系
        batch_sizes = [t['batch_size'] for t in inference_times]
        per_sample_times = [t['per_sample_time_ms'] for t in inference_times]

        # 理想情况下，per_sample_time应该随batch_size增大而减小
        time_improvement_ratio = per_sample_times[0] / per_sample_times[-1] if per_sample_times[-1] > 0 else 1

        # 评分基于批量处理效率
        if time_improvement_ratio >= 5:
            return 100.0
        elif time_improvement_ratio >= 3:
            return 80.0
        elif time_improvement_ratio >= 2:
            return 60.0
        elif time_improvement_ratio >= 1.5:
            return 40.0
        else:
            return 20.0

    def _generate_overall_assessment(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成总体评估"""
        assessment = {
            'validation_passed': True,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'overall_score': 0.0,
            'deployment_ready': False
        }

        # 检查各项验证结果
        scores = []

        # 性能指标权重50%
        perf_results = all_results.get('performance_metrics', {})
        if not perf_results.get('failed', False):
            perf_score = perf_results.get('performance_score', 0)
            scores.append(('performance', perf_score, 0.5))

            # 检查关键性能指标
            if not perf_results.get('r2_passed', False):
                assessment['critical_issues'].append(f"R²未达标: {perf_results.get('r2', 0):.4f} < {self.target_metrics['r2_min']}")
                assessment['validation_passed'] = False

            if not perf_results.get('cv_rmse_passed', False):
                assessment['critical_issues'].append(f"CV(RMSE)未达标: {perf_results.get('cv_rmse', 0):.4f} > {self.target_metrics['cv_rmse_max']}")

        # 数值稳定性权重20%
        stability_results = all_results.get('numerical_stability', {})
        if not stability_results.get('failed', False):
            stability_score = stability_results.get('numerical_stability_score', 0)
            scores.append(('stability', stability_score, 0.2))

            if not stability_results.get('numerical_stability_passed', False):
                assessment['critical_issues'].append("数值稳定性不合格")
                assessment['validation_passed'] = False

        # 资源使用权重15%
        resource_results = all_results.get('resource_usage', {})
        if not resource_results.get('failed', False):
            # 基于推理时间和内存使用计算资源评分
            resource_score = 100
            if not resource_results.get('inference_time_passed', True):
                resource_score -= 30
            if not resource_results.get('memory_efficient', True):
                resource_score -= 20
            if not resource_results.get('cpu_efficient', True):
                resource_score -= 20

            scores.append(('resource', resource_score, 0.15))

        # 预测一致性权重10%
        consistency_results = all_results.get('prediction_consistency', {})
        if not consistency_results.get('failed', False):
            consistency_score = consistency_results.get('consistency_score', 0)
            scores.append(('consistency', consistency_score, 0.1))

        # 边界情况权重5%
        edge_case_results = all_results.get('edge_cases', {})
        if not edge_case_results.get('failed', False):
            edge_score = edge_case_results.get('edge_case_score', 0)
            scores.append(('edge_cases', edge_score, 0.05))

        # 计算加权总分
        if scores:
            weighted_sum = sum(score * weight for name, score, weight in scores)
            total_weight = sum(weight for name, score, weight in scores)
            assessment['overall_score'] = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            assessment['overall_score'] = 0

        # 生成建议
        if assessment['overall_score'] >= 85:
            assessment['deployment_ready'] = len(assessment['critical_issues']) == 0
            assessment['recommendations'].append("模型性能优秀，建议部署到生产环境")
        elif assessment['overall_score'] >= 70:
            assessment['recommendations'].append("模型性能良好，建议进行小规模试运行")
        elif assessment['overall_score'] >= 60:
            assessment['recommendations'].append("模型性能一般，建议进一步优化后部署")
        else:
            assessment['recommendations'].append("模型性能不达标，不建议部署，需要重新优化")

        # 生成具体改进建议
        if len(assessment['critical_issues']) > 0:
            assessment['recommendations'].append("优先解决关键问题：" + "; ".join(assessment['critical_issues']))

        return assessment

    def _generate_validation_report(self, comprehensive_results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report_lines = [
            "=" * 80,
            f"自动化验证报告 - {comprehensive_results['validation_name']}",
            "=" * 80,
            f"验证时间: {comprehensive_results['timestamp']}",
            f"总验证用时: {comprehensive_results['total_validation_time']:.2f}秒",
            "",
            "## 验证结果摘要",
            ""
        ]

        # 总体评估
        overall = comprehensive_results['overall_assessment']
        report_lines.extend([
            f"总体评分: {overall['overall_score']:.1f}/100",
            f"验证状态: {'✅ 通过' if overall['validation_passed'] else '❌ 未通过'}",
            f"部署就绪: {'✅ 是' if overall['deployment_ready'] else '❌ 否'}",
            ""
        ])

        # 详细结果
        perf = comprehensive_results.get('performance_metrics', {})
        if not perf.get('failed', False):
            report_lines.extend([
                "## 📊 性能指标",
                f"- R²: {perf.get('r2', 0):.4f} {'✅' if perf.get('r2_passed', False) else '❌'}",
                f"- CV(RMSE): {perf.get('cv_rmse', 0):.4f} ({perf.get('cv_rmse', 0):.1%}) {'✅' if perf.get('cv_rmse_passed', False) else '❌'}",
                f"- NMBE: {perf.get('nmbe', 0):.4f} ({perf.get('nmbe', 0):.1%}) {'✅' if perf.get('nmbe_passed', False) else '❌'}",
                f"- MAPE: {perf.get('mape', 0):.4f} ({perf.get('mape', 0):.1%}) {'✅' if perf.get('mape_passed', False) else '❌'}",
                f"- 性能等级: {perf.get('performance_grade', 'N/A')}",
                ""
            ])

        stability = comprehensive_results.get('numerical_stability', {})
        if not stability.get('failed', False):
            report_lines.extend([
                "## 🔢 数值稳定性",
                f"- NaN预测: {'❌ 发现' if stability.get('has_nan_predictions', False) else '✅ 无'}",
                f"- Inf预测: {'❌ 发现' if stability.get('has_inf_predictions', False) else '✅ 无'}",
                f"- 预测一致性: {'✅ 稳定' if stability.get('repeated_prediction_consistency', True) else '❌ 不稳定'}",
                f"- 数值精度: {'✅ 正常' if not stability.get('numerical_precision_loss', False) else '❌ 损失'}",
                f"- 稳定性评分: {stability.get('numerical_stability_score', 0):.1f}/100",
                ""
            ])

        resource = comprehensive_results.get('resource_usage', {})
        if not resource.get('failed', False):
            report_lines.extend([
                "## 💻 资源使用",
                f"- 内存使用: {resource.get('memory_usage_mb', 0):.1f}MB",
                f"- 推理时间: {resource.get('large_batch_per_sample_ms', 0):.2f}ms/样本 {'✅' if resource.get('inference_time_passed', False) else '❌'}",
                f"- CPU效率: {'✅ 高效' if resource.get('cpu_efficient', True) else '❌ 低效'}",
                f"- 可扩展性评分: {resource.get('scalability_score', 0):.1f}/100",
                ""
            ])

        # 关键问题和建议
        if overall['critical_issues']:
            report_lines.extend([
                "## ⚠️ 关键问题",
                ""
            ])
            for issue in overall['critical_issues']:
                report_lines.append(f"- {issue}")
            report_lines.append("")

        if overall['recommendations']:
            report_lines.extend([
                "## 💡 建议",
                ""
            ])
            for rec in overall['recommendations']:
                report_lines.append(f"- {rec}")
            report_lines.append("")

        report_lines.extend([
            "=" * 80,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])

        return "\\n".join(report_lines)

    def _save_validation_results(self, results: Dict[str, Any], validation_name: str):
        """保存验证结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"validation_{validation_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # 确保所有数据都可以JSON序列化
        serializable_results = self._make_json_serializable(results)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"    验证结果已保存: {filepath}")

    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj


# 便捷函数
def validate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                  X_train: np.ndarray = None, y_train: np.ndarray = None,
                  target_metrics: Dict[str, float] = None) -> Dict[str, Any]:
    """
    便捷的模型验证函数

    Args:
        model: 待验证的模型
        X_test: 测试特征
        y_test: 测试目标
        X_train: 训练特征（可选）
        y_train: 训练目标（可选）
        target_metrics: 目标指标（可选）

    Returns:
        验证结果
    """
    validator = AutomatedValidation(target_metrics=target_metrics)
    return validator.run_full_validation(
        model, X_test, y_test, X_train, y_train, "quick_validation"
    )


# 测试代码
if __name__ == '__main__':
    print("=== 自动化验证框架测试 ===")

    # 创建模拟模型
    class MockModel:
        def predict(self, X):
            # 简单的线性模型模拟
            return np.sum(X * [1, 2, -1, 0.5, -0.5, 1], axis=1) + np.random.randn(len(X)) * 0.1

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 6

    X_test = np.random.randn(n_samples, n_features)
    y_test = np.sum(X_test * [1, 2, -1, 0.5, -0.5, 1], axis=1) + np.random.randn(n_samples) * 0.2

    X_train = np.random.randn(n_samples * 2, n_features)
    y_train = np.sum(X_train * [1, 2, -1, 0.5, -0.5, 1], axis=1) + np.random.randn(n_samples * 2) * 0.2

    print(f"测试数据: X_test{X_test.shape}, y_test{y_test.shape}")
    print(f"训练数据: X_train{X_train.shape}, y_train{y_train.shape}")

    # 创建模型
    model = MockModel()

    # 运行验证
    validation_results = validate_model(
        model, X_test, y_test, X_train, y_train
    )

    print("\\n验证完成!")
    print(f"总体评分: {validation_results['overall_assessment']['overall_score']:.1f}/100")
    print(f"验证通过: {validation_results['overall_assessment']['validation_passed']}")
    print(f"部署就绪: {validation_results['overall_assessment']['deployment_ready']}")

    print("\\n=== 自动化验证框架测试完成 ===")