"""
端到端基准测试系统 - LNG优化方案性能评估核心
提供全流程基准测试，从数据预处理到模型训练再到最终预测
支持多种优化策略对比、性能分析、资源使用评估等功能
专为RTX 4060 8GB环境优化设计
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
import logging
import json
import os
from pathlib import Path
import warnings
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 导入所有组件模块
from ..dataio.data_loader import LNGDataLoader
from ..preprocessing.enhanced_hampel import EnhancedHampelFilter
from ..features.dual_channel_selector import DualChannelFeatureSelector
from ..features.incremental_calculator import IncrementalFeatureExtractor
from ..models.adaptive_strategy import SampleSizeAdaptiveModelFactory
from ..models.hgbr_baseline import AdaptiveHGBRBaseline
from ..models.residual_framework import ResidualModelingFramework
from ..models.tcn_attention import TCNWithLinearAttention
from ..training.unified_framework import UnifiedTrainingFramework, UnifiedTrainingConfig
from ..monitoring.performance_monitor import PerformanceMonitoringSystem

warnings.filterwarnings('ignore')


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 数据配置
    data_size_variants: List[int] = None  # [1000, 5000, 10000, 50000]
    feature_size_variants: List[int] = None  # [10, 50, 100, 200]
    
    # 模型配置
    test_models: List[str] = None  # ['hgbr', 'residual', 'adaptive', 'tcn']
    optimization_levels: List[str] = None  # ['baseline', 'optimized', 'full']
    
    # 基准测试设置
    n_runs_per_config: int = 3
    use_synthetic_data: bool = True
    include_memory_profiling: bool = True
    include_speed_profiling: bool = True
    
    # 硬件配置
    target_gpu_memory_gb: float = 8.0
    max_cpu_cores: int = 8
    
    def __post_init__(self):
        if self.data_size_variants is None:
            self.data_size_variants = [1000, 5000, 10000]
        if self.feature_size_variants is None:
            self.feature_size_variants = [20, 50, 100]
        if self.test_models is None:
            self.test_models = ['hgbr', 'residual', 'tcn']
        if self.optimization_levels is None:
            self.optimization_levels = ['baseline', 'optimized']


@dataclass
class ComponentBenchmark:
    """单组件基准测试结果"""
    component_name: str
    execution_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    throughput_samples_per_sec: float = 0.0
    accuracy_metrics: Dict[str, float] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.accuracy_metrics is None:
            self.accuracy_metrics = {}


@dataclass
class PipelineBenchmark:
    """流水线基准测试结果"""
    pipeline_name: str
    total_time_s: float
    component_benchmarks: List[ComponentBenchmark]
    final_accuracy: Dict[str, float]
    peak_memory_gb: float
    optimization_level: str
    data_size: int
    feature_size: int
    
    def get_total_component_time(self) -> float:
        return sum(cb.execution_time_ms for cb in self.component_benchmarks) / 1000.0
    
    def get_memory_efficiency(self) -> float:
        return self.data_size / (self.peak_memory_gb * 1024)  # samples per MB


class PerformanceComparison:
    """性能对比分析"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_results = {}
        self.optimized_results = {}
    
    def add_baseline_result(self, key: str, result: PipelineBenchmark):
        self.baseline_results[key] = result
    
    def add_optimized_result(self, key: str, result: PipelineBenchmark):
        self.optimized_results[key] = result
    
    def calculate_improvements(self) -> Dict[str, Dict[str, float]]:
        """计算优化改进"""
        improvements = {}
        
        for key in self.baseline_results:
            if key in self.optimized_results:
                baseline = self.baseline_results[key]
                optimized = self.optimized_results[key]
                
                speed_improvement = baseline.total_time_s / optimized.total_time_s
                memory_reduction = 1 - (optimized.peak_memory_gb / baseline.peak_memory_gb)
                
                # 准确性对比
                accuracy_change = {}
                for metric in baseline.final_accuracy:
                    if metric in optimized.final_accuracy:
                        if metric in ['r2_score']:  # 越高越好
                            accuracy_change[metric] = (optimized.final_accuracy[metric] - baseline.final_accuracy[metric]) / baseline.final_accuracy[metric]
                        else:  # RMSE, MAE 等越低越好
                            accuracy_change[metric] = (baseline.final_accuracy[metric] - optimized.final_accuracy[metric]) / baseline.final_accuracy[metric]
                
                improvements[key] = {
                    'speed_improvement': speed_improvement,
                    'memory_reduction_percent': memory_reduction * 100,
                    'accuracy_changes': accuracy_change
                }
        
        return improvements
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        improvements = self.calculate_improvements()
        
        report = {
            'summary': {
                'total_comparisons': len(improvements),
                'avg_speed_improvement': np.mean([imp['speed_improvement'] for imp in improvements.values()]),
                'avg_memory_reduction': np.mean([imp['memory_reduction_percent'] for imp in improvements.values()]),
                'performance_targets_met': self._check_performance_targets(improvements)
            },
            'detailed_results': improvements,
            'baseline_results': {k: asdict(v) for k, v in self.baseline_results.items()},
            'optimized_results': {k: asdict(v) for k, v in self.optimized_results.items()}
        }
        
        return report
    
    def _check_performance_targets(self, improvements: Dict) -> Dict[str, bool]:
        """检查是否达到性能目标"""
        targets = {
            'speed_3x_improvement': False,
            'memory_40_percent_reduction': False,
            'accuracy_maintained': False
        }
        
        if improvements:
            avg_speed = np.mean([imp['speed_improvement'] for imp in improvements.values()])
            avg_memory_reduction = np.mean([imp['memory_reduction_percent'] for imp in improvements.values()])
            
            targets['speed_3x_improvement'] = avg_speed >= 3.0
            targets['memory_40_percent_reduction'] = avg_memory_reduction >= 40.0
            
            # 检查准确性是否下降超过5%
            accuracy_maintained = True
            for imp in improvements.values():
                for metric, change in imp['accuracy_changes'].items():
                    if change < -0.05:  # 下降超过5%
                        accuracy_maintained = False
                        break
                if not accuracy_maintained:
                    break
            
            targets['accuracy_maintained'] = accuracy_maintained
        
        return targets


class EndToEndBenchmarkSuite:
    """
    端到端基准测试套件
    集成所有优化组件的综合性能评估系统
    """
    
    def __init__(self, 
                 config: Optional[BenchmarkConfig] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or BenchmarkConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitoringSystem(logger=self.logger)
        
        # 结果存储
        self.benchmark_results = []
        self.comparison = PerformanceComparison(logger)
        
        self.logger.info("端到端基准测试套件初始化完成")
    
    def generate_synthetic_data(self, n_samples: int, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成测试数据"""
        np.random.seed(42)
        
        # 生成特征数据（模拟LNG系统数据特征）
        X = np.random.randn(n_samples, n_features)
        
        # 添加一些相关性和模式
        if n_features >= 4:
            # 模拟温度、压力、流量、效率等关键变量
            X[:, 0] = np.random.uniform(-10, 40, n_samples)  # 温度
            X[:, 1] = np.random.uniform(0.1, 2.0, n_samples)  # 压力
            X[:, 2] = np.random.uniform(10, 1000, n_samples)  # 流量
            X[:, 3] = X[:, 0] * 0.1 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1  # 效率相关
        
        # 生成目标变量（能耗）
        y = (X[:, 0] * 0.2 + X[:, 1] * 0.5 + 
             (X[:, 2] / 100) * 0.3 + 
             np.random.randn(n_samples) * 0.1)
        
        return X, y
    
    def benchmark_component(self, 
                          component_name: str,
                          component_func: Callable,
                          *args, **kwargs) -> ComponentBenchmark:
        """基准测试单个组件"""
        
        # 内存监控开始
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_gpu_memory = 0
        
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # 执行组件
        start_time = time.time()
        
        try:
            result = component_func(*args, **kwargs)
            error_message = ""
        except Exception as e:
            result = None
            error_message = str(e)
            self.logger.error(f"组件 {component_name} 执行失败: {e}")
        
        execution_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 内存监控结束
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        final_gpu_memory = 0
        gpu_memory_usage = 0
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_usage = final_gpu_memory - initial_gpu_memory
        
        # CPU利用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 计算吞吐量（如果适用）
        throughput = 0.0
        if hasattr(kwargs.get('X'), '__len__') and execution_time > 0:
            throughput = len(kwargs['X']) * 1000 / execution_time  # samples per second
        
        # 准确性指标（如果是预测组件）
        accuracy_metrics = {}
        if result is not None and hasattr(result, 'score'):
            try:
                accuracy_metrics['default_score'] = float(result.score(kwargs.get('X'), kwargs.get('y')))
            except:
                pass
        
        return ComponentBenchmark(
            component_name=component_name,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory_usage,
            cpu_utilization_percent=cpu_percent,
            throughput_samples_per_sec=throughput,
            accuracy_metrics=accuracy_metrics,
            error_message=error_message
        )
    
    def benchmark_pipeline(self, 
                          pipeline_name: str,
                          optimization_level: str,
                          data_size: int, 
                          feature_size: int,
                          model_type: str) -> PipelineBenchmark:
        """基准测试完整流水线"""
        
        self.logger.info(f"基准测试流水线: {pipeline_name} "
                        f"(优化级别: {optimization_level}, 数据: {data_size}x{feature_size}, 模型: {model_type})")
        
        pipeline_start_time = time.time()
        component_benchmarks = []
        
        # 1. 数据生成
        data_benchmark = self.benchmark_component(
            "data_generation",
            self.generate_synthetic_data,
            data_size, feature_size
        )
        component_benchmarks.append(data_benchmark)
        
        X, y = self.generate_synthetic_data(data_size, feature_size)
        
        # 2. 数据预处理
        if optimization_level in ['optimized', 'full']:
            hampel_detector = HampelOutlierDetector()
            preprocessing_benchmark = self.benchmark_component(
                "hampel_preprocessing",
                hampel_detector.detect_and_remove_outliers,
                X=X, y=y
            )
            component_benchmarks.append(preprocessing_benchmark)
            X, y = hampel_detector.detect_and_remove_outliers(X, y)
        
        # 3. 特征工程
        processed_X = X
        if optimization_level in ['optimized', 'full']:
            # 双通道特征选择
            selector = DualChannelFeatureSelector()
            mid_point = X.shape[1] // 2
            vt_features = X[:, :mid_point]
            vs_features = X[:, mid_point:]
            
            feature_selection_benchmark = self.benchmark_component(
                "dual_channel_feature_selection",
                selector.fit_transform,
                vt_features, vs_features, y
            )
            component_benchmarks.append(feature_selection_benchmark)
            processed_X = selector.fit_transform(vt_features, vs_features, y)
        
        # 4. 模型训练
        training_config = UnifiedTrainingConfig(
            model_type=model_type,
            mixed_precision=(optimization_level == 'full'),
            onecycle_lr=(optimization_level == 'full'),
            use_purged_validation=(optimization_level in ['optimized', 'full']),
            max_epochs=20 if optimization_level != 'baseline' else 10
        )
        
        trainer = UnifiedTrainingFramework(training_config, self.logger)
        
        training_benchmark = self.benchmark_component(
            f"{model_type}_training",
            trainer.train,
            X=processed_X, y=y
        )
        component_benchmarks.append(training_benchmark)
        
        # 获取训练结果
        training_results = trainer.train(processed_X, y)
        model = training_results['training_results'].get('model')
        
        # 5. 模型评估
        final_accuracy = {}
        if model is not None:
            # 简单的评估（使用部分数据作为测试集）
            test_size = min(200, len(processed_X) // 4)
            X_test = processed_X[-test_size:]
            y_test = y[-test_size:]
            
            evaluation_benchmark = self.benchmark_component(
                "model_evaluation",
                trainer.evaluate_model,
                model, X_test, y_test
            )
            component_benchmarks.append(evaluation_benchmark)
            
            final_accuracy = trainer.evaluate_model(model, X_test, y_test)
        
        # 计算总时间和峰值内存
        total_time = time.time() - pipeline_start_time
        
        # 峰值内存使用
        peak_memory = max(cb.memory_usage_mb for cb in component_benchmarks) / 1024  # GB
        if torch.cuda.is_available():
            gpu_memory = max(cb.gpu_memory_mb for cb in component_benchmarks) / 1024  # GB
            peak_memory = max(peak_memory, gpu_memory)
        
        return PipelineBenchmark(
            pipeline_name=pipeline_name,
            total_time_s=total_time,
            component_benchmarks=component_benchmarks,
            final_accuracy=final_accuracy,
            peak_memory_gb=peak_memory,
            optimization_level=optimization_level,
            data_size=data_size,
            feature_size=feature_size
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合基准测试"""
        self.logger.info("开始综合基准测试...")
        
        all_results = []
        
        # 遍历所有配置组合
        for data_size in self.config.data_size_variants:
            for feature_size in self.config.feature_size_variants:
                for model_type in self.config.test_models:
                    for opt_level in self.config.optimization_levels:
                        
                        # 多次运行取平均
                        run_results = []
                        for run_id in range(self.config.n_runs_per_config):
                            
                            pipeline_name = f"{model_type}_{opt_level}_{data_size}_{feature_size}_run{run_id}"
                            
                            try:
                                result = self.benchmark_pipeline(
                                    pipeline_name, opt_level, data_size, feature_size, model_type
                                )
                                run_results.append(result)
                                
                                # 存储到对比分析中
                                config_key = f"{model_type}_{data_size}_{feature_size}"
                                if opt_level == 'baseline':
                                    self.comparison.add_baseline_result(config_key, result)
                                elif opt_level in ['optimized', 'full']:
                                    self.comparison.add_optimized_result(config_key, result)
                                
                            except Exception as e:
                                self.logger.error(f"基准测试失败 {pipeline_name}: {e}")
                        
                        if run_results:
                            # 计算平均结果
                            avg_result = self._average_pipeline_results(run_results)
                            all_results.append(avg_result)
        
        # 生成综合报告
        comprehensive_report = self._generate_comprehensive_report(all_results)
        
        self.logger.info("综合基准测试完成")
        return comprehensive_report
    
    def _average_pipeline_results(self, results: List[PipelineBenchmark]) -> PipelineBenchmark:
        """计算多次运行的平均结果"""
        if len(results) == 1:
            return results[0]
        
        # 平均时间
        avg_time = np.mean([r.total_time_s for r in results])
        
        # 平均内存
        avg_memory = np.mean([r.peak_memory_gb for r in results])
        
        # 平均组件性能
        avg_components = []
        component_names = [cb.component_name for cb in results[0].component_benchmarks]
        
        for comp_name in component_names:
            comp_times = []
            comp_memories = []
            comp_gpu_memories = []
            
            for result in results:
                for cb in result.component_benchmarks:
                    if cb.component_name == comp_name:
                        comp_times.append(cb.execution_time_ms)
                        comp_memories.append(cb.memory_usage_mb)
                        comp_gpu_memories.append(cb.gpu_memory_mb)
            
            if comp_times:
                avg_component = ComponentBenchmark(
                    component_name=comp_name,
                    execution_time_ms=np.mean(comp_times),
                    memory_usage_mb=np.mean(comp_memories),
                    gpu_memory_mb=np.mean(comp_gpu_memories),
                    cpu_utilization_percent=0.0,
                    throughput_samples_per_sec=0.0
                )
                avg_components.append(avg_component)
        
        # 平均准确性
        avg_accuracy = {}
        accuracy_keys = set()
        for result in results:
            accuracy_keys.update(result.final_accuracy.keys())
        
        for key in accuracy_keys:
            values = [r.final_accuracy[key] for r in results if key in r.final_accuracy]
            if values:
                avg_accuracy[key] = np.mean(values)
        
        # 使用第一个结果的其他属性
        first_result = results[0]
        
        return PipelineBenchmark(
            pipeline_name=f"avg_{first_result.pipeline_name}",
            total_time_s=avg_time,
            component_benchmarks=avg_components,
            final_accuracy=avg_accuracy,
            peak_memory_gb=avg_memory,
            optimization_level=first_result.optimization_level,
            data_size=first_result.data_size,
            feature_size=first_result.feature_size
        )
    
    def _generate_comprehensive_report(self, results: List[PipelineBenchmark]) -> Dict[str, Any]:
        """生成综合报告"""
        
        # 性能对比分析
        comparison_report = self.comparison.generate_comparison_report()
        
        # 按优化级别分组统计
        stats_by_optimization = defaultdict(list)
        for result in results:
            stats_by_optimization[result.optimization_level].append(result)
        
        optimization_summary = {}
        for opt_level, level_results in stats_by_optimization.items():
            optimization_summary[opt_level] = {
                'count': len(level_results),
                'avg_time_s': np.mean([r.total_time_s for r in level_results]),
                'avg_memory_gb': np.mean([r.peak_memory_gb for r in level_results]),
                'avg_accuracy': {}
            }
            
            # 准确性统计
            accuracy_keys = set()
            for result in level_results:
                accuracy_keys.update(result.final_accuracy.keys())
            
            for key in accuracy_keys:
                values = [r.final_accuracy[key] for r in level_results if key in r.final_accuracy]
                if values:
                    optimization_summary[opt_level]['avg_accuracy'][key] = np.mean(values)
        
        # 模型性能排名
        model_rankings = self._rank_models_by_performance(results)
        
        # 系统资源分析
        resource_analysis = self._analyze_system_resources()
        
        comprehensive_report = {
            'benchmark_summary': {
                'total_configurations_tested': len(results),
                'test_completion_time': datetime.now().isoformat(),
                'target_hardware': f"RTX 4060 {self.config.target_gpu_memory_gb}GB",
                'optimization_targets': {
                    'speed_improvement': '3-5x',
                    'memory_reduction': '40-60%',
                    'accuracy_loss_tolerance': '<5%'
                }
            },
            'performance_comparison': comparison_report,
            'optimization_level_analysis': optimization_summary,
            'model_rankings': model_rankings,
            'system_resource_analysis': resource_analysis,
            'detailed_results': [asdict(result) for result in results],
            'recommendations': self._generate_optimization_recommendations(comparison_report)
        }
        
        return comprehensive_report
    
    def _rank_models_by_performance(self, results: List[PipelineBenchmark]) -> Dict[str, Any]:
        """按性能对模型进行排名"""
        model_performance = defaultdict(list)
        
        for result in results:
            if result.optimization_level in ['optimized', 'full']:  # 只考虑优化版本
                model_name = result.pipeline_name.split('_')[0]  # 提取模型名称
                
                # 计算综合性能分数
                time_score = 1 / result.total_time_s  # 时间越短分数越高
                memory_score = 1 / result.peak_memory_gb  # 内存越少分数越高
                accuracy_score = result.final_accuracy.get('r2_score', 0)  # 准确性分数
                
                composite_score = (time_score * 0.4 + memory_score * 0.3 + accuracy_score * 0.3)
                
                model_performance[model_name].append({
                    'composite_score': composite_score,
                    'time_s': result.total_time_s,
                    'memory_gb': result.peak_memory_gb,
                    'accuracy': accuracy_score
                })
        
        # 计算每个模型的平均性能
        model_averages = {}
        for model, performances in model_performance.items():
            model_averages[model] = {
                'avg_composite_score': np.mean([p['composite_score'] for p in performances]),
                'avg_time_s': np.mean([p['time_s'] for p in performances]),
                'avg_memory_gb': np.mean([p['memory_gb'] for p in performances]),
                'avg_accuracy': np.mean([p['accuracy'] for p in performances])
            }
        
        # 按综合分数排序
        ranked_models = sorted(model_averages.items(), 
                             key=lambda x: x[1]['avg_composite_score'], 
                             reverse=True)
        
        return {
            'ranking': ranked_models,
            'best_model': ranked_models[0] if ranked_models else None,
            'performance_details': dict(ranked_models)
        }
    
    def _analyze_system_resources(self) -> Dict[str, Any]:
        """分析系统资源使用"""
        analysis = {
            'cpu_info': {
                'logical_cores': psutil.cpu_count(logical=True),
                'physical_cores': psutil.cpu_count(logical=False),
                'current_utilization': psutil.cpu_percent(interval=1)
            },
            'memory_info': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'current_utilization': psutil.virtual_memory().percent
            },
            'gpu_info': {},
            'recommendations': []
        }
        
        if torch.cuda.is_available():
            analysis['gpu_info'] = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'allocated_memory_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_memory_gb': torch.cuda.memory_reserved() / (1024**3)
            }
            
            # GPU资源建议
            gpu_utilization = analysis['gpu_info']['allocated_memory_gb'] / analysis['gpu_info']['total_memory_gb']
            if gpu_utilization > 0.9:
                analysis['recommendations'].append("GPU内存使用率过高，建议启用混合精度训练")
            elif gpu_utilization < 0.3:
                analysis['recommendations'].append("GPU内存使用率较低，可以增加批处理大小")
        
        # CPU资源建议  
        if analysis['cpu_info']['current_utilization'] > 80:
            analysis['recommendations'].append("CPU使用率较高，建议减少并行处理线程数")
        
        # 内存建议
        memory_utilization = analysis['memory_info']['current_utilization']
        if memory_utilization > 85:
            analysis['recommendations'].append("系统内存使用率较高，建议减少批处理大小")
        
        return analysis
    
    def _generate_optimization_recommendations(self, comparison_report: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        summary = comparison_report['summary']
        
        # 速度优化建议
        if summary['avg_speed_improvement'] < 3.0:
            recommendations.append("当前速度提升未达到3x目标，建议：启用混合精度训练、使用OneCycle学习率调度")
        
        # 内存优化建议
        if summary['avg_memory_reduction'] < 40.0:
            recommendations.append("当前内存减少未达到40%目标，建议：启用梯度检查点、优化批处理大小")
        
        # 准确性建议
        if not summary['performance_targets_met']['accuracy_maintained']:
            recommendations.append("准确性有所下降，建议：调整混合精度训练参数、使用更保守的优化策略")
        
        # 模型特定建议
        if comparison_report.get('detailed_results'):
            for config, results in comparison_report['detailed_results'].items():
                if 'tcn' in config and results['speed_improvement'] < 2.0:
                    recommendations.append("TCN模型速度提升不足，建议：减少序列长度、使用线性注意力机制")
        
        # 硬件特定建议
        recommendations.append("针对RTX 4060 8GB：建议批处理大小不超过512，启用混合精度训练，使用梯度累积")
        
        return recommendations
    
    def save_benchmark_results(self, save_dir: str):
        """保存基准测试结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_path = save_path / 'benchmark_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            serializable_results = [asdict(result) for result in self.benchmark_results]
            json.dump(serializable_results, f, indent=2, default=str)
        
        # 保存对比报告
        comparison_path = save_path / 'performance_comparison.json'
        comparison_report = self.comparison.generate_comparison_report()
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        self.logger.info(f"基准测试结果已保存到: {save_path}")
    
    def visualize_benchmark_results(self, save_path: Optional[str] = None):
        """可视化基准测试结果"""
        if not self.benchmark_results:
            self.logger.warning("没有基准测试结果可可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 优化级别vs执行时间
        ax1 = axes[0, 0]
        opt_levels = [r.optimization_level for r in self.benchmark_results]
        times = [r.total_time_s for r in self.benchmark_results]
        
        opt_level_data = defaultdict(list)
        for opt, time in zip(opt_levels, times):
            opt_level_data[opt].append(time)
        
        ax1.boxplot([opt_level_data[level] for level in opt_level_data.keys()], 
                   labels=list(opt_level_data.keys()))
        ax1.set_title('优化级别 vs 执行时间')
        ax1.set_ylabel('执行时间 (秒)')
        
        # 2. 内存使用分布
        ax2 = axes[0, 1]
        memories = [r.peak_memory_gb for r in self.benchmark_results]
        ax2.hist(memories, bins=10, alpha=0.7)
        ax2.set_title('峰值内存使用分布')
        ax2.set_xlabel('内存使用 (GB)')
        ax2.set_ylabel('频次')
        
        # 3. 模型类型性能对比
        ax3 = axes[1, 0]
        model_types = []
        model_times = []
        for result in self.benchmark_results:
            model_type = result.pipeline_name.split('_')[0]
            model_types.append(model_type)
            model_times.append(result.total_time_s)
        
        model_data = defaultdict(list)
        for model, time in zip(model_types, model_times):
            model_data[model].append(time)
        
        if model_data:
            ax3.boxplot([model_data[model] for model in model_data.keys()], 
                       labels=list(model_data.keys()))
            ax3.set_title('模型类型 vs 执行时间')
            ax3.set_ylabel('执行时间 (秒)')
        
        # 4. 数据规模vs性能
        ax4 = axes[1, 1]
        data_sizes = [r.data_size for r in self.benchmark_results]
        ax4.scatter(data_sizes, times, alpha=0.6)
        ax4.set_title('数据规模 vs 执行时间')
        ax4.set_xlabel('数据样本数')
        ax4.set_ylabel('执行时间 (秒)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()


# 便捷函数
def create_benchmark_suite(config: Optional[BenchmarkConfig] = None,
                         logger: Optional[logging.Logger] = None) -> EndToEndBenchmarkSuite:
    """创建基准测试套件实例"""
    return EndToEndBenchmarkSuite(config, logger)


def run_quick_benchmark(data_sizes: List[int] = [1000, 5000],
                       models: List[str] = ['hgbr', 'tcn'],
                       logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """运行快速基准测试"""
    config = BenchmarkConfig(
        data_size_variants=data_sizes,
        feature_size_variants=[50],
        test_models=models,
        optimization_levels=['baseline', 'optimized'],
        n_runs_per_config=1
    )
    
    suite = create_benchmark_suite(config, logger)
    return suite.run_comprehensive_benchmark()


def compare_optimization_strategies(X: np.ndarray, y: np.ndarray,
                                  logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """对比优化策略性能"""
    
    results = {}
    data_size, feature_size = X.shape
    
    # 基准版本
    baseline_config = UnifiedTrainingConfig(
        model_type='hgbr',
        mixed_precision=False,
        onecycle_lr=False,
        use_purged_validation=False
    )
    
    baseline_trainer = UnifiedTrainingFramework(baseline_config, logger)
    start_time = time.time()
    baseline_result = baseline_trainer.train(X, y)
    baseline_time = time.time() - start_time
    
    results['baseline'] = {
        'training_time': baseline_time,
        'result': baseline_result
    }
    
    # 优化版本
    optimized_config = UnifiedTrainingConfig(
        model_type='hgbr',
        mixed_precision=True,
        onecycle_lr=True,
        use_purged_validation=True
    )
    
    optimized_trainer = UnifiedTrainingFramework(optimized_config, logger)
    start_time = time.time()
    optimized_result = optimized_trainer.train(X, y)
    optimized_time = time.time() - start_time
    
    results['optimized'] = {
        'training_time': optimized_time,
        'result': optimized_result
    }
    
    # 性能对比
    speed_improvement = baseline_time / optimized_time
    
    results['comparison'] = {
        'speed_improvement': speed_improvement,
        'baseline_time': baseline_time,
        'optimized_time': optimized_time,
        'meets_speed_target': speed_improvement >= 3.0
    }
    
    return results