"""
LNG能耗预测项目 - 主模块
基于技术路线的标准实现
"""

from .dataio import LNGDataLoader, load_lng_data
from .preprocessing import LNGPreprocessor
from .features import DynamicFeatureExtractor, StaticFeatureExtractor, LNGFeatureEngine, create_lng_features
from .models import (
    LNGTransformerFusion, MultipleLinearRegression, GaussianProcessRegression,
    AdaptiveHGBRBaseline, SampleSizeAdaptiveModelFactory,
    ResidualModelingFramework, OptimizedTCNModel
)
from .training import (
    UnifiedTrainingFramework, UnifiedTrainingConfig, create_unified_trainer, 
    quick_train, MixedPrecisionTrainer, OneCycleLR, PurgedTimeSeriesCV
)
from .eval import LNGEvaluator, EvaluationConfig, calculate_lng_metrics, MetricsReport
from .monitoring import PerformanceMonitoringSystem, create_performance_monitor
# 暂时移除基准测试模块以避免导入错误
# from .benchmarking import EndToEndBenchmarkSuite, create_benchmark_suite, run_quick_benchmark

__all__ = [
    # 数据I/O
    'LNGDataLoader',
    'load_lng_data',
    
    # 预处理
    'LNGPreprocessor',
    
    # 特征工程
    'DynamicFeatureExtractor',
    'StaticFeatureExtractor', 
    'LNGFeatureEngine',
    'create_lng_features',
    
    # 模型
    'LNGTransformerFusion',
    'MultipleLinearRegression',
    'GaussianProcessRegression',
    'AdaptiveHGBRBaseline',
    'SampleSizeAdaptiveModelFactory',
    'ResidualModelingFramework',
    'OptimizedTCNModel',
    
    # 训练优化
    'UnifiedTrainingFramework',
    'UnifiedTrainingConfig',
    'create_unified_trainer',
    'quick_train',
    'MixedPrecisionTrainer',
    'OneCycleLR',
    'PurgedTimeSeriesCV',
    
    # 评估
    'LNGEvaluator',
    'EvaluationConfig',
    'calculate_lng_metrics',
    'MetricsReport',
    
    # 监控系统
    'PerformanceMonitoringSystem',
    'create_performance_monitor'
    
    # 基准测试模块暂时移除
    # 'EndToEndBenchmarkSuite',
    # 'create_benchmark_suite',
    # 'run_quick_benchmark'
]