"""
集成并行处理器与向量化特征提取
Day 3 优化集成模块
"""
import numpy as np
import pandas as pd
import time
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import warnings

from .vectorized_extraction import VectorizedFeatureExtractor, BatchFeatureProcessor
from .parallel_processor import MemoryEfficientParallelProcessor, AdaptiveParallelProcessor
from ..monitoring.resource_monitor import ResourceMonitor, ProcessResourceTracker


class OptimizedFeatureExtractor:
    """
    Day 1-3 优化集成的特征提取器
    结合数值稳定性、向量化处理和并行优化
    """

    def __init__(self,
                 window_size=180,
                 stride=30,
                 enable_parallel=True,
                 max_workers=None,
                 memory_efficient=True,
                 enable_monitoring=True,
                 cache_enabled=True):
        """
        初始化优化特征提取器

        Args:
            window_size: 窗口大小
            stride: 窗口步长
            enable_parallel: 是否启用并行处理
            max_workers: 最大工作线程数
            memory_efficient: 是否启用内存优化
            enable_monitoring: 是否启用资源监控
            cache_enabled: 是否启用特征缓存
        """
        self.window_size = window_size
        self.stride = stride
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.enable_monitoring = enable_monitoring

        # 核心处理器
        if enable_parallel:
            self.parallel_processor = MemoryEfficientParallelProcessor(
                window_size=window_size,
                stride=stride,
                max_workers=max_workers,
                chunk_size_method='adaptive' if memory_efficient else 'fixed'
            )
            self.adaptive_processor = AdaptiveParallelProcessor(self.parallel_processor)
        else:
            # 回退到向量化处理器
            self.batch_processor = BatchFeatureProcessor(
                window_size=window_size,
                stride=stride,
                use_cache=cache_enabled
            )

        # 资源监控
        if enable_monitoring:
            self.resource_monitor = ResourceMonitor(
                monitor_interval=0.5,
                history_size=1000,
                enable_alerts=True
            )
        else:
            self.resource_monitor = None

        # 性能统计
        self.performance_history = {
            'extraction_times': [],
            'speedup_factors': [],
            'memory_usage': [],
            'data_sizes': [],
            'processing_modes': []
        }

        print(f"优化特征提取器初始化:")
        print(f"  并行处理: {'启用' if enable_parallel else '禁用'}")
        print(f"  内存优化: {'启用' if memory_efficient else '禁用'}")
        print(f"  资源监控: {'启用' if enable_monitoring else '禁用'}")
        print(f"  窗口大小: {window_size}, 步长: {stride}")

    def extract_features(self,
                        data: np.ndarray,
                        adaptive_mode: bool = True,
                        progress_callback: Optional[callable] = None) -> Tuple[np.ndarray, List[str]]:
        """
        智能特征提取：根据数据特征选择最优处理策略

        Args:
            data: 输入数据 (n_samples, n_features)
            adaptive_mode: 是否启用自适应模式
            progress_callback: 进度回调函数

        Returns:
            features: 提取的特征
            feature_names: 特征名称列表
        """
        print(f"\n=== 启动优化特征提取 ===")
        print(f"数据形状: {data.shape}")

        # 启动资源监控
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()

        start_time = time.time()

        try:
            # 智能选择处理策略
            processing_strategy = self._select_processing_strategy(data, adaptive_mode)
            print(f"选择处理策略: {processing_strategy}")

            with ProcessResourceTracker().context_manager() as process_tracker:
                # 根据策略执行特征提取
                if processing_strategy == 'parallel_adaptive':
                    features, feature_names = self.adaptive_processor.process_with_adaptation(data)
                elif processing_strategy == 'parallel_fixed':
                    features, feature_names = self.parallel_processor.process_parallel(data, progress_callback)
                elif processing_strategy == 'vectorized_batch':
                    features, feature_names = self.batch_processor.process_with_caching(data)
                else:
                    # 回退到基础向量化
                    extractor = VectorizedFeatureExtractor(self.window_size, self.stride)
                    features, feature_names = extractor.extract_all_windows_vectorized(data)

        finally:
            # 停止资源监控
            if self.resource_monitor:
                time.sleep(1)  # 让监控收集最后的数据点
                self.resource_monitor.stop_monitoring()

        # 记录性能统计
        total_time = time.time() - start_time
        self._record_performance_stats(data.shape, total_time, processing_strategy)

        print(f"\n特征提取完成:")
        print(f"  输出形状: {features.shape}")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  处理策略: {processing_strategy}")

        return features, feature_names

    def _select_processing_strategy(self, data: np.ndarray, adaptive_mode: bool) -> str:
        """
        智能选择处理策略

        Args:
            data: 输入数据
            adaptive_mode: 是否启用自适应模式

        Returns:
            strategy: 处理策略名称
        """
        n_samples, n_features = data.shape
        data_size_mb = data.nbytes / 1024 / 1024
        n_windows = (n_samples - self.window_size) // self.stride + 1

        print(f"策略选择分析:")
        print(f"  数据大小: {data_size_mb:.1f}MB")
        print(f"  窗口数量: {n_windows}")
        print(f"  特征维度: {n_features}")

        # 获取系统资源状态
        current_resources = {}
        if self.resource_monitor:
            current_resources = self.resource_monitor.get_current_stats()
        else:
            import psutil
            current_resources = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available': psutil.virtual_memory().available
            }

        cpu_usage = current_resources.get('cpu_percent', 0)
        memory_usage = current_resources.get('memory_percent', 0)
        available_memory_gb = current_resources.get('memory_available', 0) / (1024**3)

        print(f"  CPU使用率: {cpu_usage:.1f}%")
        print(f"  内存使用率: {memory_usage:.1f}%")
        print(f"  可用内存: {available_memory_gb:.1f}GB")

        # 决策逻辑
        if not self.enable_parallel:
            return 'vectorized_basic'

        # 大数据集优先使用并行处理
        if data_size_mb > 100 and n_windows > 1000:
            if adaptive_mode and (cpu_usage > 80 or memory_usage > 80):
                return 'parallel_adaptive'
            else:
                return 'parallel_fixed'

        # 中等数据集：根据系统负载决定
        if data_size_mb > 20 and n_windows > 100:
            if cpu_usage < 50 and memory_usage < 70:
                return 'parallel_fixed'
            else:
                return 'vectorized_batch'

        # 小数据集：使用向量化批处理
        return 'vectorized_batch'

    def _record_performance_stats(self, data_shape: Tuple, processing_time: float, strategy: str):
        """记录性能统计"""
        data_size_mb = (data_shape[0] * data_shape[1] * 8) / 1024 / 1024  # 假设float64

        self.performance_history['extraction_times'].append(processing_time)
        self.performance_history['data_sizes'].append(data_size_mb)
        self.performance_history['processing_modes'].append(strategy)

        # 计算加速比（相对于基准性能）
        baseline_time_per_mb = 0.5  # 秒/MB 基准（需要根据实际情况调整）
        expected_time = data_size_mb * baseline_time_per_mb
        speedup = expected_time / processing_time if processing_time > 0 else 1.0

        self.performance_history['speedup_factors'].append(speedup)
        self.performance_history['memory_usage'].append(0)  # 稍后从监控器获取

    def benchmark_processing_modes(self, test_data: np.ndarray) -> Dict[str, Dict]:
        """
        基准测试不同处理模式的性能

        Args:
            test_data: 测试数据

        Returns:
            benchmark_results: 各模式的性能指标
        """
        print(f"\n=== 处理模式基准测试 ===")
        print(f"测试数据形状: {test_data.shape}")

        results = {}

        # 测试向量化基础模式
        print("测试向量化基础模式...")
        start_time = time.time()
        extractor = VectorizedFeatureExtractor(self.window_size, self.stride)
        features_basic, _ = extractor.extract_all_windows_vectorized(test_data)
        basic_time = time.time() - start_time

        results['vectorized_basic'] = {
            'processing_time': basic_time,
            'features_shape': features_basic.shape,
            'mode': 'vectorized_basic'
        }

        # 测试向量化批处理模式
        if hasattr(self, 'batch_processor'):
            print("测试向量化批处理模式...")
            start_time = time.time()
            features_batch, _ = self.batch_processor.process_with_caching(test_data, cache_key='benchmark')
            batch_time = time.time() - start_time

            results['vectorized_batch'] = {
                'processing_time': batch_time,
                'features_shape': features_batch.shape,
                'mode': 'vectorized_batch'
            }

        # 测试并行处理模式（如果启用）
        if self.enable_parallel:
            print("测试并行处理模式...")
            start_time = time.time()
            features_parallel, _ = self.parallel_processor.process_parallel(test_data)
            parallel_time = time.time() - start_time

            results['parallel_fixed'] = {
                'processing_time': parallel_time,
                'features_shape': features_parallel.shape,
                'mode': 'parallel_fixed',
                'parallel_stats': self.parallel_processor.get_performance_stats()
            }

            # 测试自适应并行模式
            print("测试自适应并行模式...")
            start_time = time.time()
            features_adaptive, _ = self.adaptive_processor.process_with_adaptation(test_data)
            adaptive_time = time.time() - start_time

            results['parallel_adaptive'] = {
                'processing_time': adaptive_time,
                'features_shape': features_adaptive.shape,
                'mode': 'parallel_adaptive'
            }

        # 计算相对性能
        baseline_time = results['vectorized_basic']['processing_time']
        for mode, result in results.items():
            result['speedup_vs_basic'] = baseline_time / result['processing_time']
            result['efficiency'] = result['speedup_vs_basic']

        # 打印结果摘要
        print(f"\n=== 基准测试结果 ===")
        for mode, result in results.items():
            print(f"{mode:20s}: {result['processing_time']:6.2f}s (加速比: {result['speedup_vs_basic']:4.1f}x)")

        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.performance_history['extraction_times']:
            return {"error": "没有性能数据"}

        report = {
            'total_extractions': len(self.performance_history['extraction_times']),
            'average_extraction_time': np.mean(self.performance_history['extraction_times']),
            'total_processing_time': sum(self.performance_history['extraction_times']),
            'average_speedup': np.mean(self.performance_history['speedup_factors']),
            'processing_mode_distribution': {}
        }

        # 处理模式分布
        from collections import Counter
        mode_counts = Counter(self.performance_history['processing_modes'])
        total_runs = len(self.performance_history['processing_modes'])

        for mode, count in mode_counts.items():
            report['processing_mode_distribution'][mode] = {
                'count': count,
                'percentage': (count / total_runs) * 100
            }

        # 资源监控摘要（如果可用）
        if self.resource_monitor:
            resource_summary = self.resource_monitor.get_resource_summary()
            report['resource_usage'] = resource_summary

        return report

    def export_performance_data(self, filepath: str):
        """导出性能数据"""
        report = self.get_performance_report()

        # 添加详细历史数据
        report['detailed_history'] = {
            'extraction_times': self.performance_history['extraction_times'],
            'speedup_factors': self.performance_history['speedup_factors'],
            'data_sizes_mb': self.performance_history['data_sizes'],
            'processing_modes': self.performance_history['processing_modes']
        }

        # 资源监控历史数据
        if self.resource_monitor:
            report['resource_history'] = self.resource_monitor.get_historical_stats()

        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"性能数据已导出至: {filepath}")

    def validate_optimization_gains(self, baseline_time: Optional[float] = None) -> Dict[str, Any]:
        """
        验证优化效果

        Args:
            baseline_time: 基线时间（如果未提供，使用第一次运行作为基线）

        Returns:
            validation_results: 验证结果
        """
        if not self.performance_history['extraction_times']:
            return {"error": "没有性能数据进行验证"}

        times = self.performance_history['extraction_times']
        speedups = self.performance_history['speedup_factors']

        if baseline_time is None:
            baseline_time = times[0]

        current_avg = np.mean(times[-5:]) if len(times) >= 5 else np.mean(times)
        improvement = (baseline_time - current_avg) / baseline_time * 100

        validation = {
            'baseline_time': baseline_time,
            'current_average_time': current_avg,
            'improvement_percentage': improvement,
            'average_speedup': np.mean(speedups),
            'max_speedup': np.max(speedups),
            'consistency_score': 100 - (np.std(times) / np.mean(times)) * 100,
            'optimization_target_met': improvement >= 25.0,  # Day 3目标：25-30%提升
            'recommendations': []
        }

        # 生成建议
        if improvement < 25:
            validation['recommendations'].append("优化效果未达到目标25%，建议调整并行参数")
        if np.std(speedups) > 0.5:
            validation['recommendations'].append("性能波动较大，建议启用自适应模式")
        if current_avg > 10.0:  # 假设10秒为合理上限
            validation['recommendations'].append("处理时间仍较长，考虑增加并行度或优化数据分块")

        return validation


# 使用示例和综合测试
if __name__ == '__main__':
    print("=== 优化特征提取器综合测试 ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 30000  # 中等大小数据
    n_features = 6
    test_data = np.random.randn(n_samples, n_features).astype(np.float32)

    # 添加一些真实的模式
    t = np.linspace(0, 10*np.pi, n_samples)
    for i in range(n_features):
        test_data[:, i] += np.sin((i+1) * t) * np.exp(-t/10000) * (i+1)

    print(f"测试数据: 形状{test_data.shape}, 大小{test_data.nbytes/1024**2:.1f}MB")

    # 创建优化特征提取器
    optimizer = OptimizedFeatureExtractor(
        window_size=180,
        stride=30,
        enable_parallel=True,
        memory_efficient=True,
        enable_monitoring=True,
        cache_enabled=True
    )

    # 基准测试所有处理模式
    benchmark_results = optimizer.benchmark_processing_modes(test_data)

    # 多次提取测试一致性
    print(f"\n=== 一致性测试 ===")
    for i in range(3):
        print(f"运行 {i+1}/3...")
        features, feature_names = optimizer.extract_features(test_data, adaptive_mode=True)
        print(f"  特征形状: {features.shape}, 特征数: {len(feature_names)}")

    # 性能报告
    report = optimizer.get_performance_report()
    print(f"\n=== 性能报告 ===")
    print(f"总提取次数: {report['total_extractions']}")
    print(f"平均提取时间: {report['average_extraction_time']:.2f}秒")
    print(f"平均加速比: {report['average_speedup']:.2f}x")

    print(f"处理模式分布:")
    for mode, stats in report['processing_mode_distribution'].items():
        print(f"  {mode}: {stats['count']}次 ({stats['percentage']:.1f}%)")

    # 验证优化效果
    validation = optimizer.validate_optimization_gains()
    print(f"\n=== 优化验证 ===")
    print(f"性能提升: {validation['improvement_percentage']:.1f}%")
    print(f"平均加速比: {validation['average_speedup']:.2f}x")
    print(f"目标达成: {'✅' if validation['optimization_target_met'] else '❌'}")

    if validation['recommendations']:
        print("建议:")
        for rec in validation['recommendations']:
            print(f"  • {rec}")

    # 导出性能数据
    optimizer.export_performance_data("optimization_performance_report.json")

    print(f"\n=== 综合测试完成 ===")