"""
自动化性能基准测试套件 - Week 2 优化
提供全面的性能对比和基准测试功能
"""

import time
import numpy as np
import pandas as pd
import psutil
import warnings
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime


class PerformanceBenchmark:
    """
    性能基准测试套件
    对比优化前后的性能差异，生成详细的基准报告
    """

    def __init__(self, test_data_sizes: List[int] = None, output_dir: str = "benchmark_results"):
        """
        初始化性能基准测试

        Args:
            test_data_sizes: 测试数据大小列表
            output_dir: 输出目录
        """
        self.test_data_sizes = test_data_sizes or [1000, 5000, 10000, 20000]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 测试结果存储
        self.baseline_results = {}
        self.optimized_results = {}
        self.comparison_results = {}

        # 资源监控
        self.resource_monitor = ResourceMonitor()

        print(f"  性能基准测试初始化: 测试大小={self.test_data_sizes}")

    def run_complete_benchmark(self, baseline_extractor: Callable, optimized_extractor: Callable,
                             test_data_generator: Callable = None) -> Dict[str, Any]:
        """
        运行完整的基准测试

        Args:
            baseline_extractor: 基线特征提取器
            optimized_extractor: 优化后特征提取器
            test_data_generator: 测试数据生成器

        Returns:
            完整的基准测试结果
        """
        print("=== 开始完整性能基准测试 ===")

        if test_data_generator is None:
            test_data_generator = self._default_data_generator

        benchmark_start = time.time()

        # 1. 运行基线测试
        print("\n--- 基线版本测试 ---")
        self.baseline_results = self._run_extractor_benchmark(
            baseline_extractor, test_data_generator, "baseline"
        )

        # 2. 运行优化版本测试
        print("\n--- 优化版本测试 ---")
        self.optimized_results = self._run_extractor_benchmark(
            optimized_extractor, test_data_generator, "optimized"
        )

        # 3. 生成对比分析
        print("\n--- 生成对比分析 ---")
        self.comparison_results = self._generate_comparison_analysis()

        # 4. 生成报告
        total_benchmark_time = time.time() - benchmark_start
        report = self._generate_comprehensive_report(total_benchmark_time)

        # 5. 保存结果
        self._save_benchmark_results()

        # 6. 生成可视化
        self._generate_visualizations()

        print(f"\n=== 基准测试完成，总用时: {total_benchmark_time:.2f}秒 ===")

        return {
            'baseline_results': self.baseline_results,
            'optimized_results': self.optimized_results,
            'comparison_results': self.comparison_results,
            'report': report
        }

    def _default_data_generator(self, size: int, n_features: int = 6) -> np.ndarray:
        """默认测试数据生成器"""
        np.random.seed(42)  # 确保结果可重复

        # 生成具有时序特征的数据
        t = np.linspace(0, 10*np.pi, size)
        data = np.zeros((size, n_features))

        # 添加不同类型的信号
        data[:, 0] = np.sin(t) + 0.1 * np.random.randn(size)  # 正弦信号
        data[:, 1] = np.cos(2*t) + 0.1 * np.random.randn(size)  # 余弦信号
        data[:, 2] = np.cumsum(np.random.randn(size)) * 0.1  # 随机游走
        data[:, 3] = np.where(np.random.rand(size) > 0.95, 5, 0) + np.random.randn(size) * 0.1  # 稀疏脉冲
        data[:, 4] = np.linspace(0, 5, size) + np.random.randn(size) * 0.2  # 线性趋势
        data[:, 5] = np.random.randn(size)  # 白噪声

        return data

    def _run_extractor_benchmark(self, extractor: Callable, data_generator: Callable,
                                version_name: str) -> Dict[str, Any]:
        """
        运行单个提取器的基准测试

        Args:
            extractor: 特征提取器函数
            data_generator: 数据生成器
            version_name: 版本名称

        Returns:
            基准测试结果
        """
        results = {}

        for data_size in self.test_data_sizes:
            print(f"    测试数据大小: {data_size}")

            # 生成测试数据
            test_data = data_generator(data_size)

            # 启动资源监控
            self.resource_monitor.start_monitoring()

            try:
                # 预热运行
                _ = extractor(test_data[:min(100, data_size)])

                # 正式测试（多次运行取平均值）
                run_times = []
                memory_peaks = []
                cpu_peaks = []

                n_runs = 3 if data_size <= 10000 else 1  # 大数据集只运行1次

                for run_idx in range(n_runs):
                    print(f"      第 {run_idx + 1}/{n_runs} 次运行...")

                    # 重置资源监控
                    self.resource_monitor.reset_stats()

                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024**2

                    # 执行特征提取
                    features = extractor(test_data)

                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024**2

                    # 记录性能指标
                    run_time = end_time - start_time
                    memory_usage = end_memory - start_memory

                    run_times.append(run_time)
                    memory_peaks.append(memory_usage)

                    # 获取资源监控结果
                    monitor_stats = self.resource_monitor.get_summary()
                    cpu_peaks.append(monitor_stats.get('max_cpu', 0))

                    # 验证特征形状
                    if hasattr(features, 'shape'):
                        feature_shape = features.shape
                    elif isinstance(features, (list, tuple)):
                        feature_shape = (len(features[0]) if features else 0, len(features) if features else 0)
                    else:
                        feature_shape = (0, 0)

                # 计算统计量
                avg_time = np.mean(run_times)
                std_time = np.std(run_times)
                avg_memory = np.mean(memory_peaks)
                max_cpu = np.max(cpu_peaks)

                results[data_size] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'min_time': np.min(run_times),
                    'max_time': np.max(run_times),
                    'all_times': run_times,
                    'avg_memory_mb': avg_memory,
                    'max_cpu_percent': max_cpu,
                    'throughput_samples_per_sec': data_size / avg_time,
                    'feature_shape': feature_shape,
                    'data_shape': test_data.shape,
                    'runs_count': n_runs
                }

                print(f"      平均用时: {avg_time:.3f}±{std_time:.3f}秒")
                print(f"      内存使用: {avg_memory:.1f}MB")
                print(f"      CPU峰值: {max_cpu:.1f}%")

            except Exception as e:
                print(f"      ❌ 测试失败: {e}")
                results[data_size] = {
                    'error': str(e),
                    'avg_time': float('inf'),
                    'failed': True
                }

            finally:
                self.resource_monitor.stop_monitoring()

        return results

    def _generate_comparison_analysis(self) -> Dict[str, Any]:
        """生成对比分析结果"""
        comparison = {}

        for data_size in self.test_data_sizes:
            baseline = self.baseline_results.get(data_size, {})
            optimized = self.optimized_results.get(data_size, {})

            if baseline.get('failed') or optimized.get('failed'):
                comparison[data_size] = {'error': 'One or both tests failed'}
                continue

            baseline_time = baseline.get('avg_time', float('inf'))
            optimized_time = optimized.get('avg_time', float('inf'))

            if baseline_time == 0 or optimized_time == 0:
                speedup = 1.0
            else:
                speedup = baseline_time / optimized_time

            time_saved = baseline_time - optimized_time
            efficiency_gain = ((baseline_time - optimized_time) / baseline_time) * 100

            baseline_throughput = baseline.get('throughput_samples_per_sec', 0)
            optimized_throughput = optimized.get('throughput_samples_per_sec', 0)

            throughput_improvement = (optimized_throughput - baseline_throughput) / max(baseline_throughput, 1) * 100

            comparison[data_size] = {
                'speedup': speedup,
                'time_saved_seconds': time_saved,
                'efficiency_gain_percent': efficiency_gain,
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'baseline_throughput': baseline_throughput,
                'optimized_throughput': optimized_throughput,
                'throughput_improvement_percent': throughput_improvement,
                'memory_comparison': {
                    'baseline_mb': baseline.get('avg_memory_mb', 0),
                    'optimized_mb': optimized.get('avg_memory_mb', 0),
                    'memory_reduction_mb': baseline.get('avg_memory_mb', 0) - optimized.get('avg_memory_mb', 0)
                },
                'cpu_comparison': {
                    'baseline_max_cpu': baseline.get('max_cpu_percent', 0),
                    'optimized_max_cpu': optimized.get('max_cpu_percent', 0)
                }
            }

        return comparison

    def _generate_comprehensive_report(self, total_time: float) -> str:
        """生成综合性能报告"""
        report_lines = [
            "=" * 80,
            "LNG项目优化性能基准测试报告",
            "=" * 80,
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"总测试用时: {total_time:.2f}秒",
            f"测试数据规模: {self.test_data_sizes}",
            "",
            "## 性能对比总结",
            ""
        ]

        # 汇总统计
        all_speedups = []
        all_efficiency_gains = []
        all_throughput_improvements = []

        valid_comparisons = []
        for data_size, comp in self.comparison_results.items():
            if 'error' not in comp:
                valid_comparisons.append((data_size, comp))
                all_speedups.append(comp['speedup'])
                all_efficiency_gains.append(comp['efficiency_gain_percent'])
                all_throughput_improvements.append(comp['throughput_improvement_percent'])

        if valid_comparisons:
            avg_speedup = np.mean(all_speedups)
            avg_efficiency = np.mean(all_efficiency_gains)
            avg_throughput_improvement = np.mean(all_throughput_improvements)

            report_lines.extend([
                f"平均加速比: {avg_speedup:.2f}x",
                f"平均效率提升: {avg_efficiency:.1f}%",
                f"平均吞吐量提升: {avg_throughput_improvement:.1f}%",
                "",
                "## 详细性能数据",
                "",
                f"{'数据规模':<10} {'基线用时':<12} {'优化用时':<12} {'加速比':<10} {'效率提升':<12} {'吞吐量提升':<12}",
                "-" * 80
            ])

            for data_size, comp in valid_comparisons:
                report_lines.append(
                    f"{data_size:<10} "
                    f"{comp['baseline_time']:<12.3f} "
                    f"{comp['optimized_time']:<12.3f} "
                    f"{comp['speedup']:<10.2f} "
                    f"{comp['efficiency_gain_percent']:<12.1f}% "
                    f"{comp['throughput_improvement_percent']:<12.1f}%"
                )

            report_lines.extend([
                "",
                "## 内存使用对比",
                ""
            ])

            for data_size, comp in valid_comparisons:
                mem_comp = comp['memory_comparison']
                report_lines.append(
                    f"数据规模 {data_size}: "
                    f"基线 {mem_comp['baseline_mb']:.1f}MB → "
                    f"优化 {mem_comp['optimized_mb']:.1f}MB "
                    f"(节省 {mem_comp['memory_reduction_mb']:.1f}MB)"
                )

            report_lines.extend([
                "",
                "## 性能评估",
                ""
            ])

            if avg_speedup >= 2.0:
                performance_rating = "🚀 优秀 (>2x)"
            elif avg_speedup >= 1.5:
                performance_rating = "✅ 良好 (1.5-2x)"
            elif avg_speedup >= 1.2:
                performance_rating = "👍 满意 (1.2-1.5x)"
            elif avg_speedup >= 1.0:
                performance_rating = "⚠️  一般 (1.0-1.2x)"
            else:
                performance_rating = "❌ 性能下降"

            report_lines.extend([
                f"整体性能评级: {performance_rating}",
                f"推荐部署: {'是' if avg_speedup >= 1.2 else '需要进一步优化'}",
                "",
                "## 生产环境预期收益",
                ""
            ])

            # 基于最大数据规模预估生产环境收益
            largest_test = valid_comparisons[-1][1] if valid_comparisons else None
            if largest_test:
                production_estimate = self._estimate_production_benefits(largest_test)
                report_lines.extend(production_estimate)

        else:
            report_lines.extend([
                "❌ 所有测试均失败，无法生成对比报告",
                ""
            ])

        report_lines.extend([
            "",
            "=" * 80,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])

        return "\\n".join(report_lines)

    def _estimate_production_benefits(self, largest_comparison: Dict) -> List[str]:
        """基于测试结果估算生产环境收益"""
        speedup = largest_comparison['speedup']
        time_saved = largest_comparison['time_saved_seconds']

        # 假设原特征提取时间为82分钟（基于项目文档）
        original_time_minutes = 82
        optimized_time_minutes = original_time_minutes / speedup

        time_saved_minutes = original_time_minutes - optimized_time_minutes

        # 假设原项目总时间为2.3小时
        original_total_hours = 2.3
        total_time_saved_hours = time_saved_minutes / 60
        new_total_hours = original_total_hours - total_time_saved_hours

        percentage_improvement = (total_time_saved_hours / original_total_hours) * 100

        return [
            f"基于测试结果预估生产环境收益:",
            f"- 原特征提取时间: {original_time_minutes}分钟",
            f"- 优化后预估时间: {optimized_time_minutes:.1f}分钟",
            f"- 预计节省时间: {time_saved_minutes:.1f}分钟",
            f"- 原项目总时间: {original_total_hours}小时",
            f"- 优化后预估总时间: {new_total_hours:.1f}小时",
            f"- 总体改进预估: {percentage_improvement:.1f}%"
        ]

    def _save_benchmark_results(self):
        """保存基准测试结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存详细结果
        results_data = {
            'timestamp': timestamp,
            'test_data_sizes': self.test_data_sizes,
            'baseline_results': self.baseline_results,
            'optimized_results': self.optimized_results,
            'comparison_results': self.comparison_results
        }

        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"    详细结果已保存: {results_file}")

        # 保存报告
        report = self._generate_comprehensive_report(0)
        report_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"    基准报告已保存: {report_file}")

    def _generate_visualizations(self):
        """生成性能对比可视化图表"""
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('LNG项目优化性能基准测试结果', fontsize=16, fontweight='bold')

            # 准备数据
            data_sizes = []
            baseline_times = []
            optimized_times = []
            speedups = []
            throughput_baseline = []
            throughput_optimized = []

            for size, comp in self.comparison_results.items():
                if 'error' not in comp:
                    data_sizes.append(size)
                    baseline_times.append(comp['baseline_time'])
                    optimized_times.append(comp['optimized_time'])
                    speedups.append(comp['speedup'])
                    throughput_baseline.append(comp['baseline_throughput'])
                    throughput_optimized.append(comp['optimized_throughput'])

            if not data_sizes:
                print("    无有效数据，跳过可视化生成")
                return

            # 1. 运行时间对比
            axes[0, 0].plot(data_sizes, baseline_times, 'o-', label='基线版本', color='red', linewidth=2)
            axes[0, 0].plot(data_sizes, optimized_times, 'o-', label='优化版本', color='green', linewidth=2)
            axes[0, 0].set_xlabel('数据规模')
            axes[0, 0].set_ylabel('运行时间 (秒)')
            axes[0, 0].set_title('运行时间对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')

            # 2. 加速比
            axes[0, 1].plot(data_sizes, speedups, 'o-', color='blue', linewidth=2, markersize=8)
            axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='无改进线')
            axes[0, 1].set_xlabel('数据规模')
            axes[0, 1].set_ylabel('加速比')
            axes[0, 1].set_title('性能加速比')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 在每个点上标注加速比数值
            for i, (size, speedup) in enumerate(zip(data_sizes, speedups)):
                axes[0, 1].annotate(f'{speedup:.1f}x',
                                   (size, speedup),
                                   textcoords="offset points",
                                   xytext=(0, 10),
                                   ha='center')

            # 3. 吞吐量对比
            x_pos = np.arange(len(data_sizes))
            width = 0.35

            axes[1, 0].bar(x_pos - width/2, throughput_baseline, width, label='基线版本', color='red', alpha=0.7)
            axes[1, 0].bar(x_pos + width/2, throughput_optimized, width, label='优化版本', color='green', alpha=0.7)
            axes[1, 0].set_xlabel('数据规模')
            axes[1, 0].set_ylabel('吞吐量 (样本/秒)')
            axes[1, 0].set_title('处理吞吐量对比')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(data_sizes)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 效率提升百分比
            efficiency_gains = [comp['efficiency_gain_percent'] for comp in self.comparison_results.values() if 'error' not in comp]

            axes[1, 1].bar(range(len(data_sizes)), efficiency_gains, color='purple', alpha=0.7)
            axes[1, 1].set_xlabel('数据规模')
            axes[1, 1].set_ylabel('效率提升 (%)')
            axes[1, 1].set_title('效率提升百分比')
            axes[1, 1].set_xticks(range(len(data_sizes)))
            axes[1, 1].set_xticklabels(data_sizes)
            axes[1, 1].grid(True, alpha=0.3)

            # 在每个柱子上标注数值
            for i, gain in enumerate(efficiency_gains):
                axes[1, 1].annotate(f'{gain:.1f}%',
                                   (i, gain),
                                   textcoords="offset points",
                                   xytext=(0, 5),
                                   ha='center')

            plt.tight_layout()

            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_file = self.output_dir / f"benchmark_charts_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    性能图表已保存: {chart_file}")

        except Exception as e:
            print(f"    可视化生成失败: {e}")


class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.1):
        """开始监控系统资源"""
        if self.monitoring:
            return

        self.monitoring = True
        self.stats = {'cpu_usage': [], 'memory_usage': [], 'timestamps': []}

        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def reset_stats(self):
        """重置统计数据"""
        self.stats = {'cpu_usage': [], 'memory_usage': [], 'timestamps': []}

    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                self.stats['cpu_usage'].append(cpu_percent)
                self.stats['memory_usage'].append(memory_percent)
                self.stats['timestamps'].append(time.time())

                time.sleep(interval)
            except:
                break

    def get_summary(self) -> Dict[str, float]:
        """获取监控摘要"""
        if not self.stats['cpu_usage']:
            return {'avg_cpu': 0, 'max_cpu': 0, 'avg_memory': 0, 'max_memory': 0}

        return {
            'avg_cpu': np.mean(self.stats['cpu_usage']),
            'max_cpu': np.max(self.stats['cpu_usage']),
            'avg_memory': np.mean(self.stats['memory_usage']),
            'max_memory': np.max(self.stats['memory_usage'])
        }


# 测试代码
if __name__ == '__main__':
    print("=== 性能基准测试系统测试 ===")

    # 模拟特征提取函数
    def baseline_extractor(data):
        """模拟基线特征提取器（较慢）"""
        time.sleep(0.001 * len(data) / 1000)  # 模拟处理时间
        return np.random.randn(len(data) // 10, 20)

    def optimized_extractor(data):
        """模拟优化特征提取器（较快）"""
        time.sleep(0.0003 * len(data) / 1000)  # 更快的处理
        return np.random.randn(len(data) // 10, 20)

    # 创建基准测试
    benchmark = PerformanceBenchmark(
        test_data_sizes=[1000, 2000, 5000],
        output_dir="test_benchmark_results"
    )

    # 运行基准测试
    results = benchmark.run_complete_benchmark(
        baseline_extractor=baseline_extractor,
        optimized_extractor=optimized_extractor
    )

    print("\\n基准测试完成!")
    print(f"结果保存在: {benchmark.output_dir}")

    print("\\n=== 性能基准测试系统测试完成 ===")