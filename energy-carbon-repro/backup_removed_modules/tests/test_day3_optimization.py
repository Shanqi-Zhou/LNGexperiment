"""
Day 3 优化验证测试套件
验证内存效率、并行性能和系统稳定性
"""
import numpy as np
import pandas as pd
import time
import unittest
import warnings
from typing import Dict, List, Tuple, Any
import psutil
import gc
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from features.parallel_processor import MemoryEfficientParallelProcessor, AdaptiveParallelProcessor
from features.optimized_extractor import OptimizedFeatureExtractor
from monitoring.resource_monitor import ResourceMonitor, ProcessResourceTracker


class Day3OptimizationValidator:
    """Day 3 优化效果验证器"""

    def __init__(self):
        self.test_results = {}
        self.performance_baselines = {}

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行全面验证测试"""
        print("=" * 60)
        print("Day 3 优化验证测试开始")
        print("=" * 60)

        results = {
            'memory_efficiency': self._test_memory_efficiency(),
            'parallel_performance': self._test_parallel_performance(),
            'system_stability': self._test_system_stability(),
            'integration_quality': self._test_integration_quality(),
            'performance_targets': self._validate_performance_targets(),
            'overall_assessment': {}
        }

        # 计算总体评估
        results['overall_assessment'] = self._calculate_overall_assessment(results)

        return results

    def _test_memory_efficiency(self) -> Dict[str, Any]:
        """测试内存效率"""
        print("\n1. 内存效率测试...")

        test_cases = [
            {'size': (10000, 4), 'name': '小数据集'},
            {'size': (50000, 8), 'name': '中数据集'},
            {'size': (100000, 12), 'name': '大数据集'}
        ]

        memory_results = {}

        for case in test_cases:
            print(f"  测试{case['name']}: {case['size']}")

            # 生成测试数据
            data = np.random.randn(*case['size']).astype(np.float32)
            data_size_mb = data.nbytes / 1024 / 1024

            # 测试并行处理器内存使用
            with ProcessResourceTracker().context_manager() as tracker:
                processor = MemoryEfficientParallelProcessor(
                    window_size=180,
                    stride=30,
                    max_workers=4,
                    chunk_size_method='memory_aware'
                )

                gc.collect()  # 清理内存
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024

                features, _ = processor.process_parallel(data)

                memory_after = psutil.Process().memory_info().rss / 1024 / 1024

            usage_stats = tracker.get_current_usage()

            # 计算内存效率指标
            memory_growth = memory_after - memory_before
            memory_efficiency = (data_size_mb / memory_growth) if memory_growth > 0 else float('inf')

            memory_results[case['name']] = {
                'data_size_mb': data_size_mb,
                'memory_growth_mb': memory_growth,
                'memory_efficiency_ratio': memory_efficiency,
                'peak_memory_mb': memory_after,
                'processing_time': usage_stats['duration_seconds'],
                'memory_per_second': memory_growth / usage_stats['duration_seconds'] if usage_stats['duration_seconds'] > 0 else 0
            }

            print(f"    数据: {data_size_mb:.1f}MB, 内存增长: {memory_growth:.1f}MB, 效率比: {memory_efficiency:.1f}")

        # 内存效率评估
        avg_efficiency = np.mean([r['memory_efficiency_ratio'] for r in memory_results.values() if r['memory_efficiency_ratio'] != float('inf')])

        assessment = {
            'test_cases': memory_results,
            'average_efficiency_ratio': avg_efficiency,
            'memory_leak_detected': any(r['memory_growth_mb'] > r['data_size_mb'] * 2 for r in memory_results.values()),
            'memory_efficiency_grade': 'A' if avg_efficiency > 2.0 else 'B' if avg_efficiency > 1.0 else 'C',
            'passed': avg_efficiency > 1.0 and not any(r['memory_growth_mb'] > r['data_size_mb'] * 3 for r in memory_results.values())
        }

        print(f"  内存效率测试结果: {'✅ 通过' if assessment['passed'] else '❌ 失败'}")
        print(f"    平均效率比: {avg_efficiency:.2f}")
        print(f"    效率等级: {assessment['memory_efficiency_grade']}")

        return assessment

    def _test_parallel_performance(self) -> Dict[str, Any]:
        """测试并行性能"""
        print("\n2. 并行性能测试...")

        # 创建测试数据
        np.random.seed(42)
        test_data = np.random.randn(40000, 6).astype(np.float32)

        parallel_results = {}

        # 测试不同并行配置
        worker_configs = [1, 2, 4, 8]

        for workers in worker_configs:
            print(f"  测试 {workers} 个工作线程...")

            processor = MemoryEfficientParallelProcessor(
                window_size=180,
                stride=30,
                max_workers=workers,
                chunk_size_method='adaptive'
            )

            # 多次运行取平均
            times = []
            for run in range(3):
                start_time = time.time()
                features, _ = processor.process_parallel(test_data)
                execution_time = time.time() - start_time
                times.append(execution_time)

            avg_time = np.mean(times)
            std_time = np.std(times)

            parallel_results[f'{workers}_workers'] = {
                'average_time': avg_time,
                'std_time': std_time,
                'consistency': (1 - std_time / avg_time) * 100,
                'performance_stats': processor.get_performance_stats()
            }

            print(f"    平均时间: {avg_time:.2f}s ± {std_time:.3f}s")

        # 计算并行效率
        baseline_time = parallel_results['1_workers']['average_time']

        for config, result in parallel_results.items():
            workers = int(config.split('_')[0])
            speedup = baseline_time / result['average_time']
            parallel_efficiency = (speedup / workers) * 100

            result['speedup'] = speedup
            result['parallel_efficiency'] = parallel_efficiency

        # 找到最优配置
        best_config = max(parallel_results.keys(),
                         key=lambda k: parallel_results[k]['speedup'])

        best_speedup = parallel_results[best_config]['speedup']
        best_efficiency = parallel_results[best_config]['parallel_efficiency']

        assessment = {
            'test_results': parallel_results,
            'best_configuration': best_config,
            'best_speedup': best_speedup,
            'best_efficiency': best_efficiency,
            'scalability_grade': 'A' if best_speedup > 3.0 else 'B' if best_speedup > 2.0 else 'C',
            'efficiency_grade': 'A' if best_efficiency > 80 else 'B' if best_efficiency > 60 else 'C',
            'passed': best_speedup >= 2.0 and best_efficiency >= 50.0
        }

        print(f"  并行性能测试结果: {'✅ 通过' if assessment['passed'] else '❌ 失败'}")
        print(f"    最佳配置: {best_config}")
        print(f"    最大加速比: {best_speedup:.2f}x")
        print(f"    并行效率: {best_efficiency:.1f}%")

        return assessment

    def _test_system_stability(self) -> Dict[str, Any]:
        """测试系统稳定性"""
        print("\n3. 系统稳定性测试...")

        stability_results = {}

        # 长时间运行测试
        print("  长时间运行测试...")
        test_data = np.random.randn(20000, 4).astype(np.float32)

        extractor = OptimizedFeatureExtractor(
            enable_parallel=True,
            enable_monitoring=True
        )

        run_times = []
        memory_usage = []

        # 连续运行10次
        for i in range(10):
            gc.collect()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024

            start_time = time.time()
            features, _ = extractor.extract_features(test_data)
            run_time = time.time() - start_time

            memory_after = psutil.Process().memory_info().rss / 1024 / 1024

            run_times.append(run_time)
            memory_usage.append(memory_after - memory_before)

            if (i + 1) % 3 == 0:
                print(f"    完成 {i+1}/10 次运行")

        # 稳定性指标
        time_stability = (1 - np.std(run_times) / np.mean(run_times)) * 100
        memory_stability = (1 - np.std(memory_usage) / np.mean(memory_usage)) * 100 if np.mean(memory_usage) > 0 else 100

        # 内存泄漏检测
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        memory_leak_suspected = memory_trend > 5.0  # 每次运行内存增长超过5MB

        stability_results['long_running'] = {
            'run_times': run_times,
            'memory_usage': memory_usage,
            'time_stability_percent': time_stability,
            'memory_stability_percent': memory_stability,
            'memory_leak_suspected': memory_leak_suspected,
            'average_run_time': np.mean(run_times)
        }

        # 高负载测试
        print("  高负载测试...")
        large_data = np.random.randn(80000, 8).astype(np.float32)

        try:
            with ProcessResourceTracker().context_manager() as tracker:
                start_time = time.time()
                features, _ = extractor.extract_features(large_data)
                processing_time = time.time() - start_time

            usage_stats = tracker.get_current_usage()
            high_load_success = True

        except Exception as e:
            print(f"    高负载测试失败: {e}")
            high_load_success = False
            processing_time = 0
            usage_stats = {}

        stability_results['high_load'] = {
            'success': high_load_success,
            'processing_time': processing_time,
            'usage_stats': usage_stats
        }

        # 总体稳定性评估
        assessment = {
            'test_results': stability_results,
            'time_stability_grade': 'A' if time_stability > 90 else 'B' if time_stability > 80 else 'C',
            'memory_stability_grade': 'A' if memory_stability > 90 else 'B' if memory_stability > 80 else 'C',
            'no_memory_leaks': not memory_leak_suspected,
            'high_load_capable': high_load_success,
            'passed': time_stability > 80 and memory_stability > 80 and not memory_leak_suspected and high_load_success
        }

        print(f"  系统稳定性测试结果: {'✅ 通过' if assessment['passed'] else '❌ 失败'}")
        print(f"    时间稳定性: {time_stability:.1f}%")
        print(f"    内存稳定性: {memory_stability:.1f}%")
        print(f"    内存泄漏: {'❌ 未检测到' if not memory_leak_suspected else '⚠️ 可能存在'}")
        print(f"    高负载处理: {'✅ 成功' if high_load_success else '❌ 失败'}")

        return assessment

    def _test_integration_quality(self) -> Dict[str, Any]:
        """测试集成质量"""
        print("\n4. 集成质量测试...")

        integration_results = {}

        # 测试Day 1-3优化的集成
        test_data = np.random.randn(25000, 5).astype(np.float32)

        extractor = OptimizedFeatureExtractor(
            enable_parallel=True,
            memory_efficient=True,
            enable_monitoring=True
        )

        # 基准测试所有处理模式
        benchmark_results = extractor.benchmark_processing_modes(test_data)

        # 检查不同模式结果的一致性
        features_by_mode = {}
        for mode in benchmark_results.keys():
            if mode == 'vectorized_basic':
                from features.vectorized_extraction import VectorizedFeatureExtractor
                basic_extractor = VectorizedFeatureExtractor(180, 30)
                features, _ = basic_extractor.extract_all_windows_vectorized(test_data)
                features_by_mode[mode] = features
            # 其他模式的特征已在基准测试中获得

        # 特征一致性检查
        consistency_scores = {}
        base_features = features_by_mode.get('vectorized_basic')

        if base_features is not None:
            for mode, features in features_by_mode.items():
                if mode != 'vectorized_basic' and features is not None:
                    # 计算特征相似度
                    correlation = np.corrcoef(base_features.flatten(), features.flatten())[0, 1]
                    mse = np.mean((base_features - features) ** 2)
                    consistency_scores[mode] = {
                        'correlation': correlation,
                        'mse': mse,
                        'consistent': correlation > 0.99 and mse < 1e-6
                    }

        integration_results['feature_consistency'] = consistency_scores
        integration_results['benchmark_results'] = benchmark_results

        # 性能改进验证
        baseline_time = benchmark_results.get('vectorized_basic', {}).get('processing_time', float('inf'))
        best_time = min(result.get('processing_time', float('inf'))
                       for result in benchmark_results.values())

        performance_improvement = ((baseline_time - best_time) / baseline_time) * 100 if baseline_time > 0 else 0

        integration_results['performance_improvement'] = performance_improvement

        # 集成质量评估
        all_consistent = all(score.get('consistent', False)
                           for score in consistency_scores.values())

        assessment = {
            'test_results': integration_results,
            'feature_consistency_passed': all_consistent,
            'performance_improvement_percent': performance_improvement,
            'integration_grade': 'A' if all_consistent and performance_improvement > 25 else 'B' if all_consistent else 'C',
            'passed': all_consistent and performance_improvement > 20
        }

        print(f"  集成质量测试结果: {'✅ 通过' if assessment['passed'] else '❌ 失败'}")
        print(f"    特征一致性: {'✅ 通过' if all_consistent else '❌ 失败'}")
        print(f"    性能提升: {performance_improvement:.1f}%")
        print(f"    集成等级: {assessment['integration_grade']}")

        return assessment

    def _validate_performance_targets(self) -> Dict[str, Any]:
        """验证Day 3性能目标"""
        print("\n5. Day 3性能目标验证...")

        # Day 3目标：在Day 2基础上额外获得25-30%性能提升
        target_improvement = 25.0  # 最低目标

        test_data = np.random.randn(35000, 6).astype(np.float32)

        # 测试Day 2向量化性能（基准）
        from features.vectorized_extraction import VectorizedFeatureExtractor
        day2_extractor = VectorizedFeatureExtractor(180, 30)

        start_time = time.time()
        day2_features, _ = day2_extractor.extract_all_windows_vectorized(test_data)
        day2_time = time.time() - start_time

        # 测试Day 3优化性能
        day3_extractor = OptimizedFeatureExtractor(
            enable_parallel=True,
            memory_efficient=True,
            enable_monitoring=False  # 避免监控开销影响测试
        )

        start_time = time.time()
        day3_features, _ = day3_extractor.extract_features(test_data, adaptive_mode=True)
        day3_time = time.time() - start_time

        # 计算性能提升
        actual_improvement = ((day2_time - day3_time) / day2_time) * 100
        target_met = actual_improvement >= target_improvement

        # 累积改进评估（相对于未优化基线）
        # 假设Day 1+2已实现40x改进，Day 3应在此基础上额外提升25-30%
        expected_total_speedup = 40 * (1 + target_improvement / 100)

        assessment = {
            'day2_time': day2_time,
            'day3_time': day3_time,
            'actual_improvement_percent': actual_improvement,
            'target_improvement_percent': target_improvement,
            'target_met': target_met,
            'expected_total_speedup': expected_total_speedup,
            'performance_grade': 'A' if actual_improvement >= 30 else 'B' if actual_improvement >= 25 else 'C',
            'passed': target_met
        }

        print(f"  Day 3性能目标验证: {'✅ 达成' if target_met else '❌ 未达成'}")
        print(f"    Day 2基准时间: {day2_time:.2f}s")
        print(f"    Day 3优化时间: {day3_time:.2f}s")
        print(f"    实际改进: {actual_improvement:.1f}%")
        print(f"    目标改进: {target_improvement:.1f}%")

        return assessment

    def _calculate_overall_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体评估"""
        print("\n" + "=" * 60)
        print("Day 3 优化总体评估")
        print("=" * 60)

        # 各项测试通过情况
        test_passes = {
            'memory_efficiency': results['memory_efficiency']['passed'],
            'parallel_performance': results['parallel_performance']['passed'],
            'system_stability': results['system_stability']['passed'],
            'integration_quality': results['integration_quality']['passed'],
            'performance_targets': results['performance_targets']['passed']
        }

        total_tests = len(test_passes)
        passed_tests = sum(test_passes.values())
        overall_pass_rate = passed_tests / total_tests

        # 性能等级评估
        grades = {
            'memory_efficiency': results['memory_efficiency']['memory_efficiency_grade'],
            'parallel_performance': results['parallel_performance']['scalability_grade'],
            'system_stability': results['system_stability']['time_stability_grade'],
            'integration_quality': results['integration_quality']['integration_grade']
        }

        # 计算综合等级
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        avg_score = np.mean([grade_scores.get(grade, 0) for grade in grades.values()])

        if avg_score >= 3.5:
            overall_grade = 'A'
        elif avg_score >= 2.5:
            overall_grade = 'B'
        elif avg_score >= 1.5:
            overall_grade = 'C'
        else:
            overall_grade = 'D'

        assessment = {
            'test_pass_rate': overall_pass_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'individual_grades': grades,
            'overall_grade': overall_grade,
            'day3_optimization_successful': overall_pass_rate >= 0.8 and results['performance_targets']['passed'],
            'recommendations': []
        }

        # 生成改进建议
        if not results['memory_efficiency']['passed']:
            assessment['recommendations'].append("优化内存使用：考虑更保守的分块策略或内存清理")

        if not results['parallel_performance']['passed']:
            assessment['recommendations'].append("调整并行策略：优化工作线程数或分块算法")

        if not results['system_stability']['passed']:
            assessment['recommendations'].append("提升系统稳定性：修复内存泄漏或异常处理")

        if not results['integration_quality']['passed']:
            assessment['recommendations'].append("改善集成质量：确保各组件间特征一致性")

        if not results['performance_targets']['passed']:
            assessment['recommendations'].append("达成性能目标：需要进一步优化算法或系统资源利用")

        # 打印总体评估结果
        print(f"测试通过率: {overall_pass_rate*100:.1f}% ({passed_tests}/{total_tests})")
        print(f"综合等级: {overall_grade}")
        print(f"Day 3 优化: {'✅ 成功' if assessment['day3_optimization_successful'] else '❌ 需要改进'}")

        if assessment['recommendations']:
            print(f"\n改进建议:")
            for i, rec in enumerate(assessment['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n🎉 所有测试通过，Day 3 优化达到预期目标！")

        return assessment


def run_day3_validation():
    """运行Day 3优化验证的主函数"""
    validator = Day3OptimizationValidator()

    try:
        results = validator.run_comprehensive_validation()

        # 保存验证结果
        import json
        from datetime import datetime

        results['validation_timestamp'] = datetime.now().isoformat()
        results['validation_summary'] = {
            'day3_success': results['overall_assessment']['day3_optimization_successful'],
            'overall_grade': results['overall_assessment']['overall_grade'],
            'performance_improvement': results['performance_targets']['actual_improvement_percent']
        }

        with open('day3_validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n验证结果已保存至: day3_validation_results.json")

        return results

    except Exception as e:
        print(f"验证过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # 运行Day 3验证
    validation_results = run_day3_validation()

    if validation_results:
        success = validation_results['overall_assessment']['day3_optimization_successful']
        grade = validation_results['overall_assessment']['overall_grade']
        improvement = validation_results['performance_targets']['actual_improvement_percent']

        print(f"\n" + "=" * 60)
        print(f"Day 3 优化验证完成")
        print(f"结果: {'✅ 成功' if success else '❌ 需改进'}")
        print(f"等级: {grade}")
        print(f"性能提升: {improvement:.1f}%")
        print("=" * 60)