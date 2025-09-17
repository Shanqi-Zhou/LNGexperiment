"""
Week 2 优化集成脚本
整合所有Week 2优化模块，提供统一的接口和工作流
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime

# 导入Week 2优化模块
try:
    from .preprocessing.robust_outlier_detection import OptimizedHampelFilter, AdaptiveHampelFilter
    from .models.production_gpr import ProductionGPR, create_production_gpr
    from .features.production_cache import ProductionFeatureCache, CacheManager
    from .benchmarking.performance_benchmark import PerformanceBenchmark
    from .validation.automated_validation import AutomatedValidation, validate_model
    from .features.enhanced_feature_extraction import EnhancedFeatureExtractor
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from preprocessing.robust_outlier_detection import OptimizedHampelFilter, AdaptiveHampelFilter
    from models.production_gpr import ProductionGPR, create_production_gpr
    from features.production_cache import ProductionFeatureCache, CacheManager
    from benchmarking.performance_benchmark import PerformanceBenchmark
    from validation.automated_validation import AutomatedValidation, validate_model


class Week2OptimizationPipeline:
    """
    Week 2 完整优化流水线
    整合Hampel滤波、生产级GPR、特征缓存、基准测试和验证框架
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化Week 2优化流水线

        Args:
            config: 配置参数字典
        """
        # 默认配置
        default_config = {
            # Hampel滤波配置
            'hampel_window': 15,
            'hampel_sigma': 3.0,
            'use_adaptive_hampel': True,

            # GPR配置
            'gpr_auto_tune': True,
            'gpr_validation_split': 0.2,

            # 缓存配置
            'cache_dir': 'cache/week2_features',
            'cache_max_size_gb': 2.0,
            'enable_caching': True,

            # 特征提取配置
            'feature_window_size': 180,
            'feature_stride': 30,

            # 验证配置
            'target_metrics': {
                'r2_min': 0.75,
                'cv_rmse_max': 0.06,
                'nmbe_max': 0.006,
                'mae_max_percent': 5.0,
                'inference_time_per_sample_ms': 10.0
            },

            # 输出配置
            'output_dir': 'week2_optimization_results',
            'enable_benchmarking': True,
            'enable_validation': True
        }

        self.config = {**default_config, **(config or {})}

        # 创建输出目录
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self._initialize_components()

        # 记录处理历史
        self.processing_history = []

        print(f"Week 2 优化流水线初始化完成")
        print(f"输出目录: {self.output_dir}")

    def _initialize_components(self):
        """初始化各个优化组件"""

        # Hampel异常检测器
        if self.config['use_adaptive_hampel']:
            self.hampel_filter = AdaptiveHampelFilter(
                base_window=self.config['hampel_window']
            )
        else:
            self.hampel_filter = OptimizedHampelFilter(
                window=self.config['hampel_window'],
                n_sigma=self.config['hampel_sigma']
            )

        # 生产级特征缓存
        if self.config['enable_caching']:
            self.cache_manager = CacheManager(
                cache_dir=self.config['cache_dir'],
                max_size_gb=self.config['cache_max_size_gb']
            )
        else:
            self.cache_manager = None

        # 增强特征提取器（集成Day1-3优化）
        self.feature_extractor = EnhancedFeatureExtractor(
            window_size=self.config['feature_window_size'],
            stride=self.config['feature_stride']
        )

        # 自动化验证框架
        if self.config['enable_validation']:
            self.validator = AutomatedValidation(
                target_metrics=self.config['target_metrics'],
                output_dir=str(self.output_dir / 'validation')
            )
        else:
            self.validator = None

        # 性能基准测试
        if self.config['enable_benchmarking']:
            self.benchmark = PerformanceBenchmark(
                output_dir=str(self.output_dir / 'benchmarks')
            )
        else:
            self.benchmark = None

        print("    所有组件初始化完成")

    def process_complete_pipeline(self, data: Union[np.ndarray, pd.DataFrame],
                                target_column: str = 'energy',
                                pipeline_name: str = 'week2_pipeline') -> Dict[str, Any]:
        """
        执行完整的Week 2优化流水线

        Args:
            data: 输入数据
            target_column: 目标列名
            pipeline_name: 流水线名称

        Returns:
            完整的处理结果
        """
        print(f"=== 开始Week 2完整优化流水线: {pipeline_name} ===")
        pipeline_start = time.time()

        results = {
            'pipeline_name': pipeline_name,
            'start_time': datetime.now().isoformat(),
            'config': self.config.copy(),
            'stages': {}
        }

        try:
            # Stage 1: 数据预处理（Hampel异常检测）
            print("\n--- Stage 1: Hampel异常检测与数据清理 ---")
            stage1_start = time.time()

            if isinstance(data, pd.DataFrame):
                # 分离特征和目标
                if target_column in data.columns:
                    feature_data = data.drop(columns=[target_column])
                    target_data = data[target_column].values
                else:
                    feature_data = data
                    target_data = None
            else:
                feature_data = data
                target_data = None

            # 应用Hampel滤波
            if self.config['use_adaptive_hampel']:
                cleaned_features = self.hampel_filter.filter_adaptive(feature_data)
            else:
                cleaned_features = self.hampel_filter.filter_vectorized(feature_data)

            stage1_time = time.time() - stage1_start
            hampel_stats = self.hampel_filter.get_outlier_statistics()

            results['stages']['hampel_filtering'] = {
                'processing_time': stage1_time,
                'outlier_statistics': hampel_stats,
                'input_shape': feature_data.shape,
                'output_shape': cleaned_features.shape,
                'status': 'completed'
            }

            print(f"    Hampel滤波完成，用时: {stage1_time:.2f}秒")
            print(f"    异常值检测: {hampel_stats.get('outliers_detected', 0)} 个")

            # Stage 2: 特征提取（集成Day1-3优化）
            print("\n--- Stage 2: 增强特征提取 ---")
            stage2_start = time.time()

            # 使用缓存进行特征提取
            if self.cache_manager:
                features = self.cache_manager.cache_features(
                    self._extract_features_wrapper,
                    cleaned_features,
                    f"{pipeline_name}_features"
                )
            else:
                features = self._extract_features_wrapper(cleaned_features)

            stage2_time = time.time() - stage2_start

            results['stages']['feature_extraction'] = {
                'processing_time': stage2_time,
                'input_shape': cleaned_features.shape,
                'output_shape': features.shape,
                'cache_used': self.cache_manager is not None,
                'status': 'completed'
            }

            print(f"    特征提取完成，用时: {stage2_time:.2f}秒")
            print(f"    生成特征形状: {features.shape}")

            # Stage 3: 模型训练（生产级GPR）
            if target_data is not None:
                print("\n--- Stage 3: 生产级GPR模型训练 ---")
                stage3_start = time.time()

                # 创建并训练GPR模型
                gpr_model = ProductionGPR(
                    n_samples=len(features),
                    auto_tune=self.config['gpr_auto_tune']
                )

                validation_results = gpr_model.fit_with_validation(
                    features, target_data,
                    validation_split=self.config['gpr_validation_split']
                )

                stage3_time = time.time() - stage3_start
                gpr_performance = gpr_model.get_performance_summary()

                results['stages']['gpr_training'] = {
                    'processing_time': stage3_time,
                    'validation_results': validation_results,
                    'performance_summary': gpr_performance,
                    'status': 'completed'
                }

                print(f"    GPR训练完成，用时: {stage3_time:.2f}秒")
                print(f"    模型性能: {gpr_performance}")

                # Stage 4: 自动化验证
                if self.validator:
                    print("\n--- Stage 4: 自动化模型验证 ---")
                    stage4_start = time.time()

                    # 分割数据进行验证
                    n_test = min(len(features) // 4, 1000)  # 使用25%或最多1000个样本测试
                    X_test = features[-n_test:]
                    y_test = target_data[-n_test:]
                    X_train = features[:-n_test]
                    y_train = target_data[:-n_test]

                    validation_comprehensive = self.validator.run_full_validation(
                        gpr_model, X_test, y_test, X_train, y_train,
                        f"{pipeline_name}_validation"
                    )

                    stage4_time = time.time() - stage4_start

                    results['stages']['automated_validation'] = {
                        'processing_time': stage4_time,
                        'validation_results': validation_comprehensive,
                        'status': 'completed'
                    }

                    print(f"    自动化验证完成，用时: {stage4_time:.2f}秒")
                    print(f"    验证评分: {validation_comprehensive['overall_assessment']['overall_score']:.1f}/100")

                # Stage 5: 性能基准测试
                if self.benchmark and len(features) > 1000:
                    print("\n--- Stage 5: 性能基准测试 ---")
                    stage5_start = time.time()

                    # 创建基线和优化版本的对比函数
                    def baseline_extractor(data):
                        """基线特征提取器（模拟传统方法）"""
                        return self.feature_extractor.extract_features_single_threaded(data)

                    def optimized_extractor(data):
                        """优化特征提取器（Week 2完整优化）"""
                        # 应用Hampel滤波 + 增强特征提取
                        filtered_data = self.hampel_filter.filter_vectorized(data)
                        return self.feature_extractor.extract_features_optimized(filtered_data)

                    benchmark_results = self.benchmark.run_complete_benchmark(
                        baseline_extractor=baseline_extractor,
                        optimized_extractor=optimized_extractor
                    )

                    stage5_time = time.time() - stage5_start

                    results['stages']['performance_benchmark'] = {
                        'processing_time': stage5_time,
                        'benchmark_results': benchmark_results,
                        'status': 'completed'
                    }

                    print(f"    基准测试完成，用时: {stage5_time:.2f}秒")

            else:
                print("\n--- 跳过模型训练阶段（无目标数据） ---")
                results['stages']['gpr_training'] = {'status': 'skipped', 'reason': 'no_target_data'}
                results['stages']['automated_validation'] = {'status': 'skipped', 'reason': 'no_target_data'}

            # 计算总体处理时间
            total_pipeline_time = time.time() - pipeline_start
            results['total_processing_time'] = total_pipeline_time
            results['end_time'] = datetime.now().isoformat()
            results['status'] = 'completed'

            # 保存结果
            self._save_pipeline_results(results, pipeline_name)

            # 生成摘要报告
            summary_report = self._generate_pipeline_summary(results)
            print(summary_report)

            # 记录到历史
            self.processing_history.append(results)

            print(f"\n=== Week 2优化流水线完成，总用时: {total_pipeline_time:.2f}秒 ===")

            return results

        except Exception as e:
            error_time = time.time() - pipeline_start
            results.update({
                'status': 'failed',
                'error': str(e),
                'error_time': error_time,
                'end_time': datetime.now().isoformat()
            })

            print(f"\n❌ 流水线执行失败: {e}")
            self._save_pipeline_results(results, f"{pipeline_name}_failed")

            return results

    def _extract_features_wrapper(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """特征提取包装器，用于缓存系统"""
        return self.feature_extractor.extract_features_optimized(data)

    def _save_pipeline_results(self, results: Dict[str, Any], pipeline_name: str):
        """保存流水线结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_results_{pipeline_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        # 确保结果可以JSON序列化
        serializable_results = self._make_json_serializable(results)

        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"    流水线结果已保存: {filepath}")

    def _make_json_serializable(self, obj):
        """使对象可JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 1000 else f"array_shape_{obj.shape}"  # 大数组只保存形状
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):  # datetime对象
            return obj.isoformat()
        else:
            try:
                return str(obj)
            except:
                return "non_serializable_object"

    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """生成流水线执行摘要"""
        lines = [
            "=" * 80,
            f"Week 2 优化流水线执行摘要 - {results['pipeline_name']}",
            "=" * 80,
            f"执行时间: {results['start_time']} - {results['end_time']}",
            f"总处理时间: {results['total_processing_time']:.2f}秒",
            f"流水线状态: {'✅ 成功' if results['status'] == 'completed' else '❌ 失败'}",
            "",
            "## 各阶段处理时间",
            ""
        ]

        # 各阶段时间统计
        total_stage_time = 0
        for stage_name, stage_data in results['stages'].items():
            if isinstance(stage_data, dict) and 'processing_time' in stage_data:
                stage_time = stage_data['processing_time']
                total_stage_time += stage_time
                status_icon = "✅" if stage_data.get('status') == 'completed' else "⏭️" if stage_data.get('status') == 'skipped' else "❌"
                lines.append(f"{stage_name:25} {stage_time:8.2f}秒 {status_icon}")

        lines.extend([
            f"{'总阶段时间':<25} {total_stage_time:8.2f}秒",
            "",
            "## 关键优化成果",
            ""
        ])

        # Hampel滤波结果
        hampel_stage = results['stages'].get('hampel_filtering', {})
        if hampel_stage.get('status') == 'completed':
            outlier_stats = hampel_stage.get('outlier_statistics', {})
            outlier_rate = outlier_stats.get('outlier_rate_percent', 0)
            lines.append(f"异常检测: 发现 {outlier_stats.get('outliers_detected', 0)} 个异常值 ({outlier_rate:.1f}%)")

        # 特征提取结果
        feature_stage = results['stages'].get('feature_extraction', {})
        if feature_stage.get('status') == 'completed':
            input_shape = feature_stage.get('input_shape', [0, 0])
            output_shape = feature_stage.get('output_shape', [0, 0])
            lines.append(f"特征提取: {input_shape} → {output_shape}")

        # GPR训练结果
        gpr_stage = results['stages'].get('gpr_training', {})
        if gpr_stage.get('status') == 'completed':
            perf_summary = gpr_stage.get('performance_summary', {})
            if 'validation_scores' in perf_summary:
                val_scores = perf_summary['validation_scores']
                lines.append(f"GPR性能: 训练R² {val_scores.get('train_r2', 0):.3f}, 验证R² {val_scores.get('val_r2', 0):.3f}")

        # 验证结果
        validation_stage = results['stages'].get('automated_validation', {})
        if validation_stage.get('status') == 'completed':
            val_results = validation_stage.get('validation_results', {})
            overall_assessment = val_results.get('overall_assessment', {})
            score = overall_assessment.get('overall_score', 0)
            passed = overall_assessment.get('validation_passed', False)
            lines.append(f"自动化验证: 评分 {score:.1f}/100 {'✅ 通过' if passed else '❌ 未通过'}")

        # 基准测试结果
        benchmark_stage = results['stages'].get('performance_benchmark', {})
        if benchmark_stage.get('status') == 'completed':
            bench_results = benchmark_stage.get('benchmark_results', {})
            comparison = bench_results.get('comparison_results', {})
            if comparison:
                # 计算平均加速比
                speedups = [comp.get('speedup', 1) for comp in comparison.values() if isinstance(comp, dict) and 'speedup' in comp]
                if speedups:
                    avg_speedup = np.mean(speedups)
                    lines.append(f"性能基准: 平均加速比 {avg_speedup:.2f}x")

        lines.extend([
            "",
            "## 系统优化效果",
            ""
        ])

        # 基于结果计算预期生产环境收益
        if results['status'] == 'completed':
            total_time = results['total_processing_time']
            lines.extend([
                f"Week 2完整流水线处理时间: {total_time:.2f}秒",
                "相比Day 1-3基础优化额外提升:",
                "- 数据质量: Hampel异常检测改善数据质量",
                "- 模型鲁棒性: 生产级GPR增强预测稳定性",
                "- 系统可靠性: 自动化验证保证质量",
                "- 运维效率: 特征缓存减少重复计算",
                ""
            ])

        lines.extend([
            "=" * 80,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])

        return "\\n".join(lines)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化效果总结"""
        if not self.processing_history:
            return {"message": "尚未执行任何流水线"}

        latest_result = self.processing_history[-1]

        summary = {
            'total_pipelines_executed': len(self.processing_history),
            'latest_pipeline': {
                'name': latest_result['pipeline_name'],
                'status': latest_result['status'],
                'total_time': latest_result.get('total_processing_time', 0),
                'stages_completed': len([s for s in latest_result['stages'].values() if s.get('status') == 'completed'])
            },
            'week2_optimizations': {
                'hampel_anomaly_detection': '✅ 实现',
                'production_gpr': '✅ 实现',
                'feature_caching': '✅ 实现',
                'automated_validation': '✅ 实现',
                'performance_benchmarking': '✅ 实现'
            }
        }

        return summary

    def cleanup_cache(self):
        """清理缓存和临时文件"""
        if self.cache_manager:
            self.cache_manager.cache.clear()
            print("    特征缓存已清理")

    def print_config(self):
        """打印当前配置"""
        print("=== Week 2 优化流水线配置 ===")
        for section, params in self.config.items():
            if isinstance(params, dict):
                print(f"\n{section}:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
            else:
                print(f"{section}: {params}")


# 便捷函数
def run_week2_optimization(data: Union[np.ndarray, pd.DataFrame],
                          target_column: str = 'energy',
                          config: Dict[str, Any] = None,
                          pipeline_name: str = 'week2_optimization') -> Dict[str, Any]:
    """
    便捷的Week 2优化执行函数

    Args:
        data: 输入数据
        target_column: 目标列名
        config: 配置参数
        pipeline_name: 流水线名称

    Returns:
        优化结果
    """
    pipeline = Week2OptimizationPipeline(config)
    return pipeline.process_complete_pipeline(data, target_column, pipeline_name)


# 测试代码
if __name__ == '__main__':
    print("=== Week 2 优化集成测试 ===")

    # 生成测试数据
    np.random.seed(42)
    n_samples = 5000
    n_features = 6

    # 创建带异常值的测试数据
    test_data = np.random.randn(n_samples, n_features)

    # 添加信号模式
    t = np.linspace(0, 10*np.pi, n_samples)
    test_data[:, 0] += np.sin(t)
    test_data[:, 1] += 0.5 * np.cos(2*t)
    test_data[:, 2] += np.cumsum(np.random.randn(n_samples)) * 0.01

    # 人工添加异常值
    outlier_indices = np.random.choice(n_samples, size=100, replace=False)
    test_data[outlier_indices] += np.random.randn(100, n_features) * 5

    # 生成目标变量
    target = (
        2 * test_data[:, 0] +
        1.5 * test_data[:, 1] -
        0.5 * test_data[:, 2] +
        np.random.randn(n_samples) * 0.1
    )

    # 创建DataFrame
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    test_df = pd.DataFrame(test_data, columns=feature_columns)
    test_df['energy'] = target

    print(f"测试数据形状: {test_df.shape}")
    print(f"添加异常值: {len(outlier_indices)} 个")

    # 运行Week 2完整优化流水线
    results = run_week2_optimization(
        data=test_df,
        target_column='energy',
        pipeline_name='integration_test'
    )

    print("\\n=== 集成测试完成 ===")
    print(f"流水线状态: {results['status']}")
    print(f"处理时间: {results.get('total_processing_time', 0):.2f}秒")

    if results['status'] == 'completed':
        print("✅ Week 2 优化集成成功")
    else:
        print("❌ Week 2 优化集成失败")
        if 'error' in results:
            print(f"错误: {results['error']}")

    print("\\n=== Week 2 优化集成测试完成 ===")