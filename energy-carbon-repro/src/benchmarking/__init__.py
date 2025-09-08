"""基准测试模块初始化"""

from .end_to_end_benchmark import (
    BenchmarkConfig,
    ComponentBenchmark,
    PipelineBenchmark,
    PerformanceComparison,
    EndToEndBenchmarkSuite,
    create_benchmark_suite,
    run_quick_benchmark,
    compare_optimization_strategies
)

__all__ = [
    'BenchmarkConfig',
    'ComponentBenchmark',
    'PipelineBenchmark', 
    'PerformanceComparison',
    'EndToEndBenchmarkSuite',
    'create_benchmark_suite',
    'run_quick_benchmark',
    'compare_optimization_strategies'
]