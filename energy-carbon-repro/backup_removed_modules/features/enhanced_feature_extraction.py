"""
增强版特征提取器
集成Day 1-3的所有优化：数值稳定性 + 向量化计算 + 并行处理
"""
import pandas as pd
import numpy as np
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Tuple, Optional
import psutil

# 导入优化模块
from .numerical_stability import NumericallyStableFeatures, safe_statistical_features
from .vectorized_extraction import VectorizedFeatureExtractor, BatchFeatureProcessor
from .intelligent_chunking import IntelligentChunker, TimeSeriesChunker
from ..monitoring.resource_monitor import ResourceMonitor, MemoryProfiler


class EnhancedFeatureExtractor:
    """
    增强版特征提取器
    集成Day 1-3的所有优化：
    - Day 1: 数值稳定性
    - Day 2: 向量化批量计算
    - Day 3: 内存优化和并行处理
    """

    def __init__(self,
                 window_size: int = 180,
                 stride: int = 30,
                 use_parallel: bool = True,
                 max_workers: Optional[int] = None,
                 chunk_memory_percent: float = 40.0,
                 enable_monitoring: bool = True,
                 cache_features: bool = True):
        """
        初始化增强版特征提取器

        Args:
            window_size: 滑动窗口大小
            stride: 窗口步长
            use_parallel: 是否启用并行处理
            max_workers: 最大工作线程数，默认为CPU核心数
            chunk_memory_percent: 单个块的目标内存使用百分比
            enable_monitoring: 是否启用资源监控
            cache_features: 是否启用特征缓存
        """
        self.window_size = window_size
        self.stride = stride
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(psutil.cpu_count(), 8)  # 限制最大工作线程
        self.chunk_memory_percent = chunk_memory_percent
        self.enable_monitoring = enable_monitoring
        self.cache_features = cache_features

        # 系统信息
        self.cpu_count = psutil.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # 初始化组件
        self._init_components()

        # 性能统计
        self.performance_stats = {
            'total_processing_time': 0.0,
            'feature_extraction_time': 0.0,
            'parallel_efficiency': 0.0,
            'memory_peak_mb': 0.0,
            'chunks_processed': 0,
            'total_windows': 0,
            'optimization_level': 'enhanced_parallel'
        }

        print(f"  🚀 增强版特征提取器初始化完成:")
        print(f"    系统配置: {self.cpu_count}核CPU, {self.total_memory_gb:.1f}GB内存")
        print(f"    并行处理: {'启用' if use_parallel else '禁用'} ({self.max_workers} 工作线程)")
        print(f"    资源监控: {'启用' if enable_monitoring else '禁用'}")
        print(f"    特征缓存: {'启用' if cache_features else '禁用'}")

    def _init_components(self):
        """初始化各个组件"""
        # 向量化处理器（Day 2优化）
        self.batch_processor = BatchFeatureProcessor(
            self.window_size,
            self.stride,
            use_cache=self.cache_features
        )

        # 智能分块器（Day 3优化）
        self.chunker = TimeSeriesChunker(
            window_size=self.window_size,
            stride=self.stride,
            target_memory_usage_percent=self.chunk_memory_percent,
            min_chunk_size=max(1000, self.window_size * 2),
            max_chunk_size=50000
        )

        # 资源监控器（Day 3优化）
        if self.enable_monitoring:
            self.monitor = ResourceMonitor(monitoring_interval=0.5)
            self.profiler = MemoryProfiler()
        else:
            self.monitor = None
            self.profiler = None

        # 线程池（延迟初始化）
        self.executor = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        主要的特征提取接口
        自动选择最优处理策略：单线程向量化 vs 并行分块处理

        Args:
            df: 输入时序数据

        Returns:
            提取的特征DataFrame
        """
        start_time = time.time()

        if self.enable_monitoring:
            self.monitor.start_monitoring()
            self.profiler.take_snapshot("transform_start")

        print(f"  🎯 开始增强版特征提取")
        print(f"    输入数据形状: {df.shape}")
        print(f"    估算内存大小: {df.memory_usage(deep=True).sum() / (1024**2):.1f}MB")

        try:
            # 选择处理策略
            processing_strategy = self._select_processing_strategy(df)
            print(f"    选择处理策略: {processing_strategy}")

            if processing_strategy == "single_vectorized":
                result_df = self._single_vectorized_processing(df)
            elif processing_strategy == "parallel_chunked":
                result_df = self._parallel_chunked_processing(df)
            else:  # fallback
                result_df = self._fallback_processing(df)

            # 记录性能统计
            total_time = time.time() - start_time
            self.performance_stats['total_processing_time'] = total_time
            self.performance_stats['total_windows'] = len(result_df)

            if self.enable_monitoring:
                self.profiler.take_snapshot("transform_end")
                monitoring_summary = self.monitor.stop_monitoring()
                self._update_performance_stats(monitoring_summary)

            self._print_performance_summary(total_time)

            return result_df

        except Exception as e:
            if self.enable_monitoring and self.monitor.monitoring:
                self.monitor.stop_monitoring()
            print(f"  ❌ 特征提取失败: {e}")
            raise

    def _select_processing_strategy(self, df: pd.DataFrame) -> str:
        """
        根据数据大小和系统资源选择处理策略

        Args:
            df: 输入数据

        Returns:
            处理策略: "single_vectorized", "parallel_chunked", "fallback"
        """
        data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
        available_memory_mb = psutil.virtual_memory().available / (1024**2)

        # 估算处理内存需求（向量化处理通常需要3-5倍原始数据大小）
        estimated_processing_memory = data_size_mb * 4

        print(f"    策略分析:")
        print(f"      数据大小: {data_size_mb:.1f}MB")
        print(f"      可用内存: {available_memory_mb:.1f}MB")
        print(f"      估算处理内存: {estimated_processing_memory:.1f}MB")

        # 策略选择逻辑
        if not self.use_parallel:
            return "single_vectorized"

        # 如果数据很小，直接使用向量化
        if data_size_mb < 100:
            return "single_vectorized"

        # 如果内存充足，使用向量化
        if estimated_processing_memory < available_memory_mb * 0.6:
            return "single_vectorized"

        # 否则使用并行分块
        return "parallel_chunked"

    def _single_vectorized_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """单线程向量化处理（Day 2优化）"""
        print(f"  ⚡ 使用向量化特征提取")

        start_time = time.time()

        # 确保数据是数值类型
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("DataFrame中没有数值列")

        # 使用向量化处理器
        features, feature_names = self.batch_processor.process_with_caching(
            numeric_df.values,
            cache_key=f"vectorized_{hash(str(df.values.tobytes()))}"
        )

        # 创建结果DataFrame
        result_df = pd.DataFrame(features, columns=feature_names)

        extraction_time = time.time() - start_time
        self.performance_stats['feature_extraction_time'] = extraction_time

        print(f"    向量化处理完成: {extraction_time:.2f}秒")
        print(f"    生成特征形状: {result_df.shape}")

        return result_df

    def _parallel_chunked_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """并行分块处理（Day 3优化）"""
        print(f"  🔄 使用并行分块处理")

        start_time = time.time()

        # 确保数据是数值类型
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("DataFrame中没有数值列")

        # 创建数据块
        chunks_info = []
        chunk_data_list = []

        for start_idx, end_idx, chunk_data in self.chunker.create_windowed_chunks(numeric_df):
            chunks_info.append((start_idx, end_idx))
            chunk_data_list.append(chunk_data.values)  # 转换为numpy数组

        print(f"    数据分块完成: {len(chunk_data_list)} 个块")

        # 并行处理块
        chunk_results = self._process_chunks_parallel(chunk_data_list)

        # 合并结果
        result_df = self._merge_chunk_results(chunk_results, chunks_info)

        extraction_time = time.time() - start_time
        self.performance_stats['feature_extraction_time'] = extraction_time
        self.performance_stats['chunks_processed'] = len(chunk_data_list)

        # 计算并行效率
        theoretical_serial_time = extraction_time * self.max_workers
        if theoretical_serial_time > 0:
            self.performance_stats['parallel_efficiency'] = extraction_time / theoretical_serial_time

        print(f"    并行处理完成: {extraction_time:.2f}秒")
        print(f"    处理块数: {len(chunk_data_list)}")
        print(f"    生成特征形状: {result_df.shape}")

        return result_df

    def _process_chunks_parallel(self, chunk_data_list: List[np.ndarray]) -> List[np.ndarray]:
        """并行处理数据块"""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        print(f"    启动 {self.max_workers} 个工作线程...")

        try:
            # 提交任务到线程池
            futures = []
            for i, chunk_data in enumerate(chunk_data_list):
                future = self.executor.submit(self._process_single_chunk, chunk_data, i)
                futures.append(future)

            # 收集结果
            chunk_results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    chunk_results.append(result)
                    print(f"      块 {i+1}/{len(futures)} 处理完成")
                except Exception as e:
                    print(f"      块 {i+1} 处理失败: {e}")
                    # 使用空结果占位
                    chunk_results.append(np.array([]))

            return chunk_results

        finally:
            # 清理线程池
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None

    def _process_single_chunk(self, chunk_data: np.ndarray, chunk_id: int) -> np.ndarray:
        """处理单个数据块"""
        try:
            # 为每个线程创建独立的处理器
            chunk_processor = VectorizedFeatureExtractor(self.window_size, self.stride)

            # 处理块
            features, _ = chunk_processor.extract_all_windows_vectorized(chunk_data)

            return features

        except Exception as e:
            print(f"      块 {chunk_id} 处理异常: {e}")
            return np.array([])

    def _merge_chunk_results(self, chunk_results: List[np.ndarray], chunks_info: List[Tuple[int, int]]) -> pd.DataFrame:
        """合并块处理结果"""
        valid_results = [result for result in chunk_results if result.size > 0]

        if not valid_results:
            raise ValueError("所有块处理都失败了")

        print(f"    合并 {len(valid_results)} 个有效结果...")

        # 垂直堆叠特征
        all_features = np.vstack(valid_results)

        # 生成特征名称（使用第一个有效结果的维度）
        n_features = valid_results[0].shape[1]
        feature_names = self._generate_feature_names(n_features)

        result_df = pd.DataFrame(all_features, columns=feature_names)

        print(f"    合并后特征形状: {result_df.shape}")

        return result_df

    def _generate_feature_names(self, n_features: int) -> List[str]:
        """生成特征名称"""
        # 假设每列有9个特征 (与vectorized_extraction.py保持一致)
        features_per_col = 9
        n_cols = n_features // features_per_col

        feature_names = []
        for col_idx in range(n_cols):
            col_name = f'feature_{col_idx}'
            feature_names.extend([
                f'{col_name}_mean',
                f'{col_name}_std',
                f'{col_name}_skew',
                f'{col_name}_kurtosis',
                f'{col_name}_rms',
                f'{col_name}_mean_change_rate',
                f'{col_name}_change_rate_std',
                f'{col_name}_imf1_freq',
                f'{col_name}_imf1_energy_ratio'
            ])

        return feature_names

    def _fallback_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """降级处理（使用原始向量化方法）"""
        print(f"  🔄 使用降级处理方法")

        # 直接使用Day 2的向量化处理器
        extractor = VectorizedFeatureExtractor(self.window_size, self.stride)

        numeric_df = df.select_dtypes(include=[np.number])
        features, feature_names = extractor.extract_all_windows_vectorized(numeric_df.values)

        return pd.DataFrame(features, columns=feature_names)

    def _update_performance_stats(self, monitoring_summary: Dict[str, Any]):
        """更新性能统计"""
        self.performance_stats['memory_peak_mb'] = monitoring_summary.get('peak_memory_percent', 0) * self.total_memory_gb * 1024 / 100

    def _print_performance_summary(self, total_time: float):
        """打印性能摘要"""
        print(f"\n  📊 增强版特征提取性能摘要:")
        print(f"    总处理时间: {total_time:.2f}秒")
        print(f"    特征提取时间: {self.performance_stats['feature_extraction_time']:.2f}秒")
        print(f"    生成窗口数: {self.performance_stats['total_windows']}")

        if self.performance_stats['chunks_processed'] > 0:
            print(f"    处理块数: {self.performance_stats['chunks_processed']}")
            print(f"    并行效率: {self.performance_stats['parallel_efficiency']:.2%}")

        if self.performance_stats['memory_peak_mb'] > 0:
            print(f"    峰值内存使用: {self.performance_stats['memory_peak_mb']:.0f}MB")

        # 计算相对于Day 2的性能提升
        windows_per_second = self.performance_stats['total_windows'] / total_time
        print(f"    处理速度: {windows_per_second:.0f} 窗口/秒")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()

    def clear_cache(self):
        """清除缓存"""
        if hasattr(self.batch_processor, 'cache'):
            self.batch_processor.cache.clear()
            print("  🗑️ 特征缓存已清除")

    def __del__(self):
        """清理资源"""
        if self.executor:
            self.executor.shutdown(wait=False)


# 保持向后兼容性
class FeatureExtractor:
    """
    向后兼容的特征提取器接口
    内部使用增强版实现
    """
    def __init__(self, window_size=180, stride=30, use_stable_features=True, use_vectorized=True):
        # 映射到增强版参数
        self.enhanced_extractor = EnhancedFeatureExtractor(
            window_size=window_size,
            stride=stride,
            use_parallel=True,  # 默认启用并行
            enable_monitoring=False,  # 为了兼容性，默认关闭监控
            cache_features=True
        )

    def transform(self, df):
        """兼容接口"""
        return self.enhanced_extractor.transform(df)

    def get_extraction_stats(self):
        """兼容接口"""
        return self.enhanced_extractor.get_performance_stats()


# 使用示例
if __name__ == '__main__':
    print("=== 增强版特征提取器测试 ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 20000
    n_features = 6
    test_data = np.random.randn(n_samples, n_features)

    # 添加一些模式
    t = np.linspace(0, 10*np.pi, n_samples)
    test_data[:, 0] += np.sin(t)
    test_data[:, 1] += 0.5 * np.cos(2*t)

    test_df = pd.DataFrame(test_data, columns=[f'sensor_{i}' for i in range(n_features)])

    print(f"测试数据形状: {test_df.shape}")
    print(f"测试数据大小: {test_df.memory_usage(deep=True).sum() / (1024**2):.1f}MB")

    # 测试增强版提取器
    extractor = EnhancedFeatureExtractor(
        window_size=180,
        stride=30,
        use_parallel=True,
        enable_monitoring=True
    )

    # 运行特征提取
    start_time = time.time()
    features_df = extractor.transform(test_df)
    end_time = time.time()

    print(f"\n=== 测试结果 ===")
    print(f"提取时间: {end_time - start_time:.3f}秒")
    print(f"输出特征形状: {features_df.shape}")
    print(f"特征示例:")
    print(features_df.head())

    # 获取性能统计
    stats = extractor.get_performance_stats()
    print(f"\n=== 性能统计 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\n=== 增强版特征提取器测试完成 ===")