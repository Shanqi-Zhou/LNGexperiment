"""
内存高效并行处理器
Day 3 优化：内存和并行计算优化，预期25-30%性能提升
"""
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import psutil
import warnings
from typing import Optional, Tuple, List, Callable, Dict, Any
from tqdm import tqdm
import gc

from .vectorized_extraction import VectorizedFeatureExtractor


class MemoryEfficientParallelProcessor:
    """内存高效的并行特征提取处理器"""

    def __init__(self,
                 window_size=180,
                 stride=30,
                 max_workers=None,
                 max_memory_usage=0.8,
                 chunk_size_method='adaptive',
                 use_threads=True):
        """
        初始化并行处理器

        Args:
            window_size: 窗口大小
            stride: 窗口步长
            max_workers: 最大工作进程/线程数
            max_memory_usage: 最大内存使用比例(0.0-1.0)
            chunk_size_method: 分块大小计算方法 ('fixed', 'adaptive', 'memory_aware')
            use_threads: 是否使用线程池（True）或进程池（False）
        """
        self.window_size = window_size
        self.stride = stride
        self.use_threads = use_threads
        self.max_memory_usage = max_memory_usage
        self.chunk_size_method = chunk_size_method

        # CPU核心数和工作进程数配置
        self.n_cores = mp.cpu_count()
        self.max_workers = max_workers or min(self.n_cores, 8)

        # 内存信息
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available

        # 基础特征提取器
        self.base_extractor = VectorizedFeatureExtractor(window_size, stride)

        # 性能统计
        self.performance_stats = {
            'total_processing_time': 0.0,
            'parallel_efficiency': 0.0,
            'memory_efficiency': 0.0,
            'cpu_utilization': 0.0,
            'chunk_processing_times': [],
            'max_memory_used_mb': 0.0,
            'worker_stats': {},
            'speedup_factor': 1.0
        }

        # 线程安全
        self._stats_lock = Lock()

        print(f"并行处理器初始化:")
        print(f"  CPU核心数: {self.n_cores}")
        print(f"  最大工作线程/进程数: {self.max_workers}")
        print(f"  总内存: {self.total_memory / 1024**3:.1f}GB")
        print(f"  可用内存: {self.available_memory / 1024**3:.1f}GB")
        print(f"  使用{'线程池' if self.use_threads else '进程池'}")

    def process_parallel(self, data: np.ndarray,
                        progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, List[str]]:
        """
        并行处理大型数据集的特征提取

        Args:
            data: 输入数据 (n_samples, n_features)
            progress_callback: 进度回调函数

        Returns:
            features: 提取的特征 (n_windows, n_features)
            feature_names: 特征名称列表
        """
        start_time = time.time()
        print(f"\n=== 内存高效并行特征提取 ===")
        print(f"输入数据形状: {data.shape}")

        # 计算窗口信息
        n_samples, n_cols = data.shape
        n_windows = (n_samples - self.window_size) // self.stride + 1

        # 智能分块策略
        chunks = self._create_intelligent_chunks(data, n_windows)
        print(f"数据分割为 {len(chunks)} 个块进行并行处理")

        # 生成特征名称
        feature_names = self.base_extractor._generate_feature_names(n_cols)
        n_features_total = len(feature_names)

        # 预分配结果数组
        all_features = np.zeros((n_windows, n_features_total))

        # 并行处理分块
        processed_windows = 0

        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_chunk = {
                executor.submit(self._process_chunk_worker, chunk_data, chunk_idx, processed_windows + chunk_start):
                (chunk_idx, chunk_start, chunk_end)
                for chunk_idx, (chunk_data, chunk_start, chunk_end) in enumerate(chunks)
            }

            # 收集结果
            with tqdm(total=len(chunks), desc="处理数据块") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_idx, chunk_start, chunk_end = future_to_chunk[future]

                    try:
                        chunk_features, processing_time, memory_used = future.result()

                        # 将结果写入总数组
                        all_features[chunk_start:chunk_end] = chunk_features

                        # 更新统计
                        with self._stats_lock:
                            self.performance_stats['chunk_processing_times'].append(processing_time)
                            self.performance_stats['max_memory_used_mb'] = max(
                                self.performance_stats['max_memory_used_mb'], memory_used
                            )

                        pbar.update(1)

                        if progress_callback:
                            progress_callback(chunk_idx + 1, len(chunks))

                    except Exception as e:
                        print(f"块 {chunk_idx} 处理失败: {e}")
                        raise

        # 计算最终性能统计
        total_time = time.time() - start_time
        self._calculate_final_stats(total_time, n_windows)

        print(f"\n并行处理完成:")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  处理窗口数: {n_windows}")
        print(f"  并行加速比: {self.performance_stats['speedup_factor']:.2f}x")
        print(f"  内存效率: {self.performance_stats['memory_efficiency']:.1f}%")
        print(f"  CPU利用率: {self.performance_stats['cpu_utilization']:.1f}%")

        return all_features, feature_names

    def _create_intelligent_chunks(self, data: np.ndarray, n_windows: int) -> List[Tuple]:
        """
        智能数据分块策略

        Args:
            data: 输入数据
            n_windows: 总窗口数

        Returns:
            chunks: 分块数据列表 [(chunk_data, start_idx, end_idx), ...]
        """
        chunks = []

        if self.chunk_size_method == 'adaptive':
            # 自适应分块：基于CPU核心数和内存
            base_chunk_size = max(n_windows // (self.max_workers * 2), 1)
            chunk_size = self._adjust_chunk_size_for_memory(data, base_chunk_size)

        elif self.chunk_size_method == 'memory_aware':
            # 内存感知分块：基于可用内存动态调整
            chunk_size = self._calculate_memory_aware_chunk_size(data)

        else:  # fixed
            chunk_size = max(n_windows // self.max_workers, 1)

        print(f"计算得到的分块大小: {chunk_size} 个窗口")

        # 创建分块
        for start_idx in range(0, n_windows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_windows)

            # 计算数据范围（考虑窗口重叠）
            data_start = start_idx * self.stride
            data_end = min((end_idx - 1) * self.stride + self.window_size, data.shape[0])

            chunk_data = data[data_start:data_end].copy()
            chunks.append((chunk_data, start_idx, end_idx))

        return chunks

    def _adjust_chunk_size_for_memory(self, data: np.ndarray, base_chunk_size: int) -> int:
        """根据内存约束调整分块大小"""
        # 估算单个窗口的内存需求
        bytes_per_sample = data.dtype.itemsize * data.shape[1]
        window_memory = self.window_size * bytes_per_sample

        # 估算特征内存需求（假设float64）
        features_per_window = data.shape[1] * 9  # 9个统计特征
        feature_memory = features_per_window * 8  # 8 bytes per float64

        total_memory_per_window = window_memory + feature_memory

        # 计算内存安全的分块大小
        available_for_processing = self.available_memory * self.max_memory_usage
        safe_chunk_size = int(available_for_processing // (total_memory_per_window * self.max_workers))

        # 使用保守值
        final_chunk_size = min(base_chunk_size, max(safe_chunk_size, 1))

        print(f"  基础分块大小: {base_chunk_size}")
        print(f"  内存安全分块大小: {safe_chunk_size}")
        print(f"  最终分块大小: {final_chunk_size}")

        return final_chunk_size

    def _calculate_memory_aware_chunk_size(self, data: np.ndarray) -> int:
        """基于当前内存使用情况动态计算分块大小"""
        current_memory = psutil.virtual_memory()
        available_memory = current_memory.available

        # 估算处理一个窗口的内存开销
        sample_memory = data.nbytes / data.shape[0] * self.window_size

        # 保守估计：预留50%缓冲
        safe_memory = available_memory * 0.5
        max_windows_per_chunk = int(safe_memory / sample_memory / self.max_workers)

        return max(max_windows_per_chunk, 10)  # 最少10个窗口

    def _process_chunk_worker(self, chunk_data: np.ndarray,
                             chunk_idx: int,
                             global_start_idx: int) -> Tuple[np.ndarray, float, float]:
        """
        工作进程/线程处理单个数据块

        Args:
            chunk_data: 块数据
            chunk_idx: 块索引
            global_start_idx: 全局起始索引

        Returns:
            features: 提取的特征
            processing_time: 处理时间
            memory_used: 使用的内存(MB)
        """
        start_time = time.time()
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024

        try:
            # 创建局部特征提取器
            local_extractor = VectorizedFeatureExtractor(self.window_size, self.stride)

            # 提取特征
            features, _ = local_extractor.extract_all_windows_vectorized(chunk_data)

            # 内存清理
            del local_extractor
            gc.collect()

            processing_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            return features, processing_time, memory_used

        except Exception as e:
            print(f"工作进程 {chunk_idx} 处理失败: {e}")
            raise

    def _calculate_final_stats(self, total_time: float, n_windows: int):
        """计算最终性能统计"""
        # 并行效率：实际加速比 vs 理想加速比
        if self.performance_stats['chunk_processing_times']:
            avg_chunk_time = np.mean(self.performance_stats['chunk_processing_times'])
            sequential_estimate = avg_chunk_time * len(self.performance_stats['chunk_processing_times'])
            self.performance_stats['speedup_factor'] = sequential_estimate / total_time

            # 并行效率
            ideal_speedup = self.max_workers
            actual_speedup = self.performance_stats['speedup_factor']
            self.performance_stats['parallel_efficiency'] = (actual_speedup / ideal_speedup) * 100

        # 内存效率
        memory_used = self.performance_stats['max_memory_used_mb']
        memory_available = self.available_memory / 1024 / 1024
        self.performance_stats['memory_efficiency'] = max(0, (1 - memory_used / memory_available) * 100)

        # CPU利用率（估算）
        self.performance_stats['cpu_utilization'] = min(self.performance_stats['speedup_factor'] / self.max_workers * 100, 100)

        # 总处理时间
        self.performance_stats['total_processing_time'] = total_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats.copy()

    def estimate_processing_time(self, data_shape: Tuple[int, int]) -> Dict[str, float]:
        """估算处理时间"""
        n_samples, n_features = data_shape
        n_windows = (n_samples - self.window_size) // self.stride + 1

        # 基于经验公式估算（需要根据实际硬件调整）
        base_time_per_window = 0.001  # 秒
        sequential_time = n_windows * base_time_per_window
        parallel_time = sequential_time / min(self.max_workers, n_windows)

        return {
            'estimated_sequential_time': sequential_time,
            'estimated_parallel_time': parallel_time,
            'estimated_speedup': sequential_time / parallel_time,
            'total_windows': n_windows
        }

    def benchmark_chunk_sizes(self, data: np.ndarray, chunk_sizes: List[int] = None) -> Dict[int, float]:
        """
        基准测试不同分块大小的性能

        Args:
            data: 测试数据
            chunk_sizes: 要测试的分块大小列表

        Returns:
            performance_results: {chunk_size: processing_time}
        """
        if chunk_sizes is None:
            n_windows = (data.shape[0] - self.window_size) // self.stride + 1
            chunk_sizes = [
                max(n_windows // 16, 1),
                max(n_windows // 8, 1),
                max(n_windows // 4, 1),
                max(n_windows // 2, 1)
            ]

        results = {}
        original_method = self.chunk_size_method

        print(f"\n=== 分块大小基准测试 ===")

        for chunk_size in chunk_sizes:
            print(f"测试分块大小: {chunk_size}")

            # 临时设置固定分块大小
            self.chunk_size_method = 'fixed'
            self._fixed_chunk_size = chunk_size

            start_time = time.time()
            try:
                features, _ = self.process_parallel(data)
                processing_time = time.time() - start_time
                results[chunk_size] = processing_time

                print(f"  处理时间: {processing_time:.2f}秒")

            except Exception as e:
                print(f"  失败: {e}")
                results[chunk_size] = float('inf')

        # 恢复原设置
        self.chunk_size_method = original_method
        if hasattr(self, '_fixed_chunk_size'):
            delattr(self, '_fixed_chunk_size')

        return results


class AdaptiveParallelProcessor:
    """自适应并行处理器，动态调整策略"""

    def __init__(self, base_processor: MemoryEfficientParallelProcessor):
        self.base_processor = base_processor
        self.performance_history = []

    def process_with_adaptation(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        自适应处理：根据系统负载动态调整并行策略
        """
        # 监控系统状态
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # 动态调整工作线程数
        if cpu_percent > 90:
            # CPU高负载，减少工作线程
            self.base_processor.max_workers = max(1, self.base_processor.max_workers // 2)
        elif cpu_percent < 50:
            # CPU低负载，可以增加工作线程
            self.base_processor.max_workers = min(
                self.base_processor.n_cores,
                self.base_processor.max_workers * 2
            )

        # 动态调整内存使用策略
        if memory_percent > 85:
            self.base_processor.max_memory_usage = 0.5  # 保守模式
            self.base_processor.chunk_size_method = 'memory_aware'
        else:
            self.base_processor.max_memory_usage = 0.8  # 正常模式
            self.base_processor.chunk_size_method = 'adaptive'

        print(f"自适应调整: CPU={cpu_percent:.1f}%, 内存={memory_percent:.1f}%")
        print(f"调整后工作线程数: {self.base_processor.max_workers}")

        # 执行处理
        return self.base_processor.process_parallel(data)


# 使用示例和测试代码
if __name__ == '__main__':
    print("=== 并行处理器测试 ===")

    # 创建测试数据
    np.random.seed(42)
    n_samples = 50000
    n_features = 8
    test_data = np.random.randn(n_samples, n_features).astype(np.float32)

    # 添加模式数据
    t = np.linspace(0, 20*np.pi, n_samples)
    for i in range(n_features):
        test_data[:, i] += np.sin((i+1) * t) * (i+1) * 0.5

    print(f"测试数据形状: {test_data.shape}")
    print(f"数据大小: {test_data.nbytes / 1024**2:.1f} MB")

    # 测试并行处理器
    processor = MemoryEfficientParallelProcessor(
        window_size=180,
        stride=30,
        max_workers=4,
        chunk_size_method='adaptive'
    )

    # 估算处理时间
    estimates = processor.estimate_processing_time(test_data.shape)
    print(f"\n时间估算:")
    for key, value in estimates.items():
        print(f"  {key}: {value:.2f}")

    # 执行并行处理
    start_time = time.time()
    features, feature_names = processor.process_parallel(test_data)
    total_time = time.time() - start_time

    print(f"\n=== 测试结果 ===")
    print(f"输出特征形状: {features.shape}")
    print(f"特征名称数量: {len(feature_names)}")
    print(f"总处理时间: {total_time:.2f}秒")

    # 性能统计
    stats = processor.get_performance_stats()
    print(f"\n性能统计:")
    for key, value in stats.items():
        if key != 'chunk_processing_times':  # 跳过详细时间列表
            print(f"  {key}: {value}")

    # 质量验证
    print(f"\n质量验证:")
    print(f"  特征中NaN数量: {np.sum(np.isnan(features))}")
    print(f"  特征中Inf数量: {np.sum(np.isinf(features))}")
    print(f"  特征范围: [{np.min(features):.3f}, {np.max(features):.3f}]")

    print("\n=== 并行处理器测试完成 ===")