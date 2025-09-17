"""
智能数据分块策略
Day 3 优化：为大数据集提供内存友好的分块处理
"""
import numpy as np
import pandas as pd
import psutil
from typing import List, Tuple, Iterator, Any, Optional, Union
from pathlib import Path
import warnings


class IntelligentChunker:
    """智能数据分块器，基于系统资源动态调整块大小"""

    def __init__(self,
                 target_memory_usage_percent: float = 60.0,
                 min_chunk_size: int = 1000,
                 max_chunk_size: int = 100000,
                 overlap_ratio: float = 0.1):
        """
        初始化智能分块器

        Args:
            target_memory_usage_percent: 目标内存使用百分比
            min_chunk_size: 最小块大小
            max_chunk_size: 最大块大小
            overlap_ratio: 块之间的重叠比例（用于时序数据）
        """
        self.target_memory_usage = target_memory_usage_percent
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio

        # 系统信息
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_count = psutil.cpu_count()

        # 分块统计
        self.chunking_stats = {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'memory_efficiency': 0.0,
            'processing_time': 0.0
        }

        print(f"  📦 智能分块器初始化:")
        print(f"    系统内存: {self.total_memory_gb:.1f}GB")
        print(f"    CPU核心数: {self.cpu_count}")
        print(f"    目标内存使用率: {self.target_memory_usage}%")

    def calculate_optimal_chunk_size(self,
                                   data_shape: Tuple[int, ...],
                                   dtype: np.dtype,
                                   processing_factor: float = 2.0) -> int:
        """
        计算最优分块大小

        Args:
            data_shape: 数据形状
            dtype: 数据类型
            processing_factor: 处理过程中的内存放大因子

        Returns:
            最优分块大小（行数）
        """
        # 计算单行内存占用
        bytes_per_row = np.prod(data_shape[1:]) * dtype.itemsize

        # 获取可用内存
        available_memory = psutil.virtual_memory().available
        target_memory = available_memory * (self.target_memory_usage / 100.0)

        # 考虑处理过程中的内存放大
        effective_memory = target_memory / processing_factor

        # 计算理论最优块大小
        optimal_size = int(effective_memory / bytes_per_row)

        # 应用约束
        chunk_size = max(self.min_chunk_size,
                        min(optimal_size, self.max_chunk_size))

        print(f"  🧮 块大小计算:")
        print(f"    数据形状: {data_shape}")
        print(f"    单行内存: {bytes_per_row} bytes")
        print(f"    可用内存: {available_memory/(1024**2):.0f}MB")
        print(f"    目标内存: {target_memory/(1024**2):.0f}MB")
        print(f"    计算块大小: {chunk_size} 行")

        return chunk_size

    def create_chunks(self,
                     data: Union[np.ndarray, pd.DataFrame],
                     chunk_size: Optional[int] = None) -> Iterator[Tuple[int, int, Any]]:
        """
        创建数据块迭代器

        Args:
            data: 要分块的数据
            chunk_size: 手动指定块大小，不指定则自动计算

        Yields:
            (start_idx, end_idx, chunk_data)
        """
        if isinstance(data, pd.DataFrame):
            data_shape = data.shape
            dtype = data.dtypes.iloc[0] if len(data.dtypes) > 0 else np.float64
            total_rows = len(data)
        else:
            data_shape = data.shape
            dtype = data.dtype
            total_rows = data.shape[0]

        # 计算块大小
        if chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(data_shape, dtype)

        # 计算重叠大小
        overlap_size = int(chunk_size * self.overlap_ratio)

        print(f"  🔄 创建数据块:")
        print(f"    总行数: {total_rows}")
        print(f"    块大小: {chunk_size}")
        print(f"    重叠大小: {overlap_size}")

        chunk_count = 0
        start_idx = 0

        while start_idx < total_rows:
            # 计算结束索引
            end_idx = min(start_idx + chunk_size, total_rows)

            # 提取数据块
            if isinstance(data, pd.DataFrame):
                chunk_data = data.iloc[start_idx:end_idx].copy()
            else:
                chunk_data = data[start_idx:end_idx].copy()

            yield start_idx, end_idx, chunk_data

            chunk_count += 1

            # 更新下次开始位置（考虑重叠）
            if end_idx >= total_rows:
                break

            start_idx = end_idx - overlap_size

        # 更新统计信息
        self.chunking_stats['total_chunks'] = chunk_count
        self.chunking_stats['avg_chunk_size'] = chunk_size

        print(f"  ✅ 分块完成，共生成 {chunk_count} 个块")

    def create_balanced_chunks(self,
                             data: Union[np.ndarray, pd.DataFrame],
                             target_chunks: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        创建均衡的数据块索引（用于并行处理）

        Args:
            data: 要分块的数据
            target_chunks: 目标块数，默认为CPU核心数

        Returns:
            块索引列表 [(start, end), ...]
        """
        if target_chunks is None:
            target_chunks = self.cpu_count

        total_rows = len(data)
        base_chunk_size = total_rows // target_chunks
        remainder = total_rows % target_chunks

        chunks = []
        start_idx = 0

        for i in range(target_chunks):
            # 前remainder个块多分配一行
            chunk_size = base_chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size

            chunks.append((start_idx, min(end_idx, total_rows)))
            start_idx = end_idx

            if start_idx >= total_rows:
                break

        print(f"  ⚖️ 均衡分块:")
        print(f"    目标块数: {target_chunks}")
        print(f"    实际块数: {len(chunks)}")
        print(f"    块大小范围: {min(end-start for start, end in chunks)} - {max(end-start for start, end in chunks)}")

        return chunks

    def estimate_processing_memory(self,
                                 chunk_shape: Tuple[int, ...],
                                 dtype: np.dtype,
                                 window_size: int = 180,
                                 feature_multiplier: float = 8.0) -> float:
        """
        估算处理特定块时的内存需求

        Args:
            chunk_shape: 数据块形状
            dtype: 数据类型
            window_size: 滑动窗口大小
            feature_multiplier: 特征提取的内存放大倍数

        Returns:
            估算内存需求（MB）
        """
        # 原始数据内存
        raw_memory = np.prod(chunk_shape) * dtype.itemsize

        # 滑动窗口内存（需要创建窗口视图）
        window_memory = chunk_shape[0] * window_size * chunk_shape[1] * dtype.itemsize

        # 特征提取内存（通常是原始数据的数倍）
        feature_memory = raw_memory * feature_multiplier

        # 总内存需求
        total_memory = (raw_memory + window_memory + feature_memory) / (1024**2)

        return total_memory

    def validate_chunk_memory(self,
                            chunk_shape: Tuple[int, ...],
                            dtype: np.dtype) -> bool:
        """
        验证块是否在内存限制内

        Args:
            chunk_shape: 数据块形状
            dtype: 数据类型

        Returns:
            是否在内存限制内
        """
        estimated_memory = self.estimate_processing_memory(chunk_shape, dtype)
        available_memory = psutil.virtual_memory().available / (1024**2)

        memory_ok = estimated_memory < (available_memory * 0.8)  # 80%安全边际

        if not memory_ok:
            print(f"  ⚠️ 内存警告:")
            print(f"    估算需求: {estimated_memory:.0f}MB")
            print(f"    可用内存: {available_memory:.0f}MB")

        return memory_ok

    def adaptive_chunk_size(self,
                          data: Union[np.ndarray, pd.DataFrame],
                          processing_function: callable,
                          initial_chunk_size: Optional[int] = None) -> int:
        """
        自适应调整块大小，基于实际处理性能

        Args:
            data: 数据
            processing_function: 处理函数
            initial_chunk_size: 初始块大小

        Returns:
            优化后的块大小
        """
        import time

        if isinstance(data, pd.DataFrame):
            data_shape = data.shape
            dtype = data.dtypes.iloc[0]
        else:
            data_shape = data.shape
            dtype = data.dtype

        # 使用初始块大小或计算默认值
        if initial_chunk_size is None:
            chunk_size = self.calculate_optimal_chunk_size(data_shape, dtype)
        else:
            chunk_size = initial_chunk_size

        print(f"  🔧 自适应块大小调整:")
        print(f"    初始块大小: {chunk_size}")

        # 测试几个不同大小的块
        test_sizes = [
            chunk_size // 2,
            chunk_size,
            min(chunk_size * 2, self.max_chunk_size)
        ]

        best_size = chunk_size
        best_efficiency = 0.0

        for test_size in test_sizes:
            if test_size < self.min_chunk_size:
                continue

            # 创建测试块
            test_chunk = data.iloc[:test_size] if isinstance(data, pd.DataFrame) else data[:test_size]

            # 检查内存约束
            if not self.validate_chunk_memory(test_chunk.shape, dtype):
                continue

            # 测试处理性能
            try:
                start_time = time.time()
                _ = processing_function(test_chunk)
                end_time = time.time()

                processing_time = end_time - start_time
                efficiency = test_size / processing_time  # 行/秒

                print(f"    测试块大小 {test_size}: {efficiency:.0f} 行/秒")

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_size = test_size

            except Exception as e:
                print(f"    测试块大小 {test_size}: 处理失败 - {e}")
                continue

        print(f"    最佳块大小: {best_size} (效率: {best_efficiency:.0f} 行/秒)")
        return best_size

    def get_chunking_stats(self) -> dict:
        """获取分块统计信息"""
        return self.chunking_stats.copy()


class TimeSeriesChunker(IntelligentChunker):
    """专门用于时序数据的分块器"""

    def __init__(self,
                 window_size: int = 180,
                 stride: int = 30,
                 **kwargs):
        """
        初始化时序数据分块器

        Args:
            window_size: 滑动窗口大小
            stride: 窗口步长
            **kwargs: 其他参数传递给父类
        """
        super().__init__(**kwargs)
        self.window_size = window_size
        self.stride = stride

        # 确保重叠足够覆盖窗口需求
        min_overlap = window_size * 1.5  # 1.5倍窗口大小的重叠
        self.overlap_ratio = max(self.overlap_ratio, min_overlap / self.min_chunk_size)

        print(f"  📈 时序分块器配置:")
        print(f"    滑动窗口: {self.window_size}")
        print(f"    步长: {self.stride}")
        print(f"    重叠比例: {self.overlap_ratio:.3f}")

    def create_windowed_chunks(self,
                             data: Union[np.ndarray, pd.DataFrame],
                             ensure_complete_windows: bool = True) -> Iterator[Tuple[int, int, Any]]:
        """
        创建考虑滑动窗口的时序数据块

        Args:
            data: 时序数据
            ensure_complete_windows: 是否确保每个块都能生成完整的窗口

        Yields:
            (start_idx, end_idx, chunk_data)
        """
        total_rows = len(data)

        # 计算基础块大小
        if isinstance(data, pd.DataFrame):
            chunk_size = self.calculate_optimal_chunk_size(data.shape, data.dtypes.iloc[0])
        else:
            chunk_size = self.calculate_optimal_chunk_size(data.shape, data.dtype)

        # 如果需要完整窗口，调整块大小
        if ensure_complete_windows:
            # 确保块大小至少能生成一些窗口
            min_size_for_windows = self.window_size + self.stride * 2
            chunk_size = max(chunk_size, min_size_for_windows)

        print(f"  🪟 时序分块:")
        print(f"    调整后块大小: {chunk_size}")
        print(f"    确保完整窗口: {ensure_complete_windows}")

        # 创建重叠块
        overlap_size = max(self.window_size, int(chunk_size * self.overlap_ratio))

        chunk_count = 0
        start_idx = 0

        while start_idx < total_rows:
            # 计算结束索引
            end_idx = min(start_idx + chunk_size, total_rows)

            # 检查是否能生成足够的窗口
            if ensure_complete_windows and (end_idx - start_idx) < self.window_size:
                print(f"    跳过块 {chunk_count}: 数据不足以生成窗口")
                break

            # 提取数据块
            if isinstance(data, pd.DataFrame):
                chunk_data = data.iloc[start_idx:end_idx].copy()
            else:
                chunk_data = data[start_idx:end_idx].copy()

            yield start_idx, end_idx, chunk_data
            chunk_count += 1

            # 更新下次开始位置
            if end_idx >= total_rows:
                break

            start_idx = end_idx - overlap_size

        self.chunking_stats['total_chunks'] = chunk_count
        self.chunking_stats['avg_chunk_size'] = chunk_size

        print(f"  ✅ 时序分块完成，共生成 {chunk_count} 个重叠块")

    def estimate_window_count(self, chunk_size: int) -> int:
        """估算块能生成的窗口数量"""
        if chunk_size < self.window_size:
            return 0
        return (chunk_size - self.window_size) // self.stride + 1

    def validate_temporal_consistency(self,
                                    chunk_results: List[Any],
                                    overlap_size: int) -> bool:
        """
        验证重叠块之间的时序一致性

        Args:
            chunk_results: 各块的处理结果
            overlap_size: 重叠大小

        Returns:
            是否时序一致
        """
        if len(chunk_results) < 2:
            return True

        # 简单的一致性检查：比较重叠区域的特征
        try:
            for i in range(len(chunk_results) - 1):
                current_result = chunk_results[i]
                next_result = chunk_results[i + 1]

                # 这里应该根据具体的结果格式进行一致性检查
                # 目前只做基本的形状检查
                if hasattr(current_result, 'shape') and hasattr(next_result, 'shape'):
                    if current_result.shape[1:] != next_result.shape[1:]:
                        return False

            return True
        except Exception:
            return False


# 使用示例和测试
if __name__ == '__main__':
    print("=== 智能分块器测试 ===")

    # 创建测试数据
    np.random.seed(42)
    test_data = np.random.randn(50000, 8)  # 50k行，8列
    test_df = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(8)])

    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据大小: {test_data.nbytes / (1024**2):.1f}MB")

    # 测试普通分块器
    print("\n--- 测试普通分块器 ---")
    chunker = IntelligentChunker(target_memory_usage_percent=50.0)

    chunk_count = 0
    total_processed = 0

    for start_idx, end_idx, chunk_data in chunker.create_chunks(test_df):
        chunk_count += 1
        total_processed += len(chunk_data)

        if chunk_count <= 3:  # 只显示前3个块的信息
            print(f"  块 {chunk_count}: 行 {start_idx}-{end_idx}, 大小 {len(chunk_data)}")

    print(f"总共处理 {chunk_count} 个块，{total_processed} 行数据")

    # 测试时序分块器
    print("\n--- 测试时序分块器 ---")
    ts_chunker = TimeSeriesChunker(
        window_size=180,
        stride=30,
        target_memory_usage_percent=50.0
    )

    chunk_count = 0
    for start_idx, end_idx, chunk_data in ts_chunker.create_windowed_chunks(test_df):
        chunk_count += 1
        window_count = ts_chunker.estimate_window_count(len(chunk_data))

        if chunk_count <= 3:
            print(f"  时序块 {chunk_count}: 行 {start_idx}-{end_idx}, "
                  f"大小 {len(chunk_data)}, 可生成窗口 {window_count}")

    print(f"时序分块生成 {chunk_count} 个块")

    # 测试均衡分块
    print("\n--- 测试均衡分块 ---")
    balanced_chunks = chunker.create_balanced_chunks(test_df, target_chunks=4)
    for i, (start, end) in enumerate(balanced_chunks):
        print(f"  均衡块 {i+1}: 行 {start}-{end}, 大小 {end-start}")

    print("\n=== 智能分块器测试完成 ===")