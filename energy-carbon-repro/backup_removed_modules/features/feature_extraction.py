
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
# from PyEMD import EMD # EMD为计算密集型，后续单独集成

# 导入数值稳定的特征计算模块
from .numerical_stability import NumericallyStableFeatures, safe_statistical_features
from .vectorized_extraction import VectorizedFeatureExtractor, BatchFeatureProcessor

class FeatureExtractor:
    """
    从窗口化的时序数据中提取动态和静态特征。
    优化版本：集成数值稳定的特征计算
    """
    def __init__(self, window_size=180, stride=30, use_stable_features=True, use_vectorized=True):
        """
        初始化特征提取器。

        Args:
            window_size (int): 滑动窗口大小 (180步, 对应30分钟)。
            stride (int): 窗口滑动的步长 (30步, 对应5分钟)。
            use_stable_features (bool): 是否使用数值稳定的特征计算
            use_vectorized (bool): 是否使用向量化特征提取（Day 2 优化）
        """
        self.window_size = window_size
        self.stride = stride
        self.use_stable_features = use_stable_features
        self.use_vectorized = use_vectorized

        # 性能统计
        self.extraction_stats = {
            'total_windows': 0,
            'failed_calculations': 0,
            'stability_warnings': 0,
            'optimization_level': 'vectorized' if use_vectorized else 'stable' if use_stable_features else 'original'
        }

        # 初始化向量化提取器
        if use_vectorized:
            self.batch_processor = BatchFeatureProcessor(window_size, stride, use_cache=True)
            print("  初始化向量化特征提取器 (Day 2 优化)")
        else:
            print(f"  初始化传统特征提取器 (优化级别: {self.extraction_stats['optimization_level']})")

    def _extract_dynamic_features_from_window(self, window):
        """
        从单个数据窗口中提取所有动态特征。
        优化版本：使用数值稳定的计算方法
        """
        features = {}

        for col in window.columns:
            series = window[col]

            if self.use_stable_features:
                # 使用数值稳定的特征计算
                try:
                    stable_features = safe_statistical_features(series.values, col)
                    features.update(stable_features)

                    # 验证特征质量
                    if not NumericallyStableFeatures.validate_feature_array(stable_features):
                        self.extraction_stats['stability_warnings'] += 1
                        # 回退到基础统计量
                        features[f'{col}_mean'] = series.mean()
                        features[f'{col}_std'] = series.std()
                        features[f'{col}_skew'] = 0.0
                        features[f'{col}_kurtosis'] = 0.0
                        features[f'{col}_rms'] = np.sqrt(np.mean(series**2))
                        features[f'{col}_mean_change_rate'] = 0.0
                        features[f'{col}_std_change_rate'] = 0.0

                except Exception as e:
                    self.extraction_stats['failed_calculations'] += 1
                    # 提供安全的默认值
                    features[f'{col}_mean'] = series.mean() if not series.isna().all() else 0.0
                    features[f'{col}_std'] = series.std() if not series.isna().all() else 0.0
                    features[f'{col}_skew'] = 0.0
                    features[f'{col}_kurtosis'] = 0.0
                    features[f'{col}_rms'] = 0.0
                    features[f'{col}_mean_change_rate'] = 0.0
                    features[f'{col}_std_change_rate'] = 0.0

            else:
                # 使用原始计算方法（保持向后兼容）
                try:
                    # 1. 基础统计特征
                    features[f'{col}_mean'] = series.mean()
                    features[f'{col}_std'] = series.std()
                    features[f'{col}_skew'] = skew(series)
                    features[f'{col}_kurtosis'] = kurtosis(series)
                    features[f'{col}_rms'] = np.sqrt(np.mean(series**2))

                    # 2. 变化率特征
                    diffs = series.diff().dropna()
                    features[f'{col}_mean_rate_of_change'] = diffs.mean()
                    features[f'{col}_std_rate_of_change'] = diffs.std()
                except Exception:
                    # 提供默认值
                    self.extraction_stats['failed_calculations'] += 1
                    features[f'{col}_mean'] = 0.0
                    features[f'{col}_std'] = 0.0
                    features[f'{col}_skew'] = 0.0
                    features[f'{col}_kurtosis'] = 0.0
                    features[f'{col}_rms'] = 0.0
                    features[f'{col}_mean_rate_of_change'] = 0.0
                    features[f'{col}_std_rate_of_change'] = 0.0

            # 3. EMD特征 (占位符)
            features[f'{col}_imf1_freq'] = 0.0 # 占位
            features[f'{col}_imf1_energy_ratio'] = 0.0 # 占位

        return features

    def transform(self, df):
        """
        对整个时间序列DataFrame进行特征提取。
        优化版本：支持向量化批量计算和数值稳定性

        Args:
            df (pd.DataFrame): 包含所有传感器时间序列的DataFrame。

        Returns:
            pd.DataFrame: 每一行代表一个窗口，每一列是一个特征。
        """
        import time
        import hashlib
        start_time = time.time()

        print(f"  开始特征提取，优化级别: {self.extraction_stats['optimization_level']}")
        print(f"    数据形状: {df.shape}")

        if self.use_vectorized:
            # Day 2 优化：使用向量化批量处理
            print("  使用向量化特征提取器 (Day 2 优化)")

            # 生成缓存键
            data_hash = hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()[:16]
            cache_key = f"vectorized_{data_hash}_{self.window_size}_{self.stride}"

            # 确保数据是数值类型
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError("DataFrame中没有数值列")

            try:
                # 向量化特征提取
                features, feature_names = self.batch_processor.process_with_caching(
                    numeric_df.values, cache_key
                )

                # 创建DataFrame
                result_df = pd.DataFrame(features, columns=feature_names)

                # 更新统计信息
                self.extraction_stats['total_windows'] = len(result_df)
                extraction_time = time.time() - start_time
                self.extraction_stats['extraction_time'] = extraction_time

                print(f"  向量化特征提取完成，耗时: {extraction_time:.2f}秒")
                print(f"    生成特征数据形状: {result_df.shape}")
                print(f"    平均处理速度: {len(result_df)/extraction_time:.1f} 窗口/秒")

                # 缓存统计
                cache_stats = self.batch_processor.get_cache_stats()
                if cache_stats['cache_entries'] > 0:
                    print(f"    缓存统计: {cache_stats['cache_entries']} 条目, "
                          f"{cache_stats['total_cache_size_mb']:.1f}MB")

                return result_df

            except Exception as e:
                print(f"  向量化提取失败，回退到传统方法: {e}")
                # 回退到传统方法
                return self._transform_traditional(df, start_time)

        else:
            # 使用传统方法
            return self._transform_traditional(df, start_time)

    def _transform_traditional(self, df, start_time=None):
        """传统的逐窗口特征提取方法"""
        import time
        if start_time is None:
            start_time = time.time()

        features_list = []

        # 使用tqdm添加进度条
        from tqdm import tqdm

        # 计算窗口数量
        n_windows = (len(df) - self.window_size) // self.stride + 1
        self.extraction_stats['total_windows'] = n_windows

        print("  使用传统逐窗口特征提取...")
        if self.use_stable_features:
            print("    使用数值稳定的特征计算方法")
        else:
            print("    使用原始特征计算方法")

        for i in tqdm(range(0, len(df) - self.window_size + 1, self.stride), desc="    特征提取进度"):
            window = df.iloc[i:i + self.window_size]

            # 提取动态特征
            dynamic_features = self._extract_dynamic_features_from_window(window)

            features_list.append(dynamic_features)

        end_time = time.time()
        extraction_time = end_time - start_time

        # 打印性能统计
        print(f"  传统特征提取完成，耗时: {extraction_time:.2f}秒")
        if self.use_stable_features:
            print(f"    数值稳定性统计:")
            print(f"      总窗口数: {self.extraction_stats['total_windows']}")
            print(f"      计算失败: {self.extraction_stats['failed_calculations']}")
            print(f"      稳定性警告: {self.extraction_stats['stability_warnings']}")

            if self.extraction_stats['total_windows'] > 0:
                success_rate = 1 - (self.extraction_stats['failed_calculations'] / self.extraction_stats['total_windows'])
                print(f"      成功率: {success_rate:.2%}")

        result_df = pd.DataFrame(features_list)
        print(f"  生成特征数据形状: {result_df.shape}")

        return result_df

    def get_extraction_stats(self):
        """获取特征提取统计信息"""
        return self.extraction_stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.extraction_stats = {
            'total_windows': 0,
            'failed_calculations': 0,
            'stability_warnings': 0
        }

# 使用示例
if __name__ == '__main__':
    # 创建一个模拟的DataFrame
    data = {
        'sensor1': np.random.randn(1000),
        'sensor2': np.random.randn(1000) * 2 + 5
    }
    mock_df = pd.DataFrame(data)

    # 初始化并运行特征提取器
    extractor = FeatureExtractor(window_size=180, stride=30)
    feature_df = extractor.transform(mock_df)

    print("特征提取完成。")
    print(f"生成窗口数: {len(feature_df)}")
    print(f"生成特征数: {len(feature_df.columns)}")
    print("特征预览:")
    print(feature_df.head())
