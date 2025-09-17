
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
# from PyEMD import EMD # EMD为计算密集型，后续单独集成

class FeatureExtractor:
    """
    从窗口化的时序数据中提取动态和静态特征。
    """
    def __init__(self, window_size=180, stride=30):
        """
        初始化特征提取器。

        Args:
            window_size (int): 滑动窗口大小 (180步, 对应30分钟)。
            stride (int): 窗口滑动的步长 (30步, 对应5分钟)。
        """
        self.window_size = window_size
        self.stride = stride

    def _extract_dynamic_features_from_window(self, window):
        """
        从单个数据窗口中提取所有动态特征。
        """
        features = {}
        
        for col in window.columns:
            series = window[col]
            
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

            # 3. EMD特征 (占位符)
            # 此处需要调用EMD算法，计算IMF1的主频和能量比
            # emd = EMD()
            # imfs = emd(series.values)
            # imf1 = imfs[0]
            # ... 计算主频和能量比 ...
            features[f'{col}_imf1_freq'] = 0.0 # 占位
            features[f'{col}_imf1_energy_ratio'] = 0.0 # 占位

        return features

    def transform(self, df):
        """
        对整个时间序列DataFrame进行特征提取。

        Args:
            df (pd.DataFrame): 包含所有传感器时间序列的DataFrame。

        Returns:
            pd.DataFrame: 每一行代表一个窗口，每一列是一个特征。
        """
        features_list = []

        # 使用tqdm添加进度条
        from tqdm import tqdm

        print("  开始提取动态特征，处理滑动窗口...")
        for i in tqdm(range(0, len(df) - self.window_size + 1, self.stride), desc="    特征提取进度"):
            window = df.iloc[i:i + self.window_size]
            
            # 提取动态特征
            dynamic_features = self._extract_dynamic_features_from_window(window)
            
            # 提取静态特征 (占位符)
            # 静态特征需要从外部配置或数据库中获取，例如设备额定功率等
            # static_features = self._extract_static_features(...)
            # all_features = {**dynamic_features, **static_features}
            
            # 提取标签 (占位符)
            # 标签通常是窗口内或窗口后的能耗值
            # label = df['power_consumption'].iloc[i:i+self.window_size].sum()
            # all_features['label'] = label

            features_list.append(dynamic_features)
            
        return pd.DataFrame(features_list)

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
