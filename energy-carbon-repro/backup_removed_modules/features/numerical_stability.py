"""
数值稳定的特征计算模块
解决scipy.stats在近似相同数据上的精度损失问题
"""

import numpy as np
from scipy import stats
import warnings

class NumericallyStableFeatures:
    """数值稳定的特征计算器"""

    @staticmethod
    def safe_skew_kurtosis(data, axis=0, min_variance_threshold=1e-10):
        """
        数值稳定的偏度和峰度计算

        Args:
            data: 输入数据
            axis: 计算轴
            min_variance_threshold: 最小方差阈值，低于此值认为数据无变异性

        Returns:
            tuple: (skew_result, kurt_result)
        """
        # 确保数据为numpy数组
        if hasattr(data, 'values'):
            data = data.values
        data = np.asarray(data)

        # 检查数据有效性
        if data.size == 0:
            return 0.0, 0.0

        # 计算数据变异性
        data_std = np.std(data, axis=axis)

        # 检查是否为标量
        is_scalar = np.isscalar(data_std)
        if is_scalar:
            data_std = np.array([data_std])

        # 创建结果数组
        skew_result = np.zeros_like(data_std)
        kurt_result = np.zeros_like(data_std)

        # 识别有变异性的数据
        valid_mask = data_std > min_variance_threshold

        if np.any(valid_mask):
            # 只对有变异性的数据计算高阶矩
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                try:
                    if data.ndim == 1:
                        # 一维数据
                        if valid_mask.item() and len(data) > 3:
                            skew_val = stats.skew(data, nan_policy='omit')
                            kurt_val = stats.kurtosis(data, nan_policy='omit')

                            if not (np.isnan(skew_val) or np.isinf(skew_val)):
                                skew_result[0] = skew_val
                            if not (np.isnan(kurt_val) or np.isinf(kurt_val)):
                                kurt_result[0] = kurt_val
                    else:
                        # 多维数据
                        for i, is_valid in enumerate(valid_mask):
                            if is_valid and data.shape[axis] > 3:
                                data_slice = np.take(data, i, axis=1-axis) if axis == 0 else np.take(data, i, axis=axis)

                                skew_val = stats.skew(data_slice, nan_policy='omit')
                                kurt_val = stats.kurtosis(data_slice, nan_policy='omit')

                                if not (np.isnan(skew_val) or np.isinf(skew_val)):
                                    skew_result[i] = skew_val
                                if not (np.isnan(kurt_val) or np.isinf(kurt_val)):
                                    kurt_result[i] = kurt_val

                except Exception as e:
                    # 如果计算失败，返回默认值
                    pass

        # 低变异性数据设为理论默认值
        skew_result[~valid_mask] = 0.0  # 对称分布
        kurt_result[~valid_mask] = 0.0  # 正态分布的超额峰度（调整为0以避免-3）

        # 返回标量或数组
        if is_scalar:
            return skew_result[0], kurt_result[0]
        else:
            return skew_result, kurt_result

    @staticmethod
    def robust_rms(data, axis=0, min_value_threshold=1e-20):
        """
        鲁棒的RMS计算

        Args:
            data: 输入数据
            axis: 计算轴
            min_value_threshold: 最小值阈值，防止接近零的计算

        Returns:
            RMS值
        """
        if hasattr(data, 'values'):
            data = data.values
        data = np.asarray(data)

        # 计算平方均值，添加小的正值防止数值不稳定
        mean_square = np.mean(np.square(data), axis=axis)
        mean_square = np.maximum(mean_square, min_value_threshold)

        return np.sqrt(mean_square)

    @staticmethod
    def safe_mean_change_rate(data, axis=0):
        """
        稳定的均变率计算

        Args:
            data: 输入数据
            axis: 计算轴

        Returns:
            均变率
        """
        if hasattr(data, 'values'):
            data = data.values
        data = np.asarray(data)

        if data.shape[axis] < 2:
            return 0.0

        # 计算差分
        diff = np.diff(data, axis=axis)

        # 计算均值，处理NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_diff = np.nanmean(diff, axis=axis)

        # 替换NaN为0
        if np.isscalar(mean_diff):
            return mean_diff if not np.isnan(mean_diff) else 0.0
        else:
            mean_diff[np.isnan(mean_diff)] = 0.0
            return mean_diff

    @staticmethod
    def safe_std_change_rate(data, axis=0):
        """
        稳定的变率标准差计算

        Args:
            data: 输入数据
            axis: 计算轴

        Returns:
            变率标准差
        """
        if hasattr(data, 'values'):
            data = data.values
        data = np.asarray(data)

        if data.shape[axis] < 2:
            return 0.0

        # 计算差分
        diff = np.diff(data, axis=axis)

        # 计算标准差，处理NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std_diff = np.nanstd(diff, axis=axis)

        # 替换NaN为0
        if np.isscalar(std_diff):
            return std_diff if not np.isnan(std_diff) else 0.0
        else:
            std_diff[np.isnan(std_diff)] = 0.0
            return std_diff

    @staticmethod
    def validate_feature_array(features):
        """
        验证特征数组的数值稳定性

        Args:
            features: 特征数组或字典

        Returns:
            bool: 是否通过验证
        """
        if isinstance(features, dict):
            values = list(features.values())
        else:
            values = features

        for val in values:
            if np.isscalar(val):
                if np.isnan(val) or np.isinf(val):
                    return False
            else:
                val_array = np.asarray(val)
                if np.any(np.isnan(val_array)) or np.any(np.isinf(val_array)):
                    return False

        return True

    @staticmethod
    def get_stability_stats(data):
        """
        获取数据稳定性统计信息

        Args:
            data: 输入数据

        Returns:
            dict: 稳定性统计信息
        """
        if hasattr(data, 'values'):
            data = data.values
        data = np.asarray(data)

        stats_info = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'coefficient_of_variation': np.std(data) / (np.mean(data) + 1e-20),
            'zero_variance_risk': np.std(data) < 1e-10,
            'has_nan': np.any(np.isnan(data)),
            'has_inf': np.any(np.isinf(data))
        }

        return stats_info


# 便捷函数
def safe_statistical_features(data, column_name=""):
    """
    计算一组安全的统计特征

    Args:
        data: 输入数据
        column_name: 列名前缀

    Returns:
        dict: 特征字典
    """
    features = {}
    prefix = f"{column_name}_" if column_name else ""

    # 基础统计量
    features[f'{prefix}mean'] = np.mean(data)
    features[f'{prefix}std'] = np.std(data)

    # 安全的高阶矩
    skew_val, kurt_val = NumericallyStableFeatures.safe_skew_kurtosis(data)
    features[f'{prefix}skew'] = skew_val
    features[f'{prefix}kurtosis'] = kurt_val

    # 鲁棒RMS
    features[f'{prefix}rms'] = NumericallyStableFeatures.robust_rms(data)

    # 安全的变化率
    features[f'{prefix}mean_change_rate'] = NumericallyStableFeatures.safe_mean_change_rate(data)
    features[f'{prefix}std_change_rate'] = NumericallyStableFeatures.safe_std_change_rate(data)

    return features


if __name__ == "__main__":
    # 测试数值稳定性
    print("=== 数值稳定性测试 ===")

    # 测试1：低变异性数据
    low_var_data = np.array([1.0, 1.000001, 0.999999, 1.000002])
    print(f"低变异性数据: {low_var_data}")

    skew1, kurt1 = NumericallyStableFeatures.safe_skew_kurtosis(low_var_data)
    print(f"安全计算 - 偏度: {skew1:.6f}, 峰度: {kurt1:.6f}")

    # 对比原始scipy计算
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            skew_orig = stats.skew(low_var_data)
            kurt_orig = stats.kurtosis(low_var_data)
            print(f"原始计算 - 偏度: {skew_orig:.6f}, 峰度: {kurt_orig:.6f}")
            if w:
                print(f"原始计算产生 {len(w)} 个警告")
    except Exception as e:
        print(f"原始计算失败: {e}")

    # 测试2：正常数据
    normal_data = np.random.randn(100)
    features = safe_statistical_features(normal_data, "test")
    print(f"\n正常数据特征: {features}")

    # 测试3：验证特征
    is_valid = NumericallyStableFeatures.validate_feature_array(features)
    print(f"特征验证通过: {is_valid}")

    print("=== 测试完成 ===")