"""
生产级GPR实现 - Week 2 优化
实现鲁棒的高斯过程回归，增强噪声保护和数值稳定性
"""

import numpy as np
import pandas as pd
import warnings
import time
from typing import Tuple, Optional, Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, RBF, WhiteKernel, ConstantKernel as C,
    RationalQuadratic, ExpSineSquared, DotProduct
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib


class ProductionGPR:
    """
    生产级高斯过程回归实现
    包含自适应核选择、噪声保护、数值稳定性增强
    """

    def __init__(self, n_samples: int = 5000, auto_tune: bool = True, random_state: int = 42):
        """
        初始化生产级GPR

        Args:
            n_samples: 训练样本数量，用于自适应参数调整
            auto_tune: 是否自动调参
            random_state: 随机种子
        """
        self.n_samples = n_samples
        self.auto_tune = auto_tune
        self.random_state = random_state

        # 模型组件
        self.gpr = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        # 性能统计
        self.performance_stats = {
            'fit_time': 0.0,
            'log_marginal_likelihood': 0.0,
            'kernel_parameters': {},
            'validation_scores': {},
            'numerical_stability_check': True,
            'noise_level': 0.0
        }

        print(f"  初始化生产级GPR: n_samples={n_samples}, auto_tune={auto_tune}")

    def create_robust_kernel(self) -> Any:
        """
        创建鲁棒的核函数，根据数据规模自适应调整

        Returns:
            适合当前数据规模的核函数
        """
        if self.n_samples < 1000:
            # 小样本：更灵活的核函数组合
            kernel = (
                C(1.0, (1e-3, 1e3)) *
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
            )
            print(f"    使用小样本核配置 (n={self.n_samples})")

        elif self.n_samples < 5000:
            # 中等样本：平衡灵活性和稳定性
            kernel = (
                C(1.0, (1e-2, 1e2)) *
                Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e1), nu=2.5) +
                WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-2))
            )
            print(f"    使用中等样本核配置 (n={self.n_samples})")

        else:
            # 大样本：更保守但稳定的配置
            kernel = (
                C(1.0, (1e-1, 1e1)) *
                Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e1), nu=2.5) +
                WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-2))
            )
            print(f"    使用大样本核配置 (n={self.n_samples})")

        return kernel

    def create_composite_kernel(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        创建复合核函数，适合复杂数据模式

        Args:
            X: 特征数据
            y: 目标变量

        Returns:
            复合核函数
        """
        print("    创建复合核函数...")

        # 分析数据特征
        n_features = X.shape[1]
        y_variation = np.std(y) / (np.mean(np.abs(y)) + 1e-10)

        # 基础平滑核
        base_kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e1),
            nu=2.5
        )

        # 添加周期性核（如果数据显示周期性）
        if self._detect_periodicity(y):
            print("    检测到周期性，添加周期核")
            periodic_kernel = ExpSineSquared(
                length_scale=1.0,
                periodicity=1.0,
                length_scale_bounds=(1e-1, 1e1),
                periodicity_bounds=(1e-1, 1e1)
            )
            base_kernel = base_kernel + C(0.1, (1e-3, 1e1)) * periodic_kernel

        # 添加线性趋势核（如果数据有明显趋势）
        if self._detect_linear_trend(X, y):
            print("    检测到线性趋势，添加线性核")
            linear_kernel = C(1.0, (1e-3, 1e1)) * DotProduct(sigma_0=1.0)
            base_kernel = base_kernel + linear_kernel

        # 添加噪声核
        noise_kernel = WhiteKernel(
            noise_level=max(1e-6, y_variation * 0.01),
            noise_level_bounds=(1e-10, 1e-1)
        )

        composite_kernel = base_kernel + noise_kernel

        print(f"    复合核创建完成，包含 {len(str(composite_kernel).split('+'))} 个组件")
        return composite_kernel

    def _detect_periodicity(self, y: np.ndarray, threshold: float = 0.3) -> bool:
        """检测数据是否有周期性"""
        try:
            # 简单的自相关检测
            from scipy.signal import correlate

            autocorr = correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # 归一化

            # 寻找除了滞后0外的峰值
            peaks = np.where((autocorr[1:-1] > autocorr[:-2]) &
                           (autocorr[1:-1] > autocorr[2:]))[0] + 1

            return len(peaks) > 0 and np.max(autocorr[peaks]) > threshold
        except:
            return False

    def _detect_linear_trend(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.1) -> bool:
        """检测数据是否有明显的线性趋势"""
        try:
            # 简单线性回归检验
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            lr.fit(X, y)
            r2 = lr.score(X, y)

            return r2 > threshold
        except:
            return False

    def fit_with_validation(self, X: np.ndarray, y: np.ndarray,
                          validation_split: float = 0.2) -> Dict[str, Any]:
        """
        带验证的GPR拟合

        Args:
            X: 特征数据
            y: 目标变量
            validation_split: 验证集比例

        Returns:
            拟合结果统计
        """
        start_time = time.time()
        print(f"    开始GPR拟合，数据形状: X{X.shape}, y{y.shape}")

        # 数据预处理
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # 分割训练/验证集
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled,
                test_size=validation_split,
                random_state=self.random_state
            )
        else:
            X_train, y_train = X_scaled, y_scaled
            X_val, y_val = None, None

        # 创建核函数
        if self.auto_tune:
            kernel = self.create_composite_kernel(X_train, y_train)
        else:
            kernel = self.create_robust_kernel()

        # 创建GPR模型
        n_restarts = self._calculate_optimal_restarts(len(y_train))
        alpha_value = self._calculate_optimal_alpha(y_train)

        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_value,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state,
            normalize_y=False  # 已经手动标准化
        )

        print(f"    GPR参数: n_restarts={n_restarts}, alpha={alpha_value:.2e}")

        # 拟合模型
        try:
            self.gpr.fit(X_train, y_train)
            print("    GPR拟合成功")

            # 验证模型质量
            validation_results = self._validate_model_quality(
                X_train, y_train, X_val, y_val
            )

            # 记录性能统计
            self._record_performance_stats(start_time, validation_results)

            return validation_results

        except Exception as e:
            print(f"    GPR拟合失败: {e}")
            # 降级到简单核函数重试
            return self._fallback_fit(X_train, y_train, X_val, y_val, start_time)

    def _calculate_optimal_restarts(self, n_samples: int) -> int:
        """计算最优的优化重启次数"""
        if n_samples < 500:
            return 10
        elif n_samples < 2000:
            return 5
        elif n_samples < 5000:
            return 3
        else:
            return 2

    def _calculate_optimal_alpha(self, y: np.ndarray) -> float:
        """计算最优的噪声正则化参数"""
        y_var = np.var(y)
        y_std = np.std(y)

        # 基于数据方差的自适应alpha
        if y_var < 0.01:  # 低噪声数据
            return 1e-10
        elif y_var < 0.1:  # 中等噪声数据
            return 1e-8
        else:  # 高噪声数据
            return max(1e-6, y_std * 0.001)

    def _fallback_fit(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray], y_val: Optional[np.ndarray],
                     start_time: float) -> Dict[str, Any]:
        """降级拟合策略"""
        print("    使用降级核函数重新拟合...")

        # 使用最简单稳定的核
        simple_kernel = (
            C(1.0, (1e-1, 10.0)) *
            RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) +
            WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )

        self.gpr = GaussianProcessRegressor(
            kernel=simple_kernel,
            alpha=1e-3,  # 较大的alpha保证数值稳定性
            n_restarts_optimizer=3,
            random_state=self.random_state
        )

        try:
            self.gpr.fit(X_train, y_train)
            print("    降级拟合成功")

            validation_results = self._validate_model_quality(
                X_train, y_train, X_val, y_val
            )
            self._record_performance_stats(start_time, validation_results)

            return validation_results

        except Exception as e:
            print(f"    降级拟合也失败: {e}")
            raise RuntimeError("GPR拟合完全失败，请检查数据质量")

    def _validate_model_quality(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """验证模型质量"""
        results = {}

        # 训练集性能
        y_train_pred = self.gpr.predict(X_train)
        results['train_r2'] = r2_score(y_train, y_train_pred)
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))

        print(f"    训练集性能: R²={results['train_r2']:.4f}, RMSE={results['train_rmse']:.4f}")

        # 验证集性能（如果有）
        if X_val is not None and y_val is not None:
            y_val_pred = self.gpr.predict(X_val)
            results['val_r2'] = r2_score(y_val, y_val_pred)
            results['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))

            print(f"    验证集性能: R²={results['val_r2']:.4f}, RMSE={results['val_rmse']:.4f}")

        # 检查方差塌陷
        pred_std_sample = self.gpr.predict(X_train[:min(100, len(X_train))], return_std=True)[1]
        mean_std = np.mean(pred_std_sample)
        results['mean_prediction_std'] = mean_std

        if mean_std < 1e-6:
            print("    ⚠️ 检测到方差塌陷，调整噪声参数")
            self.gpr.alpha = max(self.gpr.alpha * 100, 1e-3)
            self.gpr.fit(X_train, y_train)  # 重新拟合
            results['variance_collapse_detected'] = True
        else:
            results['variance_collapse_detected'] = False

        # 检查数值稳定性
        log_likelihood = self.gpr.log_marginal_likelihood()
        if np.isnan(log_likelihood) or np.isinf(log_likelihood):
            results['numerical_stability_check'] = False
            print(f"    ⚠️ 数值不稳定: log_likelihood={log_likelihood}")
        else:
            results['numerical_stability_check'] = True
            print(f"    数值稳定性: ✅ (log_likelihood={log_likelihood:.2f})")

        return results

    def _record_performance_stats(self, start_time: float, validation_results: Dict[str, Any]):
        """记录性能统计信息"""
        self.performance_stats.update({
            'fit_time': time.time() - start_time,
            'log_marginal_likelihood': self.gpr.log_marginal_likelihood(),
            'kernel_parameters': dict(self.gpr.kernel_.get_params()),
            'validation_scores': validation_results,
            'numerical_stability_check': validation_results.get('numerical_stability_check', False),
            'noise_level': self._extract_noise_level()
        })

        print(f"    拟合完成，用时: {self.performance_stats['fit_time']:.2f}秒")

    def _extract_noise_level(self) -> float:
        """提取拟合后的噪声水平"""
        try:
            kernel_str = str(self.gpr.kernel_)
            if 'WhiteKernel' in kernel_str:
                # 从核参数中提取噪声水平
                for param_name, param_value in self.gpr.kernel_.get_params().items():
                    if 'noise_level' in param_name:
                        return float(param_value)
            return self.gpr.alpha if hasattr(self.gpr, 'alpha') else 0.0
        except:
            return 0.0

    def predict_with_uncertainty(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测并返回不确定性

        Args:
            X: 预测特征
            return_std: 是否返回标准差

        Returns:
            预测值和不确定性（如果requested）
        """
        if self.gpr is None:
            raise RuntimeError("模型尚未训练，请先调用fit_with_validation")

        # 标准化输入
        X_scaled = self.scaler_X.transform(X)

        # 预测
        if return_std:
            y_pred_scaled, y_std_scaled = self.gpr.predict(X_scaled, return_std=True)

            # 反标准化
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            y_std = y_std_scaled * self.scaler_y.scale_[0]

            return y_pred, y_std
        else:
            y_pred_scaled = self.gpr.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            return y_pred, None

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_stats['fit_time']:
            return {"message": "模型尚未训练"}

        summary = {
            'fit_time_seconds': self.performance_stats['fit_time'],
            'log_marginal_likelihood': self.performance_stats['log_marginal_likelihood'],
            'noise_level': self.performance_stats['noise_level'],
            'numerical_stability': self.performance_stats['numerical_stability_check'],
            'kernel_complexity': len(str(self.gpr.kernel_).split('+')),
            'validation_scores': self.performance_stats['validation_scores']
        }

        return summary

    def save_model(self, filepath: str):
        """保存模型"""
        if self.gpr is None:
            raise RuntimeError("模型尚未训练")

        model_data = {
            'gpr': self.gpr,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'performance_stats': self.performance_stats
        }

        joblib.dump(model_data, filepath)
        print(f"    模型已保存: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)

        self.gpr = model_data['gpr']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.performance_stats = model_data['performance_stats']

        print(f"    模型已加载: {filepath}")


# 便捷函数
def create_production_gpr(X: np.ndarray, y: np.ndarray,
                         auto_tune: bool = True,
                         validation_split: float = 0.2) -> ProductionGPR:
    """
    创建生产级GPR模型的便捷函数

    Args:
        X: 特征数据
        y: 目标变量
        auto_tune: 是否自动调参
        validation_split: 验证集比例

    Returns:
        训练好的ProductionGPR实例
    """
    print(f"创建生产级GPR模型: 数据量={len(X)}, 特征数={X.shape[1]}")

    gpr = ProductionGPR(n_samples=len(X), auto_tune=auto_tune)
    validation_results = gpr.fit_with_validation(X, y, validation_split)

    print("GPR训练完成")
    print("性能摘要:", gpr.get_performance_summary())

    return gpr


# 测试代码
if __name__ == '__main__':
    print("=== 生产级GPR测试 ===")

    # 生成测试数据
    np.random.seed(42)
    n_samples = 2000
    n_features = 5

    X = np.random.randn(n_samples, n_features)

    # 创建复杂的非线性关系
    y = (
        2 * X[:, 0] +
        np.sin(3 * X[:, 1]) +
        0.5 * X[:, 2] ** 2 +
        0.1 * np.random.randn(n_samples)  # 噪声
    )

    print(f"测试数据: X{X.shape}, y{y.shape}")
    print(f"目标变量统计: mean={np.mean(y):.3f}, std={np.std(y):.3f}")

    # 测试生产级GPR
    print("\n--- 生产级GPR测试 ---")

    start_time = time.time()
    production_gpr = create_production_gpr(X, y, auto_tune=True)
    total_time = time.time() - start_time

    print(f"总训练时间: {total_time:.2f}秒")

    # 测试预测
    X_test = X[:100]  # 使用前100个样本做测试
    y_pred, y_std = production_gpr.predict_with_uncertainty(X_test)

    print(f"预测测试: pred_shape={y_pred.shape}, std_shape={y_std.shape}")
    print(f"预测不确定性统计: mean_std={np.mean(y_std):.4f}")

    # 计算测试指标
    y_true = y[:100]
    test_r2 = r2_score(y_true, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"测试性能: R²={test_r2:.4f}, RMSE={test_rmse:.4f}")

    # 模型保存测试
    model_path = "test_production_gpr.pkl"
    production_gpr.save_model(model_path)

    print("\n=== 生产级GPR测试完成 ===")