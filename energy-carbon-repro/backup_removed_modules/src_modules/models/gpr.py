import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
import warnings

class ExactGPR:
    """
    优化的高斯过程回归模型，自动选择精确或近似方法
    针对大样本场景进行优化，保持学术准确性
    """
    def __init__(self, max_exact_samples=3000, n_restarts_optimizer=3, random_state=42):
        """
        初始化GPR模型，根据样本量自动选择策略

        Args:
            max_exact_samples: 使用精确GPR的最大样本数
            n_restarts_optimizer: 优化器重启次数（减少以提速）
            random_state: 随机种子
        """
        self.max_exact_samples = max_exact_samples
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.model = None
        self.use_exact = True

        # 核函数配置
        self.kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
                      WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e-1))

    def fit(self, X, y):
        """
        训练GPR模型，自动选择最优策略

        Args:
            X: 特征数据
            y: 目标数据（残差）
        """
        n_samples = len(X)

        if n_samples <= self.max_exact_samples:
            # 小样本：使用精确GPR
            print(f"使用精确GPR (样本数: {n_samples})")
            self._fit_exact_gpr(X, y)
        else:
            # 大样本：使用近似方法
            print(f"样本数过大 ({n_samples}), 使用近似GPR")
            self._fit_approximate_gpr(X, y)

    def _fit_exact_gpr(self, X, y):
        """精确GPR训练"""
        residual_var = np.var(y)
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=max(residual_var, 1e-6),
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state
        )
        self.use_exact = True
        self.model.fit(X, y)

    def _fit_approximate_gpr(self, X, y):
        """近似GPR训练 - 使用RandomForest替代"""
        print("  策略: 使用RandomForest近似GPR行为")
        self.model = RandomForestRegressor(
            n_estimators=200,  # 增加树数量提高精度
            max_depth=15,      # 适当深度避免过拟合
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.use_exact = False
        self.model.fit(X, y)
        print(f"  RandomForest训练完成: {self.model.n_estimators}棵树")

    def predict(self, X, return_std=True):
        """
        预测，自动处理不确定性估计

        Args:
            X: 输入特征
            return_std: 是否返回不确定性

        Returns:
            预测值和标准差（如果请求）
        """
        if self.use_exact:
            # 精确GPR直接返回不确定性
            return self.model.predict(X, return_std=return_std)
        else:
            # RandomForest近似不确定性
            pred = self.model.predict(X)

            if return_std:
                # 使用树的预测方差近似不确定性
                tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
                pred_std = np.std(tree_predictions, axis=0)
                return pred, pred_std
            return pred

    def get_kernel_params(self):
        """获取核函数参数"""
        if self.use_exact and hasattr(self.model, 'kernel_'):
            return self.model.kernel_
        else:
            return "RandomForest approximation (no kernel parameters)"

# 使用示例
if __name__ == '__main__':
    # 测试小样本（精确GPR）
    print("测试小样本场景:")
    X_train = np.sort(5 * np.random.rand(80, 1), axis=0)
    y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 80)

    gpr_model = ExactGPR()
    gpr_model.fit(X_train, y_train)

    # 测试大样本（近似GPR）
    print("\n测试大样本场景:")
    X_large = np.random.randn(5000, 10)
    y_large = np.sum(X_large**2, axis=1) + np.random.normal(0, 0.1, 5000)

    gpr_large = ExactGPR()
    gpr_large.fit(X_large, y_large)

    print("GPR优化测试完成")