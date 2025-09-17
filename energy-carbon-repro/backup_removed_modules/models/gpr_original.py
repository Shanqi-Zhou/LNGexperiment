
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

class ExactGPR:
    """
    精确高斯过程回归模型，用于小样本场景。
    采用带有噪声保护的复合核函数，确保模型稳定性。
    """
    def __init__(self, n_restarts_optimizer=8, random_state=42):
        """
        初始化模型，核函数配置来自 `LNG项目综合优化方案（收紧版）.md`。

        Args:
            n_restarts_optimizer (int): 优化器重启次数，用于寻找最优超参数。
            random_state (int): 随机种子。
        """
        # 复合核函数：Matern核捕捉平滑度，WhiteKernel对噪声建模并防止方差塌陷
        self.kernel = Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
                      WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e-1))
        
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,  # alpha将在fit方法中根据残差方差动态设置
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state
        )

    def fit(self, X, y):
        """
        训练GPR模型。
        动态设置alpha为残差方差，以增强模型的鲁棒性。

        Args:
            X (np.ndarray): 特征数据。
            y (np.ndarray): 标签数据 (在此场景中是残差)。
        """
        # 动态设置alpha为残差方差，确保噪声水平与数据匹配
        residual_var = np.var(y)
        self.model.alpha = max(residual_var, 1e-6) # 保证alpha有下界

        self.model.fit(X, y)

    def predict(self, X, return_std=True):
        """
        使用训练好的模型进行预测。

        Args:
            X (np.ndarray): 需要预测的特征数据。
            return_std (bool): 是否返回预测的标准差（不确定性）。

        Returns:
            tuple or np.ndarray: (预测均值, 预测标准差) 或 仅预测均值。
        """
        return self.model.predict(X, return_std=return_std)

    def get_kernel_params(self):
        """
        获取拟合后的核函数超参数。
        """
        return self.model.kernel_ 

# 使用示例
if __name__ == '__main__':
    # 创建模拟数据 (模拟残差)
    X_train = np.sort(5 * np.random.rand(80, 1), axis=0)
    y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 80)

    # 初始化、训练和预测
    gpr_model = ExactGPR()
    gpr_model.fit(X_train, y_train)

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_pred, y_std = gpr_model.predict(X_test)

    print("ExactGPR模型预测完成。")
    print(f"在X=2.5处的预测值: {gpr_model.predict([[2.5]], return_std=False)[0]:.4f}")
    print(f"在X=2.5处的不确定性 (std): {gpr_post.predict([[2.5]])[1][0]:.4f}")
    print("学习到的核函数参数:", gpr_model.get_kernel_params())
