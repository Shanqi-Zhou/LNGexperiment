
import numpy as np

class ResidualModelingPipeline:
    """
    残差建模框架。
    该框架首先使用一个简单模型拟合数据的线性或主体部分，
    然后用一个更复杂的模型来拟合前者的预测残差。
    """
    def __init__(self, base_model, residual_model):
        """
        初始化残差建模管线。

        Args:
            base_model: 用于第一阶段拟合的简单模型实例 (例如 MLR)。
            residual_model: 用于拟合残差的复杂模型实例 (例如 GPR)。
        """
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        """
        训练整个残差管线。

        Args:
            X (np.ndarray): 特征数据。
            y (np.ndarray): 真实标签数据。
        """
        # 1. 训练基础模型
        print("Fitting base model...")
        self.base_model.fit(X, y)

        # 2. 计算残差
        y_base_pred = self.base_model.predict(X)
        residuals = y - y_base_pred
        print(f"Base model fitted. Residuals mean: {np.mean(residuals):.4f}, std: {np.std(residuals):.4f}")

        # 3. 训练残差模型
        print("Fitting residual model on residuals...")
        self.residual_model.fit(X, residuals)
        print("Residual model fitted.")

    def predict(self, X, return_std=False):
        """
        进行最终预测。

        Args:
            X (np.ndarray): 需要预测的特征数据。
            return_std (bool): 是否返回不确定性 (如果残差模型支持)。

        Returns:
            预测结果。如果return_std=True且残差模型支持，则返回 (均值, 标准差)。
        """
        # 1. 基础模型预测
        y_base_pred = self.base_model.predict(X)

        # 2. 残差模型预测
        if return_std and hasattr(self.residual_model, 'predict') and 'return_std' in self.residual_model.predict.__code__.co_varnames:
            y_residual_pred, y_residual_std = self.residual_model.predict(X, return_std=True)
            y_final_pred = y_base_pred + y_residual_pred
            return y_final_pred, y_residual_std
        else:
            y_residual_pred = self.residual_model.predict(X, return_std=False)
            y_final_pred = y_base_pred + y_residual_pred
            return y_final_pred

# 使用示例
if __name__ == '__main__':
    from .mlr import MLR
    from .gpr import ExactGPR

    # 1. 创建一个包含线性和非线性部分的数据集
    X_train = np.linspace(0, 10, 100).reshape(-1, 1)
    # 线性部分: 2*x + 5
    # 非线性部分: sin(x)
    # 噪声
    y_train = 2 * X_train.ravel() + 5 + np.sin(X_train).ravel() + np.random.normal(0, 0.1, 100)

    # 2. 初始化模型
    base_model = MLR()
    residual_model = ExactGPR()

    # 3. 初始化并训练残差管线
    pipeline = ResidualModelingPipeline(base_model=base_model, residual_model=residual_model)
    pipeline.fit(X_train, y_train)

    # 4. 预测
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    y_pred, y_std = pipeline.predict(X_test, return_std=True)

    print("\nResidual pipeline prediction complete.")
    
    # 5. 可视化 (需要安装matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.scatter(X_train, y_train, c='k', label='Training Data', zorder=10)
        plt.plot(X_test, y_pred, c='b', label='Pipeline Prediction')
        plt.fill_between(X_test.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, 
                         alpha=0.2, color='blue', label='95% Confidence Interval')
        plt.plot(X_test, base_model.predict(X_test), 'g--', label='Base Model (MLR) Fit')
        plt.title('Residual Modeling Pipeline Demonstration')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        # plt.show() # 取消注释以显示图像
        print("\nPlot generated. (Matplotlib required to display)")
    except ImportError:
        print("\nMatplotlib not found. Skipping plot generation.")
