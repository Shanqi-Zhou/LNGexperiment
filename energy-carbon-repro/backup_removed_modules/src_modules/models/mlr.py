
from sklearn.linear_model import Ridge
import numpy as np

class MLR:
    """
    多元线性回归（Ridge回归）模型的一个简单封装。
    用作残差建模的第一步，拟合数据中的线性部分。
    """
    def __init__(self, alpha=1.0):
        """
        初始化模型。

        Args:
            alpha (float): L2正则化强度。alpha=0时为普通线性回归。
        """
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        """
        训练模型。

        Args:
            X (np.ndarray or pd.DataFrame): 特征数据。
            y (np.ndarray or pd.Series): 标签数据。
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        Args:
            X (np.ndarray or pd.DataFrame): 需要预测的特征数据。

        Returns:
            np.ndarray: 预测结果。
        """
        return self.model.predict(X)

    def get_coefficients(self):
        """
        获取模型的回归系数和截距，用于合规性报告。

        Returns:
            tuple: (回归系数, 截距)
        """
        return self.model.coef_, self.model.intercept_

# 使用示例
if __name__ == '__main__':
    # 创建模拟数据
    X_train = np.random.rand(100, 10)
    y_train = np.dot(X_train, np.arange(10)) + np.random.randn(100) * 0.1 + 5

    # 初始化、训练和预测
    mlr_model = MLR(alpha=1.0)
    mlr_model.fit(X_train, y_train)
    
    X_test = np.random.rand(10, 10)
    predictions = mlr_model.predict(X_test)

    print("MLR模型预测完成。")
    print("预测结果:", predictions)

    coef, intercept = mlr_model.get_coefficients()
    print("回归系数 (前5个):", coef[:5])
    print("截距:", intercept)
