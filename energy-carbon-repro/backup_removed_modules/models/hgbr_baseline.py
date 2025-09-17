
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np

class HGBR:
    """
    对 scikit-learn 的 HistGradientBoostingRegressor 的封装。
    使用经过验证的工业化配置，作为强大的基线模型。
    """
    def __init__(self, random_state=42, **kwargs):
        """
        初始化模型，参数来自 `LNG项目综合优化方案（收紧版）.md`。

        Args:
            random_state (int): 随机种子，确保可复现性。
            **kwargs: 其他可以传递给HistGradientBoostingRegressor的参数。
        """
        default_params = {
            'max_iter': 500,
            'learning_rate': 0.05,
            'max_bins': 255,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'verbose': 1,
            'random_state': random_state
        }
        # 允许用户覆盖默认参数
        final_params = {**default_params, **kwargs}
        self.model = HistGradientBoostingRegressor(**final_params)

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

# 使用示例
if __name__ == '__main__':
    # 创建模拟数据
    X_train = np.random.rand(5000, 20)
    y_train = np.sin(X_train[:, 0]) * 10 + np.random.randn(5000)

    # 初始化、训练和预测
    hgbr_model = HGBR()
    hgbr_model.fit(X_train, y_train)
    
    X_test = np.random.rand(10, 20)
    predictions = hgbr_model.predict(X_test)

    print("HGBR模型预测完成。")
    print("预测结果:", predictions)
    print("模型使用的参数:", hgbr_model.model.get_params())
