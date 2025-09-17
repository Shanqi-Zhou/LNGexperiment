
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

class ComprehensiveEvaluator:
    """
    全面的模型性能评估器。
    负责计算所有关键指标，并支持分层评估。
    """
    def __init__(self, target_scaler):
        """
        初始化评估器。

        Args:
            target_scaler: 一个已经fit过的scikit-learn scaler对象，
                           用于将标准化的标签和预测值反转回原始尺度。
        """
        self.target_scaler = target_scaler

    def _inverse_transform(self, y_scaled):
        "将一维数组反转回原始尺度"
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def _compute_metrics(self, y_true, y_pred):
        "在原始尺度上计算核心指标"
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # 避免除以零
        mean_true = np.mean(y_true)
        if abs(mean_true) < 1e-9: mean_true = 1e-9

        return {
            'rmse': rmse,
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'nmbe': np.mean(y_pred - y_true) / mean_true,
            'cv_rmse': rmse / mean_true
        }

    def _compute_picp(self, y_true, y_pred_mean, y_pred_std, confidence_level=0.90):
        "计算预测区间覆盖概率 (PICP)"
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        lower_bound = y_pred_mean - z_score * y_pred_std
        upper_bound = y_pred_mean + z_score * y_pred_std
        covered = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
        return covered / len(y_true)

    def evaluate(self, y_true_scaled, y_pred_scaled, y_pred_std_scaled=None, regime_labels=None):
        """
        执行全面评估。

        Args:
            y_true_scaled (np.ndarray): 标准化后的真实标签。
            y_pred_scaled (np.ndarray): 标准化后的模型预测均值。
            y_pred_std_scaled (np.ndarray, optional): 标准化后的模型预测标准差。
            regime_labels (np.ndarray, optional): 用于分层评估的工况标签。

        Returns:
            dict: 包含所有评估结果的字典。
        """
        # 1. 反标准化到原始尺度
        y_true = self._inverse_transform(y_true_scaled)
        y_pred = self._inverse_transform(y_pred_scaled)
        
        results = {'overall': self._compute_metrics(y_true, y_pred)}

        # 2. 不确定性评估
        if y_pred_std_scaled is not None:
            # 标准差的缩放只与scale_有关，与mean_无关
            y_pred_std = y_pred_std_scaled * self.target_scaler.scale_[0]
            results['overall']['picp_90'] = self._compute_picp(y_true, y_pred, y_pred_std, 0.90)
            results['overall']['picp_95'] = self._compute_picp(y_true, y_pred, y_pred_std, 0.95)

        # 3. 分工况评估
        if regime_labels is not None:
            results['by_regime'] = {}
            for regime in np.unique(regime_labels):
                mask = (regime_labels == regime)
                if np.sum(mask) > 0:
                    regime_metrics = self._compute_metrics(y_true[mask], y_pred[mask])
                    results['by_regime'][f'regime_{regime}'] = regime_metrics
        
        return results

# 使用示例
if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    # 1. 创建模拟原始数据和scaler
    y_original = np.random.rand(1000) * 50 + 100 # 均值125
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_original.reshape(-1, 1)).flatten()

    # 2. 创建模拟的预测输出 (已标准化)
    y_pred_s = y_scaled + np.random.normal(0, 0.1, 1000)
    y_std_s = np.random.rand(1000) * 0.1
    regimes = np.random.randint(0, 3, 1000)

    # 3. 初始化评估器并执行评估
    evaluator = ComprehensiveEvaluator(target_scaler=scaler)
    final_results = evaluator.evaluate(y_scaled, y_pred_s, y_std_s, regimes)

    import json
    print("--- Comprehensive Evaluation Results ---")
    print(json.dumps(final_results, indent=2))

    # 4. 检查验收指标
    kpis = final_results['overall']
    print("\n--- KPI Check ---")
    print(f"R²: {kpis['r2']:.3f} (Target: >= 0.75)")
    print(f"CV(RMSE): {kpis['cv_rmse']:.3f} (Target: <= 0.06)")
    print(f"NMBE: {kpis['nmbe']:.3f} (Target: [-0.006, 0.006])")
    print(f"PICP@90%: {kpis['picp_90']:.3f} (Target: [0.85, 0.95])")
