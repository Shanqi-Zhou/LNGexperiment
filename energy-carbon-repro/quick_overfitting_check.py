#!/usr/bin/env python3
"""
快速过拟合检测
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 加载数据
data = pd.read_csv('data/sim_lng/raw/full_simulation_data.csv', low_memory=False)
print(f"数据形状: {data.shape}")

# 检查目标变量
target = 'orv_Q_MW'
if target not in data.columns:
    print(f"目标变量 {target} 不存在")
    print("可用列:", data.columns.tolist()[:10])
    exit(1)
    
# 移除缺失值和目标列
data_clean = data.dropna()
print(f"清理后数据形状: {data_clean.shape}")

# 准备特征和目标
X = data_clean.drop(columns=[target])
y = data_clean[target]

# 只保留数值列
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]
print(f"数值特征数量: {X.shape[1]}")
print(f"目标变量统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}, 范围=[{y.min():.4f}, {y.max():.4f}]")

# 数据泄露检测: 检查是否有完全相关的特征
corr_with_target = X.corrwith(y).abs()
high_corr_features = corr_with_target[corr_with_target > 0.99].sort_values(ascending=False)
print(f"\n与目标变量高度相关(>0.99)的特征 ({len(high_corr_features)} 个):")
for feature, corr in high_corr_features.items():
    print(f"  {feature}: {corr:.6f}")

# 检查是否有重复行
duplicate_rows = X.duplicated().sum()
print(f"\n重复行数量: {duplicate_rows}")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 简单模型测试
models = {
    'LinearRegression': LinearRegression(),
    'HGBR': HistGradientBoostingRegressor(random_state=42, max_iter=100)
}

print(f"\n=== 交叉验证结果 ===")
for name, model in models.items():
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"{name}:")
    print(f"  CV R² = {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    # 训练和测试
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"  训练 R² = {train_score:.6f}")
    print(f"  测试 R² = {test_score:.6f}")
    print(f"  过拟合程度 = {train_score - test_score:.6f}")
    print()

# 检查特征重要性（对于HGBR）
if hasattr(models['HGBR'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': models['HGBR'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("前10个最重要特征:")
    print(feature_importance.head(10))