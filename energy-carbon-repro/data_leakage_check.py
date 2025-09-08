#!/usr/bin/env python3
"""
数据泄露检测
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 加载数据
data = pd.read_csv('data/sim_lng/raw/full_simulation_data.csv', low_memory=False)
print(f"原始数据形状: {data.shape}")

# 获取数值列
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
data_clean = data[numeric_columns].copy()

# 处理缺失值
missing_count = data_clean.isnull().sum().sum()
if missing_count > 0:
    print(f"处理 {missing_count} 个缺失值")
    data_clean = data_clean.fillna(data_clean.median())

print(f"清理后数据规模: {data_clean.shape}")
print(f"数值特征: {numeric_columns}")

# 目标变量
target = 'orv_Q_MW'
y = data_clean[target]
X = data_clean.drop(columns=[target])

print(f"\n=== 数据泄露检测 ===")
print(f"特征数量: {X.shape[1]}")
print(f"目标变量: {target}")

# 检查与目标变量的相关性
correlations = X.corrwith(y).abs().sort_values(ascending=False)
print(f"\n前20个与目标变量最相关的特征:")
for i, (feature, corr) in enumerate(correlations.head(20).items()):
    status = "LEAK!" if corr > 0.99 else "HIGH" if corr > 0.95 else "MID" if corr > 0.8 else ""
    print(f"{i+1:2d}. {feature:<40} {corr:.6f} {status}")

# 检查ORV相关特征
orv_features = [col for col in X.columns if 'orv' in col.lower()]
print(f"\nORV相关特征 ({len(orv_features)} 个):")
for feature in orv_features:
    corr = X[feature].corr(y)
    status = "LEAK!" if abs(corr) > 0.99 else "HIGH" if abs(corr) > 0.95 else ""
    print(f"  {feature:<40} {corr:7.4f} {status}")

# 尝试线性回归看看是否有完美预测
lr = LinearRegression()
lr.fit(X, y)
r2 = lr.score(X, y)
print(f"\n线性回归 R² = {r2:.6f}")

# 查看最重要的特征系数
feature_coef = pd.DataFrame({
    'feature': X.columns,
    'coef': np.abs(lr.coef_)
}).sort_values('coef', ascending=False)

print(f"\n线性回归前10个最重要特征:")
for _, row in feature_coef.head(10).iterrows():
    print(f"  {row['feature']:<40} {row['coef']:10.4f}")