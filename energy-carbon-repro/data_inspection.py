#!/usr/bin/env python3
"""
数据检查脚本
"""
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('data/sim_lng/raw/full_simulation_data.csv', low_memory=False)
print(f"原始数据形状: {data.shape}")
print(f"数据列: {list(data.columns)}")

# 检查目标变量
target = 'orv_Q_MW'
if target in data.columns:
    print(f"\n目标变量 {target} 统计:")
    print(f"  总计: {len(data[target])}")
    print(f"  非空值: {data[target].notna().sum()}")
    print(f"  缺失值: {data[target].isna().sum()}")
    if data[target].notna().any():
        valid_target = data[target].dropna()
        print(f"  有效值范围: [{valid_target.min():.6f}, {valid_target.max():.6f}]")
        print(f"  均值: {valid_target.mean():.6f}")
        print(f"  标准差: {valid_target.std():.6f}")
        print(f"  前10个值: {valid_target.head(10).tolist()}")
else:
    print(f"\n目标变量 {target} 不存在!")
    print("包含 'orv' 的列:")
    orv_cols = [col for col in data.columns if 'orv' in col.lower()]
    print(orv_cols)

# 检查所有列的缺失情况
print(f"\n各列缺失值统计 (前20列):")
missing_info = data.isnull().sum().sort_values(ascending=False)
print(missing_info.head(20))

# 检查数据类型
print(f"\n数据类型统计:")
print(data.dtypes.value_counts())

# 看看实际有多少完全无缺失的行
print(f"\n完全无缺失值的行数: {data.dropna().shape[0]}")

# 检查是否有时间戳或ID列可能导致完美预测
print(f"\n前5行数据预览:")
print(data.head())