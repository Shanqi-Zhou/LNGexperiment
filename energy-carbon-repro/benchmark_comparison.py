#!/usr/bin/env python3
"""
LNG项目综合基准测试脚本
对比多种模型在大样本数据上的性能
"""

import pandas as pd
import numpy as np
import logging
import time
import json
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import sys

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

from src.models.hgbr_baseline import AdaptiveHGBRBaseline

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data():
    """加载数据"""
    logger = setup_logging()
    logger.info("加载LNG数据...")
    
    data = pd.read_csv('data/sim_lng/full_simulation_data.csv', low_memory=False)
    
    # 处理数值特征
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data_clean = data[numeric_columns].copy()
    data_clean = data_clean.fillna(data_clean.median())
    
    X = data_clean.drop(columns=['orv_Q_MW'])
    y = data_clean['orv_Q_MW']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def benchmark_models():
    """基准测试多种模型"""
    logger = setup_logging()
    logger.info("开始综合基准测试...")
    
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    logger.info(f"数据规模: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    results = []
    
    # 1. 自适应HGBR基线 (已训练的最佳模型)
    logger.info("1. 评估自适应HGBR基线...")
    start_time = time.time()
    
    model_hgbr = AdaptiveHGBRBaseline(logger=logger)
    model_hgbr.fit(X_train.values, y_train.values)
    
    y_pred = model_hgbr.predict(X_test.values)
    training_time = time.time() - start_time
    
    results.append({
        'model': 'Adaptive HGBR',
        'r2_score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'training_time': training_time,
        'best_config': model_hgbr.best_model_name
    })
    
    # 2. 随机森林基线
    logger.info("2. 评估随机森林基线...")
    start_time = time.time()
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rf_time = time.time() - start_time
    
    results.append({
        'model': 'Random Forest',
        'r2_score': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'training_time': rf_time,
        'best_config': 'n_estimators=100'
    })
    
    # 3. Ridge回归基线
    logger.info("3. 评估Ridge回归基线...")
    start_time = time.time()
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    ridge_time = time.time() - start_time
    
    results.append({
        'model': 'Ridge Regression',
        'r2_score': r2_score(y_test, y_pred_ridge),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        'mae': mean_absolute_error(y_test, y_pred_ridge),
        'training_time': ridge_time,
        'best_config': 'alpha=1.0'
    })
    
    # 创建结果DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('r2_score', ascending=False)
    
    # 保存结果
    output_dir = Path('results')
    df_results.to_csv(output_dir / 'benchmark_comparison.csv', index=False)
    
    with open(output_dir / 'benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 输出结果
    logger.info("=" * 80)
    logger.info("综合基准测试结果")
    logger.info("=" * 80)
    logger.info(f"{'模型':<20} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'训练时间(s)':<12}")
    logger.info("-" * 80)
    
    for _, row in df_results.iterrows():
        logger.info(f"{row['model']:<20} {row['r2_score']:<8.4f} {row['rmse']:<10.6f} {row['mae']:<10.6f} {row['training_time']:<12.2f}")
    
    logger.info("=" * 80)
    logger.info(f"最佳模型: {df_results.iloc[0]['model']}")
    logger.info(f"最佳R² Score: {df_results.iloc[0]['r2_score']:.6f}")
    logger.info("=" * 80)
    
    return df_results

if __name__ == '__main__':
    results = benchmark_models()