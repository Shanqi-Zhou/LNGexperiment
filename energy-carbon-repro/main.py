#!/usr/bin/env python3
"""
LNG过拟合修正实验
移除数据泄露特征，进行公平的模型评估
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
import sys

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def identify_leaky_features(X, y, threshold=0.99):
    """识别可能存在数据泄露的特征"""
    logger = setup_logging()
    
    # 计算与目标变量的相关性
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    leaky_features = correlations[correlations > threshold].index.tolist()
    
    logger.info(f"发现 {len(leaky_features)} 个可能存在数据泄露的特征 (|r| > {threshold}):")
    for i, feature in enumerate(leaky_features, 1):
        logger.info(f"  {i}. {feature}: {correlations[feature]:.6f}")
    
    return leaky_features, correlations

def clean_experiment():
    """清洁版本的实验，移除数据泄露特征"""
    logger = setup_logging()
    
    logger.info("🧹 开始清洁版LNG实验 - 移除数据泄露特征")
    logger.info("=" * 80)
    
    # 加载数据
    data = pd.read_csv('data/sim_lng/full_simulation_data.csv', low_memory=False)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data_clean = data[numeric_columns].copy()
    data_clean = data_clean.fillna(data_clean.median())
    
    target_col = 'orv_Q_MW'
    X = data_clean.drop(columns=[target_col])
    y = data_clean[target_col]
    
    logger.info(f"原始数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    # 识别泄露特征
    leaky_features, correlations = identify_leaky_features(X, y, threshold=0.99)
    
    # 移除泄露特征
    X_clean = X.drop(columns=leaky_features)
    logger.info(f"移除 {len(leaky_features)} 个泄露特征后: {X_clean.shape[1]} 特征")
    
    # 进一步移除高相关特征 (0.95-0.99)
    high_corr_features = []
    remaining_correlations = X_clean.corrwith(y).abs().sort_values(ascending=False)
    
    for feature, corr in remaining_correlations.items():
        if corr > 0.95:
            high_corr_features.append(feature)
    
    if len(high_corr_features) > 0:
        logger.info(f"\n发现 {len(high_corr_features)} 个高相关特征 (0.95 < |r| < 0.99):")
        for feature in high_corr_features[:10]:  # 只显示前10个
            logger.info(f"  - {feature}: {remaining_correlations[feature]:.4f}")
        
        # 可选：移除一些高相关特征
        # 这里我们保留它们，但在日志中标记
        logger.info("保留高相关特征以进行对比分析")
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    logger.info(f"\n数据分割: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    # 测试多种模型
    results = []
    
    # 1. 自适应HGBR
    logger.info("\n1. 测试自适应HGBR (清洁特征)...")
    start_time = time.time()
    
    from src.models.hgbr_baseline import AdaptiveHGBRBaseline
    model_hgbr = AdaptiveHGBRBaseline(logger=logger)
    model_hgbr.fit(X_train.values, y_train.values)
    
    y_pred_train_hgbr = model_hgbr.predict(X_train.values)
    y_pred_test_hgbr = model_hgbr.predict(X_test.values)
    
    hgbr_time = time.time() - start_time
    
    results.append({
        'model': 'HGBR_Clean',
        'train_r2': r2_score(y_train, y_pred_train_hgbr),
        'test_r2': r2_score(y_test, y_pred_test_hgbr),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_hgbr)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_hgbr)),
        'training_time': hgbr_time,
        'features_used': X_clean.shape[1],
        'best_config': model_hgbr.best_model_name
    })
    
    # 2. 随机森林
    logger.info("2. 测试随机森林 (清洁特征)...")
    start_time = time.time()
    
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred_train_rf = rf.predict(X_train)
    y_pred_test_rf = rf.predict(X_test)
    
    rf_time = time.time() - start_time
    
    results.append({
        'model': 'RandomForest_Clean',
        'train_r2': r2_score(y_train, y_pred_train_rf),
        'test_r2': r2_score(y_test, y_pred_test_rf),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_rf)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_rf)),
        'training_time': rf_time,
        'features_used': X_clean.shape[1],
        'best_config': 'n_estimators=100'
    })
    
    # 3. Ridge回归 (基线)
    logger.info("3. 测试Ridge回归 (清洁特征)...")
    start_time = time.time()
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_train_ridge = ridge.predict(X_train_scaled)
    y_pred_test_ridge = ridge.predict(X_test_scaled)
    
    ridge_time = time.time() - start_time
    
    results.append({
        'model': 'Ridge_Clean',
        'train_r2': r2_score(y_train, y_pred_train_ridge),
        'test_r2': r2_score(y_test, y_pred_test_ridge),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train_ridge)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_ridge)),
        'training_time': ridge_time,
        'features_used': X_clean.shape[1],
        'best_config': 'alpha=1.0'
    })
    
    # 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("清洁实验结果 (移除数据泄露特征)")
    logger.info("=" * 80)
    
    df_results = pd.DataFrame(results)
    df_results['overfitting_ratio'] = df_results['test_rmse'] / df_results['train_rmse']
    df_results['performance_drop'] = df_results['train_r2'] - df_results['test_r2']
    
    logger.info("模型性能对比 (训练集 vs 测试集):")
    logger.info("=" * 100)
    logger.info(f"{'模型':<15} {'训练R²':<8} {'测试R²':<8} {'训练RMSE':<10} {'测试RMSE':<10} {'过拟合比':<8} {'性能下降':<8}")
    logger.info("-" * 100)
    
    for _, row in df_results.iterrows():
        logger.info(f"{row['model']:<15} {row['train_r2']:<8.4f} {row['test_r2']:<8.4f} "
                   f"{row['train_rmse']:<10.6f} {row['test_rmse']:<10.6f} "
                   f"{row['overfitting_ratio']:<8.3f} {row['performance_drop']:<8.4f}")
    
    # 过拟合分析
    logger.info("\n过拟合风险评估:")
    for _, row in df_results.iterrows():
        if row['overfitting_ratio'] > 2.0:
            risk = "严重过拟合 ❌"
        elif row['overfitting_ratio'] > 1.5:
            risk = "中度过拟合 ⚠️"
        elif row['overfitting_ratio'] > 1.2:
            risk = "轻微过拟合 ?"
        else:
            risk = "泛化良好 ✅"
        
        logger.info(f"  {row['model']}: {risk} (测试/训练RMSE = {row['overfitting_ratio']:.3f})")
    
    # 保存结果
    output_dir = Path('results')
    
    # 保存清洁实验结果
    clean_results = {
        'removed_leaky_features': leaky_features,
        'feature_correlations': correlations.to_dict(),
        'model_results': df_results.to_dict('records'),
        'summary': {
            'original_features': X.shape[1],
            'clean_features': X_clean.shape[1],
            'removed_features': len(leaky_features),
            'best_model': df_results.loc[df_results['test_r2'].idxmax(), 'model'],
            'best_test_r2': df_results['test_r2'].max()
        }
    }
    
    with open(output_dir / 'clean_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)
    
    df_results.to_csv(output_dir / 'clean_model_comparison.csv', index=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("清洁实验总结")
    logger.info("=" * 80)
    logger.info(f"移除特征: {len(leaky_features)} 个数据泄露特征")
    logger.info(f"使用特征: {X_clean.shape[1]} 个清洁特征")
    logger.info(f"最佳模型: {clean_results['summary']['best_model']}")
    logger.info(f"最佳测试R²: {clean_results['summary']['best_test_r2']:.4f}")
    logger.info(f"结果保存: {output_dir}/clean_experiment_results.json")
    
    return clean_results

if __name__ == '__main__':
    results = clean_experiment()