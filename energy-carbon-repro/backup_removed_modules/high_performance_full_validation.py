#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能完整数据集验证
High-Performance Full Dataset Validation

使用高性能特征引擎处理完整1.55M行数据集
"""

import sys
import pandas as pd
import numpy as np
import torch
import logging
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """使用高性能引擎的完整数据集验证"""
    logger.info("=" * 60)
    logger.info("LNG论文复现 - 高性能完整数据集验证")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    # 1. 加载完整现实化数据
    logger.info("\n--- 阶段 1: 加载完整现实化数据 ---")
    df = pd.read_csv('data/sim_lng/enhanced_simulation_data.csv', parse_dates=['ts'])

    power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
    df['energy'] = df[power_columns].sum(axis=1)

    logger.info(f"完整数据集: {len(df):,} 行")
    logger.info(f"现实化能耗: 均值={df['energy'].mean():.2f} kW, CV={df['energy'].std()/df['energy'].mean():.4f}")

    # 2. 高性能特征提取
    logger.info("\n--- 阶段 2: 高性能特征提取 ---")
    from src.features.high_performance_engine import HighPerformanceFeatureEngine

    engine = HighPerformanceFeatureEngine(window_size=180, stride=30)
    feature_df = df.drop(columns=['ts', 'energy'])

    logger.info("开始高性能特征提取...")
    start_time = time.time()
    X_dynamic, X_static = engine.fast_extract_features(feature_df)
    feature_time = time.time() - start_time

    # 3. 生成对应标签
    logger.info("\n--- 阶段 3: 标签生成 ---")
    y_list = []
    for i in range(0, len(df) - 180 + 1, 30):
        y_list.append(df['energy'].iloc[i:i+180].mean())
    y = np.array(y_list[:len(X_dynamic)])

    logger.info(f"数据准备完成:")
    logger.info(f"  动态特征: {X_dynamic.shape}")
    logger.info(f"  静态特征: {X_static.shape}")
    logger.info(f"  标签: {y.shape}")

    # 4. 模型对比验证
    logger.info("\n--- 阶段 4: 多模型性能对比 ---")

    # 训练验证分割
    indices = np.arange(len(X_dynamic))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_dynamic_train, X_dynamic_val = X_dynamic[train_idx], X_dynamic[val_idx]
    X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 为传统模型准备特征
    X_flat_train = X_dynamic_train.reshape(len(X_dynamic_train), -1)
    X_flat_val = X_dynamic_val.reshape(len(X_dynamic_val), -1)

    # 标准化
    scaler_X = StandardScaler().fit(X_flat_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = scaler_X.transform(X_flat_train)
    X_val_s = scaler_X.transform(X_flat_val)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    results = {}

    # 4.1 RandomForest基准
    logger.info("\n4.1 RandomForest基准模型")
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_start = time.time()
    rf.fit(X_train_s, y_train_s)
    rf_pred = rf.predict(X_val_s)
    rf_time = time.time() - rf_start

    rf_pred_orig = scaler_y.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
    y_val_orig = scaler_y.inverse_transform(y_val_s.reshape(-1, 1)).flatten()

    results['RandomForest'] = {
        'r2': r2_score(y_val_orig, rf_pred_orig),
        'cv_rmse': np.sqrt(mean_squared_error(y_val_orig, rf_pred_orig)) / np.mean(y_val_orig),
        'nmbe': np.mean(rf_pred_orig - y_val_orig) / np.mean(y_val_orig),
        'time': rf_time
    }

    # 4.2 跨模态融合模型
    logger.info("\n4.2 跨模态融合模型 (论文核心)")
    from src.models.cross_modal_fusion import CrossModalFusionWithResidual

    model = CrossModalFusionWithResidual(
        dynamic_dim=X_dynamic.shape[-1],
        static_dim=X_static.shape[-1],
        seq_len=X_dynamic.shape[1],
        d_model=128,
        n_heads=4,
        n_layers=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # 数据转换
    train_dynamic = torch.FloatTensor(X_dynamic_train).to(device)
    train_static = torch.FloatTensor(X_static_train).to(device)
    train_labels = torch.FloatTensor(y_train_s).unsqueeze(1).to(device)

    # 训练
    cm_start = time.time()
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        pred, _ = model(train_dynamic, train_static)
        loss = criterion(pred, train_labels)

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"  CrossModal Epoch {epoch+1}/30, Loss: {loss.item():.6f}")

    # 评估
    model.eval()
    with torch.no_grad():
        val_dynamic = torch.FloatTensor(X_dynamic_val).to(device)
        val_static = torch.FloatTensor(X_static_val).to(device)
        cm_pred, _ = model(val_dynamic, val_static)
        cm_pred_cpu = cm_pred.cpu().numpy().flatten()

    cm_time = time.time() - cm_start

    # 处理NaN
    if np.any(np.isnan(cm_pred_cpu)):
        logger.warning("跨模态预测包含NaN，使用均值替代")
        cm_pred_cpu = np.nan_to_num(cm_pred_cpu, nan=np.mean(y_train_s))

    cm_pred_orig = scaler_y.inverse_transform(cm_pred_cpu.reshape(-1, 1)).flatten()

    results['CrossModalFusion'] = {
        'r2': r2_score(y_val_orig, cm_pred_orig),
        'cv_rmse': np.sqrt(mean_squared_error(y_val_orig, cm_pred_orig)) / np.mean(y_val_orig),
        'nmbe': np.mean(cm_pred_orig - y_val_orig) / np.mean(y_val_orig),
        'time': cm_time
    }

    # 5. 结果对比
    logger.info("\n" + "=" * 60)
    logger.info("--- 阶段 5: 完整数据集模型性能对比 ---")
    logger.info("=" * 60)

    logger.info(f"\n📊 特征提取性能:")
    logger.info(f"  处理时间: {feature_time:.2f}秒")
    logger.info(f"  处理速度: {len(X_dynamic)/feature_time:.1f} 窗口/秒")
    logger.info(f"  数据规模: {len(X_dynamic):,} 窗口")

    logger.info(f"\n📊 模型性能对比 (现实化数据):")
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  R² Score: {metrics['r2']:.4f} {'✅' if metrics['r2'] >= 0.75 else '❌'}")
        logger.info(f"  CV(RMSE): {metrics['cv_rmse']:.4f} {'✅' if metrics['cv_rmse'] <= 0.06 else '❌'}")
        logger.info(f"  NMBE: {metrics['nmbe']:.4f} {'✅' if abs(metrics['nmbe']) <= 0.006 else '❌'}")
        logger.info(f"  训练时间: {metrics['time']:.2f}秒")

    # 6. 保存结果
    import yaml
    final_results = {
        'dataset': 'enhanced_simulation_data.csv',
        'feature_extraction_time': feature_time,
        'processing_speed_windows_per_sec': len(X_dynamic)/feature_time,
        'models': results,
        'data_characteristics': {
            'rows': len(df),
            'windows': len(X_dynamic),
            'energy_cv': df['energy'].std()/df['energy'].mean()
        }
    }

    results_path = Path("results") / f"full_dataset_comparison_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(final_results, f, allow_unicode=True)

    logger.info(f"\n完整对比结果已保存: {results_path}")
    logger.info("\n🎯 高性能完整数据集验证完成")

if __name__ == "__main__":
    main()