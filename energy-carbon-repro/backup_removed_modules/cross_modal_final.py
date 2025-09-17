#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化跨模态验证
Memory-Optimized Cross-Modal Validation

专注于跨模态融合模型，使用分批处理适配RTX 4060 8GB显存
"""

import sys
import pandas as pd
import numpy as np
import torch
import logging
import time
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """内存优化的跨模态融合验证"""
    logger.info("=" * 60)
    logger.info("LNG跨模态融合模型验证 - RTX 4060内存优化")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # GPU内存管理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU内存清理完成")

    # 1. 加载完整现实化数据
    logger.info("\n--- 阶段 1: 加载现实化数据 ---")
    df = pd.read_csv('data/sim_lng/enhanced_simulation_data.csv', parse_dates=['ts'])

    power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
    df['energy'] = df[power_columns].sum(axis=1)

    logger.info(f"数据集: {len(df):,} 行, 能耗CV: {df['energy'].std()/df['energy'].mean():.4f}")

    # 2. 高性能特征提取
    logger.info("\n--- 阶段 2: 高性能特征提取 ---")
    from src.features.high_performance_engine import HighPerformanceFeatureEngine

    engine = HighPerformanceFeatureEngine(window_size=180, stride=30)
    feature_df = df.drop(columns=['ts', 'energy'])

    start_time = time.time()
    X_dynamic, X_static = engine.fast_extract_features(feature_df)
    feature_time = time.time() - start_time

    # 生成标签
    y_list = []
    for i in range(0, len(df) - 180 + 1, 30):
        y_list.append(df['energy'].iloc[i:i+180].mean())
    y = np.array(y_list[:len(X_dynamic)])

    logger.info(f"特征提取完成: {feature_time:.2f}秒, 速度: {len(X_dynamic)/feature_time:.1f} 窗口/秒")
    logger.info(f"数据形状: 动态{X_dynamic.shape}, 静态{X_static.shape}, 标签{y.shape}")

    # 3. 内存优化的跨模态训练
    logger.info("\n--- 阶段 3: 跨模态融合训练 (内存优化) ---")
    from src.models.cross_modal_fusion import CrossModalFusionWithResidual

    # 训练验证分割
    train_size = int(0.8 * len(X_dynamic))
    indices = np.random.permutation(len(X_dynamic))
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    X_dynamic_val = X_dynamic[val_idx]
    X_static_val = X_static[val_idx]
    y_val = y[val_idx]

    # 标准化
    y_scaler = StandardScaler().fit(y[train_idx].reshape(-1, 1))
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).flatten()

    # 创建模型 (内存优化)
    model = CrossModalFusionWithResidual(
        dynamic_dim=X_dynamic.shape[-1],
        static_dim=X_static.shape[-1],
        seq_len=X_dynamic.shape[1],
        d_model=64,  # 减小模型维度
        n_heads=4,
        n_layers=2   # 减少层数
    ).to(device)

    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 分批训练 (内存优化)
    batch_size = 512  # 适应RTX 4060 8GB
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    logger.info(f"开始分批训练 (batch_size={batch_size})...")

    model.train()
    train_loss_total = 0
    n_batches = 0

    for epoch in range(20):
        epoch_loss = 0
        epoch_batches = 0

        # 随机打乱训练数据
        train_indices = np.random.permutation(train_idx)

        for i in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[i:i+batch_size]

            # 获取批次数据
            batch_dynamic = torch.FloatTensor(X_dynamic[batch_indices]).to(device)
            batch_static = torch.FloatTensor(X_static[batch_indices]).to(device)
            batch_y = y_scaler.transform(y[batch_indices].reshape(-1, 1)).flatten()
            batch_labels = torch.FloatTensor(batch_y).unsqueeze(1).to(device)

            # 前向传播
            optimizer.zero_grad()
            pred, _ = model(batch_dynamic, batch_static)
            loss = criterion(pred, batch_labels)

            # 反向传播
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            # 清理GPU内存
            del batch_dynamic, batch_static, batch_labels
            torch.cuda.empty_cache()

        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1}/20, 平均Loss: {avg_loss:.6f}")

    # 4. 分批评估
    logger.info("\n--- 阶段 4: 跨模态融合评估 ---")
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(val_idx), batch_size):
            end_idx = min(i + batch_size, len(val_idx))
            batch_indices = val_idx[i:end_idx]

            val_dynamic = torch.FloatTensor(X_dynamic[batch_indices]).to(device)
            val_static = torch.FloatTensor(X_static[batch_indices]).to(device)

            pred, uncertainty = model(val_dynamic, val_static)
            predictions.append(pred.cpu().numpy())

            del val_dynamic, val_static
            torch.cuda.empty_cache()

    # 合并预测结果
    y_pred_s = np.concatenate(predictions).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_val_orig = y_scaler.inverse_transform(y_val_s.reshape(-1, 1)).flatten()

    # 计算学术指标
    r2 = r2_score(y_val_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    cv_rmse = rmse / np.mean(y_val_orig)
    nmbe = np.mean(y_pred_orig - y_val_orig) / np.mean(y_val_orig)

    # 5. 结果报告
    logger.info("\n" + "=" * 60)
    logger.info("跨模态融合模型验证结果")
    logger.info("=" * 60)

    logger.info(f"\n📊 数据特性:")
    logger.info(f"  数据规模: {len(df):,} 行 → {len(X_dynamic):,} 窗口")
    logger.info(f"  现实化能耗: 均值={np.mean(y_val_orig):.2f} kW, CV=0.5841")

    logger.info(f"\n🚀 性能指标:")
    logger.info(f"  特征提取: {feature_time:.2f}秒 ({len(X_dynamic)/feature_time:.1f} 窗口/秒)")
    logger.info(f"  GPU训练: 分批优化，适配8GB显存")

    logger.info(f"\n📊 跨模态融合学术指标:")
    logger.info(f"  R² Score: {r2:.4f} {'✅' if r2 >= 0.75 else '❌'} (目标 ≥ 0.75)")
    logger.info(f"  CV(RMSE): {cv_rmse:.4f} {'✅' if cv_rmse <= 0.06 else '❌'} (目标 ≤ 0.06)")
    logger.info(f"  NMBE: {nmbe:.4f} {'✅' if abs(nmbe) <= 0.006 else '❌'} (目标 ∈ [-0.006, 0.006])")

    # 论文要求检查
    requirements_met = (r2 >= 0.75 and cv_rmse <= 0.06 and abs(nmbe) <= 0.006)

    if requirements_met:
        logger.info("\n🎉 恭喜！跨模态融合模型达到论文要求！")
    else:
        logger.info("\n📈 跨模态融合模型在现实数据上显示挑战性，为进一步研究提供方向")

    # 保存结果
    import yaml
    final_results = {
        'model': 'CrossModalFusion',
        'dataset': 'enhanced_simulation_data.csv',
        'data_size': len(df),
        'windows': len(X_dynamic),
        'performance': {
            'feature_extraction_time': feature_time,
            'processing_speed': len(X_dynamic)/feature_time
        },
        'academic_metrics': {
            'r2': float(r2),
            'cv_rmse': float(cv_rmse),
            'nmbe': float(nmbe)
        },
        'requirements_met': requirements_met,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    results_path = Path("results") / f"cross_modal_final_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(final_results, f, allow_unicode=True)

    logger.info(f"\n结果已保存: {results_path}")
    logger.info("\n🎯 跨模态融合模型验证完成")

    # 清理GPU内存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()