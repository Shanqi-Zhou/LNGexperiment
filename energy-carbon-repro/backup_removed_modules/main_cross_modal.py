#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的主程序 - 使用论文核心架构
Enhanced Main Program with Paper's Core Architecture

集成跨模态融合 + 高级特征工程 + MLR/GPR残差建模
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import warnings
import os
import sys
import io
import time
import gc
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import logging

# 设置UTF-8编码，解决Windows gbk编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.features.advanced_feature_engineering import AdvancedFeatureEngineering, FeatureConfig
from src.models.adaptive_strategy import create_adaptive_model
from src.models.cross_modal_fusion import CrossModalFusionWithResidual
from src.training.unified_framework import UnifiedTrainer
from src.training.purged_validation import PurgedWalkForwardCV
from src.eval.evaluator import ComprehensiveEvaluator

def setup_device():
    """配置GPU设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        logger.info("使用CPU训练")
    return device

def load_config(path):
    """加载YAML配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data_with_progress(data_path):
    """加载数据并显示进度"""
    logger.info(f"加载数据: {data_path}")
    start_time = time.time()

    chunk_size = 100000
    chunks = []
    total_rows = 0

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f) - 1

        logger.info(f"数据集总行数: {total_lines:,}")

        for chunk in pd.read_csv(data_path, parse_dates=['ts'], chunksize=chunk_size):
            chunks.append(chunk)
            total_rows += len(chunk)
            progress = (total_rows / total_lines) * 100
            logger.info(f"  加载进度: {progress:.1f}% ({total_rows:,}/{total_lines:,} 行)")

        df = pd.concat(chunks, ignore_index=True)
        elapsed = time.time() - start_time
        logger.info(f"数据加载完成: {len(df):,} 行, 耗时 {elapsed:.2f} 秒")

        # 内存优化
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        return df

    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise

def extract_features_advanced(df, config):
    """使用高级特征工程系统"""
    logger.info("开始高级特征提取...")

    # 创建特征工程配置
    feature_config = FeatureConfig(
        window_size=config['data_processing']['window_size'],
        stride=config['data_processing']['stride']
    )

    # 初始化高级特征工程
    feature_engine = AdvancedFeatureEngineering(feature_config)
    logger.info("启用论文完整特征体系: 9类动态特征 + 32维静态特征")

    # 处理数据集，生成跨模态特征
    feature_df = df.drop(columns=['ts', 'energy'])
    X_dynamic, X_static = feature_engine.process_dataset(feature_df)

    # 生成对应的标签
    y_list = []
    window_size = feature_config.window_size
    stride = feature_config.stride

    for i in range(0, len(df) - window_size + 1, stride):
        window_energy = df['energy'].iloc[i:i+window_size].mean()
        y_list.append(window_energy)

    y = np.array(y_list[:len(X_dynamic)])

    logger.info(f"高级特征提取完成:")
    logger.info(f"  动态特征形状: {X_dynamic.shape}")
    logger.info(f"  静态特征形状: {X_static.shape}")
    logger.info(f"  标签形状: {y.shape}")

    return X_dynamic, X_static, y

def train_fold_cross_modal(fold_idx, X_dynamic_train, X_static_train, y_train,
                          X_dynamic_val, X_static_val, y_val, config, device):
    """使用跨模态融合训练"""
    logger.info(f"\n===== 跨模态融合训练 FOLD {fold_idx + 1} =====")

    # 数据标准化
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)

    # 创建跨模态融合模型
    n_dynamic_features = X_dynamic_train.shape[-1]
    n_static_features = X_static_train.shape[-1]
    seq_len = X_dynamic_train.shape[1]

    model = CrossModalFusionWithResidual(
        dynamic_dim=n_dynamic_features,
        static_dim=n_static_features,
        seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_layers=3
    )

    logger.info(f"跨模态融合模型初始化:")
    logger.info(f"  动态特征维度: {n_dynamic_features}")
    logger.info(f"  静态特征维度: {n_static_features}")
    logger.info(f"  序列长度: {seq_len}")

    # GPU训练
    model = model.to(device)

    # 准备数据
    train_dynamic = torch.FloatTensor(X_dynamic_train).to(device)
    train_static = torch.FloatTensor(X_static_train).to(device)
    train_labels = torch.FloatTensor(y_train_s).unsqueeze(1).to(device)

    val_dynamic = torch.FloatTensor(X_dynamic_val).to(device)
    val_static = torch.FloatTensor(X_static_val).to(device)
    val_labels = torch.FloatTensor(y_val_s).unsqueeze(1).to(device)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(50):  # 快速训练
        optimizer.zero_grad()

        pred, uncertainty = model(train_dynamic, train_static)
        loss = criterion(pred, train_labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/50, Loss: {loss.item():.6f}")

    # 评估
    model.eval()
    with torch.no_grad():
        val_pred, _ = model(val_dynamic, val_static)
        val_pred_cpu = val_pred.cpu().numpy().flatten()

    # 反标准化
    y_pred_original = y_scaler.inverse_transform(val_pred_cpu.reshape(-1, 1)).flatten()
    y_val_original = y_scaler.inverse_transform(y_val_s.reshape(-1, 1)).flatten()

    # 计算指标
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_val_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
    cv_rmse = rmse / np.mean(y_val_original)
    nmbe = np.mean(y_pred_original - y_val_original) / np.mean(y_val_original)

    results = {
        'r2': r2,
        'rmse': rmse,
        'cv_rmse': cv_rmse,
        'nmbe': nmbe
    }

    logger.info(f"Fold {fold_idx + 1} 跨模态融合指标:")
    logger.info(f"  R²: {r2:.4f} (目标 ≥ 0.75)")
    logger.info(f"  CV(RMSE): {cv_rmse:.4f} (目标 ≤ 0.06)")
    logger.info(f"  NMBE: {nmbe:.4f} (目标 ∈ [-0.006, 0.006])")

    # 清理GPU内存
    torch.cuda.empty_cache()

    return {'overall': results}

def main(config):
    """优化的主执行函数 - 使用论文完整架构"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("LNG论文复现项目 - 优化架构验证")
    logger.info("=" * 60)

    device = setup_device()

    # 1. 加载现实化数据
    logger.info("\n--- 阶段 1: 加载现实化数据 ---")
    data_path = Path("data/sim_lng/enhanced_simulation_data.csv")
    df = load_data_with_progress(data_path)

    # 2. 计算能耗
    logger.info("\n--- 阶段 2: 能耗计算 ---")
    power_columns = [
        'booster_pump_power_kw',
        'hp_pump_power_kw',
        'bog_compressor_total_power_kw'
    ]

    df['energy'] = df[power_columns].sum(axis=1)
    logger.info(f"现实化能耗统计: 均值={df['energy'].mean():.2f} kW, 变异系数={df['energy'].std()/df['energy'].mean():.4f}")

    # 3. 高级特征工程
    logger.info("\n--- 阶段 3: 高级特征工程 (论文完整体系) ---")
    X_dynamic, X_static, y = extract_features_advanced(df, config)

    del df
    gc.collect()

    # 4. 跨模态融合训练
    logger.info("\n--- 阶段 4: 跨模态融合交叉验证 ---")
    cv_config = config['validation']
    cv = PurgedWalkForwardCV(
        n_splits=cv_config['n_splits'],
        embargo_size=cv_config['embargo_size']
    )

    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_dynamic)):
        X_dynamic_train = X_dynamic[train_idx]
        X_static_train = X_static[train_idx]
        y_train = y[train_idx]

        X_dynamic_val = X_dynamic[val_idx]
        X_static_val = X_static[val_idx]
        y_val = y[val_idx]

        logger.info(f"\n跨模态数据: 训练{len(train_idx):,}, 验证{len(val_idx):,}")

        results = train_fold_cross_modal(
            fold_idx, X_dynamic_train, X_static_train, y_train,
            X_dynamic_val, X_static_val, y_val, config, device
        )

        all_fold_results.append(results['overall'])
        gc.collect()

    # 5. 最终评估
    logger.info("\n" + "=" * 60)
    logger.info("--- 阶段 5: 跨模态融合最终指标 ---")
    logger.info("=" * 60)

    avg_metrics = {
        'r2': np.mean([r['r2'] for r in all_fold_results]),
        'cv_rmse': np.mean([r['cv_rmse'] for r in all_fold_results]),
        'nmbe': np.mean([r['nmbe'] for r in all_fold_results]),
    }

    logger.info("\n📊 跨模态融合学术指标:")
    logger.info("-" * 40)
    logger.info(f"  R² Score:        {avg_metrics['r2']:.4f}  {'✅' if avg_metrics['r2'] >= 0.75 else '❌'} (目标 ≥ 0.75)")
    logger.info(f"  CV(RMSE):        {avg_metrics['cv_rmse']:.4f}  {'✅' if avg_metrics['cv_rmse'] <= 0.06 else '❌'} (目标 ≤ 0.06)")
    logger.info(f"  NMBE:            {avg_metrics['nmbe']:.4f}  {'✅' if -0.006 <= avg_metrics['nmbe'] <= 0.006 else '❌'} (目标 ∈ [-0.006, 0.006])")

    requirements_met = (
        avg_metrics['r2'] >= 0.75 and
        avg_metrics['cv_rmse'] <= 0.06 and
        -0.006 <= avg_metrics['nmbe'] <= 0.006
    )

    if requirements_met:
        logger.info("\n🎉 恭喜！跨模态融合模型达到论文要求！")
    else:
        logger.info("\n⚠️ 需要进一步优化跨模态融合架构")

    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    final_results = {
        'architecture': 'CrossModalFusion + MLR/GPR',
        'feature_system': '9_categories_dynamic + 32_static',
        'metrics': avg_metrics,
        'all_folds': all_fold_results,
        'requirements_met': requirements_met,
        'timestamp': datetime.now().isoformat()
    }

    results_path = results_dir / f"cross_modal_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(final_results, f, allow_unicode=True)

    logger.info(f"\n结果已保存至: {results_path}")
    total_time = time.time() - start_time
    logger.info(f"\n总运行时间: {total_time/60:.2f} 分钟")

    return avg_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LNG论文复现 - 跨模态融合验证")
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        logger.info("调试模式：使用缩减参数")
        config['data_processing']['window_size'] = 60
        config['validation']['n_splits'] = 2

    try:
        metrics = main(config)
        sys.exit(0 if metrics['r2'] >= 0.75 else 1)
    except KeyboardInterrupt:
        logger.info("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)