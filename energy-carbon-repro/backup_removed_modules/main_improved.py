#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNG论文复现项目 - 改进的主程序
解决编码问题，添加进度显示，优化内存使用
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
from src.features.advanced_feature_engineering import AdvancedFeatureEngineering
from src.models.adaptive_strategy import create_adaptive_model
from src.training.purged_validation import PurgedWalkForwardCV
from src.eval.evaluator import ComprehensiveEvaluator

def setup_device():
    """配置GPU设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 设置混合精度以适配RTX 4060 8GB
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

    # 分块读取大文件
    chunk_size = 100000
    chunks = []
    total_rows = 0

    try:
        # 先获取总行数
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f) - 1  # 减去header

        logger.info(f"数据集总行数: {total_lines:,}")

        # 分块读取
        for chunk in pd.read_csv(data_path, parse_dates=['ts'], chunksize=chunk_size):
            chunks.append(chunk)
            total_rows += len(chunk)
            progress = (total_rows / total_lines) * 100
            logger.info(f"  加载进度: {progress:.1f}% ({total_rows:,}/{total_lines:,} 行)")

        df = pd.concat(chunks, ignore_index=True)
        elapsed = time.time() - start_time
        logger.info(f"数据加载完成: {len(df):,} 行, 耗时 {elapsed:.2f} 秒")

        # 内存优化：转换数据类型
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')

        return df

    except FileNotFoundError:
        logger.error(f"找不到数据文件: {data_path}")
        raise
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        raise

def extract_features_optimized(df, config):
    """优化的特征提取"""
    logger.info("开始特征提取...")
    feature_config = config['data_processing']

    # 显示配置信息
    logger.info(f"  窗口大小: {feature_config['window_size']}")
    logger.info(f"  步长: {feature_config['stride']}")

    # 计算预期窗口数
    expected_windows = (len(df) - feature_config['window_size']) // feature_config['stride'] + 1
    logger.info(f"  预期窗口数: {expected_windows:,}")

    # 特征提取
    feature_engine = AdvancedFeatureEngineering(
        window_size=feature_config['window_size'],
        stride=feature_config['stride']
    )

    # 分批处理以减少内存压力
    batch_size = 50000
    n_batches = (len(df) - 1) // batch_size + 1

    feature_batches = []
    label_batches = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size + feature_config['window_size'], len(df))

        batch_df = df.iloc[start_idx:end_idx]

        if len(batch_df) >= feature_config['window_size']:
            # 提取特征
            X_dynamic, X_static = feature_engine.extract_features_from_window(batch_df.drop(columns=['ts', 'energy']))

            # 提取标签
            batch_labels = batch_df['energy'].rolling(
                feature_config['window_size']
            ).mean().iloc[::feature_config['stride']]

            feature_batches.append(batch_features)
            label_batches.append(batch_labels)

            logger.info(f"  批次 {i+1}/{n_batches} 完成 ({(i+1)/n_batches*100:.1f}%)")

            # 清理内存
            gc.collect()

    # 合并所有批次
    feature_df = pd.concat(feature_batches, ignore_index=True)
    labels = pd.concat(label_batches, ignore_index=True)

    # 清理NaN
    labels = labels.dropna()

    # 对齐长度
    min_len = min(len(feature_df), len(labels))
    X = feature_df.iloc[:min_len].values
    y = labels.iloc[:min_len].values

    logger.info(f"特征提取完成: X shape={X.shape}, y shape={y.shape}")

    return X, y

def train_fold_with_checkpoint(fold_idx, X_train, y_train, X_val, y_val, config, device):
    """训练单个fold并保存checkpoint"""
    logger.info(f"\n===== 开始训练 FOLD {fold_idx + 1} =====")

    # 数据标准化
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train_s = x_scaler.transform(X_train).astype(np.float32)
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    X_val_s = x_scaler.transform(X_val).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)

    # 创建自适应模型
    n_samples = len(X_train)
    model = create_adaptive_model(n_samples)

    # 训练模型
    results = {}
    if isinstance(model, torch.nn.Module):
        # PyTorch模型
        model = model.to(device)

        # 准备数据集
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_s),
            torch.FloatTensor(y_train_s).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_s),
            torch.FloatTensor(y_val_s).unsqueeze(1)
        )

        # 调整batch_size以适应RTX 4060
        batch_size = min(config['training_loop']['batch_size'], 32)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=2
        )

        # 训练
        trainer = UnifiedTrainer(model, train_loader, val_loader, config['training_loop'])
        trainer.train()

        # 评估
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val_s).to(device)
            y_pred_s = model(X_val_tensor).cpu().numpy()

        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    else:
        # Scikit-learn模型
        logger.info(f"训练 {type(model).__name__} 模型...")
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_val_s)

    # 评估结果
    evaluator = ComprehensiveEvaluator(target_scaler=y_scaler)
    results = evaluator.evaluate(y_val_s, y_pred_s.flatten())

    # 显示核心学术指标
    logger.info(f"Fold {fold_idx + 1} 学术指标:")
    logger.info(f"  R²: {results['overall']['r2']:.4f} (目标 ≥ 0.75)")
    logger.info(f"  CV(RMSE): {results['overall']['cv_rmse']:.4f} (目标 ≤ 0.06)")
    logger.info(f"  NMBE: {results['overall']['nmbe']:.4f} (目标 ∈ [-0.006, 0.006])")

    # 保存checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'fold': fold_idx,
        'model_state': model.state_dict() if isinstance(model, torch.nn.Module) else None,
        'results': results,
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"  Checkpoint 保存至: {checkpoint_path}")

    return results

def main(config):
    """改进的主执行函数"""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("LNG论文复现项目 - 学术指标验证")
    logger.info("=" * 60)

    # 配置设备
    device = setup_device()

    # 1. 加载数据
    logger.info("\n--- 阶段 1: 数据加载 ---")
    data_path = Path("data/sim_lng/enhanced_simulation_data.csv")
    df = load_data_with_progress(data_path)

    # 2. 计算能耗
    logger.info("\n--- 阶段 2: 能耗计算 ---")
    power_columns = [
        'booster_pump_power_kw',
        'hp_pump_power_kw',
        'bog_compressor_total_power_kw'
    ]

    # 检查列是否存在
    missing_cols = [col for col in power_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"缺失功耗列: {missing_cols}")
        raise ValueError(f"数据集缺失必需的功耗列")

    df['energy'] = df[power_columns].sum(axis=1)
    logger.info(f"能耗计算完成: 平均值 = {df['energy'].mean():.2f} kW")

    # 3. 特征提取
    logger.info("\n--- 阶段 3: 特征工程 ---")
    X, y = extract_features_optimized(df, config)

    # 释放原始数据内存
    del df
    gc.collect()

    # 4. 交叉验证训练
    logger.info("\n--- 阶段 4: 交叉验证与模型训练 ---")
    cv_config = config['validation']
    cv = PurgedWalkForwardCV(
        n_splits=cv_config['n_splits'],
        embargo_size=cv_config['embargo_size']
    )

    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"\n训练集大小: {len(train_idx):,}, 验证集大小: {len(val_idx):,}")

        # 训练并评估
        results = train_fold_with_checkpoint(
            fold_idx, X_train, y_train, X_val, y_val, config, device
        )

        all_fold_results.append(results['overall'])

        # 清理内存
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 5. 最终评估
    logger.info("\n" + "=" * 60)
    logger.info("--- 阶段 5: 最终学术指标评估 ---")
    logger.info("=" * 60)

    # 计算平均指标
    avg_metrics = {
        'r2': np.mean([r['r2'] for r in all_fold_results]),
        'cv_rmse': np.mean([r['cv_rmse'] for r in all_fold_results]),
        'nmbe': np.mean([r.get('nmbe', 0) for r in all_fold_results]),
        'mape': np.mean([r.get('mape', 0) for r in all_fold_results])
    }

    # 显示最终结果
    logger.info("\n📊 学术指标验证结果:")
    logger.info("-" * 40)
    logger.info(f"  R² Score:        {avg_metrics['r2']:.4f}  {'✅' if avg_metrics['r2'] >= 0.75 else '❌'} (目标 ≥ 0.75)")
    logger.info(f"  CV(RMSE):        {avg_metrics['cv_rmse']:.4f}  {'✅' if avg_metrics['cv_rmse'] <= 0.06 else '❌'} (目标 ≤ 0.06)")
    logger.info(f"  NMBE:            {avg_metrics['nmbe']:.4f}  {'✅' if -0.006 <= avg_metrics['nmbe'] <= 0.006 else '❌'} (目标 ∈ [-0.006, 0.006])")
    logger.info(f"  MAPE:            {avg_metrics['mape']:.4f}")
    logger.info("-" * 40)

    # 检查是否达到论文要求
    requirements_met = (
        avg_metrics['r2'] >= 0.75 and
        avg_metrics['cv_rmse'] <= 0.06 and
        -0.006 <= avg_metrics['nmbe'] <= 0.006
    )

    if requirements_met:
        logger.info("\n🎉 恭喜！所有学术指标均达到论文要求！")
    else:
        logger.info("\n⚠️ 部分指标未达到论文要求，需要进一步优化")

    # 保存最终结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    final_results = {
        'metrics': avg_metrics,
        'all_folds': all_fold_results,
        'requirements_met': requirements_met,
        'timestamp': datetime.now().isoformat()
    }

    results_path = results_dir / f"academic_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(final_results, f, allow_unicode=True)

    logger.info(f"\n结果已保存至: {results_path}")

    # 显示总耗时
    total_time = time.time() - start_time
    logger.info(f"\n总运行时间: {total_time/60:.2f} 分钟")

    return avg_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LNG论文复现 - 学术指标验证")
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                      help='训练配置文件路径')
    parser.add_argument('--debug', action='store_true',
                      help='调试模式，使用小数据集')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 调试模式下减小数据规模
    if args.debug:
        logger.info("调试模式：使用缩减的数据集")
        config['data_processing']['window_size'] = 60
        config['data_processing']['stride'] = 30
        config['validation']['n_splits'] = 2
        config['training_loop']['epochs'] = 5

    try:
        # 运行主程序
        metrics = main(config)

        # 返回成功状态
        sys.exit(0 if metrics['r2'] >= 0.75 else 1)

    except KeyboardInterrupt:
        logger.info("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)