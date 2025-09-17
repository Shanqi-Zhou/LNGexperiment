#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能对比基准
Model Performance Comparison Benchmark

为模型对比研究提供快速实验平台
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """模型性能对比基准"""

    def __init__(self, data_size=50000):
        """
        初始化基准测试

        Args:
            data_size: 用于对比的数据量
        """
        self.data_size = data_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_benchmark_data(self):
        """加载基准数据"""
        logger.info(f"加载基准数据 ({self.data_size} 行)...")

        df = pd.read_csv('data/sim_lng/enhanced_simulation_data.csv',
                         parse_dates=['ts'], nrows=self.data_size)

        power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
        df['energy'] = df[power_columns].sum(axis=1)

        logger.info(f"数据特性: 均值={df['energy'].mean():.2f}, CV={df['energy'].std()/df['energy'].mean():.4f}")
        return df

    def extract_baseline_features(self, df):
        """提取基准特征 (快速版本)"""
        logger.info("提取基准特征...")

        # 简化的窗口特征
        window_size, stride = 180, 30
        features_list = []
        labels_list = []

        for i in range(0, len(df) - window_size + 1, stride):
            window = df.iloc[i:i+window_size]

            # 快速统计特征
            feature_data = window.drop(columns=['ts', 'energy']).values
            features = np.concatenate([
                np.mean(feature_data, axis=0),    # 均值
                np.std(feature_data, axis=0),     # 标准差
                np.min(feature_data, axis=0),     # 最小值
                np.max(feature_data, axis=0),     # 最大值
            ])

            features_list.append(features)
            labels_list.append(window['energy'].mean())

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"基准特征: {X.shape}, 标签: {y.shape}")
        return X, y

    def benchmark_models(self, X, y):
        """对比多个模型性能"""
        logger.info("\n" + "=" * 60)
        logger.info("模型性能对比基准测试")
        logger.info("=" * 60)

        # 训练验证分割
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # 标准化
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train_s = scaler_X.transform(X_train)
        X_val_s = scaler_X.transform(X_val)
        y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

        results = {}

        # 1. 线性回归基准
        logger.info("\n1. 线性回归基准")
        lr = LinearRegression()
        start_time = time.time()
        lr.fit(X_train_s, y_train_s)
        lr_pred = lr.predict(X_val_s)
        lr_time = time.time() - start_time

        lr_pred_orig = scaler_y.inverse_transform(lr_pred.reshape(-1, 1)).flatten()
        y_val_orig = scaler_y.inverse_transform(y_val_s.reshape(-1, 1)).flatten()

        results['LinearRegression'] = {
            'r2': r2_score(y_val_orig, lr_pred_orig),
            'cv_rmse': np.sqrt(mean_squared_error(y_val_orig, lr_pred_orig)) / np.mean(y_val_orig),
            'training_time': lr_time
        }

        # 2. 随机森林
        logger.info("\n2. 随机森林")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        start_time = time.time()
        rf.fit(X_train_s, y_train_s)
        rf_pred = rf.predict(X_val_s)
        rf_time = time.time() - start_time

        rf_pred_orig = scaler_y.inverse_transform(rf_pred.reshape(-1, 1)).flatten()

        results['RandomForest'] = {
            'r2': r2_score(y_val_orig, rf_pred_orig),
            'cv_rmse': np.sqrt(mean_squared_error(y_val_orig, rf_pred_orig)) / np.mean(y_val_orig),
            'training_time': rf_time
        }

        # 3. 跨模态融合 (如果可用)
        logger.info("\n3. 跨模态融合 (简化)")
        try:
            from src.models.cross_modal_fusion import CrossModalFusionWithResidual

            # 为跨模态准备数据 (简化版本)
            X_dynamic = X_train_s.reshape(len(X_train_s), -1, 1)  # 简化为单特征时序
            X_static = np.random.randn(len(X_train_s), 32)  # 随机静态特征

            model = CrossModalFusionWithResidual(
                dynamic_dim=1,
                static_dim=32,
                seq_len=X_dynamic.shape[1],
                d_model=64,  # 减小模型以快速训练
                n_heads=2,
                n_layers=2
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()

            # 转换数据
            train_dynamic = torch.FloatTensor(X_dynamic).to(self.device)
            train_static = torch.FloatTensor(X_static).to(self.device)
            train_labels = torch.FloatTensor(y_train_s).unsqueeze(1).to(self.device)

            start_time = time.time()
            model.train()
            for epoch in range(10):  # 快速训练
                optimizer.zero_grad()
                pred, _ = model(train_dynamic, train_static)
                loss = criterion(pred, train_labels)
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()

            cross_modal_time = time.time() - start_time

            # 简化评估
            model.eval()
            X_val_dynamic = X_val_s.reshape(len(X_val_s), -1, 1)
            X_val_static = np.random.randn(len(X_val_s), 32)

            with torch.no_grad():
                val_dynamic = torch.FloatTensor(X_val_dynamic).to(self.device)
                val_static = torch.FloatTensor(X_val_static).to(self.device)
                cm_pred, _ = model(val_dynamic, val_static)
                cm_pred_cpu = cm_pred.cpu().numpy().flatten()

            cm_pred_orig = scaler_y.inverse_transform(cm_pred_cpu.reshape(-1, 1)).flatten()

            results['CrossModalFusion'] = {
                'r2': r2_score(y_val_orig, cm_pred_orig),
                'cv_rmse': np.sqrt(mean_squared_error(y_val_orig, cm_pred_orig)) / np.mean(y_val_orig),
                'training_time': cross_modal_time
            }

        except Exception as e:
            logger.warning(f"跨模态融合测试失败: {e}")
            results['CrossModalFusion'] = {'error': str(e)}

        return results

    def print_comparison_results(self, results):
        """打印对比结果"""
        logger.info("\n" + "=" * 60)
        logger.info("模型性能对比结果")
        logger.info("=" * 60)

        for model_name, metrics in results.items():
            if 'error' in metrics:
                logger.info(f"\n{model_name}: 测试失败 ({metrics['error']})")
                continue

            logger.info(f"\n{model_name}:")
            logger.info(f"  R² Score: {metrics['r2']:.4f}")
            logger.info(f"  CV(RMSE): {metrics['cv_rmse']:.4f}")
            logger.info(f"  训练时间: {metrics['training_time']:.2f}秒")

            # 评估是否达标
            r2_pass = "✅" if metrics['r2'] >= 0.75 else "❌"
            cv_pass = "✅" if metrics['cv_rmse'] <= 0.06 else "❌"

            logger.info(f"  论文指标: R² {r2_pass}, CV(RMSE) {cv_pass}")

def run_model_benchmark():
    """运行模型对比基准"""
    benchmark = ModelBenchmark(data_size=20000)  # 快速对比用较小数据集

    # 加载数据
    df = benchmark.load_benchmark_data()

    # 提取特征
    X, y = benchmark.extract_baseline_features(df)

    # 运行对比
    results = benchmark.benchmark_models(X, y)

    # 显示结果
    benchmark.print_comparison_results(results)

    logger.info("\n🎯 对比基准建立完成，可用于模型研究")

if __name__ == "__main__":
    run_model_benchmark()