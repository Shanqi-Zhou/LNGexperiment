#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNG论文复现项目 - 统一入口
LNG Paper Reproduction Project - Unified Entry Point

单一入口文件集成所有优化：
- 现实化数据 + 高性能特征工程 + 跨模态融合 + GPU加速
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import gc
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lng_experiment.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LNGExperiment:
    """LNG论文复现实验主类"""

    def __init__(self, random_seed=42):
        """初始化实验环境"""
        # 设置随机种子确保可重现性
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
        logger.info(f"随机种子设置为: {random_seed}")
        
        # GPU设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.info("使用CPU模式")

    def load_enhanced_data(self):
        """加载现实化数据"""
        logger.info("加载现实化LNG数据...")

        # 检查数据文件
        data_path = Path("data/sim_lng/anti_overfitting_data.csv")
        if not data_path.exists():
            # 如果没有增强数据，生成它
            logger.info("增强数据不存在，正在生成...")
            self.generate_enhanced_data()

        # 加载数据
        df = pd.read_csv(data_path, parse_dates=['ts'])
        power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
        df['energy'] = df[power_columns].sum(axis=1)

        logger.info(f"数据规模: {len(df):,} 行")
        logger.info(f"能耗特性: 均值={df['energy'].mean():.2f} kW, CV={df['energy'].std()/df['energy'].mean():.4f}")

        return df

    def generate_enhanced_data(self):
        """生成现实化增强数据"""
        logger.info("开始生成现实化数据...")

        # 检查原始数据
        original_path = Path("data/sim_lng/full_simulation_data.csv")
        if not original_path.exists():
            raise FileNotFoundError(f"原始数据文件不存在: {original_path}")

        # 使用内置的数据增强逻辑
        df = pd.read_csv(original_path, parse_dates=['ts'])
        enhanced_df = self.apply_realistic_enhancement(df)

        # 保存增强数据
        output_path = Path("data/sim_lng/anti_overfitting_data.csv")
        enhanced_df.to_csv(output_path, index=False)
        logger.info(f"现实化数据已生成: {output_path}")

    def apply_realistic_enhancement(self, df):
        """应用现实化增强"""
        # 重新设置随机种子确保数据增强的一致性
        np.random.seed(self.random_seed + 1)
        
        enhanced_df = df.copy()
        n_samples = len(df)

        # 1. 船舶装卸影响 (每10天一次)
        ship_schedule = np.ones(n_samples)
        for i in range(0, n_samples, 1440):  # 每10天
            loading_duration = np.random.randint(36, 72)  # 6-12小时
            end_idx = min(i + loading_duration, n_samples)
            ship_schedule[i:end_idx] = np.random.uniform(2.0, 4.0)

        # 2. 日周期和季节变化
        hours = np.arange(n_samples) / 6
        daily_variation = 1.0 + 0.3 * np.sin(2 * np.pi * hours / 24)

        days = np.arange(n_samples) / 144
        seasonal_variation = 1.0 + 0.2 * np.sin(2 * np.pi * days / 365)

        # 3. 设备动态和噪声
        efficiency_variation = np.random.normal(1.0, 0.08, n_samples)
        measurement_noise = np.random.normal(1.0, 0.02, n_samples)

        # 应用所有变化
        total_variation = (ship_schedule * daily_variation *
                          seasonal_variation * efficiency_variation * measurement_noise)

        enhanced_df['booster_pump_power_kw'] *= total_variation
        enhanced_df['hp_pump_power_kw'] *= total_variation
        enhanced_df['bog_compressor_total_power_kw'] *= np.random.normal(1.0, 0.15, n_samples)

        return enhanced_df

    def extract_features_fast(self, df):
        """高性能特征提取"""
        logger.info("开始高性能特征提取...")

        window_size, stride = 180, 30
        feature_df = df.drop(columns=['ts', 'energy'])

        # 使用滑动窗口创建视图
        from numpy.lib.stride_tricks import sliding_window_view
        data = feature_df.values
        windows = sliding_window_view(data, (window_size, data.shape[1]))
        windows = windows[::stride, 0, :, :]

        # 快速统计特征计算
        features_list = []
        for feature_idx in range(data.shape[1]):
            window_data = windows[:, :, feature_idx]

            # 基础统计
            feat = np.column_stack([
                np.mean(window_data, axis=1),
                np.std(window_data, axis=1),
                np.min(window_data, axis=1),
                np.max(window_data, axis=1),
                np.median(window_data, axis=1)
            ])
            features_list.append(feat)

        X = np.concatenate(features_list, axis=1)

        # 生成对应标签
        y_list = []
        for i in range(0, len(df) - window_size + 1, stride):
            y_list.append(df['energy'].iloc[i:i+window_size].mean())
        y = np.array(y_list[:len(X)])

        logger.info(f"特征提取完成: {X.shape}, 标签: {y.shape}")
        return X, y

    def create_cross_modal_model(self, input_dim, hidden_dim=128):
        """创建简化的跨模态融合模型"""
        class SimpleCrossModalFusion(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                self.feedforward = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                self.output_head = nn.Linear(hidden_dim, 1)
                self.layer_norm = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                # x shape: [batch, features]
                x = self.input_projection(x)  # [batch, hidden_dim]
                x = x.unsqueeze(1)  # [batch, 1, hidden_dim] for attention

                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.layer_norm(x + attn_out)

                # Feedforward
                ff_out = self.feedforward(x)
                x = self.layer_norm(x + ff_out)

                # Output
                output = self.output_head(x.squeeze(1))  # [batch, 1]
                return output

        return SimpleCrossModalFusion(input_dim, hidden_dim)

    def train_model(self, X, y):
        """训练跨模态融合模型"""
        logger.info("开始跨模态融合模型训练...")
        
        # 重新设置随机种子确保训练过程的一致性
        np.random.seed(self.random_seed + 2)
        torch.manual_seed(self.random_seed + 2)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed + 2)

        # 时间序列感知的验证分割（防止数据泄漏）
        # 使用时间序列分割而非随机分割
        split_point = int(len(X) * 0.8)  # 前80%作为训练，后20%作为验证
        X_train, X_val = X[:split_point], X[split_point:]
        y_train, y_val = y[:split_point], y[split_point:]

        logger.info(f"时间序列分割: 训练集{len(X_train)}样本, 验证集{len(X_val)}样本")

        # 标准化
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

        X_train_s = scaler_X.transform(X_train).astype(np.float32)
        X_val_s = scaler_X.transform(X_val).astype(np.float32)
        y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)

        # 创建模型
        model = self.create_cross_modal_model(input_dim=X_train_s.shape[1]).to(self.device)
        logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

        # 训练设置
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        batch_size = 512  # RTX 4060适配

        # 训练循环
        model.train()
        for epoch in range(30):
            epoch_loss = 0
            n_batches = 0

            # 分批训练
            for i in range(0, len(X_train_s), batch_size):
                batch_X = torch.FloatTensor(X_train_s[i:i+batch_size]).to(self.device)
                batch_y = torch.FloatTensor(y_train_s[i:i+batch_size]).unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                del batch_X, batch_y
                torch.cuda.empty_cache()

            if n_batches > 0 and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"  Epoch {epoch+1}/30, Loss: {avg_loss:.6f}")

        # 评估
        logger.info("开始模型评估...")
        model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(X_val_s), batch_size):
                batch_X = torch.FloatTensor(X_val_s[i:i+batch_size]).to(self.device)
                pred = model(batch_X)
                predictions.append(pred.cpu().numpy())
                del batch_X
                torch.cuda.empty_cache()

        y_pred_s = np.concatenate(predictions).flatten()

        # 反标准化
        y_pred_orig = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_val_orig = scaler_y.inverse_transform(y_val_s.reshape(-1, 1)).flatten()

        # 计算学术指标
        r2 = r2_score(y_val_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
        cv_rmse = rmse / np.mean(y_val_orig)
        nmbe = np.mean(y_pred_orig - y_val_orig) / np.mean(y_val_orig)

        return {
            'r2': r2,
            'cv_rmse': cv_rmse,
            'nmbe': nmbe,
            'rmse': rmse
        }

    def run_experiment(self):
        """运行完整实验"""
        logger.info("=" * 60)
        logger.info("LNG论文复现项目 - 统一实验入口")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # 1. 数据加载
            logger.info("\n--- 阶段 1: 数据准备 ---")
            df = self.load_enhanced_data()

            # 2. 特征工程
            logger.info("\n--- 阶段 2: 高性能特征工程 ---")
            feature_start = time.time()
            X, y = self.extract_features_fast(df)
            feature_time = time.time() - feature_start
            logger.info(f"特征提取完成: {feature_time:.2f}秒, 速度: {len(X)/feature_time:.1f} 窗口/秒")

            # 3. 模型训练与评估
            logger.info("\n--- 阶段 3: 跨模态融合训练 ---")
            results = self.train_model(X, y)

            # 4. 结果报告
            logger.info("\n" + "=" * 60)
            logger.info("LNG跨模态融合模型最终结果")
            logger.info("=" * 60)

            logger.info(f"\n📊 学术指标:")
            logger.info(f"  R² Score: {results['r2']:.4f} {'✅' if results['r2'] >= 0.75 else '❌'} (目标 ≥ 0.75)")
            logger.info(f"  CV(RMSE): {results['cv_rmse']:.4f} {'✅' if results['cv_rmse'] <= 0.06 else '❌'} (目标 ≤ 0.06)")
            logger.info(f"  NMBE: {results['nmbe']:.4f} {'✅' if abs(results['nmbe']) <= 0.006 else '❌'} (目标 ∈ [-0.006, 0.006])")

            # 论文要求检查
            requirements_met = (
                results['r2'] >= 0.75 and
                results['cv_rmse'] <= 0.06 and
                abs(results['nmbe']) <= 0.006
            )

            if requirements_met:
                logger.info("\n🎉 恭喜！跨模态融合模型达到论文要求！")
            else:
                logger.info("\n📈 模型在现实数据上展现挑战性，为进一步研究提供方向")

            # 5. 保存结果
            final_results = {
                'experiment': 'LNG_CrossModal_Fusion',
                'dataset': {
                    'source': 'anti_overfitting_data.csv',
                    'rows': len(df),
                    'windows': len(X),
                    'energy_cv': df['energy'].std()/df['energy'].mean()
                },
                'performance': {
                    'feature_extraction_time': feature_time,
                    'processing_speed': len(X)/feature_time
                },
                'academic_metrics': {
                    'r2': float(results['r2']),
                    'cv_rmse': float(results['cv_rmse']),
                    'nmbe': float(results['nmbe']),
                    'rmse': float(results['rmse'])
                },
                'requirements_met': requirements_met,
                'total_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

            # 保存结果
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / f"lng_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"

            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(final_results, f, allow_unicode=True)

            logger.info(f"\n结果已保存: {results_path}")
            logger.info(f"总运行时间: {(time.time() - start_time)/60:.2f} 分钟")

            return results

        except Exception as e:
            logger.error(f"实验执行失败: {e}", exc_info=True)
            raise

def main():
    """主入口函数"""
    try:
        # 创建并运行实验
        experiment = LNGExperiment()
        results = experiment.run_experiment()

        # 根据结果返回退出码
        if results['r2'] >= 0.75:
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 需要改进

    except KeyboardInterrupt:
        logger.info("\n用户中断实验")
        sys.exit(1)
    except Exception as e:
        logger.error(f"实验失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()