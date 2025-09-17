#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防过拟合数据生成器
Anti-Overfitting Data Generator

整合方案A、B、C，生成更具挑战性、去相关的LNG数据
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_anti_overfitting_data():
    """生成防过拟合的现实化数据"""
    logger.info("=" * 60)
    logger.info("生成防过拟合LNG仿真数据")
    logger.info("=" * 60)

    # 1. 加载原始数据
    original_path = Path("data/sim_lng/full_simulation_data.csv")
    if not original_path.exists():
        raise FileNotFoundError(f"原始数据不存在: {original_path}")

    df = pd.read_csv(original_path, parse_dates=['ts'])
    logger.info(f"原始数据: {len(df)} 行")

    # 2. 应用防过拟合增强
    enhanced_df = apply_anti_overfitting_enhancement(df)

    # 3. 验证数据质量
    power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
    enhanced_df['energy'] = enhanced_df[power_columns].sum(axis=1)

    # 计算质量指标
    correlation = np.corrcoef(enhanced_df['energy'][:-1], enhanced_df['energy'][1:])[0, 1]
    cv = enhanced_df['energy'].std() / enhanced_df['energy'].mean()

    logger.info(f"\n数据质量验证:")
    logger.info(f"  时间相关性: {correlation:.4f} (目标: 0.3-0.7)")
    logger.info(f"  变异系数: {cv:.4f}")
    logger.info(f"  能耗范围: [{enhanced_df['energy'].min():.1f}, {enhanced_df['energy'].max():.1f}] kW")

    # 4. 保存改进数据
    output_path = Path("data/sim_lng/anti_overfitting_data.csv")
    enhanced_df.to_csv(output_path, index=False)
    logger.info(f"\n防过拟合数据已保存: {output_path}")

    return enhanced_df

def apply_anti_overfitting_enhancement(df):
    """应用防过拟合增强策略"""
    enhanced_df = df.copy()
    n_samples = len(df)

    # 方案A: 增加真实随机性
    logger.info("  方案A: 独立随机事件")

    # 1. 随机冲击事件 (5%概率，独立发生)
    shock_probability = 0.05
    random_shocks = np.where(
        np.random.random(n_samples) < shock_probability,
        np.random.exponential(0.4, n_samples),  # 指数分布冲击
        0
    )

    # 2. 设备随机故障 (泊松过程)
    failure_events = np.random.poisson(0.0005, n_samples)
    failure_impact = np.where(
        failure_events > 0,
        np.random.choice([0.6, 0.8, 1.2, 1.5], n_samples),  # 离散故障影响
        1.0
    )

    # 3. 市场需求波动 (对数正态分布)
    market_volatility = np.random.lognormal(0, 0.15, n_samples)

    # 方案B: 打破时间依赖性
    logger.info("  方案B: 去相关化处理")

    # 4. 强白噪声 (20%强度)
    white_noise = np.random.normal(1.0, 0.2, n_samples)

    # 5. 随机跳跃 (3%概率，打破平滑性)
    jump_probability = 0.03
    jump_events = np.where(
        np.random.random(n_samples) < jump_probability,
        np.random.choice([0.5, 0.7, 1.4, 1.8], n_samples),
        1.0
    )

    # 6. 独立随机船舶调度
    ship_events = np.random.exponential(1.0, n_samples)
    ship_threshold = np.percentile(ship_events, 85)  # 15%时间有船舶影响
    ship_impact = np.where(
        ship_events > ship_threshold,
        np.random.uniform(2.0, 4.0, n_samples),
        1.0
    )

    # 方案C: 复杂非线性模式
    logger.info("  方案C: 非线性复杂模式")

    # 7. 多尺度时间模式
    hours = np.arange(n_samples) / 6
    days = np.arange(n_samples) / 144

    # 复杂时间模式 (非简单周期)
    hourly_chaos = np.sin(2 * np.pi * hours / 24) + 0.3 * np.sin(6.7 * np.pi * hours / 24)
    daily_chaos = np.sin(2 * np.pi * days / 7) + 0.2 * np.sin(3.3 * np.pi * days / 7)
    seasonal_chaos = 0.1 * np.sin(2 * np.pi * days / 365) + 0.05 * np.sin(5.1 * np.pi * days / 365)

    complex_temporal = 1.0 + 0.15 * (hourly_chaos + daily_chaos + seasonal_chaos)

    # 8. 非线性交互效应
    interaction_effects = np.tanh(0.1 * (random_shocks + white_noise - 1))

    # 最终组合：非线性、去相关、高随机性
    pump_variation = ((1 + random_shocks) * failure_impact *
                     white_noise * jump_events * ship_impact *
                     complex_temporal * (1 + interaction_effects))

    bog_variation = (market_volatility *
                    np.random.gamma(1.5, 0.8, n_samples) *
                    (1 + 0.5 * interaction_effects))

    # 应用变化
    enhanced_df['booster_pump_power_kw'] *= pump_variation
    enhanced_df['hp_pump_power_kw'] *= pump_variation
    enhanced_df['bog_compressor_total_power_kw'] *= bog_variation

    # 添加测量噪声 (最后一步，确保去相关)
    measurement_noise = np.random.normal(1.0, 0.03, n_samples)
    for col in ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']:
        enhanced_df[col] *= measurement_noise

    return enhanced_df

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 生成防过拟合数据
    enhanced_data = generate_anti_overfitting_data()
    print("防过拟合数据生成完成")