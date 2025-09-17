#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现实化仿真数据增强模块
Realistic Simulation Data Enhancement Module

基于分析报告优化仿真数据，增加现实工况的动态特性
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class RealisticDataEnhancer:
    """
    现实化数据增强器
    为仿真数据添加真实LNG接收站的动态特性
    """

    def __init__(self, random_seed=42):
        """
        初始化数据增强器

        Args:
            random_seed: 随机种子，确保可复现性
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed

    def generate_ship_schedule(self, total_days=180, cycle_days=10,
                              loading_hours=(6, 12)) -> np.ndarray:
        """
        生成船舶装卸调度

        Args:
            total_days: 总仿真天数
            cycle_days: 船舶到达周期（天）
            loading_hours: 装卸时长范围（小时）

        Returns:
            ship_factor: 每天的船舶装卸系数 [0-3.0]
        """
        days = np.arange(total_days)
        ship_factor = np.ones(total_days)  # 基础系数1.0

        # 每cycle_days天安排一次船舶作业
        for day in range(0, total_days, cycle_days):
            # 随机装卸时长
            loading_duration = np.random.uniform(*loading_hours)
            loading_days = int(np.ceil(loading_duration / 24))

            # 装卸期间需求激增
            start_day = day
            end_day = min(day + loading_days, total_days)

            # 装卸系数：2.0-3.5倍正常需求
            ship_factor[start_day:end_day] = np.random.uniform(2.0, 3.5)

        return ship_factor

    def generate_daily_profile(self, hours_per_day=24) -> np.ndarray:
        """
        生成日负荷曲线

        Args:
            hours_per_day: 每天小时数

        Returns:
            daily_profile: 24小时负荷系数 [0.6-1.6]
        """
        hours = np.arange(hours_per_day)

        # 双峰模式：早高峰(8-10h) + 晚高峰(18-20h)
        morning_peak = 1.4 * np.exp(-((hours - 9)**2) / (2 * 1.5**2))
        evening_peak = 1.6 * np.exp(-((hours - 19)**2) / (2 * 1.5**2))

        # 基础负荷 + 峰值
        daily_profile = 0.7 + 0.3 * np.sin(2 * np.pi * hours / 24) + morning_peak + evening_peak

        # 限制在合理范围
        daily_profile = np.clip(daily_profile, 0.6, 1.8)

        return daily_profile

    def generate_seasonal_variation(self, total_days=180) -> np.ndarray:
        """
        生成季节性变化

        Args:
            total_days: 总天数

        Returns:
            seasonal_factor: 季节系数 [0.8-1.3]
        """
        days = np.arange(total_days)

        # 夏季需求高，冬季需求低
        seasonal_base = 1.0 + 0.15 * np.sin(2 * np.pi * days / 365)

        # 随机波动
        seasonal_noise = np.random.normal(0, 0.05, total_days)

        seasonal_factor = seasonal_base + seasonal_noise
        return np.clip(seasonal_factor, 0.8, 1.3)

    def add_equipment_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为设备添加动态特性

        Args:
            df: 原始仿真数据

        Returns:
            enhanced_df: 增强后的数据
        """
        enhanced_df = df.copy()
        n_samples = len(df)

        # 1. 泵效率动态变化 (±5-15%)
        pump_efficiency_variation = np.random.normal(1.0, 0.08, n_samples)
        pump_efficiency_variation = np.clip(pump_efficiency_variation, 0.7, 1.15)

        # 2. 设备老化效应（渐进式效率下降）
        aging_factor = 1.0 - np.linspace(0, 0.05, n_samples)  # 5%老化

        # 3. 维护事件（周期性效率恢复）
        maintenance_schedule = np.zeros(n_samples)
        maintenance_days = [30, 90, 150]  # 维护日
        for day in maintenance_days:
            if day * 144 < n_samples:  # 144 = 24h * 6 (10分钟间隔)
                start_idx = day * 144
                end_idx = min(start_idx + 144, n_samples)
                maintenance_schedule[start_idx:end_idx] = 0.15  # 维护期间15%效率损失

        # 应用到泵功率
        total_efficiency = pump_efficiency_variation * aging_factor * (1 - maintenance_schedule)

        enhanced_df['booster_pump_power_kw'] *= total_efficiency
        enhanced_df['hp_pump_power_kw'] *= total_efficiency

        return enhanced_df

    def add_operational_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加操作变化和环境影响

        Args:
            df: 输入数据

        Returns:
            enhanced_df: 增强数据
        """
        enhanced_df = df.copy()
        n_samples = len(df)

        # 1. 船舶装卸影响
        # 假设每10天一次装卸，影响持续6-12小时
        ship_schedule = np.ones(n_samples)
        for i in range(0, n_samples, 1440):  # 每10天 (1440 = 10天 * 144)
            loading_duration = np.random.randint(36, 72)  # 6-12小时 * 6 (10分钟间隔)
            end_idx = min(i + loading_duration, n_samples)
            # 装卸期间需求增加2-4倍
            ship_schedule[i:end_idx] = np.random.uniform(2.0, 4.0)

        # 2. 日周期负荷（24小时周期）
        hours = np.arange(n_samples) / 6  # 每小时6个点（10分钟间隔）
        daily_variation = 1.0 + 0.3 * np.sin(2 * np.pi * hours / 24)
        daily_variation += 0.2 * np.sin(4 * np.pi * hours / 24)  # 双峰

        # 3. 季节性变化
        days = np.arange(n_samples) / 144  # 每天144个点
        seasonal_variation = 1.0 + 0.2 * np.sin(2 * np.pi * days / 365)

        # 4. 环境噪声
        env_noise = np.random.normal(1.0, 0.05, n_samples)

        # 5. 测量噪声（传感器误差）
        measurement_noise = np.random.normal(1.0, 0.02, n_samples)

        # 组合所有变化
        total_variation = (ship_schedule * daily_variation *
                          seasonal_variation * env_noise * measurement_noise)

        # 应用到能耗数据
        enhanced_df['booster_pump_power_kw'] *= total_variation
        enhanced_df['hp_pump_power_kw'] *= total_variation

        # BOG压缩机受环境影响更大
        bog_variation = env_noise * seasonal_variation * np.random.normal(1.0, 0.15, n_samples)
        enhanced_df['bog_compressor_total_power_kw'] *= bog_variation

        return enhanced_df

    def add_fault_injection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        注入故障事件

        Args:
            df: 输入数据

        Returns:
            enhanced_df: 包含故障的数据
        """
        enhanced_df = df.copy()
        n_samples = len(df)

        # 故障事件定义（基于论文要求）
        fault_events = [
            {'day': 60, 'type': 'leak', 'duration_hours': 48, 'energy_impact': 1.05},
            {'day': 120, 'type': 'orv_fouling', 'duration_hours': 72, 'energy_impact': 1.08},
            {'day': 160, 'type': 'pump_cavitation', 'duration_hours': 24, 'energy_impact': 1.12}
        ]

        for fault in fault_events:
            start_idx = fault['day'] * 144  # 每天144个点
            duration_points = fault['duration_hours'] * 6  # 每小时6个点
            end_idx = min(start_idx + duration_points, n_samples)

            if start_idx < n_samples:
                # 应用故障影响
                enhanced_df.loc[start_idx:end_idx, 'booster_pump_power_kw'] *= fault['energy_impact']
                enhanced_df.loc[start_idx:end_idx, 'hp_pump_power_kw'] *= fault['energy_impact']

                if fault['type'] == 'orv_fouling':
                    # ORV结垢影响BOG系统
                    enhanced_df.loc[start_idx:end_idx, 'bog_compressor_total_power_kw'] *= 1.15

        return enhanced_df

    def enhance_simulation_data(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """
        完整的数据增强流程

        Args:
            input_file: 原始仿真数据文件
            output_file: 输出文件路径（可选）

        Returns:
            enhanced_df: 增强后的数据
        """
        logger.info("开始现实化数据增强...")

        # 1. 加载原始数据
        logger.info(f"加载原始数据: {input_file}")
        df = pd.read_csv(input_file, parse_dates=['ts'])
        original_stats = self._calculate_stats(df)
        logger.info(f"原始数据统计: {original_stats}")

        # 2. 添加设备动态特性
        logger.info("添加设备动态特性...")
        df = self.add_equipment_dynamics(df)

        # 3. 添加操作变化
        logger.info("添加操作变化和环境影响...")
        df = self.add_operational_variations(df)

        # 4. 注入故障事件
        logger.info("注入故障事件...")
        df = self.add_fault_injection(df)

        # 5. 重新计算总能耗
        power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']
        df['energy'] = df[power_columns].sum(axis=1)

        # 6. 统计增强效果
        enhanced_stats = self._calculate_stats(df)
        logger.info(f"增强后统计: {enhanced_stats}")

        # 7. 保存增强数据
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"增强数据已保存: {output_file}")

        # 8. 生成对比报告
        self._generate_comparison_report(original_stats, enhanced_stats)

        return df

    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """计算数据统计信息"""
        power_columns = ['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']

        if 'energy' not in df.columns:
            df['energy'] = df[power_columns].sum(axis=1)

        stats = {}
        for col in power_columns + ['energy']:
            if col in df.columns:
                values = df[col]
                stats[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'cv': values.std() / values.mean() if values.mean() > 0 else 0,
                    'min': values.min(),
                    'max': values.max(),
                    'range': values.max() - values.min()
                }

        return stats

    def _generate_comparison_report(self, original_stats: Dict, enhanced_stats: Dict):
        """生成对比报告"""
        logger.info("\n" + "="*60)
        logger.info("数据增强效果对比")
        logger.info("="*60)

        for var in ['energy', 'booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']:
            if var in original_stats and var in enhanced_stats:
                orig = original_stats[var]
                enhanced = enhanced_stats[var]

                logger.info(f"\n{var}:")
                logger.info(f"  变异系数: {orig['cv']:.6f} → {enhanced['cv']:.6f} (提升{enhanced['cv']/orig['cv']:.1f}倍)")
                logger.info(f"  标准差: {orig['std']:.3f} → {enhanced['std']:.3f}")
                logger.info(f"  范围: {orig['range']:.3f} → {enhanced['range']:.3f}")

def create_enhanced_dataset():
    """
    创建增强数据集的便捷函数
    """
    enhancer = RealisticDataEnhancer(random_seed=2025)

    # 增强现有数据
    enhanced_df = enhancer.enhance_simulation_data(
        input_file='data/sim_lng/full_simulation_data.csv',
        output_file='data/sim_lng/enhanced_simulation_data.csv'
    )

    logger.info(f"\n✅ 增强数据集创建完成")
    logger.info(f"   原始文件: data/sim_lng/full_simulation_data.csv")
    logger.info(f"   增强文件: data/sim_lng/enhanced_simulation_data.csv")
    logger.info(f"   数据量: {len(enhanced_df):,} 行")

    return enhanced_df

# 测试函数
def test_enhancement():
    """测试数据增强效果"""
    print("="*60)
    print("测试现实化数据增强模块")
    print("="*60)

    # 创建测试数据
    n_samples = 1000
    test_data = {
        'ts': pd.date_range('2024-01-01', periods=n_samples, freq='10min'),
        'booster_pump_power_kw': np.full(n_samples, 220.0),  # 恒定值
        'hp_pump_power_kw': np.full(n_samples, 1200.0),     # 恒定值
        'bog_compressor_total_power_kw': np.random.normal(5.0, 0.05, n_samples)  # 轻微变化
    }

    df = pd.DataFrame(test_data)
    df['energy'] = df[['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']].sum(axis=1)

    print(f"原始数据总能耗CV: {df['energy'].std()/df['energy'].mean():.8f}")

    # 应用增强
    enhancer = RealisticDataEnhancer()
    df = enhancer.add_equipment_dynamics(df)
    df = enhancer.add_operational_variations(df)
    df = enhancer.add_fault_injection(df)

    # 重新计算总能耗
    df['energy'] = df[['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']].sum(axis=1)

    print(f"增强后总能耗CV: {df['energy'].std()/df['energy'].mean():.6f}")
    print(f"变异性提升: {(df['energy'].std()/df['energy'].mean()) / (0.00005):.0f}倍")

    print("\n✅ 数据增强测试完成")
    return df

if __name__ == "__main__":
    # 运行测试
    test_enhancement()