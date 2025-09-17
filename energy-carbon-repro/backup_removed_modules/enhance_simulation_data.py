#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用现实化数据增强
Apply Realistic Data Enhancement

基于分析报告，对完整仿真数据集进行现实化增强
"""

import sys
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, 'src')

from sim.realistic_enhancer import RealisticDataEnhancer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主执行函数"""
    logger.info("="*60)
    logger.info("LNG仿真数据现实化增强")
    logger.info("="*60)

    # 检查输入文件
    input_file = Path("data/sim_lng/full_simulation_data.csv")
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return

    # 输出文件
    output_file = Path("data/sim_lng/enhanced_simulation_data.csv")

    # 创建增强器
    enhancer = RealisticDataEnhancer(random_seed=2025)

    try:
        # 执行数据增强
        enhanced_df = enhancer.enhance_simulation_data(
            str(input_file),
            str(output_file)
        )

        logger.info("\n🎉 数据增强完成！")
        logger.info(f"   增强数据已保存: {output_file}")
        logger.info(f"   数据规模: {len(enhanced_df):,} 行")

        # 验证增强效果
        total_energy = enhanced_df[['booster_pump_power_kw', 'hp_pump_power_kw', 'bog_compressor_total_power_kw']].sum(axis=1)
        cv_enhanced = total_energy.std() / total_energy.mean()

        logger.info(f"\n📊 增强效果验证:")
        logger.info(f"   总能耗变异系数: {cv_enhanced:.6f}")
        logger.info(f"   变异性改善: {cv_enhanced / 0.00005:.0f}倍")

        if cv_enhanced > 0.1:
            logger.info("✅ 数据现实性显著改善，适合机器学习验证")
        else:
            logger.warning("⚠️ 变异性仍需进一步提升")

    except Exception as e:
        logger.error(f"数据增强失败: {e}")
        return

if __name__ == "__main__":
    main()