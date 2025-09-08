#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LNG仿真数据生成脚本

根据技术路线要求生成LNG全链路工况仿真数据，包括：
- 环境气象数据
- 储罐运行数据
- 泵系统数据
- ORV气化器数据
- BOG压缩机数据
- 再冷凝器数据
- 外输计量数据

生成的数据将保存为CSV格式，用于后续的机器学习模型训练。
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# 添加src路径到系统路径
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from sim import LNGSystemSimulator
from sim.control import LNGSystemController

def load_simulation_config(config_path: str) -> dict:
    """加载仿真配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        # 创建默认配置
        default_config = {
            'simulation': {
                'start_time': '2024-01-01T00:00:00',
                'duration_days': 180,
                'time_step': 60.0,  # 1分钟步长
                'random_seed': 2025
            },
            'tank': {
                'capacity_m3': 2000,
                'initial_level_pct': 85.0,
                'design_pressure_kPa': 150.0,
                'insulation_thickness_m': 0.3,
                'ambient_heat_leak_factor': 1.0
            },
            'pumps': {
                'booster': {
                    'rated_flow_m3h': 800,
                    'rated_head_m': 50,
                    'rated_power_kw': 200,
                    'efficiency': 0.75,
                    'npsh_required_m': 2.0
                },
                'high_pressure': {
                    'rated_flow_m3h': 600,
                    'rated_head_m': 1500,
                    'rated_power_kw': 2500,
                    'efficiency': 0.78,
                    'npsh_required_m': 3.0
                }
            },
            'orv': {
                'design_capacity_tph': 50,
                'tube_count': 200,
                'tube_length_m': 12,
                'tube_diameter_m': 0.05,
                'seawater_flow_m3h': 8000,
                'initial_fouling_factor': 1.0,
                'fouling_rate_per_day': 0.001
            },
            'bog_compressor': {
                'stage1': {
                    'design_flow_kgh': 2000,
                    'compression_ratio': 3.0,
                    'efficiency': 0.75,
                    'rated_power_kw': 800
                },
                'stage2': {
                    'design_flow_kgh': 2000,
                    'compression_ratio': 2.5,
                    'efficiency': 0.78,
                    'rated_power_kw': 600
                },
                'intercooler_efficiency': 0.85
            },
            'recondenser': {
                'design_capacity_kgh': 1500,
                'spray_nozzle_count': 20,
                'column_diameter_m': 2.0,
                'column_height_m': 8.0,
                'heat_transfer_coeff': 2000
            },
            'environment': {
                'temp_base': 25.0,
                'seawater_temp': 15.0,
                'humidity_base': 70.0,
                'wind_speed_base': 5.0
            },
            'export': {
                'pressure_setpoint': 6500.0,
                'flow_base': 1000.0
            },
            'control': {
                'tank_pressure_setpoint': 120.0,
                'export_pressure_setpoint': 6500.0,
                'export_flow_setpoint': 1000.0,
                'orv_temp_setpoint': 5.0,
                'recondenser_temp_setpoint': -100.0
            }
        }
        
        # 保存默认配置
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        print(f"Created default config file: {config_path}")
        
        return default_config
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

def setup_output_directories(base_dir: str) -> dict:
    """设置输出目录
    
    Args:
        base_dir: 基础输出目录
        
    Returns:
        输出目录字典
    """
    dirs = {
        'base': base_dir,
        'raw': os.path.join(base_dir, 'raw'),
        'processed': os.path.join(base_dir, 'processed'),
        'features': os.path.join(base_dir, 'features'),
        'models': os.path.join(base_dir, 'models'),
        'results': os.path.join(base_dir, 'results')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def run_simulation(config: dict, output_dirs: dict, verbose: bool = True) -> pd.DataFrame:
    """运行LNG系统仿真
    
    Args:
        config: 仿真配置
        output_dirs: 输出目录
        verbose: 是否显示详细信息
        
    Returns:
        仿真结果DataFrame
    """
    if verbose:
        print("\n=== 开始LNG系统仿真 ===")
        print(f"仿真时长: {config['simulation']['duration_days']} 天")
        print(f"时间步长: {config['simulation']['time_step']} 秒")
        
    # 创建仿真器
    simulator = LNGSystemSimulator(config)
    
    # 创建控制系统
    controller = LNGSystemController(config['control'])
    
    # 运行仿真
    total_steps = int(config['simulation']['duration_days'] * 24 * 3600 / config['simulation']['time_step'])
    
    if verbose:
        print(f"总仿真步数: {total_steps}")
        
    # 使用tqdm显示进度
    with tqdm(total=total_steps, desc="仿真进度", disable=not verbose) as pbar:
        def progress_callback(progress, elapsed_days):
            pbar.set_postfix({
                'Day': f"{elapsed_days}/{config['simulation']['duration_days']}",
                'Progress': f"{progress:.1f}%"
            })
            pbar.update(1000)  # 每1000步更新一次
            
        # 运行仿真
        results_df = simulator.run_simulation(progress_callback)
        
    if verbose:
        print(f"\n仿真完成！生成了 {len(results_df)} 条记录")
        print(f"数据时间范围: {results_df['timestamp'].min()} 到 {results_df['timestamp'].max()}")
        
    return results_df

def save_simulation_data(df: pd.DataFrame, output_dirs: dict, verbose: bool = True):
    """保存仿真数据
    
    Args:
        df: 仿真结果DataFrame
        output_dirs: 输出目录
        verbose: 是否显示详细信息
    """
    if verbose:
        print("\n=== 保存仿真数据 ===")
        
    # 按照技术路线要求的格式保存数据
    data_files = {
        'env_weather.csv': {
            'columns': ['timestamp', 'environment_T_amb_C', 'environment_seawater_T_C', 
                       'environment_humidity_pct', 'environment_wind_mps'],
            'description': '环境气象数据'
        },
        'tanks.csv': {
            'columns': ['timestamp', 'tank_level_pct', 'tank_p_top_kPa', 'tank_bog_rate_kgph', 
                       'tank_T_liquid_C', 'tank_T_vapor_C'],
            'description': 'LNG储罐数据'
        },
        'pumps_booster.csv': {
            'columns': ['timestamp', 'pumps_booster_flow_m3h', 'pumps_booster_head_m', 
                       'pumps_booster_power_kw', 'pumps_booster_efficiency', 'pumps_booster_npsh_a_m'],
            'description': '增压泵数据'
        },
        'pumps_hp.csv': {
            'columns': ['timestamp', 'pumps_hp_flow_m3h', 'pumps_hp_head_m', 
                       'pumps_hp_power_kw', 'pumps_hp_efficiency', 'pumps_hp_npsh_a_m'],
            'description': '高压泵数据'
        },
        'orv.csv': {
            'columns': ['timestamp', 'orv_m_LNG_tph', 'orv_T_out_C', 'orv_Q_MW', 
                       'orv_U_eff_WK', 'orv_fouling_factor'],
            'description': 'ORV气化器数据'
        },
        'bog_compressor.csv': {
            'columns': ['timestamp', 'bog_compressor_total_flow_kgh', 'bog_compressor_total_power_kw',
                       'bog_compressor_stage1_pressure_ratio', 'bog_compressor_stage2_pressure_ratio'],
            'description': 'BOG压缩机数据'
        },
        'recondenser.csv': {
            'columns': ['timestamp', 'recondenser_m_bog_in_kgph', 'recondenser_m_condensed_kgph',
                       'recondenser_T_bog_out_C', 'recondenser_spray_flow_tph'],
            'description': '再冷凝器数据'
        },
        'export_meter.csv': {
            'columns': ['timestamp', 'export_demand_Nm3h', 'orv_T_out_C', 'total_power_kw', 'energy_MJ'],
            'description': '外输计量数据'
        }
    }
    
    # 保存各个数据文件
    for filename, file_info in data_files.items():
        # 选择存在的列
        available_columns = [col for col in file_info['columns'] if col in df.columns]
        
        if available_columns:
            subset_df = df[available_columns].copy()
            
            # 保存到raw目录
            raw_path = os.path.join(output_dirs['raw'], filename)
            subset_df.to_csv(raw_path, index=False)
            
            if verbose:
                print(f"保存 {file_info['description']}: {filename} ({len(subset_df)} 条记录)")
        else:
            if verbose:
                print(f"警告: {filename} 的列不存在，跳过保存")
                
    # 保存完整数据
    full_path = os.path.join(output_dirs['raw'], 'full_simulation_data.csv')
    df.to_csv(full_path, index=False)
    
    if verbose:
        print(f"保存完整仿真数据: full_simulation_data.csv ({len(df)} 条记录)")
        print(f"数据文件保存在: {output_dirs['raw']}")
        
    # 生成数据统计报告
    generate_data_report(df, output_dirs, verbose)

def generate_data_report(df: pd.DataFrame, output_dirs: dict, verbose: bool = True):
    """生成数据统计报告
    
    Args:
        df: 仿真结果DataFrame
        output_dirs: 输出目录
        verbose: 是否显示详细信息
    """
    if verbose:
        print("\n=== 生成数据统计报告 ===")
        
    # 基本统计信息
    report = {
        'simulation_info': {
            'total_records': len(df),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'duration_days': (pd.to_datetime(df['timestamp'].max()) - 
                            pd.to_datetime(df['timestamp'].min())).days,
            'columns_count': len(df.columns)
        },
        'data_quality': {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
    }
    
    # 关键变量统计
    key_variables = [
        'tank_level_pct', 'tank_p_top_kPa', 'tank_bog_rate_kgph',
        'pumps_booster_power_kw', 'pumps_hp_power_kw',
        'orv_Q_MW', 'orv_fouling_factor',
        'bog_compressor_total_power_kw',
        'total_power_kw', 'energy_MJ'
    ]
    
    stats_data = {}
    for var in key_variables:
        if var in df.columns:
            stats_data[var] = {
                'mean': float(df[var].mean()),
                'std': float(df[var].std()),
                'min': float(df[var].min()),
                'max': float(df[var].max()),
                'median': float(df[var].median())
            }
            
    report['key_statistics'] = stats_data
    
    # 保存报告
    report_path = os.path.join(output_dirs['base'], 'simulation_report.yaml')
    with open(report_path, 'w', encoding='utf-8') as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True)
        
    if verbose:
        print(f"数据统计报告保存至: {report_path}")
        print(f"\n关键统计信息:")
        print(f"  总记录数: {report['simulation_info']['total_records']:,}")
        print(f"  仿真天数: {report['simulation_info']['duration_days']}")
        print(f"  数据列数: {report['simulation_info']['columns_count']}")
        
        # 显示关键变量统计
        if stats_data:
            print(f"\n关键变量统计 (前5个):")
            for i, (var, stats) in enumerate(list(stats_data.items())[:5]):
                print(f"  {var}: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LNG仿真数据生成脚本')
    parser.add_argument('--config', '-c', 
                       default='../configs/simulation_params.yaml',
                       help='仿真配置文件路径')
    parser.add_argument('--output', '-o',
                       default='../data/sim_lng',
                       help='输出目录路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细信息')
    parser.add_argument('--seed', type=int, default=2025,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, args.config))
    output_path = os.path.abspath(os.path.join(script_dir, args.output))
    
    if args.verbose:
        print("=== LNG仿真数据生成脚本 ===")
        print(f"配置文件: {config_path}")
        print(f"输出目录: {output_path}")
        print(f"随机种子: {args.seed}")
        
    # 设置随机种子
    np.random.seed(args.seed)
    
    try:
        # 加载配置
        config = load_simulation_config(config_path)
        config['simulation']['random_seed'] = args.seed
        
        # 设置输出目录
        output_dirs = setup_output_directories(output_path)
        
        # 运行仿真
        start_time = datetime.now()
        results_df = run_simulation(config, output_dirs, args.verbose)
        
        # 保存数据
        save_simulation_data(results_df, output_dirs, args.verbose)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if args.verbose:
            print(f"\n=== 仿真完成 ===")
            print(f"总耗时: {duration}")
            print(f"生成记录数: {len(results_df):,}")
            print(f"输出目录: {output_path}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()