
import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到Python路径，以便导入src模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sim.system import LNGSystem

def generate_schedules(total_steps, dt_s):
    """生成环境温度和外输需求的计划曲线"""
    time_hours = np.arange(total_steps) * dt_s / 3600.0
    time_days = time_hours / 24.0

    # 1. 环境和海水温度曲线
    # 季节性变化 (夏季26°C, 冬季10°C)
    seasonal_T_sea = 18 + 8 * np.cos(2 * np.pi * time_days / 180)
    # 日夜变化 (幅值4°C)
    daily_T_amb = 4 * np.cos(2 * np.pi * time_hours / 24)
    T_ambient = seasonal_T_sea + daily_T_amb
    T_seawater = seasonal_T_sea

    # 2. 外输负荷曲线
    # 日负荷 (早晚峰，峰谷比1.6)
    base_load = 1000 # t/h
    daily_demand = base_load * (1.3 + 0.3 * np.sin(2 * np.pi * time_hours / 24 - np.pi/2) + 0.15 * np.sin(4 * np.pi * time_hours / 24))
    # 周末效应 (系数0.9)
    day_of_week = (time_days % 7).astype(int)
    weekend_mask = (day_of_week >= 5)
    daily_demand[weekend_mask] *= 0.9
    
    return T_ambient, T_seawater, daily_demand

def main():
    parser = argparse.ArgumentParser(description="生成LNG接收站仿真数据")
    parser.add_argument('--out', type=str, required=True, help='输出CSV文件的目录路径')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # --- 仿真设置 ---
    SIM_DAYS = 180
    DT_S = 1  # 仿真步长
    LOG_INTERVAL_S = 10 # 记录间隔
    TOTAL_STEPS = SIM_DAYS * 24 * 3600 // DT_S

    # --- 初始化 ---
    system = LNGSystem()
    T_ambient, T_seawater, export_demand = generate_schedules(TOTAL_STEPS, DT_S)
    
    records = []

    # --- 仿真循环 ---
    for t in tqdm(range(TOTAL_STEPS), desc="Running Simulation"):
        # 1. 获取当前外部输入
        current_demand = export_demand[t]
        current_T_amb = T_ambient[t]
        current_T_sea = T_seawater[t]

        # 2. 故障注入
        current_day = (t * DT_S) / (24 * 3600)
        if 60 <= current_day < 61:
            system.orv.fouling_decay_beta = 6.0e-6 # ORV结垢增强
        elif 120 <= current_day < 121:
            # 模拟泄漏，例如增加泵的功耗
            system.hp_pump.power_kW *= 1.05

        # 3. 更新系统状态
        system.update(current_demand, current_T_amb, current_T_sea, DT_S)

        # 4. 记录数据
        if t % LOG_INTERVAL_S == 0:
            timestamp = pd.to_datetime('2024-01-01') + pd.to_timedelta(t * DT_S, 's')
            states = system.get_all_states()
            
            # 将所有状态平铺到一个记录中
            flat_record = {'ts': timestamp}
            for component, state_dict in states.items():
                if 'stages' in state_dict: # 特殊处理压缩机
                    flat_record[f'{component}_total_power_kw'] = state_dict['total_power_kw']
                else:
                    for key, value in state_dict.items():
                        flat_record[f'{component}_{key}'] = value
            records.append(flat_record)

    # --- 保存数据 ---
    df = pd.DataFrame(records)
    
    # 根据文档拆分并保存为不同的CSV文件
    def save_csv(cols, filename):
        subset_cols = ['ts'] + [c for c in cols if c in df.columns]
        df[subset_cols].to_csv(os.path.join(args.out, filename), index=False, date_format='%Y-%m-%dT%H:%M:%S')

    save_csv(['tank_level_pct', 'tank_p_top_kPa', 'tank_bog_rate_kgph'], 'tanks.csv')
    save_csv(['booster_pump_flow_m3h', 'booster_pump_head_m', 'booster_pump_power_kw', 'booster_pump_npsh_a_m'], 'pumps_booster.csv')
    save_csv(['hp_pump_flow_m3h', 'hp_pump_head_m', 'hp_pump_power_kw', 'hp_pump_npsh_a_m'], 'pumps_hp.csv')
    save_csv(['orv_m_LNG_tph', 'orv_T_out_C', 'orv_Q_W', 'orv_U_eff_WK'], 'orv.csv')
    # ... 此处可添加其他CSV的保存逻辑
    
    # 为了演示，我们先保存一个完整的数据文件
    df.to_csv(os.path.join(args.out, 'full_simulation_data.csv'), index=False, date_format='%Y-%m-%dT%H:%M:%S')
    print(f"\n仿真完成！数据已保存到 {args.out}")

if __name__ == "__main__":
    main()
