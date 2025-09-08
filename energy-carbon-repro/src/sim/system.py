import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import random

from .tank import LNGTank
from .pumps import PumpSystem
from .orv import ORVVaporizer
from .bog_compressor import BOGCompressorSystem
from .recondenser import SprayRecondenser

class LNGSystemSimulator:
    """LNG全链路系统仿真器
    
    集成储罐、泵、气化器、BOG压缩机、再冷凝器等子系统
    """
    
    def __init__(self, params: Dict[str, Any]):
        # 系统参数
        self.simulation_params = params.get('simulation', {})
        self.start_time = datetime.fromisoformat(self.simulation_params.get('start_time', '2024-01-01T00:00:00'))
        self.time_step = self.simulation_params.get('time_step', 1.0)  # s
        self.total_duration = self.simulation_params.get('duration_days', 180) * 24 * 3600  # s
        
        # 初始化子系统
        self.tank = LNGTank(params.get('tank', {}))
        self.pump_system = PumpSystem(params.get('pumps', {}))
        self.orv = ORVVaporizer(params.get('orv', {}))
        self.bog_compressor = BOGCompressorSystem(params.get('bog_compressor', {}))
        self.recondenser = SprayRecondenser(params.get('recondenser', {}))
        
        # 环境和操作参数
        self.ambient_temp_base = params.get('environment', {}).get('temp_base', 25.0)  # °C
        self.seawater_temp_base = params.get('environment', {}).get('seawater_temp', 15.0)  # °C
        self.export_pressure_setpoint = params.get('export', {}).get('pressure_setpoint', 6500.0)  # kPa
        
        # 运行日历
        self.ship_schedule = self._generate_ship_schedule()
        self.fault_schedule = self._generate_fault_schedule()
        
        # 状态变量
        self.current_time = self.start_time
        self.simulation_step = 0
        self.is_ship_unloading = False
        self.active_faults = set()
        
        # 数据记录
        self.history = []
        
        # 随机种子
        random.seed(2025)
        np.random.seed(2025)
        
    def _generate_ship_schedule(self) -> List[Dict[str, Any]]:
        """生成船舶靠泊时间表"""
        schedule = []
        current_day = 0
        
        while current_day < self.simulation_params.get('duration_days', 180):
            # 每10天一艘船
            arrival_day = current_day + 10
            if arrival_day < self.simulation_params.get('duration_days', 180):
                # 随机选择卸载时长（6-12小时）
                unloading_duration = random.uniform(6, 12)
                
                schedule.append({
                    'arrival_day': arrival_day,
                    'arrival_hour': random.uniform(6, 18),  # 白天到达
                    'unloading_duration_hours': unloading_duration,
                    'cargo_volume_m3': random.uniform(800, 1200)  # 缩放后货量
                })
                
            current_day = arrival_day
            
        return schedule
        
    def _generate_fault_schedule(self) -> List[Dict[str, Any]]:
        """生成故障注入时间表"""
        faults = [
            {
                'day': 60,
                'type': 'leak',
                'equipment': 'pipeline',
                'severity': 1.05,  # 能损增加5%
                'duration_hours': 24 * 30  # 持续30天
            },
            {
                'day': 120,
                'type': 'fouling_enhanced',
                'equipment': 'orv',
                'severity': 2.0,  # 结垢速率翻倍
                'duration_hours': 24 * 40  # 持续40天
            },
            {
                'day': 160,
                'type': 'cavitation',
                'equipment': 'hp_pump',
                'severity': 0.9,  # 效率降低10%
                'duration_hours': 24 * 20  # 持续20天
            }
        ]
        return faults
        
    def get_environmental_conditions(self, current_time: datetime) -> Dict[str, float]:
        """获取环境条件
        
        Args:
            current_time: 当前时间
            
        Returns:
            环境条件字典
        """
        # 季节性变化
        day_of_year = current_time.timetuple().tm_yday
        seasonal_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 春分为基准
        
        # 日周期变化
        hour_of_day = current_time.hour + current_time.minute / 60
        daily_factor = np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # 早6点最低
        
        # 环境温度
        seasonal_temp = self.ambient_temp_base + seasonal_factor * 10  # ±10°C季节变化
        ambient_temp = seasonal_temp + daily_factor * 4  # ±4°C日变化
        
        # 海水温度（滞后于环境温度）
        seawater_temp = self.seawater_temp_base + seasonal_factor * 8  # ±8°C季节变化
        
        # 湿度和风速
        humidity = 60 + seasonal_factor * 20 + np.random.normal(0, 5)
        wind_speed = 5 + np.abs(np.random.normal(0, 3))
        
        return {
            'T_amb_C': ambient_temp,
            'seawater_T_C': seawater_temp,
            'humidity_pct': max(20, min(95, humidity)),
            'wind_mps': max(0, wind_speed)
        }
        
    def get_export_demand(self, current_time: datetime) -> float:
        """获取外输需求
        
        Args:
            current_time: 当前时间
            
        Returns:
            外输流量需求 (Nm³/h)
        """
        # 基础需求
        base_demand = 1000.0  # Nm³/h
        
        # 日周期（早晚峰）
        hour = current_time.hour
        if 7 <= hour <= 9 or 18 <= hour <= 21:  # 早晚峰
            daily_factor = 1.6
        elif 22 <= hour <= 6:  # 夜间低谷
            daily_factor = 0.6
        else:  # 平时
            daily_factor = 1.0
            
        # 周末系数
        weekday = current_time.weekday()
        weekend_factor = 0.9 if weekday >= 5 else 1.0
        
        # 随机波动
        random_factor = 1 + np.random.normal(0, 0.05)
        
        demand = base_demand * daily_factor * weekend_factor * random_factor
        return max(0, demand)
        
    def check_ship_operations(self, current_time: datetime) -> Dict[str, Any]:
        """检查船舶操作状态
        
        Args:
            current_time: 当前时间
            
        Returns:
            船舶操作状态
        """
        current_day = (current_time - self.start_time).days
        current_hour = current_time.hour + current_time.minute / 60
        
        for ship in self.ship_schedule:
            arrival_time = ship['arrival_day'] * 24 + ship['arrival_hour']
            departure_time = arrival_time + ship['unloading_duration_hours']
            current_time_hours = current_day * 24 + current_hour
            
            if arrival_time <= current_time_hours <= departure_time:
                # 正在卸载
                unloading_rate = ship['cargo_volume_m3'] / ship['unloading_duration_hours']  # m³/h
                return {
                    'is_unloading': True,
                    'unloading_rate_m3h': unloading_rate,
                    'ship_id': f"Ship_{ship['arrival_day']}"
                }
                
        return {
            'is_unloading': False,
            'unloading_rate_m3h': 0.0,
            'ship_id': None
        }
        
    def check_fault_status(self, current_time: datetime) -> Dict[str, Any]:
        """检查故障状态
        
        Args:
            current_time: 当前时间
            
        Returns:
            故障状态字典
        """
        current_day = (current_time - self.start_time).days
        active_faults = {}
        
        for fault in self.fault_schedule:
            fault_start_day = fault['day']
            fault_end_day = fault_start_day + fault['duration_hours'] / 24
            
            if fault_start_day <= current_day <= fault_end_day:
                fault_key = f"{fault['equipment']}_{fault['type']}"
                active_faults[fault_key] = {
                    'type': fault['type'],
                    'equipment': fault['equipment'],
                    'severity': fault['severity'],
                    'start_day': fault_start_day,
                    'current_day': current_day
                }
                
        return active_faults
        
    def add_measurement_noise(self, value: float, noise_level: float = 0.005) -> float:
        """添加测量噪声
        
        Args:
            value: 原始值
            noise_level: 噪声水平（相对标准差）
            
        Returns:
            带噪声的测量值
        """
        if value == 0:
            return 0
            
        # 系统偏差（0.3%）
        bias = value * 0.003
        
        # 随机噪声（0.5%FS，假设FS为测量值的2倍）
        random_noise = np.random.normal(0, value * noise_level)
        
        return value + bias + random_noise
        
    def simulate_step(self, dt: float = None) -> Dict[str, Any]:
        """仿真一个时间步
        
        Args:
            dt: 时间步长 (s)，默认使用配置值
            
        Returns:
            仿真结果
        """
        if dt is None:
            dt = self.time_step
            
        # 获取环境条件
        env_conditions = self.get_environmental_conditions(self.current_time)
        
        # 获取外输需求
        export_demand = self.get_export_demand(self.current_time)
        
        # 检查船舶操作
        ship_status = self.check_ship_operations(self.current_time)
        
        # 检查故障状态
        fault_status = self.check_fault_status(self.current_time)
        
        # 储罐仿真
        tank_inputs = {
            'ambient_temp': env_conditions['T_amb_C'],
            'lng_input_m3h': ship_status['unloading_rate_m3h'],
            'lng_output_m3h': export_demand / 600,  # 简化转换
            'bog_removal_kgh': 0  # 将由BOG系统计算
        }
        tank_outputs = self.tank.simulate_step(tank_inputs, dt)
        
        # ORV气化器仿真
        orv_inputs = {
            'lng_flow_tph': tank_inputs['lng_output_m3h'] * 0.45,  # m³/h to t/h
            'lng_inlet_temp': -162.0,
            'seawater_temp': env_conditions['seawater_T_C'],
            'fouling_enhanced': fault_status.get('orv_fouling_enhanced', {}).get('severity', 1.0)
        }
        orv_outputs = self.orv.simulate_step(orv_inputs, dt)
        
        # BOG压缩机仿真
        bog_inputs = {
            'bog_flow_kgh': tank_outputs['bog_rate_kgph'],
            'suction_pressure_kPa': tank_outputs['p_top_kPa'],
            'suction_temp_C': -150.0,
            'discharge_pressure_setpoint': 500.0,
            'start_command': tank_outputs['bog_rate_kgph'] > 10,  # 自动启动
            'ambient_temp_C': env_conditions['T_amb_C']
        }
        bog_outputs = self.bog_compressor.simulate_step(bog_inputs, dt)
        
        # 再冷凝器仿真
        recondenser_inputs = {
            'bog_flow_kgh': bog_outputs['total_flow_kgh'],
            'bog_temp_C': -100.0,  # 压缩后温度
            'lng_spray_flow_tph': 0,  # 自动控制
            'lng_spray_temp_C': -162.0,
            'auto_spray_control': True
        }
        recondenser_outputs = self.recondenser.simulate_step(recondenser_inputs, dt)
        
        # 泵系统仿真
        pump_inputs = {
            'booster_flow_setpoint': tank_inputs['lng_output_m3h'] * 0.5,
            'hp_flow_setpoint': tank_inputs['lng_output_m3h'] * 0.5,
            'tank_pressure_kPa': tank_outputs['p_top_kPa'],
            'export_pressure_kPa': self.export_pressure_setpoint,
            'liquid_temp_C': -162.0,
            'booster_start': tank_inputs['lng_output_m3h'] > 0,
            'hp_start': tank_inputs['lng_output_m3h'] > 0
        }
        pump_outputs = self.pump_system.simulate_step(pump_inputs, dt)
        
        # 应用故障影响
        if 'hp_pump_cavitation' in fault_status:
            pump_outputs['hp']['efficiency'] *= fault_status['hp_pump_cavitation']['severity']
            pump_outputs['hp']['power_kw'] /= fault_status['hp_pump_cavitation']['severity']
            
        # 更新BOG移除量
        total_bog_removed = (recondenser_outputs['m_condensed_kgph'] + 
                           max(0, bog_outputs['total_flow_kgh'] - recondenser_outputs['m_bog_in_kgph']))
        tank_inputs['bog_removal_kgh'] = total_bog_removed
        
        # 添加测量噪声
        noisy_outputs = {}
        for key, value in tank_outputs.items():
            if isinstance(value, (int, float)):
                noisy_outputs[f'tank_{key}'] = self.add_measurement_noise(value)
                
        # 组装输出数据
        step_data = {
            'timestamp': self.current_time.isoformat(),
            'simulation_step': self.simulation_step,
            'environment': env_conditions,
            'tank': tank_outputs,
            'pumps': pump_outputs,
            'orv': orv_outputs,
            'bog_compressor': bog_outputs,
            'recondenser': recondenser_outputs,
            'ship_status': ship_status,
            'fault_status': fault_status,
            'export_demand_Nm3h': export_demand,
            'total_power_kw': (pump_outputs['total_power_kw'] + bog_outputs['total_power_kw']),
            'energy_MJ': (pump_outputs['total_power_kw'] + bog_outputs['total_power_kw']) * dt / 1000 * 3.6
        }
        
        # 更新时间和步数
        self.current_time += timedelta(seconds=dt)
        self.simulation_step += 1
        
        # 记录历史数据
        self.history.append(step_data)
        
        return step_data
        
    def run_simulation(self, progress_callback=None) -> pd.DataFrame:
        """运行完整仿真
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            仿真结果DataFrame
        """
        print(f"Starting LNG system simulation...")
        print(f"Duration: {self.total_duration/3600/24:.1f} days")
        print(f"Time step: {self.time_step} seconds")
        
        total_steps = int(self.total_duration / self.time_step)
        
        for step in range(total_steps):
            # 仿真一步
            self.simulate_step()
            
            # 进度报告
            if step % 1000 == 0 or step == total_steps - 1:
                progress = (step + 1) / total_steps * 100
                elapsed_days = (self.current_time - self.start_time).days
                print(f"Progress: {progress:.1f}% - Day {elapsed_days}")
                
                if progress_callback:
                    progress_callback(progress, elapsed_days)
                    
        print("Simulation completed!")
        
        # 转换为DataFrame
        return self.get_results_dataframe()
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """获取结果DataFrame"""
        if not self.history:
            return pd.DataFrame()
            
        # 展平嵌套字典
        flattened_data = []
        for record in self.history:
            flat_record = {'timestamp': record['timestamp']}
            
            for key, value in record.items():
                if key == 'timestamp':
                    continue
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            for subsubkey, subsubvalue in subvalue.items():
                                flat_record[f'{key}_{subkey}_{subsubkey}'] = subsubvalue
                        else:
                            flat_record[f'{key}_{subkey}'] = subvalue
                else:
                    flat_record[key] = value
                    
            flattened_data.append(flat_record)
            
        df = pd.DataFrame(flattened_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    def save_results_to_csv(self, output_dir: str):
        """保存结果到CSV文件
        
        Args:
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.get_results_dataframe()
        
        # 按照技术路线要求的格式保存
        csv_files = {
            'env_weather.csv': ['timestamp', 'environment_T_amb_C', 'environment_seawater_T_C', 
                               'environment_humidity_pct', 'environment_wind_mps'],
            'tanks.csv': ['timestamp', 'tank_level_pct', 'tank_p_top_kPa', 'tank_bog_rate_kgph'],
            'pumps_booster.csv': ['timestamp', 'pumps_booster_flow_m3h', 'pumps_booster_head_m', 
                                 'pumps_booster_power_kw', 'pumps_booster_npsh_a_m'],
            'pumps_hp.csv': ['timestamp', 'pumps_hp_flow_m3h', 'pumps_hp_head_m', 
                            'pumps_hp_power_kw', 'pumps_hp_npsh_a_m'],
            'orv.csv': ['timestamp', 'orv_m_LNG_tph', 'orv_T_out_C', 'orv_Q_MW', 'orv_U_eff_WK'],
            'export_meter.csv': ['timestamp', 'export_demand_Nm3h', 'export_pressure_kPa', 
                               'orv_T_out_C', 'energy_MJ']
        }
        
        for filename, columns in csv_files.items():
            # 选择存在的列
            available_columns = [col for col in columns if col in df.columns]
            if available_columns:
                subset_df = df[available_columns].copy()
                subset_df.to_csv(os.path.join(output_dir, filename), index=False)
                print(f"Saved {filename} with {len(subset_df)} records")
                
        print(f"All CSV files saved to {output_dir}")