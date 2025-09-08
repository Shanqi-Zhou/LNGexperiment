import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BOGCompressorStage:
    """BOG压缩机单级模块
    
    实现单级离心压缩机的热力学模型
    """
    
    def __init__(self, stage_id: int, params: Dict[str, Any]):
        self.stage_id = stage_id
        self.pressure_ratio = params.get('pressure_ratio', 2.2)  # 级压比
        self.isentropic_efficiency = params.get('isentropic_efficiency', 0.72)  # 等熵效率
        self.mechanical_efficiency = params.get('mechanical_efficiency', 0.95)  # 机械效率
        
        # 气体物性参数
        self.gas_cp = 2200.0  # J/(kg·K)
        self.gas_cv = 1650.0  # J/(kg·K)
        self.gamma = self.gas_cp / self.gas_cv  # 比热比
        self.gas_constant = 518.0  # J/(kg·K)，甲烷近似
        
        # 运行状态
        self.inlet_pressure = 100.0  # kPa
        self.outlet_pressure = 220.0  # kPa
        self.inlet_temp = -150.0  # °C
        self.outlet_temp = -100.0  # °C
        self.mass_flow = 0.0  # kg/h
        self.power_consumption = 0.0  # kW
        
    def calculate_compression(self, mass_flow_kgh: float, inlet_pressure_kPa: float, 
                            inlet_temp_C: float) -> Dict[str, float]:
        """计算压缩过程
        
        Args:
            mass_flow_kgh: 质量流量 (kg/h)
            inlet_pressure_kPa: 入口压力 (kPa)
            inlet_temp_C: 入口温度 (°C)
            
        Returns:
            压缩结果
        """
        if mass_flow_kgh <= 0:
            return {
                'outlet_pressure_kPa': inlet_pressure_kPa,
                'outlet_temp_C': inlet_temp_C,
                'power_kw': 0.0,
                'efficiency': 0.0
            }
            
        # 温度转换为绝对温度
        T1 = inlet_temp_C + 273.15  # K
        P1 = inlet_pressure_kPa  # kPa
        P2 = P1 * self.pressure_ratio  # kPa
        
        # 等熵压缩温度
        T2s = T1 * (P2 / P1) ** ((self.gamma - 1) / self.gamma)
        
        # 实际压缩温度（考虑等熵效率）
        T2 = T1 + (T2s - T1) / self.isentropic_efficiency
        
        # 质量流量转换
        mass_flow_kgs = mass_flow_kgh / 3600  # kg/s
        
        # 压缩功计算
        # 等熵压缩功
        work_isentropic = mass_flow_kgs * self.gas_cp * (T2s - T1)  # W
        
        # 实际压缩功
        work_actual = work_isentropic / self.isentropic_efficiency  # W
        
        # 轴功率（考虑机械效率）
        shaft_power = work_actual / self.mechanical_efficiency / 1000  # kW
        
        # 更新状态
        self.inlet_pressure = P1
        self.outlet_pressure = P2
        self.inlet_temp = inlet_temp_C
        self.outlet_temp = T2 - 273.15  # 转换回摄氏度
        self.mass_flow = mass_flow_kgh
        self.power_consumption = shaft_power
        
        return {
            'outlet_pressure_kPa': P2,
            'outlet_temp_C': self.outlet_temp,
            'power_kw': shaft_power,
            'efficiency': self.isentropic_efficiency,
            'compression_ratio': self.pressure_ratio
        }
        
    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典"""
        return {
            'inlet_pressure': self.inlet_pressure,
            'outlet_pressure': self.outlet_pressure,
            'inlet_temp': self.inlet_temp,
            'outlet_temp': self.outlet_temp,
            'mass_flow': self.mass_flow,
            'power_consumption': self.power_consumption
        }


class BOGCompressorSystem:
    """BOG压缩机系统（2级压缩）
    
    实现2级离心压缩机系统，包括级间冷却
    """
    
    def __init__(self, params: Dict[str, Any]):
        # 系统参数
        self.num_stages = params.get('num_stages', 2)
        self.overall_pressure_ratio = params.get('overall_pressure_ratio', 4.84)  # 2.2^2
        self.intercooler_efficiency = params.get('intercooler_efficiency', 0.8)
        self.intercooler_temp = params.get('intercooler_temp', 15.0)  # °C
        
        # 创建压缩机级
        self.stages = []
        for i in range(self.num_stages):
            stage_params = params.get(f'stage_{i+1}', {})
            stage_params.setdefault('pressure_ratio', 2.2)
            stage_params.setdefault('isentropic_efficiency', 0.72)
            self.stages.append(BOGCompressorStage(i+1, stage_params))
            
        # 运行状态
        self.total_power = 0.0
        self.is_running = False
        self.suction_pressure = 110.0  # kPa
        self.discharge_pressure = 500.0  # kPa
        
    def calculate_intercooling(self, hot_temp_C: float, cooling_temp_C: float) -> float:
        """计算级间冷却效果
        
        Args:
            hot_temp_C: 热气体温度 (°C)
            cooling_temp_C: 冷却介质温度 (°C)
            
        Returns:
            冷却后温度 (°C)
        """
        # 冷却效率模型
        temp_drop = (hot_temp_C - cooling_temp_C) * self.intercooler_efficiency
        cooled_temp = hot_temp_C - temp_drop
        return max(cooling_temp_C, cooled_temp)
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, Any]:
        """仿真压缩机系统
        
        Args:
            inputs: 输入参数
                - bog_flow_kgh: BOG流量 (kg/h)
                - suction_pressure_kPa: 吸入压力 (kPa)
                - suction_temp_C: 吸入温度 (°C)
                - discharge_pressure_setpoint: 排出压力设定 (kPa)
                - start_command: 启动命令
                - ambient_temp_C: 环境温度 (°C)
            dt: 时间步长 (s)
            
        Returns:
            压缩机系统输出
        """
        bog_flow = inputs.get('bog_flow_kgh', 0.0)
        suction_pressure = inputs.get('suction_pressure_kPa', 110.0)
        suction_temp = inputs.get('suction_temp_C', -150.0)
        discharge_setpoint = inputs.get('discharge_pressure_setpoint', 500.0)
        start_command = inputs.get('start_command', False)
        ambient_temp = inputs.get('ambient_temp_C', 25.0)
        
        # 启停逻辑
        if start_command and bog_flow > 0:
            self.is_running = True
        elif bog_flow <= 0:
            self.is_running = False
            
        if not self.is_running:
            # 停机状态
            self.total_power = 0.0
            stage_outputs = []
            for stage in self.stages:
                stage_outputs.append({
                    'stage': stage.stage_id,
                    'p_in_kPa': 0.0,
                    'p_out_kPa': 0.0,
                    'm_kgph': 0.0,
                    'power_kw': 0.0,
                    'T_in_C': ambient_temp,
                    'T_out_C': ambient_temp
                })
        else:
            # 运行状态
            stage_outputs = []
            current_pressure = suction_pressure
            current_temp = suction_temp
            current_flow = bog_flow
            total_power = 0.0
            
            for i, stage in enumerate(self.stages):
                # 计算当前级压缩
                compression_result = stage.calculate_compression(
                    current_flow, current_pressure, current_temp
                )
                
                # 记录级输出
                stage_output = {
                    'stage': stage.stage_id,
                    'p_in_kPa': current_pressure,
                    'p_out_kPa': compression_result['outlet_pressure_kPa'],
                    'm_kgph': current_flow,
                    'power_kw': compression_result['power_kw'],
                    'T_in_C': current_temp,
                    'T_out_C': compression_result['outlet_temp_C'],
                    'efficiency': compression_result['efficiency']
                }
                stage_outputs.append(stage_output)
                
                # 累计功率
                total_power += compression_result['power_kw']
                
                # 更新下一级入口条件
                current_pressure = compression_result['outlet_pressure_kPa']
                current_temp = compression_result['outlet_temp_C']
                
                # 级间冷却（除了最后一级）
                if i < len(self.stages) - 1:
                    current_temp = self.calculate_intercooling(current_temp, ambient_temp)
                    
            self.total_power = total_power
            self.suction_pressure = suction_pressure
            self.discharge_pressure = current_pressure
            
        return {
            'stages': stage_outputs,
            'total_power_kw': self.total_power,
            'suction_pressure_kPa': self.suction_pressure,
            'discharge_pressure_kPa': self.discharge_pressure,
            'total_flow_kgh': bog_flow if self.is_running else 0.0,
            'is_running': self.is_running,
            'overall_compression_ratio': self.discharge_pressure / self.suction_pressure if self.suction_pressure > 0 else 1.0
        }
        
    def get_stage_data(self) -> List[Dict[str, float]]:
        """获取各级详细数据"""
        stage_data = []
        for stage in self.stages:
            stage_data.append({
                'stage_id': stage.stage_id,
                'pressure_ratio': stage.pressure_ratio,
                'efficiency': stage.isentropic_efficiency,
                **stage.get_state_dict()
            })
        return stage_data
        
    def inject_performance_degradation(self, stage_id: int, efficiency_factor: float = 0.9):
        """注入性能衰减故障
        
        Args:
            stage_id: 级号 (1-based)
            efficiency_factor: 效率衰减因子
        """
        if 1 <= stage_id <= len(self.stages):
            stage = self.stages[stage_id - 1]
            stage.isentropic_efficiency *= efficiency_factor
            stage.mechanical_efficiency *= efficiency_factor
            
    def reset_performance(self):
        """重置性能参数"""
        for stage in self.stages:
            stage.isentropic_efficiency = 0.72
            stage.mechanical_efficiency = 0.95
            
    def get_system_state(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'total_power': self.total_power,
            'is_running': self.is_running,
            'suction_pressure': self.suction_pressure,
            'discharge_pressure': self.discharge_pressure,
            'stages': self.get_stage_data()
        }