import numpy as np
import pandas as pd
from typing import Dict, Any

class LNGTank:
    """LNG储罐仿真模块
    
    实现全容式16万m³储罐的热力学模型，包括BOG生成、液位变化等
    """
    
    def __init__(self, params: Dict[str, Any]):
        # 储罐基本参数（缩放后）
        self.total_volume = params.get('total_volume', 1600)  # m³ (缩放后)
        self.vapor_fraction = params.get('vapor_fraction', 0.08)  # 顶空体积分数
        self.heat_leak_coeff = params.get('UA_tank', 85.0)  # W/K
        self.vap_enthalpy = params.get('delta_h_vap', 5.05e5)  # J/kg
        
        # LNG物性参数
        self.lng_density = 450.0  # kg/m³
        self.lng_cp = 3500.0  # J/(kg·K)
        self.vapor_density = 2.0  # kg/m³
        self.vapor_cp = 2200.0  # J/(kg·K)
        
        # 状态变量初始化
        self.liquid_level = 0.92  # 液位百分比
        self.vapor_pressure = 110.0  # kPa
        self.liquid_temp = -162.0  # °C
        self.vapor_temp = -160.0  # °C
        
        # 历史数据存储
        self.history = []
        
    def get_liquid_volume(self) -> float:
        """计算液体体积"""
        return self.total_volume * self.liquid_level
        
    def get_vapor_volume(self) -> float:
        """计算气体体积"""
        return self.total_volume * (1 - self.liquid_level)
        
    def get_liquid_mass(self) -> float:
        """计算液体质量"""
        return self.get_liquid_volume() * self.lng_density
        
    def get_vapor_mass(self) -> float:
        """计算气体质量"""
        return self.get_vapor_volume() * self.vapor_density
        
    def calculate_bog_rate(self, ambient_temp: float, dt: float) -> float:
        """计算BOG生成速率
        
        Args:
            ambient_temp: 环境温度 (°C)
            dt: 时间步长 (s)
            
        Returns:
            BOG生成速率 (kg/h)
        """
        # 温差驱动的热入侵
        temp_diff = ambient_temp - self.liquid_temp
        heat_input = self.heat_leak_coeff * temp_diff  # W
        
        # BOG生成量
        bog_mass_rate = heat_input / self.vap_enthalpy  # kg/s
        bog_rate_kgh = bog_mass_rate * 3600  # kg/h
        
        return max(0, bog_rate_kgh)
        
    def update_pressure(self, bog_rate_kgh: float, bog_removal_kgh: float, dt: float):
        """更新储罐压力
        
        Args:
            bog_rate_kgh: BOG生成速率 (kg/h)
            bog_removal_kgh: BOG移除速率 (kg/h)
            dt: 时间步长 (s)
        """
        # 净BOG累积
        net_bog_kgs = (bog_rate_kgh - bog_removal_kgh) * dt / 3600
        
        # 压力变化（简化理想气体模型）
        vapor_volume = self.get_vapor_volume()
        if vapor_volume > 0:
            # 压力变化与质量变化成正比
            pressure_change = net_bog_kgs * 8.314 * (self.vapor_temp + 273.15) / (vapor_volume * 0.016)  # kPa
            self.vapor_pressure += pressure_change
            
        # 压力限制
        self.vapor_pressure = max(100.0, min(150.0, self.vapor_pressure))
        
    def update_level(self, lng_input_m3h: float, lng_output_m3h: float, 
                    bog_rate_kgh: float, dt: float):
        """更新液位
        
        Args:
            lng_input_m3h: LNG进料速率 (m³/h)
            lng_output_m3h: LNG出料速率 (m³/h)
            bog_rate_kgh: BOG生成速率 (kg/h)
            dt: 时间步长 (s)
        """
        # 体积变化
        volume_change_m3 = (lng_input_m3h - lng_output_m3h) * dt / 3600
        
        # BOG导致的液体损失
        bog_volume_loss = (bog_rate_kgh * dt / 3600) / self.lng_density
        
        # 更新液位
        new_liquid_volume = self.get_liquid_volume() + volume_change_m3 - bog_volume_loss
        self.liquid_level = max(0.05, min(0.95, new_liquid_volume / self.total_volume))
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """仿真一个时间步
        
        Args:
            inputs: 输入参数字典
                - ambient_temp: 环境温度 (°C)
                - lng_input_m3h: LNG进料速率 (m³/h)
                - lng_output_m3h: LNG出料速率 (m³/h)
                - bog_removal_kgh: BOG移除速率 (kg/h)
            dt: 时间步长 (s)
            
        Returns:
            输出状态字典
        """
        ambient_temp = inputs.get('ambient_temp', 25.0)
        lng_input = inputs.get('lng_input_m3h', 0.0)
        lng_output = inputs.get('lng_output_m3h', 0.0)
        bog_removal = inputs.get('bog_removal_kgh', 0.0)
        
        # 计算BOG生成
        bog_rate = self.calculate_bog_rate(ambient_temp, dt)
        
        # 更新状态
        self.update_pressure(bog_rate, bog_removal, dt)
        self.update_level(lng_input, lng_output, bog_rate, dt)
        
        # 输出状态
        outputs = {
            'level_pct': self.liquid_level * 100,
            'p_top_kPa': self.vapor_pressure,
            'bog_rate_kgph': bog_rate,
            'liquid_volume_m3': self.get_liquid_volume(),
            'vapor_volume_m3': self.get_vapor_volume(),
            'liquid_mass_kg': self.get_liquid_mass(),
            'vapor_mass_kg': self.get_vapor_mass()
        }
        
        return outputs
        
    def get_state_dict(self) -> Dict[str, float]:
        """获取当前状态字典"""
        return {
            'liquid_level': self.liquid_level,
            'vapor_pressure': self.vapor_pressure,
            'liquid_temp': self.liquid_temp,
            'vapor_temp': self.vapor_temp
        }
        
    def set_state_dict(self, state: Dict[str, float]):
        """设置状态字典"""
        self.liquid_level = state.get('liquid_level', self.liquid_level)
        self.vapor_pressure = state.get('vapor_pressure', self.vapor_pressure)
        self.liquid_temp = state.get('liquid_temp', self.liquid_temp)
        self.vapor_temp = state.get('vapor_temp', self.vapor_temp)