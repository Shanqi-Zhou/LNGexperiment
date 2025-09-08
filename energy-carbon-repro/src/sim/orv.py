import numpy as np
import pandas as pd
from typing import Dict, Any

class ORVVaporizer:
    """ORV海水浴式气化器仿真模块
    
    实现海水与LNG之间的传热过程，包括结垢效应和传热衰减
    """
    
    def __init__(self, params: Dict[str, Any]):
        # 传热参数
        self.UA_initial = params.get('UA_orv', 4.0e6)  # W/K，初始传热系数×面积
        self.UA_current = self.UA_initial
        self.fouling_rate = params.get('fouling_rate', 3.0e-6)  # s^-1，结垢衰减系数
        self.fouling_enhanced = False  # 结垢增强标志
        
        # 设计参数
        self.design_flow = params.get('design_flow', 100.0)  # t/h，设计流量
        self.design_temp_rise = params.get('design_temp_rise', 150.0)  # K，设计温升
        
        # 海水参数
        self.seawater_cp = 4180.0  # J/(kg·K)
        self.seawater_density = 1025.0  # kg/m³
        self.seawater_flow = params.get('seawater_flow', 2000.0)  # m³/h
        
        # LNG参数
        self.lng_cp_liquid = 3500.0  # J/(kg·K)
        self.lng_cp_vapor = 2200.0  # J/(kg·K)
        self.lng_density = 450.0  # kg/m³
        self.lng_vap_enthalpy = 5.05e5  # J/kg
        self.lng_boiling_point = -162.0  # °C
        
        # 运行状态
        self.inlet_temp = -162.0  # °C
        self.outlet_temp = -100.0  # °C
        self.heat_duty = 0.0  # W
        self.seawater_temp_drop = 0.0  # K
        
        # 历史数据
        self.operating_hours = 0.0
        
    def update_fouling(self, dt: float, enhanced_factor: float = 1.0):
        """更新结垢效应
        
        Args:
            dt: 时间步长 (s)
            enhanced_factor: 结垢增强因子（故障注入时使用）
        """
        # 结垢导致的传热系数衰减
        fouling_decay = self.fouling_rate * enhanced_factor * dt
        self.UA_current *= (1 - fouling_decay)
        
        # 限制最小传热系数
        min_UA = self.UA_initial * 0.3
        self.UA_current = max(min_UA, self.UA_current)
        
        # 更新运行时间
        self.operating_hours += dt / 3600
        
    def calculate_heat_transfer(self, lng_flow_tph: float, lng_inlet_temp: float, 
                              seawater_temp: float) -> Dict[str, float]:
        """计算传热过程
        
        Args:
            lng_flow_tph: LNG流量 (t/h)
            lng_inlet_temp: LNG入口温度 (°C)
            seawater_temp: 海水温度 (°C)
            
        Returns:
            传热计算结果
        """
        if lng_flow_tph <= 0:
            return {
                'lng_outlet_temp': lng_inlet_temp,
                'heat_duty_MW': 0.0,
                'seawater_temp_drop': 0.0,
                'effectiveness': 0.0
            }
            
        # LNG质量流量
        lng_mass_flow = lng_flow_tph * 1000 / 3600  # kg/s
        
        # 海水质量流量
        seawater_mass_flow = self.seawater_flow * self.seawater_density / 3600  # kg/s
        
        # 热容流率
        lng_heat_capacity_rate = lng_mass_flow * self.lng_cp_liquid  # W/K
        seawater_heat_capacity_rate = seawater_mass_flow * self.seawater_cp  # W/K
        
        # 最小热容流率
        C_min = min(lng_heat_capacity_rate, seawater_heat_capacity_rate)
        C_max = max(lng_heat_capacity_rate, seawater_heat_capacity_rate)
        C_ratio = C_min / C_max
        
        # NTU计算
        NTU = self.UA_current / C_min
        
        # 换热器效率（逆流换热器）
        if C_ratio < 1.0:
            effectiveness = (1 - np.exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * np.exp(-NTU * (1 - C_ratio)))
        else:
            effectiveness = NTU / (1 + NTU)
            
        # 最大可能传热量
        max_temp_diff = seawater_temp - lng_inlet_temp
        Q_max = C_min * max_temp_diff  # W
        
        # 实际传热量
        Q_actual = effectiveness * Q_max  # W
        
        # LNG出口温度
        lng_temp_rise = Q_actual / lng_heat_capacity_rate
        lng_outlet_temp = lng_inlet_temp + lng_temp_rise
        
        # 海水温降
        seawater_temp_drop = Q_actual / seawater_heat_capacity_rate
        
        # 检查是否发生相变
        if lng_outlet_temp > self.lng_boiling_point:
            # 部分气化
            sensible_heat = lng_mass_flow * self.lng_cp_liquid * (self.lng_boiling_point - lng_inlet_temp)
            remaining_heat = Q_actual - sensible_heat
            
            if remaining_heat > 0:
                # 计算气化量
                vaporized_fraction = remaining_heat / (lng_mass_flow * self.lng_vap_enthalpy)
                vaporized_fraction = min(1.0, vaporized_fraction)
                
                # 混合温度（简化处理）
                lng_outlet_temp = self.lng_boiling_point + remaining_heat / (lng_mass_flow * self.lng_cp_vapor)
            else:
                vaporized_fraction = 0.0
        else:
            vaporized_fraction = 0.0
            
        return {
            'lng_outlet_temp': lng_outlet_temp,
            'heat_duty_MW': Q_actual / 1e6,
            'seawater_temp_drop': seawater_temp_drop,
            'effectiveness': effectiveness,
            'vaporized_fraction': vaporized_fraction,
            'UA_current': self.UA_current
        }
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """仿真一个时间步
        
        Args:
            inputs: 输入参数
                - lng_flow_tph: LNG流量 (t/h)
                - lng_inlet_temp: LNG入口温度 (°C)
                - seawater_temp: 海水温度 (°C)
                - fouling_enhanced: 结垢增强因子
            dt: 时间步长 (s)
            
        Returns:
            输出状态
        """
        lng_flow = inputs.get('lng_flow_tph', 0.0)
        lng_inlet_temp = inputs.get('lng_inlet_temp', -162.0)
        seawater_temp = inputs.get('seawater_temp', 15.0)
        fouling_factor = inputs.get('fouling_enhanced', 1.0)
        
        # 更新结垢
        self.update_fouling(dt, fouling_factor)
        
        # 计算传热
        heat_transfer_results = self.calculate_heat_transfer(lng_flow, lng_inlet_temp, seawater_temp)
        
        # 更新状态
        self.inlet_temp = lng_inlet_temp
        self.outlet_temp = heat_transfer_results['lng_outlet_temp']
        self.heat_duty = heat_transfer_results['heat_duty_MW'] * 1e6  # W
        self.seawater_temp_drop = heat_transfer_results['seawater_temp_drop']
        
        return {
            'm_LNG_tph': lng_flow,
            'T_in_C': lng_inlet_temp,
            'T_out_C': self.outlet_temp,
            'Q_MW': heat_transfer_results['heat_duty_MW'],
            'U_eff_WK': self.UA_current,
            'effectiveness': heat_transfer_results['effectiveness'],
            'vaporized_fraction': heat_transfer_results['vaporized_fraction'],
            'seawater_temp_drop': self.seawater_temp_drop,
            'operating_hours': self.operating_hours
        }
        
    def inject_fouling_fault(self, enhancement_factor: float = 2.0):
        """注入结垢故障
        
        Args:
            enhancement_factor: 结垢增强因子
        """
        self.fouling_enhanced = True
        self.fouling_rate *= enhancement_factor
        
    def get_performance_degradation(self) -> float:
        """获取性能衰减百分比
        
        Returns:
            性能衰减百分比 (0-100)
        """
        degradation = (1 - self.UA_current / self.UA_initial) * 100
        return max(0, degradation)
        
    def reset_fouling(self):
        """重置结垢（清洗后）"""
        self.UA_current = self.UA_initial * 0.95  # 清洗后恢复95%
        self.fouling_enhanced = False
        
    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典"""
        return {
            'UA_current': self.UA_current,
            'operating_hours': self.operating_hours,
            'inlet_temp': self.inlet_temp,
            'outlet_temp': self.outlet_temp,
            'heat_duty': self.heat_duty
        }
        
    def set_state_dict(self, state: Dict[str, float]):
        """设置状态字典"""
        self.UA_current = state.get('UA_current', self.UA_current)
        self.operating_hours = state.get('operating_hours', self.operating_hours)
        self.inlet_temp = state.get('inlet_temp', self.inlet_temp)
        self.outlet_temp = state.get('outlet_temp', self.outlet_temp)
        self.heat_duty = state.get('heat_duty', self.heat_duty)