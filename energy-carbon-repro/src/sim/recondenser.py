import numpy as np
import pandas as pd
from typing import Dict, Any

class SprayRecondenser:
    """喷淋式再冷凝器仿真模块
    
    实现BOG与LNG液体的直接接触冷凝过程
    """
    
    def __init__(self, params: Dict[str, Any]):
        # 设计参数
        self.design_capacity = params.get('design_capacity', 8.0)  # t/h，设计冷凝能力
        self.spray_efficiency = params.get('spray_efficiency', 0.85)  # 喷淋效率
        self.heat_transfer_coeff = params.get('heat_transfer_coeff', 2000.0)  # W/(m²·K)
        self.contact_area = params.get('contact_area', 100.0)  # m²，接触面积
        
        # 物性参数
        self.lng_density = 450.0  # kg/m³
        self.lng_cp = 3500.0  # J/(kg·K)
        self.bog_density = 2.0  # kg/m³
        self.bog_cp = 2200.0  # J/(kg·K)
        self.condensation_enthalpy = 5.05e5  # J/kg
        
        # 操作参数
        self.spray_ratio = params.get('spray_ratio', 10.0)  # LNG喷淋量与BOG量的比值
        self.min_spray_temp = -162.0  # °C，最低喷淋温度
        self.max_condensation_temp = -150.0  # °C，最高冷凝温度
        
        # 运行状态
        self.bog_inlet_temp = -140.0  # °C
        self.lng_spray_temp = -162.0  # °C
        self.condensed_temp = -160.0  # °C
        self.actual_capacity = 0.0  # t/h
        self.spray_flow = 0.0  # t/h
        self.heat_duty = 0.0  # MW
        
        # 性能参数
        self.fouling_factor = 1.0  # 污垢系数
        self.is_operating = False
        
    def calculate_condensation(self, bog_flow_kgh: float, bog_temp_C: float, 
                             lng_spray_flow_tph: float, lng_spray_temp_C: float) -> Dict[str, float]:
        """计算冷凝过程
        
        Args:
            bog_flow_kgh: BOG流量 (kg/h)
            bog_temp_C: BOG温度 (°C)
            lng_spray_flow_tph: LNG喷淋流量 (t/h)
            lng_spray_temp_C: LNG喷淋温度 (°C)
            
        Returns:
            冷凝计算结果
        """
        if bog_flow_kgh <= 0 or lng_spray_flow_tph <= 0:
            return {
                'condensed_flow_kgh': 0.0,
                'uncondensed_flow_kgh': bog_flow_kgh,
                'mixed_temp_C': bog_temp_C,
                'heat_duty_MW': 0.0,
                'condensation_efficiency': 0.0
            }
            
        # 质量流量转换
        bog_mass_flow = bog_flow_kgh / 3600  # kg/s
        lng_mass_flow = lng_spray_flow_tph * 1000 / 3600  # kg/s
        
        # 热容流率
        bog_heat_capacity = bog_mass_flow * self.bog_cp  # W/K
        lng_heat_capacity = lng_mass_flow * self.lng_cp  # W/K
        
        # 温差
        temp_diff = bog_temp_C - lng_spray_temp_C
        
        # 最大可能传热量（BOG完全冷却到LNG温度）
        max_sensible_heat = bog_heat_capacity * temp_diff  # W
        
        # 可用冷却能力（LNG可吸收的热量）
        lng_heating_capacity = lng_heat_capacity * (self.max_condensation_temp - lng_spray_temp_C)  # W
        
        # 实际传热量
        available_heat_transfer = min(max_sensible_heat, lng_heating_capacity)
        
        # 考虑传热效率和污垢系数
        effective_heat_transfer = available_heat_transfer * self.spray_efficiency * self.fouling_factor
        
        # 计算可冷凝的BOG量
        if temp_diff > 0:
            # BOG冷却到冷凝温度所需的显热
            sensible_cooling = bog_mass_flow * self.bog_cp * (bog_temp_C - self.max_condensation_temp)
            
            if effective_heat_transfer >= sensible_cooling:
                # 有足够冷量进行冷凝
                remaining_heat = effective_heat_transfer - sensible_cooling
                condensable_mass_flow = remaining_heat / self.condensation_enthalpy  # kg/s
                condensable_mass_flow = min(condensable_mass_flow, bog_mass_flow)
            else:
                # 只能部分冷却，无法冷凝
                condensable_mass_flow = 0.0
        else:
            condensable_mass_flow = 0.0
            
        # 冷凝效率
        condensation_efficiency = condensable_mass_flow / bog_mass_flow if bog_mass_flow > 0 else 0.0
        
        # 未冷凝的BOG
        uncondensed_mass_flow = bog_mass_flow - condensable_mass_flow
        
        # 混合温度计算（能量平衡）
        if condensable_mass_flow > 0:
            # 冷凝液温度接近LNG温度
            condensed_temp = lng_spray_temp_C + 2.0  # 略高于喷淋温度
            
            # 未冷凝气体温度
            if uncondensed_mass_flow > 0:
                uncondensed_temp = bog_temp_C - effective_heat_transfer / (uncondensed_mass_flow * self.bog_cp)
                uncondensed_temp = max(lng_spray_temp_C, uncondensed_temp)
            else:
                uncondensed_temp = condensed_temp
                
            # 混合温度（质量加权平均）
            total_liquid_mass = condensable_mass_flow + lng_mass_flow
            mixed_temp = (condensable_mass_flow * condensed_temp + lng_mass_flow * lng_spray_temp_C) / total_liquid_mass
        else:
            # 无冷凝，只是气体冷却
            mixed_temp = bog_temp_C - effective_heat_transfer / bog_heat_capacity
            mixed_temp = max(lng_spray_temp_C, mixed_temp)
            
        return {
            'condensed_flow_kgh': condensable_mass_flow * 3600,
            'uncondensed_flow_kgh': uncondensed_mass_flow * 3600,
            'mixed_temp_C': mixed_temp,
            'heat_duty_MW': effective_heat_transfer / 1e6,
            'condensation_efficiency': condensation_efficiency,
            'lng_temp_rise': effective_heat_transfer / lng_heat_capacity if lng_heat_capacity > 0 else 0.0
        }
        
    def calculate_required_spray_flow(self, bog_flow_kgh: float, bog_temp_C: float, 
                                    target_condensation_rate: float = 0.95) -> float:
        """计算所需的喷淋流量
        
        Args:
            bog_flow_kgh: BOG流量 (kg/h)
            bog_temp_C: BOG温度 (°C)
            target_condensation_rate: 目标冷凝率
            
        Returns:
            所需LNG喷淋流量 (t/h)
        """
        if bog_flow_kgh <= 0:
            return 0.0
            
        # 目标冷凝量
        target_condensed_kgh = bog_flow_kgh * target_condensation_rate
        
        # 所需冷量
        sensible_heat = bog_flow_kgh / 3600 * self.bog_cp * (bog_temp_C - self.max_condensation_temp)
        latent_heat = target_condensed_kgh / 3600 * self.condensation_enthalpy
        total_heat_required = sensible_heat + latent_heat  # W
        
        # 考虑效率损失
        actual_heat_required = total_heat_required / (self.spray_efficiency * self.fouling_factor)
        
        # LNG可提供的冷量
        lng_cooling_capacity_per_kg = self.lng_cp * (self.max_condensation_temp - self.min_spray_temp)
        
        # 所需LNG流量
        required_lng_kgs = actual_heat_required / lng_cooling_capacity_per_kg
        required_lng_tph = required_lng_kgs * 3600 / 1000
        
        return max(0, required_lng_tph)
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """仿真一个时间步
        
        Args:
            inputs: 输入参数
                - bog_flow_kgh: BOG流量 (kg/h)
                - bog_temp_C: BOG温度 (°C)
                - lng_spray_flow_tph: LNG喷淋流量 (t/h)
                - lng_spray_temp_C: LNG喷淋温度 (°C)
                - auto_spray_control: 自动喷淋控制
            dt: 时间步长 (s)
            
        Returns:
            输出状态
        """
        bog_flow = inputs.get('bog_flow_kgh', 0.0)
        bog_temp = inputs.get('bog_temp_C', -140.0)
        lng_spray_flow = inputs.get('lng_spray_flow_tph', 0.0)
        lng_spray_temp = inputs.get('lng_spray_temp_C', -162.0)
        auto_control = inputs.get('auto_spray_control', True)
        
        # 自动控制喷淋流量
        if auto_control and bog_flow > 0:
            required_spray = self.calculate_required_spray_flow(bog_flow, bog_temp)
            lng_spray_flow = min(required_spray, self.design_capacity * 2)  # 限制最大喷淋量
            
        # 运行状态判断
        self.is_operating = bog_flow > 0 and lng_spray_flow > 0
        
        if self.is_operating:
            # 计算冷凝过程
            condensation_result = self.calculate_condensation(
                bog_flow, bog_temp, lng_spray_flow, lng_spray_temp
            )
            
            # 更新状态
            self.bog_inlet_temp = bog_temp
            self.lng_spray_temp = lng_spray_temp
            self.actual_capacity = condensation_result['condensed_flow_kgh'] / 1000  # t/h
            self.spray_flow = lng_spray_flow
            self.heat_duty = condensation_result['heat_duty_MW']
            self.condensed_temp = condensation_result['mixed_temp_C']
            
            # 检查是否超出设计能力
            capacity_utilization = self.actual_capacity / self.design_capacity
            
            return {
                'm_bog_in_kgph': bog_flow,
                'm_liq_in_tph': lng_spray_flow,
                'm_condensed_kgph': condensation_result['condensed_flow_kgh'],
                'm_uncondensed_kgph': condensation_result['uncondensed_flow_kgh'],
                'T_mixed_C': condensation_result['mixed_temp_C'],
                'Q_MW': condensation_result['heat_duty_MW'],
                'condensation_efficiency': condensation_result['condensation_efficiency'],
                'capacity_utilization': capacity_utilization,
                'is_operating': self.is_operating,
                'lng_temp_rise': condensation_result['lng_temp_rise']
            }
        else:
            # 停运状态
            self.actual_capacity = 0.0
            self.spray_flow = 0.0
            self.heat_duty = 0.0
            
            return {
                'm_bog_in_kgph': bog_flow,
                'm_liq_in_tph': 0.0,
                'm_condensed_kgph': 0.0,
                'm_uncondensed_kgph': bog_flow,  # 全部送火炬
                'T_mixed_C': bog_temp,
                'Q_MW': 0.0,
                'condensation_efficiency': 0.0,
                'capacity_utilization': 0.0,
                'is_operating': False,
                'lng_temp_rise': 0.0
            }
            
    def inject_fouling_fault(self, fouling_factor: float = 0.8):
        """注入污垢故障
        
        Args:
            fouling_factor: 污垢系数（<1表示性能下降）
        """
        self.fouling_factor = fouling_factor
        
    def clean_heat_exchanger(self):
        """清洗换热器（恢复性能）"""
        self.fouling_factor = 1.0
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'design_capacity_tph': self.design_capacity,
            'actual_capacity_tph': self.actual_capacity,
            'capacity_utilization': self.actual_capacity / self.design_capacity if self.design_capacity > 0 else 0.0,
            'spray_efficiency': self.spray_efficiency,
            'fouling_factor': self.fouling_factor,
            'heat_duty_MW': self.heat_duty
        }
        
    def get_state_dict(self) -> Dict[str, float]:
        """获取状态字典"""
        return {
            'bog_inlet_temp': self.bog_inlet_temp,
            'lng_spray_temp': self.lng_spray_temp,
            'condensed_temp': self.condensed_temp,
            'actual_capacity': self.actual_capacity,
            'spray_flow': self.spray_flow,
            'heat_duty': self.heat_duty,
            'fouling_factor': self.fouling_factor,
            'is_operating': self.is_operating
        }
        
    def set_state_dict(self, state: Dict[str, float]):
        """设置状态字典"""
        self.bog_inlet_temp = state.get('bog_inlet_temp', self.bog_inlet_temp)
        self.lng_spray_temp = state.get('lng_spray_temp', self.lng_spray_temp)
        self.condensed_temp = state.get('condensed_temp', self.condensed_temp)
        self.actual_capacity = state.get('actual_capacity', self.actual_capacity)
        self.spray_flow = state.get('spray_flow', self.spray_flow)
        self.heat_duty = state.get('heat_duty', self.heat_duty)
        self.fouling_factor = state.get('fouling_factor', self.fouling_factor)
        self.is_operating = state.get('is_operating', self.is_operating)