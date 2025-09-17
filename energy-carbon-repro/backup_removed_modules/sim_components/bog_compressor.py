
import numpy as np

class BOGCompressor:
    """
    BOG（蒸发气）压缩机模型。
    模拟一个固定的多级离心式压缩机。
    """
    def __init__(self, 
                 num_stages=2, 
                 pressure_ratio_per_stage=2.2, 
                 isentropic_efficiency=0.72):
        """
        初始化BOG压缩机模型。

        Args:
            num_stages (int): 压缩机级数。
            pressure_ratio_per_stage (float): 每级的固定压比。
            isentropic_efficiency (float): 每级的等熵效率。
        """
        self.num_stages = num_stages
        self.pressure_ratio_per_stage = pressure_ratio_per_stage
        self.isentropic_efficiency = isentropic_efficiency

        # BOG (主要为甲烷) 的物性参数 (近似值)
        self.gas_constant_R = 518  # J/(kg·K) for Methane
        self.specific_heat_ratio_k = 1.31 # Methane

        # 状态变量
        self.total_power_kW = 0.0
        self.mass_flow_kgph = 0.0
        self.stage_data = []

    def update(self, mass_flow_kg_s, inlet_temp_K, inlet_pressure_kPa):
        """
        根据给定的流量和入口条件，计算总功耗和各级状态。

        Args:
            mass_flow_kg_s (float): BOG质量流量 (kg/s)。
            inlet_temp_K (float): BOG入口温度 (K)。通常接近LNG沸点，约111K。
            inlet_pressure_kPa (float): BOG入口压力 (kPa)。通常略高于储罐压力。
        """
        self.mass_flow_kgph = mass_flow_kg_s * 3600
        self.stage_data = []
        self.total_power_kW = 0.0

        if mass_flow_kg_s <= 0:
            return

        current_temp_K = inlet_temp_K
        current_pressure_kPa = inlet_pressure_kPa
        k = self.specific_heat_ratio_k
        eff = self.isentropic_efficiency

        for i in range(self.num_stages):
            # 计算等熵功 (W)
            # W_s = m * R * T_in * (k/(k-1)) * (pr^((k-1)/k) - 1)
            work_isentropic_W = (mass_flow_kg_s * self.gas_constant_R * current_temp_K * 
                                 (k / (k - 1)) * 
                                 (self.pressure_ratio_per_stage**((k - 1) / k) - 1))
            
            # 计算实际功 (W)
            work_actual_W = work_isentropic_W / eff
            self.total_power_kW += work_actual_W / 1000.0

            # 计算出口温度
            # T_out = T_in * (1 + (pr^((k-1)/k) - 1) / eff)
            outlet_temp_K = current_temp_K * (1 + (self.pressure_ratio_per_stage**((k - 1) / k) - 1) / eff)
            
            # 计算出口压力
            outlet_pressure_kPa = current_pressure_kPa * self.pressure_ratio_per_stage

            # 保存该级数据
            self.stage_data.append({
                'stage': i + 1,
                'p_in_kPa': current_pressure_kPa,
                'p_out_kPa': outlet_pressure_kPa,
                'm_kgph': self.mass_flow_kgph,
                'power_kw': work_actual_W / 1000.0
            })

            # 更新下一级的入口条件
            current_temp_K = outlet_temp_K
            current_pressure_kPa = outlet_pressure_kPa

    def get_state(self):
        """
        返回压缩机整体和各级的状态。
        """
        return {
            'total_power_kw': self.total_power_kW,
            'stages': self.stage_data
        }
