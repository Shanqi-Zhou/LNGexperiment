import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class LNGPump:
    """LNG泵仿真基类
    
    实现泵的性能曲线、效率计算和功率消耗模型
    """
    
    def __init__(self, pump_type: str, params: Dict[str, Any]):
        self.pump_type = pump_type
        self.rated_power = params.get('rated_power', 1000)  # kW
        self.rated_flow = params.get('rated_flow', 100)  # m³/h
        self.rated_head = params.get('rated_head', 100)  # m
        self.min_efficiency = params.get('min_efficiency', 0.65)
        self.max_efficiency = params.get('max_efficiency', 0.8)
        
        # NPSH参数
        self.npsh_required = params.get('npsh_required', 3.0)  # m
        self.cavitation_factor = params.get('cavitation_factor', 0.1)
        
        # 运行状态
        self.is_running = False
        self.current_flow = 0.0
        self.current_head = 0.0
        self.current_power = 0.0
        self.current_efficiency = 0.0
        self.npsh_available = 10.0
        
    def calculate_performance_curve(self, flow_m3h: float) -> Tuple[float, float]:
        """计算泵的性能曲线
        
        Args:
            flow_m3h: 流量 (m³/h)
            
        Returns:
            (扬程, 效率)
        """
        if flow_m3h <= 0:
            return 0.0, 0.0
            
        # 流量比
        flow_ratio = flow_m3h / self.rated_flow
        
        # 扬程曲线（二次多项式）
        head = self.rated_head * (1.2 - 0.3 * flow_ratio - 0.1 * flow_ratio**2)
        head = max(0, head)
        
        # 效率曲线（在额定点附近最高）
        efficiency_curve = -(flow_ratio - 1.0)**2 + 1.0
        efficiency = self.min_efficiency + (self.max_efficiency - self.min_efficiency) * max(0, efficiency_curve)
        
        return head, efficiency
        
    def calculate_npsh_effect(self, npsh_available: float) -> float:
        """计算NPSH对效率的影响
        
        Args:
            npsh_available: 有效NPSH (m)
            
        Returns:
            效率修正系数
        """
        if npsh_available >= self.npsh_required * 1.5:
            return 1.0
        elif npsh_available >= self.npsh_required:
            # 线性衰减
            ratio = (npsh_available - self.npsh_required) / (self.npsh_required * 0.5)
            return 1.0 - self.cavitation_factor * (1 - ratio)
        else:
            # 严重汽蚀
            return 1.0 - self.cavitation_factor * 2
            
    def calculate_power(self, flow_m3h: float, head_m: float, efficiency: float) -> float:
        """计算功率消耗
        
        Args:
            flow_m3h: 流量 (m³/h)
            head_m: 扬程 (m)
            efficiency: 效率
            
        Returns:
            功率 (kW)
        """
        if flow_m3h <= 0 or efficiency <= 0:
            return 0.0
            
        # 液体密度（LNG）
        density = 450.0  # kg/m³
        gravity = 9.81  # m/s²
        
        # 水力功率
        flow_m3s = flow_m3h / 3600
        hydraulic_power = density * gravity * flow_m3s * head_m / 1000  # kW
        
        # 轴功率
        shaft_power = hydraulic_power / efficiency
        
        return min(shaft_power, self.rated_power * 1.1)  # 限制最大功率
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, float]:
        """仿真一个时间步
        
        Args:
            inputs: 输入参数
                - flow_setpoint_m3h: 流量设定值 (m³/h)
                - suction_pressure_kPa: 吸入压力 (kPa)
                - discharge_pressure_kPa: 排出压力 (kPa)
                - liquid_temp_C: 液体温度 (°C)
                - start_command: 启动命令 (bool)
            dt: 时间步长 (s)
            
        Returns:
            输出状态
        """
        flow_setpoint = inputs.get('flow_setpoint_m3h', 0.0)
        suction_pressure = inputs.get('suction_pressure_kPa', 100.0)
        discharge_pressure = inputs.get('discharge_pressure_kPa', 200.0)
        liquid_temp = inputs.get('liquid_temp_C', -162.0)
        start_command = inputs.get('start_command', False)
        
        # 启停逻辑
        if start_command and flow_setpoint > 0:
            self.is_running = True
        elif flow_setpoint <= 0:
            self.is_running = False
            
        if not self.is_running:
            self.current_flow = 0.0
            self.current_head = 0.0
            self.current_power = 0.0
            self.current_efficiency = 0.0
        else:
            # 计算实际流量（考虑系统阻力）
            self.current_flow = flow_setpoint
            
            # 计算性能
            head, efficiency = self.calculate_performance_curve(self.current_flow)
            
            # 计算NPSH
            self.npsh_available = self._calculate_npsh_available(suction_pressure, liquid_temp)
            npsh_factor = self.calculate_npsh_effect(self.npsh_available)
            
            # 修正效率
            self.current_efficiency = efficiency * npsh_factor
            self.current_head = head
            
            # 计算功率
            self.current_power = self.calculate_power(self.current_flow, self.current_head, self.current_efficiency)
            
        return {
            'flow_m3h': self.current_flow,
            'head_m': self.current_head,
            'power_kw': self.current_power,
            'efficiency': self.current_efficiency,
            'npsh_a_m': self.npsh_available,
            'is_running': self.is_running
        }
        
    def _calculate_npsh_available(self, suction_pressure_kPa: float, liquid_temp_C: float) -> float:
        """计算有效NPSH
        
        Args:
            suction_pressure_kPa: 吸入压力 (kPa)
            liquid_temp_C: 液体温度 (°C)
            
        Returns:
            有效NPSH (m)
        """
        # 简化计算，实际应考虑饱和蒸汽压
        vapor_pressure_kPa = 101.3 * np.exp((liquid_temp_C + 162) / 20)  # 简化蒸汽压曲线
        
        # NPSH = (P_suction - P_vapor) / (ρ * g) + elevation
        density = 450.0  # kg/m³
        gravity = 9.81  # m/s²
        
        npsh = (suction_pressure_kPa - vapor_pressure_kPa) * 1000 / (density * gravity)
        return max(0, npsh)


class BoosterPump(LNGPump):
    """Booster泵（低压泵）"""
    
    def __init__(self, params: Dict[str, Any]):
        # 默认参数
        default_params = {
            'rated_power': 220,  # kW
            'rated_flow': 150,   # m³/h
            'rated_head': 50,    # m
            'min_efficiency': 0.65,
            'max_efficiency': 0.8,
            'npsh_required': 2.5
        }
        default_params.update(params)
        super().__init__('booster', default_params)


class HighPressurePump(LNGPump):
    """高压泵（HP泵）"""
    
    def __init__(self, params: Dict[str, Any]):
        # 默认参数
        default_params = {
            'rated_power': 1200,  # kW
            'rated_flow': 200,    # m³/h
            'rated_head': 800,    # m
            'min_efficiency': 0.65,
            'max_efficiency': 0.8,
            'npsh_required': 4.0
        }
        default_params.update(params)
        super().__init__('high_pressure', default_params)
        
    def calculate_performance_curve(self, flow_m3h: float) -> Tuple[float, float]:
        """高压泵的性能曲线（更陡峭）"""
        if flow_m3h <= 0:
            return 0.0, 0.0
            
        flow_ratio = flow_m3h / self.rated_flow
        
        # 高压泵扬程曲线更陡峭
        head = self.rated_head * (1.3 - 0.5 * flow_ratio - 0.2 * flow_ratio**2)
        head = max(0, head)
        
        # 效率曲线
        efficiency_curve = -(flow_ratio - 0.9)**2 + 0.81  # 最佳效率点在90%额定流量
        efficiency = self.min_efficiency + (self.max_efficiency - self.min_efficiency) * max(0, efficiency_curve)
        
        return head, efficiency


class PumpSystem:
    """泵系统管理器"""
    
    def __init__(self, params: Dict[str, Any]):
        self.booster_pump = BoosterPump(params.get('booster', {}))
        self.hp_pump = HighPressurePump(params.get('hp', {}))
        
    def simulate_step(self, inputs: Dict[str, float], dt: float) -> Dict[str, Any]:
        """仿真泵系统
        
        Args:
            inputs: 输入参数
                - booster_flow_setpoint: Booster泵流量设定
                - hp_flow_setpoint: HP泵流量设定
                - tank_pressure_kPa: 储罐压力
                - export_pressure_kPa: 外输压力
                - liquid_temp_C: 液体温度
            dt: 时间步长
            
        Returns:
            泵系统输出状态
        """
        # Booster泵输入
        booster_inputs = {
            'flow_setpoint_m3h': inputs.get('booster_flow_setpoint', 0.0),
            'suction_pressure_kPa': inputs.get('tank_pressure_kPa', 110.0),
            'discharge_pressure_kPa': inputs.get('tank_pressure_kPa', 110.0) + 200,  # 增压200kPa
            'liquid_temp_C': inputs.get('liquid_temp_C', -162.0),
            'start_command': inputs.get('booster_start', False)
        }
        
        # HP泵输入
        hp_inputs = {
            'flow_setpoint_m3h': inputs.get('hp_flow_setpoint', 0.0),
            'suction_pressure_kPa': booster_inputs['discharge_pressure_kPa'],
            'discharge_pressure_kPa': inputs.get('export_pressure_kPa', 6500.0),
            'liquid_temp_C': inputs.get('liquid_temp_C', -162.0),
            'start_command': inputs.get('hp_start', False)
        }
        
        # 仿真两台泵
        booster_outputs = self.booster_pump.simulate_step(booster_inputs, dt)
        hp_outputs = self.hp_pump.simulate_step(hp_inputs, dt)
        
        return {
            'booster': booster_outputs,
            'hp': hp_outputs,
            'total_power_kw': booster_outputs['power_kw'] + hp_outputs['power_kw']
        }