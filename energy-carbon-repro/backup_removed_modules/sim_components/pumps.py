
import numpy as np

class Pump:
    """
    通用LNG泵模型。
    可用于模拟增压泵 (Booster) 和高压泵 (HP)。
    """
    def __init__(self, 
                 rated_power_kW,
                 max_flow_m3h,
                 min_efficiency=0.65,
                 max_efficiency=0.80,
                 lng_density_kg_m3=450.0):
        """
        初始化泵模型。

        Args:
            rated_power_kW (float): 泵的额定功率 (kW)。
            max_flow_m3h (float): 泵的最大设计流量 (m³/h)。
            min_efficiency (float): 最小效率 (通常在零流量或低流量时)。
            max_efficiency (float): 最高效率 (通常在最佳效率点)。
            lng_density_kg_m3 (float): LNG液体密度 (kg/m³)。
        """
        self.rated_power_kW = rated_power_kW
        self.max_flow_m3h = max_flow_m3h
        self.min_efficiency = min_efficiency
        self.max_efficiency = max_efficiency
        self.lng_density_kg_m3 = lng_density_kg_m3
        
        # 状态变量
        self.flow_m3h = 0.0
        self.head_m = 0.0
        self.power_kW = 0.0
        self.npsh_a_m = 10.0  # 假设一个健康的初始NPSHa

    def _calculate_efficiency(self, flow_m3h):
        """
        根据流量计算泵的效率 (线性模型)。
        """
        # 简单的线性插值
        efficiency = self.min_efficiency + (self.max_efficiency - self.min_efficiency) * (flow_m3h / self.max_flow_m3h)
        return np.clip(efficiency, self.min_efficiency, self.max_efficiency)

    def update(self, flow_m3h, head_m):
        """
        根据需求流量和扬程，更新泵的运行状态和功耗。

        Args:
            flow_m3h (float): 当前需求的流量 (m³/h)。
            head_m (float): 当前需求的扬程 (m)。
        """
        self.flow_m3h = np.clip(flow_m3h, 0, self.max_flow_m3h)
        self.head_m = head_m

        if self.flow_m3h <= 0:
            self.power_kW = 0.0
            return

        # 1. 计算水力功率 (kW)
        # P_hyd = Q * H * rho * g
        flow_m3_s = self.flow_m3h / 3600.0
        gravity = 9.81
        hydraulic_power_W = flow_m3_s * self.head_m * self.lng_density_kg_m3 * gravity
        hydraulic_power_kW = hydraulic_power_W / 1000.0

        # 2. 计算效率
        efficiency = self._calculate_efficiency(self.flow_m3h)

        # 3. 计算电机功率 (kW)
        if efficiency > 0:
            electric_power_kW = hydraulic_power_kW / efficiency
        else:
            electric_power_kW = 0.0
        
        # 4. 功耗不能超过额定功率
        self.power_kW = np.clip(electric_power_kW, 0, self.rated_power_kW)
        
        # NPSHa的计算比较复杂，依赖上游罐压和液位，此处暂时设为固定值
        # 在系统集成时可以进一步完善
        # self.npsh_a_m = ...

    def get_state(self):
        """
        返回当前泵的状态。
        """
        return {
            "flow_m3h": self.flow_m3h,
            "head_m": self.head_m,
            "power_kw": self.power_kW,
            "npsh_a_m": self.npsh_a_m
        }
