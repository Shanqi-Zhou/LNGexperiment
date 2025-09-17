
import numpy as np

class ORV:
    """
    海水浴式气化器 (Open Rack Vaporizer) 模型。
    模拟LNG吸收海水热量完成气化，并考虑结垢导致的性能衰减。
    """
    def __init__(self, 
                 ua_W_K=4.0e6, 
                 fouling_decay_beta=3.0e-6,
                 lng_inlet_temp_C=-162.0):
        """
        初始化ORV模型。

        Args:
            ua_W_K (float): 初始总传热系数×面积 (W/K)。
            fouling_decay_beta (float): 结垢衰减系数 (1/s)。
            lng_inlet_temp_C (float): LNG入口温度 (°C)。
        """
        self.initial_ua_W_K = ua_W_K
        self.fouling_decay_beta = fouling_decay_beta
        self.lng_inlet_temp_C = lng_inlet_temp_C

        # 状态变量
        self.lng_flow_tph = 0.0
        self.outlet_temp_C = 0.0
        self.heat_transfer_W = 0.0
        self.effective_ua_W_K = self.initial_ua_W_K

    def update(self, lng_flow_tph, seawater_temp_C, time_elapsed_s):
        """
        根据输入条件，更新ORV的运行状态。

        Args:
            lng_flow_tph (float): 流经ORV的LNG质量流量 (t/h)。
            seawater_temp_C (float): 当前海水温度 (°C)。
            time_elapsed_s (float): 从仿真开始经过的总时间 (s)。
        """
        self.lng_flow_tph = lng_flow_tph

        if self.lng_flow_tph <= 0:
            self.heat_transfer_W = 0.0
            self.outlet_temp_C = seawater_temp_C # 无流量时，出口温度等于海水温度
            return

        # 1. 计算因结垢而衰减的有效传热系数
        # U_eff = U_initial * exp(-β * t)
        self.effective_ua_W_K = self.initial_ua_W_K * np.exp(-self.fouling_decay_beta * time_elapsed_s)

        # 2. 计算总传热量 (Q = U_eff * ΔT)
        # ΔT 使用海水温度和LNG进口温度的差值
        delta_T = seawater_temp_C - self.lng_inlet_temp_C
        self.heat_transfer_W = self.effective_ua_W_K * delta_T

        # 3. 计算出口温度 (简化模型)
        # 假设出口天然气温度比海水温度低一个固定的差值，例如5°C
        # 这是一个简化，实际出口温度与流量和传热量严格相关
        self.outlet_temp_C = seawater_temp_C - 5.0
        
        # 也可以根据热平衡反算出口温度，但这需要天然气的比热容(cp)和汽化潜热
        # Q = m * (h_vap + cp_gas * (T_out - T_boil))
        # 为保持模型简洁，我们暂时采用上述简化方法

    def get_state(self):
        """
        返回当前ORV的状态。
        """
        return {
            "m_LNG_tph": self.lng_flow_tph,
            "T_out_C": self.outlet_temp_C,
            "Q_W": self.heat_transfer_W,
            "U_eff_WK": self.effective_ua_W_K
        }
