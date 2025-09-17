
# Import all component models
from .tank import Tank
from .pumps import Pump
from .orv import ORV
from .bog_compressor import BOGCompressor
from .recondenser import Recondenser

class LNGSystem:
    """
    完整的LNG接收站系统模型。
    负责初始化所有设备组件，并在仿真循环中协调它们的交互。
    """
    def __init__(self):
        """
        初始化系统，创建所有设备实例。
        参数均来自 `论文复现技术路线.md`。
        """
        # 1. 储罐
        self.tank = Tank(
            volume_m3=1600, 
            initial_level_pct=0.8,
            ua_W_K=85.0
        )

        # 2. 泵 (Booster & HP)
        # 假设最大流量，实际流量由需求决定
        self.booster_pump = Pump(rated_power_kW=220, max_flow_m3h=2500)
        self.hp_pump = Pump(rated_power_kW=1200, max_flow_m3h=2500)

        # 3. 气化器
        self.orv = ORV(ua_W_K=4.0e6, fouling_decay_beta=3.0e-6)

        # 4. BOG 压缩机
        self.bog_compressor = BOGCompressor(
            num_stages=2, 
            pressure_ratio_per_stage=2.2, 
            isentropic_efficiency=0.72
        )

        # 5. 再冷凝器
        self.recondenser = Recondenser(max_capacity_tph=8.0)

        # 6. 系统时间
        self.time_elapsed_s = 0.0

    def update(self, export_demand_tph, T_ambient_C, T_seawater_C, dt_s):
        """
        在单个时间步长内，按逻辑顺序更新整个系统状态。

        Args:
            export_demand_tph (float): 外部需求的天然气流量 (t/h)。
            T_ambient_C (float): 环境温度 (°C)。
            T_seawater_C (float): 海水温度 (°C)。
            dt_s (float): 时间步长 (s)。
        """
        # --- 核心仿真逻辑顺序 ---

        # 1. BOG产生: 首先根据环境温度计算储罐的自然BOG产生率
        bog_natural_kg_s = self.tank.calculate_bog(T_ambient_C)

        # 2. BOG处理: 将产生的BOG送去压缩和再冷凝
        #    入口温度假设为LNG沸点(111K), 入口压力假设为罐顶压力
        inlet_pressure_kPa = self.tank.get_state()['p_top_kPa']
        self.bog_compressor.update(bog_natural_kg_s, inlet_temp_K=111, inlet_pressure_kPa=inlet_pressure_kPa)
        bog_flow_tph = bog_natural_kg_s * 3.6
        self.recondenser.update(bog_flow_tph)

        # 3. LNG外输: 根据外部需求，驱动HP泵和ORV
        #    扬程需满足6.5MPa出口压力要求，此处为简化值
        self.hp_pump.update(flow_m3h=export_demand_tph * 1000 / 450, head_m=1500)
        self.orv.update(export_demand_tph, T_seawater_C, self.time_elapsed_s)

        # 4. Booster泵: 其流量需满足HP泵和再冷凝器的总需求
        lng_for_recondenser_tph = self.recondenser.get_state()['m_liq_in_tph']
        total_lng_demand_tph = export_demand_tph + lng_for_recondenser_tph
        self.booster_pump.update(flow_m3h=total_lng_demand_tph * 1000 / 450, head_m=300)

        # 5. 储罐状态更新: 最后，根据所有流入流出，更新储罐的最终状态
        outflow_kg_s = (total_lng_demand_tph * 1000) / 3600
        # 返回到储罐的包括被冷凝的BOG和用于冷却的LNG
        inflow_kg_s = (self.recondenser.get_state()['m_condensed_kgph'] + 
                       self.recondenser.get_state()['m_liq_in_tph'] * 1000) / 3600
        self.tank.update(T_ambient_C, outflow_kg_s, inflow_kg_s, dt_s)

        # 6. 更新系统时间
        self.time_elapsed_s += dt_s

    def get_all_states(self):
        """
        收集并返回所有组件的当前状态，用于数据记录。
        """
        states = {
            'tank': self.tank.get_state(),
            'booster_pump': self.booster_pump.get_state(),
            'hp_pump': self.hp_pump.get_state(),
            'orv': self.orv.get_state(),
            'bog_compressor': self.bog_compressor.get_state(),
            'recondenser': self.recondenser.get_state()
        }
        return states
