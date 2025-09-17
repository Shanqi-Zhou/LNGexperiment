
import numpy as np

class Tank:
    """
    LNG储罐模型。
    模拟储罐的热量侵入、BOG生成、液位和压力变化。
    """
    def __init__(self, 
                 volume_m3=1600,  # 缩放后体积 (16万m³ -> 1600)
                 initial_level_pct=0.8, 
                 ua_W_K=85.0, 
                 latent_heat_J_kg=5.05e5,
                 lng_density_kg_m3=450.0):
        """
        初始化储罐模型。
        参数来自 `论文复现技术路线.md`。

        Args:
            volume_m3 (float): 储罐总容积 (m³)。
            initial_level_pct (float): 初始液位百分比。
            ua_W_K (float): 总传热系数×面积 (W/K)。
            latent_heat_J_kg (float): LNG汽化潜热 (J/kg)。
            lng_density_kg_m3 (float): LNG液体密度 (kg/m³)。
        """
        self.volume_m3 = volume_m3
        self.ua_W_K = ua_W_K
        self.latent_heat_J_kg = latent_heat_J_kg
        self.lng_density_kg_m3 = lng_density_kg_m3

        # 初始状态
        self.liquid_volume_m3 = self.volume_m3 * initial_level_pct
        self.level_pct = initial_level_pct
        
        # 假设初始压力为大气压之上一个稳定值
        self.pressure_kPa = 110.0 
        
        # BOG相关
        self.bog_rate_kg_s = 0.0

    def calculate_bog(self, T_ambient_C, T_lng_C=-162.0):
        """
        根据环境温差计算BOG产生率。

        Args:
            T_ambient_C (float): 环境温度 (°C)。
            T_lng_C (float): LNG温度 (°C)。

        Returns:
            float: BOG产生率 (kg/s)。
        """
        delta_T = T_ambient_C - T_lng_C
        heat_ingress_W = self.ua_W_K * delta_T
        
        # BOG (kg/s) = 热量 (J/s) / 汽化潜热 (J/kg)
        bog_kg_s = heat_ingress_W / self.latent_heat_J_kg
        return bog_kg_s

    def update(self, T_ambient_C, outflow_kg_s, inflow_kg_s, dt_s):
        """
        更新储罐在单个时间步长的状态。

        Args:
            T_ambient_C (float): 当前环境温度 (°C)。
            outflow_kg_s (float): 从储罐流出的质量流量 (kg/s)。
            inflow_kg_s (float): 流入储罐的质量流量 (kg/s)。
            dt_s (float): 时间步长 (s)。
        """
        # 1. 计算BOG产生率
        self.bog_rate_kg_s = self.calculate_bog(T_ambient_C)

        # 2. 更新液体体积和液位
        net_flow_kg_s = inflow_kg_s - outflow_kg_s
        net_volume_change_m3 = (net_flow_kg_s / self.lng_density_kg_m3) * dt_s
        self.liquid_volume_m3 += net_volume_change_m3
        
        # 保证体积在合理范围内
        self.liquid_volume_m3 = np.clip(self.liquid_volume_m3, 0, self.volume_m3)
        self.level_pct = self.liquid_volume_m3 / self.volume_m3

        # 3. 更新压力 (简化模型：压力与顶空体积和BOG有关)
        # 这是一个简化的理想气体状态方程近似
        vapor_volume_m3 = self.volume_m3 - self.liquid_volume_m3
        if vapor_volume_m3 > 0:
            # 假设BOG进入顶空，压力会轻微上升
            # 实际模型会更复杂，这里仅为示意
            pressure_increase_factor = (self.bog_rate_kg_s * dt_s) / (vapor_volume_m3 * 0.1) # 0.1为简化系数
            self.pressure_kPa += pressure_increase_factor
        
        # 压力也会因BOG被抽出而下降
        # 此部分逻辑将在BOG压缩机模型中体现

    def get_state(self):
        """
        返回当前储罐状态。
        """
        return {
            "level_pct": self.level_pct * 100,
            "p_top_kPa": self.pressure_kPa,
            "bog_rate_kgph": self.bog_rate_kg_s * 3600
        }

