
import numpy as np

class Recondenser:
    """
    再冷凝器模型。
    利用过冷LNG将压缩后的BOG重新冷凝成液体。
    """
    def __init__(self, max_capacity_tph=8.0, lng_bog_ratio=5.0):
        """
        初始化再冷凝器模型。

        Args:
            max_capacity_tph (float): 最大BOG处理能力 (吨/小时)。
            lng_bog_ratio (float): 经验参数，表示冷凝1kg的BOG需要多少kg的过冷LNG。
                                   通常在4-6之间。
        """
        self.max_capacity_tph = max_capacity_tph
        self.lng_bog_ratio = lng_bog_ratio

        # 状态变量
        self.bog_in_kgph = 0.0
        self.liq_in_tph = 0.0
        self.condensed_kgph = 0.0
        self.flared_kgph = 0.0

    def update(self, bog_flow_tph):
        """
        根据输入的BOG流量，更新再冷凝器的运行状态。

        Args:
            bog_flow_tph (float): 从压缩机来的BOG流量 (吨/小时)。
        """
        self.bog_in_kgph = bog_flow_tph * 1000

        if bog_flow_tph <= 0:
            self.liq_in_tph = 0.0
            self.condensed_kgph = 0.0
            self.flared_kgph = 0.0
            return

        # 1. 确定实际能够被冷凝的BOG量
        # 取决于BOG来量和设备最大能力
        condensed_tph = min(bog_flow_tph, self.max_capacity_tph)
        self.condensed_kgph = condensed_tph * 1000

        # 2. 计算送往火炬的BOG量
        flared_tph = bog_flow_tph - condensed_tph
        self.flared_kgph = flared_tph * 1000

        # 3. 根据经验系数估算所需的过冷LNG流量
        self.liq_in_tph = condensed_tph * self.lng_bog_ratio

    def get_state(self):
        """
        返回当前再冷凝器的状态。
        """
        return {
            "m_bog_in_kgph": self.bog_in_kgph,
            "m_liq_in_tph": self.liq_in_tph,
            "m_condensed_kgph": self.condensed_kgph
        }
    
    def get_flared_flow_kgph(self):
        """
        返回送往火炬的BOG流量。
        """
        return self.flared_kgph
