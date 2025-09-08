"""LNG仿真模块

包含LNG全链路系统的各个子系统仿真模型：
- tank: LNG储罐热力学模型
- pumps: 泵系统模型（增压泵、高压泵）
- orv: ORV海水浴式气化器模型
- bog_compressor: BOG压缩机系统模型
- recondenser: 喷淋式再冷凝器模型
- system: 系统集成仿真器
- control: 控制系统模型
"""

from .tank import LNGTank
from .pumps import LNGPump, BoosterPump, HighPressurePump, PumpSystem
from .orv import ORVVaporizer
from .bog_compressor import BOGCompressorStage, BOGCompressorSystem
from .recondenser import SprayRecondenser
from .system import LNGSystemSimulator
from .control import PIDController, CascadeController, SelectorController, LNGSystemController

__all__ = [
    'LNGTank',
    'LNGPump', 'BoosterPump', 'HighPressurePump', 'PumpSystem',
    'ORVVaporizer',
    'BOGCompressorStage', 'BOGCompressorSystem',
    'SprayRecondenser',
    'LNGSystemSimulator',
    'PIDController', 'CascadeController', 'SelectorController', 'LNGSystemController'
]

__version__ = '1.0.0'
__author__ = 'LNG Simulation Team'
__description__ = 'LNG全链路工况仿真系统'