import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ControlMode(Enum):
    """控制模式枚举"""
    MANUAL = "manual"
    AUTO = "auto"
    CASCADE = "cascade"
    OVERRIDE = "override"

@dataclass
class PIDParameters:
    """PID参数"""
    kp: float = 1.0  # 比例增益
    ki: float = 0.1  # 积分增益
    kd: float = 0.01  # 微分增益
    output_min: float = 0.0  # 输出下限
    output_max: float = 100.0  # 输出上限
    integral_min: float = -100.0  # 积分限幅下限
    integral_max: float = 100.0  # 积分限幅上限
    derivative_filter_tau: float = 0.1  # 微分滤波时间常数
    
class PIDController:
    """PID控制器
    
    实现标准PID控制算法，包含积分限幅、微分滤波等功能
    """
    
    def __init__(self, params: PIDParameters, name: str = "PID"):
        self.params = params
        self.name = name
        
        # 状态变量
        self.setpoint = 0.0
        self.process_value = 0.0
        self.output = 0.0
        self.mode = ControlMode.AUTO
        
        # 内部状态
        self.error = 0.0
        self.error_prev = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.derivative_filtered = 0.0
        
        # 时间相关
        self.dt = 1.0
        self.last_time = 0.0
        
        # 性能指标
        self.iae = 0.0  # 积分绝对误差
        self.ise = 0.0  # 积分平方误差
        self.max_error = 0.0  # 最大误差
        
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.derivative_filtered = 0.0
        self.error_prev = 0.0
        self.iae = 0.0
        self.ise = 0.0
        self.max_error = 0.0
        
    def set_parameters(self, params: PIDParameters):
        """设置PID参数"""
        self.params = params
        
    def set_setpoint(self, setpoint: float):
        """设置设定值"""
        self.setpoint = setpoint
        
    def set_mode(self, mode: ControlMode):
        """设置控制模式"""
        if mode != self.mode:
            if mode == ControlMode.AUTO:
                # 切换到自动模式时，重置积分项为当前输出
                self.integral = self.output
            self.mode = mode
            
    def update(self, process_value: float, dt: float = None) -> float:
        """更新控制器
        
        Args:
            process_value: 过程变量值
            dt: 时间步长
            
        Returns:
            控制器输出
        """
        if dt is not None:
            self.dt = dt
            
        self.process_value = process_value
        
        if self.mode != ControlMode.AUTO:
            return self.output
            
        # 计算误差
        self.error = self.setpoint - process_value
        
        # 比例项
        proportional = self.params.kp * self.error
        
        # 积分项
        self.integral += self.error * self.dt
        # 积分限幅
        self.integral = np.clip(self.integral, 
                               self.params.integral_min / max(self.params.ki, 1e-6),
                               self.params.integral_max / max(self.params.ki, 1e-6))
        integral = self.params.ki * self.integral
        
        # 微分项（带滤波）
        if self.dt > 0:
            derivative_raw = (self.error - self.error_prev) / self.dt
            # 一阶滤波
            alpha = self.dt / (self.params.derivative_filter_tau + self.dt)
            self.derivative_filtered = (1 - alpha) * self.derivative_filtered + alpha * derivative_raw
            derivative = self.params.kd * self.derivative_filtered
        else:
            derivative = 0.0
            
        # PID输出
        output_raw = proportional + integral + derivative
        
        # 输出限幅
        self.output = np.clip(output_raw, self.params.output_min, self.params.output_max)
        
        # 抗积分饱和（条件积分）
        if (output_raw > self.params.output_max and self.error > 0) or \
           (output_raw < self.params.output_min and self.error < 0):
            # 输出饱和时，停止积分累积
            self.integral -= self.error * self.dt
            
        # 更新历史值
        self.error_prev = self.error
        
        # 更新性能指标
        self.iae += abs(self.error) * self.dt
        self.ise += self.error**2 * self.dt
        self.max_error = max(self.max_error, abs(self.error))
        
        return self.output
        
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            'name': self.name,
            'mode': self.mode.value,
            'setpoint': self.setpoint,
            'process_value': self.process_value,
            'output': self.output,
            'error': self.error,
            'proportional': self.params.kp * self.error,
            'integral': self.params.ki * self.integral,
            'derivative': self.params.kd * self.derivative_filtered,
            'iae': self.iae,
            'ise': self.ise,
            'max_error': self.max_error
        }

class CascadeController:
    """串级控制器
    
    主控制器输出作为副控制器设定值
    """
    
    def __init__(self, master_params: PIDParameters, slave_params: PIDParameters, 
                 name: str = "Cascade"):
        self.master = PIDController(master_params, f"{name}_Master")
        self.slave = PIDController(slave_params, f"{name}_Slave")
        self.name = name
        
    def update(self, master_setpoint: float, master_pv: float, 
               slave_pv: float, dt: float = None) -> float:
        """更新串级控制器
        
        Args:
            master_setpoint: 主控制器设定值
            master_pv: 主控制器过程变量
            slave_pv: 副控制器过程变量
            dt: 时间步长
            
        Returns:
            副控制器输出
        """
        # 主控制器计算
        self.master.set_setpoint(master_setpoint)
        slave_setpoint = self.master.update(master_pv, dt)
        
        # 副控制器计算
        self.slave.set_setpoint(slave_setpoint)
        output = self.slave.update(slave_pv, dt)
        
        return output
        
    def get_status(self) -> Dict[str, Any]:
        """获取串级控制器状态"""
        return {
            'name': self.name,
            'master': self.master.get_status(),
            'slave': self.slave.get_status()
        }

class SelectorController:
    """选择器控制器
    
    实现高选、低选、中选等逻辑
    """
    
    def __init__(self, selector_type: str = "high", name: str = "Selector"):
        self.selector_type = selector_type  # "high", "low", "median"
        self.name = name
        self.inputs = []
        self.selected_value = 0.0
        self.selected_index = 0
        
    def update(self, inputs: List[float]) -> float:
        """更新选择器
        
        Args:
            inputs: 输入值列表
            
        Returns:
            选择的输出值
        """
        self.inputs = inputs.copy()
        
        if not inputs:
            return 0.0
            
        if self.selector_type == "high":
            self.selected_value = max(inputs)
            self.selected_index = inputs.index(self.selected_value)
        elif self.selector_type == "low":
            self.selected_value = min(inputs)
            self.selected_index = inputs.index(self.selected_value)
        elif self.selector_type == "median":
            sorted_inputs = sorted(inputs)
            n = len(sorted_inputs)
            if n % 2 == 0:
                self.selected_value = (sorted_inputs[n//2-1] + sorted_inputs[n//2]) / 2
            else:
                self.selected_value = sorted_inputs[n//2]
            self.selected_index = inputs.index(min(inputs, key=lambda x: abs(x - self.selected_value)))
        else:
            self.selected_value = inputs[0]
            self.selected_index = 0
            
        return self.selected_value
        
    def get_status(self) -> Dict[str, Any]:
        """获取选择器状态"""
        return {
            'name': self.name,
            'type': self.selector_type,
            'inputs': self.inputs,
            'selected_value': self.selected_value,
            'selected_index': self.selected_index
        }

class LNGSystemController:
    """LNG系统控制器
    
    集成各子系统的控制逻辑
    """
    
    def __init__(self, control_config: Dict[str, Any]):
        self.config = control_config
        
        # 储罐压力控制（BOG压缩机启停）
        tank_pressure_params = PIDParameters(
            kp=0.5, ki=0.1, kd=0.05,
            output_min=0.0, output_max=100.0
        )
        self.tank_pressure_controller = PIDController(tank_pressure_params, "TankPressure")
        
        # 外输压力控制（高压泵转速）
        export_pressure_params = PIDParameters(
            kp=1.0, ki=0.2, kd=0.1,
            output_min=30.0, output_max=100.0  # 泵转速百分比
        )
        self.export_pressure_controller = PIDController(export_pressure_params, "ExportPressure")
        
        # 外输流量控制（增压泵转速）
        export_flow_params = PIDParameters(
            kp=0.8, ki=0.15, kd=0.08,
            output_min=20.0, output_max=100.0
        )
        self.export_flow_controller = PIDController(export_flow_params, "ExportFlow")
        
        # ORV出口温度控制（LNG流量）
        orv_temp_params = PIDParameters(
            kp=2.0, ki=0.3, kd=0.15,
            output_min=0.1, output_max=2.0  # 流量倍数
        )
        self.orv_temp_controller = PIDController(orv_temp_params, "ORVTemp")
        
        # 再冷凝器控制（喷淋流量）
        recondenser_params = PIDParameters(
            kp=1.5, ki=0.25, kd=0.1,
            output_min=0.0, output_max=10.0  # t/h
        )
        self.recondenser_controller = PIDController(recondenser_params, "Recondenser")
        
        # 控制设定值
        self.setpoints = {
            'tank_pressure_kPa': control_config.get('tank_pressure_setpoint', 120.0),
            'export_pressure_kPa': control_config.get('export_pressure_setpoint', 6500.0),
            'export_flow_Nm3h': control_config.get('export_flow_setpoint', 1000.0),
            'orv_outlet_temp_C': control_config.get('orv_temp_setpoint', 5.0),
            'recondenser_temp_C': control_config.get('recondenser_temp_setpoint', -100.0)
        }
        
        # 控制器状态
        self.controllers_enabled = {
            'tank_pressure': True,
            'export_pressure': True,
            'export_flow': True,
            'orv_temp': True,
            'recondenser': True
        }
        
        # 报警和联锁
        self.alarms = {
            'tank_pressure_high': False,
            'tank_pressure_low': False,
            'export_pressure_low': False,
            'pump_cavitation': False,
            'orv_fouling_severe': False
        }
        
        self.interlocks = {
            'emergency_shutdown': False,
            'bog_compressor_trip': False,
            'pump_trip': False
        }
        
    def update_setpoints(self, new_setpoints: Dict[str, float]):
        """更新控制设定值"""
        self.setpoints.update(new_setpoints)
        
    def enable_controller(self, controller_name: str, enabled: bool = True):
        """启用/禁用控制器"""
        if controller_name in self.controllers_enabled:
            self.controllers_enabled[controller_name] = enabled
            
    def check_alarms_interlocks(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """检查报警和联锁
        
        Args:
            system_state: 系统状态
            
        Returns:
            报警联锁状态
        """
        # 储罐压力报警
        tank_pressure = system_state.get('tank', {}).get('p_top_kPa', 0)
        self.alarms['tank_pressure_high'] = tank_pressure > 150.0
        self.alarms['tank_pressure_low'] = tank_pressure < 80.0
        
        # 外输压力报警
        export_pressure = system_state.get('export_pressure_kPa', 0)
        self.alarms['export_pressure_low'] = export_pressure < 6000.0
        
        # 泵汽蚀报警
        pumps_state = system_state.get('pumps', {})
        booster_npsh = pumps_state.get('booster', {}).get('npsh_a_m', 10)
        hp_npsh = pumps_state.get('hp', {}).get('npsh_a_m', 10)
        self.alarms['pump_cavitation'] = booster_npsh < 2.0 or hp_npsh < 3.0
        
        # ORV结垢报警
        orv_state = system_state.get('orv', {})
        fouling_factor = orv_state.get('fouling_factor', 1.0)
        self.alarms['orv_fouling_severe'] = fouling_factor < 0.7
        
        # 联锁逻辑
        self.interlocks['emergency_shutdown'] = (
            self.alarms['tank_pressure_high'] or 
            system_state.get('emergency_stop', False)
        )
        
        self.interlocks['bog_compressor_trip'] = (
            self.alarms['tank_pressure_low'] or
            self.interlocks['emergency_shutdown']
        )
        
        self.interlocks['pump_trip'] = (
            self.alarms['pump_cavitation'] or
            self.interlocks['emergency_shutdown']
        )
        
        return {
            'alarms': self.alarms.copy(),
            'interlocks': self.interlocks.copy()
        }
        
    def calculate_control_outputs(self, system_state: Dict[str, Any], 
                                dt: float = 1.0) -> Dict[str, Any]:
        """计算控制输出
        
        Args:
            system_state: 系统状态
            dt: 时间步长
            
        Returns:
            控制输出
        """
        control_outputs = {}
        
        # 检查报警联锁
        alarm_status = self.check_alarms_interlocks(system_state)
        
        # 储罐压力控制
        if self.controllers_enabled['tank_pressure'] and not self.interlocks['bog_compressor_trip']:
            tank_pressure = system_state.get('tank', {}).get('p_top_kPa', 0)
            self.tank_pressure_controller.set_setpoint(self.setpoints['tank_pressure_kPa'])
            bog_compressor_speed = self.tank_pressure_controller.update(tank_pressure, dt)
            control_outputs['bog_compressor_speed_pct'] = bog_compressor_speed
        else:
            control_outputs['bog_compressor_speed_pct'] = 0.0
            
        # 外输压力控制
        if self.controllers_enabled['export_pressure'] and not self.interlocks['pump_trip']:
            export_pressure = system_state.get('export_pressure_kPa', 0)
            self.export_pressure_controller.set_setpoint(self.setpoints['export_pressure_kPa'])
            hp_pump_speed = self.export_pressure_controller.update(export_pressure, dt)
            control_outputs['hp_pump_speed_pct'] = hp_pump_speed
        else:
            control_outputs['hp_pump_speed_pct'] = 0.0
            
        # 外输流量控制
        if self.controllers_enabled['export_flow'] and not self.interlocks['pump_trip']:
            export_flow = system_state.get('export_flow_Nm3h', 0)
            self.export_flow_controller.set_setpoint(self.setpoints['export_flow_Nm3h'])
            booster_pump_speed = self.export_flow_controller.update(export_flow, dt)
            control_outputs['booster_pump_speed_pct'] = booster_pump_speed
        else:
            control_outputs['booster_pump_speed_pct'] = 0.0
            
        # ORV出口温度控制
        if self.controllers_enabled['orv_temp']:
            orv_temp = system_state.get('orv', {}).get('T_out_C', 0)
            self.orv_temp_controller.set_setpoint(self.setpoints['orv_outlet_temp_C'])
            lng_flow_factor = self.orv_temp_controller.update(orv_temp, dt)
            control_outputs['orv_lng_flow_factor'] = lng_flow_factor
        else:
            control_outputs['orv_lng_flow_factor'] = 1.0
            
        # 再冷凝器控制
        if self.controllers_enabled['recondenser']:
            bog_temp = system_state.get('recondenser', {}).get('T_bog_out_C', -50)
            self.recondenser_controller.set_setpoint(self.setpoints['recondenser_temp_C'])
            spray_flow = self.recondenser_controller.update(bog_temp, dt)
            control_outputs['recondenser_spray_flow_tph'] = spray_flow
        else:
            control_outputs['recondenser_spray_flow_tph'] = 0.0
            
        # 添加控制器状态信息
        control_outputs['controller_status'] = {
            'tank_pressure': self.tank_pressure_controller.get_status(),
            'export_pressure': self.export_pressure_controller.get_status(),
            'export_flow': self.export_flow_controller.get_status(),
            'orv_temp': self.orv_temp_controller.get_status(),
            'recondenser': self.recondenser_controller.get_status()
        }
        
        control_outputs['alarms_interlocks'] = alarm_status
        
        return control_outputs
        
    def get_control_performance(self) -> Dict[str, Any]:
        """获取控制性能指标"""
        return {
            'tank_pressure': {
                'iae': self.tank_pressure_controller.iae,
                'ise': self.tank_pressure_controller.ise,
                'max_error': self.tank_pressure_controller.max_error
            },
            'export_pressure': {
                'iae': self.export_pressure_controller.iae,
                'ise': self.export_pressure_controller.ise,
                'max_error': self.export_pressure_controller.max_error
            },
            'export_flow': {
                'iae': self.export_flow_controller.iae,
                'ise': self.export_flow_controller.ise,
                'max_error': self.export_flow_controller.max_error
            },
            'orv_temp': {
                'iae': self.orv_temp_controller.iae,
                'ise': self.orv_temp_controller.ise,
                'max_error': self.orv_temp_controller.max_error
            },
            'recondenser': {
                'iae': self.recondenser_controller.iae,
                'ise': self.recondenser_controller.ise,
                'max_error': self.recondenser_controller.max_error
            }
        }