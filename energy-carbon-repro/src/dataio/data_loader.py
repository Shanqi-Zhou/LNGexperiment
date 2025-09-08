"""
数据I/O模块 - CSV读取与校核
基于技术路线标准数据格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class LNGDataLoader:
    """LNG仿真数据加载器"""
    
    def __init__(self, data_dir: str, logger: Optional[logging.Logger] = None):
        self.data_dir = Path(data_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # 标准数据文件定义
        self.data_files = {
            'env_weather': 'env_weather.csv',
            'tanks': 'tanks.csv', 
            'pumps_booster': 'pumps_booster.csv',
            'pumps_hp': 'pumps_hp.csv',
            'orv': 'orv.csv',
            'bog_compressor': 'bog_compressor.csv',
            'recondenser': 'recondenser.csv',
            'export_meter': 'export_meter.csv',
            'valves_controls': 'valves_controls.csv',
            'events': 'events.csv',
            'topology_edges': 'topology_edges.csv',
            'emission_factors': 'emission_factors.csv'
        }
        
        # 必需的数据文件
        self.required_files = [
            'env_weather', 'tanks', 'pumps_booster', 'pumps_hp', 
            'orv', 'bog_compressor', 'recondenser', 'export_meter'
        ]
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件"""
        self.logger.info("加载LNG仿真数据...")
        
        data_dict = {}
        
        for name, filename in self.data_files.items():
            filepath = self.data_dir / filename
            
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    
                    # 标准化时间戳列
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    elif 'ts' in df.columns:
                        df['ts'] = pd.to_datetime(df['ts'])
                        df = df.rename(columns={'ts': 'timestamp'})
                    
                    data_dict[name] = df
                    self.logger.info(f"加载 {filename}: {df.shape}")
                    
                except Exception as e:
                    self.logger.error(f"加载 {filename} 失败: {e}")
                    if name in self.required_files:
                        raise
            else:
                if name in self.required_files:
                    raise FileNotFoundError(f"必需文件 {filepath} 不存在")
                else:
                    self.logger.warning(f"可选文件 {filepath} 不存在")
        
        return data_dict
    
    def merge_timeseries_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并时序数据"""
        self.logger.info("合并时序数据...")
        
        # 以env_weather作为基准时间轴
        if 'env_weather' not in data_dict:
            raise ValueError("缺少基准时序数据 env_weather.csv")
        
        merged_df = data_dict['env_weather'].copy()
        
        # 依次合并其他时序数据
        timeseries_files = [
            'tanks', 'pumps_booster', 'pumps_hp', 'orv', 
            'bog_compressor', 'recondenser', 'export_meter'
        ]
        
        for name in timeseries_files:
            if name in data_dict:
                df = data_dict[name]
                merged_df = pd.merge(merged_df, df, on='timestamp', how='outer', suffixes=('', f'_{name}'))
        
        # 按时间排序
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"合并后数据形状: {merged_df.shape}")
        self.logger.info(f"时间范围: {merged_df['timestamp'].min()} 到 {merged_df['timestamp'].max()}")
        
        return merged_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """数据质量校验"""
        self.logger.info("执行数据质量校验...")
        
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_timestamps': df['timestamp'].duplicated().sum(),
            'time_gaps': self._check_time_gaps(df),
            'value_ranges': self._check_value_ranges(df),
            'data_consistency': self._check_data_consistency(df)
        }
        
        return quality_report
    
    def _check_time_gaps(self, df: pd.DataFrame) -> Dict[str, any]:
        """检查时间间隔"""
        if len(df) < 2:
            return {'status': 'insufficient_data'}
        
        time_diffs = df['timestamp'].diff().dropna()
        expected_interval = pd.Timedelta(seconds=10)  # 10s采样
        
        gaps = time_diffs[time_diffs > expected_interval * 1.5]
        
        return {
            'expected_interval': str(expected_interval),
            'actual_median_interval': str(time_diffs.median()),
            'num_gaps': len(gaps),
            'max_gap': str(gaps.max()) if len(gaps) > 0 else None
        }
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """检查数值范围合理性"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 预定义的合理范围
        expected_ranges = {
            'tank_level_pct': (0, 100),
            'tank_p_top_kPa': (50, 200),
            'orv_Q_MW': (0, 500),
            'pumps_booster_power_kw': (0, 300),
            'pumps_hp_power_kw': (0, 1500),
            'bog_compressor_total_power_kw': (0, 1000),
            'T_amb_C': (-10, 50),
            'seawater_T_C': (5, 35)
        }
        
        range_check = {}
        for col in numeric_cols:
            if col in expected_ranges:
                min_val, max_val = expected_ranges[col]
                actual_min = df[col].min()
                actual_max = df[col].max()
                
                range_check[col] = {
                    'expected_min': min_val,
                    'expected_max': max_val,
                    'actual_min': actual_min,
                    'actual_max': actual_max,
                    'within_range': min_val <= actual_min and actual_max <= max_val
                }
        
        return range_check
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, bool]:
        """检查数据一致性"""
        consistency_checks = {}
        
        # 检查功率与流量的合理性
        if all(col in df.columns for col in ['pumps_booster_flow_m3h', 'pumps_booster_power_kw']):
            # 流量为0时功率应该很小
            zero_flow_mask = df['pumps_booster_flow_m3h'] < 1
            high_power_at_zero_flow = df.loc[zero_flow_mask, 'pumps_booster_power_kw'] > 50
            consistency_checks['booster_pump_power_flow'] = not high_power_at_zero_flow.any()
        
        # 检查罐液位与BOG的关系
        if all(col in df.columns for col in ['tank_level_pct', 'tank_bog_rate_kgph']):
            # 液位变化与BOG产生应该有相关性
            level_change = df['tank_level_pct'].diff().abs()
            bog_rate = df['tank_bog_rate_kgph']
            correlation = level_change.corr(bog_rate)
            consistency_checks['tank_level_bog_correlation'] = abs(correlation) > 0.1
        
        return consistency_checks


def load_lng_data(data_dir: str, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Dict]:
    """便捷函数：加载并验证LNG数据"""
    loader = LNGDataLoader(data_dir, logger)
    
    # 加载所有数据
    data_dict = loader.load_all_data()
    
    # 合并时序数据
    merged_df = loader.merge_timeseries_data(data_dict)
    
    # 数据质量校验
    quality_report = loader.validate_data_quality(merged_df)
    
    return merged_df, quality_report