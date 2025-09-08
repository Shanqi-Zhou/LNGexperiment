"""配置管理器 - 优化配置文件加载和验证工具"""

import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import os

@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class OptimizedConfigManager:
    """优化配置管理器"""
    
    def __init__(self, config_dir: str = "configs/optimized", logger: Optional[logging.Logger] = None):
        self.config_dir = Path(config_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # 预定义配置模板
        self.config_templates = {
            "small": "small_dataset.yaml",
            "medium": "medium_dataset.yaml", 
            "large": "large_dataset.yaml",
            "rtx4060": "rtx4060_8gb.yaml",
            "production": "production.yaml",
            "debug": "debug.yaml",
            "benchmark": "benchmark.yaml"
        }
        
        self.logger.info(f"配置管理器初始化完成，配置目录: {self.config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """加载配置文件"""
        
        # 支持别名
        if config_name in self.config_templates:
            config_file = self.config_templates[config_name]
        else:
            config_file = config_name if config_name.endswith('.yaml') else f"{config_name}.yaml"
        
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"成功加载配置: {config_file}")
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件格式错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            raise
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """合并多个配置文件"""
        merged_config = {}
        
        for config_name in config_names:
            config = self.load_config(config_name)
            merged_config = self._deep_merge(merged_config, config)
        
        self.logger.info(f"合并了 {len(config_names)} 个配置文件")
        return merged_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_config(self, config: Union[str, Dict[str, Any]]) -> ConfigValidationResult:
        """验证配置文件"""
        
        if isinstance(config, str):
            config_dict = self.load_config(config)
            config_name = config
        else:
            config_dict = config
            config_name = "provided_config"
        
        errors = []
        warnings = []
        suggestions = []
        
        # 必需字段检查
        required_fields = {
            'model': ['type'],
            'training': ['batch_size', 'learning_rate', 'max_epochs'],
            'preprocessing': ['normalization'],
            'features': [],
            'performance_targets': []
        }
        
        for section, fields in required_fields.items():
            if section not in config_dict:
                errors.append(f"缺少必需的配置节: {section}")
                continue
            
            for field in fields:
                if field not in config_dict[section]:
                    errors.append(f"缺少必需字段: {section}.{field}")
        
        # 类型检查
        if 'model' in config_dict and 'type' in config_dict['model']:
            model_type = config_dict['model']['type']
            valid_types = ['mlr', 'hgbr', 'gpr', 'tcn', 'transformer', 'adaptive', 'residual', 'hybrid', 'ensemble', 'production', 'debug']
            if model_type not in valid_types:
                errors.append(f"无效的模型类型: {model_type}，支持的类型: {valid_types}")
        
        # 硬件兼容性检查
        if 'training' in config_dict:
            training = config_dict['training']
            
            if 'mixed_precision' in training and training['mixed_precision']:
                if not self._check_gpu_support():
                    warnings.append("启用了混合精度训练但未检测到支持的GPU")
            
            batch_size = training.get('batch_size', 32)
            if batch_size > 512:
                warnings.append(f"批处理大小较大({batch_size})，可能导致内存不足")
            elif batch_size < 16:
                suggestions.append("批处理大小较小，考虑增加以提高GPU利用率")
        
        # 性能目标检查
        if 'performance_targets' in config_dict:
            targets = config_dict['performance_targets']
            
            if 'min_r2_score' in targets:
                r2_target = targets['min_r2_score']
                if r2_target > 0.95:
                    warnings.append(f"R²目标过高({r2_target})，可能难以达到")
                elif r2_target < 0.5:
                    suggestions.append("R²目标较低，考虑提高以获得更好的模型性能")
        
        # 配置建议
        if config_name == "debug.yaml":
            suggestions.append("检测到调试配置，仅适用于开发阶段")
        
        if config_name == "production.yaml":
            suggestions.append("检测到生产配置，确保已充分测试")
        
        is_valid = len(errors) == 0
        
        return ConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _check_gpu_support(self) -> bool:
        """检查GPU支持"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def recommend_config(self, data_size: int, gpu_memory_gb: float = 0, 
                        target: str = "performance") -> str:
        """推荐合适的配置"""
        
        # 基于数据规模的推荐
        if data_size <= 10000:
            base_config = "small"
        elif data_size <= 100000:
            base_config = "medium"
        else:
            base_config = "large"
        
        # 基于GPU内存的调整
        if gpu_memory_gb > 0:
            if gpu_memory_gb <= 8:
                if base_config in ["medium", "large"]:
                    base_config = "rtx4060"  # RTX 4060特殊优化
            elif gpu_memory_gb >= 24:
                if base_config == "small":
                    base_config = "medium"  # 利用更大显存
        
        # 基于目标的调整
        if target == "production":
            base_config = "production"
        elif target == "debug":
            base_config = "debug"
        elif target == "benchmark":
            base_config = "benchmark"
        
        recommendation = self.config_templates[base_config]
        
        self.logger.info(f"推荐配置: {recommendation} (数据规模: {data_size}, GPU内存: {gpu_memory_gb}GB, 目标: {target})")
        
        return recommendation
    
    def get_available_configs(self) -> List[str]:
        """获取可用配置列表"""
        configs = []
        
        if self.config_dir.exists():
            for config_file in self.config_dir.glob("*.yaml"):
                configs.append(config_file.stem)
        
        return sorted(configs)
    
    def export_config(self, config: Dict[str, Any], output_path: str):
        """导出配置到文件"""
        output_path = Path(output_path)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info(f"配置已导出到: {output_path}")
            
        except Exception as e:
            self.logger.error(f"配置导出失败: {e}")
            raise
    
    def create_custom_config(self, base_config: str, modifications: Dict[str, Any], 
                           output_name: str) -> str:
        """基于现有配置创建自定义配置"""
        
        # 加载基础配置
        base = self.load_config(base_config)
        
        # 应用修改
        custom_config = self._deep_merge(base, modifications)
        
        # 导出自定义配置
        output_path = self.config_dir / f"{output_name}.yaml"
        self.export_config(custom_config, output_path)
        
        self.logger.info(f"创建自定义配置: {output_name}.yaml")
        
        return str(output_path)
    
    def compare_configs(self, config1: str, config2: str) -> Dict[str, Any]:
        """比较两个配置文件的差异"""
        
        cfg1 = self.load_config(config1)
        cfg2 = self.load_config(config2)
        
        differences = {
            'only_in_config1': {},
            'only_in_config2': {},
            'different_values': {},
            'same_values': {}
        }
        
        all_keys = set(self._get_all_keys(cfg1)) | set(self._get_all_keys(cfg2))
        
        for key in all_keys:
            val1 = self._get_nested_value(cfg1, key)
            val2 = self._get_nested_value(cfg2, key)
            
            if val1 is None:
                differences['only_in_config2'][key] = val2
            elif val2 is None:
                differences['only_in_config1'][key] = val1
            elif val1 != val2:
                differences['different_values'][key] = {'config1': val1, 'config2': val2}
            else:
                differences['same_values'][key] = val1
        
        return differences
    
    def _get_all_keys(self, d: Dict, prefix: str = "") -> List[str]:
        """获取嵌套字典的所有键路径"""
        keys = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            keys.append(key)
            if isinstance(v, dict):
                keys.extend(self._get_all_keys(v, key))
        return keys
    
    def _get_nested_value(self, d: Dict, key: str):
        """获取嵌套字典的值"""
        keys = key.split('.')
        value = d
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None

# 便捷函数
def load_optimized_config(config_name: str, config_dir: str = "configs/optimized") -> Dict[str, Any]:
    """加载优化配置文件"""
    manager = OptimizedConfigManager(config_dir)
    return manager.load_config(config_name)

def validate_optimized_config(config: Union[str, Dict], config_dir: str = "configs/optimized") -> ConfigValidationResult:
    """验证优化配置文件"""
    manager = OptimizedConfigManager(config_dir)
    return manager.validate_config(config)

def recommend_optimized_config(data_size: int, gpu_memory_gb: float = 0, 
                             target: str = "performance", config_dir: str = "configs/optimized") -> str:
    """推荐优化配置文件"""
    manager = OptimizedConfigManager(config_dir)
    return manager.recommend_config(data_size, gpu_memory_gb, target)