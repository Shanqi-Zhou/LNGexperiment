import os
import yaml
import json
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """设置日志配置
    
    Args:
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: 日志文件路径，如果为None则只输出到控制台
        
    Returns:
        配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger('lng_energy_prediction')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径 (.yaml, .yml, .json)
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """保存配置文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    config_path = Path(config_path)
    
    # 确保目录存在
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

def save_results(results: Dict[str, Any], output_dir: str, experiment_name: str = None):
    """保存实验结果
    
    Args:
        results: 结果字典
        output_dir: 输出目录
        experiment_name: 实验名称，如果为None则使用时间戳
    """
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存结果为JSON
    results_file = output_path / 'results.json'
    
    # 处理不能JSON序列化的对象
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = float(value)
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            serializable_results[key] = _make_serializable(value)
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # 保存原始结果为pickle（包含所有对象）
    pickle_file = output_path / 'results.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    return str(output_path)

def _make_serializable(obj: Any) -> Any:
    """递归地使对象可JSON序列化"""
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        if isinstance(obj, torch.Tensor):
            obj = obj.cpu().numpy()
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj

def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """创建实验目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        实验目录路径
    """
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / 'models').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'configs').mkdir(exist_ok=True)
    
    return str(exp_dir)

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """绘制训练历史
    
    Args:
        history: 训练历史字典，包含loss, metrics等
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8')
    
    # 确定子图数量
    n_metrics = len(history)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(history.items()):
        ax = axes[i]
        ax.plot(values, linewidth=2, marker='o', markersize=4)
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # 添加最佳值标注
        if 'loss' in metric_name.lower():
            best_idx = np.argmin(values)
            best_val = values[best_idx]
        else:
            best_idx = np.argmax(values)
            best_val = values[best_idx]
            
        ax.annotate(f'Best: {best_val:.4f}\nEpoch: {best_idx}',
                   xy=(best_idx, best_val), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 隐藏多余的子图
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                              title: str = 'Predictions vs Actual', 
                              save_path: str = None):
    """绘制预测值vs实际值散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=0.6, s=30)
    
    # 完美预测线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 计算R²
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置相等的坐标轴比例
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                  title: str = 'Residual Analysis', 
                  save_path: str = None):
    """绘制残差分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8')
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 残差vs预测值
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差直方图
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差vs实际值
    axes[1, 1].scatter(y_true, residuals, alpha=0.6, s=30)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Actual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算回归指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 平均绝对百分比误差
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # 对称平均绝对百分比误差
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'smape': smape
    }

def set_random_seed(seed: int = 42):
    """设置随机种子
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(use_gpu: bool = True) -> torch.device:
    """获取计算设备
    
    Args:
        use_gpu: 是否使用GPU
        
    Returns:
        torch设备对象
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def print_model_summary(model, input_shape: tuple = None):
    """打印模型摘要
    
    Args:
        model: 模型对象
        input_shape: 输入形状
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    else:
        print(f"Model: {model.__class__.__name__}")
        
        # 计算参数数量
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    
    print("="*50 + "\n")

def format_time(seconds: float) -> str:
    """格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"