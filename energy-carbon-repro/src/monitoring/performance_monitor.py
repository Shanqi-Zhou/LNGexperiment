"""
性能监控系统 - LNG优化方案实时监控核心
提供训练过程、模型性能、系统资源的全方位监控
支持实时可视化、异常检测、性能分析等功能
"""

import torch
import psutil
import time
import threading
import json
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SystemMetrics:
    """系统资源指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0


@dataclass  
class TrainingMetrics:
    """训练过程指标"""
    epoch: int
    batch: int
    train_loss: float
    val_loss: float = 0.0
    learning_rate: float = 0.0
    batch_time: float = 0.0
    epoch_time: float = 0.0
    timestamp: float = 0.0


@dataclass
class ModelMetrics:
    """模型性能指标"""
    model_name: str
    r2_score: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    timestamp: float = 0.0


class RealTimeMonitor:
    """实时监控器 - 系统资源监控"""
    
    def __init__(self, 
                 sampling_interval: float = 1.0,
                 max_history: int = 1000,
                 logger: Optional[logging.Logger] = None):
        self.sampling_interval = sampling_interval
        self.max_history = max_history
        self.logger = logger or logging.getLogger(__name__)
        
        # 监控数据存储
        self.system_metrics = deque(maxlen=max_history)
        self.alerts = []
        
        # 监控控制
        self.monitoring = False
        self.monitor_thread = None
        
        # GPU检测
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.logger.info(f"检测到 {self.gpu_count} 个GPU设备")
    
    def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring:
            self.logger.warning("监控已在运行中")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("实时监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("实时监控已停止")
    
    def _monitoring_loop(self):
        """监控主循环"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                self._check_alerts(metrics)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # 磁盘IO
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0.0
        disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0.0
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb
        )
        
        # GPU指标
        if self.gpu_available:
            try:
                gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # GPU利用率（简化版）
                gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                
                metrics.gpu_memory_used_gb = gpu_memory_used
                metrics.gpu_memory_total_gb = gpu_memory_total
                metrics.gpu_utilization_percent = gpu_utilization
            except Exception as e:
                self.logger.debug(f"GPU指标收集失败: {e}")
        
        return metrics
    
    def _check_alerts(self, metrics: SystemMetrics):
        """检查告警条件"""
        alerts = []
        
        # CPU使用率告警
        if metrics.cpu_percent > 90:
            alerts.append({
                'type': 'cpu_high',
                'message': f'CPU使用率过高: {metrics.cpu_percent:.1f}%',
                'severity': 'warning',
                'timestamp': metrics.timestamp
            })
        
        # 内存使用率告警
        if metrics.memory_percent > 90:
            alerts.append({
                'type': 'memory_high',
                'message': f'内存使用率过高: {metrics.memory_percent:.1f}%',
                'severity': 'warning',
                'timestamp': metrics.timestamp
            })
        
        # GPU内存告警
        if self.gpu_available and metrics.gpu_utilization_percent > 95:
            alerts.append({
                'type': 'gpu_memory_high',
                'message': f'GPU内存使用率过高: {metrics.gpu_utilization_percent:.1f}%',
                'severity': 'critical',
                'timestamp': metrics.timestamp
            })
        
        self.alerts.extend(alerts)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        if self.system_metrics:
            return self.system_metrics[-1]
        return None
    
    def get_metrics_history(self, minutes: int = 10) -> List[SystemMetrics]:
        """获取指定时间范围的指标历史"""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.system_metrics if m.timestamp >= cutoff_time]


class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 训练指标存储
        self.training_metrics = []
        self.epoch_summaries = []
        
        # 实时统计
        self.current_epoch = 0
        self.epoch_start_time = 0
        self.batch_times = deque(maxlen=100)
        
        # 性能分析
        self.loss_history = {'train': [], 'val': []}
        self.lr_history = []
    
    def start_epoch(self, epoch: int):
        """开始一个新的epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.batch_times.clear()
    
    def log_batch(self, batch: int, train_loss: float, 
                  learning_rate: float = 0.0, batch_time: float = 0.0):
        """记录批次训练结果"""
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            batch=batch,
            train_loss=train_loss,
            learning_rate=learning_rate,
            batch_time=batch_time,
            timestamp=time.time()
        )
        
        self.training_metrics.append(metrics)
        self.batch_times.append(batch_time)
        self.loss_history['train'].append(train_loss)
        self.lr_history.append(learning_rate)
    
    def end_epoch(self, val_loss: float = 0.0):
        """结束当前epoch"""
        epoch_time = time.time() - self.epoch_start_time
        
        # 计算epoch统计
        epoch_metrics = self._calculate_epoch_stats(val_loss, epoch_time)
        self.epoch_summaries.append(epoch_metrics)
        
        if val_loss > 0:
            self.loss_history['val'].append(val_loss)
        
        self.logger.info(f"Epoch {self.current_epoch} 完成: "
                        f"训练损失={epoch_metrics['avg_train_loss']:.6f}, "
                        f"验证损失={val_loss:.6f}, "
                        f"耗时={epoch_time:.1f}s")
    
    def _calculate_epoch_stats(self, val_loss: float, epoch_time: float) -> Dict[str, Any]:
        """计算epoch统计信息"""
        epoch_batches = [m for m in self.training_metrics if m.epoch == self.current_epoch]
        
        if not epoch_batches:
            return {}
        
        stats = {
            'epoch': self.current_epoch,
            'total_batches': len(epoch_batches),
            'avg_train_loss': np.mean([b.train_loss for b in epoch_batches]),
            'min_train_loss': np.min([b.train_loss for b in epoch_batches]),
            'max_train_loss': np.max([b.train_loss for b in epoch_batches]),
            'val_loss': val_loss,
            'epoch_time': epoch_time,
            'avg_batch_time': np.mean(list(self.batch_times)) if self.batch_times else 0,
            'final_lr': epoch_batches[-1].learning_rate if epoch_batches else 0,
            'timestamp': time.time()
        }
        
        return stats
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.epoch_summaries:
            return {}
        
        return {
            'total_epochs': len(self.epoch_summaries),
            'total_batches': sum(e['total_batches'] for e in self.epoch_summaries),
            'best_train_loss': min(e['avg_train_loss'] for e in self.epoch_summaries),
            'best_val_loss': min(e['val_loss'] for e in self.epoch_summaries if e['val_loss'] > 0),
            'total_training_time': sum(e['epoch_time'] for e in self.epoch_summaries),
            'avg_epoch_time': np.mean([e['epoch_time'] for e in self.epoch_summaries]),
            'avg_batch_time': np.mean([m.batch_time for m in self.training_metrics if m.batch_time > 0]),
        }


class ModelPerformanceMonitor:
    """模型性能监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.model_metrics = []
        self.benchmark_results = {}
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, 
                      y_pred: np.ndarray, inference_time_ms: float = 0.0) -> ModelMetrics:
        """评估模型性能"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # 计算性能指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = ModelMetrics(
            model_name=model_name,
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            inference_time_ms=inference_time_ms,
            timestamp=time.time()
        )
        
        self.model_metrics.append(metrics)
        
        self.logger.info(f"模型 {model_name} 评估完成: "
                        f"R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        return metrics
    
    def benchmark_inference_speed(self, model, X_test: np.ndarray, 
                                 n_runs: int = 100) -> Dict[str, float]:
        """基准测试推理速度"""
        times = []
        
        # 预热
        for _ in range(5):
            if hasattr(model, 'predict'):
                _ = model.predict(X_test[:1])
            else:
                with torch.no_grad():
                    _ = model(torch.tensor(X_test[:1], dtype=torch.float32))
        
        # 基准测试
        for _ in range(n_runs):
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                _ = model.predict(X_test)
            else:
                with torch.no_grad():
                    _ = model(torch.tensor(X_test, dtype=torch.float32))
            
            times.append((time.time() - start_time) * 1000)  # 转换为毫秒
        
        benchmark = {
            'avg_inference_time_ms': np.mean(times),
            'std_inference_time_ms': np.std(times),
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times),
            'throughput_samples_per_sec': len(X_test) * 1000 / np.mean(times)
        }
        
        return benchmark
    
    def compare_models(self) -> pd.DataFrame:
        """比较所有模型性能"""
        if not self.model_metrics:
            return pd.DataFrame()
        
        data = []
        for metrics in self.model_metrics:
            data.append(asdict(metrics))
        
        df = pd.DataFrame(data)
        
        # 按R²得分排序
        df = df.sort_values('r2_score', ascending=False)
        
        return df


class PerformanceMonitoringSystem:
    """
    性能监控系统 - 统一监控框架
    集成系统资源、训练过程、模型性能监控
    """
    
    def __init__(self,
                 monitoring_config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = monitoring_config or {}
        
        # 初始化各监控组件
        self.system_monitor = RealTimeMonitor(
            sampling_interval=self.config.get('system_sampling_interval', 1.0),
            logger=self.logger
        )
        self.training_monitor = TrainingMonitor(self.logger)
        self.model_monitor = ModelPerformanceMonitor(self.logger)
        
        # 监控状态
        self.monitoring_active = False
        self.monitoring_session_id = None
        self.session_start_time = None
        
        # 数据存储
        self.monitoring_data = {
            'system': [],
            'training': [],
            'models': []
        }
        
        self.logger.info("性能监控系统初始化完成")
    
    def start_monitoring_session(self, session_id: str = None):
        """启动监控会话"""
        self.monitoring_session_id = session_id or f"session_{int(time.time())}"
        self.session_start_time = time.time()
        
        # 启动系统监控
        self.system_monitor.start_monitoring()
        self.monitoring_active = True
        
        self.logger.info(f"监控会话已启动: {self.monitoring_session_id}")
    
    def stop_monitoring_session(self):
        """停止监控会话"""
        if not self.monitoring_active:
            return
        
        self.system_monitor.stop_monitoring()
        self.monitoring_active = False
        
        # 保存监控数据
        self._save_session_data()
        
        session_duration = time.time() - self.session_start_time
        self.logger.info(f"监控会话已停止，耗时: {session_duration:.1f}秒")
    
    def log_training_batch(self, epoch: int, batch: int, train_loss: float,
                          learning_rate: float = 0.0, batch_time: float = 0.0):
        """记录训练批次数据"""
        if epoch != self.training_monitor.current_epoch:
            if self.training_monitor.current_epoch > 0:
                self.training_monitor.end_epoch()
            self.training_monitor.start_epoch(epoch)
        
        self.training_monitor.log_batch(batch, train_loss, learning_rate, batch_time)
    
    def log_validation_result(self, val_loss: float):
        """记录验证结果"""
        self.training_monitor.end_epoch(val_loss)
    
    def evaluate_model(self, model_name: str, model, X_test: np.ndarray, 
                      y_test: np.ndarray, benchmark_speed: bool = True) -> Dict[str, Any]:
        """评估模型并记录性能"""
        # 生成预测
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
        else:
            model.eval()
            with torch.no_grad():
                predictions = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
        
        # 基准测试推理速度
        inference_time = 0.0
        speed_benchmark = {}
        if benchmark_speed:
            speed_benchmark = self.model_monitor.benchmark_inference_speed(model, X_test)
            inference_time = speed_benchmark['avg_inference_time_ms']
        
        # 评估性能
        metrics = self.model_monitor.evaluate_model(
            model_name, y_test, predictions, inference_time
        )
        
        result = {
            'model_metrics': asdict(metrics),
            'speed_benchmark': speed_benchmark
        }
        
        return result
    
    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """获取实时仪表板数据"""
        dashboard_data = {
            'timestamp': time.time(),
            'session_id': self.monitoring_session_id,
            'system_metrics': {},
            'training_summary': {},
            'model_comparison': [],
            'alerts': []
        }
        
        # 系统指标
        current_system = self.system_monitor.get_current_metrics()
        if current_system:
            dashboard_data['system_metrics'] = asdict(current_system)
        
        # 训练摘要
        dashboard_data['training_summary'] = self.training_monitor.get_training_summary()
        
        # 模型比较
        model_comparison_df = self.model_monitor.compare_models()
        if not model_comparison_df.empty:
            dashboard_data['model_comparison'] = model_comparison_df.to_dict('records')
        
        # 告警信息
        dashboard_data['alerts'] = self.system_monitor.alerts[-10:]  # 最近10条告警
        
        return dashboard_data
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        report = {
            'session_info': {
                'session_id': self.monitoring_session_id,
                'start_time': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'duration_seconds': time.time() - self.session_start_time,
                'monitoring_active': self.monitoring_active
            },
            'system_performance': self._analyze_system_performance(),
            'training_analysis': self._analyze_training_performance(),
            'model_performance': self._analyze_model_performance(),
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """分析系统性能"""
        metrics_history = self.system_monitor.get_metrics_history(60)  # 最近60分钟
        
        if not metrics_history:
            return {}
        
        cpu_usage = [m.cpu_percent for m in metrics_history]
        memory_usage = [m.memory_percent for m in metrics_history]
        gpu_usage = [m.gpu_utilization_percent for m in metrics_history if m.gpu_utilization_percent > 0]
        
        analysis = {
            'cpu': {
                'avg_usage_percent': np.mean(cpu_usage),
                'max_usage_percent': np.max(cpu_usage),
                'usage_stability': 1 - (np.std(cpu_usage) / np.mean(cpu_usage))
            },
            'memory': {
                'avg_usage_percent': np.mean(memory_usage),
                'max_usage_percent': np.max(memory_usage),
                'min_available_gb': min(m.memory_available_gb for m in metrics_history)
            },
            'gpu': {
                'avg_usage_percent': np.mean(gpu_usage) if gpu_usage else 0,
                'max_usage_percent': np.max(gpu_usage) if gpu_usage else 0,
                'peak_memory_gb': max((m.gpu_memory_used_gb for m in metrics_history), default=0)
            }
        }
        
        return analysis
    
    def _analyze_training_performance(self) -> Dict[str, Any]:
        """分析训练性能"""
        return self.training_monitor.get_training_summary()
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """分析模型性能"""
        if not self.model_monitor.model_metrics:
            return {}
        
        metrics_df = self.model_monitor.compare_models()
        
        analysis = {
            'best_model': metrics_df.iloc[0].to_dict() if not metrics_df.empty else {},
            'model_count': len(metrics_df),
            'performance_range': {
                'r2_range': [float(metrics_df['r2_score'].min()), float(metrics_df['r2_score'].max())],
                'rmse_range': [float(metrics_df['rmse'].min()), float(metrics_df['rmse'].max())],
                'speed_range': [float(metrics_df['inference_time_ms'].min()), float(metrics_df['inference_time_ms'].max())]
            } if not metrics_df.empty else {}
        }
        
        return analysis
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 系统性能建议
        current_metrics = self.system_monitor.get_current_metrics()
        if current_metrics:
            if current_metrics.cpu_percent > 80:
                recommendations.append("CPU使用率较高，考虑减少并行处理或增加批处理大小")
            
            if current_metrics.memory_percent > 85:
                recommendations.append("内存使用率较高，考虑减少批处理大小或启用内存优化")
            
            if current_metrics.gpu_utilization_percent > 90:
                recommendations.append("GPU内存使用率很高，建议启用混合精度训练")
        
        # 训练性能建议
        training_summary = self.training_monitor.get_training_summary()
        if training_summary.get('avg_batch_time', 0) > 1.0:
            recommendations.append("批处理时间较长，考虑优化数据加载或模型结构")
        
        # 模型性能建议
        model_comparison = self.model_monitor.compare_models()
        if not model_comparison.empty:
            best_r2 = model_comparison.iloc[0]['r2_score']
            if best_r2 < 0.8:
                recommendations.append("模型性能有提升空间，考虑特征工程或模型调参")
        
        return recommendations
    
    def _save_session_data(self):
        """保存会话数据"""
        if not self.monitoring_session_id:
            return
        
        session_data = {
            'session_id': self.monitoring_session_id,
            'start_time': self.session_start_time,
            'end_time': time.time(),
            'system_metrics': [asdict(m) for m in self.system_monitor.system_metrics],
            'training_metrics': [asdict(m) for m in self.training_monitor.training_metrics],
            'model_metrics': [asdict(m) for m in self.model_monitor.model_metrics],
            'epoch_summaries': self.training_monitor.epoch_summaries,
            'alerts': self.system_monitor.alerts
        }
        
        # 这里可以保存到文件或数据库
        self.monitoring_data = session_data
    
    def visualize_performance(self, save_path: Optional[str] = None):
        """可视化性能数据"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 系统CPU使用率
        ax1 = axes[0, 0]
        metrics_history = self.system_monitor.get_metrics_history(30)
        if metrics_history:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics_history]
            cpu_usage = [m.cpu_percent for m in metrics_history]
            ax1.plot(timestamps, cpu_usage, 'b-', linewidth=2)
            ax1.set_title('CPU使用率趋势')
            ax1.set_ylabel('CPU (%)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 内存使用率
        ax2 = axes[0, 1]
        if metrics_history:
            memory_usage = [m.memory_percent for m in metrics_history]
            ax2.plot(timestamps, memory_usage, 'g-', linewidth=2)
            ax2.set_title('内存使用率趋势')
            ax2.set_ylabel('Memory (%)')
            ax2.tick_params(axis='x', rotation=45)
        
        # GPU使用率
        ax3 = axes[0, 2]
        if metrics_history:
            gpu_usage = [m.gpu_utilization_percent for m in metrics_history]
            if any(g > 0 for g in gpu_usage):
                ax3.plot(timestamps, gpu_usage, 'r-', linewidth=2)
                ax3.set_title('GPU使用率趋势')
                ax3.set_ylabel('GPU (%)')
                ax3.tick_params(axis='x', rotation=45)
        
        # 训练损失曲线
        ax4 = axes[1, 0]
        if self.training_monitor.loss_history['train']:
            train_loss = self.training_monitor.loss_history['train']
            ax4.plot(train_loss, 'b-', label='Train Loss', linewidth=2)
            
            if self.training_monitor.loss_history['val']:
                val_loss = self.training_monitor.loss_history['val']
                ax4.plot(val_loss, 'r-', label='Val Loss', linewidth=2)
            
            ax4.set_title('训练损失曲线')
            ax4.set_ylabel('Loss')
            ax4.set_xlabel('Epoch/Batch')
            ax4.legend()
        
        # 学习率变化
        ax5 = axes[1, 1]
        if self.training_monitor.lr_history:
            ax5.plot(self.training_monitor.lr_history, 'orange', linewidth=2)
            ax5.set_title('学习率变化')
            ax5.set_ylabel('Learning Rate')
            ax5.set_xlabel('Batch')
        
        # 模型性能比较
        ax6 = axes[1, 2]
        model_comparison = self.model_monitor.compare_models()
        if not model_comparison.empty:
            models = model_comparison['model_name'].tolist()
            r2_scores = model_comparison['r2_score'].tolist()
            ax6.bar(models, r2_scores)
            ax6.set_title('模型R²得分比较')
            ax6.set_ylabel('R² Score')
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"性能可视化图表已保存: {save_path}")
        
        plt.show()


# 便捷函数
def create_performance_monitor(config: Optional[Dict[str, Any]] = None,
                             logger: Optional[logging.Logger] = None) -> PerformanceMonitoringSystem:
    """创建性能监控系统实例"""
    return PerformanceMonitoringSystem(config, logger)


def quick_monitoring_session(training_function: Callable,
                           session_id: str = None,
                           logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """快速监控会话包装器"""
    monitor = create_performance_monitor(logger=logger)
    
    monitor.start_monitoring_session(session_id)
    
    try:
        # 执行训练函数
        training_result = training_function()
        
        # 生成报告
        performance_report = monitor.generate_performance_report()
        
        return {
            'training_result': training_result,
            'performance_report': performance_report,
            'dashboard_data': monitor.get_realtime_dashboard_data()
        }
    
    finally:
        monitor.stop_monitoring_session()