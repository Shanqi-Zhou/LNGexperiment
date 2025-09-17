"""
高级资源监控和管理框架
Day 3 优化：实时监控系统资源使用情况
"""
import time
import threading
import psutil
from typing import Dict, List, Any, Optional
from collections import deque
import json
from pathlib import Path

class ResourceMonitor:
    """实时资源监控器"""

    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 300):
        """
        初始化资源监控器

        Args:
            monitoring_interval: 监控间隔（秒）
            history_size: 历史记录保存数量
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 性能历史记录
        self.performance_history = {
            'timestamps': deque(maxlen=history_size),
            'cpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'memory_used_mb': deque(maxlen=history_size),
            'memory_available_mb': deque(maxlen=history_size),
            'disk_io_read_mb': deque(maxlen=history_size),
            'disk_io_write_mb': deque(maxlen=history_size),
            'network_sent_mb': deque(maxlen=history_size),
            'network_recv_mb': deque(maxlen=history_size)
        }

        # 基准值（用于计算差值）
        self.baseline_disk_io = None
        self.baseline_network = None

        # 性能统计
        self.performance_stats = {
            'peak_cpu': 0.0,
            'peak_memory': 0.0,
            'avg_cpu': 0.0,
            'avg_memory': 0.0,
            'monitoring_duration': 0.0,
            'warnings_triggered': []
        }

    def start_monitoring(self) -> None:
        """开始资源监控"""
        if self.monitoring:
            return

        self.monitoring = True
        self._reset_baseline()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print(f"  🔍 资源监控已启动，间隔: {self.monitoring_interval}秒")

    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回统计信息"""
        if not self.monitoring:
            return self.get_performance_summary()

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # 计算最终统计
        self._calculate_final_stats()
        summary = self.get_performance_summary()
        print(f"  📊 资源监控已停止，监控时长: {summary['monitoring_duration']:.1f}秒")
        return summary

    def _reset_baseline(self) -> None:
        """重置基准值"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.baseline_disk_io = {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                }

            network_io = psutil.net_io_counters()
            if network_io:
                self.baseline_network = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv
                }
        except Exception:
            # 如果无法获取IO统计，使用默认值
            self.baseline_disk_io = {'read_bytes': 0, 'write_bytes': 0}
            self.baseline_network = {'bytes_sent': 0, 'bytes_recv': 0}

    def _monitoring_loop(self) -> None:
        """监控循环"""
        start_time = time.time()

        while self.monitoring:
            try:
                timestamp = time.time()

                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)

                # 内存使用情况
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / (1024 * 1024)
                memory_available_mb = memory.available / (1024 * 1024)

                # 磁盘IO（计算差值）
                disk_read_mb, disk_write_mb = self._get_disk_io_delta()

                # 网络IO（计算差值）
                network_sent_mb, network_recv_mb = self._get_network_io_delta()

                # 记录数据
                self.performance_history['timestamps'].append(timestamp)
                self.performance_history['cpu_percent'].append(cpu_percent)
                self.performance_history['memory_percent'].append(memory_percent)
                self.performance_history['memory_used_mb'].append(memory_used_mb)
                self.performance_history['memory_available_mb'].append(memory_available_mb)
                self.performance_history['disk_io_read_mb'].append(disk_read_mb)
                self.performance_history['disk_io_write_mb'].append(disk_write_mb)
                self.performance_history['network_sent_mb'].append(network_sent_mb)
                self.performance_history['network_recv_mb'].append(network_recv_mb)

                # 更新峰值
                self.performance_stats['peak_cpu'] = max(self.performance_stats['peak_cpu'], cpu_percent)
                self.performance_stats['peak_memory'] = max(self.performance_stats['peak_memory'], memory_percent)

                # 检查警告条件
                self._check_performance_warnings(cpu_percent, memory_percent, memory_available_mb)

            except Exception as e:
                # 监控失败不应该影响主程序
                pass

            time.sleep(self.monitoring_interval)

        # 记录监控总时长
        self.performance_stats['monitoring_duration'] = time.time() - start_time

    def _get_disk_io_delta(self) -> tuple:
        """获取磁盘IO增量（MB）"""
        try:
            if not self.baseline_disk_io:
                return 0.0, 0.0

            current_io = psutil.disk_io_counters()
            if not current_io:
                return 0.0, 0.0

            read_delta = (current_io.read_bytes - self.baseline_disk_io['read_bytes']) / (1024 * 1024)
            write_delta = (current_io.write_bytes - self.baseline_disk_io['write_bytes']) / (1024 * 1024)

            return max(0, read_delta), max(0, write_delta)
        except Exception:
            return 0.0, 0.0

    def _get_network_io_delta(self) -> tuple:
        """获取网络IO增量（MB）"""
        try:
            if not self.baseline_network:
                return 0.0, 0.0

            current_net = psutil.net_io_counters()
            if not current_net:
                return 0.0, 0.0

            sent_delta = (current_net.bytes_sent - self.baseline_network['bytes_sent']) / (1024 * 1024)
            recv_delta = (current_net.bytes_recv - self.baseline_network['bytes_recv']) / (1024 * 1024)

            return max(0, sent_delta), max(0, recv_delta)
        except Exception:
            return 0.0, 0.0

    def _check_performance_warnings(self, cpu_percent: float, memory_percent: float, memory_available_mb: float) -> None:
        """检查性能警告条件"""
        warnings = []

        # CPU使用率警告
        if cpu_percent > 90:
            warnings.append(f"高CPU使用率: {cpu_percent:.1f}%")

        # 内存使用率警告
        if memory_percent > 85:
            warnings.append(f"高内存使用率: {memory_percent:.1f}%")

        # 可用内存警告
        if memory_available_mb < 500:  # 少于500MB可用内存
            warnings.append(f"低可用内存: {memory_available_mb:.0f}MB")

        # 记录新警告
        for warning in warnings:
            if warning not in self.performance_stats['warnings_triggered']:
                self.performance_stats['warnings_triggered'].append(warning)
                print(f"  ⚠️ 性能警告: {warning}")

    def _calculate_final_stats(self) -> None:
        """计算最终统计信息"""
        if not self.performance_history['cpu_percent']:
            return

        # 计算平均值
        cpu_values = list(self.performance_history['cpu_percent'])
        memory_values = list(self.performance_history['memory_percent'])

        self.performance_stats['avg_cpu'] = sum(cpu_values) / len(cpu_values)
        self.performance_stats['avg_memory'] = sum(memory_values) / len(memory_values)

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前资源状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024 * 1024 * 1024),
                'monitoring_active': self.monitoring
            }
        except Exception:
            return {'error': '无法获取系统资源信息'}

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'peak_cpu_percent': self.performance_stats['peak_cpu'],
            'peak_memory_percent': self.performance_stats['peak_memory'],
            'avg_cpu_percent': self.performance_stats['avg_cpu'],
            'avg_memory_percent': self.performance_stats['avg_memory'],
            'monitoring_duration_seconds': self.performance_stats['monitoring_duration'],
            'total_data_points': len(self.performance_history['cpu_percent']),
            'warnings_count': len(self.performance_stats['warnings_triggered']),
            'warnings': self.performance_stats['warnings_triggered'].copy()
        }

        # 添加趋势信息
        if len(self.performance_history['cpu_percent']) > 10:
            recent_cpu = list(self.performance_history['cpu_percent'])[-10:]
            recent_memory = list(self.performance_history['memory_percent'])[-10:]

            summary['recent_avg_cpu'] = sum(recent_cpu) / len(recent_cpu)
            summary['recent_avg_memory'] = sum(recent_memory) / len(recent_memory)

        return summary

    def export_performance_data(self, file_path: str) -> bool:
        """导出性能数据到文件"""
        try:
            export_data = {
                'metadata': {
                    'monitoring_interval': self.monitoring_interval,
                    'export_timestamp': time.time(),
                    'total_data_points': len(self.performance_history['timestamps'])
                },
                'performance_history': {
                    key: list(values) for key, values in self.performance_history.items()
                },
                'performance_stats': self.performance_stats,
                'summary': self.get_performance_summary()
            }

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"  💾 性能数据已导出到: {file_path}")
            return True
        except Exception as e:
            print(f"  ❌ 导出失败: {e}")
            return False


class MemoryProfiler:
    """内存使用分析器"""

    def __init__(self):
        self.memory_snapshots = []
        self.process = psutil.Process()

    def take_snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        """拍摄内存快照"""
        try:
            memory_info = self.process.memory_info()
            virtual_memory = psutil.virtual_memory()

            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'process_virtual_mb': memory_info.vms / (1024 * 1024),
                'system_memory_percent': virtual_memory.percent,
                'system_available_mb': virtual_memory.available / (1024 * 1024)
            }

            self.memory_snapshots.append(snapshot)
            return snapshot
        except Exception as e:
            return {'error': str(e), 'timestamp': time.time(), 'label': label}

    def get_memory_growth(self) -> Dict[str, Any]:
        """分析内存增长情况"""
        if len(self.memory_snapshots) < 2:
            return {'error': '需要至少2个快照来分析增长情况'}

        first = self.memory_snapshots[0]
        last = self.memory_snapshots[-1]

        if 'error' in first or 'error' in last:
            return {'error': '快照包含错误数据'}

        growth_mb = last['process_memory_mb'] - first['process_memory_mb']
        time_elapsed = last['timestamp'] - first['timestamp']

        return {
            'memory_growth_mb': growth_mb,
            'time_elapsed_seconds': time_elapsed,
            'growth_rate_mb_per_second': growth_mb / time_elapsed if time_elapsed > 0 else 0,
            'initial_memory_mb': first['process_memory_mb'],
            'final_memory_mb': last['process_memory_mb'],
            'snapshots_count': len(self.memory_snapshots)
        }

    def clear_snapshots(self) -> None:
        """清除所有快照"""
        self.memory_snapshots.clear()


# 使用示例和测试
if __name__ == '__main__':
    print("=== 资源监控器测试 ===")

    # 创建监控器
    monitor = ResourceMonitor(monitoring_interval=0.5)
    profiler = MemoryProfiler()

    # 获取当前状态
    current_status = monitor.get_current_status()
    print(f"当前系统状态:")
    for key, value in current_status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # 开始监控
    profiler.take_snapshot("start")
    monitor.start_monitoring()

    # 模拟一些工作负载
    print("\n模拟工作负载...")
    import numpy as np
    for i in range(3):
        # 模拟内存密集型操作
        data = np.random.randn(1000, 1000)
        result = np.dot(data, data.T)
        del data, result
        time.sleep(1)

        profiler.take_snapshot(f"step_{i+1}")

    # 停止监控
    profiler.take_snapshot("end")
    summary = monitor.stop_monitoring()

    # 打印结果
    print(f"\n=== 性能监控摘要 ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        elif isinstance(value, list) and value:
            print(f"{key}: {len(value)} 项")
        else:
            print(f"{key}: {value}")

    # 内存分析
    memory_growth = profiler.get_memory_growth()
    print(f"\n=== 内存增长分析 ===")
    for key, value in memory_growth.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n=== 资源监控器测试完成 ===")