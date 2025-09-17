"""
生产级特征缓存系统 - Week 2 优化
实现SQLite-based的特征缓存，提供持久化存储和智能管理
"""

import sqlite3
import pickle
import hashlib
import time
import os
import threading
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import warnings


class ProductionFeatureCache:
    """
    生产级特征缓存系统
    提供高性能的特征存储、检索和管理功能
    """

    def __init__(self, cache_dir: str = "cache/features", max_size_gb: float = 2.0,
                 cleanup_threshold: float = 0.8, compression: bool = True):
        """
        初始化特征缓存系统

        Args:
            cache_dir: 缓存目录
            max_size_gb: 最大缓存大小（GB）
            cleanup_threshold: 触发清理的阈值（占最大容量的比例）
            compression: 是否启用数据压缩
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.cleanup_threshold = cleanup_threshold
        self.compression = compression

        # 数据库连接
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.connection_lock = threading.Lock()

        # 性能统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletions': 0,
            'cleanup_operations': 0,
            'total_bytes_stored': 0,
            'total_bytes_retrieved': 0
        }

        self._init_database()
        print(f"  特征缓存初始化: 目录={cache_dir}, 最大容量={max_size_gb}GB")

    def _init_database(self):
        """初始化缓存元数据数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    access_time REAL NOT NULL,
                    creation_time REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    data_hash TEXT,
                    metadata TEXT,
                    is_compressed BOOLEAN DEFAULT FALSE
                )
            """)

            # 创建索引提高查询性能
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_time ON cache_entries(access_time)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hit_count ON cache_entries(hit_count)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_size_bytes ON cache_entries(size_bytes)
            """)

            conn.commit()

    def _generate_key(self, data_identifier: Union[str, Dict, np.ndarray, pd.DataFrame]) -> str:
        """
        生成数据唯一标识键

        Args:
            data_identifier: 数据标识符（可以是字符串、字典、数组等）

        Returns:
            唯一的缓存键
        """
        if isinstance(data_identifier, str):
            content = data_identifier
        elif isinstance(data_identifier, dict):
            # 将字典转换为排序的字符串表示
            content = str(sorted(data_identifier.items()))
        elif isinstance(data_identifier, (np.ndarray, pd.DataFrame)):
            # 对于数据数组，使用形状、数据类型和部分数据生成哈希
            if isinstance(data_identifier, pd.DataFrame):
                data_identifier = data_identifier.values

            shape_str = str(data_identifier.shape)
            dtype_str = str(data_identifier.dtype)
            # 使用数据的统计特征而不是原始数据来生成键
            stats = {
                'mean': float(np.mean(data_identifier)),
                'std': float(np.std(data_identifier)),
                'min': float(np.min(data_identifier)),
                'max': float(np.max(data_identifier))
            }
            content = f"{shape_str}_{dtype_str}_{str(stats)}"
        else:
            content = str(data_identifier)

        # 生成SHA256哈希
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

    def _calculate_data_hash(self, data: Any) -> str:
        """计算数据的哈希值，用于验证数据完整性"""
        try:
            if isinstance(data, (np.ndarray, pd.DataFrame)):
                if isinstance(data, pd.DataFrame):
                    data_bytes = data.values.tobytes()
                else:
                    data_bytes = data.tobytes()
            else:
                data_bytes = pickle.dumps(data)

            return hashlib.md5(data_bytes).hexdigest()
        except:
            return "unknown"

    def _compress_data(self, data: Any) -> bytes:
        """压缩数据"""
        try:
            import gzip
            raw_data = pickle.dumps(data)

            if self.compression:
                compressed_data = gzip.compress(raw_data)
                # 只有压缩比超过10%才使用压缩数据
                if len(compressed_data) < len(raw_data) * 0.9:
                    return compressed_data, True
                else:
                    return raw_data, False
            else:
                return raw_data, False
        except Exception as e:
            print(f"    数据压缩失败: {e}")
            return pickle.dumps(data), False

    def _decompress_data(self, data: bytes, is_compressed: bool) -> Any:
        """解压缩数据"""
        try:
            if is_compressed:
                import gzip
                raw_data = gzip.decompress(data)
            else:
                raw_data = data

            return pickle.loads(raw_data)
        except Exception as e:
            print(f"    数据解压失败: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存的数据，如果不存在则返回None
        """
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)

            try:
                cursor = conn.execute(
                    "SELECT file_path, is_compressed, data_hash FROM cache_entries WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()

                if result:
                    file_path, is_compressed, data_hash = result
                    file_path = Path(file_path)

                    if file_path.exists():
                        try:
                            # 读取数据
                            with open(file_path, 'rb') as f:
                                data_bytes = f.read()

                            data = self._decompress_data(data_bytes, is_compressed)

                            # 验证数据完整性
                            if data_hash and data_hash != "unknown":
                                current_hash = self._calculate_data_hash(data)
                                if current_hash != data_hash:
                                    print(f"    ⚠️ 数据完整性验证失败: {key}")
                                    self._remove_entry(conn, key, file_path)
                                    self.stats['misses'] += 1
                                    return None

                            # 更新访问统计
                            conn.execute("""
                                UPDATE cache_entries
                                SET access_time = ?, hit_count = hit_count + 1
                                WHERE key = ?
                            """, (time.time(), key))

                            self.stats['hits'] += 1
                            self.stats['total_bytes_retrieved'] += len(data_bytes)

                            return data

                        except Exception as e:
                            print(f"    缓存读取失败 {key}: {e}")
                            self._remove_entry(conn, key, file_path)
                    else:
                        # 清理失效条目
                        self._remove_entry(conn, key)

                self.stats['misses'] += 1
                return None

            finally:
                conn.commit()
                conn.close()

    def set(self, key: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            data: 要缓存的数据
            metadata: 额外的元数据

        Returns:
            是否成功设置
        """
        try:
            # 压缩数据
            data_bytes, is_compressed = self._compress_data(data)
            data_size = len(data_bytes)

            # 检查单个数据是否过大
            if data_size > self.max_size_bytes * 0.1:  # 不允许单个数据超过总容量的10%
                print(f"    数据过大无法缓存: {data_size / 1024**2:.2f}MB (key: {key})")
                return False

            # 计算数据哈希
            data_hash = self._calculate_data_hash(data)

            # 生成文件路径
            file_path = self.cache_dir / f"{key}.pkl"

            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(data_bytes)

            # 更新数据库
            with self.connection_lock:
                conn = sqlite3.connect(self.db_path)

                try:
                    current_time = time.time()
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries
                        (key, file_path, size_bytes, access_time, creation_time,
                         data_hash, metadata, is_compressed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, str(file_path), data_size, current_time, current_time,
                        data_hash, str(metadata) if metadata else None, is_compressed
                    ))

                    self.stats['sets'] += 1
                    self.stats['total_bytes_stored'] += data_size

                    # 检查是否需要清理
                    if self._check_cleanup_needed(conn):
                        self._cleanup_cache(conn)

                    return True

                finally:
                    conn.commit()
                    conn.close()

        except Exception as e:
            print(f"    缓存设置失败 {key}: {e}")
            return False

    def _remove_entry(self, conn: sqlite3.Connection, key: str, file_path: Optional[Path] = None):
        """从缓存中删除条目"""
        if file_path:
            file_path.unlink(missing_ok=True)

        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        self.stats['deletions'] += 1

    def _check_cleanup_needed(self, conn: sqlite3.Connection) -> bool:
        """检查是否需要清理缓存"""
        total_size = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
        ).fetchone()[0]

        return total_size > self.max_size_bytes * self.cleanup_threshold

    def _cleanup_cache(self, conn: sqlite3.Connection):
        """智能缓存清理"""
        print("    执行缓存清理...")

        # 获取当前缓存统计
        current_size = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries"
        ).fetchone()[0]

        target_size = int(self.max_size_bytes * self.cleanup_threshold * 0.7)  # 清理到70%
        bytes_to_clean = current_size - target_size

        if bytes_to_clean <= 0:
            return

        # 清理策略：优先删除访问次数少且最旧的条目
        # 使用加权评分：低访问次数 + 旧访问时间 = 高删除优先级
        entries_to_delete = conn.execute("""
            SELECT key, file_path, size_bytes
            FROM cache_entries
            ORDER BY
                (access_time * 0.3 + hit_count * 0.7) ASC,
                creation_time ASC
        """).fetchall()

        bytes_cleaned = 0
        cleaned_count = 0

        for key, file_path, size_bytes in entries_to_delete:
            if bytes_cleaned >= bytes_to_clean:
                break

            self._remove_entry(conn, key, Path(file_path))
            bytes_cleaned += size_bytes
            cleaned_count += 1

        self.stats['cleanup_operations'] += 1
        print(f"    缓存清理完成: 删除{cleaned_count}个条目, 释放{bytes_cleaned/1024**2:.2f}MB")

    def clear(self):
        """清空所有缓存"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)

            try:
                # 获取所有文件路径
                file_paths = conn.execute("SELECT file_path FROM cache_entries").fetchall()

                # 删除所有文件
                for (file_path,) in file_paths:
                    Path(file_path).unlink(missing_ok=True)

                # 清空数据库
                conn.execute("DELETE FROM cache_entries")

                print("    缓存已清空")

                # 重置统计
                for key in self.stats:
                    if key.endswith('_operations') or key.endswith('deletions'):
                        continue  # 保留操作计数
                    self.stats[key] = 0

            finally:
                conn.commit()
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)

            try:
                db_stats = conn.execute("""
                    SELECT
                        COUNT(*) as entries,
                        COALESCE(SUM(size_bytes), 0) as total_bytes,
                        COALESCE(AVG(hit_count), 0) as avg_hits,
                        COALESCE(SUM(hit_count), 0) as total_hits,
                        COALESCE(MAX(access_time) - MIN(creation_time), 0) as age_range
                    FROM cache_entries
                """).fetchone()

                entries, total_bytes, avg_hits, total_hits, age_range = db_stats

                stats = {
                    'cache_entries': entries,
                    'total_size_mb': total_bytes / 1024**2,
                    'total_size_gb': total_bytes / 1024**3,
                    'capacity_usage_percent': (total_bytes / self.max_size_bytes) * 100,
                    'avg_hits_per_entry': avg_hits,
                    'total_hits': total_hits,
                    'cache_age_hours': age_range / 3600 if age_range else 0,
                    'hit_rate_percent': (self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)) * 100,
                    **self.stats
                }

                return stats

            finally:
                conn.close()

    def get_detailed_info(self) -> List[Dict]:
        """获取详细的缓存条目信息"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)

            try:
                entries = conn.execute("""
                    SELECT key, size_bytes, access_time, creation_time, hit_count, metadata
                    FROM cache_entries
                    ORDER BY access_time DESC
                """).fetchall()

                detailed_info = []
                current_time = time.time()

                for key, size_bytes, access_time, creation_time, hit_count, metadata in entries:
                    info = {
                        'key': key,
                        'size_mb': size_bytes / 1024**2,
                        'age_hours': (current_time - creation_time) / 3600,
                        'last_access_hours': (current_time - access_time) / 3600,
                        'hit_count': hit_count,
                        'metadata': metadata
                    }
                    detailed_info.append(info)

                return detailed_info

            finally:
                conn.close()

    def optimize_cache(self):
        """优化缓存性能"""
        print("  开始缓存优化...")

        with self.connection_lock:
            conn = sqlite3.connect(self.db_path)

            try:
                # 重建索引
                conn.execute("REINDEX")

                # 清理碎片
                conn.execute("VACUUM")

                # 更新统计信息
                conn.execute("ANALYZE")

                print("  缓存优化完成")

            finally:
                conn.close()


class CacheManager:
    """缓存管理器，提供高级缓存功能"""

    def __init__(self, cache_dir: str = "cache/features", max_size_gb: float = 2.0):
        self.cache = ProductionFeatureCache(cache_dir, max_size_gb)

    def cache_features(self, extractor_func, data, cache_key_base: str, **kwargs):
        """
        缓存特征提取结果的装饰器风格函数

        Args:
            extractor_func: 特征提取函数
            data: 输入数据
            cache_key_base: 缓存键基础
            **kwargs: 传递给特征提取函数的参数

        Returns:
            特征提取结果
        """
        # 生成缓存键
        key_components = {
            'base': cache_key_base,
            'data_shape': data.shape if hasattr(data, 'shape') else str(type(data)),
            'params': kwargs
        }
        cache_key = self.cache._generate_key(key_components)

        # 尝试从缓存获取
        print(f"    检查特征缓存: {cache_key_base}")
        cached_result = self.cache.get(cache_key)

        if cached_result is not None:
            print(f"    ✅ 缓存命中: {cache_key_base}")
            return cached_result

        # 缓存未命中，执行特征提取
        print(f"    ❌ 缓存未命中，执行特征提取: {cache_key_base}")

        start_time = time.time()
        result = extractor_func(data, **kwargs)
        extraction_time = time.time() - start_time

        # 缓存结果
        metadata = {
            'extraction_time': extraction_time,
            'extractor_func': extractor_func.__name__ if hasattr(extractor_func, '__name__') else str(extractor_func),
            'data_info': str(type(data))
        }

        success = self.cache.set(cache_key, result, metadata)
        if success:
            print(f"    💾 结果已缓存: {cache_key_base} (用时: {extraction_time:.2f}s)")
        else:
            print(f"    ⚠️ 缓存失败: {cache_key_base}")

        return result

    def print_cache_report(self):
        """打印缓存报告"""
        stats = self.cache.get_stats()

        print("\n=== 缓存性能报告 ===")
        print(f"缓存条目: {stats['cache_entries']}")
        print(f"使用容量: {stats['total_size_mb']:.2f}MB ({stats['capacity_usage_percent']:.1f}%)")
        print(f"命中率: {stats['hit_rate_percent']:.1f}%")
        print(f"总命中: {stats['total_hits']}, 平均命中/条目: {stats['avg_hits_per_entry']:.1f}")
        print(f"缓存年龄: {stats['cache_age_hours']:.1f}小时")
        print(f"清理操作: {stats['cleanup_operations']}次")


# 测试代码
if __name__ == '__main__':
    print("=== 生产级特征缓存测试 ===")

    # 创建测试缓存
    cache_manager = CacheManager(cache_dir="test_cache", max_size_gb=0.1)  # 100MB测试

    # 模拟特征提取函数
    def dummy_feature_extractor(data, window_size=10, feature_type='basic'):
        """模拟的特征提取函数"""
        print(f"        执行特征提取: window_size={window_size}, feature_type={feature_type}")
        time.sleep(0.1)  # 模拟计算时间

        if isinstance(data, np.ndarray):
            # 简单的特征计算
            features = {
                'mean': np.mean(data, axis=0),
                'std': np.std(data, axis=0),
                'shape': data.shape
            }
        else:
            features = {'type': str(type(data)), 'value': str(data)[:100]}

        return features

    # 创建测试数据
    test_data_1 = np.random.randn(1000, 5)
    test_data_2 = np.random.randn(500, 10)

    print("\n--- 缓存性能测试 ---")

    # 第一次提取（缓存未命中）
    print("第一次特征提取:")
    start_time = time.time()
    features_1 = cache_manager.cache_features(
        dummy_feature_extractor, test_data_1, "test_features_1",
        window_size=15, feature_type='advanced'
    )
    time_1 = time.time() - start_time
    print(f"用时: {time_1:.3f}秒")

    # 第二次提取相同数据（缓存命中）
    print("\n第二次提取相同数据:")
    start_time = time.time()
    features_1_cached = cache_manager.cache_features(
        dummy_feature_extractor, test_data_1, "test_features_1",
        window_size=15, feature_type='advanced'
    )
    time_2 = time.time() - start_time
    print(f"用时: {time_2:.3f}秒")

    # 验证结果一致性
    print(f"结果一致性: {str(features_1) == str(features_1_cached)}")
    print(f"加速比: {time_1/time_2:.1f}x")

    # 第三次提取不同数据
    print("\n第三次提取不同数据:")
    features_2 = cache_manager.cache_features(
        dummy_feature_extractor, test_data_2, "test_features_2",
        window_size=10, feature_type='basic'
    )

    # 打印缓存报告
    cache_manager.print_cache_report()

    # 获取详细信息
    print("\n--- 缓存详细信息 ---")
    detailed_info = cache_manager.cache.get_detailed_info()
    for info in detailed_info:
        print(f"键: {info['key'][:20]}... | 大小: {info['size_mb']:.3f}MB | "
              f"命中: {info['hit_count']}次 | 年龄: {info['age_hours']:.1f}h")

    # 清理测试
    print("\n--- 缓存清理测试 ---")
    cache_manager.cache.clear()
    print("缓存已清理")

    print("\n=== 生产级特征缓存测试完成 ===")