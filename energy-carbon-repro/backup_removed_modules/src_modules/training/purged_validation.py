
import numpy as np

class PurgedWalkForwardCV:
    """
    时序数据专用的步进交叉验证方法，带有清洗(Purging)和禁运(Embargo)期。
    确保训练集和验证集之间有明确的时间间隔，防止因标签重叠等问题导致的数据泄露。
    """
    def __init__(self, n_splits=5, embargo_size=100):
        """
        初始化交叉验证分割器。

        Args:
            n_splits (int): 折数，即生成(训练集, 验证集)的对数。
            embargo_size (int): 禁运期大小。在训练集结束和验证集开始之间的样本数间隔，
                                用于防止信息泄露。
        """
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        返回交叉验证的折数。
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        生成训练集和验证集的索引。

        Args:
            X (array-like): 数据集，只需要其形状来确定样本总数。

        Yields:
            tuple: (train_indices, validation_indices) 对。
        """
        n_samples = len(X)
        
        # 我们将数据分成 n_splits+1 个大致相等的部分。
        # 每次迭代，我们使用更多的数据进行训练，并用紧邻的一块数据进行验证。
        # 最后一块数据通常被保留为最终的测试集，不参与CV。
        fold_size = n_samples // (self.n_splits + 1)

        if fold_size <= self.embargo_size:
            raise ValueError(
                f"fold_size={fold_size} must be greater than embargo_size={self.embargo_size}. "
                f"Consider reducing n_splits or embargo_size."
            )

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # 训练集从头开始，每次向后延伸一个fold
            train_end = (i + 1) * fold_size
            train_indices = indices[0:train_end]

            # 验证集在训练集之后，并跳过一个禁运期
            val_start = train_end + self.embargo_size
            val_end = val_start + fold_size
            
            # 确保验证集不会超出样本范围
            if val_end > n_samples:
                # 如果这是最后一个可能的折叠，就用到结尾
                val_end = n_samples
            
            val_indices = indices[val_start:val_end]

            # 如果验证集为空，则停止迭代
            if len(val_indices) == 0:
                break

            yield train_indices, val_indices

# 使用示例
if __name__ == '__main__':
    # 创建一个有1000个样本的模拟数据集
    X_dummy = np.random.rand(1000, 2)
    n_folds = 5
    embargo = 10

    print(f"--- PurgedWalkForwardCV Demonstration (n_samples=1000, n_splits={n_folds}, embargo={embargo}) ---")
    
    purged_cv = PurgedWalkForwardCV(n_splits=n_folds, embargo_size=embargo)

    for i, (train_idx, val_idx) in enumerate(purged_cv.split(X_dummy)):
        print(f"\nFold {i+1}/{n_folds}")
        print(f"  Train indices: {train_idx[0]} ... {train_idx[-1]} (size: {len(train_idx)})")
        print(f"  Validation indices: {val_idx[0]} ... {val_idx[-1]} (size: {len(val_idx)})")
        print(f"  Gap between train_end and val_start: {val_idx[0] - train_idx[-1] - 1} (>= embargo_size)")
