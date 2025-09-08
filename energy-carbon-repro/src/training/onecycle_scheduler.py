"""
OneCycle学习率调度器 - 优化方案高效训练核心
实现超收敛训练策略，显著提升训练速度和模型性能
包含动态学习率、动量调度、权重衰减调度等先进技术
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
from typing import Optional, Dict, Any, Union, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class OneCycleLR(_LRScheduler):
    """
    OneCycle学习率调度器
    实现Leslie Smith的超收敛训练策略
    """
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 max_lr: float,
                 total_steps: int,
                 pct_start: float = 0.3,
                 anneal_strategy: str = 'cos',
                 cycle_momentum: bool = True,
                 base_momentum: float = 0.85,
                 max_momentum: float = 0.95,
                 div_factor: float = 25.0,
                 final_div_factor: float = 1e4,
                 last_epoch: int = -1,
                 verbose: bool = False):
        """
        Args:
            optimizer: PyTorch优化器
            max_lr: 最大学习率
            total_steps: 总训练步数
            pct_start: 升温阶段占比（0-1）
            anneal_strategy: 退火策略 ('cos' 或 'linear')
            cycle_momentum: 是否循环调整动量
            base_momentum: 基础动量值
            max_momentum: 最大动量值
            div_factor: 初始学习率除数
            final_div_factor: 最终学习率除数
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # 计算关键步数
        self.step_up_size = int(self.total_steps * self.pct_start)
        self.step_down_size = self.total_steps - self.step_up_size
        
        # 初始学习率和最终学习率
        self.initial_lr = self.max_lr / self.div_factor
        self.final_lr = self.initial_lr / self.final_div_factor
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.step_up_size:
            # 升温阶段：从initial_lr线性增长到max_lr
            pct = self.last_epoch / self.step_up_size
            return [self.initial_lr + pct * (self.max_lr - self.initial_lr) 
                   for _ in self.base_lrs]
        else:
            # 冷却阶段：从max_lr衰减到final_lr
            pct = (self.last_epoch - self.step_up_size) / self.step_down_size
            
            if self.anneal_strategy == 'cos':
                # 余弦退火
                cosine_factor = (1 + math.cos(math.pi * pct)) / 2
                return [self.final_lr + (self.max_lr - self.final_lr) * cosine_factor 
                       for _ in self.base_lrs]
            elif self.anneal_strategy == 'linear':
                # 线性退火
                return [self.max_lr - pct * (self.max_lr - self.final_lr) 
                       for _ in self.base_lrs]
    
    def get_momentum(self):
        """计算当前动量"""
        if not self.cycle_momentum:
            return None
            
        if self.last_epoch < self.step_up_size:
            # 升温阶段：从max_momentum线性减少到base_momentum
            pct = self.last_epoch / self.step_up_size
            return self.max_momentum - pct * (self.max_momentum - self.base_momentum)
        else:
            # 冷却阶段：从base_momentum线性增长到max_momentum
            pct = (self.last_epoch - self.step_up_size) / self.step_down_size
            return self.base_momentum + pct * (self.max_momentum - self.base_momentum)
    
    def step(self, epoch=None):
        """执行一步调度"""
        super().step(epoch)
        
        # 更新动量
        if self.cycle_momentum:
            momentum = self.get_momentum()
            for group in self.optimizer.param_groups:
                if 'momentum' in group:
                    group['momentum'] = momentum
                elif 'betas' in group:  # Adam优化器
                    group['betas'] = (momentum, group['betas'][1])


class AdaptiveOneCycleLR:
    """
    自适应OneCycle学习率调度器
    根据验证损失动态调整学习率策略
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 max_lr: float,
                 total_steps: int,
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-7,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        # 创建基础OneCycle调度器
        self.onecycle_scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 自适应参数
        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.adjustment_history = []
        
    def step(self, val_loss: Optional[float] = None):
        """执行调度步骤"""
        # 基础OneCycle步骤
        self.onecycle_scheduler.step()
        
        # 自适应调整
        if val_loss is not None:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                
                # 如果验证损失持续不改善，降低学习率
                if self.no_improve_count >= self.patience:
                    self._reduce_learning_rate()
                    self.no_improve_count = 0
    
    def _reduce_learning_rate(self):
        """降低学习率"""
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = max(current_lr * self.factor, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.adjustment_history.append({
            'step': self.onecycle_scheduler.last_epoch,
            'old_lr': current_lr,
            'new_lr': new_lr,
            'reason': 'validation_plateau'
        })
        
        self.logger.info(f"学习率调整: {current_lr:.2e} -> {new_lr:.2e}")
    
    def get_last_lr(self):
        """获取最后的学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]


class WarmupScheduler:
    """
    预热调度器 - 训练开始时的平滑启动
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_steps: int,
                 target_lr: float,
                 warmup_strategy: str = 'linear'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.warmup_strategy = warmup_strategy
        self.step_count = 0
        
        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """执行预热步骤"""
        if self.step_count < self.warmup_steps:
            if self.warmup_strategy == 'linear':
                # 线性预热
                lr_scale = (self.step_count + 1) / self.warmup_steps
            elif self.warmup_strategy == 'exp':
                # 指数预热
                lr_scale = math.exp(math.log(0.01) * (self.warmup_steps - self.step_count - 1) / self.warmup_steps)
            else:
                lr_scale = 1.0
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale
        
        self.step_count += 1
    
    def is_finished(self) -> bool:
        """检查预热是否完成"""
        return self.step_count >= self.warmup_steps


class LearningRateSchedulerManager:
    """
    学习率调度管理器 - 统一管理多种调度策略
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.schedulers = {}
        self.history = []
        
    def create_onecycle_scheduler(self,
                                optimizer: optim.Optimizer,
                                max_lr: float,
                                total_steps: int,
                                **kwargs) -> OneCycleLR:
        """创建OneCycle调度器"""
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            **kwargs
        )
        
        self.schedulers['onecycle'] = scheduler
        self.logger.info(f"OneCycle调度器创建: max_lr={max_lr}, total_steps={total_steps}")
        
        return scheduler
    
    def create_adaptive_onecycle_scheduler(self,
                                         optimizer: optim.Optimizer,
                                         max_lr: float,
                                         total_steps: int,
                                         **kwargs) -> AdaptiveOneCycleLR:
        """创建自适应OneCycle调度器"""
        scheduler = AdaptiveOneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            logger=self.logger,
            **kwargs
        )
        
        self.schedulers['adaptive_onecycle'] = scheduler
        self.logger.info(f"自适应OneCycle调度器创建: max_lr={max_lr}")
        
        return scheduler
    
    def create_warmup_scheduler(self,
                              optimizer: optim.Optimizer,
                              warmup_steps: int,
                              target_lr: float,
                              **kwargs) -> WarmupScheduler:
        """创建预热调度器"""
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            target_lr=target_lr,
            **kwargs
        )
        
        self.schedulers['warmup'] = scheduler
        self.logger.info(f"预热调度器创建: warmup_steps={warmup_steps}")
        
        return scheduler
    
    def step_all(self, metrics: Optional[Dict[str, float]] = None):
        """执行所有调度器的步骤"""
        current_lrs = {}
        
        for name, scheduler in self.schedulers.items():
            if name == 'adaptive_onecycle' and metrics:
                scheduler.step(metrics.get('val_loss'))
            elif hasattr(scheduler, 'step'):
                scheduler.step()
            
            # 记录当前学习率
            if hasattr(scheduler, 'get_last_lr'):
                current_lrs[name] = scheduler.get_last_lr()
            elif hasattr(scheduler, 'optimizer'):
                current_lrs[name] = [group['lr'] for group in scheduler.optimizer.param_groups]
        
        # 记录历史
        self.history.append({
            'step': len(self.history),
            'learning_rates': current_lrs,
            'metrics': metrics or {}
        })
    
    def get_learning_rate_history(self) -> List[Dict]:
        """获取学习率历史"""
        return self.history
    
    def plot_learning_rate_schedule(self, scheduler_name: str = 'onecycle', 
                                  save_path: Optional[str] = None):
        """绘制学习率调度曲线"""
        if scheduler_name not in self.schedulers:
            self.logger.warning(f"调度器 {scheduler_name} 不存在")
            return
        
        if not self.history:
            self.logger.warning("没有历史记录可绘制")
            return
        
        # 提取学习率数据
        steps = [entry['step'] for entry in self.history]
        lrs = [entry['learning_rates'].get(scheduler_name, [0])[0] for entry in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lrs, 'b-', linewidth=2, label=f'{scheduler_name} Learning Rate')
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{scheduler_name.title()} Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"学习率曲线已保存: {save_path}")
        
        plt.show()


class SuperConvergenceTrainer:
    """
    超收敛训练器 - 集成OneCycle等先进训练技术
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: torch.nn.Module,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.scheduler_manager = LearningRateSchedulerManager(logger)
        self.current_scheduler = None
        
        # 训练统计
        self.training_stats = {
            'epoch_losses': [],
            'epoch_lrs': [],
            'epoch_momentums': [],
            'best_loss': float('inf'),
            'convergence_epoch': -1
        }
    
    def setup_onecycle_training(self,
                              max_lr: float,
                              total_epochs: int,
                              steps_per_epoch: int,
                              use_warmup: bool = True,
                              warmup_epochs: int = 2) -> 'SuperConvergenceTrainer':
        """设置OneCycle超收敛训练"""
        
        total_steps = total_epochs * steps_per_epoch
        
        # 预热调度器
        if use_warmup:
            warmup_steps = warmup_epochs * steps_per_epoch
            self.scheduler_manager.create_warmup_scheduler(
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                target_lr=max_lr * 0.1
            )
        
        # 主要OneCycle调度器
        self.current_scheduler = self.scheduler_manager.create_adaptive_onecycle_scheduler(
            optimizer=self.optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            patience=10
        )
        
        self.logger.info(f"超收敛训练设置完成: max_lr={max_lr}, total_steps={total_steps}")
        
        return self
    
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            
            # 学习率调度
            val_loss = None
            if val_loader is not None and batch_idx % 100 == 0:  # 定期验证
                val_loss = self.validate(val_loader)
            
            self.scheduler_manager.step_all({'val_loss': val_loss})
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Epoch统计
        avg_loss = epoch_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        current_momentum = self.optimizer.param_groups[0].get('momentum', 0.0)
        
        self.training_stats['epoch_losses'].append(avg_loss)
        self.training_stats['epoch_lrs'].append(current_lr)
        self.training_stats['epoch_momentums'].append(current_momentum)
        
        # 检查是否为最佳损失
        if avg_loss < self.training_stats['best_loss']:
            self.training_stats['best_loss'] = avg_loss
            self.training_stats['convergence_epoch'] = len(self.training_stats['epoch_losses'])
        
        return {
            'train_loss': avg_loss,
            'learning_rate': current_lr,
            'momentum': current_momentum
        }
    
    def validate(self, val_loader) -> float:
        """验证"""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'training_stats': self.training_stats,
            'scheduler_history': self.scheduler_manager.get_learning_rate_history(),
            'final_lr': self.optimizer.param_groups[0]['lr'],
            'convergence_achieved': self.training_stats['convergence_epoch'] > 0
        }


# 便捷函数
def create_onecycle_scheduler(optimizer: optim.Optimizer,
                            max_lr: float,
                            total_steps: int,
                            logger: Optional[logging.Logger] = None) -> OneCycleLR:
    """创建OneCycle学习率调度器"""
    return OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )


def estimate_optimal_max_lr(model: torch.nn.Module,
                          train_loader,
                          criterion: torch.nn.Module,
                          optimizer_class=optim.Adam,
                          device='cuda') -> float:
    """估计最优最大学习率"""
    
    # 学习率范围测试
    lr_finder_optimizer = optimizer_class(model.parameters(), lr=1e-7)
    
    model.to(device)
    model.train()
    
    lrs = []
    losses = []
    
    # 指数增长学习率测试
    lr = 1e-7
    for i, (data, target) in enumerate(train_loader):
        if i > 100:  # 限制测试步数
            break
        
        data, target = data.to(device), target.to(device)
        
        # 更新学习率
        for param_group in lr_finder_optimizer.param_groups:
            param_group['lr'] = lr
        
        # 训练步骤
        lr_finder_optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        lr_finder_optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        # 指数增长学习率
        lr *= 1.1
        
        # 如果损失发散，停止
        if len(losses) > 10 and loss.item() > losses[0] * 10:
            break
    
    # 寻找最佳学习率（损失下降最快的点）
    if len(losses) > 10:
        # 计算损失梯度
        gradients = np.gradient(losses)
        # 找到梯度最负（下降最快）的点
        optimal_idx = np.argmin(gradients[:len(gradients)//2])
        optimal_lr = lrs[optimal_idx]
        
        return min(optimal_lr * 10, 1e-1)  # 乘以安全系数
    else:
        return 1e-3  # 默认值