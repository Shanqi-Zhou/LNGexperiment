
import torch
import torch.nn as nn
from tqdm import tqdm

class UnifiedTrainer:
    """
    用于PyTorch模型的统一训练框架。
    集成了混合精度(AMP)、OneCycleLR、梯度裁剪等优化技术。
    """
    def __init__(self, model, train_loader, val_loader, config):
        """
        初始化训练器。

        Args:
            model (nn.Module): 需要训练的PyTorch模型。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            config (dict): 包含所有训练超参数的字典。
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training on device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 损失函数 (SmoothL1Loss, 来自文档)
        self.criterion = nn.SmoothL1Loss()

        # 优化器 (AdamW, 来自文档)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.config['learning_rate'], 
                                           weight_decay=self.config.get('weight_decay', 0.01))

        # 学习率调度器 (OneCycleLR, 来自文档)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.config['max_lr'], 
            steps_per_epoch=len(self.train_loader),
            epochs=self.config['epochs'],
            pct_start=self.config.get('pct_start', 0.1)
        )

        # 混合精度梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

    def _train_epoch(self):
        """执行单个训练轮次"""
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪 (来自文档)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _evaluate(self):
        """执行评估"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        """完整的训练流程"""
        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch()
            val_loss = self._evaluate()

            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"  -> New best model saved with val_loss: {best_val_loss:.4f}")

# 使用示例
if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    # 假设我们有一个简单的模型
    dummy_model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
    
    # 创建模拟数据加载器
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16)

    X_val = torch.randn(50, 10)
    y_val = torch.randn(50, 1)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 训练配置
    train_config = {
        'learning_rate': 1e-4,
        'max_lr': 1e-3,
        'epochs': 5,
        'weight_decay': 0.01,
        'gradient_clip': 1.0
    }

    # 初始化并开始训练
    trainer = UnifiedTrainer(dummy_model, train_loader, val_loader, train_config)
    trainer.train()
