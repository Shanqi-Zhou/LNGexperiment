# 配置文件使用指南

## 优化配置文件概述

本目录包含针对不同场景优化的配置文件，每个配置文件都经过精心调优以达到最佳性能。

### 配置文件列表

#### 1. `rtx4060_8gb.yaml` - RTX 4060 8GB专用配置
- **适用场景**: RTX 4060 8GB显卡环境
- **优化重点**: 内存效率、混合精度训练、GPU利用率最大化
- **预期性能**: 3-5x训练加速，40-60%内存减少
- **推荐数据规模**: 10K-100K样本

#### 2. `small_dataset.yaml` - 小数据集优化配置
- **适用场景**: ≤10,000样本的小数据集
- **优化策略**: MLR + Exact GPR残差建模
- **特点**: 精确建模、不需要复杂优化技术
- **训练时间**: <5分钟

#### 3. `medium_dataset.yaml` - 中等数据集配置
- **适用场景**: 10,000-100,000样本
- **优化策略**: HGBR + Nyström GPR混合建模
- **特点**: 平衡准确性和效率
- **训练时间**: 10-20分钟

#### 4. `large_dataset.yaml` - 大数据集配置
- **适用场景**: >100,000样本
- **优化策略**: HGBR + Local-SVGP + TCN+Attention集成
- **特点**: 最大化模型容量和优化技术
- **训练时间**: 1-2小时

#### 5. `production.yaml` - 生产环境配置
- **适用场景**: 生产部署环境
- **优化重点**: 稳定性、可靠性、快速推理
- **特点**: 保守设置、故障处理、监控集成
- **推理时间**: <5ms

#### 6. `benchmark.yaml` - 基准测试配置
- **适用场景**: 性能测试和对比分析
- **特点**: 多维度测试、详细性能分析
- **用途**: 验证优化效果、硬件对比

#### 7. `debug.yaml` - 开发调试配置
- **适用场景**: 开发阶段快速迭代
- **特点**: 快速训练、详细日志、错误调试
- **训练时间**: <30秒

### 使用指南

#### 基础使用

```bash
# 使用RTX 4060优化配置
python main.py --config configs/optimized/rtx4060_8gb.yaml --data-dir data --output-dir results

# 小数据集快速训练
python main.py --config configs/optimized/small_dataset.yaml --data-dir data/small --quick

# 生产环境部署
python main.py --config configs/optimized/production.yaml --mode deploy
```

#### 高级使用

```bash
# 基准测试
python scripts/benchmark.py --config configs/optimized/benchmark.yaml

# 开发调试
python main.py --config configs/optimized/debug.yaml --debug --verbose
```

#### 配置文件定制

1. **复制基础配置**
   ```bash
   cp configs/optimized/rtx4060_8gb.yaml configs/my_config.yaml
   ```

2. **修改关键参数**
   - 调整 `batch_size` 和 `learning_rate`
   - 修改 `model.type` 选择不同模型
   - 调整 `performance_targets` 设定目标

3. **验证配置**
   ```bash
   python scripts/validate_config.py --config configs/my_config.yaml
   ```

### 性能目标对照表

| 配置文件 | 速度提升 | 内存减少 | R²目标 | 推理时间 |
|---------|---------|---------|--------|----------|
| RTX4060 | 3-5x    | 40-60%  | ≥0.75  | <10ms    |
| Small   | 2-3x    | 20-30%  | ≥0.70  | <5ms     |
| Medium  | 3-4x    | 30-40%  | ≥0.80  | <8ms     |
| Large   | 4-6x    | 50-60%  | ≥0.85  | <15ms    |
| Production | 2x   | 20%     | ≥0.75  | <5ms     |

### 硬件要求

#### 最低要求
- CPU: 4核心
- 内存: 8GB
- GPU: 可选

#### 推荐配置
- CPU: 8核心
- 内存: 16GB
- GPU: RTX 4060 8GB或更高

#### 高性能配置
- CPU: 16核心
- 内存: 32GB
- GPU: RTX 4090 24GB

### 常见问题

#### Q: 如何选择合适的配置文件？
A: 根据数据规模和硬件环境：
- 数据<10K：使用 `small_dataset.yaml`
- 数据10K-100K：使用 `medium_dataset.yaml`
- 数据>100K：使用 `large_dataset.yaml`
- RTX 4060用户：优先 `rtx4060_8gb.yaml`

#### Q: 配置文件可以组合使用吗？
A: 可以，使用 `--config` 参数指定多个配置文件：
```bash
python main.py --config configs/optimized/rtx4060_8gb.yaml --config configs/optimized/production.yaml
```

#### Q: 如何调试配置问题？
A: 使用debug配置进行快速验证：
```bash
python main.py --config configs/optimized/debug.yaml --data-dir data
```

### 更新日志

- **v1.0**: 初始版本，包含7种优化配置
- 针对RTX 4060 8GB环境深度优化
- 实现3-5x训练加速目标
- 集成所有Phase 1-4优化组件