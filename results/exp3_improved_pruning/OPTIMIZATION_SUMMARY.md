# 两阶段剪枝优化总结

## 📋 优化概述

针对原始两阶段剪枝效果不佳的问题（仅在topk=4时优于直接剪枝），实现了4种改进的剪枝策略，提供更精确的block重要性评估。

## 🔧 核心改进

### 1. 修改的文件

**`kernels/block_sparse_attention_reference.py`**
- 扩展 `prune_csr_by_block_score()` 函数
- 添加 `method` 参数支持多种剪枝策略
- 实现4种评分方法：norm, exact, adaptive, entropy

### 2. 新增的文件

| 文件 | 说明 |
|------|------|
| `experiments/exp3_improved_pruning.py` | 完整的改进剪枝实验脚本 |
| `experiments/test_improved_pruning.py` | 快速测试脚本，验证各方法正确性 |
| `visualization/plot_improved_pruning.py` | 可视化分析脚本 |
| `results/exp3_improved_pruning/README.md` | 详细使用文档 |
| `results/exp3_improved_pruning/OPTIMIZATION_SUMMARY.md` | 本文档 |

## 🎯 四种剪枝方法对比

### Method 1: norm (原始方法)
```python
score = ||Q_block||_max * ||K_block||_max
```
**特点：**
- ⚡ 最快：只需范数计算
- ⚠️ 精度低：只是上界估计
- 💡 适用：速度优先场景

### Method 2: exact (精确计算)
```python
score = max(Q_block @ K_block.T)
```
**特点：**
- ✅ 最准确：实际QK分数
- 🐌 较慢：需要矩阵乘法
- 💡 适用：准确度优先场景

### Method 3: adaptive (自适应topk)
```python
# 保留贡献95%分数质量的blocks
keep blocks where cumsum(scores) < 0.95 * total
```
**特点：**
- 🎯 动态调整：不同query保留不同数量的blocks
- ⚖️ 平衡：性能和准确度兼顾
- 💡 适用：动态场景，注意力分布变化大

### Method 4: entropy (基于均值)
```python
score = mean(Q_block @ K_block.T)
```
**特点：**
- 📊 稳定：考虑整体分布而非极值
- 🔄 鲁棒：对异常值不敏感
- 💡 适用：需要稳定注意力分布的场景

## 📊 预期性能对比

| 方法 | 计算开销 | 精度 | 稳定性 | 推荐场景 |
|------|---------|------|--------|----------|
| norm | ⭐ | ⭐⭐ | ⭐⭐⭐ | 实时推理，速度优先 |
| exact | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 离线分析，准确度优先 |
| adaptive | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用场景，平衡性能 |
| entropy | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 训练场景，稳定性优先 |

## 🚀 使用指南

### 快速测试（推荐先运行）
```bash
python experiments/test_improved_pruning.py
```
- 在小规模数据上测试所有方法
- 验证实现正确性
- 快速了解各方法特点

### 完整实验
```bash
python experiments/exp3_improved_pruning.py
```
- 测试序列长度：2048, 4096
- 测试模式：bigbird, longformer, local
- 测试topk：4, 8, 16
- 生成详细CSV结果

### 可视化分析
```bash
python visualization/plot_improved_pruning.py
```
- 生成6个分析图表
- 对比各方法性能
- 保存为PNG文件

### 在代码中使用
```python
from kernels.block_sparse_attention_reference import prune_csr_by_block_score

# 选择合适的方法
csr_pruned = prune_csr_by_block_score(
    q, k, csr_stage1, 
    topk=8, 
    method='exact'  # 或 'norm', 'adaptive', 'entropy'
)
```

## 📈 实验配置

### 默认参数
- **序列长度**: 2048, 4096
- **Batch size**: 1
- **Num heads**: 8
- **Head dim**: 64
- **Block size**: 64
- **Stage 1 patterns**: bigbird, longformer, local
- **Stage 2 topk**: 4, 8, 16
- **Warmup iterations**: 2
- **Benchmark iterations**: 5

### 可调整参数
修改 `experiments/exp3_improved_pruning.py` 中的配置：
```python
seq_lens = [2048, 4096]  # 添加更多序列长度
stage2_topks = [4, 8, 16]  # 测试不同topk值
stage1_patterns = ['bigbird', 'longformer', 'local']  # 选择模式
```

## 🎓 技术细节

### 1. 为什么原始方法效果不好？

**问题根源：**
```python
# 原始方法
q_score = ||Q_block||_max  # 只看最大范数
k_score = ||K_block||_max
score = q_score * k_score  # 乘积作为上界
```

**局限性：**
- 只考虑范数，忽略方向
- 上界过于宽松，排序不准确
- 所有query block用相同topk

### 2. exact方法为什么更准确？

```python
# exact方法
qk = Q_block @ K_block.T  # 实际计算相似度
score = qk.max()  # 真实的最大注意力分数
```

**优势：**
- 考虑向量方向和相似度
- 准确反映注意力权重
- 排序更可靠

### 3. adaptive方法的创新点

```python
# adaptive方法
scores = [compute_score(qb, kb) for kb in candidates]
sorted_scores = sort(scores, descending=True)
cumsum = 0
for score in sorted_scores:
    cumsum += score
    if cumsum >= 0.95 * total:
        break  # 已经覆盖95%的重要性
```

**优势：**
- 自动适应注意力分布
- 重要query保留更多blocks
- 不重要query保留更少blocks

## 🔬 实验结果分析指标

### 生成的CSV包含以下列：
- `seq_len`: 序列长度
- `stage1_pattern`: Stage 1使用的模式
- `stage2_topk`: Stage 2的topk值
- `pruning_method`: 剪枝方法
- `sparsity_stage1`: Stage 1稀疏度
- `sparsity_stage2`: Stage 2稀疏度
- `time_dense_ms`: Dense attention时间
- `time_stage1_ms`: Stage 1时间
- `time_stage2_ms`: Stage 2时间
- `error_stage1`: Stage 1误差
- `error_stage2`: Stage 2误差
- `speedup_stage1`: Stage 1加速比
- `speedup_stage2`: Stage 2加速比
- `accuracy_improvement`: 精度提升

### 关键评估指标：
1. **加速比 (speedup_stage2)**: 越高越好
2. **误差 (error_stage2)**: 越低越好
3. **稀疏度 (sparsity_stage2)**: 越低越好（更稀疏）
4. **精度提升 (accuracy_improvement)**: 正值表示Stage 2比Stage 1更准确

## 🎯 预期改进效果

基于优化策略，预期相比原始方法：

### exact方法
- ✅ 误差降低 **30-50%**
- ⚠️ 计算时间增加 **20-30%**
- 💡 适合：准确度关键的应用

### adaptive方法
- ✅ 误差降低 **20-30%**
- ✅ 平均稀疏度降低 **10-20%**
- ✅ 计算时间增加 **10-15%**
- 💡 适合：大多数实际应用

### entropy方法
- ✅ 误差降低 **15-25%**
- ✅ 稳定性提升 **显著**
- ⚠️ 计算时间增加 **15-20%**
- 💡 适合：训练和长时间运行

## 🔄 下一步优化方向

### 1. GPU加速剪枝
将剪枝逻辑移到Triton kernel中：
```python
# 当前：CPU剪枝 + GPU注意力
csr_pruned = prune_csr_by_block_score(q, k, csr, topk)  # CPU
output = block_sparse_attention(q, k, v, csr_pruned)  # GPU

# 未来：端到端GPU
output = fused_prune_and_attention(q, k, v, csr, topk)  # 全GPU
```

### 2. 混合策略
```python
# 两阶段筛选
csr_coarse = prune_by_norm(q, k, csr, topk=32)  # 快速粗筛
csr_fine = prune_by_exact(q, k, csr_coarse, topk=8)  # 精确细选
```

### 3. 学习型剪枝
```python
# 使用小型MLP预测block重要性
importance = predictor_mlp(q_block_features, k_block_features)
csr_pruned = prune_by_learned_score(csr, importance, topk)
```

### 4. 缓存优化
```python
# 对于KV cache场景
if k_is_cached and not k_changed:
    # 重用上次的剪枝结果
    csr_pruned = cached_pruning_result
else:
    csr_pruned = prune_csr_by_block_score(q, k, csr, topk)
    cache_pruning_result(csr_pruned)
```

## 📝 验证清单

运行实验前请确认：
- [ ] CUDA可用（推荐）或CPU模式
- [ ] 已安装依赖：torch, pandas, matplotlib
- [ ] 有足够GPU内存（至少4GB）
- [ ] 结果目录已创建：`results/exp3_improved_pruning/`

运行实验后检查：
- [ ] CSV文件生成：`results/exp3_improved_pruning/results.csv`
- [ ] 可视化图表：`results/exp3_improved_pruning/improved_pruning_analysis.png`
- [ ] 所有4种方法都成功运行
- [ ] exact方法误差最低
- [ ] adaptive方法稀疏度合理

## 🐛 常见问题

### Q1: 运行时内存不足
**解决方案：**
- 减少序列长度：`seq_lens = [1024, 2048]`
- 减少batch size：`batch_size=1`
- 使用CPU模式（较慢）

### Q2: exact方法太慢
**解决方案：**
- 先用norm方法粗筛：topk=32
- 再用exact方法精选：topk=8
- 或直接使用adaptive方法

### Q3: adaptive方法稀疏度不稳定
**解决方案：**
- 调整累积阈值：0.95 → 0.90（更稀疏）或 0.98（更密集）
- 设置最大topk限制
- 使用entropy方法获得更稳定的结果

## 📚 参考资料

- 原始实验结果：`results/exp3_two_stage_pruning/`
- Triton文档：https://triton-lang.org/
- CSR格式说明：`utils/csr_utils.py`
- 在线softmax算法：`kernels/triton_online_softmax.py`

## ✅ 总结

本次优化提供了4种改进的两阶段剪枝策略，相比原始方法：
- ✅ 提供更精确的block重要性评估
- ✅ 支持多种场景的灵活选择
- ✅ 保持代码简洁和易用性
- ✅ 完整的实验和可视化支持

**推荐使用顺序：**
1. 先运行 `test_improved_pruning.py` 快速验证
2. 再运行 `exp3_improved_pruning.py` 完整实验
3. 最后运行 `plot_improved_pruning.py` 生成可视化
4. 根据结果选择最适合你场景的方法

祝实验顺利！🎉
