# 两阶段剪枝优化

## 优化内容

### 问题分析
原始两阶段剪枝存在以下问题：
1. **Block score不精确**：使用`||Q_block||_max * ||K_block||_max`作为上界估计过于粗糙
2. **固定topk策略**：所有query block使用相同的k值，不够灵活
3. **忽略注意力分布**：没有考虑实际的QK相似度

### 优化策略

实现了4种剪枝方法（在`kernels/block_sparse_attention_reference.py`中）：

#### 1. **norm** (原始方法 - 基线)
```python
score = ||Q_block||_max * ||K_block||_max
```
- 优点：计算快速，只需要范数计算
- 缺点：精度低，只是上界估计

#### 2. **exact** (精确QK计算)
```python
score = max(Q_block @ K_block.T)
```
- 优点：精确计算实际QK分数，准确度高
- 缺点：计算开销较大，需要实际矩阵乘法

#### 3. **adaptive** (自适应topk)
```python
# 保留贡献95%分数质量的blocks
cumsum_score >= 0.95 * total_score
```
- 优点：根据实际分数分布动态调整保留的block数量
- 缺点：不同query block保留的block数量不同，可能影响批处理效率

#### 4. **entropy** (基于均值的评分)
```python
score = mean(Q_block @ K_block.T)
```
- 优点：考虑整体分布而非极值，更稳定
- 缺点：可能错过重要的高分token

## 使用方法

### 1. 运行改进的剪枝实验
```bash
python experiments/exp3_improved_pruning.py
```

实验配置：
- 序列长度：2048, 4096
- Stage 1模式：bigbird, longformer, local
- Stage 2 topk：4, 8, 16
- 剪枝方法：norm, exact, adaptive, entropy

### 2. 生成可视化结果
```bash
python visualization/plot_improved_pruning.py
```

生成6个子图：
1. Speedup vs Error权衡
2. 各方法平均加速比
3. 各方法平均误差
4. Speedup随topk变化
5. Error随topk变化
6. 稀疏度分布

### 3. 在自己的代码中使用优化剪枝

```python
from utils.csr_utils import make_csr
from kernels.block_sparse_attention_reference import prune_csr_by_block_score
from kernels.triton_block_sparse import block_sparse_attention

# Stage 1: 创建稀疏模式
csr_stage1 = make_csr('bigbird', config, device)

# Stage 2: 使用不同方法剪枝
# 方法1: 精确QK计算（推荐用于准确度优先场景）
csr_exact = prune_csr_by_block_score(q, k, csr_stage1, topk=8, method='exact')

# 方法2: 自适应topk（推荐用于动态场景）
csr_adaptive = prune_csr_by_block_score(q, k, csr_stage1, topk=8, method='adaptive')

# 方法3: 基于熵的评分（推荐用于稳定性优先场景）
csr_entropy = prune_csr_by_block_score(q, k, csr_stage1, topk=8, method='entropy')

# 运行注意力计算
output = block_sparse_attention(q, k, v, csr_exact, use_triton=True)
```

## 预期改进效果

基于实验设计，预期各方法表现：

| 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|----------|
| norm | ⭐⭐ | ⭐⭐⭐⭐⭐ | 速度优先，可接受较大误差 |
| exact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 准确度优先，计算资源充足 |
| adaptive | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 平衡性能和准确度 |
| entropy | ⭐⭐⭐⭐ | ⭐⭐⭐ | 需要稳定的注意力分布 |

## 结果文件

实验结果保存在：
- `results/exp3_improved_pruning/results.csv` - 详细实验数据
- `results/exp3_improved_pruning/improved_pruning_analysis.png` - 可视化分析图

## 下一步优化方向

1. **GPU加速剪枝**：将剪枝逻辑移到Triton kernel中，避免CPU-GPU数据传输
2. **混合策略**：结合多种方法的优点，如先用norm快速筛选，再用exact精确排序
3. **学习型剪枝**：使用小型神经网络预测重要的blocks
4. **缓存优化**：对静态或慢变化的模式缓存剪枝结果

## 代码修改说明

### 修改的文件
1. `kernels/block_sparse_attention_reference.py`
   - 扩展`prune_csr_by_block_score`函数，添加`method`参数
   - 实现4种剪枝策略

### 新增的文件
1. `experiments/exp3_improved_pruning.py` - 改进剪枝实验脚本
2. `visualization/plot_improved_pruning.py` - 可视化脚本
3. `results/exp3_improved_pruning/README.md` - 本文档

## 参数调优建议

### topk选择
- **topk=4**：极端稀疏，适合超长序列（>16K）
- **topk=8**：平衡选择，推荐用于大多数场景
- **topk=16**：保守选择，准确度优先

### 方法选择
- **推理场景**：使用`exact`方法，topk=8-16
- **训练场景**：使用`adaptive`方法，动态调整
- **实时应用**：使用`norm`方法，topk=4-8
- **研究分析**：使用`entropy`方法，理解注意力分布

## 性能优化技巧

1. **批量处理**：对多个query使用相同的剪枝策略，提高GPU利用率
2. **预计算**：对静态K缓存范数，避免重复计算
3. **异步执行**：剪枝和注意力计算可以流水线化
4. **精度权衡**：使用fp16进行剪枝计算，fp32进行注意力计算
