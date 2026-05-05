# 统一CSR块稀疏注意力实验设计

## 核心创新点

本项目的特别之处在于：
1. **统一CSR表示**：将所有块稀疏模式（local, bigbird, longformer等）统一编码为CSR格式
2. **通用Triton kernel**：单一kernel处理所有模式，而非每种模式一个专用实现
3. **在线softmax块级累积**：使用online softmax算法高效处理任意稀疏模式
4. **两阶段动态稀疏**：静态模式 + 动态剪枝的组合策略

## 实验设计

### 实验一：统一CSR框架 vs 专用实现

**目标**：证明统一CSR表示的通用性和效率

**对比组**：
- **Baseline**: 每种模式的专用实现（如FlashAttention的local、xFormers的各种模式）
- **Ours**: 统一CSR + 通用Triton kernel

**测试维度**：
```python
configs = {
    'seq_lens': [1024, 2048, 4096, 8192, 16384],
    'patterns': ['local', 'bigbird', 'longformer', 'grouped', 'custom_hybrid'],
    'block_sizes': [32, 64, 128],
    'batch_sizes': [1, 4, 8],
    'num_heads': [8, 16, 32]
}
```

**评估指标**：
- 延迟（ms）
- 吞吐量（tokens/s）
- 内存占用（MB）
- 代码复杂度（LoC, kernel数量）
- 模式切换开销（同一kernel处理不同模式的优势）

**预期结果**：
- 统一kernel在模式切换时零开销
- 代码量显著减少（1个kernel vs N个专用kernel）
- 性能与专用实现相当或更优（得益于Triton优化）

---

### 实验二：在线Softmax块级累积的正确性与效率

**目标**：验证在线softmax算法在块稀疏场景下的优势

**算法对比**：
- **方法A（朴素）**: 先计算所有QK分数 → 应用稀疏掩码 → softmax → 乘V
- **方法B（在线softmax）**: 按CSR行遍历，逐块累积 (m, l) 统计量 → 一次性归一化

**实现细节**：
```python
# 在线softmax伪代码
for q_block in range(num_q_blocks):
    m_i = -inf  # running max
    l_i = 0.0   # running sum of exp
    acc = zeros(block_size, head_dim)
    
    for k_block in csr.row_blocks(q_block):  # 只遍历非零块
        qk = Q[q_block] @ K[k_block].T  # [bs, bs]
        m_new = max(m_i, qk.max())
        
        # 重新缩放之前的累积值
        scale_old = exp(m_i - m_new)
        acc *= scale_old
        l_i *= scale_old
        
        # 累积当前块
        p = exp(qk - m_new)
        acc += p @ V[k_block]
        l_i += p.sum()
        m_i = m_new
    
    O[q_block] = acc / l_i
```

**测试场景**：
- 不同稀疏度（5%, 10%, 25%, 50%, 100%）
- 不同块大小（32, 64, 128）
- 长序列（8K, 16K, 32K tokens）

**评估指标**：
- 数值精度（与PyTorch softmax的最大误差）
- 内存峰值（在线算法应显著降低）
- 计算效率（FLOPs利用率）
- 长序列扩展性（O(N) vs O(N²)内存）

---

### 实验三：两阶段稀疏的有效性

**目标**：量化两阶段剪枝的性能-精度权衡

**实验设置**：
```
Stage 1 (静态模式) → Stage 2 (动态剪枝) → 最终稀疏度
```

**对比方案**：
1. **Dense baseline**: 100%密集注意力
2. **Stage 1 only**: 仅使用静态模式（bigbird: 15%稀疏度）
3. **Stage 2 only**: 从dense直接剪枝到5%
4. **Two-stage**: bigbird(15%) → 动态剪枝到5%

**剪枝策略**：
- **Top-k per row**: 每个query块保留top-k个key块
- **Threshold**: 保留分数 > threshold的块
- **Adaptive**: 根据query块的注意力分布自适应调整k

**评估任务**：
- 语言建模困惑度（WikiText-103）
- 长文档问答（NarrativeQA）
- 长序列分类（LRA benchmark）

**评估指标**：
- 任务性能（准确率/困惑度）
- 推理速度（tokens/s）
- 内存占用
- 稀疏度分布（可视化哪些块被保留）

**关键问题**：
- Stage 1的模式选择如何影响Stage 2的效果？
- 两阶段是否比单阶段达到相同稀疏度时效果更好？
- 动态剪枝的开销是否值得？

---

### 实验四：CSR编码开销分析

**目标**：量化CSR编码的时间成本，证明其可忽略

**测试内容**：
- 不同模式的CSR编码时间
- 编码时间 vs 注意力计算时间的比例
- 是否可以预计算并缓存CSR掩码

**实验设置**：
```python
patterns_complexity = {
    'dense': 'O(N²)',           # 最简单
    'local': 'O(N)',            # 线性扫描
    'bigbird': 'O(N)',          # 局部+全局+随机
    'longformer': 'O(N)',       # 局部+全局
    'grouped': 'O(N)',          # 分组采样
    'custom': 'O(N·log(N))'     # 复杂启发式
}
```

**评估指标**：
- 编码时间（ms）
- 编码时间占总时间的百分比
- 不同序列长度下的扩展性

**预期结果**：
- 编码时间 < 1% 总时间（对于seq_len <= 16K）
- 可以在模型初始化时预计算静态模式

---

### 实验五：Triton Kernel性能剖析

**目标**：展示Triton实现相比PyTorch的性能提升

**对比实现**：
- **PyTorch reference**: 当前的`block_sparse_attention_reference`
- **Triton fused**: 融合的在线softmax kernel
- **Triton tiled**: 分块优化版本

**性能优化点**：
1. **内存访问优化**：
   - 共享内存缓存Q/K/V块
   - 合并全局内存访问
   - 避免中间结果物化

2. **计算优化**：
   - 融合QK计算、softmax、乘V
   - 向量化操作
   - 利用Tensor Core（fp16/bf16）

3. **CSR遍历优化**：
   - 预取下一个块的索引
   - 分支预测友好的循环结构

**测试配置**：
```python
benchmark_configs = {
    'dtypes': ['float16', 'bfloat16', 'float32'],
    'seq_lens': [1024, 2048, 4096, 8192],
    'sparsity': [0.05, 0.1, 0.25, 0.5],
    'block_sizes': [32, 64, 128]
}
```

**评估指标**：
- 端到端延迟
- 内存带宽利用率
- 计算吞吐量（TFLOPS）
- 与理论峰值的差距

**预期加速比**：
- vs PyTorch: 3-10x（得益于融合和内存优化）
- vs 专用实现: 0.8-1.2x（通用性的小代价）

---

### 实验六：混合稀疏模式探索

**目标**：展示CSR统一表示的灵活性，探索新的稀疏模式

**创新模式**：
1. **Adaptive local**: 窗口大小根据位置自适应
2. **Hierarchical**: 不同层使用不同稀疏模式
3. **Content-aware**: 基于token相似度的稀疏模式
4. **Hybrid**: 组合多种模式（如local + random + learned）

**实现示例**：
```python
def adaptive_local_csr(seq_len, block_size, importance_scores):
    """窗口大小根据token重要性自适应"""
    rows = []
    for i in range(num_blocks):
        # 重要token使用更大窗口
        window = base_window * (1 + importance_scores[i])
        rows.append(get_neighbors(i, window))
    return build_csr(rows)

def hierarchical_csr(layer_id, seq_len, block_size):
    """不同层使用不同稀疏度"""
    if layer_id < 6:  # 浅层：局部模式
        return local_window_csr(seq_len, block_size, window=2)
    else:  # 深层：全局模式
        return bigbird_csr(seq_len, block_size, ...)
```

**评估**：
- 这些模式能否用统一的CSR + Triton kernel实现？
- 性能如何？
- 是否比标准模式更好？

---

## 实验实施计划

### Phase 1: 基础设施（2周）
- [ ] 实现Triton在线softmax kernel
- [ ] 实现CSR格式的高效遍历
- [ ] 建立基准测试框架

### Phase 2: 核心实验（3周）
- [ ] 实验一：统一框架对比
- [ ] 实验二：在线softmax验证
- [ ] 实验三：两阶段稀疏评估

### Phase 3: 深入分析（2周）
- [ ] 实验四：编码开销分析
- [ ] 实验五：Triton性能剖析
- [ ] 实验六：混合模式探索

### Phase 4: 论文撰写（2周）
- [ ] 整理实验结果
- [ ] 绘制性能曲线和可视化
- [ ] 撰写论文草稿

---

## 预期贡献

1. **系统贡献**：统一的块稀疏注意力框架，支持任意CSR表示的模式
2. **算法贡献**：两阶段稀疏策略，平衡静态启发式和动态剪枝
3. **工程贡献**：高性能Triton实现，证明通用kernel可以媲美专用实现
4. **实验贡献**：全面的性能-精度权衡分析，为实际应用提供指导

---

## 关键卖点（论文摘要）

> 现有块稀疏注意力方法通常为每种稀疏模式设计专用kernel，导致代码冗余且难以扩展。
> 我们提出统一的CSR表示框架，将所有块稀疏模式编码为统一格式，并设计通用Triton kernel
> 使用在线softmax进行块级累积。进一步，我们引入两阶段稀疏策略：第一阶段使用启发式
> 模式（如bigbird）快速筛选候选块，第二阶段基于实际QK分数动态剪枝。实验表明，我们的
> 方法在保持与专用实现相当性能的同时，显著提升了灵活性和可扩展性，并在长序列任务上
> 实现了3-10x加速和50%内存节省。

