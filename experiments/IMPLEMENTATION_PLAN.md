# 实验实现计划

## 代码结构扩展

```
SparseAttentionProject/
├── kernels/
│   ├── triton_online_softmax.py          # 新增：Triton在线softmax kernel
│   ├── triton_block_sparse.py            # 更新：实际的Triton实现
│   └── block_sparse_attention_reference.py
├── experiments/
│   ├── exp1_unified_framework.py         # 实验一：统一框架对比
│   ├── exp2_online_softmax.py            # 实验二：在线softmax验证
│   ├── exp3_two_stage_pruning.py         # 实验三：两阶段稀疏
│   ├── exp4_csr_encoding_overhead.py     # 实验四：编码开销
│   ├── exp5_triton_profiling.py          # 实验五：Triton性能剖析
│   ├── exp6_hybrid_patterns.py           # 实验六：混合模式
│   └── run_all_experiments.py            # 运行所有实验
├── baselines/
│   ├── flash_attention_local.py          # FlashAttention局部注意力
│   ├── xformers_patterns.py              # xFormers各种模式
│   └── naive_sparse.py                   # 朴素稀疏实现
├── visualization/
│   ├── plot_performance.py               # 性能曲线绘制
│   ├── plot_sparsity_patterns.py         # 稀疏模式可视化
│   └── plot_two_stage_analysis.py        # 两阶段分析可视化
└── results/
    ├── exp1_unified_framework/
    ├── exp2_online_softmax/
    ├── exp3_two_stage_pruning/
    ├── exp4_csr_encoding/
    ├── exp5_triton_profiling/
    └── exp6_hybrid_patterns/
```

---

## 核心实现：Triton在线Softmax Kernel

### 关键技术点

1. **CSR格式遍历**：
   - 每个warp处理一个query块
   - 根据row_ptr和col_ind动态加载key块
   - 避免访问零块

2. **在线softmax累积**：
   - 维护running max (m) 和 running sum (l)
   - 每处理一个块就更新统计量
   - 最后一次性归一化

3. **内存优化**：
   - 使用共享内存缓存Q块（复用）
   - 流式加载K/V块（只用一次）
   - 累积器保持在寄存器中

### Triton Kernel伪代码

```python
@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, O,                    # [batch, heads, seq_len, head_dim]
    csr_row_ptr, csr_col_ind,      # CSR格式
    seq_len, head_dim, block_size,
    BLOCK_M: tl.constexpr,         # Q块大小
    BLOCK_N: tl.constexpr,         # K块大小
    BLOCK_D: tl.constexpr,         # head_dim块大小
):
    # 每个program处理一个query块
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # 加载Q块到共享内存
    q_block = tl.load(Q + offsets, mask=...)  # [BLOCK_M, head_dim]
    
    # 初始化在线softmax状态
    m_i = -float('inf')  # running max
    l_i = 0.0            # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # 遍历当前query块的所有非零key块
    row_start = tl.load(csr_row_ptr + pid_m)
    row_end = tl.load(csr_row_ptr + pid_m + 1)
    
    for pos in range(row_start, row_end):
        # 获取key块索引
        k_block_idx = tl.load(csr_col_ind + pos)
        
        # 加载K块和V块
        k_block = tl.load(K + k_offsets(k_block_idx), mask=...)
        v_block = tl.load(V + v_offsets(k_block_idx), mask=...)
        
        # 计算QK^T
        qk = tl.dot(q_block, tl.trans(k_block))  # [BLOCK_M, BLOCK_N]
        
        # 更新在线softmax统计量
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # 重新缩放之前的累积值
        scale_old = tl.exp(m_i - m_new)
        acc = acc * scale_old[:, None]
        l_i = l_i * scale_old
        
        # 计算当前块的贡献
        p = tl.exp(qk - m_new[:, None])
        acc += tl.dot(p, v_block)
        l_i += tl.sum(p, axis=1)
        
        # 更新running max
        m_i = m_new
    
    # 最终归一化
    acc = acc / l_i[:, None]
    
    # 写回输出
    tl.store(O + offsets, acc, mask=...)
```

---

## 实验一：统一框架对比

### 实现步骤

```python
# experiments/exp1_unified_framework.py

def compare_unified_vs_specialized():
    """对比统一CSR框架 vs 专用实现"""
    
    patterns = ['local', 'bigbird', 'longformer']
    seq_lens = [1024, 2048, 4096, 8192]
    
    results = []
    
    for pattern in patterns:
        for seq_len in seq_lens:
            # 1. 我们的统一实现
            csr = make_csr(pattern, config)
            time_ours = benchmark(
                lambda: triton_block_sparse_attention(q, k, v, csr)
            )
            
            # 2. 专用实现（如果存在）
            time_specialized = benchmark_specialized(pattern, q, k, v)
            
            # 3. 记录结果
            results.append({
                'pattern': pattern,
                'seq_len': seq_len,
                'time_ours': time_ours,
                'time_specialized': time_specialized,
                'speedup': time_specialized / time_ours,
                'code_lines': count_lines(pattern)  # 代码复杂度
            })
    
    # 4. 测试模式切换开销
    switch_overhead = measure_pattern_switching(patterns)
    
    return pd.DataFrame(results), switch_overhead
```

### 关键指标

- **性能对比**：我们的实现 vs FlashAttention/xFormers
- **代码复杂度**：1个通用kernel vs N个专用kernel
- **模式切换开销**：切换模式时是否需要重新编译kernel

---

## 实验二：在线Softmax验证

### 实现步骤

```python
# experiments/exp2_online_softmax.py

def verify_online_softmax():
    """验证在线softmax的正确性和效率"""
    
    # 1. 正确性测试
    for sparsity in [0.05, 0.1, 0.25, 0.5, 1.0]:
        csr = random_csr(seq_len, sparsity)
        
        # 方法A：朴素实现（物化所有中间结果）
        out_naive = naive_sparse_attention(q, k, v, csr)
        
        # 方法B：在线softmax
        out_online = triton_online_softmax(q, k, v, csr)
        
        # 验证数值精度
        max_error = (out_naive - out_online).abs().max()
        assert max_error < 1e-3, f"Error too large: {max_error}"
    
    # 2. 内存效率测试
    memory_results = []
    for seq_len in [4096, 8192, 16384, 32768]:
        mem_naive = measure_memory(naive_sparse_attention, seq_len)
        mem_online = measure_memory(triton_online_softmax, seq_len)
        
        memory_results.append({
            'seq_len': seq_len,
            'mem_naive_mb': mem_naive,
            'mem_online_mb': mem_online,
            'memory_saving': (mem_naive - mem_online) / mem_naive
        })
    
    # 3. 计算效率测试
    flops_results = benchmark_flops_utilization(
        triton_online_softmax, seq_lens=[1024, 2048, 4096]
    )
    
    return memory_results, flops_results
```

### 预期结果

- 数值误差 < 1e-3（fp16）或 < 1e-6（fp32）
- 内存节省：30-50%（长序列）
- FLOPs利用率：60-80%（接近理论峰值）

---

## 实验三：两阶段稀疏评估

### 实现步骤

```python
# experiments/exp3_two_stage_pruning.py

def evaluate_two_stage_pruning():
    """评估两阶段剪枝的效果"""
    
    # 配置
    stage1_patterns = ['bigbird', 'longformer', 'local']
    stage2_topks = [4, 8, 16, 32]
    
    results = []
    
    for pattern in stage1_patterns:
        # Stage 1: 静态模式
        csr_stage1 = make_csr(pattern, config)
        sparsity_stage1 = csr_stage1.density
        
        # 计算Stage 1输出
        out_stage1 = triton_block_sparse_attention(q, k, v, csr_stage1)
        error_stage1 = (out_stage1 - out_dense).abs().max()
        time_stage1 = benchmark(lambda: triton_block_sparse_attention(q, k, v, csr_stage1))
        
        for topk in stage2_topks:
            # Stage 2: 动态剪枝
            csr_stage2 = prune_csr_by_block_score(q, k, csr_stage1, topk)
            sparsity_stage2 = csr_stage2.density
            
            # 计算Stage 2输出
            out_stage2 = triton_block_sparse_attention(q, k, v, csr_stage2)
            error_stage2 = (out_stage2 - out_dense).abs().max()
            time_stage2 = benchmark(lambda: triton_block_sparse_attention(q, k, v, csr_stage2))
            
            # 对比：直接从dense剪枝到相同稀疏度
            csr_direct = prune_from_dense(q, k, topk)
            out_direct = triton_block_sparse_attention(q, k, v, csr_direct)
            error_direct = (out_direct - out_dense).abs().max()
            
            results.append({
                'stage1_pattern': pattern,
                'stage2_topk': topk,
                'sparsity_stage1': sparsity_stage1,
                'sparsity_stage2': sparsity_stage2,
                'error_stage1': error_stage1,
                'error_stage2': error_stage2,
                'error_direct': error_direct,
                'time_stage1': time_stage1,
                'time_stage2': time_stage2,
                'two_stage_better': error_stage2 < error_direct
            })
    
    return pd.DataFrame(results)
```

### 关键问题

1. **两阶段 vs 单阶段**：在相同最终稀疏度下，两阶段是否更准确？
2. **Stage 1模式选择**：哪种初始模式最适合作为Stage 1？
3. **剪枝开销**：Stage 2的计算开销是否值得？

---

## 实验四：CSR编码开销分析

```python
# experiments/exp4_csr_encoding_overhead.py

def analyze_csr_encoding_overhead():
    """分析CSR编码的时间开销"""
    
    results = []
    
    for pattern in ['dense', 'local', 'bigbird', 'longformer', 'grouped']:
        for seq_len in [1024, 2048, 4096, 8192, 16384]:
            config = make_config(seq_len)
            
            # 测量编码时间
            encode_time = benchmark(lambda: make_csr(pattern, config))
            
            # 测量注意力计算时间
            csr = make_csr(pattern, config)
            attention_time = benchmark(
                lambda: triton_block_sparse_attention(q, k, v, csr)
            )
            
            results.append({
                'pattern': pattern,
                'seq_len': seq_len,
                'encode_ms': encode_time,
                'attention_ms': attention_time,
                'overhead_pct': encode_time / attention_time * 100
            })
    
    return pd.DataFrame(results)
```

### 预期结果

- 编码时间 < 1% 注意力计算时间（对于seq_len <= 16K）
- 可以在模型初始化时预计算并缓存

---

## 可视化脚本

```python
# visualization/plot_performance.py

def plot_unified_framework_comparison(results_df):
    """绘制统一框架 vs 专用实现的性能对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 延迟对比
    for pattern in results_df['pattern'].unique():
        data = results_df[results_df['pattern'] == pattern]
        axes[0, 0].plot(data['seq_len'], data['time_ours'], 
                        label=f'{pattern} (Ours)', marker='o')
        axes[0, 0].plot(data['seq_len'], data['time_specialized'], 
                        label=f'{pattern} (Specialized)', marker='x', linestyle='--')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Latency (ms)')
    axes[0, 0].set_title('Latency Comparison')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log', base=2)
    
    # 2. 加速比
    speedup_data = results_df.groupby('pattern')['speedup'].mean()
    axes[0, 1].bar(speedup_data.index, speedup_data.values)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    axes[0, 1].set_ylabel('Speedup')
    axes[0, 1].set_title('Average Speedup vs Specialized')
    axes[0, 1].legend()
    
    # 3. 代码复杂度
    code_complexity = results_df.groupby('pattern')['code_lines'].first()
    axes[1, 0].bar(code_complexity.index, code_complexity.values)
    axes[1, 0].set_ylabel('Lines of Code')
    axes[1, 0].set_title('Code Complexity')
    
    # 4. 内存占用
    for pattern in results_df['pattern'].unique():
        data = results_df[results_df['pattern'] == pattern]
        axes[1, 1].plot(data['seq_len'], data['memory_mb'], 
                        label=pattern, marker='o')
    axes[1, 1].set_xlabel('Sequence Length')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('results/unified_framework_comparison.png', dpi=300)
```

---

## 运行所有实验

```python
# experiments/run_all_experiments.py

def main():
    print("=" * 80)
    print("统一CSR块稀疏注意力实验套件")
    print("=" * 80)
    
    # 实验一：统一框架对比
    print("\n[1/6] 运行实验一：统一框架对比...")
    exp1_results = compare_unified_vs_specialized()
    exp1_results.to_csv('results/exp1_unified_framework.csv')
    plot_unified_framework_comparison(exp1_results)
    
    # 实验二：在线softmax验证
    print("\n[2/6] 运行实验二：在线softmax验证...")
    exp2_memory, exp2_flops = verify_online_softmax()
    pd.DataFrame(exp2_memory).to_csv('results/exp2_memory.csv')
    pd.DataFrame(exp2_flops).to_csv('results/exp2_flops.csv')
    
    # 实验三：两阶段稀疏评估
    print("\n[3/6] 运行实验三：两阶段稀疏评估...")
    exp3_results = evaluate_two_stage_pruning()
    exp3_results.to_csv('results/exp3_two_stage.csv')
    plot_two_stage_analysis(exp3_results)
    
    # 实验四：CSR编码开销
    print("\n[4/6] 运行实验四：CSR编码开销...")
    exp4_results = analyze_csr_encoding_overhead()
    exp4_results.to_csv('results/exp4_encoding_overhead.csv')
    
    # 实验五：Triton性能剖析
    print("\n[5/6] 运行实验五：Triton性能剖析...")
    exp5_results = profile_triton_kernel()
    exp5_results.to_csv('results/exp5_triton_profiling.csv')
    
    # 实验六：混合模式探索
    print("\n[6/6] 运行实验六：混合模式探索...")
    exp6_results = explore_hybrid_patterns()
    exp6_results.to_csv('results/exp6_hybrid_patterns.csv')
    
    print("\n" + "=" * 80)
    print("所有实验完成！结果已保存到 results/ 目录")
    print("=" * 80)

if __name__ == '__main__':
    main()
```

---

## 下一步行动

1. **立即开始**：实现Triton在线softmax kernel（核心）
2. **并行进行**：准备baseline实现（FlashAttention, xFormers）
3. **逐步推进**：按实验顺序依次实现和验证
4. **持续优化**：根据实验结果调整kernel实现

需要我开始实现Triton kernel吗？
