# 统一CSR块稀疏注意力实验

本项目实现了基于统一CSR表示的块稀疏注意力机制，并提供了完整的实验套件来评估其性能。

## 核心创新点

1. **统一CSR表示**：将所有块稀疏模式（local, bigbird, longformer等）统一编码为CSR格式
2. **通用Triton kernel**：单一kernel处理所有模式，而非每种模式一个专用实现
3. **在线softmax块级累积**：使用online softmax算法高效处理任意稀疏模式
4. **两阶段动态稀疏**：静态模式 + 动态剪枝的组合策略

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行原始基准测试

```bash
# 基础运行
python experiments/run_experiment.py --device cuda

# 完整基准测试（包含两阶段剪枝）
python experiments/run_experiment.py --device cuda --seq-lens 1024 2048 4096 --patterns dense local bigbird longformer grouped --enable-two-stage
```

### 运行新实验套件

```bash
# 运行所有实验
python experiments/run_all_experiments.py

# 或单独运行某个实验
python experiments/exp1_unified_framework.py      # 实验1：统一框架对比
python experiments/exp2_online_softmax.py         # 实验2：在线softmax验证
python experiments/exp3_two_stage_pruning.py      # 实验3：两阶段稀疏评估
python experiments/exp4_csr_encoding_overhead.py  # 实验4：CSR编码开销分析
```

### 生成可视化

```bash
python visualization/plot_performance.py
```

## 实验说明

### 实验1：统一框架对比

**目标**：证明统一CSR表示的通用性和效率

**对比**：
- Triton实现（统一kernel）vs PyTorch参考实现
- 不同稀疏模式的性能
- 模式切换开销

**输出**：
- `results/exp1_unified_framework/results.csv`
- `results/exp1_unified_framework/unified_framework_comparison.png`

### 实验2：在线Softmax验证

**目标**：验证在线softmax算法的正确性和效率

**测试内容**：
- 数值精度（与PyTorch softmax对比）
- 内存效率（在线 vs 物化中间结果）
- 计算吞吐量

**输出**：
- `results/exp2_online_softmax/correctness.csv`
- `results/exp2_online_softmax/memory.csv`
- `results/exp2_online_softmax/compute.csv`
- `results/exp2_online_softmax/online_softmax_results.png`

### 实验3：两阶段稀疏评估

**目标**：量化两阶段剪枝的性能-精度权衡

**对比方案**：
1. Dense baseline（100%密集）
2. Stage 1 only（仅静态模式）
3. Stage 2 only（从dense直接剪枝）
4. Two-stage（静态 + 动态）

**输出**：
- `results/exp3_two_stage_pruning/results.csv`
- `results/exp3_two_stage_pruning/two_stage_analysis.png`

### 实验4：CSR编码开销分析

**目标**：量化CSR编码的时间成本

**测试内容**：
- 不同模式的编码时间
- 编码时间占总时间的比例
- 不同序列长度下的扩展性

**输出**：
- `results/exp4_csr_encoding/results.csv`
- `results/exp4_csr_encoding/csr_encoding_overhead.png`

## 代码结构

```
SparseAttentionProject/
├── kernels/
│   ├── triton_online_softmax.py          # Triton在线softmax kernel
│   ├── triton_block_sparse.py            # Triton块稀疏注意力入口
│   └── block_sparse_attention_reference.py  # PyTorch参考实现
├── experiments/
│   ├── exp1_unified_framework.py         # 实验1
│   ├── exp2_online_softmax.py            # 实验2
│   ├── exp3_two_stage_pruning.py         # 实验3
│   ├── exp4_csr_encoding_overhead.py     # 实验4
│   ├── run_all_experiments.py            # 运行所有实验
│   └── run_experiment.py                 # 原始基准测试
├── visualization/
│   └── plot_performance.py               # 可视化脚本
├── models/
│   ├── config.py                         # 配置
│   └── data_loader.py                    # 数据生成
├── utils/
│   ├── csr_utils.py                      # CSR工具
│   └── memory_utils.py                   # 内存工具
└── results/                              # 实验结果输出目录
```

## 预期结果

### 性能

- **Triton vs PyTorch**：3-10x加速（得益于融合和内存优化）
- **统一kernel vs 专用实现**：0.8-1.2x（通用性的小代价）
- **模式切换开销**：零开销（同一kernel）

### 内存

- **在线softmax**：长序列节省30-50%内存
- **块稀疏**：内存占用与稀疏度成正比

### 精度

- **数值误差**：< 1e-3（fp16）或 < 1e-6（fp32）
- **两阶段 vs 单阶段**：在相同稀疏度下，两阶段更准确

### 编码开销

- **编码时间**：< 1% 总时间（对于seq_len <= 16K）
- **可预计算**：静态模式可在初始化时缓存

## 论文贡献点

1. **系统贡献**：统一的块稀疏注意力框架，支持任意CSR表示的模式
2. **算法贡献**：两阶段稀疏策略，平衡静态启发式和动态剪枝
3. **工程贡献**：高性能Triton实现，证明通用kernel可以媲美专用实现
4. **实验贡献**：全面的性能-精度权衡分析，为实际应用提供指导

## 注意事项

1. **Triton可用性**：如果Triton不可用，会自动回退到PyTorch参考实现
2. **CUDA要求**：Triton kernel需要CUDA支持，CPU会使用PyTorch实现
3. **内存限制**：长序列（>16K）可能需要大量GPU内存
4. **因果掩码**：Triton kernel暂不支持因果掩码，会回退到PyTorch实现

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{unified_csr_sparse_attention,
  title={Unified CSR Block Sparse Attention with Two-Stage Pruning},
  author={Your Name},
  year={2026}
}
```

## 许可证

MIT License
