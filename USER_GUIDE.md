# 使用指南 / User Guide

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

需要的包 / Required packages:
- `torch>=2.1` - PyTorch
- `triton>=2.1` - Triton (可选，用于GPU加速)
- `numpy` - 数值计算
- `pandas` - 数据处理
- `matplotlib` - 可视化
- `psutil` - 内存监控

### 2. 快速测试 / Quick Test

验证代码是否正常工作：

```bash
python experiments/test_quick.py
```

这会测试：
- 密集注意力
- 稀疏注意力（local, bigbird, longformer）
- 两阶段剪枝
- Triton kernel（如果可用）

### 3. 运行实验 / Run Experiments

#### 选项A：运行所有实验（推荐）

```bash
python experiments/run_all_experiments.py
```

这会依次运行4个实验，大约需要10-30分钟（取决于硬件）。

#### 选项B：运行单个实验

```bash
# 实验1：统一框架对比（约5分钟）
python experiments/exp1_unified_framework.py

# 实验2：在线softmax验证（约5分钟）
python experiments/exp2_online_softmax.py

# 实验3：两阶段稀疏评估（约10分钟）
python experiments/exp3_two_stage_pruning.py

# 实验4：CSR编码开销（约5分钟）
python experiments/exp4_csr_encoding_overhead.py
```

#### 选项C：运行原始基准测试

```bash
python experiments/run_experiment.py --device cuda --seq-lens 1024 2048 4096 --patterns dense local bigbird longformer grouped --enable-two-stage
```

### 4. 生成可视化 / Generate Visualizations

```bash
python visualization/plot_performance.py
```

这会为每个实验生成PNG图表，保存在 `results/` 目录下。

## 实验说明 / Experiment Details

### 实验1：统一框架对比

**目标**：证明统一CSR表示的通用性和效率

**测试内容**：
- Triton实现 vs PyTorch参考实现
- 不同稀疏模式的性能对比
- 模式切换开销

**输出**：
- `results/exp1_unified_framework/results.csv`
- `results/exp1_unified_framework/unified_framework_comparison.png`

**关键指标**：
- 延迟（ms）
- 加速比（Triton vs PyTorch）
- 内存使用（MB）
- 稀疏度 vs 加速比

### 实验2：在线Softmax验证

**目标**：验证在线softmax算法的正确性和效率

**测试内容**：
- 数值精度（vs PyTorch和dense）
- 内存效率（在线 vs 物化中间结果）
- 计算吞吐量（tokens/s）

**输出**：
- `results/exp2_online_softmax/correctness.csv`
- `results/exp2_online_softmax/memory.csv`
- `results/exp2_online_softmax/compute.csv`
- `results/exp2_online_softmax/online_softmax_results.png`

**关键指标**：
- 最大绝对误差
- 内存节省百分比
- 吞吐量（tokens/s）

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

**关键指标**：
- 精度（最大绝对误差）
- 加速比
- 两阶段胜率（vs 直接剪枝）
- 稀疏度变化

### 实验4：CSR编码开销分析

**目标**：量化CSR编码的时间成本

**测试内容**：
- 不同模式的编码时间
- 编码时间占总时间的比例
- 序列长度扩展性

**输出**：
- `results/exp4_csr_encoding/results.csv`
- `results/exp4_csr_encoding/csr_encoding_overhead.png`

**关键指标**：
- 编码时间（ms）
- 开销百分比（编码时间 / 注意力时间）
- 编码时间 vs NNZ块数

## 预期结果 / Expected Results

### 性能 / Performance

- **Triton vs PyTorch**：3-10x加速（得益于融合和内存优化）
- **统一kernel vs 专用实现**：0.8-1.2x（通用性的小代价）
- **模式切换开销**：零开销（同一kernel）

### 内存 / Memory

- **在线softmax**：长序列节省30-50%内存
- **块稀疏**：内存占用与稀疏度成正比

### 精度 / Accuracy

- **数值误差**：< 1e-3（fp16）或 < 1e-6（fp32）
- **两阶段 vs 单阶段**：在相同稀疏度下，两阶段更准确

### 编码开销 / Encoding Overhead

- **编码时间**：< 1% 总时间（对于seq_len <= 16K）
- **可预计算**：静态模式可在初始化时缓存

## 常见问题 / FAQ

### Q1: Triton不可用怎么办？

**A**: 代码会自动回退到PyTorch参考实现。你仍然可以运行所有实验，只是性能会慢一些。

### Q2: 内存不足怎么办？

**A**: 尝试：
- 减小序列长度：`--seq-lens 1024 2048`
- 减小batch size：`--batch-size 1`
- 使用更稀疏的模式
- 使用CPU：`--device cpu`

### Q3: 如何修改实验配置？

**A**: 编辑实验脚本中的配置：
```python
config = ExperimentConfig(
    batch_size=1,      # 修改这里
    num_heads=8,       # 修改这里
    seq_len=2048,      # 修改这里
    head_dim=64,       # 修改这里
    block_size=64,     # 修改这里
)
```

### Q4: 如何添加新的稀疏模式？

**A**: 在 `utils/csr_utils.py` 中添加新的CSR生成函数：
```python
def my_pattern_csr(seq_len, block_size, device="cpu"):
    st = time.perf_counter()
    n = (seq_len + block_size - 1) // block_size
    rows = []
    
    # 定义你的稀疏模式
    for i in range(n):
        # 为每个query块选择key块
        selected_blocks = [...]  # 你的逻辑
        rows.append(selected_blocks)
    
    return _finish(rows, n, block_size, "my_pattern", st, device)
```

然后在 `make_csr` 函数中添加：
```python
def make_csr(pattern: str, cfg, device: str = "cpu"):
    # ... 现有代码 ...
    if pattern == "my_pattern":
        return my_pattern_csr(cfg.seq_len, cfg.block_size, device)
```

### Q5: 如何在自己的模型中使用？

**A**: 导入并使用 `block_sparse_attention` 函数：
```python
from kernels.triton_block_sparse import block_sparse_attention
from utils.csr_utils import make_csr

# 创建稀疏模式
csr = make_csr('bigbird', config, device)

# 使用块稀疏注意力
output = block_sparse_attention(q, k, v, csr, use_triton=True)
```

### Q6: 如何调试Triton kernel？

**A**: 
1. 设置环境变量查看生成的代码：
   ```bash
   export TRITON_INTERPRET=1
   ```

2. 使用PyTorch实现验证正确性：
   ```python
   out_triton = block_sparse_attention(q, k, v, csr, use_triton=True)
   out_pytorch = block_sparse_attention(q, k, v, csr, use_triton=False)
   error = (out_triton - out_pytorch).abs().max()
   print(f"Error: {error}")
   ```

## 性能优化建议 / Performance Tips

1. **使用fp16**：在GPU上使用fp16可以获得更好的性能
   ```python
   q = q.half()
   k = k.half()
   v = v.half()
   ```

2. **调整块大小**：较大的块大小（64, 128）通常更快，但灵活性较低

3. **预计算CSR掩码**：如果稀疏模式是静态的，可以预计算并缓存

4. **批处理**：增加batch size可以提高GPU利用率

## 故障排除 / Troubleshooting

### 错误：CUDA out of memory

**解决方案**：
- 减小序列长度或batch size
- 使用更稀疏的模式
- 使用CPU

### 错误：Triton import error

**解决方案**：
- 安装Triton：`pip install triton`
- 或者让代码回退到PyTorch：代码会自动处理

### 错误：数值误差过大

**解决方案**：
- 使用fp32而不是fp16
- 检查稀疏模式是否合理
- 验证CSR掩码是否正确

## 联系与反馈 / Contact

如有问题或建议，请：
1. 查看文档：`experiments/README.md`, `CLAUDE.md`
2. 运行快速测试：`python experiments/test_quick.py`
3. 检查实验结果：查看 `results/` 目录

## 引用 / Citation

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{unified_csr_sparse_attention,
  title={Unified CSR Block Sparse Attention with Two-Stage Pruning},
  author={Your Name},
  year={2026}
}
```
