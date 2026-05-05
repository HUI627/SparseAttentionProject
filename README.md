# SparseAttentionProject

Unified CSR block sparse attention experiments with optional two-stage pruning.

## 核心创新 / Key Innovations

1. **统一CSR表示** / Unified CSR Representation：将所有块稀疏模式统一编码为CSR格式
2. **通用Triton Kernel** / Generic Triton Kernel：单一kernel处理所有模式
3. **在线Softmax** / Online Softmax：块级累积，内存高效
4. **两阶段稀疏** / Two-Stage Pruning：静态模式 + 动态剪枝

## 快速开始 / Quick Start

### 安装 / Installation

```bash
pip install -r requirements.txt
```

### 运行基准测试 / Run Benchmarks

```bash
# 基础运行 / Basic run
python experiments/run_experiment.py --device cuda --seq-lens 1024 2048 4096 --patterns dense local bigbird longformer grouped --enable-two-stage
```

### 运行完整实验套件 / Run Full Experiment Suite

```bash
# 运行所有实验 / Run all experiments
python experiments/run_all_experiments.py

# 生成可视化 / Generate visualizations
python visualization/plot_performance.py
```

详细说明请参考 `experiments/README.md`

## Run

```bash
pip install -r requirements.txt
python experiments/run_experiment.py --device cuda --seq-lens 1024 2048 4096 --patterns dense local bigbird longformer grouped --enable-two-stage
```

If CUDA/Triton is unavailable, the runner falls back to PyTorch reference kernels on CPU/GPU for correctness and CSV generation.

## Outputs

- `experiments/benchmark_results.csv`
- `results/latency_results.csv`
- `results/accuracy_results.csv`

## 新实验 / New Experiments

- **实验1**：统一框架对比 / Unified Framework Comparison
- **实验2**：在线Softmax验证 / Online Softmax Verification  
- **实验3**：两阶段稀疏评估 / Two-Stage Pruning Evaluation
- **实验4**：CSR编码开销分析 / CSR Encoding Overhead Analysis

详见 `experiments/README.md` 和 `experiments/EXPERIMENT_DESIGN.md`

