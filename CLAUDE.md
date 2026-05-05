# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Block sparse attention research project using CSR (Compressed Sparse Row) format. Benchmarks different sparse attention patterns against dense attention, with optional two-stage pruning.

**Key Innovation**: Unified CSR representation + Triton online softmax kernel + two-stage dynamic pruning.

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Quick Test
```bash
# Verify implementation works
python experiments/test_quick.py
```

### Run Original Benchmark
```bash
# Basic run with default patterns
python experiments/run_experiment.py --device cuda

# Full benchmark with all patterns and two-stage pruning
python experiments/run_experiment.py --device cuda --seq-lens 1024 2048 4096 --patterns dense local bigbird longformer grouped --enable-two-stage
```

### Run New Experiment Suite
```bash
# Run all experiments
python experiments/run_all_experiments.py

# Run individual experiments
python experiments/exp1_unified_framework.py      # Unified framework comparison
python experiments/exp2_online_softmax.py         # Online softmax verification
python experiments/exp3_two_stage_pruning.py      # Two-stage pruning evaluation
python experiments/exp4_csr_encoding_overhead.py  # CSR encoding overhead

# Generate visualizations
python visualization/plot_performance.py
```

### Key Arguments
- `--device`: cuda or cpu (falls back to cpu if CUDA unavailable)
- `--seq-lens`: Sequence lengths to benchmark (default: 1024 2048 4096)
- `--patterns`: Sparse patterns to test (dense, local, bigbird, longformer, grouped)
- `--enable-two-stage`: Enable second-stage dynamic pruning
- `--block-size`: Block size for sparse attention (default: 64)
- `--topk-blocks`: Top-k blocks to keep in stage 2 pruning (default: 8)

## Architecture

### Core Data Structure: CSRMask
Defined in `utils/csr_utils.py`. Represents block-level sparsity in CSR format:
- `row_ptr`: CSR row pointers (length: num_q_blocks + 1)
- `col_ind`: CSR column indices (length: nnz_blocks)
- `num_q_blocks`, `num_k_blocks`: Number of blocks in query/key dimensions
- `block_size`: Size of each block (typically 64)
- `encode_ms`: Time to encode the pattern
- `pattern`: Pattern name (dense, local, bigbird, etc.)

### Sparse Patterns
All patterns implemented in `utils/csr_utils.py`:
- **dense**: Full attention (all blocks)
- **local**: Local window attention (configurable window size)
- **bigbird**: Local + global + random blocks
- **longformer**: Local + designated global tokens with full attention
- **grouped**: Group-based attention with random sampling per group

### Two-Stage Pruning
1. **Stage 1**: Create a sparse pattern (e.g., bigbird with local + global + random)
2. **Stage 2**: Dynamically prune blocks based on QK scores
   - Implemented in `kernels/block_sparse_attention_reference.py::prune_csr_by_block_score`
   - Uses `||Q_block||_max * ||K_block||_max` as upper-bound score proxy
   - Keeps top-k blocks or blocks above threshold per query block

### Kernel Implementations
- **Reference**: `kernels/block_sparse_attention_reference.py`
  - PyTorch implementations for correctness validation
  - `dense_attention`: Standard scaled dot-product attention
  - `block_sparse_attention_reference`: Sparse attention using CSRMask
  - `prune_csr_by_block_score`: Second-stage pruning logic
- **Triton**: `kernels/triton_online_softmax.py` + `kernels/triton_block_sparse.py`
  - **NEW**: Actual Triton implementation with online softmax algorithm
  - Fused QK computation, softmax, and attention output
  - CSR format traversal for arbitrary sparse patterns
  - Automatic fallback to PyTorch if Triton unavailable
  - Memory efficient: avoids materializing O(N²) attention matrix

### New Experiment Suite
Four comprehensive experiments to evaluate the unified CSR framework:

1. **exp1_unified_framework.py**: Compare unified CSR approach vs specialized implementations
   - Triton vs PyTorch performance
   - Pattern switching overhead
   - Memory usage comparison

2. **exp2_online_softmax.py**: Verify online softmax correctness and efficiency
   - Numerical accuracy (vs PyTorch and dense)
   - Memory efficiency (online vs materialized)
   - Compute throughput

3. **exp3_two_stage_pruning.py**: Evaluate two-stage pruning effectiveness
   - Compare: dense baseline, stage 1 only, stage 2 only, two-stage
   - Performance-accuracy tradeoff analysis
   - Win rate of two-stage vs direct pruning

4. **exp4_csr_encoding_overhead.py**: Measure CSR encoding cost
   - Encoding time vs attention time
   - Overhead percentage
   - Scalability with sequence length

### Visualization
- **plot_performance.py**: Generate all experiment visualizations
  - Latency comparison, speedup, memory usage
  - Numerical accuracy, memory efficiency, throughput
  - Two-stage analysis, win rate, sparsity reduction
  - Encoding overhead, time vs NNZ blocks

### Experiment Flow
1. Parse CLI arguments (`experiments/experiment_config.py`)
2. For each sequence length:
   - Generate Q, K, V tensors (`models/data_loader.py`)
   - Run dense attention baseline
   - For each sparse pattern:
     - Create CSRMask (`utils/csr_utils.py::make_csr`)
     - Run sparse attention
     - Optionally run two-stage pruning
   - Measure latency, memory, accuracy
3. Output CSV files to `experiments/` and `results/`

## Output Files
- `experiments/benchmark_results.csv`: Full results from original benchmark
- `results/latency_results.csv`: Latency and throughput metrics
- `results/accuracy_results.csv`: Max absolute error vs dense attention
- `results/exp1_unified_framework/`: Experiment 1 results and plots
- `results/exp2_online_softmax/`: Experiment 2 results and plots
- `results/exp3_two_stage_pruning/`: Experiment 3 results and plots
- `results/exp4_csr_encoding/`: Experiment 4 results and plots

## Key Implementation Details

### Triton Online Softmax Algorithm
The core innovation is the online softmax algorithm that processes blocks incrementally:
```python
for each query block:
    m_i = -inf  # running max
    l_i = 0     # running sum
    acc = 0     # output accumulator
    
    for each non-zero key block in CSR:
        qk = Q @ K.T
        m_new = max(m_i, max(qk))
        
        # Rescale previous accumulator
        acc *= exp(m_i - m_new)
        l_i *= exp(m_i - m_new)
        
        # Accumulate current block
        p = exp(qk - m_new)
        acc += p @ V
        l_i += sum(p)
        m_i = m_new
    
    O = acc / l_i
```

This avoids materializing the full O(N²) attention matrix, saving memory.

### Two-Stage Pruning Strategy
1. **Stage 1**: Create sparse pattern (e.g., bigbird with 15% density)
2. **Stage 2**: Dynamically prune based on QK scores (e.g., top-k=8, final 5% density)

Stage 2 uses `||Q_block||_max * ||K_block||_max` as a cheap upper-bound score proxy.

## Important Notes

- **Triton Availability**: Triton kernel requires CUDA. Automatically falls back to PyTorch on CPU or if Triton unavailable.
- **Causal Masking**: Triton kernel does not yet support causal masking. Will fall back to PyTorch reference.
- **Memory**: Long sequences (>16K) may require significant GPU memory.
- **Numerical Precision**: fp16 may have larger errors. Use fp32 for correctness validation.

## Documentation

- `experiments/README.md`: Detailed experiment guide
- `experiments/EXPERIMENT_DESIGN.md`: Full experiment design with 6 experiments
- `experiments/IMPLEMENTATION_PLAN.md`: Implementation details and code structure
- `IMPLEMENTATION_SUMMARY.md`: Summary of all implemented code

## Key Configuration Parameters
Defined in `models/config.py::ExperimentConfig`:
- `batch_size`, `num_heads`, `seq_len`, `head_dim`: Standard attention dimensions
- `block_size`: Block size for sparse attention (default: 64)
- `local_window_blocks`: Window size for local patterns (default: 2)
- `global_blocks`: Number of global blocks for bigbird/longformer (default: 2)
- `random_blocks`: Number of random blocks for bigbird (default: 2)
- `grouped_num_groups`: Number of groups for grouped pattern (default: 8)
- `topk_blocks`: Top-k blocks to keep in stage 2 (default: 8)
- `threshold`: Optional threshold for stage 2 pruning (default: None)
- `causal`: Enable causal masking (default: False)
