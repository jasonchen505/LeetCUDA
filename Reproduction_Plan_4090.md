# LeetCUDA 项目复现 Plan

> 基于 8×RTX 4090 算力资源，完整复现 LLM 推理核心 CUDA 算子

---

## 📋 目录

1. [硬件资源评估](#1-硬件资源评估)
2. [项目模块分析与复现难度](#2-项目模块分析与复现难度)
3. [完整复现计划](#3-完整复现计划)
4. [环境配置指南](#4-环境配置指南)
5. [每日执行计划](#5-每日执行计划)
6. [验证与Benchmark](#6-验证与benchmark)
7. [风险与应对](#7-风险与应对)

---

## 1. 硬件资源评估

### 1.1 RTX 4090 硬件特性

```
┌─────────────────────────────────────────────────────────────┐
│                    RTX 4090 关键参数                        │
├─────────────────────────────────────────────────────────────┤
│ 架构: Ada Lovelace (SM 8.9)                                │
│ CUDA Cores: 16384                                          │
│ Tensor Cores: 512 (第 4 代)                                │
│ 显存: 24GB GDDR6X                                          │
│ 显存带宽: 1008 GB/s                                        │
│ L2 Cache: 73 MB                                            │
│ Shared Memory per SM: 100 KB (可配置到 228 KB)             │
│ Register File per SM: 256 KB (65536 × 32-bit)              │
│ Max Threads per SM: 1536                                   │
│ Max Warps per SM: 48                                       │
│ FP16 Tensor Core: 330 TFLOPS (理论峰值)                    │
│ FP32: 82.6 TFLOPS                                          │
│ 理论 SMEM 带宽: ~19 TB/s                                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 4090 vs 项目测试硬件对比

| 特性 | RTX 3080 (Laptop) | L20 | RTX 4090 | 你的资源 |
|------|-------------------|-----|----------|----------|
| 架构 | Ampere (SM 8.6) | Ada (SM 8.9) | Ada (SM 8.9) | 8×4090 |
| FP16 Tensor Core | 119 TFLOPS | 119.5 TFLOPS | 330 TFLOPS | 2640 TFLOPS |
| 显存 | 16 GB | 48 GB | 24 GB | 192 GB |
| 显存带宽 | 760 GB/s | 864 GB/s | 1008 GB/s | 8064 GB/s |
| L2 Cache | 6 MB | 98 MB | 73 MB | 584 MB |
| SMEM per SM | 100 KB | 100 KB | 100 KB | - |

### 1.3 4090 复现优势与限制

**优势**：
- FP16 Tensor Core 性能是 L20 的 2.76x
- 显存带宽比 L20 高 17%
- 支持所有 Ampere/Ada 特性（MMA m16n8k16, cp.async 等）
- 8 卡可以做多卡并行测试

**限制**：
- 不支持 Hopper 特性（WGMMA, TMA, Warp Specialization）
- 单卡显存 24GB，大模型测试受限
- 需要跳过 `hgemm_wgmma` 相关代码

---

## 2. 项目模块分析与复现难度

### 2.1 模块分类与优先级

```
┌─────────────────────────────────────────────────────────────┐
│                    项目模块优先级矩阵                        │
├─────────────────────────────────────────────────────────────┤
│ P0 - 核心必做（面试高频 + LLM 核心）                        │
│   ├── softmax (naive → safe → online)                      │
│   ├── rms-norm / layer-norm                                │
│   ├── rope                                                 │
│   ├── embedding                                            │
│   ├── reduce (warp_reduce, block_reduce)                   │
│   └── elementwise (relu, gelu, etc.)                       │
│                                                            │
│ P1 - 重点突破（性能优化 + 面试亮点）                        │
│   ├── hgemm (naive → mma → stages → swizzle)              │
│   ├── flash-attn (split_kv → split_q → share_qkv)        │
│   ├── sgemm (naive → tiling → vec4 → bcf)                 │
│   └── sgemv / hgemv                                        │
│                                                            │
│ P2 - 扩展提升（差异化 + 深度理解）                          │
│   ├── triton (softmax, layer-norm, merge-attn-states)     │
│   ├── cutlass/cute (hgemm, flash-attn)                    │
│   ├── mat-transpose (bank conflict 专题)                   │
│   └── nms                                                  │
│                                                            │
│ P3 - 跳过（硬件不支持）                                     │
│   ├── hgemm_wgmma (需要 Hopper SM 9.0)                    │
│   └── ws-hgemm (Warp Specialization, 需要 Hopper)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 各模块复现难度评估

| 模块 | 难度 | 预计时间 | 依赖 | 4090 兼容性 |
|------|------|----------|------|-------------|
| **P0: 基础原语** |
| reduce (warp/block) | ⭐ | 2h | 无 | ✅ 完全兼容 |
| elementwise | ⭐ | 1h | 无 | ✅ 完全兼容 |
| embedding | ⭐ | 1h | 无 | ✅ 完全兼容 |
| softmax (naive/safe/online) | ⭐⭐ | 4h | reduce | ✅ 完全兼容 |
| rms-norm | ⭐⭐ | 2h | reduce | ✅ 完全兼容 |
| layer-norm | ⭐⭐ | 2h | reduce | ✅ 完全兼容 |
| rope | ⭐ | 2h | 无 | ✅ 完全兼容 |
| dot-product | ⭐⭐ | 2h | reduce | ✅ 完全兼容 |
| **P1: 核心算子** |
| sgemm (naive→vec4) | ⭐⭐⭐ | 4h | 无 | ✅ 完全兼容 |
| sgemm (bcf+dbuf) | ⭐⭐⭐⭐ | 6h | sgemm | ✅ 完全兼容 |
| sgemv (k32/k128/k16) | ⭐⭐⭐ | 3h | reduce | ✅ 完全兼容 |
| hgemv | ⭐⭐⭐ | 2h | sgemv | ✅ 完全兼容 |
| hgemm (naive→mma) | ⭐⭐⭐⭐ | 8h | sgemm | ✅ 完全兼容 |
| hgemm (stages+swizzle) | ⭐⭐⭐⭐⭐ | 12h | hgemm | ✅ 完全兼容 |
| flash-attn (split_kv) | ⭐⭐⭐⭐⭐ | 8h | online_softmax, hgemm | ✅ 完全兼容 |
| flash-attn (split_q) | ⭐⭐⭐⭐⭐ | 8h | flash-attn | ✅ 完全兼容 |
| flash-attn (share_qkv) | ⭐⭐⭐⭐⭐ | 8h | flash-attn | ✅ 完全兼容 |
| **P2: 扩展** |
| mat-transpose | ⭐⭐⭐ | 3h | 无 | ✅ 完全兼容 |
| triton_softmax | ⭐⭐ | 2h | 无 | ✅ 完全兼容 |
| triton_layer_norm | ⭐⭐ | 2h | 无 | ✅ 完全兼容 |
| merge-attn-states | ⭐⭐⭐ | 3h | 无 | ✅ 完全兼容 |
| cutlass/cute hgemm | ⭐⭐⭐⭐⭐ | 8h | hgemm | ✅ 完全兼容 |
| **P3: 跳过** |
| hgemm_wgmma | - | - | - | ❌ 需要 Hopper |
| ws-hgemm | - | - | - | ❌ 需要 Hopper |

### 2.3 总体时间估算

```
P0 基础原语:    ~16 小时 (2 天)
P1 核心算子:    ~62 小时 (8 天)
P2 扩展提升:    ~18 小时 (2-3 天)
总计:           ~96 小时 (12-13 天)
```

---

## 3. 完整复现计划

### 3.1 Phase 1: 环境搭建 + 基础原语 (Day 1-2)

**目标**：搭建环境，复现所有 P0 基础算子

**Day 1: 环境搭建 + reduce + elementwise + embedding**

```bash
# 上午 (3h): 环境搭建
1. 安装 CUDA 12.x + PyTorch 2.x
2. 验证 4090 硬件特性
3. 克隆项目，熟悉代码结构
4. 编译运行 notes-v2.cu 面试笔记

# 下午 (4h): 基础算子
1. 实现 warp_reduce_sum/max (1h)
2. 实现 block_reduce_sum/max (1h)
3. 实现 elementwise (relu, add) (1h)
4. 实现 embedding (1h)
```

**Day 2: Softmax + Norm + RoPE**

```bash
# 上午 (3h): Softmax 三种实现
1. 实现 naive softmax (1h)
2. 实现 safe softmax (1h)
3. 实现 online safe softmax (1h)

# 下午 (4h): Norm + RoPE
1. 实现 RMSNorm (1.5h)
2. 实现 LayerNorm (1.5h)
3. 实现 RoPE (1h)
```

**验证点**：
```bash
# 运行面试笔记验证
cd kernels/interview
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda notes-v2.cu -o notes_v2_sm89.bin
CUDA_VISIBLE_DEVICES=0 ./notes_v2_sm89.bin
# 预期：所有 P0 测试 PASS
```

### 3.2 Phase 2: SGEMM 优化路径 (Day 3-4)

**目标**：从 naive SGEMM 到高性能 SGEMM，理解 GEMM 优化全路径

**Day 3: SGEMM 基础优化**

```bash
# 上午 (3h): Naive + Tiling
1. 实现 sgemm_naive_f32 (1h)
2. 实现 sgemm_sliced_k_f32 (tiling) (2h)

# 下午 (4h): Thread Tile + Vec4
1. 实现 sgemm_t_8x8_sliced_k_f32x4 (2h)
2. 实现 sgemm_t_8x8_sliced_k_f32x4_bcf (2h)
```

**Day 4: SGEMM 高级优化**

```bash
# 上午 (3h): Double Buffer + Async
1. 实现 sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf (1.5h)
2. 实现 sgemm_async (cp.async) (1.5h)

# 下午 (4h): 测试 + Profiling
1. 运行完整 benchmark (1h)
2. 使用 ncu 分析性能瓶颈 (2h)
3. 对比 cuBLAS baseline (1h)
```

**验证点**：
```bash
cd kernels/sgemm
python3 sgemm.py --all --plot
# 预期：vec4 版本达到 cuBLAS 60%+ 性能
```

### 3.3 Phase 3: HGEMM + Tensor Cores (Day 5-7)

**目标**：实现高性能 HGEMM，理解 Tensor Core 编程

**Day 5: HGEMM 基础 + WMMA**

```bash
# 上午 (3h): HGEMM Naive + Sliced K
1. 实现 hgemm_naive_f16 (1h)
2. 实现 hgemm_sliced_k_f16 (1h)
3. 实现 hgemm_t_8x8_sliced_k_f16x4 (1h)

# 下午 (4h): WMMA API
1. 实现 hgemm_wmma_m16n16k16_naive (2h)
2. 实现 hgemm_wmma_m16n16k16_mma4x2 (2h)
```

**Day 6: HGEMM MMA PTX**

```bash
# 上午 (4h): MMA m16n8k16
1. 理解 MMA 指令格式 (1h)
2. 实现 hgemm_mma_m16n8k16_naive (1.5h)
3. 实现 hgemm_mma_m16n8k16_mma2x4 (1.5h)

# 下午 (4h): MMA + Stages
1. 实现 hgemm_mma_stages (2h)
2. 实现 hgemm_mma_stages_dsmem (2h)
```

**Day 7: HGEMM Swizzle + Benchmark**

```bash
# 上午 (3h): Swizzle 优化
1. 理解 SMEM Swizzle 原理 (1h)
2. 实现 hgemm_mma_stages_swizzle (2h)

# 下午 (4h): CuTe + Benchmark
1. 实现 hgemm_mma_stage_tn_cute (2h)
2. 运行完整 benchmark (1h)
3. 对比 cuBLAS，目标 98%+ (1h)
```

**验证点**：
```bash
cd kernels/hgemm
python3 hgemm.py --mma-all --plot --topk 8
# 预期：MMA 版本达到 cuBLAS 95%+ 性能
# CuTe 版本达到 cuBLAS 98%+ 性能
```

### 3.4 Phase 4: FlashAttention (Day 8-10)

**目标**：实现 FlashAttention-2，理解 Attention 优化

**Day 8: FlashAttention 基础**

```bash
# 上午 (3h): 理论 + Split KV
1. 理解 FlashAttention 算法 (1h)
2. 实现 flash_attn_mma_split_kv (2h)

# 下午 (4h): Split Q
1. 理解 Split Q vs Split KV 区别 (1h)
2. 实现 flash_attn_mma_split_q (3h)
```

**Day 9: FlashAttention 优化**

```bash
# 上午 (3h): Shared KV SMEM
1. 理解 SRAM 共享策略 (1h)
2. 实现 flash_attn_mma_share_kv (2h)

# 下午 (4h): Shared QKV SMEM
1. 理解 Fully Shared QKV (1h)
2. 实现 flash_attn_mma_share_qkv (3h)
```

**Day 10: FlashAttention 进阶 + Benchmark**

```bash
# 上午 (3h): Fine-grained Tiling
1. 理解 QK/QKV Tiling (1h)
2. 实现 flash_attn_mma_tiling_qk (2h)

# 下午 (4h): Swizzle + Benchmark
1. 实现 flash_attn swizzle 变体 (2h)
2. 运行完整 benchmark (1h)
3. 对比官方 FA2 (1h)
```

**验证点**：
```bash
cd kernels/flash-attn
python3 flash_attn_mma.py --D 64 --iters 10 --torch
# 预期：D=64 时达到 FA2 90%+ 性能
# 精度误差 < 1e-3
```

### 3.5 Phase 5: Triton + 扩展 (Day 11-12)

**目标**：学习 Triton 编程，扩展视野

**Day 11: Triton 基础**

```bash
# 上午 (3h): Triton Softmax
1. 理解 Triton 编程模型 (1h)
2. 实现 triton_fused_softmax (2h)

# 下午 (4h): Triton LayerNorm + Merge Attn
1. 实现 triton_fused_layer_norm (2h)
2. 实现 triton_merge_attn_states (2h)
```

**Day 12: 其他扩展 + 总结**

```bash
# 上午 (3h): Matrix Transpose + NMS
1. 实现 mat_transpose (bank conflict 专题) (2h)
2. 实现 nms (1h)

# 下午 (4h): 总结 + 面试准备
1. 整理所有 benchmark 数据 (2h)
2. 准备面试回答 (2h)
```

### 3.6 Phase 6: 多卡并行测试 (Day 13)

**目标**：利用 8 卡资源做并行测试

```bash
# 上午 (3h): 多卡 Benchmark
1. 使用 CUDA_VISIBLE_DEVICES 测试不同卡
2. 对比 8 卡性能一致性
3. 测试多卡并行 HGEMM

# 下午 (4h): 大规模测试
1. 测试大矩阵 HGEMM (16384×16384)
2. 测试长序列 FlashAttention (N=16384)
3. 整理最终 benchmark 报告
```

---

## 4. 环境配置指南

### 4.1 基础环境

```bash
# 1. 检查 CUDA 版本
nvcc --version
nvidia-smi

# 2. 安装 PyTorch (推荐 2.5+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 验证 GPU
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# 4. 安装依赖
pip install matplotlib numpy
```

### 4.2 编译环境

```bash
# 设置 CUDA 架构 (4090 = sm_89)
export TORCH_CUDA_ARCH_LIST=Ada

# 或者在编译时指定
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda your_kernel.cu -o output.bin
```

### 4.3 Profiling 工具

```bash
# Nsight Compute (kernel 级分析)
ncu --set full ./your_binary

# Nsight Systems (系统级分析)
nsys profile --stats=true ./your_binary

# 常用指标
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__throughput.avg.pct_of_peak_sustained_elapsed \
  ./your_binary
```

---

## 5. 每日执行计划

### 5.1 每日时间分配

```
09:00 - 12:00  核心学习 + 实现 (3h)
12:00 - 13:00  午餐
13:00 - 17:00  实现 + 测试 (4h)
17:00 - 18:00  整理笔记 + 准备第二天
```

### 5.2 每日 Checklist

```bash
□ 代码实现
  - 今日目标 kernel 实现完成
  - 通过单元测试
  - 性能 benchmark 完成

□ 学习笔记
  - 记录实现过程中的问题
  - 记录优化技巧和经验
  - 整理面试可能问到的点

□ Git 提交
  - 代码提交
  - 笔记提交
```

---

## 6. 验证与 Benchmark

### 6.1 验证方法

**正确性验证**：
```python
# PyTorch 参考实现
import torch

def verify_kernel(output, reference, atol=1e-3, rtol=1e-3):
    """验证 kernel 输出与 PyTorch 参考的一致性"""
    max_diff = torch.max(torch.abs(output - reference)).item()
    mean_diff = torch.mean(torch.abs(output - reference)).item()
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    assert torch.allclose(output, reference, atol=atol, rtol=rtol), \
        f"Verification failed! Max diff: {max_diff}"
    print("Verification passed!")
```

**性能验证**：
```python
import torch
import time

def benchmark_kernel(func, *args, warmup=10, iters=100):
    """Benchmark kernel 性能"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters * 1000  # ms
    print(f"Average time: {avg_time:.3f} ms")
    return avg_time
```

### 6.2 Benchmark 矩阵

| 算子 | 测试规模 | 指标 | 目标 |
|------|----------|------|------|
| Softmax | N=1024/2048/4096 | GB/s | > 80% 带宽利用率 |
| Norm | N=1024/2048/4096 | GB/s | > 80% 带宽利用率 |
| SGEMM | M=N=K=1024/4096/16384 | TFLOPS | > 60% cuBLAS |
| HGEMM | M=N=K=1024/4096/16384 | TFLOPS | > 95% cuBLAS |
| FlashAttn | B=1,H=8,N=8192,D=64 | TFLOPS | > 90% FA2 |

### 6.3 预期性能数据 (4090)

```
┌─────────────────────────────────────────────────────────────┐
│                    预期性能 (4090)                          │
├─────────────────────────────────────────────────────────────┤
│ SGEMM (naive):     ~10 TFLOPS  (12% cuBLAS)               │
│ SGEMM (vec4):      ~50 TFLOPS  (60% cuBLAS)               │
│ SGEMM (bcf+dbuf):  ~65 TFLOPS  (79% cuBLAS)               │
│                                                             │
│ HGEMM (naive):     ~15 TFLOPS  (5% cuBLAS)                │
│ HGEMM (mma):       ~280 TFLOPS (85% cuBLAS)               │
│ HGEMM (stages):    ~310 TFLOPS (94% cuBLAS)               │
│ HGEMM (swizzle):   ~320 TFLOPS (97% cuBLAS)               │
│                                                             │
│ FlashAttn (split_q): ~200 TFLOPS (D=64)                   │
│ FlashAttn (share_qkv): ~250 TFLOPS (D=64)                 │
│                                                             │
│ cuBLAS HGEMM:      ~330 TFLOPS (理论峰值)                  │
│ cuBLAS SGEMM:      ~82 TFLOPS (理论峰值)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 风险与应对

### 7.1 技术风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| MMA PTX 编译错误 | 中 | 延迟 1-2h | 参考项目已有代码，逐步调试 |
| Bank Conflict 难以定位 | 中 | 延迟 2-4h | 使用 ncu profiling，对比项目实现 |
| FlashAttention 精度问题 | 高 | 延迟 4-8h | 检查 Online Softmax，确保 FP32 累加 |
| 性能达不到预期 | 中 | 延迟 2-4h | 使用 ncu 定位瓶颈，对比 cuBLAS |
| 4090 特定问题 | 低 | 延迟 1-2h | 检查 SM 8.9 特性，调整编译参数 |

### 7.2 时间风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| HGEMM 优化超时 | 中 | 压缩 P2 时间 | 优先保证功能正确，性能优化可选 |
| FlashAttention 实现困难 | 高 | 压缩 P2 时间 | 先实现 split_kv，再优化 |
| 环境问题 | 低 | 延迟 1 天 | 提前准备，预留缓冲时间 |

### 7.3 应急方案

**如果时间不够**：
1. 优先完成 P0 + P1 核心模块
2. P2 扩展模块可以选择性完成
3. 跳过 CuTe 实现，专注 MMA PTX

**如果遇到困难**：
1. 参考项目已有代码，理解设计思路
2. 使用 ncu 定位性能瓶颈
3. 简化实现，先保证正确性

---

## 📝 复现 Checklist

### Phase 1: 基础原语
□ reduce (warp_reduce, block_reduce)
□ elementwise (relu, gelu, add)
□ embedding
□ softmax (naive, safe, online)
□ rms-norm
□ layer-norm
□ rope
□ dot-product

### Phase 2: SGEMM
□ sgemm_naive
□ sgemm_sliced_k (tiling)
□ sgemm_t_8x8 (thread tile)
□ sgemm_vec4 (vectorize)
□ sgemm_bcf (bank conflict free)
□ sgemm_dbuf (double buffer)

### Phase 3: HGEMM
□ hgemm_naive
□ hgemm_wmma
□ hgemm_mma (m16n8k16)
□ hgemm_mma_stages
□ hgemm_mma_swizzle
□ hgemm_cute (optional)

### Phase 4: FlashAttention
□ flash_attn_split_kv
□ flash_attn_split_q
□ flash_attn_share_kv
□ flash_attn_share_qkv
□ flash_attn_tiling_qk (optional)

### Phase 5: 扩展
□ triton_softmax
□ triton_layer_norm
□ merge_attn_states
□ mat_transpose
□ nms (optional)

### Phase 6: 验证
□ 所有单元测试通过
□ 性能 benchmark 完成
□ 面试笔记整理完成

---

**祝复现顺利！🚀**
