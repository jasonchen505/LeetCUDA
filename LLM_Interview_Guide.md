# LLM 算法实习面试准备指南

> 基于 LeetCUDA 项目，面向 LLM & Agent 应用或后训练方向的算法实习面试

---

## 📋 目录

1. [项目概述与技术栈](#1-项目概述与技术栈)
2. [LLM 核心组件的 CUDA 实现](#2-llm-核心组件的-cuda-实现)
3. [关键优化技术深度解析](#3-关键优化技术深度解析)
4. [面试高频问题与深挖点](#4-面试高频问题与深挖点)
5. [LLM 推理优化专题](#5-llm-推理优化专题)
6. [算法岗面试差异化准备](#6-算法岗面试差异化准备)
7. [项目亮点包装与回答模板](#7-项目亮点包装与回答模板)

---

## 1. 项目概述与技术栈

### 1.1 项目定位

LeetCUDA 是一个 **CUDA 学习项目**，包含 200+ CUDA kernel 实现，覆盖了 LLM 训练和推理中的核心算子。项目核心亮点：

- **HGEMM**：达到 cuBLAS 98%~100% 性能的半精度矩阵乘法
- **FlashAttention**：使用纯 MMA PTX 实现的 FlashAttention-2
- **LLM 基础算子**：Softmax、LayerNorm、RMSNorm、RoPE、Embedding 等

### 1.2 技术栈速查

```
┌─────────────────────────────────────────────────────────────┐
│                    LeetCUDA 技术栈                          │
├─────────────────────────────────────────────────────────────┤
│ 编程语言: CUDA C++ (主要), Python (测试/绑定)              │
│ 精度格式: FP32, FP16, BF16, FP8 (E4M3/E5M2), TF32          │
│ Tensor Cores: WMMA (m16n16k16), MMA (m16n8k16), WGMMA      │
│ 优化技术: Tiling, Vectorize, Swizzle, Pipeline, etc.       │
│ 框架集成: PyTorch (pybind11 绑定)                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 面试中如何介绍这个项目

**30秒版本**（电梯演讲）：
> "我学习并实现了一个 CUDA 算子库 LeetCUDA，包含 LLM 训练推理中的核心算子。我深入研究了如何用 Tensor Cores 实现高性能 HGEMM 和 FlashAttention，理解了从 naive 实现到达到 cuBLAS 98%+ 性能的完整优化路径。这些工作让我对 LLM 推理的底层实现有深入理解。"

---

## 2. LLM 核心组件的 CUDA 实现

### 2.1 Embedding 层

**代码位置**: `kernels/embedding/embedding.cu`

**核心实现**:
```cuda
// Embedding 查表操作
__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight,
                                             float *output, int n, int emb_size) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  // 使用 128-bit 加载，提高带宽利用率
  LDST128BITS(output[bx * emb_size + 4 * tx]) =
      LDST128BITS(weight[offset + 4 * tx]);
}
```

**面试要点**:
- **问题**: "Embedding 的 CUDA 实现有什么优化空间？"
- **回答要点**:
  1. **访存模式**: Embedding 是 memory-bound 操作，优化重点在带宽利用
  2. **向量化加载**: 使用 `float4` (128-bit) 减少指令数
  3. **合并访问**: 确保同一 warp 的线程访问连续内存
  4. **实际瓶颈**: 在 LLM 中，Embedding 通常不是瓶颈，但 LM Head（输出层）的 softmax 是

**深挖问题**:
- Q: "Embedding 和 Linear 层在 CUDA 实现上有什么本质区别？"
- A: Embedding 是查表操作（gather），本质是不规则访存；Linear 是矩阵乘法，可以高度规整化。Embedding 更难优化因为访存模式取决于输入 index。

### 2.2 Softmax 实现

**代码位置**: `kernels/softmax/softmax.cu`

**三种实现递进**:

```cuda
// 1. Naive Softmax（数值不稳定）
exp_val = expf(x[idx]);
exp_sum = block_reduce_sum(exp_val);
y[idx] = exp_val / exp_sum;

// 2. Safe Softmax（减去最大值）
max_val = block_reduce_max(x[idx]);
exp_val = expf(x[idx] - max_val);
exp_sum = block_reduce_sum(exp_val);
y[idx] = exp_val / exp_sum;

// 3. Online Softmax（单 pass，用于 FlashAttention）
// 使用 MD 结构体维护 (max, sum) 状态
struct MD { float m; float d; };
// 在线更新：new_d = old_d * exp(old_m - new_m) + exp(x - new_m)
```

**面试要点**:

**问题1**: "为什么需要 Safe Softmax？"
```
当 x 很大时，exp(x) 会溢出（FP16 最大值 65504）
解决：减去最大值 max(x)，保证 exp 的输入 <= 0
数学等价性：exp(x_i) / sum(exp(x_j)) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

**问题2**: "Online Softmax 是怎么工作的？为什么 FlashAttention 需要它？"
```
Online Softmax 核心思想：
- 传统 softmax 需要两遍扫描：第一遍找 max，第二遍计算
- Online Softmax 单遍扫描，动态维护 (max, d) 对
- 当发现新的 max 时，对之前的 d 进行 rescale

FlashAttention 需要它：
- FlashAttention 分块处理 Q@K^T，每个块的 max 可能不同
- 需要 Online Softmax 在处理完所有块后得到正确的 softmax 结果
- 还需要保存 m, l 用于最后的 rescale
```

**问题3**: "block_reduce_sum 的实现细节？"
```
两级归约：
1. Warp 内使用 __shfl_xor_sync 做蝶形归约（O(log32)=5 步）
2. Warp leader 写入 shared memory
3. 第一个 warp 读取 shared memory，再次 warp reduce
4. 最后 __shfl_sync broadcast 结果到所有线程

关键细节：
- 为什么用 __shfl_xor_sync 而不是 __shfl_down_sync？
  XOR 模式所有线程做相同工作量，更均衡
- 为什么最后需要 broadcast？
  否则只有 warp0 的 lane0 知道结果，其他线程无法使用
```

### 2.3 LayerNorm / RMSNorm

**代码位置**: `kernels/layer-norm/layer_norm.cu`, `kernels/rms-norm/rms_norm.cu`

**LayerNorm 公式**:
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

**RMSNorm 公式**（LLaMA 等模型使用）:
```
y = x / sqrt(mean(x^2) + eps) * gamma
```

**CUDA 实现核心**:
```cuda
// RMSNorm: 一次 block reduce 计算 sum(x^2)
float variance = value * value;
variance = block_reduce_sum(variance);
if (tid == 0)
  s_variance = rsqrtf(variance / (float)K + epsilon);
y[idx] = (value * s_variance) * g;

// LayerNorm: 两次 block reduce
// 第一次：计算 mean
float sum = block_reduce_sum(value);
if (tid == 0) s_mean = sum / (float)K;
// 第二次：计算 variance
float variance = (value - s_mean) * (value - s_mean);
variance = block_reduce_sum(variance);
```

**面试要点**:

**问题1**: "RMSNorm 和 LayerNorm 的区别？为什么 LLaMA 用 RMSNorm？"
```
区别：
- LayerNorm: 减均值，除标准差，有 bias 项
- RMSNorm: 不减均值，用 RMS 值归一化，无 bias 项

LLaMA 用 RMSNorm 的原因：
1. 计算量少：省去一次 mean 计算（少一次 reduce）
2. 实验表明效果相当
3. 更适合大规模模型，减少通信开销
```

**问题2**: "Norm 层在训练和推理时有什么区别？"
```
训练时：
- 需要计算梯度，实现更复杂
- 通常使用 FP32 保持精度（mixed precision training）

推理时：
- 只需要前向传播
- 可以用 FP16/BF16
- 可以和前面的 Linear 层融合（kernel fusion）
```

### 2.4 RoPE 旋转位置编码

**代码位置**: `kernels/rope/rope.cu`

**数学公式**:
```
对于位置 pos，维度 i：
theta_i = 10000^(2i/d)
q_complex = q[2i] + j*q[2i+1]
k_complex = k[2i] + j*k[2i+1]

旋转后：
q' = q * exp(j * pos * theta_i)
k' = k * exp(j * pos * theta_i)

展开为实数运算：
out[2i]   = x[2i]*cos(pos*theta_i) - x[2i+1]*sin(pos*theta_i)
out[2i+1] = x[2i]*sin(pos*theta_i) + x[2i+1]*cos(pos*theta_i)
```

**CUDA 实现**:
```cuda
__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];
  int token_pos = idx / N;
  int token_idx = idx % N;
  float exp_v = 1.0f / powf(theta, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);
  out[idx * 2] = x1 * cos_v - x2 * sin_v;
  out[idx * 2 + 1] = x1 * sin_v + x2 * cos_v;
}
```

**面试要点**:

**问题1**: "RoPE 的核心思想是什么？为什么比绝对位置编码好？"
```
核心思想：
- 将位置信息编码到旋转矩阵中
- 通过复数乘法实现相对位置编码
- q·k 的内积只依赖于相对位置 (pos_k - pos_q)

优势：
1. 相对位置编码：泛化到更长序列
2. 远程衰减：距离越远，注意力越小
3. 无需额外参数
4. 可以高效实现（复数乘法 = 4次乘法 + 2次加法）
```

**问题2**: "RoPE 在 CUDA 实现上有什么优化点？"
```
1. 向量化：使用 float4 同时处理 2 对 (x1, x2)
2. 预计算：sin/cos 值可以预先计算并缓存
3. 融合：可以和 QKV Linear 融合
4. 长序列优化：对于长序列，可以分块计算
```

### 2.5 FlashAttention

**代码位置**: `kernels/flash-attn/`

**核心算法**:
```
输入: Q, K, V [B, H, N, d]
输出: O [B, H, N, d]

核心思想：
1. 分块处理：将 Q, K, V 分成小块（tile）
2. Online Softmax：逐块计算 Q@K^T，维护 running max 和 sum
3. 避免 materialize 完整的 attention matrix
4. 使用 SRAM 存储中间结果，减少 HBM 访问

算法流程：
for each Q_tile:
    for each K_tile, V_tile:
        S_tile = Q_tile @ K_tile^T
        m_tile = max(S_tile)  # 当前块的 max
        P_tile = exp(S_tile - m_tile)  # 当前块的 softmax
        l_tile = sum(P_tile)  # 当前块的 sum
        # 更新 running stats
        m_new = max(m_old, m_tile)
        l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
        # 更新输出
        O = (l_old * exp(m_old - m_new) * O + exp(m_tile - m_new) * P_tile @ V) / l_new
```

**Split Q vs Split KV**:
```
Split KV（FlashAttention-1）：
- 所有 MMA(Warps) 共同处理 Q, K, V
- 需要通过 shared memory 通信
- 性能较差

Split Q（FlashAttention-2）：
- 每个 Warp 处理不同的 Q slice
- 所有 Warp 共享 K, V
- 减少通信，性能更好
```

**面试要点**:

**问题1**: "FlashAttention 为什么能加速？减少了多少复杂度？"
```
加速原因：
1. 减少 HBM 访问：从 O(N^2) 降到 O(N)
2. 不 materialize N×N 的 attention matrix
3. 利用 SRAM 的高带宽

复杂度分析：
- 标准 Attention: O(N^2) 存储，O(N^2 d) 计算
- FlashAttention: O(N) 存储，O(N^2 d) 计算（计算量相同，但访存减少）

实际加速：
- 取决于 SRAM 大小和序列长度
- 通常 2-4x 加速，长序列更明显
```

**问题2**: "FlashAttention 的 Online Softmax 是怎么工作的？"
```
关键数据结构：
- m: running max（当前看到的最大值）
- l: running sum（exp(x - m) 的累加和）

更新规则（当处理新块时）：
1. 计算新块的 max_m 和 sum_l
2. 更新全局 max: m_new = max(m_old, max_m)
3. Rescale 旧的 l: l_old' = l_old * exp(m_old - m_new)
4. Rescale 新的 l: l_new' = sum_l * exp(max_m - m_new)
5. 更新 l: l_new = l_old' + l_new'

为什么需要 rescale：
- 因为 max 变了，之前计算的 exp 值需要调整
- 这就是 "online" 的代价：额外的 rescale 操作
```

**问题3**: "你实现的 FlashAttention 和官方 FA2 有什么区别？"
```
主要区别：
1. 精度：官方使用 FP32 累加，我的实现支持 FP16/FP32
2. 优化程度：官方有更多工程优化（如 warp specialization）
3. 硬件特性：官方利用更多硬件特性（如 TMA on Hopper）
4. 性能：大规模场景官方更快，小规模场景我的实现可能更快

我实现的特色：
1. 多种 SRAM 共享策略（Shared KV, Shared QKV）
2. Fine-grained Tiling 支持更大的 HeadDim
3. 完整的 Swizzle 支持消除 bank conflicts
```

---

## 3. 关键优化技术深度解析

### 3.1 Memory Coalescing（合并访问）

**核心概念**:
```
Warp 内 32 个线程的内存访问会被硬件合并为 1-32 次内存事务。

最佳情况（合并访问）：
- 线程 0 访问 addr[0]，线程 1 访问 addr[1]，...
- 32 个连续地址 → 1 次 128B 事务

最坏情况（离散访问）：
- 线程 0 访问 addr[0]，线程 1 访问 addr[1000]，...
- 32 个分散地址 → 最多 32 次事务
```

**代码示例**（来自 `kernels/relu/relu.cu`）:
```cuda
// 合并访问：相邻线程访问相邻地址
int idx = blockIdx.x * blockDim.x + threadIdx.x;  // ✓
y[idx] = fmaxf(0.0f, x[idx]);

// 非合并访问：需要转置时
int idx = blockIdx.x * blockDim.x + threadIdx.x;
y[idx * N] = x[idx * N];  // ✗ 跨步访问

// 解决方案：使用 shared memory 做转置
```

**面试问题**:
- Q: "如何判断一个 kernel 是否有合并访问问题？"
- A: 使用 `ncu` (Nsight Compute) profiling，查看 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` 指标。理想情况是每次 warp 访问产生 1 个 sector（32B），如果远大于 1 说明有合并问题。

### 3.2 Bank Conflict

**核心概念**:
```
Shared Memory 分为 32 个 bank，每个 bank 宽 4 bytes。
同一 warp 的线程如果访问同一 bank 的不同地址 → bank conflict → 串行化。

Bank 计算：addr / 4 % 32

示例：
float s[32][32];
// 线程 i 访问 s[0][i] → 访问 bank i → 无 conflict ✓
// 线程 i 访问 s[i][0] → 访问 bank 0 → 32-way conflict ✗
```

**解决方案**:
```cuda
// 方案 1: Padding（最常用）
__shared__ float s[32][32 + 1];  // +1 打破对齐

// 方案 2: Swizzle（CUTLASS/CuTe 使用）
// 通过位运算重新映射 bank 地址
// 优点：不浪费空间，更灵活
```

**代码示例**（来自 SGEMM 实现）:
```cuda
// 使用 padding 避免 bank conflict
__shared__ float s_a[BK][BM + OFFSET];  // OFFSET 通常是 1 或 4

// 分析 bank layout：
// s_a[8][128] 有 4 bytes/bank, 32 banks
// 每行 128 floats = 128/32 = 4 个 bank layer
// 如果不做 padding，相邻线程访问同一 bank
```

**面试问题**:
- Q: "为什么 HGEMM 实现中，NN layout 用 padding，TN layout 用 swizzle？"
- A: NN layout（A和B都是行主序）时，bank conflict 模式固定，padding 简单有效。TN layout（A行主序B列主序）时，conflict 模式复杂，swizzle 更通用且不浪费空间。

### 3.3 Vectorized Memory Access（向量化访问）

**核心思想**:
```
使用向量类型（float4, half2）一次加载多个元素，减少指令数。

float4 = 128 bits = 4 × 32-bit float
half2 = 32 bits = 2 × 16-bit half
int4 = 128 bits = 4 × 32-bit int

优势：
1. 减少 load/store 指令数（1条 vs 4条）
2. 提高带宽利用率（128-bit 对齐访问更高效）
3. 减少指令调度开销
```

**代码示例**:
```cuda
// 标量版本
for (int i = 0; i < 4; i++) {
  y[idx + i] = x[idx + i];  // 4 条 load + 4 条 store
}

// 向量化版本
float4 reg = FLOAT4(x[idx]);  // 1 条 128-bit load
FLOAT4(y[idx]) = reg;          // 1 条 128-bit store

// 宏定义
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
```

### 3.4 Double Buffering（双缓冲）

**核心思想**:
```
在计算当前数据块的同时，预取下一个数据块到另一个 buffer。
通过计算和访存重叠，隐藏内存延迟。

实现：
1. 分配 2 倍的 shared memory
2. 主循环：计算 buffer[sel] 的同时加载 buffer[sel^1]
3. sel = sel ^ 1 交替使用
```

**代码结构**（来自 HGEMM 实现）:
```cuda
// 主循环
for (int bk = 1; bk < num_blocks; bk++) {
  // 1. 加载下一块到 buffer[sel^1]
  load_to_smem(s_a[sel^1], ...);
  load_to_smem(s_b[sel^1], ...);
  
  // 2. 计算当前块 buffer[sel]
  compute(s_a[sel], s_b[sel], ...);
  
  // 3. 切换 buffer
  sel ^= 1;
  __syncthreads();
}
```

**面试问题**:
- Q: "双缓冲为什么能加速？"
- A: GPU 不能像 CPU 那样乱序执行。如果先加载再计算，加载指令会阻塞后续计算指令。双缓冲让加载和计算使用不同的 buffer，硬件可以并行执行它们。

### 3.5 Tensor Cores（WMMA / MMA）

**核心概念**:
```
Tensor Cores 是专门做矩阵乘法的硬件单元。

WMMA API（高级 API）：
- Warp 级操作，32 个线程协作完成矩阵乘
- m16n16k16: 16×16 × 16×16 = 16×16 矩阵乘
- 每条指令完成 4096 次乘加

MMA PTX（底层指令）：
- 更细粒度控制
- m16n8k16: 16×8 × 8×16 = 16×8 矩阵乘
- 可以更灵活地组织数据

WGMMA（Hopper）：
- Warpgroup 级操作，128 个线程
- m64n128k16: 更大的矩阵乘
- 异步执行
```

**代码示例**（MMA 指令）:
```cuda
// MMA m16n8k16 指令
// A: 16×16, B: 16×8, C: 16×8
// 每个线程持有 A 的 4 个 32-bit 寄存器，B 的 2 个，C 的 2 个
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
  "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
  : "=r"(R_C[0]), "=r"(R_C[1])
  : "r"(R_A[0]), "r"(R_A[1]), "r"(R_A[2]), "r"(R_A[3]),
    "r"(R_B[0]), "r"(R_B[1]),
    "r"(R_C[0]), "r"(R_C[1])
);
```

**面试问题**:
- Q: "WMMA 和 MMA 的区别？为什么项目里用 MMA 而不是 WMMA？"
- A: WMMA 是高级 API，易用但灵活性低。MMA 是底层 PTX 指令，可以更精细地控制数据布局和指令调度。项目用 MMA 是因为：
  1. 可以实现更复杂的 tiling 策略
  2. 可以减少寄存器使用
  3. 性能通常更好（5-10%）

### 3.6 SMEM Swizzle

**核心概念**:
```
Swizzle 是一种通过位运算重新映射 shared memory 地址的技术，可以完全消除 bank conflict。

原理：
- 传统布局：addr = row * stride + col
- Swizzle 布局：addr = row * stride + (col XOR row)

效果：
- 相邻行的访问会映射到不同的 bank
- 完全消除 bank conflict，且不浪费空间
```

**面试问题**:
- Q: "Swizzle 和 Padding 的区别？"
- A: 
  - Padding：简单，在每行末尾加空元素打破对齐。缺点：浪费空间，可能影响 occupancy
  - Swizzle：通过位运算重映射，不浪费空间。缺点：实现复杂，需要理解 bank layout
  - CUTLASS/CuTe 使用 Swizzle 是因为更通用、更高效

---

## 4. 面试高频问题与深挖点

### 4.1 关于 Softmax 的深挖

**Q1**: "Softmax 的数值稳定性问题是什么？怎么解决？"
```
问题：
- exp(x) 在 x 很大时会溢出（FP16: 65504, FP32: 3.4e38）
- exp(x) 在 x 很小时会下溢为 0

解决方案：
1. Safe Softmax：减去最大值 max(x)
   - 保证 exp 的输入 <= 0
   - 不会溢出，但可能下溢（下溢是安全的，因为 0 是有效值）

2. 数学等价性证明：
   exp(x_i) / sum(exp(x_j))
   = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
   = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

3. Online Softmax：
   - 在分块处理时，max 可能变化
   - 需要对之前的 sum 进行 rescale
```

**Q2**: "Softmax 在 LLM 推理中是瓶颈吗？"
```
分析：
- Softmax 是 memory-bound 操作（AI ≈ 0.625 FLOPS/Byte）
- 在 Attention 中，Softmax 的复杂度是 O(N^2)
- 对于长序列（N > 4096），Softmax 可能成为瓶颈

优化方向：
1. FlashAttention：避免 materialize 完整的 softmax
2. Kernel Fusion：和 Q@K^T 融合
3. 使用 FP16 计算，FP32 累加
```

### 4.2 关于 GEMM 的深挖

**Q1**: "从 naive SGEMM 到高性能 HGEMM 的优化路径？"
```
优化路径（LeetCUDA 项目展示了完整路径）：

1. Naive SGEMM：
   - 每个线程计算一个 C 元素
   - 性能：~5% cuBLAS

2. Block Tile + K Tile：
   - 使用 shared memory 缓存数据
   - 减少 global memory 访问
   - 性能：~20% cuBLAS

3. Thread Tile：
   - 每个线程计算多个元素（如 8×8）
   - 提高计算密度
   - 性能：~40% cuBLAS

4. Vectorize（float4）：
   - 128-bit 加载/存储
   - 减少指令数
   - 性能：~60% cuBLAS

5. Double Buffering：
   - 计算和访存重叠
   - 性能：~80% cuBLAS

6. Tensor Cores（WMMA/MMA）：
   - 利用专用硬件
   - 性能：~90% cuBLAS

7. Swizzle + Pipeline：
   - 消除 bank conflict
   - 多级流水线
   - 性能：~98% cuBLAS
```

**Q2**: "为什么 HGEMM 比 SGEMM 更适合 LLM？"
```
原因：
1. 精度足够：LLM 推理通常 FP16 足够
2. 带宽减半：FP16 只需 FP32 一半的带宽
3. Tensor Cores：专为 FP16 设计，吞吐量更高
4. 存储减半：KV Cache 等用 FP16 存储

性能对比：
- SGEMM: 理论峰值 ~30 TFLOPS（FP32 Tensor Cores）
- HGEMM: 理论峰值 ~120 TFLOPS（FP16 Tensor Cores）
- 4x 吞吐量差距
```

### 4.3 关于 FlashAttention 的深挖

**Q1**: "FlashAttention 的 SRAM 使用策略？"
```
SRAM 使用：
1. Standard FA2：
   - Q: O(Br × d)
   - K: O(Bc × d) × stages
   - V: O(Bc × d)
   - 总计: O(4 × Br × d)

2. Shared KV SMEM（项目实现）：
   - K, V 共享同一块 SMEM
   - SRAM 减半：O(2 × Br × d)

3. Shared QKV SMEM（项目实现）：
   - Q, K, V 全部共享
   - SRAM 降为 1/4：O(Br × d)

4. Fine-grained Tiling（项目实现）：
   - 在 MMA 级别做 Tiling
   - SRAM 复杂度 O(16 × d)
   - 可以支持更大的 HeadDim（1024+）
```

**Q2**: "FlashAttention 的精度问题？"
```
精度来源：
1. MMA 累加精度：
   - FP16 累加：快但精度低
   - FP32 累加：慢但精度高（官方 FA2 使用）

2. Softmax 精度：
   - 必须使用 FP32 计算 max 和 sum
   - FP16 的 exp 精度不够

3. 项目实现的精度选择：
   - MMA Acc F16：性能更好，精度 ~1e-3
   - MMA Acc F32：精度更高，但性能下降 10-20%

4. 与官方 FA2 的精度差异：
   - 官方：MMA F32 + Softmax F32
   - 项目：MMA F16/F32 + Softmax F32
   - 典型误差：< 1e-3
```

### 4.4 关于 LLM 推理的深挖

**Q1**: "LLM 推理的主要瓶颈是什么？"
```
瓶颈分析：

1. Prefill 阶段（处理 prompt）：
   - 瓶颈：Compute-bound
   - 主要计算：Q@K^T, P@V, Linear
   - 优化方向：FlashAttention, Tensor Cores

2. Decode 阶段（生成 token）：
   - 瓶颈：Memory-bound
   - 主要瓶颈：KV Cache 读取
   - 优化方向：KV Cache 量化, GQA, PagedAttention

3. 内存瓶颈：
   - KV Cache：O(N × L × d)，N=seq_len, L=层数
   - 权重：O(L × d^2)
   - 激活值：O(B × N × d)
```

**Q2**: "KV Cache 优化有哪些方法？"
```
方法分类：

1. 减少 KV Cache 大小：
   - GQA（Grouped Query Attention）：多个 Q head 共享 K, V
   - MQA（Multi Query Attention）：所有 Q head 共享 K, V
   - CLA（Cross Layer Attention）：层间共享 KV Cache

2. KV Cache 量化：
   - FP8 量化：E4M3 或 E5M2
   - INT8 量化：更激进，需要校准
   - Per-channel 或 Per-token 量化

3. KV Cache 管理：
   - PagedAttention（vLLM）：按页管理，减少碎片
   - Prefix Caching：共享公共前缀
   - Sliding Window：只保留最近的 KV

4. 计算优化：
   - FlashDecoding：并行化 decode 阶段
   - Continuous Batching：动态 batch
```

**Q3**: "如何做 LLM 的后训练（Post-Training）？"
```
后训练方法：

1. SFT（Supervised Fine-Tuning）：
   - 使用高质量指令数据微调
   - 保持 base model 的能力，学习指令跟随

2. RLHF（Reinforcement Learning from Human Feedback）：
   - 训练 Reward Model
   - 使用 PPO 优化 policy
   - 需要大量人类偏好数据

3. DPO（Direct Preference Optimization）：
   - 直接从偏好数据优化
   - 无需训练 Reward Model
   - 更简单稳定

4. 量化后训练：
   - PTQ（Post-Training Quantization）：训练后量化
   - QAT（Quantization Aware Training）：量化感知训练

5. 对齐技术：
   - Constitutional AI
   - RLAIF（RL from AI Feedback）
   - Self-Play
```

---

## 5. LLM 推理优化专题

### 5.1 推理框架对比

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM 推理框架对比                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│    特性      │   vLLM       │  TensorRT-LLM│   TGI          │
├──────────────┼──────────────┼──────────────┼────────────────┤
│   核心技术   │ PagedAttention│ TRT优化      │ 连续批处理     │
│   KV Cache   │ 按页管理     │ 静态/动态    │ 连续内存       │
│   量化       │ GPTQ/AWQ     │ FP8/INT8     │ GPTQ/AWQ      │
│   多机       │ 支持         │ 支持         │ 有限           │
│   易用性     │ 高           │ 中           │ 高             │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### 5.2 量化技术详解

**Q1**: "LLM 量化的挑战是什么？"
```
挑战：
1. 离群值（Outliers）：
   - LLM 的激活值有少量离群值
   - 直接量化会丢失这些重要信息
   - 解决：SmoothQuant, AWQ

2. 权重量化 vs 激活量化：
   - 权重：静态已知，可以离线校准
   - 激活：动态变化，需要在线处理
   - 通常只量化权重（Weight-Only Quantization）

3. 精度-性能权衡：
   - INT8：几乎无损，2x 压缩
   - INT4：需要校准，4x 压缩
   - FP8：硬件支持，接近 FP16 精度

4. Kernel 实现：
   - 量化/反量化需要高效实现
   - 需要和 Linear 融合
   - 需要硬件支持（如 INT8 Tensor Cores）
```

**Q2**: "GPTQ 和 AWQ 的区别？"
```
GPTQ（GPT Quantization）：
- 基于 OBQ（Optimal Brain Quantization）
- 逐层量化，最小化重建误差
- 需要校准数据
- 支持 INT4/INT3

AWQ（Activation-Aware Weight Quantization）：
- 观察：少数权重通道对输出影响大
- 保护这些"重要"通道的精度
- 混合精度：重要通道用高精度，其他用低精度
- 通常比 GPTQ 精度更好

选择建议：
- 如果需要最激进的压缩（INT3/INT2）：GPTQ
- 如果需要更好的精度（INT4）：AWQ
```

### 5.3 Speculative Decoding

**Q1**: "什么是 Speculative Decoding？"
```
核心思想：
- 使用小模型（draft model）快速生成多个候选 token
- 使用大模型（target model）并行验证这些候选
- 如果候选正确，一次生成多个 token

算法流程：
1. Draft Model 生成 K 个候选 token
2. Target Model 并行计算这 K 个 token 的概率
3. 从后往前验证，接受概率 > 阈值的 token
4. 如果被拒绝，从拒绝位置重新采样

加速效果：
- 取决于 draft model 的准确率
- 通常 2-3x 加速
- 保证输出分布不变（理论上）
```

### 5.4 Continuous Batching

**Q1**: "Continuous Batching 和 Static Batching 的区别？"
```
Static Batching：
- 所有请求必须等最长的那个完成
- 短请求完成后 GPU 空闲
- 浪费计算资源

Continuous Batching：
- 请求级别调度，不是 batch 级别
- 短请求完成后立即插入新请求
- GPU 始终保持忙碌

实现细节：
1. 追踪每个请求的状态（prefill/decode/完成）
2. 动态调整 batch 组成
3. 需要 PagedAttention 支持动态 KV Cache 管理
```

---

## 6. 算法岗面试差异化准备

### 6.1 算法 vs Infra 的区别

```
Infra 岗位关注：
- 性能优化：TFLOPS, 延迟, 吞吐量
- 系统设计：分布式, 调度, 内存管理
- 工程实现：CUDA kernel, 编译优化

算法岗位关注：
- 模型设计：架构选择, 精度-效率权衡
- 训练策略：数据, 损失函数, 优化器
- 应用效果：下游任务, 用户体验

你的优势：
- 理解底层实现，能做出更 informed 的算法决策
- 知道什么优化是可能的，什么是有代价的
- 能和 Infra 团队有效沟通
```

### 6.2 算法岗需要补充的知识

**Q1**: "你了解的 LLM 训练技术？"
```
需要了解：

1. 预训练技术：
   - 数据配比：代码、数学、多语言
   - 训练稳定性：Loss spike, 学习率调整
   - Scaling Laws：模型大小 vs 数据量 vs 计算量

2. 后训练技术：
   - SFT：数据质量 > 数量
   - RLHF/DPO：对齐技术
   - 安全性：红队测试, 安全对齐

3. 高效训练：
   - 混合精度训练
   - 梯度累积
   - ZeRO 优化器
   - 序列并行
```

**Q2**: "Agent 应用中的技术挑战？"
```
挑战：

1. 长上下文处理：
   - 长文档理解
   - 多轮对话管理
   - KV Cache 管理

2. 工具调用：
   - 函数调用格式
   - 错误处理
   - 多步推理

3. 规划与推理：
   - 任务分解
   - 链式思考（CoT）
   - 自我纠错

4. 记忆管理：
   - 短期记忆（上下文窗口）
   - 长期记忆（向量数据库）
   - 工作记忆（当前任务）

5. 多模态：
   - 图像理解
   - 代码执行
   - 外部 API 调用
```

### 6.3 如何展示你的技术深度

**策略1**: 从算法角度解释 infra 优化
```
例如，解释 FlashAttention 时，不要只说"减少 HBM 访问"，
而是要说：

"FlashAttention 对算法研究的影响：
1. 使得长序列训练成为可能（之前受内存限制）
2. 改变了模型设计：可以使用更长的上下文
3. 影响了位置编码的选择：RoPE 更适合长序列
4. 催生了长上下文模型：如 Llama 3.1 128K"
```

**策略2**: 用具体数字说明理解深度
```
例如：
"在实现 HGEMM 时，我发现：
- 使用 MMA PTX 比 WMMA API 快 5-10%
- SMEM Padding 可以减少 50% 的 bank conflict
- Double Buffering 可以隐藏 80% 的内存延迟
- 最终达到 cuBLAS 98% 的性能"
```

**策略3**: 展示 problem-solving 能力
```
例如：
"在实现 FlashAttention 时，我遇到了精度问题：
- 问题：MMA 使用 FP16 累加导致精度损失
- 分析：误差在 softmax 的 rescale 时被放大
- 解决：关键路径使用 FP32，其他用 FP16
- 结果：精度损失从 1e-2 降到 1e-5"
```

---

## 7. 项目亮点包装与回答模板

### 7.1 项目介绍模板

**1分钟版本**：
```
"我学习并实现了一个 CUDA 算子库，包含 LLM 训练推理的核心算子。

三个核心亮点：
1. HGEMM：达到 cuBLAS 98% 性能，理解了从 naive 到高性能的完整优化路径
2. FlashAttention：实现了 FA2 的核心算法，包括 Online Softmax 和 Split Q 策略
3. LLM 基础算子：Softmax、Norm、RoPE 等，理解了它们在 LLM 中的作用和优化

通过这个项目，我不仅理解了 LLM 的底层实现，还能从算法角度评估不同优化的权衡。"
```

### 7.2 常见问题回答模板

**Q**: "你为什么要做这个项目？"
```
"三个原因：

1. 理解 LLM 底层：作为算法研究者，我想知道模型的每个组件是怎么实现的，
   这样能做出更好的设计决策。

2. 评估优化权衡：比如，FlashAttention 减少了内存但增加了计算，
   我想知道在什么场景下这个权衡是值得的。

3. 与 Infra 团队协作：理解底层实现后，我能更好地和 Infra 团队沟通，
   知道什么优化是可能的，什么是有代价的。"
```

**Q**: "这个项目对你的算法研究有什么帮助？"
```
"帮助很大，举几个例子：

1. 模型设计：知道 RMSNorm 比 LayerNorm 快 10%，在设计新模型时会优先考虑

2. 长上下文：理解 FlashAttention 后，我知道长序列训练是可行的，
   这影响了我对上下文长度的选择

3. 量化：理解量化实现后，我知道 FP8 几乎无损，但 INT4 需要仔细校准，
   这影响了我对部署方案的选择

4. 推理优化：知道 KV Cache 是主要瓶颈后，我会优先考虑 GQA、MQA 等架构"
```

**Q**: "你遇到的最大挑战是什么？"
```
"在实现 FlashAttention 时，遇到了精度问题：

问题：MMA 使用 FP16 累加，但 softmax 需要 FP32 精度
分析：误差在 softmax 的 rescale 时被放大，特别是长序列
解决：
1. 关键路径（max, sum）使用 FP32
2. MMA 累加支持 FP16/FFP32 切换
3. 最后 rescale 使用 FP32

收获：
1. 理解了数值稳定性的重要性
2. 学会了如何分析和定位精度问题
3. 知道了哪些优化是有代价的"
```

### 7.3 技术深度展示技巧

**技巧1**: 使用类比解释复杂概念
```
例如，解释 Online Softmax：
"Online Softmax 就像在跑步比赛中实时计算平均配速。
传统方法：跑完全程再算（需要两遍扫描）
Online 方法：每跑一公里就更新一次配速，但要记住之前的配速需要调整"
```

**技巧2**: 用数据支撑观点
```
例如：
"在 HGEMM 优化中，我测试了每种优化技术的效果：
- Naive: 5 TFLOPS (5% cuBLAS)
- +Tiling: 20 TFLOPS (20% cuBLAS)
- +Vectorize: 60 TFLOPS (60% cuBLAS)
- +Tensor Cores: 100 TFLOPS (90% cuBLAS)
- +Swizzle: 110 TFLOPS (98% cuBLAS)

这让我清楚地知道每种优化的贡献。"
```

**技巧3**: 展示思考过程
```
例如：
"在选择 FlashAttention 的 SRAM 策略时，我考虑了三个因素：
1. SRAM 使用量：Shared QKV 最省，但可能影响 occupancy
2. 精度：MMA F32 更准，但慢 20%
3. 实现复杂度：Shared KV 最简单

最终我选择了 Shared KV 作为默认，因为：
- SRAM 减半，occupancy 提高
- 精度足够（误差 < 1e-3）
- 实现简单，易于维护"
```

---

## 📝 面试 Checklist

### 基础知识
- [ ] GPU 架构：SM, Warp, Tensor Cores
- [ ] Memory Hierarchy：HBM, L2, L1/SMEM, Registers
- [ ] CUDA 编程：Grid, Block, Thread
- [ ] 性能分析：Roofline Model, Occupancy

### LLM 核心组件
- [ ] Embedding：查表操作，memory-bound
- [ ] Softmax：Safe/Online，数值稳定性
- [ ] Norm：LayerNorm vs RMSNorm
- [ ] Attention：标准/Flash，SRAM 策略
- [ ] RoPE：相对位置编码，复数乘法

### 优化技术
- [ ] Memory Coalescing
- [ ] Bank Conflict & Swizzle
- [ ] Vectorized Access (float4)
- [ ] Double Buffering
- [ ] Tensor Cores (WMMA/MMA)

### LLM 推理优化
- [ ] KV Cache：GQA/MQA, 量化, PagedAttention
- [ ] 量化：GPTQ/AWQ, FP8/INT8
- [ ] Speculative Decoding
- [ ] Continuous Batching

### 算法视角
- [ ] 模型设计：架构选择，精度-效率权衡
- [ ] 后训练：SFT, RLHF/DPO
- [ ] Agent：长上下文，工具调用，规划
- [ ] 与 Infra 协作：知道什么优化是可能的

---

## 📚 推荐阅读

### 论文
1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
3. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
4. **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization"
5. **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
6. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
7. **vLLM**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"

### 博客（项目中的 100+ 篇）
1. 图解:从Online-Softmax到FlashAttention V1/V2/V3
2. 高频面试题汇总-大模型手撕CUDA
3. 图解vLLM Automatic Prefix Caching
4. 原理&图解FlashDecoding/FlashDecoding++
5. GQA/YOCO/CLA/MLKV: 层内和层间KV Cache共享

---

**祝面试顺利！🚀**
