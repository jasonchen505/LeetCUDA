# LeetCUDA 增量学习笔记

> 基于前两轮面试准备文档，记录复现过程中新学习到的点

---

## 📋 笔记说明

本文档记录在复现 LeetCUDA 项目过程中，相对于前两轮文档（`LLM_Interview_Guide.md` 和 `Interview_Five_Categories_Guide.md`）**新学习到的知识点**。

**前两轮覆盖内容**：
- LLM 核心组件的 CUDA 实现（Softmax, Norm, RoPE, Embedding）
- 关键优化技术（Memory Coalescing, Bank Conflict, Vectorize, Double Buffering, Tensor Cores）
- FlashAttention 核心算法
- 面试五类问题应对策略

**本轮新增内容**：
- 4090 硬件特性与 Ada 架构深度理解
- HGEMM 完整优化路径（从 naive 到 cuBLAS 98%）
- MMA PTX 指令深度解析
- SMEM Swizzle 机制详解
- Triton 编程模型
- 多卡并行优化
- 实际复现中的问题与解决方案

---

## 1. RTX 4090 与 Ada 架构深度理解

### 1.1 Ada 架构新特性

**新增知识点**：

```
Ada Lovelace (SM 8.9) vs Ampere (SM 8.0/8.6) 的关键差异：

1. 第 4 代 Tensor Cores：
   - 支持 FP8 Tensor Cores（E4M3/E5M2）
   - FP16 吞吐量：330 TFLOPS（vs A100 312 TFLOPS）
   - 新增 Hopper 的一些 MMA 特性（部分）

2. 显存子系统：
   - L2 Cache: 73 MB（vs A100 40 MB）
   - 显存带宽: 1008 GB/s（vs A100 2039 GB/s HBM2e）
   - 注意：4090 是 GDDR6X，不是 HBM

3. SM 资源：
   - Shared Memory: 100 KB（可配置到 228 KB）
   - Register File: 256 KB
   - Max Threads per SM: 1536
   - Max Warps per SM: 48

4. 新增指令：
   - cp.async.bulk（部分支持）
   - stmatrix（部分支持）
```

**面试关联**：
```
问题：「4090 和 A100 在 LLM 推理上有什么区别？」

回答要点：
1. 计算能力：4090 FP16 Tensor Core 330 TFLOPS vs A100 312 TFLOPS
2. 显存：4090 24GB GDDR6X vs A100 80GB HBM2e
3. 带宽：4090 1008 GB/s vs A100 2039 GB/s
4. 互连：4090 无 NVLink vs A100 600 GB/s NVLink
5. 适用场景：
   - 4090：单卡推理、小模型训练、性价比高
   - A100：大模型训练、多卡并行、稳定性高
```

### 1.2 4090 性能特征

**新增知识点**：

```
4090 在不同算子上的性能特征：

1. Compute-bound 算子（GEMM, Attention）：
   - 理论峰值 330 TFLOPS（FP16）
   - 实际可达 95%+ 利用率
   - 优势：高时钟频率，Tensor Core 数量多

2. Memory-bound 算子（Softmax, Norm, Elementwise）：
   - 理论带宽 1008 GB/s
   - 实际可达 80-90% 利用率
   - 优势：大 L2 Cache（73 MB）

3. 混合算子（FlashAttention）：
   - 小规模（D=64, N<8192）：Compute-bound
   - 大规模（D=512, N>16384）：Memory-bound
   - 4090 在小规模场景优势明显
```

---

## 2. HGEMM 完整优化路径深度解析

### 2.1 从 Naive 到 cuBLAS 98% 的 7 个层次

**新增知识点**：

```
HGEMM 优化 7 层金字塔（LeetCUDA 完整展示）：

Level 0 - Naive:
├── 实现：每个线程计算一个 C 元素
├── 性能：~5% cuBLAS
├── 瓶颈：global memory 访问太多
└── 关键问题：没有数据复用

Level 1 - Block Tile + K Tile (SMEM):
├── 实现：使用 shared memory 缓存 A/B tile
├── 性能：~20% cuBLAS
├── 改进：数据从 HBM 搬到 SMEM 复用
└── 关键概念：Tiling, SMEM, __syncthreads()

Level 2 - Thread Tile (Register Blocking):
├── 实现：每个线程计算 TM×TN 个元素
├── 性能：~40% cuBLAS
├── 改进：提高计算密度，减少线程数
└── 关键概念：计算/访存比 (Arithmetic Intensity)

Level 3 - Vectorize (float4/half2):
├── 实现：128-bit 加载/存储
├── 性能：~60% cuBLAS
├── 改进：减少 load/store 指令数
└── 关键概念：LDST128BITS, 对齐访问

Level 4 - Double Buffering (Pipeline):
├── 实现：计算和访存使用不同 buffer
├── 性能：~80% cuBLAS
├── 改进：隐藏内存延迟
└── 关键概念：cp.async, 流水线

Level 5 - Tensor Cores (MMA m16n8k16):
├── 实现：使用 MMA PTX 指令
├── 性能：~90% cuBLAS
├── 改进：硬件矩阵乘单元
└── 关键概念：ldmatrix, HMMA16816

Level 6 - Multistage + Swizzle:
├── 实现：多级流水线 + SMEM Swizzle
├── 性能：~98% cuBLAS
├── 改进：消除 bank conflict，优化 L2 cache
└── 关键概念：Block Swizzle, SMEM Swizzle
```

### 2.2 MMA m16n8k16 指令深度解析

**新增知识点**：

```
MMA m16n8k16 指令详解：

1. 指令格式：
   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {D0, D1}, {A0,A1,A2,A3}, {B0,B1}, {C0,C1}

2. 寄存器布局：
   - A 矩阵 [16×16]: 4 个 32-bit 寄存器 (RA[0..3])
   - B 矩阵 [16×8]: 2 个 32-bit 寄存器 (RB[0..1])
   - C/D 矩阵 [16×8]: 2 个 32-bit 寄存器 (RC[0..1], RD[0..1])

3. 线程到数据的映射：
   - 32 个线程协作完成 16×8 的矩阵乘
   - 每个线程持有 C 的 4 个 half 值（2 个 uint32）
   - 线程 i 持有 C[i/4, (i%4)*2..(i%4)*2+1] 和 C[i/4+8, (i%4)*2..(i%4)*2+1]

4. 数据布局要求：
   - A: row-major（与指令的 .row 匹配）
   - B: col-major（与指令的 .col 匹配）
   - 如果 B 是 row-major，需要使用 .trans 版本的 ldmatrix

5. ldmatrix 指令：
   - ldmatrix.sync.aligned.x4.m8n8.shared.b16: 加载 4 个 8×8 矩阵片段
   - ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16: 转置加载 2 个 8×8 矩阵片段
```

**面试关联**：
```
问题：「MMA m16n8k16 的寄存器布局是怎样的？」

深度回答：
1. A 矩阵 [16×16]:
   - 每个线程持有 4 个 uint32（RA[0..3]）
   - RA[0] 持有行 0-7 的 8 个 half（4 对）
   - RA[1] 持有行 0-7 的另外 8 个 half
   - RA[2] 持有行 8-15 的 8 个 half
   - RA[3] 持有行 8-15 的另外 8 个 half

2. B 矩阵 [16×8]:
   - 每个线程持有 2 个 uint32（RB[0..1]）
   - RB[0] 持有列 0-3 的 4 个 half
   - RB[1] 持有列 4-7 的 4 个 half

3. C 矩阵 [16×8]:
   - 每个线程持有 2 个 uint32（RC[0..1]）
   - RC[0] 持有行 0-7 的 4 个 half
   - RC[1] 持有行 8-15 的 4 个 half
```

### 2.3 TN 布局详解

**新增知识点**：

```
TN 布局（T=A 行优先，N=B 列优先）详解：

1. BLAS 约定：
   - N = Normal（列优先，BLAS 原生格式）
   - T = Transposed（行优先，相对 BLAS 来说是"转置过的"）
   - TN 表示：A 是行优先（T），B 是列优先（N）

2. 内存布局：
   - A[M×K]: row-major → A[m][k] = A[m*K + k]
   - B[K×N]: col-major → B[k][n] = B[n*K + k]
   - 等价于 B^T[N×K]: row-major → B^T[n][k] = B[n*K + k]

3. 为什么 TN 布局适合 MMA：
   - MMA 指令要求 A row-major, B col-major
   - TN 布局天然匹配，无需转置
   - ldmatrix 加载 A 用 x4（非转置）
   - ldmatrix 加载 B 用 x2（非转置，因为 SMEM 中存的是 B^T row-major）

4. NN vs TN 布局对比：
   - NN: A row-major, B row-major
     - 需要 ldmatrix.x2.trans 来加载 B
     - 需要额外的转置开销
   - TN: A row-major, B col-major
     - ldmatrix.x2 直接加载，无需转置
     - 性能更好
```

---

## 3. SMEM Swizzle 机制详解

### 3.1 Bank Conflict 根因分析

**新增知识点**：

```
Bank Conflict 的根本原因：

1. Shared Memory 结构：
   - 32 个 bank，每个 bank 宽 4 bytes
   - 连续的 4 bytes 地址映射到连续的 bank
   - Bank 计算：bank = (addr / 4) % 32

2. Conflict 产生条件：
   - 同一 warp 的多个线程访问同一 bank 的不同地址
   - 访问被串行化，变成 n 次（n-way conflict）

3. 特殊情况：
   - 所有线程访问同一地址：broadcast，无 conflict
   - 所有线程访问同一 bank 的同一地址：broadcast，无 conflict

4. 分析工具：
   - ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
   - 手动计算：分析每个线程访问的地址，计算 bank 映射
```

### 3.2 Padding vs Swizzle

**新增知识点**：

```
Padding 和 Swizzle 的对比：

1. Padding 方案：
   - 原理：在每行末尾加空元素，打破 bank 对齐
   - 实现：s_a[BM][BK + PAD]
   - 优点：实现简单
   - 缺点：浪费 SMEM 空间，可能影响 occupancy
   - 适用：NN 布局，bank conflict 模式固定

2. Swizzle 方案：
   - 原理：通过位运算重新映射 bank 地址
   - 实现：addr_swizzled = addr XOR (row << shift)
   - 优点：不浪费空间，更通用
   - 缺点：实现复杂
   - 适用：TN 布局，复杂访问模式

3. Swizzle 参数：
   - Swizzle<B, M, S>：B=base, M=mask, S=shift
   - 例：Swizzle<3, 3, 3> 表示每 8 行一个 swizzle 周期
   - 需要与访问模式匹配

4. CuTe 中的 Swizzle：
   - 自动推导 swizzle 参数
   - 基于 Layout 代数
   - 更易用，但理解成本高
```

### 3.3 实际 Swizzle 实现

**新增知识点**：

```
项目中的 Swizzle 实现（来自 hgemm_mma_stage_swizzle.cu）：

1. SMEM 地址计算：
   // 原始地址
   int addr = row * stride + col;
   // Swizzle 后地址
   int addr_swizzled = addr ^ ((row & mask) << shift);

2. 为什么 Swizzle 能消除 bank conflict：
   - 相邻行的访问会映射到不同的 bank
   - XOR 操作保证了 bank 映射的均匀分布
   - 对于任何固定的 row 偏移，col 的 bank 映射都是均匀的

3. Block Swizzle vs SMEM Swizzle：
   - Block Swizzle：在 grid 维度做 swizzle，改善 L2 cache locality
   - SMEM Swizzle：在 shared memory 维度做 swizzle，消除 bank conflict
   - 两者可以同时使用
```

---

## 4. Multistage Pipeline 深度解析

### 4.1 cp.async 指令

**新增知识点**：

```
cp.async 指令详解：

1. 基本语法：
   cp.async.cg.shared.global [dst], [src], size
   cp.async.ca.shared.global [dst], [src], size

2. 参数说明：
   - cg (cache global): 只缓存到 L2
   - ca (cache all): 缓存到 L1 + L2
   - size: 4, 8, 16 bytes

3. 异步语义：
   - 发起异步拷贝，不阻塞当前线程
   - 需要 commit_group/wait_group 同步

4. 使用模式：
   // 发起异步拷贝
   CP_ASYNC_CG(dst, src, 16);
   // 提交组
   CP_ASYNC_COMMIT_GROUP();
   // 等待组完成
   CP_ASYNC_WAIT_GROUP(n);

5. 多级流水线：
   // 预加载 K_STAGE-1 个 stage
   for (int k = 0; k < K_STAGE-1; k++) {
     load_to_smem(s[k], ...);
     CP_ASYNC_COMMIT_GROUP();
   }
   CP_ASYNC_WAIT_GROUP(K_STAGE-2);
   __syncthreads();
   
   // 主循环
   for (int k = K_STAGE-1; k < num_tiles; k++) {
     // 计算当前 stage
     compute(s[k % K_STAGE], ...);
     // 预加载下一个 stage
     load_to_smem(s[(k+1) % K_STAGE], ...);
     CP_ASYNC_COMMIT_GROUP();
     CP_ASYNC_WAIT_GROUP(K_STAGE-2);
     __syncthreads();
   }
```

### 4.2 流水线设计模式

**新增知识点**：

```
多级流水线设计模式：

1. 双缓冲 (K_STAGE=2):
   - 优点：实现简单，SMEM 开销小
   - 缺点：隐藏延迟能力有限
   - 适用：计算密度高的场景

2. 三缓冲 (K_STAGE=3):
   - 优点：更好的延迟隐藏
   - 缺点：SMEM 开销增加 50%
   - 适用：中等计算密度

3. 四缓冲 (K_STAGE=4):
   - 优点：最佳延迟隐藏
   - 缺点：SMEM 开销大，可能影响 occupancy
   - 适用：计算密度低的场景

4. 选择策略：
   - 根据 SMEM 大小和 kernel 需求选择
   - 4090 SMEM 100KB，通常用 2-3 stage
   - Hopper SMEM 228KB，可以用 4+ stage
```

---

## 5. FlashAttention 高级优化

### 5.1 SRAM 共享策略

**新增知识点**：

```
FlashAttention SRAM 共享策略详解：

1. Standard FA2 (无共享):
   - Q: O(Br × d)
   - K: O(Bc × d) × stages
   - V: O(Bc × d)
   - 总计: O(4 × Br × d)
   - 问题：SMEM 占用大，限制 occupancy

2. Shared KV SMEM:
   - K, V 共享同一块 SMEM
   - 总计: O(2 × Br × d)
   - 优点：SMEM 减半，occupancy 提高
   - 限制：需要按顺序处理 K 和 V

3. Fully Shared QKV SMEM:
   - Q, K, V 全部共享
   - 总计: O(Br × d)
   - 优点：SMEM 降为 1/4
   - 限制：需要更复杂的调度

4. Fine-grained Tiling:
   - 在 MMA 级别做 Tiling
   - SMEM 复杂度 O(16 × d)
   - 优点：支持更大的 HeadDim（1024+）
   - 限制：实现复杂度高
```

### 5.2 Prefetch 策略

**新增知识点**：

```
FlashAttention Prefetch 策略：

1. Prefetch Q s2r (SMEM → Register):
   - 在循环开始前，将 Q 从 SMEM 加载到寄存器
   - 减少循环内 Q 的 SMEM 访问
   - 条件：(kHeadDim / kMmaAtomK) <= 8

2. Prefetch K/V g2s (Global → SMEM):
   - 在计算当前 tile 时，预取下一个 tile 的 K/V
   - 使用 cp.async 实现异步拷贝
   - 条件：kStage == 2

3. Collective Store:
   - 通过 warp shuffle + 寄存器复用实现 O 的写回
   - 不需要额外的 SMEM
   - 减少 SMEM 使用量

4. 优化效果：
   - Prefetch Q: 减少 50% 的 Q SMEM 访问
   - Prefetch KV: 隐藏 80% 的内存延迟
   - Collective Store: 节省 50% 的 SMEM
```

---

## 6. Triton 编程模型

### 6.1 Triton vs CUDA 对比

**新增知识点**：

```
Triton vs CUDA 对比：

1. 编程模型：
   - CUDA: 线程级编程，显式管理线程
   - Triton: 块级编程，自动管理线程

2. 内存管理：
   - CUDA: 显式管理 SMEM, 寄存器
   - Triton: 自动管理，通过 hint 优化

3. 性能：
   - CUDA: 极致优化，可达 98% cuBLAS
   - Triton: 接近 CUDA，通常 90-95%

4. 开发效率：
   - CUDA: 低，需要理解硬件细节
   - Triton: 高，Python 语法，自动优化

5. 适用场景：
   - CUDA: 极致性能需求，生产环境
   - Triton: 快速原型，研究实验
```

### 6.2 Triton 关键概念

**新增知识点**：

```
Triton 关键概念：

1. Program:
   - 类似 CUDA 的 block
   - 通过 program_id 获取索引
   - 可以是一维或多维

2. Block:
   - Triton 的基本计算单元
   - 自动映射到 GPU 的 block
   - 通过 BLOCK_SIZE 指定大小

3. tl.load / tl.store:
   - 自动处理边界检查
   - 支持 mask 和 other 参数
   - 自动向量化

4. tl.reduce:
   - 块内归约操作
   - 支持 sum, max, min 等
   - 自动使用 warp reduce

5. constexpr:
   - 编译时常量
   - 用于 BLOCK_SIZE, num_stages 等
   - 影响 kernel 的编译和优化
```

---

## 7. 多卡并行优化

### 7.1 数据并行 vs 模型并行

**新增知识点**：

```
多卡并行策略：

1. 数据并行 (Data Parallelism):
   - 每张卡持有完整模型
   - 数据分片到不同卡
   - 适用于：batch size 大，模型小
   - 通信：AllReduce 梯度

2. 模型并行 (Model Parallelism):
   - 模型分片到不同卡
   - 每张卡持有部分模型
   - 适用于：模型大，单卡放不下
   - 通信：AllGather/ReduceScatter

3. 张量并行 (Tensor Parallelism):
   - 矩阵乘法分片
   - 适用于：Linear 层
   - 通信：AllReduce/AllGather

4. 流水线并行 (Pipeline Parallelism):
   - 模型按层分片
   - 适用于：深层模型
   - 通信：点对点

5. 序列并行 (Sequence Parallelism):
   - 序列维度分片
   - 适用于：长序列
   - 通信：AllGather/ReduceScatter
```

### 7.2 4090 多卡优化

**新增知识点**：

```
4090 多卡优化策略：

1. 无 NVLink 的影响：
   - 卡间通信走 PCIe
   - 带宽：~32 GB/s（PCIe 4.0 x16）
   - 延迟：~10 μs
   - 影响：AllReduce 成为瓶颈

2. 优化策略：
   - 减少通信量：梯度累积，压缩通信
   - 计算通信重叠：异步通信，流水线
   - 使用高效的通信原语：NCCL

3. 适用场景：
   - 数据并行：batch size 大时有效
   - 模型并行：通信量大，效果差
   - 推理：单卡足够，无需多卡

4. 性能估算：
   - 8 卡数据并行：线性加速比 7-7.5x
   - 8 卡模型并行：加速比 3-5x（通信瓶颈）
   - 推荐：单卡推理，多卡训练
```

---

## 8. 实际复现中的问题与解决方案

### 8.1 编译问题

**问题1**: `nvcc` 编译 MMA PTX 指令报错

```
错误：error: inline assembly requires target feature +sm_80

解决：
nvcc -std=c++20 -O2 -arch=sm_89 -lcublas -lcuda your_kernel.cu
// 确保 -arch=sm_89 或更高
```

**问题2**: `cp.async` 指令不支持

```
错误：error: 'cp.async.cg.shared.global' requires .target sm_80 or higher

解决：
nvcc -std=c++20 -O2 -arch=sm_80 -lcublas -lcuda your_kernel.cu
// 至少 sm_80
```

### 8.2 性能问题

**问题1**: HGEMM 性能只有 cuBLAS 的 50%

```
原因分析：
1. 没有使用 Tensor Cores
2. Bank conflict 严重
3. 没有使用 double buffering

解决方案：
1. 使用 MMA PTX 指令
2. 添加 SMEM padding 或 swizzle
3. 实现 double buffering
```

**问题2**: FlashAttention 精度误差大

```
原因分析：
1. MMA 使用 FP16 累加
2. Softmax 没有使用 FP32

解决方案：
1. 关键路径使用 FP32 累加
2. Softmax 使用 FP32 计算 max 和 sum
3. 使用 Online Softmax
```

### 8.3 调试技巧

**新增知识点**：

```
CUDA Kernel 调试技巧：

1. 使用 assert 检查边界：
   assert(idx < N);

2. 使用 printf 调试：
   if (tid == 0) printf("value: %f\n", value);

3. 使用 cuda-memcheck 检查内存错误：
   cuda-memcheck ./your_binary

4. 使用 ncu 分析性能：
   ncu --set full ./your_binary

5. 使用 nsys 分析系统行为：
   nsys profile --stats=true ./your_binary

6. 对比 PyTorch 参考实现：
   assert(torch.allclose(output, reference, atol=1e-3))
```

---

## 9. 面试新增深挖点

### 9.1 关于 HGEMM 的深挖

**Q**: 「为什么 HGEMM 用 TN 布局而不是 NN 布局？」

```
深度回答：

1. MMA 指令要求：
   - mma.sync.aligned.m16n8k16.row.col
   - A 需要 row-major，B 需要 col-major
   - TN 布局天然匹配，无需转置

2. ldmatrix 指令效率：
   - TN 布局：ldmatrix.x2 直接加载
   - NN 布局：需要 ldmatrix.x2.trans
   - 转置加载有额外开销

3. 性能差异：
   - TN 布局：无转置开销，性能更好
   - NN 布局：有转置开销，性能略差

4. 实际选择：
   - 如果 B 矩阵是 row-major（常见），用 NN 布局
   - 如果 B 矩阵是 col-major（cuBLAS 默认），用 TN 布局
   - 项目中两种都实现了
```

### 9.2 关于 FlashAttention 的深挖

**Q**: 「FlashAttention 的 Split Q 和 Split KV 有什么区别？」

```
深度回答：

1. Split KV（FlashAttention-1）：
   - 所有 MMA(Warps) 共同处理 Q, K, V
   - 需要通过 shared memory 通信
   - 通信开销大，性能较差

2. Split Q（FlashAttention-2）：
   - 每个 Warp 处理不同的 Q slice
   - 所有 Warp 共享 K, V
   - 减少通信，性能更好

3. 性能差异：
   - Split Q 比 Split KV 快 35%
   - 原因：减少了 Warp 间通信

4. 实现细节：
   - Split Q：warp_QP = warp_id, warp_KV = 0
   - Split KV：warp_QP = warp_id / 2, warp_KV = warp_id % 2
```

### 9.3 关于 SMEM Swizzle 的深挖

**Q**: 「Swizzle 的参数怎么选择？」

```
深度回答：

1. Swizzle 参数含义：
   - B (base): 基础偏移
   - M (mask): 掩码
   - S (shift): 移位

2. 选择原则：
   - 需要与访问模式匹配
   - 保证相邻行映射到不同 bank
   - 保证同一 warp 的线程访问均匀分布

3. 常见配置：
   - Swizzle<3, 3, 3>: 每 8 行一个周期
   - Swizzle<2, 3, 3>: 每 4 行一个周期
   - 需要根据实际访问模式选择

4. 调试方法：
   - 使用 ncu 检查 bank conflict
   - 手动分析 bank 映射
   - 尝试不同配置，选择最优
```

---

## 10. 学习资源推荐

### 10.1 论文

1. **FlashAttention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
2. **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
3. **Online Softmax**: "Online normalizer calculation for softmax" (arXiv:1805.02867)
4. **CUTLASS**: "CUTLASS: CUDA Templates for Linear Algebra Subroutines"
5. **CuTe**: "CuTe: A C++ Template Library for Coordinate-free Tensor Programming"

### 10.2 博客

1. 图解:从Online-Softmax到FlashAttention V1/V2/V3
2. CUDA 入门的正确姿势：how-to-optimize-gemm
3. cutlass cute 101
4. CUDA（三）：通用矩阵乘法：从入门到熟练
5. ops(1)：LayerNorm 算子的 CUDA 实现与优化

### 10.3 代码

1. LeetCUDA: https://github.com/xlite-dev/LeetCUDA
2. flash-attention: https://github.com/Dao-AILab/flash-attention
3. CUTLASS: https://github.com/NVIDIA/cutlass
4. Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

---

## 📝 学习进度追踪

### Week 1: 基础原语 + SGEMM
□ Day 1: 环境搭建 + reduce + elementwise
□ Day 2: Softmax + Norm + RoPE
□ Day 3: SGEMM naive + tiling
□ Day 4: SGEMM vec4 + bcf + dbuf

### Week 2: HGEMM + FlashAttention
□ Day 5: HGEMM naive + WMMA
□ Day 6: HGEMM MMA PTX
□ Day 7: HGEMM stages + swizzle
□ Day 8: FlashAttention split_kv
□ Day 9: FlashAttention split_q + share_kv
□ Day 10: FlashAttention share_qkv + tiling

### Week 3: 扩展 + 总结
□ Day 11: Triton softmax + layer_norm
□ Day 12: merge_attn_states + mat_transpose
□ Day 13: 多卡测试 + benchmark + 总结

---

**祝学习顺利！🚀**
