# Kimi K3 NPU 定制算子分析

> 分支: `feature/kimi-k3-release-v0.23.0`
> PR: [#12951](https://github.com/vllm-project/vllm-ascend/pull/12951)
> 生成日期: 2026-07-28

---

## 1. 概述

Kimi K3 是 Moonshot AI 推出的新一代大语言模型，架构上采用 **KDA (Key Delta Attention)** 混合 **MLA (Multi-head Latent Attention)** 的 hybrid attention 设计，MoE 层数达到 896 个 routed experts + 2 个 shared experts，并使用 **SiTU** 自定义激活函数。

为在华为 Ascend NPU 上高效运行 Kimi K3，vLLM-Ascend 引入了 **6 个全新定制算子**（4 个 KDA Attention + 2 个量化 SiTU），并扩展了 2 个已有算子。所有定制算子通过 `torch.ops._C_ascend` 注册，底层基于 **Ascend C** 实现，支持 **A2 (Ascend 910B)**、**A3 (Ascend 910_93)** 和 **A5 (Ascend 950)** 三平台。

### 关键模型参数

| 参数 | 数值 |
|---:|---:|
| Hidden Size | 7168 |
| Routed Expert Hidden Size | 3584 |
| Routed Expert Intermediate Size | 3072 |
| Num Experts | 896 |
| Top-K | 16 |
| Shared Experts | 2 |
| Shared Intermediate Size | 6144 (2×3072) |
| EP Size | 64 |
| KDA Head Dim (K/V) | 128 |
| SiTU Beta | 4.0 |
| SiTU Linear Beta | 25.0 |

### 硬件平台支持

Kimi K3 定制算子编译时根据目标 `SOC_VERSION` 选择对应的架构实现（来源：`csrc/build_aclnn.sh`）：

| 平台 | SOC_VERSION | 设备型号 | AscendDeviceType | Kernel 架构标识 |
|---|---|---|---|---|
| **A2** | `ascend910b` | Ascend 910B1/B2/B3 | `A2 = 0` | `arch20` |
| **A3** | `ascend910_93` | Ascend 910_93 | `A3 = 1` | `arch22` |
| **A5** | `ascend950` | Ascend 950 | `A5 = 3` | `arch35` |
| **310P** | `ascend310` | Ascend 310P | `_310P` | `arch32` |

#### 算子-平台支持矩阵

| 算子 | A2 (910B) | A3 (910_93) | A5 (950) | 310P |
|---|---|---|---|---|
| `recurrent_kda` | ✅ | ✅ | ✅ | ❌ |
| `chunk_kda_fwd` | ✅ | ✅ | ✅ | ✅ |
| `kda_gate_cumsum` | ✅ | ✅ | ✅ | ✅ |
| `kda_layout_swap12` | ✅ | ✅ | ✅ | ✅ |
| `dequant_situ_quant` | ✅ | ✅ | ❌ | ❌ |
| `situ_mx_quant` | ❌ | ❌ | ✅ | ❌ |
| `chunk_gated_delta_rule_fwd_h` | ✅ (arch20) | ✅ (arch22) | ✅ (arch35) | ✅ |
| `chunk_fwd_o` | ✅ (arch20) | ✅ (arch22) | ✅ | ✅ |
| `recurrent_gated_delta_rule` | ✅ | ✅ | ✅ | ❌ |
| `fused_gdn_gating` | ✅ | ✅ | ❌ | ❌ |

> 编译来源：`csrc/build_aclnn.sh` — A2 编译段 (line 92) / A3 编译段 (line 147) / A5 编译段 (line 204) / 310P 编译段 (line 78)

#### 平台差异要点

| 差异点 | A2 | A3 | A5 |
|---|---|---|---|
| **SiTU 量化路径** | `dequant_situ_quant` (INT8) | 同 A2 | `situ_mx_quant` (MX FP8) |
| **Kernel 架构版本** | `arch20`（chunk_gated_delta、chunk_fwd_o） | `arch22` | `arch35` |
| **MoE dispatch/combine 融合** | 无专用算子 | `dispatch_ffn_combine_*` 等 A3 专属 | 无 |
| **ModelSlim FP4→INT4 旋转** | 不适用 | `kimi_k3.py:1426` A3 专属 | 不适用 |
| **GMM Swiglu Quant 融合** | `dequant_swiglu_quant` | 同 A2 | `swiglu_group_quant` |
| **数值一致性** | 不同平台 kernel 实现不同，输出**可能不完全一致** | — | — |

> **注意**：A2 和 A3 使用相同的算子接口但底层 kernel 实现不同（arch20 vs arch22），相同输入在不同平台上的数值输出可能存在微小差异（通常 < 1e-3）。A5 差异更大，使用了不同的量化方案（MX FP8 vs INT8）。

#### 310P 不支持 Kimi K3

310P 平台缺少 3 个关键算子：`recurrent_kda`、`dequant_situ_quant`、`situ_mx_quant`。其中 `recurrent_kda` 是 KDA decode 的必需算子，缺少它会导致 decode 阶段无法运行。因此 **Kimi K3 无法在 310P 上运行**。

---

## 2. 算子全景图

```
Kimi K3 Forward
│
├── Input Embedding
│
├── [DecoderLayer × N]
│   │
│   ├── ─── Attention ───
│   │   │
│   │   ├── KDA Attention (部分层)
│   │   │   ├── Conv1D ──► npu_causal_conv1d_custom  [复用]
│   │   │   ├── Decode ──► recurrent_kda             [新 · 算子1]
│   │   │   └── Prefill ─► kda_gate_cumsum           [新 · 算子2]
│   │   │                └► chunk_kda_fwd             [新 · 算子3]
│   │   │
│   │   └── MLA Attention (部分层)
│   │       └── 标准 MLA + Output Gate
│   │
│   └── ─── FFN ───
│       │
│       ├── MoE (Mixture of Experts)
│       │   ├── Router → gating logits
│       │   ├── Routed Experts → FusedMoE(situ)
│       │   │   └── A3: dequant_situ_quant           [新 · 算子4]
│       │   │       A5: situ_mx_quant                 [新 · 算子5]
│       │   └── Shared Experts
│       │       └── A3: dequant_situ_quant           [新 · 算子4]
│       │           A5: situ_mx_quant                 [新 · 算子5]
│       │
│       └── MLP (Dense layers, 使用 SiluAndMul)
│
├── Final RMSNorm
└── LM Head
```

---

## 3. KDA Attention 算子详解

### 3.1 背景：KDA (Key Delta Attention)

KDA 是一种带有状态记忆的线性注意力机制。与标准 Self-Attention 不同，KDA 维护一个可更新的 **循环状态矩阵 S**（shape `[V, K]`），在每个时间步执行：

```
S_t = exp(gate_t) · S_{t-1}
δ_t = beta_t · (v_t - S_t @ k_t)
S_t = S_t + δ_t ⊗ k_t          (outer product update)
o_t = S_t @ (q_t · scale)       (output)
```

- **Prefill** 阶段：对长序列做 chunk 级并行 prefix-scan
- **Decode** 阶段：逐 token 循环更新状态

Kimi K3 的 KDA 与标准 GLA (Gated Linear Attention) 的主要区别在于使用了 **bounded sigmoid gate** 替代 softplus gate：

```
Kimi K3 gate: lower_bound · sigmoid(exp(A_log) · (raw_gate + dt_bias))
Standard KDA: -exp(A_log) · softplus(raw_gate + dt_bias)
```

---

### 3.2 算子 1：`recurrent_kda` — 逐 Token 循环解码

**文件位置：** `csrc/attention/recurrent_kda/`

| 属性 | 值 |
|---|---|
| 阶段 | **Decode**（含 speculative decode） |
| 序列长度 | ≤ 8 tokens / 序列 |
| 数据精度 | q/k/v/out: BF16; gate/beta: FP32/BF16/FP16; A_log/dt_bias: FP32 |
| Bound 类型 | **Compute-bound**（在 AI Core UB 内完成） |

#### 接口签名

```python
out = torch.ops._C_ascend.recurrent_kda(
    query,                    # BF16  [B,T,H,K]  或 [T,H,K]
    key,                      # BF16  同上
    value,                    # BF16  [B,T,HV,V] 或 [T,HV,V]
    gate,                     # FP32  同上 (raw gate)
    beta,                     # FP32  [B,T,HV]   或 [T,HV]
    initial_state,            # FP32  [state_capacity, HV, V, K]  可变，in-place 更新
    cu_seqlens,               # INT32 [seq_num+1] 设备端累积偏移
    ssm_state_indices,        # INT32 [T] 或 [seq_num, max_step] 状态池索引
    A_log,                    # FP32  [HV]
    dt_bias,                  # FP32  [HV*K] 或 [HV,K]
    *,
    num_accepted_tokens=None, # 可选 INT32 [seq_num]
    scale=128**-0.5,          # 默认 0.088388
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,  # Kimi K3 在 kernel 内转换 raw → log gate
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=True,           # Kimi K3 使用 bounded sigmoid
    lower_bound=-5.0,
) -> Tensor out               # BF16，与 value 同 shape
```

#### 核心设计

1. **单 AI Core Kernel 融合**：状态衰减 + delta-rule 更新 + 输出计算在一个 kernel 内完成，避免中间张量写 DRAM。
2. **In-place 状态更新**：`initial_state` 是 mutable input，命中槽位原地更新，未命中槽位保持不变，兼容 AOT functionalization 和 NPU 图编译。
3. **容量型状态池**：`[state_capacity, HV, V, K]` 的状态池通过 `ssm_state_indices` 按需索引，支持变长请求批处理。
4. **内联 Q/K L2Norm**：`use_qk_l2norm_in_kernel=True` 时在 kernel 内对 q/k 做 L2 归一化，省去额外的 launch 开销。
5. **Speculative Decode 支持**：`ssm_state_indices` 可为 `[seq_num, max_step]` 二维索引，`num_accepted_tokens` 指示有效 token 数。

#### 调用流程（`kimi_kda.py:_run_recurrent`）

```
raw_gate, beta = projections(hidden_states)
conv_out = causal_conv1d(mixed_qkv)       ← npu_causal_conv1d_custom
q, k, v = split_projections(conv_out)
out = recurrent_kda(q, k, v, raw_gate, beta, state, cu_seqlens, indices, ...)
```

#### 约束

- `K=V=128`（当前 Kimi K3 torch 入口收窄为此值）
- 每条 recurrent 有效序列长度 ≤ 8
- `HV` 必须能被 `H` 整除
- `cu_seqlens` 首项必须为 0，offset 单调不减

---

### 3.3 算子 2：`kda_gate_cumsum` — 门控累积和

**文件位置：** `csrc/attention/kda_gate_cumsum/`

| 属性 | 值 |
|---|---|
| 阶段 | **Prefill 前置**（chunk_kda_fwd 的预处理） |
| 数据精度 | gate: BF16/FP16/FP32 输入；FP32 输出 |
| Bound 类型 | **Memory-bound**（segment-wise cumsum，主要受显存带宽约束） |

#### 接口签名

```python
gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
    g,                     # BF16/FP16/FP32  [B,T,HV,K] 或 rank-3
    chunk_size,            # int: 32/64/128
    *, A_log=None,         # FP32 [HV]     （use_gate_in_kernel=True 时必选）
    dt_bias=None,          # FP32 [HV*K]   （可选）
    cu_seqlens=None,       # int[]  host 端
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=-5.0,
    layout="BSND",
) -> Tensor gate_cumsum    # FP32 与 g 同 shape
```

#### 两种模式

| 模式 | `use_gate_in_kernel` | 输入含义 | 适用场景 |
|---|---|---|---|
| **Cumsum Only** | `False` | 预计算的 step log gate | 标准 KDA（Kimi K2 系列） |
| **Gate + Cumsum** | `True` | raw gate + `A_log`/`dt_bias` | **Kimi K3** bounded sigmoid |

Kimi K3 使用 **Gate + Cumsum 模式**，在算子内部完成：
```
gate = lower_bound · sigmoid(exp(A_log) · (g + dt_bias))
gate_cumsum = chunk_segment_cumsum(gate, chunk_size)
```

#### 调用流程（`kimi_kda.py:_run_prefill`）

```python
if self.gate_lower_bound is not None:  # Kimi K3 路径
    gate_cumsum = kda_gate_cumsum(
        raw_gate, chunk_size,
        A_log=A_log, dt_bias=dt_bias,
        use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    )
else:  # 标准 KDA 路径
    gate = fused_kda_gate(raw_gate, ...)  # Triton 计算
    gate_cumsum = kda_gate_cumsum(gate, chunk_size)
```

---

### 3.4 算子 3：`chunk_kda_fwd` — 分块并行 Prefill

**文件位置：** `csrc/attention/chunk_kda_fwd/`

| 属性 | 值 |
|---|---|
| 阶段 | **Prefill**（长序列首 token 处理） |
| Chunk 大小 | 32 / 64 / 128（当前使用 64） |
| 返回张量数 | **12** 个 |
| Bound 类型 | **Compute-bound**（chunk 级并行扫描 + 矩阵运算） |

#### 接口签名

```python
(o, final_state, g, aqk, akk, w, u, qg, kg, v_new, h, initial_state_out) = \
    torch.ops._C_ascend.chunk_kda_fwd(
        q,                          # BF16/FP16  [B,T,H,K]
        k,                          # 同 q
        v,                          # BF16/FP16  [B,T,HV,V]
        gk,                         # FP32       [B,T,HV,K]  → kda_gate_cumsum 输出
        beta,                       #            [B,T,HV]
        scale,                      # float      (K ** -0.5)
        chunk_size,                 # int: 64
        layout="BSND",
        *, initial_state=None,      # FP32 [seq_num, HV, K, V]
        output_final_state=True,
        cu_seqlens=None,            # int[] host 端
        chunk_indices=None,         # int[] host 端
        return_intermediate=False,  # Kimi K3 forward 不需要中间结果
        safe_gate=False,
        transpose_state_layout=False,
    )
```

#### 返回值说明

| 索引 | 名称 | Shape | 说明 |
|---:|---|---|---|
| 0 | `o` | `[B,T,HV,V]` | **主输出** |
| 1 | `final_state` | `[seq_num,HV,K,V]` | 最终状态（写入 recurrent state pool） |
| 2 | `g` | 同 gk | gate 的 FP32 副本 |
| 3 | `aqk` | `[B,T,HV,BT]` | Q·K 注意力分数（用于计算输出） |
| 4 | `akk` | `[B,T,HV,BT]` | K·K 注意力分数 |
| 5 | `w` | `[B,T,HV,K]` | 权重中间量 |
| 6 | `u` | `[B,T,HV,V]` | 值中间量 |
| 7 | `qg` | `[B,T,HV,K]` | gate 加权后的 Q |
| 8 | `kg` | `[B,T,HV,K]` | gate 加权后的 K |
| 9 | `v_new` | `[B,T,HV,V]` | 变换后的 V |
| 10 | `h` | `[B,num_chunks,HV,K,V]` | 每个 chunk 的隐藏状态 |
| 11 | `initial_state_out` | 同 initial_state | 初始状态副本 |

#### 内部计算流程

```
Phase 1: Intra-chunk 计算
  └── 每个 chunk 内：
      ├── Q/K 的 gate 加权
      ├── chunk_local_cumsum(gate)
      ├── scaled_dot_kkt (chunk_kda_scaled_dot_kkt_fwd)
      │   ├── intra-sub-intra kernel  → 计算对角线块
      │   └── intra-sub-inter kernel  → 计算非对角线块
      └── solve_tril_kda → 解下三角系统，得到注意力矩阵 A

Phase 2: Inter-chunk 传播
  └── recompute_w_u_fwd:
      ├── w = A @ (kb * beta)  → 权重
      └── u = A @ (vb * beta)  → 加权值

Phase 3: Chunk Gated Delta Rule
  └── chunk_gated_delta_rule_fwd_h  [复用已有算子]
      ├── 读取 initial_state
      ├── 逐 chunk 计算 h_state = gdn_fwd_h(k, w, u, prev_state)
      ├── 输出 v_new
      └── 输出 final_state

Phase 4: 输出组合
  └── chunk_gla_fwd_o_gk:
      └── o = Aqk @ v_new + h @ qg  (chunk GLA 输出)
```

#### 调用流程（`kimi_kda.py:_run_prefill`）

```python
# 1. Q/K L2 Normalize
q = l2norm_fwd(q)
k = l2norm_fwd(k)

# 2. 门控预处理
gate_cumsum = kda_gate_cumsum(raw_gate, chunk_size, ...)

# 3. Chunk KDA 前向
result = chunk_kda_fwd(q, k, v, gate_cumsum, beta, scale,
                        initial_state=initial_state_kv,
                        output_final_state=True, ...)
o, final_state = result[0], result[1]

# 4. 状态回写 (transpose 回 [H,V,K])
recurrent_state[state_indices] = final_state.transpose(-1,-2)
```

#### 状态布局转换

```
recurrent state pool: [state_capacity, HV, V, K]   ← 状态池（decode 使用）
                           ↓ transpose(-1,-2)
initial_state_kv:     [seq_num, HV, K, V]           ← chunk_kda_fwd 输入
                           ↓ chunk_kda_fwd 内部计算
final_state:           [seq_num, HV, K, V]           ← chunk_kda_fwd 输出
                           ↓ transpose(-1,-2)
recurrent_state:      [state_capacity, HV, V, K]    ← 写回状态池
```

---

### 3.5 算子 4：`kda_layout_swap12` — 布局转换

**文件位置：** `csrc/attention/kda_layout_swap12/`

| 属性 | 值 |
|---|---|
| 阶段 | 辅助（已注册，forward 中未直接调用） |
| 数据精度 | FP32/FP16/BF16 |
| Bound 类型 | **Memory-bound** |

```python
out = torch.ops._C_ascend.kda_layout_swap12(
    x,                  # rank-3: [a,b,c] → [b,a,c] | rank-4: [B,a,b,c] → [B,b,a,c]
    *, dependency=None  # 可选，约束输出 shape
) -> Tensor y           # 同 dtype
```

---

## 4. 量化 SiTU 激活算子详解

### 4.1 背景：SiTU 激活函数

Kimi K3 使用自定义的 **SiTU (Sigmoid-Tanh Unit)** 激活函数：

```
gate, up = split(x, 2, axis=-1)
gate = beta · tanh(gate / beta) · sigmoid(gate)
up   = linear_beta · tanh(up / linear_beta)        (linear_beta > 0 时)
situ = gate * up
```

| 参数 | Kimi K3 取值 | 说明 |
|---|---|---|
| `beta` | 4.0 | gate 的 tanh 缩放因子 |
| `linear_beta` | 25.0 | up 的 tanh 缩放因子（接近线性区间） |

---

### 4.2 算子 5：`dequant_situ_quant` — A2/A3 平台（反量化 + SiTU + INT8 量化）

**文件位置：** `csrc/moe/dequant_situ_quant/`

| 属性 | 值 |
|---|---|
| 硬件 | **A2** (Ascend 910B) / **A3** (Ascend 910_93) |
| 精度路径 | INT32 → (dequant) → BF16 → SiTU → INT8 |
| Bound 类型 | **Compute-bound** |

#### 接口签名

```python
y, scale = torch.ops._C_ascend.dequant_situ_quant(
    x,                      # INT32 或 BF16  [M, 2H]，H 为偶数
    *, weight_scale=None,   # FP32 [2H/TP]   (A3 shared expert 时使用)
    activation_scale=None,  # FP32 [M]        (A3 shared expert 时使用)
    bias=None,              # 可选 FP32
    quant_scale=None,       # 动态模式不使用
    quant_offset=None,
    group_index=None,
    beta=4.0,               # SiTU beta
    linear_beta=25.0,       # SiTU linear_beta
    activate_left=True,     # gate 在左半，up 在右半
    quant_mode="dynamic",
) -> (Tensor y, Tensor scale)
# y: INT8 [M, H]
# scale: FP32 [M]  每行一个 scale
```

#### 两种输入模式

**A. Shared Experts — INT32 输入（需要反量化）**

```
QMM → INT32 accumulator [M, 2H]
     → dequant(weight_scale, activation_scale)
     → BF16 gate/up
     → SiTU(gate, up)
     → dynamic INT8 quant [M, H]
```

输入 `x` 为 QMM 的 INT32 累加器，需要 `weight_scale`（FP32 per-channel）和 `activation_scale`（FP32 per-token）做反量化。

| TP | x | y | scale | gate 宽度 | up 宽度 |
|---:|---:|---:|---:|---:|---:|
| 1 | `[M, 12288]` | `[M, 6144]` | `[M]` | 6144 | 6144 |
| 2 | `[M, 6144]` | `[M, 3072]` | `[M]` | 3072 | 3072 |
| 4 | `[M, 3072]` | `[M, 1536]` | `[M]` | 1536 | 1536 |
| 8 | `[M, 1536]` | `[M, 768]` | `[M]` | 768 | 768 |
| 16 | `[M, 768]` | `[M, 384]` | `[M]` | 384 | 384 |

**B. Routed Experts — BF16 输入（跳过反量化）**

```
W4 GMM1 → BF16 [M, 2H]  (已由 GMM 完成 dequant + scale bias)
       → SiTU(gate, up)
       → dynamic INT8 quant [M, H]
```

输入已经是 BF16，不传 `weight_scale`/`activation_scale`。

| 阶段 | x | y | scale |
|---|---|---|---|
| decode | `[M, 6144]` | `[M, 3072]` | `[M]` |
| prefill | `[M, 6144]` | `[M, 3072]` | `[M]` |

其中 `M` 上限为 `tokens * min(top_k(16), local_experts(14)) = tokens * 14`。

#### 动态量化输出公式

```
scale = max(abs(situ), axis=-1) / 127
scale = 1  (整行为零时)
y     = clamp(round(situ / scale), -128, 127).astype(INT8)
```

---

### 4.3 算子 6：`situ_mx_quant` — A5 平台专用（SiTU + MX FP8 量化）

**文件位置：** `csrc/moe/situ_mx_quant/`

| 属性 | 值 |
|---|---|
| 硬件 | **A5** (Ascend 950)，**A2/A3 不可用** |
| 精度路径 | BF16 → SiTU → MX FP8 (E4M3FN/E5M2 + E8M0 block scale) |
| Bound 类型 | **Compute-bound** |

#### 接口签名

```python
y, mxscale = torch.ops._C_ascend.situ_mx_quant(
    x,                   # BF16 [N..., 2H]，末维偶数
    beta=4.0,            # SiTU beta
    linear_beta=25.0,    # SiTU linear_beta
    activate_left=True,  # gate 在左半
    dst_type=36,         # 36=FLOAT8_E4M3FN, 35=FLOAT8_E5M2
) -> (Tensor y, Tensor mxscale)
# y: FP8 E4M3FN [N..., H]
# mxscale: FP8 E8M0 [N..., ceil(H/64), 2]
```

#### 与 `dequant_situ_quant` 的区别

| | `dequant_situ_quant` (A3) | `situ_mx_quant` (A5) |
|---|---|---|
| 输入类型 | INT32 或 BF16 | BF16（已反量化） |
| 反量化 | 支持（shared expert） | 不支持 |
| 输出量化 | INT8 动态量化 | MX FP8 (E4M3FN/E5M2) |
| Scale 粒度 | per-row FP32 | per-64-element E8M0 block scale |
| Scale 格式 | `[M]` FP32 | `[M, ceil(H/64), 2]` FP8 E8M0 |

#### A5 Shared Expert 形状

| TP | x | y | mxscale |
|---:|---|---|---|
| 1 | `[M, 12288]` | `[M, 6144]` | `[M, 96, 2]` |
| 2 | `[M, 6144]` | `[M, 3072]` | `[M, 48, 2]` |
| 4 | `[M, 3072]` | `[M, 1536]` | `[M, 24, 2]` |
| 8 | `[M, 1536]` | `[M, 768]` | `[M, 12, 2]` |
| 16 | `[M, 768]` | `[M, 384]` | `[M, 6, 2]` |

#### A5 Routed Expert 形状（TP 不变）

```
x=[M, 6144]  →  y=[M, 3072]  →  mxscale=[M, 48, 2]
```

---

## 5. 复用的已有算子

### 5.1 `npu_causal_conv1d_custom`

**调用位置：** `kimi_kda.py:231`

KDA Attention 中 q/k/v projection 后的短因果卷积（kernel size=4）。在 `_run_causal_conv1d` 中调用，将 mixed_qkv 通过 concat 后的 1D 卷积权重处理。

```python
torch.ops._C_ascend.npu_causal_conv1d_custom(
    out, mixed_qkv, conv_weights_t,
    conv_state=conv_state,
    cache_indices_opt=metadata.cache_indices,
    ...
)
```

### 5.2 `chunk_gated_delta_rule_fwd_h` — 三平台均有定制版本

**调用位置：** `chunk_kda_fwd` 内部

Chunk Gated Delta Rule 的 H 矩阵更新 kernel。原本在 `csrc/moe/` 下已有实现，Kimi K3 PR 对其做了大量扩展，按架构划分为三个实现：

| 架构 | 平台 | 文件 |
|---|---|---|
| **arch20** | A2 (910B) | `op_kernel/arch20/` |
| **arch22** | A3 (910_93) | `op_kernel/arch22/` — 修改 epilogue（`block_epilogue_gdn_fwdh_update/vnew.hpp`），新增 `tiling_processor.h` |
| **arch35** | A5 (950) | `op_kernel/arch35/` — **全新实现** epilogue（`block_epilogue_gdn_fwdh_update/vnew.hpp`）和 gemm kernel（`gdn_fwd_h_kernel.hpp`，609 行） |

另外新增了通用工具：`kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp`

---

## 6. 算子调用时序

### 6.1 Decode 阶段（单 token / 短序列）

```
hidden_states
  │
  ├── projections ─► q, k, v, raw_gate, beta, output_gate
  │
  ├── _run_causal_conv1d(qkv_mixed)        ← npu_causal_conv1d_custom
  │
  ├── _run_recurrent(q, k, v, raw_gate, beta)
  │   └── recurrent_kda                    ← 新算子 #1
  │       ├── in-kernel: L2Norm(q), L2Norm(k)
  │       ├── in-kernel: gate = safe_sigmoid(raw_gate, A_log, dt_bias)
  │       ├── in-kernel: state decay + delta update
  │       └── in-kernel: output = state @ q * scale
  │   输出: attention_output
  │   state pool 原位更新 (state_indices → recurrent_state pool)
  │
  └── output gate → output projection → residual add
```

### 6.2 Prefill 阶段（长序列首 token）

```
hidden_states
  │
  ├── projections ─► q, k, v, raw_gate, beta, output_gate
  │
  ├── _run_causal_conv1d(qkv_mixed)        ← npu_causal_conv1d_custom
  │
  ├── L2Norm(q), L2Norm(k)  (Python 侧)
  │
  ├── _run_prefill(q, k, v, raw_gate, beta)
  │   │
  │   ├── kda_gate_cumsum(raw_gate, ...)   ← 新算子 #2
  │   │   输出: gate_cumsum [B,T,HV,K] FP32
  │   │
  │   ├── chunk_kda_fwd(q,k,v,gate_cumsum,beta) ← 新算子 #3
  │   │   ├── chunk_local_cumsum(gate)
  │   │   ├── chunk_kda_scaled_dot_kkt_fwd (intra-chunk)
  │   │   ├── solve_tril_kda
  │   │   ├── recompute_w_u_fwd
  │   │   ├── chunk_gated_delta_rule_fwd_h  ← 复用算子
  │   │   └── chunk_gla_fwd_o_gk
  │   │   输出: o [B,T,HV,V], final_state [seq_num,HV,K,V]
  │   │
  │   └── recurrent_state[state_indices] = final_state.transpose(-1,-2)
  │
  └── output gate → output projection → residual add
```

### 6.3 MoE 前向

```
hidden_states [M, 7168]
  │
  ├── Router(ReplicatedLinear) → router_logits [M, 896]
  │
  ├── Routed Expert 路径:
  │   ├── routed_expert_down_proj: [M, 7168] → [M, 3584]  (压缩)
  │   ├── FusedMoE (activation="situ", EP=64, 每 rank 14 experts)
  │   │   ├── dispatch: tokens → expert groups
  │   │   ├── GMM1 (W4): gate+up matmul → BF16 [M, 6144]
  │   │   ├── [A3] dequant_situ_quant  ← 新算子 #4
  │   │   │      或
  │   │   │   [A5] situ_mx_quant       ← 新算子 #5
  │   │   │       输出: INT8/FP8 [M, 3072]
  │   │   ├── GMM2: down matmul → BF16 [M, 3584]
  │   │   └── combine: experts → tokens
  │   ├── routed_expert_norm (RMSNorm, 可选)
  │   └── routed_expert_up_proj: [M, 3584] → [M, 7168]  (解压缩)
  │
  ├── Shared Expert 路径:
  │   ├── gate_up_proj (MergedColumnParallel): [M, 7168] → [M, 12288]
  │   └── [A3] dequant_situ_quant  ← 新算子 #4
  │          或
  │       [A5] situ_mx_quant       ← 新算子 #5
  │          (shared: 反量化 INT32 → SiTU → INT8/MXFP8)
  │       → down_proj: [M, 6144] → [M, 7168] (reduce_results)
  │
  └── output = routed + shared (residual add)
```

---

## 7. Torch Binding 注册汇总

所有算子均注册在 `csrc/torch_binding.cpp`，使用 `torch::kPrivateUse1` (NPU) dispatch key。非 310P 平台（A2/A3/A5）共用一个注册段（L2563-L2644），310P 平台使用独立注册段（L3294-L3326），支持的算子较少。

| 算子 | 注册行（非310P） | 注册行（310P） | 类型 | 支持平台 |
|---|---|---|---|---|
| `chunk_kda_fwd` | L2573 | L3304 | **新** | A2/A3/A5/310P |
| `kda_gate_cumsum` | L2578 | L3309 | **新** | A2/A3/A5/310P |
| `kda_layout_swap12` | L2583 | L3314 | **新** | A2/A3/A5/310P |
| `recurrent_kda` | L2622 | ❌ | **新** | A2/A3/A5 |
| `dequant_situ_quant` | L2636 | ❌ | **新** | A2/A3 |
| `situ_mx_quant` | L2644 | ❌ | **新** | A5 |
| `chunk_gated_delta_rule_fwd_h` | L2563 | L3294 | **扩展** | A2(arch20)/A3(arch22)/A5(arch35)/310P |
| `npu_causal_conv1d_custom` | (已有) | (已有) | **复用** | 全平台 |

另有对应的 Meta 实现（`csrc/torch_binding_meta.cpp`），用于 AOT 图追踪和 shape 推导。

---

## 8. 设计要点总结

1. **KDA 的 Decode/Prefill 分离**：短序列（≤8 tokens）走 `recurrent_kda` 逐 token 计算，长序列走 `chunk_kda_fwd` 分块并行扫描，两者共享同一个 `[state_capacity, HV, V, K]` 状态池。

2. **Kimi K3 的 bounded sigmoid gate**：相比标准 KDA 的 `-exp(A_log) * softplus`，Kimi K3 使用 `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`，数值更稳定，`kda_gate_cumsum` 和 `recurrent_kda` 都需要支持 `safe_gate` 模式。

3. **SiTU 激活的融合量化**：将"激活函数 + 量化"融合为单个 NPU kernel，避免中间 BF16 结果写 DRAM。A2/A3 使用 `dequant_situ_quant`（INT8），A5 使用 `situ_mx_quant`（MX FP8），共享相同的 SiTU 激活计算逻辑。

4. **三平台适配（A2/A3/A5）**：通过 `csrc/build_aclnn.sh` 的 `SOC_VERSION` 匹配选择算子集合和架构版本。
   - A2 (ascend910b): arch20 kernel → `dequant_situ_quant` (INT8)
   - A3 (ascend910_93): arch22 kernel → `dequant_situ_quant` (INT8)
   - A5 (ascend950): arch35 kernel → `situ_mx_quant` (MX FP8)
   - 310P: **不支持 Kimi K3**（缺少 `recurrent_kda`、量化算子）
   - 同一算子在 A2/A3/A5 上的数值输出可能不完全一致，因为底层 kernel 实现不同。

5. **AOT 图编译兼容**：所有算子的 `initial_state` 使用 mutable input 语义（不返回 aliased output），`cu_seqlens` 使用设备端 tensor（兼容 ACLGraph replay 时 host 不可读），Meta 实现确保 shape 推演正确。
