# Kimi K3 NPU 算子差异分析：训练 vs 推理

> **训练端**: MindSpeed-MM (`/Users/kevin/code/MindSpeed-MM`)
> **推理端**: vLLM-Ascend (`/Users/kevin/code/vllm-ascend`, PR #12951)
> 分析日期: 2026-07-28

---

## 一、总体差异概览

| 维度 | MindSpeed-MM (训练) | vLLM-Ascend (推理) |
|---|---|---|
| 核心目标 | 大规模分布式训练 | 高吞吐低延迟推理 |
| NPU 算子实现方式 | `torch_npu` 通用算子 + 外部 `triton_ascend_kernels` | 自研 AscendC 定制算子 (`torch.ops._C_ascend`) |
| KDA 算子粒度 | 单个融合 `chunk_kda`（训练 prefill 模式） | 拆分为 3 个独立算子（decode/ prefill/ preprocess） |
| SiTU 激活 | 纯 PyTorch，无 NPU 融合 | 融合到量化 pipeline（A3: `dequant_situ_quant`, A5: `situ_mx_quant`） |
| MoE 实现 | `npu_moe_token_permute` + `npu_grouped_matmul` + `npu_swiglu` | `FusedMoE` + W4A8 GMM → 融合 SiTU+量化算子 |
| ShortConv | 标准 Triton JIT（含 backward） | AscendC 定制算子（仅 forward） |
| 状态管理 | 训练时全序列一次性处理，无持久状态池 | 容量型状态池 + in-place 更新 + ssm_state_indices 寻址 |
| 并行策略 | FSDP2 + EP + CP + SP | TP + EP + DP + PCP |

---

## 二、KDA Attention 算子对比

这是差异最大的部分。两边的 KDA 计算数学语义相同，但实现方式和算子拆分完全不同。

### 2.1 算子来源与形态

| | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **实现来源** | 外部包 `triton_ascend_kernels.attention.fla.kda.chunk` | 自研 AscendC C++ 源码 `csrc/attention/` |
| **调用接口** | 标准 Python 函数 `chunk_kda(q, k, v, g, beta, ...)` | 3 个独立 `torch.ops._C_ascend.*` 调用 |
| **可用模式** | `"fused"` (triton_ascend) / `"naive"` (PyTorch 参考) | AscendC 一种实现，通过拆分算子覆盖全部场景 |
| **Backward** | 外部包提供（Triton→Ascend 编译） | 无（推理专用，不需要 backward） |

### 2.2 算子拆分对比

MindSpeed-MM 训练时只用一个融合算子（整序列一次性处理），而 vLLM-Ascend 为了推理效率将其拆为 3 个独立算子：

| 功能 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **短序列 decode** (≤8 tokens) | 不适用（训练无 decode） | `recurrent_kda` — 逐 token 循环，单 kernel 融合 |
| **长序列 prefill** | `chunk_kda` — 单个融合算子处理全部 | `kda_gate_cumsum` + `chunk_kda_fwd` — 门控预处理 + chunk 并行 |
| **门控计算** | `chunk_kda` 内部完成 | `kda_gate_cumsum` — 独立算子，支持 Kimi K3 的 safe_gate 模式 |
| **状态管理** | 无持久状态池 | `[state_capacity, HV, V, K]` 容量型状态池，`ssm_state_indices` 寻址 |
| **布局转换** | `chunk_kda` 内部处理 | `kda_layout_swap12` — 独立 BSND↔BNSD 转换算子 |

### 2.3 关键实现差异

#### A. Kimi K3 Safe Gate 处理

MindSpeed-MM 在调用 `chunk_kda` 前用 PyTorch 预处理 beta sigmoid，但 gate 转换在 kernel 内：

```python
# MindSpeed-MM: modeling_kimi_linear.py line 695, 705
beta = torch.sigmoid(beta)  # 外部预处理，不在 kernel 内
# gate 转换在 triton_ascend_kernels 内部，参数为：
#   use_gate_in_kernel=True, safe_gate=..., lower_bound=...
```

vLLM-Ascend 的 `kda_gate_cumsum` 同样支持 `safe_gate` 模式，但额外在 `recurrent_kda` 的 decode 路径也内置了 Kimi K3 的 bounded sigmoid gate：

```python
# vLLM-Ascend: kimi_kda.py line 282
out = torch.ops._C_ascend.recurrent_kda(
    ..., safe_gate=True, lower_bound=-5.0,
    use_gate_in_kernel=True,
)
```

#### B. use_beta_sigmoid_in_kernel 差异

MindSpeed-MM 文档和代码明确记录了 `triton_ascend_kernels` 的 `chunk_kda` **不支持** `use_beta_sigmoid_in_kernel` 参数（会被 `**kwargs` 吞掉），beta sigmoid 始终在外部完成。

vLLM-Ascend 的 `recurrent_kda` 显式支持 `use_beta_sigmoid_in_kernel` 参数，但 Kimi K3 路径设为 `False`，保持了一致的外部处理方式。

#### C. 推理特有功能（vLLM-Ascend 独占）

| 功能 | 说明 |
|---|---|
| **Decode/Prefill 混合批处理** | 通过 `cu_seqlens` 区分，同一 batch 内 decode 走 `recurrent_kda`、prefill 走 `chunk_kda_fwd` |
| **Speculative Decode** | `ssm_state_indices` 支持 `[seq_num, max_step]` 二维索引，`num_accepted_tokens` 控制有效 token 数 |
| **ACLGraph Replay** | `cu_seqlens` 使用设备端 tensor，图编译时 host 不可读 |
| **Mooncake Connector** | KV-cache 接口集成，支持 PD 分离 |
| **AOT Functionalization** | `initial_state` 使用 mutable input 语义，兼容 `torch.compile` |

#### D. 训练特有功能（MindSpeed-MM 独占）

| 功能 | 说明 |
|---|---|
| **Backward Graph** | `chunk_kda` 提供完整 forward+backward，支持梯度回传 |
| **Selective Recompute** | `skip_kda_recompute=True` 跳过 KDA 重计算，减少显存 |
| **FSDP2 Prefetch** | 层间 forward/backward prefetch chains |
| **Context Parallel** | Ring Attention + Ulysses SP 混合并行 |

---

## 三、SiTU 激活函数对比

这是差异第二大的部分。训练端完全没有 NPU 融合，推理端则将 SiTU 深度融合进量化 pipeline。

| 维度 | MindSpeed-MM (训练) | vLLM-Ascend (推理) |
|---|---|---|
| **实现位置** | `modeling_kimi_linear.py:145` `SituAndMul` 类 | `csrc/moe/dequant_situ_quant/` + `csrc/moe/situ_mx_quant/` |
| **计算设备** | **CPU/GPU (fp32)**，纯 PyTorch | **NPU (A3) / NPU (A5)**，AscendC kernel |
| **是否独立算子** | 是，注册为 `ACT2FN["situ"]` | 否，融合在量化算子内部 |
| **量化** | 不涉及（训练精度） | A3: 动态 INT8; A5: MX FP8 (E4M3FN/E5M2) |
| **反量化** | 不涉及 | A3 shared expert: INT32→BF16 反量化 |
| **Backward** | ✅ autograd 支持 | ❌ 推理专用 |

### 3.1 MindSpeed-MM 实现

```python
# modeling_kimi_linear.py line 145-200
class SituAndMul(nn.Module):
    def __init__(self, beta=1.0, linear_beta=None):
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x):
        # x shape: [..., 2H]
        d = x.shape[-1] // 2
        gate, up = x[..., :d].float(), x[..., d:].float()  # ← 强制 fp32
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(x.dtype)
```

**问题**：每次调用都需要 `float()` 转换 + tanh + sigmoid，计算结果写回 DRAM 再被下游 MoE 读取。训练时精度要求高可以接受，推理时则成为延迟瓶颈。

### 3.2 vLLM-Ascend 实现

```python
# A3 平台：三合一融合 kernel
y_int8, scale_fp32 = torch.ops._C_ascend.dequant_situ_quant(
    x_int32_or_bf16, weight_scale, activation_scale,
    beta=4.0, linear_beta=25.0, activate_left=True, quant_mode="dynamic"
)
# kernel 内部：dequant → SiTU → dynamic INT8，全部在 AI Core 内完成

# A5 平台：SiTU + MXFP8 融合 kernel
y_fp8, mxscale_e8m0 = torch.ops._C_ascend.situ_mx_quant(
    x_bf16, beta=4.0, linear_beta=25.0, dst_type=36  # FLOAT8_E4M3FN
)
```

**优势**：消除了 SiTU 中间结果的 DRAM 写回，以及后续量化步骤的额外 kernel launch。

### 3.3 数学公式一致性

两边的 SiTU 公式等价：

```
gate = beta * tanh(gate / beta) * sigmoid(gate)
up   = (linear_beta * tanh(up / linear_beta)) if linear_beta else up
result = gate * up
```

但 Kimi K3 的实际参数取值不同：
- MindSpeed-MM 训练：`beta=1.0, linear_beta=None`（取决于具体训练配置）
- vLLM-Ascend 推理：`beta=4.0, linear_beta=25.0`（部署时使用 W8A8 量化权重）

---

## 四、MoE 算子对比

### 4.1 核心差异

| 维度 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **实现文件** | `kimi_moe_patch.py` + `ops/npu_patch/` | `kimi_k3.py:KimiK3MoE` + `FusedMoE` |
| **NPU 算子** | `npu_moe_token_permute` / `npu_grouped_matmul` / `npu_swiglu` / `npu_moe_token_unpermute` | 自研 AscendC: `dequant_situ_quant` (A3) / `situ_mx_quant` (A5) |
| **激活函数** | **SwiGLU**（用 `npu_swiglu`，不是 SiTU！） | **SiTU**（融合在量化算子内） |
| **权重布局** | 3D tensor `[E, H, 2*I]` + `[E, I, H]` | W4A8 压缩权重 + GMM path |
| **量化** | 训练时不量化（fp16/bf16） | W4A8 INT4 权重 + INT8/FP8 激活 |
| **Expert 并行** | EP: all-to-all v GMM + all-to-all dispatch | EP: 每 rank 14 experts, AllGather fallback (>512 experts) |

### 4.2 关键发现：MindSpeed-MM 训练时不用 SiTU 做 fused MoE！

查看 `PatchKimiMoeExperts._forward_fused`：

```python
# kimi_moe_patch.py
# gate_up_proj → swiglu(gate_up) → down_proj
intermediate_hidden_states = grouped_matmul(permuted, gate_up_proj, tokens_per_expert)
intermediate_activations = swiglu(intermediate_hidden_states)  # ← 调用 npu_swiglu！
```

训练时 MoE 的 `forward_fused` 路径实际调用的是 `npu_swiglu`（标准 SiLU·gate），而非 SiTU。**这是因为 `torch_npu` 没有 `npu_situ` 算子**，且训练时 SiTU 的数值特性使得其更适合在 fp32 下精确计算。

SiTU 在训练时是在 `KimiBlockSparseMLP`（单个 expert 的 MLP，非 fused 路径）中使用的：

```python
# KimiBlockSparseMLP.forward()  用于普通 MLP 层（非 MoE 层），或者 eager fallback
gate_up = gate_up_proj(hidden_states)
act = ACT2FN["situ"](gate_up)  # ← SituAndMul (pure PyTorch fp32)
hidden_states = down_proj(act)
```

### 4.3 算子链对比

**MindSpeed-MM 训练（fused MoE + EP）**：
```
hidden_states
  → torch_npu.npu_moe_token_permute           ← NPU 通用算子
  → all-to-all (EP dispatch)                  ← 通信
  → torch_npu.npu_grouped_matmul(gate_up)      ← NPU 通用算子
  → torch_npu.npu_swiglu                       ← NPU 通用算子 (SiluAndMul, not SiTU!)
  → torch_npu.npu_grouped_matmul(down)         ← NPU 通用算子
  → all-to-all (EP combine)                   ← 通信
  → torch_npu.npu_moe_token_unpermute          ← NPU 通用算子
  → output
```

**vLLM-Ascend 推理（FusedMoE + SiTU + W4A8）**：
```
hidden_states
  → gate (ReplicatedLinear) → router_logits
  → routed_expert_down_proj (压缩: 7168→3584)
  → FusedMoE dispatch
  → W4 GMM1 (INT4 weights) → BF16 gate_up [M, 6144]
  → torch.ops._C_ascend.dequant_situ_quant   ← 自研 AscendC (A3)
    或 torch.ops._C_ascend.situ_mx_quant      ← 自研 AscendC (A5)
  → W4 GMM2 (INT4 weights) → BF16 down [M, 3584]
  → FusedMoE combine
  → routed_expert_up_proj (解压缩: 3584→7168)
  → + shared_expert (同 SiTU 量化路径)
  → output
```

---

## 五、ShortConv (Causal Conv1D) 对比

| 维度 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **实现方式** | 标准 Triton JIT kernel (`@triton.autotune`) | AscendC 定制算子 |
| **文件位置** | `ops/gdn/triton/convolution.py` | `csrc/` (已有算子，非 Kimi K3 新增) |
| **调用接口** | `causal_conv1d(x, weight, bias, ...)` (autograd.Function) | `torch.ops._C_ascend.npu_causal_conv1d_custom(...)` |
| **Forward** | `causal_conv1d_fwd_impl` (Triton) | AscendC kernel |
| **Backward** | `causal_conv1d_bwd_impl` (Triton) ✅ | ❌ (推理不需要) |
| **A5 (arch35) 支持** | ❌ (显式 raise NotImplementedError) | ✅ (已有算子支持) |

MindSpeed-MM 的 ShortConv 用标准 Triton JIT 实现（`import triton; import triton.language as tl`），不是 `triton_ascend`。这意味着训练时的因果卷积依赖 Triton DSL 在 Ascend 上的兼容层，且 arch35 尚不支持。

---

## 六、MLA / Flash Attention 对比

| 维度 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **实现方式** | `flash_attention_2` → `npu_fusion_attention` | 同左，NPU FA |
| **布局支持** | BNSD / BSND / TND / NTD / 1TND / 1NTD | BSND / TND (KDA 相关) |
| **Context Parallel** | ✅ Ring Attention + Ulysses SP | ❌ (推理不需要) |
| **Output Gate** | ✅ sigmoid(g_proj(hidden_states)) | ✅ 同左 |
| **RoPE** | interleave 模式 (`npu_rotary_mul`) | 参考 vLLM 上游实现 |

MLA 部分两边实现高度一致，都属于标准 MLA + NPU FA。差异主要在训练侧多了 CP 支持。

---

## 七、RMSNorm 对比

| 维度 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **KDA Output Norm** | `KimiK_3_MoeRMSNormGated`: `npu_rms_norm` + `sigmoid(gate)` | vLLM 上游 `RMSNorm`（非 NPU fused） |
| **MLA Sub-Layer Norm** | `KimiRMSNorm`: 纯 PyTorch fp32 | vLLM 上游 `RMSNorm` |

训练端针对 KDA output 专门使用了 `npu_rms_norm` 融合，推理端则使用 vLLM 框架的标准 RMSNorm。

---

## 八、资源管理对比

| 维度 | MindSpeed-MM | vLLM-Ascend |
|---|---|---|
| **KDA 状态** | 无持久状态（全序列一次性处理） | `[state_capacity, HV, V, K]` 容量型状态池，支持变长请求 |
| **状态寻址** | 不适用 | `ssm_state_indices` 设备端索引，支持 packed / speculative 两种模式 |
| **状态更新** | `chunk_kda` 返回 `recurrent_state` 供下一 chunk | in-place: 命中槽位原地更新，未命中保持不变 |
| **KV Cache** | 不适用（训练） | MLA: 标准 KV cache; KDA: 状态池通过 Mooncake connector 集成 |

---

## 九、总结：关键差异一览

| # | 对比项 | MindSpeed-MM (训练) | vLLM-Ascend (推理) |
|---:|---|---|---|
| 1 | **KDA 算子数** | 1 个融合算子 (`chunk_kda`) | 3 个独立算子 (recurrent / chunk / cumsum) |
| 2 | **KDA 实现来源** | 外部 `triton_ascend_kernels` | 自研 AscendC C++ |
| 3 | **KDA Backward** | ✅ 支持 | ❌ 不需要 |
| 4 | **KDA Decode 路径** | `fused_recurrent_kda` (fla) | `recurrent_kda` (自研 AscendC) |
| 5 | **SiTU 算子** | 纯 PyTorch fp32，无 NPU 融合 | A3/A5 双平台 AscendC 融合量化 |
| 6 | **MoE 激活** | `npu_swiglu`（SiLU，非 SiTU） | 真实 SiTU（融合在量化内） |
| 7 | **MoE 算子链** | `permute → GMM → swiglu → GMM → unpermute` | `W4GMM → dequant_situ_quant/situ_mx_quant → W4GMM` |
| 8 | **ShortConv** | Triton JIT (含 backward) | AscendC (仅 forward) |
| 9 | **状态池** | 无 | `[state_capacity, HV, V, K]` + 索引寻址 |
| 10 | **混合批处理** | 不支持 | Decode/Prefill 同 batch |
| 11 | **Speculative Decode** | 不支持 | ✅ 二维索引 + num_accepted_tokens |
| 12 | **EP 策略** | all-to-all GMM dispatch/combine | 固定 EP=64, 每 rank 14 experts |
| 13 | **量化** | 无（fp16/bf16 训练） | W4A8: INT4 weights + INT8/MXFP8 activations |
| 14 | **CP (Context Parallel)** | ✅ Ring Attention | ❌ |
| 15 | **A5 (arch35) 支持** | ShortConv 不支持 | 全算子兼容 |

### 核心结论

1. **KDA 是两边实现差异最大的算子**：训练用单一 `triton_ascend_kernels` 融合算子，推理用 3 个自研 AscendC 算子拆分处理。这种拆分是推理场景的特殊需求驱动的——需要区分 decode（短序列逐 token）和 prefill（长序列 chunk 并行），并支持混合批处理和状态池管理。

2. **SiTU 在训练时没有 NPU 融合**：训练端 MoE fused 路径实际调用 `npu_swiglu`（标准 SiLU），SiTU 只在非 fused MLP 或 eager fallback 中用纯 PyTorch 计算。推理端则通过 `dequant_situ_quant`/`situ_mx_quant` 将 SiTU 深度融合到量化 pipeline，消除中间结果写 DRAM。

3. **算子层次不同**：训练端使用 `torch_npu` 通用 NPU 算子（`npu_moe_token_permute`, `npu_grouped_matmul` 等），推理端大量使用自研 AscendC 定制算子（`torch.ops._C_ascend.*`）。训练侧 `triton_ascend_kernels` 是对 fla 的 Triton→Ascend 移植。

4. **训练需要 backward，推理需要状态管理**：这是最根本的差异来源。训练要求每个算子都有完整前向+反向、支持 autograd 和 FSDP；推理要求高效的状态管理（状态池、in-place 更新）、混合批处理、低延迟 decode。
