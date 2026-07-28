# Kimi K3 训推精度对齐分析

> 分析日期: 2026-07-28
> 训练端: MindSpeed-MM `/Users/kevin/code/MindSpeed-MM`
> 推理端: vLLM-Ascend PR #12951 `feature/kimi-k3-release-v0.23.0`
>
> **更新 (2026-07-28)**: 已从 ModelScope `moonshotai/Kimi-K3` 的 config.json 直接验证关键参数。

---

## 一、总览：精度风险矩阵

| # | 差异点 | 严重度 | 状态 | 说明 |
|---:|---|---|---|---|
| 1 | ~~KDA Gate 公式~~ | ~~🔴~~ | ✅ **已确认一致** | config 中 `gate_lower_bound: -5.0` |
| 2 | ~~SiTU 参数值~~ | ~~🔴~~ | ✅ **已确认一致** | config 中 `beta=4.0, linear_beta=25.0` |
| 3 | **SiTU 内部计算精度** | 🟡 中等 | 待验证 | fp32 (训练) vs AscendC 融合 kernel (推理) |
| 4 | **KDA Prefill 算子拆分** | 🟡 中等 | 待验证 | 单融合 `chunk_kda` vs 多算子拆分 |
| 5 | **KDA Recurrent Decode 精度** | 🟡 中等 | 待验证 | `fused_recurrent_kda` vs AscendC `recurrent_kda` |
| 6 | **Chunk Gated Delta Rule** | 🟢 低 | 待验证 | Triton vs AscendC 实现差异 |
| 7 | **权重格式差异** | 🟢 低（预期内） | 已知 | 训练 BF16 vs 推理 MXFP4 (A5 原生/A2 反量化) |

---

## 二、✅ 已确认一致：KDA Gate 公式

### 2.1 结论

训练和推理使用**相同的 bounded sigmoid gate 公式**。

### 2.2 证据

从 ModelScope `moonshotai/Kimi-K3/config.json` 中提取的真实配置：

```json
{
  "text_config": {
    "linear_attn_config": {
      "head_dim": 128,
      "num_heads": 96,
      "gate_lower_bound": -5.0,
      "short_conv_kernel_size": 4,
      "use_full_rank_gate": true,
      "kda_layers": [1, 2, 3, 5, 6, 7, ...],
      "full_attn_layers": [4, 8, 12, 16, ...]
    }
  }
}
```

`gate_lower_bound: -5.0` 明确声明，训练和推理均使用：

```
gate_t = -5.0 * sigmoid(exp(A_log) * (raw_gate + dt_bias))
```

### 2.3 训练端代码验证

```python
# MindSpeed-MM modeling_kimi_linear.py line 614
self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
# → 实际训练时从 HF config 加载 → gate_lower_bound = -5.0

# line 705-706
safe_gate=self.gate_lower_bound is not None,   # → True
lower_bound=self.gate_lower_bound,              # → -5.0
```

> **之前分析中"训练 YAML 未设置 gate_lower_bound → softplus gate"的判断是错误的**。`gate_lower_bound` 在 HuggingFace 发布的 config.json 中声明，训练时通过 `trust_remote_code=True` 自动加载，不依赖训练 YAML 中的显式声明。

### 2.4 推理端代码验证

```python
# vLLM-Ascend kimi_kda.py
# _run_recurrent: safe_gate=True, lower_bound=-5.0
# _run_prefill:  kda_gate_cumsum(..., safe_gate=True, lower_bound=-5.0)
```

训推两端公式一致，**无需修改**。

---

## 三、✅ 已确认一致：SiTU 参数值

### 3.1 结论

训练和推理使用相同的 SiTU 激活参数。

### 3.2 证据

从 ModelScope `moonshotai/Kimi-K3/config.json` 中提取：

```json
{
  "text_config": {
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0
  }
}
```

### 3.3 训练端代码验证

```python
# MindSpeed-MM modeling_kimi_linear.py lines 169-172
def _get_situ_activation_params(config: KimiLinearConfig):
    beta = getattr(config, "activation_situ_beta", None)        # → 4.0
    linear_beta = getattr(config, "activation_situ_linear_beta", None)  # → 25.0
    return beta or 1.0, linear_beta
```

`SituAndMul(beta=4.0, linear_beta=25.0)` — 训练时的 `_forward_fused` 和 eager 路径都使用此值。

### 3.4 推理端代码验证

```python
# vLLM-Ascend SiTU 合约
K3_BETA = 4.0
K3_LINEAR_BETA = 25.0
# dequant_situ_quant / situ_mx_quant 均使用此值
```

训推两端参数一致，**无需修改**。

---

## 四、🟡 待验证：SiTU 内部计算精度

### 4.1 问题描述

虽然公式和参数一致，但**内部实现的精度路径不同**。

### 4.2 训练端路径

```python
# SituAndMul.forward() — 全部 fp32
gate = x[..., :d].to(torch.float32)                           # ← fp32 cast
up = x[..., d:].to(torch.float32)                             # ← fp32 cast
situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate) # ← fp32 tanh/sigmoid
if linear_beta is not None:
    up = linear_beta * torch.tanh(up / linear_beta)            # ← fp32 tanh
return (situ_a * up).to(x.dtype)                               # ← cast back to bf16
```

### 4.3 推理端路径（A2/A3: `dequant_situ_quant`）

AscendC kernel 内部融合：**反量化 → SiTU 激活 → 动态 INT8 量化**

- 输入: BF16 或 INT32
- SiTU 中间计算: fp16/fp32 取决于 AI Core 模板参数
- tanh/sigmoid: 硬件查表或多项式近似
- 输出: INT8 + per-row FP32 scale

**潜在精度差异来源**：
1. AI Core 内部的 `tanh`/`sigmoid` 实现 vs PyTorch `torch.tanh`/`torch.sigmoid`
2. fp16 vs fp32 中间累加
3. 最终的 INT8 量化舍入误差

### 4.4 推理端路径（A5: `situ_mx_quant`）

- SiTU 计算 → MX FP8 (E4M3FN) 量化
- E8M0 block scale per 64 elements

### 4.5 验证方案

```python
import torch
import torch_npu

def compare_situ_fp32_vs_ascendc():
    """对比训练侧 fp32 SiTU 和推理侧融合 kernel 的 SiTU 部分"""
    x = torch.randn(4, 6144, dtype=torch.bfloat16)
    beta, linear_beta = 4.0, 25.0

    # 训练侧: SituAndMul(fp32)
    gate = x[..., :3072].float()
    up = x[..., 3072:].float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_transformed = linear_beta * torch.tanh(up / linear_beta)
    ref = (situ_a * up_transformed).to(torch.bfloat16)

    # 推理侧: dequant_situ_quant (BF16 mode, 不反量化)
    y_int8, scale = torch.ops._C_ascend.dequant_situ_quant(
        x.npu(), beta=beta, linear_beta=linear_beta,
        weight_scale=None, activation_scale=None,
        bias=None, quant_scale=None, quant_offset=None,
        group_index=None, activate_left=True, quant_mode="dynamic"
    )
    # 反量化 INT8 → BF16 后对比
    y_bf16 = (y_int8.float() * scale.unsqueeze(-1)).to(torch.bfloat16)

    diff = (ref.npu() - y_bf16).abs()
    print(f"SiTU max diff (fp32 vs AscendC+INT8): {diff.max():.6f}")
    print(f"SiTU mean diff: {diff.mean():.6f}")

    # 期望: max diff < 0.5 (INT8 量化误差), mean diff < 0.01
```

---

## 五、🟡 待验证：KDA Prefill 算子拆分

### 5.1 差异描述

| | 训练端 | 推理端 |
|---|---|---|
| 实现 | `triton_ascend_kernels` 的 `chunk_kda` (单融合算子) | `kda_gate_cumsum` + `chunk_kda_fwd` (两步) |
| Gate 预处理 | kernel 内部 | 独立 `kda_gate_cumsum` launch |
| 中间精度 | 内部门控累积 (Triton → Ascend) | 门控 fp32 输出 → chunk_kda_fwd 接收 fp32 输入 |

### 5.2 关键对齐点

- `kda_gate_cumsum` 输出 fp32，与训练 kernel 内部精度一致 ✅
- 相同的 chunk_size=64 ✅
- 相同的 safe_gate/lower_bound ✅
- 相同的 Q/K L2Norm (fp32 normalize, bf16 output) ✅

### 5.3 待验证

预计算 gate_cumsum vs kernel 内部实时计算的**逐元素精度差异**。

---

## 六、🟡 待验证：KDA Recurrent Decode

### 6.1 差异描述

| | 训练端 (非训练路径) | 推理端 |
|---|---|---|
| 实现 | `fla.ops.kda.fused_recurrent_kda` 或 `triton_ascend_kernels` | `torch.ops._C_ascend.recurrent_kda` (AscendC) |
| 状态布局 | `[H, K, V]` (transpose_state_layout=True) | `[state_capacity, HV, V, K]` (state_v_first=True) |

### 6.2 关键对齐点

- 相同的 bounded sigmoid gate 公式 ✅
- 相同的 Q/K L2Norm ✅
- 状态转换: `transpose(-1,-2)` 对应 `transpose_state_layout` ✅

### 6.3 待验证

1. 状态矩阵乘加 (`S@k`, `outer(delta,k)`, `S@q`) 的内部累加精度
2. 多步 decode 后状态漂移

---

## 七、🟢 待验证：Chunk Gated Delta Rule

`chunk_gated_delta_rule_fwd_h` 算子在两端使用不同的 kernel 实现 (Triton vs AscendC)。由于使用 fp32 state 累加，差异预计 < 1e-3。

---

## 八、🟢 已知差异：权重格式

| 权重组件 | 训练格式 | 推理格式 (ModelScope 发布版) | 推理格式 (A2/A3 运行时) |
|---|---|---|---|
| Expert 权重 (93%) | MXFP4 (训练原生) | MXFP4-packed (磁盘) | MXFP4 → BF16 反量化 (HBM) |
| 非 Expert (7%) | BF16 (训练原生) | BF16 (磁盘) | BF16 (直接使用) |
| SiTU 激活输出 | BF16 | — | A2/A3: INT8; A5: MXFP8 |

训练用 MXFP4/MXFP8 混合精度原生训练，发布的 checkpoint 就是训练精度，不存在"全精度 → 量化"的后训练步骤。推理时 A2/A3 需从 MXFP4 反量化到 BF16 再计算；A5 可原生加速 MXFP4 计算。

---

## 九、已验证对齐的算子

| 算子 | 状态 | 说明 |
|---|---|---|
| **Q/K L2Norm** | ✅ | 两端 `use_qk_l2norm_in_kernel=True`，fp32 归一化 + bf16 cast |
| **Beta Sigmoid** | ✅ | 两端都在 kernel 外部 `sigmoid(beta)`，`use_beta_sigmoid_in_kernel=False` |
| **KDA Gate 公式** | ✅ | 两端 `gate = -5.0 * sigmoid(exp(A_log) * (g + dt_bias))` |
| **SiTU 参数** | ✅ | 两端 `beta=4.0, linear_beta=25.0, activate_left=True` |
| **ShortConv** | ✅ | 两端 depthwise causal conv1d, kernel_size=4，无 activation |
| **MLA Flash Attention** | ✅ | 两端 `npu_fusion_attention`，layout 一致 |
| **KDA Output Gate** | ✅ | `RMSNorm(hidden) * sigmoid(gate)` |
| **MoE Router** | ✅ | 标准 top-k (16) + sigmoid routing |

---

## 十、修复优先级

### 已消除（无需修复）

1. ✅ **KDA Gate 公式** — config.json 确认 `gate_lower_bound: -5.0`
2. ✅ **SiTU 参数** — config.json 确认 `beta=4.0, linear_beta=25.0`

### 第一优先：单算子精度验证

3. **SiTU kernel 精度** — 对比 fp32 SituAndMul vs AscendC `dequant_situ_quant` 的 SiTU 部分
4. **KDA Gate kernel 精度** — 对比 PyTorch gate 和 AscendC `kda_gate_cumsum` (safe_gate 模式)

### 第二优先：端到端验证

5. **KDA Prefill** — 对比 `chunk_kda` vs `kda_gate_cumsum + chunk_kda_fwd`
6. **KDA Decode** — 对比 `fused_recurrent_kda` vs `recurrent_kda`，含多步状态累积

---

## 十一、精度对比测试框架

```python
import torch
import torch.nn.functional as F

def compare_kda_gate_safe():
    """单算子精度对比: bounded sigmoid gate (safe_gate=True)"""
    H, K = 96, 128  # Kimi K3 实际值
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H * K, dtype=torch.float32)
    raw_gate = torch.randn(1, 64, H, K, dtype=torch.bfloat16)
    lower_bound = -5.0

    # 训练端: PyTorch bounded sigmoid
    A = torch.exp(A_log).view(1, 1, H, 1)
    bias = dt_bias.view(1, 1, H, K)
    gate_ref = lower_bound * torch.sigmoid(A * (raw_gate.float() + bias))

    # 推理端: AscendC kda_gate_cumsum
    gate_ascendc = torch.ops._C_ascend.kda_gate_cumsum(
        raw_gate.npu().contiguous(), 64,
        A_log=A_log.npu(), dt_bias=dt_bias.npu(),
        use_gate_in_kernel=True, safe_gate=True,
        lower_bound=lower_bound, layout="BSND"
    )

    diff = (gate_ref.npu() - gate_ascendc).abs()
    print(f"Gate max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
    # 期望: max < 1e-4, mean < 1e-6

def compare_situ_fp32_vs_ascendc():
    """单算子精度对比: SiTU fp32 vs AscendC 融合 kernel"""
    x = torch.randn(4, 6144, dtype=torch.bfloat16)
    beta, linear_beta = 4.0, 25.0
    d = x.shape[-1] // 2

    # 训练端: fp32 SituAndMul
    gate = x[..., :d].float()
    up = x[..., d:].float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_t = linear_beta * torch.tanh(up / linear_beta)
    ref = (situ_a * up_t).to(torch.bfloat16)

    # 推理端: dequant_situ_quant (BF16, no dequant)
    y_int8, scale = torch.ops._C_ascend.dequant_situ_quant(
        x.npu(), beta=beta, linear_beta=linear_beta,
        weight_scale=None, activation_scale=None,
        bias=None, quant_scale=None, quant_offset=None,
        group_index=None, activate_left=True, quant_mode="dynamic"
    )
    y_bf16 = (y_int8.float() * scale.unsqueeze(-1).float()).to(torch.bfloat16)

    diff = (ref.npu() - y_bf16).abs()
    print(f"SiTU max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
    # 期望: max < 0.5 (INT8 量化), mean < 0.01
```
