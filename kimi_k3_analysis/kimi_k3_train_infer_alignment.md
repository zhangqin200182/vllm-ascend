# Kimi K3 训推精度对齐分析

> 分析日期: 2026-07-28
> 训练端: MindSpeed-MM `/Users/kevin/code/MindSpeed-MM`
> 推理端: vLLM-Ascend PR #12951 `feature/kimi-k3-release-v0.23.0`

---

## 一、总览：精度风险矩阵

经过逐算子的代码级对比，识别出以下精度敏感点：

| # | 差异点 | 严重度 | 影响范围 | 是否会导致 token 级差异 |
|---:|---|---|---|---|
| 1 | **KDA Gate 公式不一致** | 🔴 严重 | 全部 KDA 层 | ✅ 是 |
| 2 | **SiTU 参数值不确定** | 🔴 严重 | 全部 MoE/MLP 层 | ✅ 是 |
| 3 | **SiTU 内部计算精度** | 🟡 中等 | 全部 MoE/MLP 层 | ✅ 可能 |
| 4 | **KDA Prefill 算子拆分** | 🟡 中等 | Prefill 阶段 | ✅ 可能 |
| 5 | **KDA Recurrent Decode 精度** | 🟡 中等 | Decode 阶段 | ✅ 可能 |
| 6 | **Chunk Gated Delta Rule** | 🟢 低 | Prefill 阶段 | ⚠️ 累积误差 |
| 7 | **权重量化误差** | 🟢 低（预期内） | 全模型 | ⚠️ 已知有损 |

---

## 二、🔴 严重问题 1：KDA Gate 公式不一致

### 2.1 问题描述

**训练端**和**推理端**使用了不同的 KDA gate 计算公式，这是训推对齐最严重的问题。

### 2.2 训练端代码路径

MindSpeed-MM 训练配置 (`examples/kimi_k3/kimik3_config.yaml`) 中**没有**设置 `linear_attn_config.gate_lower_bound`，导致：

```python
# modeling_kimi_linear.py line 614
self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound", None)
# → gate_lower_bound = None

# line 705-706
safe_gate=self.gate_lower_bound is not None,   # → False
lower_bound=self.gate_lower_bound,              # → None
```

训练时使用的 gate 公式为**标准 KDA softplus gate**：

```
gate_t = -exp(A_log) * softplus(raw_gate + dt_bias)
```

这在 `chunk_kda_naive.py` 的 `kda_gate` 函数中得到确认（line 99-102）：

```python
if lower_bound is not None:
    g = lower_bound * torch.sigmoid(A * g)       # safe_gate 路径
else:
    g = -A * F.softplus(g)                       # ← 训练实际使用此路径
```

### 2.3 推理端代码路径

vLLM-Ascend 的 `kimi_kda.py` 始终使用 Kimi K3 的 bounded sigmoid gate：

```python
# kimi_kda.py line 269-301 (_run_recurrent)
out = torch.ops._C_ascend.recurrent_kda(
    ...,
    safe_gate=self.gate_lower_bound is not None,  # gate_lower_bound 来自模型 config
    lower_bound=self.gate_lower_bound if self.gate_lower_bound is not None else -5.0,
)

# kimi_kda.py line 352-362 (_run_prefill)
if self.gate_lower_bound is not None:  # ← 推理时此条件为 True（config 中有此参数）
    gate_cumsum = torch.ops._C_ascend.kda_gate_cumsum(
        ..., use_gate_in_kernel=True, safe_gate=True, lower_bound=self.gate_lower_bound,
    )
```

推理时的 gate 公式：

```
gate_t = lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
       = -5.0 * sigmoid(exp(A_log) * (raw_gate + dt_bias))
```

### 2.4 影响分析

两个公式在数值上**不等价**。举例说明：

| raw_gate + dt_bias | `-exp(A)·softplus(x)` (A=2.0) | `-5.0·sigmoid(exp(A)·x)` (A=2.0) |
|---|---:|---:|---:|
| -2.0 | -0.94 | -0.09 |
| 0.0 | -5.12 | -2.50 |
| 2.0 | -15.08 | -4.91 |
| 4.0 | -30.43 | -5.00 |

- **Softplus gate**: 对大的正输入无上界，遗忘速度可以极快
- **Bounded sigmoid gate**: 有硬上限 `lower_bound`（-5.0），遗忘速度有上界

这意味着：
- 如果训练时用 softplus gate 训练的 `A_log` 权重，在推理时换用 bounded sigmoid gate，KDA 的状态衰减行为完全不同
- 会导致 **token 级输出差异**，且随序列增长而累积放大

### 2.5 根因

这个差异的根因有两种可能：

**可能 A**: 训练配置确实用了 softplus gate，但推理端的 `KimiK3TextConfig` 中自带 `gate_lower_bound=-5.0`，导致推理自动切换到 bounded sigmoid。**训练时应该也设置 `gate_lower_bound=-5.0`**。

**可能 B**: 实际训练时从 HuggingFace 加载的 pretrained config 中已经包含 `gate_lower_bound=-5.0`，但训练 YAML 示例文件不完整。需要检查实际训练使用的完整 config。

### 2.6 修复方案

```python
# vllm_ascend/ops/kimi_kda.py

# 方案 1: 从模型 config 读取 gate_lower_bound，如果未设置则不使用 safe_gate
class AscendKimiGatedDeltaNetAttention:
    def __init__(self, ...):
        ...
        # 从 vllm_config 或 model config 中读取
        gate_lower_bound = getattr(config, 'gate_lower_bound', None)
        self.gate_lower_bound = gate_lower_bound  # None → safe_gate=False

    def _run_recurrent(self, ...):
        use_safe_gate = self.gate_lower_bound is not None
        out = torch.ops._C_ascend.recurrent_kda(
            ...,
            safe_gate=use_safe_gate,
            lower_bound=self.gate_lower_bound if use_safe_gate else -5.0,
        )

# 方案 2: 与训练对齐——确保训练和推理都使用相同的 gate_lower_bound
# 在模型 config.json 中明确声明:
#   "linear_attn_config": {"gate_lower_bound": -5.0}
```

**验证方法**：取同一个 `A_log`、`dt_bias`、`raw_gate` 输入，分别在 PyTorch 和 AscendC 算子上计算 gate，对比数值误差是否在 `1e-5` 量级。

---

## 三、🔴 严重问题 2：SiTU 参数值不确定

### 3.1 问题描述

训练和推理使用的 SiTU 参数（`beta`、`linear_beta`）可能不同。

### 3.2 训练端取值

```python
# modeling_kimi_linear.py lines 169-172
def _get_situ_activation_params(config: KimiLinearConfig):
    beta = getattr(config, "activation_situ_beta", None)
    linear_beta = getattr(config, "activation_situ_linear_beta", None)
    return beta or 1.0, linear_beta
# 默认: beta=1.0, linear_beta=None
```

### 3.3 推理端取值

```python
# vllm-ascend 的 KimiK3TextConfig / HF config.json 中声明
"activation_situ_beta": 4.0,
"activation_situ_linear_beta": 25.0,
```

### 3.4 影响分析

SiTU 公式：`result = beta · tanh(gate/beta) · sigmoid(gate) · up`

| gate 值 | beta=1.0 | beta=4.0 |
|---:|---:|---:|
| 1.0 | 1.0·tanh(1.0)·sigmoid(1.0) ≈ 0.56 | 4.0·tanh(0.25)·sigmoid(1.0) ≈ 0.72 |
| 2.0 | 1.0·tanh(2.0)·sigmoid(2.0) ≈ 0.85 | 4.0·tanh(0.5)·sigmoid(2.0) ≈ 1.64 |
| 3.0 | 1.0·tanh(3.0)·sigmoid(3.0) ≈ 0.95 | 4.0·tanh(0.75)·sigmoid(3.0) ≈ 2.35 |

`beta` 越大，gate 越接近线性（tanh 被拉伸的区间更宽）。`beta=1.0 vs 4.0` 差异显著，会导致 MoE/MLP 输出的 token 级差异。

### 3.5 修复方案

**必须确认实际训练时使用的 HF config 中的真实值**。如果训练用 `beta=1.0, linear_beta=None`，推理端必须匹配：

```python
# vllm_ascend/ops/activation.py
# 确保从模型 config 读取，而不是硬编码
class SituActivationConfig:
    def __init__(self, config):
        self.beta = getattr(config, 'activation_situ_beta', 1.0)  # 默认对齐训练
        self.linear_beta = getattr(config, 'activation_situ_linear_beta', None)
```

### 3.6 验证方法

```python
# 单算子精度测试
x = torch.randn(4, 6144, device="cpu")

# 训练侧: SituAndMul (fp32)
gate, up = x[..., :3072].float(), x[..., 3072:].float()
ref = (1.0 * torch.tanh(gate) * torch.sigmoid(gate) * up).to(torch.bfloat16)

# 推理侧: dequant_situ_quant (A3)
y_npu, _ = torch.ops._C_ascend.dequant_situ_quant(
    x.to("npu").to(torch.bfloat16), beta=1.0, activate_left=True
)
# 对比 ref vs y_npu
```

---

## 四、🟡 中等问题 3：SiTU 内部计算精度

### 4.1 问题描述

训练端 SiTU 全部在 fp32 下计算，推理端融合在 AscendC kernel 内，内部精度可能为 fp16/bf16。

### 4.2 训练端精度

```python
# SituAndMul.forward()
gate = x[..., :d].to(torch.float32)      # fp32
up = x[..., d:].to(torch.float32)         # fp32
situ_a = self.beta * torch.tanh(...) * torch.sigmoid(...)  # fp32
return (situ_a * up).to(x.dtype)          # 输出转回 bf16
```

### 4.3 推理端精度

`dequant_situ_quant` 的 AscendC kernel 内部：
- 输入 BF16 → SiTU 计算 → 动态 INT8 量化
- 中间计算精度取决于 kernel 实现（通常 AI Core 内部用 fp16/fp32 取决于模板参数）
- tanh 和 sigmoid 在 AI Core 上使用硬件查表或多项式近似

### 4.4 修复方案

1. 确认 `dequant_situ_quant` kernel 内部 SiTU 计算的中间精度
2. 如果 kernel 内部用 fp16 计算 tanh/sigmoid，评估是否需要添加 fp32 fallback 路径用于精度对齐
3. 可选方案：提供一个不量化的 SiTU 算子（纯 BF16→BF16），用于精度验证

```python
# 建议新增: 纯 SiTU 激活算子（不量化），用于精度对齐
y = torch.ops._C_ascend.situ_activation(
    x,  # BF16 [..., 2H]
    beta=1.0, linear_beta=None, activate_left=True
) -> BF16 [..., H]
```

---

## 五、🟡 中等问题 4：KDA Prefill — 单算子 vs 多算子拆分

### 5.1 问题描述

训练端调用单一 `chunk_kda` 融合算子（`triton_ascend_kernels`），推理端拆分为 `kda_gate_cumsum` + `chunk_kda_fwd` 两步。

### 5.2 差异点

```
训练: chunk_kda(q, k, v, raw_gate, beta, ...)
      └── kernel 内部: l2norm → gate transform → cumsum → chunk scan → output

推理: gate_cumsum = kda_gate_cumsum(raw_gate, chunk_size, ...)    ← 独立 launch
      o, state = chunk_kda_fwd(q, k, v, gate_cumsum, beta, ...)   ← 独立 launch
      └── kernel 内部: l2norm → chunk scan → output (gate 已在外部处理)
```

### 5.3 精度影响

- `kda_gate_cumsum` 的输出 gate_cumsum 是 **fp32** 精度，与训练 kernel 内部一致
- `chunk_kda_fwd` 接收 fp32 gate_cumsum，与训练一致
- 主要风险在于两个算子之间的数据传递有无精度损失

### 5.4 修复方案

1. 确保 `kda_gate_cumsum` 的输出精度和数值与训练 kernel 内部一致
2. 建议做 intermediate value 对比：在相同输入下，提取训练 `chunk_kda` 内部的 gate_cumsum 中间值和推理 `kda_gate_cumsum` 的输出值，逐元素对比

```python
# 精度对比测试
# 1. 用相同输入分别跑训练端 chunk_kda（dump 中间 gate_cumsum）和推理端 kda_gate_cumsum
# 2. 逐元素对比 gate_cumsum 的差异
# 3. 如果 gate_cumsum 对齐，则问题出在 chunk_kda_fwd 内部
```

---

## 六、🟡 中等问题 5：KDA Decode — `fused_recurrent_kda` vs `recurrent_kda`

### 6.1 问题描述

训练端 decode 路径使用 `fused_recurrent_kda`（来自 `fla` 包或 `triton_ascend_kernels`），推理端使用自研 `torch.ops._C_ascend.recurrent_kda`。

### 6.2 差异点

两者的数学公式相同：

```
S_t = exp(gate_t) * S_{t-1}
delta_t = beta_t * (v_t - S_t @ k_t)
S_t = S_t + outer(delta_t, k_t)
o_t = S_t @ q_t * scale
```

但内部实现可能不同：
- **状态布局**: 训练端状态可能使用 `[K, V]` 布局，推理端使用 `[V, K]` 布局
- **累加精度**: 训练端 fused_recurrent_kda 可能使用 fp32 累加，推理端可能使用 bf16/fp16
- **Gate 计算**: 同严重问题 1

### 6.3 修复方案

1. 统一 gate 公式（对齐严重问题 1 的修复）
2. 验证状态布局转换是否正确（训练端 `transpose_state_layout=True` 对应推理端 `transpose(-1, -2)`）
3. 单 token decode 精度对比测试

---

## 七、🟢 低风险问题 6：Chunk Gated Delta Rule 精度

### 7.1 问题描述

`chunk_kda_fwd` 内部调用的 `chunk_gated_delta_rule_fwd_h` 算子，训练端和推理端实现不同：
- 训练端：`triton_ascend_kernels` 内部 Triton kernel
- 推理端：AscendC kernel（`csrc/moe/chunk_gated_delta_rule_fwd_h/`）

### 7.2 风险分析

此算子是 chunk 级的状态传播，使用 fp32 state 累加。由于两端的实现方式不同（Triton vs AscendC），可能存在微小的浮点差异。但由于 state 使用 fp32，且主要是矩阵乘加操作，误差应在 `1e-3` 量级。

### 7.3 修复方案

建议做单 chunk 级别的精度对比测试，确认误差在可接受范围内。

---

## 八、🟢 低风险问题 7：权重量化误差

### 8.1 问题描述

训练使用 fp16/bf16 权重，推理使用 W4A8 量化权重。这是预期的有损压缩。

### 8.2 风险分析

- INT4 权重量化误差由量化方案（GPTQ/AWQ 等）决定，不在算子层面
- 激活量化（INT8/MXFP8）由 `dequant_situ_quant`/`situ_mx_quant` 完成，误差取决于量化粒度和动态范围

### 8.3 修复方案

这是已知的精度-性能 tradeoff，不属于训推对齐的修复范围。但建议记录量化前后的精度差异（perplexity 变化等）。

---

## 九、已验证对齐的算子

以下算子经过代码验证，**已对齐**，无需修改：

| 算子 | 对齐状态 | 说明 |
|---|---|---|
| **Q/K L2Norm** | ✅ 对齐 | 两端都使用 `use_qk_l2norm_in_kernel=True`，fp32 归一化 + cast |
| **Beta Sigmoid** | ✅ 对齐 | 两端都在 kernel 外部计算 `sigmoid(beta)`，传入 `use_beta_sigmoid_in_kernel=False` |
| **ShortConv 数学语义** | ✅ 对齐 | 两端都是 depthwise causal conv1d，kernel_size=4，无 activation |
| **MLA Flash Attention** | ✅ 对齐 | 两端都走 `npu_fusion_attention`，layout 一致 |
| **KDA Output Gate (RMSNormGated)** | ✅ 对齐 | 都是 `RMSNorm(hidden) * sigmoid(gate)`，语义一致 |
| **MoE Router** | ✅ 对齐 | 标准 top-k routing，一致 |

---

## 十、修复优先级与建议顺序

### 第一优先（必须立即验证）

1. **确认训练时实际使用的 `gate_lower_bound` 值**
   - 检查实际训练使用的 HuggingFace pretrained config.json
   - 如果训练用 softplus gate（`gate_lower_bound=None`），推理端需要改为相同的 softplus gate
   - 如果训练也用 bounded sigmoid gate（`gate_lower_bound=-5.0`），则更新训练 YAML 文档

2. **确认训练时实际使用的 `activation_situ_beta` 和 `activation_situ_linear_beta` 值**
   - 从实际训练使用的 HF config 中读取
   - 确保推理端 `SituActivationConfig` 从 config 读取而不是硬编码

### 第二优先（单算子精度测试）

3. **KDA Gate 单算子精度对比**
   - 取相同输入，分别在训练 PyTorch softplus 和推理 AscendC gate 上计算
   - 确认差异在 `1e-5` 量级（如果公式一致）

4. **SiTU 单算子精度对比**
   - 取相同输入，分别在训练 `SituAndMul`(fp32→bf16) 和推理 `dequant_situ_quant` 上计算
   - 确认 SiTU 激活部分的差异在 `1e-3` 量级

### 第三优先（端到端验证）

5. **KDA Prefill 端到端精度对比**
   - 相同输入序列，对比训练 `chunk_kda` 和推理 `kda_gate_cumsum + chunk_kda_fwd` 的输出
   - 逐个中间 tensor 对比（gate_cumsum, aqk, akk, w, u, v_new）

6. **KDA Decode 端到端精度对比**
   - 相同状态初值和输入 token，对比 `fused_recurrent_kda` 和 `recurrent_kda` 的输出和状态更新
   - 多步累加后对比状态差异

---

## 十一、精度对比测试框架建议

```python
def compare_kda_gate():
    """单算子精度对比: KDA Gate"""
    # 相同输入
    H, K = 16, 128
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H * K, dtype=torch.float32)
    raw_gate = torch.randn(1, 64, H, K, dtype=torch.bfloat16)

    # 训练端计算 (softplus gate)
    A = torch.exp(A_log).view(1, 1, H, 1)
    gate_ref = -A * F.softplus(raw_gate.float() + dt_bias.view(1, 1, H, K))

    # 推理端计算 (如果使用 bounded sigmoid)
    lower_bound = -5.0
    gate_infer = lower_bound * torch.sigmoid(
        torch.exp(A_log).view(1, 1, H, 1) * (raw_gate.float() + dt_bias.view(1, 1, H, K))
    )

    diff = (gate_ref - gate_infer).abs()
    print(f"Gate max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

def compare_situ():
    """单算子精度对比: SiTU 激活"""
    x = torch.randn(4, 6144, dtype=torch.bfloat16)
    beta, linear_beta = 1.0, None

    # 训练端 SituAndMul
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    ref = (situ_a * up).to(torch.bfloat16)

    # 推理端 dequant_situ_quant (BF16 mode, no dequant)
    y_npu, _ = torch.ops._C_ascend.dequant_situ_quant(
        x.npu(), beta=beta, linear_beta=linear_beta or 0.0,
        weight_scale=None, activation_scale=None,
        activate_left=True, quant_mode="dynamic"
    )

    # 注意: y_npu 是 INT8，需要反量化回 BF16 才能对比
    # ...

    diff = (ref.npu() - y_bf16).abs()
    print(f"SiTU max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
```
