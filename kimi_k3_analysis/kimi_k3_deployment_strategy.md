# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 官方模型: ModelScope `Eco-Tech/Kimi-K3-w4a8` (1467 GB)
> 原始权重: HF `moonshotai/Kimi-K3` (1560 GB, MXFP4)

---

## 一、模型规模

| 指标 | 数值 |
|---:|---:|
| 总参数量（BF16 等效） | ~2.78T |
| 层数 | 93 (68 KDA + 24 MLA) |
| Expert/层 | 896（每层独立，共 83328 实例） |
| Attention heads | 96 |

### 官方 W4A8 模型 (`Eco-Tech/Kimi-K3-w4a8`, 1467 GB)

```
quant_model_description.json 量化分布:

  W4A8_DYNAMIC:  989,184 tensors  ← Expert (INT4, 0.5 B/param)
  W8A8_DYNAMIC:    2,227 tensors  ← Attention + MLP Linear (INT8, 1.0 B/param)
  FLOAT:           1,887 tensors  ← Norms / Embedding / LM Head / Gates (BF16)
  group_size: 0 (per-channel)
```

| 组件 | 格式 | 大小 |
|---|---|---|
| Expert 权重 | W4A8 (INT4) | ~1376 GB |
| Attention q/k/v/o_proj | W8A8 (INT8) | ~50 GB |
| MLP gate/up/down | W8A8 (INT8) | ~25 GB |
| Norms, Embedding, LM Head, Gates | FLOAT (BF16) | ~16 GB |
| **合计** | | **~1467 GB** |

---

## 二、部署路径

```
HF: moonshotai/Kimi-K3 (1560 GB, MXFP4)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   Path 1: 不转换                      Path 2: ModelSlim 转换
   MXFP4 → FP8 (A5 专用)              MXFP4 → INT4 (A2/A3)

   权重 FP8: 2752 GB                   权重 INT4: 1376 GB
   非Expert: 184 GB(BF16)              非Expert: 91 GB(W8A8+FP)
   HBM 合计: 2936 GB                   官方模型: 1467 GB
   A2 可用: ❌                          A2 可用: ✅
```

| | Path 1: MXFP4→FP8 (A5) | Path 2: INT4 (A2/A3) |
|---|---|---|
| 平台 | A5 专用 | A2 / A3 |
| 离线转换 | 不需要 | **ModelSlim (MXFP4→INT4 + rotation)** |
| Expert HBM 格式 | FP8 (1 B/elem) | INT4 (0.5 B/elem) |
| Expert HBM 大小 | 2752 GB | 1376 GB |
| 非Expert 格式 | BF16 (2 B/elem) | W8A8 (1 B/elem) + FLOAT |
| 非Expert 大小 | 184 GB | 91 GB |
| **合计 HBM** | **2936 GB** | **1467 GB** |
| **官方模型** | — | `Eco-Tech/Kimi-K3-w4a8` |
| A2 直接加载 HF | — | ❌ `npu_format_cast(FP4→FP8)` 不支持 |

> **A2/A3 不能直接 `vllm serve moonshotai/Kimi-K3`。** HF 的 MXFP4 权重会触发 `npu_format_cast(FP4→FP8)`，A2/A3 不支持此转换，且 FP8 GMM kernel 也不可用。

---

## 三、A2 64 GB 部署估算 (官方 W4A8 模型)

```python
# 权重 shape (来自 vllm_ascend/quantization/methods/w4a8_mxfp4.py:get_weight)
w13_weight: [num_experts, 2*intermediate//TP, hidden//2] uint8  # gate_up, INT4 packed
w2_weight:  [num_experts, hidden//TP, intermediate//2] uint8     # down, INT4 packed
```

| TP | EP | GPU | Expert/GPU | 非Expert/GPU | 权重合计 | +KV(15GB) | 总占用 | |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 8 | 8 | **64** | 21.5 GB | 11.4 GB | **32.9 GB** | ~48 GB | ✅ 推荐 |
| 8 | 4 | 32 | 43.0 GB | 11.4 GB | **54.4 GB** | ~69 GB | ⚠️ 临界 |
| 8 | 2 | 16 | 86.0 GB | — | — | — | ❌ |

> TP=8 必须（96 heads 被 8 整除）
> TP=4 不可行（非Expert 翻倍到 22.8 GB，虽然仍 <64GB，但 Expert 43GB 时总占用 >66GB 无 KV 余量）

### 32 GPU 场景（TP=8, EP=4）

```
权重:          54.4 GB
激活+workspace: ~3.0 GB
KV cache 余量: ~6.5 GB

MLA KV ≈ 9.3 KB/token → 6.5 GB ≈ 700K tokens
max-num-seqs=8:  context ≤ 6400
max-num-seqs=16: context ≤ 3200
max-num-seqs=32: context ≤ 1600
```

> ⚠️ 仅适合功能验证/短文本，不适合生产。

### 64 GPU 场景（TP=8, EP=8）

```
权重:          32.9 GB
激活+workspace: ~3.0 GB
KV cache 余量: ~25 GB

max-num-seqs=64:  context ≤ 32000  ✅ 生产可用
max-num-seqs=32:  context ≤ 64000  ✅
```

---

## 四、部署命令

### 64 GPU 生产部署

```bash
vllm serve Eco-Tech/Kimi-K3-w4a8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

### 32 GPU 功能验证

```bash
vllm serve Eco-Tech/Kimi-K3-w4a8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 4 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85
```

---

## 五、各平台汇总

| 平台 | 部署模型 | 最小 GPU | 推荐 GPU | 说明 |
|---|---|---|---|---|
| **A2 (910B, 64 GB)** | `Eco-Tech/Kimi-K3-w4a8` | 32 (EP=4) | **64 (EP=8)** | ModelSlim 转换后 |
| **A3 (910_93, 64 GB)** | `Eco-Tech/Kimi-K3-w4a8` | 32 (EP=4) | **64 (EP=8)** | 同上 + dispatch 融合 |
| **A5 (950)** | `moonshotai/Kimi-K3` (直载) | 待确认 | 待确认 | MXFP4 原生 |
| **310P** | ❌ | — | — | 缺少算子 |

---

## 六、常见问题

**Q: 能直接用 `vllm serve moonshotai/Kimi-K3` 吗？**

A2/A3 不能。HF 权重是 MXFP4 格式，A2/A3 没有 FP8 计算能力，且模型加载时会触发不支持的 `npu_format_cast`。必须用 `Eco-Tech/Kimi-K3-w4a8`（已 ModelSlim 转换）。

**Q: 官方 W4A8 模型比 HF 原始模型小多少？**

1467 GB vs 1560 GB，小约 6%。主要是 Attention 和 MLP 从 BF16 量化到 W8A8（INT8），省了一半。

**Q: 单节点 8 GPU 能跑吗？**

不能。Expert 1376 GB / 8 = 172 GB，远超单卡 64 GB。
