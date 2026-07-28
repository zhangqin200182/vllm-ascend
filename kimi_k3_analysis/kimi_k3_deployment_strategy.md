# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: HuggingFace `moonshotai/Kimi-K3` + ModelScope `moonshotai/Kimi-K3`
> **目标硬件: A2 (Ascend 910B), 64 GB HBM/卡**

---

## 一、模型参数与发布权重

### 1.1 参数规模

| 指标 | 数值 |
|---:|---:|
| 总参数量（BF16 等效） | ~2.78T (2780B) |
| Expert 参数量 | ~2.6T（占总量 93%） |
| 非 Expert 参数量 | ~0.18T（attention/embedding/shared/norm） |
| 纯 BF16 总大小 | ~5560 GB (2.78T × 2 bytes) |

### 1.2 发布仓库

Moonshot AI 团队在训练时使用 MXFP4/MXFP8 混合精度原生训练，发布的 checkpoint 直接就是训练精度：

| 仓库 | 平台 | 大小 | 格式 |
|---|---|---|---|
| `moonshotai/Kimi-K3` | HuggingFace | **1560 GB** | 混合精度: Expert MXFP4 (~1300 GB) + 非Expert BF16 (~260 GB) |
| `moonshotai/Kimi-K3` | ModelScope | **1560 GB** | 同上 |

> **只有一个仓库，不存在独立的 MXFP4 全量量化版本。** 之前的分析中假设存在 `Kimi-K3-MXFP4` (594 GB) 是错误的。

### 1.3 非 Expert BF16 部分详解

量化配置中明确排除了以下模块的量化（保持 BF16）：

```
ignore: ['re:.*self_attn.*', 're:.*shared_experts.*', 're:.*mlp\\.(gate|up|gate_up|down)_proj.*', 're:.*lm_head.*']
```

| 组件 | 层数 | 说明 | 估计 BF16 大小 |
|---|---|---|---|
| KDA Attention (q/k/v/o_proj, gate proj) | 68 层 | 含 full_rank_gate 输出投影 | ~100 GB |
| MLA Attention (q_a/q_b/kv_a/kv_b/o_proj) | 24 层 | 低秩压缩 | ~40 GB |
| Shared Experts (gate_up + down, ×2) | 93 层 | 每层 2 个共享专家 | ~20 GB |
| Embedding (163840 × 7168) | 1 | | ~2.3 GB |
| LM Head (7168 × 163840) | 1 | | ~2.3 GB |
| RMSNorm + Router gates + 其他 | 93 层 | | ~95 GB |
| **合计** | | | **~260 GB** |

> 这 ~260 GB BF16 权重不受 EP 切分影响——EP 只切 Expert，不切 attention/embedding/shared/norm。

### 1.4 A2 上的计算约束

A2 **不支持 MXFP4 原生计算**（MXFP4 需要 A5 硬件）。A2 指令集支持：BF16、FP16、FP32、INT8、INT4。

在 A2 上部署时：
- Expert MXFP4 权重 → HBM 中以 MXFP4-packed 格式存储
- GMM（Grouped MatMul）计算时即时反量化 MXFP4 → BF16
- 反量化发生在计算单元内部，不额外占用 HBM
- SiTU 激活通过 `dequant_situ_quant` 融合 → 输出 INT8

---

## 二、A2 (64 GB) 显存估算

### 2.1 权重占用

```
Expert (MXFP4-packed):  ~1300 GB ← EP 可切分
非Expert (BF16):        ~260 GB  ← EP 不可切分，仅 TP 可切
加载总大小:             ~1560 GB
```

### 2.2 各方案每卡显存

| TP | EP | 卡数 | Expert GB/卡 | 非Expert GB/卡 | 权重合计 | +KV Cache | +激活 | **总占用** | 可行性 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 8 | — | 8 | 162.5 | 32.5 | 195.0 | — | — | **>195** | ❌ |
| 8 | 8 | 64 | 20.3 | 32.5 | 52.8 | 5-10 | 2-3 | **~63** | ⚠️ 临界 |
| 8 | 16 | 128 | 10.2 | 32.5 | 42.7 | 10-15 | 2-3 | **~57** | ✅ |
| 8 | 32 | 256 | 5.1 | 32.5 | 37.6 | 15-25 | 2-3 | **~60** | ✅ |
| 4 | 32 | 128 | 10.2 | 65.0 | 75.2 | — | — | **>75** | ❌ |
| 4 | 64 | 256 | 5.1 | 65.0 | 70.1 | — | — | **>70** | ❌ |

> **关键发现**: TP=4 不可行——非Expert BF16 每卡 65 GB 已超单卡容量。**必须 TP≥8，且必须 EP≥16**。
>
> 瓶颈始终是**非Expert BF16 权重**（260 GB），这部分不受 EP 影响。TP=8 时每卡固定 32.5 GB，占总 HBM 的一半。

### 2.3 最小可行配置

```
TP=8 + EP=16:  128 卡 (16 节点) ≈ 57 GB/卡  ✅ 推荐
TP=8 + EP=8:    64 卡 (8 节点)  ≈ 63 GB/卡  ⚠️ 临界，需严格限制 KV cache
```

---

## 三、推荐部署命令

### 生产配置（128 卡，16 节点）

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 16 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

### 临界配置（64 卡，8 节点）

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.80
```

---

## 四、各平台对比

| 平台 | 推荐方案 | 最小卡数 | 说明 |
|---|---|---|---|
| **A2 (910B, 64 GB)** | TP=8 + EP=16 | **128 卡** | 非Expert BF16 为瓶颈 |
| A3 (910_93, 64 GB) | TP=8 + EP=16 | 128 卡 | 同 A2，有 dispatch 融合优化 |
| **A5 (950)** | TP=8 + EP=4 或更少 | **待验证, 预计 32-64 卡** | MXFP4 原生计算 + `situ_mx_quant`，减少反量化开销 |
| 310P | ❌ 不支持 | — | 缺少关键算子 |

---

## 五、总结

```
Kimi K3 部署约束:
  Expert MXFP4 (~1300 GB) → EP 可切分 → 不是瓶颈
  非Expert BF16 (~260 GB) → EP 不可切 → TP=8 下每卡固定 32.5 GB
  64 GB HBM  → 权重占用 32.5 + Expert 分片 + KV cache < 64 GB
           → EP≥16 时 Expert 分片 ≤ 10.2 GB/卡
           → 总占用 ≈ 57 GB/卡 ✅

  A2 最小生产：128 卡 (TP=8 + EP=16)
  临界验证：   64 卡 (TP=8 + EP=8, 极严格 KV cache 限制)
  单节点：     不可行（195 GB/卡）
```
