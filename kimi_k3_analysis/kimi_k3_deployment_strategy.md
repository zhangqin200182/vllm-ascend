# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: ModelScope `moonshotai/Kimi-K3` config.json + HuggingFace 权重仓库

---

## 一、模型真实参数与权重格式

### 1.1 参数规模

| 指标 | 数值 |
|---:|---:|
| **总参数量**（BF16 等效） | **~2.78T (2780B)** |
| Expert 参数量 | ~2.6T（占总参数 93%） |
| 非 Expert 参数量 | ~0.18T（attention/embedding/shared/norm） |

### 1.2 发布权重的实际格式

Moonshot 团队在训练时就使用 MXFP4/MXFP8 混合精度，发布权重直接是训练精度：

| 仓库 | 大小 | 格式 |
|---|---|---|
| `moonshotai/Kimi-K3` | **~1560 GB** | **混合精度**: Expert 权重 MXFP4 (0.5 bytes/param) + 其余 BF16 (2 bytes/param) |
| `moonshotai/Kimi-K3-MXFP4` | **~594 GB** | **全量 MXFP4** |

推演验证：
```
Expert 权重 (~2.6T params, 93%):
  MXFP4:  2.6T × 0.5 bytes ≈ 1300 GB

非 Expert 权重 (~0.18T params):
  BF16:   0.18T × 2 bytes ≈ 360 GB

合计: 1300 + 360 ≈ 1660 GB  ← 接近实际 1560 GB ✓
```

### 1.3 核心超参

| 参数 | 数值 |
|---:|---:|
| `num_hidden_layers` | 93 |
| `hidden_size` | 7168 |
| `num_attention_heads` | 96 |
| `intermediate_size` | 33792 |
| `moe_intermediate_size` | 3072 |
| `num_experts` | 896（每层独立） |
| `num_experts_per_token` | 16 |
| KDA 层数 | 68 |
| MLA 层数 | 24 |
| `gate_lower_bound` | -5.0 |
| SiTU `beta` / `linear_beta` | 4.0 / 25.0 |

---

## 二、指令精度支持

| 平台 | BF16 | FP16 | FP32 | INT8 | INT4 | MXFP4 | MXFP8 |
|---|---|---|---|---|---|---|---|
| **A2 (910B)** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **A3 (910_93)** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **A5 (950)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **关键结论: A2/A3 不支持 MXFP4 原生计算**。A5 才支持 MX 格式。
>
> 在 A2/A3 上部署 Kimi K3 时，MXFP4 权重需要**运行时反量化**为 BF16/FP16 后再计算。vLLM-Ascend 的 W4A8 + `dequant_situ_quant` 路径正是做这件事：GMM 将 INT4/MXFP4 权重反量化到 BF16，然后 `dequant_situ_quant` 算 SiTU 激活 → 动态 INT8 量化输出。这就是为什么 A2/A3 必须走 `dequant_situ_quant` 而不是 `situ_mx_quant`。

---

## 三、单卡显存估算（修订）

### 3.1 权重占用

**以 `moonshotai/Kimi-K3`（混合精度仓库）在 A2 上部署为例：**

| 存储格式 | Expert 权重 | 非 Expert 权重 | 加载总大小 |
|---|---|---|---|
| 磁盘 (ModelScope) | ~1300 GB (MXFP4-packed) | ~360 GB (BF16) | ~1660 GB |
| 运行时 (A2 NPU) | **需反量化** | ~360 GB (BF16 直接使用) | 取决于反量化策略 |

A2 上运行时有两种方案：

**方案 1: 加载时全量反量化为 BF16（内存密集型）**

```
运行时 BF16 权重总量: 2.78T × 2 bytes = ~5560 GB

TP=8:  5560 / 8 = 695 GB/卡   ← 单卡 64 GB 完全不可能
TP=64: 5560 / 64 = 87 GB/卡   ← 超越单卡限制
```

> ❌ 全量 BF16 部署不可行，必须借助 **EP + 运行时反量化**。

**方案 2: 运行时 W4A8 + 即时反量化（vLLM-Ascend 推荐）**

```
Expert 权重以压缩格式存储在 HBM 中: ~1300 GB
非 Expert BF16:                     ~360 GB
加载总大小 (HBM 存储):              ~1660 GB

TP=8（纯 TP，不开启 EP）:
  每卡: 1660 / 8 = 207.5 GB  ← 远超 64 GB

TP=8 + EP=8 (64 卡):
  Expert: 1300 / 8 / 8 = 20.3 GB/卡
  非 Expert (TP=8 shard): 360 / 8 = 45 GB/卡
  权重总计: ~65.3 GB/卡  ← 紧贴 64 GB 上限！
```

> ⚠️ **TP=8 + EP=8 下每卡 ~65 GB，刚好超出 64 GB 上限。需要在策略层面进一步优化。**

**方案 3: TP=8 + EP=16 (128 卡)**

```
Expert: 1300 / 8 / 16 = 10.2 GB/卡
非 Expert (TP=8 shard): 360 / 8 = 45 GB/卡
权重总计: ~55.3 GB/卡
```

> ✅ 配合 `gpu-memory-utilization 0.85` 可稳定运行。

**方案 4: TP=1 + EP=64 (64 卡)**

```
Expert: 1300 / 64 = 20.3 GB/卡
非 Expert (无 TP): 360 GB/卡  ← 问题！非 Expert 权重未切分
权重总计: ~380 GB/卡  ← 不可行
```

> ❌ 纯 EP 不切分 attention 权重，必须同时开启 TP。

### 3.2 最终推荐

| 方案 | 卡数 | 并行 | 权重 GB/卡 | KV Cache | 合计 | 可行性 |
|---|---|---|---|---|---|---|
| TP=8, 无 EP | 8 (1节点) | TP=8 | 207.5 | 5-10 | ~215 | ❌ |
| TP=8, EP=8 | 64 (8节点) | TP=8, EP=8 | 65.3 | 10-20 | ~80 | ⚠️ 临界 |
| **TP=8, EP=16** | **128 (16节点)** | TP=8, EP=16 | **55.3** | 15-30 | **~75** | ✅ |
| TP=4, EP=32 | 128 (16节点) | TP=4, EP=32 | 55.6 | 15-30 | ~80 | ✅ |
| TP=2, EP=64 | 128 (16节点) | TP=2, EP=64 | 55.1 | 15-30 | ~80 | ✅ |

> **推荐生产配置: TP=8 + EP=16 (128 卡)**。TP=8 切分 attention 和 embedding，EP=16 切分 expert 权重，磁盘 I/O 和运行时内存均可控。

---

## 四、部署命令

### 生产推荐配置（128 卡）

```bash
vllm serve <model-path> \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 16 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

### 功能验证（单节点，无 EP，需全量反量化模式）

> 单节点无法直接用 HBM 存储全量权重。如有 A5 平台（支持 MXFP4 原生计算），单节点可行。

---

## 五、各平台能力对比

| 平台 | MXFP4 原生计算 | SiTU 量化路径 | Kimi K3 推荐方案 |
|---|---|---|---|
| **A2 (910B)** | ❌ 不支持 | `dequant_situ_quant` (反量化→SiTU→INT8) | 128 卡: TP=8+EP=16 |
| **A3 (910_93)** | ❌ 不支持 | `dequant_situ_quant` (+ dispatch 融合) | 128 卡: TP=8+EP=16 |
| **A5 (950)** | ✅ 支持 | `situ_mx_quant` (SiTU→MXFP8) | 128 卡: TP=8+EP=16，或更少卡 (MXFP4 直接计算) |
| **310P** | ❌ | ❌ 缺少算子 | Kimi K3 不可运行 |

---

## 六、总结

| 场景 | 最小硬件 | 并行方案 | 说明 |
|---|---|---|---|
| 功能验证 | **A5: 8-16 卡** | TP=8, EP=1-2 | A5 原生支持 MXFP4 计算 |
| A2/A3 开发 | 128 卡 (16节点) | TP=8, EP=16 | ~75 GB/卡含 KV cache |
| A2/A3 生产 | 128-256 卡 | TP=8, EP=16-32 | 叠加 DP 扩展吞吐 |
| A5 生产 | 16-64 卡 | TP=8, EP=1-8 | MXFP4 原生计算大幅降低内存 |

> **由于 93% 参数在 Expert 中且为 MXFP4 格式，A2/A3 部署的核心约束是 Expert 权重的运行时反量化开销。TP + EP 组合是唯一可行的方案。AI Gateway 侧可叠加 DP 进一步扩展。**
