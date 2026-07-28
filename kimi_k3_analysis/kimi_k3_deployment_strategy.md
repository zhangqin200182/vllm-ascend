# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: ModelScope `moonshotai/Kimi-K3` config.json + HuggingFace 权重仓库
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

### 1.2 HuggingFace 两个仓库

| 仓库 | 大小 | 格式 |
|---|---|---|
| `moonshotai/Kimi-K3` | 1560 GB | **混合精度**: Expert MXFP4 (~1300 GB) + 非Expert BF16 (~260 GB) |
| `moonshotai/Kimi-K3-MXFP4` | 594 GB | **全量 MXFP4**: 所有权重 4-bit 存储 |

### 1.3 A2 上的部署约束

A2 **不支持 MXFP4 原生计算**。与 A5 不同，A2 无法直接在 MXFP4 格式上做矩阵乘。A2 的 NPU 指令集支持：BF16、FP16、FP32、INT8、INT4。

在 A2 上部署时：
- MXFP4 权重→**HBM 中以 MXFP4 格式存储**（节省存储空间）
- 计算时 GMM/W4A8 路径**即时反量化**为 BF16 → 乘加 → 输出
- 反量化是计算的一部分，不额外占用 HBM（中间结果在 L1/UB 中）
- **HBM 中存储的格式决定显存占用，不是计算格式**

### 1.4 核心超参

| 参数 | 数值 |
|---:|---:|
| `num_hidden_layers` | 93 (68 KDA + 24 MLA) |
| `hidden_size` | 7168 |
| `num_attention_heads` | 96 |
| `num_experts` | 896（每层独立） |
| `num_experts_per_token` | 16 |
| `gate_lower_bound` | -5.0 |
| SiTU `beta`/`linear_beta` | 4.0/25.0 |

---

## 二、A2 单卡显存估算

### 2.1 使用 `moonshotai/Kimi-K3`（混合精度仓库, 1560 GB）

```
Expert (MXFP4-packed):      ~1300 GB
非Expert (BF16):            ~260 GB  ← attention/embedding/shared/norm/norm 等
加载总大小 (HBM):           ~1560 GB
```

| TP | EP | 卡数 | Expert GB/卡 | 非Expert GB/卡 | 权重合计 | KV Cache | 总计 | 可行性 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 8 | 162.5 | 32.5 | **195.0** | — | >195 | ❌ |
| 8 | 8 | 64 | 20.3 | 32.5 | **52.8** | 5-10 | ~62 | ⚠️ 临界 |
| 8 | 16 | 128 | 10.2 | 32.5 | **42.7** | 10-15 | ~55 | ✅ |
| 8 | 32 | 256 | 5.1 | 32.5 | **37.6** | 15-25 | ~55 | ✅ |

> 瓶颈在**非Expert BF16 权重**（260 GB / TP=8 = 32.5 GB/卡），这部分**不受 EP 影响**——attention/embedding/shared/norm 不通过 EP 切分。

### 2.2 使用 `moonshotai/Kimi-K3-MXFP4`（全量 MXFP4, 594 GB）⭐ 推荐

```
Expert (MXFP4):              ~552 GB (93%)
非Expert (MXFP4):            ~42 GB  (7%)
加载总大小 (HBM):            ~594 GB
```

| TP | EP | 卡数 | Expert GB/卡 | 非Expert GB/卡 | 权重合计 | KV Cache | 总计 | 可行性 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1 | 8 | 69.0 | 5.3 | **74.3** | — | >74 | ❌ |
| 8 | 8 | 64 | 8.6 | 5.3 | **13.9** | 10-20 | ~30 | ✅ |
| 8 | 4 | 32 | 17.3 | 5.3 | **22.6** | 10-20 | ~40 | ✅ |
| 8 | 2 | 16 | 34.5 | 5.3 | **39.8** | 10-15 | ~55 | ✅ |
| 4 | 8 | 32 | 17.3 | 10.5 | **27.8** | 10-20 | ~45 | ✅ |
| 4 | 4 | 16 | 34.5 | 10.5 | **45.0** | 10-15 | ~60 | ⚠️ 临界 |

> **推荐 TP=8, EP=4, 32 卡 (4 节点)**。每卡 ~23 GB 权重 + ~15 GB KV cache = ~38 GB，余量充足。

### 2.3 两个仓库对比

| | 混合精度仓库 (1560 GB) | MXFP4 仓库 (594 GB) |
|---|---|---|
| 非Expert 格式 | BF16 → **A2 可直接计算** | MXFP4 → **需反量化，但 HBM 占用小** |
| TP=8 下非Expert/卡 | 32.5 GB | 5.3 GB |
| 最小可行配置 | 128 卡 (TP=8, EP=16) | **32 卡 (TP=8, EP=4)** |
| 推荐场景 | A5（原生 MXFP4 计算）/ 已有权重 | **A2 部署首选** |

> **在 A2 上，MXFP4 仓库是更好的选择**。虽然 A2 不能原生计算 MXFP4，但 GMM 反量化路径已经处理了这个问题。关键是 HBM 存储——594 GB vs 1560 GB，非Expert 部分缩小了 6 倍。

---

## 三、推荐部署方案

### 方案 A：MXFP4 仓库 + TP=8 + EP=4（推荐）

```
硬件: 4 节点 × 8 × A2 (64 GB) = 32 卡
仓库: moonshotai/Kimi-K3-MXFP4 (594 GB)
```

```bash
vllm serve moonshotai/Kimi-K3-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 4 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85
```

| 指标 | 数值 |
|---|---|
| 权重/卡 | ~23 GB |
| KV Cache 余量 | ~15-20 GB (16K × 32 seqs) |
| 激活+其他余量 | ~15-20 GB |

### 方案 B：MXFP4 仓库 + TP=8 + EP=2（最小卡数）

```
硬件: 2 节点 × 8 × A2 (64 GB) = 16 卡
```

```bash
vllm serve moonshotai/Kimi-K3-MXFP4 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 2 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85
```

| 指标 | 数值 |
|---|---|
| 权重/卡 | ~40 GB |
| KV Cache 余量 | ~8-12 GB (8K × 16 seqs) |
| 可行性 | ⚠️ 临界，需严格控制 context 和并发 |

### 方案 C：混合精度仓库 + TP=8 + EP=16（大容量）

```
硬件: 16 节点 × 8 × A2 (64 GB) = 128 卡
仓库: moonshotai/Kimi-K3 (1560 GB)
```

适用于已有混合精度权重、不愿重新下载的场景。

---

## 四、方案可行性速查

```
仓库: moonshotai/Kimi-K3-MXFP4 (594 GB) [推荐]

TP=8 + EP=4:  32 卡 ✅  权重 23 GB + KV ~15 GB = ~38 GB
TP=8 + EP=2:  16 卡 ⚠️  权重 40 GB + KV ~10 GB = ~50 GB (临界)
TP=4 + EP=8:  32 卡 ✅  权重 28 GB + KV ~15 GB = ~43 GB

仓库: moonshotai/Kimi-K3 (1560 GB, 混合精度)

TP=8 + EP=16: 128 卡 ✅  权重 43 GB + KV ~15 GB = ~58 GB
TP=8 + EP=8:   64 卡 ⚠️  权重 53 GB + KV ~10 GB = ~63 GB (临界)
TP=8 + EP=4:   32 卡 ❌  权重 73 GB 单权重已超 64 GB
```

---

## 五、总结

| 场景 | 仓库 | 最小硬件 | 并行方案 |
|---|---|---|---|
| **推荐部署** | MXFP4 (594 GB) | **32 卡 (4 节点)** | TP=8, EP=4 |
| 最小验证 | MXFP4 (594 GB) | 16 卡 (2 节点) | TP=8, EP=2, 界限配置 |
| 兼容部署 | 混合精度 (1560 GB) | 128 卡 (16 节点) | TP=8, EP=16 |

> **关键结论**: A2 上用 MXFP4 仓库 (594 GB) 比混合精度仓库 (1560 GB) 显存节省 2.6 倍，32 卡即可部署。瓶颈不在 Expert 权重（EP 可切分），而在于**非Expert 权重的 HBM 存储**——混合精度仓库中这部分用 BF16 存储占 260 GB，TP=8 下每卡固定 32.5 GB；MXFP4 仓库中仅 42 GB，TP=8 下每卡仅 5.3 GB。
