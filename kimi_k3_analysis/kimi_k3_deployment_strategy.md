# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: HuggingFace `moonshotai/Kimi-K3` config.json + 代码 shape 精确计算
> 补充分析: [kimi_k3_deployment_paths.md](kimi_k3_deployment_paths.md) — 两条部署路径的完整对比

---

## 一、模型规模

| 指标 | 数值 |
|---:|---:|
| 总参数量（BF16 等效） | ~2.78T |
| 层数 | 93（68 KDA + 24 MLA） |
| Expert/层 | 896（每层独立，共 83328 实例） |
| 单个 Expert 实例 (INT4) | gate_up 0.0110 GB + down 0.0055 GB = **0.0165 GB** |
| Expert 全量 (INT4) | 83328 × 0.0165 = **1376 GB** |
| 非Expert (BF16) | 184 GB（checkpoint 总量 1560 - Expert 1376） |
| **磁盘/HBM 合计** | **1560 GB** |

> 权重 shape 来源: `vllm_ascend/quantization/methods/w4a8_mxfp4.py:get_weight()`
>
> ```python
> w13_weight: [num_experts, 2*intermediate, hidden//2] uint8  # gate_up, INT4 packed
> w2_weight:  [num_experts, hidden, intermediate//2] uint8     # down, INT4 packed
> ```

---

## 二、部署路径

```
HF: moonshotai/Kimi-K3 (1560 GB, Expert MXFP4 + Non-Expert BF16)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
    Path 1: 不转换                     Path 2: ModelSlim 转换
    MXFP4 → FP8 (A5 专用)             MXFP4 → INT4 (A2/A3 通用)

    Expert HBM: 2752 GB (FP8)          Expert HBM: 1376 GB (INT4)
    合计 HBM:   2936 GB                合计 HBM:   1560 GB
```

> A2/A3 必须走 Path 2（ModelSlim 离线转换），详见 [kimi_k3_deployment_paths.md](kimi_k3_deployment_paths.md)。

---

## 三、A2/A3 部署估算（Path 2: INT4）

**约束**: A2 64 GB HBM，A3 64 GB HBM，权重 INT4 packed + 非Expert BF16

| TP | EP | GPU 数 | Expert/GPU | 非Expert/GPU | 权重合计 | +KV(15GB) | 总占用 | |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 8 | 16 | 128 | 10.8 GB | 23.0 GB | **33.8 GB** | ~49 GB | ✅ 推荐 |
| 8 | 8 | **64** | 21.5 GB | 23.0 GB | **44.5 GB** | ~60 GB | ✅ 可行 |
| 8 | 4 | 32 | 43.0 GB | 23.0 GB | 66.0 GB | — | ❌ 超限 |
| 4 | 32 | 128 | 10.8 GB | 46.0 GB | 56.8 GB | >72 GB | ❌ TP=4 非Expert 已超 |

> **非Expert BF16 (184 GB)** 不受 EP 切分。TP=8 下每卡固定 23 GB，是真正的硬地板。

---

## 四、推荐方案

### 生产（128 GPU, 16 节点）

```bash
vllm serve <ModelSlim-转换后的-INT4-模型> \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 16 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

每 GPU: 权重 34 GB + KV ~15 GB ≈ 49 GB / 64 GB ✅

### 有限验证（64 GPU, 8 节点）

```bash
vllm serve <ModelSlim-转换后的-INT4-模型> \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85
```

每 GPU: 权重 45 GB + KV ~15 GB ≈ 60 GB / 64 GB ⚠️ 接近上限

---

## 五、常见问题

**Q: 能直接用 `vllm serve moonshotai/Kimi-K3` 吗？**

A2/A3 上不行。HF 权重是 MXFP4 格式，需要 ModelSlim 离线转 INT4 后才能部署。

**Q: 转换前后 HBM 占用有变化吗？**

没有。MXFP4 和 INT4 都是 `uint8` 容器、`0.5 bytes/param`，大小相同。

**Q: 单节点 8 GPU 能跑吗？**

不能。Expert 1376 GB / 8 = 172 GB/GPU，加上非Expert 23 GB = 195 GB，远超 64 GB。

---

## 六、各平台汇总

| 平台 | 部署路径 | 最小 GPU | 推荐 GPU |
|---|---|---|---|
| **A2 (910B, 64 GB)** | Path 2 (INT4) | 64 (EP=8) | 128 (EP=16) |
| **A3 (910_93, 64 GB)** | Path 2 (INT4) | 64 (EP=8) | 128 (EP=16) |
| **A5 (950)** | Path 1 (FP8) 或 Path 2 (INT4) | 待验证 | 待验证 |
| **310P** | ❌ 不可用 | — | — |
