# Kimi K3 RL 场景：训练与推理的 GPU 鸿沟

> 问题记录: 2026-07-28
> 状态: 待讨论

---

## 问题

RL 需要训练和推理共用 BF16 权重（实时同步、无损精度）。但**同样的 128 GPU，训练能跑 BF16，vLLM 推理跑不了**。

## 根因

训练和推理的**非Expert 权重分片方式不同**：

```
                    训练 (MindSpeed-MM)              推理 (vLLM)
                    ──────────────────               ────────────
Expert 分片:        EP                            TP × EP
非Expert 分片:      FSDP2 (全 N 卡均匀分片)          TP (仅 TP 组内分片)

128 GPU 下:
  Expert:    EP=128 → 5200/128 = 40.6 GB    TP×EP=128 → 5200/128 = 40.6 GB
  非Expert:  FSDP2 → 260/128 = 2.0 GB      TP=8 → 260/8 = 32.5 GB  ← 差距在这里
  ───────────────────────────────────────────────────────────────────────
  合计:      42.6 GB ✅                     73.1 GB ❌
```

vLLM 的 TP 切分粒度比 FSDP2 粗得多——TP 最多到 16（头数限制），FSDP2 可以切到 128。

## GPU 需求断层

| | 训练 BF16 | vLLM BF16 | vLLM W4A8 |
|---|---|---|---|
| 最小 GPU | **128** (LoRA) | **256** (TP=16, EP=16) | **64** (TP=8, EP=8) |
| 权重/GPU | 42.6 GB | 36.6 GB | 32.9 GB |
| 精度 | BF16 无损 | BF16 无损 | INT4 有损 |
| 权重同步 | 实时 | 实时 | ModelSlim N步延迟 |
| RL 可用 | ✅ rollout 用 FSDP2 forward | ✅ vLLM serve | ✅ vLLM serve |

核心矛盾：

```
训练 128 GPU (BF16)  ←→  推理 256 GPU (BF16)
                       ↕  差 128 GPU
                       
训练 128 GPU (BF16)  ←→  推理 64 GPU (W4A8)
                       ↕  需 ModelSlim 离线量化 + 额外推理集群
```

## 可能的解决方向

### 方向 A: vLLM 支持非Expert 走 DP shard

如果 vLLM 的 DP 能把非Expert 权重也在 DP 组间切分（类似 FSDP2 的 fully shard），128 GPU 就有解了：

```
TP=8, DP=16, EP=1 (128 GPU):
  Expert:    5200/8/1 = 650 GB ❌  不行，EP 还是瓶颈

TP=8, DP=8, EP=2 (128 GPU):
  Expert:    5200/8/2 = 325 GB ❌

非Expert DP shard 可以降，但 Expert 只能靠 EP，TP×DP×EP=128 的乘积锁住了
```

最终取决于 vLLM 是否支持 TP×DP×EP 的三维并行（非Expert 用 TP+DP 分片，Expert 用 TP+EP 分片）。

### 方向 B: 训练集群用更大 HBM 的卡

如果用 A5 (128 GB)，vLLM BF16 在 128 GPU 上：

```
TP=8, EP=16, 128 GPU:
  Expert: 5200/128 = 40.6 GB
  非Expert: 260/8 = 32.5 GB
  合计: 73.1 GB / 128 GB ✅ (有 55 GB KV 余量)
```

### 方向 C: 接受 W4A8 + N 步延迟

训练 128 GPU (BF16) + 推理 64 GPU (W4A8)。
权重每 N 步做一次 ModelSlim 量化同步。
off-policy RL 可接受一定延迟。

### 方向 D: 混合精度推理

vLLM 加载时 Expert 保持量化存储（W4A8），仅非Expert 保持 BF16。
一半实时（非Expert BF16 实时更新）、一半延迟（Expert 需量化）。
Expert 更新频率低时可行。

---

## 讨论要点

1. vLLM-Ascend 是否能扩展 DP 来切分非Expert 权重？技术难度多大？
2. RL 场景是否必须在单集群上完成训练+推理？双集群可行吗？
3. 权重同步延迟对 RL 训练收敛的影响有多大？
4. A5 平台的 HBM 容量能否解决这个 gap？
