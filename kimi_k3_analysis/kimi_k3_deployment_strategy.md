# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 官方模型: ModelScope `Eco-Tech/Kimi-K3-w4a8` (1467 GB)
> 原始权重: HF `moonshotai/Kimi-K3` (1560 GB, MXFP4)
> 拓扑来源: vLLM `parallel_state.py:initialize_model_parallel` + `fused_moe/config.py:FusedMoEParallelConfig`

---

## 一、模型规模

| 指标 | 数值 |
|---:|---:|
| 总参数量（BF16 等效） | ~2.78T |
| 层数 | 93 (68 KDA + 24 MLA) |
| Expert/层 | 896（每层独立，共 83328 实例） |
| Attention heads | 96 |

| 组件 | BF16 大小 | W4A8 官方模型 |
|---|---|---|
| Expert 权重 | ~5200 GB | ~1376 GB (INT4) |
| 非Expert (Attention+MLP+Norms+Emb) | ~260 GB | ~91 GB (W8A8+FLOAT) |
| **合计** | **~5460 GB** | **~1467 GB** |

---

## 二、vLLM 并行拓扑（关键）

```
vLLM 布局: ExternalDP × DP × PP × PCP × TP

EP group = DP × PCP × TP   ← 代码: parallel_state.py line 1894-1902

EP 组内: Expert 分片到每个 rank
EP 组间 (ExternalDP): Expert 权重复制
非Expert (Attention/Norm/Emb): 仅由 TP 分片
```

**128 GPU，ExternalDP=1 时的所有合法方案（BF16 直载）：**

| TP | DP | EP | Expert/GPU | 非Expert/GPU | 权重合计 | +KV | |
|:---:|:---:|:---:|---:|---:|---:|---:|:---:|
| 8 | 16 | 128 | 40.6 | 32.5 | 73.1 | — | ❌ |
| 16 | 8 | 128 | 40.6 | 16.3 | **56.9** | ~64 | ⚠️ |
| 32 | 4 | 128 | 40.6 | 8.1 | **48.7** | ~56 | ✅ |
| 4 | 32 | 128 | 40.6 | 65.0 | 105.6 | — | ❌ |

> Expert=5200/EP=5200/128=40.6 GB 是定数（128 GPU 下 EP=DP×TP=128）。
> TP 决定非Expert：TP=8→32.5, TP=16→16.3, TP=32→8.1。
> TP=16 (6 heads/rank) 可行但 KV 紧。TP=32 (3 heads/rank) 充裕但需验证。

**256 GPU（ExternalDP=1）：**

| TP | DP | EP | Expert/GPU | 非Expert/GPU | 合计 | |
|:---:|:---:|:---:|---:|---:|---:|:---:|
| 16 | 16 | 256 | 20.3 | 16.3 | **36.6** | ✅ |
| 32 | 8 | 256 | 20.3 | 8.1 | **28.4** | ✅ |
| 8 | 32 | 256 | 20.3 | 32.5 | 52.8 | ✅ |

---

## 三、部署方案总览

### BF16 直载（RL 场景，实时权重）

| GPU | TP | DP | EP | 权重/GPU | +KV | |
|:---:|:---:|:---:|:---:|---:|---:|:---:|
| **128** | 16 | 8 | 128 | 56.9 GB | ~64 GB | ⚠️ KV 紧 |
| 128 | 32 | 4 | 128 | 48.7 GB | ~56 GB | ✅ |
| 192 | 24 | 8 | 192 | 37.9 GB | ~45 GB | ✅ |
| 256 | 16 | 16 | 256 | 36.6 GB | ~44 GB | ✅ |

### W4A8 官方模型（`Eco-Tech/Kimi-K3-w4a8`, 1467 GB）

| GPU | TP | DP | EP | 权重/GPU | +KV | |
|:---:|:---:|:---:|:---:|---:|---:|:---:|
| **64** | 8 | 8 | 64 | 32.9 GB | ~48 GB | ✅ 推荐 |
| 32 | 8 | 4 | 32 | 54.4 GB | ~69 GB | ⚠️ 临界 |
| 128 | 8 | 16 | 128 | 21.5 GB | ~37 GB | ✅ |

> W4A8 官方模型: Expert=1376 GB (INT4), 非Expert=91 GB (W8A8+FLOAT)
> Expert/GPU = 1376 / EP, 非Expert/GPU = 91 / TP

---

## 四、部署命令

### BF16 直载（RL，128 GPU, TP=16）

```bash
vllm serve <bf16-model-path> \
  --tensor-parallel-size 16 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --trust-remote-code \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85
```

### W4A8 生产（64 GPU）

```bash
vllm serve Eco-Tech/Kimi-K3-w4a8 \
  --tensor-parallel-size 8 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.85
```

---

## 五、RL 场景结论

```
训练 (MindSpeed-MM)     推理 (vLLM BF16)
  128 GPU ✅              128 GPU ✅ (TP=16 or TP=32)
  同一集群               同一集群
  实时权重               实时权重
  BF16 无损              BF16 无损

不再有 GPU 鸿沟——128 GPU 两边都能跑 BF16。
差距在 W4A8（64 GPU 就够了，但要 ModelSlim 量化有延迟）。
```

---

## 六、各平台汇总

| 场景 | 模型 | 最小 GPU | 推荐 GPU |
|---|---|---|---|
| **RL BF16 推理** | BF16 直载 | **128** (TP=16) | 192 (TP=24) |
| **生产 W4A8 推理** | `Eco-Tech/Kimi-K3-w4a8` | **64** (TP=8) | 128 |
| **A5 MXFP4 推理** | `moonshotai/Kimi-K3` | 待验证 | 待验证 |
| **310P** | — | ❌ | — |
