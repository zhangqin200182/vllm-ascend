# Kimi K3 RL 场景：训练推理统一集群分析

> 更新时间: 2026-07-28
> 状态: 已解决 — 128 GPU 可同时跑训练和 BF16 推理

---

## 结论

```
训练 (MindSpeed-MM, LoRA)     推理 (vLLM, BF16 直载)
  128 GPU ✅                      128 GPU ✅
  FSDP2 + EP=128                  TP=16 + DP=8 + EP=128
  Expert: 5200/128 = 40.6 GB      Expert: 5200/128 = 40.6 GB
  非Expert: 260/128 = 2.0 GB      非Expert: 260/16 = 16.3 GB
  优化器: ~3 GB (LoRA)            KV cache: ~7 GB
  ─────────────────────           ──────────────────────
  合计: ~60.6 GB ✅               合计: ~64 GB ⚠️

或 TP=32, DP=4:
  Expert: 40.6 GB
  非Expert: 260/32 = 8.1 GB
  合计: ~49 GB + KV(15 GB) = ~64 GB ✅
```

## 关键认知修正

**之前错误**: vLLM 的 EP 只能通过 TP 来分片 Expert，TP 和非Expert 共用同一个分片因子，导致 TP↑ = 非Expert 不够分、TP↓ = EP 太小 Expert 分不够。

**实际拓扑**: vLLM `EP = DP × TP`。EP 独立于 Attention 的 TP。DP 被 "展平" 进 EP（`flatten_tp_across_dp_and_pcp`），Expert 按 DP×TP 分片，而 Attention 仍只走 `get_tp_group()` → 仅按 TP 分片。

```
vLLM 拓扑: ExternalDP × DP × PP × PCP × TP

  Expert 分片: EP = DP × PCP × TP（FusedMoE 内部 flatten）
  Attention 分片: TP（独立，get_tp_group()）
  非Expert 复制: DP（每个 DP replica 一份）

128 GPU = ExternalDP=1 × DP=8 × PP=1 × PCP=1 × TP=16
  EP = 8 × 1 × 16 = 128 → Expert = 5200/128 = 40.6 GB ✓
  TP = 16 → 非Expert = 260/16 = 16.3 GB ✓
  合计: 56.9 GB + KV
```

## 来源

- `vllm/distributed/parallel_state.py:1894-1902` — EP group = DP × PCP × TP
- `vllm/model_executor/layers/fused_moe/config.py:1113-1121` — flatten_tp_across_dp_and_pcp
