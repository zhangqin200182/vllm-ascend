# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: ModelScope `moonshotai/Kimi-K3` config.json + PR #12951 + `KIMI_K3_SITU_QUANT_CONTRACT.md`

---

## 一、模型规格（基于真实 config.json）

### 1.1 核心超参

| 参数 | 数值 |
|---:|---|
| `num_hidden_layers` | **93** |
| `hidden_size` | 7168 |
| `intermediate_size` (MLP) | 33792 |
| `moe_intermediate_size` | 3072 |
| `routed_expert_hidden_size` | 3584 |
| `num_attention_heads` | **96** |
| `num_key_value_heads` | 96 (GQA ratio 1:1) |
| `vocab_size` | 163840 |
| `q_lora_rank` | 1536 |
| `kv_lora_rank` | 512 |
| `qk_nope_head_dim` | 128 |
| `qk_rope_head_dim` | 64 |
| `v_head_dim` | 128 |
| `max_position_embeddings` | 1048576 (1M) |
| `dtype` | bfloat16 |
| `hidden_act` | situ |

### 1.2 MoE 配置

| 参数 | 数值 |
|---:|---|
| `num_experts` | 896（**每层独立**，非跨层共享） |
| `num_experts_per_token` | 16 |
| `num_shared_experts` | 2 |
| `moe_layer_freq` | 1（每层都是 MoE） |
| `moe_router_activation_func` | sigmoid |
| `moe_renormalize` | True |
| `topk_method` | noaux_tc |
| `routed_scaling_factor` | 1.0 |

### 1.3 Attention 配置

| 参数 | 数值 |
|---:|---|
| `linear_attn_config.kda_layers` | **68 层** KDA（共 93 层） |
| `linear_attn_config.full_attn_layers` | **24 层** MLA（全注意力层） |
| `linear_attn_config.num_heads` | 96 |
| `linear_attn_config.head_dim` | 128 |
| `linear_attn_config.gate_lower_bound` | **-5.0**（confirmed!） |
| `linear_attn_config.short_conv_kernel_size` | 4 |
| `linear_attn_config.use_full_rank_gate` | True |
| `attn_res_block_size` | 12 |
| `mla_use_nope` | True |
| `mla_use_output_gate` | True |

### 1.4 量化配置（ModelScope 发布版）

```
format: mxfp4-pack-quantized (compressed-tensors)
num_bits: 4 (weights only)
group_size: 32
symmetric: True
忽略层: self_attn.*, shared_experts.*, mlp.(gate|up|gate_up|down)_proj.*, lm_head.*
```

> `lm_head`、`shared_experts`、`self_attn` 等层保持 BF16 精度。

### 1.5 模型权重大小

**ModelScope (`moonshotai/Kimi-K3`) 发布版 — MXFP4 量化后：**

| 组件 | 文件数 | 说明 |
|---|---|---|
| 96 个 safetensors 分片 | 96 | 每个 0.09 ~ 16.99 GB |
| 总权重大小 | **~1.56 TB** | BF16 等效存储（safetensors 存储 MXFP4-packed 数据） |

> 实际 INT4 有效权重约 390-400 GB；safetensors 中 uint8 压缩存储 + scale 开销导致文件总大小高于有效参数数。

---

## 二、显存估算（修正后）

### 2.1 逐层权重分解（BF16 原始）

| 组件 | 每层占用 | 合计 (93 层) |
|---|---|---|
| **KDA Attention** (68 层，每层) | ~800 MB | ~54.4 GB |
| Q/K/V 投影 | 88M × 3 = 264M params | |
| Gate 投影 (f_a, f_b, b, g_a, g_b, o_proj) | ~200M params | |
| Conv1D 权重 (kernel=4) | 极小 | |
| **MLA Attention** (24 层，每层) | ~288 MB | ~6.9 GB |
| q_a, q_b, kv_a, kv_b, o_proj | ~144M params | |
| **MoE 专家** (93 层，每层 896 experts) | ~14.8 GB | **~1376 GB** |
| 单个 expert: gate_up + down | ~66 MB (BF16) | |
| **Router + 投影** (93 层) | ~130 MB | ~12.1 GB |
| `routed_expert_down_proj` | 7168×3584 = 25.7M | |
| `routed_expert_up_proj` | 3584×7168 = 25.7M | |
| Router | 7168×896 = 6.4M | |
| **Embedding + LM Head + Vision** | — | ~6 GB |
| **其他 (RMSNorm, 共享专家…)** | — | ~10 GB |
| **总计（BF16）** | | **~1.47 TB** ≈ 1.56 TB ✓ |

### 2.2 W4A8 / MXFP4 量化后

| 组件 | 量化后占用 |
|---|---|
| 量化层的 INT4 权重 (896 experts × 93 layers) | ~344 GB |
| 非量化层 (self_attn, lm_head, shared_experts, mlp) BF16 | ~46 GB |
| **总权重** | **~390 GB** |

### 2.3 单卡可行性分析

**以 Atlas 800T A2 (64 GB HBM) 为例：**

```
                   TP=8 (单节点)          TP=8 + EP=8 (64卡)      TP=8 + EP=64 (512卡)
                   ─────────────          ─────────────────      ────────────────────
量化权重 / 卡       390/8 = 48.8 GB       Expert: 344/8/8=5.4     Expert: 344/8/64=0.7
                                         Attn: 46/8=5.8          Attn: 46/8=5.8
                                         Total: ~11.2 GB         Total: ~6.5 GB
KV Cache            5-10 GB              15-30 GB                30-50 GB
激活 + 临时         2-5 GB               3-5 GB                  3-5 GB
─────────────────────────────────────────────────────────────────────────────────
合计                ~56-64 GB            ~30-46 GB               ~40-62 GB
可行性              ⚠️ 临界               ✅ 舒适                  ✅ 舒适
```

> **结论：单节点 8×A2 (64GB) 部署 W4A8 Kimi K3 勉强可行但非常临界**。需严格控制 KV cache 和并发数，推荐开启 EP 以降低单卡内存压力。

---

## 三、推荐并行策略

### 策略 A：单节点 TP=8（最小资源，临界）

```
硬件: 1 节点 × 8 × Atlas 800T A2/A3 (64 GB)
```

| 并行维度 | 数值 | 说明 |
|---|---|---|
| TP | 8 | attention head 96 维正好被 8 整除 |
| EP | 不开启 | 896 experts 全部在每张卡上 |
| DP | 不开启 | |

```bash
vllm serve <w4a8-model-path> \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

| 限制 | 说明 |
|---|---|
| Context 长度 | ≤ 4K（KV cache 严格控制） |
| 并发数 | ≤ 16 |
| 适用场景 | 开发测试、短文本 QA |

### 策略 B：TP=8 + EP=8（推荐最低生产配置）

```
硬件: 8 节点 × 8 × Atlas 800T A2/A3 (64 GB)
总卡数: 64
```

| 并行维度 | 数值 | 说明 |
|---|---|---|
| TP | 8 | 节点内 attention head 切分 |
| EP | 8 | 896/8 = 112 experts/rank |
| DP | 可叠加 | 视节点数而定 |

```bash
vllm serve <w4a8-model-path> \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --quantization ascend \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9
```

| 优势 | 说明 |
|---|---|
| Context 长度 | 可达 32K |
| 并发数 | 64+ |
| 单卡内存 | ~30-46 GB（舒适） |

### 策略 C：纯 DP 扩展（吞吐优先）

```
硬件: 2×N 节点，TP=8，EP=8 per 8 nodes
在每个 EP 组上叠加 DP
```

```bash
# 8 节点 EP 组内
--data-parallel-size 2 \
--data-parallel-size-local 1
```

| 优势 | 说明 |
|---|---|
| 线性扩展吞吐 | 每增加 8 节点翻倍 |

### 策略 D：EP=64（完整 Expert 并行，最低单卡内存）

```
硬件: 64 节点 × 8 × Atlas 800T A2/A3 (64 GB)
总卡数: 512（但可搭配 TP 减少）
```

| 并行维度 | 数值 | 说明 |
|---|---|---|
| TP | 1-8 | 可用较少 TP 换取更低的单卡内存 |
| EP | 64 | 896/64 = 14 experts/rank |

> 512 卡是理论最大值（TP=1, EP=64, 64×8=512）。实际可调整 TP/EP/DP 组合。

---

## 四、训推精度对齐更新

### 关键发现：训练和推理使用的参数一致！

从 HuggingFace config.json 证实：

| 参数 | 训练 (MindSpeed-MM) | 推理 (vLLM-Ascend) | 对齐? |
|---|---|---|---|
| `gate_lower_bound` | **-5.0**（config 中声明） | **-5.0**（代码硬编码） | ✅ |
| `activation_situ_beta` | **4.0** | **4.0** | ✅ |
| `activation_situ_linear_beta` | **25.0** | **25.0** | ✅ |
| `safe_gate` | True | True | ✅ |

> 此前的「训推精度对齐分析」文档中关于 `gate_lower_bound` 未对齐的担忧已消除 — 训练 config 中确实声明了 `gate_lower_bound: -5.0`，训练和推理使用相同的 bounded sigmoid gate 公式。

### 仍需关注的对齐点

1. **SiTU 融合 kernel 内部精度** (`dequant_situ_quant` vs 训练侧 fp32 `SituAndMul`)
2. **KDA Chunk 算子** (`triton_ascend_kernels` vs AscendC `chunk_kda_fwd` 的中间值差异)
3. **量化误差** (MXFP4 权重 + INT8 激活 vs BF16 训练)

---

## 五、各平台适用性

| 平台 | 推荐策略 | 最小卡数 | 说明 |
|---|---|---|---|
| **A2 (910B, 64 GB)** | 策略 B (TP=8+EP=8) | **64 卡** (8 节点) | 单节点 8 卡为临界配置 |
| **A3 (910_93, 64 GB)** | 策略 B + A3 dispatch 融合 | 64 卡 | A3 有额外 MoE dispatch 融合算子 |
| **A5 (950)** | 策略 B (situ_mx_quant) | 待验证 | 使用 MX FP8 量化 |
| **310P** | ❌ 不支持 | — | 缺少 `recurrent_kda` 和 SiTU 量化算子 |

---

## 六、参考命令

### 环境变量

```bash
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=512
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

### 最小测试启动命令

```bash
# 单节点，TP=8，极简配置（仅验证功能可用）
vllm serve <w4a8-model-path> \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --trust-remote-code \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75 \
  --disable-custom-all-reduce
```

---

## 七、总结

| 场景 | 最小硬件 | 最大 Context | 并行方案 |
|---|---|---|---|
| **功能验证** | 1 节点 8×A2/64GB | 2K | TP=8, 极保守配置 |
| **开发测试** | 1 节点 8×A2/64GB | 4K | TP=8, 临界运行 |
| **生产标准** | 8 节点 64×A2/64GB | 32K | TP=8 + EP=8 |
| **大吞吐** | 16+ 节点 | 128K | TP=8 + EP=8 + DP=N |
| **训练** | MindSpeed-MM 负责 | — | FSDP2 + EP + CP + SP |

> **关键修正**：此前估计单节点 8×A2 可「舒适运行」是错误的。BF16 原始模型 1.56 TB、W4A8 量化后 ~390 GB 权重，TP=8 下每卡权重约 49 GB，加上 KV cache 后几乎填满 64 GB。**强烈建议生产环境开启 EP**。
