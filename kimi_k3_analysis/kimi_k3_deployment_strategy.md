# Kimi K3 部署并行策略与资源估算

> 分析日期: 2026-07-28
> 数据来源: PR #12951、`KIMI_K3_SITU_QUANT_CONTRACT.md`、模型配置

---

## 一、模型规格

| 参数 | 数值 |
|---:|---:|
| Hidden Size | 7168 |
| Routed Expert Hidden Size | 3584 |
| Routed Expert Intermediate Size | 3072 |
| Routed Experts 总数 | 896 |
| 每 Token 激活 Expert 数 (Top-K) | 16 |
| Shared Experts | 2 |
| Shared Intermediate Size | 6144 (2×3072) |
| KDA Head Dim (K/V) | 128 |
| SiTU Beta / Linear Beta | 4.0 / 25.0 |
| 输入精度 | BF16 |
| 推荐权重量化 | W4A8 (INT4 权重 + INT8/MXFP8 激活) |

> **注意**: HuggingFace 上发布的权重为 W4A8 量化版本（`Kimi-K3-Instruct-W4A8` 或类似命名），直接可用于部署。BF16 原始权重由于模型规模过大（总参数量估算 ~800B+），单卡/单节点无法部署。

---

## 二、显存估算

### 2.1 W4A8 量化下权重占用（per-expert）

| 权重矩阵 | Shape | INT4 大小 |
|---|---|---|
| `gate_up_proj` | `[3584, 6144]` | ~11.0 MB |
| `down_proj` | `[3072, 3584]` | ~5.5 MB |
| **每个 Expert 合计** | | **~16.5 MB** |

### 2.2 无 EP 场景（全部 896 Expert 权重）

| 组件 | INT4 权重 |
|---|---|
| 896 Routed Experts | 896 × 16.5 MB ≈ 14.8 GB |
| 2 Shared Experts | ~33 MB |
| Attention 层 (KDA + MLA) × N | ~2-4 GB (取决于层数) |
| Embedding + LM Head | ~0.5-1 GB |
| Vision Tower (多模态) | ~0.5 GB |
| **总权重大小** | **~18-22 GB** |

### 2.3 KV Cache / KDA State 占用

| 组件 | 每 Token 占用 | 典型配置 | 总占用 |
|---|---|---|---|
| MLA KV Cache | ~0.02-0.05 MB/token | 8K context × 64 seqs | ~10-25 GB |
| KDA Recurrent State | `[H, 128, 128]` × state_capacity | 1024 slots × 16 heads | ~0.01 GB (可忽略) |

### 2.4 单卡可行性（Atlas 800T A2, 64 GB HBM）

| 组件 | 占用 |
|---|---|
| 权重 (W4A8, TP=8 切分) | ~2.5-3.5 GB |
| KV Cache (8K × 32 seqs) | ~6-12 GB |
| 激活值 + 临时显存 | ~5-10 GB |
| CUDA Graph 缓存 | ~2-5 GB |
| **合计** | **~15-30 GB** |

> ✅ **单节点 (8×A2 64GB) 可部署 W4A8 量化版 Kimi K3（文本模式）。**

---

## 三、推荐并行策略

### 策略 A：单节点（最小资源）

```
硬件: 1 节点 × 8 × Atlas 800T A2 (64 GB) 或 1 节点 × 8 × Atlas 800T A3 (64 GB)
总卡数: 8
```

| 并行维度 | 配置 | 说明 |
|---|---|---|
| **TP** | 8 | 按 attention head + expert 内部切分 |
| **EP** | 不开启 | 896 个 expert 权重全量在每张卡上 |
| **DP** | 不开启 | 单节点内不需要 |

```bash
vllm serve <model-path> \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.9
```

| 限制项 | 值 |
|---|---|
| 最大 Context 长度 | ~8K tokens (取决于并发请求数) |
| 最大并发请求数 | ~32 (取决于 context 长度) |
| 适用场景 | 文本对话、短文档处理、单轮 QA |

### 策略 B：双节点 DP（推荐生产环境）

```
硬件: 2 节点 × 8 × Atlas 800T A2/A3 (64 GB)
总卡数: 16
```

| 并行维度 | 配置 | 说明 |
|---|---|---|
| **TP** | 8 | 单节点内 expert 内部切分 |
| **DP** | 2 (跨节点) | 每节点独立服务请求，吞吐翻倍 |
| **EP** | 不开启 | |
| **DP Local** | 1 | |

```bash
# 节点 0 (主)
vllm serve <model-path> \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --data-parallel-address $LOCAL_IP \
  --data-parallel-rpc-port 13389 \
  --quantization ascend \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.9

# 节点 1 (从)
vllm serve <model-path> \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 1 \
  --data-parallel-address $MASTER_IP \
  --data-parallel-rpc-port 13389 \
  --quantization ascend \
  --max-model-len 32768 \
  --headless \
  --gpu-memory-utilization 0.9
```

> 参考：同仓库 Kimi-K2.5 部署配置 `tests/e2e/nightly/multi_node/internal_dp/config/Kimi-K2_5-W4A8-A2-dual-nodes.yaml`

### 策略 C：多节点（EP 满配，最大吞吐）

```
硬件: 8 节点 × 8 × Atlas 800T A2/A3 (64 GB)
总卡数: 64
```

| 并行维度 | 配置 | 说明 |
|---|---|---|
| **TP** | 8 | 单节点内 |
| **EP** | 8 | 896/8=112 experts/rank，大幅降低单卡内存 |
| **DP** | 1-8 | 取决于节点数 |

```bash
vllm serve <model-path> \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --expert-parallel-size 8 \
  --quantization ascend \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.9
```

| 优势 | 说明 |
|---|---|
| 支持长 Context | 可达 128K tokens |
| 高并发 | 128+ 序列 |
| 单卡显存压力小 | 每 rank 仅 ~112 experts (TP 切分后) |

> **注意**: 策略 C 需要额外验证 Kimi K3 的 EP 算子（`dispatch_ffn_combine_*` 等）在目标平台上的可用性。

### 策略 D：PD 分离 + Mooncake（最低延迟）

```
硬件: P 节点 + D 节点，具体数量待验证
```

| 角色 | 负责 | 最小卡数 |
|---|---|---|
| **Prefill (P)** | 长序列 prefill，chunk 并行 | 6-8 卡/节点 × N 节点 |
| **Decode (D)** | 逐 token decode，recurrent 状态更新 | 2-4 卡/节点 × M 节点 |

> **限制**: Kimi K3 当前暂不支持 PCP (`kimi_kda.py` line 317 有显式限制)，PD 分离策略需等待后续 PR 支持。

---

## 四、各平台适用策略

| 平台 | 推荐策略 | 单卡最小 | 说明 |
|---|---|---|---|
| **A2 (910B, 64 GB)** | 策略 A 或 B | 8 卡 (1 节点) | ✅ 已验证可部署 |
| **A2 (910B, 32 GB)** | 策略 C (EP=16+) | 待验证 | 32 GB 卡需要 EP 分担 expert 权重 |
| **A3 (910_93, 64 GB)** | 策略 A 或 B + ModelSlim | 8 卡 (1 节点) | A3 有额外 MoE dispatch 融合算子 |
| **A5 (950)** | 策略 A（situ_mx_quant） | 待验证 | A5 使用 MX FP8 量化，算子集不同 |
| **310P** | ❌ 不支持 | — | 缺少 `recurrent_kda` 和 SiTU 量化算子 |

---

## 五、环境变量

```bash
# 通用
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=512

# Kimi K3 专用
export VLLM_ASCEND_ENABLE_MLAPO=1          # MLA 优化
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1     # Flash Communication

# 量化（W4A8 权重必须）
--quantization ascend
```

---

## 六、总结

| 场景 | 最小硬件 | 最大 Context | 最大并发 |
|---|---|---|---|
| **开发/测试** | 1 节点 8×A2/A3 (64 GB) | 8K | 32 |
| **生产（标准）** | 2 节点 16×A2/A3 (64 GB，DP=2) | 32K | 64 |
| **生产（高吞吐）** | 8 节点 64×A2/A3 (64 GB，EP=8) | 128K | 128+ |
| **训练** | （不在本文档范围内，MindSpeed-MM 项目负责） | — | — |

> **当前 PR 状态**: `feature/kimi-k3-release-v0.23.0` 为 OPEN 状态，以上部署策略基于代码分析，实际可用性需待 PR 合入并发布版本后验证。
