# Kimi K3 部署路径对比分析

> 分析日期: 2026-07-28
> 代码来源: `vllm_ascend/quantization/methods/w4a8_mxfp4.py` + `w4a8.py`

---

## 一、两条部署路径

vLLM-Ascend 对 Kimi K3 提供两条部署路径，分别对应不同的硬件平台：

```
HF/ModelScope: moonshotai/Kimi-K3 (1560 GB, Expert MXFP4 + Non-Expert BF16)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    路径 1: 不转换                         路径 2: ModelSlim 转换
    MXFP4 → FP8                            MXFP4 → INT4
    (A5 专用)                              (A2/A3 通用)
```

---

## 二、路径 1：MXFP4 → FP8（A5，不转换）

### 2.1 工作流程

```
HF Checkpoint (MXFP4-packed, uint8, 0.5B/param)
    │
    ▼ 加载
torch.uint8 tensor in HBM (1376 GB Expert + 184 GB Non-Expert)
    │
    ▼ process_weights_after_loading: npu_format_cast(float4→float8)
    │   convert FP4 packed → FP8 native format
    │
FP8 tensor in HBM: 2752 GB Expert (1B/param) + 184 GB Non-Expert (BF16)
    │
    ▼ 推理: situ_mx_quant (SiTU→MXFP8), GMM with FP8 weights
    │
   A5 输出
```

### 2.2 关键代码

```python
# w4a8_mxfp4.py line 262-264
w13_weight.data = torch_npu.npu_format_cast(
    w13_weight.data, 29,
    customize_dtype=torch.float8_e4m3fn,          # target: FP8
    input_dtype=torch_npu.float4_e2m1fn_x2         # source: FP4
)
```

### 2.3 HBM 占用 (A5)

FP8 = 1 byte/element，是原 FP4-packed 的 2 倍。

| 组件 | 磁盘 (MXFP4) | HBM (FP8 转换后) |
|---|---|---|
| Expert | 1376 GB | **2752 GB** |
| 非Expert | 184 GB (BF16) | 184 GB (BF16) |
| **合计** | 1560 GB | **2936 GB** |

### 2.4 GPU 估算 (A5)

| TP | EP | GPU | Expert FP8/GPU | 非Expert BF16/GPU | 权重合计 | 可行性 |
|:---:|:---:|:---:|---:|---:|---:|:---:|
| 8 | 16 | 128 | 21.5 GB | 23.0 GB | **44.5 GB** | 取决于 A5 HBM |
| 8 | 8 | 64 | 43.0 GB | 23.0 GB | **66.0 GB** | 取决于 A5 HBM |
| 8 | — | 8 | 344 GB | 23.0 GB | **367 GB** | ❌ |

### 2.5 特点

| 优点 | 缺点 |
|---|---|
| 无需离线转换，直接加载 HF 权重 | 仅 A5 可用 |
| 保留 MXFP4 训练精度 | FP8 HBM 膨胀 2× |
| 原生 FP8 计算 | A2/A3 不支持 `npu_format_cast(FP4→FP8)` |

---

## 三、路径 2：MXFP4 → INT4（A2/A3，ModelSlim 转换）

### 3.1 工作流程

```
HF Checkpoint (MXFP4-packed, uint8, 0.5B/param)
    │
    ▼ ModelSlim 离线工具
    │   FP4→INT4 格式转换 + rotation matrix
    │
ModelSlim 转换后的 checkpoint (INT4-packed, uint8, 0.5B/param)
    │
    ▼ 加载
torch.uint8 tensor in HBM: 1376 GB Expert + 184 GB Non-Expert
    │   ↑ 大小与原始 HF checkpoint 相同（都是 0.5B/param 4-bit packed）
    │
    ▼ 推理: dequant_situ_quant (SiTU→INT8), GMM with INT4→BF16 dequant
    │
   A2/A3 输出
```

### 3.2 关键代码

```python
# w4a8.py line 667-671
def process_weights_after_loading(self, layer):
    if self.is_compressed_tensors_format:
        self.process_weights_after_loading_compressed_tensors(layer)
    else:
        self.process_weights_after_loading_modelslim(layer)
```

### 3.3 HBM 占用 (A2/A3)

INT4 = 0.5 byte/element，与 MXFP4-packed 相同，**无需膨胀**。

| 组件 | 磁盘 (MXFP4) | HBM (INT4 转换后) |
|---|---|---|
| Expert | 1376 GB | **1376 GB** |
| 非Expert | 184 GB (BF16) | 184 GB (BF16) |
| **合计** | 1560 GB | **1560 GB** |

### 3.4 GPU 估算 (A2/A3, 64 GB)

| TP | EP | GPU | Expert INT4/GPU | 非Expert BF16/GPU | 权重合计 | +KV | 总占用 | 可行性 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| 8 | 16 | 128 | 10.8 GB | 23.0 GB | **33.8 GB** | ~15 GB | ~49 GB | ✅ |
| 8 | 8 | 64 | 21.5 GB | 23.0 GB | **44.5 GB** | ~15 GB | ~60 GB | ✅ |
| 8 | 4 | 32 | 43.0 GB | 23.0 GB | **66.0 GB** | — | — | ❌ |

### 3.5 特点

| 优点 | 缺点 |
|---|---|
| A2/A3/A5 全平台兼容 | 需要 ModelSlim 离线转换步骤 |
| INT4 紧凑存储，HBM 不膨胀 | FP4→INT4 存在精度损失 |
| 64 卡可部署 | 转换后的 checkpoint 需要额外存储空间 |

---

## 四、路径对比总览

| | 路径 1: MXFP4→FP8 | 路径 2: ModelSlim INT4 |
|---|---|---|
| **平台** | A5 专用 | A2 / A3 / A5 |
| **离线转换** | 不需要 | ModelSlim (FP4→INT4) |
| **Expert HBM 格式** | **FP8** (1 B/elem) | **INT4** (0.5 B/elem) |
| **Expert HBM 大小** | **2752 GB** | **1376 GB** |
| **非Expert HBM 大小** | 184 GB | 184 GB |
| **合计 HBM** | **2936 GB** | **1560 GB** |
| **TP=8 EP=16 每GPU** | 44.5 GB | 33.8 GB |
| **TP=8 EP=8 每GPU** | 66.0 GB | 44.5 GB |
| **最小 GPU (A2 64GB)** | ❌ 不支持 | **64** (TP=8, EP=8) |
| **SiTU 激活量化** | `situ_mx_quant` (MXFP8) | `dequant_situ_quant` (INT8) |
| **GMM 精度路径** | FP8 native | INT4→BF16 dequant |
| **精度** | 训练原生 FP4→FP8 | FP4→INT4 有损转换 |
| **部署前准备** | `vllm serve moonshotai/Kimi-K3` | ModelSlim 转换 → 部署转换后权重 |

---

## 五、路径选择决策树

```
目标硬件是 A5?
  ├── 是 → 路径 1 (MXFP4→FP8, 直载 HF 权重, situ_mx_quant)
  │        128 GPU (TP=8, EP=16) 推荐
  │        需要确认 A5 HBM 容量
  │
  └── 否 (A2/A3)
        └── 路径 2 (ModelSlim INT4, 离线转换, dequant_situ_quant)
              64 GPU (TP=8, EP=8)  临界可行
             128 GPU (TP=8, EP=16) 推荐生产

310P?
  └── ❌ 两条路径均不可用 (缺少 recurrent_kda + SiTU 量化算子)
```

---

## 六、总结

1. **A2/A3 必须用 ModelSlim 转换**——不能直接加载 HF 原始权重。直接加载会走 MXFP4→FP8 路径，但 A2/A3 不支持 `npu_format_cast(FP4→FP8)`。

2. **转换后 HBM 大小不变**——INT4 和 MXFP4 都是 `uint8` 容器、`0.5 bytes/param`。只是 4-bit 的编码方式从 float4 变成了 integer4。

3. **FP8 路径 HBM 膨胀 2×**——即使 A5 走路径 1，`npu_format_cast` 也会把 FP4 展开为 FP8 (1 byte/elem)，Expert HBM 从 1376 GB 变成 2752 GB。

4. **两种路径的 GPU 数差距有限**——路径 1 虽然 Expert 大 2×，但在 TP=8 EP=16 下每 GPU 只多 ~11 GB (44.5 vs 33.8)，64 卡的差距并不悬殊。
