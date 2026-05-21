# Pi3 MUSA 待处理优化项

本文记录当前 Pi3 MUSA 路径中还没有完全主线化、但会影响性能或功能覆盖的主要待处理项。

## 1. cuRoPE2D / CroCo RoPE MUSA 编译算子

当前状态：

- `pi3/models/layers/pos_embed.py` 会尝试导入 `models.curope.cuRoPE2D`。
- 原始 CroCo RoPE 代码已经下载到 `/datapool/husicheng/Pi3/third_party/croco`。
- MUSA 适配版已经放在 `/datapool/husicheng/Pi3/third_party/croco_musa`，上游 commit 为 `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb`。
- 适配版已将 `CUDAExtension/nvcc/.cu/cuda runtime` 替换为 `MUSAExtension/mcc/.mu/musa runtime`，可在 MUSA 容器内编译出：

```text
/datapool/husicheng/Pi3/third_party/croco_musa/models/curope/curope.cpython-310-x86_64-linux-gnu.so
```

当前实际路径：

- 如果不设置 `PYTHONPATH`，Pi3 仍会走 `pos_embed.py` 内部的纯 PyTorch `RoPE2D` fallback。
- 设置 `PYTHONPATH=/datapool/husicheng/Pi3/third_party/croco_musa` 后，`pi3.models.layers.pos_embed.RoPE2D` 可以解析为 `models.curope.curope2d.cuRoPE2D`。
- MUSA cuRoPE 适配版已通过 forward/backward smoke；float32 前向相对 PyTorch fallback 最大误差 `4.768e-07`，bfloat16 最大误差 `4.688e-02`、平均误差 `1.855e-03`。
- `(B,H,N,D)=(1,16,3136,64)` 下，cached PyTorch fallback 约 `0.44-0.45 ms`，MUSA cuRoPE in-place kernel 约 `0.06-0.09 ms`。

后续处理方向：

- 决定是否将 `third_party/croco_musa/models/curope` 正式接入仓库或安装流程。
- 如果正式接入，需要补充构建脚本、运行入口的 `PYTHONPATH`/安装步骤，以及 CI/smoke 验证。
- bfloat16 误差来自低精度路径，需要结合 Pi3/Pi3X 端到端输出确认是否接受。

## 2. xFormers 缺失

当前状态：

当前 MUSA `.venv` 中仍然没有 `xformers`：

```text
xformers: MISSING
local_attention_xformers False
local_block_xformers False
dinov2_attention_xformers False
dinov2_block_xformers False
dinov2_swiglu_xformers False
```

官方资料结论：

- xFormers 官方 README 当前只给出 CUDA 12.6/12.8/13.0 wheel、实验 ROCm 7.1 wheel，以及从源码构建 CUDA/ROCm 的路径；没有 MUSA/PrivateUse1 wheel 或直接安装入口。
- xFormers 源码构建逻辑基于 `CUDAExtension`/`ROCM_HOME`/`CUDA_HOME`，官方 `fmha` 注册也主要面向 CUDA dispatch；没有发现官方 MUSA backend。
- torch_musa 官方文档给出的方向是通用第三方库 MUSA 扩展：使用 `MUSAExtension`、CUDA-Porting、`PrivateUse1`、`.cu -> .mu` 等方式迁移第三方 CUDA extension。
- 摩尔线程官方支持/已 musify 的第三方仓库列表里没有 xFormers；`torchada` 可以在运行时/构建期映射 CUDA API，但不是 xFormers 的专门官方适配包。

参考来源：

- xFormers 官方仓库安装说明：<https://github.com/facebookresearch/xformers#installing-xformers>
- xFormers 官方 `setup.py`：<https://github.com/facebookresearch/xformers/blob/main/setup.py>
- torch_musa 第三方库扩展官方文档：<https://docs.mthreads.com/torchmusa/torchmusa-doc-online/third_party_lib_extension>
- torchada 官方介绍：<https://blog.mthreads.com/blog/musa/2026-02-25-torchada/>

涉及代码：

- `pi3/models/layers/attention.py`
- `pi3/models/layers/block.py`
- `pi3/models/dinov2/layers/attention.py`
- `pi3/models/dinov2/layers/block.py`
- `pi3/models/dinov2/layers/swiglu_ffn.py`

当前实际路径：

- 普通 tensor attention 会退回 PyTorch attention/MLP 实现，当前 Pi3/Pi3X 推理能跑。
- 已增加本地 PyTorch/MUSA fallback：缺少 xFormers 时，`NestedTensorBlock` 的 list/nested tensor 输入不再直接报 `AssertionError`，而是使用本地 `TorchBlockDiagonalMask`、block-wise attention、`index_select_cat` 和 `scaled_index_add` 等价路径。
- 当前主推理路径主要使用 PyTorch `scaled_dot_product_attention`，MUSA 上 bf16 SDPA 已实测可用。
- `SwiGLUFFNFused` 在缺少 xFormers 时仍走已有 PyTorch `SwiGLUFFN` fallback。

后续处理方向：

- 当前处理目标是 correctness fallback，不是 fused 性能替代；大规模 nested tensor 训练仍可能比 xFormers 慢。
- 如果未来训练强依赖 nested tensor 的吞吐，需要单独评估 MUSA fused/block-diagonal attention kernel。
- 对 `SwiGLU` fused 路径可单独评估是否值得做 MUSA fused 实现；当前 PyTorch fallback 可用但可能较慢。

## 当前不是待处理阻塞项

- MUSA `torch.svd` 缺失已通过 polar12 fast path + 少量 CPU-SVD fallback 处理。
- MUSA `torch.det`、`torch.inverse`、`torch.linalg.inv` 当前实测可用。
- PyTorch `scaled_dot_product_attention` bf16 当前实测可用。
