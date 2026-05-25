# Pi3 MUSA 优化状态

本文只记录当前 MUSA 推理路径还需要关注的事项。当前目标以 Pi3/Pi3X
推理为主，不覆盖训练优化。

## 当前结论

- Pi3/Pi3X MUSA 推理已经可用。
- cuRoPE MUSA 已默认接入，并已修复 `tokens are not contiguous` 问题。
- xFormers 缺失不阻塞当前推理路径；EPIC-KITCHENS P01 profiling 显示
  nested/list 与 block-diagonal attention 占比为 0。
- 当前不需要继续投入完整 xFormers MUSA 适配。只有新的推理 workload 真的触发
  nested/list 或 block-diagonal attention 时，再重新评估。

## cuRoPE2D

`scripts/run_musa.sh` 默认启用 `PI3_ENABLE_CROCO_MUSA=1`，并将
`third_party/croco_musa` 加入 `PYTHONPATH`。正常情况下
`pi3.models.layers.pos_embed.RoPE2D` 会解析为 MUSA 版
`models.curope.curope2d.cuRoPE2D`。缺少 `.so` 或 extension 源码比 `.so` 更新时，
脚本会自动重新构建 cuRoPE。

已完成：

- CroCo RoPE 已迁移到 `MUSAExtension/mcc/.mu`，可编译生成
  `third_party/croco_musa/models/curope/curope*.so`。
- 2026-05-25 修复 `tokens are not contiguous`：MUSA kernel 只要求最后一维
  contiguous，不再错误拒绝 `(B,H,N,D).transpose(1,2)` 得到的 strided view；
  wrapper 同时保证 `positions.contiguous()`。
- qk_norm 后的 contiguous `(B,H,N,D)=(1,16,3136,64)` smoke 已通过。
- 与 PyTorch RoPE fallback 对齐：
  - float32 最大误差 `4.768e-07`，平均误差 `1.252e-08`
  - bfloat16 最大误差 `2.266e-01`，平均误差 `4.849e-03`
- 已用 cuRoPE MUSA 重跑 EPIC-KITCHENS P01 的 `P01_01/P01_02/P01_03`。
  与 PyTorch RoPE fallback 基线的 `frame_indices` 和 `valid_mask` 完全一致。
  对齐报告：
  `outputs/curope_musa/EPIC-KITCHENS/P01/interval_300/comparison_to_xformers_fallback.json`

保留的维护项：

- 如需减少首次启动耗时，可把 `run_musa.sh` 中的首次自动构建拆成显式 build/install
  命令。
- 后续如果更换 RoPE kernel 或精度策略，继续用已保存的 P01 baseline 回归。

## xFormers

当前 MUSA `.venv` 中没有 `xformers`。项目已提供本地 PyTorch/MUSA fallback：

- 普通 dense attention 走现有 PyTorch/MUSA 路径。
- `NestedTensorBlock` 的 list/nested 输入可走 `TorchBlockDiagonalMask`、
  block-wise attention、`index_select_cat`、`scaled_index_add` fallback。
- `SwiGLUFFNFused` 缺失时回退到 PyTorch `SwiGLUFFN`。

P01/Pi3XVO 推理 profiling 结果：

- 报告路径：
  `outputs/attention_profile/EPIC-KITCHENS/P01/interval_300/attention_profile.json`
- 输入：EPIC-KITCHENS P01 的 `P01_01/P01_02/P01_03`
- `chunk_count=45`
- `sampled_frame_count_with_overlap=708`
- `attention_calls=2295`
- 全部 attention 都是 `local.FlashAttentionRope`，且 `attn_bias=None`
- `block_diagonal_attention_calls=0`
- `block_diagonal_work_ratio=0.0`
- `nested_tensor_list_calls=0`
- `index_select_cat_calls=0`
- `scaled_index_add_calls=0`

结论：

- 当前推理路径没有触发 xFormers 的核心缺失路径。
- 不做完整 xFormers MUSA fork。
- 暂不实现 `BlockDiagonalMask + memory_efficient_attention` 的 MUSA fast path。
- 暂不实现 `index_select_cat`、`scaled_index_add`、`SwiGLU fused` 的 MUSA fused 版本。

重新评估条件：

- 新的推理模型或数据路径 profiling 显示 nested/list 或 block-diagonal attention
  占比明显。
- dense attention 不再走当前可用路径，或 MUSA SDPA/FlashAttentionRope 出现性能或
  correctness 问题。

## 其它状态

- MUSA `torch.svd` 缺失已通过 polar12 fast path 和少量 CPU-SVD fallback 处理。
- MUSA `torch.det`、`torch.inverse`、`torch.linalg.inv` 当前实测可用。
- PyTorch `scaled_dot_product_attention` bf16 当前实测可用。
