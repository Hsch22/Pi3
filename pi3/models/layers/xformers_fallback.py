from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import Tensor
from torch import nn


class TorchBlockDiagonalMask:
    def __init__(self, seqlens: Sequence[int]) -> None:
        self.seqlens = tuple(int(s) for s in seqlens)
        self._batch_sizes: List[int] | None = None

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "TorchBlockDiagonalMask":
        return cls(tuple(seqlens))

    def split(self, tensor: Tensor) -> List[Tensor]:
        if tensor.shape[0] != 1:
            raise ValueError("TorchBlockDiagonalMask.split expects a batch size of 1")
        batch_sizes = self._batch_sizes or [1] * len(self.seqlens)

        outputs = []
        seq_idx = 0
        token_idx = 0
        for batch_size in batch_sizes:
            group_seqlens = self.seqlens[seq_idx : seq_idx + batch_size]
            if not group_seqlens:
                raise ValueError("empty sequence group in TorchBlockDiagonalMask")
            if len(set(group_seqlens)) != 1:
                raise ValueError("cannot reshape a group with unequal sequence lengths")

            group_tokens = sum(group_seqlens)
            chunk = tensor[:, token_idx : token_idx + group_tokens]
            outputs.append(chunk.reshape(batch_size, group_seqlens[0], *tensor.shape[2:]))

            seq_idx += batch_size
            token_idx += group_tokens
        return outputs

    def materialize(self, q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        total = sum(self.seqlens)
        if q_len != total or k_len != total:
            raise ValueError("block diagonal mask length does not match attention shape")

        min_value = torch.finfo(dtype).min
        mask = torch.full((q_len, k_len), min_value, device=device, dtype=dtype)
        offset = 0
        for seqlen in self.seqlens:
            mask[offset : offset + seqlen, offset : offset + seqlen] = 0
            offset += seqlen
        return mask.view(1, 1, q_len, k_len)


def index_select_cat(tensors: Sequence[Tensor], indices: Sequence[Tensor]) -> Tensor:
    return torch.cat([tensor.index_select(0, index).flatten() for tensor, index in zip(tensors, indices)], dim=0)


def scaled_index_add(
    input: Tensor,
    index: Tensor,
    source: Tensor,
    *,
    scaling: Tensor | None = None,
    alpha: float = 1.0,
) -> Tensor:
    if scaling is not None:
        source = source * scaling
    return torch.index_add(input, 0, index, source.to(dtype=input.dtype), alpha=alpha)


def add_attention_bias(attn: Tensor, attn_bias) -> Tensor:
    if attn_bias is None:
        return attn
    if isinstance(attn_bias, TorchBlockDiagonalMask):
        return attn + attn_bias.materialize(attn.shape[-2], attn.shape[-1], attn.device, attn.dtype)
    if isinstance(attn_bias, Tensor):
        return attn + attn_bias.to(device=attn.device, dtype=attn.dtype)
    raise TypeError(f"unsupported attention bias type: {type(attn_bias)!r}")


def block_diagonal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_bias: TorchBlockDiagonalMask,
    attn_drop: nn.Module | None = None,
) -> Tensor:
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError("TorchBlockDiagonalMask attention expects a batch size of 1")

    outputs = []
    offset = 0
    for seqlen in attn_bias.seqlens:
        end = offset + seqlen
        q_i = q[:, :, offset:end]
        k_i = k[:, :, offset:end]
        v_i = v[:, :, offset:end]
        attn_i = q_i @ k_i.transpose(-2, -1)
        attn_i = attn_i.softmax(dim=-1)
        if attn_drop is not None:
            attn_i = attn_drop(attn_i)
        outputs.append(attn_i @ v_i)
        offset = end
    return torch.cat(outputs, dim=2)
