# CroCo cuRoPE2D MUSA adaptation

Upstream snapshot:

- Source: `../croco`
- Commit: `5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb`
- Sparse paths: `models/curope`, `models/pos_embed.py`

MUSA changes:

- `setup.py` uses `torch_musa.utils.musa_extension.MUSAExtension` and `BuildExtension`.
- `kernels.cu` is ported to `kernels.mu`.
- CUDA runtime headers and launch checks are replaced with MUSA runtime equivalents.
- C++ device dispatch checks `c10::DeviceType::PrivateUse1` for MUSA tensors.
- Python import path stays compatible with CroCo/Pi3: `from models.curope import cuRoPE2D`.

Build:

```bash
cd /datapool/husicheng/Pi3/third_party/croco_musa/models/curope
/datapool/husicheng/Pi3/.venv/bin/python setup.py build_ext --inplace
```

Use from Pi3 without copying into the repo root:

```bash
PYTHONPATH=/datapool/husicheng/Pi3/third_party/croco_musa \
  /datapool/husicheng/Pi3/.venv/bin/python your_script.py
```

Validation notes:

- `pi3.models.layers.pos_embed.RoPE2D` resolves to `models.curope.curope2d.cuRoPE2D` when `PYTHONPATH` points at this directory.
- Forward numerical check passed for float32 with max absolute error `4.768e-07`.
- bfloat16 check showed max absolute error `4.688e-02` and mean absolute error `1.855e-03` versus the PyTorch fallback.
- Backward smoke test passed on MUSA with finite gradients.
- On shape `(B,H,N,D)=(1,16,3136,64)`, cached PyTorch fallback was about `0.44-0.45 ms`; MUSA cuRoPE in-place kernel was about `0.06-0.09 ms`.
