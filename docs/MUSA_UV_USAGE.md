# Pi3 MUSA uv 使用说明

本文记录 `/datapool/husicheng/Pi3` 在 Moore Threads S5000/MUSA 环境的运行方式。

## 环境边界

- 默认镜像：`registry.mthreads.com/mcctest/ai/mtwan:4.3.3-pt2.7-v0.2-mudnn3.1.7-ph1`
- 共享镜像 loader：`/datapool/shared_images/mtwan_4.3.3-pt2.7-v0.2-mudnn3.1.7-ph1/load_mtwan.sh`
- 项目 venv：`/datapool/husicheng/Pi3/.venv`
- PyPI 首选镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 代理回退：`/datapool/.config/mihomo/proxy-env.sh`

`.venv` 使用 `--system-site-packages` 创建，复用 mtwan 镜像内的 MUSA PyTorch：

```text
torch==2.7.1
torch_musa==2.7.1+9f1bb31
torchvision==0.22.1+32091f2
```

不要在项目 `.venv` 内安装 PyPI/CUDA 版 `torch`、`torchvision`、`torch_musa`、`triton` 或 `nvidia-*`/`cuda-*` 包。`requirements-musa.txt`、`constraints-musa.txt`、`excludes-musa.txt` 和 bootstrap 校验会阻止这些包进入 `.venv`。

## 初始化

从宿主机运行：

```bash
cd /datapool/husicheng/Pi3
bash scripts/bootstrap_musa_uv.sh
```

脚本会自动进入默认 mtwan 容器。Python 包安装先走国内镜像，失败后才加载代理并回退到 PyPI。

## 下载模型

模型从 ModelScope 下载，下载脚本会清掉代理环境：

```bash
cd /datapool/husicheng/Pi3
MODELSCOPE_TOKEN=<token> bash scripts/run_musa.sh bash scripts/download_modelscope.sh
```

ModelScope 临时下载目录为：

```text
/datapool/husicheng/Pi3/checkpoints/Jasonhsc/Pi3
```

下载完成后脚本会把权重移动到稳定运行路径：

```text
/datapool/husicheng/Pi3/ckpts/Pi3.safetensors
/datapool/husicheng/Pi3/ckpts/Pi3X.safetensors
```

## 运行

所有运行命令建议通过 `scripts/run_musa.sh` 进入同一 mtwan 容器：

```bash
cd /datapool/husicheng/Pi3
bash scripts/run_musa.sh .venv/bin/python example_mm.py \
  --device auto
```

示例脚本会默认优先使用本地 `ckpts/Pi3.safetensors` / `ckpts/Pi3X.safetensors`；只有这些文件不存在时才回退到原始 `from_pretrained` 逻辑。

若要进入交互 shell：

```bash
bash scripts/run_musa.sh bash
```
