import torch


def det3(matrix):
    """Compute determinants for a batch of 3x3 matrices without torch.det."""
    a00 = matrix[:, 0, 0]
    a01 = matrix[:, 0, 1]
    a02 = matrix[:, 0, 2]
    a10 = matrix[:, 1, 0]
    a11 = matrix[:, 1, 1]
    a12 = matrix[:, 1, 2]
    a20 = matrix[:, 2, 0]
    a21 = matrix[:, 2, 1]
    a22 = matrix[:, 2, 2]
    return (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )


def trace3(matrix):
    return matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]


def polar_project_so3(matrix, num_iters=12):
    """Project 3x3 matrices to SO(3) with Newton-Schulz polar iterations."""
    dtype = matrix.dtype
    x = matrix.float()
    scale = torch.sqrt((x * x).sum(dim=(1, 2), keepdim=True)).clamp_min(1e-6)
    x = x / scale

    eye = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[0], 3, 3)
    for _ in range(num_iters):
        x = 0.5 * torch.bmm(x, 3.0 * eye - torch.bmm(x.transpose(1, 2), x))

    sign = torch.where(
        det3(x) < 0,
        -torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
        torch.ones(x.shape[0], device=x.device, dtype=x.dtype),
    )
    x = torch.cat([x[:, :, :2], x[:, :, 2:3] * sign.view(-1, 1, 1)], dim=2)
    return x.to(dtype=dtype)


def rotation_error_mask(rotation, orth_threshold=1e-3, det_threshold=1e-3):
    work = rotation.float()
    eye = torch.eye(3, device=work.device, dtype=work.dtype).expand(work.shape[0], 3, 3)
    gram_error = torch.bmm(work.transpose(1, 2), work) - eye
    orth_error = torch.sqrt((gram_error * gram_error).sum(dim=(1, 2)))
    det_error = (det3(work) - 1.0).abs()
    finite = torch.isfinite(orth_error) & torch.isfinite(det_error)
    bad = (~finite) | (orth_error > orth_threshold) | (det_error > det_threshold)
    return bad, orth_error, det_error


def svd_project_so3(matrix):
    u, _, v = torch.svd(matrix)
    det = det3(torch.bmm(u, v.transpose(1, 2)))
    u = torch.cat([u[:, :, :2], u[:, :, 2:3] * det.view(-1, 1, 1)], dim=2)
    return torch.bmm(u, v.transpose(1, 2))


def svd_project_so3_cpu(matrix, device=None, dtype=None):
    target_device = matrix.device if device is None else device
    target_dtype = matrix.dtype if dtype is None else dtype
    rotation = svd_project_so3(matrix.cpu().float())
    return rotation.to(device=target_device, dtype=target_dtype)
