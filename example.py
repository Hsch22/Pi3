import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.checkpoint import load_checkpoint_state, resolve_checkpoint
from pi3.utils.device import autocast, get_amp_dtype, resolve_device
from pi3.utils.geometry import depth_normal_edge
from pi3.models.pi3 import Pi3

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='auto',
                        help="Device to run inference on ('auto', 'musa', 'cuda' or 'cpu'). Default: 'auto'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = resolve_device(args.device)
    ckpt = resolve_checkpoint("pi3", args.ckpt)
    if ckpt is not None:
        print(f"Loading checkpoint: {ckpt}")
        model = Pi3().eval()
        weight = load_checkpoint_state(ckpt)
        model.load_state_dict(weight)
        model = model.to(device)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = get_amp_dtype(device)
    with torch.no_grad():
        with autocast(device, dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_normal_edge(res['local_points'], rtol=0.03, mask=masks)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")
