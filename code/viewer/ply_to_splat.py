"""Convert an INRIA 3DGS .ply into the antimatter15 .splat format (32 bytes/gaussian)
and print suggested camera framing. Loadable by mkkellogg/gaussian-splats-3d."""
import sys, numpy as np
from plyfile import PlyData

SH_C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main(ply_path, out_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    rots = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
    fdc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    alpha = sigmoid(np.asarray(v["opacity"], dtype=np.float32))[:, None]
    rgba = np.clip(np.concatenate([rgb, alpha], axis=1) * 255.0, 0, 255).astype(np.uint8)
    rot_u8 = np.clip(rots * 128.0 + 128.0, 0, 255).astype(np.uint8)

    n = xyz.shape[0]
    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scales.view(np.uint8).reshape(n, 12)
    buf[:, 24:28] = rgba
    buf[:, 28:32] = rot_u8

    # sort by descending "size * opacity" so big/opaque splats load first (nicer progressive load)
    importance = (scales.prod(axis=1)) * alpha[:, 0]
    order = np.argsort(-importance)
    buf = buf[order]
    buf.tofile(out_path)

    c = xyz.mean(axis=0)
    ext = np.percentile(xyz, 97, axis=0) - np.percentile(xyz, 3, axis=0)
    diag = float(np.linalg.norm(ext))
    print(f"gaussians: {n}")
    print(f"out_size_MB: {n * 32 / 1e6:.1f}")
    print(f"centroid: {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}")
    print(f"extent97: {ext[0]:.3f} {ext[1]:.3f} {ext[2]:.3f}  diag={diag:.3f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
