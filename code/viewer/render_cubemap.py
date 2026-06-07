"""Render a 6-face cubemap (90 deg FOV) from a point inside the trained 3DGS scene,
for a Street-View-style 360 panorama (Pannellum cubeMap input).

Reuses the repo's Camera + render() math exactly (camera_pose = w2c quaternion+T)."""
import os, sys, math, types
import numpy as np
import torch
import torchvision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
from gaussian_renderer import render
from utils.pose_utils import get_tensor_from_camera

SCENE = sys.argv[1] if len(sys.argv) > 1 else "Simingshan"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "viewer")
PLY = os.path.join(ROOT, "output", SCENE, "point_cloud", "iteration_7000", "point_cloud.ply")
SPARSE_IMAGES = os.path.join(ROOT, "data", SCENE, "sparse", "0", "images.bin")
OUT = os.path.join(OUTDIR, "pano")
S = 1024  # face resolution


def cam_frame_from_colmap():
    """Return (center, forward, up, right) of the average camera frame in world coords."""
    extr = read_extrinsics_binary(SPARSE_IMAGES)
    centers, fwds, ups = [], [], []
    for im in extr.values():
        Rcw = qvec2rotmat(im.qvec)          # world->cam rotation
        R_c2w = Rcw.T                        # cam->world (columns = cam axes in world)
        C = -R_c2w @ np.asarray(im.tvec)     # camera center in world
        centers.append(C)
        fwds.append(R_c2w[:, 2])             # cam +Z (forward)
        ups.append(-R_c2w[:, 1])             # cam -Y (up)
    center = np.mean(centers, axis=0)
    up = np.mean(ups, axis=0); up /= np.linalg.norm(up)
    fwd = np.mean(fwds, axis=0)
    fwd = fwd - np.dot(fwd, up) * up         # level the horizon
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    return center, fwd, up, right


def make_c2w(forward, up):
    """Build cam->world rotation with columns [right(+X), down(+Y), forward(+Z)]."""
    f = forward / np.linalg.norm(forward)
    u = up / np.linalg.norm(up)
    down = -u
    right = np.cross(down, f); right /= np.linalg.norm(right)
    down = np.cross(f, right); down /= np.linalg.norm(down)
    return np.stack([right, down, f], axis=1)  # columns


def render_face(name, forward, up, center, gaussians, pipe, bg):
    R_c2w = make_c2w(forward, up)
    T = -R_c2w.T @ center                     # Camera expects R=cam2world, T=w2c translation
    dummy = torch.zeros(3, S, S)
    cam = Camera(colmap_id=0, R=R_c2w, T=T, FoVx=math.pi / 2, FoVy=math.pi / 2,
                 image=dummy, gt_alpha_mask=None, image_name=name, uid=0)
    camera_pose = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1))
    img = render(cam, gaussians, pipe, bg, camera_pose=camera_pose)["render"]
    os.makedirs(OUT, exist_ok=True)
    torchvision.utils.save_image(img, os.path.join(OUT, name + ".png"))
    print("rendered", name)


def main():
    gaussians = GaussianModel(3)
    gaussians.load_ply(PLY)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    pipe = types.SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    center, fwd, up, right = cam_frame_from_colmap()
    # optional center override: pass "x,y,z" as 3rd CLI arg (e.g., the gaussian centroid)
    if len(sys.argv) > 3:
        center = np.array([float(t) for t in sys.argv[3].split(",")], dtype=np.float64)
    print("center", center, "up", up)

    faces = {
        "front": (fwd, up),
        "right": (right, up),
        "back": (-fwd, up),
        "left": (-right, up),
        "up": (up, -fwd),
        "down": (-up, fwd),
    }
    with torch.no_grad():
        for name, (f, u) in faces.items():
            render_face(name, f, u, center, gaussians, pipe, bg)


if __name__ == "__main__":
    main()
