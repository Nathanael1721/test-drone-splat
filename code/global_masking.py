"""
Complement Global Masking (DroneSplat paper, Sec. 3.2, Eq. 11-14) — re-implementation.

The released train.py only implements Adaptive *Local* Masking. This module adds the
*Global* masking: objects whose object-wise residual exceeds a stricter global threshold
T_G = E[R] + lambda_G * Var[R] are treated as tracking candidates; SAM2's video predictor
then propagates each such object across ALL frames, so a distractor that is momentarily
static in some frame (e.g. a car stopped at a red light) is still masked everywhere.

Usage (from train.py):
    tracker = GlobalMaskTracker(image_dir, sam2_ckpt, sam2_cfg)
    tracker.add_candidates(image_name, label_map_np, residual_norm_np, lambda_g)   # per frame
    tracker.run()                       # runs SAM2 video tracking for all candidates
    keep = tracker.get_keep_mask(image_name, H, W)   # torch float (1=keep, 0=distractor)
    tracker.close()
"""
import os
import sys
import shutil
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _prompt_points_from_mask(binary_mask):
    """Center + 4 extreme edge points of a binary object mask (paper: center + 4 edges)."""
    ys, xs = np.nonzero(binary_mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    pts = [
        [cx, cy],                                   # center
        [float(xs[np.argmin(xs)]), float(ys[np.argmin(xs)])],  # leftmost
        [float(xs[np.argmax(xs)]), float(ys[np.argmax(xs)])],  # rightmost
        [float(xs[np.argmin(ys)]), float(ys[np.argmin(ys)])],  # topmost
        [float(xs[np.argmax(ys)]), float(ys[np.argmax(ys)])],  # bottommost
    ]
    return np.array(pts, dtype=np.float32)


class GlobalMaskTracker:
    def __init__(self, image_dir, sam2_ckpt, sam2_cfg="configs/sam2/sam2_hiera_l.yaml", device="cuda"):
        from submodules.sam2.sam2.build_sam import build_sam2_video_predictor
        self.device = device
        self.predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)

        # SAM2's video loader requires integer frame filenames ("<idx>.jpg"); our images are
        # named like "2411006_18_001.jpg", so build a temp dir of numerically-named frames.
        names = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        self.tmpdir = tempfile.mkdtemp(prefix="dronesplat_gm_")
        self.idx2name, self.name2idx = {}, {}
        for i, n in enumerate(names):
            Image.open(os.path.join(image_dir, n)).convert("RGB").save(
                os.path.join(self.tmpdir, f"{i:05d}.jpg"), quality=95)
            stem = os.path.splitext(n)[0]
            self.idx2name[i] = stem
            self.name2idx[stem] = i
        self.num_frames = len(names)

        self._candidates = []          # list of (frame_idx, points) to track
        self._tracked_keys = set()     # dedup (frame_idx, instance_id)
        self.global_masks = {}         # stem -> bool np.array (H,W), True = distractor
        self.state = None

    # ---- candidate collection (called per training frame at the warmup checkpoint) ----
    def add_candidates(self, image_name, label_map, residual_norm, lambda_g):
        """label_map: HxW int instance ids; residual_norm: HxW float in [0,1] (object-wise)."""
        stem = os.path.splitext(image_name)[0]
        if stem not in self.name2idx:
            return
        fidx = self.name2idx[stem]
        rmean, rstd = float(residual_norm.mean()), float(residual_norm.std())
        t_g = rmean + lambda_g * rstd          # T_G = E[R] + lambda_G * Var[R]
        for inst in np.unique(label_map):
            if inst == 0:
                continue
            m = label_map == inst
            if m.sum() < 50:                    # ignore tiny specks
                continue
            if float(residual_norm[m].mean()) <= t_g:
                continue
            key = (fidx, int(inst))
            if key in self._tracked_keys:
                continue
            pts = _prompt_points_from_mask(m)
            if pts is not None:
                self._candidates.append((fidx, pts))
                self._tracked_keys.add(key)

    # ---- run SAM2 video tracking for every collected candidate ----
    @torch.no_grad()
    def run(self):
        if not self._candidates:
            print("[GlobalMask] no candidates above T_G — nothing to track.")
            return
        print(f"[GlobalMask] tracking {len(self._candidates)} distractor candidate(s) across {self.num_frames} frames...")
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast.__enter__()
        self.state = self.predictor.init_state(
            self.tmpdir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        for ci, (fidx, pts) in enumerate(self._candidates):
            try:
                self.predictor.reset_state(self.state)
                self.predictor.add_new_points_or_box(
                    self.state, frame_idx=fidx, obj_id=1,
                    points=pts, labels=np.ones(len(pts), dtype=np.int32))
                self._collect(self.predictor.propagate_in_video(self.state, start_frame_idx=fidx))
                self._collect(self.predictor.propagate_in_video(self.state, start_frame_idx=fidx, reverse=True))
            except Exception as e:
                print(f"[GlobalMask] candidate {ci} (frame {fidx}) failed: {e}")
        autocast.__exit__(None, None, None)
        n_px = sum(int(m.sum()) for m in self.global_masks.values())
        print(f"[GlobalMask] done. global distractor pixels: {n_px} across {len(self.global_masks)} frames.")

    def _collect(self, gen):
        for out_idx, _obj_ids, mask_logits in gen:
            m = (mask_logits[0] > 0.0).squeeze().detach().cpu().numpy().astype(bool)
            stem = self.idx2name[int(out_idx)]
            if stem not in self.global_masks:
                self.global_masks[stem] = np.zeros_like(m, dtype=bool)
            else:
                if self.global_masks[stem].shape != m.shape:
                    continue
            self.global_masks[stem] |= m

    # ---- consume during training ----
    def has_masks(self):
        return len(self.global_masks) > 0

    def get_keep_mask(self, image_name, H, W):
        """Return float tensor (H,W) on cuda: 1=keep (static), 0=distractor."""
        stem = os.path.splitext(image_name)[0]
        if stem not in self.global_masks:
            return None
        dm = torch.from_numpy(self.global_masks[stem].astype(np.float32))[None, None]
        dm = F.interpolate(dm, size=(H, W), mode="nearest")[0, 0]
        return (1.0 - dm).cuda()

    def release_model(self):
        """Free the SAM2 video model + inference state from GPU, keep the numpy masks."""
        try:
            del self.state
            self.state = None
            del self.predictor
            self.predictor = None
            torch.cuda.empty_cache()
        except Exception:
            pass

    def close(self):
        try:
            del self.state
            self.state = None
            torch.cuda.empty_cache()
        except Exception:
            pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)
