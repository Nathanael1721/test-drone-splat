# DroneSplat — Explainer Website + Reconstruction Code

This repository hosts an **interactive explainer website** for the DroneSplat paper
(*3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery*, CVPR 2025)
together with our reconstruction process, results, and **re-implementation of the paper components
missing from the official code**.

## 🌐 Live website (Vercel)

The website lives at the **repository root**, so connecting this repo to Vercel deploys it
automatically — no configuration needed (framework preset: **Other**).

- `index.html` — explainer (English) + embedded live 3D viewers
- `id.html` — Indonesian version
- `orbit-*.html` / `pano-*.html` — standalone 3DGS orbit & 360° viewers
- `models/` — `.splat` 3D Gaussian models (Simingshan 44 MB, Sculpture 94 MB)
- `assets/` — figures, comparisons, the ghost before/after crop
- `lib/` — Pannellum (vendored); 3DGS rendering via the GaussianSplats3D CDN

Local preview:
```bash
python -m http.server 8080   # then open http://localhost:8080/
```

## 🧪 `code/` — reconstruction pipeline + our additions

The full DroneSplat code we ran, plus the components we **re-implemented from the paper** because
they are absent from the public release:

- `code/global_masking.py` — **Complement Global Masking** (SAM2 video tracking), new.
- `code/train.py` — hooks for `--use_global_mask` and `--use_voxel`.
- `code/scene/gaussian_model.py` — **Voxel-guided optimization** (gradient decay + ghost-voxel opacity pruning).
- `code/explainer*.html`, `code/_make_*.py`, `code/_crop_compare.py` — explainer + comparison tooling.

**Not included** (too large for Git — download separately): model checkpoints, the `data/` scenes,
trained `output/` point clouds (`.ply`), and `masks.json`. The git **submodules** (DUSt3R, SAM2,
diff-gaussian-rasterization, simple-knn) must be obtained from the upstream
[BITyia/DroneSplat](https://github.com/BITyia/DroneSplat) repo to run the code.

Run the full re-implemented pipeline:
```bash
python code/train.py -s data/Simingshan -m output/Simingshan_full --scene Simingshan \
  --iterations 7000 --use_masks --use_global_mask --global_track_iter 1500 --use_voxel
```

## Credit
Paper: Tang et al., CVPR 2025. Built on 3D Gaussian Splatting, DUSt3R, SAM2, InstantSplat.
Numbers quoted from the official CVPR Open Access PDF; result images & 3D models are our own reconstruction.
