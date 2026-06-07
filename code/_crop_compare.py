"""Zoomed road-region crop comparison: GT | baseline | full, for a few frames."""
import os, sys
from PIL import Image, ImageDraw

base = r"D:\OneDrive\College\S2-TaipeiTech\2nd Semester\Computer Vision\DroneSplat"
scene, m1, m2 = "Simingshan", "Simingshan", "Simingshan_full"
gt_dir = os.path.join(base, "data", scene, "images")
r1_dir = os.path.join(base, "output", m1, "render_test")
r2_dir = os.path.join(base, "output", m2, "render_test")

frames = sorted(os.listdir(r2_dir))[:4]
box = (560, 470, 1360, 1010)   # road region (x0,y0,x1,y1) for 1920x1080
labels = ["GROUND TRUTH", "BASELINE (local only)", "FULL (+global +voxel)"]
out_w = 600
rows = []
for n in frames:
    crops = []
    for d in (gt_dir, r1_dir, r2_dir):
        im = Image.open(os.path.join(d, n)).convert("RGB").crop(box)
        h = int(im.height * out_w / im.width)
        crops.append(im.resize((out_w, h)))
    h = crops[0].height
    row = Image.new("RGB", (out_w * 3 + 12, h + 24), (18, 18, 18))
    dr = ImageDraw.Draw(row)
    for i, im in enumerate(crops):
        x = i * (out_w + 6)
        row.paste(im, (x, 24))
        col = (255, 255, 255) if i == 0 else ((160, 175, 205) if i == 1 else (120, 230, 120))
        dr.text((x + 5, 6), labels[i] + ("   " + n if i == 0 else ""), fill=col)
    rows.append(row)

W = rows[0].width
H = sum(r.height for r in rows) + 5 * (len(rows) - 1)
canvas = Image.new("RGB", (W, H), (18, 18, 18))
y = 0
for r in rows:
    canvas.paste(r, (0, y)); y += r.height + 5
out = os.path.join(base, "output", "Simingshan_full", "crop_compare.jpg")
canvas.save(out, quality=92)
print("saved:", out, canvas.size)
