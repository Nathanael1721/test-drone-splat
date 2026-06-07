"""3-way comparison: GT | baseline render | global-mask render, for the test set."""
import os, sys
from PIL import Image, ImageDraw

base = r"D:\OneDrive\College\S2-TaipeiTech\2nd Semester\Computer Vision\DroneSplat"
scene = sys.argv[1] if len(sys.argv) > 1 else "Simingshan"
m1 = sys.argv[2] if len(sys.argv) > 2 else "Simingshan"        # baseline output dir name
m2 = sys.argv[3] if len(sys.argv) > 3 else "Simingshan_gm"     # global-mask output dir name
maxn = int(sys.argv[4]) if len(sys.argv) > 4 else 999

gt_dir = os.path.join(base, "data", scene, "images")
r1_dir = os.path.join(base, "output", m1, "render_test")
r2_dir = os.path.join(base, "output", m2, "render_test")
names = sorted(os.listdir(r2_dir))[:maxn]

tile_w, hgap = 460, 6
labels = ["GROUND TRUTH", "BASELINE (local mask)", "+ GLOBAL MASK (ours)"]
rows = []
for n in names:
    imgs = []
    for d in (gt_dir, r1_dir, r2_dir):
        p = os.path.join(d, n)
        imgs.append(Image.open(p).convert("RGB") if os.path.exists(p) else Image.new("RGB", (tile_w, 260), (40, 0, 0)))
    h = int(imgs[0].height * tile_w / imgs[0].width)
    imgs = [im.resize((tile_w, h)) for im in imgs]
    row = Image.new("RGB", (tile_w * 3 + hgap * 2, h + 24), (20, 20, 20))
    d = ImageDraw.Draw(row)
    for i, im in enumerate(imgs):
        x = i * (tile_w + hgap)
        row.paste(im, (x, 24))
        col = (255, 255, 255) if i == 0 else ((150, 170, 200) if i == 1 else (120, 230, 120))
        d.text((x + 4, 6), labels[i] + ("  " + n if i == 0 else ""), fill=col)
    rows.append(row)

W = rows[0].width
H = sum(r.height for r in rows) + 4 * (len(rows) - 1)
canvas = Image.new("RGB", (W, H), (20, 20, 20))
y = 0
for r in rows:
    canvas.paste(r, (0, y)); y += r.height + 4
out = os.path.join(base, "output", m2, "compare3_test.jpg")
canvas.save(out, quality=90)
print("saved:", out, canvas.size)
