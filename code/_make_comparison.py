import os, json, sys
from PIL import Image, ImageDraw

base = r"D:\OneDrive\College\S2-TaipeiTech\2nd Semester\Computer Vision\DroneSplat"
scene = sys.argv[1] if len(sys.argv) > 1 else "Simingshan"
maxn = int(sys.argv[2]) if len(sys.argv) > 2 else 999
render_dir = os.path.join(base, "output", scene, "render_test")
gt_dir = os.path.join(base, "data", scene, "images")
metrics = json.load(open(os.path.join(base, "output", scene, "metrics.json")))
mdict = {m["image_name"]: m for m in metrics if "image_name" in m}

names = sorted(os.listdir(render_dir))[:maxn]
tile_w = 480
rows = []
for n in names:
    gt = Image.open(os.path.join(gt_dir, n)).convert("RGB")
    rd = Image.open(os.path.join(render_dir, n)).convert("RGB")
    h = int(gt.height * tile_w / gt.width)
    gt = gt.resize((tile_w, h)); rd = rd.resize((tile_w, h))
    row = Image.new("RGB", (tile_w * 2 + 8, h + 22), (20, 20, 20))
    row.paste(gt, (0, 22)); row.paste(rd, (tile_w + 8, 22))
    d = ImageDraw.Draw(row)
    m = mdict.get(n, {})
    d.text((4, 4), f"GT  {n}", fill=(255, 255, 255))
    d.text((tile_w + 12, 4),
           f"RENDER  PSNR={m.get('psnr', 0):.1f}  SSIM={m.get('ssim', 0):.2f}  LPIPS={m.get('lpips', 0):.2f}",
           fill=(120, 230, 120))
    rows.append(row)

W = rows[0].width
H = sum(r.height for r in rows) + 4 * (len(rows) - 1)
canvas = Image.new("RGB", (W, H), (20, 20, 20))
y = 0
for r in rows:
    canvas.paste(r, (0, y)); y += r.height + 4
out = os.path.join(base, "output", scene, "comparison_test.jpg")
canvas.save(out, quality=90)
print("saved:", out, canvas.size)
