"""Stitch the 6 cubemap faces into one equirectangular panorama for Pannellum."""
import os, numpy as np
from PIL import Image
import py360convert

import sys
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
PANO = os.path.join(OUT, "pano")


def load(name):
    return np.array(Image.open(os.path.join(PANO, name + ".png")).convert("RGB"))

# py360convert 'dict' expects faces oriented so they tile seamlessly.
# Our render uses up = world-up for the 4 side faces; U/D need a roll to match.
faces = {
    "F": load("front"),
    "R": load("right"),
    "B": load("back"),
    "L": load("left"),
    "U": load("up"),
    "D": load("down"),
}

equi = py360convert.c2e(faces, h=2048, w=4096, cube_format="dict")
equi = np.clip(equi, 0, 255).astype(np.uint8)
Image.fromarray(equi).save(os.path.join(OUT, "equirect.png"))
print("saved equirect.png", equi.shape)
