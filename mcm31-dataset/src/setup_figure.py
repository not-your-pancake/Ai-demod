"""
setup_figure.py - turn the bench photograph into a publication figure.

A raw phone photograph is not a figure. This crops away the room behind the
bench, which carries no information, and labels the instruments a reader needs
to identify: the oscilloscope, the supply, the probe points, and the units under
test. Labels are drawn as vector text so they stay sharp in the PDF.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "photos",
                   "5_selected_boards_after_completion_of_dataset.jpeg")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2 = "#0b0b0b", "#52514e"
CALL = "#eb6834"          # callout colour, same orange as the other figures

# crop: keep the bench, drop the room behind it
CROP = (0, 300, 1240, 940)      # left, upper, right, lower
im = Image.open(SRC).crop(CROP)
W, H = im.size

fig, ax = plt.subplots(figsize=(7.1, 7.1 * H / W))
ax.imshow(im)
ax.set_axis_off()

# (text x, text y, arrow-tip x, arrow-tip y, label) in cropped pixel coords
CALLOUTS = [
    (960,  30, 890,  75, "Digital oscilloscope\n(TP-20 and TP-9, simultaneous)"),
    (430,  35, 600,  70, "Universal power supply"),
    (760, 235, 700, 190, "Probe pair"),
    (150, 300, 260, 250, "MCM31/EV units under test"),
    (600, 560, 640, 500, "Bench logbook"),
]
for tx, ty, ax_, ay, label in CALLOUTS:
    ax.annotate(
        label, xy=(ax_, ay), xytext=(tx, ty),
        fontsize=7, color=INK, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=CALL, lw=0.9,
                  alpha=0.93),
        arrowprops=dict(arrowstyle="-|>", color=CALL, lw=1.1,
                        shrinkA=2, shrinkB=2))

fig.tight_layout(pad=0.1)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig0_setup.{ext}"), dpi=300,
                bbox_inches="tight")
plt.close(fig)
print(f"fig0_setup.pdf / .png written  (cropped from {Image.open(SRC).size} "
      f"to {(W, H)})")
