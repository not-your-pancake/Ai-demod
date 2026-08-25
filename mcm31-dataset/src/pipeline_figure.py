"""
pipeline_figure.py - the processing chain, as a block diagram.

Two lanes share one capture. The upper lane produces ground-truth labels by
coherent averaging; the lower lane produces the network's input. They are drawn
separately because the separation is the point: labels never touch the network's
input path, and the network never receives the carrier, the pulse shape or the
constellation that the labelling path uses.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(7.1, 2.75))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()

BW, BH = 0.148, 0.19


def box(x, y, text, edge, fill="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fill, ec=edge, lw=lw, zorder=3))
    ax.text(x + BW / 2, y + BH / 2, text, ha="center", va="center",
            fontsize=6.5, color=INK, zorder=4, linespacing=1.35)


def arrow(x0, y0, x1, y1, colour=INK2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 color=colour, lw=0.9, mutation_scale=8,
                                 shrinkA=0, shrinkB=0, zorder=2))


TOP, BOT = 0.63, 0.16
XS = [0.20, 0.38, 0.56, 0.74]

# shared source
box(0.015, (TOP + BOT) / 2, "6.0 s capture\nTP-20 + TP-9\n87.26 kSa/s", INK2)
ax.text(0.089, (TOP + BOT) / 2 + BH + 0.035, "hardware", ha="center",
        fontsize=6.3, color=INK2, style="italic")

# ---- upper lane: ground truth -------------------------------------------
box(XS[0], TOP, "word period\nrefinement", ORANGE)
box(XS[1], TOP, "coherent average\nK = 57–450", ORANGE)
box(XS[2], TOP, "blind symbol\ntiming recovery", ORANGE)
box(XS[3], TOP, "quantise\n$\\rightarrow$ labels", ORANGE, fill="#fdf0ea")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + BW, TOP + BH / 2, b, TOP + BH / 2, ORANGE)
arrow(0.015 + BW, (TOP + BOT) / 2 + BH * 0.62, XS[0], TOP + BH / 2, ORANGE)
ax.text((XS[0] + XS[3] + BW) / 2, TOP + BH + 0.055,
        "ground-truth path — never seen by the network",
        ha="center", fontsize=6.5, color=ORANGE, style="italic")

# ---- lower lane: network input ------------------------------------------
box(XS[0], BOT, "anti-alias +\ndecimate\n145 $\\rightarrow$ 20 sps", BLUE)
box(XS[1], BOT, "AGC over\nthe capture", BLUE)
box(XS[2], BOT, "window\n$N$ symbols\n(the RF cap)", BLUE, fill="#eaf2fc", lw=1.5)
box(XS[3], BOT, "1-D CNN\n$\\approx$21k params", BLUE, fill="#eaf2fc")
for a, b in zip(XS[:-1], XS[1:]):
    arrow(a + BW, BOT + BH / 2, b, BOT + BH / 2, BLUE)
arrow(0.015 + BW, (TOP + BOT) / 2 + BH * 0.38, XS[0], BOT + BH / 2, BLUE)
ax.text((XS[0] + XS[3] + BW) / 2, BOT - 0.075,
        "signal path — passband throughout; no carrier, pulse shape or "
        "constellation supplied",
        ha="center", fontsize=6.5, color=BLUE, style="italic")

# software AWGN injected into the signal path only
ax.text(XS[1] + BW / 2, BOT + BH + 0.10, "software AWGN\n0–30 dB", ha="center",
        va="center", fontsize=6.3, color=AQUA)
arrow(XS[1] + BW / 2, BOT + BH + 0.045, XS[1] + BW / 2, BOT + BH, AQUA)

# labels join the network at the loss
arrow(XS[3] + BW / 2, TOP, XS[3] + BW / 2, BOT + BH, INK2, style="-|>")
ax.text(XS[3] + BW / 2 + 0.012, (TOP + BOT + BH) / 2, "loss only",
        ha="left", va="center", fontsize=6.3, color=INK2)

# output
arrow(XS[3] + BW, BOT + BH / 2, 0.945, BOT + BH / 2, BLUE)
ax.text(0.952, BOT + BH / 2, "symbol", ha="left", va="center",
        fontsize=6.8, color=INK)

fig.tight_layout(pad=0.15)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_pipeline.{ext}"))
plt.close(fig)
print("fig_pipeline.pdf / .png written")
