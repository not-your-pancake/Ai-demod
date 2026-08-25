"""
constellation_figure.py - the measured constellations, from the real captures.

The paper asserts a four-phase two-amplitude star with a measured radius ratio
of 1.89. Until now the reader has had to take that on trust. This plots it from
the data: every symbol of every capture, downconverted and derotated by the same
pipeline that produces the training labels, pooled across all five boards so the
device-to-device spread is visible rather than hidden.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
meta = pd.read_csv(os.path.join(ROOT, "dataset", "captures.csv"))
SPS = int(d["sps_out"])
wave, wo = d["wave"], d["wave_offset"]
rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))
TITLE = {"BPSK": "BPSK", "QPSK": "QPSK", "8QAM": "8-QAM"}

for ax, mod in zip(axes, ("BPSK", "QPSK", "8QAM")):
    pool = []
    for i in meta.index[meta.modulation == mod]:
        w = wave[wo[i]:wo[i + 1]].reshape(-1, SPS).astype(np.float64)
        n = np.arange(SPS)
        s = (w * np.exp(-1j * 2 * np.pi * M.CYCLES_PER_SYMBOL
                        * (n + 0.5) / SPS)[None, :]).mean(axis=1)
        s = M.derotate(s, mod)
        pool.append(s / np.sqrt((np.abs(s) ** 2).mean()))   # unit RMS per capture
    z = np.concatenate(pool)
    sel = rng.choice(len(z), size=min(6000, len(z)), replace=False)
    ax.scatter(z[sel].real, z[sel].imag, s=1.2, c=BLUE, alpha=0.16,
               linewidths=0, rasterized=True, zorder=2)

    lim = 2.2
    ax.axhline(0, color=GRID, lw=0.6, zorder=1)
    ax.axvline(0, color=GRID, lw=0.6, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase")
    ax.set_title(f"{TITLE[mod]}  ({len(z):,} symbols)", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if mod == "8QAM":
        r = np.abs(z)
        med = np.median(r)
        ratio = r[r > med].mean() / r[r <= med].mean()
        for rad, st in ((r[r <= med].mean(), ":"), (r[r > med].mean(), "--")):
            th = np.linspace(0, 2 * np.pi, 256)
            ax.plot(rad * np.cos(th), rad * np.sin(th), st, color=ORANGE,
                    lw=0.9, zorder=3)
        ax.text(0, -lim * 0.88, f"measured radius ratio {ratio:.2f}",
                ha="center", fontsize=6.6, color=ORANGE)

axes[0].set_ylabel("Quadrature")
fig.text(0.5, -0.07,
         "All captures pooled across the five boards, normalised per capture to "
         "unit RMS. Same downconversion\nand derotation used to generate the "
         "training labels. No noise added.",
         ha="center", fontsize=6.5, color=INK2)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_constellation.{ext}"))
plt.close(fig)
print("fig_constellation.pdf / .png written")
