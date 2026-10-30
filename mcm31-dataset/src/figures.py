"""
figures.py - publication figures for the MCM31 neural demodulator paper.

Design decisions and why:
  * Vector PDF for the journal, PNG alongside for viewing.
  * Identity is NEVER colour-alone: every series carries a distinct line style
    AND marker, so the figures survive greyscale printing and CVD readers.
  * Log SER axis. A measured SER of exactly zero cannot be drawn on a log
    scale, so those points are plotted as hollow down-triangles on a fixed
    floor line set below every non-zero value, and called out in the caption
    as "no errors observed" - never silently clipped to a small positive
    number. The floor is a plotting convention, not a confidence bound.
  * Only 0-9 dB is plotted for the main result; that is where the receivers
    differ. Above it both make a handful of errors in tens of thousands.
  * Figures 1-3 use the protocol of the revised paper (2026-10-30): learned
    receiver over three training seeds (results/revision_runs.csv), classical
    reference on the same leave-one-pattern-out records
    (results/classical_lopo.csv), shuffled-label control over five
    permutation seeds (results/experiments.csv).

Palette: validated categorical slots 1-3 (blue/orange/aqua), all-pairs pass.
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
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
})

exp = pd.read_csv(os.path.join(ROOT, "results", "experiments.csv"))
rev = pd.read_csv(os.path.join(ROOT, "results", "revision_runs.csv"))
cls = pd.read_csv(os.path.join(ROOT, "results", "classical_lopo.csv"))
main = exp[exp.control == "none"]
runs = pd.concat([main, rev[rev.kind == "seed"]], ignore_index=True)   # seeds 0-2
SNRS = [0, 3, 6, 9]
COLS = [f"snr{s}" for s in SNRS]
MODS = ["BPSK", "QPSK", "8QAM"]
TITLE = {"BPSK": "BPSK", "QPSK": "QPSK", "8QAM": "8-QAM"}
FLOOR = 1.5e-6   # below every non-zero mean (one error in one of three seeds ~3e-6)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def seed_curve(mod, split, drop_fold=None):
    """Fold-mean SER per seed -> mean, min, max over seeds, per SNR."""
    d = runs[(runs.modulation == mod) & (runs.split == split)]
    if drop_fold:
        d = d[d.fold != drop_fold]
    g = d.groupby("seed")[COLS].mean()
    return g.mean().to_numpy(), g.min().to_numpy(), g.max().to_numpy()


def plot_series(ax, x, y, colour, style, marker, label, dx=0.0, lo=None, hi=None):
    """Draw a SER curve; zeros go on the floor line as hollow triangles.

    `dx` shifts the series sideways. Without it two series that both reach
    zero at the same SNR stack exactly on top of each other and one silently
    disappears - which reads as missing data, not as a tie. `lo`/`hi` draw a
    vertical bar spanning the runs.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float) + dx
    pos = y > 0
    zero = ~pos
    ax.plot(x[pos], y[pos], style, color=colour, marker=marker,
            ms=4.5, lw=1.6, mew=0, label=label, zorder=3)
    if lo is not None:
        ok = pos & (np.asarray(lo) > 0)
        ax.vlines(x[ok], np.asarray(lo)[ok], np.asarray(hi)[ok], color=colour, lw=0.9, zorder=3)
    if zero.any():
        ax.plot(x[zero], np.full(int(zero.sum()), FLOOR), "v",
                mfc="none", mec=colour, ms=5, mew=1.2, ls="none", zorder=3)


# ---------------------------------------------------------------- figure 1
print("figure 1 - SER vs SNR, learned vs classical reference")
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True)
for ax, mod in zip(axes, MODS):
    m, lo, hi = seed_curve(mod, "leave_one_pattern_out")
    plot_series(ax, SNRS, m, BLUE, "-", "o", "Learned (this work)", dx=-0.22, lo=lo, hi=hi)
    c = cls[cls["mod"] == mod].set_index("snr")
    plot_series(ax, SNRS, [c.loc[s, "macro_mean"] for s in SNRS], ORANGE, "--", "s",
                "Classical reference", dx=+0.22)
    chance = 1 - 1 / M.n_classes(mod)
    ax.axhline(chance, color=INK2, lw=0.7, ls=":", zorder=1)
    ax.text(-0.5, chance * 1.35, "chance", va="bottom", ha="left",
            fontsize=6.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1.6)
    ax.set_xlim(-0.8, 9.8)
    ax.set_xticks(SNRS)
    ax.set_title(TITLE[mod], color=INK)
    ax.set_xlabel("SNR (dB)")
    ax.grid(True, which="major", axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Symbol error rate")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.13), handlelength=2.2)
fig.text(0.5, -0.30,
         "Hollow triangles: no errors observed, drawn on a fixed floor line (a visual "
         "convention, not a bound).\nLearned: mean of 3 training seeds, bars span the "
         "seeds. Classical: mean of 5 noise realisations.\nSoftware AWGN on hardware "
         "captures; leave-one-pattern-out, unweighted mean over folds.",
         ha="center", fontsize=6.3, color=INK2)
save(fig, "fig1_ser_vs_snr")

# ---------------------------------------------------------------- figure 2
print("figure 2 - 8-QAM by split")
fig, ax = plt.subplots(figsize=(3.5, 2.7))
q = runs[runs.modulation == "8QAM"]
rnd = pd.concat([q[q.split == "random"][COLS], rev[rev.kind == "randsplit"][COLS]])
plot_series(ax, SNRS, rnd.mean(), AQUA, ":", "^", f"Random 80/20 ({len(rnd)} runs)",
            dx=-0.15, lo=rnd.min(), hi=rnd.max())
m, lo, hi = seed_curve("8QAM", "leave_one_board_out")
plot_series(ax, SNRS, m, ORANGE, "--", "s", "Leave-one-board-out", dx=0.15, lo=lo, hi=hi)
m, lo, hi = seed_curve("8QAM", "leave_one_pattern_out")
plot_series(ax, SNRS, m, BLUE, "-", "o", "Leave-one-pattern-out", lo=lo, hi=hi)
# The two qualifications the text rests on: the board gap is one board (B5),
# and the pattern gap is one thin fold (T1+T2 trains on four records).
ax.plot(SNRS, seed_curve("8QAM", "leave_one_board_out", "board=B5")[0], "-.",
        color=ORANGE, lw=1.0, alpha=0.7, label="Leave-one-board-out, B5 excluded")
ax.plot(SNRS, seed_curve("8QAM", "leave_one_pattern_out", "group=T1+T2")[0], "-.",
        color=BLUE, lw=1.0, alpha=0.7, label="Leave-one-pattern-out, T3–T6 only")
ax.set_yscale("log")
ax.set_ylim(1e-5, 0.6)
ax.set_xticks(SNRS)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Symbol error rate")
ax.set_title("8-QAM symbol error rate by evaluation split", fontsize=8)
ax.grid(True, which="major", axis="y", alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left", handlelength=2.2, fontsize=6.4)
save(fig, "fig2_no_memorisation")

# ---------------------------------------------------------------- figure 3
print("figure 3 - controls on leave-one-pattern-out")
fig, ax = plt.subplots(figsize=(3.6, 2.7))
x = np.arange(3)
w = 0.26
lopo = runs[runs.split == "leave_one_pattern_out"]
model = [lopo[lopo.modulation == m].groupby("seed").snr6.mean().mean() for m in MODS]
pos = [main[(main.modulation == m) &
            (main.split == "leave_one_pattern_out")]["position_only_ser"].mean()
       for m in MODS]
# Shuffled: mean over five permutation seeds of the fold-mean; bars span the
# five seeds. Single runs scatter much more widely - see the text.
shu = [exp[(exp.control == "shuffled_labels") & (exp.modulation == m)]
       .groupby("seed").clean.mean() for m in MODS]
ax.bar(x - w, model, w * 0.9, color=BLUE, label="Learned receiver (6 dB)", zorder=3)
ax.bar(x, pos, w * 0.9, color=ORANGE, label="Position-only (sees no signal)", zorder=3)
ax.bar(x + w, [s.mean() for s in shu], w * 0.9, color=AQUA,
       label="Shuffled training labels", zorder=3)
ax.errorbar(x + w, [s.mean() for s in shu],
            yerr=[[s.mean() - s.min() for s in shu], [s.max() - s.mean() for s in shu]],
            fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=5)
for i, m in enumerate(MODS):
    ch = 1 - 1 / M.n_classes(m)
    ax.plot([i - 1.55 * w, i + 1.55 * w], [ch, ch], color=INK, lw=1.2,
            ls=":", zorder=4)
ax.plot([], [], color=INK, lw=1.2, ls=":", label="Chance, 1 - 1/K")
for i, v in enumerate(model):
    lab = "0" if v == 0 else ("<0.0001" if v < 1e-4 else f"{v:.4f}")
    ax.text(i - w, v + 0.03, lab, ha="center", fontsize=6, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([TITLE[m] for m in MODS])
ax.set_ylabel("Symbol error rate")
ax.set_ylim(0, 1.05)
ax.set_title("Controls on leave-one-pattern-out folds", fontsize=8)
ax.grid(True, axis="y", alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.44),
          fontsize=6.6, ncol=2, columnspacing=1.0, handlelength=1.4)
save(fig, "fig3_controls")

# ---------------------------------------------------------------- figure 4
print("figure 4 - hardware lock threshold across boards")
# Measured TP-20 RMS vs TP-9 agreement with PRBS63 (src/diagnose_tp9.py)
LOCK = {"B1": [(655.1, 1.000), (500.2, 0.667), (329.0, 0.651)],
        "B2": [(391.2, 0.667), (315.3, 0.667), (191.8, 0.714)],
        "B3": [(650.5, 1.000), (502.5, 0.651), (319.3, 0.635)],
        "B4": [(662.8, 0.921), (508.7, 0.730), (315.4, 0.683)],
        "B5": [(626.1, 1.000), (485.2, 0.698), (295.0, 0.683)]}
fig, ax = plt.subplots(figsize=(3.6, 2.7))
marks = ["o", "s", "^", "D", "v"]
for (b, pts), mk in zip(LOCK.items(), marks):
    a = np.array(pts)
    ax.plot(a[:, 0], a[:, 1] * 100, mk, ms=5, mfc="none", mec=BLUE, mew=1.2,
            ls="none", label=b, zorder=3)
ax.axhspan(0, 69.8, color=ORANGE, alpha=0.09, zorder=1)
ax.axhline(69.8, color=ORANGE, lw=0.9, ls="--", zorder=2)
ax.text(702, 66.5, "chance (95th pct)", fontsize=6.3, color=ORANGE,
        ha="right", va="top")
ax.axvspan(508.7, 626.1, color=GRID, alpha=0.6, zorder=0)
ax.text(567, 84, "lock threshold\n509-626 mV", ha="center", fontsize=6.3,
        color=INK2)
ax.set_xlabel("TP-20 RMS (mV)")
ax.set_ylabel("TP-9 agreement with PRBS63 (%)")
ax.set_title("Demodulator lock threshold, 5 boards", fontsize=8)
# Non-zero baseline is fine here: this is a scatter of an agreement rate, not
# a bar chart, and all the data lives between 63% and 100%.
ax.set_ylim(55, 106)
ax.set_xlim(150, 710)
ax.grid(True, axis="y", alpha=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.34),
          fontsize=6.5, columnspacing=0.9, handletextpad=0.1)
save(fig, "fig4_lock_threshold")

print(f"\nwritten to {FIG}")


# ---------------------------------------------------------------- figure 5
print("figure 5 - errors follow the constellation geometry")
import train as T  # noqa: E402  (constellation table)

conf = np.loadtxt(os.path.join(ROOT, "results", "confusion_8QAM_0dB.csv"),
                  delimiter=",")
C = T.constellation("8QAM").numpy()
z = C[:, 0] + 1j * C[:, 1]
n = len(z)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 3.0))

# -- panel A: the constellation, with confusion flows drawn on it -----------
pair = {}
for i in range(n):
    for j in range(i + 1, n):
        c = conf[i, j] + conf[j, i]
        if c > 0:
            pair[(i, j)] = c
mx = max(pair.values())
for (i, j), c in sorted(pair.items(), key=lambda kv: kv[1]):
    if c < 5:
        continue
    axA.plot([z[i].real, z[j].real], [z[i].imag, z[j].imag],
             color=ORANGE, lw=0.4 + 3.4 * (c / mx), alpha=0.75,
             solid_capstyle="round", zorder=2)
axA.scatter(z.real, z.imag, s=64, facecolor="white", edgecolor=BLUE,
            linewidth=1.6, zorder=3)
for k in range(n):
    axA.annotate(str(k), (z[k].real, z[k].imag), ha="center", va="center",
                 fontsize=6.2, color=INK, zorder=4)
lim = 1.55
axA.set_xlim(-lim, lim)
axA.set_ylim(-lim, lim)
axA.set_aspect("equal")
axA.set_xlabel("I")
axA.set_ylabel("Q")
axA.set_title("Where the errors go (8-QAM, 0 dB)", fontsize=8)
axA.grid(True, alpha=0.5)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.text(0, -lim * 0.93, "line width proportional to confusion count",
         ha="center", fontsize=6.3, color=INK2)

# -- panel B: confusion count against geometric distance -------------------
d_obs, c_obs = [], []
for i in range(n):
    for j in range(n):
        if i != j:
            d_obs.append(abs(z[i] - z[j]))
            c_obs.append(conf[i, j])
d_obs = np.round(np.array(d_obs), 3)
c_obs = np.array(c_obs, dtype=float)
total = c_obs.sum()
levels = np.unique(d_obs)
obs = np.array([c_obs[d_obs == d].sum() for d in levels]) / total
unif = np.array([(d_obs == d).sum() for d in levels]) / len(d_obs)

w = 0.055
axB.bar(levels - w / 2, obs * 100, w, color=BLUE, label="Observed", zorder=3)
axB.bar(levels + w / 2, unif * 100, w, color=ORANGE,
        label="If errors were random", zorder=3)
axB.set_xlabel("Distance between constellation points")
axB.set_ylabel("Share of all errors (%)")
axB.set_title("Errors concentrate at the shortest hop", fontsize=8)
axB.grid(True, axis="y", alpha=0.7)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.set_ylim(0, 92)
axB.legend(frameon=False, loc="upper right", fontsize=7,
           bbox_to_anchor=(1.02, 0.80))
axB.annotate("ring error\n(same phase)", xy=(levels[0] + 0.03, obs[0] * 100 * 0.62),
             xytext=(levels[0] + 0.34, obs[0] * 100 * 0.62),
             fontsize=6.3, color=INK2, va="center",
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
fig.subplots_adjust(wspace=0.34)
fig.text(0.5, -0.14,
         "79.5% of errors land on the nearest constellation point, against a "
         "14.3% chance level (5.6x).\nThe classifier loss never supplies the "
         "constellation geometry - it is recovered from the data.",
         ha="center", fontsize=6.5, color=INK2)
save(fig, "fig5_error_geometry")
