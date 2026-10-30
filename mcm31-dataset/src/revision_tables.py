"""
revision_tables.py - every number in the revised Tables VII, VIII and X, and the
text around them, from the result files. Nothing is typed in by hand.

Reads results/experiments.csv (seed-0 folds + shuffled-label control),
results/revision_runs.csv (extra seeds, partitions, latent check) and
results/classical_lopo.csv. Run after run_experiments.py, run_revision.py and
classical_lopo.py.

Aggregation: a "mean" is the unweighted mean of per-fold SERs within one run;
"+-" is the standard deviation of that mean across runs (seeds or partitions).
Error counts are pooled over folds so 0.0000 is never confused with no errors.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
exp = pd.read_csv(os.path.join(R, "experiments.csv"))
rev = pd.read_csv(os.path.join(R, "revision_runs.csv"))
cls = pd.read_csv(os.path.join(R, "classical_lopo.csv"))
main = exp[exp.control == "none"]
runs = pd.concat([main, rev[rev.kind == "seed"]], ignore_index=True)   # seeds 0, 1, 2
MODS = ("BPSK", "QPSK", "8QAM")
C4 = ["snr0", "snr3", "snr6", "snr9"]


def per_seed(df, cols):
    return df.groupby("seed")[cols].mean()


print("=" * 78)
print("TABLE VII - leave-one-pattern-out, learned (3 seeds) vs classical (5 realisations)")
print("=" * 78)
for mod in MODS:
    d = runs[(runs.modulation == mod) & (runs.split == "leave_one_pattern_out")]
    for snr in (0, 3, 6, 9, 12):
        c = f"snr{snr}"
        g = d.groupby("seed")[c].mean()
        err = d.groupby("seed").apply(lambda x: int((x[c] * x.n_test).round().sum()), include_groups=False)
        k = cls[(cls["mod"] == mod) & (cls.snr == snr)].iloc[0]
        print(f"  {mod:<5} {snr:>2} dB  L {g.mean():.6f} +- {g.std(ddof=1):.6f}  errors/seed "
              f"{err.tolist()}   C {k.macro_mean:.6f} +- {k.macro_sd:.6f}")

print("\n" + "=" * 78)
print("TABLE VIII - 8-QAM by split (LaTeX rows: name (folds, runs) & 0 & 3 & 6 & 9 dB)")
print("=" * 78)
q = runs[runs.modulation == "8QAM"]
rnd = pd.concat([q[q.split == "random"][C4], rev[rev.kind == "randsplit"][C4]])
lines = [
    (r"Random $80/20$ (1, 7)", rnd),
    (r"Leave-one-pattern-out (5, 3)", per_seed(q[q.split == "leave_one_pattern_out"], C4)),
    (r"\quad T3--T6 only (4, 3)",
     per_seed(q[(q.split == "leave_one_pattern_out") & (q.fold != "group=T1+T2")], C4)),
    (r"Leave-one-board-out (5, 3)", per_seed(q[q.split == "leave_one_board_out"], C4)),
    (r"\quad without B5 (4, 3)",
     per_seed(q[(q.split == "leave_one_board_out") & (q.fold != "board=B5")], C4)),
    (r"Leave-one-atten.-out (2, 1)", per_seed(q[q.split == "leave_one_attenuation_out"], C4)),
]
for name, r in lines:
    m, s = r.mean(), r.std(ddof=1)
    cells = [f"{m[c]:.4f}$\\pm${s[c]:.4f}" if len(r) > 1 else f"{m[c]:.4f}" for c in C4]
    print("  " + name + " & " + " & ".join(cells) + r" \\")
t = q[(q.split == "leave_one_pattern_out") & (q.fold == "group=T1+T2")]
print(f"  T1+T2 fold alone: 0 dB {t.snr0.mean():.4f} +- {t.snr0.std(ddof=1):.4f}")
print("  leave-one-board-out per board, mean of 3 seeds:")
print(q[q.split == "leave_one_board_out"].groupby("fold")[["snr0", "snr6", "snr9"]]
      .mean().round(4).to_string())

print("\n" + "=" * 78)
print("TABLE X - shuffled-label control (training labels permuted, test as measured)")
print("=" * 78)
sh = exp[exp.control == "shuffled_labels"]
for mod in MODS:
    d = sh[sh.modulation == mod]
    fm = d.groupby("seed").clean.mean()
    lopo = runs[(runs.modulation == mod) & (runs.split == "leave_one_pattern_out")]
    print(f"  {mod:<5} {len(d)} runs  mean {fm.mean():.3f} [{fm.min():.3f}, {fm.max():.3f}]  "
          f"single runs {d.clean.min():.3f}-{d.clean.max():.3f}  | "
          f"receiver clean {lopo.clean.max():.4f}, 6 dB {lopo.groupby('seed').snr6.mean().mean():.6f}")
print(f"  best shuffled run overall {sh.clean.min():.3f}")

print("\n" + "=" * 78)
print("LATENT HEAD (lambda 0.2, ablation fold) - one point for every class?")
print("=" * 78)
for _, r in rev[rev.kind == "latent"].iterrows():
    means = np.array(json.loads(r.latent_means))
    print(f"  {r.modulation:<5} class means span {np.ptp(means, axis=0).round(3).tolist()} "
          f"around {means.mean(0).round(2).tolist()}; max within-class spread "
          f"{max(json.loads(r.latent_spread)):.4f}")
