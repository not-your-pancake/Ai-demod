"""Dump every number the Results section quotes, so nothing is written from memory."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
import splits as SP
import model as MD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
cap = pd.read_csv(os.path.join(ROOT, "dataset", "captures.csv"))
exp = pd.read_csv(os.path.join(R, "experiments.csv"))
cls = pd.read_csv(os.path.join(R, "classical_snr.csv"))
abl = pd.read_csv(os.path.join(R, "ablation_physics.csv"))
main = exp[exp.control == "none"]
meta = SP.load_meta(ROOT)

print("=" * 70)
print("DATASET")
print("=" * 70)
print(f"captures            {len(cap)}")
print(f"boards              {cap.board.nunique()}  ({sorted(cap.board.unique())})")
print(f"symbols total       {len(d['label']):,}")
print(f"samples/symbol      {int(d['sps_out'])}")
for m in ("BPSK", "QPSK", "8QAM"):
    s = cap[cap.modulation == m]
    g = meta[meta.modulation == m]
    print(f"  {m:<5} {len(s):>2} captures  {int(s.n_symbols.sum()):>7,} symbols  "
          f"{s.pattern.nunique()} words  {g.group.nunique()} independent patterns")
print(f"\ntemplate SNR        {cap.template_snr.min():.1f} - {cap.template_snr.max():.1f} dB "
      f"(median {cap.template_snr.median():.1f})")
print(f"carrier (blind)     {cap.rms_mv.min():.0f} - {cap.rms_mv.max():.0f} mV TP-20 RMS")
print(f"symbol timing tau   BPSK {cap[cap.modulation=='BPSK'].tau.mean():.4f}  "
      f"QPSK {cap[cap.modulation=='QPSK'].tau.mean():.4f}")

print("\n" + "=" * 70)
print("MODEL")
print("=" * 70)
for m in ("BPSK", "QPSK", "8QAM"):
    c = M.MAX_RECEPTIVE_FIELD_SYMBOLS[m]
    net = MD.build(m, 20, c)
    print(f"  {m:<5} cap {c} symbol(s) = {c*M.BITS_PER_SYMBOL[m]} bits, "
          f"window {net.window} samples, RF {net.receptive_field()}, "
          f"{net.n_parameters():,} params")

print("\n" + "=" * 70)
print("SER BY SPLIT  (mean over folds)")
print("=" * 70)
SN = ["snr0", "snr3", "snr6", "snr9"]
for m in ("BPSK", "QPSK", "8QAM"):
    print(f"\n{m}   chance {1-1/M.n_classes(m):.3f}")
    for sp in ("leave_one_pattern_out", "leave_one_board_out",
               "leave_one_attenuation_out", "random"):
        r = main[(main.modulation == m) & (main.split == sp)]
        if not len(r):
            continue
        print(f"  {sp:<26} n={len(r)}  " +
              "  ".join(f"{c[3:]}dB {r[c].mean():.4f}" for c in SN))
    c = cls[cls.modulation == m].iloc[0]
    print(f"  {'classical (blind)':<26} n=-  " +
          "  ".join(f"{s[3:]}dB {c[s]:.4f}" for s in SN))

print("\n" + "=" * 70)
print("CONTROLS")
print("=" * 70)
for m in ("BPSK", "QPSK", "8QAM"):
    ch = 1 - 1 / M.n_classes(m)
    sh = exp[(exp.control == "shuffled_labels") & (exp.modulation == m)]["clean"].iloc[0]
    po = main[(main.modulation == m) &
              (main.split == "leave_one_pattern_out")]["position_only_ser"].mean()
    md = main[(main.modulation == m) &
              (main.split == "leave_one_pattern_out")]["snr6"].mean()
    print(f"  {m:<5} chance {ch:.3f} | shuffled {sh:.4f} | position-only {po:.4f} "
          f"| model@6dB {md:.4f}")

print("\n" + "=" * 70)
print("PHYSICS ABLATION")
print("=" * 70)
for _, r in abl.iterrows():
    print(f"  {r.modulation:<5} lambda {r.lam:>4.1f}  "
          f"0dB {r.snr0:.4f}  3dB {r.snr3:.4f}  6dB {r.snr6:.4f}")

print("\n" + "=" * 70)
print("ERROR GEOMETRY (8-QAM, 0 dB)")
print("=" * 70)
conf = np.loadtxt(os.path.join(R, "confusion_8QAM_0dB.csv"), delimiter=",")
tot = conf.sum() - np.trace(conf)
print(f"  total errors {int(tot):,} of {int(conf.sum()):,} symbols "
      f"(SER {tot/conf.sum():.4f})")
print(f"  zero-count long-distance pairs: "
      f"{int((conf - np.diag(np.diag(conf)) == 0).sum() - 8)} of 56 off-diagonal cells")

print("\n" + "=" * 70)
print("THIN FOLDS TO DECLARE")
print("=" * 70)
for _, r in main[main.n_train_captures <= 2].iterrows():
    print(f"  {r.modulation:<5} {r.split:<24} {r.fold:<16} "
          f"{r.n_train_captures} training capture(s), {r.n_train:,} symbols")
