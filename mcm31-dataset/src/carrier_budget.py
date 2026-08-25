"""
carrier_budget.py - how much signal does the classical receiver need BEFORE it
can demodulate anything, and how accurate must its carrier estimate be?

This replaces the weaker "hand it a wrong number and watch it break" test. A
reviewer can dismiss that with "a tracking loop would follow it". What a
tracking loop cannot do is acquire from nothing: the classical receiver must
estimate fc from some observation window before the first symbol decision, and
that estimate has an error that shrinks with observation length.

Two curves, both measured:
  A. carrier-estimate error vs observation length
  B. SER vs carrier error  (the tolerance the receiver actually has)

Together they give the number that matters: the shortest observation from which
a classical receiver can demodulate at all. The learned receiver needs no
estimate, so its requirement is one window - 20 samples for 8-QAM, 1.7 ms.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
from classical import blind_carrier, demodulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
meta = pd.read_csv(os.path.join(ROOT, "dataset", "captures.csv"))
SPS = int(d["sps_out"]); wave, label = d["wave"], d["label"]
wo, lo = d["wave_offset"], d["label_offset"]
fs = SPS * M.SYMBOL_RATE

def ser(pred, y, mod):
    K = 2 if mod == "BPSK" else 4
    best = 0.0
    for k in range(K):
        p = (pred + k) % 2 if mod == "BPSK" else (
            (pred + k) % 4 if mod == "QPSK" else (pred % 4 + k) % 4 + 4 * (pred // 4))
        best = max(best, float((p == y).mean()))
    return 1 - best

print("A) CARRIER-ESTIMATE ERROR vs OBSERVATION LENGTH")
print(f"   {'symbols':>8} {'ms':>8} {'median |err| Hz':>17}")
print("   " + "-" * 36)
rowsA = []
for nsym in (8, 16, 32, 64, 128, 256, 512, 1024, 3596):
    errs = []
    for i in meta.index[meta.modulation == "QPSK"][:6]:
        w = wave[wo[i]:wo[i] + nsym * SPS]
        try:
            errs.append(abs(blind_carrier(w, SPS, fs) - M.FC_NOMINAL))
        except Exception:
            pass
    e = float(np.median(errs))
    rowsA.append(dict(symbols=nsym, ms=nsym / M.SYMBOL_RATE * 1e3, err_hz=e))
    print(f"   {nsym:>8} {nsym / M.SYMBOL_RATE * 1e3:>8.1f} {e:>17.4f}")

print("\nB) SER vs CARRIER ERROR  (full capture, clean)")
print(f"   {'|df| Hz':>10} {'BPSK':>9} {'QPSK':>9} {'8QAM':>9}")
print("   " + "-" * 40)
rowsB = []
for df in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    line = {}
    for mod in ("BPSK", "QPSK", "8QAM"):
        s = []
        for i in meta.index[meta.modulation == mod][:4]:
            w = wave[wo[i]:wo[i+1]]; y = label[lo[i]:lo[i+1]]
            s.append(ser(demodulate(w, mod, SPS, M.FC_NOMINAL + df, fs), y, mod))
        line[mod] = float(np.mean(s))
    rowsB.append(dict(df_hz=df, **line))
    print(f"   {df:>10.3f} {line['BPSK']:>9.4f} {line['QPSK']:>9.4f} {line['8QAM']:>9.4f}")

print("\nC) END TO END - estimate fc from N symbols, then demodulate THOSE symbols")
print(f"   {'symbols':>8} {'ms':>8} {'BPSK':>9} {'QPSK':>9} {'8QAM':>9}")
print("   " + "-" * 48)
rowsC = []
for nsym in (8, 16, 32, 64, 128, 256, 512, 1024, 3596):
    line = {}
    for mod in ("BPSK", "QPSK", "8QAM"):
        s = []
        for i in meta.index[meta.modulation == mod][:4]:
            w = wave[wo[i]:wo[i] + nsym * SPS]
            y = label[lo[i]:lo[i] + nsym]
            try:
                fc = blind_carrier(w, SPS, fs)
                s.append(ser(demodulate(w, mod, SPS, fc, fs), y, mod))
            except Exception:
                s.append(np.nan)
        line[mod] = float(np.nanmean(s))
    rowsC.append(dict(symbols=nsym, ms=nsym / M.SYMBOL_RATE * 1e3, **line))
    print(f"   {nsym:>8} {nsym / M.SYMBOL_RATE * 1e3:>8.1f} "
          f"{line['BPSK']:>9.4f} {line['QPSK']:>9.4f} {line['8QAM']:>9.4f}")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
pd.DataFrame(rowsA).to_csv(os.path.join(ROOT,"results","carrier_est_vs_length.csv"),index=False)
pd.DataFrame(rowsB).to_csv(os.path.join(ROOT,"results","ser_vs_carrier_error.csv"),index=False)
pd.DataFrame(rowsC).to_csv(os.path.join(ROOT,"results","classical_vs_length.csv"),index=False)
print("\n   learned receiver, for comparison: 1 window = "
      f"{1/M.SYMBOL_RATE*1e3:.2f} ms for 8-QAM, no carrier estimate at all")
