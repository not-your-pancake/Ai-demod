"""
classical_snr.py - the blind classical receiver's SER vs SNR, measured with
exactly the same noise the network sees, so the two curves are comparable.

Noise is drawn from the CAPTURE reference level, identical to data.py, and the
carrier is estimated blind from the noisy waveform - the receiver is given no
more than the network is.
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
SNRS = [0, 3, 6, 9, 12, 15, 20, 25, 30, None]

def ser(pred, y, mod):
    K = 2 if mod == "BPSK" else 4
    best = 0.0
    for k in range(K):
        p = ((pred + k) % 2 if mod == "BPSK" else
             (pred + k) % 4 if mod == "QPSK" else
             (pred % 4 + k) % 4 + 4 * (pred // 4))
        best = max(best, float((p == y).mean()))
    return 1 - best

rng = np.random.default_rng(0)
rows = []
print("BLIND CLASSICAL RECEIVER - SER vs SNR (mean over all captures)")
print(f"  {'mod':<6} " + "".join(f"{('clean' if s is None else f'{s}dB'):>9}" for s in SNRS))
print("  " + "-" * 96)
for mod in ("BPSK", "QPSK", "8QAM"):
    sel = meta.index[meta.modulation == mod]
    line = {}
    for snr in SNRS:
        s = []
        for i in sel:
            w = wave[wo[i]:wo[i+1]].astype(np.float64)
            y = label[lo[i]:lo[i+1]]
            ref = np.sqrt((w ** 2).mean())
            wn = w if snr is None else w + rng.standard_normal(len(w)) * ref / np.sqrt(10 ** (snr / 10))
            fc = blind_carrier(wn, SPS, fs)          # blind, from the NOISY signal
            s.append(ser(demodulate(wn, mod, SPS, fc, fs), y, mod))
        line["clean" if snr is None else f"snr{snr}"] = float(np.mean(s))
    rows.append(dict(modulation=mod, **line))
    print(f"  {mod:<6} " + "".join(f"{line['clean' if s is None else f'snr{s}']:>9.4f}" for s in SNRS))

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "classical_snr.csv"), index=False)
print(f"\n  chance SER: BPSK 0.500  QPSK 0.750  8-QAM 0.875")
print(f"  saved -> results/classical_snr.csv")
