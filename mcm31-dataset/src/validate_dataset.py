"""
validate_dataset.py - prove the built dataset is fit for training.

Two separate questions, which the first version of this script wrongly mixed:

  A. ALIGNMENT - do the stored labels describe the stored waveform? Tested by
     decoding the decimated waveform with a plain textbook demodulator and
     comparing. Class NAMING is allowed to differ by a constellation rotation,
     because the demodulator here uses its own phase reference.

  B. CONSISTENCY - does the same physical symbol carry the same label in every
     capture? This is the one that matters for training: if capture A calls a
     45-degree symbol "class 0" and capture B calls it "class 1", the network
     gets contradictory targets. Tested by requiring the rotation found in (A)
     to be IDENTICAL across every capture of a modulation.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
meta = pd.read_csv(os.path.join(ROOT, "dataset", "captures.csv"))
SPS = int(d["sps_out"]); wave, label = d["wave"], d["label"]
wo, lo = d["wave_offset"], d["label_offset"]
fail = []

print(f"loaded {len(label):,} symbols, {SPS} samples/symbol, {len(meta)} captures\n")

w0 = wave[wo[0]:wo[1]].reshape(-1, SPS)
sp = np.abs(np.fft.rfft(w0 - w0.mean(axis=1, keepdims=True), axis=1)).mean(axis=0)
print(f"carrier survived decimation: per-symbol spectrum peaks at bin "
      f"{int(np.argmax(sp))} (expect {M.CYCLES_PER_SYMBOL})\n")

def rot_class(y, k, mod):
    if mod == "BPSK": return (y + k) % 2
    if mod == "QPSK": return (y + k) % 4
    return (y % 4 + k) % 4 + 4 * (y // 4)

rows = []
for i, r in meta.iterrows():
    w = wave[wo[i]:wo[i+1]].reshape(-1, SPS).astype(np.float64)
    y = label[lo[i]:lo[i+1]]; mod = r["modulation"]
    n = np.arange(SPS)
    s = M.derotate((w * np.exp(-1j*2*np.pi*M.CYCLES_PER_SYMBOL*(n+0.5)/SPS)[None,:])
                   .mean(axis=1), mod)
    p = M.quantise_symbols(s, mod)
    K = 2 if mod == "BPSK" else 4
    best, k = max((float((rot_class(p, j, mod) == y).mean()), j) for j in range(K))
    rows.append(dict(f=r["filename"], mod=mod, agree=best, rot=k))

print("A) ALIGNMENT - labels vs an independent textbook decode")
worst = min(rows, key=lambda x: x["agree"])
print(f"   lowest agreement over all {len(rows)} captures: "
      f"{worst['agree']*100:.2f}%  ({worst['f']})")
if worst["agree"] < 0.99:
    fail.append(f"alignment: {worst['f']} at {worst['agree']*100:.2f}%")
    for x in sorted(rows, key=lambda x: x["agree"])[:6]:
        print(f"     {x['f']:<20} {x['agree']*100:6.2f}%")
print(f"   {'PASS' if worst['agree'] >= 0.99 else 'FAIL'}\n")

print("B) CONSISTENCY - class naming identical across captures")
for mod in ("BPSK", "QPSK", "8QAM"):
    ks = sorted({x["rot"] for x in rows if x["mod"] == mod})
    ok = len(ks) == 1
    print(f"   {mod:<5} rotation needed by each capture: {ks}   "
          f"{'PASS' if ok else 'FAIL - same symbol has different labels'}")
    if not ok:
        fail.append(f"consistency: {mod} labels differ by rotation across captures")
print()

print("class balance")
for mod in ("BPSK", "QPSK", "8QAM"):
    idx = np.concatenate([np.arange(lo[i], lo[i+1])
                          for i in meta.index[meta.modulation == mod]])
    c = np.bincount(label[idx], minlength=M.n_classes(mod))
    print(f"   {mod:<5} {c.tolist()}   rarest/commonest = {c.min()/c.max():.3f}")

print("\nreceptive-field budget")
for mod, cap in M.MAX_RECEPTIVE_FIELD_SYMBOLS.items():
    bits = cap * M.BITS_PER_SYMBOL[mod]
    print(f"   {mod:<5} <= {cap} symbol(s) = {cap*SPS:>3} samples, "
          f"{bits} bits visible  {'OK' if bits <= 5 else 'UNSAFE'}")

print()
if fail:
    print(f"{len(fail)} FAILURE(S):")
    for f in fail: print(f"  - {f}")
    raise SystemExit(1)
print("PASS - labels describe the waveform, and the same physical symbol")
print("       carries the same label in every capture. Ready for training.")
