"""
Does the DIP switch word map to phase the same way the on-board PRBS63 does?

Handbook Section 10 says the switch -> symbol mapping is unresolved. That test
was run on QPSK/8-QAM, where bit-grouping adds a second unknown. B1_BPSK_S1.CSV
is the clean case: a 24-bit DIP word sent as plain BPSK, one bit per phase.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "raw", "B1", "B1_BPSK_S1.CSV")

cap = M.load_capture(path)
period = M.refine_period(cap.ch1, M.word_period_seconds("S1", "BPSK") * cap.fs)
tmpl = M.coherent_template(cap.ch1, period)
sym, tau = M.extract_symbols(tmpl.values, 24, return_tau=True)
sym = M.derotate(sym, "BPSK")
bits = (sym.real < 0).astype(np.int8)

print(f"word period : {period:.2f} samples = {period/cap.fs*1e3:.3f} ms "
      f"(expected 40.003)")
print(f"template    : K = {tmpl.n_periods} periods, SNR = {tmpl.snr_db:.1f} dB")
print(f"timing tau  : {tau:.4f} symbols")
print(f"recovered   : {''.join(map(str, bits))}")
print(f"S1 on paper : {M.PATTERNS['S1']}")

print("\nMatched against every known 24-bit word (24 rotations x 2 polarities):")
print(f"  {'word':<6} {'score':>7} {'rot':>5} {'inv':>6}   {'chance p95':>10}")
print("  " + "-" * 46)
results = []
for pid in M.PATTERNS:
    m = M.match_word(bits, pid)
    results.append(m)
    star = "  <-- MATCH" if m["above_chance"] else ""
    print(f"  {pid:<6} {m['score']*100:>6.1f}% {m['rotation']:>5} "
          f"{str(m['inverted']):>6}   {m['chance_p95']*100:>9.1f}%{star}")

best = max(results, key=lambda r: r["score"])
runner = sorted(results, key=lambda r: -r["score"])[1]
print(f"\nbest        : {best['pattern']} at {best['score']*100:.1f}%")
print(f"runner-up   : {runner['pattern']} at {runner['score']*100:.1f}%")
print(f"chance      : {best['chance_mean']*100:.1f}% mean, "
      f"{best['chance_p95']*100:.1f}% p95")

if best["pattern"] == "S1" and best["score"] == 1.0:
    print("\n=> S1 recovered EXACTLY. The DIP word maps to phase directly.")
    print("   This RESOLVES the BPSK half of the switch-to-symbol mapping.")
elif best["score"] > best["chance_p95"]:
    print(f"\n=> partial match only - {best['score']*100:.1f}% is above chance "
          f"but not exact.")
else:
    print("\n=> nothing above chance. Mapping stays unresolved, as the handbook says.")
