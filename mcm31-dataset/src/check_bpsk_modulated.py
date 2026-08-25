"""Is the BPSK carrier actually being modulated, or is it a bare sine?

A modulated PRBS63 word pushes most of its energy into sidebands around the
carrier. An unmodulated carrier puts essentially all of it in one bin. This
separates "the board sent data" from "the board sent nothing" with a single
number, and does not depend on the switch-to-symbol mapping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["B1_BPSK_A0.CSV", "B1_BPSK_AMID.CSV", "B1_BPSK_AMAX.CSV"]

print(f"{'file':<20} {'RMS mV':>7} {'carrier bin':>12} {'sidebands':>10} "
      f"{'radii spread':>13} {'fingerprint':>14}")
print("-" * 82)

for f in FILES:
    path = os.path.join(ROOT, "raw", "B1", f)
    cap = M.load_capture(path)
    period = M.refine_period(cap.ch1, M.word_period_seconds("PRBS63", "BPSK") * cap.fs)
    tmpl = M.coherent_template(cap.ch1, period)

    # one word holds exactly 126 carrier cycles (2 per symbol x 63 symbols),
    # so the carrier lands exactly on FFT bin 126 of the folded template
    spec = np.abs(np.fft.rfft(tmpl.values - tmpl.values.mean())) ** 2
    total = spec.sum()
    carrier = spec[126] / total * 100
    sidebands = (total - spec[125:128].sum()) / total * 100

    sym = M.derotate(M.extract_symbols(tmpl.values, 63), "BPSK")
    r = np.abs(sym)
    rms = np.sqrt(np.mean((cap.ch1 - cap.ch1.mean()) ** 2)) * 1000

    print(f"{f:<20} {rms:>7.1f} {carrier:>11.2f}% {sidebands:>9.2f}% "
          f"{r.max() / r.min():>13.3f} "
          f"{str(M.observed_point_counts(sym, 'BPSK')):>14}")

print()
print("Modulated PRBS63 -> most energy in the sidebands, fingerprint [31, 32].")
print("Bare carrier     -> ~all energy in one bin, fingerprint [0, 63], spread 1.000.")
