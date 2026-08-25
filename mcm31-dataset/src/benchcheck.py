"""
benchcheck.py - fast check to run AT THE BENCH, right after saving a file.

Takes about 10 seconds. Catches every mistake that has actually happened on
this project so far:

    CH2 switched off          -> wrong column count
    wrong time base           -> wrong row count
    CH1 not re-ranged         -> too few voltage levels
    data not loaded / no START-> unmodulated carrier, or the wrong word

Run it before you move on to the next file, not after you get home.

Usage:
    <venv>\\python.exe src\\benchcheck.py raw\\B1\\B1_BPSK_AMID.CSV
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN_TAGS = {"Q2", "Q3", "T2", "T3", "S1",
                "T4", "T5", "T6", "Q4", "Q5"}
DEFAULT_PATTERN = {"BPSK": "PRBS63", "QPSK": "Q1", "8QAM": "T1"}

GREEN, RED, YELL, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1] if os.path.isabs(argv[1]) else os.path.join(ROOT, argv[1])
    if not os.path.exists(path):
        print(f"{RED}file not found:{RESET} {path}")
        return 2

    stem = os.path.splitext(os.path.basename(path))[0].upper()
    parts = stem.split("_")
    mod = next((p for p in parts if p in M.BITS_PER_SYMBOL), None)
    if mod is None:
        print(f"{RED}cannot tell the modulation from the filename:{RESET} {stem}")
        return 2
    tag = parts[-1] if len(parts) > 2 else ""
    pattern = tag if tag in PATTERN_TAGS else DEFAULT_PATTERN[mod]

    print(f"\n  {os.path.basename(path)}   ({mod}, expecting pattern {pattern})")
    print("  " + "-" * 62)

    problems = []

    def report(label, value, ok, hint=""):
        colour = GREEN if ok else RED
        mark = "ok  " if ok else "BAD "
        print(f"  {colour}{mark}{RESET} {label:<22} {value}")
        if not ok:
            problems.append(hint)

    cap = M.load_capture(path)

    report("columns", cap.meta["n_columns"], cap.meta["n_columns"] >= 3,
           "CH2 was switched off - turn CH2 on and recapture")
    report("rows", f"{cap.n:,}", 520_000 <= cap.n <= 526_000,
           "wrong time base - set 500 ms/div, Max Sa Rate, recapture")

    levels = int(np.unique(cap.ch1).size)
    pkpk = float(cap.ch1.max() - cap.ch1.min())
    report("CH1 levels", f"{levels}  (pk-pk {pkpk:.2f} V)", levels >= 90,
           "CH1 too small on screen - turn CH1 VOLTS/DIV until the wave fills "
           "about 6 of the 8 squares, then recapture")

    fc = M.estimate_carrier_4th(cap.ch1, cap.fs)
    report("carrier", f"{fc:.2f} Hz", 1199.5 <= fc <= 1200.5,
           "carrier is wrong - check SW6 / J3 and the modulation setting")

    # does the waveform actually repeat at the expected word period? This is
    # what catches a wrong SW3: with SW3 on 24 BIT a BPSK capture repeats every
    # 24 (or 48) bits, not every 63, and every later check then reads garbage.
    n_sym = M.symbols_per_word(pattern, mod)
    period = M.word_period_seconds(pattern, mod) * cap.fs
    lag = int(np.floor(period)) + 1
    idx = np.arange(cap.n - lag, dtype=np.float64)
    a = cap.ch1[: len(idx)]
    b = np.interp(idx + period, np.arange(cap.n, dtype=np.float64), cap.ch1)
    a = a - a.mean(); b = b - b.mean()
    rep = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    sw3 = "64 BIT" if pattern == "PRBS63" else "24 BIT"
    report("word period repeats", f"r = {rep:.3f}  ({period / cap.fs * 1e3:.2f} ms)",
           rep >= 0.95,
           f"the signal does NOT repeat at the {pattern} period - SW3 is almost "
           f"certainly on the wrong setting. For this file SW3 must be {sw3}. "
           f"Set it, PRESS START, and recapture")

    # the decisive one: is the carrier actually carrying data?
    tmpl = M.coherent_template(cap.ch1, period)
    spec = np.abs(np.fft.rfft(tmpl.values - tmpl.values.mean())) ** 2
    car_bin = M.CYCLES_PER_SYMBOL * n_sym
    sidebands = (spec.sum() - spec[car_bin - 1:car_bin + 2].sum()) / spec.sum() * 100
    report("modulation depth", f"{sidebands:.1f}% in sidebands", sidebands >= 50,
           "THE CARRIER IS BARE - no data is reaching the modulator. Check SW3 "
           "(64 BIT for BPSK, 24 BIT otherwise), J1 and SW8, then PRESS START "
           "and recapture")

    sym = M.derotate(M.extract_symbols(tmpl.values, n_sym), mod)
    ident = M.identify_pattern(sym, mod)
    report("pattern fingerprint", f"{ident['observed']} -> "
           f"{ident['consistent_with'] or 'NOTHING'}",
           pattern in ident["consistent_with"],
           f"the loaded word is not {pattern} - reset the DIP switches and PRESS START")

    print("  " + "-" * 62)
    if problems:
        print(f"\n  {RED}>>> RECAPTURE THIS FILE{RESET}")
        for p in dict.fromkeys(problems):
            print(f"      - {p}")
        print()
        return 1
    print(f"\n  {GREEN}>>> GOOD - go on to the next file{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
