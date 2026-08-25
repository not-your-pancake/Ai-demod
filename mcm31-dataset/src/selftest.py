"""
selftest.py - prove the analysis pipeline is correct BEFORE any real capture.

This builds a SYNTHETIC capture with known ground truth, runs the real
pipeline over it, and asserts that the recovered bits are exactly PRBS63.

*** THE FILES THIS WRITES ARE NOT DATA. ***
They live in selftest/ and are never used for training, never quoted as a
result, and never mixed into raw/. Their only job is to fail loudly if
mcm31.py has a bug, so that a failure on the real capture points at the
hardware instead of at the code.

Run:  E:\\Ai demod\\.venv\\Scripts\\python.exe src\\selftest.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "selftest")


def make_synthetic(pattern_id: str, modulation: str, snr_db: float = 35.0,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a capture that obeys the handbook's measured facts:
    fs = 87260, fc = 1199.91, exactly 2 carrier cycles per symbol, 523600
    samples, 8-bit quantisation."""
    from scipy.signal import firwin, lfilter

    rng = np.random.default_rng(seed)
    fs, n = M.FS_NOMINAL, M.N_SAMPLES_NOMINAL
    bits = M.bits_of(pattern_id)
    bps = M.BITS_PER_SYMBOL[modulation]
    n_sym = len(bits) // bps
    sps = fs / M.SYMBOL_RATE

    # bits -> constellation points (handbook Section 2.5)
    grouped = bits.reshape(n_sym, bps)
    idx = np.array([int("".join(map(str, g)), 2) for g in grouped])
    if modulation == "BPSK":
        const = np.array([1.0 + 0j, -1.0 + 0j])
    elif modulation == "QPSK":
        const = np.exp(1j * np.deg2rad([45, 135, 225, 315]))
    else:  # 8-QAM: 4-phase, 2-amplitude STAR, radius ratio 1.965
        ph = np.deg2rad([45, 135, 225, 315])
        const = np.concatenate([np.exp(1j * ph), 1.965 * np.exp(1j * ph)])
    sym = const[idx % len(const)]

    # hold each symbol, allowing a fractional samples-per-symbol
    samp_idx = (np.arange(n) / sps).astype(np.int64) % n_sym
    base = sym[samp_idx]

    # mild pulse shaping, like the board's output filter
    taps = firwin(65, 1.5 * M.SYMBOL_RATE, fs=fs)
    base = lfilter(taps, [1.0], base.real) + 1j * lfilter(taps, [1.0], base.imag)

    t = np.arange(n) / fs
    carrier = np.exp(1j * 2 * np.pi * M.FC_NOMINAL * t)
    x = (base * carrier).real

    # additive noise at the requested per-sample SNR
    sigma = np.sqrt(x.var()) * 10 ** (-snr_db / 20.0)
    x = x + rng.normal(0.0, sigma, n)

    # 8-bit scope quantisation over a screen that the signal nearly fills
    full = 1.15 * np.abs(x).max()
    x = np.round(x / (2 * full / 255)) * (2 * full / 255)

    # CH2 = TP-9, the board's recovered data as a 0/5 V line, delayed a bit
    bit_sps = sps / bps
    bit_idx = (np.arange(n) / bit_sps).astype(np.int64) % len(bits)
    ch2 = bits[bit_idx].astype(np.float64) * 5.0
    ch2 = np.roll(ch2, int(0.4 * bit_sps))
    ch2 += rng.normal(0.0, 0.02, n)

    return t, x, ch2


def write_csv(path: str, t, ch1, ch2) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(M.EXPECTED_HEADER + "\n")
        np.savetxt(f, np.column_stack([t, ch1, ch2]), fmt="%.7g", delimiter=",")


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def main() -> int:
    failures = []

    # -- 0. the handbook's own PRBS claims ---------------------------------
    banner("0. Re-deriving the handbook's PRBS63 claims from the sequence")
    p = M.verify_prbs63()
    print(f"  length                  : {p['length']}          (claim: 63)")
    print(f"  ones / zeros            : {p['ones']} / {p['zeros']}      (claim: 31 / 32)")
    print(f"  run spectrum            : {p['run_spectrum']}")
    print(f"  number of runs          : {p['n_runs']}          (claim: 32)")
    print(f"  complement obeys LFSR   : {p['complement_satisfies_lfsr']}   (claim: True)")
    expect_spec = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1, 6: 1}
    for label, got, want in [
        ("length", p["length"], 63),
        ("ones", p["ones"], 31),
        ("zeros", p["zeros"], 32),
        ("run spectrum", p["run_spectrum"], expect_spec),
        ("n_runs", p["n_runs"], 32),
        ("lfsr", p["complement_satisfies_lfsr"], True),
    ]:
        if got != want:
            failures.append(f"PRBS63 {label}: got {got}, handbook says {want}")

    # -- word periods ------------------------------------------------------
    banner("1. Word periods recomputed from the symbol rate")
    for pat, mod, claim_ms in [("PRBS63", "BPSK", 105.0), ("S1", "BPSK", 40.0),
                               ("Q1", "QPSK", 20.0), ("T1", "8QAM", 13.33)]:
        got = M.word_period_seconds(pat, mod) * 1e3
        ok = abs(got - claim_ms) < 0.02
        print(f"  {pat:7s} {mod:5s}: {got:8.4f} ms   handbook {claim_ms:6.2f} ms   {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"word period {pat}/{mod}: {got:.4f} ms vs {claim_ms} ms")

    # -- 2. end-to-end on synthetic BPSK -----------------------------------
    banner("2. End-to-end pipeline on SYNTHETIC BPSK (known answer = PRBS63)")
    t, ch1, ch2 = make_synthetic("PRBS63", "BPSK", snr_db=35.0)
    path = os.path.join(OUT_DIR, "SYNTHETIC_selftest_BPSK.CSV")
    write_csv(path, t, ch1, ch2)
    print(f"  wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")

    cap = M.load_capture(path)
    print(f"  header     : {cap.header}")
    print(f"  samples    : {cap.n:,}      duration {cap.duration:.4f} s      fs {cap.fs:,.1f} Hz")
    print(f"  CH1 levels : {np.unique(cap.ch1).size}")

    fc = M.estimate_carrier_4th(cap.ch1, cap.fs)
    print(f"  carrier (4th power, blind) : {fc:.3f} Hz   truth 1199.910 Hz")
    if not (1199.5 <= fc <= 1200.5):
        failures.append(f"carrier estimate {fc:.3f} Hz outside 1199.5-1200.5")

    nominal_P = M.word_period_seconds("PRBS63", "BPSK") * cap.fs
    refined_P = M.refine_period(cap.ch1, nominal_P)
    print(f"  word period: nominal {nominal_P:.3f} -> refined {refined_P:.3f} samples")

    tmpl = M.coherent_template(cap.ch1, refined_P)
    print(f"  template   : K = {tmpl.n_periods} periods, SNR = {tmpl.snr_db:.1f} dB   (target >= 15 dB)")
    if tmpl.snr_db < 15:
        failures.append(f"template SNR {tmpl.snr_db:.1f} dB below 15 dB on clean synthetic data")

    sym, tau = M.extract_symbols(tmpl.values, 63, return_tau=True)
    print(f"  blind symbol timing tau = {tau:.4f} symbols")
    sym = M.derotate(sym, "BPSK")
    rec = (sym.real < 0).astype(np.int8)      # phase 180 == bit 1
    m = M.match_prbs63(rec)
    print(f"  PRBS63 match: {m['score']*100:.1f}%   rotation {m['rotation']}, inverted {m['inverted']}")
    print(f"  chance level: {m['chance_mean']*100:.1f}% mean, {m['chance_p95']*100:.1f}% p95")
    print(f"  above chance: {m['above_chance']}")
    if m["score"] < 1.0:
        failures.append(f"PRBS63 recovery only {m['score']*100:.1f}% on noise-free-ish synthetic data")

    # radius of every symbol - catches the wrap-boundary bug (handbook 10.1 step 5)
    r = np.abs(sym)
    print(f"  symbol radii: min {r.min():.3f}, max {r.max():.3f}, ratio {r.max()/r.min():.2f}")
    if r.min() < 0.5 * np.median(r):
        failures.append("a symbol has collapsed radius - circular extension may be broken")

    tp9 = M.tp9_self_consistency(cap.ch2, refined_P)
    print(f"  TP-9 self-consistency: {tp9['consistency']:.4f}  (shuffled control {tp9['chance']:.4f})")

    # -- 3. QPSK and 8-QAM -------------------------------------------------
    for pat, mod, npts in [("Q1", "QPSK", 4), ("T1", "8QAM", 8)]:
        banner(f"3. End-to-end on SYNTHETIC {mod} (pattern {pat})")
        t, ch1, ch2 = make_synthetic(pat, mod, snr_db=35.0, seed=1)
        cap = M.Capture(path="<mem>", header=M.EXPECTED_HEADER, t=t, ch1=ch1, ch2=ch2,
                        n=len(t), fs=M.FS_NOMINAL, duration=(len(t) - 1) / M.FS_NOMINAL)
        fc = M.estimate_carrier_4th(cap.ch1, cap.fs)
        P = M.refine_period(cap.ch1, M.word_period_seconds(pat, mod) * cap.fs)
        tm = M.coherent_template(cap.ch1, P)
        ns = M.symbols_per_word(pat, mod)
        s, tau = M.extract_symbols(tm.values, ns, return_tau=True)
        s = M.derotate(s, mod)
        r = np.abs(s) / np.abs(s).mean()
        print(f"  carrier {fc:.3f} Hz | period {P:.2f} samples | K = {tm.n_periods} | SNR {tm.snr_db:.1f} dB")
        print(f"  blind symbol timing tau = {tau:.4f} symbols")
        print(f"  {ns} symbols, radii: {np.round(np.sort(r), 3)}")
        ang = np.sort(np.rad2deg(np.angle(s)) % 360)
        print(f"  phases (deg): {np.round(ang, 1)}   (want clusters near 45/135/225/315)")
        if mod == "8QAM":
            ratio = np.sort(r)[-4:].mean() / np.sort(r)[:4].mean()
            print(f"  ring radius ratio: {ratio:.3f}   handbook measured 1.965")
            if not (1.6 <= ratio <= 2.3):
                failures.append(f"8-QAM ring ratio {ratio:.3f} far from the measured 1.965")
        else:
            spread = r.max() / r.min()
            print(f"  radius spread: {spread:.3f}   (QPSK is single-ring, want ~1.0)")
            if spread > 1.35:
                failures.append(f"QPSK radii spread {spread:.3f} - single ring expected")
        if r.min() < 0.5:
            failures.append(f"{mod}: collapsed symbol radius - check circular extension")

    # -- 4. pattern identity, and its chance level -------------------------
    banner("4. Check 9 - pattern identity, and what a WRONG pattern scores")
    _, a1, _ = make_synthetic("Q1", "QPSK", snr_db=35.0, seed=2)
    _, a2, _ = make_synthetic("Q1", "QPSK", snr_db=35.0, seed=3)
    _, b1, _ = make_synthetic("Q2", "QPSK", snr_db=35.0, seed=4)
    P = M.refine_period(a1, M.word_period_seconds("Q1", "QPSK") * M.FS_NOMINAL)
    tA1 = M.coherent_template(a1, P).values
    tA2 = M.coherent_template(a2, P).values
    tB1 = M.coherent_template(b1, P).values
    same = M.template_similarity(tA1, tA2)
    diff = M.template_similarity(tA1, tB1)
    print(f"  Q1 vs Q1 (two captures) : {same:.4f}   pass needs >= 0.99")
    print(f"  Q1 vs Q2 (wrong word)   : {diff:.4f}   must be clearly lower")
    if same < 0.99:
        failures.append(f"same-pattern similarity only {same:.4f}")
    if diff > 0.95:
        failures.append(f"different patterns score {diff:.4f} - check 9 cannot discriminate")

    # -- 5. receptive field caps ------------------------------------------
    banner("5. Receptive-field caps (handbook Section 11.1)")
    for mod, cap_sym in M.MAX_RECEPTIVE_FIELD_SYMBOLS.items():
        bits_seen = cap_sym * M.BITS_PER_SYMBOL[mod]
        print(f"  {mod:5s}: {cap_sym} symbol(s) = {bits_seen} bits seen   "
              f"(6 bits would determine the 7th -> {'SAFE' if bits_seen <= 5 else 'UNSAFE'})")
        if bits_seen > 5:
            failures.append(f"{mod} cap of {cap_sym} symbols exposes {bits_seen} bits")

    # -- verdict -----------------------------------------------------------
    banner("VERDICT")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed. The pipeline recovers known ground truth exactly.")
    print("  Any failure on a REAL capture now points at the hardware, not the code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
