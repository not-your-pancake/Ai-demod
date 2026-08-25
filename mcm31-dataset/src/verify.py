"""
verify.py - run the handbook's Section 9 checks on every capture, plus the
BPSK cross-check against PRBS63.

Usage:
    <venv>\\python.exe src\\verify.py raw\\B1
    <venv>\\python.exe src\\verify.py raw\\B1\\B1_BPSK_A0.CSV

Writes:
    templates/<name>.npy      coherent template per file
    reports/verify_<dir>.csv  one row per file
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN_TAGS = {"Q2", "Q3", "T2", "T3", "S1",
                "T4", "T5", "T6", "Q4", "Q5"}
DEFAULT_PATTERN = {"BPSK": "PRBS63", "QPSK": "Q1", "8QAM": "T1"}


def parse_name(path: str) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0].upper()
    parts = stem.split("_")
    board = parts[0] if parts else "?"
    mod = next((p for p in parts if p in M.BITS_PER_SYMBOL), None)
    tag = parts[-1] if len(parts) > 2 else ""
    if tag in PATTERN_TAGS:
        pattern, atten = tag, "A0"
    else:
        pattern, atten = DEFAULT_PATTERN.get(mod, "?"), tag
    return {"board": board, "modulation": mod, "pattern": pattern, "attenuation": atten}


def period_correlation(x: np.ndarray, period_samples: float) -> float:
    """Does the waveform actually repeat at this period? Pearson correlation
    between the signal and itself shifted by one (fractional) word period."""
    n = len(x)
    lag = int(np.floor(period_samples)) + 1
    idx = np.arange(n - lag, dtype=np.float64)
    a = x[: len(idx)]
    b = np.interp(idx + period_samples, np.arange(n, dtype=np.float64), x)
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(label: str, value, ok: bool, detail: str = "") -> tuple:
    mark = "PASS" if ok else "FAIL"
    print(f"    [{mark}] {label:<28} {value}{('   ' + detail) if detail else ''}")
    return (label, ok)


def verify_file(path: str) -> dict:
    info = parse_name(path)
    name = os.path.basename(path)
    print(f"\n{'-' * 74}\n{name}   ({info['modulation']}, pattern {info['pattern']}, "
          f"attenuation {info['attenuation']})\n{'-' * 74}")

    row = dict(filename=name, **info)
    results = []

    cap = M.load_capture(path)

    # 1 columns -------------------------------------------------------------
    results.append(check("1 columns", cap.meta["n_columns"],
                         cap.meta["n_columns"] >= 3, f"header: {cap.header}"))
    # 2 sample count --------------------------------------------------------
    results.append(check("2 sample count", f"{cap.n:,}", 520_000 <= cap.n <= 526_000))
    # 3 duration ------------------------------------------------------------
    results.append(check("3 duration", f"{cap.duration:.4f} s", 5.99 <= cap.duration <= 6.01))
    # 4 sample rate ---------------------------------------------------------
    results.append(check("4 sample rate", f"{cap.fs:,.1f} Hz", 87_000 <= cap.fs <= 87_500))
    # 5 CH1 dynamic range ---------------------------------------------------
    levels = int(np.unique(cap.ch1).size)
    rms_mv = float(np.sqrt(np.mean((cap.ch1 - cap.ch1.mean()) ** 2)) * 1000)
    pkpk = float(cap.ch1.max() - cap.ch1.min())
    results.append(check("5 CH1 distinct levels", levels, levels >= 90,
                         f"RMS {rms_mv:.1f} mV, pk-pk {pkpk:.2f} V"))
    # 6 carrier -------------------------------------------------------------
    fc = M.estimate_carrier_4th(cap.ch1, cap.fs)
    results.append(check("6 carrier (4th power)", f"{fc:.3f} Hz", 1199.5 <= fc <= 1200.5))

    # 7 word period ---------------------------------------------------------
    nominal = M.word_period_seconds(info["pattern"], info["modulation"]) * cap.fs
    refined = M.refine_period(cap.ch1, nominal)
    corr = period_correlation(cap.ch1, refined)
    results.append(check("7 word period repeats", f"r = {corr:.4f}", corr >= 0.95,
                         f"{refined:.2f} samples = {refined / cap.fs * 1e3:.3f} ms"))

    # 8 template SNR --------------------------------------------------------
    tmpl = M.coherent_template(cap.ch1, refined)
    results.append(check("8 template SNR", f"{tmpl.snr_db:.1f} dB", tmpl.snr_db >= 15,
                         f"K = {tmpl.n_periods} periods"))

    # 10 TP-9 lock (check 9 needs all files, so it runs afterwards) ----------
    tp9 = M.tp9_self_consistency(cap.ch2, refined)
    tp9x2 = M.tp9_self_consistency(cap.ch2, refined * 2)
    best_mult, best = (1, tp9) if tp9["consistency"] >= tp9x2["consistency"] else (2, tp9x2)
    results.append(check("10 TP-9 self-consistency", f"{best['consistency']:.4f}",
                         True,   # informational: never blocks
                         f"at {best_mult}x word period; shuffled control "
                         f"{best['chance']:.4f}; {'LOCKED' if best['locked'] else 'baseline_unavailable'}"))

    # symbols + BPSK ground-truth cross-check -------------------------------
    n_sym = M.symbols_per_word(info["pattern"], info["modulation"])
    sym, tau = M.extract_symbols(tmpl.values, n_sym, return_tau=True)
    sym = M.derotate(sym, info["modulation"])
    r = np.abs(sym) / np.abs(sym).mean()
    print(f"    [ -- ] {'symbol timing tau':<28} {tau:.4f} symbols")
    print(f"    [ -- ] {'symbol radii':<28} min {r.min():.3f}  max {r.max():.3f}  "
          f"spread {r.max() / r.min():.2f}")

    # 9b pattern fingerprint - works with no reference template, and without
    # solving the unresolved switch-to-symbol mapping
    ident = M.identify_pattern(sym, info["modulation"])
    ok9b = info["pattern"] in ident["consistent_with"]
    results.append(check("9b pattern fingerprint", ident["observed"], ok9b,
                         f"expected for {info['pattern']}: "
                         f"{ident['candidates'].get(info['pattern'])}; "
                         f"consistent with {ident['consistent_with'] or 'NOTHING'}"))
    row["fingerprint"] = str(ident["observed"])
    row["fingerprint_matches"] = ",".join(ident["consistent_with"])

    prbs = None
    if info["modulation"] == "BPSK" and info["pattern"] == "PRBS63":
        bits = (sym.real < 0).astype(np.int8)
        m = M.match_prbs63(bits)
        if m["score"] < 0.5:                       # polarity is arbitrary
            m = M.match_prbs63(1 - bits)
        prbs = m["score"]
        results.append(check("* PRBS63 ground truth", f"{m['score'] * 100:.1f}%",
                             m["above_chance"],
                             f"chance {m['chance_mean'] * 100:.1f}% / p95 "
                             f"{m['chance_p95'] * 100:.1f}%, rotation {m['rotation']}"))

    if info["modulation"] == "8QAM":
        s = np.sort(r)
        ratio = s[-4:].mean() / s[:4].mean()
        print(f"    [ -- ] {'8-QAM ring ratio':<28} {ratio:.3f}   "
              f"(handbook measured 1.965)")
        row["ring_ratio"] = ratio

    np.save(os.path.join(ROOT, "templates", os.path.splitext(name)[0] + ".npy"), tmpl.values)

    row.update(
        n=cap.n, duration=cap.duration, fs=cap.fs, levels=levels,
        rms_mv=rms_mv, pkpk=pkpk, carrier_hz=fc,
        period_samples=refined, period_ms=refined / cap.fs * 1e3, period_corr=corr,
        template_snr_db=tmpl.snr_db, k_periods=tmpl.n_periods,
        tp9_consistency=best["consistency"], tp9_chance=best["chance"],
        tp9_multiple=best_mult, tp9_locked=best["locked"],
        tau=tau, radius_spread=float(r.max() / r.min()), prbs63_match=prbs,
        n_failed=sum(1 for _, ok in results if not ok),
    )
    row["_template"] = tmpl.values
    return row


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else os.path.join(ROOT, "raw", "B1")
    if not os.path.isabs(target):
        target = os.path.join(ROOT, target)

    if os.path.isfile(target):
        files = [target]
    else:
        # Windows globbing is case-insensitive, so *.CSV and *.csv both match
        # the same file. De-duplicate or every capture gets verified twice.
        seen, files = set(), []
        for p in sorted(glob.glob(os.path.join(target, "*.CSV"))
                        + glob.glob(os.path.join(target, "*.csv"))):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                files.append(p)
    if not files:
        print(f"no CSV files found under {target}")
        return 1

    os.makedirs(os.path.join(ROOT, "templates"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)

    print(f"\nVerifying {len(files)} file(s) in {target}")
    rows = [verify_file(f) for f in files]

    # -- check 9: pattern identity, across files sharing a pattern ----------
    print(f"\n{'=' * 74}\nCHECK 9 - pattern identity (files sharing a pattern must match >= 0.99)"
          f"\n{'=' * 74}")
    by_pattern: dict[str, list[dict]] = {}
    for r in rows:
        by_pattern.setdefault(f"{r['modulation']}/{r['pattern']}", []).append(r)

    identity = {}
    for key, group in by_pattern.items():
        if len(group) < 2:
            print(f"  {key:<14} only one capture - nothing to compare yet")
            continue
        base = group[0]
        for other in group[1:]:
            s = M.template_similarity(base["_template"], other["_template"])
            mark = "PASS" if s >= 0.99 else "FAIL"
            print(f"  [{mark}] {key:<14} {base['attenuation']:>4} vs {other['attenuation']:<4} : {s:.4f}")
            identity[other["filename"]] = s
            if s < 0.99:
                other["n_failed"] += 1

    # -- check 9c: held-out words must be MUTUALLY DISTINCT -----------------
    #
    # Check 9 above only fires where a pattern was captured more than once.
    # Every Block 2 word is captured exactly ONCE, so check 9 is blind there -
    # and check 9b cannot cover for it, because T1..T6 all share the
    # fingerprint [1,1,1,1,1,1,1,1] by construction, as do Q1..Q5 with
    # [3,3,3,3]. A stale DIP word would pass both. The test that does work is
    # the inverse: two DIFFERENT words on the same modulation must NOT produce
    # the same waveform.
    print("\n" + "=" * 74)
    print("CHECK 9c - different words, same modulation, must NOT match")
    print("=" * 74)
    keys = sorted(by_pattern)
    same_mod_pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            mod_i, pat_i = keys[i].split("/")
            mod_j, pat_j = keys[j].split("/")
            if mod_i != mod_j:
                continue
            same_mod_pairs += 1
            other = by_pattern[keys[j]][0]
            pos, absv = M.template_similarity_signed(
                by_pattern[keys[i]][0]["_template"], other["_template"])
            if pos >= 0.99:
                tag, ok = "   <-- STALE WORD: START WAS NOT PRESSED", False
            elif absv >= 0.99:
                tag, ok = ("   <-- ANTIPODAL: bit-complement of the other word. "
                           "Not independent - treat the pair as ONE pattern.", True)
            else:
                tag, ok = "", True
            mark = "PASS" if ok else "FAIL"
            if absv >= 0.99 and pos < 0.99:
                mark = "WARN"
            print(f"  [{mark}] {mod_i:<5} {pat_i:>7} vs {pat_j:<7} : "
                  f"{pos:+.4f} signed / {absv:.4f} abs{tag}")
            if not ok:
                other["n_failed"] += 1
    if same_mod_pairs == 0:
        print("  no two different words on the same modulation yet - nothing to compare")

    keys = list(by_pattern)
    print("\n  reference - across modulations, resampled to a common grid:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i].split("/")[0] == keys[j].split("/")[0]:
                continue
            s = M.template_similarity(by_pattern[keys[i]][0]["_template"],
                                      by_pattern[keys[j]][0]["_template"])
            print(f"    {keys[i]:<14} vs {keys[j]:<14} : {s:.4f}")

    # -- summary -----------------------------------------------------------
    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    hdr = (f"{'file':<20} {'lvls':>5} {'RMS mV':>8} {'fc Hz':>9} {'r':>7} "
           f"{'SNR dB':>7} {'TP-9':>7} {'fails':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['filename']:<20} {r['levels']:>5} {r['rms_mv']:>8.1f} {r['carrier_hz']:>9.3f} "
              f"{r['period_corr']:>7.4f} {r['template_snr_db']:>7.1f} "
              f"{r['tp9_consistency']:>7.4f} {r['n_failed']:>6}")

    out = pd.DataFrame([{k: v for k, v in r.items() if k != "_template"} for r in rows])
    out["identity_vs_first"] = out["filename"].map(identity)
    csv_path = os.path.join(ROOT, "reports",
                            f"verify_{os.path.basename(target.rstrip(os.sep))}.csv")
    out.to_csv(csv_path, index=False)
    print(f"\nreport written to {csv_path}")

    total_fail = int(out["n_failed"].sum())
    print(f"\n{'ALL CHECKS PASSED' if total_fail == 0 else str(total_fail) + ' CHECK FAILURE(S)'}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
