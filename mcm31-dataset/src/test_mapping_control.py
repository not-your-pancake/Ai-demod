"""
Corrected chance control for the switch->symbol mapping search.

The first control drew uniformly random symbol sequences. That is the WRONG
null, because it does not preserve how many times each point occurs. The search
is a bijection search, so the count structure is exactly what determines whether
a perfect fit is even possible.

The honest null shuffles the reference word's own symbol sequence, which keeps
the counts identical and only destroys the ORDER - the thing actually being
tested.

Worked example of why this matters: T1 contains all 8 tribits exactly once, and
the observed 8-QAM symbols are also 8 distinct points. Whenever both sides are
permutations of 0..7, a perfect bijection ALWAYS exists, for any rotation. A
100% score there is arithmetic, not evidence.
"""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_match(obs, ref, n_pts):
    n = len(ref)
    best = -1.0
    for r in range(n):
        rr = np.roll(ref, r)
        counts = np.zeros((n_pts, n_pts))
        for p, g in zip(obs, rr):
            counts[p, g] += 1
        row, col = linear_sum_assignment(-counts)
        best = max(best, float(counts[row, col].sum() / n))
    return best


def observed(path, pattern, mod):
    cap = M.load_capture(path)
    P = M.refine_period(cap.ch1, M.word_period_seconds(pattern, mod) * cap.fs)
    tm = M.coherent_template(cap.ch1, P)
    n = M.symbols_per_word(pattern, mod)
    s = M.derotate(M.extract_symbols(tm.values, n), mod)
    q = np.floor((np.angle(s) % (2 * np.pi)) / (np.pi / 2)).astype(int) % 4
    if mod == "8QAM":
        return q + 4 * (np.abs(s) > np.median(np.abs(s))).astype(int)
    return q


def run(label, path, pattern, mod, n_pts, n_null=20000):
    obs = observed(path, pattern, mod)
    bps = M.BITS_PER_SYMBOL[mod]
    ref = np.array([int("".join(map(str, g)), 2)
                    for g in M.bits_of(pattern).reshape(-1, bps)])

    score = best_match(obs, ref, n_pts)

    rng = np.random.default_rng(0)
    null = np.array([best_match(rng.permutation(ref), ref, n_pts)
                     for _ in range(n_null)])
    p_value = float((null >= score).mean())

    print(f"\n{label}   word {pattern}, {len(ref)} symbols")
    print(f"  observed points        : {obs.tolist()}")
    print(f"  {pattern} groups from switches : {ref.tolist()}")
    print(f"  best memoryless fit    : {score * 100:.1f}%")
    print(f"  HONEST null (shuffles preserving the counts, n={n_null:,}):")
    print(f"     mean {null.mean() * 100:.1f}%   p95 {np.percentile(null, 95) * 100:.1f}%   "
          f"perfect fits {float((null >= 1.0).mean()) * 100:.1f}% of the time")
    print(f"  p-value                : {p_value:.5f}")
    if p_value < 0.001:
        print("  => REAL. The switch word maps to symbols by a fixed bijection.")
    elif float((null >= 1.0).mean()) > 0.5:
        print("  => VACUOUS. A perfect fit is guaranteed by the count structure;")
        print("     this test cannot say anything for this word.")
    else:
        print("  => not significant.")


run("QPSK ", os.path.join(ROOT, "raw", "B1", "B1_QPSK_A0.CSV"), "Q1", "QPSK", 4)
run("8-QAM", os.path.join(ROOT, "raw", "B1", "B1_8QAM_A0.CSV"), "T1", "8QAM", 8, n_null=5000)
