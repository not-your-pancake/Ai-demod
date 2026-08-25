"""
WARNING - the chance control in this script is WRONG for 8-QAM. Superseded by
test_mapping_control.py. Kept only because it is the clean test of the
memoryless-vs-differential question, where the two agree.

The null here draws uniformly random symbol sequences, which does not preserve
how often each constellation point occurs. For 8-QAM that inflates the result
into a false positive: T1 holds all 8 tribits exactly once and the observed
symbols are 8 distinct points, so a perfect bijection exists for ANY rotation.
The honest count-preserving null fits perfectly 100% of the time -> p = 1.00.

Trust from this script ONLY the memoryless-vs-differential comparison:
  QPSK   memoryless 100.0%   differential 66.7%
  8-QAM  memoryless 100.0%   differential 50.0%
i.e. the board is NOT differentially encoded. For whether the memoryless fit is
significant, run test_mapping_control.py.

BPSK proved the DIP word reaches the modulator faithfully (S1 = 100%).
So why did the handbook's QPSK/8-QAM mapping search score at chance?

Hypothesis: multi-level modes use DIFFERENTIAL encoding. Many PSK modulators
encode the phase CHANGE rather than the absolute phase for QPSK/8-QAM, while
leaving BPSK absolute. That would look exactly like this: BPSK resolvable,
QPSK/8-QAM at chance under a memoryless search.

Tested here with a chance control on both hypotheses (handbook rule 3).
"""
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def symbol_indices(path, pattern, mod):
    cap = M.load_capture(path)
    P = M.refine_period(cap.ch1, M.word_period_seconds(pattern, mod) * cap.fs)
    tm = M.coherent_template(cap.ch1, P)
    n = M.symbols_per_word(pattern, mod)
    s = M.derotate(M.extract_symbols(tm.values, n), mod)
    quad = (np.floor((np.angle(s) % (2 * np.pi)) / (np.pi / 2)).astype(int)) % 4
    if mod == "8QAM":
        return quad + 4 * (np.abs(s) > np.median(np.abs(s))).astype(int), tm
    return quad, tm


def best_match(obs, ref, n_pts, differential):
    """Best achievable score over ALL point->group bijections and all rotations.

    Brute-forcing 8! permutations is far too slow, so for each rotation the
    optimal bijection is found exactly by solving the assignment problem on the
    co-occurrence counts. Same answer, seconds instead of hours.
    """
    x = (obs - np.roll(obs, 1)) % n_pts if differential else obs
    n = len(ref)
    best = -1.0
    for r in range(n):
        rr = np.roll(ref, r)
        counts = np.zeros((n_pts, n_pts))
        for p, g in zip(x, rr):
            counts[p, g] += 1
        row, col = linear_sum_assignment(-counts)
        best = max(best, float(counts[row, col].sum() / n))
    return best


def run(label, path, pattern, mod, n_pts, n_null=500):
    obs, tm = symbol_indices(path, pattern, mod)
    bps = M.BITS_PER_SYMBOL[mod]
    ref = np.array([int("".join(map(str, g)), 2)
                    for g in M.bits_of(pattern).reshape(-1, bps)])

    rng = np.random.default_rng(0)
    print(f"\n{label}  (word {pattern}, {len(ref)} symbols, template SNR {tm.snr_db:.1f} dB)")
    print(f"  observed points : {obs.tolist()}")
    print(f"  {pattern} groups     : {ref.tolist()}")

    for diff in (False, True):
        score = best_match(obs, ref, n_pts, diff)
        null = [best_match(rng.integers(0, n_pts, len(ref)), ref, n_pts, diff)
                for _ in range(n_null)]
        p95 = float(np.percentile(null, 95))
        name = "differential" if diff else "memoryless  "
        verdict = ("beats this (WRONG) null" if score > p95 else "chance")
        print(f"  {name} : {score*100:5.1f}%   chance {np.mean(null)*100:.1f}% mean, "
              f"{p95*100:.1f}% p95   -> {verdict}")


run("QPSK", os.path.join(ROOT, "raw", "B1", "B1_QPSK_A0.CSV"), "Q1", "QPSK", 4)
run("8-QAM", os.path.join(ROOT, "raw", "B1", "B1_8QAM_A0.CSV"), "T1", "8QAM", 8, n_null=500)
