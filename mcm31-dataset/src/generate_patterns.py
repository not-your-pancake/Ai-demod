"""
generate_patterns.py - propose extra 24-bit DIP words for Block 2.

Selection criterion, stated so it can go in the paper: held-out words are chosen
to maximise coverage of distinct symbol-TRANSITION contexts. With the receptive
field capped at 1 symbol for 8-QAM, what a new word actually buys is each
constellation point being seen beside neighbours it has not sat beside before,
because inter-symbol interference is the only way context reaches the network.

Hard exclusions:
  * any cyclic rotation of an existing word - the board transmits it as the same
    periodic waveform, so it is not a held-out pattern at all, just a re-trigger
  * any rotation of an existing word's bitwise complement - structurally
    dependent (T2 is already exactly the complement of T1)
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402


def symbols_of(pattern_id: str, bps: int) -> tuple[int, ...]:
    bits = M.bits_of(pattern_id)
    return tuple(int("".join(map(str, g)), 2) for g in bits.reshape(-1, bps))


def word_bits(syms, bps: int) -> str:
    return "".join(format(s, f"0{bps}b") for s in syms)


def max_run(bits: str) -> int:
    best = cur = 1
    for i in range(1, len(bits)):
        cur = cur + 1 if bits[i] == bits[i - 1] else 1
        best = max(best, cur)
    return best


def contexts(syms) -> set:
    return {(syms[i - 1], syms[i]) for i in range(len(syms))}


def rotations(syms):
    return {tuple(syms[i:] + syms[:i]) for i in range(len(syms))}


def forbidden_set(existing, bps: int) -> set:
    mask = (1 << bps) - 1
    out = set()
    for syms in existing:
        out |= rotations(syms)
        out |= rotations(tuple(mask - s for s in syms))
    return out


def propose(modulation: str, existing_ids: list[str], n_new: int,
            run_limit: int, candidates) -> list:
    bps = M.BITS_PER_SYMBOL[modulation]
    existing = [symbols_of(p, bps) for p in existing_ids]
    banned = forbidden_set(existing, bps)
    have = set().union(*(contexts(s) for s in existing))

    pool = []
    for syms in candidates:
        if syms in banned:
            continue
        bits = word_bits(syms, bps)
        if max_run(bits) > run_limit:
            continue
        pool.append((syms, bits, contexts(syms)))

    print(f"\n{modulation}: {len(pool):,} candidate words survive the filters")
    print(f"  existing {existing_ids} cover {len(have)} of "
          f"{(2 ** bps) ** 2} possible transitions")

    chosen = []
    for _ in range(n_new):
        best = max(pool, key=lambda c: (len(c[2] - have), -max_run(c[1])))
        chosen.append(best)
        have |= best[2]
        pool = [c for c in pool if c[0] != best[0]]
    print(f"  after adding {n_new} words: {len(have)} transitions covered")
    return chosen


def dip_map(bits: str) -> str:
    banks = [bits[0:8], bits[8:16], bits[16:24]]
    labels = ["BANK 1 (left)  ", "BANK 2 (middle)", "BANK 3 (right) "]
    return "\n".join(f"      {lab}  " + " ".join(b)
                     for lab, b in zip(labels, banks))


def show(name: str, syms, bits: str, modulation: str) -> None:
    bps = M.BITS_PER_SYMBOL[modulation]
    arr = np.array([int(c) for c in bits], dtype=np.int8)
    groups = np.bincount(np.array(syms), minlength=2 ** bps)
    print(f"\n  {name} = {bits}")
    print(f"      symbols     : {list(syms)}")
    print(f"      point counts: {sorted(groups.tolist())}   "
          f"({int((groups > 0).sum())} of {2 ** bps} points covered)")
    print(f"      max bit run : {max_run(bits)}    ones/zeros: "
          f"{int(arr.sum())}/{int((1 - arr).sum())}")
    print(dip_map(bits))


# ---------------------------------------------------------------------------
# 8-QAM: every word is a permutation of the 8 tribits, so all 8 points appear
# exactly once - the property Section 3.3 requires and Q1 fails.
print("=" * 70)
print("PROPOSED EXTRA HELD-OUT WORDS  (24-bit DIP, board B1, Block 2)")
print("=" * 70)

new_8qam = propose("8QAM", ["T1", "T2", "T3"], 3, run_limit=3,
                   candidates=itertools.permutations(range(8)))
for name, (syms, bits, _) in zip(["T4", "T5", "T6"], new_8qam):
    show(name, syms, bits, "8QAM")

# QPSK: 12 dibits, three of each value -> perfectly balanced quadrants
qpsk_pool = set(itertools.permutations([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]))
new_qpsk = propose("QPSK", ["Q1", "Q2", "Q3"], 2, run_limit=3,
                   candidates=qpsk_pool)
for name, (syms, bits, _) in zip(["Q4", "Q5"], new_qpsk):
    show(name, syms, bits, "QPSK")

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CROSS-CHECK against the handbook's single-purpose rule (Section 3.3)")
print("=" * 70)
for name, (syms, bits, _) in list(zip(["T4", "T5", "T6"], new_8qam)):
    d = np.array([int(c) for c in bits], dtype=np.int8).reshape(12, 2)
    q = np.bincount(d[:, 0] * 2 + d[:, 1], minlength=4)
    print(f"  {name} viewed as QPSK dibits -> quadrants {q.tolist()}  "
          f"{'(unusable for QPSK, as intended)' if q.min() < 2 else '(would also work for QPSK)'}")
for name, (syms, bits, _) in list(zip(["Q4", "Q5"], new_qpsk)):
    t = np.array([int(c) for c in bits], dtype=np.int8).reshape(8, 3)
    idx = t[:, 0] * 4 + t[:, 1] * 2 + t[:, 2]
    cov = int((np.bincount(idx, minlength=8) > 0).sum())
    print(f"  {name} viewed as 8-QAM tribits -> {cov} of 8 points  "
          f"{'(unusable for 8-QAM, as intended)' if cov < 8 else '(would also work for 8-QAM)'}")
