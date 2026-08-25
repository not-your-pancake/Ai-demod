"""
splits.py - the evaluation splits from handbook Section 11.4.

Splitting is done at CAPTURE level, never symbol level. Two symbols from the
same capture are ~300 repeats of the same word through the same hardware; a
symbol-level split would put near-duplicates on both sides and inflate every
number.

One grouping is not obvious and matters: for 8-QAM, **T1 and T2 are one
pattern, not two**. T2 is the bitwise complement of T1, and this board maps
complementary tribits to antipodal points, so their templates correlate -0.998.
After per-window RMS normalisation they are the same data. Holding out T2 while
training on T1 would leak, and the leave-one-pattern-out number would be wrong
in the flattering direction.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# 8-QAM T1/T2 are antipodal - see module docstring
PATTERN_GROUPS = {
    "BPSK": {"PRBS63": "PRBS63", "S1": "S1"},
    "QPSK": {"Q1": "Q1", "Q2": "Q2", "Q3": "Q3"},
    "8QAM": {"T1": "T1+T2", "T2": "T1+T2",
             "T3": "T3", "T4": "T4", "T5": "T5", "T6": "T6"},
}


def load_meta(root: str) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "dataset", "captures.csv"))
    meta["group"] = [PATTERN_GROUPS[m][p]
                     for m, p in zip(meta.modulation, meta.pattern)]
    return meta


def _folds(meta: pd.DataFrame, mod: str, key: str):
    sub = meta[meta.modulation == mod]
    out = []
    for v in sorted(sub[key].unique()):
        test = sub.index[sub[key] == v].to_numpy()
        train = sub.index[sub[key] != v].to_numpy()
        if len(train) == 0 or len(test) == 0:
            continue
        out.append((f"{key}={v}", train, test))
    return out


def make_splits(meta: pd.DataFrame, mod: str) -> dict:
    """All splits for one modulation, as capture-index arrays."""
    sub = meta[meta.modulation == mod]
    rng = np.random.default_rng(0)

    idx = sub.index.to_numpy().copy()
    rng.shuffle(idx)
    cut = max(1, int(round(0.2 * len(idx))))
    random_split = [("random 80/20", idx[cut:], idx[:cut])]

    return {
        "leave_one_pattern_out": _folds(meta, mod, "group"),
        "leave_one_board_out": _folds(meta, mod, "board"),
        "leave_one_attenuation_out": _folds(meta, mod, "attenuation"),
        "random": random_split,
    }


def symbol_index(meta: pd.DataFrame, label_offset: np.ndarray,
                 captures: np.ndarray) -> np.ndarray:
    """Expand capture indices into symbol indices into the flat label array."""
    return np.concatenate([np.arange(label_offset[c], label_offset[c + 1])
                           for c in captures])


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = load_meta(ROOT)
    d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
    lo = d["label_offset"]

    warn = []
    for mod in ("BPSK", "QPSK", "8QAM"):
        print(f"\n{'=' * 74}\n{mod}\n{'=' * 74}")
        for name, folds in make_splits(meta, mod).items():
            print(f"\n  {name}   ({len(folds)} fold{'s' if len(folds) != 1 else ''})")
            if not folds:
                print("    none available")
                continue
            print(f"    {'fold':<26} {'train cap':>10} {'test cap':>9} "
                  f"{'train sym':>11} {'test sym':>10}")
            for fname, tr, te in folds:
                ntr = len(symbol_index(meta, lo, tr))
                nte = len(symbol_index(meta, lo, te))
                flag = ""
                if len(tr) <= 2:
                    flag = "  <-- thin"
                    warn.append(f"{mod} {name} {fname}: only {len(tr)} training capture(s)")
                print(f"    {fname:<26} {len(tr):>10} {len(te):>9} "
                      f"{ntr:>11,} {nte:>10,}{flag}")

    print(f"\n{'=' * 74}")
    if warn:
        print("THIN FOLDS - report these with their training size stated:")
        for w in warn:
            print(f"  - {w}")
    else:
        print("no thin folds")
