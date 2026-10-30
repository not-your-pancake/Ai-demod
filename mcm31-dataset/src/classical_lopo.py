"""
classical_lopo.py - the classical reference receiver on exactly the test records
the learned receiver is scored on, aggregated the same way.

classical_snr.py averages over EVERY record of a modulation; the learned numbers
are the unweighted mean over leave-one-pattern-out folds. Those are different
estimands, so Table VII compares like with like from this file instead:
same test records per fold, mean over folds, five noise realisations.

What the reference receiver is given that the network is not (say so in the
paper): the constellation decision regions, a +-20 Hz carrier search band,
carrier and phase estimates from the whole ~3,600-symbol record, and the M-fold
phase ambiguity resolved against the TRUE labels (best rotation, below). Noise
follows the same convention as data.py but not the same realisations.

Output: results/classical_lopo.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
import splits as SP
from classical import blind_carrier, demodulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
wave, label, wo, lo = d["wave"], d["label"], d["wave_offset"], d["label_offset"]
SPS = int(d["sps_out"])
fs = SPS * M.SYMBOL_RATE
meta = SP.load_meta(ROOT)
SNRS = [0, 3, 6, 9, 12, 15]
REALISATIONS = 5


def errors(pred, y, mod):
    """Errors after the best global rotation - the genie phase step."""
    K = 2 if mod == "BPSK" else 4
    best = 0
    for k in range(K):
        p = ((pred + k) % 2 if mod == "BPSK" else
             (pred + k) % 4 if mod == "QPSK" else
             (pred % 4 + k) % 4 + 4 * (pred // 4))
        best = max(best, int((p == y).sum()))
    return len(y) - best, len(y)


rows = []
print("CLASSICAL REFERENCE, leave-one-pattern-out test records, mean over folds")
for mod in ("BPSK", "QPSK", "8QAM"):
    folds = SP.make_splits(meta, mod)["leave_one_pattern_out"]
    for snr in SNRS:
        macro, micro = [], []
        for rep in range(REALISATIONS):
            rng = np.random.default_rng(100 + rep)
            per_fold, e_all, n_all = [], 0, 0
            for _, _, te in folds:
                e_f = n_f = 0
                for c in te:
                    w = wave[wo[c]:wo[c + 1]].astype(np.float64)
                    y = label[lo[c]:lo[c + 1]]
                    ref = np.sqrt((w ** 2).mean())
                    wn = w + rng.standard_normal(len(w)) * ref / np.sqrt(10 ** (snr / 10))
                    e, n = errors(demodulate(wn, mod, SPS, blind_carrier(wn, SPS, fs), fs), y, mod)
                    e_f += e; n_f += n
                per_fold.append(e_f / n_f); e_all += e_f; n_all += n_f
            macro.append(np.mean(per_fold)); micro.append(e_all / n_all)
        rows.append(dict(mod=mod, snr=snr, macro_mean=np.mean(macro),
                         macro_sd=np.std(macro, ddof=1), micro_mean=np.mean(micro),
                         errors_last=e_all, n=n_all))
        print(f"  {mod:<5} {snr:>2} dB  {np.mean(macro):.6f} (sd {np.std(macro, ddof=1):.6f})  "
              f"pooled {np.mean(micro):.6f}  errors {e_all}/{n_all} (last realisation)")

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "classical_lopo.csv"), index=False)
print("\n  saved -> results/classical_lopo.csv")
