"""Run every fold and every control. Saves incrementally to results/.

Main folds use training seed 0. Extra seeds and partitions are in
run_revision.py -> results/revision_runs.csv.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M, splits as SP, train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "experiments.csv")

d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
arrays = (d["wave"], d["label"], d["wave_offset"], d["label_offset"])
SPS = int(d["sps_out"])
meta = SP.load_meta(ROOT)

rows, t_start = [], time.time()

def save():
    pd.DataFrame(rows).to_csv(CSV, index=False)

for mod in ("BPSK", "QPSK", "8QAM"):
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    n_word = meta["n_sym_word"].to_numpy()   # per record, not per modulation
    for split_name, folds in SP.make_splits(meta, mod).items():
        for fold_name, tr, te in folds:
            t0 = time.time()
            net, res, ntr, nte = T.train_fold(mod, cap, tr, te, arrays, SPS)
            pos = T.position_baseline(mod, cap, tr, te, arrays, SPS, n_word)
            rows.append(dict(modulation=mod, split=split_name, fold=fold_name,
                             control="none", seed=0, n_train=ntr, n_test=nte,
                             n_train_captures=len(tr), position_only_ser=pos,
                             secs=round(time.time() - t0, 1), **res))
            save()
            print(f"{mod:<5} {split_name:<26} {fold_name:<16} "
                  f"clean SER {res['clean']:.4f}  snr6 {res['snr6']:.4f}  "
                  f"pos-only {pos:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# Shuffled-label control: every leave-one-pattern-out fold, five permutation
# seeds, training labels permuted and test labels as measured (train.py). One
# run per fold is not enough: with 8-63 distinct clean windows per word a single
# run can land anywhere from ~0.2 to 1.0, so the mean over seeds is the number.
SHUFFLE_SEEDS = (0, 1, 2, 3, 4)
for mod in ("BPSK", "QPSK", "8QAM"):
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    for fold_name, tr, te in SP.make_splits(meta, mod)["leave_one_pattern_out"]:
        for seed in SHUFFLE_SEEDS:
            t0 = time.time()
            net, res, ntr, nte = T.train_fold(mod, cap, tr, te, arrays, SPS,
                                              seed=seed, shuffle_labels=True)
            rows.append(dict(modulation=mod, split="leave_one_pattern_out",
                             fold=fold_name, control="shuffled_labels",
                             seed=seed, n_train=ntr, n_test=nte,
                             n_train_captures=len(tr), position_only_ser=np.nan,
                             secs=round(time.time() - t0, 1), **res))
            save()
            print(f"{mod:<5} SHUFFLED-LABEL CONTROL {fold_name:<14} seed {seed}  "
                  f"clean SER {res['clean']:.4f}  "
                  f"(chance {1 - 1 / M.n_classes(mod):.4f})", flush=True)
    shu = pd.DataFrame(rows)
    shu = shu[(shu.control == "shuffled_labels") & (shu.modulation == mod)]
    best = shu.clean.min()
    print(f"{mod:<5} shuffled mean {shu.clean.mean():.4f}, best single run {best:.4f} "
          f"-> {'PASS, nowhere near the receiver' if best > 0.1 else 'LEAK?'}",
          flush=True)

save()
print(f"\nDONE in {(time.time()-t_start)/60:.1f} min -> {CSV}")
