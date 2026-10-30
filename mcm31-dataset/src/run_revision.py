"""
run_revision.py - the extra runs added after the two technical reviews (2026-10-30).

run_experiments.py trains every fold once, with seed 0. That cannot say whether
a 0.005 difference between splits is real. This adds:

  seed       training seeds 1 and 2 for every leave-one-pattern-out fold, and
             for every 8-QAM leave-one-board-out and random fold
  randsplit  four more random 80/20 capture partitions for 8-QAM (train seed 0)
  latent     does the latent (I, Q) head collapse? lambda = 0.2, the ablation
             fold (8-QAM and QPSK, board B2 held out, 18 epochs)

The latent check matters because the constellation loss pulls each output to
the NEAREST point, untied from the true class, so a head that returns one
constant point pays no penalty. It does exactly that: every class mean lands on
the same point, spread <= 0.005.

Output: results/revision_runs.csv. Four worker processes x four threads, about
an hour on a 16-core CPU.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "revision_runs.csv")
LOPO_FOLDS = {"BPSK": 2, "QPSK": 3, "8QAM": 5}


def job(spec):
    import torch
    torch.set_num_threads(4)
    import mcm31 as M, splits as SP, train as T
    d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
    arrays = (d["wave"], d["label"], d["wave_offset"], d["label_offset"])
    sps = int(d["sps_out"])
    meta = SP.load_meta(ROOT)

    kind, mod = spec["kind"], spec["mod"]
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    if kind == "randsplit":
        # same construction as splits.make_splits, other partition seeds
        sub = meta[meta.modulation == mod]
        idx = sub.index.to_numpy().copy()
        np.random.default_rng(spec["part_seed"]).shuffle(idx)
        cut = max(1, int(round(0.2 * len(idx))))
        fold, tr, te = f"random part_seed={spec['part_seed']}", idx[cut:], idx[:cut]
    else:
        fold, tr, te = SP.make_splits(meta, mod)[spec["split"]][spec["fold_i"]]
    seed = spec.get("seed", 0)
    epochs = spec.get("epochs", T.EPOCHS)

    t0 = time.time()
    net, res, _, nte = T.train_fold(mod, cap, tr, te, arrays, sps, seed=seed, epochs=epochs)
    row = dict(kind=kind, modulation=mod, split=spec.get("split", "random"), fold=fold,
               seed=seed, epochs=epochs, lam=T.LAMBDA_CONST, n_test=nte,
               secs=round(time.time() - t0, 1), **res)

    if kind == "latent":
        import data as DT
        wave, label, wo, lo = arrays
        ws = DT.WindowSet(wave, label, wo, lo, te, sps, cap)
        net.eval()
        with torch.no_grad():
            iq = net(torch.from_numpy(ws.gather(np.arange(len(ws)))))[1].numpy()
        z = iq[:, 0] + 1j * iq[:, 1]
        z = z / np.sqrt((np.abs(z) ** 2).mean())        # as const_loss scales it
        means = [z[ws.y == k].mean() for k in range(M.n_classes(mod))]
        spread = [float(np.sqrt((np.abs(z[ws.y == k] - means[k]) ** 2).mean()))
                  for k in range(M.n_classes(mod))]
        row.update(latent_means=json.dumps([[round(float(m.real), 4), round(float(m.imag), 4)]
                                            for m in means]),
                   latent_spread=json.dumps([round(s, 4) for s in spread]))
    return row


def specs():
    out = [dict(kind="latent", mod=m, split="leave_one_board_out", fold_i=1, epochs=18)
           for m in ("8QAM", "QPSK")]
    for s in (1, 2):
        for mod, n in LOPO_FOLDS.items():
            out += [dict(kind="seed", mod=mod, split="leave_one_pattern_out", fold_i=i, seed=s)
                    for i in range(n)]
        out.append(dict(kind="seed", mod="8QAM", split="random", fold_i=0, seed=s))
        out += [dict(kind="seed", mod="8QAM", split="leave_one_board_out", fold_i=i, seed=s)
                for i in range(5)]
    out += [dict(kind="randsplit", mod="8QAM", part_seed=p) for p in (1, 2, 3, 4)]
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(job, s): s for s in specs()}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            pd.DataFrame(rows).sort_values(["kind", "modulation", "split", "fold", "seed"]) \
                .to_csv(CSV, index=False)
            print(f"{len(rows):>3}/{len(futs)} {r['kind']:<9} {r['modulation']:<5} "
                  f"{r['fold']:<20} seed {r['seed']}  snr0 {r['snr0']:.4f}  "
                  f"({r['secs']:.0f}s)", flush=True)
    print(f"saved -> {CSV}")
