"""
isi_readability.py - can the 8-QAM window read a neighbour through ISI?

The 8-QAM window holds the target symbol only. The one route by which a
neighbouring symbol can still reach it is the trace that the transmitter's
pulse and the anti-alias filter leave in the target's 20 samples, 20-28 dB
below the symbol (context_bound.py section 6). This script asks the paper's
network to do the one thing the shortcut needs: name the label of the
PREVIOUS (or NEXT) symbol from that window.

The test is set up to favour the shortcut:
  * one network per record, so the word, the board and the attenuation are
    all fixed;
  * each record is split in time - first 75% train, last 25% test, 8 symbols
    apart - so the test windows repeat the training windows' positions in the
    word, with fresh noise.

It is scored only where the target's own class cannot give the answer away:
positions whose constellation point occurs twice in the word with different
neighbours. That is 4 of the 8 positions in each of T3-T6. It never happens in
T1/T2, where every point occurs once, so those records are not used. At those
positions, the best predictor that knows the target's class exactly is right
half the time (`ceiling`, fitted on the training windows). Anything above that
must come from the trace.

`visible` is the check that the probe itself works: the same training with the
neighbour inside the window ([t-1, t] or [t, t+1]), which it should name
almost perfectly. `target only, noise-free training` asks whether the trace is
there at all: the same probe trained on noise-free windows only, where a weak
but deterministic trace is easiest to learn.

Output: results/isi_readability.csv
"""
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window_control as W      # Windows, the data arrays and meta

CSV = os.path.join(W.ROOT, "results", "isi_readability.csv")
WORDS = ("T3", "T4", "T5", "T6")
OFFSETS = (-1, +1)
CONFIGS = ("target only", "visible", "target only, noise-free training")
SEEDS = (0, 1, 2)
TRAIN_FRAC, GAP = 0.75, 8
EVAL_SNR = [0, 3, 6, 9, 12, 15, 20, 30]
WORKERS, THREADS = 8, 2


def setup(word, off, config):
    c = int(W.meta.index[(W.meta.modulation == "8QAM") & (W.meta.pattern == word)][0])
    before, after = (0, 0) if config.startswith("target only") else ((1, 0) if off < 0 else (0, 1))
    ws = W.Windows([c], before, after, margin=1)
    j = (ws.starts - W.wo[c]) // W.SPS + before          # target index in the record
    y = W.label[W.lo[c] + j + off].astype(np.int64)      # the neighbour's label
    n = int(W.meta.n_sym_word[c])
    w = W.label[W.lo[c]:W.lo[c] + n].astype(int)
    assert (W.label[W.lo[c] + j] == w[j % n]).all()      # labels tile the word exactly
    nb = defaultdict(set)
    for p in range(n):
        nb[w[p]].add(w[(p + off) % n])
    amb = np.array([len(nb[w[p]]) > 1 for p in j % n])
    cut = int(TRAIN_FRAC * (W.lo[c + 1] - W.lo[c]))
    tr, te = np.flatnonzero(j < cut - GAP), np.flatnonzero(j >= cut)
    # best predictor that knows the target's class: fitted on train, scored on test
    tab = defaultdict(Counter)
    for k in tr:
        tab[w[j[k] % n]][y[k]] += 1
    te_amb = te[amb[te]]
    ceiling = float(np.mean([tab[w[j[k] % n]].most_common(1)[0][0] == y[k] for k in te_amb]))
    return ws, y, tr, te, amb, ceiling


def job(spec):
    import torch
    import torch.nn as nn
    import data as DT
    import model as MD
    import train as T
    torch.set_num_threads(THREADS)

    word, off, config, seed = spec
    ws, y, tr, te, amb, ceiling = setup(word, off, config)
    torch.manual_seed(seed)
    net = MD.CappedCNN(8, ws.window)          # the paper's network for this window
    C = T.constellation("8QAM")
    cnt = np.bincount(y[tr], minlength=8).astype(np.float64)
    cw = cnt.sum() / (8 * np.maximum(cnt, 1))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw / cw.mean(), dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), T.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T.EPOCHS)
    rng = np.random.default_rng(seed)
    for _ in range(T.EPOCHS):
        for b in DT.epoch_batches(len(tr), T.BATCH, rng):
            idx = tr[b]
            snr = None if "noise-free" in config else rng.uniform(*T.TRAIN_SNR, size=len(idx))
            x = torch.from_numpy(ws.gather(idx, snr, rng))
            opt.zero_grad()
            logit, iq = net(x)
            (ce(logit, torch.from_numpy(y[idx])) +
             T.LAMBDA_CONST * T.const_loss(iq, C)).backward()
            opt.step()
        sched.step()

    net.eval()
    erng = np.random.default_rng(10_000 + seed)
    rows = []
    with torch.no_grad():
        for snr in EVAL_SNR + [None]:
            hit = []
            for _ in range(5):                 # five noise draws per SNR
                p = net(torch.from_numpy(ws.gather(te, snr, erng)))[0].argmax(1).numpy()
                hit.append(p == y[te])
                if snr is None:
                    break
            hit = np.mean(hit, axis=0)
            rows.append(dict(word=word, offset=off, config=config, seed=seed,
                             snr="clean" if snr is None else snr,
                             acc_ambiguous=float(hit[amb[te]].mean()),
                             acc_all=float(hit.mean()), n_ambiguous=int(amb[te].sum()),
                             ceiling=ceiling))
    return rows


if __name__ == "__main__":
    specs = [(w, o, c, s) for w in WORDS for o in OFFSETS for c in CONFIGS for s in SEEDS]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for out in ex.map(job, specs) for r in out]
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)

    order = [str(s) for s in EVAL_SNR] + ["clean"]
    df["snr"] = df.snr.astype(str)
    t = (df.groupby(["config", "offset", "snr"])
           .agg(acc=("acc_ambiguous", "mean"), lo=("acc_ambiguous", "min"),
                hi=("acc_ambiguous", "max"), ceiling=("ceiling", "mean"),
                n=("n_ambiguous", "sum")).reset_index())
    t["snr"] = pd.Categorical(t.snr, order, ordered=True)
    print("accuracy naming the neighbour, ambiguous positions, mean over 4 words x 3 seeds")
    print("(ceiling = best predictor that knows the target's class)\n")
    print(t.sort_values(["config", "offset", "snr"]).round(3).to_string(index=False))
    print(f"\nsaved -> {CSV}")
