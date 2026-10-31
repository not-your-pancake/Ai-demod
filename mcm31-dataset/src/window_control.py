"""
window_control.py - positive control for the bounded input context.

The paper argues (Section V-A) that the receiver cannot memorise the transmitted
word because its window holds fewer neighbouring symbols than are needed to
determine the label. This script tests the other half of that argument: give the
SAME network a window long enough to determine every label from its neighbours,
and check that (a) it then does use the sequence, and (b) the
leave-one-pattern-out split catches it.

Three inputs, same CNN family, loss, optimiser, epochs and training SNR range
as train.py:

  short   the paper's window (BPSK 2+2, QPSK 1+0, 8-QAM 0+0)
  long    +-L symbols, target visible
  guard   +-L symbols with the target AND its two nearest neighbours blanked
          (clean waveform set to zero before the noise is added). Whatever this
          network gets right, it got from the sequence, not from the symbol.

Every recording repeats one short word tens to hundreds of times, so the ways a
long window can leak were each closed and checked:

  1. Captures, never symbols, are split; train and test captures are asserted
     disjoint.
  2. Words are enumerated under cyclic shift and constellation rotation. Only
     T1/T2 coincide (already one group in splits.py); every other withheld word
     is new to the network in every equivalent form.
  3. The blanked span is the target +-1 symbol. The anti-alias filter reaches
     +-0.88 symbol (context_bound.py section 5), so no filtered copy of the
     target reaches a visible symbol.
  4. All three inputs are scored on the identical target symbols: every capture
     loses the same L edge symbols at each end, whatever the window.
  5. A label-level lookup ceiling is fitted on the training captures only and
     scored on the test captures, plain and closed under rotation: the best any
     context-only predictor could do on that fold.
  6. Each test capture is flagged by whether its exact word occurs in
     training. A random 80/20 partition can withhold a word that has a single
     record; those captures are reported apart, not averaged in. T2 counts as
     unseen when only T1 was trained: it is T1 rotated by 180 degrees, and the
     long-window network did not carry T1's memorised sequence over to it.

Usage:
  window_control.py --audit     label-level checks only, no training
  window_control.py             audit, then training (resumable)
  window_control.py --summary   tables from results/window_control.csv

Output: results/window_control.csv, one row per (run, test capture).
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
import splits as SP
import data as DT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "window_control.csv")
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
wave, label, wo, lo = d["wave"], d["label"], d["wave_offset"], d["label_offset"]
SPS = int(d["sps_out"])
meta = SP.load_meta(ROOT)

MODS = ("8QAM", "QPSK", "BPSK")
CONFIGS = ("short", "long", "guard")
SHORT = {m: DT.CONTEXT[M.MAX_RECEPTIVE_FIELD_SYMBOLS[m]] for m in MODS}   # (before, after)
LONG = {"BPSK": 10, "QPSK": 6, "8QAM": 4}    # +-L; the audit asserts each is enough
GUARD = 1
PART_SEEDS = (0, 1, 2)
SEEDS = (0,)
EVAL_SNR = [-6, -3, 0, 3, 6, 9]              # -6 and -3 dB lie below the training range
WORKERS, THREADS = 4, 4


def rotate(y, mod, k):
    """Constellation rotation by k quarter-turns (BPSK: sign flip), as classical_snr.py."""
    y = np.asarray(y)
    if mod == "BPSK":
        return (y + k) % 2
    if mod == "QPSK":
        return (y + k) % 4
    return (y % 4 + k) % 4 + 4 * (y // 4)


def geometry(mod, config):
    """(before, after, blanked offsets) of the input window."""
    if config == "short":
        return SHORT[mod] + ((),)
    L = LONG[mod]
    blank = tuple(range(-GUARD, GUARD + 1)) if config == "guard" else ()
    return L, L, blank


def context_offsets(mod, config):
    """Neighbour offsets a context-only predictor could read in this window."""
    b, a, blank = geometry(mod, config)
    return [o for o in range(-b, a + 1) if o != 0 and o not in blank]


def folds(mod):
    out = [("leave_one_pattern_out", name, tr, te)
           for name, tr, te in SP.make_splits(meta, mod)["leave_one_pattern_out"]]
    sub = meta.index[meta.modulation == mod].to_numpy()
    for ps in PART_SEEDS:       # part_seed 0 is splits.make_splits' own partition
        idx = sub.copy()
        np.random.default_rng(ps).shuffle(idx)
        cut = max(1, int(round(0.2 * len(idx))))
        out.append(("random", f"random part_seed={ps}", idx[cut:], idx[:cut]))
    return out


# ------------------------------------------------------------------ windows
class Windows:
    """data.WindowSet with any (before, after), a common edge margin, and an
    optional blanked span. AGC and noise follow data.py exactly."""

    def __init__(self, caps, before, after, margin, blank=()):
        assert margin >= max(before, after)
        self.window = (before + after + 1) * SPS
        st, ys, cs = [], [], []
        for c in caps:
            n = int(lo[c + 1] - lo[c])
            assert wo[c + 1] - wo[c] == n * SPS
            j = np.arange(margin, n - margin, dtype=np.int64)
            st.append(wo[c] + (j - before) * SPS)
            ys.append(label[lo[c] + j])
            cs.append(np.full(len(j), c, dtype=np.int32))
        self.starts = np.concatenate(st)
        self.y = np.concatenate(ys).astype(np.int64)
        self.capture = np.concatenate(cs)
        rms = {int(c): float(np.sqrt((wave[wo[c]:wo[c + 1]].astype(np.float64) ** 2).mean()))
               for c in caps}
        self.ref = np.array([rms[int(c)] for c in self.capture], dtype=np.float32)
        self.grid = np.arange(self.window, dtype=np.int64)
        self.mask = np.ones(self.window, dtype=np.float32)
        for o in blank:
            s = (before + o) * SPS
            self.mask[s:s + SPS] = 0.0

    def __len__(self):
        return len(self.starts)

    def gather(self, idx, snr_db=None, rng=None):
        x = wave[self.starts[idx][:, None] + self.grid[None, :]].astype(np.float32)
        x *= self.mask                      # blank the CLEAN signal, then add noise
        ref = self.ref[idx][:, None]
        if snr_db is not None:
            snr = np.asarray(snr_db, dtype=np.float32).reshape(-1, 1)
            x += rng.standard_normal(x.shape).astype(np.float32) * (ref / np.sqrt(10.0 ** (snr / 10.0)))
        return (x / ref)[:, None, :]

    def class_weights(self, n_classes):
        c = np.bincount(self.y, minlength=n_classes).astype(np.float64)
        w = c.sum() / (len(c) * np.maximum(c, 1))
        return (w / w.mean()).astype(np.float32)


def build(mod, window):
    """CappedCNN with stride plans for the long windows. model.build() refuses
    windows over the cap - that refusal is the point of the paper, and this
    control bypasses it on purpose. Short windows keep the paper's plans."""
    import model as MD

    class Net(MD.CappedCNN):
        _STRIDES = {**MD.CappedCNN._STRIDES,
                    180: (2, 2, 2, 2), 260: (2, 2, 2, 2), 420: (2, 2, 2, 2)}
    return Net(M.n_classes(mod), window)


# ------------------------------------------------------------------ training
def job(spec):
    import torch
    import torch.nn as nn
    import train as T
    torch.set_num_threads(THREADS)

    mod, config, seed = spec["mod"], spec["config"], spec["seed"]
    split, fold, tr, te = spec["split"], spec["fold"], spec["tr"], spec["te"]
    assert not set(tr.tolist()) & set(te.tolist())
    b, a, blank = geometry(mod, config)
    ws_tr = Windows(tr, b, a, LONG[mod], blank)
    ws_te = Windows(te, b, a, LONG[mod], blank)

    t0 = time.time()
    torch.manual_seed(seed)
    net = build(mod, ws_tr.window)
    C = T.constellation(mod)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(ws_tr.class_weights(net.n_classes)))
    opt = torch.optim.Adam(net.parameters(), T.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T.EPOCHS)
    rng = np.random.default_rng(seed)
    for _ in range(T.EPOCHS):
        for bt in DT.epoch_batches(len(ws_tr), T.BATCH, rng):
            x = torch.from_numpy(ws_tr.gather(bt, rng.uniform(*T.TRAIN_SNR, size=len(bt)), rng))
            opt.zero_grad()
            logit, iq = net(x)
            (ce(logit, torch.from_numpy(ws_tr.y[bt])) +
             T.LAMBDA_CONST * T.const_loss(iq, C)).backward()
            opt.step()
        sched.step()

    net.eval()
    erng = np.random.default_rng(10_000 + seed)
    wrong = {}
    with torch.no_grad():
        for snr in EVAL_SNR + [None]:
            pred = np.concatenate([
                net(torch.from_numpy(ws_te.gather(np.arange(i, min(i + 4096, len(ws_te))),
                                                  snr, erng)))[0].argmax(1).numpy()
                for i in range(0, len(ws_te), 4096)])
            wrong["clean" if snr is None else f"snr{snr}"] = pred != ws_te.y

    train_words = set(meta.pattern[tr])
    rows = []
    for c in te:
        sel = ws_te.capture == c
        rows.append(dict(modulation=mod, config=config, split=split, fold=fold, seed=seed,
                         capture=int(c), board=meta.board[c], word=meta.pattern[c],
                         word_in_train=meta.pattern[c] in train_words, n=int(sel.sum()),
                         params=net.n_parameters(), window=ws_te.window,
                         secs=round(time.time() - t0, 1),
                         **{k: float(v[sel].mean()) for k, v in wrong.items()}))
    return rows


# ------------------------------------------------------------------ audit
def min_symmetric(w, guard):
    """Smallest L such that the cyclic +-L context, minus the target +-guard,
    determines the target everywhere in the word."""
    n = len(w)
    for L in range(guard + 1, 4 * n):
        offs = [o for o in range(-L, L + 1) if abs(o) > guard]
        seen = {}
        if all(seen.setdefault(tuple(w[(i + o) % n] for o in offs), w[i]) == w[i]
               for i in range(n)):
            return L
    return None


def lookup_ser(mod, tr, te, offs, closed):
    """Context -> label table fitted on training captures, scored on test
    captures, same target symbols as the networks. `closed`: every training
    capture is also entered under each constellation rotation."""
    K = 2 if mod == "BPSK" else 4
    m = LONG[mod]
    tab = defaultdict(Counter)
    for c in tr:
        y0 = label[lo[c]:lo[c + 1]].astype(int)
        for k in (range(K) if closed else (0,)):
            y = rotate(y0, mod, k)
            for i in range(m, len(y) - m):
                tab[tuple(y[i + o] for o in offs)][y[i]] += 1
    prior = Counter()
    for v in tab.values():
        prior.update(v)
    ok = tot = 0
    for c in te:
        y = label[lo[c]:lo[c + 1]].astype(int)
        for i in range(m, len(y) - m):
            k = tuple(y[i + o] for o in offs)
            ok += (tab[k] if k in tab else prior).most_common(1)[0][0] == y[i]
            tot += 1
    return 1 - ok / tot


def audit():
    print("1. word equivalence under cyclic shift + constellation rotation")
    for mod in MODS:
        K = 2 if mod == "BPSK" else 4
        cls = defaultdict(set)
        for c in meta.index[meta.modulation == mod]:
            w = label[lo[c]:lo[c] + int(meta.n_sym_word[c])].astype(int)
            forms = [tuple(np.roll(rotate(w, mod, k), s)) for k in range(K) for s in range(len(w))]
            cls[min(forms)].add(meta.group[c])
        groups = [sorted(v) for v in cls.values()]
        assert all(len(g) == 1 for g in groups), f"{mod}: equivalent words in different groups {groups}"
        print(f"   {mod:<5} {len(groups)} classes, one per pattern group: {sorted(sum(groups, []))}")

    print(f"\n2. smallest +-L whose neighbours determine the target (cyclic, per word)")
    for mod in MODS:
        need = {}
        for g in sorted(meta.group[meta.modulation == mod].unique()):
            c = meta.index[(meta.modulation == mod) & (meta.group == g)][0]
            w = list(label[lo[c]:lo[c] + int(meta.n_sym_word[c])].astype(int))
            need[g] = (min_symmetric(w, 0), min_symmetric(w, GUARD))
        worst = max(v[1] for v in need.values())
        print(f"   {mod:<5} " + "  ".join(f"{g} {a}/{b}" for g, (a, b) in need.items()) +
              f"   (target only / target+-{GUARD} blanked);  L = {LONG[mod]}")
        assert LONG[mod] >= worst, f"{mod}: L = {LONG[mod]} < {worst} needed"

    print("\n3. context-only lookup ceiling, SER, train captures -> test captures")
    print("   plain / closed under rotation; chance BPSK 0.500, QPSK 0.750, 8-QAM 0.875")
    for mod in MODS:
        for config in CONFIGS:
            offs = context_offsets(mod, config)
            res = defaultdict(list)
            for split, name, tr, te in folds(mod):
                res[split].append((name, lookup_ser(mod, tr, te, offs, False),
                                   lookup_ser(mod, tr, te, offs, True)))
            for split, v in res.items():
                p = np.mean([x[1] for x in v]); q = np.mean([x[2] for x in v])
                per = "  ".join(f"{n.split('=')[-1]} {a:.3f}/{b:.3f}" for n, a, b in v)
                print(f"   {mod:<5} {config:<6} {split:<22} {p:.3f}/{q:.3f}   {per}")


# ------------------------------------------------------------------ summary
def summary():
    r = pd.read_csv(CSV)
    # exact word, recomputed from the fold definitions (rows written before
    # 2026-10-31 stored the pattern GROUP here, which counts T2 as seen via T1)
    tr_of = {(m, name): tr for m in MODS for _, name, tr, _ in folds(m)}
    r["word"] = meta.pattern[r.capture].to_numpy()
    r["word_in_train"] = [meta.pattern[c] in set(meta.pattern[tr_of[(m, f)]])
                          for m, f, c in zip(r.modulation, r.fold, r.capture)]
    cols = [c for c in ["snr-6", "snr-3", "snr0", "snr3", "snr6", "clean"] if c in r]
    for c in cols:
        r[c + "_err"] = r[c] * r.n
    # fold SER pooled over its test captures; random partitions split by whether
    # the test capture's word was in training
    r["part"] = np.where(r.split == "leave_one_pattern_out", "LOPO (word withheld)",
                         np.where(r.word_in_train, "random, word seen", "random, word unseen"))
    g = r.groupby(["modulation", "config", "part", "fold", "seed"])
    fold = g[[c + "_err" for c in cols]].sum().div(g.n.sum(), axis=0)
    fold.columns = cols
    out = fold.groupby(["modulation", "config", "part"]).mean()
    out["folds"] = fold.groupby(["modulation", "config", "part"]).size()
    pd.set_option("display.width", 160)
    print(out.round(4).to_string())


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        sys.exit()
    audit()
    if args.audit:
        sys.exit()

    done = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    key = set(zip(done.modulation, done.config, done.fold, done.seed)) if len(done) else set()
    specs = [dict(mod=mod, config=cf, seed=s, split=sp, fold=name, tr=tr, te=te)
             for mod in MODS for sp, name, tr, te in folds(mod)
             for cf in CONFIGS for s in SEEDS if (mod, cf, name, s) not in key]
    print(f"\n{len(specs)} runs to do, {WORKERS} workers x {THREADS} threads", flush=True)
    rows = done.to_dict("records")
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(job, s): s for s in specs}
        for f in as_completed(futs):
            s, out = futs[f], f.result()
            rows += out
            pd.DataFrame(rows).to_csv(CSV, index=False)
            n = sum(o["n"] for o in out)
            ser = {k: sum(o[k] * o["n"] for o in out) / n for k in ("snr0", "clean")}
            print(f"{s['mod']:<5} {s['config']:<6} {s['fold']:<22} seed {s['seed']}  "
                  f"snr0 {ser['snr0']:.4f}  clean {ser['clean']:.4f}  "
                  f"({out[0]['secs']:.0f}s, {(time.time() - t0) / 60:.1f} min)", flush=True)
    print()
    summary()
