"""
physics_learned.py - did the network learn the constellation GEOMETRY, or just
a lookup from waveform to class label?

The ablation showed the constellation loss does not change the error rate, and
that its effect on the latent I/Q head cannot be read at lambda = 0 (the head
gets no gradient there and stays at initialisation). So that test says nothing.

This one does not touch the I/Q head at all. It asks a question the classifier
alone can answer:

    WHEN THE NETWORK IS WRONG, IS IT WRONG TOWARDS THE NEAREST NEIGHBOUR?

A model that had merely memorised a waveform-to-label lookup would scatter its
errors uniformly over the wrong classes. A model that has internalised the
constellation must confuse a point with its geometric neighbours, because that
is where additive noise actually moves a symbol. Measured against the exact
chance level for each modulation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
import model as MD
import splits as SP
import data as DT
import train as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
arrays = (d["wave"], d["label"], d["wave_offset"], d["label_offset"])
SPS = int(d["sps_out"])
meta = SP.load_meta(ROOT)


def neighbour_rank(mod):
    """For each constellation point, the other points ordered by distance."""
    C = T.constellation(mod).numpy()
    z = C[:, 0] + 1j * C[:, 1]
    n = len(z)
    order = np.zeros((n, n - 1), dtype=int)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        order[i] = sorted(others, key=lambda j: abs(z[i] - z[j]))
    return order


print("Are the network's errors geometrically structured?\n")
print(f"  {'mod':<6} {'errors':>8} {'-> nearest':>11} {'chance':>8} "
      f"{'ratio':>7}  verdict")
print("  " + "-" * 62)

for mod in ("QPSK", "8QAM"):
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    fold, tr, te = SP.make_splits(meta, mod)["leave_one_board_out"][1]
    net, res, _, _ = T.train_fold(mod, cap, tr, te, arrays, SPS, epochs=18)

    wave, label, wo, lo = arrays
    ws = DT.WindowSet(wave, label, wo, lo, te, SPS, cap)
    rng = np.random.default_rng(1)
    net.eval()
    preds, ys = [], []
    idx = np.arange(len(ws))
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            b = idx[i:i + 4096]
            x = torch.from_numpy(ws.gather(b, snr_db=0.0, rng=rng))
            preds.append(net(x)[0].argmax(1).numpy())
            ys.append(ws.y[b])
    pred = np.concatenate(preds)
    y = np.concatenate(ys)

    wrong = pred != y
    order = neighbour_rank(mod)
    n_cls = M.n_classes(mod)

    # save the full confusion matrix so the figure can be drawn from it
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    np.savetxt(os.path.join(ROOT, "results", f"confusion_{mod}_0dB.csv"),
               conf, fmt="%d", delimiter=",")
    # how many wrong predictions landed on the SINGLE nearest other point
    # (ties at equal distance are counted together, as they must be)
    C = T.constellation(mod).numpy()
    z = C[:, 0] + 1j * C[:, 1]
    hit, tot, chance_p = 0, 0, []
    for t in range(n_cls):
        sel = wrong & (y == t)
        if not sel.any():
            continue
        dist = np.array([abs(z[t] - z[j]) if j != t else np.inf
                         for j in range(n_cls)])
        nearest = np.flatnonzero(np.isclose(dist, dist.min()))
        hit += int(np.isin(pred[sel], nearest).sum())
        tot += int(sel.sum())
        chance_p.append(len(nearest) / (n_cls - 1) * sel.sum())
    frac = hit / max(tot, 1)
    ch = sum(chance_p) / max(tot, 1)
    print(f"  {mod:<6} {tot:>8,} {frac*100:>10.1f}% {ch*100:>7.1f}% "
          f"{frac/ch:>7.2f}x  "
          f"{'GEOMETRIC' if frac > ch * 1.3 else 'not distinguishable'}")

print("\n  'nearest' = the wrong prediction was the closest other constellation")
print("  point (all tied-closest points counted). Chance is the exact fraction")
print("  of wrong classes that are nearest neighbours, per true symbol.")
