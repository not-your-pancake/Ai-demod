"""
ablate_physics.py - does the constellation loss actually do anything?

The paper wants to call this a "physics-guided" CNN. The only physics term in
the model is the constellation loss:

    L = L_CE(symbol) + lambda * min over c in C of ||(I_hat, Q_hat) - c||^2

with C the measured Section 2.5 geometry (BPSK antipodal pair, QPSK single
ring of four, 8-QAM two-ring star at the MEASURED ratio 1.89).

Calling a model physics-guided because a term exists in the loss is decoration.
This measures two things instead:

  1. Does lambda change the symbol error rate at all?
  2. With lambda = 0 - no physics whatsoever - does the latent I/Q head still
     organise itself into the constellation? If it does, the geometry is IN the
     data and the loss term is merely restating it. If it does not, the term is
     supplying real structure.

Verdict is printed, not assumed.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def run(mod, lam, seed=0, epochs=18):
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    fold, tr, te = SP.make_splits(meta, mod)["leave_one_board_out"][1]
    wave, label, wo, lo = arrays
    ws_tr = DT.WindowSet(wave, label, wo, lo, tr, SPS, cap)
    ws_te = DT.WindowSet(wave, label, wo, lo, te, SPS, cap)

    torch.manual_seed(seed)
    net = MD.build(mod, SPS, cap)
    C = T.constellation(mod)
    w = torch.tensor(ws_tr.class_weights(net.n_classes))
    opt = torch.optim.Adam(net.parameters(), T.LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        for b in DT.epoch_batches(len(ws_tr), T.BATCH, rng):
            snr = rng.uniform(*T.TRAIN_SNR, size=len(b))
            x = torch.from_numpy(ws_tr.gather(b, snr_db=snr, rng=rng))
            y = torch.from_numpy(ws_tr.y[b])
            opt.zero_grad()
            logit, iq = net(x)
            loss = ce(logit, y)
            if lam > 0:
                loss = loss + lam * T.const_loss(iq, C)
            loss.backward()
            opt.step()
        sched.step()

    res = T.evaluate(net, ws_te, rng, [0, 3, 6])

    # how well does the latent I/Q head organise into the constellation?
    net.eval()
    with torch.no_grad():
        sel = np.arange(0, len(ws_te), max(1, len(ws_te) // 6000))
        iq = net(torch.from_numpy(ws_te.gather(sel)))[1].numpy()
    z = iq[:, 0] + 1j * iq[:, 1]
    z = z / np.sqrt((np.abs(z) ** 2).mean())
    Cz = C[:, 0].numpy() + 1j * C[:, 1].numpy()
    # best rotation, then mean distance to the nearest ideal point
    best = None
    for th in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        dist = np.abs((z * np.exp(1j * th))[:, None] - Cz[None, :]).min(axis=1).mean()
        best = dist if best is None or dist < best else best
    # a null: how far would randomly-placed points of the same power be?
    rs = np.random.default_rng(0)
    rnd = rs.standard_normal(len(z)) + 1j * rs.standard_normal(len(z))
    rnd = rnd / np.sqrt((np.abs(rnd) ** 2).mean())
    null = np.abs(rnd[:, None] - Cz[None, :]).min(axis=1).mean()
    return res, float(best), float(null), ws_te.y[sel], z


print("Does the constellation loss change anything?  "
      "(8-QAM and QPSK, leave-one-board-out)\n")
print(f"  {'mod':<6} {'lambda':>7} {'SER@0dB':>9} {'SER@3dB':>9} {'SER@6dB':>9} "
      f"{'IQ->const dist':>15} {'random null':>12}")
print("  " + "-" * 76)
rows = []
for mod in ("8QAM", "QPSK"):
    for lam in (0.0, 0.2, 1.0):
        t0 = time.time()
        res, dist, null, _, _ = run(mod, lam)
        rows.append(dict(modulation=mod, lam=lam, iq_dist=dist, null=null, **res))
        print(f"  {mod:<6} {lam:>7.1f} {res['snr0']:>9.4f} {res['snr3']:>9.4f} "
              f"{res['snr6']:>9.4f} {dist:>15.3f} {null:>12.3f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "ablation_physics.csv"),
                          index=False)
print("\n  IQ->const dist: mean distance from the latent I/Q head to the nearest")
print("  ideal constellation point, after the best global rotation. Lower means")
print("  the head has organised itself into the constellation. The random null")
print("  is what unstructured points of the same power would score.")
print("\n  saved -> results/ablation_physics.csv")
