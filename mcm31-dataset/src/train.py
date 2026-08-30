"""
train.py - Step 3: train every fold, run every control.

Loss (handbook Section 11.3):
    L = L_CE(symbol) + lambda * L_const
    L_const = min over c in C of ||(I_hat, Q_hat) - c||^2
with the predicted (I,Q) normalised per batch. C is the Section 2.5 table:
BPSK two antipodal points, QPSK a single ring of four, 8-QAM a TWO-RING STAR -
never a 16-QAM square grid, which would fight the data term.

The ring ratio used is 1.89, the mean MEASURED across all 15 8-QAM captures
(range 1.867-1.914), not the handbook's quoted 1.965.

Controls that must FAIL, per Section 11.4:
  * shuffled labels   - break the waveform/label link; anything above chance
                        means a leak
  * position-only     - the best possible predictor that sees NO signal, just
                        the symbol's index within the word. On a random split
                        it scores 100% because the word repeats ~300 times;
                        that is precisely why the random split is inflated.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")

RING_RATIO = 1.89          # measured mean over all 8-QAM captures
EVAL_SNR = [0, 3, 6, 9, 12, 15, 20, 25, 30]
TRAIN_SNR = (0.0, 30.0)
EPOCHS = 25
BATCH = 512
LR = 2e-3
LAMBDA_CONST = 0.2


def constellation(mod: str) -> torch.Tensor:
    if mod == "BPSK":
        pts = np.array([1.0, -1.0], dtype=np.complex128)
    else:
        ph = np.exp(1j * np.deg2rad([45, 135, 225, 315]))
        pts = ph if mod == "QPSK" else np.concatenate([ph, RING_RATIO * ph])
    pts = pts / np.sqrt((np.abs(pts) ** 2).mean())      # unit RMS
    return torch.tensor(np.stack([pts.real, pts.imag], 1), dtype=torch.float32)


def const_loss(iq: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    iq = iq / iq.pow(2).mean().sqrt().clamp_min(1e-8)
    d = (iq[:, None, :] - C[None, :, :]).pow(2).sum(-1)
    return d.min(dim=1).values.mean()


def evaluate(net, ws, rng, snr_list) -> dict:
    net.eval()
    out = {}
    sel = np.arange(len(ws))
    with torch.no_grad():
        for snr in snr_list + [None]:
            correct = 0
            for i in range(0, len(sel), 4096):
                b = sel[i:i + 4096]
                x = torch.from_numpy(ws.gather(b, snr_db=snr, rng=rng))
                correct += int((net(x)[0].argmax(1).numpy() == ws.y[b]).sum())
            out["clean" if snr is None else f"snr{snr}"] = 1.0 - correct / len(sel)
    net.train()
    return out


def train_fold(mod, cap, tr_caps, te_caps, arrays, sps, seed=0,
               shuffle_labels=False, epochs=EPOCHS):
    wave, label, wo, lo = arrays
    if shuffle_labels:
        # Permute labels ONLY among the symbols of the captures in this fold.
        # Shuffling the global array would hand a BPSK capture an 8-QAM label
        # of 7, which is not a control - it is a crash, and before that a
        # different experiment entirely. The alphabet must stay correct; only
        # the waveform-to-label link may break.
        rs = np.random.default_rng(seed + 999)
        label = label.copy()
        span = np.concatenate([np.arange(lo[c], lo[c + 1])
                               for c in list(tr_caps) + list(te_caps)])
        label[span] = rs.permutation(label[span])
    ws_tr = DT.WindowSet(wave, label, wo, lo, tr_caps, sps, cap)
    ws_te = DT.WindowSet(wave, label, wo, lo, te_caps, sps, cap)
    assert not (set(ws_tr.capture.tolist()) & set(ws_te.capture.tolist()))

    torch.manual_seed(seed)
    net = MD.build(mod, sps, cap)
    C = constellation(mod)
    w = torch.tensor(ws_tr.class_weights(net.n_classes))
    opt = torch.optim.Adam(net.parameters(), LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        for b in DT.epoch_batches(len(ws_tr), BATCH, rng):
            snr = rng.uniform(*TRAIN_SNR, size=len(b))
            x = torch.from_numpy(ws_tr.gather(b, snr_db=snr, rng=rng))
            y = torch.from_numpy(ws_tr.y[b])
            opt.zero_grad()
            logit, iq = net(x)
            (ce(logit, y) + LAMBDA_CONST * const_loss(iq, C)).backward()
            opt.step()
        sched.step()
    return net, evaluate(net, ws_te, rng, EVAL_SNR), len(ws_tr), len(ws_te)


def position_baseline(mod, cap, tr_caps, te_caps, arrays, sps, n_word_by_capture):
    """Predictor that sees NO waveform - only the symbol's index within the word.

    Meaningful ONLY on leave-one-pattern-out. Labels are a periodic tiling of the
    transmitted word, and two records carrying the same word have label sequences
    that are identical up to a cyclic shift. So wherever the test partition
    reuses a training word - every leave-one-board-out, leave-one-attenuation-out
    and random fold - position determines the label exactly and the attainable
    error rate is zero, not chance. What this function returns on those splits is
    set by an arbitrary word-phase offset between records and means nothing. Read
    it only where the withheld word is absent from training.

    The modulus is each TEST record's own word length: BPSK carries a 63-symbol
    pseudorandom sequence in some records and a 24-symbol word in others, so one
    modulus per modulation is wrong.
    """
    wave, label, wo, lo = arrays
    n_cls = M.n_classes(mod)
    ws_tr = DT.WindowSet(wave, label, wo, lo, tr_caps, sps, cap)
    correct = total = 0
    for c in te_caps:
        n_word = int(n_word_by_capture[c])
        table = np.zeros((n_word, n_cls), dtype=np.int64)
        np.add.at(table, (ws_tr.symbol_index % n_word, ws_tr.y), 1)
        ws_te = DT.WindowSet(wave, label, wo, lo, [c], sps, cap)
        pred = table.argmax(1)[ws_te.symbol_index % n_word]
        correct += int((pred == ws_te.y).sum())
        total += len(ws_te.y)
    return 1.0 - correct / total
