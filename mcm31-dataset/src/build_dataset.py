"""
build_dataset.py - Step 1: turn 43 raw captures into a training-ready dataset.

What this produces, per capture:
  * the whole 6-second waveform resampled to exactly SPS_OUT samples per symbol,
    stored CONTIGUOUSLY so a window can include neighbouring symbols
  * one label per symbol, read off the coherent template (never the DIP switches)
  * metadata: board, modulation, pattern, attenuation

Deliberate choices, and why:

  PASSBAND, NOT BASEBAND. We keep the raw modulated waveform. Converting to I/Q
  would mean handing the network the carrier frequency, which is exactly the
  parameter the paper claims it never receives.

  NO NOISE ADDED HERE. Software AWGN is applied on the fly during training. That
  keeps this file small, lets the SNR axis change without a rebuild, and means
  every epoch sees fresh noise instead of the same 8 frozen copies.

  NO PER-WINDOW NORMALISATION HERE either - it depends on the window length,
  which depends on the receptive field, which is a training-time choice.

Run:  <venv>\python.exe src\build_dataset.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dataset")

SPS_OUT = 20                 # samples per symbol after decimation (handbook: 16-24)
ANTIALIAS_HZ = 4_000.0       # well above the 1500 Hz signal, well below the new
                             # Nyquist of 20 * 599.955 / 2 = 6000 Hz

PATTERN_TAGS = {"Q2", "Q3", "T2", "T3", "S1", "T4", "T5", "T6", "Q4", "Q5"}
DEFAULT_PATTERN = {"BPSK": "PRBS63", "QPSK": "Q1", "8QAM": "T1"}


def parse_name(path: str) -> dict:
    stem = os.path.splitext(os.path.basename(path))[0].upper()
    parts = stem.split("_")
    mod = next(p for p in parts if p in M.BITS_PER_SYMBOL)
    tag = parts[-1]
    if tag in PATTERN_TAGS:
        pattern, atten = tag, "A0"
    else:
        pattern, atten = DEFAULT_PATTERN[mod], tag
    return {"board": parts[0], "modulation": mod, "pattern": pattern,
            "attenuation": atten}


def build_one(path: str) -> dict:
    info = parse_name(path)
    mod, pat = info["modulation"], info["pattern"]
    cap = M.load_capture(path)

    # --- timing from the coherent template -------------------------------
    # The template is only used for the SYMBOL TIMING (tau) and the period.
    # Labels are derived further down from the whole capture, because the
    # rotation estimate must not come from 8 symbols - see below.
    n_sym_word = M.symbols_per_word(pat, mod)
    period = M.refine_period(cap.ch1, M.word_period_seconds(pat, mod) * cap.fs)
    tmpl = M.coherent_template(cap.ch1, period)
    _, tau = M.extract_symbols(tmpl.values, n_sym_word, return_tau=True)

    # --- anti-alias, then resample onto an exact symbol grid -------------
    # Zero-phase filtering (filtfilt) so symbol timing is not shifted.
    taps = firwin(129, ANTIALIAS_HZ, fs=cap.fs)
    x = filtfilt(taps, [1.0], cap.ch1.astype(np.float64))

    sps_in = period / n_sym_word           # raw samples per symbol, fractional
    # symbol i occupies [(i + tau - 0.5) * sps_in, (i + tau + 0.5) * sps_in)
    first = int(np.ceil((0.5 - tau) + 1))                       # keep in range
    last = int(np.floor((cap.n - 1) / sps_in - tau - 0.5)) - 1
    n_symbols = last - first
    if n_symbols < 100:
        raise RuntimeError(f"{os.path.basename(path)}: only {n_symbols} symbols")

    s = np.arange(first, last, dtype=np.float64)[:, None]
    m = np.arange(SPS_OUT, dtype=np.float64)[None, :]
    pos = (s + tau - 0.5) * sps_in + (m + 0.5) * (sps_in / SPS_OUT)
    wave = np.interp(pos.ravel(), np.arange(cap.n, dtype=np.float64), x)

    # --- labels ----------------------------------------------------------
    # Downconvert every symbol in the capture, then average over the ~300..450
    # repeats of the word. That is the same coherent averaging the template
    # does, so the SNR is unchanged, but it keeps ONE phase convention and lets
    # the rotation be estimated from all ~3,600 symbols instead of 8.
    #
    # Why that matters: the Mth-power rotation estimate has an M-fold
    # ambiguity. From 8 symbols it is noisy, and for the framing-affected
    # 8-QAM words the constellation is unbalanced (6 of 8 points, two doubled),
    # which biases it enough to pick a different quadrant. Measured across the
    # real captures that produced rotations 0,1,2,3 for 8-QAM - i.e. the same
    # physical symbol carried a different label in different captures, which
    # would poison training. Estimating from the full capture fixes it.
    w = wave.reshape(-1, SPS_OUT)
    lo_n = np.arange(SPS_OUT)
    lo_c = np.exp(-1j * 2 * np.pi * M.CYCLES_PER_SYMBOL * (lo_n + 0.5) / SPS_OUT)
    s_all = (w * lo_c[None, :]).mean(axis=1)

    order = M._POWER_ORDER[mod]
    tgt = order * M._REFERENCE_PHASE[mod]
    rot = np.angle(np.mean(s_all ** order) * np.exp(-1j * tgt)) / order

    n_reps = len(s_all) // n_sym_word
    s_word = s_all[:n_reps * n_sym_word].reshape(n_reps, n_sym_word).mean(axis=0)
    word_labels = M.quantise_symbols(s_word * np.exp(-1j * rot), mod)
    labels = word_labels[np.arange(n_symbols) % n_sym_word]

    # free consistency check: per-symbol decisions on the un-averaged capture
    # must agree with the tiled word labels
    per_sym = M.quantise_symbols(s_all * np.exp(-1j * rot), mod)
    self_agree = float((per_sym == labels).mean())

    return dict(
        wave=wave.astype(np.float32),
        labels=labels.astype(np.int8),
        n_symbols=int(n_symbols),
        first_symbol=int(first),
        sps_in=float(sps_in),
        tau=float(tau),
        period=float(period),
        template_snr=float(tmpl.snr_db),
        word_labels=word_labels,
        n_sym_word=int(n_sym_word),
        rms_mv=float(np.sqrt(np.mean((cap.ch1 - cap.ch1.mean()) ** 2)) * 1000),
        rotation_deg=float(np.rad2deg(rot)),
        self_agree=self_agree,
        **info,
    )


def main() -> int:
    files = sorted(
        {os.path.normcase(p): p
         for p in glob.glob(os.path.join(ROOT, "raw", "B*", "*.CSV"))}.values())
    if not files:
        print("no captures found")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Building dataset from {len(files)} captures  "
          f"(passband, {SPS_OUT} samples/symbol)\n")
    print(f"  {'file':<20} {'sym':>6} {'labels':<24} {'self':>7}  {'check'}")
    print("  " + "-" * 76)

    waves, labels, rows, offs_w, offs_l = [], [], [], [0], [0]
    problems = []

    for path in files:
        d = build_one(path)
        waves.append(d["wave"])
        labels.append(d["labels"])
        offs_w.append(offs_w[-1] + len(d["wave"]))
        offs_l.append(offs_l[-1] + len(d["labels"]))

        counts = np.bincount(d["word_labels"],
                             minlength=M.n_classes(d["modulation"]))
        note = ""
        # BPSK against PRBS63 is a free correctness check on the labels
        if d["modulation"] == "BPSK" and d["pattern"] == "PRBS63":
            mm = M.match_prbs63(d["word_labels"])
            if mm["score"] < 0.5:
                mm = M.match_prbs63(1 - d["word_labels"])
            note = f"PRBS63 {mm['score'] * 100:.0f}%"
            if mm["score"] < 1.0:
                problems.append(f"{d['board']}_{d['pattern']}: PRBS63 only "
                                f"{mm['score'] * 100:.1f}%")
        if d["self_agree"] < 0.99:
            problems.append(f"{os.path.basename(path)}: per-symbol decisions "
                            f"agree with tiled labels only "
                            f"{d['self_agree']*100:.1f}%")
        print(f"  {os.path.basename(path):<20} {d['n_symbols']:>6,} "
              f"{str(counts.tolist()):<24} {d['self_agree']*100:>6.2f}%  {note}")

        rows.append({k: d[k] for k in
                     ("board", "modulation", "pattern", "attenuation",
                      "n_symbols", "first_symbol", "sps_in", "tau", "period",
                      "template_snr", "n_sym_word", "rms_mv",
                      "rotation_deg", "self_agree")}
                    | {"filename": os.path.basename(path)})

    meta = pd.DataFrame(rows)
    wave_all = np.concatenate(waves)
    label_all = np.concatenate(labels)

    np.savez_compressed(
        os.path.join(OUT_DIR, "windows.npz"),
        wave=wave_all, label=label_all,
        wave_offset=np.array(offs_w, dtype=np.int64),
        label_offset=np.array(offs_l, dtype=np.int64),
        sps_out=np.int64(SPS_OUT))
    meta.to_csv(os.path.join(OUT_DIR, "captures.csv"), index=False)

    print(f"\n  total symbols : {len(label_all):,}")
    print(f"  waveform      : {len(wave_all):,} samples "
          f"({wave_all.nbytes / 1e6:.0f} MB as float32)")
    print(f"  written to    : {OUT_DIR}")

    print("\n  symbols per modulation:")
    for mod in ("BPSK", "QPSK", "8QAM"):
        sel = meta["modulation"] == mod
        n = int(meta.loc[sel, "n_symbols"].sum())
        print(f"    {mod:<5} {n:>8,} symbols   "
              f"{sel.sum()} captures   {meta.loc[sel,'pattern'].nunique()} patterns")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("\n  Every BPSK capture's labels reproduce PRBS63 exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
