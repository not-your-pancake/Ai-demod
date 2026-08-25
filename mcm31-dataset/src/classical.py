"""
classical.py - the textbook receiver, and how it depends on knowing the carrier.

Two things get measured:

  1. BLIND CLASSICAL RECEIVER. 4th-power carrier recovery, matched filter,
     hard decision. Given nothing but the waveform, exactly like the network.
     This is the honest baseline and it is expected to TIE - the handbook
     already killed any claim of beating it.

  2. CARRIER SENSITIVITY. The same receiver, but told a carrier that is wrong
     by 1, 5 or 20 Hz. Its constellation then rotates through the capture and
     the decision boundaries stop meaning anything.

     Framing matters here, and the honest framing is a SENSITIVITY ANALYSIS,
     not "we broke the baseline". A classical receiver must obtain fc from
     somewhere, and that estimate has an error bar. This measures how much
     error it tolerates. The learned receiver has no fc to be wrong about, so
     the question does not arise for it - that is the actual contribution.

     A tracking loop would follow a static offset, and the paper must say so.
     What it could not follow is not knowing the symbol rate or pulse shape
     either, which is the regime the network operates in.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M

FS_DEC = None   # filled from sps at call time


def blind_carrier(wave: np.ndarray, sps: int, fs_dec: float) -> float:
    """4th-power carrier estimate on the decimated waveform. No hints."""
    x = wave.astype(np.float64) - wave.mean()
    from scipy.signal import hilbert
    z = hilbert(x) ** 4
    nfft = 1 << int(np.ceil(np.log2(len(z))))
    spec = np.abs(np.fft.rfft(z.real, nfft) + 1j * np.fft.rfft(z.imag, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs_dec)
    band = (f > 4 * M.FC_NOMINAL - 80) & (f < 4 * M.FC_NOMINAL + 80)
    i = np.flatnonzero(band)[np.argmax(spec[band])]
    a, b, c = spec[i - 1], spec[i], spec[i + 1]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den else 0.0
    return float((f[i] + delta * (f[1] - f[0])) / 4)


def demodulate(wave: np.ndarray, mod: str, sps: int, fc: float,
               fs_dec: float) -> np.ndarray:
    """Downconvert at the SUPPLIED fc, matched-filter, hard-decide."""
    n = np.arange(len(wave), dtype=np.float64)
    bb = wave.astype(np.float64) * np.exp(-2j * np.pi * fc * n / fs_dec)
    sym = bb.reshape(-1, sps).mean(axis=1)
    return M.quantise_symbols(M.derotate(sym, mod), mod)


def add_noise(wave: np.ndarray, snr_db, rng) -> np.ndarray:
    if snr_db is None:
        return wave
    ref = np.sqrt((wave.astype(np.float64) ** 2).mean())
    return wave + rng.standard_normal(len(wave)) * ref / np.sqrt(10 ** (snr_db / 10))


if __name__ == "__main__":
    import pandas as pd
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
    meta = pd.read_csv(os.path.join(ROOT, "dataset", "captures.csv"))
    SPS = int(d["sps_out"]); wave, label = d["wave"], d["label"]
    wo, lo = d["wave_offset"], d["label_offset"]
    fs_dec = SPS * M.SYMBOL_RATE
    rng = np.random.default_rng(0)

    print(f"decimated sample rate {fs_dec:.1f} Hz, carrier {M.FC_NOMINAL} Hz\n")
    print("1) BLIND CARRIER RECOVERY on the decimated waveform")
    est = [blind_carrier(wave[wo[i]:wo[i+1]], SPS, fs_dec) for i in range(6)]
    print(f"   first 6 captures: {np.round(est, 3).tolist()}")
    print(f"   error vs 1199.91 Hz: max {max(abs(np.array(est)-M.FC_NOMINAL)):.4f} Hz\n")

    print("2) SER vs CARRIER ERROR  (clean signal, blind receiver otherwise)")
    print(f"   {'mod':<6} " + "".join(f"{f'{dfh:+g} Hz':>12}" for dfh in (0, 1, 5, 20)))
    print("   " + "-" * 56)
    rows = []
    for mod in ("BPSK", "QPSK", "8QAM"):
        sel = meta.index[meta.modulation == mod][:6]
        line = {}
        for dfh in (0, 1, 5, 20):
            sers = []
            for i in sel:
                w = wave[wo[i]:wo[i+1]]
                y = label[lo[i]:lo[i+1]]
                p = demodulate(w, mod, SPS, M.FC_NOMINAL + dfh, fs_dec)
                # allow the free global rotation a real receiver would resolve
                K = 2 if mod == "BPSK" else 4
                best = max(float((((p % 4 + k) % 4 + 4*(p//4)) if mod == "8QAM"
                                  else (p + k) % K == y).mean()
                                 if mod != "8QAM" else
                                 (((p % 4 + k) % 4 + 4*(p//4)) == y).mean())
                           for k in range(K))
                sers.append(1 - best)
            line[dfh] = float(np.mean(sers))
        print(f"   {mod:<6} " + "".join(f"{line[dfh]:>12.4f}" for dfh in (0, 1, 5, 20)))
        rows.append(dict(modulation=mod, **{f"df_{k}Hz": v for k, v in line.items()}))
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "carrier_sensitivity.csv"),
                              index=False)
    print(f"\n   chance SER: BPSK 0.500  QPSK 0.750  8-QAM 0.875")
