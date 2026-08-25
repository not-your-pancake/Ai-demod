"""
diagnose_tp9.py - work out what TP-9 (CH2) is actually doing in each capture.

Why this exists: handbook check 10 scores TP-9 by SELF-consistency, i.e. does it
repeat. A demodulator that has lost lock and is free-running on mains pickup
also repeats, perfectly. So self-consistency cannot separate "locked" from
"stuck", and on B1 it produced a non-monotonic lock/unlock/lock pattern that no
demodulator should show.

This measures TP-9 against things that cannot be faked:
  * how many edges it produces, versus how many the transmitted word requires
  * its own dominant period, found blind
  * for BPSK, its BER against the PRBS63 ground truth, with a chance control
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATTERN = {"BPSK": "PRBS63", "QPSK": "Q1", "8QAM": "T1"}


def digitize(ch2: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(ch2, [2, 98])
    return (ch2 > 0.5 * (lo + hi)).astype(np.int8)


def dominant_period(bits: np.ndarray, lo_lag: int, hi_lag: int) -> tuple[float, float]:
    """Blind period estimate by normalised autocorrelation."""
    x = bits.astype(np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft)[:n]
    ac /= ac[0] + 1e-30
    hi_lag = min(hi_lag, n - 1)
    seg = ac[lo_lag:hi_lag]
    if len(seg) == 0:
        return float("nan"), float("nan")
    k = int(np.argmax(seg)) + lo_lag
    return float(k), float(ac[k])


def tp9_bits_at(ch2: np.ndarray, period_samples: float, n_bits: int) -> np.ndarray:
    """Fold TP-9 at a period, then read n_bits with a blind timing search."""
    seg = M.fold(digitize(ch2).astype(np.float64), period_samples)
    avg = seg.mean(axis=0)
    m = len(avg)
    ext = np.tile(avg, 3)
    eidx = np.arange(len(ext), dtype=np.float64)
    sps = m / n_bits

    def sample(tau):
        return np.interp(m + (np.arange(n_bits) + tau) * sps, eidx, ext)

    taus = np.arange(0.0, 1.0, 1.0 / 128)
    tau = float(taus[int(np.argmax([np.abs(sample(t) - 0.5).mean() for t in taus]))])
    return (sample(tau) > 0.5).astype(np.int8)


def main() -> int:
    files = []
    seen = set()
    for p in sorted(glob.glob(os.path.join(ROOT, "raw", "B1", "*.CSV"))):
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            files.append(p)

    print(f"{'file':<20} {'RMS mV':>7} {'edges':>7} {'expect':>7} {'dom.period ms':>14} "
          f"{'r':>6} {'TP-9 vs PRBS63':>15}")
    print("-" * 90)

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0].upper()
        mod = next(p for p in stem.split("_") if p in M.BITS_PER_SYMBOL)
        pattern = DEFAULT_PATTERN[mod]

        cap = M.load_capture(path)
        rms = np.sqrt(np.mean((cap.ch1 - cap.ch1.mean()) ** 2)) * 1000
        bits = digitize(cap.ch2)

        n_edges = int(np.count_nonzero(np.diff(bits)))

        # how many edges the transmitted word actually requires
        word = M.bits_of(pattern)
        runs_per_word = int(np.count_nonzero(np.diff(np.concatenate([word, word[:1]]))))
        word_P = M.word_period_seconds(pattern, mod) * cap.fs
        n_words = cap.n / word_P
        expected_edges = int(round(runs_per_word * n_words))

        bit_period = cap.fs / (M.SYMBOL_RATE * M.BITS_PER_SYMBOL[mod])
        lag, r = dominant_period(bits, int(0.5 * bit_period), 30_000)
        dom_ms = lag / cap.fs * 1e3

        # autocorrelation at exact multiples of the transmitted word period -
        # a mis-framed but locked TP-9 (the handbook's 48-bit 8-QAM case) shows
        # up here even when the global peak sits somewhere useless
        xf = bits.astype(np.float64)
        xf -= xf.mean()
        nfft = 1 << int(np.ceil(np.log2(2 * len(xf))))
        sp = np.fft.rfft(xf, n=nfft)
        acf = np.fft.irfft(sp * np.conj(sp), n=nfft)[: len(xf)]
        acf /= acf[0] + 1e-30
        mults = " ".join(
            f"{k}x:{np.interp(k * word_P, np.arange(len(acf)), acf):+.2f}"
            for k in (1, 2, 3, 4)
        )

        verdict = ""
        if mod == "BPSK" and pattern == "PRBS63":
            rec = tp9_bits_at(cap.ch2, word_P, 63)
            m = M.match_prbs63(rec)
            if m["score"] < 0.5:
                m = M.match_prbs63(1 - rec)
            verdict = (f"{m['score'] * 100:5.1f}%"
                       f"{'  ABOVE' if m['above_chance'] else '  chance'}")

        print(f"{os.path.basename(path):<20} {rms:>7.1f} {n_edges:>7,} {expected_edges:>7,} "
              f"{dom_ms:>11.3f} ms {r:>6.3f} {verdict:>15}")
        print(f"{'':<20} autocorr at word-period multiples:  {mults}")

    print("\nNotes:")
    print("  edges   = transitions on TP-9 across the whole 6 s capture")
    print("  expect  = transitions the transmitted word requires over the same span")
    print("  edges far BELOW expect => TP-9 is stuck or free-running, not decoding")
    print("  TP-9 vs PRBS63 chance level is ~66.6% (best of 63 rotations x 2 polarities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
