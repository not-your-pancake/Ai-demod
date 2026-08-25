"""
data.py - turn the flat dataset into batches of symbol windows.

A window is `cap_symbols` consecutive symbols, arranged as symmetrically as the
cap allows around the symbol being labelled:

    BPSK   cap 5 -> [t-2 t-1  T  t+1 t+2]   100 samples
    QPSK   cap 2 -> [t-1        T       ]    40 samples
    8-QAM  cap 1 -> [           T       ]    20 samples

Symbols too close to a capture's edge to fill a window are dropped, which costs
at most 4 of ~3,600 symbols per capture.

Noise is added HERE, not in the stored dataset, so every epoch sees a fresh
draw rather than the same frozen copies, and the SNR axis can change without
rebuilding anything. This is the software AWGN the paper must declare: the
hardware supplies the waveform, we supply the noise.
"""

from __future__ import annotations

import numpy as np


CONTEXT = {  # (symbols before the target, symbols after)
    1: (0, 0),
    2: (1, 0),
    5: (2, 2),
}


class WindowSet:
    def __init__(self, wave, label, wave_offset, label_offset,
                 captures, sps: int, cap_symbols: int):
        before, after = CONTEXT[cap_symbols]
        # Reference level for BOTH normalisation and noise, taken over the
        # whole capture - i.e. an AGC, exactly what a real receiver runs.
        #
        # It must NOT be per-window. An 8-QAM window is one symbol, so dividing
        # by its own RMS makes every symbol unit-energy and erases the inner /
        # outer ring - half the constellation information. Measured: with
        # per-window normalisation, 8-QAM SER sat at 0.50 and did not improve
        # from 0 dB to clean, because the ring bit was pure guesswork.
        # A capture-level reference removes board gain (B2 runs 40% low) and
        # attenuation, while preserving symbol-to-symbol amplitude ratios.
        self._cap_rms = {}
        for c in captures:
            seg = wave[wave_offset[c]:wave_offset[c + 1]]
            self._cap_rms[int(c)] = float(np.sqrt((seg.astype(np.float64) ** 2).mean()))
        self.window = cap_symbols * sps
        self.wave = wave
        starts, ys, caps = [], [], []
        for c in captures:
            n_sym = int(label_offset[c + 1] - label_offset[c])
            j = np.arange(before, n_sym - after, dtype=np.int64)
            starts.append(wave_offset[c] + (j - before) * sps)
            ys.append(label[label_offset[c] + j])
            caps.append(np.full(len(j), c, dtype=np.int32))
        self.starts = np.concatenate(starts)
        self.y = np.concatenate(ys).astype(np.int64)
        self.capture = np.concatenate(caps)
        self.symbol_index = np.concatenate(
            [np.arange(before, int(label_offset[c + 1] - label_offset[c]) - after)
             for c in captures]).astype(np.int64)
        self._grid = np.arange(self.window, dtype=np.int64)
        self.ref = np.array([self._cap_rms[int(c)] for c in self.capture],
                            dtype=np.float32)

    def __len__(self):
        return len(self.starts)

    def gather(self, idx: np.ndarray, snr_db=None, rng=None) -> np.ndarray:
        """(len(idx), 1, window) float32, AGC-normalised, optionally with AWGN.

        Noise power is set from the CAPTURE reference too, not the window's own
        power. Otherwise an inner-ring symbol would receive less noise than an
        outer-ring one, which no physical channel does.
        """
        x = self.wave[self.starts[idx][:, None] + self._grid[None, :]]
        x = x.astype(np.float32, copy=True)
        ref = self.ref[idx][:, None]
        if snr_db is not None:
            rng = rng or np.random.default_rng()
            snr = np.asarray(snr_db, dtype=np.float32).reshape(-1, 1)
            sigma = ref / np.sqrt(10.0 ** (snr / 10.0))
            x += rng.standard_normal(x.shape).astype(np.float32) * sigma
        return (x / ref)[:, None, :]

    def class_weights(self, n_classes: int) -> np.ndarray:
        """Inverse-frequency weights. QPSK is skewed 0.565 rarest/commonest
        because Q1's quadrants are 2/3/3/4 and Q1 dominates the captures."""
        c = np.bincount(self.y, minlength=n_classes).astype(np.float64)
        w = c.sum() / (len(c) * np.maximum(c, 1))
        return (w / w.mean()).astype(np.float32)


def epoch_batches(n: int, batch: int, rng) -> list[np.ndarray]:
    idx = rng.permutation(n)
    return [idx[i:i + batch] for i in range(0, n, batch)]
