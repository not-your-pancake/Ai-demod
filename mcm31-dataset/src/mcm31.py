"""
mcm31.py - core library for the MCM31 neural demodulator project.

Design rules taken from the handbook (Section 0):
  * Never assume a number. Every constant here is either quoted from the
    handbook's *measured* table or recomputed from the data at runtime.
  * Ground truth comes from the coherent template, never from the DIP switches.
  * Every claimed match must be compared against a chance-level control.

Nothing in this file trains anything. It only reads captures, verifies them,
and produces ground-truth templates and symbol labels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Measured constants (handbook Section 2). Do not re-derive from theory.
# ---------------------------------------------------------------------------

FS_NOMINAL = 87_260.0          # Hz, oscilloscope sample rate
FC_NOMINAL = 1_199.91          # Hz, carrier
CYCLES_PER_SYMBOL = 2          # exact, and this is what makes coherence work
SYMBOL_RATE = FC_NOMINAL / CYCLES_PER_SYMBOL   # 599.955 baud

N_SAMPLES_NOMINAL = 523_600
EXPECTED_HEADER = "[s],CH1[V],CH2[V]"

# Handbook Section 2.4. Verified at import time by verify_prbs63().
PRBS63 = "001000100101101100011101000011010111001111011111000000101010011"

# Handbook Section 3.2. 24-bit DIP words.
PATTERNS = {
    "Q1": "110010000101101011011110",
    "T1": "000111001110010101011100",
    "Q2": "001001111000110101110010",
    "Q3": "110100101100011000111001",
    "T2": "111000110001101010100011",
    "T3": "010101000111011100110001",
    "S1": "011000101010111100001110",
    # added 2026-08-16 for Block 2 - chosen to maximise symbol-transition
    # coverage (8-QAM went from 20/64 transitions to 43/64). See
    # generate_patterns.py for the selection criterion and exclusions.
    "T4": "000100101110111010001011",
    "T5": "110011001000101100010111",
    "T6": "011010110101001100111000",
    "Q4": "110100101101001001001011",
    "Q5": "001101011010110100101100",
}

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "8QAM": 3}

# Handbook Section 11.1 - correctness constraint, not a hyperparameter.
MAX_RECEPTIVE_FIELD_SYMBOLS = {"BPSK": 5, "QPSK": 2, "8QAM": 1}


# ---------------------------------------------------------------------------
# Self-check of the handbook's own PRBS claim (handbook rule 1 applied to it)
# ---------------------------------------------------------------------------

def verify_prbs63() -> dict:
    """Re-derive every claim the handbook makes about PRBS63 from the string."""
    s = np.array([int(c) for c in PRBS63], dtype=np.int8)
    out = {"length": len(s), "ones": int(s.sum()), "zeros": int((1 - s).sum())}

    # run-length spectrum over one *circular* period
    runs, cur = [], 1
    ext = np.concatenate([s, s[:1]])
    for i in range(1, len(s)):
        if ext[i] == ext[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    spec = {}
    for r in runs:
        spec[r] = spec.get(r, 0) + 1
    out["run_spectrum"] = dict(sorted(spec.items()))
    out["n_runs"] = len(runs)

    # the COMPLEMENT should satisfy s[i+6] = s[i] XOR s[i+5]  (mask 33)
    c = 1 - s
    ok = all(c[(i + 6) % 63] == (c[i] ^ c[(i + 5) % 63]) for i in range(63))
    out["complement_satisfies_lfsr"] = bool(ok)
    return out


def bits_of(pattern_id: str) -> np.ndarray:
    """Return the bit array for a pattern ID ('PRBS63' or a 24-bit DIP word)."""
    src = PRBS63 if pattern_id.upper() == "PRBS63" else PATTERNS[pattern_id.upper()]
    return np.array([int(c) for c in src], dtype=np.int8)


def word_period_seconds(pattern_id: str, modulation: str) -> float:
    """Word period = (bits / bits-per-symbol) symbols at the symbol rate."""
    n_bits = len(bits_of(pattern_id))
    n_sym = n_bits / BITS_PER_SYMBOL[modulation]
    if abs(n_sym - round(n_sym)) > 1e-9:
        raise ValueError(
            f"{pattern_id} has {n_bits} bits, which does not group evenly into "
            f"{BITS_PER_SYMBOL[modulation]}-bit {modulation} symbols"
        )
    return n_sym / SYMBOL_RATE


def symbols_per_word(pattern_id: str, modulation: str) -> int:
    return len(bits_of(pattern_id)) // BITS_PER_SYMBOL[modulation]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Capture:
    path: str
    header: str
    t: np.ndarray
    ch1: np.ndarray          # TP-20, the modulated waveform (network input)
    ch2: np.ndarray          # TP-9, the board's recovered data (baseline only)
    n: int
    fs: float
    duration: float
    meta: dict = field(default_factory=dict)


def load_capture(path: str) -> Capture:
    """Read an R&S HMO1002 CSV export. Tolerant about the exact header text,
    strict about there being three numeric columns."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline().strip()

    def _is_numeric_row(line: str) -> bool:
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        try:
            [float(p) for p in parts if p != ""]
            return len(parts) >= 2
        except ValueError:
            return False

    has_header = not _is_numeric_row(first)
    sep = ";" if first.count(";") > first.count(",") else ","

    df = pd.read_csv(
        path,
        sep=sep,
        header=0 if has_header else None,
        encoding="utf-8-sig",
        engine="c",
    )
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)

    if df.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)}: only {df.shape[1]} numeric column(s)")

    t = df.iloc[:, 0].to_numpy(dtype=np.float64)
    ch1 = df.iloc[:, 1].to_numpy(dtype=np.float64)
    ch2 = (df.iloc[:, 2].to_numpy(dtype=np.float64)
           if df.shape[1] >= 3 else np.full_like(ch1, np.nan))

    good = np.isfinite(t) & np.isfinite(ch1)
    t, ch1, ch2 = t[good], ch1[good], ch2[good]

    n = len(t)
    duration = float(t[-1] - t[0])
    fs = (n - 1) / duration if duration > 0 else float("nan")

    return Capture(
        path=path,
        header=first if has_header else "(no header row)",
        t=t, ch1=ch1, ch2=ch2,
        n=n, fs=fs, duration=duration,
        meta={"n_columns": int(df.shape[1])},
    )


# ---------------------------------------------------------------------------
# Carrier estimate - 4th power (handbook Section 9, check 6)
# ---------------------------------------------------------------------------

def estimate_carrier_4th(x: np.ndarray, fs: float, fc_hint: float = FC_NOMINAL) -> float:
    """Raising to the 4th power collapses every constellation used here:
    BPSK phases {0,180} and the 4-phase star {45,135,225,315} all map to a
    single tone at 4*fc. No knowledge of the modulation is needed."""
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    # Work on a power-of-two-ish slice for FFT speed; 2^19 = 524288 >= 523600
    z = hilbert(x) ** 4
    nfft = 1 << int(np.ceil(np.log2(len(z))))
    spec = np.abs(np.fft.rfft(z.real, n=nfft) + 1j * np.fft.rfft(z.imag, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    target = 4.0 * fc_hint
    band = (freqs > target - 60) & (freqs < target + 60)
    if not band.any():
        return float("nan")
    idx = np.flatnonzero(band)[np.argmax(spec[band])]

    # parabolic interpolation for sub-bin accuracy
    if 0 < idx < len(spec) - 1:
        a, b, c = spec[idx - 1], spec[idx], spec[idx + 1]
        denom = (a - 2 * b + c)
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return float((freqs[idx] + delta * (freqs[1] - freqs[0])) / 4.0)


# ---------------------------------------------------------------------------
# Folding, period refinement, coherent template (handbook Section 10.1)
# ---------------------------------------------------------------------------

def fold(x: np.ndarray, period_samples: float, n_bins: int | None = None,
         max_periods: int | None = None) -> np.ndarray:
    """Cut x into K consecutive periods of *fractional* length `period_samples`
    and resample each onto a common grid of `n_bins` points.

    The period is never a whole number of samples (e.g. QPSK is 1745.2), so
    plain reshaping would slowly drift out of alignment. Interpolation fixes it.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    P = float(period_samples)
    if n_bins is None:
        n_bins = int(round(P))
    k_max = int(np.floor((n - 1) / P))
    if max_periods is not None:
        k_max = min(k_max, max_periods)
    if k_max < 2:
        raise ValueError(f"only {k_max} full period(s) fit in the capture")

    k = np.arange(k_max)[:, None]
    m = np.arange(n_bins)[None, :]
    pos = k * P + m * (P / n_bins)
    vals = np.interp(pos.ravel(), np.arange(n, dtype=np.float64), x)
    return vals.reshape(k_max, n_bins)


def refine_period(x: np.ndarray, period_samples: float,
                  span: float = 6.0, coarse: float = 0.2,
                  fine: float = 0.02) -> float:
    """Handbook 10.1 step 2: maximise the energy of the coherent average over a
    +/-6 sample window in 0.02-sample steps. Done as coarse-then-fine so it
    takes seconds rather than minutes.

    n_bins is pinned to the nominal period so that every candidate is scored on
    the same grid - otherwise longer periods win by simply having more bins.
    """
    n_bins = int(round(period_samples))

    def energy(P: float) -> float:
        try:
            return float(fold(x, P, n_bins=n_bins).mean(axis=0).var())
        except ValueError:
            return -np.inf

    best = period_samples
    for step, width in ((coarse, span), (fine, 1.5 * coarse)):
        cands = np.arange(best - width, best + width + 0.5 * step, step)
        best = float(cands[int(np.argmax([energy(P) for P in cands]))])
    return best


@dataclass
class Template:
    values: np.ndarray       # one word period of the averaged passband waveform
    period_samples: float
    n_periods: int           # K
    snr_db: float
    residual_var: float


def coherent_template(x: np.ndarray, period_samples: float) -> Template:
    """Average all K periods of the RAW PASSBAND waveform (handbook 10.1 step 3).

    SNR uses the handbook's estimator, which removes the averaging bias:
        10*log10( (var(template) - var(residual)/K) / var(residual) )
    so the number reported is the single-period SNR, not the flattered
    post-averaging one.
    """
    seg = fold(x, period_samples)
    k = seg.shape[0]
    tmpl = seg.mean(axis=0)
    resid = seg - tmpl[None, :]
    resid_var = float(resid.var(ddof=1))
    signal_var = float(tmpl.var()) - resid_var / k
    snr = 10.0 * np.log10(signal_var / resid_var) if signal_var > 0 and resid_var > 0 else -np.inf
    return Template(values=tmpl, period_samples=float(period_samples),
                    n_periods=int(k), snr_db=float(snr), residual_var=resid_var)


def circular_extend(template: np.ndarray, times: int = 3) -> np.ndarray:
    """Handbook 10.1 step 5. Without this the last symbol lands on the wrap
    boundary and comes out corrupted - this produced a spurious radius of 0.07
    in an early run."""
    return np.tile(np.asarray(template), times)


# ---------------------------------------------------------------------------
# Symbol extraction from a template
# ---------------------------------------------------------------------------

def extract_symbols(template: np.ndarray, n_symbols: int,
                    tau: float | None = None,
                    return_tau: bool = False):
    """Downconvert the template and return one complex sample per symbol.

    Nothing is supplied from outside. The carrier is fixed by the geometry of
    the word: there are exactly CYCLES_PER_SYMBOL carrier cycles per symbol, so
    one word holds n_symbols*2 cycles whatever the scope's sample rate was.

    The sampling instant is NOT known, because the board's transmit filter has
    group delay. `tau` (0..1, in symbols) is therefore recovered blind, by
    maximising the mean symbol energy at the matched-filter output - the
    classic non-data-aided timing criterion. Skipping this step makes symbols
    land part-way through a transition and produces the collapsed radii that
    the handbook blames on the wrap boundary; both faults look identical, so
    fix the timing first and only then suspect the extension.
    """
    tmpl = np.asarray(template, dtype=np.float64)
    m = len(tmpl)
    ext = circular_extend(tmpl, 3)

    # phase advances by exactly 2*pi*CYCLES_PER_SYMBOL per symbol; because
    # cycles_per_word is an integer, this stays continuous across the copies
    cycles_per_word = CYCLES_PER_SYMBOL * n_symbols
    idx = np.arange(len(ext), dtype=np.float64)
    bb = ext * np.exp(-1j * 2 * np.pi * cycles_per_word * idx / m)

    # matched filter for a rectangular pulse = boxcar of one symbol
    sps = m / n_symbols
    win = max(1, int(round(sps)))
    bb = np.convolve(bb, np.ones(win) / win, mode="same")

    def _sample(t: float) -> np.ndarray:
        centres = m + (np.arange(n_symbols) + t) * sps      # middle copy only
        return (np.interp(centres, idx, bb.real)
                + 1j * np.interp(centres, idx, bb.imag))

    if tau is None:
        grid = np.arange(0.0, 1.0, 1.0 / 256)
        tau = float(grid[int(np.argmax([np.abs(_sample(t)).mean() ** 2 for t in grid]))])
        fine = np.linspace(tau - 1.0 / 256, tau + 1.0 / 256, 33)
        tau = float(fine[int(np.argmax([np.abs(_sample(t)).mean() ** 2 for t in fine]))])

    sym = _sample(tau)
    return (sym, tau) if return_tau else sym


# 4th-power collapses the star's phases onto 180 deg, not 0 deg, because the
# reference phase is 45 deg. Getting this wrong rotates the whole constellation
# by 45 deg and silently misaligns the Section 2.5 loss.
_REFERENCE_PHASE = {"BPSK": 0.0, "QPSK": np.pi / 4, "8QAM": np.pi / 4}
_POWER_ORDER = {"BPSK": 2, "QPSK": 4, "8QAM": 4}


def derotate(sym: np.ndarray, modulation: str) -> np.ndarray:
    """Remove the arbitrary absolute phase with an Mth-power estimate, which
    needs no labels, and land the result on the canonical constellation angles.

    The rotation is applied BEFORE np.angle, not subtracted after. That looks
    like a cosmetic difference and is not. For QPSK and 8-QAM the reference
    phase is 45 deg, so the 4th-power sum sits at ~180 deg - exactly on
    np.angle's branch cut. Two captures whose phase is physically identical can
    then read -178.8 and +179.7, and dividing by 4 puts their constellations
    90 deg apart, silently relabelling every symbol. Measured across the real
    captures the phase is highly reproducible (circular concentration 0.995+),
    so that 90 deg jump was pure branch instability. Rotating first moves the
    measurement to ~0 and the cut to +-180, where no data sits.

    This does not affect rotation-invariant quantities (fingerprints, radii,
    bijection searches) - only quantised labels compared ACROSS captures.
    """
    order = _POWER_ORDER[modulation]
    target = order * _REFERENCE_PHASE[modulation]
    rot = np.angle(np.mean(sym ** order) * np.exp(-1j * target)) / order
    return sym * np.exp(-1j * rot)


# ---------------------------------------------------------------------------
# BPSK cross-check against PRBS63 (handbook Section 10.1, last line)
# ---------------------------------------------------------------------------

def match_prbs63(bits: np.ndarray) -> dict:
    """Align a recovered bit sequence to PRBS63 over all 63 rotations and both
    polarities. ALWAYS reported together with the chance level (handbook rule 3).
    """
    ref = bits_of("PRBS63")
    b = np.asarray(bits, dtype=np.int8).ravel()
    if len(b) != 63:
        raise ValueError(f"expected 63 bits, got {len(b)}")

    best = {"score": -1.0, "rotation": None, "inverted": None}
    for inv in (False, True):
        cand = 1 - b if inv else b
        for r in range(63):
            score = float((np.roll(ref, r) == cand).mean())
            if score > best["score"]:
                best = {"score": score, "rotation": r, "inverted": inv}

    # chance control: same search, but on random bits
    rng = np.random.default_rng(0)
    null = []
    for _ in range(400):
        rb = rng.integers(0, 2, 63).astype(np.int8)
        s = max(float((np.roll(ref, r) == (1 - rb if inv else rb)).mean())
                for inv in (False, True) for r in range(63))
        null.append(s)
    best["chance_mean"] = float(np.mean(null))
    best["chance_p95"] = float(np.percentile(null, 95))
    best["above_chance"] = bool(best["score"] > best["chance_p95"])
    return best


# ---------------------------------------------------------------------------
# Pattern fingerprint - identify the loaded word WITHOUT the switch mapping
# ---------------------------------------------------------------------------
#
# Handbook Section 10 says the DIP-switch -> symbol mapping is unresolved, so we
# cannot say "bit pattern X must give symbol Y". But whatever that mapping is,
# it is a *bijection* from the 2^b bit groups onto the 2^b constellation points.
# A bijection permutes the labels; it cannot change how many symbols share a
# point. So the sorted multiset of per-point counts is a fingerprint of the
# loaded word that survives the unknown mapping, any constellation rotation,
# and any I/Q reflection.
#
# Caveat: it assumes the mapping is memoryless. Differential encoding would
# break it, and would show up as a mismatch on a word we know is correct.

def n_classes(modulation: str) -> int:
    return 2 ** BITS_PER_SYMBOL[modulation]


def quantise_symbols(sym: np.ndarray, modulation: str) -> np.ndarray:
    """Map derotated complex symbols to constellation-point indices.

    This IS the label, per handbook Section 10.1 step 7: "the label sequence is
    the quantised symbol index". It is read off the measured waveform, never
    from the DIP switches.

    BPSK  -> sign of the real part                      (2 classes)
    QPSK  -> phase quadrant                             (4 classes)
    8-QAM -> phase quadrant + 4 x (outer ring)          (8 classes)
    """
    if modulation == "BPSK":
        return (sym.real < 0).astype(np.int8)
    quad = np.floor((np.angle(sym) % (2 * np.pi)) / (np.pi / 2)).astype(np.int8) % 4
    if modulation == "QPSK":
        return quad
    outer = (np.abs(sym) > np.median(np.abs(sym))).astype(np.int8)
    return quad + 4 * outer


def observed_point_counts(sym: np.ndarray, modulation: str) -> list[int]:
    """Count how many of the extracted symbols land on each constellation point.
    `sym` must already be derotated."""
    idx = quantise_symbols(sym, modulation)
    return sorted(np.bincount(idx, minlength=n_classes(modulation)).tolist())


def expected_point_counts(pattern_id: str, modulation: str) -> list[int]:
    """The same fingerprint, predicted from the word's bits."""
    bits = bits_of(pattern_id)
    bps = BITS_PER_SYMBOL[modulation]
    groups = bits.reshape(-1, bps)
    idx = np.array([int("".join(map(str, g)), 2) for g in groups])
    return sorted(np.bincount(idx, minlength=2 ** bps).tolist())


def identify_pattern(sym: np.ndarray, modulation: str) -> dict:
    """Which of the handbook's words is consistent with these symbols?"""
    obs = observed_point_counts(sym, modulation)
    n_sym = len(sym)
    candidates = {}
    for pid in ["PRBS63"] + list(PATTERNS):
        try:
            if symbols_per_word(pid, modulation) != n_sym:
                continue
            candidates[pid] = expected_point_counts(pid, modulation)
        except (ValueError, KeyError):
            continue
    matches = [pid for pid, exp in candidates.items() if exp == obs]
    return {"observed": obs, "candidates": candidates, "consistent_with": matches}


# ---------------------------------------------------------------------------
# TP-9 self-consistency (handbook Section 9, check 10)
# ---------------------------------------------------------------------------

def tp9_self_consistency(ch2: np.ndarray, period_samples: float) -> dict:
    """How reliably does the board's own recovered data repeat, period to period?

    Reported next to a shuffled control, because majority-vote agreement on
    pure noise is well above 0.5 and would otherwise look like a result.
    """
    x = np.asarray(ch2, dtype=np.float64)
    if not np.isfinite(x).any():
        return {"consistency": float("nan"), "chance": float("nan"), "n_periods": 0}

    thr = 0.5 * (np.nanmin(x) + np.nanmax(x))
    bits = (x > thr).astype(np.float64)

    seg = fold(bits, period_samples) > 0.5
    maj = seg.mean(axis=0) > 0.5
    consistency = float((seg == maj[None, :]).mean())

    rng = np.random.default_rng(0)
    shuf = rng.permutation(bits)
    seg_s = fold(shuf, period_samples) > 0.5
    maj_s = seg_s.mean(axis=0) > 0.5
    chance = float((seg_s == maj_s[None, :]).mean())

    return {"consistency": consistency, "chance": chance,
            "n_periods": int(seg.shape[0]),
            "locked": bool(consistency >= 0.6 and consistency > chance + 0.05)}


# ---------------------------------------------------------------------------
# Pattern identity (handbook Section 9, check 9 - the one that caught a real bug)
# ---------------------------------------------------------------------------

def template_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Best normalised cross-correlation over all circular shifts.

    Shifts are allowed because the scope trigger phase is arbitrary; a stale
    DIP word still shows up, because the *shape* would be identical to the
    previous pattern rather than to the intended one.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        grid = np.linspace(0, 1, len(a), endpoint=False)
        b = np.interp(grid, np.linspace(0, 1, len(b), endpoint=False), b, period=1.0)
    a = (a - a.mean()) / (np.linalg.norm(a - a.mean()) + 1e-30)
    b = (b - b.mean()) / (np.linalg.norm(b - b.mean()) + 1e-30)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    return float(np.max(np.abs(cc)))


# ---------------------------------------------------------------------------
# General sequence match - the DIP-switch mapping question
# ---------------------------------------------------------------------------
#
# Handbook Section 10 declares the switch -> symbol mapping unresolved: a joint
# search over both known words scored 14/24 against a chance level of 14.1/24.
# But that search was run on QPSK/8-QAM, where bits are first grouped into
# dibits/tribits and the group -> point assignment is a second unknown.
#
# BPSK has no grouping: one bit, one phase. And PRBS63 already decodes at 100%,
# which proves the on-board sequence maps straight to phase. So a 24-bit DIP
# word sent as plain BPSK is the clean test of whether the SWITCHES map the same
# way. If it does, that resolves the BPSK half of the mapping.

def match_word(bits: np.ndarray, pattern_id: str, n_null: int = 2000,
               seed: int = 0) -> dict:
    """Align a recovered bit sequence to a known word over all circular
    rotations and both polarities, with an empirical chance control."""
    ref = bits_of(pattern_id)
    b = np.asarray(bits, dtype=np.int8).ravel()
    if len(b) != len(ref):
        raise ValueError(f"{pattern_id} is {len(ref)} bits, got {len(b)}")
    n = len(ref)

    def best_score(cand: np.ndarray) -> tuple[float, int, bool]:
        best = (-1.0, 0, False)
        for inv in (False, True):
            c = 1 - cand if inv else cand
            for r in range(n):
                s = float((np.roll(ref, r) == c).mean())
                if s > best[0]:
                    best = (s, r, inv)
        return best

    score, rot, inv = best_score(b)

    rng = np.random.default_rng(seed)
    null = [best_score(rng.integers(0, 2, n).astype(np.int8))[0]
            for _ in range(n_null)]
    return {"pattern": pattern_id, "score": score, "rotation": rot,
            "inverted": inv, "chance_mean": float(np.mean(null)),
            "chance_p95": float(np.percentile(null, 95)),
            "above_chance": bool(score > np.percentile(null, 95))}


def template_similarity_signed(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(best positive correlation, best absolute) over all circular shifts.

    The two differ when one waveform is the NEGATION of the other. On this board
    complementing every bit of a word maps each constellation point to its
    antipode, so a word and its bit-complement transmit waveforms that are
    exact negatives - measured at -0.998 for T1 vs T2. Such a pair is NOT two
    independent patterns: after per-window normalisation and the learnable
    global rotation of Section 11.3 they are the same data, and holding one out
    while training on the other would leak.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        grid = np.linspace(0, 1, len(a), endpoint=False)
        b = np.interp(grid, np.linspace(0, 1, len(b), endpoint=False), b, period=1.0)
    a = a - a.mean(); a = a / (np.linalg.norm(a) + 1e-30)
    b = b - b.mean(); b = b / (np.linalg.norm(b) + 1e-30)
    cc = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n=len(a))
    return float(cc.max()), float(np.abs(cc).max())
