"""
context_bound.py - what the input-window cap does and does not rule out.

The paper's bound (Section V-A): a window holds fewer NEIGHBOURING symbols than
are needed to determine its label, for every transmitted word. This script
checks that by enumeration on the MEASURED label sequences, then measures the
two things the bound leaves open. No training; about a minute.

  1. PRBS63 five-bit patterns       why "each 5-bit window occurs twice" was wrong
  2. configured bit strings         minimum preceding BITS that fix the next bit
  3. measured label words           kappa- / kappa+- in SYMBOLS, and the best lookup
                                    from the window's own neighbours -> Table VI
  4. context-only lookup per split  fitted on training records, scored on test
  5. anti-alias FIR support         what filtfilt can smear across a boundary
  6. measured ISI                   neighbour trace in the clean, word-averaged waveform
  7. position-only null             cyclic shifts of the lookup; rank of the true one

Window geometry is data.CONTEXT: BPSK 2 before + 2 after, QPSK 1 before, 8-QAM
none. The target's own waveform is the signal, not context.

Output: results/context_complexity.csv (the Table VI rows).
"""
import itertools
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.signal import firwin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcm31 as M
import splits as SP
import data as DT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(ROOT, "dataset", "windows.npz"))
wave, label, wo, lo = d["wave"], d["label"], d["wave_offset"], d["label_offset"]
SPS = int(d["sps_out"])
meta = SP.load_meta(ROOT)
MODS = ("BPSK", "QPSK", "8QAM")
CTX = {m: DT.CONTEXT[M.MAX_RECEPTIVE_FIELD_SYMBOLS[m]] for m in MODS}   # (before, after)


def min_context(seq, two_sided=False):
    """Smallest k such that the k-symbol context determines the next symbol
    everywhere in the cyclic sequence. Two-sided: floor(k/2) before, rest after."""
    n = len(seq)
    for k in range(n):
        a, b = (k // 2, k - k // 2) if two_sided else (k, 0)
        seen = {}
        for i in range(n):
            key = (tuple(seq[(i - j) % n] for j in range(1, a + 1)) +
                   tuple(seq[(i + j) % n] for j in range(1, b + 1)))
            if seen.setdefault(key, seq[i]) != seq[i]:
                break
        else:
            return k
    return n


# ---- 1 ----------------------------------------------------------------------
print("1. PRBS63 five-bit patterns (stored string, and its complement)")
for name, s in (("stored", M.bits_of("PRBS63")), ("complement", 1 - M.bits_of("PRBS63"))):
    n = len(s)
    rec = all(s[(i + 6) % n] == (s[i] ^ s[(i + 5) % n]) for i in range(n))
    occ = Counter(tuple(int(s[(i + j) % n]) for j in range(5)) for i in range(n))
    once = [w for w, c in occ.items() if c == 1]
    print(f"   {name:<10} satisfies s[i+6]=s[i]^s[i+5]: {rec!s:<5}  "
          f"{sum(c == 2 for c in occ.values())} patterns twice, {len(once)} once: {once}")

# ---- 2 ----------------------------------------------------------------------
print("\n2. minimum preceding bits that determine the next bit (configured strings)")
print("   " + "  ".join(f"{p} {min_context(list(M.bits_of(p)))}"
                      for p in ["PRBS63", "S1", "Q1", "Q2", "Q3", "T1", "T2", "T3", "T4", "T5", "T6"]))

# ---- 3 ----------------------------------------------------------------------
print("\n3. measured label words (Table VI)")
word = {c: label[lo[c]:lo[c] + int(meta.n_sym_word[c])].astype(int) for c in meta.index}
rows = []
for mod in MODS:
    bef, aft = CTX[mod]
    for g in sorted(meta[meta.modulation == mod].group.unique()):
        c = meta.index[(meta.modulation == mod) & (meta.group == g)][0]
        w = list(word[c]); n = len(w)
        key = lambda i: (tuple(w[(i - j) % n] for j in range(1, bef + 1)) +
                         tuple(w[(i + j) % n] for j in range(1, aft + 1)))
        tab = defaultdict(Counter)
        for i in range(n):
            tab[key(i)][w[i]] += 1
        lookup = sum(tab[key(i)].most_common(1)[0][0] == w[i] for i in range(n)) / n
        rows.append(dict(modulation=mod, word=g, n_symbols=n, kappa_minus=min_context(w),
                         kappa_pm=min_context(w, True), window=f"{bef}+{aft}",
                         lookup=round(lookup, 4), chance=1 / M.n_classes(mod)))
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
tbl.to_csv(os.path.join(ROOT, "results", "context_complexity.csv"), index=False)
print("   (T3-T6 give identical rows and are one line in the paper)")


# ---- 4 ----------------------------------------------------------------------
def context_lookup_ser(tr, te, bef, aft):
    """Best lookup from the true labels of a window's neighbours to its target,
    fitted on training records and scored on test records."""
    tab = defaultdict(Counter)

    def items(c):
        y = label[lo[c]:lo[c + 1]].astype(int)
        for i in range(bef, len(y) - aft):
            yield tuple(y[i - bef:i]) + tuple(y[i + 1:i + 1 + aft]), y[i]
    for c in tr:
        for k, t in items(c):
            tab[k][t] += 1
    prior = Counter()
    for v in tab.values():
        prior.update(v)
    ok = tot = 0
    for c in te:
        for k, t in items(c):
            ok += (tab[k] if k in tab else prior).most_common(1)[0][0] == t
            tot += 1
    return 1 - ok / tot


print("\n4. context-only lookup SER, mean over folds (chance BPSK 0.500, QPSK 0.750, 8-QAM 0.875)")
for mod in MODS:
    for sp, folds in SP.make_splits(meta, mod).items():
        v = [context_lookup_ser(tr, te, *CTX[mod]) for _, tr, te in folds]
        print(f"   {mod:<5} {sp:<26} {np.mean(v):.4f}")
print("   8-QAM IF the previous symbol were readable through ISI (context 1+0):")
for sp, folds in SP.make_splits(meta, "8QAM").items():
    v = [context_lookup_ser(tr, te, 1, 0) for _, tr, te in folds]
    print(f"   8QAM  {sp:<26} {np.mean(v):.4f}  folds {np.round(v, 3).tolist()}")

# ---- 5 ----------------------------------------------------------------------
print("\n5. anti-alias FIR (build_dataset.py: firwin(129, 4 kHz), filtfilt)")
h = firwin(129, 4000.0, fs=M.FS_NOMINAL)
g = np.convolve(h, h[::-1])
sps_in = float(meta.sps_in.mean())
t = np.arange(len(g)) - (len(g) - 1) / 2
out = (g[np.abs(t) > 0.5 * sps_in] ** 2).sum() / (g ** 2).sum()
print(f"   effective support {len(g)} raw samples = +-{(len(g) - 1) / 2 / sps_in:.2f} symbol; "
      f"energy beyond +-0.5 symbol {10 * np.log10(out):.1f} dB")

# ---- 6 ----------------------------------------------------------------------
print("\n6. neighbour-induced deviation of a symbol's 20 samples from its class mean")
print("   (clean waveform averaged over word repetitions; only classes that repeat in")
print("    a word can show it, so 8-QAM T1/T2 - each point once - drop out)")
for mod in MODS:
    rel, half = [], []
    for c in meta.index[meta.modulation == mod]:
        n = int(meta.n_sym_word[c])
        x = wave[wo[c]:wo[c + 1]].astype(np.float64).reshape(-1, SPS)
        tmpl = x[:len(x) // n * n].reshape(-1, n, SPS).mean(0)
        y = word[c]
        cm = {k: tmpl[y == k].mean(0) for k in np.unique(y)}
        rep = [i for i in range(n) if (y == y[i]).sum() > 1]
        if not rep:
            continue
        dev = np.array([tmpl[i] - cm[y[i]] for i in rep])
        dmin = min(np.linalg.norm(cm[a] - cm[b]) for a, b in itertools.combinations(cm, 2))
        rms = np.sqrt((dev ** 2).sum(1).mean())
        rel.append(20 * np.log10(rms / np.sqrt((tmpl[rep] ** 2).sum(1).mean())))
        half.append(rms / (dmin / 2))
    print(f"   {mod:<5} {min(rel):.1f} to {max(rel):.1f} dB of the symbol's RMS; "
          f"{min(half):.2f}-{max(half):.2f} of half the minimum class distance")

# ---- 7 ----------------------------------------------------------------------
print("\n7. position-only control vs its cyclic-shift null (leave-one-pattern-out)")
for mod in MODS:
    cap = M.MAX_RECEPTIVE_FIELD_SYMBOLS[mod]
    K = M.n_classes(mod)
    null_means = []
    for name, tr, te in SP.make_splits(meta, mod)["leave_one_pattern_out"]:
        ws = DT.WindowSet(wave, label, wo, lo, tr, SPS, cap)
        err = defaultdict(lambda: [0, 0])
        for c in te:
            nw = int(meta.n_sym_word[c])
            tab = np.zeros((nw, K), int)
            np.add.at(tab, (ws.symbol_index % nw, ws.y), 1)
            wt = DT.WindowSet(wave, label, wo, lo, [c], SPS, cap)
            for r in range(nw):
                p = np.roll(tab.argmax(1), r)[wt.symbol_index % nw]
                err[r][0] += int((p != wt.y).sum())
                err[r][1] += len(wt.y)
        v = np.array([a / b for a, b in err.values()])
        null_means.append(v.mean())
        print(f"   {mod:<5} {name:<14} observed {v[0]:.4f}  null mean {v.mean():.4f}  "
              f"{int((v < v[0] - 1e-9).sum())} of {len(v) - 1} other shifts better")
    print(f"   {mod:<5} null mean over folds {np.mean(null_means):.4f}")
