# 4. Results

> **Draft for the MCM31 neural demodulator paper (AEU submission).**
> Every number below is taken from `results/` and was produced by the scripts in
> `src/`. Nothing is quoted from memory. Passages marked ⚠️ are limitations that
> should stay in the paper rather than be edited out — each of them pre-empts a
> question a referee will otherwise ask.

---

## 4.1 Dataset and ground truth

Ten MCM31/EV communication trainer units were sourced. **Five were functional; the
remaining five were excluded** after individual inspection, in each case because
the on-board data generator had failed — either the sequence load (START) did not
latch or the 64-bit PRBS generator produced no output. On those units the
modulator receives no data and TP-20 carries an unmodulated carrier, so no
usable waveform exists to capture. All results below therefore rest on **five
physically distinct boards**, and leave-one-board-out is a five-fold evaluation.

The final corpus is **43 oscilloscope captures**, each 6.0004 s at 87,260 samples/s
(523,600 samples), acquired simultaneously on TP-20 (the modulated waveform) and
TP-9 (the board's own recovered data). Board B1 contributes 15 captures — seven
main-campaign files plus eight held-out data words — and boards B2–B5 contribute
seven each. Segmented at the symbol rate this yields **154,628 labelled symbols**:

| Modulation | Captures | Symbols | Data words | Independent patterns |
|---|---|---|---|---|
| BPSK | 16 | 57,536 | 2 | 2 |
| QPSK | 12 | 43,152 | 3 | 3 |
| 8-QAM | 15 | 53,940 | 6 | **5** |

The 8-QAM row distinguishes *words* from *independent patterns*. Words T1 and T2
are bitwise complements of one another, and on this hardware complementary
tribits map to antipodal constellation points: their coherent templates
correlate at **−0.998**. After per-capture amplitude normalisation the two carry
the same information, so they are treated as a **single** pattern throughout. Had
they been counted separately, the leave-one-pattern-out result would have been
optimistically biased.

### Ground truth

Labels are obtained by coherent averaging of the transmitted waveform, never from
the DIP-switch setting. For each capture the word period is refined by maximising
the energy of the coherent average, the raw passband waveform is averaged over
its 57–450 repetitions, and symbols are extracted after blind timing recovery.
Template SNR across the corpus is **22.7–43.2 dB (median 30.2 dB)**.

Two independent checks confirm the labels:

1. **All 16 BPSK captures reproduce the board's 63-bit maximal-length sequence at
   100%** (chance level 66.6%, 95th percentile 69.8%, over 63 rotations and both
   polarities). The carrier is recovered blind by fourth-power estimation at
   1199.915–1199.918 Hz across all five boards.
2. A single memoryless bijection from constellation point to dibit fits **all
   three QPSK words simultaneously at 100%** — 36 symbols on one 4-point mapping.
   Across 20,000 count-preserving label shuffles a perfect fit never occurred
   (**p < 5 × 10⁻⁵**).

⚠️ **Symbol timing recovery is a prerequisite, not an optional refinement.** Extracting
symbols without it placed decision instants part-way through transitions and
collapsed two of the eight 8-QAM constellation radii (0.20 against a nominal
1.31). This failure is visually identical to the template wrap-boundary artefact
and must be excluded before that diagnosis is reached.

---

## 4.2 The receptive-field bound

The board's sequence is an order-6 maximal-length sequence: any six consecutive
bits determine the seventh exactly through `s[i+6] = s[i] ⊕ s[i+5]`, while any
five do not. A receiver that observes at most five bits therefore **cannot**
predict a symbol from context and must read the waveform.

The bound is enforced by the **input**, not by kernel arithmetic. The network is
presented with a window of exactly *N* symbols and asked to label the symbol
within it; it cannot attend to a sixth bit because no sixth bit is present in
its input. This is verifiable by inspection and survives any later change to the
layer stack.

| Modulation | Symbols visible | Bits visible | Window | Receptive field | Parameters |
|---|---|---|---|---|---|
| BPSK | 5 | 5 | 100 samples | 93 | 20,932 |
| QPSK | 2 | 4 | 40 samples | 39 | 22,022 |
| 8-QAM | 1 | 3 | 20 samples | 20 | 20,106 |

All three sit at or below the five-bit limit. The identical architecture is used
throughout: four convolutional layers, 32 channels, a full-width learned
combiner, and two heads (symbol logits and a latent I/Q pair). Waveforms are
decimated from 145.4 to **20 samples per symbol** and presented in **passband**;
converting to baseband would amount to supplying the carrier frequency.

---

## 4.3 Symbol error rate against a blind classical receiver

The reference receiver performs fourth-power carrier recovery, matched filtering
and hard decision, and is given no parameters the network is not given. Noise is
software AWGN referenced to each capture's RMS, identical for both receivers.
Figure 1 reports leave-one-pattern-out performance.

| SNR | BPSK learned | BPSK classical | QPSK learned | QPSK classical | 8-QAM learned | 8-QAM classical |
|---|---|---|---|---|---|---|
| 0 dB | 0.0001 | 0.0000 | 0.0093 | 0.0037 | 0.1319 | 0.1219 |
| 3 dB | 0.0000 | 0.0000 | 0.0011 | 0.0003 | 0.0432 | 0.0381 |
| 6 dB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0065 | 0.0058 |
| 9 dB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0003 |

**The two receivers are equivalent.** The classical receiver retains a small
advantage at 0 dB — 0.1219 against 0.1319 for 8-QAM, a relative difference of
8% — which is expected, since a matched filter is near-optimal for additive
Gaussian noise. Above 9 dB the learned receiver reaches exactly zero errors while
the classical receiver retains a residual floor of 2 × 10⁻⁴.

⚠️ **We make no performance claim.** The contribution is that this parity is
achieved without supplying the carrier frequency, the pulse shape, the
constellation geometry, or the decision boundaries. All quantities saturate above
9 dB, so only the 0–9 dB region is informative and only that region is plotted.

---

## 4.4 Generalisation

Splits are constructed at **capture level**. Symbols within a capture are
repetitions of one word through one signal path; a symbol-level split would place
near-duplicates on both sides of the partition.

8-QAM symbol error rate, mean over folds:

| Split | Folds | 0 dB | 3 dB | 6 dB | 9 dB |
|---|---|---|---|---|---|
| Random 80/20 | 1 | 0.1269 | 0.0387 | 0.0054 | 0.0005 |
| Leave-one-pattern-out | 5 | 0.1319 | 0.0432 | 0.0065 | 0.0003 |
| Leave-one-board-out | 5 | 0.1327 | 0.0451 | 0.0105 | 0.0021 |
| Leave-one-attenuation-out | 2 | 0.1367 | 0.0452 | 0.0071 | 0.0004 |

**Holding out an entire data word costs essentially nothing** (Figure 2). At 0 dB
the leave-one-pattern-out and random splits differ by 0.005 in SER — within
run-to-run variation. This is the empirical counterpart of Section 4.2: a model
exploiting sequence structure would score markedly better on a random split,
where training and test share the same word. It does not.

**Cross-device transfer is measurably harder.** Leave-one-board-out degrades to
0.0021 at 9 dB against 0.0003 for leave-one-pattern-out — roughly seven-fold.
The effect is small in absolute terms but consistent, and is the only split where
a systematic gap appears.

⚠️ Two folds are thin and are reported with their training size: BPSK with PRBS63
held out trains on **one** capture (3,592 symbols), and QPSK with Q1 held out trains
on **two** (7,190 symbols). Both nonetheless reach 0.0001 and 0.0174 respectively at
0 dB. The 8-QAM folds, where the receptive-field bound is tightest and the result
matters most, are the best conditioned: four of five train on 14 captures.

---

## 4.5 Negative controls

Two controls must fail. Both do (Figure 3).

| Modulation | Chance | Shuffled labels | Position-only | Model at 6 dB |
|---|---|---|---|---|
| BPSK | 0.500 | 0.4979 | 0.5151 | 0.0000 |
| QPSK | 0.750 | 0.7349 | 0.7862 | 0.0000 |
| 8-QAM | 0.875 | 0.8834 | 0.8523 | 0.0065 |

**Shuffled labels.** Permuting labels among the symbols of each fold leaves
performance at chance, confirming no path from input to target other than the
waveform.

**Position-only.** The strongest possible predictor that observes no signal — a
lookup table from the symbol's index within the word to its most frequent label —
also remains at chance, on every split including the random one. Because splitting
is at capture level and different captures carry different words, symbol position
alone carries no transferable information.

---

## 4.6 Learned constellation geometry

The loss includes a constellation term, `λ · min_c ‖(Î,Q̂) − c‖²`, evaluated against
the measured geometry (a four-phase two-ring star of measured radius ratio **1.89**,
not a 16-QAM square grid). Its contribution was ablated:

| λ | 8-QAM 0 dB | 3 dB | 6 dB | QPSK 0 dB |
|---|---|---|---|---|
| 0.0 | 0.1246 | 0.0370 | 0.0060 | 0.0058 |
| 0.2 | 0.1261 | 0.0359 | 0.0056 | 0.0051 |
| 1.0 | 0.1276 | 0.0364 | 0.0058 | 0.0054 |

⚠️ **The constellation prior does not improve symbol error rate.** Differences across
two decades of λ lie within run-to-run variation. Its role is interpretability: it
aligns the latent I/Q head with the physical constellation. We do not claim it as
a performance mechanism.

The geometry is nevertheless recovered — from the data, by the classifier alone.
Examining *where* errors go at 0 dB (Figure 5), **79.5% of 8-QAM errors land on the
geometrically nearest constellation point, against a chance level of 14.3% — an
enrichment of 5.6×**. For QPSK the figure is 100% against 66.7%. Of the 56
off-diagonal cells in the confusion matrix, **36 are exactly zero**; the populated
cells are the same-phase ring transition and the adjacent-quadrant transition,
which are the two shortest distances in the constellation.

A model implementing a memorised waveform-to-label mapping would distribute its
errors uniformly over incorrect classes. This one confuses each point with its
metric neighbours — precisely where additive noise transports a symbol.

---

## 4.7 Hardware characterisation

The captures also characterise the boards' own demodulators, measured against the
known transmitted sequence rather than by self-consistency (Figure 4). Across all
five units, TP-9 agreement with PRBS63 is at chance for every capture below
**509 mV** TP-20 RMS and reaches 92–100% for every capture above **626 mV**. A
single threshold in that interval accounts for all five boards, and brackets the
552–570 mV cliff previously reported for a single unit.

Board-level behaviour differs in kind, not merely in degree. Board B2's BPSK path
delivers 391 mV where the other four deliver 626–663 mV, and never reaches
threshold; its QPSK and 8-QAM outputs are indistinguishable from the other boards
(cross-board waveform correlation +0.9987 and +0.9991, against +0.9769 for BPSK).
Board B4 crosses the threshold but locks imperfectly, at 92.1%.

⚠️ **Self-consistency is not a valid lock criterion**, and we recommend against it. A
demodulator that has lost lock also repeats. On board B1 at 329 mV, TP-9 self-consistency
read 0.979 while agreement with the true sequence was at chance: the unlocked
demodulator had entered a stable period-2 state, inverting its output every word
period (autocorrelation −0.93 at one word period, +0.90 at two). Four separate
captures in this study would have been misclassified as locked by that metric.

⚠️ The board's own QPSK demodulator failed to lock on word Q1 on **all five units**,
at both attenuation settings, although the pattern fingerprint confirms Q1 was
correctly loaded. No classical baseline from the hardware is therefore available
for QPSK.

---

## 4.8 Limitations

**Content diversity.** The corpus is large in noise realisations and hardware
variation but small in distinct transmitted content: 8-QAM comprises five
independent 24-bit words, or 40 distinct symbol instances. Leave-one-pattern-out
for 8-QAM is a five-fold test on that basis.

**The SNR axis is synthetic.** The hardware supplies the waveform — carrier,
filter response, quantisation, board-to-board variation — and additive white
Gaussian noise is applied in software. The board's own noise control adds only
+0.02 dB within the signal band and is therefore unusable as an SNR axis.

**Symbol timing is supplied.** The network is given neither the carrier frequency
nor the pulse shape, but its input window is symbol-synchronous, which supplies
the symbol rate and the sampling phase. This is not incidental: under a
one-symbol receptive field for 8-QAM, blind timing recovery is impossible, so the
receptive-field bound and timing-blindness cannot both hold. The system should be
described as operating on symbol-synchronous windows from a conventional timing
front-end.

**Carrier robustness was tested and does not support a claim.** We measured whether
a classical receiver's dependence on the carrier constitutes a practical
disadvantage. Estimating the carrier blind from only eight symbols (13.3 ms) gives
an error of 1.02 Hz, but demodulating those same eight symbols with that estimate
yields SER 0.0000 — over 13.3 ms, 1 Hz of error is 4.9° of phase rotation. Estimate
error and tolerance contract together, so the classical receiver is self-consistent
at every observation length tested (8 to 3,596 symbols). Reporting a collapse under
an externally imposed carrier offset would misrepresent a receiver that does not
make that error.

**Operating regime.** A single carrier frequency, a single symbol rate, and an
additive Gaussian channel with no multipath or fading. Generalisation beyond this
regime is not evidenced here.

---

## Figures referenced

| Figure | Content |
|---|---|
| 1 | Symbol error rate against SNR, learned and classical receivers, three modulations |
| 2 | 8-QAM error rate across evaluation splits |
| 3 | Negative controls against chance |
| 4 | Board demodulator lock threshold across five units |
| 5 | Error distribution over the constellation |
