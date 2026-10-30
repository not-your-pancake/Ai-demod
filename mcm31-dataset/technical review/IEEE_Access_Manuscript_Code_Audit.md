# Technical audit of the IEEE Access manuscript and accompanying code

**Manuscript:** *Neural Demodulation of Hardware-Captured PSK and Star-QAM Waveforms Under a Provable Memorisation Bound*  
**Authors shown in manuscript:** Mohammad Khalid Hasan Nabil, Ivan Grgić, and Md. Maynul Islam  
**Audit date:** 30 October 2026  
**Materials examined:** 14-page manuscript PDF, `code+results.zip`, and separate `captures.csv` supplied by the author.

> **Purpose and status.** This is a detailed, independent technical review to support author revision and a second opinion from Claude or another reviewer. It is not an official IEEE decision, and I am not an IEEE Access editor or reviewer. The verdict below is a readiness assessment based on the material actually supplied. I distinguish source-code findings, calculations independently made from supplied pattern strings/results, and issues that still require raw oscilloscope waveforms or a complete rerun.

---

## 0. How to read the references in this audit

### Manuscript page and line references

The PDF does **not** contain author-inserted continuous line numbers. To make the review traceable, page references below use **PDF page number plus line number within that page as extracted by `pdftotext -layout`**. Line numbering restarts on each PDF page. It is a locator generated for this audit, not an official manuscript line-numbering scheme. For example, `PDF p. 7, Section V-A, extracted lines 46–66` means page 7 of the submitted PDF and lines 46–66 of that page's layout-preserving text extraction.

Where a reference is to a table, equation or figure, that identifier is also given. If you revise the PDF, page-local line numbers will move; the section/table/equation identifier should remain the primary locator.

### Code references

Code locations refer to the file paths inside `code+results.zip`, with 1-based line numbers as shown by `nl -ba`. For example, `src/model.py:111–128` means lines 111–128 of `src/model.py` in the ZIP archive.

### Evidence labels

- **Confirmed in supplied source:** directly visible in the supplied code or PDF.
- **Recalculated from supplied CSV/results:** independently calculated from the 43-row capture manifest or saved result CSVs; not a fresh model-training run.
- **Reviewer-side diagnostic:** a new calculation on pattern strings or saved results, not a claim reported by the manuscript.
- **Needs raw data / rerun:** cannot be conclusively resolved from the supplied archive.

---

# 1. Executive assessment

## 1.1 Provisional reviewer recommendation

**Do not submit the current version to IEEE Access yet.** My recommendation is **substantial technical revision before submission**. If this exact PDF were submitted now, I would expect a significant risk of rejection because the principal contribution is a *provable* restriction on sequence memorisation, but the manuscript and implementation do not yet establish the guarantee for the actual end-to-end signal path and the symbol contexts used in the experiments.

This is not a recommendation to abandon the project. The hardware dataset, measured-label pipeline, small waveform CNN, device-held-out experiments and the attempt to test sequence memorisation are all worth preserving. The problem is the distance between the strongest claim and the evidence currently documented.

## 1.2 Overall strengths

1. **A real laboratory acquisition campaign.** The manuscript reports 43 dual-channel captures from five functional MCM31/EV trainer boards, spanning BPSK, QPSK and star 8-QAM.
2. **Symbol-level labels are derived from the measured waveform.** The label-generation pipeline does not simply copy the transmitter's DIP-switch configuration.
3. **The data are split by capture rather than by individual symbol.** This avoids the most obvious within-record train/test contamination.
4. **The authors disclose meaningful limitations.** In particular, the manuscript acknowledges the small number of independent words, the five-board sample, software-added AWGN and symbol-synchronous input.
5. **The implementation contains a real input-window cap.** The CNN is not simply described as having a short receptive field; the code constructs bounded-length input windows and checks the model dimensions.

## 1.3 Main reasons the current paper is not ready

| Priority | Finding | Why it matters |
|---|---|---|
| **Critical** | The order-six PRBS recurrence is used as a global justification for BPSK, QPSK and 8-QAM caps. | The theorem must be tied to the actual pattern and actual symbol/context presented to the classifier. |
| **Critical** | The zero-phase FIR preprocessing can mix information across nominal window boundaries; the code itself notes that ISI is how context reaches the one-symbol 8-QAM input. | The full front-end support must be included in any structural “cannot use context” guarantee. |
| **Major** | The shuffled-label control trains and evaluates against labels that have both been shuffled. | The reported chance result is not the clean train-shuffled/test-ground-truth control described in the paper. |
| **Major** | The neural and classical results are aggregated differently, and use different noise realizations and observation budgets. | Claims of parity/equivalence are not yet demonstrated under a fully matched comparison protocol. |
| **Major** | Most additional held-out words appear only on board B1. | Pattern generalisation is confounded with device identity for several LOPO folds. |
| **Major** | Only one training seed and one random 80/20 split are used in the principal run. | Reported performance and split stability are not characterised adequately. |
| **Major** | The 8-QAM configured-word-to-symbol sequence mapping remains unresolved. | A bit-pattern bound must be shown to apply to the labels/waveforms actually used in the model. |
| **Moderate** | The supplied archive omits raw waveforms and `dataset/windows.npz`. | I could not independently rebuild the data or rerun the full pipeline. |
| **Moderate** | Several claims need narrower wording, especially “blind,” “matched-filter,” “indistinguishable,” and “mathematically impossible.” | The paper should not imply resources or guarantees that the experiment does not establish. |

---

# 2. Materials received and audit limits

## 2.1 Files examined

| Supplied item | What it contains | Audit result |
|---|---|---|
| `AI_Demod_Paper (1).pdf` | 14-page manuscript | Reviewed for claims, methods, tables, conclusions, references, acknowledgements and biographies. |
| `code+results.zip` | Python source under `src/`, saved outputs under `results/`, and extracted manuscript text | Source inspected; saved CSVs reconciled against key manuscript tables. |
| `captures.csv` | 43-row capture manifest | Recalculated record, symbol, board, modulation, pattern and attenuation counts. |

## 2.2 What was not included in the supplied ZIP

The ZIP contains source code and result files, but I did not find:

- the original oscilloscope waveform CSVs referred to by the manifest's `filename` column;
- `dataset/windows.npz`, which the training and evaluation scripts expect;
- saved trained model checkpoints;
- a `requirements.txt`, `environment.yml`, `pyproject.toml` or equivalent pinned environment file;
- a top-level `README` with a complete reproduction sequence.

The separately supplied `captures.csv` is a manifest, not a replacement for the raw waveforms. As a result, this audit **does not constitute an end-to-end rerun of dataset creation, model training or raw-signal label validation**. I inspected the relevant source and reconciled saved outputs where possible.

## 2.3 What was independently checked

- `captures.csv` contains 43 rows and sums to **154,628** symbols in `n_symbols`.
- The modulation totals agree with manuscript Table V: BPSK 57,536; QPSK 43,152; 8-QAM 53,940.
- `experiments.csv` contains 38 rows: 35 main evaluation rows plus 3 shuffled-label-control rows.
- The 35 main folds are consistent with the split code: 10 LOPO + 15 LOBO + 7 leave-one-attenuation-out + 3 random = 35.
- The saved LOPO neural results reproduce the manuscript's Table VII numbers to the displayed rounding when averaged equally across pattern folds.
- The saved `classical_snr.csv` reproduces the classical values in Table VII to the displayed rounding, but its averaging procedure is not the same as the LOPO macro-average used for the neural results.
- The saved `experiments.csv` shuffled-label values correspond to the three control rows generated from the first leave-one-board-out fold for each modulation, rather than leave-one-pattern-out folds as Table X and Fig. 6 imply.

---

# 3. Critical finding 1: the central mathematical guarantee needs to be reformulated and proved for the actual task

**Severity: Critical — resolve before any submission.**

## 3.1 Where the manuscript makes the claim

- **PDF p. 1, Abstract, extracted lines 17–26:** says the transmitted sequence has linear order six, any six consecutive bits determine the seventh, and restricting the input to at most five bits makes prediction-from-context mathematically impossible.
- **PDF p. 1, Abstract, extracted lines 43–47:** lists a receptive-field bound that “makes sequence memorisation impossible” as a contribution.
- **PDF p. 7, Section V-A, extracted lines 46–66, Equation (8):** introduces `s[i+6] = s[i] XOR s[i+5]`, states that six consecutive bits determine the seventh and no five do, and concludes that the capped receiver must extract the answer from the waveform.
- **PDF p. 12, Section IX, extracted lines 14–30:** repeats the conclusion that sequence memorisation is excluded by construction.

## 3.2 What the implementation actually enforces

The input cap is genuine:

- `src/mcm31.py:56–59` defines bits per symbol and caps of 5 BPSK symbols, 2 QPSK symbols and 1 8-QAM symbol.
- `src/model.py:28` sets `MAX_BITS_VISIBLE = 5`.
- `src/model.py:111–128` asserts that `cap_symbols * bits_per_symbol` does not exceed five bits and checks the CNN receptive field against the input window.
- `src/model.py:137–150` prints the PRBS63 order-six argument as the justification for all three caps.
- `src/validate_dataset.py:82–86` checks only that the total configured input-bit budget is no greater than five.

**What is not implemented:** a routine that derives the minimum prediction context independently for each configured pattern and then checks that the actual permitted context is shorter than that value. The source code's global assertion is a five-bit budget check, not a per-pattern complexity proof.

## 3.3 Reviewer-side calculation from the supplied pattern strings

I independently evaluated the cyclic bit strings defined in `src/mcm31.py`. For each pattern, I searched for the smallest number `k` of *immediately preceding bits* such that the same length-`k` context never occurs with two different next bits anywhere in the cyclic word. This is a direct lookup-complexity calculation on the configured bit strings, not a new hardware measurement or a trained-model result.

| Modulation | Pattern | Bit length | Minimum preceding bits that determine the next bit in the supplied cyclic string |
|---|---|---:|---:|
| BPSK | PRBS63 | 63 | 6 |
| BPSK | S1 | 24 | 7 |
| QPSK | Q1 | 24 | 8 |
| QPSK | Q2 | 24 | 6 |
| QPSK | Q3 | 24 | 9 |
| 8-QAM | T1 | 24 | 9 |
| 8-QAM | T2 | 24 | 9 |
| 8-QAM | T3 | 24 | 8 |
| 8-QAM | T4 | 24 | 8 |
| 8-QAM | T5 | 24 | 9 |
| 8-QAM | T6 | 24 | 8 |

This calculation is consistent with the 6–9-bit numbers in the WorldCUR abstract. It is useful evidence that the abstract's pattern-specific numbers may be supportable.

**But it does not, by itself, prove the claim made in the journal paper.** There are three distinct objects that must not be conflated:

1. a configured cyclic **bit string**;
2. the **symbol-label sequence** recovered from the oscilloscope waveform;
3. the **waveform samples** supplied to the CNN after timing, filtering, resampling and normalisation.

A theorem about the first object only establishes a claim about the third object if the mapping, framing, front-end support and network's actual context are rigorously connected.

## 3.4 The actual windows are not simply “five previous bits,” “four previous bits” and “three previous bits”

`src/data.py:25–29` sets the context as:

```python
CONTEXT = {
    1: (0, 0),
    2: (1, 0),
    5: (2, 2),
}
```

`src/data.py:32–67` constructs each target window from this context. The code's own module docstring (`src/data.py:4–10`) describes the windows as:

| Modulation | Actual input | What is explicitly present around the target |
|---|---|---|
| BPSK | `[t-2, t-1, T, t+1, t+2]` | Two preceding symbols and two following symbols, with the target in the centre. |
| QPSK | `[t-1, T]` | One preceding symbol and the target. |
| 8-QAM | `[T]` | The target symbol only. |

This is different from a model that receives a one-sided history of five bits and predicts the seventh bit. In particular:

- BPSK uses a **centred** five-symbol window. The recurrence argument in Section V-A is expressed in terms of preceding bits, but the CNN also receives future-neighbour waveform samples.
- QPSK's four-bit count includes the target symbol's two waveform bits as well as one preceding two-bit symbol. The target waveform is meant to contain the desired signal; it should not be described as context information.
- 8-QAM has one target-symbol window. The explicit crop contains no adjacent symbol, but the prefilter and physical pulse response can make adjacent symbols influence samples inside the crop; see Finding 2.

The paper should define precisely whether the bound concerns (a) all bits in the tensor, (b) only non-target transmitted symbols, or (c) every transmitted bit that can influence the target waveform after the analogue and digital front end. Those are not equivalent definitions.

## 3.5 Context-only reviewer diagnostics

I also calculated simple lookup baselines from the supplied pattern strings. These are **not trained neural-network results**, and they do not validate the hardware label alignment. They are included to show why “not perfectly predictable” and “no useful context information” are different claims.

- With the BPSK central context consisting of two preceding and two following bits, a majority lookup over the cyclic strings gives about **50.8%** for PRBS63 and **62.5%** for S1. BPSK chance is 50%. Thus, on these bit strings, the available two-sided context is not a perfect deterministic lookup for every position, but S1 context is predictive above chance.
- With one preceding two-bit QPSK symbol as a lookup context, the simple pattern-string accuracies are approximately **41.7% for Q1, 58.3% for Q2 and 33.3% for Q3**, against 25% four-class chance.

These diagnostics do not say the CNN is exploiting context. They show that the paper should test that possibility directly instead of treating a past-bit recurrence theorem as proof that all context information is useless.

## 3.6 Important issue: next-bit complexity is not automatically next-symbol complexity

The abstract's 6–9-bit figures are expressed as the number of bits required to determine the next bit. The model outputs a constellation-symbol class, which may represent two or three bits per symbol. The journal paper needs to state the precise prediction problem and prove the implication it uses.

For QPSK, the supplied strings can also be grouped into two-bit symbols. Under straightforward grouping at the beginning of the 24-bit word, the minimum previous-symbol contexts that uniquely determine the next grouped symbol are 3 symbols for Q1, 3 for Q2 and 2 for Q3. That is a different quantity from the bit-level next-bit complexity (8, 6 and 9 bits respectively). The paper should avoid implying that the two quantities are interchangeable without proof.

For 8-QAM, the concern is stronger because the manuscript itself says that the actual transmitter framing offset does not coincide with the configured word order. A symbol-context calculation on an assumed grouping is not definitive until it is tied to measured symbol labels and the actual board framing.

A further reviewer-side diagnostic illustrates why this matters. When the 24-bit words are grouped into cyclic triplets at bit offsets 1 or 2 (rather than offset 0), the pattern occupancy matches the manuscript's description: T1/T2 contain all eight triplets, while T3–T6 contain six distinct triplets. Under either of those candidate alignments, the triplet sequence in T1/T2 has no repeated triplet, so one preceding triplet (3 bits) uniquely identifies the next triplet; for T3–T6, two preceding triplets (6 bits) are sufficient in the supplied cyclic strings. This is a **conditional diagnostic**, not a verified hardware mapping: the actual offset and measured label order still need to be established. It shows why the 6–9-bit *all-bit-position next-bit* table cannot simply be substituted for a *symbol-boundary next-symbol* complexity table. The prediction task being bounded must match the network's classification target.

## 3.7 PRBS63 recurrence polarity should be made explicit

- `src/mcm31.py:35` stores the PRBS63 bit string.
- `src/mcm31.py:87–90` checks the recurrence on the **complement** of the stored string, not on the literal stored string.

The complement satisfies the recurrence in the code. The stored string itself does not satisfy the same zero-offset equation as written. Complementing all bits preserves whether a context uniquely predicts a successor, so this is not necessarily a fatal result. However, define `s[i]` consistently in Equation (8), and explain the polarity convention. A reviewer should not have to infer that the proof uses the complement of the string stored in the code.

## 3.8 Required revision

Add a subsection, for example:

**V-A. Pattern-Specific Context-Complexity Bound**

It should:

1. Define the cyclic transmitted bit string and the actual measured symbol-label sequence for each pattern.
2. Define the target index and exactly which past/future symbols are available in a model input.
3. Define the context-complexity quantity mathematically. For example, for a symbolic label sequence `y_i`, define the smallest context length for which every repeated context has only one possible target label. For centred windows, define a two-sided context key explicitly instead of using a one-sided recurrence without adjustment.
4. Calculate the result for every independent pattern, including all cyclic phases and the actual grouping/framing offset.
5. State separately whether the theorem prevents **perfect deterministic lookup** or prevents **any predictive information**. The former does not imply the latter.
6. Implement the calculation in the repository and add assertions/tests that fail if a configured pattern or window changes.
7. Report a context-only baseline (past-only, future-only if available, and the exact two-sided context used by the CNN) so readers can see whether the available context predicts labels above chance.

**Do not simply add the 6–9 numbers to a table and call the proof complete.** The table must use the actual label sequence and exact input-context geometry.

---

# 4. Critical finding 2: the zero-phase FIR and inter-symbol interference must be included in the guarantee

**Severity: Critical — resolve before defending a structural impossibility claim.**

## 4.1 Manuscript locations

- **PDF p. 6, Fig. 3 and caption, extracted lines 31–50:** describes the separate ground-truth and signal paths, the `N`-symbol CNN window, and the passband input.
- **PDF p. 7, Section V-B, “Signal conditioning,” extracted lines 23–45:** states that the waveform is low-pass filtered with a zero-phase FIR and decimated from 145.44 to 20 samples per symbol.
- **PDF p. 7, Table VI, extracted lines 5–14:** reports 100, 40 and 20 input samples and CNN receptive fields 93, 39 and 20 samples.
- **PDF p. 12, Section VIII, “Synthetic noise” and “Symbol timing,” extracted lines approximately 53–69:** acknowledges software-added AWGN and supplied symbol timing.

## 4.2 Code evidence

`src/build_dataset.py:77–93`:

```python
taps = firwin(129, ANTIALIAS_HZ, fs=cap.fs)
x = filtfilt(taps, [1.0], cap.ch1.astype(np.float64))
...
wave = np.interp(pos.ravel(), np.arange(cap.n, dtype=np.float64), x)
```

The filter is applied to the full recording before the sample windows are constructed. `filtfilt` is a forward-backward, zero-phase operation. For a finite 129-tap FIR, the equivalent forward-backward impulse response has nominal support of 257 raw samples, i.e. a half-width of 128 raw samples on either side (ignoring special edge handling at the recording boundaries).

The capture metadata reports approximately 145.44 raw samples per symbol. Thus, the nominal half-support is:

`128 / 145.44 ≈ 0.88 symbol periods`.

This means a sample inside a cropped one-symbol 8-QAM window may depend on raw samples from outside the nominal symbol interval. The crop length is still one symbol, but its values are not necessarily independent of neighbouring transmitted symbols.

`src/generate_patterns.py:4–14` makes the authors' own concern explicit: it says that with a one-symbol 8-QAM receptive field, inter-symbol interference is the only way context reaches the network.

## 4.3 Interpretation — important qualification

This finding does **not** prove that the model has memorised labels, or that the experiment is invalid. It proves that the theoretical statement “the model cannot see more than one symbol” is not enough to establish what transmitted information can influence the samples in that one-symbol crop.

The complete effective context is determined by:

- the transmitter's pulse shape and analogue filtering;
- the hardware channel and measured transmit-filter delay;
- the anti-alias FIR;
- forward-backward filtering;
- resampling and timing alignment;
- the final crop supplied to the CNN.

A structural theorem needs to cover the whole signal-processing chain that generates the CNN input, not merely the last tensor dimension.

## 4.4 Required experiments

1. **Impulse-response/support report:** publish the exact FIR coefficients or reproducible filter design, its effective two-sided support and the corresponding support in symbol periods.
2. **Context-ablation experiment:** create a control that retains the target symbol's useful waveform but suppresses or randomises neighbouring-symbol contributions as far as the signal model permits.
3. **Context-only baseline:** use the measured symbol-label sequence to test how accurately the exact available neighbour context predicts the target label without the target waveform.
4. **Front-end sensitivity:** compare the present zero-phase filtering with a causally implementable front end, or explicitly state that the receiver is offline/noncausal and quantify the resulting context support.
5. **Boundary and pulse-shape analysis:** inspect how adjacent symbols contribute to the waveform around decision times. A one-symbol crop must not be treated as a one-symbol information dependency without this analysis.

If the result is that the front end allows neighbouring symbols to influence the target crop, narrow the claim. A defensible conclusion may be that the **digital CNN crop** is bounded, not that all sequence-derived context is mathematically excluded end-to-end.

---

# 5. Major finding 3: the shuffled-label control does not match the manuscript's description

**Severity: Major.**

## 5.1 Manuscript locations

- **PDF p. 8, Section VI-C, “Negative controls,” extracted lines 35–37 and 38–50:** says permuting labels among symbols leaves performance at chance.
- **PDF p. 10, Table X, extracted lines 38–47:** caption explicitly says the controls are on leave-one-pattern-out folds; the shuffled-label results are shown as part of that table.
- **PDF p. 10, Fig. 6 caption, extracted lines 77–78:** says both negative controls are shown for leave-one-pattern-out folds.

## 5.2 Code evidence

`src/run_experiments.py:38–49` runs the shuffled-label control only once per modulation, using the **first leave-one-board-out fold**:

```python
fold_name, tr, te = SP.make_splits(meta, mod)["leave_one_board_out"][0]
net, res, ntr, nte = T.train_fold(..., shuffle_labels=True)
```

`src/train.py:87–99` creates a copied label array, constructs a combined index of all train **and test** capture spans, permutes labels across that combined span, and then uses the modified array to build both training and test windows:

```python
span = np.concatenate([np.arange(lo[c], lo[c + 1])
                       for c in list(tr_caps) + list(te_caps)])
label[span] = rs.permutation(label[span])
ws_tr = DT.WindowSet(wave, label, ..., tr_caps, ...)
ws_te = DT.WindowSet(wave, label, ..., te_caps, ...)
```

This is not the standard control described in the manuscript, where the training labels are shuffled to break the waveform/target relationship and the held-out test labels remain the original ground truth. The current script changes the targets in both partitions before evaluation.

## 5.3 Why this matters

A negative control should answer a clearly specified question. If the aim is to test whether a network can learn the true physical waveform-to-label mapping, the training targets should be corrupted while the test targets remain the independently retained ground truth. Otherwise the evaluation score is measured against a corrupted target array rather than the original held-out label sequence.

There is a second mismatch: the saved shuffled-control rows are LOBO folds, while Table X and Fig. 6 identify their controls as LOPO. `src/figures.py:147–181` draws the LOPO model and position-only values alongside the shuffled values selected from those three LOBO control rows.

## 5.4 Required correction

- Preserve the original `y_test` before any permutation.
- Shuffle only training labels, within each training capture or under a documented permutation scheme that preserves the class alphabet.
- Evaluate against the untouched original test labels.
- Repeat over several independent shuffle seeds.
- Run the control on the same held-out pattern folds used by Table X, or change the table/figure/caption so they truthfully describe the actual LOBO control.
- Save the train/test capture identifiers and random seed for every control run.
- Regenerate Table X and Fig. 6 from the corrected output files.

Do not retain the current Table X/Fig. 6 captions if the control remains the one implemented in the supplied code.

---

# 6. Major finding 4: neural and classical comparisons need a matched evaluation protocol

**Severity: Major.**

## 6.1 Manuscript locations

- **PDF p. 8, Table VII, extracted lines 5–18:** neural and classical SER values.
- **PDF p. 8, Section VI-A, “Error rate against a blind classical receiver,” extracted lines 49–61:** says the two receivers use identical noise and reports parity/equivalence.
- **PDF p. 12, Section VIII, “Symbol timing,” extracted lines approximately 58–69:** acknowledges symbol timing is supplied and describes the system as symbol-synchronous.

## 6.2 What the implementation does

### Neural path

- `src/data.py:72–87` creates AWGN separately for each extracted window, using the capture-wide RMS reference.
- `src/train.py:68–80` evaluates the network on windows, adding noise through `WindowSet.gather`.
- `src/train.py:84–121` trains one model for a given fold; the default seed is 0.

### Classical path

- `src/classical_snr.py:33–50` evaluates the classical receiver on complete capture waveforms, not the per-symbol crops used for the CNN.
- At each SNR, it adds noise to the entire stored record, estimates the carrier from the noisy full waveform, and demodulates the whole record.
- `src/classical.py:39–52` estimates carrier frequency from the supplied trace using a fourth-power estimator.

The code uses the same *SNR scale/reference convention* (capture-level RMS) but does not use identical noise samples. The CNN receives independent random noise on each window; the classical routine receives a separately generated continuous-noise trace. Therefore, the paper's phrase “Noise is identical for both” is not literally true at the waveform-realisation level.

## 6.3 Aggregation mismatch

`src/train.py`/`src/run_experiments.py` produce CNN results per split/fold. Table VII's learned values equal an **unweighted mean across the LOPO pattern folds**.

`src/classical_snr.py:35–50` instead computes a per-record SER and averages across all records of the modulation. That is not the same averaging unit as the neural LOPO pattern-fold mean.

The following figures were recalculated from `results/experiments.csv`. “Macro LOPO” is the arithmetic mean of the pattern-fold SERs; “micro LOPO” weights each fold by `n_test`; classical is the value from `results/classical_snr.csv` (per-record mean over all captures).

| Modulation | SNR | Neural macro LOPO | Neural micro LOPO (weighted by test windows) | Classical result CSV |
|---|---:|---:|---:|---:|
| BPSK | 0 dB | 0.0000557 | 0.0001044 | 0.0000173 |
| QPSK | 0 dB | 0.0093185 | 0.0153686 | 0.0036847 |
| 8-QAM | 0 dB | 0.1319345 | 0.1452354 | 0.1218576 |
| BPSK | 6 dB | 0 | 0 | 0 |
| QPSK | 6 dB | 0.0000278 | 0.0000695 | 0.0000231 |
| 8-QAM | 6 dB | 0.0065477 | 0.0101038 | 0.0058027 |
| 8-QAM | 9 dB | 0.0003438 | 0.0012607 | 0.0003337 |

These micro values are not replacement results; they simply demonstrate that the aggregation choice matters, especially for 8-QAM, where the fold sizes are unequal. The classical per-record mean is a third aggregation scheme. The journal should define the estimand and apply it consistently.

## 6.4 Observation-budget mismatch

The CNN receives windows of 20, 40 or 100 samples (one, two or five symbols). The classical SNR script estimates the carrier from an entire approximately six-second capture. The capture manifest reports approximately 3,596 symbols per recording.

A classical receiver may legitimately use a long preamble or long carrier-estimation interval. But this is not the same observation budget as a neural model that classifies a target window with a few neighbouring symbol intervals. The comparison should therefore be framed as one of the following:

- a performance comparison with explicitly different synchronization/observation resources; or
- a matched-budget comparison in which both receivers are given the same waveform duration, same timing information and same permissible calibration data.

The paper should not imply equal information/latency budgets unless they are made equal.

## 6.5 “Matched filter” description needs verification

`src/classical.py:55–61` downconverts the real passband waveform and then computes:

```python
sym = bb.reshape(-1, sps).mean(axis=1)
```

That is a rectangular integrate-and-dump operation over each symbol interval. It is a matched filter only if the relevant transmitted pulse shape and receive filter make this the appropriate matched operation (for example, an appropriate rectangular pulse). The manuscript does not sufficiently establish that condition.

Required action:

- report the measured transmit pulse shape and the receive filter used by the classical benchmark;
- either justify the integrate-and-dump as the correct matched filter for this hardware, or implement the matched filter corresponding to the measured pulse response;
- document timing, carrier and phase estimation resources given to the baseline;
- avoid describing the comparison as fully observation-matched if the classical receiver uses the full record.

## 6.6 Required correction

Create one evaluation script that uses the same held-out captures and aggregation convention for both systems. Ideally, use the same noisy waveform realisation for paired comparisons. Report both macro-pattern and micro-symbol metrics if both are scientifically relevant. Add confidence intervals that respect capture/pattern clustering rather than treating every repeated symbol as independent.

---

# 7. Major finding 5: pattern and board are confounded in the current capture manifest

**Severity: Major.**

## 7.1 Manuscript locations

- **PDF p. 5, Section III-D, “Corpus,” extracted lines 31–68:** describes 43 captures, board distribution and distinct pattern handling.
- **PDF p. 5, Table V, extracted lines 5–15:** shows 2 independent BPSK patterns, 3 QPSK patterns and 5 independent 8-QAM patterns.
- **PDF p. 9, Table IX, extracted lines 35–69:** reports the individual LOPO folds and their training/test record counts.
- **PDF p. 12, Section VIII, extracted lines 44–52:** acknowledges content diversity and device-count limitations.

## 7.2 What the 43-row manifest shows

The record distribution in the supplied `captures.csv` is:

| Modulation | Pattern | Total records | Distribution across boards |
|---|---|---:|---|
| BPSK | PRBS63 | 15 | 3 on each of B1–B5 |
| BPSK | S1 | 1 | B1 only |
| QPSK | Q1 | 10 | 2 on each of B1–B5 |
| QPSK | Q2 | 1 | B1 only |
| QPSK | Q3 | 1 | B1 only |
| 8-QAM | T1 | 10 | 2 on each of B1–B5 |
| 8-QAM | T2 | 1 | B1 only |
| 8-QAM | T3 | 1 | B1 only |
| 8-QAM | T4 | 1 | B1 only |
| 8-QAM | T5 | 1 | B1 only |
| 8-QAM | T6 | 1 | B1 only |

T1/T2 are grouped together as one independent pattern in `src/splits.py:24–30`, which is sensible given the stated antipodal relationship. Even so, most additional words (S1, Q2, Q3 and T3–T6) are only represented on B1.

## 7.3 Why this matters for LOPO

For several held-out patterns, the test set consists of just one B1 recording. Those results therefore demonstrate performance on a new word **on B1**, not necessarily generalisation to that word across independent hardware devices.

Table IX makes the thin folds visible:

- BPSK PRBS63 held out: only one training capture (S1) remains.
- QPSK Q1 held out: only two training captures (Q2 and Q3) remain.
- QPSK Q2 and Q3 held out: each is a single-record B1 test pattern.
- 8-QAM T3, T4, T5 and T6 held out: each is a single-record B1 test pattern.

This is not a reason to discard the results. It is a reason to qualify the generalisation claim and design a more crossed experiment.

## 7.4 Required correction

**Best remedy:** record each independent pattern on multiple boards, ideally under multiple attenuation settings, so pattern and device effects can be separated.

If new captures are not possible:

- make clear that several LOPO results are single-capture, B1-only demonstrations;
- do not generalise those folds to unseen hardware;
- present the leave-one-board-out result as a separate, limited device-generalisation experiment;
- state that the design is not fully crossed and that pattern identity is confounded with board identity for the additional words;
- avoid claiming that the entire pattern family has been validated across devices.

---

# 8. Major finding 6: single-seed training and one random split are not enough to characterise stability

**Severity: Major.**

## 8.1 Code evidence

- `src/splits.py:52–67` creates one random split with `np.random.default_rng(0)`.
- `src/train.py:84–85` defaults to `seed=0`.
- `src/train.py:102–120` sets the Torch seed and trains a single network for the supplied fold.
- `src/run_experiments.py:21–36` executes each fold once.

The 35-fold count itself is correct. The issue is not the number of folds; it is the absence of repeated random partitions and repeated initialisations.

## 8.2 Required correction

- Train at least 5 independent seeds per fold if computationally feasible; report the chosen number and rationale.
- Repeat random 80/20 capture splits across multiple fixed, recorded seeds.
- Report mean, standard deviation and/or confidence intervals for the variability across seeds and partitions.
- Preserve the exact split manifests and seeds in the repository.
- For the two thin LOPO folds, avoid presenting a single result as a stable estimate. Explicitly describe them as low-training-data cases.

Because symbols in one capture are repeated transmissions through one board and one signal path, do not calculate overly narrow confidence intervals by treating all symbols as independent samples. Use capture-level or pattern-level resampling (cluster bootstrap) for uncertainty on generalisation.

---

# 9. Major finding 7: 8-QAM ground-truth mapping and pattern framing need to be tied to the proof

**Severity: Major.**

## 9.1 Manuscript locations

- **PDF p. 5, Section III-D, extracted lines approximately 45–60:** says two of six 8-QAM words are bitwise complements; four of six words deliver six rather than eight constellation points because the fixed symbol-grouping offset does not coincide with configuration order.
- **PDF p. 7, Section IV-C, “Verification of the labels,” extracted lines 23–42:** reports BPSK and QPSK checks, then says that the 8-QAM configured-word mapping cannot be resolved by the stated bijection check.
- **PDF p. 7, Section V-A, extracted lines approximately 46–66:** uses the bit-order recurrence to justify the 8-QAM cap as well as the other modulations.

## 9.2 Code evidence

- `src/mcm31.py:427–436` says the DIP-switch-to-symbol mapping is unresolved and assumes a memoryless bijection.
- `src/mcm31.py:442–459` quantises the measured complex symbol into BPSK, QPSK or 8-QAM classes.
- `src/mcm31.py:462` onward calculates constellation-point occupancy fingerprints.
- `src/build_dataset.py:95–125` estimates phase/rotation over the full capture, obtains averaged word labels and tiles those labels across the recording.

The occupancy fingerprint can identify a loaded word without knowing the exact permutation from configuration groups to constellation-point classes. It does not, on its own, prove the **time-ordered mapping** between every configured bit group and every measured symbol label.

## 9.3 Required correction

For 8-QAM, provide one of the following:

1. a measured, time-indexed mapping from configured 3-bit groups to the label sequence, with the framing offset explicitly established; or
2. a proof that the context-complexity property being claimed is invariant under every mapping and offset that is still consistent with the observed data.

Then calculate the context property on the actual cyclic sequence used in the learning target. State whether the mapping is a global memoryless bijection, a fixed rotation/reflection, an offset in grouping, or something more complex. A general claim of “unknown bijection” is not enough if the order or grouping phase is uncertain.

Also clarify the role of T1/T2: the manuscript treats them as antipodal equivalents and the split code groups them together. Explain precisely how this equivalence affects labels, pattern counting and the context proof.

---

# 10. Additional methodological issues to address

## 10.1 Capture-level RMS normalisation uses the complete capture

**Code:** `src/data.py:36–49` computes RMS from each complete capture and `src/data.py:72–87` uses it to normalise the input window and scale AWGN.

This is not label leakage, because it does not use test labels. However, it is a full-record calibration statistic. It is not a strictly causal online operation if the receiver is intended to process a stream before the full record has arrived.

**Recommendation:** describe this as capture-level AGC/calibration and state that the complete capture's RMS is available. If online deployment is claimed, replace it with a causal or rolling AGC and report whether performance changes.

## 10.2 Ground-truth creation is offline and uses the full recording

**Code:** `src/build_dataset.py:68–75` uses the coherent template for timing/period; lines 95–120 use coherent averaging and a rotation estimate from the full record to obtain word labels.

This may be a reasonable offline ground-truth procedure. The CNN does not receive the label-generation path directly, as Fig. 3 intends. But the manuscript should explicitly distinguish:

- offline label annotation using the full waveform;
- the information supplied to the trained CNN at inference;
- calibration/timing quantities that are derived from the full recording and then used to form the windows.

A full-record global phase/rotation estimate for label annotation is not the same thing as a per-window receiver estimate. Explain why the resulting labels are independent enough for the purpose of the study and which checks confirm them.

## 10.3 `self_agree` is a consistency diagnostic, not an independent ground-truth measurement

The manifest's `self_agree` field ranges from approximately 0.99944 to 1.0. In `src/build_dataset.py:122–125`, it compares per-symbol quantisation on the unaveraged capture with labels obtained from the averaged word template.

This is useful evidence that the averaged labels are self-consistent with the capture. It is not an external ground-truth measurement independent of the same waveform and quantisation procedure. The manuscript should not describe it as equivalent to a known transmitted reference sequence.

## 10.4 Clarify the meaning of “blind”

The neural network does not receive carrier frequency or a supplied constellation, but it does receive symbol-synchronous windows. The manuscript acknowledges this in Section VIII.

Use accurate wording such as:

> carrier-blind, constellation-agnostic classification of symbol-synchronous passband waveform windows

Avoid using “fully blind receiver” without immediately qualifying the timing/front-end assumptions.

## 10.5 “No carrier frequency, pulse shape or constellation information” needs qualification

The neural network may not be given numerical parameter values explicitly, but it receives waveforms shaped by the physical transmitter/filter and a symbol-synchronous sampling grid. This is not the same as having no information about pulse shape or symbol timing anywhere in the overall pipeline.

Explain the role of the known symbol rate, external trigger, timing estimate and measured waveform shape.

## 10.6 Software AWGN limits the channel-generalisation claim

**Manuscript:** PDF p. 12, Section VIII, extracted lines 53–57.  
**Code:** `src/data.py:14–17` and `src/data.py:72–87`; `src/classical_snr.py:41–50`.

The real hardware supplies the carrier, filter response, quantisation and board-specific variations. AWGN is added in software. This is valid as an experimental procedure, but the conclusions are limited to the tested single-carrier/single-symbol-rate additive-noise regime. The current paper already acknowledges the absence of multipath and fading; keep that limitation prominent in the abstract and conclusion.

## 10.7 “Zero errors” needs finite-sample wording

At several SNR values the saved result has zero observed errors. That should be written as “no errors were observed in the evaluated test samples” and accompanied by the test count. A finite test set cannot establish a true SER of exactly zero.

Report:

- number of errors;
- total number of tested symbols/windows;
- confidence interval or a stated upper bound when zero errors are observed;
- how uncertainty is clustered by capture/pattern.

---

# 11. Numerical and table checks

## 11.1 Dataset totals

**Manuscript:** PDF p. 5, Table V, extracted lines 5–15.  
**Manifest:** `captures.csv`, fields `modulation`, `pattern`, `board`, `n_symbols`.

| Modulation | Records | Symbols | Independent patterns stated in manuscript |
|---|---:|---:|---:|
| BPSK | 16 | 57,536 | 2 |
| QPSK | 12 | 43,152 | 3 |
| 8-QAM | 15 | 53,940 | 5 |
| **Total** | **43** | **154,628** | — |

The manifest totals match the manuscript.

## 11.2 Evaluation fold count

**This is a verified pass, not a discrepancy.**

From `src/splits.py:24–66` and the capture manifest:

- LOPO: BPSK 2 + QPSK 3 + 8-QAM 5 = 10 folds;
- LOBO: 5 boards × 3 modulations = 15 folds;
- leave-one-attenuation-out: BPSK 3 + QPSK 2 + 8-QAM 2 = 7 folds;
- random 80/20: one per modulation = 3 folds.

**Total = 35 main folds.** `results/experiments.csv` has 38 rows because it adds three shuffled-label-control rows.

Do not change the manuscript's “35 folds” merely on the basis of the earlier preliminary concern. The count is consistent.

## 11.3 Table VII values

**Manuscript:** PDF p. 8, Table VII.  
**Source outputs:** `results/experiments.csv` and `results/classical_snr.csv`.

The neural values in Table VII match the unweighted LOPO fold means to four decimal places. The classical figures match `classical_snr.csv`, which averages by record across the modulation rather than by LOPO fold. The issue is the aggregation convention, not a simple transcription error.

## 11.4 Table VIII / IX

The saved fold results support the broad ordering in Table VIII: LOPO is similar to the random split, while LOBO is worse at higher SNR. Table IX's thin folds are visible in the training-record counts. These points should be discussed with the pattern/board confounding explained in Finding 5.

## 11.5 Table X / Fig. 6

The shuffled-label figures do not come from the LOPO split described in their captions. They come from a single first LOBO fold per modulation and the training/test target arrays are both modified. Regenerate this table and figure after correcting the control.

---

# 12. Presentation and manuscript corrections

## 12.1 Abstract and title

- The title is technically clear but the phrase “Provable Memorisation Bound” places the highest burden on the exact mathematical proof. Keep it only if the revised proof truly supports the full end-to-end claim.
- The abstract currently states that the order-six recurrence applies to the experiment as a whole (PDF p. 1, lines 17–31). Rewrite it to explain the pattern-specific analysis and timing/front-end assumptions.
- Report the AWGN limitation and supplied symbol timing in one concise sentence.
- Avoid saying “matches” or “indistinguishable” unless the statistical comparison supports equivalence/non-inferiority under a defined margin and matched protocol.

## 12.2 Introduction and contribution list

**PDF p. 1, extracted lines 43–58.**

The contribution list is currently strong and declarative. After revising the guarantee, make each item correspond to a reproducible deliverable:

1. hardware dataset and provenance;
2. waveform-derived ground-truth method plus independently justified checks;
3. formally defined pattern/context bound and its executable verifier;
4. device-demodulator findings;
5. negative controls that actually follow the stated split and target protocol.

Do not claim “memorisation impossible” if the result only shows no perfect deterministic lookup from a restricted bit context.

## 12.3 Methods section needs more implementation detail

Add details that another research group needs to reproduce the front end:

- exact FIR design (129 taps, 4 kHz cutoff, window/design details and coefficient-generation library/version);
- forward-backward filtering operation and its effective temporal support;
- interpolation/resampling procedure and boundary handling;
- precise context geometry and target location for each modulation;
- precise pulse shape or measured response used by the classical receiver;
- exact random seeds and train/test capture manifest;
- training hardware/software versions and model checkpoint generation procedure.

## 12.4 Author name inconsistency

- **PDF p. 1, author line 9:** `Mohammad Khalid Hasan Nabil`.
- **PDF p. 14, first biography, extracted lines 17–20:** `Md. Khalid Hasan Nabil`.

Standardise the author's published name across title page, submission portal, ORCID, biography and data/code records. Also standardise punctuation in `Md. Maynul Islam` between the author line and biography.

## 12.5 References and literature review

I have not independently checked every one of the 49 references against the publisher's official record. That is a separate reference-validation task. Before submission:

- confirm author order, title, journal/conference, year, volume, issue, pages/article number and DOI;
- distinguish peer-reviewed articles from preprints (particularly newer 2026 material);
- confirm every cited work supports the precise statement attached to it;
- re-check the 23-paper survey table for classification errors and ensure `n.s.` really means the paper does not state a property;
- cite the WorldCUR accepted abstract or any subsequently published proceedings/abstract record where required, and clearly distinguish the journal's expanded technical contribution.

---

# 13. WorldCUR 2027 abstract and journal submission

The author reports that the WorldCUR 2027 abstract was accepted. This is positive, but it does not replace journal peer review of the full method.

The WorldCUR abstract is written for a broad interdisciplinary audience and uses a more accessible explanation of the proposed rule. The IEEE Access abstract can be more technical. However, the WorldCUR claims about the 6–9-bit contexts must be traceable to a reproducible per-pattern calculation in the journal submission.

Before submitting to IEEE:

1. Establish whether the accepted WorldCUR abstract will appear in proceedings, an online abstract book, or another archival publication.
2. Disclose the accepted/presented/submitted WorldCUR material in the IEEE submission if it is considered related prior dissemination.
3. State how the journal manuscript differs: complete mathematical analysis, complete hardware method, detailed evaluation, additional results and limitations.
4. Do not submit substantially identical material elsewhere concurrently.

IEEE Access explicitly permits qualifying expanded versions of an author's conference publications, but requires appropriate citation/disclosure and a clear explanation of the differences. The author should confirm how an *accepted abstract-only* contribution is treated in this specific case; do not assume that abstract acceptance automatically bars the journal article or automatically requires no disclosure.

Official IEEE guidance: [IEEE Access — Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/) and [IEEE Author Center — Submission and Peer Review Policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/).

---

# 14. AI-assisted writing and research-integrity disclosure

The author says ChatGPT, Claude and Gemini were used for grammar correction and polishing, and that the hardware data were collected by the author. The latter is a separate experimental contribution; it does not by itself determine what AI disclosure is required.

IEEE's current public guideline says that AI-generated content in an article (including text, figures, images and code) must be disclosed in the acknowledgements, identifying the AI system, the affected sections and the level of assistance. It separately says that language editing and grammar enhancement are generally outside the intent of that policy, although disclosure is recommended.

**Practical recommendation:** accurately record what each tool did. If the tools only corrected grammar and improved clarity, a short transparency statement is appropriate. If any tool drafted substantive text, literature synthesis, equations, code, figures or interpretations, disclose those contributions in the manner IEEE requires. Do not describe substantive generation as “grammar polishing.” The human authors remain responsible for every claim, result and citation.

Possible wording **only if factually accurate**:

> During manuscript preparation, the authors used ChatGPT, Claude and Gemini for language editing, grammar correction and improvements in writing clarity. The authors reviewed and verified the final wording and take responsibility for the accuracy and integrity of the manuscript.

If AI generated substantive content, this sentence is insufficient; specify the affected sections and the actual level of generation.

Official source: [IEEE — Author Guidelines for Artificial Intelligence (AI)-Generated Text](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/).

---

# 15. IEEE Access submission requirements checklist

The following is a practical checklist based on the IEEE Access pages checked on 30 October 2026. Always re-check the live instructions immediately before submission.

| Requirement | Current manuscript/status | Action |
|---|---|---|
| Original research contribution | Appears to be an original experimental/methodological research article, subject to resolving the bound claim and confirming novelty against literature. | Explain the specific novelty and difference from WorldCUR material. |
| Research Article type | Appropriate category. | Select “Research Article” in the submission system. |
| IEEE Access template | Current PDF is 14 pages and visually follows IEEE two-column style, but the editable source was not supplied for audit. | Prepare final Word/LaTeX source using the official IEEE Access template and ensure source/PDF match. |
| File size | Not checked for final source/PDF pair. | IEEE submission checklist says each submission file should be no more than 40 MB. |
| Abbreviations | Review required. | Define each acronym/abbreviation on first use in the main article, even if it appeared in the abstract. |
| Keywords | Index terms are present in PDF p. 1. | Enter 3–10 suitable manuscript keywords in submission portal. |
| Author list | Present. | Verify every author qualifies and approve final author order and affiliations. |
| Corresponding author ORCID | Not verifiable from PDF/ZIP. | Submitting author must associate a publicly visible, populated ORCID with the account. |
| Author biographies | Present on PDF p. 14. | Standardise names and verify biography information for all authors. |
| Grammar and references | Editorial/reference pass still needed. | Proofread and verify every reference. |
| AI disclosure | Depends on actual tool usage. | Disclose substantive AI-generated content; language-editing disclosure is recommended. |
| Concurrent submission | Not known. | Do not submit the same manuscript to another journal simultaneously. |
| Supplementary materials | Code and DOI are listed, but the supplied archive is incomplete for reproduction. | Prepare raw data, scripts, environment, readme, model artefacts and supplemental files. |
| Page length | 14 pages. | Under IEEE Access's recommended 20 pages; no formal page limit. |
| APC | Current page lists USD 2,160 plus applicable local taxes. | Confirm funding and institutional discounts before submission. |

Official links:

- [IEEE Access — Submission Guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/)
- [IEEE Access — Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
- [IEEE Access — Article Processing Charges](https://ieeeaccess.ieee.org/about/article-processing-charges/)

The live IEEE Access pages checked for this report state: double-column/single-spaced official template; Word or LaTeX source plus matching PDF; maximum file size 40 MB; populated public ORCID for submitting author; biographies for all authors; 3–10 keywords; no simultaneous journal submission; no formal page limit but a recommendation to remain under 20 pages; and USD 2,160 APC plus local taxes. These are administrative requirements, not a guarantee of acceptance.

---

# 16. Recommended repair plan — in priority order

## Phase 1: define the scientific claim correctly

**Do this before rewriting the abstract.**

1. Write down the exact target label for every modulation.
2. Define the target waveform interval and explicit neighbouring symbols in every CNN crop.
3. Derive the bit-to-symbol grouping and framing offset for all patterns.
4. Compute pattern-specific next-bit and next-symbol context complexity from the actual cyclic label sequences.
5. Compute the two-sided context complexity that corresponds to the BPSK centred window.
6. Explain the 8-QAM mapping ambiguity and establish a valid invariant or obtain the mapping.
7. Decide what is genuinely proved: no perfect pattern-only lookup, no use of a particular bounded digital context, or a stronger end-to-end impossibility statement. Use only the strongest statement actually justified.

## Phase 2: account for the physical and digital front end

1. Quantify the pulse response and the anti-alias FIR support.
2. Include forward-backward filtering, interpolation and sample timing in the information-support analysis.
3. Evaluate whether neighbouring symbols influence the target crop through ISI.
4. Run context-only and front-end ablation tests.
5. If the guarantee cannot be maintained end-to-end, narrow the title, abstract and conclusion to a verifiable digital-window cap rather than universal impossibility.

## Phase 3: repair controls and baseline comparisons

1. Correct the shuffled-label procedure so only training labels are permuted and test labels remain true.
2. Regenerate Table X and Fig. 6 under LOPO if that is what the manuscript claims.
3. Compute the classical baseline on the same held-out captures and same aggregation convention.
4. Make noise paired where possible, or remove the phrase “identical noise.”
5. Clarify the classical matched-filter implementation and observation duration.
6. Run multiple training seeds and repeated random splits.
7. Report confidence intervals, error counts and sample counts.

## Phase 4: improve experimental design and generalisation claims

1. Collect additional patterns on boards B2–B5 if possible.
2. Avoid confounding novel pattern identity with B1 alone.
3. If additional hardware collection is impossible, explicitly qualify B1-only single-capture results.
4. Consider a crossed pattern-by-board design in a follow-up experiment.

## Phase 5: reproducibility and editorial preparation

1. Add all raw captures or stable download instructions and licence/usage information.
2. Include `dataset/windows.npz` or exact commands to generate it from the raw captures.
3. Add pinned dependencies, Python/PyTorch/SciPy versions and tested OS/hardware.
4. Add a README with one-command or clearly sequenced reproduction steps.
5. Save exact split IDs, seeds, checkpoints, predictions and result-generation scripts.
6. Correct author names and affiliations.
7. Verify every reference.
8. Align the WorldCUR and IEEE disclosures.
9. Re-run the PDF/source comparison before submission.

---

# 17. Questions and tasks for Claude to cross-verify

Use the ZIP and manuscript together. Ask Claude not to assume this audit is correct; it should independently check every claim and provide counter-evidence where appropriate.

1. **Re-derive the pattern-context table.** Independently compute the minimum cyclic preceding-bit context for every pattern in `src/mcm31.py`. Check PRBS63, S1, Q1–Q3 and T1–T6. State the exact recurrence/context definition and whether the result is invariant to cyclic rotation and bit complement.
2. **Check the actual model context.** Read `src/data.py` and explain precisely what BPSK, QPSK and 8-QAM tensors contain relative to the target symbol. Does the proof in PDF p. 7 Section V-A match the actual context geometry?
3. **Prove or refute the end-to-end claim.** Include the filter in `src/build_dataset.py`, the zero-phase `filtfilt` operation, interpolation, physical pulse response and ISI. Can neighbouring transmitted symbols affect the target crop? What assumptions are needed to call the bound structural?
4. **Verify the 8-QAM grouping.** Determine the grouping offset consistent with the manuscript's claim that four of six words generate six of eight points. Verify this against the actual label arrays if the full dataset can be made available. Is the context-complexity property invariant under the unresolved mapping?
5. **Audit the shuffled-label control.** Trace `src/run_experiments.py:38–49`, `src/train.py:87–99` and `src/figures.py:147–181`. Do Table X/Fig. 6 describe the actual experiment? Propose a correct control and exact test labels.
6. **Reproduce Tables VII–IX.** Recalculate means from `results/experiments.csv`, show the aggregation unit and compare to manuscript tables. Check that Table VII's classical values use the same test captures/fold weighting.
7. **Check the noise statement.** Trace random-number generation in neural evaluation and `classical_snr.py`. Are the same noise samples used or only the same SNR convention?
8. **Check observation budget.** How many samples does the classical carrier estimator use compared with each neural inference input? Is the comparison fair, and what would be the matched-budget experiment?
9. **Check the matched-filter term.** Is `reshape(...).mean(axis=1)` a valid matched filter for the measured MCM31/EV transmit pulse? What extra measurement/derivation is required?
10. **Check the pattern/board cross-tab.** Independently reproduce the pattern-by-board counts from `captures.csv`; assess whether LOPO is confounded with board B1.
11. **Check statistical inference.** Which uncertainty estimates are appropriate given repeated cyclic symbols within a single capture and five devices? Propose a cluster bootstrap or other appropriate method.
12. **Check reproducibility gaps.** List all files/scripts/data/checkpoints that are required to run the project but absent from `code+results.zip`.
13. **Check manuscript wording.** Flag every statement that claims “impossible,” “provable,” “blind,” “matched filter,” “identical,” “parity” or “indistinguishable” and evaluate whether the implementation supports it.
14. **Check IEEE Access submission rules.** Re-check the official instructions current on the date of the review; do not rely on third-party summaries.
15. **Give an independent recommendation.** State what would cause reject, major revision, or a defensible submission, and separate fatal issues from correctable reporting issues.

Ask Claude to give **file paths and line numbers**, not just general comments. Require it to classify each result as confirmed, contradicted, uncertain, or requiring the raw waveform data.

---

# 18. Final decision and bottom line

The research is not “dead.” The supplied code shows a real bounded-window CNN, a thoughtful capture-level split strategy, and a substantial effort to measure labels from hardware waveforms. The 43-record manifest and the principal fold count are consistent with the manuscript. The core idea of preventing a receiver from exploiting repeated test sequences is worth pursuing.

However, the manuscript currently presents a stronger guarantee than the supplied implementation and evidence establish. The most important unresolved question is whether the **actual waveform window after the complete transmitter/front-end/preprocessing chain** is structurally incapable of exploiting the relevant sequence context. The zero-phase filter and the code's own ISI discussion make this a necessary analysis, not an optional refinement.

The shuffled-label control also needs to be rerun as described, and the learned/classical comparison should be re-aggregated and qualified. The single-board concentration of additional patterns limits the current LOPO generalisation evidence.

**My recommendation: do not submit this PDF yet. First repair the mathematical definition and end-to-end context analysis; then rerun the controls and fair baseline comparisons.** If the stronger theorem fails, retain the dataset and experiments but narrow the paper's central claim. A precise, defensible methodological contribution is more valuable than an impressive but overextended “impossibility” statement.

---

# 19. External policy sources checked on 30 October 2026

1. **IEEE Access — Submission Guidelines:** https://ieeeaccess.ieee.org/authors/submission-guidelines/  
   Template, source/PDF requirements, AI-generated text disclosure, author/ORCID/biography requirements, abbreviations, keywords, manuscript type and page guidance.
2. **IEEE Access — Preparing Your Article:** https://ieeeaccess.ieee.org/authors/preparing-your-article/  
   Acceptance criteria; originality, technical standard, conclusions supported by data, references, supplementary material, and the policy for expanded conference versions.
3. **IEEE Author Center — Submission and Peer Review Policies:** https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/  
   Prior-publication disclosure, explanation of differences and multiple-submission policy.
4. **IEEE — Author Guidelines for Artificial Intelligence (AI)-Generated Text:** https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/  
   Disclosure of AI-generated article content and separate treatment of grammar/language editing.
5. **IEEE Access — Article Processing Charges:** https://ieeeaccess.ieee.org/about/article-processing-charges/  
   Current listed APC and page-length guidance.

*Policy figures and requirements can change. Confirm them again in the live IEEE Access submission portal immediately before submitting.*
