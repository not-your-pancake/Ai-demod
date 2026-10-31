# Point-by-point response to the two technical reviews

Revision of 2026-10-30. **R1** = `technical review/IEEE_Access_Manuscript_Code_Audit.md`.
**R2** = `technical review/Neural_Demodulation_Review_for_Author.docx`.

Every review claim was checked against the code in `src/`, the saved results in
`results/`, and the dataset in `dataset/windows.npz` (built from the raw captures).
Each claim gets one verdict:

- **Confirmed**: the claim is correct.
- **Partly**: part of the claim is correct.
- **Contradicted**: the evidence shows the claim is wrong.
- **Not checkable here**: needs data or information we don't have.

Every new number is reproduced by a script in `src/`:

| Script | Produces |
|---|---|
| `context_bound.py` | Table VI (`results/context_complexity.csv`), the PRBS63 pattern counts, filter support, measured ISI, context-only and position-only controls |
| `classical_lopo.py` | Table VII classical column (`results/classical_lopo.csv`) |
| `run_experiments.py` | the seed-0 folds and the corrected shuffled-label control (`results/experiments.csv`: 35 + 50 rows) |
| `run_revision.py` | extra seeds, random partitions and the latent-head check (`results/revision_runs.csv`) |
| `revision_tables.py` | prints Tables VII, VIII and X from the CSVs above |
| `figures.py` | Figs. 4–6 (`fig1`–`fig3`) |
| `window_control.py` | the positive control, Section VI and `tab:positive` (`results/window_control.csv`; `--summary` prints the table, `--audit` the leakage checks) |
| `isi_readability.py` | the trace-readability control in Section VI-C (`results/isi_readability.csv`) |

`train.py` itself was fixed, so a fresh run of `run_experiments.py` produces the
corrected control. The three old joint-shuffle rows were removed from
`experiments.csv` and remain in git history.

## 1. The memorisation bound (R1 §3, R2 #1). Confirmed. Rewritten.

| Claim | Verdict | Evidence |
|---|---|---|
| "each five-bit window occurs twice with distinct successors" is false | **Confirmed** | 31 of the 32 five-bit patterns occur twice. The all-zero pattern (all-ones in the stored polarity) occurs once and is followed by a one. |
| The PRBS63 string in the code is the complement of the sequence that satisfies Eq. (8) | **Confirmed** | `mcm31.verify_prbs63` checks the complement. Eq. (8) is now stated for the complement. |
| The 24-bit words don't satisfy Eq. (8), so the argument can't be extended to QPSK/8-QAM | **Confirmed** | Minimum preceding bits that determine the next bit: PRBS63 6, S1 7, Q1 8, Q2 6, Q3 9, T1 9, T2 9, T3 8, T4 8, T5 9, T6 8. This reproduces R1's table exactly. |
| The windows are centred (BPSK), include the target's own bits (QPSK), and context can arrive through ISI (8-QAM) | **Confirmed** | `data.CONTEXT`. The neighbouring content is 2+2 symbols for BPSK, 1+0 for QPSK and 0 for 8-QAM. |
| Next-bit complexity is not next-symbol complexity | **Confirmed** | Computed on the *measured* label sequences, in symbols: κ⁻ = 6/7 (PRBS63/S1), 3/3/2 (Q1–Q3), 1 (T1+T2), 2 (T3–T6). Two-sided κ± = 10/7, 4/3/3, 1, 2. |
| "Not determined" ≠ "uninformative" | **Confirmed** | The best in-word lookup from the window's own neighbours scores 0.508 (PRBS63), 0.625 (S1), 0.417/0.583/0.333 (Q1–Q3), against chance of 0.5 and 0.25. |

**Change.** New title: "…With a Bounded Input Context". Section V-A is rewritten
around the per-word context complexity of the measured labels, with a new
Table VI (`tab:context`). Every window is shorter than its word's complexity, so
deterministic lookup is excluded. The section then states plainly what the bound
does *not* exclude. The abstract, introduction, contributions, related work,
discussion and conclusion all drop "provable / impossible / excluded by
construction". One new result: the margins explain the caps. A third QPSK symbol
would determine Q3, and a second 8-QAM symbol would determine T1/T2.

## 2. Preprocessing and ISI (R1 §4, R2 #2). Confirmed. Quantified.

- The FIR is 129 taps, Hamming window (`firwin`), with a 4 kHz cut-off at 87.26 kSa/s, applied with `filtfilt`. The effective support is 257 raw samples, ±0.88 symbol, but only −50 dB of its energy lies beyond ±0.5 symbol.
- The transmitter's pulse ISI, measured on the word-averaged clean waveform, is −18.5 to −18.9 dB (BPSK), −21.0 to −25.2 dB (QPSK) and −24.2 to −27.9 dB (8-QAM) relative to the symbol's own RMS. That is 0.12, 0.08–0.16 and 0.17–0.21 of half the minimum class distance. **A one-symbol 8-QAM window does carry a weak trace of its neighbours.**
- The implication R1 hints at is checked. On splits that reuse a training word, a lookup from the *previous 8-QAM label* gives SER 0 on 4 of the 5 LOBO folds (in T1 each point occurs once per word). On LOPO it is at chance (0.90). The LOPO-versus-random comparison is therefore the empirical test of ISI exploitation, and the paper now says so.
- Section V-B now gives the FIR design, `filtfilt`, the linear-interpolation grid formula, edge handling, target position, the noise formula, and the injection point.

## 3. Constellation penalty collapse (R2 #3). Confirmed, and worse than the review said.

- The 8-QAM latent distance in `ablation_physics.csv` is 0.2501 / 0.2501. That is exactly the value you get when every output has the same magnitude (unit circle against the outer ring at 1.250).
- Retrained at λ = 0.2 with the class means dumped (`src/run_revision.py`, kind `latent`). **All eight 8-QAM class means sit at (0.71, 0.71), and all four QPSK class means at (−0.71, 0.71). The within-class spread is ≤ 0.005.** The latent head outputs a constant.
- **Change.** The claim "aligns the latent head with the physical constellation" is removed. Sections V-C and VI-D explain the degenerate solution, and the contribution list now reports the collapse as part of the negative result.

## 4. Neighbouring errors ≠ no memorisation (R2 #4). Confirmed.

The "memorised mapping would scatter errors uniformly" sentence is removed. Fig. 7
is now described as consistent with geometric discrimination. Its caption gives
its provenance: LOBO B2 fold, 18 epochs, λ = 0.2, one noise realisation, 863
errors in 7,192 symbols. Both confusion directions are combined, with 9.6%
directional asymmetry. Pairs with fewer than 5 confusions are not drawn.

## 5. Label verification (R2 #5; R1 §9, §10.2–10.3). Confirmed. Clarified.

- "All sixteen BPSK records reproduce PRBS63" is **wrong**: only 15 are PRBS63. The S1 record was checked separately and matches the S1 word at 100% (complement, cyclic shift 18). The text now says this.
- What is verified per modulation is now stated: BPSK bits, QPSK a dibit mapping, 8-QAM only constellation classes. 8-QAM results are framed as class recovery.
- The context analysis uses the measured class sequences. It is invariant to cyclic shift and to any fixed relabelling, so the unresolved tribit mapping and framing offset do not affect it (R1 §9).
- Labelling convention: a fixed φ_ref. The evidence is that 148 same-word pairs agree up to a cyclic shift.
- The independence of the classical baseline is now stated. It is built from the same operations as the label path, so it is a reference, not an independent check.
- The label path was described inaccurately. The code derives labels from the filtered, resampled *record* (template used only for period and τ), and the timing criterion maximises mean |s|, not mean |s|². Both are fixed. `self_agree` was not claimed in the manuscript, so nothing to change.

## 6. Equivalence claims, classical comparison (R2 #6, #7; R1 §6). Confirmed.

| Claim | Verdict | Evidence |
|---|---|---|
| "Noise is identical for both" | **Confirmed false** | Same convention, different realisations. The network gets a fresh draw per window; classical gets one continuous record. |
| Aggregation mismatch | **Confirmed** | Learned = macro over LOPO folds. Classical CSV = mean per record over all records. Classical was recomputed on the same LOPO test records, macro, 5 noise realisations: 0 dB BPSK 1.9e-5, QPSK 0.0027, 8-QAM 0.1206. |
| Observation budget differs | **Confirmed** | Classical uses whole-record carrier and phase (~3,600 symbols). The network sees 1–5 symbols plus record-level AGC/timing. Now stated. |
| "Matched filter" | **Confirmed** | `reshape(...).mean()` is integrate-and-dump, matched only to a rectangular pulse. Now described that way. |
| Not stated by either review | — | The classical receiver also resolves the M-fold phase ambiguity **against the true labels** (`classical_snr.ser`), searches the carrier within ±20 Hz of nominal, and uses fixed decision regions. All three are now stated. |
| "Above 9 dB the learned receiver reaches zero errors" | **Wrong** | 6 errors in 53,940 at 12 dB (T1+T2 fold), none from 15 dB. Classical keeps 9 errors per realisation at ≥12 dB. Fixed. |
| SNR definition (R2 #7) | — | Per-sample ratio at 20 sps over 0–6 kHz, so **Eₛ/N₀ = SNR + 10 dB**. This explains the BPSK ~1e-5 at "0 dB". Added. |
| AGC (R1 §10.1) | **Confirmed**, plus one more issue | The AGC denominator is the *noise-free* record RMS, i.e. idealised. Now stated. |

The equivalence wording ("statistically indistinguishable", "equivalent", "parity",
"matches") is removed throughout. Seed variability is reported (see §8).

## 7. Shuffled-label control (R1 §5). Confirmed. Rerun.

- **Confirmed:** `train.py` permuted labels across train *and* test, on LOBO B1 only, while the caption said LOPO. With test labels shuffled, chance is guaranteed whatever the model does, so the old control could not fail.
- **Rerun:** training labels only are permuted, test labels are left as measured, on all 10 LOPO folds × 5 seeds (50 runs). Clean SER, mean over folds and seeds, with the range of the 5 per-seed fold-means:
  - BPSK: **0.549** [0.376, 0.750]
  - QPSK: **0.710** [0.579, 0.846]
  - 8-QAM: **0.922** [0.875, 0.977]

  Chance is 0.500 / 0.750 / 0.875.
- **New finding: single runs scatter far from chance in both directions,** from 0.169 to 1.000. Each word offers only 8–63 distinct noise-free windows, so a network trained on permuted labels learns an arbitrary but systematic function. It is often a near-constant class, which scores 0.75 or 1.0 on a 6-point 8-QAM word, and occasionally an inverted or partial demodulator. One shuffled run is therefore not a reliable chance reference on this corpus. What the control does establish: the best shuffled run (0.169) is nowhere near the trained receiver (0 clean, ≤ 0.0071 at 6 dB).
- A third, context-only control was added. A lookup from the neighbours' true labels gives 0.4959 / 0.6943 / 0.9001 on withheld words.
- Table X, Fig. 6 and Section VI-C have been rewritten. The "Both must fail, and both do" caption is gone.

## 8. Seeds and splits (R1 §8; R2 #6, Fig. 5). Confirmed. Seeds added.

Two extra seeds were run on every LOPO fold and on every 8-QAM LOBO/random fold, plus 4 extra 8-QAM random partitions. Variability is reported in Tables VII–VIII and Figs. 4–5. Results at 0 dB, 8-QAM:

- Random split, 7 runs: 0.1256 ± 0.0022.
- LOPO, all 5 folds, 3 seeds: 0.1310 ± 0.0008.

**The old claim that the gap was "within run-to-run variation" was not supported as stated.** The 0.0054 gap is larger than the run spread. All of it comes from the T1+T2 fold (0.1570 ± 0.0058). That fold trains on only 4 B1 records and tests on all 5 boards. The four folds that train on 14 records give 0.1245 ± 0.0024, no worse than random at any SNR. The paper now makes exactly this argument. That is the comparison that tests ISI or context exploitation, because the random split offers neighbour cues that withheld words do not.

## 9. Pattern/board confound (R1 §7). Confirmed. Disclosed.

The cross-tab is reproduced exactly: S1, Q2, Q3, T2–T6 are B1 only. This is now stated in Section III-D, the Table IX caption and the Limitations.

**Not raised by either review:** the LOBO gap is driven entirely by board B5
(8-QAM 6 dB: 0.0302 for B5 against 0.0049–0.0060 for the others; 9 dB: 0.0102 against
≤0.0002). The claim that it was "consistent across folds" was false and has
been corrected.

## 10. Numerical table checks (R2 table; R1 §11)

| Item | Verdict | Change |
|---|---|---|
| Table II "smallest" vs Allahham's 3 devices | **Confirmed** | Caption fixed |
| BPSK −64 / QPSK −12 symbols | **Confirmed**: edge trimming (2+2 per BPSK record × 16; 1 per QPSK record × 12) | Explained in V-B; Table IX says windows |
| Table VIII LOBO 6 dB 0.0105 vs Table XI 0.0056–0.0060 | **Confirmed, explained** | The ablation is a single LOBO fold (B2), 18 epochs, one seed. Caption and text say so. |
| "two decades of λ" | **Confirmed wrong** | Now "λ ∈ {0, 0.2, 1.0}" |
| Architecture not reconstructible | **Confirmed** | Layers, strides, lengths and heads are listed. RF and parameter counts verified by hand (20,932 / 22,022 / 20,106). |
| 35 folds | **Correct (R1 §11.2 agrees)** | Per-modulation breakdown added |
| Macro vs micro means; 0.0000 vs no errors | **Confirmed** | Means defined as unweighted; error counts given |
| 43 records / 154,628 symbols / Table VII learned values | **Correct** | — |

## 11. Figures (R2)

| Fig. | Change |
|---|---|
| 1 | **Not fixed (needs a photo edit):** add B1–B5 labels. A visible draft note marks it. |
| 2 | Caption: individual symbols, per-record derotation and unit RMS, 6,000 per panel |
| 3 | Caption: τ and the word period are shared, the label path reads the same resampled waveform, noise is added before the AGC (whose gain comes from the clean record) |
| 4 | Regenerated: matched classical aggregation, 3-seed bars, floor line described as a visual convention, saturation sentence removed |
| 5 | Regenerated: seed/partition ranges, B5-excluded LOBO line, neutral title |
| 6 | Regenerated with the corrected control and run-to-run range. The position-only control now has a cyclic-shift null (T4 is the only fold where the true alignment is best, at 1/8 odds). |
| 7 | Provenance, error count and directionality added (see §4) |
| 8 | Identified as BPSK/PRBS63; 509–626 mV called an observed separation interval; chance reference explained. **Also found:** two of the eleven sub-threshold records sit marginally above the 95th percentile (71.4%, 73.0%). Now stated. |

## 12. Literature (R2 #8)

- RML22 (dataset paper) and Hoydis et al. (vision article) are removed from Table I. All counts are recomputed from the worksheet: 21 systems; 9 baseband / 7 frequency / 4 symbols / 1 passband; carrier given 15; timing given 14; constellation given 18; pulse family 17; simulated 19; random split 11/21.
- Cammerer et al.: the abstract (arXiv 1911.13055) states training "on the actual wireless channel" with SDRs, so it is now **hw-train**, not hw-valid.
- [44], [45] are now "radio map" labels with devices n/a, and the text says so.

## 13. Added after the reviews: positive control (2026-10-31)

Neither review asked for this, but it closes the obvious next objection: the controls show the bounded receiver does not exploit the sequence, yet nothing showed that the evaluation would catch it if it did. The new subsection "Positive control: a longer window" trains the same network three ways: the bounded window; a long window (±4 / ±6 / ±10 symbols); and that long window with the target and its two neighbours blanked. Seed 0, every leave-one-pattern-out fold, three random partitions.

Leakage checks (`window_control.py --audit`):
- No withheld word is a shift or rotation of a training word (apart from T1/T2, already grouped).
- A label lookup from the long context stays at chance on leave-one-pattern-out.
- The blanked span exceeds the filter support.
- All inputs are scored on identical symbols.

Results, 8-QAM at 0 dB:

| Input | Seen word (random) | Withheld word | Withheld, noise-free |
|---|---|---|---|
| Bounded | 0.1231 | 0.1301 | 0 |
| Long | 0.0010 | 0.6221 | 0.5756 |
| Blanked | 0.0052 | 0.8809 | 0.8976 |

QPSK and BPSK show the same pattern. The network memorises once its input allows, even though noise is added throughout training. The random split rewards that by two orders of magnitude, and leave-one-pattern-out exposes it. Under the bounded window, neither signature appears.

**Trace readability (added the same day).** This tests directly whether the 8-QAM network can read a neighbour through the ISI trace.
- **Setup:** the network is trained to name the previous or next symbol's label from its target-only window. There is one network per record, with a time split (75/25), so word, board and positions are the ones it trained on.
- **Scoring:** only at positions where the target's class cannot reveal the neighbour (4 of 8 in T3–T6). There, a predictor that knows the target's class is right 0.501 of the time.
- **Results:**
  - The network scores 0.500 / 0.501 on noise-free windows, and 0.507 / 0.497 when trained noise-free.
  - With the neighbour visible in the window, it scores 0.89–0.93 at 0 dB and ≥0.999 from 15 dB, so the probe works.
- **Physics:** from 0 to 12 dB, where 8-QAM errors occur, the trace is 2–18 dB below the noise per symbol.
- **−6 dB, positive-control runs:** with the bounded window, seen and withheld words agree once the thin folds are left out: 0.408/0.407 for 8-QAM and 0.142/0.143 for QPSK.

Edits made:
- New subsection and table in Section VI.
- "Trace readability" paragraph in VI-C, and the −6 dB sentence in VI-D.
- One sentence in the abstract, which stays at 250 words.
- Introduction and contribution 3.
- Forward reference in V-A.
- Discussion, Limitations and Conclusion.

## 14. Presentation comments (third AI review, 2026-10-31)

| Comment | Change |
|---|---|
| Dense; Sections V–VI hard to follow | Roadmap paragraphs open both sections. VI-E (constellation geometry) and VI-F (the boards' own demodulators) are condensed and marked as side results. The Discussion's hardware paragraph is merged into VI-F, and the carrier paragraph in Limitations is shortened. |
| Abstract overloaded; contributions 4–5 dilute | The abstract is rewritten at 222 words; the only numbers left are the symbol and board counts. Contributions 4 and 5 become a short "side results" paragraph after the list. |
| 2025–2026 references must be verifiable | Every DOI entry was checked against Crossref (title, year, volume, issue, pages) and every arXiv entry against arXiv; all match. Utrilla was checked against DataCite. Morin was checked against Crossref and its DOI added. Proakis (now 5th ed., 2008), Mengali and Golomb were corrected. The Table I codings for the 2025–2026 studies (Tonix-Gleason, Liu) agree with their abstracts and full text. The Shulman preprint is now introduced as a preprint. |
| "We are not aware of prior work…" | Now "Among the learned-receiver studies reviewed here, none…". No other first-or-novel claims were found. |

## 15. Not resolved here

- **Fig. 1 B1–B5 annotations.** Needs a photo edit.
- **Reference verification** against IEEE Xplore (R1 §12.5). Still outstanding, as noted in the file header.
- **AI-use disclosure and WorldCUR prior-dissemination disclosure** (R1 §13–14). Visible draft notes are in the Acknowledgment. This revision itself was drafted with Claude, which has to be disclosed.
- **Author names.** The biographies now match the author line ("Mohammad Khalid Hasan Nabil", "Md. Maynul Islam"). Confirm these are the forms you want on ORCID and in the portal.
- **Reproducibility package** (R1 §2.2, Phase 5): requirements file, README, checkpoints, split manifests. The revision scripts are now in `src/`, but these items are still missing.
- **New captures** (crossed pattern × board design, R1 §7.4). This needs bench time.
- **`../paper-ieeetran/`** has not been updated to match.
