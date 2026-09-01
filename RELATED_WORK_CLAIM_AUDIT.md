# Related Work — every claim and the cell it came from

Written from `related_work_worksheet - related_work_worksheet.csv` on 2026-09-01,
per `worksheet_rules.md`. Rows with `read_status != DONE` were not used. A24 is
`DONE` but every axis reads `not-stated`, so it is cited in prose and left out of
Table I.

## Two claims have no cell

Both come from the instructions in `worksheet_rules.md`, not from the worksheet.
They are claims about the absence of prior work, which no cell can carry.

| Claim | Where | Source |
|---|---|---|
| "none constrains the receiver's input context relative to the complexity of the transmitted sequence" | end of II-A | `worksheet_rules.md`, II-A: "Close by noting that these systems are evaluated on held-out partitions without a structural bound on context." |
| "We are not aware of prior work in learned communications that constrains receiver context below the linear complexity of the test sequence" | end of II-B | `worksheet_rules.md`, II-B: "no prior work in learned communications bounds receiver context below the linear complexity of the test sequence" |

Both are hedged as written ("none reports", "we are not aware of"). Neither can
be checked from the worksheet; they rest on your reading of the 44 papers.

## One claim in the rules that the worksheet does not support — omitted

`worksheet_rules.md` asks II-B to say that "one wireless dataset built fixed and
random payload variants apparently for the same reason (B02)". B02's cells do not
record that. `my_one_line` describes a CNN trained on preambles from SDR testbeds
for transmitter identification, and `page_evidence` covers the axis values only.
**Not written.** B02 appears in Table II via its II-C row (C06) instead.

## II-A — Learned physical-layer processing

| Claim | Cell |
|---|---|
| "twenty-three recent learned receivers and demodulators" | II-A rows with axis values (24 rows less A24) |
| "Ten operate on baseband in-phase and quadrature samples" | `input_domain = baseband-IQ` × 10 |
| "eight on frequency-domain subcarriers" | `input_domain = frequency-domain` × 8 |
| "four take symbols already at the output of a matched filter" | `input_domain = post-matched-filter-symbols` × 4 |
| "one operates on passband samples" | `input_domain = passband` × 1 (A09) |
| "Seventeen are given the carrier frequency" | `carrier_supplied = yes` × 17 |
| "sixteen the symbol timing" | `timing_supplied = yes` × 16 |
| "Twenty are given the constellation" | `constellation_supplied = yes` × 20 |
| "nineteen fix a known pulse-shape family without presenting it to the network" | `pulse_shape_supplied = fixed-family` × 19, wording from the rules' definition |
| "Twenty-one are evaluated on simulated waveforms" | `data_source = simulated` × 21 |
| "the two exceptions use hardware captures for validation only" | `data_source = hardware-validation-only` × 2 (A03, A21) |
| Zhang: sliding-window 1-D CNN on raw passband, no explicit carrier sync or symbol-period grouping | A09 `my_one_line` |
| "closest to ours in input domain" | A09 and C13 are the only `passband` rows |
| "differs in being evaluated on simulated waveforms and in being given the constellation" | A09 `data_source = simulated`, `constellation_supplied = yes`; C13 `hardware-primary`, `no` |
| Xiao: blind time/frequency compensation, joint modulation type and symbol index | A04 `my_one_line` |
| Dakic: LoRa symbols from baseband, robust to emulated offsets, no explicit sync modules | A10 `my_one_line` |
| Zhu: bit sequences from oversampled baseband, blind timing via shifted binary cross-entropy | A02 `my_one_line` |
| Hanna: carrier offset, timing and symbol rate blind, then minimum-distance decoding | A03 `my_one_line` |
| "one of the two entries whose evaluation involves hardware" | A03 `data_source = hardware-validation-only` |
| DeepRx: joint channel estimation, equalisation, soft demapping, LLRs for 5G | A19 `my_one_line` |
| MIMO extension: combining-based or fully learned transformation layer | A20 `my_one_line` |
| Cammerer: bit-wise autoencoder, geometric shaping and bit labelling with iterative demapping | A21 `my_one_line` |
| "one of the three tabulated systems not given a constellation" | `constellation_supplied = no` × 3 (A01, A04, A21) |
| "A further group learns the demapping alone, from symbols already at the matched-filter output" | A11–A14 `input_domain = post-matched-filter-symbols` |
| "Twelve of the twenty-three report a random held-out split" | `evaluation_split = held-out-random` × 12 |
| "the remaining eleven do not state one" | `evaluation_split = not-stated` × 11 |
| "None reports a partition grouped by recording or by device" | no II-A row carries a `grouped-by-*` value |
| "synthetic datasets used to benchmark it" | A23 `my_one_line` (RML22 corrects flaws in prior synthetic datasets) |
| "Broad reviews of the area" | A22, B03, B04 `my_one_line` (overview / tutorial / review) |
| "of distribution shift specifically" | A24 `my_one_line` (tutorial and survey on domain generalization) |

Table I values are generated directly from the axis columns — no value was typed
by hand.

## II-B — Evaluation validity and memorisation

| Claim | Cell |
|---|---|
| Shulman: segment-level cross-validation causes leakage and optimism in drone benchmarks; recording-grouped evaluation collapses the inflated accuracies | B01 `my_one_line` |
| "the only one that reports a recording-grouped split" | B01 is the only row with `evaluation_split = grouped-by-recording` |
| Brookshire: segmenting recordings without grouping by subject causes leakage and inflated performance in translational EEG | B07 `my_one_line` |
| Carlini: quantitative framework for testing unintended memorisation of rare training sequences | B05 `my_one_line` |
| Arpit: gradient-based optimisation learns simple patterns before memorising noise | B06 `my_one_line` |

B03 and B04 are surveys; they are cited in II-A rather than padding II-B.

## II-C — Hardware-captured RF datasets

| Claim | Cell |
|---|---|
| "twelve public radio-frequency datasets" | II-C rows less C13 |
| "Nine captured primarily from hardware, one uses hardware for validation only, and two are simulated" | `data_source` counts: 9 / 1 / 2 |
| "Seven label at device level, three at modulation-class level, and two do not state a granularity" | `label_granularity` counts: 7 / 3 / 2 |
| "None labels individual symbols" | no II-C row other than C13 has `label_granularity = symbol` |
| WiSig: ten million packets, 174 transmitters, 41 receivers, over a month | C01 `my_one_line`, `devices = 174` |
| Elmaghbub: 25 LoRa transceivers, indoor, outdoor and wired testbeds | C04 `my_one_line`, `devices = 25` |
| Zahid: 27 radios | C05 `devices = 27` |
| Morin: twenty | C06 `devices = 20` |
| Kashani: thirteen | C09 `devices = 13` |
| Jagannath: ten commercial chipsets | C10 `my_one_line`, `devices = 10` |
| "Our five units are fewer than any of these, and far fewer than the 174" | C13 `devices = 5` against C01 `devices = 174` |
| closing paragraph on what a label denotes | C13 `label_granularity = symbol` against the device and modulation-class rows |

## Worksheet issues found while writing

| Issue | Row | Action |
|---|---|---|
| BibTeX entry had no `year` field | A02 `Zhu2022EndtoEnd` | fixed: `year = {2022}`, verified against Crossref 10.1049/cmu2.12424 (IET Communications 16(11) 1315-1324, issued 2022-05-21) |
| `evaluation_split` holds two fused values | C13 | not used by either table; repair for your records |
| `read_status` blank | C13 | used anyway, as the rules direct |
| CSV export doubled the quotes in one entry | B02 / C06 | undone on import, content unchanged |
| Same paper in two sections | B02 and C06 | one bib entry, one table row (II-C) |
| `carrier_supplied` has a trailing space | A07 | normalised on read |

## Build fixes applied outside the worksheet

- `refs.bib`: `IEEE Communications Surveys & Tutorials` → `\&`. LaTeX reads a bare
  `&` as a column separator, which halted the build. The journal name is unchanged.
- `manuscript.tex`: added `\usepackage[T1]{fontenc}`, needed for the ogonek in
  Jastrz\k{e}bski (B06). Without it LaTeX errors with "Command \k unavailable in
  encoding OT1".
