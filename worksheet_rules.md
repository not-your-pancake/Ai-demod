# How to fill the worksheet, and how Claude Code writes from it

44 rows. 24 for II-A, 7 for II-B, 13 for II-C. Two rows are already filled as
worked examples: A03 (the DPN paper, filled from the arXiv full text) and C13
(your own work).

## Rule 1 — you fill the cells, not a model

Every axis cell must come from a PDF you opened. Not from an abstract, not
from Elicit, not from Gemini, not from me. If you have not read it, the cell
stays `TODO` and the paper does not get a row in the published table.

## Rule 2 — controlled vocabulary only

Do not write sentences in the axis columns. Pick one value.

| Column | Allowed values |
|---|---|
| `read_status` | `TODO` / `DONE` / `NO-ACCESS` / `DROPPED` |
| `carrier_supplied` | `yes` / `no` / `partial-residual-offset-estimated` |
| `timing_supplied` | `yes` / `no-recovered-blind` |
| `pulse_shape_supplied` | `yes` / `no` / `fixed-family` |
| `constellation_supplied` | `yes` / `no` |
| `input_domain` | `passband` / `baseband-IQ` / `post-matched-filter-symbols` / `frequency-domain` |
| `data_source` | `simulated` / `hardware-primary` / `hardware-validation-only` |
| `devices` | an integer, or `n/a` |
| `label_granularity` | `symbol` / `modulation-class` / `device` / `protocol` |
| `evaluation_split` | `held-out-random` / `grouped-by-recording` / `grouped-by-device` / `not-stated` |

`fixed-family` means the paper fixes a known pulse shape or a small known set
of them, without feeding it to the network. That is the case A03 turned out to
be, and it is the distinction two AI summaries got wrong.

If a paper genuinely does not state something after you have read the methods,
write `not-stated`. That is different from `no`. Do not collapse the two.

## Rule 3 — two free-text columns, and only two

- `my_one_line` — one sentence, your words, what the system does.
- `page_evidence` — where you found the axis values. Section number is enough.

`page_evidence` is the important one. It is what lets you defend any row if a
reviewer challenges it, six months from now when you have forgotten.

## Rule 4 — add the bibkey yourself

`bibkey` stays empty until you have the verified BibTeX entry. A row with a
`TODO` bibkey does not get cited.

---

# Instructions for Claude Code

Write Sections II-A, II-B and II-C of the manuscript from
`related_work_worksheet.csv`.

## Hard constraints

1. **The worksheet is the only source.** Do not use the Elicit CSVs. Do not
   use your own knowledge of any paper in the list. Do not search.
2. **Ignore every row where `read_status` is not `DONE`.** They are not
   evidence yet.
3. **Never write a claim that is not a cell value.** Every factual sentence
   about another paper must trace to a cell. If you want to say something the
   worksheet does not contain, stop and ask.
4. **`not-stated` means say nothing.** Do not convert `not-stated` into "does
   not use", "does not require", or "is not given". This exact conversion has
   already produced two false claims in this project.
5. **Do not cite a row whose `bibkey` is empty.**

## What to produce

**II-A.** Two to three paragraphs plus a table. The table columns are
`carrier_supplied`, `timing_supplied`, `pulse_shape_supplied`,
`constellation_supplied`, `input_domain`, `data_source`. Include the C13 row so
this work appears in the same table. Discuss five or six systems in prose; the
rest are table rows and a grouped citation. Close by noting that these systems
are evaluated on held-out partitions without a structural bound on context.

**II-B.** One or two paragraphs. The honest position: one wireless paper
demonstrates recording-level leakage in RF sensing (B01), one wireless dataset
built fixed and random payload variants apparently for the same reason (B02),
the concern is long-standing in machine learning generally (B05, B06), and no
prior work in learned communications bounds receiver context below the linear
complexity of the test sequence. Do not pad this section. If the worksheet does
not support a sentence, leave it out.

**II-C.** One or two paragraphs plus a table. Table columns: `devices`,
`label_granularity`, `data_source`. **The primary axis is label granularity,
not ground-truth provenance.** Every dataset in the list labels at device,
modulation-class or protocol level. None labels individual symbols. That is the
gap. Include the C13 row. Report the device count honestly; WiSig has far more
devices than this work.

## Style

Match the existing manuscript: IEEE Access, British spelling, plain declarative
sentences, no hype. The carrier here is 1199.91 Hz, which is audio band. Use
"RF" only when describing other people's work, never for this work.

## When you are done

List every claim you wrote alongside the cell it came from. One line each.
If any claim has no cell, say so instead of removing it quietly.
