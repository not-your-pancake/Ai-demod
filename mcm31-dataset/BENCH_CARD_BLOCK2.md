# BENCH CARD — Block 2, board B1 only

**8 files, about 50 minutes.** Same board, same bench settings, same workflow as
Block 1. Nothing here is new except the DIP words.

**Fixed for every file in this block:**

| Item | Setting |
|---|---|
| Attenuation | **fully counter-clockwise (A0)** — never moved in this block |
| Noise knob | fully counter-clockwise, untouched |
| Time base | 500 ms/div, Max Sa Rate |
| CH1 / CH2 / trigger | TP-20 / TP-9 / EXT from TP-1 |

Only TP-20 matters here, so a word that breaks TP-9 is fine.

---

## Do them in this order — it minimises switch changes

### Group A — 8-QAM  (SW3 = 24 BIT, J1 = TRIBIT, SW8 = DI/TRIBIT)

| # | Save as | Word | BANK 1 (left) | BANK 2 (middle) | BANK 3 (right) |
|---|---|---|---|---|---|
| 1 | `B1_8QAM_T2.CSV` | T2 | `1 1 1 0 0 0 1 1` | `0 0 0 1 1 0 1 0` | `1 0 1 0 0 0 1 1` |
| 2 | `B1_8QAM_T3.CSV` | T3 | `0 1 0 1 0 1 0 0` | `0 1 1 1 0 1 1 1` | `0 0 1 1 0 0 0 1` |
| 3 | `B1_8QAM_T4.CSV` | **T4** | `0 0 0 1 0 0 1 0` | `1 1 1 0 1 1 1 0` | `1 0 0 0 1 0 1 1` |
| 4 | `B1_8QAM_T5.CSV` | **T5** | `1 1 0 0 1 1 0 0` | `1 0 0 0 1 0 1 1` | `0 0 0 1 0 1 1 1` |
| 5 | `B1_8QAM_T6.CSV` | **T6** | `0 1 1 0 1 0 1 1` | `0 1 0 1 0 0 1 1` | `0 0 1 1 1 0 0 0` |

T4, T5, T6 are the new words. All three cover all 8 constellation points, have
max bit run 3, and are balanced 12 ones / 12 zeros — the same properties the
handbook requires of T1–T3.

### Group B — QPSK  (SW3 = 24 BIT, **J1 = DIBIT**, SW8 = DI/TRIBIT)

| # | Save as | Word | BANK 1 (left) | BANK 2 (middle) | BANK 3 (right) |
|---|---|---|---|---|---|
| 6 | `B1_QPSK_Q2.CSV` | Q2 | `0 0 1 0 0 1 1 1` | `1 0 0 0 1 1 0 1` | `0 1 1 1 0 0 1 0` |
| 7 | `B1_QPSK_Q3.CSV` | Q3 | `1 1 0 1 0 0 1 0` | `1 1 0 0 0 1 1 0` | `0 0 1 1 1 0 0 1` |

### Group C — BPSK  ⚠️ THE TRAP

| # | Save as | Word | BANK 1 (left) | BANK 2 (middle) | BANK 3 (right) |
|---|---|---|---|---|---|
| 8 | `B1_BPSK_S1.CSV` | S1 | `0 1 1 0 0 0 1 0` | `1 0 1 0 1 1 1 1` | `0 0 0 0 1 1 1 0` |

🔴 **File 8 uses SW3 = 24 BIT, J1 = BIT, SW8 = BIT.**
This is the **only** BPSK file in the whole campaign that uses 24 BIT.

🔴 **When you are finished, put SW3 back to 64 BIT.**
Leaving it on 24 BIT is exactly what produced two bare-carrier files earlier.

---

## Every file, every time

1. Set the switches
2. **Press START**
3. Look at CH1 — it must be **messy**, not a clean smooth sine
4. Re-range CH1 if needed (attenuation does not move in this block, but the
   modulation change between groups does change the signal size)
5. Save
6. Run the bench check before moving on:

```powershell
& "E:\Ai demod\.venv\Scripts\python.exe" "E:\Ai demod\mcm31-dataset\src\benchcheck.py" "E:\Ai demod\mcm31-dataset\raw\B1\B1_8QAM_T4.CSV"
```

All six lines green and `>>> GOOD` before you touch the next switch.

---

## Why the three extra 8-QAM words are worth 20 minutes

The network's receptive field is capped at **1 symbol** for 8-QAM, so the only
way neighbouring symbols reach it is through inter-symbol interference. What a
new word buys is each constellation point sitting beside neighbours it has not
sat beside before.

| | Symbol transitions covered | Leave-one-pattern-out |
|---|---|---|
| T1 + T2 + T3 | **20 of 64** | 3 folds |
| plus T4 + T5 + T6 | **43 of 64** | **6 folds** |

Also worth knowing: **T2 is exactly the bitwise complement of T1**, so the
existing three words are less independent than they look. T4–T6 were selected
to exclude every rotation and complement of the existing set.

QPSK was checked the same way and needs nothing: Q1 + Q2 + Q3 already cover
**14 of 16** transitions. Adding more would buy 2 transitions. Not worth it.
