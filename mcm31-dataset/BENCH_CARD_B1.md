# BENCH CARD — Board B1, Session 1

Open this on your phone at the bench. Work top to bottom. Do not skip.

**Goal of this session:** 7 good files from board B1. Nothing else.
**Stop rule:** Do NOT start board B2 until I have checked B1's files in software.

---

## PART A — Wire it up (do once)

You need **three** connections, not two. People forget the trigger.

| # | Connect | To | Note |
|---|---|---|---|
| 1 | Scope **CH1** probe | **TP-20** | the modulated signal — this is what the AI will learn from |
| 2 | Scope **CH2** probe | **TP-9** | board's own recovered data — comparison only |
| 3 | Scope **EXT TRIG** input | **TP-1** | ← easy to forget |
| 4 | All ground clips | board ground | bad ground = fake noise in the data |

---

## PART B — Scope settings (set once, never change again)

| Setting | Value |
|---|---|
| Time base | **500 ms/div** |
| Acquire mode | **Max Sa Rate** |
| Trigger source | **External** |
| Trigger level | **2.5 V** |
| Trigger slope | **Rising** |
| Channels on | CH1 **and** CH2 (both must be ON) |

⚠️ **500 ms/div, not 1 ms/div and not 1 s/div.** A wrong time base was already tested and gave only 7 bits of data. The whole file becomes rubbish.

---

## PART C — Board switches (set once, never change again)

| Item | Setting |
|---|---|
| **Noise knob** | **Fully counter-clockwise, until it physically stops.** Then never touch it again, all session, all 8 boards. |
| SW1 | **TTL** |
| SW2 | **NORMAL** (not DIFFERENTIAL) |
| SW6 | **PSK / QPSK / QAM** |
| J3 | **PSK/Q** position |
| J4 | **link fitted** |
| SW7, J6 | leave as-found — but **photograph them** (see below) |

### Trimmers — hands off

Do **not** turn the **PHASE** trimmer. Do not try to "tune" the board to make it look nicer on screen. Where the trimmers sit right now is exactly the board-to-board difference we are trying to measure. Turning one destroys the measurement.

> **Attenuation — read carefully.** There is a control you *will* turn (the attenuation knob, three positions in Part F) and there may be a small screwdriver **trimmer** you must **not** turn. Before you touch anything, photograph the attenuation control in its as-found position. If it turns out to be a screwdriver preset rather than a knob, message me and we will decide together — do not guess.

---

## PART D — Photographs (B1 is the reference board for all others)

Take these **before** you change anything:

1. `B1_panel_asfound.jpg` — the whole switch panel, sharp, well lit, readable
2. Close-up of **SW7**
3. Close-up of **J6**
4. Close-up of the **attenuation control** in its as-found position
5. Close-up of the **noise knob** at its counter-clockwise stop

Then after you set everything in Part C:

6. `B1_panel_asset.jpg` — the whole panel again, as set

**These photos are how boards B2–B8 get matched to B1.** Without them the cross-device comparison is meaningless. Put them in `E:\Ai demod\mcm31-dataset\photos\`.

---

## PART E — The START button test (30 seconds, MANDATORY)

The DIP switches are **latched**. Flipping a switch changes nothing until you press **START**. This has already ruined one capture: two "different" patterns turned out to be the identical waveform (cross-correlation +0.9999).

Do this now, in **QPSK** mode:

| Step | Do this | You must see |
|---|---|---|
| 1 | Set QPSK mode: **SW3 = 24 BIT**, **J1 = DIBIT**, **SW8 = DI/TRIBIT**. Attenuation fully CCW. Look at CH1 (TP-20). | a modulated (messy-looking) trace |
| 2 | Turn **all 24 DIP switches OFF**. Press **START**. | the trace **collapses into a clean, smooth sine wave** |
| 3 | Set pattern **Q1** (map below). Press **START**. | the trace becomes messy/modulated again |

**If step 2 does not visibly change the screen, the load failed.** Press START again. If still nothing: press RESET. If still nothing: power-cycle the board. Repeat until step 2 works. Do not carry on until it works.

### Q1 switch map — `110010000101101011011110`

Three 8-way banks, **left to right**, and within each bank **left switch = first bit**:

```
BANK 1 (left)    1 1 0 0 1 0 0 0
BANK 2 (middle)  0 1 0 1 1 0 1 0
BANK 3 (right)   1 1 0 1 1 1 1 0
```

`1` = ON/up, `0` = OFF/down.

> Don't worry if your board's ON/OFF is the other way round — the handbook says the switch-to-symbol mapping is unknown anyway, and I get the real labels from the waveform itself. **What matters is that you use the same convention on all 8 boards.** Pick one now and stick to it.

### 🔴 From here on: press **START** after EVERY switch change. Every single time.

---

## PART F — The 7 files

Attenuation has exactly three positions. **Do not hunt for a "good" position by eye** — you physically cannot see the failure point on a scope.

- **A0** = attenuation knob fully **counter-clockwise**, at the hard stop
- **AMID** = roughly **halfway** between the two stops → **photograph the slot angle**
- **AMAX** = fully **clockwise**, at the hard stop

| # | Save as | Mod | SW3 | J1 | SW8 | Pattern | Attenuation |
|---|---|---|---|---|---|---|---|
| 1 | `B1_BPSK_A0.CSV` | BPSK | 64 BIT | BIT | BIT | on-board PRBS | **A0** |
| 2 | `B1_BPSK_AMID.CSV` | BPSK | 64 BIT | BIT | BIT | on-board PRBS | **AMID** 📷 |
| 3 | `B1_BPSK_AMAX.CSV` | BPSK | 64 BIT | BIT | BIT | on-board PRBS | **AMAX** 📷 |
| 4 | `B1_QPSK_A0.CSV` | QPSK | 24 BIT | DIBIT | DI/TRIBIT | **Q1** + START | **A0** |
| 5 | `B1_QPSK_AMAX.CSV` | QPSK | 24 BIT | DIBIT | DI/TRIBIT | **Q1** | **AMAX** |
| 6 | `B1_8QAM_A0.CSV` | 8-QAM | 24 BIT | TRIBIT | DI/TRIBIT | **T1** + START | **A0** |
| 7 | `B1_8QAM_AMAX.CSV` | 8-QAM | 24 BIT | TRIBIT | DI/TRIBIT | **T1** | **AMAX** |

Save into `E:\Ai demod\mcm31-dataset\raw\B1\`. Use these exact filenames, capital `.CSV`.

### T1 switch map — `000111001110010101011100`

```
BANK 1 (left)    0 0 0 1 1 1 0 0
BANK 2 (middle)  1 1 1 0 0 1 0 1
BANK 3 (right)   0 1 0 1 1 1 0 0
```

---

## PART G — Before you press SAVE, every single time

- [ ] Both CH1 and CH2 are switched ON (otherwise you get 2 columns instead of 3 and the file is dead)
- [ ] Time base still **500 ms/div**
- [ ] **CH1 fills most of the screen height but does NOT touch the top or bottom edge.** Too small = not enough voltage steps, file rejected. Clipped = waveform destroyed, file rejected.
- [ ] **CH1 looks MESSY, not like a clean smooth sine wave.** A clean sine means no data is reaching the modulator — see the warning below.
- [ ] Filename is exactly right
- [ ] Row added to the logbook

### 🔴 Re-range CH1 after EVERY change — modulation *and* attenuation

Turning the attenuation knob shrinks the wave. If you don't re-range, the scope
describes it with too few voltage steps and the file is rejected. This already
cost one set of recaptures: all three A0 files passed, and all four attenuated
files failed.

### 🔴 A clean smooth sine wave means the board is sending NOTHING

This is the one failure you *can* see by eye. It looks exactly like step 2 of
the START test. If CH1 goes smooth and regular:

- **BPSK:** SW3 must be **64 BIT**, J1 = **BIT**, SW8 = **BIT**
- **QPSK / 8-QAM:** SW3 must be **24 BIT**, and the DIP word must not be all-OFF
- Then **press START** and look again

This already cost two BPSK files that came back as a bare carrier with zero data.

---

## PART H — Check EVERY file before you move on

Run this on the laptop after saving each file. It takes about 10 seconds and
catches every mistake that has actually happened on this project:

```powershell
& "E:\Ai demod\.venv\Scripts\python.exe" "E:\Ai demod\mcm31-dataset\src\benchcheck.py" "E:\Ai demod\mcm31-dataset\raw\B1\B1_BPSK_A0.CSV"
```

You want all six lines green and `>>> GOOD`:

```
  ok   columns                3
  ok   rows                   523,600
  ok   CH1 levels             143  (pk-pk 2.84 V)
  ok   carrier                1199.92 Hz
  ok   modulation depth       96.6% in sidebands     <- the important one
  ok   pattern fingerprint    [31, 32] -> ['PRBS63']
```

If anything says `BAD`, it tells you exactly what to fix. **Recapture there and
then** — it is 2 minutes at the bench and a whole trip back if you find out later.

Do not leave the lab until all 7 files say GOOD.

---

## PART I — Logbook

Open `E:\Ai demod\mcm31-dataset\logbook.csv` and add one line per file, at the bench, while you remember.

---

## PART J — End of session

1. Copy all 7 CSVs into `E:\Ai demod\mcm31-dataset\raw\B1\`
2. Copy all photos into `E:\Ai demod\mcm31-dataset\photos\`
3. **Message me.** I run the full 10-check verification and build the ground-truth pipeline on B1.
4. **Wait for my green light before starting board B2.**

Why wait? If something in the setup is systematically wrong, I will see it in B1's data. Better to lose 45 minutes than to discover it after 6 hours and 8 boards.

---

## Quick reminders

| Trap | What happens | Prevention |
|---|---|---|
| **CH1 goes clean and smooth** | **Bare carrier — the file contains no data at all** | **SW3 / J1 / SW8 for the right modulation, then START** |
| **Turned attenuation, didn't re-range** | Too few voltage levels, file rejected | Re-range CH1 after every knob move |
| Forgot START | Two "different" patterns are secretly identical | Press START after every switch change |
| CH2 off | File has 2 columns, unusable | Check both channels ON before saving |
| Wrong time base | Only a few bits captured | 500 ms/div, always |
| CH1 too small | Not enough voltage levels | Fill the screen, don't clip |
| Touched the noise knob | Breaks comparison across boards | Fully CCW, then never again |
| Turned the PHASE trimmer | Destroys the cross-device measurement | Never turn it |
| TP-9 "looks perfect" on screen | It can look perfect while being completely broken | **Never judge TP-9 by eye. Ever.** |
