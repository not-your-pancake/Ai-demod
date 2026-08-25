"""Summarise every fold and control into the tables the paper needs."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0,".")
import mcm31 as M
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d=pd.read_csv(os.path.join(ROOT,"results","experiments.csv"))
cl=pd.read_csv(os.path.join(ROOT,"results","classical_snr.csv"))
SNR=["snr0","snr3","snr6","snr9","snr12","snr15","snr20","snr25","snr30","clean"]
HDR=["0dB","3dB","6dB","9dB","12","15","20","25","30","clean"]
main=d[d.control=="none"]

for mod in ("BPSK","QPSK","8QAM"):
    ch=1-1/M.n_classes(mod)
    print(f"\n{'='*94}\n{mod}   (chance SER {ch:.3f})\n{'='*94}")
    print(f"  {'split':<28} {'folds':>5} " + "".join(f"{h:>7}" for h in HDR))
    print("  "+"-"*92)
    for sp in ("leave_one_pattern_out","leave_one_board_out",
               "leave_one_attenuation_out","random"):
        r=main[(main.modulation==mod)&(main.split==sp)]
        if not len(r): continue
        tag = sp + ("  <-INFLATED" if sp=="random" else "")
        print(f"  {tag:<28} {len(r):>5} " + "".join(f"{r[c].mean():>7.4f}" for c in SNR))
    c=cl[cl.modulation==mod].iloc[0]
    print(f"  {'blind CLASSICAL receiver':<28} {'-':>5} " + "".join(f"{c[s]:>7.4f}" for s in SNR))

print(f"\n\n{'='*94}\nCONTROLS THAT MUST FAIL\n{'='*94}")
print(f"  {'modulation':<12} {'shuffled-label SER':>20} {'chance':>9}  verdict")
print("  "+"-"*60)
for mod in ("BPSK","QPSK","8QAM"):
    s=d[(d.control=="shuffled_labels")&(d.modulation==mod)]
    ch=1-1/M.n_classes(mod)
    v=float(s["clean"].iloc[0])
    print(f"  {mod:<12} {v:>20.4f} {ch:>9.3f}  "
          f"{'PASS - at chance, no leak' if v>ch-0.05 else 'LEAK'}")

print(f"\n  {'modulation':<12} {'split':<28} {'position-only SER':>18} {'chance':>9}")
print("  "+"-"*70)
for mod in ("BPSK","QPSK","8QAM"):
    ch=1-1/M.n_classes(mod)
    for sp in ("leave_one_pattern_out","leave_one_board_out","random"):
        r=main[(main.modulation==mod)&(main.split==sp)]
        if not len(r): continue
        print(f"  {mod:<12} {sp:<28} {r.position_only_ser.mean():>18.4f} {ch:>9.3f}")

print(f"\n\n{'='*94}\nPER-FOLD DETAIL - leave-one-pattern-out (the primary result)\n{'='*94}")
print(f"  {'mod':<6} {'fold':<12} {'train caps':>10} {'test sym':>9} "
      f"{'snr0':>8} {'snr6':>8} {'clean':>8} {'pos-only':>9}")
print("  "+"-"*76)
for mod in ("BPSK","QPSK","8QAM"):
    for _,r in main[(main.modulation==mod)&(main.split=="leave_one_pattern_out")].iterrows():
        thin = "  <- thin" if r.n_train_captures<=2 else ""
        print(f"  {mod:<6} {r.fold.replace('group=',''):<12} {r.n_train_captures:>10} "
              f"{r.n_test:>9,} {r.snr0:>8.4f} {r.snr6:>8.4f} {r.clean:>8.4f} "
              f"{r.position_only_ser:>9.4f}{thin}")
