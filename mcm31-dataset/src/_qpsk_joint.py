import os, sys, itertools
import numpy as np
sys.path.insert(0, ".")
import mcm31 as M

R = r"E:\Ai demod\mcm31-dataset\raw\B1"
files = {"Q1":"B1_QPSK_A0.CSV","Q2":"B1_QPSK_Q2.CSV","Q3":"B1_QPSK_Q3.CSV"}
obs={}
for w,f in files.items():
    cap=M.load_capture(os.path.join(R,f))
    P=M.refine_period(cap.ch1, M.word_period_seconds(w,"QPSK")*cap.fs)
    s=M.derotate(M.extract_symbols(M.coherent_template(cap.ch1,P).values,12),"QPSK")
    obs[w]=(np.floor((np.angle(s)%(2*np.pi))/(np.pi/2)).astype(int)%4)
    print(f"{w}: observed quadrants {obs[w].tolist()}")

def dibits(word,k):
    """word's bits rotated by k, grouped into dibits. k covers both the
    symbol rotation (even k) and the framing offset (odd k)."""
    b="".join(map(str,M.bits_of(word).tolist())); b=b[k:]+b[:k]
    return np.array([int(b[i:i+2],2) for i in range(0,24,2)])

def fit(pi, points, word):
    mapped=np.array([pi[int(p)] for p in points])
    best=(-1.0,None)
    for k in range(24):
        sc=float((dibits(word,k)==mapped).mean())
        if sc>best[0]: best=(sc,k)
    return best

best=None
for perm in itertools.permutations(range(4)):
    pi=dict(enumerate(perm))
    res={w:fit(pi,obs[w],w) for w in files}
    tot=sum(r[0] for r in res.values())
    if best is None or tot>best[0]: best=(tot,pi,res)
tot,pi,res=best

print(f"\nONE SHARED BIJECTION across Q1, Q2, Q3")
print(f"  quadrant -> dibit : {dict(sorted(pi.items()))}")
print(f"  {'word':>4} {'fit':>7} {'bit-rot':>8} {'framing':>8}")
for w in files:
    sc,k=res[w]
    print(f"  {w:>4} {sc*100:>6.1f}% {k:>8} {'offset' if k%2 else 'aligned':>8}")
print(f"  mean over 3 words : {tot/3*100:.1f}%")

rng=np.random.default_rng(0); null=[]
for _ in range(20000):
    sh={w:rng.permutation(obs[w]) for w in files}
    null.append(max(sum(fit(dict(enumerate(p)),sh[w],w)[0] for w in files)
                    for p in itertools.permutations(range(4))))
null=np.array(null)/3
print(f"\nCHANCE CONTROL (quadrant sequences shuffled, n=20,000)")
print(f"  mean {null.mean()*100:.1f}%   p95 {np.percentile(null,95)*100:.1f}%   "
      f"best ever {null.max()*100:.1f}%")
print(f"  p-value for {tot/3*100:.1f}% : {float((null>=tot/3).mean()):.5f}")

def ctx(s): return {(s[i-1],s[i]) for i in range(len(s))}
print("\nMEASURED QPSK coverage")
for n in (1,2,3):
    ws=list(files)[:n]
    print(f"  {'+'.join(ws):<12} {len(set().union(*(set(obs[w].tolist()) for w in ws))):>2}/4 points"
          f"   {len(set().union(*(ctx(obs[w].tolist()) for w in ws))):>3}/16 transitions")
