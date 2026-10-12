"""Summarise search_index logs in logs/ into markdown tables.

Usage (from the repository root): python3 scripts/summarize_search_logs.py
Reads logs/search_<config>_rep<k>.log, prints per-config tables (median latency
over repeats) and the cheapest setting reaching each recall target, and writes
logs/summary.json.
"""
import re,glob,statistics as st,collections
rows=collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob('logs/search_*_rep*.log'):
    cfg=re.search(r'search_(.+)_rep\d',f).group(1)
    for line in open(f):
        m=re.match(r'\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$',line)
        if m:
            L,r,c,a,p=m.groups(); rows[cfg][int(L)].append((float(r),float(c),float(a),float(p)))
out={}
for cfg in sorted(rows):
    print(f"\n### {cfg}  (reps: {len(next(iter(rows[cfg].values())))})")
    print("| L | Recall@10 | dist cmps | mean us (median of reps) | P99 us (median) | QPS/core |")
    print("|---|---|---|---|---|---|")
    out[cfg]={}
    for L in sorted(rows[cfg]):
        v=rows[cfg][L]; r=v[0][0]; c=v[0][1]; a=st.median(x[2] for x in v); p=st.median(x[3] for x in v)
        out[cfg][L]=(r,c,a,p)
        print(f"| {L} | {r:.4f} | {c:.0f} | {a:.0f} | {p:.0f} | {1e6/a:.0f} |")
import json; json.dump({k:{str(L):v for L,v in d.items()} for k,d in out.items()},open('logs/summary.json','w'))
# equal-recall comparison: min latency config reaching target recall
def best(cfg,t):
    c=[(v[2],L,v) for L,v in out.get(cfg,{}).items() if v[0]>=t]
    return min(c) if c else None
print()
for t in (0.95,0.98,0.99,0.995):
    line=[f"target {t}:"]
    for cfg in sorted(out):
        b=best(cfg,t)
        line.append(f"{cfg}: " + (f"L={b[1]} {b[0]:.0f}us p99 {b[2][3]:.0f} rec {b[2][0]:.4f}" if b else "n/a"))
    print(" | ".join(line))
