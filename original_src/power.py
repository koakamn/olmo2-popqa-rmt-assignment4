"""Sample-size / precision calculation by simulation. The discordance rate d (share of items on which two stages
give different correctness) is NOT known yet, so a range is explored; results are planning values, not findings."""
import numpy as np
from scipy.stats import binomtest
rng = np.random.default_rng(42)

def power(n, delta, d, sims=1500):
    hits = 0
    for _ in range(sims):
        nd = rng.binomial(n, d)                       # discordant pairs
        n01 = rng.binomial(nd, 0.5 + delta * n / (2 * max(nd, 1)) if nd else 0.5)
        n01 = min(max(n01, 0), nd)
        if nd and binomtest(n01, nd, 0.5).pvalue < 0.05: hits += 1
    return hits / sims

print("Power of exact McNemar (alpha=.05), n paired items; delta = true accuracy difference")
for n in (1000, 3003):
    for d in (0.10, 0.20, 0.30):
        row = {delta: power(n, delta, d) for delta in (0.01, 0.02, 0.03, 0.05)}
        print(f"n={n:5d} discordance={d:.2f} ->", {k: round(v, 2) for k, v in row.items()})
print("\n95% CI half-width for a paired difference (normal approx.): 1.96*sqrt(d/n)")
for n in (1000, 3003):
    print(f"n={n}:", {d: round(1.96 * np.sqrt(d / n), 4) for d in (0.10, 0.20, 0.30)})
