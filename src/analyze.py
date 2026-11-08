"""Statistical analysis of stage conditions. Input: preds_<condition>.jsonl files (schema: id, condition, outcome).
Outputs: rates with Wilson CIs, paired stage differences (clustered bootstrap CI + exact McNemar, Holm-adjusted),
transition table, and logistic model popularity x stage with subject-clustered SE."""
import json, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm
import statsmodels.formula.api as smf

B, SEED = 2000, 42

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)

def load(paths, test):
    frames = [pd.read_json(p, lines=True) for p in paths]
    preds = pd.concat(frames)
    return preds.merge(test[["id", "prop", "pop_tertile", "log_pop", "s_uri"]], on="id", validate="m:1")

def rates(preds):
    rows = []
    for (cond, grp), g in preds.assign(grp="all").groupby(["condition", "grp"]):
        pass
    for cond, g in preds.groupby("condition"):
        for name, sub in [("all", g)] + [(t, g[g.pop_tertile == t]) for t in ["rare", "medium", "popular"]]:
            n = len(sub)
            for oc in ["correct", "incorrect", "abstain"]:
                k = int((sub.outcome == oc).sum()); lo, hi = wilson(k, n)
                rows.append(dict(condition=cond, subset=name, outcome=oc, n=n, k=k, rate=k / n, lo=lo, hi=hi))
    return pd.DataFrame(rows)

def paired_diff(preds, a, b, outcome="correct", subset=None, rng=None):
    """difference in rate of `outcome` (b minus a) on identical items; cluster bootstrap over subjects; exact McNemar."""
    pa = preds[preds.condition == a].set_index("id"); pb = preds[preds.condition == b].set_index("id")
    ids = pa.index.intersection(pb.index)
    pa, pb = pa.loc[ids], pb.loc[ids]
    if subset is not None:
        keep = pa.pop_tertile == subset; pa, pb = pa[keep], pb[keep]
    xa, xb = (pa.outcome == outcome).values.astype(int), (pb.outcome == outcome).values.astype(int)
    diff = xb.mean() - xa.mean()
    n01, n10 = int(((xa == 0) & (xb == 1)).sum()), int(((xa == 1) & (xb == 0)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    groups = pa.s_uri.values; ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    rng = rng or np.random.default_rng(SEED); boots = []
    for _ in range(B):
        pick = rng.choice(ug, len(ug), replace=True); ii = np.concatenate([idx[g] for g in pick])
        boots.append(xb[ii].mean() - xa[ii].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(a=a, b=b, outcome=outcome, subset=subset or "all", n=len(ids), diff=diff, ci_lo=lo, ci_hi=hi,
                gained=n01, lost=n10, p_mcnemar=p)

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1, run)
    return adj

def stage_table(preds, chain, margin=0.03):
    rows = []
    for a, b in zip(chain[:-1], chain[1:]):
        for oc in ["correct", "incorrect", "abstain"]:
            rows.append(paired_diff(preds, a, b, oc))
    t = pd.DataFrame(rows); t["p_holm"] = holm(t.p_mcnemar.values)
    t["ci_within_margin"] = (t.ci_lo > -margin) & (t.ci_hi < margin)      # H1 criterion (pre-specified +-3 pp)
    return t

def popularity_model(preds):
    d = preds.copy(); d["correct"] = (d.outcome == "correct").astype(int); d["lp"] = d.log_pop - d.log_pop.mean()
    m = smf.logit("correct ~ lp * C(condition) + C(prop)", data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.s_uri)[0]})
    return m

if __name__ == "__main__":
    test = pd.read_json("../data/processed/test_sample.jsonl", lines=True)
    chain = sys.argv[1].split(",")                      # e.g. base,sft,dpo
    paths = [f"../results/preds_{c}.jsonl" for c in chain]
    P = load(paths, test)
    print(rates(P).round(4).to_string(index=False)); print(stage_table(P, chain).round(4).to_string(index=False))
    print(popularity_model(P).summary().tables[1])
