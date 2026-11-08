"""Verification of analyze.py on SYNTHETIC inputs with a known injected effect.
These files are test fixtures built from the real prior-baseline labels; they are NOT language-model results."""
import json, numpy as np, pandas as pd
from analyze import load, stage_table, rates, popularity_model
rng = np.random.default_rng(0)
test = pd.read_json("../data/processed/test_sample.jsonl", lines=True)
base = pd.read_json("../results/preds_prior_baseline.jsonl", lines=True)

def fixture(name, gain=0, loss=0):
    d = base.copy(); d["condition"] = name
    inc = d.index[d.outcome == "incorrect"].to_numpy(); cor = d.index[d.outcome == "correct"].to_numpy()
    d.loc[rng.choice(inc, gain, replace=False), "outcome"] = "correct"
    d.loc[rng.choice(cor, loss, replace=False), "outcome"] = "incorrect"
    return d

A = fixture("synth_A"); B = fixture("synth_B_plus5pp", gain=150); C = fixture("synth_C_null_swap", gain=60, loss=60)
P = pd.concat([A, B, C]).merge(test[["id", "prop", "pop_tertile", "log_pop", "s_uri"]], on="id")
t1 = stage_table(P, ["synth_A", "synth_B_plus5pp"]); t2 = stage_table(P, ["synth_A", "synth_C_null_swap"])
c = t1[t1.outcome == "correct"].iloc[0]; z = t2[t2.outcome == "correct"].iloc[0]
print(f"injected +150 gains (+4.99 pp): diff={c['diff']:.4f}  CI=[{c.ci_lo:.4f},{c.ci_hi:.4f}]  gained={c.gained} lost={c.lost}  p={c.p_mcnemar:.2e}")
print(f"symmetric swap 60/60 (0 pp)   : diff={z['diff']:.4f}  CI=[{z.ci_lo:.4f},{z.ci_hi:.4f}]  gained={z.gained} lost={z.lost}  p={z.p_mcnemar:.3f}  within +-3pp margin: {bool(z.ci_within_margin)}")
assert abs(c["diff"] - 150 / 3003) < 1e-9 and c.ci_lo > 0 and c.gained == 150 and c.lost == 0 and c.p_mcnemar < 1e-10
assert abs(z["diff"]) < 1e-9 and z.p_mcnemar > 0.05 and bool(z.ci_within_margin)
m = popularity_model(P); print("popularity logit fitted, n_obs =", int(m.nobs), "| converged:", m.mle_retvals["converged"])
print("ALL ANALYSIS CHECKS PASSED")
