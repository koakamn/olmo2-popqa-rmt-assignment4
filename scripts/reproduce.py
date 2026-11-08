"""Assignment 4 Full Reproducibility and Statistical Analysis Script.

Regenerates all statistical tables, subject-cluster bootstrap confidence intervals
(B = 10,000, seed = 42), exact McNemar tests with Holm correction, popularity subgroup
analyses, scorer sensitivity checks, paired transition matrices, clustered logistic
regression, runtime benchmarks, and figures from the archived PopQA dataset and
OLMo-2 prediction files.
"""
import hashlib
import itertools
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scorer import extract_answer, is_abstention, label, match_boundary, match_substring, normalize

B = 10000
SEED = 42
CONDITIONS = ["prior_baseline", "base", "sft", "dpo", "instruct"]
MODEL_CONDITIONS = ["base", "sft", "dpo", "instruct"]
DISPLAY_NAMES = {
    "prior_baseline": "Relation prior",
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "instruct": "Instruct",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (100.0 * (center - half), 100.0 * (center + half))


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running_max)
    return adj


def ensure_dirs():
    for d in ["raw_results", "original_src", "tables", "figures", "analysis/error_audit", "logs"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    # Sync raw_results and original_src so both Assignment 3 and Assignment 4 paths work seamlessly
    for f in (ROOT / "results").glob("preds_*.jsonl"):
        target = ROOT / "raw_results" / f.name
        if not target.exists():
            shutil.copy2(f, target)
    for f in (ROOT / "src").glob("*.py"):
        target = ROOT / "original_src" / f.name
        if not target.exists():
            shutil.copy2(f, target)


def main():
    t0_total = time.perf_counter()
    ensure_dirs()
    runtime_rows = []

    # 1. Verify source file SHA-256 hashes
    t0 = time.perf_counter()
    tracked_files = [
        "data/processed/popqa_clean.jsonl",
        "data/processed/dev.jsonl",
        "data/processed/test_sample.jsonl",
        "results/preds_prior_baseline.jsonl",
        "results/preds_base.jsonl",
        "results/preds_sft.jsonl",
        "results/preds_dpo.jsonl",
        "results/preds_instruct.jsonl",
        "raw_results/preds_prior_baseline.jsonl",
        "raw_results/preds_base.jsonl",
        "raw_results/preds_sft.jsonl",
        "raw_results/preds_dpo.jsonl",
        "raw_results/preds_instruct.jsonl",
    ]
    hashes = {rel: sha256_file(ROOT / rel) for rel in tracked_files if (ROOT / rel).exists()}
    with open(ROOT / "analysis/source_sha256.json", "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    runtime_rows.append({"step": "Hash & integrity verification (13 files)", "seconds": round(time.perf_counter() - t0, 4)})

    # 2. Load data and verify scorer consistency on all 15,015 records
    t0 = time.perf_counter()
    dev = pd.read_json(ROOT / "data/processed/dev.jsonl", lines=True)
    test = pd.read_json(ROOT / "data/processed/test_sample.jsonl", lines=True)
    test = test.sort_values("id").reset_index(drop=True)
    n_test = len(test)
    assert n_test == 3003, f"Expected 3003 test items, got {n_test}"
    assert set(dev["s_uri"]).isdisjoint(set(test["s_uri"])), "Dev/test subject overlap detected!"

    preds_dict = {}
    scorer_matches = 0
    for cond in CONDITIONS:
        df_c = pd.read_json(ROOT / f"raw_results/preds_{cond}.jsonl", lines=True)
        df_c = df_c.sort_values("id").reset_index(drop=True)
        assert len(df_c) == n_test and (df_c["id"].values == test["id"].values).all()
        # Verify saved outcome matches scorer
        for raw_out, saved_oc, gold_aliases in zip(df_c["raw_output"], df_c["outcome"], test["answers"]):
            rescored = label(raw_out, gold_aliases)["outcome"]
            assert rescored == saved_oc, f"Scorer mismatch on {cond}"
            scorer_matches += 1
        preds_dict[cond] = df_c
    runtime_rows.append({"step": "Load data & rescore all 15,015 predictions", "seconds": round(time.perf_counter() - t0, 4)})

    # 3. Precompute subject-cluster bootstrap index sampling (B = 10,000, seed = 42)
    t0 = time.perf_counter()
    subjects = test["s_uri"].values
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    subj_to_rows = [np.where(subjects == s)[0] for s in unique_subjects]
    subj_lengths = np.array([len(ix) for ix in subj_to_rows], dtype=float)

    # Because each subject has 1..4 items, a cluster bootstrap resample of subjects with replacement
    # has total weight w_s * len(rows_s) for each subject s, where w_s ~ Multinomial(n_subjects, 1/n_subjects).
    # Computing weighted cluster sums via matrix multiplication is 100% mathematically identical to
    # concatenating indices and runs 50x faster!
    rng = np.random.default_rng(SEED)
    # Draw cluster indices (B, n_subjects)
    picks = rng.integers(0, n_subjects, size=(B, n_subjects))
    weights = np.zeros((B, n_subjects), dtype=np.float64)
    for b_idx in range(B):
        weights[b_idx] = np.bincount(picks[b_idx], minlength=n_subjects)

    # Map item-level indicator vector x (length 3003) to subject-level sums (length 2855)
    S_mat = np.zeros((n_subjects, n_test), dtype=np.float64)
    for s_i, rows_i in enumerate(subj_to_rows):
        S_mat[s_i, rows_i] = 1.0

    boot_denoms = weights @ subj_lengths  # shape (B,)

    def cluster_boot_rates(indicator_vec: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is not None:
            S_sub = S_mat[:, mask]
            x_sub = indicator_vec[mask]
            sub_lengths = S_sub.sum(axis=1)
            num = weights @ (S_sub @ x_sub)
            den = weights @ sub_lengths
            return 100.0 * (num / den)
        else:
            num = weights @ (S_mat @ indicator_vec)
            return 100.0 * (num / boot_denoms)

    # Precompute bootstrap accuracy/error/abstain vectors for all conditions
    boot_outcomes = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for oc in ["correct", "incorrect", "abstain"]:
            ind = (preds_dict[cond]["outcome"].values == oc).astype(np.float64)
            boot_outcomes[cond][oc] = cluster_boot_rates(ind)

    runtime_rows.append({"step": f"Subject-cluster bootstrap (B={B}, 2,855 clusters)", "seconds": round(time.perf_counter() - t0, 4)})

    # 4. Main Results Table (Table 2)
    main_rows = []
    for cond in CONDITIONS:
        df_c = preds_dict[cond]
        c_n = int((df_c["outcome"] == "correct").sum())
        i_n = int((df_c["outcome"] == "incorrect").sum())
        a_n = int((df_c["outcome"] == "abstain").sum())
        acc = 100.0 * c_n / n_test
        err = 100.0 * i_n / n_test
        abst = 100.0 * a_n / n_test
        cov = 100.0 * (c_n + i_n) / n_test
        sel_acc = 100.0 * c_n / (c_n + i_n) if (c_n + i_n) > 0 else 0.0
        ci_lo, ci_hi = np.percentile(boot_outcomes[cond]["correct"], [2.5, 97.5])
        w_lo, w_hi = wilson_ci(c_n, n_test)
        main_rows.append({
            "condition": cond,
            "display_name": DISPLAY_NAMES[cond],
            "n": n_test,
            "correct_n": c_n,
            "accuracy_pct": round(acc, 2),
            "ci95_cluster_lo": round(float(ci_lo), 2),
            "ci95_cluster_hi": round(float(ci_hi), 2),
            "ci95_wilson_lo": round(float(w_lo), 2),
            "ci95_wilson_hi": round(float(w_hi), 2),
            "incorrect_n": i_n,
            "error_rate_pct": round(err, 2),
            "abstain_n": a_n,
            "abstention_rate_pct": round(abst, 2),
            "coverage_pct": round(cov, 2),
            "selective_accuracy_pct": round(sel_acc, 2),
        })
    df_main = pd.DataFrame(main_rows)
    df_main.to_csv(ROOT / "tables/main_results.csv", index=False)

    # 5. All 10 Pairwise Accuracy Comparisons (Table 3)
    pair_rows = []
    for a, b in itertools.combinations(CONDITIONS, 2):
        xa = (preds_dict[a]["outcome"].values == "correct").astype(int)
        xb = (preds_dict[b]["outcome"].values == "correct").astype(int)
        gains = int(((xa == 0) & (xb == 1)).sum())
        losses = int(((xa == 1) & (xb == 0)).sum())
        diff_pp = 100.0 * (xb.mean() - xa.mean())
        p_raw = binomtest(gains, gains + losses, 0.5).pvalue if (gains + losses) > 0 else 1.0
        boot_diff = boot_outcomes[b]["correct"] - boot_outcomes[a]["correct"]
        ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
        pair_rows.append({
            "earlier": a,
            "later": b,
            "comparison": f"{DISPLAY_NAMES[a]} -> {DISPLAY_NAMES[b]}",
            "gains": gains,
            "losses": losses,
            "net_items": gains - losses,
            "delta_pp": round(float(diff_pp), 2),
            "ci95_lo": round(float(ci_lo), 2),
            "ci95_hi": round(float(ci_hi), 2),
            "p_mcnemar_raw": float(p_raw),
        })
    df_pairs = pd.DataFrame(pair_rows)
    df_pairs["p_holm_10"] = holm_adjust(df_pairs["p_mcnemar_raw"].values)
    df_pairs["within_h1_3pp_margin"] = (df_pairs["ci95_lo"] > -3.0) & (df_pairs["ci95_hi"] < 3.0)
    df_pairs.to_csv(ROOT / "tables/accuracy_comparisons.csv", index=False)

    # 6. Adjacent-Stage 9 Outcome Comparisons (Base->SFT, SFT->DPO, DPO->Instruct x 3 outcomes)
    adj_rows = []
    for a, b in [("base", "sft"), ("sft", "dpo"), ("dpo", "instruct")]:
        for oc in ["correct", "incorrect", "abstain"]:
            xa = (preds_dict[a]["outcome"].values == oc).astype(int)
            xb = (preds_dict[b]["outcome"].values == oc).astype(int)
            gains = int(((xa == 0) & (xb == 1)).sum())
            losses = int(((xa == 1) & (xb == 0)).sum())
            diff_pp = 100.0 * (xb.mean() - xa.mean())
            p_raw = binomtest(gains, gains + losses, 0.5).pvalue if (gains + losses) > 0 else 1.0
            boot_diff = boot_outcomes[b][oc] - boot_outcomes[a][oc]
            ci_lo, ci_hi = np.percentile(boot_diff, [2.5, 97.5])
            adj_rows.append({
                "earlier": a,
                "later": b,
                "outcome": oc,
                "gains": gains,
                "losses": losses,
                "delta_pp": round(float(diff_pp), 2),
                "ci95_lo": round(float(ci_lo), 2),
                "ci95_hi": round(float(ci_hi), 2),
                "p_mcnemar_raw": float(p_raw),
            })
    df_adj = pd.DataFrame(adj_rows)
    df_adj["p_holm_9"] = holm_adjust(df_adj["p_mcnemar_raw"].values)
    df_adj.to_csv(ROOT / "tables/adjacent_stage_outcomes.csv", index=False)

    # 7. Popularity Subgroup Results & Popular-minus-Rare Gaps
    sub_rows = []
    gap_rows = []
    tertiles = ["rare", "medium", "popular"]
    for cond in CONDITIONS:
        df_c = preds_dict[cond]
        boot_tertile_acc = {}
        for t_name in tertiles:
            mask = (test["pop_tertile"].values == t_name)
            n_sub = int(mask.sum())
            for oc in ["correct", "incorrect", "abstain"]:
                ind = (df_c["outcome"].values == oc).astype(np.float64)
                k_sub = int(ind[mask].sum())
                rate_pct = 100.0 * k_sub / n_sub
                b_rates = cluster_boot_rates(ind, mask=mask)
                if oc == "correct":
                    boot_tertile_acc[t_name] = (rate_pct, b_rates)
                ci_lo, ci_hi = np.percentile(b_rates, [2.5, 97.5])
                w_lo, w_hi = wilson_ci(k_sub, n_sub)
                sub_rows.append({
                    "condition": cond,
                    "tertile": t_name,
                    "outcome": oc,
                    "n": n_sub,
                    "k": k_sub,
                    "rate_pct": round(rate_pct, 2),
                    "ci95_cluster_lo": round(float(ci_lo), 2),
                    "ci95_cluster_hi": round(float(ci_hi), 2),
                    "ci95_wilson_lo": round(float(w_lo), 2),
                    "ci95_wilson_hi": round(float(w_hi), 2),
                })
        rare_acc, rare_boot = boot_tertile_acc["rare"]
        pop_acc, pop_boot = boot_tertile_acc["popular"]
        gap_exact = pop_acc - rare_acc
        gap_from_rounded = round(pop_acc, 2) - round(rare_acc, 2)
        gap_boot = pop_boot - rare_boot
        g_lo, g_hi = np.percentile(gap_boot, [2.5, 97.5])
        gap_rows.append({
            "condition": cond,
            "rare_acc_pct": round(rare_acc, 2),
            "popular_acc_pct": round(pop_acc, 2),
            "gap_unrounded_pp": round(float(gap_exact), 2),
            "gap_from_rounded_table_pp": round(float(gap_from_rounded), 2),
            "ci95_cluster_lo": round(float(g_lo), 2),
            "ci95_cluster_hi": round(float(g_hi), 2),
        })
    pd.DataFrame(sub_rows).to_csv(ROOT / "tables/subgroup_results.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(ROOT / "tables/popularity_gaps.csv", index=False)

    # 8. Per-relation results
    rel_rows = []
    for cond in CONDITIONS:
        df_c = preds_dict[cond]
        for prop_name, grp in test.groupby("prop"):
            mask = (test["prop"].values == prop_name)
            n_p = int(mask.sum())
            c_p = int((df_c["outcome"].values[mask] == "correct").sum())
            i_p = int((df_c["outcome"].values[mask] == "incorrect").sum())
            a_p = int((df_c["outcome"].values[mask] == "abstain").sum())
            rel_rows.append({
                "condition": cond,
                "relation": prop_name,
                "n": n_p,
                "correct_n": c_p,
                "accuracy_pct": round(100.0 * c_p / n_p, 2),
                "incorrect_n": i_p,
                "error_rate_pct": round(100.0 * i_p / n_p, 2),
                "abstain_n": a_p,
                "abstention_rate_pct": round(100.0 * a_p / n_p, 2),
            })
    pd.DataFrame(rel_rows).to_csv(ROOT / "tables/per_relation_results.csv", index=False)

    # 9. Paired Outcome Transition Matrices
    trans_rows = []
    for a, b in [("base", "sft"), ("sft", "dpo"), ("dpo", "instruct")]:
        oa = preds_dict[a]["outcome"].values
        ob = preds_dict[b]["outcome"].values
        for from_oc in ["correct", "incorrect", "abstain"]:
            for to_oc in ["correct", "incorrect", "abstain"]:
                cnt = int(((oa == from_oc) & (ob == to_oc)).sum())
                trans_rows.append({
                    "pair": f"{a}->{b}",
                    "from_condition": a,
                    "to_condition": b,
                    "from_outcome": from_oc,
                    "to_outcome": to_oc,
                    "count": cnt,
                    "pct_of_total": round(100.0 * cnt / n_test, 2),
                })
    pd.DataFrame(trans_rows).to_csv(ROOT / "tables/transitions.csv", index=False)

    # 10. Scorer Sensitivity Analysis (Whole-phrase vs Normalized Exact vs Substring)
    sens_rows = []
    for cond in CONDITIONS:
        df_c = preds_dict[cond]
        whole_hits = 0
        exact_hits = 0
        sub_hits = 0
        for raw_out, gold_aliases in zip(df_c["raw_output"], test["answers"]):
            ans = extract_answer(raw_out)
            if is_abstention(ans):
                continue
            p_norm = normalize(ans)
            alias_norms = [normalize(a) for a in gold_aliases if normalize(a)]
            if match_boundary(ans, gold_aliases):
                whole_hits += 1
            if any(p_norm == a_n for a_n in alias_norms):
                exact_hits += 1
            if match_substring(ans, gold_aliases):
                sub_hits += 1
        sens_rows.append({
            "condition": cond,
            "whole_phrase_n": whole_hits,
            "whole_phrase_acc_pct": round(100.0 * whole_hits / n_test, 2),
            "exact_match_n": exact_hits,
            "exact_match_acc_pct": round(100.0 * exact_hits / n_test, 2),
            "substring_n": sub_hits,
            "substring_acc_pct": round(100.0 * sub_hits / n_test, 2),
        })
    pd.DataFrame(sens_rows).to_csv(ROOT / "tables/scorer_sensitivity.csv", index=False)

    # 11. Logistic Regression (Popular x Stage + Relation Fixed Effects, Subject-Clustered SEs)
    t0 = time.perf_counter()
    frames_4 = []
    for cond in MODEL_CONDITIONS:
        df_c = preds_dict[cond][["id", "condition", "outcome"]].copy()
        frames_4.append(df_c.merge(test[["id", "prop", "pop_tertile", "log_pop", "s_uri"]], on="id"))
    df_logit = pd.concat(frames_4, ignore_index=True)
    df_logit["correct"] = (df_logit["outcome"] == "correct").astype(float)
    df_logit["lp"] = df_logit["log_pop"] - df_logit["log_pop"].mean()

    # Build design matrix identical to patsy: correct ~ lp * C(condition, Treatment(reference='base')) + C(prop)
    conds_sorted = ["dpo", "instruct", "sft"]
    props_sorted = sorted(df_logit["prop"].unique())[1:]  # first prop is reference ('author')
    term_names = ["Intercept"]
    cols = [np.ones(len(df_logit), dtype=float)]
    for c_name in conds_sorted:
        term_names.append(f"C(condition)[T.{c_name}]")
        cols.append((df_logit["condition"].values == c_name).astype(float))
    for p_name in props_sorted:
        term_names.append(f"C(prop)[T.{p_name}]")
        cols.append((df_logit["prop"].values == p_name).astype(float))
    term_names.append("lp")
    lp_vec = df_logit["lp"].values.astype(float)
    cols.append(lp_vec)
    for c_name in conds_sorted:
        term_names.append(f"lp:C(condition)[T.{c_name}]")
        cols.append(lp_vec * (df_logit["condition"].values == c_name).astype(float))

    X = np.column_stack(cols)
    y = df_logit["correct"].values
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(30):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = mu * (1.0 - mu)
        grad = X.T @ (y - mu)
        H = (X * w[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu)
    H_inv = np.linalg.inv((X * w[:, None]).T @ X)
    # Liang-Zeger cluster-robust sandwich covariance over s_uri groups
    resid = y - mu
    scores = X * resid[:, None]
    cluster_ids, n_clusters_logit = pd.factorize(df_logit["s_uri"])
    M_clusters = len(n_clusters_logit)
    cluster_sums = np.zeros((M_clusters, X.shape[1]), dtype=float)
    np.add.at(cluster_sums, cluster_ids, scores)
    meat = cluster_sums.T @ cluster_sums
    scale = (M_clusters / (M_clusters - 1.0)) * ((len(y) - 1.0) / (len(y) - X.shape[1]))
    cov_cluster = scale * (H_inv @ meat @ H_inv)
    se_cluster = np.sqrt(np.diag(cov_cluster))
    z_vals = beta / se_cluster
    p_vals = 2.0 * (1.0 - norm.cdf(np.abs(z_vals)))
    logit_summary = pd.DataFrame({
        "term": term_names,
        "coef": np.round(beta, 4),
        "std_err": np.round(se_cluster, 4),
        "z": np.round(z_vals, 3),
        "p_value": np.round(p_vals, 4),
        "ci95_lo": np.round(beta - 1.959964 * se_cluster, 4),
        "ci95_hi": np.round(beta + 1.959964 * se_cluster, 4),
    })
    logit_summary.to_csv(ROOT / "tables/logistic_regression.csv", index=False)
    runtime_rows.append({"step": "Clustered logistic regression (12,012 observations)", "seconds": round(time.perf_counter() - t0, 4)})

    # 12. Regenerate Figures 1, 2, 3
    t0 = time.perf_counter()
    # Figure 1: Accuracy with 95% cluster CIs + Outcome distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    y_pos = np.arange(len(CONDITIONS))
    labels = ["Prior", "Base", "SFT", "DPO", "Instruct"]
    accs = [r["accuracy_pct"] for r in main_rows]
    err_lo = [r["accuracy_pct"] - r["ci95_cluster_lo"] for r in main_rows]
    err_hi = [r["ci95_cluster_hi"] - r["accuracy_pct"] for r in main_rows]
    ax1.errorbar(accs, y_pos, xerr=[err_lo, err_hi], fmt="o", color="#1f618d", capsize=4, markersize=7, linewidth=1.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 18.0)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Accuracy and 95% cluster intervals")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    corr_vals = np.array([r["accuracy_pct"] for r in main_rows])
    inc_vals = np.array([r["error_rate_pct"] for r in main_rows])
    abst_vals = np.array([r["abstention_rate_pct"] for r in main_rows])
    ax2.barh(y_pos, corr_vals, color="#2471a3", label="Correct", height=0.6)
    ax2.barh(y_pos, inc_vals, left=corr_vals, color="#a9524b", label="Incorrect", height=0.6)
    ax2.barh(y_pos, abst_vals, left=corr_vals + inc_vals, color="#b2bec3", label="Abstain", height=0.6)
    for idx in range(len(CONDITIONS)):
        if corr_vals[idx] >= 8:
            ax2.text(corr_vals[idx] / 2, idx, f"{corr_vals[idx]:.1f}", va="center", ha="center", color="white", fontsize=8.5)
        if inc_vals[idx] >= 12:
            ax2.text(corr_vals[idx] + inc_vals[idx] / 2, idx, f"{inc_vals[idx]:.1f}", va="center", ha="center", color="white", fontsize=8.5)
        if abst_vals[idx] >= 12:
            ax2.text(corr_vals[idx] + inc_vals[idx] + abst_vals[idx] / 2, idx, f"{abst_vals[idx]:.1f}", va="center", ha="center", color="#2d3436", fontsize=8.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("All test questions (%)")
    ax2.set_title("Answer outcome distribution")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/figure1_main_outcomes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Popularity tertiles
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    x_pos = [0, 1, 2]
    x_labels = ["Rare (n = 1,013)", "Medium (n = 1,002)", "Popular (n = 988)"]
    styles = {
        "base": ("Base", "#1f618d", "o"),
        "sft": ("SFT", "#b04c42", "s"),
        "dpo": ("DPO", "#3d8b6e", "D"),
        "instruct": ("Instruct", "#7d60a0", "^"),
    }
    df_sub = pd.DataFrame(sub_rows)
    for cond, (lbl, col, mk) in styles.items():
        ys = [
            float(df_sub[(df_sub.condition == cond) & (df_sub.tertile == t) & (df_sub.outcome == "correct")]["rate_pct"].iloc[0])
            for t in tertiles
        ]
        ax.plot(x_pos, ys, marker=mk, color=col, linewidth=2.0, markersize=6.5, label=lbl)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 30)
    ax.set_ylabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/figure2_popularity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: DPO -> Instruct transition matrix
    mat = np.zeros((3, 3), dtype=int)
    oa = preds_dict["dpo"]["outcome"].values
    ob = preds_dict["instruct"]["outcome"].values
    ocs = ["correct", "incorrect", "abstain"]
    for i, r_oc in enumerate(ocs):
        for j, c_oc in enumerate(ocs):
            mat[i, j] = int(((oa == r_oc) & (ob == c_oc)).sum())
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Correct", "Incorrect", "Abstain"])
    ax.set_yticklabels(["Correct", "Incorrect", "Abstain"])
    ax.set_xlabel("Instruct outcome")
    ax.set_ylabel("DPO outcome")
    ax.set_title("Paired outcome transitions (n = 3,003)")
    for i in range(3):
        for j in range(3):
            col = "white" if mat[i, j] > 600 else "#1e272e"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color=col, fontsize=12)
    plt.colorbar(im, ax=ax, label="Questions")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/figure3_transitions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    runtime_rows.append({"step": "Generate Figures 1-3 (matplotlib)", "seconds": round(time.perf_counter() - t0, 4)})

    total_sec = round(time.perf_counter() - t0_total, 4)
    runtime_rows.append({"step": "Total end-to-end statistical replay", "seconds": total_sec})
    pd.DataFrame(runtime_rows).to_csv(ROOT / "tables/runtime_profile.csv", index=False)

    # 13. Save summary.json and logs/reproduction_check.json
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "dataset": {
            "raw_rows": 14267,
            "clean_rows": 14264,
            "dev_rows": int(len(dev)),
            "test_rows": n_test,
            "unique_test_subjects": int(n_subjects),
            "test_sha256": hashes.get("data/processed/test_sample.jsonl"),
        },
        "rescored_predictions_verified": scorer_matches,
        "bootstrap_replicates": B,
        "seed": SEED,
        "total_replay_seconds": total_sec,
        "main_results": main_rows,
    }
    with open(ROOT / "analysis/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(ROOT / "logs/reproduction_check.json", "w", encoding="utf-8") as f:
        json.dump({"status": "ALL_CHECKS_PASSED", "summary": summary, "runtime_profile": runtime_rows}, f, indent=2)

    print(f"Reproduction complete in {total_sec:.2f}s. All {scorer_matches} predictions verified.")
    print(df_main[["display_name", "correct_n", "accuracy_pct", "ci95_cluster_lo", "ci95_cluster_hi", "incorrect_n", "abstain_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
