import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
full = pd.read_json("../data/processed/popqa_clean.jsonl", lines=True)
res = json.load(open("../results/baseline_prior_results.json"))

# Fig 2: popularity by relation
order = full.groupby("prop").log_pop.median().sort_values().index
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.boxplot([full[full.prop == p].log_pop for p in order], vert=False, labels=order, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#cfe0f3", color="#2b5c8a"), medianprops=dict(color="#c0392b"))
ax.set_xlabel("log10 of Wikipedia page views of the subject (s_pop)"); ax.grid(axis="x", alpha=.3)
ax.set_title("PopQA: popularity differs strongly between relation types (n = 14,264)")
plt.tight_layout(); plt.savefig("../figures/fig2_popularity_by_relation.png", dpi=200); plt.close()

# Fig 3: prior baseline accuracy by relation
d = pd.DataFrame(res["by_relation"]).sort_values("acc")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.barh(d.prop + "  (n=" + d.n.astype(int).astype(str) + ")", d.acc * 100, color="#5b8fc7")
for i, v in enumerate(d.acc * 100): ax.text(v + .8, i, f"{v:.1f}%", va="center", fontsize=8)
ax.axvline(res["accuracy"] * 100, color="#c0392b", ls="--"); ax.text(res["accuracy"] * 100 + 1, 0.2, f"overall {res['accuracy']*100:.1f}%", color="#c0392b", fontsize=8)
ax.set_xlabel("Accuracy of the no-model 'most frequent answer' baseline (%)"); ax.set_xlim(0, 80)
ax.set_title("Reference baseline on the held-out test sample (n = 3,003)")
plt.tight_layout(); plt.savefig("../figures/fig3_prior_baseline_by_relation.png", dpi=200); plt.close()

# Fig 1: pipeline
fig, ax = plt.subplots(figsize=(7.8, 4.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)
def box(x, y, w, h, t, c="#e8f0fa", ec="#2b5c8a", fs=6.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=c, ec=ec, lw=1)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1))
G, Y = "#e3f4e1", "#fff3d6"
xs = [1, 21, 41, 61, 81]; txt = ["PopQA .tsv\n14,267 Q-A pairs\n(GitHub, MIT)", "Clean + parse\n3 rows dropped\n= 14,264", "Popularity tertiles\nwithin each of\n16 relations",
       "Subject-grouped\nsplit: dev 2,891 /\ntest 11,373", "Stratified test\nsample\nn = 3,003"]
for x, t in zip(xs, txt): box(x, 48, 17, 11, t, G)
for x in xs[:-1]: arr(x + 17.6, 53.5, x + 20, 53.5)
box(1, 28, 30, 13, "Reference baseline\n(most frequent object per relation,\nfitted on dev only)\nEXECUTED - real results", G)
box(36, 28, 33, 13, "OLMo-2 checkpoints\nbase -> SFT -> DPO -> Instruct\nfixed few-shot prompt\nSCRIPT READY - not yet run", Y)
box(74, 28, 25, 13, "Scorer\nextract -> normalise ->\nwhole-phrase match ->\ncorrect / incorrect / abstain\n7 unit tests PASSED", G)
arr(89, 47.6, 20, 41.6); arr(89, 47.6, 52, 41.6)
box(1, 15, 68, 6, "predictions.jsonl  (one schema for every condition: id, condition, prompt_set, raw_output)", "#f4f4f4", "#888", fs=6.0)
arr(16, 27.6, 16, 21.6); arr(52, 27.6, 52, 21.6); arr(69.6, 18, 74.4, 27.6)
box(32, 1, 67, 9, "Analysis: rates + Wilson CI | paired stage differences (clustered bootstrap CI)\nexact McNemar with Holm correction | popularity x stage logit (subject-clustered SE)\nVERIFIED on synthetic fixtures with a known effect", G, fs=6.4)
arr(86, 27.6, 86, 10.6)
ax.text(1, 6, "green = executed and evidenced\nyellow = implemented,\nawaiting GPU run", fontsize=6.0, va="center")
plt.savefig("../figures/fig1_pipeline.png", dpi=200, bbox_inches="tight"); plt.close()
print("figures ok")
