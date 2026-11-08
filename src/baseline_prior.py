"""Reference baseline (no language model): for each relation predict the most frequent object seen in the DEV split.
Fitted on dev only, evaluated on the held-out test sample. Writes predictions in the same JSONL schema the LLM runner uses."""
import json, collections
import pandas as pd
from scorer import label, match_substring, match_boundary, is_abstention, normalize

dev = pd.read_json("../data/processed/dev.jsonl", lines=True)
test = pd.read_json("../data/processed/test_sample.jsonl", lines=True)

mode = {p: collections.Counter(g["obj"].dropna()).most_common(1)[0][0] for p, g in dev.groupby("prop")}
json.dump(mode, open("../results/baseline_prior_modes.json", "w"), indent=1, ensure_ascii=False)

rows = []
for r in test.itertuples():
    raw = mode.get(r.prop, "")
    lab = label(raw, r.answers)
    rows.append({"id": int(r.id), "condition": "prior_baseline", "prompt_set": "none", "raw_output": raw,
                 "answer": lab["answer"], "outcome": lab["outcome"]})
with open("../results/preds_prior_baseline.jsonl", "w") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")

res = pd.DataFrame(rows).merge(test[["id", "prop", "pop_tertile", "answers", "s_pop"]], on="id")
out = {"n": len(res), "accuracy": float((res.outcome == "correct").mean())}
out["by_tertile"] = res.groupby("pop_tertile").apply(lambda g: float((g.outcome == "correct").mean())).round(4).to_dict()
out["by_relation"] = (res.groupby("prop").apply(lambda g: pd.Series({"n": len(g), "acc": round(float((g.outcome == "correct").mean()), 4)}))
                      .sort_values("acc", ascending=False).reset_index().to_dict("records"))

# matching-rule audit on real data: substring (original PopQA rule) vs whole-phrase boundary rule
sub = [match_substring(r.answer, r.answers) for r in res.itertuples()]
bnd = [match_boundary(r.answer, r.answers) for r in res.itertuples()]
out["substring_acc"], out["boundary_acc"] = sum(sub) / len(sub), sum(bnd) / len(bnd)
out["substring_only_hits"] = int(sum(s and not b for s, b in zip(sub, bnd)))

# how risky is the matcher on the whole dataset? (gold-alias length, abstention lexicon vs gold answers)
full = pd.read_json("../data/processed/popqa_clean.jsonl", lines=True)
out["share_questions_with_alias_le3_chars"] = float(full.answers.apply(lambda a: any(len(normalize(x)) <= 3 for x in a)).mean())
out["gold_objects_triggering_abstain_lexicon"] = int(full["obj"].dropna().apply(is_abstention).sum())
json.dump(out, open("../results/baseline_prior_results.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "by_relation"}, indent=1))
print(pd.DataFrame(out["by_relation"]).to_string(index=False))
print("modes:", mode)
