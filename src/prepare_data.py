"""Step 1: load PopQA, clean, assign within-relation popularity tertiles,
make a subject-grouped dev/test split and a stratified test sample."""
import json, hashlib, sys
import numpy as np, pandas as pd

SEED = 42
RAW = "data/popQA.tsv"
N_TEST_SAMPLE = 3000

df = pd.read_csv(RAW, sep="\t")
n_raw = len(df)
df = df.dropna(subset=["subj"]).drop_duplicates("id").copy()        # 3 rows have no subject
df["answers"] = df["possible_answers"].apply(json.loads)
df["log_pop"] = np.log10(df["s_pop"].astype(float))

# popularity tertiles computed WITHIN each relation (popularity differs a lot between relations)
df["pop_tertile"] = df.groupby("prop")["s_pop"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 3, labels=["rare", "medium", "popular"]).astype(str))

# subject-grouped dev/test split (the same subject can have several questions) -> no subject leakage
rng = np.random.default_rng(SEED)
subjects = np.array(df["s_uri"].unique(), dtype=object)
rng.shuffle(subjects)
n_dev = int(0.2 * len(subjects))
dev_subj = set(subjects[:n_dev])
df["split"] = np.where(df["s_uri"].isin(dev_subj), "dev", "test")
assert set(df[df.split == "dev"].s_uri).isdisjoint(set(df[df.split == "test"].s_uri))

# stratified (relation x tertile) test sample, proportional allocation
test = df[df.split == "test"]
frac = N_TEST_SAMPLE / len(test)
sample = (test.groupby(["prop", "pop_tertile"], group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=SEED)))
df["in_test_sample"] = df["id"].isin(sample["id"])

cols = ["id", "subj", "prop", "question", "answers", "s_uri", "s_pop", "log_pop", "pop_tertile", "split", "in_test_sample", "obj"]
df[cols].to_json("data/processed/popqa_clean.jsonl", orient="records", lines=True, force_ascii=False)
df[df.split == "dev"][cols].to_json("data/processed/dev.jsonl", orient="records", lines=True, force_ascii=False)
df[df.in_test_sample][cols].to_json("data/processed/test_sample.jsonl", orient="records", lines=True, force_ascii=False)

stats = {
    "raw_rows": n_raw, "after_cleaning": len(df), "dropped": n_raw - len(df),
    "n_subjects": int(df.s_uri.nunique()), "n_relations": int(df.prop.nunique()),
    "dev_rows": int((df.split == "dev").sum()), "test_rows": int((df.split == "test").sum()),
    "test_sample_rows": int(df.in_test_sample.sum()),
    "test_sample_by_tertile": df[df.in_test_sample].pop_tertile.value_counts().to_dict(),
    "test_sample_by_relation": df[df.in_test_sample].prop.value_counts().to_dict(),
    "median_s_pop_by_tertile_in_sample": df[df.in_test_sample].groupby("pop_tertile").s_pop.median().to_dict(),
    "sha256_raw": hashlib.sha256(open(RAW, "rb").read()).hexdigest(), "seed": SEED,
}
json.dump(stats, open("results/data_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
