"""LLM runner (Colab/GPU). Same JSONL schema as the baseline. Greedy decoding, fixed few-shot prompt built from DEV items only.
Usage:  python run_llm.py --model allenai/OLMo-2-0425-1B --condition base --prompt_set 1
        python run_llm.py --dry-run          (builds and prints prompts, needs no GPU/model)"""
import argparse, json, random
import pandas as pd
from scorer import label

INSTR = 'Answer each question with a short phrase. If you are not sure of the answer, reply "I don\'t know".\n\n'
MODELS = {"base": "allenai/OLMo-2-0425-1B", "sft": "allenai/OLMo-2-0425-1B-SFT",
          "dpo": "allenai/OLMo-2-0425-1B-DPO", "instruct": "allenai/OLMo-2-0425-1B-Instruct"}

def build_shots(dev, prompt_set, k=5):
    """k exemplars from dev, one per distinct relation, seeded by prompt_set; the last one is an abstention exemplar
    taken from a rare dev question so that the model sees that abstaining is allowed."""
    rnd = random.Random(1000 + prompt_set)
    props = rnd.sample(sorted(dev.prop.unique()), k)
    shots = []
    for i, p in enumerate(props):
        g = dev[(dev.prop == p) & (dev.pop_tertile == ("rare" if i == k - 1 else "popular"))]
        r = g.iloc[rnd.randrange(len(g))]
        shots.append((r.question, "I don't know" if i == k - 1 else r.answers[0]))
    rnd.shuffle(shots)
    return shots

def make_prompt(shots, question):
    return INSTR + "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in shots) + f"Question: {question}\nAnswer:"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="base"); ap.add_argument("--model"); ap.add_argument("--prompt_set", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dev = pd.read_json("../data/processed/dev.jsonl", lines=True)
    test = pd.read_json("../data/processed/test_sample.jsonl", lines=True)
    shots = build_shots(dev, a.prompt_set)
    if a.dry_run:
        print(make_prompt(shots, test.iloc[0].question)); raise SystemExit
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(42)
    name = a.model or MODELS[a.condition]
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    out = open(f"../results/preds_{a.condition}.jsonl", "w")
    for i in range(0, len(test), a.batch):
        chunk = test.iloc[i:i + a.batch]
        enc = tok([make_prompt(shots, q) for q in chunk.question], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, g in zip(chunk.itertuples(), gen):
            raw = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True)
            lab = label(raw, r.answers)
            out.write(json.dumps({"id": int(r.id), "condition": a.condition, "prompt_set": a.prompt_set, "model": name,
                                  "raw_output": raw, **lab}, ensure_ascii=False) + "\n")
