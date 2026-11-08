# How Do Training Stages Shape What LLMs Know? A Comparative Study of Factual Knowledge in Pre-trained, Fine-tuned and Aligned Language Models

**Course:** Research Methods and Tools (RMT) — Assignment 4 (Experiment Execution, Results and Discussion)  
**Institution:** Astana IT University, School of Software Engineering  
**Authors:** Rustam Seitzhan, Abzal Bakhytov, Adilzhan Amangeldi  
**Group:** SE-2414  
**Release Tag:** `assignment-4`

---

## 1. Overview and Purpose

This repository contains the complete reproducibility package, raw prediction logs, statistical analysis scripts, figures, and qualitative error audit for our study of factual question answering across four released checkpoints of the **OLMo-2 1B** model lineage (`allenai/OLMo-2-0425-1B`, `-SFT`, `-DPO`, and `-Instruct`) plus a development-fitted **relation-prior baseline** on the **PopQA** benchmark ($n = 3,003$ shared test questions across $2,855$ distinct entities and $16$ relation types).

---

## 2. Repository Structure

```text
.
├── README.md                                 # Main documentation and reproduction guide
├── README.txt                                # Original Assignment 3 quick-run note
├── LICENSE                                   # MIT License
├── requirements.txt                          # Pinned Python dependencies for analysis replay
├── Assignment3_OLMo2_PopQA_Colab.ipynb       # Google Colab T4 GPU execution notebook
├── data/
│   └── processed/
│       ├── popqa_clean.jsonl                 # Cleaned PopQA dataset (14,264 rows)
│       ├── dev.jsonl                         # Subject-grouped development split (2,891 rows)
│       └── test_sample.jsonl                 # Shared evaluation sample (3,003 rows; SHA-256: 9c664749...)
├── raw_results/                              # Unchanged JSONL predictions for all 5 conditions
│   ├── preds_prior_baseline.jsonl
│   ├── preds_base.jsonl
│   ├── preds_sft.jsonl
│   ├── preds_dpo.jsonl
│   └── preds_instruct.jsonl
├── results/                                  # Assignment 3 results directory (mirrored for compatibility)
├── scripts/
│   └── reproduce.py                          # One-command statistical & figure regeneration script
├── analysis/
│   ├── source_sha256.json                    # SHA-256 manifest of all input data and prediction files
│   ├── summary.json                          # Complete machine-readable summary and environment record
│   └── error_audit/
│       ├── build_error_audit.py              # Script generating the 32-case qualitative error audit
│       ├── error_audit_cases.csv             # Full 32 audited errors with aliases, outputs, and rationales
│       └── error_audit_summary.csv           # Error taxonomy counts and percentages (Table B1)
├── tables/                                   # Derived CSV tables matching the Assignment 4 report
│   ├── main_results.csv                      # Table 2: Main outcomes & 95% subject-cluster bootstrap CIs
│   ├── accuracy_comparisons.csv              # Table 3: All 10 pairwise exact McNemar + Holm comparisons
│   ├── adjacent_stage_outcomes.csv           # 9 adjacent-stage tests across Correct / Incorrect / Abstain
│   ├── subgroup_results.csv                  # Popularity tertile outcomes (Rare, Medium, Popular)
│   ├── popularity_gaps.csv                   # Popular-minus-rare accuracy gaps with 95% cluster CIs
│   ├── per_relation_results.csv              # Breakdown across all 16 Wikidata relation types
│   ├── transitions.csv                       # Paired outcome transition matrices (e.g. DPO -> Instruct)
│   ├── scorer_sensitivity.csv                # Evaluator sensitivity (whole-phrase vs exact vs substring)
│   ├── logistic_regression.csv               # Clustered logistic regression (12,012 observations)
│   └── runtime_profile.csv                   # Execution time profile for each pipeline stage
├── figures/
│   ├── figure1_main_outcomes.png             # Figure 1: Accuracy CIs and 3-way outcome distribution
│   ├── figure2_popularity.png                # Figure 2: Accuracy across within-relation popularity tertiles
│   └── figure3_transitions.png               # Figure 3: DPO to Instruct paired transition matrix
├── src/                                      # Core pipeline modules (scorer, LLM runner, data prep)
└── logs/                                     # Execution, unit-test, and reproduction verification logs
    ├── reproduction_check.json               # Automated replay verification report
    ├── llm_analysis.log                      # Original Assignment 3 statistical output log
    ├── test_scorer.log                       # Unit test log (7/7 passed)
    └── verify_analysis.log                   # Synthetic verification fixture log
```

---

## 3. Installation and Exact Reproduction Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.12 and Python 3.13 across Windows x86_64 and Linux x86_64)

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Regenerate all tables, statistical tests, and figures
From the repository root, run:
```bash
python scripts/reproduce.py
python analysis/error_audit/build_error_audit.py
```

### Expected Output
`python scripts/reproduce.py` verifies the SHA-256 hashes of all 13 data/prediction files, rescores all 15,015 saved predictions against `src/scorer.py`, runs 10,000 subject-cluster bootstrap replicates (`seed=42` over 2,855 unique subjects), computes exact McNemar tests with Holm step-down adjustments, fits the clustered logistic regression model (`n = 12,012`), and writes all CSV tables in `tables/` and figures in `figures/`:

| Condition | Correct $n$ | Accuracy % | 95% Cluster CI | Incorrect $n$ (%) | Abstain $n$ (%) | Selective Acc % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Relation prior** | 219 | 7.29% | [6.38, 8.24] | 2,784 (92.71%) | 0 (0.00%) | 7.29% |
| **Base** | 430 | 14.32% | [13.05, 15.66] | 1,832 (61.01%) | 741 (24.68%) | 19.01% |
| **SFT** | 294 | 9.79% | [8.70, 10.93] | 689 (22.94%) | 2,020 (67.27%) | 29.91% |
| **DPO** | 343 | 11.42% | [10.25, 12.64] | 1,670 (55.61%) | 990 (32.97%) | 17.04% |
| **Instruct** | 344 | 11.46% | [10.28, 12.68] | 1,326 (44.16%) | 1,333 (44.39%) | 20.60% |

---

## 4. Oral Defence "Pick-a-Number" Evidence Map

| Reported Claim / Number | Source File & Script |
| :--- | :--- |
| Test sample size $n = 3,003$ and SHA-256 `9c664749...` | `data/processed/test_sample.jsonl`; `analysis/source_sha256.json` |
| Base accuracy $430 / 3,003 = 14.32\%$ `[13.05, 15.66]` | `raw_results/preds_base.jsonl`; `tables/main_results.csv` |
| Base $\rightarrow$ SFT: $38$ gains, $174$ losses ($\Delta = -4.53\text{ pp}, p_{\text{Holm}} = 5.45\times 10^{-21}$) | `tables/accuracy_comparisons.csv`; `scripts/reproduce.py` |
| DPO $\rightarrow$ Instruct: $407$ errors become abstentions | `tables/transitions.csv`; `figures/figure3_transitions.png` |
| Popularity gap: Base $18.80\text{ pp}$, SFT $10.97\text{ pp}$, DPO $14.72\text{ pp}$ ($14.73\text{ pp}$ rounded), Instruct $13.62\text{ pp}$ | `tables/subgroup_results.csv`; `tables/popularity_gaps.csv` |
| Clustered logistic regression (`lp` coef $= 1.2034, p < 0.001$) | `tables/logistic_regression.csv`; `logs/llm_analysis.log` |
| 32 audited failure cases across all 4 checkpoints | `analysis/error_audit/error_audit_cases.csv`; `analysis/error_audit/error_audit_summary.csv` |
