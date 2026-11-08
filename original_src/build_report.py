import json, re, textwrap, os
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, NextPageTemplate, Preformatted, KeepTogether, ListFlowable, ListItem)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FD = "/usr/share/fonts/truetype/dejavu/"
for n, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"), ("Serif-I", "DejaVuSerif-Italic.ttf"),
             ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
             ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")

ROOT = ".."
NAVY = colors.HexColor("#1f3b5c")
body = ParagraphStyle("body", fontName="Serif", fontSize=8.6, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=4.5)
small = ParagraphStyle("small", parent=body, fontSize=7.4, leading=10, alignment=0, spaceAfter=2)
cell = ParagraphStyle("cell", parent=body, fontSize=6.9, leading=8.8, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Serif-B")
cap = ParagraphStyle("cap", parent=body, fontName="Serif-I", fontSize=7.6, leading=10, alignment=TA_CENTER, spaceAfter=7)
h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=12.2, leading=15, textColor=NAVY, spaceBefore=11, spaceAfter=5)
h2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=9.4, leading=12, textColor=NAVY, spaceBefore=6, spaceAfter=3)
h3 = ParagraphStyle("h3", fontName="Serif-B", fontSize=8.6, leading=11.5, spaceBefore=3, spaceAfter=2)
title = ParagraphStyle("title", fontName="Sans-B", fontSize=17, leading=22, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8)
sub = ParagraphStyle("sub", fontName="Serif", fontSize=10, leading=14, alignment=TA_CENTER, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=body, spaceAfter=2)
mono = ParagraphStyle("mono", fontName="Mono", fontSize=6.1, leading=7.4)

WORDS = {}      # section -> word count (for sections 3-8)
def P(t, s=body): return Paragraph(t, s)
def H1(t): return Paragraph(t, h1)
def H2(t): return Paragraph(t, h2)
def B(items, st=bullet):
    return ListFlowable([ListItem(Paragraph(i, st), leftIndent=11) for i in items], bulletType="bullet", start="•", leftIndent=11, bulletFontSize=7)
def wc(*texts):
    return sum(len(re.sub(r"<[^>]+>", " ", t).split()) for t in texts)

def tbl(data, widths, header=True, font=cell, zebra=True, extra=None):
    rows = [[Paragraph(str(c), cellb if (header and i == 0) else font) for c in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")))
    if zebra:
        for i in range(2, len(data), 2): st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f8fb")))
    if extra: st += extra
    t.setStyle(TableStyle(st)); return t

def code(path, maxw=118):
    out = []
    for line in open(path).read().rstrip("\n").split("\n"):
        out += textwrap.wrap(line, maxw, subsequent_indent="      ", replace_whitespace=False, drop_whitespace=False) or [""]
    return Preformatted("\n".join(out), mono)
def logblock(path, maxw=118, limit=None):
    txt = open(path).read().rstrip("\n")
    lines = [l for l in txt.split("\n") if "Warning" not in l and "warn" not in l.lower()]
    if limit: lines = lines[:limit]
    out = []
    for line in lines: out += textwrap.wrap(line, maxw, subsequent_indent="      ") or [""]
    return Preformatted("\n".join(out), mono)

# ---------- real numbers ----------
st = json.load(open(f"{ROOT}/results/data_stats.json")); bl = json.load(open(f"{ROOT}/results/baseline_prior_results.json"))
test = pd.read_json(f"{ROOT}/data/processed/test_sample.jsonl", lines=True); dev = pd.read_json(f"{ROOT}/data/processed/dev.jsonl", lines=True)
pred = pd.read_json(f"{ROOT}/results/preds_prior_baseline.jsonl", lines=True).merge(test[["id", "subj", "prop", "question", "answers", "pop_tertile"]], on="id")
n_correct = int((pred.outcome == "correct").sum())
modes = json.load(open(f"{ROOT}/results/baseline_prior_modes.json"))

# ---------- page templates ----------
W, Hh = A4
def footer(c, d):
    c.saveState(); c.setFont("Sans", 7); c.setFillColor(colors.HexColor("#666666"))
    c.drawString(2 * cm, 1.1 * cm, "Assignment 3 - How Do Training Stages Shape What LLMs Know? - SE-2414")
    c.drawRightString(W - 2 * cm, 1.1 * cm, f"Page {d.page}"); c.restoreState()
def footer_l(c, d):
    c.saveState(); c.setFont("Sans", 7); c.setFillColor(colors.HexColor("#666666"))
    c.drawString(2 * cm, 1.0 * cm, "Assignment 3 - How Do Training Stages Shape What LLMs Know? - SE-2414")
    c.drawRightString(Hh - 2 * cm, 1.0 * cm, f"Page {d.page}"); c.restoreState()
doc = BaseDocTemplate(f"{ROOT}/outputs/Assignment3_Seitzhan_Amangeldi_Bakhytov.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="Assignment 3: Methodology and Core Practical Implementation", author="Rustam Seitzhan, Adilzhan Amangeldi, Abzal Bakhytov")
fp = Frame(2 * cm, 1.8 * cm, W - 4 * cm, Hh - 3.6 * cm, id="p"); fl = Frame(2 * cm, 1.6 * cm, Hh - 4 * cm, W - 3.4 * cm, id="l")
doc.addPageTemplates([PageTemplate("portrait", [fp], onPage=footer, pagesize=A4), PageTemplate("land", [fl], onPage=footer_l, pagesize=landscape(A4))])
TW = W - 4 * cm
S = []

# ================= TITLE =================
S += [Spacer(1, 1.2 * cm), P("Assignment 3", sub), Paragraph("Methodology and Core Practical Implementation", title), Spacer(1, 4),
      Paragraph("How Do Training Stages Shape What LLMs Know? A Comparative Study of Factual Knowledge in Pre-trained, Fine-tuned and Aligned Language Models", ParagraphStyle("t2", parent=sub, fontName="Serif-B", fontSize=11.5, leading=15.5)),
      Spacer(1, 6),
      tbl([["Students", "Rustam Seitzhan, Adilzhan Amangeldi, Abzal Bakhytov"], ["Group", "SE-2414"], ["Research area", "Artificial Intelligence"],
           ["Research topic", "Knowledge acquisition mechanisms in Large Language Models (LLMs)"], ["Core track (Part 7)", "Computer Vision / NLP / Algorithmic Project - NLP evaluation pipeline"],
           ["Builds on", "Assignment 1 (problem and gap) and Assignment 2 (research question, aim, objectives, hypotheses)"]], [4.1 * cm, TW - 4.1 * cm], header=False, zebra=False,
          extra=[("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dfe8f3"))]), Spacer(1, 6)]

S.append(P("<b>Reading guide.</b> Sections 3-8 are the methodology section. Section 9 reports the practical implementation: everything marked <b>executed</b> was run on real data and its output is reproduced in Appendix E. "
           "The language-model inference step is implemented and its prompts were generated on real data, but the four checkpoints have <b>not yet been run</b>; this report therefore contains no model results, and Section 9.8 states what remains.", small))

# ================= 1-2 =================
S.append(H1("1. Final Research Title"))
S.append(P("<b>How Do Training Stages Shape What LLMs Know? A Comparative Study of Factual Knowledge in Pre-trained, Fine-tuned and Aligned Language Models.</b> The title is unchanged from Assignment 2."))
S.append(H1("2. Primary Research Question, Aim and Objectives"))
S.append(P("<b>Primary research question.</b> How does the factual question-answering performance of open-weight LLMs change across the training-pipeline stages (pre-trained base model, supervised instruction-tuned model and preference-aligned model), and which stage accounts for the largest share of the factual knowledge the model exhibits?"))
S.append(P("<b>Subquestion 1.</b> Do the effects of these stages differ between frequently mentioned (popular) and rarely mentioned (rare) facts? &nbsp; <b>Subquestion 2.</b> Does alignment mainly change the share of correct answers, or the balance between confidently incorrect answers and abstentions?"))
S.append(P("<b>Aim.</b> To investigate how the pre-training, supervised fine-tuning and alignment stages of the LLM training pipeline shape the factual knowledge exhibited by open-weight language models."))
S.append(B(["<b>O1.</b> Identify open-weight model families that publish comparable pre-trained, instruction-tuned and aligned versions, and summarize how each stage is documented in their technical reports.",
            "<b>O2.</b> Assess which established factual question-answering and knowledge-probing benchmarks are suitable for measuring factual knowledge across these versions.",
            "<b>O3.</b> Compare the factual QA accuracy of the pre-trained, fine-tuned and aligned versions of the same model family on the same questions.",
            "<b>O4.</b> Analyse how stage effects differ for popular and rare facts, and between correct, incorrect and abstaining answers.",
            "<b>O5.</b> Evaluate which stage accounts for the largest share of exhibited factual knowledge, and interpret the result against previous studies."]))
S.append(P("<b>Hypotheses (unchanged).</b> H1: the base model already reaches accuracy close to its fine-tuned and aligned versions (cf. Zhou et al., 2023; Gekhman et al., 2024). H2: alignment lowers the share of confidently incorrect answers and raises abstentions without a comparable gain in accuracy (cf. Ouyang et al., 2022). H3: rare facts have lower accuracy than popular facts at every stage, and no stage removes the gap (cf. Kandpal et al., 2023)."))
S.append(P("<b>Refinements made while operationalising (no change of topic or question).</b> (a) Assignment 2 named Llama 2 as an example family. Llama 2 chat models are released as one combined SFT-plus-RLHF checkpoint, which is exactly the weakness noted for Candidate Question 1, so the primary lineage is <b>OLMo 2 1B</b>, whose base, SFT, DPO and final checkpoints are all public (Team OLMo et al., 2024). "
           "(b) The three stages of Assignment 2 map to <b>Base</b> (pre-trained), <b>SFT</b> (supervised instruction-tuned) and <b>DPO</b> (preference-aligned; Rafailov et al., 2023). The final checkpoint (<b>Instruct</b>, after an additional RLVR stage) is analysed only as an exploratory fourth condition. (c) \"Popular\" and \"rare\" are defined from Wikipedia page views (Section 5).", small))

# ================= 3 Part 1 =================
S.append(H1("3. Research Approach and Design"))
t31 = ("<b>Approach.</b> The study is quantitative and computational. It evaluates existing open-weight models instead of training new ones and produces numerical evidence "
       "(rates of correct, incorrect and abstaining answers) that can be compared statistically.")
t32 = ("<b>Design.</b> A comparative benchmark with a paired (repeated-measures) structure. The same 3,003 factual questions are put to four published checkpoints of one lineage, OLMo 2 1B, "
       "which differ only in the training stage applied: pre-trained base, supervised fine-tuning (SFT), preference optimisation with DPO (the aligned stage) and the final instruct model (Team OLMo et al., 2024; Lambert et al., 2024). "
       "Because every question is answered by every checkpoint, stage differences are measured within items, which removes between-question variance.")
t33 = ("<b>Unit of analysis.</b> One answer: the closed-book response of one checkpoint to one PopQA question, labelled correct, incorrect or abstain. Answers are clustered by subject entity in all inference.")
t34 = ("<b>Why this design fits.</b> The research question asks how performance changes across stages and which stage contributes most. A paired comparison of adjacent stages on identical items, "
       "with popularity as a moderator, answers this directly (O3-O5), and the lineage fixes architecture, size and tokenizer so that training stage is the only systematic difference. "
       "The evidence produced is item-level categorical outcomes plus documentary evidence of how each stage was trained.")
t35 = ("<b>Rejected alternative.</b> A white-box probing study (linear probes, causal tracing) would show where a fact is stored, but it needs heavy GPU memory and specialist tooling "
       "(why Candidate Question 2 was rejected) and does not show whether a stage changes what the model <i>exhibits</i>.")
for t in (t31, t32, t33, t34, t35): S.append(P(t))
WORDS["3"] = wc(t31, t32, t33, t34, t35)

# ================= 4 Part 2 =================
S.append(H1("4. Sample, Dataset, Cases and Research Context"))
t41 = ("<b>Research material.</b> <i>PopQA</i> (Mallen et al., 2022): 14,267 entity-centred questions generated from Wikidata triples (Vrandečić &amp; Krötzsch, 2014), each with a list of gold answer aliases, one of 16 relation types and the number of "
       "Wikipedia page views of the subject, which serves as popularity measure. The systems studied are the four OLMo-2-0425-1B checkpoints (Base, SFT, DPO, Instruct; Allen Institute for AI, 2025). An optional OLMo 2 7B run is a replication.")
t42 = ("<b>Inclusion and exclusion.</b> A question is included if it has a unique id, a non-empty subject and at least one gold alias. Three rows without a subject were excluded (14,267 to 14,264). No question is removed after seeing model output; dev questions are never evaluated.")
t43 = ("<b>Size and rationale.</b> The data were split by subject into a development set (2,891 questions, 20% of subjects) and a test frame of 11,373 questions, from which 3,003 questions (rare 1,013, medium 1,002, popular 988) form the evaluation sample. "
       "A simulation of the exact McNemar test (Appendix E) gives about 96% power to detect a 3-point accuracy difference at 20% discordance and a 95% interval half-width of about 1.6 points; "
       "per tertile (about 1,000 items) power for 3 points is about 55%, so smaller tertile effects will be reported as inconclusive. Four checkpoints with three prompt sets mean 36,036 short generations, feasible on one free-tier GPU.")
t44 = ("<b>Selection strategy.</b> Stratified random sampling by relation and within-relation popularity tertile with proportional allocation (seed 42), after a subject-grouped dev/test split so that no subject appears on both sides; 1,492 subjects have more than one question, which an item-level split would leak.")
t45 = ("<b>Access.</b> PopQA is public (MIT licence) and the OLMo 2 weights are public (Apache 2.0); no permission or participants are needed, so access is realistic.")
t46 = ("<b>Bias.</b> Popularity is strongly confounded with relation type (Figure 2): capital and colour questions concern popular subjects, sport and place-of-birth questions rare ones. Pooled tertiles would therefore compare relations rather than popularity, so tertiles are computed within each relation. "
       "Wikidata coverage is also uneven, so results generalise to PopQA-style entity facts only.")
for t in (t41, t42, t43, t44, t45, t46): S.append(P(t))
WORDS["4"] = wc(t41, t42, t43, t44, t45, t46)

# ================= 5 Part 3 =================
S.append(H1("5. Variables, Constructs, Measures, Instruments and Metrics"))
t51 = ("<b>Independent variable.</b> Training stage, a within-item factor: Base, SFT, DPO (confirmatory) and Instruct (exploratory). <b>Moderator.</b> Popularity: log10 of subject page views, grouped into within-relation tertiles (rare, medium, popular).")
t52 = ("<b>Dependent variables</b> (per answer: correct, incorrect or abstain). Accuracy = share correct; error rate = share answered incorrectly, i.e. confidently wrong; abstention rate = share declined; "
       "selective accuracy = correct / (correct + incorrect). The <i>stage effect</i> is the paired difference of a rate between adjacent stages.")
t53 = ("<b>Controls.</b> Held fixed: architecture, parameter count, tokenizer, items, prompt text and order, decoding (greedy, at most 16 new tokens). Modelled: relation type as fixed effect. "
       "Monitored confounders: the answer prior (a no-model baseline), fit between prompt format and checkpoint, and possible benchmark contamination.")
t54 = ("<b>Instruments.</b> (i) PopQA as benchmark; (ii) a scorer that extracts the first answer line, normalises text (lower case, accents, punctuation, articles) and requires a whole-phrase match with a gold alias; "
       "(iii) an abstention lexicon (\"I don't know\", \"not sure\", empty output and similar); (iv) a reference baseline that predicts the most frequent object of each relation, fitted on the development split only; "
       "(v) a manual audit of 100 outputs per checkpoint by two raters (Cohen's kappa; Cohen, 1960).")
t55 = ("<b>Contribution to objectives.</b> Accuracy and error rates serve O3 and O5; tertile-wise and outcome-wise rates serve O4; the baseline gives the floor for O5; "
       "benchmark criteria and documentation tables serve O1-O2.")
for t in (t51, t52, t53, t54, t55): S.append(P(t))
WORDS["5"] = wc(t51, t52, t53, t54, t55)

# ================= 6 Part 4 =================
S.append(H1("6. Data Collection, Preparation and Experimental Procedure"))
t61 = "The procedure runs in chronological order; steps 1-3 are already executed (Section 9)."
S.append(P(t61))
steps = ["<b>Acquire and verify data.</b> Download PopQA from the authors' repository and record its SHA-256 hash.",
         "<b>Prepare data.</b> Parse alias lists, drop rows without a subject, compute within-relation popularity tertiles, make the subject-grouped dev/test split and the stratified test sample (fixed seed).",
         "<b>Build the instruments.</b> Implement the scorer with unit tests and fit the reference baseline on the dev split.",
         "<b>Set up inference.</b> Google Colab GPU, Transformers (Wolf et al., 2020) and PyTorch, fp16, greedy decoding; library versions are logged.",
         "<b>Pilot on dev only.</b> Run each checkpoint on 300 dev questions, check format failures, and let two raters label 100 outputs per checkpoint. Refine the abstention lexicon on these outputs, then <b>freeze</b> code and the decisions in Appendix A before the test sample is touched.",
         "<b>Main run.</b> Four checkpoints on the 3,003 test questions with prompt set 1 (confirmatory); prompt sets 2 and 3, which differ in the few-shot exemplars, are robustness replications. Every checkpoint receives the identical raw-text prompt, without a chat template; "
         "a secondary run uses native chat format zero-shot to check that the raw prompt does not disadvantage post-trained checkpoints.",
         "<b>Store.</b> Raw output, extracted answer and label are written to one JSONL schema per condition and never edited by hand.",
         "<b>Analyse</b> with the frozen script (Section 7)."]
S.append(ListFlowable([ListItem(Paragraph(s, bullet), leftIndent=14) for s in steps], bulletType="1", leftIndent=14, bulletFontSize=8))
t62 = ("<b>Fairness and consistency.</b> Only the weights change between conditions; items, prompt, order, batch size and decoding are identical. Exemplars come from the dev split only, so the test questions never appear in a prompt. "
       "Because half-precision batch inference can be non-deterministic, 200 items are re-run to estimate the noise. <b>Reproducibility record:</b> data hash, seeds, model identifiers, library versions, parameters and scripts (Appendix D-F).")
S.append(P(t62))
WORDS["6"] = wc(t61, *steps, t62)

# ================= 7 Part 5 =================
S.append(H1("7. Data Analysis Plan"))
t71 = ("All rates carry Wilson 95% intervals (Wilson, 1927). Stage effects are paired differences between adjacent stages on identical items, with 95% percentile intervals from a bootstrap that resamples <i>subjects</i> (2,000 resamples; Efron &amp; Tibshirani, 1993), "
       "and an exact McNemar test for the discordant pairs (McNemar, 1947; Dietterich, 1998). The six confirmatory tests (correct, incorrect and abstain for Base to SFT and SFT to DPO) are Holm-adjusted (Holm, 1979); everything else is exploratory and labelled so.")
S.append(P(t71))
objs = ["<b>O1</b> - Document analysis: an extraction sheet per checkpoint (stage, data, method, licence, public weights) compared in one table; a family qualifies only if all stages are public.",
        "<b>O2</b> - Criterion-by-criterion comparison of PopQA with TriviaQA, Natural Questions and LAMA (Joshi et al., 2017; Kwiatkowski et al., 2019; Petroni et al., 2019): closed-book, entity-level, aliases, popularity metadata, licence. Plus the matcher audit (Section 9).",
        "<b>O3</b> - Accuracy per stage and paired differences. H1 is supported only if the 95% interval of every accuracy difference to the later stages lies inside a pre-specified margin of &plusmn;3 points, roughly twice the expected interval half-width; margins of 2 and 5 points are sensitivity checks.",
        "<b>O4</b> - Tertile-wise outcome shares per stage; logistic regression of correctness on log-popularity, stage and their interaction with relation fixed effects and subject-clustered standard errors (Seabold &amp; Perktold, 2010); paired tests on error and abstention rates (H2), and the rare-minus-popular gap per stage (H3).",
        "<b>O5</b> - Decomposition of final accuracy into Base accuracy plus the two stage increments, each with its interval; Base share of final accuracy, also measured above the prior-baseline floor; counts of items gained and lost between stages; interpretation against Zhou et al. (2023), Gekhman et al. (2024) and Ouyang et al. (2022)."]
S.append(B(objs))
t72 = ("<b>Error analysis and leakage control.</b> Incorrect answers are sampled by relation and inspected (wrong entity, wrong type, formatting, alias miss). Leakage is prevented by the subject-grouped split, dev-only exemplars and dev-only scorer tuning. "
       "The baseline uses the same scorer and items and serves as floor, not competitor.")
S.append(P(t72))
WORDS["7"] = wc(t71, *objs, t72)

# ================= 8 Part 6 =================
S.append(H1("8. Research Quality, Ethics, Privacy and Reproducibility"))
t81 = ("<b>Validity.</b> String-match QA measures <i>exhibited</i> knowledge, not internal storage; conclusions are limited to that construct. Matching validity is protected by a whole-phrase rule (Section 9 shows that plain substring matching inflates accuracy) and by the two-rater audit with a target kappa of at least 0.80. "
       "Internal validity comes from identical items, prompts and decoding. External validity is limited to one lineage and size; the optional 7B run tests this. A known threat is contamination: Wikipedia is in pre-training and PopQA has been public since 2022 (Lewis et al., 2021), so pre-training and post-training cannot be fully separated from memorised benchmark text; this is reported as a limitation.")
t82 = ("<b>Reliability and robustness.</b> Three prompt sets quantify prompt sensitivity, a re-run of 200 items estimates hardware noise, and conclusions are called robust only if they hold in all prompt sets.")
t83 = ("<b>Ethics and privacy.</b> No human participants, no consent needed. The data are public knowledge-base facts, but many questions name real people (including their parents), so wrong model answers about them are not republished as claims; reports show aggregates and a few illustrative items about public figures. "
       "Only public data and public weights are sent to Colab, and no confidential material, source code or personal data is uploaded. AI assistants are used only for language checking under course rules; the team verifies and owns every decision. "
       "Licences (MIT, Apache 2.0) are respected; raw outputs stay in a private team repository and are deleted after the course archive period.")
for t in (t81, t82, t83): S.append(P(t))
WORDS["8"] = wc(t81, t82, t83)
TOTAL = sum(WORDS.values())
print("WORD COUNTS", WORDS, "TOTAL", TOTAL)

# ================= 9 Part 7 =================
S.append(PageBreak())
S.append(H1("9. Core Practical Component / Initial Implementation"))
S.append(P("<b>Track:</b> Computer Vision / NLP / Algorithmic Project. <b>Supports:</b> O2 (benchmark audit), and the infrastructure for O3-O5. All numbers in 9.2-9.7 come from code in the repository (Appendix D) run on the real PopQA file; none come from a language model."))
S.append(H2("9.1 What was built"))
S.append(Image(f"{ROOT}/figures/fig1_pipeline.png", width=TW * 0.97, height=TW * 0.97 * 0.575)); S.append(P("Figure 1. Pipeline. Green parts were executed and are evidenced in the appendix; the yellow part is implemented and awaits its first GPU run.", cap))
S.append(H2("9.2 Dataset evidence and preprocessing output (executed)"))
S.append(tbl([["Quantity", "Value", "Quantity", "Value"],
              ["Source file (SHA-256, first 16 hex)", st["sha256_raw"][:16] + "...", "Rows raw / after cleaning", f"{st['raw_rows']:,} / {st['after_cleaning']:,} (3 without subject dropped)"],
              ["Distinct subjects / relations", f"{st['n_subjects']:,} / {st['n_relations']}", "Dev / test frame (subject-grouped)", f"{st['dev_rows']:,} / {st['test_rows']:,}"],
              ["Evaluation sample", f"{st['test_sample_rows']:,}", "Sample by tertile (rare / medium / popular)", "1,013 / 1,002 / 988"],
              ["Median page views per tertile", "176 / 1,081.5 / 10,365", "Seed", "42"]], [4.2 * cm, 3.7 * cm, 4.6 * cm, TW - 12.5 * cm]))
S.append(Spacer(1, 4))
S.append(Image(f"{ROOT}/figures/fig2_popularity_by_relation.png", width=TW * 0.84, height=TW * 0.84 * 0.5)); S.append(P("Figure 2. Popularity (log10 page views) by relation type on the cleaned data: the medians differ by roughly two orders of magnitude, which is why popularity tertiles are computed within relations.", cap))
S.append(H2("9.3 Scorer and test evidence (executed)"))
S.append(P("The scorer extracts the first answer line (base models often continue with a new \"Question:\"), normalises it, and labels it <i>abstain</i> if it is empty or matches the abstention lexicon, otherwise <i>correct</i> if a gold alias occurs as a complete phrase, otherwise <i>incorrect</i>. "
           "Seven unit tests with hand-written strings pass (Appendix E.2); they cover answer extraction, normalisation, the substring false positive, abstention detection and the rule that an abstention beats a match."))
S.append(H2("9.4 Reference baseline: first quantitative evaluation (executed)"))
S.append(P(f"A no-model baseline predicts, for every relation, the most frequent object of the development split (for example <i>occupation</i> to \"{modes['occupation']}\", <i>sport</i> to \"{modes['sport']}\"). It was fitted on 2,891 dev questions and evaluated by the same scorer on the 3,003 test questions through the same predictions schema that the LLM runner uses, "
           f"so the harness has been exercised end to end. Overall accuracy is <b>{bl['accuracy']*100:.2f}%</b> ({n_correct} of 3,003); by tertile rare {bl['by_tertile']['rare']*100:.1f}%, medium {bl['by_tertile']['medium']*100:.1f}%, popular {bl['by_tertile']['popular']*100:.1f}%, i.e. no popularity gradient, as expected for a rule that ignores the subject."))
S.append(Image(f"{ROOT}/figures/fig3_prior_baseline_by_relation.png", width=TW * 0.84, height=TW * 0.84 * 0.5)); S.append(P("Figure 3. Accuracy of the no-model baseline by relation type. Low-cardinality relations (sport, religion, country, occupation) are partly guessable without any knowledge.", cap))
S.append(H2("9.5 Input / output examples from the real run (executed)"))
rows = [["Question (input)", "Gold aliases (first 3)", "Baseline output", "Label"]]
def ex(sub, n): return sub.head(n)
sel = pd.concat([pred[(pred.outcome == "correct")].sample(3, random_state=1), pred[(pred.outcome == "incorrect") & pred.prop.isin(["occupation", "genre", "country"])].sample(3, random_state=1)])
for r in sel.itertuples():
    rows.append([escape(r.question), escape("; ".join(r.answers[:3])), escape(r.answer), r.outcome])
S.append(tbl(rows, [5.8 * cm, 5.2 * cm, 4.3 * cm, TW - 15.3 * cm])); S.append(Spacer(1, 4))
S.append(H2("9.6 Error and example analysis: matcher audit (executed)"))
from scorer import normalize, match_substring, match_boundary
fp_rows = []
for r in pred.itertuples():
    if match_substring(r.answer, r.answers) and not match_boundary(r.answer, r.answers):
        pn = normalize(r.answer); hit = [a for a in r.answers if normalize(a) and normalize(a) in pn][0]
        fp_rows.append([escape(r.question), escape(r.answer), escape(hit), "substring: correct / whole-phrase: incorrect"])
n_fp = len(fp_rows)
S.append(P(f"Running the original PopQA-style substring rule next to the whole-phrase rule on the real baseline output exposes a defect: substring matching reports {bl['substring_acc']*100:.2f}% ({round(bl['substring_acc']*3003)} hits) against {bl['boundary_acc']*100:.2f}% ({n_correct} hits), "
           f"i.e. <b>{n_fp} false positives</b>, about 6% of all baseline hits. The cause is short aliases such as country codes (\"IT\", \"ES\", \"CA\") that occur inside ordinary words (\"uni<b>t</b>ed\" contains \"it\"): {bl['share_questions_with_alias_le3_chars']*100:.1f}% of all questions have an alias of three or fewer characters. Examples of the {n_fp} cases:"))
S.append(tbl([["Question", "Output", "Alias matched inside output", "Verdict"]] + fp_rows[:5], [5.4 * cm, 3.5 * cm, 3.3 * cm, TW - 12.2 * cm]))
S.append(P("Decision: all later evaluation uses the whole-phrase rule. Long free-text outputs from real models will make this effect larger than in the baseline, so the audit in Appendix G also checks matching errors in both directions.", small))
S.append(H2("9.7 Verification of the analysis code (executed, synthetic input)"))
S.append(P("The analysis script cannot yet be run on model predictions. It was therefore verified on <b>synthetic test fixtures</b> built from the baseline labels with a known injected effect; these are test inputs, not results. "
           "Injecting 150 additional correct answers (+4.99 points) is recovered as a difference of +5.00 points with 95% interval [+4.2, +5.8], 150 gained and 0 lost items and p &lt; 1e-10; a symmetric swap of 60 gains and 60 losses gives a difference of 0.0, interval [-0.8, +0.7] and p = 1.0, "
           "and is classified as inside the &plusmn;3-point margin. The popularity-by-stage logistic model with clustered standard errors fits and converges (Appendix E.5)."))
S.append(H2("9.8 Limitations, failures found, and next steps"))
S.append(P("<b>What did not work as expected, discovered during the first implementation:</b>", small))
S.append(B([f"<b>Substring matching inflates accuracy</b> ({n_fp} false positives in 3,003 baseline answers), so it was replaced by whole-phrase matching.",
            "<b>Accuracy is partly guessable.</b> A rule that ignores the question reaches 67.5% on sport, 42.6% on religion, 29.1% on country and 28.1% on occupation, but about 0% on director or screenwriter. A pooled accuracy would hide this, so results will be reported per relation and as accuracy above the prior baseline.",
            "<b>Popularity is confounded with relation type</b> (Figure 2), so pooled tertiles were replaced by within-relation tertiles.",
            "<b>Some relations are too small</b> in the sample (colour n = 6, mother n = 41), so per-relation estimates for them will not be interpreted.",
            "<b>The abstention lexicon is validated only on hand-written strings.</b> No gold answer triggers it (0 of 14,264), but real checkpoints may abstain in unexpected wording; the dev-only audit exists for this reason.",
            "<b>The single \"I don't know\" exemplar</b> is a real, answerable dev question, which may raise abstention rates; prompt sets 2 and 3 vary it as a sensitivity check."], small))
S.append(P("<b>Unresolved / not yet done.</b> The four OLMo-2 checkpoints have not yet been run: prompt construction was dry-run on real data (Appendix C) and the runner compiles, but no model output exists, so this report makes no claim about stage effects. "
           "<b>Next:</b> (1) run the 300-question dev pilot and the two-rater audit; (2) freeze code and Appendix A; (3) run Base, SFT and DPO on the test sample with prompt set 1, then Instruct and prompt sets 2-3; (4) run the frozen analysis and extend this section with the real tables.", small))
S.append(H2("9.9 Optional bonus (survey / interview)"))
S.append(P("Not attempted. The study is purely technical and no human feedback is needed to answer the research question; the bonus cannot replace Part 7."))

# ================= 10 Alignment table (landscape) =================
S.append(NextPageTemplate("land")); S.append(PageBreak())
S.append(H1("10. Methodology Alignment Table"))
LW = Hh - 4 * cm
rows = [["Research objective", "Data / evidence needed", "Source / sample / dataset", "Collection / instrument", "Analysis method", "Evidence for RQ"],
 ["O1. Identify open-weight families with comparable pre-trained, instruction-tuned and aligned versions; summarise stage documentation.",
  "Per checkpoint: training stage, data, method, licence, weights public or not.", "OLMo 2 family (1B, 7B, 13B, 32B) technical report and model cards; Tulu 3 post-training report.",
  "Document-extraction sheet; inclusion rule: all stages public with open weights.", "Structured comparison table; qualitative synthesis.", "Shows that stage differences are due to training stage only (internal validity) and justifies the lineage."],
 ["O2. Assess which factual-QA benchmarks suit measuring knowledge across the versions.", "Benchmark properties; matcher behaviour on real data.",
  "PopQA vs TriviaQA, Natural Questions, LAMA; PopQA file (14,264 questions after cleaning).", "Criteria checklist (closed-book, entity-level, aliases, popularity metadata, licence); matcher audit (Section 9.6).",
  "Criterion-by-criterion comparison; substring vs whole-phrase audit.", "Confirms PopQA measures exhibited facts and supports popularity analysis; fixes the scoring rule."],
 ["O3. Compare factual QA accuracy of pre-trained, fine-tuned and aligned versions on the same questions.", "Correct / incorrect / abstain per item and checkpoint.",
  "3,003 test questions x 4 checkpoints x 3 prompt sets; reference baseline.", "run_llm.py (fixed prompt, greedy) + scorer.py; JSONL schema.", "Wilson CIs; paired differences with subject-clustered bootstrap CI; exact McNemar with Holm; H1 margin of 3 points.",
  "Size and uncertainty of each stage's accuracy change: the direct answer to \"how does performance change\"."],
 ["O4. Analyse stage effects for popular vs rare facts and for correct, incorrect and abstaining answers.", "Outcome per item with popularity (page views) and relation.",
  "Same sample, within-relation tertiles; s_pop metadata.", "Scorer outcomes joined with PopQA metadata.", "Tertile-wise outcome shares; logit with popularity x stage, relation fixed effects, clustered SE; paired tests on error and abstention rates.",
  "Answers Subquestions 1 and 2 and tests H2 and H3."],
 ["O5. Evaluate which stage accounts for the largest share of exhibited knowledge; interpret against previous studies.", "Stage increments, baseline floor, gained / lost items, published findings.",
  "Outputs of O3 and O4; baseline results (7.29%); Zhou et al. (2023), Gekhman et al. (2024), Ouyang et al. (2022).", "Frozen analysis script; transition counts.",
  "Decomposition: Base accuracy + increment(SFT) + increment(DPO) with intervals; share above the prior floor; comparison with literature.", "Identifies the stage with the largest share: the answer to the primary research question."]]
S.append(tbl(rows, [4.6 * cm, 3.7 * cm, 4.6 * cm, 4.2 * cm, 5.2 * cm, LW - 22.3 * cm]))
S.append(NextPageTemplate("portrait")); S.append(PageBreak())

# ================= 11 References =================
S.append(H1("11. References"))
refst = ParagraphStyle("ref", parent=body, fontSize=7.6, leading=10.4, alignment=0, leftIndent=13, firstLineIndent=-13, spaceAfter=2.6)
refs = ["Allen Institute for AI. (2025). <i>OLMo-2-0425-1B</i> [Model card and checkpoint table]. Hugging Face. https://huggingface.co/allenai/OLMo-2-0425-1B",
 "Cohen, J. (1960). A coefficient of agreement for nominal scales. <i>Educational and Psychological Measurement, 20</i>(1), 37-46.",
 "Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. <i>Neural Computation, 10</i>(7), 1895-1923.",
 "Efron, B., &amp; Tibshirani, R. J. (1993). <i>An introduction to the bootstrap</i>. Chapman &amp; Hall.",
 "Gekhman, Z., Yona, G., Aharoni, R., Eyal, M., Feder, A., Reichart, R., &amp; Herzig, J. (2024). Does fine-tuning LLMs on new knowledge encourage hallucinations? In <i>Proceedings of EMNLP 2024</i> (pp. 7765-7784). ACL. https://doi.org/10.18653/v1/2024.emnlp-main.444",
 "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics, 6</i>(2), 65-70.",
 "Joshi, M., Choi, E., Weld, D. S., &amp; Zettlemoyer, L. (2017). TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In <i>Proceedings of ACL 2017</i> (pp. 1601-1611).",
 "Kandpal, N., Deng, H., Roberts, A., Wallace, E., &amp; Raffel, C. (2023). Large language models struggle to learn long-tail knowledge. In <i>Proceedings of ICML 2023</i> (PMLR Vol. 202, pp. 15696-15707).",
 "Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., et al. (2019). Natural Questions: A benchmark for question answering research. <i>Transactions of the Association for Computational Linguistics, 7</i>, 453-466.",
 "Lambert, N., Morrison, J., Pyatkin, V., et al. (2024). <i>Tulu 3: Pushing frontiers in open language model post-training</i>. arXiv:2411.15124.",
 "Lewis, P., Stenetorp, P., &amp; Riedel, S. (2021). <i>Question and answer test-train overlap in open-domain question answering datasets</i>. arXiv:2008.02637.",
 "Mallen, A., Asai, A., Zhong, V., Das, R., Hajishirzi, H., &amp; Khashabi, D. (2022). <i>When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories</i>. arXiv:2212.10511.",
 "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. <i>Psychometrika, 12</i>(2), 153-157.",
 "Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... Lowe, R. (2022). Training language models to follow instructions with human feedback. <i>Advances in Neural Information Processing Systems, 35</i>. arXiv:2203.02155.",
 "Petroni, F., Rocktaschel, T., Riedel, S., Lewis, P., Bakhtin, A., Wu, Y., &amp; Miller, A. (2019). Language models as knowledge bases? In <i>Proceedings of EMNLP-IJCNLP 2019</i> (pp. 2463-2473).",
 "Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., &amp; Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. <i>Advances in Neural Information Processing Systems, 36</i>. arXiv:2305.18290.",
 "Seabold, S., &amp; Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In <i>Proceedings of the 9th Python in Science Conference</i> (pp. 92-96).",
 "Team OLMo, Walsh, P., Soldaini, L., Groeneveld, D., et al. (2024). <i>2 OLMo 2 Furious</i>. arXiv:2501.00656.",
 "Vrandecic, D., &amp; Krotzsch, M. (2014). Wikidata: A free collaborative knowledgebase. <i>Communications of the ACM, 57</i>(10), 78-85.",
 "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. <i>Journal of the American Statistical Association, 22</i>(158), 209-212.",
 "Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. In <i>Proceedings of EMNLP 2020: System Demonstrations</i> (pp. 38-45).",
 "Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., ... Levy, O. (2023). LIMA: Less is more for alignment. <i>Advances in Neural Information Processing Systems, 36</i>. arXiv:2305.11206."]
S += [Paragraph(r, refst) for r in refs]

# ================= 12 Appendix =================
S.append(PageBreak()); S.append(H1("12. Appendix"))
S.append(H2("Appendix A. Decisions fixed before the test sample is evaluated"))
S.append(tbl([["Item", "Decision", "Reason"],
 ["Confirmatory contrasts", "Base to SFT and SFT to DPO; outcomes correct, incorrect, abstain (6 tests, Holm-adjusted); alpha = 0.05", "Limits false positives from multiple testing; Instruct, relations and 7B are exploratory"],
 ["H1 criterion", "95% bootstrap interval of the accuracy difference inside +-3 points; sensitivity at 2 and 5", "About twice the expected interval half-width (1.6 points), so the question is decidable at n = 3,003"],
 ["Prompt regime", "Raw-text few-shot prompt, identical for all checkpoints; native chat zero-shot as secondary", "Equal conditions; checks format disadvantage of the base model"],
 ["Prompt sets", "Set 1 confirmatory; sets 2 and 3 vary exemplars; robust only if direction agrees in all three", "Quantifies prompt sensitivity"],
 ["Decoding", "Greedy, max 16 new tokens, fp16, batch 16, torch seed 42", "Deterministic as far as hardware allows"],
 ["Matching rule", "Normalised whole-phrase match; abstention beats match; first answer line only", "Section 9.6 audit"],
 ["Bootstrap / seed", "2,000 resamples of subjects; seed 42", "Clustered uncertainty"],
 ["Scorer tuning", "On dev-split outputs only; target kappa >= 0.80 between raters", "No leakage from test outputs"]], [3.2 * cm, 8.0 * cm, TW - 11.2 * cm]))
S.append(H2("Appendix B. Dataset specification"))
fields = [["Field", "Meaning"], ["id, subj, prop, obj", "Question id, subject name, relation type (16), gold object"], ["question", "Natural-language question, e.g. \"What is Scooter Braun's occupation?\""],
          ["possible_answers", "Gold aliases (JSON list; median 2, up to 190 per question)"], ["s_pop", "Wikipedia page views of the subject (min 2, median 975, max 15.1 million)"],
          ["s_uri", "Wikidata entity of the subject, used for grouped splitting and clustering"], ["pop_tertile, split, in_test_sample", "Derived: within-relation tertile, dev/test, membership of the 3,003 sample"]]
S.append(tbl(fields, [4.6 * cm, TW - 4.6 * cm])); S.append(Spacer(1, 4))
rel = test.groupby("prop").agg(n=("id", "size"), med=("s_pop", "median")).reset_index()
bres = {d["prop"]: d["acc"] for d in bl["by_relation"]}
S.append(tbl([["Relation", "Sample n", "Median page views", "Baseline accuracy"]] + [[r.prop, int(r.n), f"{r.med:,.0f}", f"{bres[r.prop]*100:.1f}%"] for r in rel.sort_values("n", ascending=False).itertuples()],
             [4.6 * cm, 3 * cm, 4 * cm, TW - 11.6 * cm]))
S.append(H2("Appendix C. Prompt template (output of the dry run on real dev and test data)"))
S.append(logblock(f"{ROOT}/logs/dry_run_prompt.log"))
S.append(H2("Appendix D. Source code (repository src/)"))
for fn, desc in [("scorer.py", "answer extraction, matching and labelling"), ("prepare_data.py", "cleaning, tertiles, split, sample"), ("baseline_prior.py", "reference baseline and matcher audit"),
                 ("analyze.py", "rates, paired tests, bootstrap, logit"), ("run_llm.py", "prompt builder and checkpoint runner (not yet executed)")]:
    S.append(P(f"<b>D. {fn}</b> - {desc}", small)); S.append(code(f"{ROOT}/src/{fn}")); S.append(Spacer(1, 5))
S.append(P("Also in the repository: <b>test_scorer.py</b> (7 unit tests), <b>power.py</b> (power simulation), <b>verify_analysis.py</b> (synthetic-fixture verification), <b>make_figures.py</b>, <b>build_report.py</b>.", small))
S.append(H2("Appendix E. Execution logs (unedited)"))
for tag, fn in [("E.1 prepare_data.py", "prepare_data.log"), ("E.2 pytest test_scorer.py", "test_scorer.log"), ("E.3 baseline_prior.py", "baseline_prior.log"),
                ("E.4 power.py (planning values under assumed discordance 0.10-0.30; not findings)", "power.log"), ("E.5 verify_analysis.py (synthetic fixtures)", "verify_analysis.log"), ("E.6 environment", "environment.log")]:
    S.append(P(f"<b>{tag}</b>", small)); S.append(logblock(f"{ROOT}/logs/{fn}")); S.append(Spacer(1, 5))
S.append(H2("Appendix F. Repository structure and reproduction"))
S.append(Preformatted("""rmt/
  data/popQA.tsv                      raw file (github.com/AlexTMallen/adaptive-retrieval, MIT)
  data/processed/                     popqa_clean.jsonl, dev.jsonl, test_sample.jsonl
  src/                                scripts listed in Appendix D
  results/                            data_stats.json, baseline_prior_results.json, preds_<condition>.jsonl
  logs/  figures/

cd src
python prepare_data.py                      # data cleaning, split, sample
python -m pytest -v test_scorer.py          # scorer tests
python baseline_prior.py                    # reference baseline
python power.py ; python verify_analysis.py
python run_llm.py --dry-run                 # print prompt (no GPU)
python run_llm.py --condition base --prompt_set 1      # on a GPU; same for sft, dpo, instruct
python analyze.py base,sft,dpo              # frozen analysis""", mono))
S.append(H2("Appendix G. Manual audit protocol and risk register"))
S.append(tbl([["Risk", "Mitigation"],
 ["Scorer mislabels real outputs (alias miss, wording of abstention)", "Two raters label 100 dev outputs per checkpoint blind to the automatic label; kappa between raters and between consensus and scorer; refine rules on dev outputs only; report residual disagreement"],
 ["Base model does not follow the answer format", "Extraction cuts at the next \"Question:\"; format-failure rate reported per checkpoint; native-format secondary run"],
 ["Prompt choice drives the result", "Three exemplar sets; conclusions robust only if consistent"],
 ["Popularity confounded with relation", "Within-relation tertiles; relation fixed effects"],
 ["Benchmark contamination", "Reported as limitation; claims limited to exhibited knowledge on PopQA"],
 ["Compute limits", "1B models, 16-token outputs; 7B replication optional"],
 ["Hardware non-determinism", "200-item re-run; report disagreement rate"]], [6.4 * cm, TW - 6.4 * cm]))

doc.build(S)
print("PDF built")
