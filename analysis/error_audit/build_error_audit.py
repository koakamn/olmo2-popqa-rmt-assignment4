"""Generates the 32-case qualitative error audit tables (Appendix B) from the archived predictions."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Explicit human-verified taxonomy and annotations for the 32 deterministically sampled failure cases
AUDIT_CASES = [
    # Base (8 cases)
    (1, "base", 1223902, "Different occupation or genre", "Different attribute",
     "Predicts 'Actor' instead of 'journalist' for Herlyn Espinal; wrong occupation association."),
    (2, "base", 3436182, "Different named person", "Different person",
     "Predicts 'Raja Sena' instead of 'Vijay Sena' as father of Ballal Sena; dynasty entity confusion."),
    (3, "base", 4682521, "Different named person", "Different person",
     "Predicts 'Ian Anderson' instead of 'Sashadhar Mukherjee' for Leader; title ambiguity across works named 'Leader'."),
    (4, "base", 5453507, "Different named person", "Different person",
     "Predicts 'David Winning' instead of 'Gene Reynolds' for 'Pilot'; extreme title ambiguity ('Pilot' episode)."),
    (5, "base", 5921512, "Different named person", "Different person",
     "Predicts 'John Hodge' instead of 'Ed Burns' / 'David Simon' for The Detail."),
    (6, "base", 2775591, "Different named person", "Different person",
     "Predicts 'Robert Avsec' instead of 'Neil Simon' for Max Dugan Returns."),
    (7, "base", 1635529, "Potential lexical or short-form mismatch", "Possible lexical mismatch",
     "Predicts 'Catholic' whereas gold aliases list 'Catholic Church' / 'Catholicism'; adjectival short form."),
    (8, "base", 2057677, "Potential lexical or short-form mismatch", "Possible lexical mismatch",
     "Predicts 'Babice' whereas gold requires disambiguated 'Babice, Chrzanów County'."),
    # SFT (8 cases)
    (9, "sft", 1094065, "Broader attribute than gold", "Broader attribute",
     "Predicts 'Author' for Jilly Cooper whereas gold lists 'journalist', 'writer', 'novelist'; near-synonym omitted from alias set."),
    (10, "sft", 5919221, "Broader attribute than gold", "Broader attribute",
     "Predicts coarse genre 'Drama' instead of specific 'legal drama' for The Court."),
    (11, "sft", 2977722, "Different location or administrative relation", "Geographic mismatch",
     "Predicts 'Jordan' instead of 'Yemen' for Khabb wa ash Sha'af District."),
    (12, "sft", 778850, "Different named person", "Different person",
     "Predicts 'Michael Almereyda' instead of 'Stephen Daldry' as director of The Hours."),
    (13, "sft", 5760432, "Different named person", "Different person",
     "Predicts 'Nikhil Advani' instead of 'Bernardo Bertolucci' for Partner; cross-film title ambiguity."),
    (14, "sft", 2886500, "Different named person", "Different person",
     "Predicts 'Wolfgang Amadeus Mozart' instead of 'Arturo Toscanini' / 'Arrigo Boito' for Nerone."),
    (15, "sft", 5951965, "Different named person", "Different person",
     "Predicts 'Sue Townsend' instead of 'Jim Cartwright' for The Rise and Fall of Little Voice."),
    (16, "sft", 1762839, "Potential lexical or short-form mismatch", "Possible lexical mismatch",
     "Predicts 'businessman' whereas gold aliases include 'businessperson', 'entrepreneur', 'politician'."),
    # DPO (8 cases)
    (17, "dpo", 2360504, "Broader attribute than gold", "Broader attribute",
     "Predicts hypernym 'musician' instead of specific instrument role 'guitarist' for Robby Krieger."),
    (18, "dpo", 1087089, "Different occupation or genre", "Different attribute",
     "Predicts 'Western' instead of 'soap opera' for Dallas; title ambiguity (1950 Western film vs TV series)."),
    (19, "dpo", 1250393, "Different named person", "Different person",
     "Predicts 'Don Rich' instead of 'Jessica Lange' / 'William D. Wittliff' for Country."),
    (20, "dpo", 740702, "Different named person", "Different person",
     "Predicts 'John McLeod' instead of 'Howard Deutch' as director of The Great Outdoors."),
    (21, "dpo", 5265423, "Different named person", "Different person",
     "Predicts 'John August' instead of 'Abe Forsythe' as screenwriter for Ned."),
    (22, "dpo", 1565610, "Different named person", "Different person",
     "Predicts 'John Powell' instead of 'Charlie Chaplin' as composer of Pay Day."),
    (23, "dpo", 2861779, "Different named person", "Different person",
     "Predicts 'John Green' instead of 'Dean Koontz' as author of Lightning."),
    (24, "dpo", 6430975, "Broader attribute than gold", "Broader attribute",
     "Predicts hypernym 'musician' instead of 'record producer' / 'songwriter' for Teddy Riley."),
    # Instruct (8 cases)
    (25, "instruct", 1901465, "Different occupation or genre", "Different attribute",
     "Predicts 'Lawyer' instead of 'screenwriter' for Ivo Perilli."),
    (26, "instruct", 1437335, "Different occupation or genre", "Different attribute",
     "Predicts 'Pop' instead of 'hip hop music' for Ayo."),
    (27, "instruct", 1663488, "Different named person", "Different person",
     "Predicts chronicler 'Niketas Choniates' instead of father 'Konstantinos X Doukas' for Michael VII Doukas."),
    (28, "instruct", 520298, "Different named person", "Different person",
     "Predicts 'John Singleton' instead of 'Danny DeVito' as producer of Even Money."),
    (29, "instruct", 4583461, "Different location or administrative relation", "Geographic mismatch",
     "Predicts country 'Italy' instead of historical department/province 'Marengo' / 'Province of Alessandria'."),
    (30, "instruct", 4114074, "Different named person", "Different person",
     "Predicts 'Jerome Kern' instead of 'David Bowie' as composer of Five Years."),
    (31, "instruct", 57908, "Different named person", "Different person",
     "Predicts 'George Furth' instead of 'Oscar Hammerstein II' as author of Carousel."),
    (32, "instruct", 71982, "Possible alternate occupation", "Possible alternate occupation",
     "Predicts 'model' whereas gold record lists 'actor' / 'actress' for Kiko Mizuhara (who is both a model and actress)."),
]


def main():
    out_dir = ROOT / "analysis/error_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    test = {
        json.loads(line)["id"]: json.loads(line)
        for line in open(ROOT / "data/processed/test_sample.jsonl", encoding="utf-8")
    }
    preds = {
        cond: {
            json.loads(line)["id"]: json.loads(line)
            for line in open(ROOT / f"results/preds_{cond}.jsonl", encoding="utf-8")
        }
        for cond in ["base", "sft", "dpo", "instruct"]
    }

    case_rows = []
    for audit_id, cond, item_id, tax_cat, short_cat, rationale in AUDIT_CASES:
        t_row = test[item_id]
        p_row = preds[cond][item_id]
        assert p_row["outcome"] == "incorrect", f"Expected incorrect outcome for {cond} {item_id}"
        case_rows.append({
            "audit_id": audit_id,
            "condition": cond,
            "item_id": item_id,
            "relation": t_row["prop"],
            "pop_tertile": t_row["pop_tertile"],
            "question": t_row["question"],
            "gold_example": t_row["answers"][0],
            "all_gold_aliases": json.dumps(t_row["answers"], ensure_ascii=False),
            "saved_answer": p_row["answer"],
            "raw_output": p_row["raw_output"],
            "taxonomy_category": tax_cat,
            "short_category": short_cat,
            "rationale": rationale,
        })

    df_cases = pd.DataFrame(case_rows)
    df_cases.to_csv(out_dir / "error_audit_cases.csv", index=False, encoding="utf-8")

    summary_order = [
        ("Different named person", "The requested person is replaced by a different name absent from the supplied gold aliases. Incorrect entity association or title ambiguity."),
        ("Different occupation or genre", "The predicted occupation or genre differs from the benchmark attribute rather than being a broader version of it."),
        ("Different location or administrative relation", "The response supplies a different country or administrative level inconsistent with the benchmark answer."),
        ("Broader attribute than gold", "The output is a broader category related to a supplied gold alias but lacks the benchmark's specificity."),
        ("Potential lexical or short-form mismatch", "A shortened or closely related lexical form is absent from the supplied alias list."),
        ("Possible alternate occupation", "The response gives a real-world occupation of a multi-occupation entity that is absent from the Wikidata-derived gold alias list."),
    ]
    sum_rows = []
    for cat_name, desc in summary_order:
        cnt = int((df_cases["taxonomy_category"] == cat_name).sum())
        sum_rows.append({
            "category": cat_name,
            "definition_and_cause": desc,
            "count": cnt,
            "pct_of_32": round(100.0 * cnt / len(df_cases), 2),
        })
    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_dir / "error_audit_summary.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(df_cases)} verified error audit cases and {len(df_sum)} summary categories.")


if __name__ == "__main__":
    main()
