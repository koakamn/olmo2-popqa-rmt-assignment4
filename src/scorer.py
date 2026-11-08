"""Answer extraction, matching and three-way outcome labelling (correct / incorrect / abstain)."""
import re, unicodedata

ABSTAIN_PATTERNS = [
    r"\bi (do not|don't|dont) know\b", r"\bi'?m not sure\b", r"\bi am not sure\b", r"\bnot sure\b",
    r"\bi (cannot|can't|can not|couldn't|could not) (answer|say|find|determine|provide)\b",
    r"\bno (information|data|record)\b", r"\bunknown\b", r"\bunclear\b", r"\bunable to\b",
    r"\bi have no (idea|knowledge)\b", r"\bit is not (known|clear)\b", r"\bn/?a\b",
]
_ABSTAIN = re.compile("|".join(ABSTAIN_PATTERNS), re.I)
_STOP_MARKERS = re.compile(r"(?i)(?:^|\n|\s)(question|q)\s*:")

def extract_answer(raw: str) -> str:
    """Keep only the first answer line; base models often keep generating new 'Question:' blocks."""
    if raw is None:
        return ""
    raw = raw.strip()
    m = _STOP_MARKERS.search(raw)
    if m:
        raw = raw[:m.start()]
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[0] if lines else ""

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\u0400-\u04ff ]+", " ", s)       # drop punctuation (keeps Cyrillic letters)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def match_boundary(pred: str, aliases) -> bool:
    """Whole-phrase match: alias must appear as complete word(s) in the prediction."""
    p = f" {normalize(pred)} "
    return any(a_n and f" {a_n} " in p for a_n in (normalize(a) for a in aliases))

def match_substring(pred: str, aliases) -> bool:
    """Plain substring match (used by the original PopQA code); kept only to quantify false positives."""
    p = normalize(pred)
    return any(a_n and a_n in p for a_n in (normalize(a) for a in aliases))

def is_abstention(answer: str) -> bool:
    return answer == "" or bool(_ABSTAIN.search(answer))

def label(raw_output: str, aliases) -> dict:
    ans = extract_answer(raw_output)
    if is_abstention(ans):
        outcome = "abstain"
    elif match_boundary(ans, aliases):
        outcome = "correct"
    else:
        outcome = "incorrect"
    return {"answer": ans, "outcome": outcome}
