"""Unit tests for the scorer. Inputs are hand-written strings (NOT model outputs)."""
from scorer import *

def test_extract_stops_at_next_question():
    assert extract_answer("Paris\nQuestion: What is X?\nAnswer: Y") == "Paris"
    assert extract_answer("  \n Paris Question: next") == "Paris"
    assert extract_answer("") == ""

def test_normalize():
    assert normalize("The Académie, Française!") == "academie francaise"

def test_boundary_vs_substring_false_positive():
    # alias "pol" (a real PopQA alias of 'politician') must not match inside 'Polish'
    assert match_substring("Polish composer", ["politician", "pol"]) is True    # substring: false positive
    assert match_boundary("Polish composer", ["politician", "pol"]) is False   # boundary: correct

def test_boundary_true_positive():
    assert match_boundary("He was a politician.", ["politician", "pol"]) is True
    assert match_boundary("The Politician", ["politician"]) is True

def test_abstention_detection():
    for t in ["I don't know", "I'm not sure.", "Unknown", "I cannot answer that", ""]:
        assert label(t, ["Paris"])["outcome"] == "abstain"

def test_three_way_labels():
    assert label("Paris", ["Paris", "City of Light"])["outcome"] == "correct"
    assert label("London", ["Paris"])["outcome"] == "incorrect"
    assert label("Paris\nQuestion: Capital of Peru?\nAnswer: Lima", ["Lima"])["outcome"] == "incorrect"  # only 1st answer counts

def test_abstain_beats_match():
    assert label("I don't know, maybe Paris", ["Paris"])["outcome"] == "abstain"
