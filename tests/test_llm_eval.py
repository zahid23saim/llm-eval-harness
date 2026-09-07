"""Tests for the scoring harness. Run with: python -m pytest -q"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_eval as le  # noqa: E402


# --- matching -------------------------------------------------------------- #
def test_exact_match_ignores_case_and_trailing_punctuation():
    item = {"answer": "Paris", "match": "exact"}
    assert le.is_correct(item, "paris")
    assert le.is_correct(item, "  PARIS.  ")
    assert not le.is_correct(item, "Paris, France")   # exact means exact


def test_contains_finds_answer_inside_a_sentence():
    item = {"answer": "1969", "match": "contains"}
    assert le.is_correct(item, "The moon landing was in 1969.")
    assert not le.is_correct(item, "It was in 1970.")


def test_contains_matches_whole_words_not_substrings():
    # gold "8" must not match inside "18"; "no" must not match inside "know"
    assert not le.is_correct({"answer": "8", "match": "contains"}, "a byte has 18 bits")
    assert not le.is_correct({"answer": "no", "match": "contains"}, "I do not know")
    # but a genuine whole-word occurrence still matches
    assert le.is_correct({"answer": "no", "match": "contains"}, "the answer is no")


@pytest.mark.xfail(strict=True, reason="contains is a literal presence test and "
                   "cannot detect negation; use exact/numeric for answers that "
                   "could merely embed the gold string")
def test_contains_cannot_detect_negation():
    item = {"answer": "1969", "match": "contains"}
    assert not le.is_correct(item, "it was not in 1969")


def test_first_number_does_not_read_a_hyphen_in_a_token_as_minus():
    # "GPT-4" is a model name, not -4; the real number is 90
    assert le.first_number("GPT-4 scored 90") == 90.0


def test_first_number_handles_a_leading_decimal_point():
    assert le.first_number(".5") == 0.5


def test_first_number_keeps_genuine_negatives():
    assert le.first_number("the delta was -5 points") == -5.0


def test_first_number_is_none_without_digits():
    assert le.first_number("seven") is None


def test_first_number_parses_scientific_notation():
    assert le.first_number("learning rate 1e-9") == 1e-9


def test_first_number_keeps_thousands_comma_but_not_decimal_comma():
    assert le.first_number("1,000 tokens") == 1000.0
    # an ambiguous decimal comma must not be silently turned into 314
    assert le.first_number("3,14") == 3.0


def test_non_string_answers_do_not_crash():
    # a gold answer written as a bare JSON number must not blow up normalize
    assert le.normalize(1969) == "1969"
    assert le.is_correct({"answer": 1969, "match": "contains"}, "it happened in 1969")
    assert le.is_correct({"answer": 8, "match": "numeric"}, "the count is 8")


def test_cli_warns_about_answer_ids_not_in_gold(tmp_path, capsys):
    gold = _write(tmp_path, "gold.json", [{"id": "a", "answer": "x", "match": "exact"}])
    answers = _write(tmp_path, "answers.json", {"a": "x", "ghost": "y"})
    le.main([gold, answers])
    assert "ghost" in capsys.readouterr().err


def test_numeric_respects_tolerance():
    item = {"answer": "3.14", "match": "numeric", "tol": 0.01}
    assert le.is_correct(item, "about 3.14159")
    assert not le.is_correct(item, "roughly 3.2")


def test_numeric_rejects_answers_with_no_number():
    item = {"answer": "8", "match": "numeric"}
    assert not le.is_correct(item, "there is no number here")


def test_unknown_match_type_raises():
    with pytest.raises(ValueError):
        le.is_correct({"answer": "x", "match": "fuzzy"}, "x")


# --- scoring --------------------------------------------------------------- #
def test_score_reports_accuracy_and_failures():
    gold = [
        {"id": "a", "question": "?", "answer": "yes", "match": "contains"},
        {"id": "b", "question": "?", "answer": "no", "match": "contains"},
    ]
    acc, failures = le.score(gold, {"a": "the answer is yes", "b": "the answer is yes"})
    assert acc == 0.5
    assert len(failures) == 1 and failures[0]["id"] == "b"


def test_missing_answer_counts_as_a_failure():
    gold = [{"id": "a", "question": "?", "answer": "yes", "match": "contains"}]
    acc, failures = le.score(gold, {})
    assert acc == 0.0
    assert failures[0]["got"] == "(no answer)"


# --- validation ------------------------------------------------------------ #
def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_load_gold_rejects_duplicate_ids(tmp_path):
    path = _write(tmp_path, "gold.json", [
        {"id": "a", "answer": "1", "match": "exact"},
        {"id": "a", "answer": "2", "match": "exact"},
    ])
    with pytest.raises(ValueError):
        le.load_gold(path)


def test_load_gold_rejects_numeric_item_without_a_number(tmp_path):
    path = _write(tmp_path, "gold.json", [
        {"id": "a", "answer": "not a number", "match": "numeric"},
    ])
    with pytest.raises(ValueError):
        le.load_gold(path)


def test_load_gold_accepts_a_valid_file(tmp_path):
    path = _write(tmp_path, "gold.json", [
        {"id": "a", "answer": "8", "match": "numeric"},
        {"id": "b", "answer": "Paris", "match": "contains"},
    ])
    gold = le.load_gold(path)
    assert len(gold) == 2
