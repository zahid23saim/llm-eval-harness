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
