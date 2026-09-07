#!/usr/bin/env python3
"""A tiny, dependency-free harness for scoring LLM answers against a gold set.

Give it a gold file (questions with known answers and a per-question match rule)
and an answers file (the model's answers keyed by id). It prints an accuracy
score and, for every miss, the question, the expected answer, and what the model
actually produced -- so you can see *why* a run failed, not just that it did.

Standard library only. Run it directly as a CLI:

    python llm_eval.py examples/gold.json examples/answers.json

Exit status is 0 when everything passes and 1 when any item fails, so it drops
straight into a CI step or a pre-commit hook.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop trailing punctuation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".!?,;:")


def first_number(text: str):
    """Pull the first standalone number out of a string, or None if there isn't one.

    A leading ``-`` counts as a negative sign only when it is a real sign (start of
    string, or after a space or punctuation) -- a hyphen inside a token such as
    ``GPT-4`` is not read as minus four. Digits glued to a preceding word or hyphen
    are skipped, so ``"GPT-4 scored 90"`` yields ``90``. A leading decimal point is
    handled too, so ``".5"`` is ``0.5``, not ``5``.
    """
    m = re.search(r"(?<![\w.\-])-?\d*\.?\d+", text.replace(",", ""))
    return float(m.group()) if m else None


def is_correct(item: dict, model_answer: str) -> bool:
    """Judge one answer using the item's own match rule.

    match types:
      exact    -> normalized model answer equals the normalized gold answer
      contains -> normalized gold answer appears as a whole word/phrase in the
                  normalized model answer
      numeric  -> first numbers within `tol` of each other (default tol 0.0)
    """
    kind = item.get("match", "exact")
    gold = item["answer"]

    if kind == "exact":
        return normalize(model_answer) == normalize(gold)

    if kind == "contains":
        # Whole-word / phrase presence, so gold "8" does not match "18" and "no"
        # does not match "know". This is still a *literal* presence test: it does
        # not understand negation, so gold "1969" matches both "in 1969" and "not
        # in 1969". Use exact/numeric when a wrong answer could merely embed the
        # gold string.
        gold_norm = normalize(gold)
        if not gold_norm:
            return True
        pattern = r"(?<!\w)" + re.escape(gold_norm) + r"(?!\w)"
        return re.search(pattern, normalize(model_answer)) is not None

    if kind == "numeric":
        got = first_number(model_answer)
        want = first_number(gold)
        if got is None or want is None:
            return False
        return abs(got - want) <= item.get("tol", 0.0)

    raise ValueError(f"unknown match type: {kind!r}")


# --------------------------------------------------------------------------- #
# loading + validation
# --------------------------------------------------------------------------- #
VALID_MATCHES = {"exact", "contains", "numeric"}


def load_gold(path: str) -> list:
    """Load and validate a gold file. Bad eval data causes more wrong
    conclusions than bad models do, so fail loudly on it."""
    with open(path, encoding="utf-8") as fh:
        gold = json.load(fh)
    if not isinstance(gold, list) or not gold:
        raise ValueError("gold file must be a non-empty JSON array")

    seen = set()
    for item in gold:
        if "id" not in item or "answer" not in item:
            raise ValueError(f"every item needs 'id' and 'answer': {item!r}")
        if item["id"] in seen:
            raise ValueError(f"duplicate id: {item['id']!r}")
        seen.add(item["id"])
        kind = item.get("match", "exact")
        if kind not in VALID_MATCHES:
            raise ValueError(f"unknown match type {kind!r} on id {item['id']!r}")
        if kind == "numeric" and first_number(str(item["answer"])) is None:
            raise ValueError(f"numeric item {item['id']!r} has no number in its answer")
    return gold


def load_answers(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        answers = json.load(fh)
    if not isinstance(answers, dict):
        raise ValueError("answers file must be a JSON object of {id: answer}")
    return answers


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(gold: list, answers: dict):
    """Return (accuracy, failures). `failures` is a list of dicts describing
    each miss with enough context to act on it."""
    passed, failures = 0, []
    for item in gold:
        model_answer = answers.get(item["id"], "")
        if is_correct(item, model_answer):
            passed += 1
        else:
            failures.append({
                "id": item["id"],
                "question": item.get("question", ""),
                "expected": item["answer"],
                "got": model_answer or "(no answer)",
            })
    return passed / len(gold), failures


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score LLM answers against a gold set.")
    parser.add_argument("gold", help="path to the gold JSON file")
    parser.add_argument("answers", help="path to the answers JSON file")
    parser.add_argument("--quiet", action="store_true", help="print only the accuracy line")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    answers = load_answers(args.answers)
    acc, failures = score(gold, answers)

    print(f"accuracy: {acc:.0%}  ({len(gold) - len(failures)}/{len(gold)})")
    if failures and not args.quiet:
        print()
        for f in failures:
            print(f"FAIL {f['id']}: {f['question']}")
            print(f"     expected: {f['expected']!r}")
            print(f"     got:      {f['got']!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
