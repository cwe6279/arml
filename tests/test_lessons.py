"""lessons/lessons.json: every lesson's cases still hold, and every prompt rule has a lesson
behind it or is listed as unproven. Open lessons run as expected failures until fixed."""
import json
import re
from pathlib import Path

import pytest

from arml import StreamParser, format_rules, lint

ROOT = Path(__file__).resolve().parents[1]
DOC = json.loads((ROOT / "lessons" / "lessons.json").read_text())
LESSONS = DOC["lessons"]
UNPROVEN = DOC["prompt_rules_without_a_lesson"]["rules"]
KINDS = {"model", "server", "harness", "lint", "spec", "data"}
SAMPLE_TOOLS = [{"name": "weather", "description": "Current weather.",
                 "parameters": {"city": {"type": "string", "required": True}}}]
# Both versions of the prompt: some rules are worded differently for spoken replies.
PROMPTS = [format_rules(SAMPLE_TOOLS, code=True), format_rules(SAMPLE_TOOLS, code=True, spoken=True)]
PROMPT = "\n".join(PROMPTS)
# The rules are the "- " lines after the tool list.
RULE_LINES = sorted({line[2:] for p in PROMPTS for line in p.split("RESPONSE FORMAT", 1)[1].splitlines()
                     if line.startswith("- ")})
CASES = [(lesson, i, case) for lesson in LESSONS for i, case in enumerate(lesson.get("cases", []))]


def _mark(item):
    """An open lesson's prompt rules, or a case marked "open", are expected to fail until fixed.
    strict: once it passes, the test says so, and the "open" mark has to come off."""
    return [pytest.mark.xfail(strict=True, reason="open")] if item.get("status") == "open" or item.get("open") else []


def test_every_lesson_is_complete_and_ids_are_unique():
    assert len({lesson["id"] for lesson in LESSONS}) == len(LESSONS)
    for lesson in LESSONS:
        assert re.fullmatch(r"L\d{2,}", lesson["id"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", lesson["added"])
        assert lesson["kind"] in KINDS and lesson["what"] and lesson["source"] and lesson["enforced_by"]
        assert lesson.get("status", "fixed") in ("fixed", "open")


@pytest.mark.parametrize("lesson,i,case", [pytest.param(l, i, c, id=f"{l['id']}-{i}", marks=_mark(c)) for l, i, c in CASES])
def test_case(lesson, i, case):
    if "stream" in case:
        sp, shown = StreamParser(), []
        for chunk in case["stream"]:
            shown += [e.text for e in sp.feed(chunk) if e.kind == "text"]
        shown += [e.text for e in sp.close() if e.kind == "text"]
        assert "".join(shown) == case["visible"]
        return
    codes = [x.code for x in lint(case["response"], spoken=case.get("spoken", False), acted=case.get("acted", False))]
    for code in case.get("expect", []):
        assert code in codes, f"{lesson['id']}: expected {code}, lint gave {codes}"
    for code in case.get("expect_not", []):
        assert code not in codes, f"{lesson['id']}: {code} should not fire, lint gave {codes}"


def test_lint_codes_a_lesson_names_exist():
    known = set(re.findall(r"\b[A-Z]{2,}(?:_[A-Z]+)+\b|\b(?:EMPTY|UNCLOSED|SHORTHAND)\b",
                           (ROOT / "arml" / "lint.py").read_text()))
    for lesson in LESSONS:
        for code in lesson["enforced_by"].get("lint", []):
            assert code in known, f"{lesson['id']} names lint code {code}, which lint.py does not have"


@pytest.mark.parametrize("lesson", [pytest.param(l, id=l["id"], marks=_mark(l))
                                    for l in LESSONS if l["enforced_by"].get("prompt")])
def test_the_prompt_still_has_the_rules_a_lesson_relies_on(lesson):
    for rule in lesson["enforced_by"]["prompt"]:
        assert rule in PROMPT, f"{lesson['id']} relies on the prompt rule {rule!r}, which format_rules() no longer has"


def test_every_prompt_rule_has_a_lesson_or_is_listed_as_unproven():
    backed = [r for lesson in LESSONS if lesson.get("status") != "open" for r in lesson["enforced_by"].get("prompt", [])]
    for line in RULE_LINES:
        assert any(r in line for r in backed + UNPROVEN), \
            f"prompt rule with no lesson behind it: {line!r}. Add a lesson, list it as unproven, or remove it."
    for rule in UNPROVEN:
        assert rule in PROMPT, f"{rule!r} is listed as unproven but is not in format_rules() any more: remove it from the list"
