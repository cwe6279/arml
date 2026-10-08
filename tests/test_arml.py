"""ARML: visible vs silent text, streaming, shorthand tags, lint and canonical form.

Several cases are real failures from a voice assistant (running on a small home device)
that spoke text as it streamed in.
"""
import os
import random

import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arml import BlockType, StreamParser, canonical, is_clean, lint, parse

VOICE_NOTE = (
    "Got it. I'll note the reservation.\n\n<think>User booked a Denver flight; dates are now "
    "March 3 and 4. Reservation details are in their email. Stop the pending "
    "flight search.</think>\n\n<output>Should I have the agent read the email?</output>"
)

CORPUS = [
    VOICE_NOTE,
    '<think>Need the weather first.</think>\n<tool name="weather">{"city": "Paris"}</tool>',
    "<output>Here's what I found.</o>",
    "<o>Short form. Two sentences.</o>",
    '<output>Run this:\n<code lang="python">print(1 < 2)</code>\nDone.</output>',
    "Use the `<code execute=\"true\">` block for that. <output>See above.</output>",
    "<think>Something is wrong.</think><halt reason=\"loop\"/>",
    "Fine. <think>half a thought that never ends. Second sentence",
    "x < y and a <b>bold</b> word. </think> <output>ok</output>",
    "<output>Total: 3 < 5, and <think> is a word here</output>",
    "Plain text only, no tags. Two sentences.",
]


def chunked(text, rng):
    i = 0
    while i < len(text):
        n = rng.randint(1, 9)
        yield text[i:i + n]
        i += n


def stream(text, rng):
    sp = StreamParser()
    events = [e for ch in chunked(text, rng) for e in sp.feed(ch)] + sp.close()
    return sp, events


def words(s):
    return " ".join(s.split())


# ── visible vs silent ────────────────────────────────────────────

def test_silent_blocks_are_never_visible():
    r = parse(VOICE_NOTE)
    assert "Denver" not in r.visible_text()
    assert words(r.visible_text()) == "Got it. I'll note the reservation. Should I have the agent read the email?"


def test_an_unclosed_block_hides_everything_after_it():
    r = parse("Fine. <think>half a thought. Second sentence")
    assert r.visible_text() == "Fine."
    assert r.dangling == ["think"] and not r.is_well_formed()
    # text() keeps its old meaning: everything outside complete blocks
    assert "half a thought" in r.text()


def test_stray_closing_tags_are_dropped_and_reported():
    r = parse("All done. </think> <output>ok</output>")
    assert r.visible_text() == "All done.\n\nok"
    assert r.stray_closes == ["think"]


# ── shorthand ────────────────────────────────────────────────────

def test_mixed_shorthand_is_one_block():
    # Was the example in parse()'s own docstring, and matched nothing.
    r = parse("<output>Here's what I found.</o>")
    assert r.output() is not None and r.output().content == "Here's what I found."
    assert r.output().tag == "output" and r.aliased == ["o"]


def test_registered_tags_keep_their_names():
    from arml import create_parser
    p = create_parser().register("answer", BlockType.OUTPUT)
    r = p.parse("<answer>yes</answer>")
    assert r.blocks[0].tag == "answer" and r.visible_text() == "yes"


# ── streaming ────────────────────────────────────────────────────

def test_streaming_matches_the_batch_parser_for_any_chunking():
    for seed in range(40):
        rng = random.Random(seed)
        for text in CORPUS:
            sp, events = stream(text, rng)
            batch = parse(text)
            got = [(b.tag, b.content, b.attributes) for b in sp.blocks]
            want = [(b.tag, b.content, b.attributes) for b in batch.blocks]
            assert got == want, (seed, text)
            shown = "".join(e.text for e in events if e.kind == "text")
            assert words(shown) == words(batch.visible_text().replace("\n\n", " ")), (seed, text)
            assert sp.dangling == batch.dangling and sp.stray_closes == batch.stray_closes, (seed, text)


def test_a_silent_block_is_held_until_it_closes():
    sp = StreamParser()
    shown = []
    for ch in VOICE_NOTE:                       # one character at a time
        shown += [e.text for e in sp.feed(ch) if e.kind == "text"]
        assert "Denver" not in "".join(shown)
    shown += [e.text for e in sp.close() if e.kind == "text"]
    assert "read the email?" in "".join(shown)


def test_output_streams_as_it_arrives():
    sp = StreamParser()
    first = sp.feed("<output>Hello there, this is")
    assert [e.text for e in first] == ["Hello there, this is"]
    assert sp.feed(" the end.</outp") == [type(first[0])("text", text=" the end.")]
    assert [e.kind for e in sp.feed("ut>")] == ["block"]


def test_a_less_than_sign_is_not_held_forever():
    sp = StreamParser()
    assert [e.text for e in sp.feed("3 < 5 and 2 <")] == ["3 < 5 and 2 "]
    assert [e.text for e in sp.feed(" 4")] == ["< 4"]


# ── lint ─────────────────────────────────────────────────────────

def codes(text):
    return {i.code for i in lint(text)}


def test_lint_flags_what_a_parser_had_to_forgive():
    assert codes("") == {"EMPTY"}
    assert "UNCLOSED" in codes("<think>never closed")
    assert "STRAY_CLOSE" in codes("ok </tool> <output>x</output>")
    assert "TOOL_NO_NAME" in codes("<tool>{}</tool>")
    assert "TOOL_BAD_JSON" in codes('<tool name="a">not json at all</tool>')
    assert "TOOL_JSON_REPAIRED" in codes('<tool name="a">{"q": "x",}</tool>')
    assert "NO_ACTION" in codes("<think>just thinking</think>")
    assert "NO_BLOCKS" in codes("Plain text only.")
    assert "OUTPUT_WITH_TOOL" in codes('<tool name="a">{}</tool><output>done</output>')
    assert {"MIXED_ALIAS", "SHORTHAND"} <= codes("<output>x</o>")
    assert "STRAY_TEXT" in codes(VOICE_NOTE)


def test_clean_responses_pass():
    good = '<think>Need the weather.</think>\n<tool name="weather">{"city": "Paris"}</tool>'
    assert lint(good) == [] and is_clean(good)
    assert not is_clean("<think>never closed")


# ── canonical ────────────────────────────────────────────────────

def test_canonical_form_is_one_spelling():
    messy = "<think> plan it </think>\n\n\n<tool name='weather'>{\"city\":\"Paris\"}</tool><o>Done.</o>"
    want = ('<think>plan it</think>\n\n<tool name="weather">{"city": "Paris"}</tool>'
            "\n\n<output>Done.</output>")
    assert canonical(messy) == want
    assert canonical(want) == want                  # stable
    assert parse(canonical(messy)).visible_text() == parse(messy).visible_text()


def test_canonical_leaves_repaired_json_as_written():
    text = '<tool name="a">{"q": "x",}</tool>'
    assert canonical(text) == text


# ── prompt ───────────────────────────────────────────────────────

def test_format_rules_list_exact_tools_and_switch_for_voice():
    from arml.prompt import format_rules
    tools = [{"name": "weather", "description": "Current weather.",
              "parameters": {"city": {"type": "string", "required": True}}}]
    text = format_rules(tools)
    assert "- weather: Current weather." in text and "city (string, required)" in text
    assert '<tool name="weather">{"city": "value"}</tool>' in text
    assert "No <output> in the same response as a <tool>" in text and "SPOKEN" not in text
    voice = format_rules(tools, spoken=True)
    assert "may only acknowledge" in voice and "Do not read your plan out loud" in voice
    assert "<code" not in text and "<code" in format_rules(tools, code=True)


# ── turn records ─────────────────────────────────────────────────

def test_turn_records_lint_redact_and_round_trip(tmp_path):
    from arml.record import TurnRecord, append, read, training_example
    rec = TurnRecord.from_response(
        '<think>Text her.</think><tool name="send_sms">{"to": "212-555-0199", "body": "hi"}</tool>',
        model="m", harness="test/1", session="s", turn=1,
        context=[{"role": "user", "content": "text me at 212-555-0199 or a@b.com"}],
        system="You are a test.")
    assert rec.clean and rec.blocks == ["think", "tool:send_sms"]
    assert "212-555-0199" not in rec.to_json() and "a@b.com" not in rec.to_json()
    assert rec.system is None and len(rec.system_sha256) == 64
    path = str(tmp_path / "turns.jsonl")
    append(path, rec.set_outcome(tool_ok=True, user="accepted"))
    (back,) = list(read(path))
    assert back.outcome == {"tool_ok": True, "user": "accepted"}
    ex = training_example(back, system="You are a test.")
    assert ex["messages"][0]["role"] == "system" and ex["messages"][-1]["content"] == back.canonical


def test_bad_turns_are_kept_but_never_become_training_examples():
    from arml.record import TurnRecord, training_example
    broken = TurnRecord.from_response("Sure. <think>never closed", model="m", harness="t")
    assert not broken.clean and "UNCLOSED" in broken.lint and broken.canonical is None
    assert training_example(broken) is None
    corrected = TurnRecord.from_response("<output>It is 9 pm.</output>", model="m", harness="t")
    assert training_example(corrected) is not None
    corrected.set_outcome(user="corrected", note="it was 10 pm")
    assert training_example(corrected) is None
    with pytest.raises(ValueError):
        corrected.set_outcome(user="loved it")


# ── behavior checks, from a voice assistant's real logs ──────────

def test_lint_catches_what_went_wrong_in_the_field():
    # It invented a tool result: the venue's hours, as an "event", before the agent replied.
    assert "FAKE_RESULT" in codes("<output>(Event: g6f40, step 1 done: open until 10 p.m.)</output>")
    assert "FAKE_RESULT" in {i.code for i in lint("<output>[[result]] ok</output>", reserved=[r"\[\[result\]\]"])}
    # "Let me check your calendar" with no tool that could.
    assert "ANNOUNCED_NO_ACTION" in codes("<output>Let me check your calendar for tomorrow.</output>")
    assert "ANNOUNCED_NO_ACTION" not in codes(
        '<tool name="calendar">{"day": "tomorrow"}</tool><output>Let me check.</output>')
    # Claimed a text went out when nothing sent it.
    assert "CLAIMED_NO_ACTION" in codes("<output>I've sent the summary.</output>")
    # Long spoken replies, numbers read aloud.
    long = "<output>" + "word " * 45 + "</output>"
    assert "LONG_SPOKEN" in {i.code for i in lint(long, spoken=True)} and "LONG_SPOKEN" not in codes(long)
    assert "SPOKEN_NUMBER" in {i.code for i in lint("<output>Texting 212-555-0199 now.</output>", spoken=True)}


def test_deliberate_silence_is_a_valid_response():
    assert is_clean("<think>They are talking to someone else.</think><output/>")
    assert parse("<output/>").visible_text() == ""
