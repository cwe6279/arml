"""datagen/generate.py: scenarios are consistent, the checker catches wrong sessions, and
nothing overlaps the benchmark."""
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("generate", "datagen/generate.py")
B = _load("native_vs_arml", "bench/native_vs_arml.py")
KINDS = {k for k, _ in G.KINDS}


def _first(kind, seed=1):
    return next(s for s in (G.scenario(seed, i) for i in range(500)) if s["kind"] == kind)


def scripted(*replies):
    replies = list(replies)
    return lambda messages: replies.pop(0)


def test_scenarios_are_reproducible_and_varied():
    a, b = G.scenario(3, 17), G.scenario(3, 17)
    assert G.describe(a) == G.describe(b) and G.system_prompt(a) == G.system_prompt(b)
    many = [G.scenario(1, i) for i in range(600)]
    assert {s["kind"] for s in many} == KINDS
    assert len({s["ask"] for s in many}) > 350
    assert len({t["name"] for s in many for t in s["tools"]}) > 100           # names and styles vary
    assert all(2 <= len(s["tools"]) <= 8 for s in many)
    assert len({G.system_prompt(s) for s in many}) > 500


def test_nothing_overlaps_the_benchmark():
    assert G.HELD_OUT_TOOLS == set(B.TOOLS)
    bench_asks = {t["ask"].lower() for t in B.TASKS}
    for i in range(1500):
        s = G.scenario(7, i)
        assert not ({t["name"] for t in s["tools"]} & set(B.TOOLS))
        assert s["ask"].lower() not in bench_asks


def test_the_oracle_passes_every_scenario_and_every_turn_is_strict_clean():
    kinds = set()
    for i in range(700):
        s = G.scenario(11, i)
        r = G.run_session(s, G.oracle(s, random.Random(i)), "oracle")
        assert r["passed"], (G.describe(s), r["why"], r["calls"], r["final"])
        assert r["codes"] == []
        assert all(rec.good(strict=True) and rec.outcome["task_ok"] for rec in r["records"])
        kinds.add(s["kind"])
    assert kinds == KINDS


def test_the_checker_rejects_wrong_sessions():
    s = _first("lookup")
    name, args = s["need"][0]
    right = f'<tool name="{name}">{json.dumps(args)}</tool>'
    token = s["answer_has"][0]
    ok = G.run_session(s, scripted(right, f"<output>It is {token}.</output>"), "m")
    assert ok["passed"] and len(ok["records"]) == 2
    r = G.run_session(s, scripted("<output>It is 5.</output>"), "m")              # answered from memory
    assert not r["passed"] and r["why"].startswith("no correct")
    assert r["records"][0].outcome == {"task_ok": False, "note": r["why"]}
    r = G.run_session(s, scripted(right, "<output>I looked it up.</output>"), "m")
    assert r["why"] == "the answer does not say it"
    r = G.run_session(s, scripted('<tool name="crystal_ball">{}</tool>', right, f"<output>{token}</output>"), "m")
    assert r["why"] == "invented a tool: crystal_ball"
    r = G.run_session(s, scripted(*[right] * 6), "m")
    assert r["why"] == "still calling tools after the step limit"

    s = _first("action")
    name, args = s["need"][0]
    call = f'<tool name="{name}">{json.dumps(args)}</tool>'
    assert G.run_session(s, scripted(call, "<output>Done.</output>"), "m")["passed"]
    r = G.run_session(s, scripted(call, call, "<output>Done.</output>"), "m")     # the action ran twice
    assert r["why"] == f"{name} called 2 times, expected 1"
    r = G.run_session(s, scripted("<output>Done, all set.</output>"), "m")         # claimed, never did it
    assert not r["passed"]

    s = _first("missing")
    assert G.run_session(s, scripted(f"<output>{s['gold']}</output>"), "m")["passed"]
    r = G.run_session(s, scripted("<output>Done.</output>"), "m")
    assert r["why"] == "did not ask for what is missing"

    s = _first("no tool")
    tool = s["tools"][0]
    r = G.run_session(s, scripted(f'<tool name="{tool["name"]}">{{}}</tool>', f"<output>{s['gold']}</output>"), "m")
    assert not r["passed"] and "did not call for" in r["why"]

    s = _first("compare")
    r = G.run_session(s, scripted(*[f'<tool name="{n}">{json.dumps(a)}</tool>' for n, a in s["need"]],
                                  f"<output>{s['answer_not'][0]}.</output>"), "m")  # named the lower one
    assert not r["passed"]

    s = next(x for x in (G.scenario(1, i) for i in range(500)) if x["kind"] == "unknown" and x["answer_not"])
    name, args = s["need"][0]
    lie = s["answer_not"][0]
    r = G.run_session(s, scripted(f'<tool name="{name}">{json.dumps(args)}</tool>', f"<output>It is {lie}.</output>"), "m")
    assert r["why"] == f"the answer says {lie!r}, which it should not"


def test_a_passing_session_with_a_warning_keeps_only_its_clean_turns():
    s = _first("lookup")
    name, args = s["need"][0]
    early = f'<tool name="{name}">{json.dumps(args)}</tool><output>Checking.</output>'   # OUTPUT_WITH_TOOL
    r = G.run_session(s, scripted(early, f"<output>{s['answer_has'][0]}</output>"), "m")
    assert r["passed"] and r["codes"] == ["OUTPUT_WITH_TOOL"]
    assert [rec.good(strict=True) for rec in r["records"]] == [False, True]


def test_generate_writes_the_log_the_training_file_and_resumes(tmp_path):
    teacher = lambda s: G.oracle(s, random.Random(s["id"]))                       # noqa: E731
    st = G.generate(40, 5, teacher, "oracle", str(tmp_path), quiet=True)
    assert st["scenarios"] == 40 and st["passed"] == 40 and st["examples"] == st["turns_logged"] > 40
    lines = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(lines) == st["examples"]
    for ex in lines:
        roles = [m["role"] for m in ex["messages"]]
        assert roles[0] == "system" and roles[1] == "user" and roles[-1] == "assistant"
        assert "AVAILABLE TOOLS" in ex["messages"][0]["content"]
        text = json.dumps(ex)
        assert "<phone>" not in text and "<card>" not in text and "<email>" not in text and "<key>" not in text
    emails = [s for s in (G.scenario(5, i) for i in range(40)) if any(t["concept"]["key"] == "email" for t in s["tools"])]
    assert emails                                       # the email tool's addresses are kept as written
    assert "@example.org" in (tmp_path / "train.jsonl").read_text()
    again = G.generate(50, 5, teacher, "oracle", str(tmp_path), quiet=True)        # 40 are already there
    assert again["skipped"] == 40 and again["scenarios"] == 10
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == again["examples"] > st["examples"]


def test_a_failed_teacher_session_is_logged_but_not_trained_on(tmp_path):
    st = G.generate(12, 2, lambda s: (lambda messages: "I think it's fine."), "bad", str(tmp_path), quiet=True)
    assert st["turns_logged"] == 12 and st["examples"] == 0 and st["codes"] == {"NO_BLOCKS": 12}
