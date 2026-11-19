"""bench/native_vs_arml.py with a scripted model: both loops, the scoring and the report."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_vs_arml", ROOT / "bench" / "native_vs_arml.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

TASK = {t["id"]: t for t in B.TASKS}


def scripted(replies):
    """A chat function that answers with the given messages in turn and records what it was sent."""
    replies, seen = list(replies), []

    def chat(messages, tools=None):
        seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        msg = replies.pop(0)
        msg = {"role": "assistant", "content": msg} if isinstance(msg, str) else {"role": "assistant", **msg}
        return {"message": msg, "usage": {"completion_tokens": 7}, "seconds": 0.5}
    chat.seen = seen
    return chat


def call(name, args, raw=False):
    return {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": name, "arguments": args if raw else json.dumps(args)}}]}


def test_arml_loop_runs_the_tool_and_scores_the_answer():
    chat = scripted(['<think>need the tool</think>\n<tool name="weather">{"city": "Lisbon"}</tool>',
                     "<output>It is 19 C with light rain in Lisbon.</output>"])
    r = B.run_arml(TASK["weather"], chat)
    assert r["calls"] == [("weather", {"city": "Lisbon"})] and r["errors"] == [] and r["steps"] == 2
    assert B.score(TASK["weather"], r["calls"], r["final"]) == (True, "")
    assert chat.seen[0]["tools"] is None and "AVAILABLE TOOLS" in chat.seen[0]["messages"][0]["content"]
    assert chat.seen[1]["messages"][-1] == {"role": "user", "content": "Result of weather: 19 C, light rain"}
    assert r["tokens"] == 14 and r["seconds"] == 1.0


def test_arml_format_errors_are_lint_errors():
    r = B.run_arml(TASK["capital"], scripted(["Paris."]))                 # plain text: NO_BLOCKS
    assert r["errors"] == ["NO_BLOCKS"] and r["final"] == "Paris."
    assert B.score(TASK["capital"], r["calls"], r["final"])[0]            # right answer, wrong format
    r = B.run_arml(TASK["capital"], scripted(["<output>Paris"]))          # never closed: nothing is shown
    assert "UNCLOSED" in r["errors"] and r["final"] == ""
    assert B.score(TASK["capital"], r["calls"], r["final"]) == (False, "no answer")


def test_native_loop_sends_tools_and_returns_results_as_tool_messages():
    chat = scripted([call("weather", {"city": "Lisbon"}), "It is 19 C in Lisbon."])
    r = B.run_native(TASK["weather"], chat)
    assert r["calls"] == [("weather", {"city": "Lisbon"})] and r["errors"] == []
    assert B.score(TASK["weather"], r["calls"], r["final"]) == (True, "")
    assert [t["function"]["name"] for t in chat.seen[0]["tools"]] == list(B.TOOLS)
    assert chat.seen[0]["tools"][0]["function"]["parameters"]["required"] == ["city"]
    assert chat.seen[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "19 C, light rain"}
    assert "AVAILABLE TOOLS" not in chat.seen[0]["messages"][0]["content"]


def test_native_format_errors():
    r = B.run_native(TASK["weather"], scripted([call("weather", "{city: Lisbon", raw=True), "Sorry."]))
    assert r["errors"] == ["TOOL_BAD_JSON"]
    r = B.run_native(TASK["weather"], scripted(['<tool_call>{"name": "weather", "arguments": {"city": "Lisbon"}}</tool_call>']))
    assert r["errors"] == ["CALL_AS_TEXT"] and r["calls"] == []
    r = B.run_native(TASK["weather"], scripted(["<think>hmm</think>"]))
    assert r["errors"] == ["EMPTY"]
    r = B.run_native(TASK["weather"], scripted([call("clock", {})] * 6))
    assert r["errors"] == ["MAX_STEPS"] and r["steps"] == 6


def test_scoring_rules():
    s = B.score
    assert s(TASK["weather"], [("weather", {"city": "Oslo"})], "19")[1] == "no correct weather call"
    assert s(TASK["weather"], [("weather", {"city": "Lisbon"})], "It rains.")[1] == "the answer does not say it"
    assert s(TASK["capital"], [("clock", {})], "Paris")[1].startswith("called clock")
    assert s(TASK["flight"], [("book_flight", {"to": "Paris"})], "Booked.")[1] == "invented a tool: book_flight"
    assert s(TASK["flight"], [("clock", {})], "I can't book flights.")[0]
    both = [("weather", {"city": "Denver"}), ("weather", {"city": "Lisbon, Portugal"})]
    assert s(TASK["warmer"], both, "Lisbon is warmer.")[0]                 # any order
    assert not s(TASK["warmer"], both[:1] * 2, "Lisbon")[0]                # one call cannot count twice
    sent = ("send_message", {"to": "priya.n@example.com", "text": "the meeting moved to 3"})
    assert s(TASK["text"], [sent], "")[0]
    assert s(TASK["text"], [sent, sent], "")[1] == "send_message called 2 times, expected 1"
    assert s(TASK["seconds"], [("set_timer", {"minutes": "1.5"})], "")[0]
    assert not s(TASK["seconds"], [("set_timer", {"minutes": 90})], "")[0]
    assert s(TASK["multiply"], [("calculate", {"expression": "84.50 × 1.08"})], "91.26")[0]
    assert s(TASK["send-what"], [], "Who should I send it to?")[0]
    assert not s(TASK["send-what"], [], "Sent.")[0]


def test_every_task_can_be_passed_with_the_fake_tools():
    """The data behind each task supports its expected answer."""
    assert B.run_tool("calendar", {"date": "2026-03-11"}) == ("nothing scheduled", True)
    assert B.run_tool("contacts", {"name": "Sam"})[1] is False
    assert B.run_tool("calculate", {"expression": "84.50 * 1.08"}) == ("91.26", True)
    assert B.run_tool("set_timer", {"minutes": 12}) == ("timer set for 12 minutes", True)
    assert B.run_tool("nope", {}) == ("error: no tool named nope", False)
    assert len({t["id"] for t in B.TASKS}) == len(B.TASKS)
    for t in B.TASKS:
        assert all(name in B.TOOLS for name, _ in t["need"])


def test_run_all_summary_and_report(tmp_path):
    tasks = [TASK["capital"], TASK["weather"]]

    def make(seed):
        def chat(messages, tools=None):
            ask = messages[1]["content"]
            if tools:                                   # native: gets both right
                if "weather" in ask and messages[-1]["role"] != "tool":
                    msg = call("weather", {"city": "Lisbon"})
                else:
                    msg = {"content": "Paris." if "France" in ask else "19 C."}
            else:                                       # arml: skips the tool on the weather task
                msg = {"content": "<output>Paris.</output>" if "France" in ask else "<output>Sunny.</output>"}
            return {"message": {"role": "assistant", **msg}, "usage": {"completion_tokens": 5}, "seconds": 0.1}
        return chat

    rows = B.run_all("m", make, ["arml", "native"], 2, tasks, 6, True, str(tmp_path), "m__x", quiet=True)
    assert len(rows) == 8
    s = B.summarize(rows, ["arml", "native"])
    assert s["modes"]["arml"]["passed"] == 2 and s["modes"]["native"]["passed"] == 4
    assert s["paired"]["only_native"] == 2 and s["paired"]["only_arml"] == 0 and s["paired"]["p"] == 0.5
    assert s["paired"]["tasks_only_native"] == ["weather"]
    text = B.report(s, ["arml", "native"], "t")
    assert "| Tasks done right | 2 of 4 | 4 of 4 |" in text and "sign test p = 0.5" in text
    assert len((tmp_path / "m__x.transcripts.jsonl").read_text().splitlines()) == 8
    assert (tmp_path / "m__x.turns.jsonl").read_text().count('"harness": "arml-bench/1"') == 4
    from arml.record import read, training_example
    turns = list(read(str(tmp_path / "m__x.turns.jsonl")))
    assert sorted(r.outcome["task_ok"] for r in turns) == [False, False, True, True]
    failed = [r for r in turns if not r.outcome["task_ok"]]
    assert failed[0].outcome["note"] == "no correct weather call" and failed[0].clean
    assert [training_example(r) is not None for r in turns].count(True) == 2     # only the passed tasks


def test_sign_test():
    assert B.sign_test(0, 0) == 1.0 and B.sign_test(5, 5) == 1.0
    assert round(B.sign_test(10, 0), 4) == 0.002 and round(B.sign_test(1, 8), 3) == 0.039


def test_asking_counts_with_or_without_a_question_mark_and_warnings_are_kept():
    assert B.score(TASK["send-what"], [], "Please provide the recipient's email address.")[0]
    assert B.score(TASK["weather-where"], [], "Which city?")[0]
    r = B.run_arml(TASK["weather"], scripted(['<tool name="weather">{"city": "Lisbon"}</tool><output>Checking.</output>',
                                             "<output>19 C.</output>"]))
    assert r["errors"] == [] and r["warnings"] == ["OUTPUT_WITH_TOOL"]
