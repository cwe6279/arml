"""
Does a model do agent work better writing ARML, or using its server's native tool calls?

The same model gets the same tasks twice, on the same tools:

    arml     the system prompt carries format_rules(tools); the model writes <tool> and
             <output> blocks as text; results go back the way examples/minimal_harness.py
             sends them. No `tools` field is sent.
    native   the tools go in the request's `tools` field; the server's chat template and
             parser handle the calls; results go back as role "tool" messages.

The tools are fakes with fixed data, so every task has a known right answer and nothing
real is sent. Works with any OpenAI-compatible server. No dependencies.

    python bench/native_vs_arml.py --model qwen3:8b
    python bench/native_vs_arml.py --model qwen3:8b --reasoning-effort none --runs 3

What is measured, per mode: tasks done right, format errors (ARML lint errors; for native:
bad JSON arguments, a call written as text, an empty reply), invented tools, steps, output
tokens and time. Then the two modes are compared task by task. See bench/README.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import re
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arml import format_rules, lint, parse                               # noqa: E402
from arml.record import TurnRecord, append                               # noqa: E402

# ── the tools: fixed data, nothing real happens ─────────────────────────────
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod}


def calc(expression: str) -> float:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in _OPS:
            args = (ev(n.left), ev(n.right)) if isinstance(n, ast.BinOp) else (ev(n.operand),)
            return _OPS[type(n.op)](*args)
        raise ValueError("only numbers and + - * / ** %")
    text = str(expression).replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    return ev(ast.parse(text, mode="eval").body)


NOW = "Tuesday 2026-03-10 14:35"
WEATHER = {"lisbon": "19 C, light rain", "oslo": "4 C, clear", "tokyo": "23 C, cloudy",
           "denver": "11 C, windy"}
CALENDAR = {"2026-03-10": "09:00 standup; 15:30 dentist", "2026-03-11": "nothing scheduled",
            "2026-03-12": "12:00 lunch with Priya"}
CONTACTS = {"dana": "Dana Ruiz, dana.ruiz@example.com", "priya": "Priya N, priya.n@example.com"}


def _weather(p):
    city = str(p.get("city", "")).strip().lower().split(",")[0]
    return WEATHER.get(city) or f"error: no weather data for {p.get('city')!r}"


def _calendar(p):
    return CALENDAR.get(str(p.get("date", "")).strip()) or f"error: no calendar for {p.get('date')!r}; use YYYY-MM-DD"


def _contacts(p):
    name = str(p.get("name", "")).strip().lower().split()[:1]
    return CONTACTS.get(name[0] if name else "") or f"error: no contact named {p.get('name')!r}"


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


TOOLS: Dict[str, Tuple[dict, Callable[[dict], str]]] = {
    "weather": ({"description": "Current weather for a city.",
                 "parameters": {"city": {"type": "string", "required": True}}}, _weather),
    "calculate": ({"description": "Exact arithmetic on an expression such as (2 + 3) * 4.",
                   "parameters": {"expression": {"type": "string", "required": True}}},
                  lambda p: f"{calc(p['expression']):.10g}"),
    "clock": ({"description": "The current local date and time.", "parameters": {}}, lambda p: NOW),
    "calendar": ({"description": "The user's events on one date.",
                  "parameters": {"date": {"type": "string", "required": True,
                                          "description": "YYYY-MM-DD"}}}, _calendar),
    "contacts": ({"description": "Look up a person's name and email address.",
                  "parameters": {"name": {"type": "string", "required": True}}}, _contacts),
    "send_message": ({"description": "Send a text message to an email address.",
                      "parameters": {"to": {"type": "string", "required": True,
                                            "description": "email address"},
                                     "text": {"type": "string", "required": True}}},
                     lambda p: f"sent to {p['to']}"),
    "set_timer": ({"description": "Start a countdown timer.",
                   "parameters": {"minutes": {"type": "number", "required": True}}},
                  lambda p: f"timer set for {_num(p['minutes']):g} minutes"),
}
ARML_SPECS = [{"name": n, **spec} for n, (spec, _) in TOOLS.items()]


def native_specs() -> List[dict]:
    """The same tools in the OpenAI `tools` format."""
    out = []
    for name, (spec, _) in TOOLS.items():
        props, required = {}, []
        for pname, p in spec["parameters"].items():
            props[pname] = {k: v for k, v in p.items() if k != "required"}
            if p.get("required"):
                required.append(pname)
        out.append({"type": "function", "function": {
            "name": name, "description": spec["description"],
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


# ── the prompts: the same behaviour rules in both modes ─────────────────────
IDENTITY = ("You are a careful assistant. Use a tool when the request needs one. "
            "When something you need is missing, ask one short question.")
# format_rules() carries behaviour rules next to its format rules. Native mode gets the same
# behaviour rules, so the comparison is about the format and not about the prompt.
NATIVE_RULES = """RULES:
- Use only the tools you were given, with their exact parameter names.
- Nothing runs between responses. Do not announce work ("Let me check that") without calling the tool in the same response.
- An action happens only through a tool call. Never say you sent, saved or created something without the call that does it.
- Only the runtime writes tool results. Never write a result yourself; wait for it.
- Facts that change (the time, the date, a calendar, prices, opening hours) come from a tool result in this conversation, never from memory or earlier notes.
- When you hand work to a tool, put everything it needs into the arguments: the place, the dates, the names you already know.
- When a request changes (new dates, a new target), cancel or replace the work started for the old one."""


def system_prompt(mode: str, native_rules: bool = True) -> str:
    if mode == "arml":
        return IDENTITY + "\n\n" + format_rules(ARML_SPECS)
    return IDENTITY + ("\n\n" + NATIVE_RULES if native_rules else "")


# ── the tasks ───────────────────────────────────────────────────────────────
def _has(key: str, *needles: str) -> Callable[[dict], bool]:
    return lambda p: all(n.lower() in str(p.get(key, "")).lower() for n in needles)


def _calc_is(value: float) -> Callable[[dict], bool]:
    def check(p):
        try:
            return math.isclose(calc(p.get("expression", "")), value, abs_tol=0.006)
        except Exception:
            return False
    return check


def _both(*checks) -> Callable[[dict], bool]:
    return lambda p: all(c(p) for c in checks)


ANY = lambda p: True                                                     # noqa: E731
# An answer that asks for what is missing, with or without a question mark.
ASKS = r"\?|\b(provide|specify|tell me|let me know|which|who|whom|what|where|need)\b"
ALL_TOOLS = frozenset(TOOLS)
SIDE_EFFECTS = frozenset({"send_message", "set_timer"})

# need:   calls that must be made (each matched by a different call, any order)
# forbid: tools that must not be called at all
# answer: a regex the final visible answer must match (None: any answer, even none)
TASKS: List[dict] = [
    # one tool
    dict(id="weather", cat="one tool", ask="What's the weather in Lisbon?",
         need=[("weather", _has("city", "lisbon"))], answer=r"19"),
    dict(id="multiply", cat="one tool", ask="What is 84.50 times 1.08, exactly?",
         need=[("calculate", _calc_is(91.26))], answer=r"91\.26"),
    dict(id="time", cat="one tool", ask="What time is it?",
         need=[("clock", ANY)], answer=r"14:35|2:35"),
    dict(id="calendar", cat="one tool", ask="What's on my calendar on March 12th, 2026?",
         need=[("calendar", _has("date", "2026-03-12"))], answer=r"priya"),
    dict(id="timer", cat="one tool", ask="Set a timer for 12 minutes.",
         need=[("set_timer", lambda p: _num(p.get("minutes")) == 12)], answer=None),
    dict(id="email", cat="one tool", ask="What's Dana's email address?",
         need=[("contacts", _has("name", "dana"))], answer=r"dana\.ruiz@example\.com"),
    dict(id="text", cat="one tool", ask="Text priya.n@example.com: the meeting moved to 3.",
         need=[("send_message", _both(_has("to", "priya.n@example.com"), _has("text", "3")))], answer=None),
    # exact arguments
    dict(id="tokyo", cat="exact arguments", ask="How warm is it in Tokyo right now?",
         need=[("weather", _has("city", "tokyo"))], answer=r"23"),
    dict(id="seconds", cat="exact arguments", ask="Start a 90 second timer.",
         need=[("set_timer", lambda p: _num(p.get("minutes")) == 1.5)], answer=None),
    dict(id="brackets", cat="exact arguments", ask="What is (17 + 5) * 3 - 4?",
         need=[("calculate", _calc_is(62))], answer=r"62"),
    # several steps
    dict(id="tomorrow", cat="several steps", ask="Do I have anything on my calendar tomorrow?",
         need=[("clock", ANY), ("calendar", _has("date", "2026-03-11"))],
         answer=r"nothing|no\b|free|clear|don'?t|do not|empty"),
    dict(id="relay", cat="several steps", ask="Message Dana the current temperature in Oslo.",
         need=[("weather", _has("city", "oslo")), ("contacts", _has("name", "dana")),
               ("send_message", _both(_has("to", "dana.ruiz@example.com"), _has("text", "4")))], answer=None),
    dict(id="warmer", cat="several steps", ask="Is it warmer in Lisbon or in Denver right now?",
         need=[("weather", _has("city", "lisbon")), ("weather", _has("city", "denver"))], answer=r"lisbon"),
    dict(id="until", cat="several steps", ask="How many minutes until 5 pm?",
         need=[("clock", ANY)], answer=r"145"),
    dict(id="lookup-send", cat="several steps",
         ask="Look up Priya's email and send her this: Running ten minutes late.",
         need=[("contacts", _has("name", "priya")),
               ("send_message", _both(_has("to", "priya.n@example.com"), _has("text", "late")))], answer=None),
    dict(id="difference", cat="several steps",
         ask="What's the temperature difference between Tokyo and Oslo right now?",
         need=[("weather", _has("city", "tokyo")), ("weather", _has("city", "oslo"))], answer=r"19"),
    # no tool needed
    dict(id="capital", cat="no tool needed", ask="What's the capital of France?",
         need=[], forbid=ALL_TOOLS, answer=r"paris"),
    dict(id="spanish", cat="no tool needed", ask="How do you say 'good morning' in Spanish?",
         need=[], forbid=ALL_TOOLS, answer=r"buenos d[ií]as"),
    dict(id="thanks", cat="no tool needed", ask="Thanks, that's all for now.",
         need=[], forbid=ALL_TOOLS, answer=None),
    dict(id="goldfish", cat="no tool needed", ask="Suggest one name for a pet goldfish.",
         need=[], forbid=ALL_TOOLS, answer=r"\w"),
    # missing or impossible: the right move is to ask or decline, not to act
    dict(id="send-what", cat="missing or impossible", ask="Send a message.",
         need=[], forbid={"send_message"}, answer=ASKS),
    dict(id="weather-where", cat="missing or impossible", ask="What's the weather like?",
         need=[], forbid={"weather"}, answer=ASKS),
    dict(id="flight", cat="missing or impossible", ask="Book me a flight to Paris for Friday.",
         need=[], forbid=SIDE_EFFECTS, answer=r"\w"),
    dict(id="unknown-contact", cat="missing or impossible", ask="Send Sam a message saying hello.",
         need=[], forbid={"send_message"}, answer=r"\w"),
    # facts that change: must come from a tool, not from memory
    dict(id="weekday", cat="facts that change", ask="What day of the week is it?",
         need=[("clock", ANY)], answer=r"tuesday"),
    dict(id="dentist", cat="facts that change", ask="Is my dentist appointment today?",
         need=[("clock", ANY), ("calendar", _has("date", "2026-03-10"))], answer=r"yes|15:30|3:30"),
]
CATEGORIES = list(dict.fromkeys(t["cat"] for t in TASKS))


def score(task: dict, calls: List[Tuple[str, dict]], final: str) -> Tuple[bool, str]:
    """(passed, why not). calls are the tool calls that were made, in order."""
    left = list(calls)
    for name, check in task["need"]:
        hit = next((i for i, (n, p) in enumerate(left) if n == name and check(p)), None)
        if hit is None:
            return False, f"no correct {name} call"
        left.pop(hit)
    forbidden = set(task.get("forbid") or ())
    made = [n for n, _ in calls]
    bad = [n for n in made if n in forbidden]
    if bad:
        return False, f"called {bad[0]}, which this request did not call for"
    for tool in SIDE_EFFECTS:                       # an action is done once, not twice
        wanted = sum(1 for n, _ in task["need"] if n == tool)
        if made.count(tool) > wanted:
            return False, f"{tool} called {made.count(tool)} times, expected {wanted}"
    unknown = [n for n in made if n not in TOOLS]
    if unknown:
        return False, f"invented a tool: {unknown[0]}"
    if task["answer"] and not re.search(task["answer"], final or "", re.I):
        return False, "the answer does not say it" if (final or "").strip() else "no answer"
    return True, ""


# ── the model ───────────────────────────────────────────────────────────────
def make_chat(url: str, model: str, key: str = "", reasoning_effort: str = "", seed: int = 1,
              timeout: int = 300, max_tokens: int = 2048) -> Callable[[List[dict], Optional[List[dict]]], dict]:
    """chat(messages, tools) -> {"message": ..., "usage": ..., "seconds": ...}"""
    def chat(messages, tools=None):
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                "temperature": 0, "seed": seed, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(), headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
        return {"message": data["choices"][0]["message"], "usage": data.get("usage") or {},
                "seconds": time.time() - t0}
    return chat


def run_tool(name: str, params: dict) -> Tuple[str, bool]:
    """(result text, ok)"""
    if name not in TOOLS:
        return f"error: no tool named {name}", False
    try:
        result = TOOLS[name][1](params)
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", False
    return result, not result.startswith("error:")


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# A tool call that came back as text: the server's parser missed it, or the model wrote
# the call in a format the server does not parse.
_CALL_AS_TEXT_RE = re.compile(r"<tool_call>|</tool_call>|<\|tool|<function[=\s>]|<tool\b|"
                              r"\"arguments\"\s*:|\"parameters\"\s*:\s*\{", re.I)


def run_arml(task: dict, chat, max_steps: int = 6, log: Optional[Callable[[TurnRecord], None]] = None,
             model: str = "", session: str = "") -> dict:
    system = system_prompt("arml")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["ask"]}]
    calls, errors, warnings, final, usage, seconds, steps = [], [], [], "", 0, 0.0, 0
    for steps in range(1, max_steps + 1):
        r = chat(messages, None)
        raw = r["message"].get("content") or ""
        usage += int(r["usage"].get("completion_tokens") or 0)
        seconds += r["seconds"]
        issues = lint(raw)
        errors += [i.code for i in issues if i.level == "error"]
        warnings += [i.code for i in issues if i.level == "warn"]
        rec = TurnRecord.from_response(raw, model=model, harness="arml-bench/1", session=session,
                                       turn=steps, context=messages[1:], system=system) if log else None
        messages.append({"role": "assistant", "content": raw})
        parsed = parse(raw)
        tools = [] if parsed.is_halted() else parsed.tools()
        if not tools:
            final = parsed.visible_text().strip()
            if log:
                log(rec)
            break
        tool = tools[0]                                 # one <tool> per response
        try:
            params = tool.params()
        except Exception:
            params = {}
        params = params if isinstance(params, dict) else {}
        calls.append((tool.name, params))
        result, ok = run_tool(tool.name, params)
        if log:
            log(rec.set_outcome(tool_ok=ok, **({} if ok else {"tool_error": result[:200]})))
        messages.append({"role": "user", "content": f"Result of {tool.name}: {result}"})
    else:
        errors.append("MAX_STEPS")
    return {"calls": calls, "errors": errors, "warnings": warnings, "final": final, "tokens": usage,
            "seconds": seconds, "steps": steps, "messages": messages}


def run_native(task: dict, chat, max_steps: int = 6, native_rules: bool = True) -> dict:
    messages = [{"role": "system", "content": system_prompt("native", native_rules)},
                {"role": "user", "content": task["ask"]}]
    specs = native_specs()
    calls, errors, final, usage, seconds, steps = [], [], "", 0, 0.0, 0
    for steps in range(1, max_steps + 1):
        r = chat(messages, specs)
        msg = r["message"]
        usage += int(r["usage"].get("completion_tokens") or 0)
        seconds += r["seconds"]
        content = _THINK_RE.sub("", msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        if not tool_calls:
            if _CALL_AS_TEXT_RE.search(content):
                errors.append("CALL_AS_TEXT")
            elif not content:
                errors.append("EMPTY")
            final = content
            break
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name, args = str(fn.get("name") or ""), fn.get("arguments")
            try:
                params = args if isinstance(args, dict) else json.loads(args or "{}")
                if not isinstance(params, dict):
                    raise ValueError("arguments are not an object")
                calls.append((name, params))
                result, _ = run_tool(name, params)
            except ValueError:
                errors.append("TOOL_BAD_JSON")
                calls.append((name, {}))
                result = "error: the arguments were not valid JSON"
            messages.append({"role": "tool", "tool_call_id": tc.get("id") or "", "content": result})
    else:
        errors.append("MAX_STEPS")
    return {"calls": calls, "errors": errors, "final": final, "tokens": usage,
            "seconds": seconds, "steps": steps, "messages": messages}


# ── running and reporting ───────────────────────────────────────────────────
def run_all(model: str, make, modes: List[str], runs: int, tasks: List[dict], max_steps: int,
            native_rules: bool, out_dir: str, tag: str, quiet: bool = False) -> List[dict]:
    """One row per (task, run, mode). make(seed) returns a chat function."""
    rows = []
    os.makedirs(out_dir, exist_ok=True)
    transcripts = os.path.join(out_dir, f"{tag}.transcripts.jsonl")
    turns = os.path.join(out_dir, f"{tag}.turns.jsonl")
    for path in (transcripts, turns):
        if os.path.exists(path):
            os.remove(path)
    for run in range(1, runs + 1):
        chat = make(run)
        for task in tasks:
            for mode in modes:
                records: List[TurnRecord] = []
                try:
                    if mode == "arml":
                        r = run_arml(task, chat, max_steps, model=model, session=f"{tag}-{run}-{task['id']}",
                                     log=records.append)
                    else:
                        r = run_native(task, chat, max_steps, native_rules)
                except Exception as e:                  # a server error is not the model's failure
                    r = {"calls": [], "errors": ["SERVER_ERROR"], "final": "", "tokens": 0, "seconds": 0.0,
                         "steps": 0, "messages": [{"role": "error", "content": f"{type(e).__name__}: {e}"}]}
                ok, why = score(task, r["calls"], r["final"])
                if "SERVER_ERROR" in r["errors"]:
                    ok, why = False, "server error"
                for rec in records:                     # every turn carries how its task came out
                    append(turns, rec.set_outcome(task_ok=ok, **({} if ok else {"note": why})))
                row = {"task": task["id"], "cat": task["cat"], "run": run, "mode": mode, "passed": ok,
                       "why": why, "errors": r["errors"], "warnings": r.get("warnings", []),
                       "calls": [n for n, _ in r["calls"]],
                       "invented": sum(1 for n, _ in r["calls"] if n not in TOOLS),
                       "steps": r["steps"], "tokens": r["tokens"], "seconds": round(r["seconds"], 2)}
                rows.append(row)
                with open(transcripts, "a") as f:
                    f.write(json.dumps({**row, "calls": r["calls"], "final": r["final"],
                                        "messages": r["messages"]}, ensure_ascii=False) + "\n")
                if not quiet:
                    mark = "ok  " if ok else "FAIL"
                    extra = f"  [{', '.join(sorted(set(r['errors'])))}]" if r["errors"] else ""
                    print(f"  run {run} {mode:6} {task['id']:16} {mark} {why}{extra}", flush=True)
    return rows


def sign_test(wins_a: int, wins_b: int) -> float:
    """Two-sided exact sign test: how likely a split this uneven is if the modes are equal."""
    n, k = wins_a + wins_b, min(wins_a, wins_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows: List[dict], modes: List[str]) -> dict:
    out: Dict[str, Any] = {"modes": {}}
    for mode in modes:
        mine = [r for r in rows if r["mode"] == mode]
        n = len(mine) or 1
        codes: Dict[str, int] = {}
        warns: Dict[str, int] = {}
        for r in mine:
            for c in r["errors"]:
                codes[c] = codes.get(c, 0) + 1
            for c in r.get("warnings", []):
                warns[c] = warns.get(c, 0) + 1
        out["modes"][mode] = {
            "n": len(mine), "passed": sum(r["passed"] for r in mine),
            "by_category": {c: [sum(r["passed"] for r in mine if r["cat"] == c),
                                sum(1 for r in mine if r["cat"] == c)] for c in CATEGORIES},
            "clean": sum(1 for r in mine if not r["errors"]), "error_codes": codes, "warning_codes": warns,
            "invented_tools": sum(r["invented"] for r in mine),
            "steps": round(sum(r["steps"] for r in mine) / n, 2),
            "tokens": round(sum(r["tokens"] for r in mine) / n),
            "seconds": round(sum(r["seconds"] for r in mine) / n, 2)}
    if len(modes) == 2:
        a, b = modes
        key = lambda r: (r["task"], r["run"])           # noqa: E731
        pa = {key(r): r["passed"] for r in rows if r["mode"] == a}
        pb = {key(r): r["passed"] for r in rows if r["mode"] == b}
        only_a = sorted(k for k in pa if pa[k] and not pb.get(k))
        only_b = sorted(k for k in pb if pb[k] and not pa.get(k))
        out["paired"] = {f"only_{a}": len(only_a), f"only_{b}": len(only_b),
                         "both": sum(1 for k in pa if pa[k] and pb.get(k)),
                         "neither": sum(1 for k in pa if not pa[k] and not pb.get(k)),
                         "p": round(sign_test(len(only_a), len(only_b)), 3),
                         f"tasks_only_{a}": sorted({k[0] for k in only_a}),
                         f"tasks_only_{b}": sorted({k[0] for k in only_b})}
    return out


def report(summary: dict, modes: List[str], title: str) -> str:
    m = summary["modes"]
    line = lambda label, f: f"| {label} | " + " | ".join(f(m[x]) for x in modes) + " |"   # noqa: E731
    out = [f"### {title}", "", "| | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes),
           line("Tasks done right", lambda s: f"{s['passed']} of {s['n']}")]
    for c in CATEGORIES:
        out.append(line(f"- {c}", lambda s, c=c: "{} of {}".format(*s["by_category"][c])))
    out += [line("Runs with no format error", lambda s: f"{s['clean']} of {s['n']}"),
            line("Format errors", lambda s: ", ".join(f"{k} x{v}" for k, v in sorted(s["error_codes"].items())) or "none"),
            line("Lint warnings (ARML only)", lambda s: ", ".join(f"{k} x{v}" for k, v in sorted(s["warning_codes"].items())) or "none"),
            line("Invented tools", lambda s: str(s["invented_tools"])),
            line("Steps per task", lambda s: str(s["steps"])),
            line("Output tokens per task", lambda s: str(s["tokens"])),
            line("Seconds per task", lambda s: str(s["seconds"]))]
    p = summary.get("paired")
    if p:
        a, b = modes
        out += ["", f"Task by task: {a} alone got {p[f'only_{a}']} right, {b} alone got {p[f'only_{b}']}, "
                    f"both {p['both']}, neither {p['neither']} (sign test p = {p['p']}).",
                f"Only {a}: {', '.join(p[f'tasks_only_{a}']) or 'none'}. "
                f"Only {b}: {', '.join(p[f'tasks_only_{b}']) or 'none'}."]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--model", action="append", required=True, help="repeat for several models")
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", ""), help="if your server wants one")
    ap.add_argument("--modes", default="arml,native")
    ap.add_argument("--runs", type=int, default=3, help="times each task is run (the seed changes)")
    ap.add_argument("--reasoning-effort", default="", help="sent as reasoning_effort when set, e.g. none, low, high")
    ap.add_argument("--label", default="", help="names this setup in the title and the result files, "
                    "for what the request cannot say, e.g. llamacpp-q8-thinking-off")
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--no-native-rules", action="store_true",
                    help="native mode without the behaviour rules ARML's prompt carries")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    wanted = {t.strip() for t in a.tasks.split(",") if t.strip()}
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]
    for model in a.model:
        effort = a.reasoning_effort or "default"
        setup = a.label or f"reasoning-{effort}"
        tag = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{model}__{setup}")
        title = f"{model}, {a.label or 'reasoning ' + effort}, {len(tasks)} tasks x {a.runs} runs"
        print(f"\n{title}")
        make = lambda seed: make_chat(a.url, model, a.key, a.reasoning_effort, seed, a.timeout)   # noqa: E731
        rows = run_all(model, make, modes, a.runs, tasks, a.max_steps, not a.no_native_rules, a.out, tag)
        summary = summarize(rows, modes)
        with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
            json.dump({"model": model, "reasoning_effort": effort, "label": a.label, "runs": a.runs, "url": a.url,
                       "native_rules": not a.no_native_rules, "summary": summary, "rows": rows}, f, indent=1)
        text = report(summary, modes, title)
        with open(os.path.join(a.out, f"{tag}.md"), "w") as f:
            f.write(text + "\n")
        print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
