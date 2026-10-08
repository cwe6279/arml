"""
An example harness built on ARML: one file, no dependencies, about 100 lines.
It shows the whole loop: teach the model ARML, stream the visible text, run a
tool, feed the result back, and log every model response in the shared format.

Works with any server that speaks the OpenAI-compatible chat API:

    llama.cpp   llama-server -m model.gguf               --url http://localhost:8080/v1
    LM Studio   (start the local server)                 --url http://localhost:1234/v1
    Ollama                                               --url http://localhost:11434/v1
    vLLM        vllm serve <model>                       --url http://localhost:8000/v1

    python examples/minimal_harness.py --url http://localhost:11434/v1 --model qwen3:8b

After a reply, mark how it went (that is the lesson in the log):
    /good  right  (accepted)    /bad  wrong  (corrected)    /again  had to re-ask  (repeated)
Typing your next message marks the last reply accepted. /quit to end.
"""

import argparse
import ast
import json
import operator
import os
import sys
import urllib.request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arml import StreamParser, format_rules, parse                       # noqa: E402
from arml.record import TurnRecord, append                               # noqa: E402

# ── tools: swap in your own ───────────────────────────────────────────
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expression):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, (ast.BinOp, ast.UnaryOp)) and type(n.op) in _OPS:
            args = (ev(n.left), ev(n.right)) if isinstance(n, ast.BinOp) else (ev(n.operand),)
            return _OPS[type(n.op)](*args)
        raise ValueError("only numbers and + - * / **")
    return str(ev(ast.parse(expression, mode="eval").body))


TOOLS = {
    "calculate": ({"description": "Exact arithmetic.",
                   "parameters": {"expression": {"type": "string", "required": True}}},
                  lambda p: calculate(p["expression"])),
    "clock": ({"description": "The current local date and time.", "parameters": {}},
              lambda p: datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z")),
}
SYSTEM = "You are a careful assistant.\n\n" + format_rules(
    [{"name": n, **spec} for n, (spec, _) in TOOLS.items()])


# ── the model: OpenAI-compatible streaming chat ─────────────────────
def stream_chat(url, model, messages, key=""):
    body = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(f"{url}/chat/completions", body, headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0].get("delta", {})
            yield delta.get("content") or ""


def respond(args, messages):
    """Stream one response: show only visible text, return the raw response."""
    sp, raw = StreamParser(), []
    for chunk in stream_chat(args.url, args.model, messages, args.key):
        raw.append(chunk)
        for ev in sp.feed(chunk):
            print(ev.text if ev.kind == "text" else "", end="", flush=True)
    for ev in sp.close():
        print(ev.text if ev.kind == "text" else "", end="", flush=True)
    print()
    return "".join(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", ""), help="if your server wants one")
    ap.add_argument("--log", default="arml-turns.jsonl")
    ap.add_argument("--max-steps", type=int, default=6, help="tool calls per question")
    args = ap.parse_args()

    messages, pending = [{"role": "system", "content": SYSTEM}], []
    session, turn = datetime.now().strftime("%Y%m%d-%H%M%S"), 0

    def flush(user=None):                  # write the last reply's records once its outcome is known
        if user and pending and "user" not in pending[-1].outcome:
            pending[-1].set_outcome(user=user)
        for rec in pending:
            append(args.log, rec)
        pending.clear()

    print(f"ARML example harness: {args.model} at {args.url}. Logging to {args.log}.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            text = "/quit"
        if text == "/quit":
            return flush()
        if text in ("/good", "/bad", "/again"):
            flush({"/good": "accepted", "/bad": "corrected", "/again": "repeated"}[text])
            continue
        flush("accepted")
        messages.append({"role": "user", "content": text})

        for _ in range(args.max_steps):
            turn += 1
            raw = respond(args, messages)
            rec = TurnRecord.from_response(raw, model=args.model, harness="arml-example/1",
                                           session=session, turn=turn, context=messages[1:], system=SYSTEM)
            messages.append({"role": "assistant", "content": raw})
            pending.append(rec)
            if not rec.clean:
                print(f"  [arml lint] {', '.join(rec.lint)}")
            r = parse(raw)
            if r.is_halted() or not r.tools():
                break
            tool = r.tools()[0]
            spec_fn = TOOLS.get(tool.name)
            try:
                result = spec_fn[1](tool.params()) if spec_fn else f"no tool named {tool.name}"
                rec.set_outcome(tool_ok=spec_fn is not None)
            except Exception as e:
                result = f"error: {e}"
                rec.set_outcome(tool_ok=False, tool_error=str(e)[:200])
            print(f"  [{tool.name}] {result}")
            messages.append({"role": "user", "content": f"Result of {tool.name}: {result}"})


if __name__ == "__main__":
    main()
