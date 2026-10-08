"""
ARML format rules for a system prompt — the part every harness needs to teach a
model to write ARML. Your own instructions (identity, domain rules) go around it.

    from arml.prompt import format_rules

    system = my_identity + "\\n\\n" + format_rules(tools=[
        {"name": "weather", "description": "Current weather for a city.",
         "parameters": {"city": {"type": "string", "required": True}}},
    ])

The rules are the ones that held up in production (a coding agent and a
voice assistant). Each exists because a model broke it; keep them when you edit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


def tool_list(tools: Iterable[Dict[str, Any]]) -> str:
    """Tools as the model should see them: name, description, exact parameter names."""
    lines: List[str] = []
    for t in tools:
        lines.append(f"- {t['name']}: {t.get('description', '').strip()}")
        for pname, spec in (t.get("parameters") or {}).items():
            spec = spec if isinstance(spec, dict) else {"type": str(spec)}
            need = "required" if spec.get("required") else "optional"
            desc = f" — {spec['description']}" if spec.get("description") else ""
            lines.append(f"    {pname} ({spec.get('type', 'string')}, {need}){desc}")
    return "\n".join(lines)


def format_rules(tools: Optional[Iterable[Dict[str, Any]]] = None,
                 code: bool = False, spoken: bool = False) -> str:
    """The ARML section of a system prompt.

    tools:  tool specs ({name, description, parameters}); listed with exact names.
    code:   also teach <code> (only if your harness runs it).
    spoken: the visible text is read aloud as it streams (voice clients).
    """
    tools = list(tools or [])
    example_tool = tools[0]["name"] if tools else "tool_name"
    example_params = {p: "value" for p in list((tools[0].get("parameters") or {}))[:1]} if tools else {"parameter_name": "value"}
    parts = []
    if tools:
        parts.append("AVAILABLE TOOLS:\n" + tool_list(tools))
    no_output_with_tool = (
        "- With a <tool>, an <output> may only acknowledge (a few words, spoken while the tool runs); "
        "the answer comes after its result." if spoken else
        "- No <output> in the same response as a <tool>: the tool runs first, and the answer comes after its result.")
    parts.append(f"""RESPONSE FORMAT — structure every response with ARML blocks:

<think>your reasoning; the user never sees it</think>
<tool name="{example_tool}">{json.dumps(example_params)}</tool>
<output>your answer to the user</output>{'''
<code lang="python" execute="true">code to run</code>''' if code else ''}
<halt reason="...">stop: something is wrong and continuing would do harm</halt>

RULES:
- Every response uses blocks. Plain text outside blocks is not a valid response.
- Close every block you open, with the same tag name.
- Only the content of <output> reaches the user. Everything else is between you and the runtime.
- Use only tools from AVAILABLE TOOLS, with their exact parameter names.
- Tool content is valid JSON: double quotes, no trailing commas, no comments.
- One <tool> per response; wait for its result before the next step.
{no_output_with_tool}
- Nothing runs between responses. Do not announce work ("Let me check that") without doing it in the same response with a <tool>.
- An action happens only through a <tool>. Never say you sent, saved or created something without the <tool> that does it.
- Only the runtime writes tool results. Never write a result, event or "step done" yourself; wait for it.
- Facts that change (the time, the date, a calendar, prices, opening hours) come from a tool result in this conversation, never from memory or earlier notes.
- When you hand work to a tool, put everything it needs into the arguments: the place, the dates, the names you already know.
- When a request changes (new dates, a new target), cancel or replace the work started for the old one.
- To mention a tag in prose, put it in backticks: `<tool>`.""")
    if spoken:
        parts.append("""SPOKEN REPLIES:
- Your <output> is read aloud as it arrives. Keep it short and natural to hear.
- When you start a task, put the <tool> first and keep the <output> to a few words ("On it."). Do not read your plan out loud; plans go in <think>.
- Never read out phone numbers, codes, or text the user already has in a message.
- If what you heard is garbled, ask one short question instead of acting on it. If the user is talking to someone else, answer with <output/>: nothing is said.
- Use a camera or screen only when the request needs it.""")
    return "\n\n".join(parts)
