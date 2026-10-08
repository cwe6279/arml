"""
ARML format validator — deterministic, no LLM call.

Classifies a raw model response *after* the tag parser has found nothing
dispatchable, so the ReAct loop can make an explicit decision instead of
silently promoting whatever came back to a final answer. Motivating
failure (r19p): qwen3.8 emitted a complete ``write_file`` JSON payload
whose ``<tool>`` wrapper was missing / whose ``</tool>`` was cut off
after a 17 KB ``content`` value. The catch-all turned the JSON into the
user-facing answer, the file was never written, and the next turn the
model claimed it had been.

Verdicts:
    OK                 tag parser already found blocks — nothing to do
    EMPTY              no text at all (thinking-only generations, cut-offs)
    UNTERMINATED_TOOL  ``<tool name="X">`` opened, never closed
    BARE_TOOL_JSON     whole response is a JSON object whose keys match
                       exactly one registered tool's schema
    PLAIN_TEXT         prose, no tags — legitimately a final answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .types import _repair_json_newlines, _repair_json_structure

OK = "OK"
EMPTY = "EMPTY"
UNTERMINATED_TOOL = "UNTERMINATED_TOOL"
BARE_TOOL_JSON = "BARE_TOOL_JSON"
PLAIN_TEXT = "PLAIN_TEXT"

_OPEN_TOOL_RE = re.compile(r'<tool\s+[^>]*?name\s*=\s*"([^"]+)"[^>]*>', re.IGNORECASE)
_CLOSE_TOOL_RE = re.compile(r'</tool\s*>', re.IGNORECASE)
_FENCE_RE = re.compile(r'^\s*```(?:json|arml|xml|html)?\s*\n?(.*?)\n?\s*```\s*$', re.DOTALL | re.IGNORECASE)


@dataclass
class Verdict:
    kind: str
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    detail: str = ""
    # Which schema keys (if any) tie-broke a BARE_TOOL_JSON match.
    matched_keys: list = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        """True when a tool call can be dispatched from this verdict."""
        return self.tool is not None and isinstance(self.params, dict)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _try_json(raw: str) -> Optional[Dict[str, Any]]:
    """json.loads with the same repair ladder ToolBlock.params() uses."""
    if not raw or not raw.strip():
        return None
    attempts = (
        lambda s: s,
        _repair_json_structure,
        _repair_json_newlines,
        lambda s: _repair_json_structure(_repair_json_newlines(s)),
    )
    for fix in attempts:
        try:
            obj = json.loads(fix(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _close_truncated_json(raw: str) -> str:
    """Best-effort closer for JSON cut off mid-generation.

    Terminates an open string, then appends the missing ``]``/``}``
    in stack order. Content past the cut is lost, but a ``write_file``
    whose ``content`` was truncated still dispatches with a partial
    body — the caller decides whether that's acceptable.
    """
    stack = []
    in_str = False
    esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in '{[':
            stack.append(ch)
        elif ch in '}]':
            if stack:
                stack.pop()
    out = raw
    if in_str:
        out += '"'
    out = re.sub(r',\s*$', '', out)
    for ch in reversed(stack):
        out += '}' if ch == '{' else ']'
    return out


def _match_tool_by_keys(params: Dict[str, Any], tool_schemas: Dict[str, dict]) -> tuple:
    """Return (tool_name, matched_keys) if params fit exactly one schema.

    A schema matches when every *required* key is present and every
    present key is declared. Among matches, prefer the one with the most
    required keys satisfied; ambiguity → no match (better to ask than
    to fire the wrong side-effecting tool).
    """
    keys = set(params.keys())
    if not keys:
        return None, []
    scored = []
    for name, schema in (tool_schemas or {}).items():
        if not isinstance(schema, dict) or not schema:
            continue
        declared = {k for k, v in schema.items() if isinstance(v, dict)}
        required = {k for k, v in schema.items()
                    if isinstance(v, dict) and v.get("required")}
        if not required.issubset(keys):
            continue
        if not keys.issubset(declared):
            continue
        scored.append((len(required), len(declared), name, sorted(required)))
    if not scored:
        return None, []
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None, []  # tie → ambiguous
    return scored[0][2], scored[0][3]


def validate_arml(raw: str, tool_schemas: Optional[Dict[str, dict]] = None,
                  parsed_blocks: int = 0) -> Verdict:
    """Classify a raw response the tag parser could not dispatch.

    Args:
        raw: model text, already stripped of the provider prefix.
        tool_schemas: ``{tool_name: {param: {type, required, ...}}}`` —
            the same dict-schemas ``Tool.parameters`` carries.
        parsed_blocks: number of ARML blocks the parser found; > 0 → OK.
    """
    text = (raw or "").strip()
    if not text:
        return Verdict(EMPTY, detail="empty response")

    # --- unterminated <tool name="..."> ------------------------------
    # Checked before the parsed_blocks shortcut: the usual shape is a
    # well-formed <think> followed by a <tool> whose close was cut off,
    # and the parser happily reports the <think> as a block.
    opens = list(_OPEN_TOOL_RE.finditer(text))
    closes = list(_CLOSE_TOOL_RE.finditer(text))
    if opens and len(closes) < len(opens):
        last = opens[-1]
        name = last.group(1).strip().lower()
        body = text[last.end():]
        detail = (f"<tool name=\"{name}\"> was opened but never closed "
                  f"({len(body)} chars of body)")
        params = _try_json(body)
        if params is not None:
            # Body is complete JSON — only the closing tag went missing.
            # Safe to dispatch.
            return Verdict(UNTERMINATED_TOOL, tool=name, params=params,
                           detail=detail + "; JSON body intact")
        # Body itself is cut off. We could force-close it, but dispatching
        # a write_file with half its content would silently produce a
        # truncated file that the model then reports as done. Refuse and
        # let the retry prompt ask for a complete (or split) payload.
        cut = _try_json(_close_truncated_json(body))
        how = "cut off mid-JSON" if cut is not None else "unrecoverable"
        return Verdict(UNTERMINATED_TOOL, tool=name, params=None,
                       detail=f"{detail}; JSON body {how}")

    if parsed_blocks > 0:
        return Verdict(OK)

    # --- bare JSON object matching a tool schema ----------------------
    candidate = _strip_fence(text)
    if candidate.lstrip().startswith('{'):
        params = _try_json(candidate)
        if params is None and candidate.count('{') > candidate.count('}'):
            params = _try_json(_close_truncated_json(candidate))
        if params is not None:
            name, matched = _match_tool_by_keys(params, tool_schemas or {})
            if name:
                return Verdict(BARE_TOOL_JSON, tool=name, params=params,
                               matched_keys=matched,
                               detail=f"bare JSON matched tool '{name}' by keys {matched}")
            return Verdict(PLAIN_TEXT,
                           detail="bare JSON object that matches no tool schema")

    return Verdict(PLAIN_TEXT)
