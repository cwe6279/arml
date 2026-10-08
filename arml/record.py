"""
ARML turn records — one shared log format, so logs from every harness built on
ARML can be pooled and used to train smaller models on agent tasks and
conversation.

One JSON line per model response (schema "arml.turn/1"):

    from arml.record import TurnRecord, append

    rec = TurnRecord.from_response(
        response,                       # the model's raw text
        model="claude-haiku-5-5", harness="my-harness/1.0",
        session="s-42", turn=7,
        context=[{"role": "user", "content": "What's the weather?"}],
        system=system_prompt,           # stored as a hash unless keep_system=True
    )
    ...run the tool, show the output...
    rec.set_outcome(tool_ok=True, user="accepted")
    append("arml-turns.jsonl", rec)

Text is redacted before it is stored (emails, phone numbers, card-like numbers,
API keys); pass your own `redact` to do more, or redact=None to keep text as is.

Why the outcome matters: lint tells you a response was well-formed, not that it
was right. A clean response the user had to correct, interrupt or repeat is a
lesson too, and the one a small model most needs.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional

from .lint import canonical, lint
from .parser import _default_parser
from .types import BlockType

SCHEMA = "arml.turn/1"

# How the user took the response, when the harness can tell.
USER_REACTIONS = {"accepted", "corrected", "repeated", "interrupted", "ignored"}

_REDACTIONS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (re.compile(r"\b(?:sk|pk|xox[bap]|ghp|AKIA)[-_A-Za-z0-9]{12,}\b"), "<key>"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "<number>"),
    (re.compile(r"(?<!\w)(?:\+?\d{1,2}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"), "<phone>"),
]


def basic_redact(text: str) -> str:
    """Emails, API keys, card-like numbers and phone numbers. Not a privacy review."""
    for pattern, label in _REDACTIONS:
        text = pattern.sub(label, text)
    return text


@dataclass
class TurnRecord:
    response: str
    model: str
    harness: str
    session: str = ""
    turn: int = 0
    context: List[Dict[str, str]] = field(default_factory=list)
    system_sha256: str = ""
    system: Optional[str] = None
    canonical: Optional[str] = None
    lint: List[str] = field(default_factory=list)
    clean: bool = False
    blocks: List[str] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema: str = SCHEMA

    @classmethod
    def from_response(cls, response: str, *, model: str, harness: str, session: str = "",
                      turn: int = 0, context: Optional[List[Dict[str, str]]] = None,
                      system: Optional[str] = None, keep_system: bool = False,
                      redact: Optional[Callable[[str], str]] = basic_redact,
                      tags: Optional[List[str]] = None, spoken: bool = False) -> "TurnRecord":
        clean_text = redact or (lambda s: s)
        issues = lint(response, spoken=spoken)
        errors = [i for i in issues if i.level == "error"]
        parsed = _default_parser.parse(response)
        return cls(
            response=clean_text(response),
            model=model, harness=harness, session=session, turn=turn,
            context=[{"role": m.get("role", "user"), "content": clean_text(str(m.get("content", "")))}
                     for m in (context or [])],
            system_sha256=hashlib.sha256(system.encode()).hexdigest() if system else "",
            system=clean_text(system) if (system and keep_system) else None,
            canonical=clean_text(canonical(response)) if not errors else None,
            lint=[i.code for i in issues],
            clean=not errors,
            blocks=[f"tool:{b.get('name', '')}" if b.block_type == BlockType.TOOL else b.tag
                    for b in parsed.blocks],
            tags=list(tags or []) + (["spoken"] if spoken else []),
        )

    def set_outcome(self, *, tool_ok: Optional[bool] = None, tool_error: Optional[str] = None,
                    user: Optional[str] = None, note: Optional[str] = None) -> "TurnRecord":
        """What happened next: the tool's result and how the user took the response."""
        if user is not None and user not in USER_REACTIONS:
            raise ValueError(f"user reaction must be one of {sorted(USER_REACTIONS)}")
        for key, value in (("tool_ok", tool_ok), ("tool_error", tool_error),
                           ("user", user), ("note", note)):
            if value is not None:
                self.outcome[key] = value
        return self

    def good(self) -> bool:
        """Clean, and nothing in the outcome says it went wrong."""
        return (self.clean and self.outcome.get("tool_ok") is not False
                and self.outcome.get("user") not in {"corrected", "repeated", "interrupted"})

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def append(path: str, record: TurnRecord) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def read(path: str) -> Iterator[TurnRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("schema") == SCHEMA:
                    yield TurnRecord(**d)


def training_example(record: TurnRecord, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A chat-format example (system, context, assistant = canonical ARML) from a good
    record, else None. Pass the system prompt if the record only kept its hash."""
    if not record.good() or not record.canonical:
        return None
    messages = []
    sys_text = system if system is not None else record.system
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages += [dict(m) for m in record.context]
    messages.append({"role": "assistant", "content": record.canonical})
    return {"messages": messages, "meta": {"model": record.model, "harness": record.harness,
                                           "id": record.id}}
