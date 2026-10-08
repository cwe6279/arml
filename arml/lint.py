"""
ARML lint and canonical form — deterministic, no LLM call.

Two jobs, both for collecting ARML logs as training data for smaller models:

    from arml.lint import lint, canonical

    issues = lint(response)                 # what is wrong with it, if anything
    if not any(i.level == "error" for i in issues):
        keep(canonical(response))           # one consistent spelling of the same response

`lint` never repairs; it says what a parser would have had to forgive. A log
kept for training should have no errors, because a small model learns exactly
what it is shown, forgiven mistakes included.

`canonical` rewrites a response that parses into one spelling: canonical tag
names (<o> -> <output>), attributes as key="value", tool JSON compact and with
keys in the order written, one blank line between parts. It changes form only,
never meaning: tool JSON that needs repair is left as written (and linted).

Codes:
    error  EMPTY             no text at all
    error  UNCLOSED          a block was opened and never closed
    error  STRAY_CLOSE       a closing tag with no opening tag
    error  TOOL_NO_NAME      <tool> without name="..."
    error  TOOL_BAD_JSON     tool content is not JSON, even after the usual repairs
    error  NO_ACTION         no tool, output, code, validate or halt: nothing to do
    error  NO_BLOCKS         plain text only (the runtime accepts it as an answer, a model
                             trained on it learns to skip the format)
    warn   TOOL_JSON_REPAIRED tool content parses only after repair
    warn   MIXED_ALIAS       <output>...</o>: shorthand and full name in one block
    warn   STRAY_TEXT        plain text outside blocks next to blocks
    warn   MANY_OUTPUTS      more than one <output>
    warn   OUTPUT_WITH_TOOL  <output> in the same response as a <tool> (the tool runs first,
                             the output is ignored until a later turn)
    info   SHORTHAND         shorthand tags used (<o>)

Behavior checks (from a voice assistant's real logs; regexes, so warnings, not proof):
    error  FAKE_RESULT       visible text writes what only the runtime may write: a tool
                             result, event or step report ("(Event: ...", "step 1 done").
                             The assistant once invented a venue's hours this way.
    warn   ANNOUNCED_NO_ACTION "Let me check ..." / "I'll ..." with no tool, code or validate
                             block: nothing runs between responses, so nothing happens.
    warn   CLAIMED_NO_ACTION "I've sent / saved / booked ..." with no block that does it.
    warn   LONG_SPOKEN       spoken=True: the visible text is over the word budget.
    warn   SPOKEN_NUMBER     spoken=True: a phone number or long code read aloud.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from .parser import ARMLParser, _default_parser
from .types import BlockType, ToolBlock

ACTION_TYPES = {BlockType.TOOL, BlockType.OUTPUT, BlockType.CODE, BlockType.VALIDATE, BlockType.HALT}
DOING_TYPES = {BlockType.TOOL, BlockType.CODE, BlockType.VALIDATE}

# Text only the runtime writes. A harness adds its own markers with lint(reserved=[...]).
RESERVED_MARKERS = [
    r"\(\s*Event\s*:", r"\bstep\s+\d+\s+(?:done|complete[d]?|failed)\b",
    r"\[\s*(?:tool|observation|result)\b[^\]]*\]", r"^\s*Observation\s*:",
]
_ANNOUNCE_RE = re.compile(
    r"\b(?:let me|i'll|i will|i'm going to|i am going to)\s+(?:now\s+|just\s+|quickly\s+)?"
    r"(?:check|look|find|search|pull|get|fetch|send|text|email|book|schedule|create|run|try|ask|"
    r"verify|confirm|research|update|cancel|open|read|set)\b", re.IGNORECASE)
_CLAIM_RE = re.compile(
    r"\b(?:i(?:'ve| have)?\s+(?:sent|saved|booked|scheduled|created|emailed|texted|updated|"
    r"cancelled|canceled|added|noted|set up)|(?:it's|it is)\s+on its way)\b", re.IGNORECASE)


@dataclass
class Issue:
    code: str
    level: str          # "error", "warn" or "info"
    message: str

    def __str__(self) -> str:
        return f"{self.level} {self.code}: {self.message}"


def lint(text: str, parser: Optional[ARMLParser] = None, *, spoken: bool = False,
         word_budget: int = 30, reserved: Optional[List[str]] = None) -> List[Issue]:
    """spoken: the visible text is read aloud (adds LONG_SPOKEN, SPOKEN_NUMBER).
    reserved: extra regexes for text only your runtime writes (FAKE_RESULT)."""
    p = parser or _default_parser
    if not (text or "").strip():
        return [Issue("EMPTY", "error", "no text at all")]
    r = p.parse(text)
    issues: List[Issue] = []

    for tag in r.dangling:
        issues.append(Issue("UNCLOSED", "error", f"<{tag}> opened and never closed"))
    for tag in r.stray_closes:
        issues.append(Issue("STRAY_CLOSE", "error", f"</{tag}> with no <{tag}>"))

    for b in r.tools():
        if not b.get("name"):
            issues.append(Issue("TOOL_NO_NAME", "error", "<tool> without name=\"...\""))
        body = b.content.strip()
        if not body:
            continue
        try:
            json.loads(body)
        except (json.JSONDecodeError, TypeError):
            if b.params():
                issues.append(Issue("TOOL_JSON_REPAIRED", "warn",
                                    f"<tool name=\"{b.get('name', '')}\"> JSON parses only after repair"))
            else:
                issues.append(Issue("TOOL_BAD_JSON", "error",
                                    f"<tool name=\"{b.get('name', '')}\"> content is not JSON"))

    if not r.blocks and not r.dangling:
        issues.append(Issue("NO_BLOCKS", "error", "plain text, no ARML blocks"))
    elif r.blocks and not any(b.block_type in ACTION_TYPES for b in r.blocks):
        issues.append(Issue("NO_ACTION", "error",
                            "blocks but no tool, output, code, validate or halt"))

    outputs = r.by_type(BlockType.OUTPUT)
    if len(outputs) > 1:
        issues.append(Issue("MANY_OUTPUTS", "warn", f"{len(outputs)} <output> blocks"))
    if outputs and r.tools():
        issues.append(Issue("OUTPUT_WITH_TOOL", "warn",
                            "<output> next to a <tool>; the output is not used this turn"))
    if r.blocks and any(isinstance(part, str) for part in r.parts):
        issues.append(Issue("STRAY_TEXT", "warn", "plain text outside blocks"))

    if r.aliased:
        if _mixed_alias(text, p):
            issues.append(Issue("MIXED_ALIAS", "warn", "a block opened and closed with different names"))
        issues.append(Issue("SHORTHAND", "info", "shorthand tags: " + ", ".join(sorted(set(r.aliased)))))

    visible = r.visible_text()
    doing = any(b.block_type in DOING_TYPES for b in r.blocks)
    for marker in RESERVED_MARKERS + list(reserved or []):
        if re.search(marker, visible, re.IGNORECASE | re.MULTILINE):
            issues.append(Issue("FAKE_RESULT", "error",
                                "visible text writes a runtime result; only the harness may"))
            break
    if not doing and _ANNOUNCE_RE.search(visible):
        issues.append(Issue("ANNOUNCED_NO_ACTION", "warn",
                            f"announces work with no block that does it: \"{_ANNOUNCE_RE.search(visible).group(0)}\""))
    if not doing and _CLAIM_RE.search(visible):
        issues.append(Issue("CLAIMED_NO_ACTION", "warn",
                            f"claims an action with no block that does it: \"{_CLAIM_RE.search(visible).group(0)}\""))
    if spoken:
        n = len(visible.split())
        if n > word_budget:
            issues.append(Issue("LONG_SPOKEN", "warn", f"{n} words spoken (budget {word_budget})"))
        if re.search(r"(?<!\w)\+?\d[\d ().-]{8,}\d(?!\w)", visible):
            issues.append(Issue("SPOKEN_NUMBER", "warn", "a phone number or long code read aloud"))
    return issues


def is_clean(text: str, parser: Optional[ARMLParser] = None) -> bool:
    """No errors: fit to keep as a training example."""
    return not any(i.level == "error" for i in lint(text, parser))


def canonical(text: str, parser: Optional[ARMLParser] = None) -> str:
    """The same response in canonical spelling. Remains of broken blocks are not kept."""
    p = parser or _default_parser
    r = p.parse(text)
    out = []
    for part in r.parts:
        if isinstance(part, str):
            out.append(part)
            continue
        attrs = "".join(f' {k}="{v}"' for k, v in part.attributes.items())
        content = part.content
        if isinstance(part, ToolBlock) and content:
            try:
                content = json.dumps(json.loads(content), ensure_ascii=False, separators=(", ", ": "))
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(f"<{part.tag}{attrs}/>" if not content else f"<{part.tag}{attrs}>{content}</{part.tag}>")
    return "\n\n".join(out)


def _mixed_alias(text: str, p: ARMLParser) -> bool:
    """An opening and closing tag of one block spelled differently (<output>...</o>)."""
    for m in p.open_re.finditer(text):
        if m.group(3):
            continue
        names = [t for t in p.tag_to_type if p.canonical_tag(t) == p.canonical_tag(m.group(1))]
        if len(names) < 2:
            continue
        close = re.compile(rf"</({'|'.join(map(re.escape, names))})\s*>", re.IGNORECASE).search(text, m.end())
        if close and close.group(1).lower() != m.group(1).lower():
            return True
    return False
