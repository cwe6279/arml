"""
ARML Types

Core data structures for ARML blocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum, auto
import re


def _repair_json_newlines(raw: str) -> str:
    """Escape literal newlines inside JSON string values.

    LLMs sometimes emit bare newlines inside JSON strings instead of the
    required ``\\n`` escape.  This function walks through the raw text,
    tracks whether we are inside a quoted string, and replaces any bare
    ``\\n`` (or ``\\r``) with the proper escape sequence.
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == '\\' and in_string and i + 1 < n:
            # Already-escaped character — keep as-is
            out.append(ch)
            out.append(raw[i + 1])
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if in_string and ch == '\n':
            out.append('\\n')
            i += 1
            continue
        if in_string and ch == '\r':
            out.append('\\r')
            i += 1
            continue
        if in_string and ch == '\t':
            out.append('\\t')
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def _repair_json_structure(raw: str) -> str:
    """Fix common LLM JSON structural errors.

    Handles:
    - Double braces from Llama: {{...}} → {...}
    - Trailing commas: {"a": 1, } → {"a": 1}
    - Stray quotes before close: {"a": 1, "} → {"a": 1}
    - Missing closing brace: {"a": 1 → {"a": 1}
    """
    fixed = raw.strip()

    # Fix double braces (common with Llama models)
    while fixed.startswith('{{') and fixed.endswith('}}'):
        fixed = fixed[1:-1]

    # Fix trailing comma variations: , } or ,"} or , "}
    fixed = re.sub(r',\s*"?\s*}', '}', fixed)

    # Fix missing closing brace
    open_count = fixed.count('{')
    close_count = fixed.count('}')
    if open_count > close_count:
        fixed += '}' * (open_count - close_count)

    return fixed


class BlockType(Enum):
    """Types of ARML blocks.

    Cleaned up 2026-06-15. Six block types are runtime-active; one
    (GUARDRAIL) is reserved for future LLM-emitted refusal signals.
    Anything else the LLM emits is parsed as plain text.
    """

    THINK = auto()      # Internal reasoning, dropped from history
    TOOL = auto()       # ReAct tool invocation
    OUTPUT = auto()     # Final answer; closes the turn
    HALT = auto()       # Kill switch — stop execution
    CODE = auto()       # Run Python (sandbox or container) — see <code>
    VALIDATE = auto()   # Hard QA gate — see <validate task_id="N">

    # Reserved (parser keeps the tag, no runtime path yet)
    GUARDRAIL = auto()  # Future: LLM-emitted refusal signal

    # Custom/Unknown — parser fallback for tags it doesn't recognize
    CUSTOM = auto()


@dataclass
class Block:
    """
    Base ARML block.
    
    Attributes:
        block_type: The type of block (from BlockType enum)
        tag: The original tag name (e.g., "think", "tool")
        content: The text content inside the block
        attributes: Dict of tag attributes
        raw: The original raw text including tags
    """
    block_type: BlockType
    tag: str
    content: str
    attributes: Dict[str, str] = field(default_factory=dict)
    raw: str = ""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(key, default)
    
    def __getitem__(self, key: str) -> str:
        """Get attribute via indexing: block['name']"""
        return self.attributes[key]
    
    def has(self, key: str) -> bool:
        """Check if attribute exists."""
        return key in self.attributes


# =============================================================================
# Specialized Block Types (optional convenience classes)
# =============================================================================

@dataclass
class ThinkBlock(Block):
    """Reasoning/thinking block."""
    
    @property
    def depth(self) -> str:
        return self.get("depth", "normal")
    
    @property
    def visible(self) -> bool:
        return self.get("visible", "true").lower() == "true"


@dataclass
class ToolBlock(Block):
    """Tool invocation block."""
    
    @property
    def name(self) -> str:
        return self.get("name", "")
    
    @property
    def action(self) -> str:
        return self.get("action", "")
    
    @property
    def is_async(self) -> bool:
        return self.get("async", "false").lower() == "true"
    
    def params(self) -> Dict[str, Any]:
        """Parse content as JSON parameters.

        Handles common LLM issues:
        - Literal newlines inside JSON string values (should be \\n)
        - Double braces: {{...}} → {...}
        - Trailing commas: {"a": 1, } → {"a": 1}
        - Stray quotes: {"a": 1, "} → {"a": 1}
        - Missing closing brace
        """
        import json
        raw = self.content
        if not raw or not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try structural repairs (double braces, trailing commas, etc.)
        try:
            repaired = _repair_json_structure(raw)
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try newline repairs
        try:
            repaired = _repair_json_newlines(raw)
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try both repairs combined
        try:
            repaired = _repair_json_structure(_repair_json_newlines(raw))
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}


@dataclass
class OutputBlock(Block):
    """Final output block."""
    
    @property
    def format(self) -> str:
        return self.get("format", "text")
    
    @property
    def audience(self) -> str:
        return self.get("audience", "user")


@dataclass
class GuardrailBlock(Block):
    """Guardrail/safety check block. Reserved — no runtime path yet."""

    @property
    def guardrail_type(self) -> str:
        return self.get("type", "")

    @property
    def action(self) -> str:
        return self.get("action", "warn")


@dataclass
class CodeBlock(Block):
    """Executable code block: the harness runs it (lang, execute, sandbox attributes)."""

    @property
    def language(self) -> str:
        return self.get("lang", "python")

    @property
    def execute(self) -> bool:
        return self.get("execute", "false").lower() == "true"

    @property
    def sandbox(self) -> bool:
        return self.get("sandbox", "true").lower() == "true"


@dataclass
class ValidateBlock(Block):
    """Validation gate block: checks the harness runs before a task counts as done.

    Inline `<check type="...">expected</check>` children are parsed
    out of `content` by the harness (no separate
    block type — checks live only inside <validate>).
    """

    @property
    def task_id(self) -> str:
        return self.get("task_id", "")



@dataclass
class HaltBlock(Block):
    """
    Kill switch block - immediately stop all execution.
    
    When parser encounters this, execution should stop.
    No pending tools should run. No output should be produced.
    """
    
    @property
    def reason(self) -> str:
        """security, error, policy, user, resource, anomaly"""
        return self.get("reason", "error")
    
    @property
    def code(self) -> str:
        """Machine-readable error code."""
        return self.get("code", "")
    
    @property
    def recoverable(self) -> bool:
        """Can execution resume?"""
        return self.get("recoverable", "false").lower() == "true"
    
    @property
    def message(self) -> str:
        """Human-readable halt message."""
        return self.content
