"""
ARML - AI Response Markup Language

A markup language for structuring AI agent responses.

Usage:
    from arml import parse

    result = parse(llm_response)

    # Get specific blocks
    tools = result.tools()
    output = result.output()
    code_blocks = result.by_tag("code")
    validate_blocks = result.by_tag("validate")

    # Iterate all blocks
    for block in result:
        print(block.tag, block.content)

Active block types: think, tool, output (alias `o`), halt, code,
validate. Reserved: guardrail. The spec, with the grammar as
regexes and a guide to building a harness, is README.md.
"""

from .types import (
    # Enum
    BlockType,

    # Base class
    Block,

    # Specialized blocks (matches the active TAG_TO_TYPE map in parser.py)
    ThinkBlock,
    ToolBlock,
    OutputBlock,
    HaltBlock,
    CodeBlock,
    ValidateBlock,
    GuardrailBlock,
)

from .parser import (
    parse,
    create_parser,
    ParseResult,
    ARMLParser,
    VISIBLE_TYPES,
)
from .stream import StreamParser, Event
from .lint import lint, is_clean, canonical, Issue
from .prompt import format_rules
from .record import TurnRecord

__version__ = "2.1.0"
__all__ = [
    # Main function
    "parse",
    "create_parser",

    # Result
    "ParseResult",
    "ARMLParser",
    "VISIBLE_TYPES",

    # Streaming, lint and canonical form (arml/README.md)
    "StreamParser",
    "Event",
    "lint",
    "is_clean",
    "canonical",
    "Issue",

    # Building a harness: prompt rules and the shared turn-log format
    "format_rules",
    "TurnRecord",

    # Types
    "BlockType",
    "Block",
    "ThinkBlock",
    "ToolBlock",
    "OutputBlock",
    "HaltBlock",
    "CodeBlock",
    "ValidateBlock",
    "GuardrailBlock",
]
