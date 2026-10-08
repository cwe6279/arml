"""
ARML Parser

Extracts ARML blocks from LLM response text. The grammar as regexes,
and how a harness dispatches the blocks, are in README.md.

Usage:
    from arml import parse

    result = parse(llm_response)

    for block in result.blocks:
        print(block.tag, block.content)

    # Convenience methods
    tools = result.tools()
    output = result.output()

    # Generic accessor for newer block types
    code_blocks = result.by_tag("code")
    validate_blocks = result.by_tag("validate")
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .types import (
    Block,
    BlockType,
    ThinkBlock,
    ToolBlock,
    OutputBlock,
    HaltBlock,
    CodeBlock,
    ValidateBlock,
    GuardrailBlock,
)


# =============================================================================
# Tag Mappings
# =============================================================================

# Map tag names to BlockType.
#
# Cleaned up 2026-06-15: dropped <plan>, <step>, <delegate>, <memory>,
# <error>/<e>, <status>, <file> — these were registered without runtime
# paths so the LLM could emit them and they'd be silently discarded by
# the dispatch ladder. <code> joined the active set in r12; <validate>
# joined in this commit. <guardrail> is the one remaining placeholder
# (reserved for future LLM-emitted refusal signals).
#
TAG_TO_TYPE: Dict[str, BlockType] = {
    "think": BlockType.THINK,
    "tool": BlockType.TOOL,
    "output": BlockType.OUTPUT,
    "o": BlockType.OUTPUT,            # shorthand
    "halt": BlockType.HALT,
    "code": BlockType.CODE,
    "validate": BlockType.VALIDATE,
    "guardrail": BlockType.GUARDRAIL,  # placeholder — no runtime today
}

# Map BlockType to specialized class. Same cleanup applied.
TYPE_TO_CLASS: Dict[BlockType, Type[Block]] = {
    BlockType.THINK: ThinkBlock,
    BlockType.TOOL: ToolBlock,
    BlockType.OUTPUT: OutputBlock,
    BlockType.HALT: HaltBlock,
    BlockType.CODE: CodeBlock,
    BlockType.VALIDATE: ValidateBlock,
    BlockType.GUARDRAIL: GuardrailBlock,
}


# Block types whose content the user sees. Every other block is the agent talking to
# the runtime (reasoning, tool calls, code, checks) and is never shown or spoken.
# A voice client that read a silent block aloud is how this rule came about.
VISIBLE_TYPES = {BlockType.OUTPUT}


# =============================================================================
# Parse Result
# =============================================================================

@dataclass
class ParseResult:
    """
    Result of parsing ARML from text.
    
    Attributes:
        blocks: List of all parsed blocks
        text_segments: List of plain text between blocks
        raw: Original input text
    """
    blocks: List[Block] = field(default_factory=list)
    text_segments: List[str] = field(default_factory=list)
    raw: str = ""
    # Plain text and blocks in the order they appeared. Plain text here has
    # fragments of broken blocks removed (see `dangling` and `stray_closes`).
    parts: List[Union[str, Block]] = field(default_factory=list)
    dangling: List[str] = field(default_factory=list)      # tags opened, never closed
    stray_closes: List[str] = field(default_factory=list)  # closing tags with no opening
    aliased: List[str] = field(default_factory=list)       # shorthand tags rewritten (o -> output)
    
    # Convenience methods
    
    def tools(self) -> List[ToolBlock]:
        """Get all tool blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.TOOL]
    
    def output(self) -> Optional[OutputBlock]:
        """Get the output block (usually last one)."""
        outputs = [b for b in self.blocks if b.block_type == BlockType.OUTPUT]
        return outputs[-1] if outputs else None

    def think(self) -> List[ThinkBlock]:
        """Get all think blocks."""
        return [b for b in self.blocks if b.block_type == BlockType.THINK]

    def halt(self) -> Optional["HaltBlock"]:
        """Get halt block if present. CHECK THIS FIRST."""
        halts = [b for b in self.blocks if b.block_type == BlockType.HALT]
        return halts[0] if halts else None
    
    def is_halted(self) -> bool:
        """Check if execution should stop. CHECK THIS FIRST."""
        return self.halt() is not None
    
    def by_type(self, block_type: BlockType) -> List[Block]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks if b.block_type == block_type]
    
    def by_tag(self, tag: str) -> List[Block]:
        """Get all blocks with a specific tag name."""
        return [b for b in self.blocks if b.tag == tag]
    
    def first(self, block_type: BlockType) -> Optional[Block]:
        """Get first block of a type."""
        matches = self.by_type(block_type)
        return matches[0] if matches else None
    
    def has(self, block_type: BlockType) -> bool:
        """Check if any block of type exists."""
        return any(b.block_type == block_type for b in self.blocks)
    
    def text(self) -> str:
        """Get all plain text (non-ARML content) joined."""
        return "\n".join(self.text_segments)
    
    def visible_text(self) -> str:
        """What the user may see or hear: plain text and the content of visible
        blocks, in order. Never the content of a silent block, and never the
        remains of a block that was opened and not closed."""
        pieces = []
        for part in self.parts:
            if isinstance(part, str):
                pieces.append(part)
            elif part.block_type in VISIBLE_TYPES and part.content:
                pieces.append(part.content)
        return "\n\n".join(pieces)

    def is_well_formed(self) -> bool:
        """Every block that was opened was closed, and nothing was closed that was not opened."""
        return not self.dangling and not self.stray_closes

    def __len__(self) -> int:
        return len(self.blocks)
    
    def __iter__(self):
        return iter(self.blocks)
    
    def __bool__(self) -> bool:
        return len(self.blocks) > 0


# =============================================================================
# Parser
# =============================================================================

class ARMLParser:
    """
    Parser for ARML markup.

    Usage:
        parser = ARMLParser()
        result = parser.parse(text)

        # Or register custom tags
        parser.register("mytag", BlockType.CUSTOM, MyCustomBlock)
        result = parser.parse(text)
    """

    def __init__(self):
        self.tag_to_type = dict(TAG_TO_TYPE)
        self.type_to_class = dict(TYPE_TO_CLASS)
        self._custom_handlers: Dict[str, Callable] = {}
        self._pattern_cache: Optional[re.Pattern] = None
    
    def register(
        self,
        tag: str,
        block_type: BlockType = BlockType.CUSTOM,
        block_class: Type[Block] = None,
        handler: Callable[[str, Dict], Block] = None,
    ):
        """
        Register a custom tag.

        Args:
            tag: Tag name (e.g., "mytag" for <mytag>)
            block_type: BlockType to assign
            block_class: Optional specialized Block subclass
            handler: Optional custom handler function(content, attributes) -> Block
        """
        self.tag_to_type[tag] = block_type
        if block_class:
            self.type_to_class[block_type] = block_class
        if handler:
            self._custom_handlers[tag] = handler
        # Invalidate pattern cache so it rebuilds with new tag
        self._pattern_cache = None
        return self
    
    def _get_pattern(self) -> re.Pattern:
        """Get compiled regex pattern, building/caching on first use."""
        if self._pattern_cache is None:
            tags = "|".join(re.escape(tag) for tag in self.tag_to_type.keys())
            # Use backreference \1 to match closing tag to opening tag name
            # This ensures <tool>...</tool> won't match </think> etc.
            # Self-closing tags handled separately: <tag/>
            # The (?<!`) lookbehind keeps backtick-quoted mentions in prose
            # (e.g. "the `<code execute=\"true\">` block needs...") from
            # parsing as real blocks — without it, the following prose gets
            # dispatched as executable code.
            pattern = rf'(?<!`)<({tags})(\s[^>]*)?>(.+?)</\1\s*>|(?<!`)<({tags})(\s[^>]*)?/>'
            self._pattern_cache = re.compile(pattern, re.DOTALL | re.IGNORECASE)
            self._open_re = re.compile(rf'(?<!`)<({tags})(\s[^>]*?)?(/?)>', re.IGNORECASE)
            self._close_re = re.compile(rf'(?<!`)</({tags})\s*>', re.IGNORECASE)
            # Only the built-in shorthands; a tag someone registers keeps its own name.
            aliases = [t for t in TAG_TO_TYPE if self.tag_to_type.get(t) == TAG_TO_TYPE[t]
                       and t not in self._custom_handlers and t != self.canonical_tag(t)]
            self._alias_re = (re.compile(rf'(?<!`)<(/?)({"|".join(map(re.escape, aliases))})(?=[\s/>])',
                                         re.IGNORECASE) if aliases else None)
        return self._pattern_cache

    def canonical_tag(self, tag: str) -> str:
        """The first tag registered for a block type is its canonical name: o -> output."""
        block_type = self.tag_to_type.get(tag.lower())
        if block_type is None or block_type == BlockType.CUSTOM:
            return tag.lower()
        return next(t for t, bt in self.tag_to_type.items() if bt == block_type)

    @property
    def open_re(self) -> re.Pattern:
        """An opening tag of any known block: groups (tag, attributes, '/' if self-closing)."""
        self._get_pattern()
        return self._open_re

    @property
    def close_re(self) -> re.Pattern:
        """A closing tag of any known block: group (tag)."""
        self._get_pattern()
        return self._close_re

    def normalize_aliases(self, text: str) -> tuple:
        """Rewrite shorthand tags to canonical ones, so <output>...</o> is one block.
        Returns (text, [aliases rewritten])."""
        self._get_pattern()
        if not self._alias_re or not text:
            return text, []
        found = []

        def swap(m):
            found.append(m.group(2).lower())
            return f"<{m.group(1)}{self.canonical_tag(m.group(2))}"
        return self._alias_re.sub(swap, text), found

    def _plain(self, segment: str, result: "ParseResult") -> str:
        """Plain text between blocks, minus the remains of broken blocks: an opening
        tag with no close hides everything after it, a stray closing tag is dropped."""
        for m in self._close_re.finditer(segment):
            result.stray_closes.append(self.canonical_tag(m.group(1)))
        segment = self._close_re.sub("", segment)
        m = self._open_re.search(segment)
        if m and not m.group(3):
            result.dangling.append(self.canonical_tag(m.group(1)))
            segment = segment[:m.start()]
        return segment.strip()

    def parse(self, text: str) -> ParseResult:
        """
        Parse ARML blocks from text.

        Args:
            text: Input text potentially containing ARML blocks

        Returns:
            ParseResult with blocks and text segments
        """
        if not text:
            return ParseResult(raw=text)

        result = ParseResult(raw=text)
        pattern = self._get_pattern()
        text, result.aliased = self.normalize_aliases(text)

        last_end = 0

        for match in pattern.finditer(text):
            # Capture text before this block
            if match.start() > last_end:
                text_segment = text[last_end:match.start()].strip()
                if text_segment:
                    result.text_segments.append(text_segment)
                    plain = self._plain(text_segment, result)
                    if plain:
                        result.parts.append(plain)

            # Parse the block
            if match.group(1):  # Normal tag with content
                tag = match.group(1).lower()
                attrs_str = match.group(2) or ""
                content = match.group(3).strip()
            else:  # Self-closing tag
                tag = match.group(4).lower()
                attrs_str = match.group(5) or ""
                content = ""

            # Parse attributes
            attributes = self._parse_attributes(attrs_str)

            # Create block
            block = self._create_block(tag, content, attributes, match.group(0))
            result.blocks.append(block)
            result.parts.append(block)

            last_end = match.end()

        # Capture trailing text
        if last_end < len(text):
            text_segment = text[last_end:].strip()
            if text_segment:
                result.text_segments.append(text_segment)
                plain = self._plain(text_segment, result)
                if plain:
                    result.parts.append(plain)

        return result
    
    def _parse_attributes(self, attrs_str: str) -> Dict[str, str]:
        """Parse attributes from string like ' name="value" flag'."""
        attrs = {}
        if not attrs_str:
            return attrs
        
        # Match key="value" or key='value' or key=value or just key (flag)
        pattern = r'(\w+)(?:=(?:"([^"]*)"|\'([^\']*)\'|(\S+)))?'
        
        for match in re.finditer(pattern, attrs_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4) or "true"
            attrs[key] = value
        
        return attrs
    
    def _create_block(
        self, 
        tag: str, 
        content: str, 
        attributes: Dict[str, str],
        raw: str,
    ) -> Block:
        """Create appropriate Block instance."""
        
        # Check for custom handler
        if tag in self._custom_handlers:
            return self._custom_handlers[tag](content, attributes)
        
        # Get block type
        block_type = self.tag_to_type.get(tag, BlockType.CUSTOM)
        
        # Get block class
        block_class = self.type_to_class.get(block_type, Block)
        
        return block_class(
            block_type=block_type,
            tag=tag,
            content=content,
            attributes=attributes,
            raw=raw,
        )


# =============================================================================
# Convenience Function
# =============================================================================

# Default parser instance
_default_parser = ARMLParser()


def parse(text: str) -> ParseResult:
    """
    Parse ARML blocks from text.
    
    This is the main entry point for parsing.
    
    Args:
        text: Input text potentially containing ARML blocks
        
    Returns:
        ParseResult with blocks and text segments
        
    Example:
        from arml import parse
        
        result = parse('''
            <think>Let me analyze this...</think>
            <tool name="search">{"query": "test"}</tool>
            <output>Here's what I found.</output>
        ''')
        
        print(result.tools())   # [ToolBlock(...)]
        print(result.output())  # OutputBlock(...)
    """
    return _default_parser.parse(text)


def create_parser() -> ARMLParser:
    """Create a new parser instance (for custom tag registration)."""
    return ARMLParser()
