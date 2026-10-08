"""
ARML streaming parser.

Reads a response while the model is still writing it, for clients that show or
speak text as it arrives (chat UIs, voice). Uses the same tags and regexes as
`arml.parse`, so a streamed response yields the same blocks as the whole one.

    from arml.stream import StreamParser

    sp = StreamParser()
    for chunk in llm_stream:
        for ev in sp.feed(chunk):
            if ev.kind == "text":
                speak(ev.text)          # plain text and <output> content only
            else:
                run(ev.block)           # a block, once its closing tag arrives
    for ev in sp.close():
        ...

Rules (battle-tested on a voice assistant, see arml/README.md):

1. From an opening tag until its closing tag, nothing is released. A silent
   block can hold whole sentences; a client that splits text at full stops
   before blocks are removed cuts the block in half, the halves no longer look
   like a block, and the assistant reads its own notes aloud.
2. The content of a visible block (<output>) is released as it arrives, minus
   a possible partial closing tag at the end of the buffer.
3. A "<" that could still become a tag is held until it is decided.
4. A block never closed is dropped, not released, and reported in `dangling`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .parser import ARMLParser, VISIBLE_TYPES, _default_parser
from .types import Block


@dataclass
class Event:
    kind: str                       # "text" or "block"
    text: str = ""
    block: Optional[Block] = None


class StreamParser:
    def __init__(self, parser: Optional[ARMLParser] = None):
        self.parser = parser or _default_parser
        self.blocks: List[Block] = []
        self.dangling: List[str] = []
        self.stray_closes: List[str] = []
        self._buf = ""
        self._prev = ""             # last character released, for the `<tag>` backtick rule
        self._open = None           # (tag, attributes, opening tag text) while inside a block
        self._content = ""          # content of the open block so far
        self._tags = [t.lower() for t in self.parser.tag_to_type]

    # ── public ─────────────────────────────────────────────────────
    def feed(self, chunk: str) -> List[Event]:
        self._buf += chunk or ""
        out: List[Event] = []
        while self._step(out):
            pass
        return out

    def close(self) -> List[Event]:
        """End of the response: release held plain text, drop an unclosed block."""
        out: List[Event] = []
        if self._open is not None:
            self.dangling.append(self._open[0])
            if self._visible(self._open[0]):
                self._release_text(self._buf, out)   # the user already has the start of it
            self._open, self._content, self._buf = None, "", ""
        elif self._buf:
            self._release_text(self._buf, out)
            self._buf = ""
        return out

    def is_well_formed(self) -> bool:
        return not self.dangling and not self.stray_closes

    # ── one step: True when the buffer changed and another step may help ──
    def _step(self, out: List[Event]) -> bool:
        p = self.parser
        if self._open is None:
            m = self._find_open(self._buf)
            if m is None:
                hold = self._partial_tag_at(self._buf)
                self._release_text(self._buf[:hold], out)
                self._buf = self._buf[hold:]
                return False
            self._release_text(self._buf[:m.start()], out)
            tag = p.canonical_tag(m.group(1))
            attrs = m.group(2) or ""
            self._buf = self._buf[m.end():]
            if m.group(3):                                  # <tag/>
                self._emit_block(tag, "", attrs, m.group(0), out)
            else:
                self._open, self._content = (tag, attrs, m.group(0)), ""
            return True

        tag, attrs, opening = self._open
        close = re.compile(rf"</(?:{self._names(tag)})\s*>", re.IGNORECASE).search(self._buf)
        if close is None:
            if self._visible(tag):
                hold = self._partial_tag_at(self._buf)
                self._release_text(self._buf[:hold], out)
                self._content += self._buf[:hold]
                self._buf = self._buf[hold:]
            return False
        body = self._buf[:close.start()]
        if self._visible(tag):
            self._release_text(body, out)
        self._content += body
        self._buf = self._buf[close.end():]
        self._open = None
        if self._content.strip():
            self._emit_block(tag, self._content.strip(), attrs,
                             f"{opening}{self._content}</{tag}>", out)
        # An empty <tag></tag> is not a block in arml.parse either.
        self._content = ""
        return True

    # ── helpers ────────────────────────────────────────────────────
    def _find_open(self, text: str):
        for m in self.parser.open_re.finditer(text):
            if m.start() == 0 and self._prev == "`":
                continue                                    # `<tag>` quoted in prose
            return m
        return None

    def _names(self, canonical: str) -> str:
        """Closing names for a block: its canonical tag and any shorthand (output|o)."""
        names = [t for t in self.parser.tag_to_type if self.parser.canonical_tag(t) == canonical]
        return "|".join(re.escape(n) for n in names or [canonical])

    def _visible(self, tag: str) -> bool:
        return self.parser.tag_to_type.get(tag) in VISIBLE_TYPES

    def _partial_tag_at(self, text: str) -> int:
        """Index from which the end of text could still become a tag; len(text) if none."""
        i = text.rfind("<")
        if i < 0 or ">" in text[i:]:
            return len(text)
        m = re.match(r"/?([A-Za-z]*)(.*)$", text[i + 1:], re.DOTALL)
        name, rest = m.group(1).lower(), m.group(2)
        if not rest:
            could = any(t.startswith(name) for t in self._tags)
        else:
            could = name in self._tags and rest[0] in " \t\r\n/"
        return i if could else len(text)

    def _release_text(self, text: str, out: List[Event]) -> None:
        if not text:
            return
        if self._open is None:
            for m in self.parser.close_re.finditer(text):
                self.stray_closes.append(self.parser.canonical_tag(m.group(1)))
            text = self.parser.close_re.sub("", text)
        if text:
            out.append(Event("text", text=text))
            self._prev = text[-1]

    def _emit_block(self, tag: str, content: str, attrs: str, raw: str, out: List[Event]) -> None:
        block = self.parser._create_block(tag, content, self.parser._parse_attributes(attrs), raw)
        self.blocks.append(block)
        out.append(Event("block", block=block))
