# ARML 2.1 — Agent Response Markup Language

ARML is a small markup that a model writes and a harness parses: reasoning,
tool calls and the answer to the user, each in its own block. Use it when you
build your own agent harness, and log your turns in the shared format below, so
that everyone's real-world lessons can be pooled to train smaller local models
to be better at agent tasks and conversation.

It is meant to be parsed by regular expressions, the same way every time, and to
be written the same way every time. The second half matters as much as the
first: a model learns exactly what it is shown, forgiven mistakes included.

Plain Python, no dependencies, Python 3.9+:

```bash
pip install git+https://github.com/cwe6279/arml.git
```

or copy the `arml/` folder into your project. Tests: `pytest tests/`.

## Build a harness with ARML

**1. Teach the model the format.** `format_rules()` gives the ARML section of
a system prompt; your own instructions go around it.

```python
from arml import format_rules

TOOLS = [{"name": "weather", "description": "Current weather for a city.",
          "parameters": {"city": {"type": "string", "required": True}}}]
system = "You are a helpful assistant.\n\n" + format_rules(TOOLS)   # spoken=True for voice
```

**2. Parse each response and act on it.**

```python
from arml import parse

r = parse(response)
if r.is_halted():
    stop(r.halt().get("reason"))
elif r.tools():
    tool = r.tools()[0]
    result = run_tool(tool.name, tool.params())   # feed the result back as the next message
else:
    show(r.visible_text())                        # never shows <think> or broken blocks
```

**3. If you show or speak text as it streams, use the streaming parser.** It
holds back a block until it closes, so reasoning never leaks (section 3).

```python
from arml import StreamParser

sp = StreamParser()
for chunk in model_stream:
    for ev in sp.feed(chunk):
        show(ev.text) if ev.kind == "text" else handle(ev.block)
for ev in sp.close():
    ...
```

**4. Log every turn in the shared format, with what happened next.**

```python
from arml.record import TurnRecord, append

rec = TurnRecord.from_response(response, model=MODEL, harness="myapp/1.0",
                               session=sid, turn=n, context=messages, system=system)
rec.set_outcome(tool_ok=True, user="accepted")   # or corrected / repeated / interrupted / ignored
append("arml-turns.jsonl", rec)
```

Each line holds the response, its lint codes, its canonical form, the context
the model saw, and the outcome. Text is redacted (emails, phone numbers, keys,
card-like numbers) before it is written; the system prompt is kept as a hash
unless you pass `keep_system=True`.

**5. Turn logs into training data.** `training_example(rec)` returns a
chat-format example (system, context, assistant = canonical ARML) only for good
turns: well-formed, with a tool that worked and a user who did not have to
correct, repeat or interrupt. Keep the rest. Their lint codes and outcomes are
the lessons: which mistakes models make, and the negative examples.

```python
from arml.record import read, training_example

examples = [ex for rec in read("arml-turns.jsonl") if (ex := training_example(rec, system=SYSTEM))]
```

The outcome is what turns logs into lessons. Lint can only say a response was
well-formed, not that it was right. A clean reply that the user cut off because
it read its plan aloud is exactly what a small model needs to learn from.

## 1. Grammar

Known tags (`TAG_TO_TYPE` in `arml/parser.py`): `think`, `tool`, `output`
(shorthand `o`), `halt`, `code`, `validate`, `guardrail`. `TAGS` below is
those names joined with `|`.

| What | Regex (case-insensitive) |
|---|---|
| Block | `` (?<!`)<(TAGS)(\s[^>]*)?>(.+?)</\1\s*> `` (dot matches newline) |
| Self-closing block | `` (?<!`)<(TAGS)(\s[^>]*)?/> `` |
| Opening tag | `` (?<!`)<(TAGS)(\s[^>]*?)?(/?)> `` |
| Closing tag | `` (?<!`)</(TAGS)\s*> `` |
| Attribute | `(\w+)(?:=(?:"([^"]*)"\|'([^']*)'\|(\S+)))?` |

- A block closes at the first closing tag with its own name. Tags inside a
  block are content: `<output>` may hold a `<code>` example.
- A tag quoted in backticks (`` `<code>` ``) is prose, not a tag.
- Shorthand is rewritten before matching: `` (?<!`)<(/?)(o)(?=[\s/>]) `` →
  `<\1output`. So `<output>…</o>` is one block, where before 2.1 it matched
  nothing and the answer leaked out with its tags. Tags someone registers keep
  their own names.
- Tool content is JSON. The parser repairs common slips (`ToolBlock.params()`),
  but lint reports every repair (section 4).

## 2. Visible and silent

Each block type is visible or silent (`VISIBLE_TYPES`):

| Visible: the user sees or hears it | Silent: the agent talking to the runtime |
|---|---|
| `output`, and plain text outside blocks | `think`, `tool`, `code`, `validate`, `halt`, `guardrail`, anything registered |

`ParseResult.visible_text()` returns plain text and visible block content, in
order. It never returns:

- the content of a silent block;
- anything after an opening tag that is never closed (reported in `dangling`);
- a closing tag with no opening (dropped, reported in `stray_closes`).

`ParseResult.text()` is unchanged: all text outside complete blocks.

## 3. Streaming

`arml.StreamParser` reads a response while the model is still writing it, for
chat UIs and voice. Its rules, each from a real failure:

1. **Nothing is released from an opening tag until its closing tag.** A voice
   assistant split replies at full stops, then stripped blocks from each
   sentence. A note block holding whole sentences was cut at its own full stops,
   the halves no longer looked like blocks, and the assistant read its private
   note aloud: 73 words spoken for a 40-word reply. Blocks with no full stop
   inside were fine, so the bug showed up only now and then.
2. **Visible content streams.** `<output>` text is released as it arrives,
   minus a possible partial closing tag at the end of the buffer.
3. **A `<` that could still become a tag is held** until it is decided:
   `<thi` waits, `< 4` and `<b>` go out.
4. **A block never closed is dropped**, not released, and reported in
   `dangling`. Visible text already released stays released.

For any response and any chunk sizes, streaming gives the same blocks, the same
visible text, and the same `dangling`/`stray_closes` as `parse()`. The tests
check this with random chunkings.

Ordering advice for agents whose output is spoken: put the action block
(`<tool>`) before a long visible reply, and keep the visible part short ("On
it."). The client can start the work while the words play, instead of after
them.

## 4. Lint

`arml.lint(text)` → a list of `Issue(code, level, message)`; `is_clean(text)` is
true when there are no errors. Deterministic, no model call, never repairs.

| Level | Code | Meaning |
|---|---|---|
| error | `EMPTY` | no text |
| error | `UNCLOSED` | a block opened, never closed |
| error | `STRAY_CLOSE` | a closing tag with no opening |
| error | `TOOL_NO_NAME` | `<tool>` without `name` |
| error | `TOOL_BAD_JSON` | tool content is not JSON even after repair |
| error | `NO_ACTION` | blocks, but no tool, output, code, validate or halt |
| error | `NO_BLOCKS` | plain text only (accepted as an answer at run time; a model trained on it learns to skip the format) |
| warn | `TOOL_JSON_REPAIRED` | tool JSON parses only after repair |
| warn | `MIXED_ALIAS` | `<output>…</o>` |
| warn | `STRAY_TEXT` | plain text outside blocks, next to blocks |
| warn | `MANY_OUTPUTS` | more than one `<output>` |
| warn | `OUTPUT_WITH_TOOL` | `<output>` next to `<tool>`; the tool runs first and the output is not used this turn |
| info | `SHORTHAND` | shorthand tags used |

Behavior checks, from the field (section 7). They are regexes, so treat a hit as
"look at this turn", not proof:

| Level | Code | Meaning |
|---|---|---|
| error | `FAKE_RESULT` | visible text writes what only the runtime writes: `(Event: …`, `step 1 done`, `[tool result]`. Add your harness's markers with `lint(text, reserved=[...])` |
| warn | `ANNOUNCED_NO_ACTION` | "Let me check…" / "I'll send…" with no `<tool>`, `<code>` or `<validate>` |
| warn | `CLAIMED_NO_ACTION` | "I've sent / saved / booked…" with no block that did it |
| warn | `LONG_SPOKEN` | `spoken=True`: more visible words than `word_budget` (30) |
| warn | `SPOKEN_NUMBER` | `spoken=True`: a phone number or long code read aloud |

A deliberately silent turn is `<output/>`: valid, and nothing is shown or said.

## 5. Canonical form

`arml.canonical(text)` writes a response that parses in one spelling:

- canonical tag names (`<o>` → `<output>`);
- attributes as `key="value"`, in the order written;
- block content trimmed; tool JSON compact (`{"a": 1, "b": 2}`), keys in the
  order written;
- one blank line between parts;
- remains of broken blocks are not kept.

It changes form, never meaning: tool JSON that needs repair is left as written.
`canonical(canonical(x)) == canonical(x)`.

## 6. Building a training set

```python
from arml import lint, is_clean, canonical

for response in logged_responses:
    if is_clean(response):
        keep(canonical(response))
    else:
        reject(response, [str(i) for i in lint(response)])   # counts per code show what to fix in prompts
```

Keep the rejected ones with their codes. They show which mistakes each model
makes, and they are the negative examples if you ever train a format checker.

## 7. Lessons from the field

From a voice assistant running on a small home device, which
has its own block syntax (`{{note …}}`, `{{task …}}`). It hands work to a coding
agent. Four days of logs: 20 sessions, 406 user turns, scanned on 2026-10-07.
Each lesson is now a rule in `format_rules()`, a lint check, or an outcome label.

| What happened | How often | Now in ARML |
|---|---|---|
| Blocks with a full stop inside were split mid-block, read aloud, and never run; the reply still said "On it." | 13 of 29 note/task blocks | Streaming rule 1; `UNCLOSED`; a dropped block must fail visibly in the harness |
| Invented a tool result ("step 1 done: open until 10 p.m.") before the real one came back, which said it could not confirm | 1, and the user acted on it | `FAKE_RESULT`; prompt: only the runtime writes results |
| "Let me check your calendar" with no calendar tool (checked the clock instead) | 7 | `ANNOUNCED_NO_ACTION`; prompt: nothing runs between responses |
| "Noted." / "in motion" with no block behind it | 9 | `CLAIMED_NO_ACTION` |
| Gave the time from earlier context, wrong by hours | 3 | prompt: changing facts come from a tool this turn |
| Spoken replies over 40 words or 15 s, phone numbers read aloud | 38 long, 13 with numbers | `LONG_SPOKEN`, `SPOKEN_NUMBER`; spoken prompt rules |
| Answered someone else's video call: 126 turns, 31 % of all turns | 1 session | spoken prompt: answer `<output/>` when the user talks to someone else; outcome `ignored` |
| Used the camera on filler speech, acted on garbled transcripts; the user said "what?" or "not what I asked" | about 25 repairs | spoken prompt: ask one short question; outcome `corrected` / `repeated` |
| A task left out the place the assistant already knew; an old task kept running after the dates changed | 2 | prompt: put known context in the arguments; cancel or replace old work |
| Sub-agent tasks hit 900 s limits, and goals were lost on restart | 8 failed, 6 lost | harness: persist goals, report failures; outcome `tool_ok=False` |

The harness lessons in the last column apply to any harness built on ARML:
- Never speak or show a block that failed to parse; say the action failed.
- Keep work that outlives a turn somewhere that survives a restart.
- Record the outcome of every turn. Most of these failures were only visible
  through the user's reaction.
