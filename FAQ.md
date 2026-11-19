# ARML FAQ

## What is ARML?

A plain-text format for what a model writes when it acts as an agent. Its reasoning, its tool
calls and its answer each go in their own block:

```
<think>They want the total with tax. I need exact math.</think>
<tool name="calculate">{"expression": "84.50 * 1.08"}</tool>
```

Your harness parses the blocks with regular expressions and acts on them. The repository has
the spec, a batch and a streaming parser, a lint, the prompt rules that teach a model the
format, and a shared log format. Plain Python, no dependencies.

## What problem does it solve?

Small models are often fine at deciding which tool to use, and bad at writing the call
reliably. In our first benchmark, qwen3:8b got most tasks right while about a third of its
replies broke the format. Native tool calling fails in its own ways, and differently on each
model and server:

- calls that come back as text the server never parsed;
- Gemma 4 returning an empty reply after a tool result when thinking is off, on both Ollama and
  llama.cpp;
- Qwen slipping its own `<tool_call>` tags into its replies.

ARML gives a model one simple, strict format to learn, and parsing that works the same on every
server.

## Why does an open format matter?

So you don't have to depend on a major provider to build an agent.

Today, reliable tool calling mostly comes from large hosted models, and each model family
defines its own tool-call format. An agent built that way depends on that provider's pricing,
rate limits, policies and model retirements. It also sends your data to them on every step.

A small model running on your own hardware, trained on a format nobody owns, changes that:
- **You own it.** The model, the training data and the format are yours to keep, copy and change.
- **It keeps working.** Nothing is deprecated, repriced or switched off under you.
- **Your data stays local.** Steps run offline, on your machine.
- **You can switch models.** Any model trained on ARML fits the same harness, because the format
  doesn't belong to a model family.

Large hosted models still have a place. For example, a big model can plan a task and hand the
routine steps to a small local one. But the routine steps don't have to depend on anyone else.

## Why not just use my model's native tool calling?

If it works reliably for your model and server, keep using it. ARML is for when it doesn't, or
when you need the same behaviour across several models and servers:

- **Native calling depends on the chat template and the server's parser.** Change either and the
  behaviour changes.
- **ARML depends only on the text.** The same reply parses the same way in llama.cpp, Ollama,
  LM Studio, vLLM, or a hosted API.

On untuned models, the first benchmark found the two about even when thinking is on. ARML's
real claim is about models trained on it.

## Isn't parsing with regular expressions fragile?

Not once the model writes the format. The parse is deterministic, needs no library, and is
cheap enough to run on every streamed chunk. The streaming parser gives the same result as the
batch parser for any chunking; the tests check this with random chunk sizes. What can go wrong
is the model not following the format, and that is what lint measures and training fixes.

There is one known gap: a block ends at the first closing tag with its name, so content that
contains its own closing tag (a tool argument holding `</tool>`) cuts the block short. A fix is
planned for the next version of the spec.

## Does my model need fine-tuning to use ARML?

No. `format_rules()` writes the section of a system prompt that teaches the format, and most
capable models follow it. Small models follow it less reliably. That's what fine-tuning on ARML
is for.

## Which models and servers work?

Any server with the OpenAI-compatible chat API: llama.cpp, Ollama, LM Studio, vLLM. The model
only has to write text, so it needs no tool-calling support of its own. The first benchmark
ran qwen3:8b, gemma4 8B and gemma4 E4B, on Ollama and on llama.cpp.
[examples/minimal_harness.py](examples/minimal_harness.py) is a complete harness in about 100
lines.

## Can I train a tiny model on my own tools?

That's where ARML is headed, but it isn't packaged yet. The plan: you describe your 10 to 20
tools in one file. A local model writes training examples, including the cases where the right
move is not to call a tool. You fine-tune a small model and get a file you can run in Ollama.

What exists today:
- [datagen/](datagen/) generates checked training data for a built-in set of example tools.
- [bench/](bench/) measures a model writing ARML against the same model using native tool calls.

## How much training data does it take?

Rules of thumb for LoRA fine-tuning, not measurements:
- about 500 to 2,000 examples to write the format reliably;
- about 5,000 to 20,000 for good decisions across many tools.

Give the cases where no tool should be called about as many examples as the cases where one
should. Include near misses: requests that sound like a tool request but aren't.

## What about reasoning models?

ARML's `<think>` block is where the model reasons, and the user never sees it. Many local
reasoning models already write `<think>…</think>` in their text, which ARML accepts.

Some servers move the reasoning into a separate field. That's fine for the parse, but the
reasoning is then missing from the text, so it doesn't reach your logs.

## What if my tools change during a conversation?

Don't rewrite the system prompt. A local server reuses its prompt cache only while the start of
the prompt is unchanged, so rewriting the tool list makes it reprocess the whole conversation.
Send `tools_notice()` as a runtime message instead. `format_rules()` already tells the model
that the notice overrides the tool list. See step 6 of
[Build a harness with ARML](README.md#build-a-harness-with-arml).

## Does it work for voice assistants?

Yes. `format_rules(spoken=True)` adds rules for replies that are read aloud. The streaming
parser holds back every block until it closes, so a private note is never spoken: that rule
came from a real voice assistant that read its notes aloud. Lint warns about spoken replies
that run long or read out phone numbers.

## Are my logs private? Should I share them?

Logs stay wherever your harness writes them; nothing is sent anywhere.

Turn records redact emails, phone numbers, card-like numbers and API keys before they're
written, but not names, addresses or anything else private. Read a log file before you share
it. The point of the shared log format is that logs from many people can be pooled into
training data for better small models.

## What's the license?

The code is MIT: use it for anything, commercial or not, as long as you keep the copyright
notice.

A model fine-tuned with ARML keeps its base model's license, whatever that is. Qwen models are
Apache 2.0, for example, and Gemma comes under Google's Gemma terms.

## Is ARML a standard? Who maintains it?

It's a spec at version 2.1, maintained by one person, and it changes as it meets real use.
Issues and pull requests are welcome, especially:
- failure cases from real models;
- logs from harnesses built on ARML.
