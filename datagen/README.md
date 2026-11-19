# Generating ARML training data

`datagen/generate.py` builds training data for fine-tuning a model to write ARML. A teacher
model answers generated requests in ARML. Every session is checked, and only clean turns from
sessions that came out right become training examples. Using a stronger model's checked
answers to train a smaller one is distillation.

```bash
python datagen/generate.py --show 5                                     # look at scenarios, run nothing
python datagen/generate.py --teacher oracle --n 2000                    # templated answers, no model
python datagen/generate.py --url http://localhost:8002/v1 --model qwen3.8-flash-next --n 500
```

Output goes to `datagen/out/`, which is not committed:

| File | What it holds |
|---|---|
| `turns.jsonl` | every teacher turn in the shared log format, passed or not, with its outcome |
| `train.jsonl` | chat-format training examples: the turns clean under `strict` from sessions that passed |
| `stats.json` | counts per kind of request, why sessions failed, lint codes |

A run picks up where the last one stopped: scenarios already in `turns.jsonl` are skipped, and
`train.jsonl` is rebuilt from the whole log. A different `--seed` gives different scenarios.

## How a scenario is built

Code builds every scenario, not a model, so each one has a known right answer:

- **A toolset** of 2 to 8 tools drawn from a catalog of 30 lookups (share prices, flights, parcels,
  orders, server health and so on) and 23 actions (thermostat, reminders, bookings, transfers,
  email, taxis).
  Each tool's name and parameter names vary (`next_bus(stop)`, `busArrival(stopName)`), so a
  model learns to read the tool list it is given rather than memorise one.
- **A request** of one of twelve kinds:

  | Kind | Example | Right behaviour |
  |---|---|---|
  | lookup | "Is flight LH 455 on time?" | the right tool, then an answer with the fact |
  | action | "Set the office to 21 degrees." | the action once, with the right arguments |
  | chain | "Check the build status of atlas, then post it to the releases channel." | a lookup whose result goes into the action |
  | compare | "Which garage has more free spaces, North or Market?" | two lookups, then the right name |
  | no tool | "What's the capital of Japan?" | an answer, no tool call |
  | missing | "Change the thermostat." | a question back, no call |
  | impossible | a request no tool in the list can do | say so, call nothing |
  | unknown | a flight number the tool has no data for | look it up, then say nothing was found instead of inventing |
  | removed | after an earlier lookup, the runtime's notice removes that tool; "And Oslo?" | don't call it from memory; say it's no longer available |
  | renamed | the notice renames the tool used earlier | use the new name, never the old one |
  | added | the notice adds a tool the request needs | use it, though the system prompt never listed it |
  | tool error | the tool says its service isn't responding | try at most once more, then say so without inventing a result |

  The last four start partway through a conversation: an earlier exchange is already in the
  context, and the system prompt still lists the tools as they were at the start. Tool
  changes arrive as a `tools_notice()` message, as the main README describes.

- **Distractors:** the toolset always holds tools the request doesn't need.
- **One of ten identity lines** opens the system prompt, followed by `format_rules(tools)`.

The tools are fakes with fixed data, so nothing real happens.

## How a session is checked

A session passes when every needed call was made with the right arguments, no call was made
that the request didn't ask for, no tool was invented, each action ran exactly as often as
asked, the answer says what it must and doesn't say what it mustn't, and the teacher stopped
within 6 steps. Every turn gets `task_ok` in its outcome, so a turn from a failed session
never becomes a training example.

## Held out from the benchmark

No generated tool shares a name with the benchmark's tools (`HELD_OUT_TOOLS`), and no request
repeats a benchmark task. A model trained on this data can therefore be scored on
`bench/native_vs_arml.py` (arm 3 in `bench/README.md`). The tests check the overlap.

## Choosing a teacher

- **A real model** gives natural wording and real reasoning in `<think>`. It is slow: the
  first trial with Qwen3.8 Flash Next on llama.cpp took about 25 seconds per scenario, so
  2,000 scenarios is about 14 hours. Its turns that leak its own native tool format (a
  `<tool_call>` tag in front of the ARML block, seen in that trial) are dropped by the strict
  filter.
- **The oracle** (`--teacher oracle`) needs no model and always passes. Its answers are
  templates, so it teaches the format and the moves, not natural language. Use it for a quick
  first set, or mix a little of it with a real teacher's sessions.

## How much data

Rules of thumb for LoRA fine-tuning, not measurements:

- about 500 to 2,000 examples to make a model write the format reliably;
- about 5,000 to 20,000 for good agent behaviour in it, with wide variety in tools and
  requests mattering more than the count.

In 2,000 scenarios that gives about 1,270 distinct requests and 310 distinct tool names. More tool concepts, request phrasings and identities widen what
the model sees, and they are plain data at the top of `generate.py`.
