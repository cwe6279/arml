# Benchmark: ARML or native tool calls?

Local model servers (llama.cpp, Ollama, vLLM, LM Studio) parse tool calls themselves, using
each model's own chat template. That is the format the model was trained on. ARML asks the
model to write its tool calls as text instead. This benchmark runs the same model on the same
tasks both ways.

```bash
python bench/native_vs_arml.py --model qwen3:8b                           # the server's default thinking
python bench/native_vs_arml.py --model qwen3:8b --reasoning-effort none   # thinking off
python bench/native_vs_arml.py --model qwen3:8b --model gemma4:latest --runs 5 --url http://localhost:8080/v1
```

No dependencies. Any server with the OpenAI-compatible chat API works. Results go to
`bench/results/`: a summary (`.md`, `.json`), every conversation (`.transcripts.jsonl`) and the
ARML turns in the shared log format (`.turns.jsonl`). Each turn's outcome carries `task_ok`,
whether its task passed, so `training_example(rec, strict=True)` keeps only clean turns from
tasks that came out right. The `.jsonl` files are not committed.

## This is a baseline, not a verdict

A model off the shelf was trained on its own tool format and has never seen ARML. Comparing
the two on that model gives native tool calls a head start. ARML's claim is about a model
trained on it, so the fair test has three arms:

| Arm | Model | Format | Status |
|---|---|---|---|
| 1 | base model | native tool calls | measured here |
| 2 | base model | ARML | measured here |
| 3 | the same model, fine-tuned on ARML | ARML | needs a tuned model |

- **Arm 3 beats arm 1:** ARML has earned its place as a format models write.
- **Arm 3 only matches arm 1:** the case for ARML rests on pooled, comparable logs alone.
- **Arm 2 against arm 1,** which is all this benchmark can measure today, shows the cost of
  asking for ARML without training. It says nothing about which format is better.

What arms 1 and 2 are still good for:

- **The starting gap.** How far an untuned model is from writing ARML correctly is the
  distance fine-tuning has to close.
- **What to train on.** The failed ARML turns are logged with their lint codes
  (`.turns.jsonl`). They show which mistakes each model makes.
- **The "before" number** for arm 3.

A tuned model is one more `--model`, so arm 3 needs no new code. It does need tasks the model
never saw in training: hold these 26 out of the training set, or add new ones before scoring.

## The two modes

| | `arml` | `native` |
|---|---|---|
| How the model learns the tools | `format_rules(tools)` in the system prompt | the request's `tools` field |
| How it calls one | a `<tool name="…">{json}</tool>` block in its text | the server's `tool_calls` |
| How the result comes back | a user message, `Result of weather: …`, as in `examples/minimal_harness.py` | a `tool` message |
| Calls per response | the first `<tool>` only (ARML's rule) | every call in the response |

Both modes get the same identity line and the same behaviour rules ("nothing runs between
responses", "facts that change come from a tool"). ARML's prompt carries those rules next to
its format rules, so native mode is given them too. Otherwise the comparison would measure
the prompt, not the format. `--no-native-rules` leaves them out.

## The tasks

26 tasks on 7 fake tools with fixed data (weather, calculate, clock, calendar, contacts,
send_message, set_timer). Nothing real is sent, and every task has a known right answer.

| Group | Tasks | What it tests |
|---|---|---|
| One tool | 7 | the right tool with the right argument |
| Exact arguments | 3 | "90 seconds" into a `minutes` parameter, an expression kept intact |
| Several steps | 6 | chains where one result feeds the next call |
| No tool needed | 4 | answering without reaching for a tool |
| Missing or impossible | 4 | asking or declining instead of acting on a guess |
| Facts that change | 2 | the day and the calendar come from a tool, not from memory |

A task passes when every needed call was made with correct arguments, no call was made that
the request did not ask for, no tool was invented, an action such as `send_message` ran once
and not twice, and the final answer says the right thing.

## What is reported

- **Tasks done right**, in total and per group.
- **Format errors.** For ARML these are its own lint errors, such as `NO_BLOCKS` (a plain-text
  answer) or `UNCLOSED`. For native: `TOOL_BAD_JSON`, `CALL_AS_TEXT` (the call came back as
  text because the server's parser missed it) and `EMPTY`. `MAX_STEPS` in either mode means
  the model was still calling tools after 6 steps.
- **Invented tools**, steps, output tokens and seconds per task.
- **Task by task:** how many task runs only one mode got right, with a sign test. With 26
  tasks and 3 runs, a difference of a few tasks is noise. Read the p value before the totals.

## Reading the result

- A format error is not a failed task. A right answer in plain text is `NO_BLOCKS` in ARML:
  the user got the answer, but the turn is not one to train on.
- Thinking changes the picture. With thinking off, ARML's `<think>` block is the only place the
  model can reason, and native mode has none. Run both settings.
- One server's parser is not every server's. `CALL_AS_TEXT` depends on the server and the
  model's template as much as on the model.
