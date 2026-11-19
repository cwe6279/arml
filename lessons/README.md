# Lessons and regression cases

[lessons.json](lessons.json) is a running list of what has gone wrong with models, servers,
harnesses and ARML itself, and what ARML does about each. Every lesson carries test cases, and
[tests/test_lessons.py](../tests/test_lessons.py) runs them on every test run. A failure that was
fixed once can't quietly come back.

## How ARML treats its rules

- **A rule starts from a failure.** Every lesson is something that actually happened: a model
  read its private note aloud, invented a tool result, or wrote no blocks. We don't add rules for
  problems nobody has had.
- **Code enforces a rule; a prompt only asks.** A rule that lives only in the prompt is a request
  the model may ignore. Lint codes, parser behaviour and tests are what hold, so each lesson says
  what enforces it, and its cases check that the enforcement works.
- **Every prompt rule earns its place.** The tests fail if a rule in `format_rules()` has no
  lesson behind it, unless it's listed under `prompt_rules_without_a_lesson`. Those are on
  probation: a lesson or a measurement should come to back them, or they go. This matters more
  for small models than large ones, because the prompt is what a tiny model reads on every step.
- **The list and the code must agree.** When they drift apart, the tests fail. This has already
  happened: the list said lint caught "Noted." with no action behind it. It didn't, and the first
  run of these tests found it.
- **Prune as well as add.** When a lesson stops applying, because the cause was removed or a
  better rule covers it, delete it and its rule together.
- **Fewer, sharper tools.** The same evidence-first view applies to toolsets. A small model
  choosing among a few clear tools does better than one choosing among many overlapping ones. The
  benchmark is the place to measure this for a given model.

## A lesson

```json
{
 "id": "L13",
 "added": "2026-11-18",
 "source": "where it was seen",
 "kind": "model",
 "what": "what happened, in a sentence or two",
 "how_often": "13 of 29 blocks",
 "enforced_by": {"lint": ["OUTPUT_WITH_TOOL"], "prompt": ["No <output> in the same response as a <tool>"],
                 "harness": "what a harness must do", "training": "what training data must do"},
 "cases": [{"response": "<tool …>…</tool><output>…</output>", "expect": ["OUTPUT_WITH_TOOL"]}]
}
```

- **`kind`** is one of:
  - `model`: what a model wrote
  - `server`: a chat template or server quirk
  - `harness`: the program around the model
  - `lint`: ARML's own checks
  - `spec`: the format
  - `data`: making training data
- **`enforced_by.prompt`** holds short phrases from `format_rules()`. The tests check that the
  prompt still contains them.
- **Each entry in `cases`** is one of:
  - **a response** with the lint codes it must produce (`expect`) and must not produce
    (`expect_not`). Options: `"spoken": true` and `"acted": true`.
  - **a streamed reply** (`stream`: a list of chunks) and the only text allowed to reach the
    user (`visible`).
- **A lesson that's still open** gets `"status": "open"`. Its prompt rules and any case marked
  `"open": true` run as expected failures. When the fix lands, the test reports that it now
  passes, and the open mark comes off.

## Adding one

1. **Find a real failure:** in your logs, the benchmark or a training run. Note where it came
   from and how often it happened.
2. **Write the smallest response that shows it** as a case, and run the tests. If lint already
   catches it, you've documented a rule that works. If not, the case fails and shows exactly what
   to fix.
3. **Fix it in code** (lint, the parser, or the harness), not only in the prompt.
4. **If you change a prompt rule,** update the lesson that relies on it, or the tests will tell
   you.

Lessons that point to a benchmark or data-generator kind (`enforced_by.bench`) are the start of
regression benchmarks. The plan is for the benchmark to draw tasks from this list, so each
lesson is measured on real models as well as checked by lint.
