# Real-caller S2S benchmark

Speech-to-speech models on phone calls where every caller line is a recording of a
real person: banking and healthcare scenarios, each model run three times on the same
agent build. Calls are run and judged on the Cekura platform; this project turns the
exported runs into the website's data files.

| Command | What it does |
| --- | --- |
| `s2s-human-bench site-export` | Scores every call, rates the models and writes the website's data files: leaderboard, insights, reports, per-language view, per-skill view and reviewed example calls |

## Scoring

All settings below live in [`config/human-bench.json`](config/human-bench.json); the code
holds none of them.

* **Call score (0-100)**: the weighted mean of four skill families (Hearing, Turn Taking,
  Task Handling, Speaking; 25% each) over the families the call reached. A family's value
  is the share of its reached skills that passed. Task Handling is half skills, half the
  call's task outcome: the share of the scenario's expected-outcome lines met (lines not
  reached are left out) or, with a task file (`--tasks`), 100 when the agent completed the
  scenario's main task and 0 when it did not.
* **Main task (`--tasks`)**: decided from the call's tool calls. The private task file
  lists, per scenario, the tool calls the task needs and the values each must carry. It can
  also list tools that must not run (the caller took the action back first) and the undo
  expected after an action. Values are compared after normalizing formats (phone numbers,
  dates, times, amounts, ids, street abbreviations). The format is described in
  [`tasks.py`](src/s2s_human_bench/tasks.py).
* **Pass**: call score of at least 80. **pass³**: the scenario passed in all three runs.
* **ELO**: per scenario, a model's call score averaged over its runs; two models draw when
  their scores are within 5 points, otherwise the higher wins. Ranked models get
  Bradley-Terry ratings (1000 = average) with a 95% interval from a seeded bootstrap
  over scenarios. A reference system (the cascade) is rated afterwards against the ranked
  models with their ratings held fixed, so it moves no one else. A model's `scenarioSwap`
  takes the listed scenarios from another agent's runs, run by run; the cascade uses it to
  answer English callers with English speech-to-text and language switchers with multilingual.
* **Runs**: a model is published only with three usable runs. A run is usable when it was
  fully exported and the agent spoke on all but at most 7 calls; a run where it stayed
  silent on more is a throttled or broken run, not model behaviour.
* **Languages**: a scenario switches language when the Language Switch skill applies to it.
  The config names each switching scenario's language, and the export stops if those lists
  disagree with where the skill applied.

## Running

```sh
cd s2s-human-bench
git -C <website checkout> fetch origin
uv run --locked s2s-human-bench site-export \
  --runs <exported runs dir> \
  --public-s2s <website checkout> \
  --out <output dir>
```

* `--runs` holds one folder per run (`<run>/calls/<scenario>/<call id>/` with `call.json`,
  `metrics.json`, `transcript.json`, `expected_outcome.json`), named as the config's models
  list says, and optionally `scenario_clips.json` (`--clips`) for the recorded-clip count
  and `scenario_titles.json` (`--titles`) for the scenario titles shown with example calls.
* `--public-s2s` is the public S2S benchmark's `s2s-benchmark.json`, or a website checkout
  to read it from at `origin/main`; it supplies model metadata and published prices.
* `--name KIND=FILE` renames an output (defaults in the config's `outputs`), `--date` pins
  `generatedAt`, `--exclude-unhealthy` leaves out calls whose simulation-health check
  failed, `--clip-picks` also writes the example moments' clip windows, and
  `--reviewed-examples` lists the example calls (`{site id: [call id, ...]}`) a person has
  listened to; without it no example calls are published.
* `--tasks` scores Task Outcome as pass or fail on each scenario's main task (see Scoring).
  The file names each scenario's tools and the values the caller gives, so it is test-set
  grading detail and stays out of this repository; without the flag nothing changes.

Output depends only on the inputs, the config, `--date` and the task file. Run data,
transcripts, the task file and clip picks stay out of this repository.

```sh
uv run --locked pytest
```
