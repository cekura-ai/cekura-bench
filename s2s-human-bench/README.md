# Real-caller S2S benchmark

Speech-to-speech models on phone calls where every caller line is a recording of a
real person: banking and healthcare scenarios, each model run three times on the same
agent build. Calls are run and judged on the Cekura platform; this project turns the
exported runs into the website's data files.

| Command | What it does |
| --- | --- |
| `s2s-human-bench site-export` | Scores every call, rates the models and writes the website's data files: leaderboard, insights, reports, per-language view, per-skill view and reviewed example calls |
| `s2s-human-bench task-metric` | Writes the Cekura custom-code metric that checks the main task on the platform, built from the scorer's own checker |

## Scoring

All settings below live in [`config/human-bench.json`](config/human-bench.json); the code
holds none of them.

* **Call score (0-100)**: what someone running the model in production would check, in four
  parts over the parts the call reached: **Got It Done** (35%), the call's task outcome, 100
  when the agent completed the scenario's main task and 0 when it did not; **Right and Safe**
  (25%), the share of its judged skills passed (names and numbers, asking when unsure,
  corrections, cancellations, consent, handoff, honesty about failures), zeroed on a call
  where the agent claimed an action worked when it did not; **Call Flow** (25%), half the share
  of its judged skills passed, half measured from the call's timing; and **Understood and
  Clear** (15%), the share of its judged skills passed. Runs without a main-task verdict
  (neither a task file nor the platform metric) use the share of the scenario's
  expected-outcome lines met, leaving out lines not reached.
* **Measured call flow** ([`flow.py`](src/s2s_human_bench/flow.py)), from the transcript's word
  timings only: the wait before each reply (3 s counts half a problem, 5 s one, 8 s two), the
  agent talking over a caller who is still mid-turn, the agent jumping into a pause the caller
  then fills, and how long the agent keeps talking after the caller cuts in (2 s counts half,
  3 s one). Short acknowledgements ("mhm", "okay sure") on either side are not turns. The
  measured half of Call Flow is the share of turns with no problem.
* **Main task**: decided from the call's tool calls. Each scenario's task lists the tool
  calls it needs and the values each must carry. It can also list tools that must not run
  (the caller took the action back first) and the undo expected after an action. Values are
  compared after normalizing formats (phone numbers, dates, times, amounts, ids, street
  abbreviations); the format is described in [`tasks.py`](src/s2s_human_bench/tasks.py).
  The tasks are test-set grading detail and stay out of this repository: offline they come
  from a private file (`--tasks`); on the platform, each scenario's test profile carries its
  task (`eval_target_task`) and the metric in [`metrics/`](metrics/task_completion_metric.py),
  built from the same checker by `task-metric`, scores a run when its metrics are rescored
  after the calls, like the other skill metrics; the platform does not score it on its own.
  The scorer reads that metric's verdict (`metrics.task` in the config) unless a task file
  is given. Calls without a verdict from it fall back to the expected-outcome share, and
  `site-export` warns about them.
* **Pass**: call score of at least 80. **pass³**: the scenario passed in all three runs.
* **ELO**: per scenario, a model's call score averaged over its runs; each pair's result on a
  scenario follows the size of the gap, `1 / (1 + 10^(-gap / 12))`, so a near-tie counts
  close to a draw and a wide gap close to a win. Ranked models get Bradley-Terry ratings
  (1000 = average) with a 95% interval and a range of places from a seeded bootstrap over
  scenarios. A model's **place** is 1 plus the number of models whose call score is
  separably above it (the comparison below), so models the runs cannot tell apart share a
  place. A reference system (the cascade) is rated afterwards against the ranked
  models with their ratings held fixed, so it moves no one else. A model's `scenarioSwap`
  takes the listed scenarios from another agent's runs, run by run; the cascade uses it to
  answer English callers with English speech-to-text and language switchers with multilingual.
* **Comparisons**: for every pair of models, per scenario the call score averaged over runs, one
  minus the other, then the mean over scenarios with a 95% t-interval; *separable* when the
  interval excludes zero. The same for each family, the task outcome and reply time. A skill
  change is listed only when every run of one model beats every run of the other and the pooled
  pass rates differ with a Benjamini-Hochberg q under `comparisons.skillQ`; skills that move in
  every run but within chance are only counted (`withinChance`).
* **Board** (`board`): per model, the call score, task done and each part with 95% ranges (the
  scenario is the unit), the measured flow per 10 calls, and the skills and flow measures
  where it is clearly better or worse than the median of the other ranked models. Per call
  type ("hard moment"), over every call of the scenarios built for it (where at least a
  quarter of all calls reached the moment): how often it came up, how often it was handled
  when it did, and how often the task got done with the moment handled. Per use case (groups
  of scenarios in the config), task done and call score.
* **Scenarios** (`scenarios`): every scenario with each model's runs: task done, call score
  and the skills missed.
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
