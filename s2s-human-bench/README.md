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

* **Call score (0-100)**: the weighted mean of four skill families (Hearing, Turn Taking,
  Task Handling, Speaking; 25% each) over the families the call reached. A family's value
  is the share of its reached skills that passed. Task Handling is half skills, half the
  call's task outcome: 100 when the agent completed the scenario's main task and 0 when it
  did not. Runs without a main-task verdict (neither a task file nor the platform metric)
  use the share of the scenario's expected-outcome lines met, leaving out lines not reached.
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
* **Entity capture** (with `--entities`): the skill the entity file names (1.2) passes when the
  agent's first use of the detail the caller gave is right, whether that use is a tool call or
  a line it says; reading it back is not required. A line is read from the agent's own words
  when the run carries them (the reference agent writes them into the run record; older runs
  have them in the execution log), else from the transcript; a value blanked out of a log
  leaves that line to the transcript. Where a service only produces audio, its text is a
  transcription and counts as a second hearing: when it disagrees with the transcript, a later
  tool value decides, and without one the call is not reached. When only the transcript says
  the first use was wrong, the model's next tool value can settle it: letters often misheard
  (M/N, B/P/D) are a fail only if the tool carries the same letters, and any other wrong reading
  is a pass if the tool carries the right value and the caller did not give it again in between.
  See [`entities.py`](src/s2s_human_bench/entities.py). Without the flag the skill comes from
  its platform metric.
* **Pass**: call score of at least 80. **pass³**: the scenario passed in all three runs.
* **ELO**: per scenario, a model's call score averaged over its runs; two models draw when
  their scores are within 5 points, otherwise the higher wins. Ranked models get
  Bradley-Terry ratings (1000 = average) with a 95% interval from a seeded bootstrap
  over scenarios. A reference system (the cascade) is rated afterwards against the ranked
  models with their ratings held fixed, so it moves no one else. A model's `scenarioSwap`
  takes the listed scenarios from another agent's runs, run by run (a list takes them from
  several agents, each its own scenarios); the cascade uses it to answer English callers with
  English speech-to-text and language switchers with multilingual.
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

* `--entities` scores the skill it names from the call itself (see Scoring). Like the task
  file, it holds each scenario's true detail, so it stays out of this repository. The agent's
  own words come from `call_additional_metadata.json` next to `call.json` when the export has it.
* `--agent-logs` (with `--entities`) is a folder of execution logs exported from the platform,
  one `<call id>.json` per call, for runs recorded before the agent wrote its words into the run
  record; the agent's "agent said" lines are read instead.

Output depends only on the inputs, the config, `--date`, the task file, the entity file and the
agent logs. Run data, transcripts, logs, the task and entity files and clip picks stay out of this
repository.

```sh
uv run --locked pytest
```
