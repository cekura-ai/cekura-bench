# turn-taking-bench

Measures how well speech-to-speech models handle the moments that decide whether a phone call feels natural:

- **Waits through pauses**: the caller pauses mid-thought and the model stays quiet (pause handling).
- **Replies when the caller is done**: the model answers real turn endings (turn taking).
- **Stops when the caller cuts in**: a barge-in while the model is speaking (barge-in yield, stop latency).
- **Keeps going through "mm-hmm"**: backchannels such as "mm-hmm", "okay", "got it" (backchannel continue rate).

End of turn (the first two) is scored with [Sesame's TurnBench](https://github.com/SesameAILabs/turnbench) published scorer, unchanged. Interruptions (the last two) use our own method and are reported separately; the two are never blended into one rank.

## Data

Conversations are authored, not recorded: each template in `datasets/authored/specs/` scripts a customer-service call with exact pauses, turn endings, cut-ins and backchannels, voiced with ElevenLabs and assembled so every event sits at a known time. Each conversation is built for several voice pairings and played under six caller audio conditions (clean, telephone, noise at 20 dB and 10 dB SNR, reverb, packet dropouts).

Templates are split per template, stratified by family (see `datasets/authored/TAXONOMY.md`):

- **Public** (this repository, 15 templates): anyone can build and run them. The rendered public set is released as a CC BY 4.0 dataset.
- **Private** (25 leaderboard templates and 10 holdout templates): kept outside this repository so results cannot be tuned to them. Their file checksums are published with the public dataset.

To run against the private templates, point `--specs` at the private specs directory (its own `splits.json` lists the test ids). The template-specific label tests run when `TURN_TAKING_BENCH_PRIVATE_SPECS` names that directory, and skip otherwise.

## Setup

```bash
cd turn-taking-bench
uv sync --locked --extra s2s --extra smart-turn
uv run --no-sync turn-taking-bench setup smart_turn_v3     # pinned model assets into .assets/ (or TURN_TAKING_BENCH_ASSETS_DIR)
```

Provider keys are read from a dotenv file (`--env`); values never enter reports.

## Commands

| Command | What it does |
|---|---|
| `turn-taking-bench author build` | Render specs with TTS and write a local dataset directory |
| `turn-taking-bench author check` | Planted-vs-derived gold table and the oracle round trip through TurnBench's scorer |
| `turn-taking-bench author degrade` / `split` | Audio-condition variants of a dataset; public / test / holdout splits |
| `turn-taking-bench run` | Stream a split through a model (reference detector or `s2s:<provider>`), score end of turn |
| `turn-taking-bench s2s t2-run` | Live cut-in and backchannel sessions for one model; `--condition`, `--turn-detection` |
| `turn-taking-bench s2s t2-rescore` / `t2-combine` | Re-score saved interruption runs offline; pool caller voices |
| `turn-taking-bench s2s scorecard` | One model's end-of-turn and interruption results side by side |
| `turn-taking-bench report` / `compare` | Rescore a finished run with confidence intervals; compare two runs |
| `turn-taking-bench models` | List registered adapters |

Reference systems (`adapters/`): a classic cascade's end-of-turn decision (Deepgram Flux), Smart Turn v3 and an energy VAD.
Speech-to-speech adapters (`s2s/`): OpenAI Realtime, GPT-Live, Azure Realtime, Gemini Live (incl. extended thinking), Grok, Amazon Nova Sonic, Phonic and Deepslate, each run at its provider's default turn settings unless a flag says otherwise.

## Running the public set

The public set is on Hugging Face at `cekura/turn-taking-bench` (CC BY 4.0). The downloaded folder loads directly as a dataset, and `calls_by_test.json` says which calls feed which test:

```bash
# end of turn: the 24 pause / turn-ending / long-turn calls
turn-taking-bench run --dataset <downloaded folder> --model s2s:<provider> \
  --ids "$(jq -r '.end_of_turn.conversation_ids|join(",")' <downloaded folder>/calls_by_test.json)"
# cut-ins and "mm-hmm"s: once per caller voice (f-us, m-us, f-gb), with the published caller clips
turn-taking-bench s2s t2-run --specs datasets/authored/specs --cache-dir <downloaded folder>/interruption-clips/renders \
  --split dev --caller-voice f-us --provider <provider> --out runs/<model>-f-us
```

Scoring end of turn on all 45 calls mixes in the interruption scripts and will not match the published numbers.

## Reproducing results

- **Rescoring saved predictions is exact**: the end-of-turn scorer is pinned and confidence intervals use a fixed seed (`turn-taking-bench report`, `turn-taking-bench s2s t2-rescore`).
- **Re-running live models lands within the 95% intervals, not exactly**: models vary run to run, and cut-ins are timed to each model's live reply.
- **Location and machine matter**: reply and stop times include the network path; published results were run from US East on dedicated cloud machines. Caller audio is paced in real time, and a send more than 25 ms late drops the moments around it, so run on a machine that is not CPU-starved.
- **Models change**: providers can update a model behind the same name; results record the model id and run date.
- **The private set** (the leaderboard) cannot be re-run outside Cekura by design; compare against the public-set results.

## Tests

```bash
uv run --no-sync python -m pytest -q
```

Tests that need local model assets or cached renders skip when those are absent.

## Licences

Code: as the repository. TurnBench data is under its own non-commercial licence and is used only to calibrate timing; it is not redistributed here.

Not affiliated with or endorsed by Sesame or the TurnBench authors. Its scorer is used, and two of its baselines are ported, under its MIT licence (see `THIRD_PARTY_NOTICES.md`).
