# turn-bench

Measures how well speech-to-speech models handle the moments that decide whether a phone call feels natural:

- **Waits through pauses**: the caller pauses mid-thought and the model stays quiet (pause handling).
- **Replies when the caller is done**: the model answers real turn endings (turn taking).
- **Stops when the caller cuts in**: a barge-in while the model is speaking (barge-in yield, stop latency).
- **Keeps going through "mm-hmm"**: backchannels such as "mm-hmm", "okay", "got it" (backchannel continue rate).

End of turn (the first two) is scored with [Sesame TurnBench](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev)'s published scorer, unchanged. Interruptions (the last two) use our own method and are reported separately; the two are never blended into one rank.

## Data

Conversations are authored, not recorded: each template in `datasets/authored/specs/` scripts a customer-service call with exact pauses, turn endings, cut-ins and backchannels, voiced with ElevenLabs and assembled so every event sits at a known time. Each conversation is built for several voice pairings and played under six caller audio conditions (clean, telephone, noise at 20 dB and 10 dB SNR, reverb, packet dropouts).

Templates are split per template, stratified by family (see `datasets/authored/TAXONOMY.md`):

- **Public** (this repository, 15 templates): anyone can build and run them. The rendered public set is released as a CC BY 4.0 dataset.
- **Private** (25 leaderboard templates and 10 holdout templates): kept outside this repository so results cannot be tuned to them. Their file checksums are published with the public dataset.

To run against the private templates, point `--specs` at the private specs directory (its own `splits.json` lists the test ids). The template-specific label tests run when `TURN_BENCH_PRIVATE_SPECS` names that directory, and skip otherwise.

## Setup

```bash
cd turn-bench
uv sync --locked --extra s2s --extra smart-turn
uv run --no-sync turn-bench setup smart_turn_v3     # pinned model assets into .assets/ (or TURN_BENCH_ASSETS_DIR)
```

Provider keys are read from a dotenv file (`--env`); values never enter reports.

## Commands

| Command | What it does |
|---|---|
| `turn-bench author build` | Render specs with TTS and write a local dataset directory |
| `turn-bench author check` | Planted-vs-derived gold table and the oracle round trip through Sesame's scorer |
| `turn-bench author degrade` / `split` | Audio-condition variants of a dataset; public / test / holdout splits |
| `turn-bench run` | Stream a split through a model (reference detector or `s2s:<provider>`), score end of turn |
| `turn-bench s2s t2-run` | Live cut-in and backchannel sessions for one model; `--condition`, `--turn-detection` |
| `turn-bench s2s t2-rescore` / `t2-combine` | Re-score saved interruption runs offline; pool caller voices |
| `turn-bench s2s scorecard` | One model's end-of-turn and interruption results side by side |
| `turn-bench report` / `compare` | Rescore a finished run with confidence intervals; compare two runs |
| `turn-bench models` | List registered adapters |

Reference systems (`adapters/`): a classic cascade's end-of-turn decision (Deepgram Flux), Smart Turn v3 and an energy VAD.
Speech-to-speech adapters (`s2s/`): OpenAI Realtime, GPT-Live, Azure Realtime, Gemini Live (incl. extended thinking), Grok, Amazon Nova Sonic, Phonic and Deepslate, each run at its provider's default turn settings unless a flag says otherwise.

## Tests

```bash
uv run --no-sync python -m pytest -q
```

Tests that need local model assets or cached renders skip when those are absent.

## Licences

Code: as the repository. Sesame TurnBench data is under its own non-commercial licence and is used only to calibrate timing; it is not redistributed here.
