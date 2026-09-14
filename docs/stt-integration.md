# STT project layout

The voice-agent runner lives at the repository root. The speech-to-text benchmark
lives in `stt-bench/`, with its own Python dependencies, configuration, tests, and
results. Run each project's commands from its respective directory.

From the repository root:

```sh
cd stt-bench
uv sync --locked
uv run --locked stt-bench models
```

See the [STT README](../stt-bench/README.md) for dataset preparation, provider
credentials, measurement definitions, and benchmark commands. The
[script guide](../stt-bench/scripts/README.md) covers remote execution and reporting.

Keep the STT `.env`, `.venv`, prepared audio, run checkpoints, and reports inside
`stt-bench/`. These local files are ignored by Git. Some integration tests require
FLEURS audio; dashboard generation requires saved reports.

The project originated in [cekura-ai/stt-bench](https://github.com/cekura-ai/stt-bench)
at commit `821fa89a77af6ddfa792fcb04520c598be6b1865`. The repositories do not
automatically synchronize changes.
