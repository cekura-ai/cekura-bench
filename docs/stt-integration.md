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

The benchmark code is synchronized from [cekura-ai/stt-bench](https://github.com/cekura-ai/stt-bench)
at commit `8e09cc7ea9cb6cbbe2ddaa5c6258af41cc8dfad1`, with documentation adapted for this repository
and experimental or incident-specific helpers excluded. The repositories do not
automatically synchronize changes.
