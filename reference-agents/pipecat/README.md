# Pipecat reference agent

Canonical Pipecat implementation of Ava. Set `AGENT_DIR` to `appointments` or `insurance`; the worker loads that frozen definition and serves fixture-backed tool responses.

Install with `uv sync`, configure the required Pipecat Cloud, model-provider, and Cekura environment variables, then run the worker as specified by the selected deployment TOML. Do not commit `.env` files or deployed-call data.
