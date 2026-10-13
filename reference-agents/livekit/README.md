# LiveKit reference agent

Canonical LiveKit Agents implementation of Ava. `agent.py` selects the `appointments` or `insurance` definition per call, dynamically registers the frozen tools, and resolves them through `mock_backend.py`.

Install dependencies with `uv sync`, configure the required LiveKit, model-provider, and Cekura environment variables, then run `uv run python agent.py dev`. Do not commit `.env` files or deployed-call data.
