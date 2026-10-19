"""Create (or find) the ElevenLabs agent the ``s2s:elevenlabs-agents`` model talks to.

The agent is the configuration in turn-taking-bench/elevenlabs-agent.local.json
(or ELEVENLABS_AGENT_CONFIG) with turn bench's prompt. Creating it is free; conversations are billed. An
agent with the same name is reused, so running this twice creates nothing.
Prints the agent id; put it in ``turn-taking-bench/.env`` as ELEVENLABS_AGENT_ID.

    uv run python scripts/elevenlabs_agent.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

from turn_taking_bench.s2s import SYSTEM_PROMPT
from turn_taking_bench.s2s._vendor_env import setting
from turn_taking_bench.s2s.elevenlabs_agents import AGENT_NAME, API, CREDENTIAL_ENV, agent_payload, load_agent_config


def call(method: str, path: str, key: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{API}{path}", method=method, headers={"xi-api-key": key, "Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"{method} {path}: HTTP {exc.code}: {exc.read().decode(errors='replace')[:2000]}") from exc


def main() -> int:
    key = setting(CREDENTIAL_ENV)
    if not key:
        print(f"{CREDENTIAL_ENV} is not set", file=sys.stderr)
        return 1
    found = call("GET", f"/v1/convai/agents?{urllib.parse.urlencode({'search': AGENT_NAME})}", key)
    for agent in found.get("agents", []):
        if agent.get("name") == AGENT_NAME:
            print(agent["agent_id"])
            return 0
    print(call("POST", "/v1/convai/agents/create", key, agent_payload(SYSTEM_PROMPT, load_agent_config()))["agent_id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
