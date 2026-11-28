#!/usr/bin/env python3
"""Fetches the agent under attack, tau2-bench's airline agent by Sierra, into redteam/tau2_airline/: from
sierra-research/tau2-bench at the commit the published results ran on, under its MIT license.

    python scripts/fetch_tau2_airline.py

- policy.md, db.json and LICENSE are copied unchanged.
- system-prompt.txt is the policy inside tau2's agent instructions, as its LLM agent builds it (SYSTEM_PROMPT and
  AGENT_INSTRUCTION in src/tau2/agent/llm_agent.py, read from the source, not run).
- first-message.txt is tau2's default first message from the agent (src/tau2/orchestrator/orchestrator.py).
- tool-definitions.json is the schema tau2's toolkit produces for each of its 14 tools. This step imports tau2's own
  code, so it needs tau2's light dependencies (requirements.txt, "fetch"), and runs in a separate process.

Every file is checked against the SHA-256 of the file the published runs used; a mismatch stops with an error."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f"
URL = f"https://codeload.github.com/sierra-research/tau2-bench/tar.gz/{COMMIT}"
OUT = Path(__file__).resolve().parents[1] / "redteam" / "tau2_airline"
SHA256 = {
    "policy.md": "10dc0525421521208be39cee235bba84a16e2bcba9899eb93d92cd81d2f62fc4",
    "db.json": "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b",
    "system-prompt.txt": "5815a1954ea05866e89e6cbf441ec558fcb41d00dc0e38b044dea8654bd3f990",
    "first-message.txt": "3347431a93ea418a6ceeb9489bbf7a110cb42f58b11f0655aa9a912e5c05ec72",
    "tool-definitions.json": "d866dbd3a400b4f9fd58eae98352f1640e4ce5fee5a8891e6a71ace6b3223340",
    "LICENSE": "e67c5aa0074dfcaefd3c3a1aedb94cb539234aecd15d5a972574e3200e6252fe",
}
# Run in its own process, on tau2's source: the toolkit's own schema for each tool, in the order tools.py defines them
SCHEMAS = """
import json, sys
sys.path.insert(0, sys.argv[1])
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
tools = AirlineTools(FlightDB.load(sys.argv[2])).get_tools().values()
print(json.dumps([t.openai_schema["function"] for t in tools], indent=1))
"""


def strings(path: Path) -> dict[str, str]:
    """A module's top-level string constants, "...".strip() included, read from its source without running it."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "strip"
                    and isinstance(value.func.value, ast.Constant)):
                out[node.targets[0].id] = value.func.value.value.strip()
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[node.targets[0].id] = value.value
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        print(f"downloading tau2-bench {COMMIT[:10]}")
        with urllib.request.urlopen(URL, timeout=300) as response:
            data = response.read()
        prefix = f"tau2-bench-{COMMIT}/"
        wanted = (prefix + "src/", prefix + "LICENSE", prefix + "data/tau2/domains/airline/")
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith(wanted) and (m.isfile() or m.isdir())]
            tar.extractall(tmp, members=members, filter="data")
        root = Path(tmp) / prefix
        airline = root / "data/tau2/domains/airline"

        OUT.mkdir(parents=True, exist_ok=True)
        files = {name: (airline / name).read_bytes() for name in ("policy.md", "db.json")}
        files["LICENSE"] = (root / "LICENSE").read_bytes()
        agent = strings(root / "src/tau2/agent/llm_agent.py")
        prompt = agent["SYSTEM_PROMPT"].format(domain_policy=files["policy.md"].decode(),
                                               agent_instruction=agent["AGENT_INSTRUCTION"])
        files["system-prompt.txt"] = (prompt + "\n").encode()
        greeting = re.search(r'role="assistant", content="([^"]+)"', (root / "src/tau2/orchestrator/orchestrator.py").read_text())
        files["first-message.txt"] = (greeting.group(1) + "\n").encode()
        made = subprocess.run([sys.executable, "-c", SCHEMAS, str(root / "src"), str(airline / "db.json")],
                              capture_output=True, text=True)
        if made.returncode:
            raise SystemExit("tau2's tool schemas could not be built (install the fetch dependencies in "
                             f"requirements.txt):\n{made.stderr[-2000:]}")
        files["tool-definitions.json"] = (made.stdout.rstrip("\n") + "\n").encode()

        wrong = [name for name, body in files.items() if hashlib.sha256(body).hexdigest() != SHA256[name]]
        for name, body in files.items():
            (OUT / name).write_bytes(body)
        if wrong:
            raise SystemExit(f"written, but not the files the published runs used: {', '.join(wrong)}")
        print(f"wrote {', '.join(sorted(files))} to {OUT}; all match the published runs")


if __name__ == "__main__":
    main()
