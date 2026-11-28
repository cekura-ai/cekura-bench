"""Keys, data paths, the spend ledger and the chat helper every part of the harness shares.

REDTEAM_ENV names the keys file (one KEY=value per line; values are never printed). RED_TEAM_DATA names the directory
for run data, outside the repository: runs/<round>.jsonl, logs/, ledger.jsonl and the platform agent ids. Run rows
hold attack text, so that directory stays private."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # this benchmark's folder
CONFIGS = ROOT / "agent-configurations"
ENV_FILE = Path(os.environ.get("REDTEAM_ENV") or Path.home() / ".redteam.env")
DATA = Path(os.environ.get("RED_TEAM_DATA") or Path.home() / "red-team-data")
RUNS, LOGS, LEDGER = DATA / "runs", DATA / "logs", DATA / "ledger.jsonl"

# OpenAI-compatible endpoints: (base url, key name in the keys file)
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_AUTHORIZATION"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
}

# List prices, USD per million tokens (input, output), for the ledger; an unknown model counts as 0.5/1.5
PRICES = {"gemini-2.5-flash": (0.30, 2.50), "deepseek-ai/DeepSeek-V4-Pro": (0.5, 1.5), "gpt-4.1": (2.00, 8.00),
          "gpt-5.2": (1.75, 14.00), "gpt-6-sol": (5.0, 40.0), "gpt-realtime-2.1": (4.0, 16.0),
          "gpt-realtime-2.1-mini": (0.6, 2.4), "models/gemini-3.8-live-extended-thinking": (1.0, 12.0)}
CONCURRENCY = {"gemini": 16, "deepinfra": 32, "openai": 16, "anthropic": 4}

_clients = None
_sems: dict[str, asyncio.Semaphore] = {}


def load_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return values


def clients() -> dict:
    global _clients
    if _clients is None:
        from openai import AsyncOpenAI
        env = load_env(ENV_FILE)
        _clients = {name: AsyncOpenAI(base_url=url, api_key=env[key], max_retries=2, timeout=180)
                    for name, (url, key) in PROVIDERS.items() if env.get(key)}
    return _clients


def _sem(provider: str) -> asyncio.Semaphore:
    if provider not in _sems:
        _sems[provider] = asyncio.Semaphore(CONCURRENCY.get(provider, 8))
    return _sems[provider]


def log_usage(provider: str, model: str, purpose: str, usage) -> None:
    pin, pout = PRICES.get(model, (0.5, 1.5))
    tin = getattr(usage, "prompt_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    tout = total - tin if total else (getattr(usage, "completion_tokens", 0) or 0)  # includes hidden thinking tokens
    append_jsonl(LEDGER, {"t": round(time.time()), "provider": provider, "model": model, "purpose": purpose,
                          "in": tin, "out": tout, "usd": round((tin * pin + tout * pout) / 1e6, 6)})


def reasoning_kwargs(provider: str, model: str, reasoning: str | None, max_tokens: int) -> dict:
    """Thinking settings. "auto": Gemini Flash models think not at all, Gemini Pro a little with room added to
    max_tokens, OpenAI's reasoning models low."""
    if reasoning != "auto":
        return {"reasoning_effort": reasoning} if reasoning else {}
    if provider == "gemini" and "flash" in model:
        return {"reasoning_effort": "none"}
    if provider == "gemini" and "pro" in model:
        return {"reasoning_effort": "low", "max_tokens": max_tokens + 1500}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if provider == "openai" and model.startswith(("gpt-5", "gpt-6")):
        return {"reasoning_effort": "low"}
    return {}


async def chat(provider: str, model: str, messages: list[dict], purpose: str, temperature: float | None = 0.9,
               max_tokens: int = 700, attempts: int = 5, reasoning: str | None = "auto") -> str:
    """One chat completion with backoff on rate limits and transient errors; usage goes to the ledger."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    kwargs.update(reasoning_kwargs(provider, model, reasoning, max_tokens))
    if provider == "openai" and model.startswith(("gpt-5", "gpt-6")):
        # OpenAI's reasoning models take max_completion_tokens, which also covers their thinking, and no temperature
        kwargs["max_completion_tokens"], temperature = kwargs.pop("max_tokens"), None
    if temperature is not None and provider != "anthropic":
        kwargs["temperature"] = temperature
    last = None
    for attempt in range(attempts):
        try:
            async with _sem(provider):
                response = await clients()[provider].chat.completions.create(**kwargs)
            log_usage(provider, model, purpose, response.usage)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            await asyncio.sleep(min(60, 3 * 2 ** attempt))
    raise last


def spend(purpose_prefix: str = "") -> float:
    return round(sum(r["usd"] for r in read_jsonl(LEDGER) if r["purpose"].startswith(purpose_prefix)), 2) \
        if LEDGER.exists() else 0.0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _timestamp(value: str) -> str | None:
    """The one spelling of a time slot, or None when this is not one (as cekura-bench's mock-tool server reads it)."""
    if "-" not in value:
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if moment.utcoffset():
        return None
    return moment.replace(tzinfo=None, microsecond=0).isoformat()


def normalize(value):
    """An argument as its meaning: digit strings as digits, other text case- and punctuation-insensitive, lists as
    sets (cekura-bench mock_tools.server._normalize, unchanged)."""
    if isinstance(value, str):
        stripped = value.strip()
        slot = _timestamp(stripped)
        if slot is not None:
            return slot
        digits = "".join(ch for ch in stripped if ch.isdigit())
        if len(digits) >= 7 and not any(ch.isalpha() for ch in stripped):
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            return digits
        return " ".join("".join(ch if ch.isalnum() else " " for ch in stripped).lower().split())
    if isinstance(value, list):
        return sorted((normalize(item) for item in value), key=repr)
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    return value
