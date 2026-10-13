"""The shape of a tool, shared by both lanes.

It lives here rather than in an adapter because the tool contract is the thing
both lanes have in common: the service bench hands these to a realtime session, the agent bench hands
them to a framework, and the mock server that serves the contract produces them.
Keeping it in the harness would mean the deployed reference agent imported the
thing measuring it, which is backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
