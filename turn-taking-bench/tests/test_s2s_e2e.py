"""Track 1 end to end against the service bench's scripted agent (``service/adapters/fake.py``):
a session per conversation, the caller channel streamed in real time, onsets
read off the recorded output, TurnBench's scorer on the result. The fake agent's
reply timing is known (it answers ``endpoint_ms`` after the caller goes quiet),
so the EOT latencies and the cut-in during a scripted pause are checkable
against ground truth. Real-time by construction, so these tests take about as
long as the audio they stream."""

import json
from pathlib import Path

import numpy as np
import pytest

from turn_taking_bench.author.labels import LABEL_TURN
from turn_taking_bench.author.package import DatasetWriter, flac_bytes
from turn_taking_bench.runner import run_benchmark, speaker_events, stream_speaker
from turn_taking_bench.s2s import default_config
from turn_taking_bench.s2s.model import S2SModel

RATE = 48000
FAKE = {"reply_ms": 1200.0, "endpoint_ms": 500.0, "chunk_ms": 40.0}


def channel(spans: list[tuple[float, float]], seconds: float, amplitude: float = 0.3) -> np.ndarray:
    """A caller channel: a 220 Hz tone over each (start, end) span, silence elsewhere."""
    samples = np.zeros(int(RATE * seconds), dtype=np.float32)
    t = np.arange(samples.size) / RATE
    for start, end in spans:
        lo, hi = int(start * RATE), int(end * RATE)
        samples[lo:hi] = amplitude * np.sin(2 * np.pi * 220.0 * t[lo:hi])
    return samples


def turns(spans: list[tuple[float, float]]) -> list[dict]:
    return [{"start_s": s, "end_s": e, "label": LABEL_TURN, "text": ""} for s, e in spans]


CONVERSATIONS = {
    # caller: three turns; the gap 4.5 -> 5.5 is a mid-turn pause (nobody else speaks)
    "1": {"seconds": 8.0, "caller": [(0.5, 1.5), (3.5, 4.5), (5.5, 6.5)], "agent": [(2.0, 3.0)]},
    "2": {"seconds": 4.0, "caller": [(0.5, 1.5)], "agent": [(2.0, 3.0)]},
}


def write_dataset(out: Path) -> Path:
    writer = DatasetWriter(out, name="s2s-e2e")
    for cid, spec in CONVERSATIONS.items():
        row = {
            "conversation_id": cid,
            "speaker_1_audio": {"bytes": flac_bytes(channel(spec["caller"], spec["seconds"]), RATE), "path": f"{cid}_speaker_1.flac"},
            "speaker_2_audio": {"bytes": flac_bytes(channel(spec["agent"], spec["seconds"]), RATE), "path": f"{cid}_speaker_2.flac"},
            "metadata": {
                "conversation_type": "Authored/Scripted",
                "speaker_1_actor_id": "test:caller",
                "speaker_2_actor_id": "test:agent",
                "speaker_1_actor_gender": "f",
                "speaker_2_actor_gender": "m",
            },
            "authored": json.dumps({"spec_id": f"E2E-{cid}"}),
        }
        for annotator in "abc":
            row[f"speaker_1_annotation_{annotator}"] = turns(spec["caller"])
            row[f"speaker_2_annotation_{annotator}"] = turns(spec["agent"])
        writer.add(row, {"spec_id": f"E2E-{cid}", "family": "e2e", "difficulty": "easy"}, spec["seconds"])
    writer.finish()
    return out


def test_fake_provider_session_yields_onsets_as_eots(tmp_path):
    config = {**default_config("fake"), "trailing_grace_s": 0.8, "adapter_kwargs": FAKE}
    model = S2SModel(config)
    assert model.begin_channel("1", 1, tmp_path / "artifacts") is True
    samples = channel(CONVERSATIONS["1"]["caller"], CONVERSATIONS["1"]["seconds"])
    run = stream_speaker(model, samples, RATE, realtime=False)  # the model paces itself

    events, _deduped = speaker_events(run.events)
    assert events.interruption == []
    # A reply 500 ms after each quiet: the one at ~5.0 s lands in the scripted pause.
    assert len(events.eot) == 3
    for got, expected in zip(events.eot, (2.0, 5.0, 7.0)):
        assert expected - 0.02 <= got <= expected + 0.25, events.eot
    diagnostics = run.diagnostics
    assert diagnostics["readout"] == "onset" and diagnostics["responses"] == 3
    assert diagnostics["interrupted"] == 1  # the caller's third turn barged into the second reply
    assert diagnostics["pacing"]["pacing_void"] is False
    assert all(not o["caller_active"] for o in diagnostics["onsets"])

    # The agent channel is not streamed.
    assert model.begin_channel("1", 2, tmp_path / "artifacts") is False
    assert model.push(samples[:960], 0.02) == [] and model.flush(0.02) == []

    artifacts = tmp_path / "artifacts" / "1" / "speaker_1"
    for name in ("caller.wav", "agent.wav", "timelines.json", "events.jsonl", "raw.jsonl", "readout.json"):
        assert (artifacts / name).is_file(), name
    readout = json.loads((artifacts / "readout.json").read_text())
    assert readout["events"]["eot"] == events.eot
    assert readout["session"]["requested"]["instructions_sha256"]


def test_run_benchmark_scores_fake_provider_with_turnbench_scorer(tmp_path):
    dataset = write_dataset(tmp_path / "dataset")
    out = tmp_path / "run"
    info = run_benchmark(
        model_name="s2s:fake",
        workers=2,  # forced to 1: one session at a time
        out=out,
        realtime=True,
        config_overrides={"trailing_grace_s": 0.8, "adapter_kwargs": FAKE},
        dataset_dir=dataset,
    )
    assert info["failures"] == []
    assert info["workers"] == 1
    assert info["model"]["provenance"]["adapter"] == "fake"
    assert info["model"]["provenance"]["prompt_sha256"]

    report = json.loads((out / "report.json").read_text())
    caller = report["blocks"]["caller"]["aggregate"]["eot"]
    assert (caller["tp"], caller["fn"], caller["fp"], caller["tn"]) == (3, 0, 1, 0)
    assert 450 <= caller["latency_ms"]["p50"] <= 800
    card = report["blocks"]["caller"]["scorecard"]
    assert card["turn_taking"]["value"] == 1.0
    assert card["pause_handling"]["value"] == 0.0
    assert card["interruption_handling"]["value"] is None  # INT undefined for S2S
    # The skipped agent channel is recorded as such, with no events.
    for record in info["conversations"]:
        assert record["speakers"]["2"]["raw_events"] == {"eot": 0, "interruption": 0}
    lines = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert {line["speaker"]: ("skipped" in (line.get("diagnostics") or {})) for line in lines if line["conversation_id"] == "1"} == {1: False, 2: True}
    assert (out / "artifacts" / "1" / "speaker_1" / "agent.wav").is_file()
