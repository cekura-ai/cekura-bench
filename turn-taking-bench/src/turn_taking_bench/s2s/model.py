"""``S2SModel``: a live speech-to-speech provider behind the ``TurnModel`` interface.

One instance serves one conversation. The runner tells it which channel is
about to be streamed (``begin_channel``, an optional hook ``turn_taking_bench.runner``
calls before ``reset``); for a channel the model wants it opens one provider
session through the service bench's adapter and from then on every ``push``
forwards one 20 ms frame — resampled by the published filter — to the socket,
at real-time pace. The session lives on a background thread running its own
asyncio loop, because the adapters are asynchronous and the harness interface
is not; ``push`` never waits on the network.

Nothing is decided during streaming. ``flush`` streams ``trailing_grace_s`` of
silence (the provider must hear silence to endpoint the caller's last turn, and
the last reply has to arrive), closes the session, and reads the events off the
recorded artifacts with ``turn_taking_bench.s2s.onsets``:

* ``readout: onset`` — EOT at every onset of model speech while the caller was
  quiet (TurnBench's generative protocol, ``baselines/gemini_vad``). No INT.
* ``readout: vad_events`` — EOT at the provider's ``speech_stopped``, INT at its
  ``speech_started`` (TurnBench's ``openai_server_vad`` / ``openai_semantic_vad``).

Both readouts' raw timings are kept in the diagnostics and in ``readout.json``,
so a run can be re-read the other way without a new session.

**Clocks.** The runner pushes frame *i* no earlier than ``start + (i + 1) * 20 ms``
— at the *end* of the frame's audio interval, which is the causal reading: a
live model cannot hear a sample before it has been spoken. The model's audio
clock is anchored there: ``origin`` is the wall instant at which audio time 0
lies, fixed at the first push, and every provider-side instant ``t`` maps to
audio time ``t - origin``. A commit time is that audio time, clamped to the
audio pushed so far (TurnBench's ``appended_ms`` rule), so causality holds by
construction and the harness's check is a second opinion. The service bench
sends a chunk at the *start* of its interval instead, so a latency measured
here is up to one frame (20 ms) more conservative than the service bench's for
the same provider; the convention is recorded with every run.

If the runner is not pacing (``--realtime`` not given) the model paces itself,
sleeping until each frame's audio time has elapsed: an S2S model is real-time
by nature. Pacing slip (a push later than its frame time) is measured and
published with the service bench's thresholds; a conversation whose host fell
behind by more than ``VOID_SLIP_MS`` is flagged ``pacing_void`` rather than
blamed on the provider.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from turn_taking_bench.author import service_bridge
from turn_taking_bench.model import FRAME_S, Event
from turn_taking_bench.s2s import MODES, READOUTS, S2S_PROVIDERS, SYSTEM_PROMPT, prompt_sha256
from turn_taking_bench.s2s.credentials import credential, describe
from turn_taking_bench.s2s.onsets import active_at, active_mask, output_onsets
from turn_taking_bench.s2s.resample import StreamingResampler

CONNECT_TIMEOUT_S = 40.0
CLOSE_TIMEOUT_S = 30.0


class SessionError(RuntimeError):
    """The provider refused, closed or broke the session; the conversation is void."""


# ---- adapter construction --------------------------------------------------------------


def _extra_providers() -> dict[str, Any]:
    """turn-taking-bench's own adapters (turn_taking_bench/s2s/*), registered here so the
    service bench's registry stays untouched."""
    from service.registry import ProviderEntry

    from turn_taking_bench.s2s import azure_realtime, deepslate, gemini_thinking, gpt_live, nova_sonic, phonic

    return {
        "azure-realtime": azure_realtime.provider_entry(),
        "gemini-live-thinking": gemini_thinking.provider_entry(),
        "gpt-live": gpt_live.provider_entry(),
        "nova-sonic": ProviderEntry(
            key="nova-sonic", adapter=nova_sonic.NovaSonicAdapter, default_model=nova_sonic.DEFAULT_MODEL,
            credential_env="AWS_ACCESS_KEY_ID", default_voice=nova_sonic.DEFAULT_VOICE,
        ),
        "phonic": ProviderEntry(
            key="phonic", adapter=phonic.PhonicAdapter, default_model=phonic.MODEL,
            credential_env="PHONIC_API_KEY", default_voice=phonic.VOICE,
        ),
        "deepslate": ProviderEntry(
            key="deepslate", adapter=deepslate.DeepslateAdapter, default_model=deepslate.MODEL,
            credential_env="DEEPSLATE_API_KEY", default_voice=None,
        ),
    }


def provider_entry(provider: str):
    if provider not in S2S_PROVIDERS:
        raise KeyError(f"unknown s2s provider {provider!r}; known: {', '.join(S2S_PROVIDERS)}")
    service_providers = service_bridge.registry().PROVIDERS
    if provider in service_providers:
        return service_providers[provider]
    return _extra_providers()[provider]


def vad_only_adapter(base: type) -> type:
    """``base`` with TurnBench's baseline session: text output and no responses, so
    the only thing the provider does is run its endpointer over our audio
    (``baselines/openai_realtime.py::_session_config``). Only an OpenAI-shaped
    protocol has ``create_response``; the base adapter's own payload is kept and
    the two keys are changed in place, so the shape stays the adapter's."""

    class VadOnlyAdapter(base):  # type: ignore[misc,valid-type]
        name = f"{base.name}-vad-only"

        def _session_payload(self) -> dict[str, Any]:
            session = super()._session_payload()
            detection = session.get("audio", {}).get("input", {}).get("turn_detection")
            if detection is None:
                detection = session.get("turn_detection")
            if not isinstance(detection, dict):
                raise ValueError(f"{base.name} has no server turn detection to run alone")
            detection["create_response"] = False
            if "output_modalities" in session:
                session["output_modalities"] = ["text"]
                session.get("audio", {}).pop("output", None)
            return session

    return VadOnlyAdapter


def session_config(config: dict[str, Any], entry: Any):
    base = service_bridge.adapters_base()
    detection = base.TurnDetection(
        mode=config.get("turn_detection") or "server_vad",
        silence_duration_ms=config.get("silence_duration_ms"),
        threshold=config.get("threshold"),
        prefix_padding_ms=config.get("prefix_padding_ms"),
        create_response=config.get("create_response"),
        interrupt_response=config.get("interrupt_response"),
    )
    if config["mode"] == "vad_only":
        return base.SessionConfig(
            instructions="",
            first_message=None,
            voice=None,
            tools=(),
            turn_detection=detection,
            modality="text",
            transcribe_input=False,
        )
    return base.SessionConfig(
        instructions=config.get("instructions") or SYSTEM_PROMPT,
        first_message=None,
        voice=config.get("voice") or entry.default_voice,
        tools=(),
        turn_detection=detection,
        modality="audio",
        transcribe_input=bool(config.get("transcribe_input", True)),
    )


def resolve_model(config: dict[str, Any]) -> str:
    return config.get("model") or provider_entry(config["provider"]).default_model


def adapter_class(config: dict[str, Any]) -> type:
    entry = provider_entry(config["provider"])
    cls = entry.adapter
    return vad_only_adapter(cls) if config["mode"] == "vad_only" else cls


def api_key(config: dict[str, Any]) -> str:
    entry = provider_entry(config["provider"])
    if config["provider"] == "fake":
        return ""
    return credential(entry.credential_env, config.get("env_file"))


def model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """What ``run.json`` records about the model: provider, resolved model id,
    adapter, mode, readout, the prompt's digest and where the credential came
    from (never the credential)."""
    entry = provider_entry(config["provider"])
    cls = adapter_class(config)
    found = None if config["provider"] == "fake" else describe(entry.credential_env, config.get("env_file"))
    return {
        "provider": config["provider"],
        "model": resolve_model(config),
        "adapter": cls.name,
        "adapter_module": entry.adapter.__module__,
        "mode": config["mode"],
        "readout": config["readout"],
        "channels": list(config.get("channels", [1])),
        "turn_detection": config.get("turn_detection") or "server_vad",
        "silence_duration_ms": config.get("silence_duration_ms"),
        "threshold": config.get("threshold"),
        "prefix_padding_ms": config.get("prefix_padding_ms"),
        "voice": config.get("voice") or entry.default_voice,
        "instructions": "" if config["mode"] == "vad_only" else (config.get("instructions") or SYSTEM_PROMPT),
        "prompt_sha256": None if config["mode"] == "vad_only" else prompt_sha256(config.get("instructions") or SYSTEM_PROMPT),
        "credential": None if found is None else {"variable": found[1], "source": "environment" if found[0] == "environment" else "dotenv"},
        "service_bench": service_bridge.service_module("provenance").harness_state(),
        "frame_push_convention": "frame pushed at the end of its audio interval; chunk t_wall is the push instant",
        "discloses": list(entry.discloses),
    }


# ---- the session thread ----------------------------------------------------------------


class _Session(threading.Thread):
    """One provider session on its own asyncio loop. ``send`` is thread-safe
    and non-blocking; ``finish`` streams the trailing silence, closes, and joins."""

    def __init__(self, make_adapter: Callable[[], Any], clock: Any, log: Any, *, grace_s: float, input_rate: int) -> None:
        super().__init__(name="s2s-session", daemon=True)
        self.make_adapter = make_adapter
        self.clock = clock
        self.log = log
        self.grace_s = grace_s
        self.input_rate = input_rate
        self.adapter: Any = None
        self.error: str | None = None
        self.ready = threading.Event()
        self.finished = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None

    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:  # noqa: BLE001 -- recorded, surfaced by the model
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self.ready.set()
            self.finished.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self.adapter = self.make_adapter()
        try:
            await asyncio.wait_for(self.adapter.connect(), CONNECT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            self.error = f"connect: {type(exc).__name__}: {exc}"
            await self._close()
            return
        self.ready.set()
        ev = service_bridge.events()
        audio = service_bridge.audio()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                pcm, t_send = item
                await self.adapter.send_audio(pcm, t_send)
            # The carrier keeps running after the last frame: silence, paced,
            # for the provider's endpointer and the final reply.
            frame = audio.silence(self.input_rate, FRAME_S * 1000.0)
            origin = self.clock.now()
            for index in range(int(round(self.grace_s / FRAME_S))):
                target = origin + index * FRAME_S
                now = self.clock.now()
                if target > now:
                    await asyncio.sleep(target - now)
                await self.adapter.send_audio(frame, max(target, self.clock.now()))
        except Exception as exc:  # noqa: BLE001 -- a dead socket ends the conversation, not the run
            self.error = f"{type(exc).__name__}: {exc}"
            self.log.emit(ev.SESSION_ERROR, error=repr(exc), where="s2s-carrier")
        finally:
            await self._close()

    async def _close(self) -> None:
        try:
            await asyncio.wait_for(self.adapter.close(), CLOSE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            self.error = self.error or f"close: {type(exc).__name__}: {exc}"

    def send(self, pcm: bytes, t_send: float) -> None:
        if self._loop is None or self._queue is None or self.finished.is_set():
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (pcm, t_send))
        except RuntimeError:  # loop already closed: the error is recorded
            pass

    def finish(self, timeout_s: float) -> None:
        if self._loop is not None and self._queue is not None and not self.finished.is_set():
            try:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            except RuntimeError:
                pass
        self.join(timeout_s)
        if self.is_alive():
            self.error = self.error or "session did not close in time"


# ---- the model ----------------------------------------------------------------------------


class S2SModel:
    """A speech-to-speech provider as a ``TurnModel``; see the module docstring."""

    def __init__(self, config: dict[str, Any]) -> None:
        if config.get("mode", "agent") not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {config.get('mode')!r}")
        if config.get("readout", "onset") not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {config.get('readout')!r}")
        self.config = dict(config)
        self.provider = config["provider"]
        self.entry = provider_entry(self.provider)
        self.adapter_cls = adapter_class(self.config)
        self.model_id = resolve_model(self.config)
        self._key: str | None = None
        self._conversation_id: str | None = None
        self._speaker: int | None = None
        self._wanted = False
        self._directory: Path | None = None
        self._session: _Session | None = None
        self._clock: Any = None
        self._resampler: StreamingResampler | None = None
        self._origin: float | None = None
        self._last_t_end = 0.0
        self._pushes = 0
        self._max_slip_ms = 0.0
        self._slip_events = 0
        self._diagnostics: dict[str, Any] = {}

    # -- runner hooks -----------------------------------------------------------------

    def begin_channel(self, conversation_id: str, speaker: int, artifacts_dir: Path | None = None) -> bool:
        """Called by the runner before ``reset``. Opens the session for a wanted
        channel; returns False for a channel the model does not stream."""
        self._conversation_id, self._speaker = conversation_id, speaker
        self._wanted = speaker in [int(c) for c in self.config.get("channels", [1])]
        self._diagnostics = {}
        if not self._wanted:
            self._diagnostics = {"skipped": f"speaker_{speaker} is not streamed: the S2S model replaces that channel"}
            return False
        if self._key is None:
            self._key = api_key(self.config)
        self._directory = None
        if artifacts_dir is not None and self.config.get("artifacts", True):
            self._directory = Path(artifacts_dir) / str(conversation_id) / f"speaker_{speaker}"
            self._directory.mkdir(parents=True, exist_ok=True)
        ev = service_bridge.events()
        self._clock = ev.Clock()
        log = ev.EventLog(
            self._clock,
            None if self._directory is None else self._directory / "events.jsonl",
            None if self._directory is None else self._directory / "raw.jsonl",
        )
        config = session_config(self.config, self.entry)
        excluded = self.adapter_cls.unsupported_reason(config)
        if excluded:
            raise SessionError(f"configuration not supported: {excluded}")

        def make_adapter() -> Any:
            return self.adapter_cls(
                model=self.model_id, api_key=self._key, log=log, config=config, **dict(self.config.get("adapter_kwargs") or {})
            )

        self._session = _Session(
            make_adapter, self._clock, log, grace_s=float(self.config.get("trailing_grace_s", 3.0)), input_rate=self.adapter_cls.input_rate
        )
        self._session.start()
        if not self._session.ready.wait(CONNECT_TIMEOUT_S + 5.0) or self._session.error:
            self._session.finish(CLOSE_TIMEOUT_S)
            raise SessionError(f"{self.provider} session refused: {self._session.error or 'connect timed out'}")
        return True

    # -- TurnModel ----------------------------------------------------------------------

    def reset(self, sample_rate: int) -> None:
        self._resampler = StreamingResampler(sample_rate, self.adapter_cls.input_rate)
        self._origin = None
        self._last_t_end = 0.0
        self._pushes = 0
        self._max_slip_ms = 0.0
        self._slip_events = 0

    def push(self, frame: np.ndarray, t_end_s: float) -> list[Event]:
        if not self._wanted:
            return []
        assert self._session is not None and self._resampler is not None and self._clock is not None
        if self._session.error:
            raise SessionError(f"{self.provider} session failed while streaming: {self._session.error}")
        now = self._clock.now()
        if self._origin is None:
            self._origin = now - t_end_s
        target = self._origin + t_end_s
        if now < target:  # the runner is not pacing; an S2S model is real-time by nature
            time.sleep(target - now)
            now = self._clock.now()
        slip_ms = (now - target) * 1000.0
        caller = service_bridge.caller()
        self._max_slip_ms = max(self._max_slip_ms, slip_ms)
        if slip_ms > caller.SLIP_WARN_MS:
            self._slip_events += 1
        pcm = service_bridge.audio().to_pcm(self._resampler.process(np.asarray(frame, dtype=np.float64) * 32767.0))
        self._session.send(pcm, now)
        self._pushes += 1
        self._last_t_end = t_end_s
        return []

    def flush(self, t_end_s: float) -> list[Event]:
        if not self._wanted:
            return []
        assert self._session is not None and self._resampler is not None
        tail = self._resampler.flush()
        if tail.size and not self._session.error:
            self._session.send(service_bridge.audio().to_pcm(tail), self._clock.now())
        self._last_t_end = max(self._last_t_end, t_end_s)
        self._session.finish(float(self.config.get("trailing_grace_s", 3.0)) + CLOSE_TIMEOUT_S)
        if self._origin is None:
            self._origin = self._clock.now() - t_end_s
        events = self._derive()
        self._write_artifacts()
        if self._session.error and self._pushes == 0:
            raise SessionError(f"{self.provider} session failed before any audio: {self._session.error}")
        return events

    def diagnostics(self) -> dict[str, Any]:
        return self._diagnostics

    # -- readout ---------------------------------------------------------------------------

    def _audio_time(self, t_wall: float) -> float:
        """A provider-side instant on the audio clock, clamped to the audio pushed."""
        assert self._origin is not None
        return max(0.0, min(t_wall - self._origin, self._last_t_end))

    def _derive(self) -> list[Event]:
        assert self._session is not None and self._origin is not None
        adapter = self._session.adapter
        log = adapter.log
        ev = service_bridge.events()
        caller = service_bridge.caller()
        metrics = service_bridge.metrics()

        onsets = output_onsets(
            bytes(adapter.agent_pcm), adapter.agent_timeline, log.events, merge_gap_s=float(self.config.get("merge_gap_s", 0.5))
        )
        mask = None
        if self.config.get("caller_active_filter", True) and adapter.caller_pcm:
            mask = active_mask(bytes(adapter.caller_pcm), adapter.input_rate, merge_gap_s=float(self.config.get("caller_merge_gap_s", 0.0)))
        onset_records: list[dict[str, Any]] = []
        onset_eots: list[float] = []
        for onset in onsets:
            audio_t = onset.t - self._origin
            active = mask is not None and active_at(mask, audio_t)
            onset_records.append(
                {**onset.as_json(), "audio_t": round(audio_t, 4), "caller_active": active, "past_audio_end": audio_t > self._last_t_end}
            )
            if not active:
                onset_eots.append(self._audio_time(onset.t))
        vad_eots = [self._audio_time(e.t) for e in log.events if e.kind == ev.VAD_SPEECH_END]
        vad_ints = [self._audio_time(e.t) for e in log.events if e.kind == ev.VAD_SPEECH_START]

        readout = self.config.get("readout", "onset")
        eots, ints = (onset_eots, []) if readout == "onset" else (vad_eots, vad_ints)
        events = [Event("eot", t) for t in _clean(eots)] + [Event("interruption", t) for t in _clean(ints)]
        events.sort(key=lambda e: e.commit_time_s)

        self._diagnostics = {
            "provider": self.provider,
            "model": self.model_id,
            "adapter": adapter.name,
            "mode": self.config["mode"],
            "readout": readout,
            "conversation_id": self._conversation_id,
            "speaker": self._speaker,
            "origin": round(self._origin, 6),
            "audio_time_s": self._last_t_end,
            "pushes": self._pushes,
            "pacing": {
                "max_slip_ms": round(self._max_slip_ms, 2),
                "slip_events": self._slip_events,
                "void_threshold_ms": caller.VOID_SLIP_MS,
                "pacing_void": self._max_slip_ms > caller.VOID_SLIP_MS,
            },
            "responses": len([e for e in log.events if e.kind == ev.AGENT_AUDIO_START]),
            "interrupted": len([e for e in log.events if e.kind == ev.AGENT_INTERRUPTED]),
            "onsets": onset_records,
            "vad_speech_end": [round(t, 4) for t in vad_eots],
            "vad_speech_start": [round(t, 4) for t in vad_ints],
            "events": {"eot": [e.commit_time_s for e in events if e.kind == "eot"], "interruption": [e.commit_time_s for e in events if e.kind == "interruption"]},
            "agent_audio_s": round(adapter.agent_timeline.duration_s, 3),
            "usage": metrics.usage(adapter),
            "transcripts": {"agent": list(adapter.agent_text), "caller_asr": list(adapter.caller_text)},
            "session": {
                "provider_session_id": adapter.session_id,
                "requested": _session_snapshot(adapter.config),
                "acknowledged": adapter.session_ack,
            },
            "error": self._session.error,
            "artifacts": None if self._directory is None else str(self._directory),
        }
        return events

    def _write_artifacts(self) -> None:
        if self._directory is None or self._session is None or self._session.adapter is None:
            return
        adapter = self._session.adapter
        audio = service_bridge.audio()
        if adapter.caller_pcm:
            audio.write_wav(self._directory / "caller.wav", bytes(adapter.caller_pcm), adapter.input_rate)
        if adapter.agent_pcm:
            audio.write_wav(self._directory / "agent.wav", bytes(adapter.agent_pcm), adapter.output_rate)
        (self._directory / "timelines.json").write_text(
            json.dumps(
                {
                    "caller": adapter.caller_timeline.as_json(),
                    "agent": adapter.agent_timeline.as_json(),
                    "wall_origin": adapter.log.clock.wall_origin,
                    "audio_origin": self._origin,
                    "frame_push_convention": "frame pushed at the end of its audio interval; chunk t_wall is the push instant",
                }
            ),
            encoding="utf-8",
        )
        (self._directory / "readout.json").write_text(json.dumps(self._diagnostics, indent=2), encoding="utf-8")
        adapter.log.close()


def _clean(times: list[float]) -> list[float]:
    """Sorted, rounded to the millisecond, exact repeats dropped (TurnBench's ``_clean``)."""
    kept: list[float] = []
    for t in sorted(round(t, 3) for t in times):
        if not kept or t > kept[-1]:
            kept.append(t)
    return kept


def _session_snapshot(config: Any) -> dict[str, Any]:
    return {
        "instructions": config.instructions,
        "instructions_sha256": prompt_sha256(config.instructions) if config.instructions else None,
        "first_message": config.first_message,
        "voice": config.voice,
        "turn_detection": config.turn_detection.label,
        "modality": config.modality,
        "transcribe_input": config.transcribe_input,
    }
