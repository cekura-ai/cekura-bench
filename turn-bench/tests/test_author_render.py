"""Renderers and the render cache: deterministic mock, cached bytes with a
sha256, voice labels shared with the service bench, ElevenLabs gated on its key."""

import os

import numpy as np
import pytest

from turn_bench.author import service_bridge
from turn_bench.author.render import (
    VOICES,
    ElevenLabsRenderer,
    MockRenderer,
    RenderCache,
    _elevenlabs_pcm_at_speed,
    make_renderer,
    read_env_key,
    render_key,
)


def test_mock_render_is_deterministic_and_breaks_are_true_silence():
    renderer = MockRenderer()
    a, rate = renderer.render("Hello there. <1s> Still here.", "f-gb")
    b, _ = renderer.render("Hello there. <1s> Still here.", "f-gb")
    assert rate == 24000 and np.array_equal(a, b)
    (s0, e0), (s1, e1) = renderer.layout("Hello there. <1s> Still here.", "f-gb")
    assert s1 - e0 == 24000 and np.abs(a[e0:s1]).max() == 0.0 and np.abs(a[s0:e0]).max() > 0.25
    other, _ = renderer.render("Hello there. <1s> Still here.", "m-us")
    assert not np.array_equal(a, other)  # a different voice renders differently


def test_cache_hits_on_the_second_request_and_records_provenance(tmp_path):
    cache = RenderCache(tmp_path)
    first = cache.get(MockRenderer(), "Mm-hmm.", "f-us")
    second = cache.get(MockRenderer(), "Mm-hmm.", "f-us")
    assert (cache.hits, cache.misses) == (1, 1)
    assert first.sha256 == second.sha256 and len(first.sha256) == 64
    assert np.array_equal(first.samples, second.samples)
    assert first.provenance["renderer"] == "mock" and first.provenance["voice"] == "f-us" and first.provenance["text"] == "Mm-hmm."
    assert cache.get(MockRenderer(), "Mm-hmm.", "m-us").sha256 != first.sha256


def test_voice_registry_covers_the_service_bench_corpus_and_the_accent_voices():
    from turn_bench.author.render import voice_registry

    published = service_bridge.corpus_voices()
    assert set(published) < set(VOICES)
    registry = voice_registry("m")
    for label, voice in VOICES.items():
        assert registry[label]["elevenlabs_voice_id"] == (published[label].voice_id if label in published else voice.elevenlabs_voice_id)
        assert registry[label]["accent"] and registry[label]["model"] == "m"
    assert ElevenLabsRenderer(api_key="x", model="m").service_voice("m-in").voice_id == VOICES["m-in"].elevenlabs_voice_id


def test_elevenlabs_renderer_requires_the_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("TURN_BENCH_ENV_FILE", str(tmp_path / "absent.env"))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        make_renderer("elevenlabs")
    renderer = ElevenLabsRenderer(api_key="x", model="m")
    with pytest.raises(KeyError, match="no ElevenLabs id"):
        renderer.render("hi", "no-such-voice")


def test_elevenlabs_key_comes_from_the_env_file_and_never_from_the_repr(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# keys\nOTHER=1\nexport ELEVENLABS_API_KEY='sk-test-value'\n")
    assert read_env_key(env, "ELEVENLABS_API_KEY") == "sk-test-value"
    assert read_env_key(env, "MISSING") is None and read_env_key(tmp_path / "nope", "X") is None
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("TURN_BENCH_ENV_FILE", str(env))
    renderer = make_renderer("elevenlabs")
    assert renderer.api_key == "sk-test-value" and renderer.model
    assert "sk-test-value" not in repr(renderer)


def test_cache_counts_renderer_calls_and_characters(tmp_path):
    cache = RenderCache(tmp_path)
    cache.get(MockRenderer(), "Mm-hmm.", "f-us")
    cache.get(MockRenderer(), "Mm-hmm.", "f-us")
    cache.get(MockRenderer(), "Okay.", "f-us")
    assert cache.stats()["renderer_calls"] == 2 and cache.stats()["characters_rendered"] == len("Mm-hmm.") + len("Okay.")


def test_speed_stretches_the_mock_speech_but_not_the_breaks_and_changes_the_cache_key(tmp_path):
    renderer = MockRenderer()
    text = "Hello there. <1s> Still here."
    (s0, e0), (s1, e1) = renderer.layout(text, "f-us")
    (t0, f0), (t1, f1) = renderer.layout(text, "f-us", speed=0.8)
    assert f0 - t0 == pytest.approx((e0 - s0) / 0.8, abs=2) and f1 - t1 == pytest.approx((e1 - s1) / 0.8, abs=2)
    assert t1 - f0 == s1 - e0 == 24000  # the scripted pause is the scripted pause at any speed
    slow, rate = renderer.render(text, "f-us", 0.8)
    assert len(slow) > len(renderer.render(text, "f-us")[0])
    with pytest.raises(ValueError, match="speed"):
        renderer.render(text, "f-us", 0.5)
    assert render_key(renderer, text, "f-us") == render_key(renderer, text, "f-us", 1.0) != render_key(renderer, text, "f-us", 0.8)
    cache = RenderCache(tmp_path)
    natural, slowed = cache.get(renderer, text, "f-us"), cache.get(renderer, text, "f-us", 0.8)
    assert natural.sha256 != slowed.sha256 and slowed.provenance["speed"] == 0.8 and natural.provenance["speed"] == 1.0
    assert cache.misses == 2


def test_elevenlabs_speed_goes_out_as_voice_settings(monkeypatch):
    import io
    import json
    import urllib.request

    captured = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["has_key"] = bool(request.get_header("Xi-api-key"))
        return Response(b"\x00\x01" * 480)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    voice = service_bridge.corpus_voices()["f-us"]
    pcm = _elevenlabs_pcm_at_speed("sk-test", "Hello. <break time=\"1s\"/> There.", voice, 0.8)
    assert len(pcm) == 960 and captured["has_key"]
    assert captured["body"]["voice_settings"] == {"speed": 0.8} and captured["body"]["model_id"] == voice.model
    assert "<break time=\"1s\"/>" in captured["body"]["text"] and "pcm_24000" in captured["url"]
    assert "sk-test" not in json.dumps(captured["body"])  # the key is a header, never in the payload
