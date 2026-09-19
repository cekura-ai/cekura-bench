"""Deadline-wait behavior using controlled clocks; no real-clock pacing trials."""
import asyncio

import pytest

import stt_bench.timing as timing


@pytest.mark.parametrize('deadline', [float('nan'), float('inf'), float('-inf')])
def test_invalid_deadline_fails_before_waiting(monkeypatch, deadline):
    async def unexpected_sleep(_):
        pytest.fail('An invalid deadline must not schedule a wait')
    monkeypatch.setattr(timing.asyncio, 'sleep', unexpected_sleep)
    with pytest.raises(ValueError, match='finite'):
        asyncio.run(timing.wait_until(deadline, lambda: 0.))


def test_already_late_deadline_does_not_sleep_or_spin(monkeypatch):
    checks = []
    def now():
        checks.append(1)
        return 1.
    async def unexpected_sleep(_):
        pytest.fail('An already-passed deadline must not sleep')
    monkeypatch.setattr(timing.asyncio, 'sleep', unexpected_sleep)
    asyncio.run(timing.wait_until(.5, now))
    assert len(checks) <= 2


def test_coarse_wait_is_cancellable(monkeypatch):
    async def scenario():
        waiting = asyncio.Event()
        async def blocked_sleep(seconds):
            assert 0 < seconds <= .001
            waiting.set()
            await asyncio.Future()
        monkeypatch.setattr(timing.asyncio, 'sleep', blocked_sleep)
        task = asyncio.create_task(timing.wait_until(1., lambda: 0.))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
    asyncio.run(scenario())


def test_final_wait_remains_bounded_when_deadline_clock_stops(monkeypatch):
    ticks = []
    def elapsed_clock():
        value = len(ticks) * .00005
        ticks.append(value)
        assert value <= .0011, 'Final spin exceeded its 1 ms budget'
        return value
    async def unexpected_sleep(_):
        pytest.fail('A sub-millisecond final wait should not use a coarse sleep')
    monkeypatch.setattr(timing.time, 'perf_counter', elapsed_clock)
    monkeypatch.setattr(timing.asyncio, 'sleep', unexpected_sleep)
    asyncio.run(timing.wait_until(.0005, lambda: 0.))
    assert .001 <= ticks[-1] <= .0011


def test_overslept_deadline_returns_without_additional_waits(monkeypatch):
    clock = [0.]
    sleeps = []
    async def delayed_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += .080
    monkeypatch.setattr(timing.asyncio, 'sleep', delayed_sleep)
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: clock[0])
    asyncio.run(timing.wait_until(.020, lambda: clock[0]))
    assert sleeps == [.001]
    assert clock[0] == .080


def test_native_wait_converts_relative_clock_without_sleep_or_spin(monkeypatch):
    calls = []
    class Timer:
        async def wait_until(self, deadline):
            calls.append(deadline)
    async def unexpected_sleep(_):
        pytest.fail('The native timer must not also poll or busy wait')
    monkeypatch.setattr(timing.asyncio, 'sleep', unexpected_sleep)
    monkeypatch.setattr(timing.time, 'perf_counter', lambda: 100.)
    asyncio.run(timing.wait_until(.020, lambda: .005, timer=Timer()))
    assert calls == [100.015]
    asyncio.run(timing.wait_until(.001, lambda: .005, timer=Timer()))
    assert calls == [100.015]


def test_portable_scope_requires_no_native_timer(monkeypatch):
    monkeypatch.setattr(timing.sys, 'platform', 'linux')
    async def scenario():
        async with timing.deadline_timer() as timer:
            assert timer is None
    asyncio.run(scenario())
    assert timing.scheduler_backend() == 'bounded-cooperative-v2'
