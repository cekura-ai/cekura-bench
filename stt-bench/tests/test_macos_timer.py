"""Native timer lifecycle checks; these assert no OS performance thresholds."""
import asyncio
import errno
import os
import sys
import time

import pytest

from stt_bench.macos_timer import MacOSDeadlineTimer


pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS public timer API")


def test_timer_event_reuse_and_context_cleanup():
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            fd = timer._fd
            assert not os.get_inheritable(fd)
            for _ in range(2):
                deadline = time.perf_counter() + .005
                assert await timer.wait_until(deadline) >= deadline
                assert timer._pending is None
            assert await timer.wait_until(0.0) >= 0
        assert timer._closed and timer._fd == -1
        assert not asyncio.get_running_loop().remove_reader(fd)
        with pytest.raises(OSError) as error:
            os.fstat(fd)
        assert error.value.errno == errno.EBADF
        timer.close()  # Idempotent on its owning loop.
    asyncio.run(scenario())


def test_cancellation_disarms_and_allows_reuse():
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            task = asyncio.create_task(timer.wait_until(time.perf_counter() + 60))
            await asyncio.sleep(0)  # Let the waiter register its future timer.
            assert timer._pending is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert timer._pending is None and timer._active_ident is None
            deadline = time.perf_counter() + .005
            assert await timer.wait_until(deadline) >= deadline
    asyncio.run(scenario())


def test_concurrent_wait_is_rejected_without_cancelling_first():
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            task = asyncio.create_task(timer.wait_until(time.perf_counter() + 60))
            await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="Concurrent waits"):
                await timer.wait_until(time.perf_counter() + 60)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    asyncio.run(scenario())


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf"), -1, True, "1"])
def test_invalid_deadline_is_rejected_without_arming(deadline):
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            with pytest.raises(ValueError, match="finite and nonnegative"):
                await timer.wait_until(deadline)
            assert timer._pending is None and timer._sequence == 0
    asyncio.run(scenario())


def test_close_cancels_waiter_and_rejects_future_use():
    async def scenario():
        timer = MacOSDeadlineTimer()
        task = asyncio.create_task(timer.wait_until(time.perf_counter() + 60))
        await asyncio.sleep(0)
        timer.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(RuntimeError, match="closed"):
            await timer.wait_until(0.0)
        assert timer._pending is None
    asyncio.run(scenario())


def test_foreign_loop_and_outside_loop_use_are_rejected():
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            def foreign_loop():
                async def attempt():
                    with pytest.raises(RuntimeError, match="owning event loop"):
                        await timer.wait_until(0.0)
                    with pytest.raises(RuntimeError, match="owning event loop"):
                        timer.close()
                asyncio.run(attempt())
            await asyncio.to_thread(foreign_loop)
            assert not timer._closed
        return timer
    timer = asyncio.run(scenario())
    with pytest.raises(RuntimeError, match="owning event loop"):
        timer.close()


def test_constructor_closes_fd_when_reader_registration_fails(monkeypatch):
    async def scenario():
        captured = []
        def fail_reader(fd, callback):
            captured.append(fd)
            raise OSError("Reader registration failed")
        monkeypatch.setattr(asyncio.get_running_loop(), "add_reader", fail_reader)
        with pytest.raises(OSError, match="registration failed"):
            MacOSDeadlineTimer()
        assert len(captured) == 1
        with pytest.raises(OSError) as error:
            os.fstat(captured[0])
        assert error.value.errno == errno.EBADF
    asyncio.run(scenario())


def test_registration_error_clears_pending_wait_and_preserves_exception(monkeypatch):
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            original = timer._change
            def fail_arm(event, *, ignore_missing=False):
                if not ignore_missing:
                    raise OSError(errno.EINVAL, "Timer registration rejected")
                return original(event, ignore_missing=ignore_missing)
            monkeypatch.setattr(timer, "_change", fail_arm)
            with pytest.raises(OSError, match="registration rejected"):
                await timer.wait_until(time.perf_counter() + 60)
            assert timer._pending is None
    asyncio.run(scenario())


def test_tick_conversion_and_overflow():
    async def scenario():
        async with MacOSDeadlineTimer() as timer:
            assert timer._ticks(0.0) == 0
            expected = (1_000_000_000 * timer._denom + timer._numer - 1) // timer._numer
            assert timer._ticks(1.0) == expected
            with pytest.raises(OverflowError):
                timer._ticks(1e100)
    asyncio.run(scenario())
