import asyncio
import unittest
from types import SimpleNamespace as NS

from probe_live_setup import drain_interaction, validate_lookup_continuation


def message(status=None, calls=None, turn_complete=False):
    return NS(
        go_away=None, tool_call_cancellation=None,
        tool_call=NS(function_calls=calls) if calls else None,
        server_content=NS(
            interaction_status=status, output_transcription=None,
            interrupted=False, turn_complete=turn_complete,
        ),
    )


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    def test_filler_and_system_error_do_not_pass(self):
        events = [
            {"stage": "mock_response_received", "name": "lookup_patient", "ms": 1,
             "result": {"upcoming_appointments": [{"provider": "Dr. Patel"}]}},
            {"stage": "output_transcription", "ms": 2,
             "text": "Let me check. The system is having trouble."},
        ]
        with self.assertRaisesRegex(RuntimeError, "no grounded"):
            validate_lookup_continuation(events)
        events.append({"stage": "output_transcription", "ms": 3, "text": "Your appointment is with Dr. Patel."})
        validate_lookup_continuation(events)

    async def test_tool_responds_before_idle_and_reader_survives_filler(self):
        responded = asyncio.Event()
        observed = []

        class Session:
            rounds = 0

            async def receive(self):
                self.rounds += 1
                if self.rounds == 1:
                    yield message("IN_PROGRESS", turn_complete=True)
                    return
                yield message(calls=[NS(id="one", name="lookup_patient")])
                await responded.wait()  # Server cannot finish before tool result.
                yield message("IDLE")

        async def respond(fc):
            observed.append(fc.id)
            responded.set()

        session = Session()
        await asyncio.wait_for(drain_interaction(session, respond, lambda *a, **k: None), 1)
        self.assertEqual(observed, ["one"])
        self.assertEqual(session.rounds, 2)

    async def test_worker_failure_propagates_without_hanging_reader(self):
        class Session:
            async def receive(self):
                yield message(calls=[NS(id="one", name="lookup_patient")])
                await asyncio.Event().wait()

        async def respond(fc):
            raise ValueError("mock failed")

        with self.assertRaises(ExceptionGroup) as error:
            await asyncio.wait_for(drain_interaction(Session(), respond, lambda *a, **k: None), 1)
        self.assertIsInstance(error.exception.exceptions[0], ValueError)

    async def test_timeout_awaits_tool_cleanup(self):
        cleaned = asyncio.Event()

        class Session:
            async def receive(self):
                yield message(calls=[NS(id="one", name="lookup_patient")])
                await asyncio.Event().wait()

        async def respond(fc):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(drain_interaction(Session(), respond, lambda *a, **k: None), .05)
        self.assertTrue(cleaned.is_set())


if __name__ == "__main__":
    unittest.main()
