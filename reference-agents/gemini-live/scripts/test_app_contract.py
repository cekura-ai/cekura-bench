import sys
import unittest
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import CallState, build_cekura_transcript_payload, serialize_tool_response, workflow_for
from google.genai import types


class TranscriptContractTests(unittest.TestCase):
    def test_sdk_tool_envelope_has_the_model_id_and_camel_case_wire_keys(self):
        envelope = serialize_tool_response([
            types.FunctionResponse(id="call_1", name="lookup_patient", response={"output": {"patient_id": "p_1003"}}),
        ])
        response = envelope["tool_response"]["functionResponses"][0]
        self.assertEqual(response["id"], "call_1")
        self.assertEqual(response["name"], "lookup_patient")
        self.assertEqual(response["response"]["output"]["patient_id"], "p_1003")

    def test_streamed_transcription_becomes_one_utterance(self):
        state = CallState(started_wall=datetime.now(timezone.utc))
        state.transcript_entry("bot", content="Your new appointment is")
        state.transcript_entry("bot", content="confirmed.")
        state.transcript_entry("bot", content="<no speech>")
        self.assertEqual(state.transcript, [{"role": "bot", "content": "Your new appointment is confirmed.", "start_time": 0.0, "end_time": 0.0}])

    def test_webhook_shape_keeps_tool_pairs_and_integer_run(self):
        state = CallState(
            workflow=workflow_for("19715717785"), session_id="our-session-id",
            caller_number="+16175550000", started_wall=datetime(2026, 9, 15, tzinfo=timezone.utc),
            cekura_run_id=3833696,
            transcript=[
                {"role": "user", "content": "hello", "start_time": 2.0, "end_time": 2.0},
                {"role": "function_call", "data": {"id": "f1", "name": "lookup_patient", "arguments": {"phone": "6175559210"}}, "start_time": 3.0, "end_time": 3.0},
                {"role": "function_call_result", "data": {"id": "f1", "name": "lookup_patient", "result": {"patient_id": "p_1003"}}, "start_time": 4.0, "end_time": 4.0},
            ],
        )
        payload = build_cekura_transcript_payload(state, datetime(2026, 9, 15, 0, 1, tzinfo=timezone.utc))
        self.assertEqual(payload["agent_id"], 23484)
        call = payload["calls"][0]
        self.assertEqual(call["id"], "our-session-id")
        self.assertIsInstance(call["run_id"], int)
        self.assertEqual([item["role"] for item in call["messages"]], ["user", "function_call", "function_call_result"])
        self.assertEqual(call["messages"][1]["data"]["id"], call["messages"][2]["data"]["id"])

    def test_run_id_is_omitted_when_not_safely_associated(self):
        state = CallState(workflow=workflow_for("19713912400"), started_wall=datetime.now(timezone.utc), transcript=[{"role": "bot", "content": "hi", "start_time": 0.0, "end_time": 0.0}])
        call = build_cekura_transcript_payload(state, datetime.now(timezone.utc))["calls"][0]
        self.assertNotIn("run_id", call)
        self.assertEqual(call["messages"][0]["role"], "bot")


if __name__ == "__main__":
    unittest.main()
