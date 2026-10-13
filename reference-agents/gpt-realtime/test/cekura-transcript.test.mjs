import assert from "node:assert/strict";
import test from "node:test";
import {
  buildCustomTranscriptPayload,
  findRunId,
  normalizePhoneNumber,
  phoneFromSipHeaders,
} from "../lib/cekura-transcript.mjs";

test("builds a Cekura custom-provider payload with sorted, paired entries", () => {
  const payload = buildCustomTranscriptPayload({
    sessionId: "openai:rtc_1",
    openaiCallId: "rtc_1",
    startedAt: "2026-08-15T23:00:00.000Z",
    endedAt: "2026-08-15T23:00:08.000Z",
    endedReason: "agent-hangup",
    callerNumber: "+17625550123",
    cekuraRunId: 12,
    entries: [
      { role: "function_call_result", content: "", data: { id: "c1", name: "lookup_patient", result: {} }, start_time: 3, end_time: 3, _sequence: 2 },
      { role: "function_call", content: "", data: { id: "c1", name: "lookup_patient", arguments: { phone: "2025550188" } }, start_time: 2, end_time: 2, _sequence: 1 },
      { role: "user", content: "I need a checkup", start_time: 1, end_time: 1, _sequence: 0 },
    ],
  }, 21644);
  assert.equal(payload.agent_id, 21644);
  assert.equal(payload.calls[0].run_id, 12);
  assert.deepEqual(payload.calls[0].messages.map((entry) => entry.role), ["user", "function_call", "function_call_result"]);
  assert.deepEqual(payload.calls[0].messages[1].data.arguments, { phone: "2025550188" });
});

test("matches an active run only by its distinct caller number", () => {
  const rows = [
    { id: 10, agent: 21644, status: "running", inbound_number: "+17625550123", agent_number: "+16205360171" },
    { id: 12, agent: 21644, status: "evaluating", inbound_number: "+17625550123", agent_number: "+16205360171" },
    { id: 11, agent: 21644, status: "completed", inbound_number: "+17625550124", agent_number: "+16205360171" },
  ];
  assert.equal(findRunId(rows, { agentId: 21644, callerNumber: "+1 (762) 555-0123", agentNumber: "+16205360171" }), 10);
  assert.equal(findRunId(rows, { agentId: 21644, callerNumber: "+17625550124", agentNumber: "+16205360171" }), null);
});

test("extracts a caller number from Twilio SIP identity headers", () => {
  assert.equal(phoneFromSipHeaders([{ name: "P-Asserted-Identity", value: "<sip:+17625550123@sip.twilio.com>" }]), "+17625550123");
  assert.equal(normalizePhoneNumber("+1 (762) 555-0123"), "17625550123");
});
