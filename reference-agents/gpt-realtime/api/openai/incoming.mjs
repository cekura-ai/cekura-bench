import OpenAI from "openai";
import { getDeadline, waitUntil } from "@vercel/functions";
import WebSocket from "ws";
import { canonicalConfig, realtimeTools, realtimeWorkflowConfigs, requireWebhookSecret } from "../../lib/canonical.mjs";
import { normalizeCanonicalToolArguments } from "../../lib/canonical-tool-arguments.mjs";
import {
  addTranscriptEntry,
  parseObject,
  phoneFromSipHeaders,
  publishCekuraTranscript,
  resolveCekuraRunId,
} from "../../lib/cekura-transcript.mjs";

// Leave a 10-second buffer before Vercel Hobby's five-minute deadline for the
// custom-provider transcript publication. This is a guard, not the normal
// duration of a benchmark call.
const HOBBY_GUARD_MS = 290_000;
// Benchmark v1's authoritative mock rows live on the shared fixture catalog.
// Provider targets retain their own tool records for Cekura configuration, but
// the deployed runtime must always call this canonical source of truth.
const CEKURA_CANONICAL_MOCK_AGENT_ID = 20909;
const CEKURA_RUN_ASSOCIATION_ATTEMPTS = 12;
const CEKURA_RUN_ASSOCIATION_DELAY_MS = 500;
const DIAGNOSTIC_SINK_URL = process.env.DIAGNOSTIC_SINK_URL;
const CEKURA_AGENT_NUMBER = process.env.CEKURA_AGENT_NUMBER || "+16205360171";
// Set this when the Medicare DID is provisioned.  Live delegation is immutable
// after acceptance, so each benchmark lane needs a deterministic called-DID.
const CEKURA_MEDICARE_AGENT_NUMBER = process.env.CEKURA_MEDICARE_AGENT_NUMBER || "";
// A dedicated deployment can share an OpenAI project webhook subscription with
// the appointments worker. In that mode it must never attempt to accept a
// non-Medicare session: the appointments worker owns those calls.
const MEDICARE_ONLY = process.env.CEKURA_MEDICARE_ONLY === "true";
const APPOINTMENTS_ONLY = process.env.CEKURA_APPOINTMENTS_ONLY === "true";
// GPT-Live is the caller-facing, full-duplex voice model. The canonical v1
// workflow prompt and function schemas deliberately remain with the delegated
// Responses backend, which is where tool selection/execution belongs in Live.
const GPT_LIVE_MODEL = "gpt-live-1";
const GPT_LIVE_BACKEND_MODEL = "gpt-5.6-terra";
const nativeOpenAIHeaders = (headers = {}) => ({
  Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  ...headers,
});
// Inbound SIP sessions live in the project named by the dialed OpenAI SIP URI.
// Keep this separate from the delegated Responses project: it scopes only Live
// accept/attach/hangup lookups and leaves the canonical workflow untouched.
const liveControlHeaders = (headers = {}) => ({
  ...nativeOpenAIHeaders(),
  ...(process.env.OPENAI_LIVE_PROJECT_ID ? { "OpenAI-Project": process.env.OPENAI_LIVE_PROJECT_ID } : {}),
  ...headers,
});
const openAIHeaders = (headers = {}) => ({
  Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  ...(process.env.OPENAI_PROJECT_ID ? { "OpenAI-Project": process.env.OPENAI_PROJECT_ID } : {}),
  ...headers,
});
const END_CALL_TOOL = {
  type: "function",
  name: "end_call",
  description: "End the phone call as soon as the caller says they need no further help. In the same final response, give a warm sign-off and call this tool; do not wait for the caller to hang up first.",
  parameters: { type: "object", properties: {}, additionalProperties: false },
};

// Vercel's historical log view can omit info-level output.  These are deliberately
// error-level, structured diagnostic records for the temporary smoke-test worker.
// They never include authorization headers or webhook signatures.
async function diagnostic(callId, stage, details = {}) {
  const record = {
    diagnostic: "openai-realtime-sip",
    at: new Date().toISOString(),
    call_id: callId,
    stage,
    ...details,
  };
  console.error(JSON.stringify(record));
  if (DIAGNOSTIC_SINK_URL) {
    // SIP Live sessions must be accepted almost immediately. Diagnostics are
    // useful for smoke runs but must never sit on the critical accept path.
    void fetch(DIAGNOSTIC_SINK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record),
    }).catch((error) => console.error("Realtime diagnostic sink failed", error.message));
  }
}

// Keep the production log useful for latency triage. Raw Live audio deltas are
// large and obscure the ordering of accept, model, and tool events.
function summarizeLiveEvent(envelope) {
  const event = envelope?.type === "response.event" ? envelope.event : envelope;
  if (!event || typeof event !== "object") return event;
  const summary = { type: event.type };
  // Responses delegation nests backend lifecycle events in response.event.
  // The outer delegation ID is the only reliable way to associate those
  // events with the Live request that caused them, so retain it in logs.
  if (envelope?.type === "response.event") summary.delegation_id = envelope.delegation_id || null;
  if (event.response?.id) {
    summary.response = {
      id: event.response.id,
      status: event.response.status || null,
      error: event.response.error || null,
      incomplete_details: event.response.incomplete_details || null,
      output_count: Array.isArray(event.response.output) ? event.response.output.length : null,
    };
  }
  if (event.delegation?.id) summary.delegation = { id: event.delegation.id, target: event.delegation.target || null, response_id: event.response_id || null };
  if (typeof event.delta === "string" && !String(event.type || "").includes("audio")) summary.delta = event.delta;
  if (event.item) {
    summary.item = { type: event.item.type, id: event.item.id || null, call_id: event.item.call_id || null, name: event.item.name || null };
    if (event.item.type === "function_call") summary.item.arguments = event.item.arguments;
    if (event.item.type === "message") {
      summary.item.status = event.item.status || null;
      summary.item.text = completedMessageText(event.item) || null;
    }
  }
  if (event.call_id) summary.call_id = event.call_id;
  if (event.name) summary.name = event.name;
  if (event.audio || (typeof event.delta === "string" && String(event.type || "").includes("audio"))) {
    const audio = event.audio || event.delta || "";
    summary.audio_bytes = Buffer.byteLength(audio, "utf8");
  }
  return summary;
}

function completedMessageText(item) {
  if (!item || typeof item !== "object") return "";
  if (typeof item.text === "string") return item.text;
  if (!Array.isArray(item.content)) return "";
  return item.content.map((part) => {
    if (typeof part?.text === "string") return part.text;
    if (typeof part?.refusal === "string") return part.refusal;
    if (typeof part?.output_text === "string") return part.output_text;
    return "";
  }).filter(Boolean).join("");
}

function shouldLogLiveEvent(event) {
  // Vercel retains only a bounded number of records per invocation. Audio and
  // transcript fragments arrive several times per second; logging each one
  // hid the terminal tool/close events that are the point of this worker.
  return ![
    "session.input_audio.append",
    "session.output_audio.delta",
    "session.input_transcript.delta",
    "session.output_transcript.delta",
    "response.output_text.delta",
  ].includes(event?.type);
}

function responseIdFor(event) {
  return event?.response?.id || event?.response_id || null;
}

function responseStatusFor(event) {
  return event?.response?.status || event?.status || null;
}

function responseFailureFor(event) {
  return event?.response?.error || event?.error || event?.response?.incomplete_details || null;
}

function timing(session, timingStage, details = {}) {
  return diagnostic(session.openaiCallId, "live_timing", {
    timing_stage: timingStage,
    elapsed_ms: Date.now() - session.startedAtMs,
    ...details,
  });
}

async function rawBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

function normalizedPhone(value) {
  const digits = String(value || "").replace(/\D/g, "");
  if (digits.length === 11 && digits.startsWith("1")) return `+${digits}`;
  return digits.length === 10 ? `+1${digits}` : "";
}

function calledNumberFromSipHeaders(headers = []) {
  // Twilio preserves the dialed DID in Diversion when it forwards the PSTN
  // call to the OpenAI project SIP URI. Fall back to To for direct calls.
  const explicitDid = headers.find((header) => String(header.name || "").toLowerCase() === "x-cekura-called-number");
  if (explicitDid) return normalizedPhone(explicitDid.value);
  const preferred = headers.find((header) => String(header.name || "").toLowerCase() === "diversion")
    || headers.find((header) => String(header.name || "").toLowerCase() === "to");
  const match = String(preferred?.value || "").match(/sip:([^@;>]+)/i);
  return normalizedPhone(match?.[1] || "");
}

function twilioCallSidFromSipHeaders(headers = []) {
  return headers.find((header) => String(header.name || "").toLowerCase() === "x-twilio-callsid")?.value || null;
}

function twilioParentCallSidFromSipHeaders(headers = []) {
  return headers.find((header) => String(header.name || "").toLowerCase() === "x-cekura-parent-call-sid")?.value || null;
}

function workflowForCalledNumber(calledNumber, workflows) {
  const medicare = workflows.find((workflow) => workflow.key === "medicare");
  if (CEKURA_MEDICARE_AGENT_NUMBER && normalizedPhone(CEKURA_MEDICARE_AGENT_NUMBER) === calledNumber && medicare) return medicare;
  return workflows.find((workflow) => workflow.key === "appointments") || workflows[0];
}

function openSocketOnce(callId, attempt) {
  return new Promise((resolve, reject) => {
    void diagnostic(callId, "sideband_connecting", { attempt });
    const socket = new WebSocket(`wss://api.openai.com/v1/realtime?call_id=${encodeURIComponent(callId)}`, {
      headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
    });
    socket.once("open", () => {
      void diagnostic(callId, "sideband_open", { attempt });
      resolve(socket);
    });
    socket.once("error", (error) => {
      void diagnostic(callId, "sideband_connect_error", { attempt, message: error.message });
      reject(error);
    });
    socket.once("unexpected-response", (_request, response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.once("end", () => {
        const message = `Unexpected server response: ${response.statusCode}`;
        void diagnostic(callId, "sideband_unexpected_response", {
          attempt,
          status: response.statusCode,
          headers: response.headers,
          body: body.slice(0, 2_000),
        });
        reject(new Error(message));
      });
    });
  });
}

function openLiveSocketOnce(sessionId, attempt) {
  return new Promise((resolve, reject) => {
    void diagnostic(sessionId, "live_sideband_connecting", { attempt });
    const socket = new WebSocket(`wss://api.openai.com/v1/live/sessions/${encodeURIComponent(sessionId)}/attach`, {
      headers: liveControlHeaders(),
    });
    socket.once("open", () => { void diagnostic(sessionId, "live_sideband_open", { attempt }); resolve(socket); });
    socket.once("error", reject);
  });
}

async function openLiveSocket(sessionId) {
  const delays = [0, 250, 500, 1_000, 2_000];
  let lastError;
  for (let index = 0; index < delays.length; index += 1) {
    if (delays[index]) await new Promise((resolve) => setTimeout(resolve, delays[index]));
    try { return await openLiveSocketOnce(sessionId, index + 1); } catch (error) { lastError = error; }
  }
  throw lastError;
}

async function openSocket(callId) {
  // The call-accept endpoint returns while the SIP leg is being established.
  // A sideband connection can briefly return 404 until that session exists.
  const retryDelaysMs = [0, 250, 500, 1_000, 2_000];
  let lastError;
  for (let attempt = 0; attempt < retryDelaysMs.length; attempt += 1) {
    if (retryDelaysMs[attempt]) await new Promise((resolve) => setTimeout(resolve, retryDelaysMs[attempt]));
    try {
      return await openSocketOnce(callId, attempt + 1);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}

function send(socket, event) {
  socket.send(JSON.stringify(event));
}

async function invokeCanonicalTool(workflow, name, argumentsObject) {
  if (!workflow.tools.some((tool) => tool.name === name)) {
    throw new Error(`Rejected unconfigured tool: ${name}`);
  }

  const result = await fetch(`https://new-prod.cekura.ai/test_framework/v1/aiagents/${CEKURA_CANONICAL_MOCK_AGENT_ID}/tool/${encodeURIComponent(name)}/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(argumentsObject),
  });
  const body = await result.text();
  if (!result.ok) throw new Error(`Canonical ${name} tool failed: ${result.status}`);
  return body;
}

async function hangup(callId) {
  await fetch(`https://api.openai.com/v1/realtime/calls/${encodeURIComponent(callId)}/hangup`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.OPENAI_API_KEY}` },
  });
}

async function hangupLive(sessionId) {
  let lastError;
  for (const delayMs of [0, 250, 750]) {
    if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
    try {
      const response = await fetch(`https://api.openai.com/v1/live/sessions/${encodeURIComponent(sessionId)}/hangup`, {
        method: "POST", headers: liveControlHeaders(),
      });
      const body = await response.text();
      if (response.ok) {
        return {
          status: response.status,
          requestId: response.headers.get("x-request-id") || response.headers.get("request-id") || null,
          body: body || null,
        };
      }
      // A client error is definitive; retries are only meaningful for a
      // transient control-plane failure.
      if (response.status < 500) throw new Error(`OpenAI Live hangup failed: ${response.status}: ${body}`);
      lastError = new Error(`OpenAI Live hangup failed: ${response.status}: ${body}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("OpenAI Live hangup failed without a response");
}

async function terminateLive(session, source) {
  if (session.liveHangupRequestedAt) return;
  session.liveHangupRequestedAt = Date.now();
  await timing(session, "live_hangup_requested", { source, transport: "rest" });
  try {
    const result = await hangupLive(session.openaiCallId);
    await timing(session, "live_hangup_accepted", {
      source,
      status: result.status,
      request_id: result.requestId,
      response_body: result.body,
    });
    return result;
  } catch (error) {
    await timing(session, "live_hangup_failed", { source, message: error.message });
    throw error;
  }
}

async function inspectTwilioSipLeg(session, source) {
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;
  if (!session.twilioCallSid || !accountSid || !authToken) {
    await timing(session, "telephone_disconnect_unverified", {
      source,
      has_call_sid: Boolean(session.twilioCallSid),
      credentials_configured: Boolean(accountSid && authToken),
    });
    return { skipped: true };
  }
  const authorization = Buffer.from(`${accountSid}:${authToken}`).toString("base64");
  const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(accountSid)}/Calls/${encodeURIComponent(session.twilioCallSid)}.json`, {
    headers: { Authorization: `Basic ${authorization}` },
  });
  const body = await response.text();
  if (!response.ok) {
    await timing(session, "telephone_disconnect_unverified", {
      source,
      status: response.status,
      twilio_call_sid: session.twilioCallSid,
    });
    return;
  }
  const call = JSON.parse(body);
  const ended = Boolean(call.end_time) || ["completed", "busy", "failed", "canceled", "no-answer"].includes(call.status);
  await timing(session, ended ? "telephone_disconnected" : "telephone_disconnect_unverified", {
    source,
    twilio_call_sid: session.twilioCallSid,
    twilio_status: call.status,
    twilio_end_time: call.end_time || null,
  });
}

async function terminateTwilioProgrammableLeg(session, source) {
  if (!session.twilioParentCallSid) return;
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;
  if (!accountSid || !authToken) {
    await timing(session, "telephone_hangup_unavailable", { source, reason: "missing_twilio_credentials" });
    return;
  }
  const authorization = Buffer.from(`${accountSid}:${authToken}`).toString("base64");
  await timing(session, "telephone_hangup_requested", { source, twilio_parent_call_sid: session.twilioParentCallSid });
  const response = await fetch(`https://api.twilio.com/2010-04-01/Accounts/${encodeURIComponent(accountSid)}/Calls/${encodeURIComponent(session.twilioParentCallSid)}.json`, {
    method: "POST",
    headers: { Authorization: `Basic ${authorization}`, "Content-Type": "application/x-www-form-urlencoded" },
    body: "Status=completed",
  });
  const body = await response.text();
  if (!response.ok) {
    await timing(session, "telephone_hangup_failed", { source, status: response.status, twilio_parent_call_sid: session.twilioParentCallSid, response_body: body || null });
    return;
  }
  const call = JSON.parse(body);
  await timing(session, "telephone_hangup_accepted", {
    source, status: response.status, twilio_parent_call_sid: session.twilioParentCallSid,
    twilio_status: call.status || null, twilio_end_time: call.end_time || null,
  });
}

function liveSessionConfig(workflow) {
  const backendCapabilities = workflow.key === "appointments"
    ? "- Appointments: look up a patient, check availability, create or cancel an appointment.\n- Closure: end the call after the caller has clearly finished."
    : "- Medicare: record permissions and qualification, route the caller, and create a handoff summary.\n- Closure: end the call after the caller has clearly finished.";
  const voiceInstructions = [
    "You are Ava, the caller-facing voice for this phone conversation.",
    `Immediately say this greeting exactly, then listen: ${workflow.firstMessage}`,
    "Speak only caller-facing information; never expose tool names or internal reasoning.",
    "Interruption policy: Stop speaking when the caller interrupts. Listen to what they say.",
    "Delegation policy:\nBackend tools:\n" + backendCapabilities + "\nDelegate to the backend when:\n- The caller needs one of the backend capabilities above.\n- The caller gives consent, qualification, coverage, election-window, contact, or routing information that must be recorded or changes a record.\n- You have said or are about to say that you will record, check, route, transfer, or otherwise complete backend work: delegate now before any such caller-facing statement.\n- The caller clearly says they have no further questions, are finished, says goodbye, or asks to end the call. Delegate closure before speaking any final sign-off.\nDo not delegate to the backend when:\n- The caller is merely greeting you, making a brief acknowledgment, or you need a brief clarification.\nDelegate before giving an answer that depends on backend work. Do not guess a backend result while waiting, and never repeat a checking/working status when no backend delegation is pending.",
  ].join("\n");
  return {
    type: "live",
    model: GPT_LIVE_MODEL,
    instructions: voiceInstructions,
    // Direct SIP negotiates the audio transport; voice remains an explicit
    // Live-session choice at acceptance.
    audio: { output: { voice: "marin" } },
    delegation: {
      type: "responses",
      responses: {
        model: GPT_LIVE_BACKEND_MODEL,
        // Byte-identical source: docs/benchmark-v1/agent-defs/<workflow>/system-prompt.txt.
        instructions: workflow.prompt,
        tools: [...realtimeTools(workflow.tools), END_CALL_TOOL],
        parallel_tool_calls: false,
      },
    },
  };
}

async function acceptLive(sessionId, workflow) {
  // GPT-Live requires `type: live` plus its delegation mode at acceptance.
  // Responses delegation is immutable after a session starts, so this cannot
  // be deferred to a sideband session.update.
  const startedAt = Date.now();
  const response = await fetch(`https://api.openai.com/v1/live/sessions/${encodeURIComponent(sessionId)}/accept`, {
    method: "POST",
    // Diagnostic probe: a project-scoped key already selects its project.
    // Do not override that selection for the SIP accept lookup.
    headers: liveControlHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ session: liveSessionConfig(workflow) }),
  });
  const body = await response.text();
  await diagnostic(sessionId, "live_accept_response", { status: response.status, elapsed_ms: Date.now() - startedAt, body });
  if (!response.ok) throw new Error(`OpenAI Live accept failed: ${response.status}: ${body}`);
}

async function finalizeCekuraTranscript(callId, session) {
  if (session.publishPromise) return session.publishPromise;
  session.endedAt ||= new Date().toISOString();
  session.publishPromise = (async () => {
    try {
      // Cekura creates a different_numbers run asynchronously after the PSTN
      // leg begins.  Do not publish an unbound transcript merely because the
      // initial lookup raced that creation.
      if (!session.cekuraRunId && session.cekuraRunAssociation) {
        await session.cekuraRunAssociation;
      }
      const result = await publishCekuraTranscript(session, {
        apiKey: process.env.CEKURA_API_KEY,
        agentId: session.workflow.agentId,
      });
      await diagnostic(callId, "cekura_transcript_published", result);
    } catch (error) {
      await diagnostic(callId, "cekura_transcript_publish_failed", { message: error.message });
    }
  })();
  return session.publishPromise;
}

async function selectWorkflow(callId, session, workflows) {
  for (let attempt = 1; attempt <= CEKURA_RUN_ASSOCIATION_ATTEMPTS; attempt += 1) {
    const matches = await Promise.all(workflows.map(async (workflow) => ({
      workflow,
      runId: await resolveCekuraRunId({
        apiKey: process.env.CEKURA_API_KEY,
        agentId: workflow.agentId,
        callerNumber: session.callerNumber,
        // The inbound SIP number is the only reliable lane discriminator.
        // Using the appointments DID here made every dedicated-Medicare call
        // miss its exact run and fall back to Cekura's unsafe content matcher.
        agentNumber: session.calledNumber || CEKURA_AGENT_NUMBER,
      }),
    })));
    const match = matches.find(({ runId }) => runId);
    if (match) {
      session.workflow = match.workflow;
      session.cekuraRunId = match.runId;
      await diagnostic(callId, "cekura_run_associated", {
        caller_number: session.callerNumber,
        run_id: match.runId,
        workflow: match.workflow.key,
        attempt,
      });
      return match.workflow;
    }
    if (attempt < CEKURA_RUN_ASSOCIATION_ATTEMPTS) {
      await new Promise((resolve) => setTimeout(resolve, CEKURA_RUN_ASSOCIATION_DELAY_MS));
    }
  }
  // A direct/manual SIP call has no Cekura run to disambiguate it. Keep the
  // pre-existing Appointment behavior rather than accepting an unconfigured call.
  session.workflow = workflows.find((workflow) => workflow.key === "appointments") || workflows[0];
  await diagnostic(callId, "cekura_run_association_missing", {
    caller_number: session.callerNumber,
    workflow: session.workflow.key,
  });
  return session.workflow;
}

async function sidebandSession(callId, workflows, session) {
  const socket = await openSocket(callId);
  const deadline = getDeadline?.();
  const deadlineGuard = deadline ? Math.max(1_000, deadline.getTime() - Date.now() - 10_000) : HOBBY_GUARD_MS;
  const guardMs = Math.min(HOBBY_GUARD_MS, deadlineGuard);
  let closingForDuration = false;

  const guard = setTimeout(() => {
    closingForDuration = true;
    session.endedReason = "duration-guard";
    diagnostic(callId, "duration_guard_fired", { guard_ms: guardMs });
    void finalizeCekuraTranscript(callId, session);
    socket.close(1000, "benchmark call duration limit");
    void hangup(callId).catch((error) => console.error("OpenAI SIP hangup failed", error));
  }, guardMs);

  try {
    // The single SIP URI cannot encode a workflow. Resolve the Cekura
    // different_numbers caller identity before configuring the Realtime session.
    // This happens before any model audio is generated, so no appointment/Medicare
    // prompt or tool schema can leak across benchmark lanes.
    const workflow = await selectWorkflow(callId, session, workflows);
    send(socket, {
      type: "session.update",
      session: {
        type: "realtime",
        instructions: workflow.prompt,
        tools: [...realtimeTools(workflow.tools), END_CALL_TOOL],
      },
    });
    // The canonical prompt specifies the greeting; this starts the first spoken turn.
    send(socket, { type: "response.create", response: { instructions: workflow.firstMessage } });
    diagnostic(callId, "greeting_requested");

    await new Promise((resolve, reject) => {
      socket.on("message", async (payload) => {
        try {
          const event = JSON.parse(payload.toString());
          diagnostic(callId, "realtime_event", { event });
          if (event.type === "conversation.item.input_audio_transcription.completed") {
            if (event.transcript) addTranscriptEntry(session, { role: "user", content: event.transcript });
            return;
          }
          if (event.type === "response.output_audio_transcript.done") {
            if (event.transcript) addTranscriptEntry(session, { role: "bot", content: event.transcript });
            return;
          }
          if (event.type !== "response.function_call_arguments.done") return;

          if (event.name === "end_call") {
            session.endedReason = "agent-hangup";
            addTranscriptEntry(session, {
              role: "function_call",
              content: "",
              data: { id: event.call_id, name: event.name, arguments: parseObject(event.arguments) },
            });
            addTranscriptEntry(session, {
              role: "function_call_result",
              content: "",
              data: { id: event.call_id, name: event.name, result: { ended: true } },
            });
            send(socket, {
              type: "conversation.item.create",
              item: { type: "function_call_output", call_id: event.call_id, output: JSON.stringify({ ended: true }) },
            });
            // A SIP hangup can leave the Realtime sideband open briefly. Start the
            // non-critical Cekura publish now, while the complete final turn is in
            // memory; socket close remains an idempotent fallback below.
            const publishing = finalizeCekuraTranscript(callId, session);
            await hangup(callId);
            await publishing;
            return;
          }

          let output;
          const rawArgumentsObject = parseObject(event.arguments);
          const argumentsObject = normalizeCanonicalToolArguments(event.name, rawArgumentsObject);
          if (JSON.stringify(rawArgumentsObject) !== JSON.stringify(argumentsObject)) {
            await diagnostic(callId, "canonical_tool_arguments_normalized", { tool_name: event.name, raw_arguments: rawArgumentsObject, executed_arguments: argumentsObject });
          }
          addTranscriptEntry(session, {
            role: "function_call",
            content: "",
            data: { id: event.call_id, name: event.name, arguments: argumentsObject },
          });
          try {
            output = await invokeCanonicalTool(session.workflow, event.name, argumentsObject);
          } catch (error) {
            output = JSON.stringify({ error: error.message });
          }
          addTranscriptEntry(session, {
            role: "function_call_result",
            content: "",
            data: { id: event.call_id, name: event.name, result: parseObject(output) },
          });

          send(socket, {
            type: "conversation.item.create",
            item: { type: "function_call_output", call_id: event.call_id, output },
          });
          send(socket, { type: "response.create" });
        } catch (error) {
          reject(error);
        }
      });
      socket.once("close", (code, reason) => {
        session.endedReason ||= "caller-hangup";
        diagnostic(callId, "sideband_closed", { code, reason: reason.toString() });
        resolve();
      });
      socket.once("error", (error) => {
        diagnostic(callId, "sideband_runtime_error", { message: error.message });
        reject(error);
      });
    });
  } finally {
    clearTimeout(guard);
    if (!closingForDuration && socket.readyState === WebSocket.OPEN) socket.close();
    await finalizeCekuraTranscript(callId, session);
  }
}

async function liveSidebandSession(sessionId, session) {
  const socket = await openLiveSocket(sessionId);
  await timing(session, "sideband_attached");
  const deadline = getDeadline?.();
  const guardMs = Math.min(HOBBY_GUARD_MS, deadline ? Math.max(1_000, deadline.getTime() - Date.now() - 10_000) : HOBBY_GUARD_MS);
  // GPT-Live documents transcript deltas as the authoritative transcript
  // interface; unlike Realtime, it deliberately emits no turn-completed event.
  // Group close-together fragments into stable provider transcript messages.
  const input = { active: null }, output = { active: null };
  const appendTranscriptDelta = (bucket, role, event) => {
    const fragment = event.delta || "";
    if (!fragment) return;
    const start = Number(event.start_ms || 0) / 1_000;
    const end = Number(event.end_ms || event.start_ms || 0) / 1_000;
    if (!bucket.active || (Number(event.start_ms || 0) - bucket.active._lastEndMs) > 1_200) {
      bucket.active = {
        role,
        content: fragment,
        _sequence: session.entries.length,
        start_time: start,
        end_time: end,
        _lastEndMs: Number(event.end_ms || event.start_ms || 0),
      };
      session.entries.push(bucket.active);
    } else {
      bucket.active.content += fragment;
      bucket.active.end_time = end;
      bucket.active._lastEndMs = Number(event.end_ms || event.start_ms || 0);
    }
  };
  const terminalCallerSignal = /\b(?:that(?:'s| is) (?:all|everything)|nothing (?:else|further)|i(?:'m| am) all set|goodbye|bye)\b/i;
  // Responses delegation events are nested in response.event. Keep a local
  // ledger because response.completed deliberately has an empty output list;
  // completed function calls must instead be collected from prior
  // response.output_item.done events (per the Live protocol).
  const delegatedResponses = new Map();
  const workPromise = /\b(?:record|save|check|look(?:ing)? up|route|transfer|hand ?off|submit|process|complete)\b/i;
  const ensureDelegatedResponse = (envelope, event) => {
    const delegationId = envelope?.type === "response.event" ? envelope.delegation_id : null;
    const responseId = responseIdFor(event);
    if (!delegationId && !responseId) return null;
    const key = `${delegationId || "unscoped"}:${responseId || "pending"}`;
    if (!delegatedResponses.has(key)) {
      delegatedResponses.set(key, {
        key, delegationId, responseId, createdAtMs: Date.now(), functionCalls: [], text: "", repairDirected: false,
      });
    }
    const state = delegatedResponses.get(key);
    if (delegationId && !state.delegationId) state.delegationId = delegationId;
    if (responseId && !state.responseId) state.responseId = responseId;
    return state;
  };
  let closureDelegationDirected = false;
  let closureDelegationFollowupRequested = false;
  const guard = setTimeout(() => {
    session.endedReason = "duration-guard";
    void finalizeCekuraTranscript(sessionId, session);
    void terminateLive(session, "duration_guard").catch((error) => console.error("OpenAI Live SIP hangup failed", error));
  }, guardMs);
  try {
    await new Promise((resolve, reject) => {
      socket.on("message", async (payload) => {
        try {
          const envelope = JSON.parse(payload.toString());
          const event = envelope.type === "response.event" ? envelope.event : envelope;
          if (shouldLogLiveEvent(event)) {
            await diagnostic(sessionId, "live_event", { event: summarizeLiveEvent(envelope) });
          }
          if (event.type === "session.delegation.created") {
            await timing(session, "delegation_created", {
              delegation_id: event.delegation?.id || null,
              target: event.delegation?.target || null,
              response_id: event.response_id || null,
              offset_ms: event.offset_ms ?? null,
            });
          }
          const delegatedResponse = ensureDelegatedResponse(envelope, event);
          if (delegatedResponse && event.type === "response.created") {
            await timing(session, "delegated_response_created", {
              delegation_id: delegatedResponse.delegationId,
              response_id: delegatedResponse.responseId,
            });
          }
          if (delegatedResponse && event.type === "response.output_text.delta" && typeof event.delta === "string") {
            delegatedResponse.text += event.delta;
          }
          if (delegatedResponse && event.type === "response.output_item.done" && event.item?.type === "message") {
            const completedText = completedMessageText(event.item);
            if (completedText) delegatedResponse.text = completedText;
          }
          if (delegatedResponse && event.type === "response.output_item.done" && event.item?.type === "function_call") {
            delegatedResponse.functionCalls.push({ call_id: event.item.call_id || null, name: event.item.name || null });
          }
          if (delegatedResponse && event.type === "response.completed") {
            const status = responseStatusFor(event);
            const failure = responseFailureFor(event);
            await timing(session, "delegated_response_completed", {
              delegation_id: delegatedResponse.delegationId,
              response_id: delegatedResponse.responseId,
              status,
              failure,
              pending_function_calls: delegatedResponse.functionCalls,
              backend_text: delegatedResponse.text || null,
              elapsed_ms: Date.now() - delegatedResponse.createdAtMs,
            });
            // A normal tool-free backend answer is legitimate. This guard is
            // intentionally narrower: when the backend promises to perform
            // work yet emitted no call, steer only the *voice* model away from
            // repeating an untrue checking status on the next turn. It never
            // fabricates a tool call, arguments, or result.
            if (!delegatedResponse.functionCalls.length && delegatedResponse.text && workPromise.test(delegatedResponse.text) && !delegatedResponse.repairDirected) {
              delegatedResponse.repairDirected = true;
              send(socket, {
                type: "session.thinking.append",
                event_id: `unfinished_backend_${Date.now()}`,
                delegation_id: null,
                content: "The backend completed without invoking a tool. Do not tell the caller work is being checked, recorded, routed, or completed. If workflow work remains, delegate it before speaking about its result.",
              });
              await timing(session, "delegation_completed_without_tool", {
                delegation_id: delegatedResponse.delegationId,
                response_id: delegatedResponse.responseId,
                backend_text: delegatedResponse.text,
              });
            }
          }
          if (event.type === "session.closed") {
            const reason = event.reason || event.session?.reason || "unknown";
            session.liveClosedAt = Date.now();
            session.liveCloseReason = reason;
            // Do not call this an agent hangup until Live has confirmed that
            // the close request actually finalized.
            if (reason === "close_requested") session.endedReason = "agent-hangup";
            await timing(session, "live_session_closed", {
              reason,
              close_wait_ms: session.liveHangupRequestedAt ? session.liveClosedAt - session.liveHangupRequestedAt : null,
            });
            return;
          }
          if (event.type === "session.input_transcript.delta") {
            if (!session.lastInputAtMs) await timing(session, "first_input_transcript");
            session.lastInputAtMs = Date.now();
            appendTranscriptDelta(input, "user", event);
            const callerText = input.active?.content || "";
            if (!closureDelegationDirected && terminalCallerSignal.test(callerText)) {
              closureDelegationDirected = true;
              // GPT-Live owns the decision to delegate. Steer that decision
              // explicitly for a terminal caller turn; the backend still owns
              // the end_call function and the worker still only hangs up after
              // that function is emitted.
              send(socket, {
                type: "session.instructions.append",
                event_id: `closure_delegate_${Date.now()}`,
                delegation_id: null,
                content: "The caller has explicitly finished the conversation. Immediately delegate closure to the backend now. Do not say a goodbye or any closing language yourself. Wait for the backend to invoke end_call.",
              });
              await timing(session, "closure_delegation_directive_sent");
            }
            return;
          }
          if (event.type === "session.output_transcript.delta") {
            if (session.awaitingOutputSinceMs) {
              await timing(session, "first_output_after_continuation", { wait_ms: Date.now() - session.awaitingOutputSinceMs });
              session.awaitingOutputSinceMs = null;
            }
            appendTranscriptDelta(output, "bot", event);
            return;
          }
          // An instructions.append affects subsequent model work, but does not
          // itself create a turn.  The caller's terminal utterance may already
          // have started a reply before the append arrived.  Once that reply
          // completes, request one follow-up turn so Live can delegate closure
          // to the unchanged backend end_call tool.
          if (event.type === "response.completed" && closureDelegationDirected && !closureDelegationFollowupRequested) {
            closureDelegationFollowupRequested = true;
            send(socket, { type: "response.create" });
            await timing(session, "closure_delegation_followup_requested");
            return;
          }
          if (event.type !== "response.output_item.done" || event.item?.type !== "function_call") return;
          const { call_id: callId, name, arguments: rawArguments } = event.item;
          if (!callId || !name) throw new Error("Live function call missing id or name");
          const rawArgumentsObject = parseObject(rawArguments);
          const argumentsObject = normalizeCanonicalToolArguments(name, rawArgumentsObject);
          if (JSON.stringify(rawArgumentsObject) !== JSON.stringify(argumentsObject)) {
            await diagnostic(sessionId, "canonical_tool_arguments_normalized", { tool_name: name, tool_call_id: callId, raw_arguments: rawArgumentsObject, executed_arguments: argumentsObject });
          }
          await timing(session, "tool_requested", {
            tool_name: name,
            tool_call_id: callId,
            raw_arguments_json: rawArguments,
            raw_arguments: rawArgumentsObject,
            executed_arguments: argumentsObject,
          });
          addTranscriptEntry(session, { role: "function_call", content: "", data: { id: callId, name, arguments: argumentsObject } });
          let result;
          if (name === "end_call") {
            result = JSON.stringify({ ended: true });
            session.endedReason = "agent-hangup";
          } else {
            const toolStartedAt = Date.now();
            try { result = await invokeCanonicalTool(session.workflow, name, argumentsObject); }
            catch (error) { result = JSON.stringify({ error: error.message }); }
            await timing(session, "tool_result_ready", { tool_name: name, tool_call_id: callId, tool_ms: Date.now() - toolStartedAt });
          }
          addTranscriptEntry(session, { role: "function_call_result", content: "", data: { id: callId, name, result: parseObject(result) } });
          await timing(session, "tool_output_submitting", {
            tool_name: name,
            tool_call_id: callId,
            raw_arguments_json: rawArguments,
            raw_arguments: rawArgumentsObject,
            executed_arguments: argumentsObject,
          });
          send(socket, { type: "response.item.create", event_id: `tool_${callId}`, item: { type: "function_call_output", call_id: callId, output: result } });
          if (name === "end_call") {
            // Use only the documented Live REST hangup.  Do not race it with
            // a sideband session.close or an unsupported attempt to control
            // an Elastic SIP Trunk leg through Twilio's Calls API.
            await terminateLive(session, "model_end_call");
            await terminateTwilioProgrammableLeg(session, "model_end_call");
            void inspectTwilioSipLeg(session, "post_live_hangup");
            // The publish remains deliberately non-blocking with respect to
            // session.closed so transcripts are not lost on an upstream
            // finalization fault. The resulting logs now distinguish a
            // requested close from a confirmed one.
            await finalizeCekuraTranscript(sessionId, session);
          } else {
            session.awaitingOutputSinceMs = Date.now();
            await timing(session, "model_continuation_requested", { tool_name: name, tool_call_id: callId });
            send(socket, { type: "response.create", event_id: `continue_${callId}` });
          }
        } catch (error) { reject(error); }
      });
      socket.once("close", (code, reason) => {
        session.endedReason ||= "caller-hangup";
        void timing(session, "sideband_closed", { code, reason: reason.toString() });
        resolve();
      });
      socket.once("error", (error) => {
        void timing(session, "sideband_error", { message: error.message });
        reject(error);
      });
    });
  } finally {
    clearTimeout(guard);
    for (const entry of session.entries) delete entry._lastEndMs;
    if (socket.readyState === WebSocket.OPEN) socket.close();
    await finalizeCekuraTranscript(sessionId, session);
  }
}

// This endpoint accepts the SIP invite quickly, then Vercel keeps the sideband
// task alive with waitUntil(). Hobby functions cap this benchmark worker at 5m.
export default async function handler(request, response) {
  if (request.method !== "POST") return response.status(405).json({ error: "POST required" });

  try {
    const config = canonicalConfig();
    const workflows = realtimeWorkflowConfigs();
    const webhookSecret = requireWebhookSecret();
    const body = await rawBody(request);
    await diagnostic("unknown", "webhook_raw_received", {
      content_length: body.length,
      webhook_id: request.headers["webhook-id"] || null,
    });
    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
      webhookSecret,
    });
    const event = await client.webhooks.unwrap(body, request.headers);
    // GPT-Live SIP uses a session ID and Live API endpoint. Keep the legacy
    // Realtime branch below during the webhook-subscription cutover so retries
    // from the old event type remain harmless.
    if (event.type === "live.transport.incoming" || event.type === "live.call.incoming") {
      const sessionId = event.data.session_id;
      if (!sessionId || (event.data.type && event.data.type !== "sip")) return response.status(400).json({ accepted: false, error: "Expected SIP Live session" });
      const session = {
        sessionId: `openai-live:${sessionId}`,
        openaiCallId: sessionId,
        callerNumber: phoneFromSipHeaders(event.data.sip_headers),
        calledNumber: calledNumberFromSipHeaders(event.data.sip_headers),
        twilioCallSid: twilioCallSidFromSipHeaders(event.data.sip_headers),
        twilioParentCallSid: twilioParentCallSidFromSipHeaders(event.data.sip_headers),
        startedAt: new Date().toISOString(), startedAtMs: Date.now(), entries: [],
      };
      await diagnostic(sessionId, "live_webhook_received", { event });
      // A dedicated Medicare DID, once provisioned, is deterministic and can
      // be accepted immediately. Until then, retain the prior shared-DID
      // benchmark routing: choose the only active Cekura run before accepting.
      // Native benchmark-key context makes that accept fast enough; this was
      // previously broken by the incorrect OpenAI-Project override.
      const dedicatedMedicare = CEKURA_MEDICARE_AGENT_NUMBER
        && normalizedPhone(CEKURA_MEDICARE_AGENT_NUMBER) === session.calledNumber;
      // The Programmable Voice bridge preserves the appointments DID in a SIP
      // header. Treat it as deterministic too: Live's acceptance window is
      // shorter than a paginated Cekura run lookup, which happens afterward.
      const dedicatedAppointments = APPOINTMENTS_ONLY
        && normalizedPhone(CEKURA_AGENT_NUMBER) === session.calledNumber;
      const dedicatedWorkflow = dedicatedMedicare || dedicatedAppointments
        ? workflowForCalledNumber(session.calledNumber, workflows)
        : null;
      if (MEDICARE_ONLY && !dedicatedWorkflow) {
        await diagnostic(sessionId, "live_session_ignored_non_medicare_did", {
          called_number: session.calledNumber || null,
        });
        return response.status(200).end();
      }
      if (APPOINTMENTS_ONLY && dedicatedMedicare) {
        await diagnostic(sessionId, "live_session_ignored_medicare_did", {
          called_number: session.calledNumber || null,
        });
        return response.status(200).end();
      }
      const workflow = dedicatedWorkflow || await selectWorkflow(sessionId, session, workflows);
      if (dedicatedWorkflow) {
        session.workflow = workflow;
      }
      await acceptLive(sessionId, workflow);
      // Acceptance is latency-critical. Associate after it so the outbound
      // Cekura lookup cannot race the ephemeral Live SIP session.
      if (dedicatedWorkflow) session.cekuraRunAssociation = selectWorkflow(sessionId, session, [workflow]);
      waitUntil(liveSidebandSession(sessionId, session).catch((error) => {
        console.error("OpenAI Live sideband error", { sessionId, error: error.message });
      }));
      return response.status(200).end();
    }
    if (event.type !== "realtime.call.incoming") return response.status(200).json({ ignored: event.type });

    const callId = event.data.call_id;
    const callerNumber = phoneFromSipHeaders(event.data.sip_headers);
    const session = {
      sessionId: `openai:${callId}`,
      openaiCallId: callId,
      callerNumber,
      startedAt: new Date().toISOString(),
      startedAtMs: Date.now(),
      entries: [],
    };
    await diagnostic(callId, "webhook_received", { event });
    const accepted = await fetch(`https://api.openai.com/v1/realtime/calls/${callId}/accept`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        type: "realtime",
        model: "gpt-realtime-2.1",
        audio: { input: { transcription: { model: "gpt-4o-mini-transcribe", language: "en" } } },
      }),
    });
    const acceptBody = await accepted.text();
    await diagnostic(callId, "accept_response", { status: accepted.status, body: acceptBody });
    if (!accepted.ok) throw new Error(`OpenAI call accept failed: ${accepted.status}: ${acceptBody}`);

    // OpenAI establishes the realtime session after the incoming-webhook ACK.
    // Defer the sideband task so this response reaches OpenAI first.
    const sideband = new Promise((resolve) => setTimeout(resolve, 1_000))
      .then(() => {
        return sidebandSession(callId, workflows, session);
      });
    waitUntil(sideband.catch((error) => {
      console.error("OpenAI Realtime sideband error", { callId, error: error.message });
    }));
    return response.status(200).end();
  } catch (error) {
    await diagnostic("unknown", "ingress_error", { message: error.message });
    console.error("OpenAI Realtime ingress error", error);
    return response.status(400).json({ accepted: false, error: error.message });
  }
}
