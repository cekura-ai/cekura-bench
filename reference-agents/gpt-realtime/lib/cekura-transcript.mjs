const WEBHOOK_URL = "https://api.cekura.ai/test_framework/custom-provider-transcript-webhook";
// Pool caller numbers are reused while previous runs are still evaluating. Only
// the actual provider-call window is safe evidence for a `different_numbers`
// association; never bind a new SIP call to an older evaluating run.
const ACTIVE_RUN_STATUSES = new Set(["running", "in_progress"]);

export function normalizePhoneNumber(value) {
  const digits = String(value || "").replace(/\D/g, "");
  return digits.length >= 7 ? digits : null;
}

export function sipHeaderValue(headers, name) {
  return (headers || []).find((header) => header?.name?.toLowerCase() === name.toLowerCase())?.value || "";
}

export function phoneFromSipHeaders(headers) {
  const identity = sipHeaderValue(headers, "P-Asserted-Identity") || sipHeaderValue(headers, "From");
  const match = identity.match(/(?:sip:|tel:)(\+?[0-9]+)/i);
  return match ? `+${match[1].replace(/^\+/, "")}` : null;
}

export function parseObject(value) {
  if (value && typeof value === "object") return value;
  if (!value) return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" ? parsed : { value: parsed };
  } catch {
    return { value: String(value) };
  }
}

export function addTranscriptEntry(session, entry) {
  const now = Date.now();
  session.entries.push({
    ...entry,
    _sequence: session.entries.length,
    start_time: Math.max(0, (now - session.startedAtMs) / 1000),
    end_time: Math.max(0, (now - session.startedAtMs) / 1000),
  });
}

export function buildCustomTranscriptPayload(session, agentId) {
  const messages = [...session.entries]
    .sort((a, b) => (a.start_time - b.start_time) || (a._sequence - b._sequence))
    .map(({ _sequence, ...entry }) => entry);
  const call = {
    id: session.sessionId,
    startedAt: session.startedAt,
    endedAt: session.endedAt || new Date().toISOString(),
    endedReason: session.endedReason || "unknown",
    messages,
    metadata: { source: "openai-realtime-sip-vercel", openai_call_id: session.openaiCallId },
  };
  if (session.callerNumber) call.from_phone_number = session.callerNumber;
  if (Number.isInteger(session.cekuraRunId)) call.run_id = session.cekuraRunId;
  return { agent_id: agentId, calls: [call] };
}

export function findRunId(rows, { agentId, callerNumber, agentNumber }) {
  const caller = normalizePhoneNumber(callerNumber);
  const destination = normalizePhoneNumber(agentNumber);
  if (!caller) return null;
  const matches = (rows || []).filter((run) => (
    run?.agent === agentId
    && ACTIVE_RUN_STATUSES.has(String(run.status || "").toLowerCase())
    && normalizePhoneNumber(run.inbound_number) === caller
    && (!destination || normalizePhoneNumber(run.agent_number) === destination)
    && Number.isInteger(run.id)
  ));
  return matches.length === 1 ? matches[0].id : null;
}

export async function resolveCekuraRunId({ apiKey, agentId, callerNumber, agentNumber, fetchImpl = fetch }) {
  if (!apiKey || !callerNumber || !Number.isInteger(agentId)) return null;
  let url = `https://api.cekura.ai/test_framework/v1/runs/?agent_id=${agentId}&page_size=100`;
  for (let page = 0; page < 5 && url; page += 1) {
    const response = await fetchImpl(url, { headers: { "X-CEKURA-API-KEY": apiKey } });
    if (!response.ok) return null;
    const payload = await response.json();
    const match = findRunId(payload.results, { agentId, callerNumber, agentNumber });
    if (match) return match;
    url = payload.next || null;
  }
  return null;
}

export async function publishCekuraTranscript(session, { apiKey, agentId, fetchImpl = fetch }) {
  if (!apiKey || !Number.isInteger(agentId) || !session.entries.length) return { skipped: true };
  const response = await fetchImpl(WEBHOOK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CEKURA-API-KEY": apiKey },
    body: JSON.stringify(buildCustomTranscriptPayload(session, agentId)),
  });
  if (!response.ok) throw new Error(`Cekura transcript publish failed: ${response.status}`);
  return { skipped: false, status: response.status };
}
