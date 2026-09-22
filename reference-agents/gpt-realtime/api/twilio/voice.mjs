const SIP_URI = process.env.OPENAI_SIP_URI || "sip:proj_Kc5M0zmLn0XC9rkVXpdrQp7C@sip.api.openai.com;transport=tls";
const ACCOUNT_SID = process.env.TWILIO_ACCOUNT_SID || "";
// The same source is deployed twice. Medicare must preserve its dedicated DID
// in the SIP header; otherwise the Medicare-only receiver treats its own call
// as an appointments call and deliberately declines the Live session.
const AGENT_NUMBER = process.env.CEKURA_MEDICARE_AGENT_NUMBER || process.env.CEKURA_AGENT_NUMBER || "+16205360171";

function escapeXml(value) {
  return String(value).replace(/[<>&"']/g, (character) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;", "'": "&apos;" })[character]);
}

export default async function handler(request, response) {
  if (request.method !== "POST") return response.status(405).send("POST required");
  const body = request.body && typeof request.body === "object" ? request.body : {};
  if (!ACCOUNT_SID || body.AccountSid !== ACCOUNT_SID || !body.CallSid) return response.status(403).send("Forbidden");
  const secureSipUri = SIP_URI.includes(";secure=true") ? SIP_URI : `${SIP_URI};secure=true`;
  const separator = secureSipUri.includes("?") ? "&" : "?";
  const target = `${secureSipUri}${separator}x-cekura-called-number=${encodeURIComponent(AGENT_NUMBER)}&x-cekura-parent-call-sid=${encodeURIComponent(body.CallSid)}`;
  response.setHeader("Content-Type", "text/xml");
  return response.status(200).send(`<?xml version="1.0" encoding="UTF-8"?><Response><Dial answerOnBridge="true"><Sip>${escapeXml(target)}</Sip></Dial></Response>`);
}
