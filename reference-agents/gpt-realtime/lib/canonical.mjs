import crypto from "node:crypto";
import { gunzipSync } from "node:zlib";

const required = [
  "OPENAI_API_KEY",
  "CANONICAL_PROMPT",
  "CANONICAL_FIRST_MESSAGE",
  "CANONICAL_TOOLS_JSON",
];

function configFromVariables({ key, label, prompt, firstMessage, toolsJson, agentId }) {
  const missing = [
    ["prompt", prompt],
    ["first message", firstMessage],
    ["tools JSON", toolsJson],
    ["Cekura agent ID", agentId],
  ].filter(([, value]) => !value).map(([name]) => name);
  if (missing.length) throw new Error(`Missing ${label} configuration: ${missing.join(", ")}`);

  let tools;
  try {
    tools = JSON.parse(toolsJson);
  } catch {
    throw new Error(`${label} tools JSON must be valid JSON`);
  }
  if (!Array.isArray(tools) || tools.length !== 4) throw new Error(`Expected exactly four ${label} tools`);

  const parsedAgentId = Number.parseInt(agentId, 10);
  if (!Number.isInteger(parsedAgentId)) throw new Error(`${label} Cekura agent ID must be an integer`);
  return {
    key,
    label,
    prompt,
    firstMessage,
    tools,
    agentId: parsedAgentId,
    promptSha256: crypto.createHash("sha256").update(prompt).digest("hex"),
  };
}

export function canonicalConfig() {
  const missing = required.filter((name) => !process.env[name]);
  if (missing.length) throw new Error(`Missing required configuration: ${missing.join(", ")}`);

  let tools;
  try {
    tools = JSON.parse(process.env.CANONICAL_TOOLS_JSON);
  } catch {
    throw new Error("CANONICAL_TOOLS_JSON must be valid JSON");
  }
  if (!Array.isArray(tools) || tools.length !== 4) throw new Error("Expected exactly four canonical tools");

  return {
    prompt: process.env.CANONICAL_PROMPT,
    firstMessage: process.env.CANONICAL_FIRST_MESSAGE,
    tools,
    promptSha256: crypto.createHash("sha256").update(process.env.CANONICAL_PROMPT).digest("hex"),
  };
}

// Appointment and Medicare deliberately share the OpenAI SIP URI and webhook.
// Cekura's different_numbers caller identity selects which of these canonical
// workflows is attached to the live call before the greeting is sent.
export function realtimeWorkflowConfigs() {
  let medicare;
  if (process.env.MEDICARE_CANONICAL_CONFIG_GZIP_BASE64) {
    try {
      medicare = JSON.parse(gunzipSync(Buffer.from(process.env.MEDICARE_CANONICAL_CONFIG_GZIP_BASE64, "base64")).toString("utf8"));
    } catch {
      throw new Error("MEDICARE_CANONICAL_CONFIG_GZIP_BASE64 must be a valid gzipped JSON configuration");
    }
  }
  return [
    configFromVariables({
      key: "appointments",
      label: "Appointment",
      prompt: process.env.CANONICAL_PROMPT,
      firstMessage: process.env.CANONICAL_FIRST_MESSAGE,
      toolsJson: process.env.CANONICAL_TOOLS_JSON,
      agentId: process.env.CEKURA_AGENT_ID,
    }),
    configFromVariables({
      key: "medicare",
      label: "Medicare",
      prompt: medicare?.prompt || process.env.MEDICARE_CANONICAL_PROMPT,
      firstMessage: medicare?.firstMessage || process.env.MEDICARE_CANONICAL_FIRST_MESSAGE,
      toolsJson: medicare?.toolsJson || process.env.MEDICARE_CANONICAL_TOOLS_JSON,
      agentId: process.env.CEKURA_MEDICARE_AGENT_ID,
    }),
  ];
}

export function realtimeTools(tools) {
  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
  }));
}

export function requireWebhookSecret() {
  if (!process.env.OPENAI_WEBHOOK_SECRET) throw new Error("Missing required configuration: OPENAI_WEBHOOK_SECRET");
  return process.env.OPENAI_WEBHOOK_SECRET;
}
