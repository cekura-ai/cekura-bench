#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const API_BASE_V1 = "https://api.cekura.ai/test_framework/v1";
const API_BASE_V2 = "https://api.cekura.ai/test_framework/v2";
const NATIVE_PROVIDERS = new Set(["vapi", "retell", "elevenlabs", "synthflow"]);
const PROVIDER_KEY_ENVS = {
  vapi: "VAPI_API_KEY",
  retell: "RETELL_API_KEY",
  elevenlabs: "ELEVENLABS_API_KEY",
  synthflow: "SYNTHFLOW_API_KEY",
};

const args = process.argv.slice(2);
const configIndex = args.indexOf("--config");
const configPath = configIndex >= 0 ? args[configIndex + 1] : "benchmark.config.json";
const execute = args.includes("--execute");
const validateOnly = args.includes("--validate");

if (!configPath || configPath.startsWith("--")) {
  throw new Error("Usage: npm run benchmark -- --config benchmark.config.json [--execute]");
}

const config = JSON.parse(await readFile(resolve(configPath), "utf8"));
const watchResults = config.watchResults !== false;
// A Pipecat Cloud run has the platform create a Daily room and start a session
// in it, so there is no number to dial and none to require.
const viaPipecat = Boolean(config.pipecat);
const required = viaPipecat
  ? ["projectId", "catalogAgentId", "suite"]
  : ["projectId", "catalogAgentId", "agentNumber", "suite"];
for (const key of required) {
  if (!config[key]) throw new Error(`Missing required config value: ${key}`);
}
if (!["appointments", "medicare"].includes(config.suite)) {
  throw new Error('suite must be either "appointments" or "medicare"');
}
const pipecatAgentName = config.pipecat?.agentName ?? config.agentSetup?.pipecat?.agentName;
if (viaPipecat && !pipecatAgentName) {
  throw new Error("pipecat.agentName is required to launch a Pipecat Cloud run");
}
// The session config replaces the agent's stored one rather than merging over
// it, so a partial config silently drops the agent's defaults. Send the whole
// configuration under test, or send none and let the agent's stored one stand.
if (viaPipecat && config.pipecat.config && typeof config.pipecat.config !== "object") {
  throw new Error("pipecat.config must be an object of session keys");
}
if (config.targetAgentId && config.agentSetup) {
  throw new Error("Use either targetAgentId or agentSetup, not both");
}

const apiKey = process.env.CEKURA_API_KEY;
if (!apiKey) throw new Error("CEKURA_API_KEY is not set");
const headers = {
  Accept: "application/json",
  "Content-Type": "application/json",
  "X-CEKURA-API-KEY": apiKey,
};

async function api(baseUrl, path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, { headers, ...options });
  const body = await response.text();
  if (!response.ok) throw new Error(`${options.method ?? "GET"} ${path}: HTTP ${response.status} ${body.slice(0, 800)}`);
  return body ? JSON.parse(body) : null;
}

function redact(value) {
  if (Array.isArray(value)) return value.map(redact);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value).map(([key, entry]) => [
    key,
    /api[ _-]?key|secret|password|token/i.test(key) ? "[REDACTED]" : redact(entry),
  ]));
}

function nativeSetupPayload(setup) {
  const provider = String(setup.provider ?? "").toLowerCase();
  if (!NATIVE_PROVIDERS.has(provider)) {
    throw new Error(`agentSetup.provider must be one of: ${[...NATIVE_PROVIDERS].join(", ")}`);
  }
  if (!setup.providerAgentId) {
    throw new Error("agentSetup.providerAgentId is required for a native provider");
  }
  const keyEnv = setup.providerApiKeyEnv ?? PROVIDER_KEY_ENVS[provider];
  const providerApiKey = process.env[keyEnv];
  if (!providerApiKey) throw new Error(`${keyEnv} is not set`);

  return {
    project: config.projectId,
    provider: {
      type: provider,
      agent_id: setup.providerAgentId,
      credentials: { api_key: providerApiKey },
      configure_from_provider: true,
    },
  };
}

function managedSetupPayload(setup = {}) {
  const provider = String(setup.provider ?? "self_hosted").toLowerCase();
  if (NATIVE_PROVIDERS.has(provider)) return nativeSetupPayload(setup);
  if (!["self_hosted", "livekit", "pipecat"].includes(provider)) {
    throw new Error("agentSetup.provider must be vapi, retell, elevenlabs, synthflow, livekit, pipecat, or self_hosted");
  }
  if (provider === "livekit" && (!setup.livekit?.url || !setup.livekit?.apiSecretEnv || !setup.livekit?.agentName)) {
    throw new Error("LiveKit setup requires livekit.url, livekit.apiSecretEnv, and livekit.agentName");
  }
  if (provider === "pipecat" && !setup.pipecat?.agentName) {
    throw new Error("Pipecat setup requires pipecat.agentName");
  }
  const providerKeyEnv = setup.providerApiKeyEnv ?? (provider === "livekit" ? "LIVEKIT_API_KEY" : undefined);

  const providerConfig = provider === "livekit" ? {
    url: setup.livekit?.url,
    api_secret: setup.livekit?.apiSecretEnv ? process.env[setup.livekit.apiSecretEnv] : undefined,
    agent_name: setup.livekit?.agentName,
    config: setup.livekit?.config,
  } : provider === "pipecat" ? {
    pipecat_agent_name: setup.pipecat?.agentName,
    webhook_url: setup.pipecat?.webhookUrl,
    config: setup.pipecat?.config,
    room_properties: setup.pipecat?.roomProperties,
  } : undefined;
  const credentials = providerConfig ? {
    ...(providerKeyEnv ? { api_key: process.env[providerKeyEnv] } : {}),
    config: Object.fromEntries(Object.entries(providerConfig).filter(([, value]) => value !== undefined)),
  } : undefined;
  if (providerKeyEnv && !process.env[providerKeyEnv]) {
    throw new Error(`${providerKeyEnv} is not set`);
  }

  return {
    name: setup.name ?? "Benchmark target agent",
    description: setup.description ?? "Voice agent registered for Cekura benchmark evaluation.",
    project: config.projectId,
    language: setup.language ?? "en",
    provider: {
      type: provider,
      ...(credentials ? { credentials } : {}),
    },
    telephony: {
      phone_number: config.agentNumber,
      inbound: setup.inbound ?? true,
      ...(setup.sipUri ? { sip_uri: setup.sipUri } : {}),
      ...(setup.sipAuth ? { sip_auth: setup.sipAuth } : {}),
      ...(setup.outboundNumbers ? { outbound_numbers: setup.outboundNumbers } : {}),
    },
  };
}

function agentIdFrom(value) {
  return value?.id ?? value?.agent_id ?? value?.agent?.id ?? value?.result?.id ?? value?.result?.agent_id ?? null;
}

const TERMINAL_RESULT_STATUSES = new Set(["completed", "failed", "cancelled", "canceled"]);
const sleep = (milliseconds) => new Promise((resolveSleep) => setTimeout(resolveSleep, milliseconds));

async function watchAndShareResult(resultId) {
  const defaultWaitMinutes = config.suite === "medicare" ? 14 * 6 : 32 * 6;
  const initialWaitMinutes = config.watchInitialWaitMinutes ?? defaultWaitMinutes;
  const pollSeconds = config.watchPollSeconds ?? 60;
  await sleep(initialWaitMinutes * 60_000);

  let result;
  for (;;) {
    result = await api(API_BASE_V1, `/results/${resultId}/`);
    if (TERMINAL_RESULT_STATUSES.has(String(result?.status ?? "").toLowerCase())) break;
    await sleep(pollSeconds * 1000);
  }

  const report = await api(API_BASE_V1, "/results/reports/share/", {
    method: "POST",
    body: JSON.stringify({
      result_ids: [resultId],
      name: config.name ?? `Cekura ${config.suite} benchmark — ${new Date().toISOString().slice(0, 10)}`,
      expire_at: null,
    }),
  });
  const shareUrl = report?.share_url ?? report?.public_url ?? report?.url ?? report?.share?.url ?? null;
  const markdownPath = resolve("data", `benchmark-report-${resultId}.md`);
  const lines = [
    "# Benchmark report",
    "",
    `- Suite: ${config.suite}`,
    `- Result: ${resultId}`,
    `- Status: ${result?.status ?? "unknown"}`,
    `- Dashboard: https://dashboard.cekura.ai/${config.projectId}/results/${resultId}`,
    `- Public report: ${shareUrl ?? "created; see share response in the launch record"}`,
    "",
  ];
  await writeFile(markdownPath, lines.join("\n"));
  return { report: redact(report), share_url: shareUrl, report_path: markdownPath };
}

async function waitForImportedAgent(progressId) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const progress = await api(API_BASE_V2, `/aiagents/import-progress/?progress_id=${encodeURIComponent(progressId)}`);
    const status = String(progress?.status ?? "").toLowerCase();
    const agentId = agentIdFrom(progress);
    if (agentId) return { progress, agentId };
    if (["failed", "error", "cancelled"].includes(status)) {
      throw new Error(`Agent import ${progressId} ended with status: ${status}`);
    }
    if (["completed", "complete", "succeeded", "success"].includes(status)) {
      throw new Error(`Agent import ${progressId} completed without returning an agent ID`);
    }
    await new Promise((resolveSleep) => setTimeout(resolveSleep, 2000));
  }
  throw new Error(`Timed out waiting for agent import ${progressId}`);
}

async function provisionTargetAgent(payload) {
  const result = await api(API_BASE_V2, "/aiagents/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  const agentId = agentIdFrom(result);
  if (agentId) return { result, agentId };
  if (result?.progress_id) return waitForImportedAgent(result.progress_id);
  throw new Error("Agent setup did not return an agent ID or import progress ID");
}

function scenarioCode(scenario) {
  return String(scenario.name ?? "").match(/^(AS|MS)\d+\s*-/)?.[0] ?? null;
}

function scenarioNumber(scenario) {
  return String(scenario.name ?? "").match(/^(AS|MS)\d+/)?.[0] ?? "";
}

const listed = await api(API_BASE_V1, `/scenarios/?agent_id=${config.catalogAgentId}&page_size=100`);
const scenarios = (listed.results ?? listed)
  .filter((scenario) => scenarioCode(scenario))
  .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { numeric: true }));
const suitePrefix = config.suite === "appointments" ? "AS" : "MS";
let suiteScenarios = scenarios.filter((scenario) => scenario.name.startsWith(suitePrefix));
if (!suiteScenarios.length) {
  throw new Error(`Catalog does not contain ${suitePrefix} scenarios for suite: ${config.suite}`);
}
// A named subset, for a shakedown launch that only has to prove the pipe carries
// every field a board is scored from. Codes, not ids: a code is stable across
// catalogs and readable in the launch record, so what a small run covered stays
// legible after the run id has aged out.
if (config.scenarios?.length) {
  const wanted = new Set(config.scenarios.map((code) => String(code).trim().toUpperCase()));
  suiteScenarios = suiteScenarios.filter((scenario) => wanted.has(scenarioNumber(scenario)));
  const found = new Set(suiteScenarios.map((scenario) => scenarioNumber(scenario)));
  const missing = [...wanted].filter((code) => !found.has(code));
  if (missing.length) {
    throw new Error(`Catalog has no ${missing.join(", ")} for suite: ${config.suite}`);
  }
}

const setupPayload = config.targetAgentId ? null : managedSetupPayload(config.agentSetup);
const provisioned = execute && setupPayload ? await provisionTargetAgent(setupPayload) : null;
const targetAgentId = config.targetAgentId ?? provisioned?.agentId ?? null;
const runName = config.name ?? `Cekura ${config.suite} benchmark — ${new Date().toISOString().slice(0, 10)}`;
// Two endpoints, because the transports differ in kind rather than in detail. A
// phone run dials a number. A Pipecat Cloud run carries a session body instead,
// which is where the configuration under test travels -- so one deployment can
// answer for every configuration and a row says which one it was.
const runPath = viaPipecat ? "/scenarios/run_scenarios_pipecat_v2/" : "/scenarios/run_scenarios/";
const payload = viaPipecat ? {
  agent: targetAgentId,
  scenarios: suiteScenarios.map((scenario) => ({ scenario: scenario.id })),
  frequency: config.frequency ?? 3,
  concurrency_limit: config.concurrencyLimit ?? 5,
  // Empty rather than omitted: omitting it mocks every tool configured on the
  // agent, and this agent answers its own tools from the published contract.
  mock_tool_names: [],
  name: runName,
  pipecat_data: {
    pipecat_agent_name: pipecatAgentName,
    ...(config.pipecat.config ? { config: config.pipecat.config } : {}),
    ...(config.pipecat.roomProperties ? { room_properties: config.pipecat.roomProperties } : {}),
  },
} : {
  agent_id: targetAgentId,
  project_id: config.projectId,
  scenarios: suiteScenarios.map((scenario) => scenario.id),
  frequency: config.frequency ?? 3,
  concurrency_limit: config.concurrencyLimit ?? 5,
  mode: config.numberMode ?? "different_numbers",
  mock_tool_names: [],
  agent_number: config.agentNumber,
  name: runName,
};
const plan = {
  mode: validateOnly ? "validate" : execute ? "execute" : "dry-run",
  project_id: config.projectId,
  catalog_agent_id: config.catalogAgentId,
  target_agent_id: targetAgentId,
  suite: config.suite,
  transport: viaPipecat ? "pipecat" : "telephony",
  suite_scenarios: suiteScenarios.length,
  ...(config.scenarios?.length ? { scenario_subset: suiteScenarios.map(scenarioNumber) } : {}),
  ...(setupPayload ? { agent_setup: redact(setupPayload) } : {}),
  ...(provisioned ? { agent_setup_result: redact(provisioned.result) } : {}),
  payload,
};

if (!validateOnly && execute) {
  const result = await api(API_BASE_V1, runPath, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  plan.result_id = result.id;
  plan.dashboard_url = `https://dashboard.cekura.ai/${config.projectId}/results/${result.id}`;
  if (watchResults) plan.report = await watchAndShareResult(result.id);
}

await mkdir(resolve("data"), { recursive: true });
const statePath = resolve("data", `benchmark-launch-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
await writeFile(statePath, `${JSON.stringify({ created_at: new Date().toISOString(), ...plan }, null, 2)}\n`);
console.log(JSON.stringify({ state_path: statePath, ...plan }, null, 2));
