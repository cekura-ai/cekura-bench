#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const API_BASE = "https://api.cekura.ai/test_framework/v1";
const args = process.argv.slice(2);
const configIndex = args.indexOf("--config");
const configPath = configIndex >= 0 ? args[configIndex + 1] : "benchmark.config.json";
const execute = args.includes("--execute");
const validateOnly = args.includes("--validate");

if (!configPath || configPath.startsWith("--")) {
  throw new Error("Usage: npm run benchmark -- --config benchmark.config.json [--execute]");
}

const config = JSON.parse(await readFile(resolve(configPath), "utf8"));
const required = ["projectId", "catalogAgentId", "targetAgentId", "agentNumber"];
for (const key of required) {
  if (!config[key]) throw new Error(`Missing required config value: ${key}`);
}
if (config.catalogAgentId === config.targetAgentId) {
  throw new Error("catalogAgentId and targetAgentId must be different agents");
}

const apiKey = process.env.CEKURA_API_KEY;
if (!apiKey) throw new Error("CEKURA_API_KEY is not set");
const headers = {
  Accept: "application/json",
  "Content-Type": "application/json",
  "X-CEKURA-API-KEY": apiKey,
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { headers, ...options });
  const body = await response.text();
  if (!response.ok) throw new Error(`${options.method ?? "GET"} ${path}: HTTP ${response.status} ${body.slice(0, 800)}`);
  return body ? JSON.parse(body) : null;
}

function scenarioCode(scenario) {
  return String(scenario.name ?? "").match(/^(AS|MS)\d+\s*-/)?.[0] ?? null;
}

const listed = await api(`/scenarios/?agent_id=${config.catalogAgentId}&page_size=100`);
const scenarios = (listed.results ?? listed)
  .filter((scenario) => scenarioCode(scenario))
  .sort((left, right) => String(left.name).localeCompare(String(right.name), undefined, { numeric: true }));

const appointmentScenarios = scenarios.filter((scenario) => scenario.name.startsWith("AS"));
const medicareScenarios = scenarios.filter((scenario) => scenario.name.startsWith("MS"));
if (!appointmentScenarios.length || !medicareScenarios.length) {
  throw new Error("Catalog must contain both AS (Appointment) and MS (Medicare) scenarios");
}

const payload = {
  agent_id: config.targetAgentId,
  project_id: config.projectId,
  scenarios: scenarios.map((scenario) => scenario.id),
  frequency: config.frequency ?? 3,
  concurrency_limit: config.concurrencyLimit ?? 5,
  mode: config.numberMode ?? "different_numbers",
  mock_tool_names: [],
  agent_number: config.agentNumber,
  name: config.name ?? `Cekura benchmark — ${new Date().toISOString().slice(0, 10)}`,
};

const plan = {
  mode: validateOnly ? "validate" : execute ? "execute" : "dry-run",
  project_id: config.projectId,
  catalog_agent_id: config.catalogAgentId,
  target_agent_id: config.targetAgentId,
  appointment_scenarios: appointmentScenarios.length,
  medicare_scenarios: medicareScenarios.length,
  total_scenarios: scenarios.length,
  payload,
};

if (!validateOnly && execute) {
  const result = await api("/scenarios/run_scenarios/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  plan.result_id = result.id;
  plan.dashboard_url = `https://dashboard.cekura.ai/${config.projectId}/results/${result.id}`;
}

await mkdir(resolve("data"), { recursive: true });
const statePath = resolve("data", `benchmark-launch-${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
await writeFile(statePath, `${JSON.stringify({ created_at: new Date().toISOString(), ...plan }, null, 2)}\n`);
console.log(JSON.stringify({ state_path: statePath, ...plan }, null, 2));

