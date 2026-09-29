import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const API_BASE = "https://api.cekura.ai/test_framework/v1";
export const TERMINAL = new Set(["completed", "failed", "timeout", "cancelled", "canceled"]);
// Keep this in lockstep with reference-agents/pipecat-s2s/bot.py Settings.KEYS.
const SESSION_KEYS = new Set(["s2s_provider", "s2s_model", "s2s_voice", "agent_dir", "s2s_backend_model", "s2s_backend_reasoning", "aws_region", "qwen_region", "qwen_workspace_id", "cascade_tts_voice", "cekura_mode"]);

export function isMain(url) { return Boolean(process.argv[1]) && url === pathToFileURL(resolve(process.argv[1])).href; }

export function fail(message) { throw new Error(message); }
export function positiveInt(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${label} must be a positive integer`);
  return value;
}
export function plainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
export function safeName(value, label) {
  if (typeof value !== "string" || !/^[a-zA-Z0-9][a-zA-Z0-9._ -]{0,119}$/.test(value)) fail(`${label} has unsupported characters`);
  return value;
}
export function displayName(value, label) {
  if (typeof value !== "string" || value.length < 1 || value.length > 180 || /[\x00-\x1f/\\]/.test(value)) fail(`${label} has unsupported characters`);
  return value;
}
export function parseFlags(argv, values, switches = []) {
  const result = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    if (switches.includes(key)) { if (result[key] !== undefined) fail(`Repeated ${key}`); result[key] = true; continue; }
    if (!values.includes(key)) fail(`Unknown option: ${key}`);
    if (result[key] !== undefined) fail(`Repeated ${key}`);
    const value = argv[++i];
    if (!value || value.startsWith("--")) fail(`${key} requires a value`);
    result[key] = value;
  }
  return result;
}
export async function readJson(path) { return JSON.parse(await readFile(resolve(path), "utf8")); }
export async function writeJsonNew(path, value) {
  await mkdir(dirname(resolve(path)), { recursive: true });
  await writeFile(resolve(path), `${JSON.stringify(value, null, 2)}\n`, { flag: "wx", mode: 0o600 });
}
export async function writeJsonAtomic(path, value) {
  const target = resolve(path);
  await mkdir(dirname(target), { recursive: true });
  const temp = `${target}.${process.pid}.${Date.now()}.tmp`;
  try {
    await writeFile(temp, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx", mode: 0o600 });
    await rename(temp, target);
  } catch (error) {
    try { await unlink(temp); } catch { /* no temporary file */ }
    throw error;
  }
}
export function validateRow(row) {
  if (!plainObject(row)) fail("Each row must be an object");
  safeName(row.key, "row.key");
  displayName(row.name, "row.name");
  safeName(row.provider, "row.provider");
  if (!["appointments", "medicare"].includes(row.suite)) fail(`${row.key}: invalid suite`);
  positiveInt(row.agent_id, `${row.key}.agent_id`);
  positiveInt(row.frequency, `${row.key}.frequency`);
  positiveInt(row.concurrency, `${row.key}.concurrency`);
  if (!Array.isArray(row.scenario_ids) || !row.scenario_ids.length || new Set(row.scenario_ids).size !== row.scenario_ids.length) fail(`${row.key}: scenario_ids must be a nonempty unique list`);
  row.scenario_ids.forEach((id) => positiveInt(id, `${row.key}.scenario_id`));
  if (!plainObject(row.config) || row.config.agent_dir !== row.suite || row.config.s2s_provider !== row.provider) fail(`${row.key}: config suite/provider mismatch`);
  if (typeof row.config.s2s_model !== "string" || !row.config.s2s_model) fail(`${row.key}: s2s_model required`);
  if (row.expected_agent_commit_prefix !== undefined && !/^[0-9a-f]{7,40}$/.test(row.expected_agent_commit_prefix)) fail(`${row.key}: expected_agent_commit_prefix must be a hex commit prefix`);
  for (const [key, value] of Object.entries(row.config)) {
    if (!SESSION_KEYS.has(key) || !["string", "number"].includes(typeof value) || String(value).trim() === "") fail(`${row.key}: unsupported or ignored session config key ${key}`);
  }
  if (row.mock_tool_names !== undefined && (JSON.stringify(row.mock_tool_names) !== "[]")) fail(`${row.key}: mock_tool_names must be []`);
  if (row.after !== undefined && !Array.isArray(row.after)) fail(`${row.key}: after must be an array`);
  return row;
}
export function validateCampaign(campaign) {
  if (!plainObject(campaign) || campaign.schema_version !== 1) fail("campaign schema_version must be 1");
  positiveInt(campaign.project_id, "project_id");
  safeName(campaign.pipecat_agent_name, "pipecat_agent_name");
  if (!Array.isArray(campaign.rows) || !campaign.rows.length) fail("campaign.rows required");
  campaign.rows.forEach(validateRow);
  const keys = new Set(campaign.rows.map((row) => row.key));
  if (keys.size !== campaign.rows.length || new Set(campaign.rows.map((row) => row.name)).size !== campaign.rows.length) fail("Campaign keys and names must be unique");
  for (const row of campaign.rows) for (const key of row.after ?? []) if (!keys.has(key) || key === row.key) fail(`${row.key}: invalid dependency ${key}`);
  return campaign;
}
export function emptyRegistry(projectId) { return { schema_version: 1, project_id: projectId, entries: [] }; }
export function validateRegistry(registry) {
  if (!plainObject(registry) || registry.schema_version !== 1) fail("registry schema_version must be 1");
  positiveInt(registry.project_id, "registry.project_id");
  if (!Array.isArray(registry.entries)) fail("registry.entries must be an array");
  const keys = new Set(), ids = new Set();
  for (const entry of registry.entries) {
    safeName(entry.key, "entry.key"); displayName(entry.name, "entry.name"); positiveInt(entry.result_id, "entry.result_id");
    if (!["appointments", "medicare"].includes(entry.suite)) fail(`${entry.key}: invalid suite`);
    safeName(entry.provider, `${entry.key}.provider`);
    const settings = entry.settings;
    if (!plainObject(settings) || !plainObject(settings.config)) fail(`${entry.key}: exact settings required`);
    positiveInt(settings.agent_id, `${entry.key}.settings.agent_id`);
    positiveInt(settings.frequency, `${entry.key}.settings.frequency`);
    positiveInt(settings.concurrency, `${entry.key}.settings.concurrency`);
    if (!Array.isArray(settings.scenario_ids) || !settings.scenario_ids.length || new Set(settings.scenario_ids).size !== settings.scenario_ids.length) fail(`${entry.key}: unique scenario_ids required`);
    settings.scenario_ids.forEach((id) => positiveInt(id, `${entry.key}.settings.scenario_id`));
    if (JSON.stringify(settings.mock_tool_names) !== "[]") fail(`${entry.key}: mock_tool_names must be []`);
    safeName(settings.pipecat_agent_name, `${entry.key}.settings.pipecat_agent_name`);
    if (settings.config.agent_dir !== entry.suite || settings.config.s2s_provider !== entry.provider || !settings.config.s2s_model) fail(`${entry.key}: settings suite/provider/model mismatch`);
    validateRow({ key: entry.key, name: entry.name, suite: entry.suite, provider: entry.provider, agent_id: settings.agent_id, scenario_ids: settings.scenario_ids, frequency: settings.frequency, concurrency: settings.concurrency, mock_tool_names: settings.mock_tool_names, config: settings.config, ...(settings.expected_agent_commit_prefix ? { expected_agent_commit_prefix: settings.expected_agent_commit_prefix } : {}) });
    if (!plainObject(entry.validation) || !["pending", "clear", "stop", "unverified"].includes(entry.validation.status)) fail(`${entry.key}: invalid validation state`);
    if (entry.validation.status === "clear" && !/^[0-9a-f]{64}$/.test(entry.validation.export_manifest_sha256 ?? "")) fail(`${entry.key}: clear requires a safe-export manifest hash`);
    if (entry.share_links !== undefined && !plainObject(entry.share_links)) fail(`${entry.key}: share_links must be an object`);
    if (keys.has(entry.key) || ids.has(entry.result_id)) fail("Duplicate registry key or result ID");
    keys.add(entry.key); ids.add(entry.result_id);
  }
  return registry;
}
export function payloadFor(row, agentName) {
  validateRow(row);
  return {
    agent: row.agent_id,
    scenarios: row.scenario_ids.map((scenario) => ({ scenario })),
    frequency: row.frequency,
    concurrency_limit: row.concurrency,
    mock_tool_names: [],
    name: row.name,
    pipecat_data: { pipecat_agent_name: agentName, config: row.config },
  };
}
export async function apiGet(path, base = API_BASE) {
  const key = process.env.CEKURA_API_KEY;
  if (!key) fail("CEKURA_API_KEY is not set");
  const response = await fetch(`${base}${path}`, { headers: { Accept: "application/json", "X-CEKURA-API-KEY": key } });
  if (!response.ok) fail(`GET ${path}: HTTP ${response.status}`);
  return response.json();
}
export async function apiPost(path, body, base = API_BASE) {
  const key = process.env.CEKURA_API_KEY;
  if (!key) fail("CEKURA_API_KEY is not set");
  const response = await fetch(`${base}${path}`, { method: "POST", headers: { Accept: "application/json", "Content-Type": "application/json", "X-CEKURA-API-KEY": key }, body: JSON.stringify(body) });
  if (!response.ok) fail(`POST ${path}: HTTP ${response.status}`);
  return response.json();
}
export function listItems(response) {
  if (Array.isArray(response)) return response;
  if (Array.isArray(response?.results)) return response.results;
  fail("Unexpected result-list response");
}
export function resultRuns(result) {
  const runs = result?.runs ?? result?.run_details;
  if (Array.isArray(runs)) return runs;
  if (plainObject(runs)) return Object.values(runs);
  fail("Result has no run list");
}
export function tally(values) { const output = {}; for (const value of values) { const key = value ?? "unavailable"; output[key] = (output[key] ?? 0) + 1; } return output; }
