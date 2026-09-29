#!/usr/bin/env node

// A single-row, fail-closed Pipecat campaign launcher. Preview makes no API call.
import { createHash } from "node:crypto";
import { mkdir, rmdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { apiGet, apiPost, emptyRegistry, fail, isMain, listItems, parseFlags, payloadFor, readJson, TERMINAL, validateCampaign, validateRegistry, writeJsonAtomic } from "../lib/s2s-toolkit.mjs";

const usage = "Usage: node bin/s2s-campaign.mjs --campaign campaign.json --registry registry.json [--row key] [--execute]\nPreview is default. --execute requires --row and CEKURA_API_KEY.\n";

async function main() {
  const flags = parseFlags(process.argv.slice(2), ["--campaign", "--registry", "--row"], ["--execute", "--help"]);
  if (flags["--help"]) return process.stdout.write(usage);
  if (!flags["--campaign"] || !flags["--registry"]) fail(usage.trim());
  if (flags["--execute"] && !flags["--row"]) fail("--execute requires --row; one result per invocation");
  const campaign = validateCampaign(await readJson(flags["--campaign"]));
  let registry;
  try { registry = validateRegistry(await readJson(flags["--registry"])); }
  catch (error) { if (error.code !== "ENOENT") throw error; registry = emptyRegistry(campaign.project_id); }
  if (registry.project_id !== campaign.project_id) fail("Campaign/registry project mismatch");
  const rows = flags["--row"] ? campaign.rows.filter((row) => row.key === flags["--row"]) : campaign.rows;
  if (!rows.length) fail("Unknown --row key");
  const plans = rows.map((row) => {
    const payload = payloadFor(row, campaign.pipecat_agent_name);
    const recorded = registry.entries.find((entry) => entry.key === row.key);
    const dependencies = (row.after ?? []).map((key) => registry.entries.find((entry) => entry.key === key));
    const dependencyClear = dependencies.length === (row.after ?? []).length && dependencies.every((entry) => entry?.validation?.status === "clear" && entry.validation.export_manifest_sha256);
    return { row: row.key, planned_calls: row.scenario_ids.length * row.frequency, recorded_result_id: recorded?.result_id ?? null, dependencies_clear: dependencyClear, eligible_locally: !recorded && dependencyClear, payload };
  });
  if (!flags["--execute"]) return process.stdout.write(`${JSON.stringify({ mode: "preview", project_id: campaign.project_id, plans }, null, 2)}\n`);

  const plan = plans[0], row = rows[0];
  if (!plan.eligible_locally) fail(`${row.key}: already recorded or dependencies are not validation-clear`);
  const lock = `${resolve(flags["--registry"])}.lock`;
  await mkdir(dirname(lock), { recursive: true });
  await mkdir(lock); // An existing lock requires human reconciliation, never an automatic retry.
  try {
    registry = validateRegistry(await readJson(flags["--registry"]).catch((error) => {
      if (error.code !== "ENOENT") throw error;
      return emptyRegistry(campaign.project_id);
    }));
    if (registry.entries.some((entry) => entry.key === row.key || entry.name === row.name)) fail(`${row.key}: registry already contains this key or name`);
    for (const key of row.after ?? []) {
      const dependency = registry.entries.find((entry) => entry.key === key);
      if (dependency?.validation?.status !== "clear" || !dependency.validation.export_manifest_sha256) fail(`${row.key}: dependency ${key} lacks a verified safe export`);
      const live = await apiGet(`/results/${dependency.result_id}/`);
      if (!TERMINAL.has(String(live.status ?? "").toLowerCase())) fail(`${row.key}: dependency ${key} is not terminal remotely`);
    }
    const query = `/results/?project_id=${campaign.project_id}&name=${encodeURIComponent(row.name)}&page_size=200`;
    const remote = await apiGet(query);
    if (remote.next) fail("Result listing is paginated; cannot prove name uniqueness");
    if (listItems(remote).some((item) => item.name === row.name)) fail(`${row.key}: an exact-name result already exists remotely`);
    const created = await apiPost("/scenarios/run_scenarios_pipecat_v2/", plan.payload);
    const resultId = created?.id ?? created?.result?.id;
    if (!Number.isSafeInteger(resultId) || resultId < 1) fail("Launch response did not contain a result ID; inspect remote name before retrying");
    const settingsHash = createHash("sha256").update(JSON.stringify(plan.payload)).digest("hex");
    registry.entries.push({ key: row.key, name: row.name, result_id: resultId, suite: row.suite, provider: row.provider, settings_sha256: settingsHash, settings: { agent_id: row.agent_id, scenario_ids: row.scenario_ids, frequency: row.frequency, concurrency: row.concurrency, mock_tool_names: [], pipecat_agent_name: campaign.pipecat_agent_name, config: row.config, ...(row.expected_agent_commit_prefix ? { expected_agent_commit_prefix: row.expected_agent_commit_prefix } : {}) }, validation: { status: "pending" }, share_links: {} });
    await writeJsonAtomic(flags["--registry"], registry);
    process.stdout.write(`${JSON.stringify({ mode: "execute", row: row.key, result_id: resultId, planned_calls: plan.planned_calls })}\n`);
  } finally { await rmdir(lock); }
}

if (isMain(import.meta.url)) main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
