import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { payloadFor, SESSION_KEYS, TERMINAL, validateCampaign } from "../lib/s2s-toolkit.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const run = (script, args) => execFileSync(process.execPath, [join(root, "bin", script), ...args], { cwd: root, encoding: "utf8", env: { ...process.env, CEKURA_API_KEY: "" } });
const result = (script, args) => spawnSync(process.execPath, [join(root, "bin", script), ...args], { cwd: root, encoding: "utf8", env: { ...process.env, CEKURA_API_KEY: "" } });
const json = (path, value) => writeFile(path, `${JSON.stringify(value)}\n`);
const row = { key: "example", name: "S2S — example — round 1 — appointments", suite: "appointments", provider: "example", agent_id: 10, scenario_ids: [1], frequency: 1, concurrency: 1, mock_tool_names: [], expected_agent_commit_prefix: "3abc123", config: { agent_dir: "appointments", s2s_provider: "example", s2s_model: "models/example-v1" } };
const settings = { agent_id: row.agent_id, scenario_ids: row.scenario_ids, frequency: row.frequency, concurrency: row.concurrency, mock_tool_names: [], pipecat_agent_name: "cekura-s2s", config: row.config, expected_agent_commit_prefix: row.expected_agent_commit_prefix };
const entry = { key: row.key, name: row.name, result_id: 99, suite: row.suite, provider: row.provider, settings, validation: { status: "unverified" } };

test("campaign preview is offline, validates session keys, and builds an exact payload", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-campaign-"));
  const campaign = join(dir, "campaign.json");
  await json(campaign, { schema_version: 1, project_id: 1, pipecat_agent_name: "cekura-s2s", rows: [row] });
  const preview = JSON.parse(run("s2s-campaign.mjs", ["--campaign", campaign, "--registry", join(dir, "registry.json")]));
  assert.equal(preview.mode, "preview");
  assert.equal(preview.plans[0].planned_calls, 1);
  assert.deepEqual(preview.plans[0].payload, payloadFor(row, "cekura-s2s"));
  assert.deepEqual(preview.plans[0].payload.scenarios, [{ scenario: 1 }]);
  assert.deepEqual(preview.plans[0].payload.mock_tool_names, []);
  assert.throws(() => validateCampaign({ schema_version: 1, project_id: 1, pipecat_agent_name: "cekura-s2s", rows: [{ ...row, config: { ...row.config, ignored_setting: "x" } }] }), /unsupported or ignored session config key/);
  assert.equal(TERMINAL.has("timeout"), true);
});

test("session config allowlist matches the bot's accepted keys", async () => {
  const bot = await readFile(join(root, "reference-agents", "pipecat-s2s", "bot.py"), "utf8");
  const declaration = bot.match(/class Settings:[\s\S]*?^    KEYS\s*=\s*\(([\s\S]*?)^    \)/m);
  assert.ok(declaration, "Settings.KEYS declaration must be readable");
  const keys = [...declaration[1].matchAll(/^\s*"([^"]+)",?\s*$/gm)].map((match) => match[1]);
  assert.ok(keys.length > 0, "Settings.KEYS must not be empty");
  assert.deepEqual([...SESSION_KEYS].sort(), keys.sort());
});

test("registry and campaign share a lock; duplicate refusal has the expected reason", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-registry-"));
  const path = join(dir, "registry.json"), campaign = join(dir, "campaign.json"), imported = join(dir, "entry.json");
  await json(campaign, { schema_version: 1, project_id: 1, pipecat_agent_name: "cekura-s2s", rows: [row] });
  run("s2s-cohort-registry.mjs", ["init", "--registry", path, "--project-id", "1"]);
  await json(imported, { ...entry, validation: { status: "clear", export_manifest_sha256: "a".repeat(64) } });
  run("s2s-cohort-registry.mjs", ["import", "--registry", path, "--entry", imported]);
  assert.equal(JSON.parse(await readFile(path, "utf8")).entries[0].validation.status, "unverified");
  const rejected = result("s2s-campaign.mjs", ["--campaign", campaign, "--registry", path, "--row", row.key, "--execute"]);
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /already recorded or dependencies are not validation-clear/);
  const premature = result("s2s-cohort-registry.mjs", ["validation", "--registry", path, "--key", row.key, "--status", "clear"]);
  assert.notEqual(premature.status, 0);
  assert.match(premature.stderr, /requires --export/);
  assert.equal((await readFile(path, "utf8")).includes("\"status\": \"clear\""), false);
});

test("one occupied registry lock blocks both launch and registry mutation", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-lock-"));
  const path = join(dir, "registry.json"), campaign = join(dir, "campaign.json"), imported = join(dir, "entry.json");
  await json(campaign, { schema_version: 1, project_id: 1, pipecat_agent_name: "cekura-s2s", rows: [row] });
  await json(path, { schema_version: 1, project_id: 1, entries: [] });
  await json(imported, entry);
  await mkdir(`${path}.lock`);
  const launch = result("s2s-campaign.mjs", ["--campaign", campaign, "--registry", path, "--row", row.key, "--execute"]);
  const importAttempt = result("s2s-cohort-registry.mjs", ["import", "--registry", path, "--entry", imported]);
  assert.notEqual(launch.status, 0);
  assert.notEqual(importAttempt.status, 0);
  assert.match(launch.stderr, /EEXIST/);
  assert.match(importAttempt.stderr, /EEXIST/);
});

async function fixture(dir) {
  const registry = join(dir, "registry.json"), raw = join(dir, "raw"), definitions = join(dir, "definitions"), campaign = join(dir, "presentation.json");
  await Promise.all([mkdir(join(raw, "appointments", "runs"), { recursive: true }), mkdir(join(definitions, "appointments"), { recursive: true }), mkdir(join(definitions, "medicare"), { recursive: true })]);
  await json(registry, { schema_version: 1, project_id: 1, entries: [entry] });
  await json(campaign, { repeats: 1, decisions: { build: "fixture" }, models: { example: { short: "Example", name: "Example", vendor: "Fixture", setting: "Fixture" } }, suites: [] });
  await json(join(definitions, "appointments", "expected-tool-calls.json"), { 1: [{ name: "lookup", arguments: { marker: "fixture-only" } }] });
  await json(join(definitions, "appointments", "tool-definitions.json"), [{ name: "lookup", parameters: { properties: { marker: { type: "string" } } } }]);
  await json(join(definitions, "medicare", "expected-tool-calls.json"), {});
  await json(join(definitions, "medicare", "tool-definitions.json"), []);
  await json(join(raw, "appointments", "result-99.json"), { id: 99, status: "timeout", runs: [{ id: 101 }] });
  const meta = { config: "example", agent_definition: "appointments", cekura_agent_id: 10, s2s_provider: "example", s2s_model: "models/example-v1", agent_commit: "3abc123", pipecat_version: "1", cekura_version: "1", integrity: { checks: ["ok"], closed_by: "agent" }, usage: {}, tool_calls: [{ name: "lookup", arguments: { marker: "fixture-only" }, resolution: "exact" }] };
  await json(join(raw, "appointments", "runs", "101.json"), { source: { suite: "appointments", provider: "example", resultId: 99, run_id: 101 }, run: { id: 101, result_id: 99, scenario: 1, status: "completed", success: true, duration: "01:30", started_at: "2026-01-01T00:00:00Z", voice_recording_url: "private-recording-url", transcript: "private caller words", transcript_object: { text: "private caller words" }, evaluation: { metrics: [{ name: "Expected Outcome", score: 5 }, { name: "Infrastructure Issues", score: 1 }, { name: "Latency (in ms)", score: 1200 }, { name: "Tool Call Accuracy", score: 5 }] }, provider_call_details: { custom_metadata: meta } } });
  return { registry, raw, definitions, campaign };
}

test("cohort export rejects a run or capture assigned to another result", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-result-identity-"));
  const { registry, raw, definitions, campaign } = await fixture(dir);
  const path = join(raw, "appointments", "runs", "101.json");
  const original = JSON.parse(await readFile(path, "utf8"));
  await json(path, { ...original, run: { ...original.run, result_id: 100 } });
  const flags = ["--registry", registry, "--campaign", campaign, "--definitions", definitions, "--raw", raw];
  const wrongRun = result("s2s-export-cohort.mjs", [...flags, "--out", join(dir, "wrong-run")]);
  assert.notEqual(wrongRun.status, 0);
  assert.match(wrongRun.stderr, /run\/source identity mismatch/);
  await json(path, { ...original, source: { ...original.source, resultId: 100 } });
  const wrongCapture = result("s2s-export-cohort.mjs", [...flags, "--out", join(dir, "wrong-capture")]);
  assert.notEqual(wrongCapture.status, 0);
  assert.match(wrongCapture.stderr, /run\/source identity mismatch/);
});

test("cohort export works when invoked outside the repository", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-external-cwd-"));
  const { registry, raw, definitions, campaign } = await fixture(dir);
  const output = join(dir, "safe");
  const summary = JSON.parse(execFileSync(process.execPath, [join(root, "bin", "s2s-export-cohort.mjs"), "--registry", registry, "--campaign", campaign, "--definitions", definitions, "--raw", raw, "--out", output], { cwd: dir, encoding: "utf8", env: { ...process.env, CEKURA_API_KEY: "" } }));
  assert.equal(summary.runs, 1);
  assert.equal(JSON.parse(await readFile(join(output, "s2s-benchmark.json"), "utf8")).schemaVersion, 2);
});

test("cohort driver emits only the canonical website contract and binds registry clear to its hash", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-export-"));
  const { registry, raw, definitions, campaign } = await fixture(dir);
  const out = join(dir, "safe"), zip = join(dir, "safe.zip");
  const summary = JSON.parse(run("s2s-export-cohort.mjs", ["--registry", registry, "--campaign", campaign, "--definitions", definitions, "--raw", raw, "--out", out, "--zip", zip]));
  assert.equal(summary.runs, 1);
  const benchmark = JSON.parse(await readFile(join(out, "s2s-benchmark.json"), "utf8"));
  assert.equal(benchmark.schemaVersion, 2);
  assert.equal(benchmark.results[0].toolCalls.scoredRuns, 1);
  assert.equal(benchmark.build.agentCommit, "3abc123");
  const bundle = execFileSync("unzip", ["-p", zip], { encoding: "utf8" });
  const names = execFileSync("unzip", ["-Z1", zip], { encoding: "utf8" }).trim().split("\n");
  assert.deepEqual(names.sort(), ["manifest.json", "s2s-benchmark.json", "scenario-matrix.jsonl", "tool-failure-breakdown.json"].sort());
  assert.doesNotMatch(bundle, /private caller words|private-recording-url|fixture-only/);
  run("s2s-cohort-registry.mjs", ["validation", "--registry", registry, "--key", row.key, "--status", "clear", "--export", out]);
  const stored = JSON.parse(await readFile(registry, "utf8"));
  assert.match(stored.entries[0].validation.export_manifest_sha256, /^[0-9a-f]{64}$/);
});

test("missing agent metadata is counted and prevents registry clear", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-missing-metadata-"));
  const { registry, raw, definitions, campaign } = await fixture(dir);
  const registryData = JSON.parse(await readFile(registry, "utf8"));
  registryData.entries[0].settings.frequency = 2;
  await json(registry, registryData);
  await json(campaign, { repeats: 2, decisions: { build: "fixture" }, models: { example: { short: "Example", name: "Example", vendor: "Fixture", setting: "Fixture" } }, suites: [] });
  await json(join(raw, "appointments", "result-99.json"), { id: 99, status: "completed", runs: [{ id: 101 }, { id: 102 }] });
  const second = JSON.parse(await readFile(join(raw, "appointments", "runs", "101.json"), "utf8"));
  second.source.run_id = 102; second.run.id = 102; second.run.started_at = "2026-01-01T00:00:01Z"; second.run.provider_call_details = {};
  await json(join(raw, "appointments", "runs", "102.json"), second);
  const out = join(dir, "safe");
  const summary = JSON.parse(run("s2s-export-cohort.mjs", ["--registry", registry, "--campaign", campaign, "--definitions", definitions, "--raw", raw, "--out", out]));
  assert.equal(summary.missing_agent_records, 1);
  const benchmark = JSON.parse(await readFile(join(out, "s2s-benchmark.json"), "utf8"));
  assert.equal(benchmark.results[0].missingAgentRecords, 1);
  const refused = result("s2s-cohort-registry.mjs", ["validation", "--registry", registry, "--key", row.key, "--status", "clear", "--export", out]);
  assert.notEqual(refused.status, 0);
  assert.match(refused.stderr, /failed run count, metadata, configuration, or settings check/);
});

test("all four bin modules can be imported without running their CLIs", async () => {
  for (const name of ["s2s-campaign", "s2s-cohort-registry", "s2s-export-cohort", "s2s-reduce-evidence"]) await import(`../bin/${name}.mjs`);
});

test("evidence reducer accepts scenario IDs as arguments and snapshots prior tools at VAD", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-reducer-"));
  const input = join(dir, "evidence.jsonl"), output = join(dir, "verdicts.json");
  const coverage = { as58_complete: true, tool_result_complete: true, response_group_complete: true, vad_complete: true };
  const as58 = { result_id: 1, run_id: 2, scenario_id: 1, provider: "example", coverage, events: [{ type: "as58_pause", at_ms: 1000, pause_end_ms: 3000 }] };
  const ms72 = { result_id: 3, run_id: 4, scenario_id: 2, provider: "example", coverage, events: [{ type: "vad_interruption", at_ms: 0 }, { type: "tool_call", at_ms: 1000, tool_name: "check", model_response_id: "r1" }, { type: "tool_call", at_ms: 2000, tool_name: "check", model_response_id: "r2" }] };
  const ms74 = { result_id: 3, run_id: 5, scenario_id: 3, provider: "example", coverage, events: [{ type: "tool_call", at_ms: 0, tool_name: "check", model_response_id: "r1" }, { type: "vad_interruption", at_ms: 1000 }, { type: "tool_call", at_ms: 2000, tool_name: "check", model_response_id: "r2" }] };
  await writeFile(input, [as58, ms72, ms74].map(JSON.stringify).join("\n") + "\n");
  run("s2s-reduce-evidence.mjs", ["--input", input, "--out", output, "--as58-scenario", "1", "--ms72-scenario", "2", "--ms74-scenario", "3"]);
  const reduced = JSON.parse(await readFile(output, "utf8"));
  assert.equal(reduced.rows[0].as58.verdict, "no phone action");
  assert.equal(reduced.rows[1].patterns.same_tool_within_5s_after_vad.count, 0);
  assert.equal(reduced.rows[2].patterns.same_tool_within_5s_after_vad.count, 1);
  await writeFile(input, `${JSON.stringify({ ...as58, transcript: "must not enter reducer" })}\n`);
  const unsafe = result("s2s-reduce-evidence.mjs", ["--input", input, "--out", join(dir, "unsafe.json"), "--as58-scenario", "1", "--ms72-scenario", "2", "--ms74-scenario", "3"]);
  assert.notEqual(unsafe.status, 0);
  assert.match(unsafe.stderr, /unexpected or unsafe field/);
});
