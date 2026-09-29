import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { payloadFor, validateCampaign } from "../lib/s2s-toolkit.mjs";

const root = resolve(".");
const run = (script, args) => execFileSync(process.execPath, [join(root, "bin", script), ...args], { cwd: root, encoding: "utf8", env: { ...process.env, CEKURA_API_KEY: "" } });
const json = (path, value) => writeFile(path, `${JSON.stringify(value)}\n`);
const row = { key: "example", name: "S2S — example — round 1 — appointments", suite: "appointments", provider: "example", agent_id: 10, scenario_ids: [311705], frequency: 1, concurrency: 1, mock_tool_names: [], config: { agent_dir: "appointments", s2s_provider: "example", s2s_model: "models/example-v1" } };

test("campaign preview is offline and Pipecat payload has exact target/settings", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-campaign-"));
  const campaign = join(dir, "campaign.json");
  await json(campaign, { schema_version: 1, project_id: 8197, pipecat_agent_name: "cekura-s2s", rows: [row] });
  const preview = JSON.parse(run("s2s-campaign.mjs", ["--campaign", campaign, "--registry", join(dir, "registry.json")]));
  assert.equal(preview.mode, "preview");
  assert.equal(preview.plans[0].planned_calls, 1);
  assert.deepEqual(preview.plans[0].payload, payloadFor(row, "cekura-s2s"));
  assert.deepEqual(preview.plans[0].payload.scenarios, [{ scenario: 311705 }]);
  assert.deepEqual(preview.plans[0].payload.mock_tool_names, []);
  assert.throws(() => validateCampaign({ schema_version: 1, project_id: 8197, pipecat_agent_name: "cekura-s2s", rows: [{ ...row, config: { ...row.config, api_key: "never" } }] }), /sensitive config key/);
});

test("registry records validation and rejects a duplicate launch before network", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-registry-"));
  const path = join(dir, "registry.json"), campaign = join(dir, "campaign.json"), entry = join(dir, "entry.json");
  await json(campaign, { schema_version: 1, project_id: 8197, pipecat_agent_name: "cekura-s2s", rows: [row] });
  run("s2s-cohort-registry.mjs", ["init", "--registry", path, "--project-id", "8197"]);
  await json(entry, { key: row.key, name: row.name, result_id: 99, suite: row.suite, provider: row.provider, settings: { agent_id: row.agent_id, scenario_ids: row.scenario_ids, frequency: 1, concurrency: 1, mock_tool_names: [], pipecat_agent_name: "cekura-s2s", config: row.config } });
  run("s2s-cohort-registry.mjs", ["import", "--registry", path, "--entry", entry]);
  run("s2s-cohort-registry.mjs", ["validation", "--registry", path, "--key", row.key, "--status", "clear"]);
  const stored = JSON.parse(await readFile(path, "utf8"));
  assert.equal(stored.entries[0].validation.status, "clear");
  assert.throws(() => run("s2s-campaign.mjs", ["--campaign", campaign, "--registry", path, "--row", row.key, "--execute"]));
});

test("export creates an allowlisted ZIP and never writes raw payload values", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-export-"));
  const registry = join(dir, "registry.json"), raw = join(dir, "raw"), out = join(dir, "safe"), zip = join(dir, "safe.zip");
  await mkdir(join(raw, "appointments", "runs"), { recursive: true });
  await json(registry, { schema_version: 1, project_id: 8197, entries: [{ key: row.key, name: row.name, result_id: 99, suite: row.suite, provider: row.provider, settings: { agent_id: row.agent_id, scenario_ids: row.scenario_ids, frequency: 1, concurrency: 1, mock_tool_names: [], pipecat_agent_name: "cekura-s2s", config: row.config }, validation: { status: "stop" } }] });
  await json(join(raw, "appointments", "result-99.json"), { id: 99, status: "completed", runs: [{ id: 101 }] });
  await json(join(raw, "appointments", "runs", "101.json"), { run: { id: 101, scenario: 311705, status: "completed", success: true, duration: "00:01:30", voice_recording_url: "secret-recording-url", transcript: "private caller words", transcript_object: { text: "private caller words" }, metadata: { agent_commit: "abc123", s2s_model: "models/example-v1", timing: { reply: { count: 1, turns: [{ ms: 100 }] }, endpointing: { count: 1, turns: [{ ms: 200 }] } }, integrity: { checks: ["ok"], closed_by: "agent", stalls: [], recording_stalls: ["private artifact"] }, tool_calls: [{ name: "update_appointment", matched: false, resolution: "fuzzy", arguments: { phone_number: "private number" } }] }, evaluation: { metrics: [{ id: 217255, score: 3 }] } } });
  const result = JSON.parse(run("s2s-export-cohort.mjs", ["--registry", registry, "--raw", raw, "--out", out, "--zip", zip]));
  assert.equal(result.runs, 1);
  const safe = await readFile(join(out, "local-diagnostics", "agent-runs.jsonl"), "utf8");
  assert.doesNotMatch(safe, /private caller words|private number|secret-recording-url|private artifact/);
  assert.match(safe, /phone_number/);
  assert.match(safe, /"s2s_model":"models\/example-v1"/);
  assert.match(safe, /"recording_based_stall_count":1/);
  assert.equal(JSON.parse(await readFile(join(out, "manifest.json"), "utf8")).total_runs, 1);
  const zipList = execFileSync("unzip", ["-Z1", zip], { encoding: "utf8" });
  const zipBody = execFileSync("unzip", ["-p", zip], { encoding: "utf8" });
  assert.match(zipList, /tool-failure-breakdown.json/);
  assert.doesNotMatch(zipList, /raw|transcript/);
  assert.doesNotMatch(zipBody, /private caller words|private number|secret-recording-url|private artifact/);
  const duplicate = spawnSync(process.execPath, [join(root, "bin", "s2s-export-cohort.mjs"), "--registry", registry, "--raw", raw, "--out", out, "--zip", zip], { encoding: "utf8" });
  assert.notEqual(duplicate.status, 0);
  assert.match(duplicate.stderr, /ZIP destination already exists/);
});

test("local reducer reports verified patterns and preserves unknown as unverified", async () => {
  const dir = await mkdtemp(join(tmpdir(), "s2s-reducer-"));
  const input = join(dir, "evidence.jsonl"), output = join(dir, "verdicts.json");
  const coverage = { as58_complete: true, tool_result_complete: true, response_group_complete: true, vad_complete: true };
  const as58 = { result_id: 1, run_id: 2, scenario_id: 311705, provider: "example", coverage, events: [{ type: "as58_pause", at_ms: 1000, pause_end_ms: 3000 }, { type: "phone_action", at_ms: 3200, argument_names: ["phone_number"], used_unspoken_digits: false, caller_finished_number: true }] };
  const ms72 = { result_id: 3, run_id: 4, scenario_id: 311732, provider: "example", coverage, events: [{ type: "tool_call", at_ms: 1000, tool_name: "save_medicare_qualification", model_response_id: "r1" }, { type: "tool_result", at_ms: 1100, tool_name: "save_medicare_qualification", routing_ready: false }, { type: "vad_interruption", at_ms: 2000 }, { type: "tool_call", at_ms: 3000, tool_name: "save_medicare_qualification", model_response_id: "r2" }, { type: "tool_call", at_ms: 3100, tool_name: "route_medicare_call", model_response_id: "r2" }] };
  const ms74 = { result_id: 3, run_id: 5, scenario_id: 311734, provider: "example", coverage: { ...coverage, vad_complete: false }, events: [] };
  await writeFile(input, [as58, ms72, ms74].map(JSON.stringify).join("\n") + "\n");
  run("s2s-reduce-evidence.mjs", ["--input", input, "--out", output]);
  const reduced = JSON.parse(await readFile(output, "utf8"));
  assert.equal(reduced.rows[0].as58.verdict, "waited through pause");
  assert.equal(reduced.rows[1].patterns.route_after_not_ready.count, 1);
  assert.equal(reduced.rows[1].patterns.multi_tool_response.count, 1);
  assert.equal(reduced.rows[1].patterns.same_tool_within_5s_after_vad.count, 1);
  assert.equal(reduced.rows[2].patterns.same_tool_within_5s_after_vad.count, null);
  assert.equal(reduced.counts.unverified_patterns, 1);
  await writeFile(input, `${JSON.stringify({ ...as58, transcript: "must not enter reducer" })}\n`);
  const unsafe = spawnSync(process.execPath, [join(root, "bin", "s2s-reduce-evidence.mjs"), "--input", input, "--out", join(dir, "unsafe.json")], { encoding: "utf8" });
  assert.notEqual(unsafe.status, 0);
  assert.match(unsafe.stderr, /unexpected or unsafe field/);
});
