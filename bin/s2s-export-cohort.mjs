#!/usr/bin/env node

// Use the canonical builder; only its four public files leave private scratch.
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve, sep } from "node:path";
import { build } from "../lib/s2s-export-contract.mjs";
import { fail, isMain, parseFlags, payloadFor, readJson, resultRuns, tally, TERMINAL, validateRegistry } from "../lib/s2s-toolkit.mjs";

const PUBLIC_FILES = ["manifest.json", "s2s-benchmark.json", "scenario-matrix.jsonl", "tool-failure-breakdown.json"];
const usage = `Usage: node bin/s2s-export-cohort.mjs --registry registry.json --campaign presentation.json --definitions private-contracts --out fresh-safe-dir [--key row-key] [--raw local-capture | --fetch] [--zip fresh.zip]
--fetch uses CEKURA_API_KEY and writes raw responses only in disposable private scratch. The ZIP contains the four website-contract files only.
`;

function command(program, args, cwd) {
  return new Promise((accept, reject) => {
    const child = spawn(program, args, { cwd, stdio: "ignore" });
    child.on("error", reject);
    child.on("exit", (code) => code === 0 ? accept() : reject(new Error(`${program} exited ${code}`)));
  });
}
const metadata = (run) => run.provider_call_details?.custom_metadata ?? run.custom_metadata ?? null;
const scenarioId = (run) => Number(run.scenario?.id ?? run.scenario ?? run.scenario_id);
function settingsHash(entry) {
  const s = entry.settings;
  const payload = payloadFor({ key: entry.key, name: entry.name, suite: entry.suite, provider: entry.provider, agent_id: s.agent_id, scenario_ids: s.scenario_ids, frequency: s.frequency, concurrency: s.concurrency, mock_tool_names: [], config: s.config }, s.pipecat_agent_name);
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

async function stageEntry(entry, source, target) {
  const result = await readJson(join(source, entry.suite, `result-${entry.result_id}.json`));
  if (Number(result.id) !== entry.result_id || !TERMINAL.has(String(result.status ?? "").toLowerCase())) fail(`${entry.key}: source result ID/status mismatch`);
  const summaries = resultRuns(result), expected = entry.settings.scenario_ids.length * entry.settings.frequency;
  if (summaries.length !== expected) fail(`${entry.key}: found ${summaries.length} runs, expected ${expected}`);
  const runIds = summaries.map((summary) => Number(summary.id ?? summary.run_id));
  if (runIds.some((id) => !Number.isSafeInteger(id) || id < 1) || new Set(runIds).size !== expected) fail(`${entry.key}: missing or repeated run ID`);
  await mkdir(join(target, entry.suite, "runs"), { recursive: true });
  const observed = []; let missingAgentRecords = 0, configurationMismatches = 0;
  for (const id of runIds) {
    const sourcePath = join(source, entry.suite, "runs", `${id}.json`);
    const raw = await readJson(sourcePath), run = raw.run ?? raw;
    if (Number(run.id) !== id || (raw.source?.result_id != null && Number(raw.source.result_id) !== entry.result_id)) fail(`${entry.key}: run/source identity mismatch`);
    const scenario = scenarioId(run);
    if (!Number.isSafeInteger(scenario)) fail(`${entry.key}: run ${id} has no scenario ID`);
    observed.push(scenario);
    const meta = metadata(run);
    if (!meta || typeof meta !== "object") missingAgentRecords += 1;
    else if ([
      meta.agent_definition !== entry.suite,
      Number(meta.cekura_agent_id) !== entry.settings.agent_id,
      meta.s2s_provider !== entry.provider,
      meta.s2s_model !== entry.settings.config.s2s_model,
      entry.settings.expected_agent_commit_prefix && !String(meta.agent_commit ?? "").startsWith(entry.settings.expected_agent_commit_prefix),
    ].some(Boolean)) configurationMismatches += 1;
    const destination = join(target, entry.suite, "runs", `${id}.json`);
    await writeFile(destination, JSON.stringify({ source: { suite: entry.suite, provider: entry.provider, resultId: entry.result_id, run_id: id }, run }), { flag: "wx", mode: 0o600 });
  }
  const counts = tally(observed);
  const distributionOk = Object.keys(counts).length === entry.settings.scenario_ids.length && entry.settings.scenario_ids.every((id) => counts[id] === entry.settings.frequency);
  if (!distributionOk) fail(`${entry.key}: scenario/repeat distribution mismatch`);
  return { key: entry.key, result_id: entry.result_id, suite: entry.suite, provider: entry.provider, run_count: summaries.length, expected_run_count: expected, scenario_distribution_ok: true, missing_agent_records: missingAgentRecords, configuration_mismatches: configurationMismatches, expected_agent_commit_prefix: entry.settings.expected_agent_commit_prefix ?? null, settings_sha256: settingsHash(entry) };
}

function assertPublicShape(filename, body) {
  const forbidden = new Set(["transcript", "transcript_object", "recording", "recording_url", "voice_recording_url", "audio_url", "execution_log", "execution_logs", "logs", "trace", "traces", "session_id", "session_identifier", "credentials", "secret", "password", "api_key", "caller_profile", "testing_agent_variables", "raw_arguments", "argument_values"]);
  const visit = (value) => {
    if (Array.isArray(value)) return value.forEach(visit);
    if (value && typeof value === "object") for (const [key, child] of Object.entries(value)) {
      if (forbidden.has(key.toLowerCase())) fail(`${filename}: forbidden public field ${key}`);
      visit(child);
    }
  };
  if (filename.endsWith(".jsonl")) body.trim().split("\n").filter(Boolean).forEach((line) => visit(JSON.parse(line)));
  else visit(JSON.parse(body));
}

async function main() {
  const flags = parseFlags(process.argv.slice(2), ["--registry", "--campaign", "--definitions", "--out", "--key", "--raw", "--zip"], ["--fetch", "--help"]);
  if (flags["--help"]) return process.stdout.write(usage);
  if (!["--registry", "--campaign", "--definitions", "--out"].every((key) => flags[key]) || Boolean(flags["--raw"]) === Boolean(flags["--fetch"])) fail(usage.trim());
  const registry = validateRegistry(await readJson(flags["--registry"]));
  const entries = flags["--key"] ? registry.entries.filter((entry) => entry.key === flags["--key"]) : registry.entries;
  if (!entries.length) fail("No registry entries selected");
  const frequency = entries[0].settings.frequency;
  if (entries.some((entry) => entry.settings.frequency !== frequency)) fail("One export requires one frequency; select a uniform cohort");
  const campaign = await readJson(flags["--campaign"]);
  if (campaign.repeats !== frequency) fail("Campaign repeats do not match registry frequency");
  const expectedRuns = {};
  for (const entry of entries) {
    const key = `${entry.suite}\u0000${entry.provider}`;
    expectedRuns[key] = (expectedRuns[key] ?? 0) + entry.settings.scenario_ids.length * frequency;
  }
  for (const [key, count] of Object.entries(expectedRuns)) if (campaign.expectedRuns?.[key] != null && campaign.expectedRuns[key] !== count) fail(`Campaign expectedRuns disagrees for ${key.replace("\u0000", "/")}`);
  const presentation = { ...campaign, expectedRuns };
  const output = resolve(flags["--out"]), zip = flags["--zip"] ? resolve(flags["--zip"]) : null;
  if (zip && (zip === output || zip.startsWith(`${output}${sep}`))) fail("ZIP must be outside output directory");
  for (const path of [output, zip].filter(Boolean)) {
    try { await access(path); fail(`Destination already exists: ${path}`); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  const scratch = await mkdtemp(join(tmpdir(), "cekura-s2s-export-"));
  try {
    const source = flags["--fetch"] ? join(scratch, "capture") : resolve(flags["--raw"]);
    if (flags["--fetch"]) {
      const args = [resolve("bin/export-s2s-results.mjs"), "--out", source, "--no-logs", "--no-traces"];
      for (const entry of entries) args.push("--result", `${entry.suite}:${entry.provider}:${entry.result_id}`);
      await command(process.execPath, args, resolve("."));
    }
    const selected = join(scratch, "selected"), built = join(scratch, "built");
    const verificationEntries = [];
    for (const entry of entries) verificationEntries.push(await stageEntry(entry, source, selected));
    const result = await build({ raw: selected, definitions: resolve(flags["--definitions"]), campaign: presentation, output: built, root: resolve(".") });
    if (result.website.schemaVersion !== 2 || result.website.calls !== verificationEntries.reduce((sum, entry) => sum + entry.run_count, 0)) fail("Contract build count/schema mismatch");
    for (const [key, count] of Object.entries(expectedRuns)) {
      const group = result.manifest.results[key];
      const expectedIds = entries.filter((entry) => `${entry.suite}\u0000${entry.provider}` === key).map((entry) => entry.result_id).sort();
      if (group?.exportedRuns !== count || group?.expectedRuns !== count || JSON.stringify([...group.resultIds].sort()) !== JSON.stringify(expectedIds)) fail("Contract build result IDs/counts mismatch");
      const benchmarkRow = result.website.results.find((item) => `${item.suite}\u0000${item.model}` === key);
      const missing = verificationEntries.filter((entry) => `${entry.suite}\u0000${entry.provider}` === key).reduce((sum, entry) => sum + entry.missing_agent_records, 0);
      if (!benchmarkRow || benchmarkRow.missingAgentRecords !== missing) fail("Contract build missing-agent count mismatch");
    }
    const publicBodies = await Promise.all(PUBLIC_FILES.map(async (name) => {
      const body = await readFile(join(built, name), "utf8"); assertPublicShape(name, body); return body;
    }));
    const manifestHash = createHash("sha256").update(publicBodies[0]).digest("hex");
    await mkdir(dirname(output), { recursive: true });
    await mkdir(output);
    await Promise.all(PUBLIC_FILES.map((name, index) => writeFile(join(output, name), publicBodies[index], { flag: "wx" })));
    await writeFile(join(output, "registry-verification.json"), `${JSON.stringify({ schema_version: 1, project_id: registry.project_id, manifest_sha256: manifestHash, entries: verificationEntries }, null, 2)}\n`, { flag: "wx" });
    if (zip) await command("zip", ["-X", "-q", zip, ...PUBLIC_FILES], output);
    process.stdout.write(`${JSON.stringify({ output, zip, results: entries.length, runs: result.website.calls, missing_agent_records: verificationEntries.reduce((sum, entry) => sum + entry.missing_agent_records, 0) })}\n`);
  } finally { await rm(scratch, { recursive: true, force: true }); }
}

if (isMain(import.meta.url)) main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
