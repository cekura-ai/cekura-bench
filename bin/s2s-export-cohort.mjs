#!/usr/bin/env node

// Shareable output is built only from named fields; raw API responses are never written.
import { access, mkdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join, resolve, sep } from "node:path";
import { apiGet, fail, METRICS, numberOrNull, parseFlags, percentile, readJson, resultRuns, tally, TERMINAL, validateRegistry } from "../lib/s2s-toolkit.mjs";

const usage = "Usage: node bin/s2s-export-cohort.mjs --registry registry.json --out fresh-dir [--key row-key] [--raw captured-dir | --fetch] [--zip fresh.zip]\n--fetch reads Cekura using CEKURA_API_KEY. --raw reads the local export-s2s-results layout. Neither mode fetches recordings.\n";
const identifier = (value) => typeof value === "string" && /^[A-Za-z_][A-Za-z0-9_.:-]{0,79}$/.test(value) ? value : null;
const modelName = (value) => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9_./:-]{0,119}$/.test(value) ? value : null;
const idsOnly = (values) => Array.isArray(values) ? values.map(identifier).filter(Boolean) : null;
const seconds = (value) => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const parts = value.split(":").map(Number);
  return parts.length === 3 && parts.every(Number.isFinite) ? parts[0] * 3600 + parts[1] * 60 + parts[2] : null;
};
const score = (run, id) => numberOrNull((run.evaluation?.metrics ?? []).find((item) => Number(item.id) === id)?.score);
const numeric = (value) => numberOrNull(value);
const metricScores = (run) => Object.fromEntries(Object.entries(METRICS).map(([name, id]) => [name, score(run, id)]));
const parity = (block) => Number.isInteger(block?.count) && Array.isArray(block?.turns) ? block.count === block.turns.length : null;
const safeTools = (tools) => Array.isArray(tools) ? tools.map((call) => ({
  name: identifier(call?.name),
  matched: typeof call?.matched === "boolean" ? call.matched : null,
  resolution: identifier(call?.resolution),
  defect_kind: identifier(call?.defect_kind),
  argument_names: Object.keys(call?.arguments ?? {}).map(identifier).filter(Boolean).sort(),
})) : [];

export function safeRun(run, entry, summary) {
  const meta = run?.metadata ?? {};
  const integrity = meta.integrity ?? {};
  const reply = meta.timing?.reply ?? {};
  const endpointing = meta.timing?.endpointing ?? {};
  const replyMs = Array.isArray(reply.turns) ? reply.turns.map((turn) => numeric(turn?.ms)).filter((value) => value !== null) : [];
  const recordingStalls = integrity.recording_stalls;
  return {
    result_id: entry.result_id, run_id: numeric(run?.id ?? summary?.id ?? summary?.run_id),
    suite: entry.suite, provider: entry.provider,
    scenario_id: numeric(typeof run?.scenario === "object" ? run.scenario?.id : run?.scenario ?? summary?.scenario),
    status: identifier(run?.status ?? summary?.status), platform_pass: typeof run?.success === "boolean" ? run.success : null,
    evaluation_status: identifier(run?.evaluation_status), duration_seconds: seconds(run?.duration),
    metrics: metricScores(run),
    provenance: {
      agent_commit: identifier(meta.agent_commit), agent_definition: identifier(meta.agent_definition),
      cekura_agent_id: numeric(meta.cekura_agent_id), system_prompt_sha256: identifier(meta.system_prompt_sha256),
      shared_rules_sha256: identifier(meta.shared_rules_sha256), s2s_provider: identifier(meta.s2s_provider),
      s2s_model: modelName(meta.s2s_model), s2s_voice: identifier(meta.s2s_voice),
    },
    timing: {
      reply_count: numeric(reply.count), reply_turns_length: Array.isArray(reply.turns) ? reply.turns.length : null,
      reply_parity: parity(reply), reply_p50_ms: numeric(reply.p50_ms), reply_p90_ms: numeric(reply.p90_ms),
      endpointing_count: numeric(endpointing.count), endpointing_turns_length: Array.isArray(endpointing.turns) ? endpointing.turns.length : null,
      endpointing_parity: parity(endpointing), endpointing_p50_ms: numeric(endpointing.p50_ms), endpointing_p90_ms: numeric(endpointing.p90_ms),
      reply_turns_ms: replyMs,
    },
    integrity: {
      checks: idsOnly(integrity.checks), closed_by: identifier(integrity.closed_by),
      stall_count: Array.isArray(integrity.stalls) ? integrity.stalls.length : null,
      recording_based_stall_count: Array.isArray(recordingStalls) ? recordingStalls.length : numeric(recordingStalls),
      hangup_tail_ms: numeric(integrity.hangup_tail_ms),
      detailed_record_present: Boolean(run && Number.isSafeInteger(run.id)),
      caller_transcript_present: Boolean(run?.transcript && run?.transcript_object),
    },
    usage: {
      input_audio_tokens: numeric(meta.usage?.input_audio_tokens), output_audio_tokens: numeric(meta.usage?.output_audio_tokens),
      reasoning_tokens: numeric(meta.usage?.reasoning_tokens), call_seconds: numeric(meta.usage?.call_seconds),
    },
    tool_calls: safeTools(meta.tool_calls),
  };
}

function assertSafe(value) {
  const forbidden = new Set(["transcript", "transcript_object", "recording", "recording_url", "voice_recording_url", "audio_url", "execution_log", "execution_logs", "logs", "trace", "traces", "session_id", "session_identifier", "credentials", "secret", "password", "api_key", "caller_profile", "testing_agent_variables", "argument_value", "argument_values"]);
  const visit = (item, path) => {
    if (Array.isArray(item)) return item.forEach((child, index) => visit(child, `${path}[${index}]`));
    if (item && typeof item === "object") for (const [key, child] of Object.entries(item)) {
      if (forbidden.has(key.toLowerCase())) fail(`Unsafe export key ${path}.${key}`);
      visit(child, `${path}.${key}`);
    }
  };
  visit(value, "$ ");
}

function resultSummary(entry, runs) {
  const turnMs = runs.flatMap((run) => run.timing.reply_turns_ms);
  const cases = new Map();
  for (const run of runs) {
    if (!cases.has(run.scenario_id)) cases.set(run.scenario_id, []);
    cases.get(run.scenario_id).push(run);
  }
  const strictCases = [...cases.values()].filter((group) => group.length === entry.settings.frequency && group.every((run) => run.metrics.tool_call_accuracy === 5)).length;
  const strictRuns = runs.filter((run) => run.metrics.tool_call_accuracy === 5).length;
  return {
    result_id: entry.result_id, label: entry.name, suite: entry.suite, provider: entry.provider,
    validation_status: entry.validation?.status ?? "unverified", run_count: runs.length,
    platform_pass_count: runs.filter((run) => run.platform_pass === true).length,
    strict_tool_at_5_count: strictRuns, strict_tool_at_5_share: strictRuns / runs.length,
    scenario_case_count: cases.size, strict_tool_all_repeats_cases: strictCases,
    expected_outcome_at_5_count: runs.filter((run) => run.metrics.expected_outcome === 5).length,
    latency_ms: { samples: turnMs.length, p50: percentile(turnMs, 0.5), p90: percentile(turnMs, 0.9) },
    closed_by: tally(runs.map((run) => run.integrity.closed_by)),
    stalled_calls: runs.filter((run) => (run.integrity.stall_count ?? 0) > 0).length,
    recording_based_stall_calls: runs.filter((run) => (run.integrity.recording_based_stall_count ?? 0) > 0).length,
    recording_stall_coverage: runs.filter((run) => run.integrity.recording_based_stall_count !== null).length,
    missing_record_or_transcript: runs.filter((run) => !run.integrity.detailed_record_present || !run.integrity.caller_transcript_present).length,
    timing_parity_failures: runs.filter((run) => run.timing.reply_parity === false || run.timing.endpointing_parity === false).length,
  };
}

function scenarioMatrix(runs, registry) {
  const groups = new Map();
  for (const run of runs) {
    const key = `${run.result_id}:${run.scenario_id}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(run);
  }
  return [...groups.values()].map((group) => {
    const entry = registry.entries.find((item) => item.result_id === group[0].result_id);
    const strict = group.filter((run) => run.metrics.tool_call_accuracy === 5).length;
    const p50 = group.map((run) => run.timing.reply_p50_ms).filter((value) => value !== null);
    return {
      result_id: group[0].result_id, suite: group[0].suite, provider: group[0].provider, scenario_id: group[0].scenario_id,
      runs: group.length, expected_repeats: entry.settings.frequency,
      platform_passes: group.filter((run) => run.platform_pass === true).length,
      strict_tool_at_5_count: strict,
      strict_tool_all_repeats: group.length === entry.settings.frequency && strict === group.length,
      expected_outcome_at_5_count: group.filter((run) => run.metrics.expected_outcome === 5).length,
      reply_p50_interval_ms: { min: p50.length ? Math.min(...p50) : null, max: p50.length ? Math.max(...p50) : null },
      stalled_calls: group.filter((run) => (run.integrity.stall_count ?? 0) > 0).length,
      closed_by: tally(group.map((run) => run.integrity.closed_by)),
    };
  });
}

function failures(runs) {
  const groups = new Map();
  for (const run of runs) for (const call of run.tool_calls) {
    if (call.matched === true && call.resolution === "exact" && !call.defect_kind) continue;
    const key = JSON.stringify([run.result_id, call.name, call.resolution, call.defect_kind]);
    if (!groups.has(key)) groups.set(key, { result_id: run.result_id, tool_name: call.name, resolution: call.resolution, defect_kind: call.defect_kind, count: 0, argument_names: new Set() });
    const group = groups.get(key); group.count += 1;
    for (const name of call.argument_names) group.argument_names.add(name);
  }
  return [...groups.values()].map((group) => ({ ...group, argument_names: [...group.argument_names].sort() }));
}

async function mapLimit(items, limit, work) {
  const output = new Array(items.length); let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(items.length, limit) }, async () => {
    while (cursor < items.length) { const index = cursor++; output[index] = await work(items[index]); }
  }));
  return output;
}

async function zipFiles(output, destination, names) {
  await new Promise((accept, reject) => {
    const child = spawn("zip", ["-X", "-q", resolve(destination), ...names], { cwd: output, stdio: "ignore" });
    child.on("error", reject);
    child.on("exit", (code) => code === 0 ? accept() : reject(new Error(`zip exited ${code}`)));
  });
}

async function main() {
  const flags = parseFlags(process.argv.slice(2), ["--registry", "--out", "--key", "--raw", "--zip"], ["--fetch", "--help"]);
  if (flags["--help"]) return process.stdout.write(usage);
  if (!flags["--registry"] || !flags["--out"] || Boolean(flags["--raw"]) === Boolean(flags["--fetch"])) fail(usage.trim());
  const registry = validateRegistry(await readJson(flags["--registry"]));
  const entries = flags["--key"] ? registry.entries.filter((entry) => entry.key === flags["--key"]) : registry.entries;
  if (!entries.length) fail("No registry entries selected");
  const output = resolve(flags["--out"]);
  if (flags["--zip"] && (resolve(flags["--zip"]) === output || resolve(flags["--zip"]).startsWith(`${output}${sep}`))) fail("ZIP must be outside output directory");
  const collected = await mapLimit(entries, 4, async (entry) => {
    if (entry.status && !TERMINAL.has(String(entry.status).toLowerCase())) fail(`${entry.key}: registry says nonterminal`);
    const result = flags["--fetch"] ? await apiGet(`/results/${entry.result_id}/`) : await readJson(join(resolve(flags["--raw"]), entry.suite, `result-${entry.result_id}.json`));
    if (Number(result.id) !== entry.result_id) fail(`${entry.key}: result ID mismatch`);
    if (!TERMINAL.has(String(result.status ?? "").toLowerCase())) fail(`${entry.key}: result is not terminal`);
    const summaries = resultRuns(result);
    const expected = entry.settings.scenario_ids.length * entry.settings.frequency;
    if (summaries.length !== expected) fail(`${entry.key}: ${summaries.length} runs, expected ${expected}`);
    const runs = await mapLimit(summaries, 8, async (summary) => {
      const id = summary.id ?? summary.run_id;
      const raw = flags["--fetch"] ? await apiGet(`/runs/${id}/`) : await readJson(join(resolve(flags["--raw"]), entry.suite, "runs", `${id}.json`));
      const record = raw.run ?? raw;
      return safeRun(record, entry, summary);
    });
    if (runs.some((run) => run.run_id === null || run.scenario_id === null)) fail(`${entry.key}: missing run or scenario identifier`);
    if (new Set(runs.map((run) => run.run_id)).size !== expected) fail(`${entry.key}: duplicate run identifiers`);
    const observed = tally(runs.map((run) => run.scenario_id));
    if (Object.keys(observed).length !== entry.settings.scenario_ids.length || entry.settings.scenario_ids.some((id) => observed[id] !== entry.settings.frequency)) fail(`${entry.key}: scenario/repeat distribution does not match registry`);
    return { entry, runs, result: resultSummary(entry, runs) };
  });
  const runs = collected.flatMap((item) => item.runs), results = collected.map((item) => item.result), matrix = scenarioMatrix(runs, registry);
  const toolFailures = { schema_version: 1, privacy: "defect kinds and argument names only", failures: failures(runs) };
  const diagnostics = { schema_version: 1, stall_source: "source integrity metadata; unavailable is not zero", by_result: results.map(({ result_id, stalled_calls, recording_based_stall_calls, recording_stall_coverage, closed_by, missing_record_or_transcript, timing_parity_failures }) => ({ result_id, stalled_calls, recording_based_stall_calls, recording_stall_coverage, closed_by, missing_record_or_transcript, timing_parity_failures })) };
  const manifest = { schema_version: 1, privacy_safe: true, project_id: registry.project_id, generated_at: new Date().toISOString(), result_ids: entries.map((entry) => entry.result_id), total_runs: runs.length, exclusions: ["transcripts", "recordings", "logs", "traces", "caller profiles", "credentials", "session IDs", "tool argument values"], files: ["manifest.json", "s2s-benchmark.json", "scenario-matrix.jsonl", "tool-failure-breakdown.json", "local-diagnostics/summary.json", "local-diagnostics/agent-runs.jsonl"] };
  const benchmark = { schema_version: 1, project_id: registry.project_id, total_runs: runs.length, results };
  for (const item of [manifest, benchmark, matrix, toolFailures, diagnostics, runs]) assertSafe(item);
  if (flags["--zip"]) {
    try { await access(resolve(flags["--zip"])); fail("ZIP destination already exists"); }
    catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  await mkdir(join(output, "local-diagnostics"));
  const json = (path, value) => writeFile(join(output, path), `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
  const jsonl = (path, values) => writeFile(join(output, path), `${values.map((value) => JSON.stringify(value)).join("\n")}\n`, { flag: "wx" });
  await Promise.all([json("manifest.json", manifest), json("s2s-benchmark.json", benchmark), jsonl("scenario-matrix.jsonl", matrix), json("tool-failure-breakdown.json", toolFailures), json("local-diagnostics/summary.json", diagnostics), jsonl("local-diagnostics/agent-runs.jsonl", runs)]);
  if (flags["--zip"]) await zipFiles(output, flags["--zip"], manifest.files);
  process.stdout.write(`${JSON.stringify({ output, zip: flags["--zip"] ?? null, results: entries.length, runs: runs.length })}\n`);
}

main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
