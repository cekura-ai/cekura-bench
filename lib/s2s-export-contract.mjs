import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

const run = promisify(execFile);
const AXES = ["overall", "task", "infra", "integrity", "toolStrict", "toolEquivalent"];
const INTEGRITY_KEYS = ["replies_drifting", "turns_unanswered", "audio_in_starved", "silent_call", "hangup_held", "service_error", "reply_stalled", "caller_audio_unacknowledged"];
const SERVICE_FAILURES = new Set(["service_error", "reply_stalled", "caller_audio_unacknowledged"]);

export const percentile = (values, q) => {
  const ordered = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!ordered.length) return null;
  const point = (ordered.length - 1) * q;
  const low = Math.floor(point), high = Math.ceil(point);
  return ordered[low] + (ordered[high] - ordered[low]) * (point - low);
};

const number = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
const list = (value) => Array.isArray(value) ? value : value == null ? [] : [value];
const value = (object, ...paths) => {
  for (const path of paths) {
    let current = object;
    for (const part of path.split(".")) current = current?.[part];
    if (current !== undefined && current !== null) return current;
  }
  return null;
};
const metricName = (metric) => String(value(metric, "name", "metric_name") ?? "").toLowerCase();
const metric = (runRecord, pattern) => list(value(runRecord, "evaluation.metrics", "metrics")).find((item) => pattern.test(metricName(item))) ?? null;
const metricPass = (item, top) => {
  if (!item) return null;
  if (typeof item.thumbs_up === "boolean") return item.thumbs_up;
  const score = number(value(item, "score", "score_normalized", "enum"));
  return score === null ? null : score === top;
};
const iso = (run) => value(run, "started_at", "start_time", "created_at", "created", "start") ?? null;
const timestamp = (run) => { const parsed = Date.parse(iso(run) ?? ""); return Number.isFinite(parsed) ? parsed : Number.MAX_SAFE_INTEGER; };
const rowId = (metadata, fallback) => metadata?.config ?? metadata?.s2s_provider ?? fallback;
const suiteId = (metadata, fallback) => metadata?.agent_definition ?? fallback;
const clean = (value) => value === undefined ? null : value;
const slug = (value) => String(value ?? "unknown").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "unknown";

export function repeatIndices(records) {
  const groups = new Map();
  for (const record of records) {
    const key = `${record.row}\u0000${record.suite}\u0000${record.scenarioId}`;
    groups.set(key, [...(groups.get(key) ?? []), record]);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => timestamp(a.platform) - timestamp(b.platform) || String(a.runId).localeCompare(String(b.runId)));
    group.forEach((record, index) => { record.repeat = index + 1; });
  }
  return records;
}

export function bootstrap(groups, statistic, seed = 20260923, resamples = 10_000) {
  if (!groups.length) return [null, null];
  let state = seed >>> 0;
  const random = () => { state |= 0; state = state + 0x6D2B79F5 | 0; let t = Math.imul(state ^ state >>> 15, 1 | state); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const values = [];
  for (let sample = 0; sample < resamples; sample += 1) {
    const selected = Array.from({ length: groups.length }, () => groups[Math.floor(random() * groups.length)]);
    values.push(statistic(selected));
  }
  return [percentile(values, 0.025), percentile(values, 0.975)];
}

async function files(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => entry.isDirectory() ? files(join(root, entry.name)) : [join(root, entry.name)]));
  return nested.flat();
}

export async function loadRaw(root) {
  const paths = (await files(root)).filter((path) => /\/(appointments|medicare)\/runs\/[^/]+\.json$/.test(path));
  const records = [];
  for (const path of paths) {
    const raw = JSON.parse(await readFile(path, "utf8"));
    const platform = raw.run ?? raw;
    const source = raw.source ?? {};
    const metadata = value(platform, "provider_call_details.custom_metadata", "custom_metadata");
    records.push({
      platform, metadata: metadata && typeof metadata === "object" ? metadata : null,
      runId: clean(source.run_id ?? platform.id ?? platform.run_id), resultId: clean(source.result_id),
      scenarioId: clean(platform.scenario?.id ?? platform.scenario ?? platform.scenario_id ?? source.scenario_id),
      suite: suiteId(metadata, source.suite), row: rowId(metadata, source.provider), provider: source.provider ?? null,
    });
  }
  return repeatIndices(records);
}

export async function contracts(definitions) {
  const result = {};
  for (const suite of ["appointments", "medicare"]) {
    const path = join(definitions, suite, "expected-tool-calls.json");
    const text = await readFile(path, "utf8");
    result[suite] = { path, sha256: createHash("sha256").update(text).digest("hex"), values: JSON.parse(text) };
  }
  return result;
}

function agentInput(records) {
  return records.filter((record) => record.metadata).map((record) => JSON.stringify({ run_id: record.runId, scenario_id: record.scenarioId, custom_metadata: record.metadata })).join("\n") + "\n";
}

export async function runScorers(records, definitions, diagnostics, root) {
  const input = join(diagnostics, "agent-runs.jsonl");
  const report = join(diagnostics, "agent-report.json");
  const scores = join(diagnostics, "agent-tool-score.json");
  await writeFile(input, agentInput(records));
  const base = resolve(root);
  await run("python3", ["-m", "agent.report", input, "--out", report], { cwd: base });
  await run("python3", ["-m", "agent.tool_score", input, "--definitions", definitions, "--out", scores], { cwd: base });
  return { report: JSON.parse(await readFile(report, "utf8")), scores: JSON.parse(await readFile(scores, "utf8")), paths: { input, report, scores } };
}

function byRunToolScores(input, expected, root) {
  // `agent.tool_score` produces row totals. Per-run scores use its exported pure scorer,
  // keeping the contract and score semantics identical to the checked scorer module.
  return run("python3", ["-c", `import json,sys; from pathlib import Path; from collections import defaultdict; from agent.tool_score import load_expected,load_schemas,score_run,_pair,value_matches,adds_nothing,OPTIONAL; from agent.report import CALL_CONTROL\ndef detail(expected,calls,mode,schemas):\n out=[]; groups=defaultdict(lambda:[[],[]])\n for e in expected: groups[e['name']][0].append(e)\n for c in calls:\n  if not c.get('cancelled') and c.get('name') not in CALL_CONTROL: groups[c.get('name')][1].append(c)\n for name,(want,got) in groups.items():\n  schema=(schemas.get(name) or {}).get('properties') or {}; mask,assignment,costs=_pair(want,got,mode,schema)\n  for i,j in enumerate(assignment):\n   if j is None: out.append({'tool':name,'defect':'unmatched_call','argument':None}); continue\n   target=want[j].get('arguments') or {}; actual=got[i].get('arguments') or {}\n   for key in set(target)|set(actual):\n    if key not in target and (mode=='strict' or not adds_nothing(key,actual[key],actual,schema)): out.append({'tool':name,'defect':'extra_argument','argument':key})\n    elif key in target and key not in actual and target[key] != OPTIONAL: out.append({'tool':name,'defect':'missing_argument','argument':key})\n    elif key in target and key in actual and not value_matches(key,target[key],actual[key],mode): out.append({'tool':name,'defect':'name_value' if key.endswith('_name') else 'other_value','argument':key})\n  for j,e in enumerate(want):\n   if not mask & (1<<j) and not e.get('optional'): out.append({'tool':name,'defect':'missing_call','argument':None})\n return out\nbase=Path(sys.argv[2]); cache={}; out=[]\nfor r in map(json.loads,open(sys.argv[1])):\n suite=r['custom_metadata']['agent_definition']; cache.setdefault(suite,(load_expected(suite,base),load_schemas(suite,base))); d,s=cache[suite]; e=d[str(r['scenario_id'])]; calls=r['custom_metadata'].get('tool_calls') or []; out.append({'run_id':r['run_id'],'scores':{m:score_run(e,calls,m,s).as_dict() for m in ('strict','equivalent')} if e else None,'details':{m:detail(e,calls,m,s) for m in ('strict','equivalent')} if e else {}})\nprint(json.dumps(out))`, input, expected], { cwd: root });
}

function axes(record, perRunScore) {
  const integrity = record.metadata?.integrity?.checks;
  const tool = perRunScore?.scores ?? null;
  return {
    overall: typeof record.platform.success === "boolean" ? record.platform.success : null,
    task: metricPass(metric(record.platform, /expected[ _-]?outcome|task[ _-]?completion/), 5),
    infra: metricPass(metric(record.platform, /infrastructure[ _-]?issues/), 1),
    integrity: integrity === undefined ? null : Array.isArray(integrity) && integrity.length === 1 && integrity[0] === "ok",
    toolStrict: tool ? tool.strict.passed : null,
    toolEquivalent: tool ? tool.equivalent.passed : null,
  };
}

function latency(record) {
  const item = metric(record.platform, /^latency \(in ms\)$|latency/);
  return number(value(item, "score", "score_normalized", "enum"));
}

function serviceFailure(record) {
  return list(record.metadata?.integrity?.checks).some((check) => SERVICE_FAILURES.has(check));
}

function aggregateAxis(records, axis, repeats, seed, resamples) {
  const values = records.map((record) => record.axis[axis]);
  const missing = values.filter((item) => item === null).length;
  const byScenario = new Map();
  for (const record of records) {
    const key = String(record.scenarioId);
    byScenario.set(key, [...(byScenario.get(key) ?? []), record]);
  }
  const groups = [...byScenario.values()].filter((group) => group.length === repeats && group.every((record) => record.axis[axis] !== null));
  const rate = groups.length ? groups.flat().filter((record) => record.axis[axis]).length / (groups.length * repeats) : null;
  const all = groups.map((group) => group.every((record) => record.axis[axis]));
  const passAll = all.filter(Boolean).length;
  const passCounts = Object.fromEntries(Array.from({ length: repeats + 1 }, (_, index) => [index, 0]));
  for (const group of groups) passCounts[group.filter((record) => record.axis[axis]).length] += 1;
  const rateBootstrap = bootstrap(groups, (sample) => sample.flat().filter((record) => record.axis[axis]).length / (sample.length * repeats), seed, resamples);
  const allBootstrap = bootstrap(groups, (sample) => sample.filter((group) => group.every((record) => record.axis[axis])).length / sample.length, seed, resamples);
  return { pass: values.filter(Boolean).length, missing, scenarios: groups.length, passAll, passCounts, intervals: { rate: rateBootstrap, passAll: allBootstrap }, rate };
}

function integrity(records) {
  const count = Object.fromEntries(INTEGRITY_KEYS.map((key) => [key, 0]));
  let ok = 0, missing = 0;
  for (const record of records) {
    const checks = record.metadata?.integrity?.checks;
    if (checks === undefined) { missing += 1; continue; }
    if (checks.length === 1 && checks[0] === "ok") ok += 1;
    for (const check of checks) if (check in count) count[check] += 1;
  }
  return { ok, ...count, missing };
}

function closeCounts(records) {
  const count = { agent: 0, harness: 0, callerOrTimeout: 0 };
  for (const record of records) {
    const value = record.metadata?.integrity?.closed_by;
    if (value === "agent") count.agent += 1;
    else if (value === "harness") count.harness += 1;
    else count.callerOrTimeout += 1;
  }
  return count;
}

function resolutions(records) {
  const count = { exact: 0, fuzzy: 0, none: 0 };
  for (const record of records) for (const call of list(record.metadata?.tool_calls)) {
    if (call.cancelled || ["end_call", "transfer_call"].includes(call.name)) continue;
    count[call.resolution] = (count[call.resolution] ?? 0) + 1;
  }
  return count;
}

function cost(report, row, suite) {
  const selected = report.rows.find((item) => item.row === row && item.suite === suite)?.cost;
  if (!selected) return { costPerCallUsd: null, costPerMinuteUsd: null, costPublished: false, costNote: "No agent-side cost record was exported." };
  const note = selected.publishable ? "" : `Not published: ${Object.keys(selected.unpriced_runs ?? {}).join(", ") || "unverified rates"}.`;
  return { costPerCallUsd: selected.per_call_usd, costPerMinuteUsd: selected.per_minute_usd, costPublished: selected.publishable, costNote: note };
}

function model(records, campaign) {
  const sample = records.find((record) => record.metadata)?.metadata ?? {};
  const configured = campaign.models?.[records[0].row] ?? {};
  return {
    id: records[0].row, short: configured.short ?? records[0].row, name: configured.name ?? records[0].row,
    vendor: configured.vendor ?? "Unknown", modelId: sample.s2s_model ?? configured.modelId ?? "unknown",
    voice: sample.s2s_voice ?? configured.voice ?? "unknown", setting: configured.setting ?? "As recorded",
    sampleRateHz: sample.pipeline_sample_rate ?? configured.sampleRateHz ?? 0, turns: sample.turn_source ?? configured.turns ?? "unknown",
  };
}

function toolSummary(records) {
  const scored = records.filter((record) => record.perRunScore?.scores);
  const result = { expected: 0, strict: 0, equivalent: 0, scoredRuns: scored.length, strictPassedRuns: 0, equivalentPassedRuns: 0 };
  for (const record of scored) for (const mode of ["strict", "equivalent"]) {
    const score = record.perRunScore.scores[mode];
    result.expected += mode === "strict" ? score.expected_calls : 0;
    result[mode] += score.matched_calls;
    result[`${mode}PassedRuns`] += score.passed ? 1 : 0;
  }
  return result;
}

function toolBreakdown(records) {
  const out = {};
  for (const record of records) {
    for (const mode of ["strict", "equivalent"]) {
      for (const entry of record.perRunScore?.details?.[mode] ?? []) {
        const tool = entry.tool ?? "unknown";
        const bucket = out[mode] ??= {};
        const item = bucket[tool] ??= { defects: {}, arguments: {} };
        item.defects[entry.defect] = (item.defects[entry.defect] ?? 0) + 1;
        if (entry.argument) item.arguments[entry.argument] = (item.arguments[entry.argument] ?? 0) + 1;
      }
    }
  }
  return out;
}

export async function build({ raw, definitions, campaign, output, root = process.cwd() }) {
  const records = await loadRaw(raw);
  const contractsBySuite = await contracts(definitions);
  if (!records.length) throw new Error(`No raw run files found under ${raw}.`);
  for (const record of records) {
    if (!record.metadata) throw new Error(`Run ${record.runId} has no agent record; raw capture is incomplete.`);
    if (!contractsBySuite[record.suite]?.values?.[String(record.scenarioId)]) throw new Error(`No expected-call contract for ${record.suite} scenario ${record.scenarioId}.`);
  }
  try {
    if ((await readdir(output)).length) throw new Error(`Refusing to overwrite non-empty export directory: ${output}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(output, { recursive: true });
  const diagnostics = join(output, "local-diagnostics"); await mkdir(diagnostics, { recursive: true });
  const scores = await runScorers(records, definitions, diagnostics, root);
  const scoreOutput = await byRunToolScores(scores.paths.input, definitions, root);
  const perRun = new Map(JSON.parse(scoreOutput.stdout).map((item) => [String(item.run_id), item]));
  records.forEach((record) => { record.perRunScore = perRun.get(String(record.runId)) ?? null; record.axis = axes(record, record.perRunScore); record.serviceFailure = serviceFailure(record); record.latencyMs = latency(record); });
  const repeats = Number(campaign.repeats);
  if (!Number.isInteger(repeats) || repeats < 1) throw new Error("campaign.repeats must be a positive integer.");
  const method = { latencyPercentile: "linear interpolation (numpy default)", bootstrap: { resamples: 10000, seed: 20260923, unit: "scenario cluster" }, repeatIndex: "sort each (row, suite, scenario) by platform start time, then run id" };
  const resultGroups = new Map();
  for (const record of records) { const key = `${record.suite}\u0000${record.row}`; resultGroups.set(key, [...(resultGroups.get(key) ?? []), record]); }
  const results = [], matrix = [], breakdown = [];
  for (const group of resultGroups.values()) {
    const [first] = group;
    const axesByName = Object.fromEntries(AXES.map((axis) => [axis, aggregateAxis(group, axis, repeats, 20260923, 10_000)]));
    const latencies = group.map((record) => record.latencyMs).filter(Number.isFinite);
    const perRepeat = Array.from({ length: repeats }, (_, index) => percentile(group.filter((record) => record.repeat === index + 1).map((record) => record.latencyMs), 0.5));
    const clean = group.filter((record) => !record.serviceFailure);
    const reduced = Object.fromEntries(["overall", "task", "infra", "integrity"].map((axis) => [axis, aggregateAxis(clean, axis, repeats, 20260923, 10_000).pass]));
    const tool = toolSummary(group), agentCost = cost(scores.report, first.row, first.suite);
    const agreement = group.reduce((total, record) => {
      const platform = metricPass(metric(record.platform, /tool[ _-]?call[ _-]?accuracy/), 5);
      return platform === null || record.axis.toolStrict === null ? total : { compared: total.compared + 1, agreed: total.agreed + Number(platform === record.axis.toolStrict) };
    }, { compared: 0, agreed: 0 });
    results.push({ suite: first.suite, model: first.row, runs: group.length, scenarios: new Set(group.map((record) => record.scenarioId)).size,
      overallPass: axesByName.overall.pass, taskPass: axesByName.task.pass, infraClean: axesByName.infra.pass,
      toolCalls: tool, latencyP50Ms: percentile(latencies, 0.5), latencyP90Ms: percentile(latencies, 0.9), latencyP95Ms: percentile(latencies, 0.95), latencyP50ByRepeat: perRepeat,
      integrity: integrity(group), closedBy: closeCounts(group), toolResolutions: resolutions(group), ...agentCost,
      missing: Object.fromEntries(AXES.map((axis) => [axis, axesByName[axis].missing])),
      passAll: Object.fromEntries(AXES.map((axis) => [axis, axesByName[axis].passAll])),
      passCounts: Object.fromEntries(AXES.map((axis) => [axis, axesByName[axis].passCounts])),
      intervals: Object.fromEntries(AXES.map((axis) => [axis, axesByName[axis].intervals])),
      serviceFailureRuns: group.filter((record) => record.serviceFailure).length, excludingServiceFailures: { runs: clean.length, ...reduced }, toolPlatformAgreement: agreement,
    });
    breakdown.push({ row: first.row, suite: first.suite, defects: toolBreakdown(group) });
    for (const record of group) matrix.push({ row: record.row, suite: record.suite, scenario_id: record.scenarioId, repeat: record.repeat, run_id: record.runId, overall: record.axis.overall, task: record.axis.task, infra: record.axis.infra, integrity: record.axis.integrity, tool_strict: record.axis.toolStrict, tool_equivalent: record.axis.toolEquivalent, service_failure: record.serviceFailure, closed_by: record.metadata?.integrity?.closed_by ?? null, latency_ms: record.latencyMs });
  }
  const allMetadata = records.map((record) => record.metadata);
  const values = (field) => [...new Set(allMetadata.map((metadata) => metadata?.[field]).filter((item) => item != null))];
  const oneValue = (field) => { const found = values(field); if (found.length !== 1) throw new Error(`Website build must have one ${field}; found ${found.length}.`); return found[0]; };
  const starts = records.map((record) => iso(record.platform)).filter(Boolean).sort();
  const models = [...new Map([...resultGroups.values()].map((group) => { const item = model(group, campaign); return [item.id, item]; })).values()];
  const website = { schemaVersion: 2, generatedAt: new Date().toISOString(), window: { firstCallUtc: starts[0] ?? null, lastCallUtc: starts.at(-1) ?? null }, build: { agentCommit: oneValue("agent_commit"), pipecatVersion: oneValue("pipecat_version"), cekuraVersion: oneValue("cekura_version") }, repeats, calls: records.length, methods: method, campaign: campaign.decisions ?? {}, suites: campaign.suites ?? [], models, results };
  const manifest = { schemaVersion: 1, generatedAt: website.generatedAt, results: Object.fromEntries([...resultGroups].map(([key, group]) => [key, { resultIds: [...new Set(group.map((record) => record.resultId))], expectedRuns: campaign.expectedRuns?.[key] ?? null, exportedRuns: group.length }])), build: website.build, window: website.window, campaign: website.campaign, exporterRef: campaign.exporterRef ?? null, expectedContracts: Object.fromEntries(Object.entries(contractsBySuite).map(([suite, contract]) => [suite, { sha256: contract.sha256 }])), methods: method, errors: [] };
  await Promise.all([writeFile(join(output, "s2s-benchmark.json"), JSON.stringify(website, null, 2) + "\n"), writeFile(join(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n"), writeFile(join(output, "scenario-matrix.jsonl"), matrix.map(JSON.stringify).join("\n") + "\n"), writeFile(join(output, "tool-failure-breakdown.json"), JSON.stringify(breakdown, null, 2) + "\n")]);
  return { website, manifest, matrix, breakdown, diagnostics: scores.paths };
}
