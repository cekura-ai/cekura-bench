#!/usr/bin/env node

// Input is a local, value-free event extract. Incomplete ordering is unverified, never zero.
import { readFile } from "node:fs/promises";
import { fail, isMain, parseFlags, positiveInt, writeJsonNew } from "../lib/s2s-toolkit.mjs";

const usage = "Usage: node bin/s2s-reduce-evidence.mjs --input local-evidence.jsonl --out safe-verdicts.json --as58-scenario ID --ms72-scenario ID --ms74-scenario ID\nSee docs/s2s-toolkit.md for the exact value-free event schema.\n";
const allowedRun = new Set(["result_id", "run_id", "scenario_id", "provider", "coverage", "events"]);
const allowedCoverage = new Set(["as58_complete", "tool_result_complete", "response_group_complete", "vad_complete"]);
const allowedEvent = new Set(["type", "at_ms", "tool_name", "argument_names", "model_response_id", "routing_ready", "used_unspoken_digits", "caller_finished_number", "pause_end_ms"]);
const validIdentifier = (value) => typeof value === "string" && /^[A-Za-z_][A-Za-z0-9_.:-]{0,79}$/.test(value);
function keys(value, allowed, label) {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some((key) => !allowed.has(key))) fail(`${label}: unexpected or unsafe field`);
}
function validate(run) {
  keys(run, allowedRun, "run");
  for (const id of ["result_id", "run_id", "scenario_id"]) positiveInt(run[id], id);
  if (!validIdentifier(run.provider)) fail("provider must be a safe identifier");
  keys(run.coverage, allowedCoverage, "coverage");
  for (const value of Object.values(run.coverage)) if (typeof value !== "boolean") fail("coverage fields must be booleans");
  if (!Array.isArray(run.events)) fail("events must be an array");
  let previous = -Infinity;
  for (const event of run.events) {
    keys(event, allowedEvent, "event");
    if (!["as58_pause", "phone_action", "tool_call", "tool_result", "vad_interruption"].includes(event.type)) fail("unknown event type");
    if (!Number.isFinite(event.at_ms) || event.at_ms < previous) fail("events must have nondecreasing at_ms");
    previous = event.at_ms;
    if (event.tool_name !== undefined && !validIdentifier(event.tool_name)) fail("unsafe tool name");
    if (event.model_response_id !== undefined && !validIdentifier(event.model_response_id)) fail("unsafe response ID");
    if (event.argument_names !== undefined && (!Array.isArray(event.argument_names) || event.argument_names.some((name) => !validIdentifier(name)))) fail("argument_names must contain field names only");
    for (const flag of ["routing_ready", "used_unspoken_digits", "caller_finished_number"]) if (event[flag] !== undefined && typeof event[flag] !== "boolean") fail(`${flag} must be boolean`);
    if (event.pause_end_ms !== undefined && (!Number.isFinite(event.pause_end_ms) || event.pause_end_ms < event.at_ms)) fail("invalid pause_end_ms");
  }
  return run;
}

export function reduce(run, scenarios) {
  validate(run);
  const base = { result_id: run.result_id, run_id: run.run_id, scenario_id: run.scenario_id, provider: run.provider };
  const events = run.events;
  const isAs58 = run.scenario_id === scenarios.as58;
  const as58 = { verdict: "unverified", argument_names: [] };
  if (isAs58 && run.coverage.as58_complete) {
    const pause = events.find((event) => event.type === "as58_pause");
    const action = events.find((event) => event.type === "phone_action");
    if (pause && !action) as58.verdict = "no phone action";
    if (pause && action && typeof action.used_unspoken_digits === "boolean" && typeof action.caller_finished_number === "boolean") {
      as58.argument_names = action.argument_names ?? [];
      if (action.used_unspoken_digits) as58.verdict = "filled in number";
      else if (action.caller_finished_number && action.at_ms >= pause.pause_end_ms) as58.verdict = "waited through pause";
      else if (!action.caller_finished_number) as58.verdict = "caller did not finish number";
    }
  }

  const result = { ...base, ...(isAs58 ? { as58 } : {}) };
  if (![scenarios.ms72, scenarios.ms74].includes(run.scenario_id)) return result;
  const patterns = {
    route_after_not_ready: { count: null, tool_names: [] },
    multi_tool_response: { count: null, tool_names: [] },
    same_tool_within_5s_after_vad: { count: null, tool_names: [] },
  };
  if (run.coverage.tool_result_complete) {
    let lastRoutingReady = null, count = 0, missingOutcome = false;
    for (const event of events) {
      if (event.type === "tool_result" && event.tool_name === "save_medicare_qualification") {
        if (typeof event.routing_ready !== "boolean") missingOutcome = true;
        else lastRoutingReady = event.routing_ready;
      }
      if (event.type === "tool_call" && event.tool_name === "route_medicare_call" && lastRoutingReady === false) count += 1;
    }
    if (!missingOutcome) patterns.route_after_not_ready = { count, tool_names: count ? ["route_medicare_call"] : [] };
  }
  if (run.coverage.response_group_complete) {
    const groups = new Map();
    const allGrouped = events.filter((event) => event.type === "tool_call").every((event) => event.model_response_id);
    for (const event of events) if (event.type === "tool_call" && event.model_response_id) {
      if (!groups.has(event.model_response_id)) groups.set(event.model_response_id, []);
      groups.get(event.model_response_id).push(event.tool_name);
    }
    const matches = [...groups.values()].filter((names) => names.length >= 2);
    if (allGrouped) patterns.multi_tool_response = { count: matches.length, tool_names: [...new Set(matches.flat())].sort() };
  }
  if (run.coverage.vad_complete) {
    const priorTools = new Set(), repeated = [];
    let interruptionAt = null, toolsBeforeInterruption = new Set();
    for (const event of events) {
      if (event.type === "vad_interruption") { interruptionAt = event.at_ms; toolsBeforeInterruption = new Set(priorTools); }
      if (event.type !== "tool_call") continue;
      if (interruptionAt !== null && event.at_ms - interruptionAt <= 5000 && toolsBeforeInterruption.has(event.tool_name)) repeated.push(event.tool_name);
      priorTools.add(event.tool_name);
    }
    patterns.same_tool_within_5s_after_vad = { count: repeated.length, tool_names: [...new Set(repeated)].sort() };
  }
  return { ...result, patterns };
}

async function main() {
  const flags = parseFlags(process.argv.slice(2), ["--input", "--out", "--as58-scenario", "--ms72-scenario", "--ms74-scenario"], ["--help"]);
  if (flags["--help"]) return process.stdout.write(usage);
  if (!["--input", "--out", "--as58-scenario", "--ms72-scenario", "--ms74-scenario"].every((key) => flags[key])) fail(usage.trim());
  const scenarios = { as58: positiveInt(Number(flags["--as58-scenario"]), "as58-scenario"), ms72: positiveInt(Number(flags["--ms72-scenario"]), "ms72-scenario"), ms74: positiveInt(Number(flags["--ms74-scenario"]), "ms74-scenario") };
  if (new Set(Object.values(scenarios)).size !== 3) fail("Scenario IDs must be distinct");
  const lines = (await readFile(flags["--input"], "utf8")).trim().split("\n").filter(Boolean);
  if (!lines.length) fail("Evidence input is empty");
  const rows = lines.map((line) => reduce(JSON.parse(line), scenarios));
  if (new Set(rows.map((row) => row.run_id)).size !== rows.length) fail("Duplicate run IDs in evidence input");
  const counts = { verified_as58: rows.filter((row) => row.as58?.verdict && row.as58.verdict !== "unverified").length, unverified_as58: rows.filter((row) => row.as58?.verdict === "unverified").length, unverified_patterns: rows.reduce((sum, row) => sum + Object.values(row.patterns ?? {}).filter((pattern) => pattern.count === null).length, 0) };
  await writeJsonNew(flags["--out"], { schema_version: 1, privacy_safe: true, counts, rows });
  process.stdout.write(`${JSON.stringify({ output: flags["--out"], rows: rows.length, ...counts })}\n`);
}

if (isMain(import.meta.url)) main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
