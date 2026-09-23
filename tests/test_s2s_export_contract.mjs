import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { bootstrap, build, percentile, repeatIndices } from "../lib/s2s-export-contract.mjs";

test("percentiles use linear interpolation and bootstrap is deterministic", () => {
  assert.equal(percentile([1, 2, 3, 4], 0.9), 3.7);
  assert.deepEqual(bootstrap([[1], [2]], (sample) => sample.length, 20260923, 20), bootstrap([[1], [2]], (sample) => sample.length, 20260923, 20));
});

test("repeat indices order a row scenario by start time then run id", () => {
  const records = repeatIndices([
    { row: "row", suite: "appointments", scenarioId: 1, runId: 2, platform: { started_at: "2026-01-01T00:00:02Z" } },
    { row: "row", suite: "appointments", scenarioId: 1, runId: 1, platform: { started_at: "2026-01-01T00:00:01Z" } },
  ]);
  assert.deepEqual(records.map((record) => record.repeat), [2, 1]);
});

test("builder keeps missing metrics separate and excludes raw values from public artifacts", async () => {
  const root = await mkdtemp(join(tmpdir(), "s2s-export-"));
  const raw = join(root, "raw"), definitions = join(root, "definitions"), out = join(root, "out");
  await Promise.all([mkdir(join(raw, "appointments", "runs"), { recursive: true }), mkdir(join(definitions, "appointments"), { recursive: true }), mkdir(join(definitions, "medicare"), { recursive: true })]);
  await writeFile(join(definitions, "appointments", "expected-tool-calls.json"), JSON.stringify({ 1: [{ name: "lookup", arguments: { marker: "fixture-only" } }] }));
  await writeFile(join(definitions, "appointments", "tool-definitions.json"), JSON.stringify([{ name: "lookup", parameters: { properties: { marker: { type: "string" } } } }]));
  await writeFile(join(definitions, "medicare", "expected-tool-calls.json"), JSON.stringify({}));
  await writeFile(join(definitions, "medicare", "tool-definitions.json"), JSON.stringify([]));
  for (const [id, startedAt, expectedMetric] of [[2, "2026-01-01T00:00:02Z", true], [1, "2026-01-01T00:00:01Z", false]]) {
    const metadata = { config: "fixture-row", s2s_provider: "fixture-row", agent_definition: "appointments", agent_commit: "fixture", pipecat_version: "1", cekura_version: "1", integrity: { checks: ["ok"], closed_by: "agent" }, usage: {}, tool_calls: [{ name: "lookup", arguments: { marker: "fixture-only" }, resolution: "exact" }] };
    const metrics = expectedMetric ? [{ name: "Expected Outcome", score: 5 }, { name: "Infrastructure Issues", score: 1 }, { name: "Latency (in ms)", score: 1000 }] : [{ name: "Infrastructure Issues", score: 1 }, { name: "Latency (in ms)", score: 2000 }];
    await writeFile(join(raw, "appointments", "runs", `${id}.json`), JSON.stringify({ source: { suite: "appointments", provider: "fixture-row", resultId: 9, run_id: id }, run: { id, scenario: 1, started_at: startedAt, success: true, evaluation: { metrics }, provider_call_details: { custom_metadata: metadata } } }));
  }
  const result = await build({ raw, definitions, output: out, root: process.cwd(), campaign: { repeats: 2, decisions: { build: "fixture" }, models: { "fixture-row": { vendor: "Fixture" } } } });
  assert.equal(result.website.results[0].missing.task, 1);
  assert.equal(result.website.results[0].latencyP90Ms, 1900);
  const publicText = await readFile(join(out, "s2s-benchmark.json"), "utf8");
  assert.equal(publicText.includes("fixture-only"), false);
});
