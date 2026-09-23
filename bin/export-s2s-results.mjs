#!/usr/bin/env node

/** Capture raw S2S result evidence for the local contract builder.
 *
 * Read-only. It never starts a run and never requests recordings. This output
 * can contain transcripts, logs, traces and tool argument values, so it is
 * intentionally ignored by git and must remain local.
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

const API = "https://api.cekura.ai/test_framework/v1";
const RECORDING = /(^|_)(recording|recordings|recording_url|voice_recording|waveform|audio_url|audio_download_url)($|_)/i;
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function usage() { return `Usage: npm run export:s2s -- --result <suite:provider:result-id> [--result ...] --out <directory>\n`; }
function parse(argv) {
  const options = { api: API, results: [], logs: true, traces: true, concurrency: 8 };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i], next = () => { const value = argv[++i]; if (!value || value.startsWith("--")) throw new Error(`${arg} requires a value.`); return value; };
    if (arg === "--help") options.help = true;
    else if (arg === "--out") options.out = resolve(next());
    else if (arg === "--api-base") options.api = next().replace(/\/$/, "");
    else if (arg === "--concurrency") options.concurrency = Number(next());
    else if (arg === "--no-logs") options.logs = false;
    else if (arg === "--no-traces") options.traces = false;
    else if (arg === "--result") { const [suite, provider, id, extra] = next().split(":"); if (extra || !["appointments", "medicare"].includes(suite) || !provider || !/^\d+$/.test(id ?? "")) throw new Error("--result is suite:provider:result-id"); options.results.push({ suite, provider, resultId: Number(id) }); }
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!options.help && (!options.out || !options.results.length)) throw new Error("--out and at least one --result are required.");
  if (!Number.isInteger(options.concurrency) || options.concurrency < 1 || options.concurrency > 32) throw new Error("--concurrency must be 1 through 32.");
  return options;
}
function noRecordings(value) { if (Array.isArray(value)) return value.map(noRecordings); if (!value || typeof value !== "object") return value; return Object.fromEntries(Object.entries(value).filter(([key]) => !RECORDING.test(key)).map(([key, item]) => [key, noRecordings(item)])); }
async function json(path, value) { await writeFile(path, JSON.stringify(value, null, 2) + "\n"); }
function runs(result) { const items = result?.runs ?? result?.run_details ?? []; return Array.isArray(items) ? items : Object.values(items); }
async function pool(items, limit, work) { let next = 0; await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => { while (next < items.length) { const item = items[next++]; await work(item); } })); }
function client(options) {
  const key = process.env.CEKURA_API_KEY; if (!key) throw new Error("CEKURA_API_KEY is not set.");
  return async (path, optional = false) => {
    let error;
    for (let attempt = 0; attempt < 4; attempt += 1) try {
      const response = await fetch(`${options.api}${path}`, { headers: { Accept: "application/json", "X-CEKURA-API-KEY": key } });
      const text = await response.text();
      if (response.ok) return text ? JSON.parse(text) : null;
      if (optional && response.status === 404) return null;
      if (response.status === 429 || response.status >= 500) { error = new Error(`${path}: HTTP ${response.status}`); await sleep(500 * 2 ** attempt); continue; }
      throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 300)}`);
    } catch (caught) { error = caught; if (attempt < 3) await sleep(500 * 2 ** attempt); }
    throw error;
  };
}
async function main() {
  const options = parse(process.argv.slice(2)); if (options.help) return process.stdout.write(usage());
  try {
    if ((await readdir(options.out)).length) throw new Error(`Refusing to overwrite non-empty export directory: ${options.out}`);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  await mkdir(options.out, { recursive: true });
  const get = client(options), manifest = { schemaVersion: 2, generatedAt: new Date().toISOString(), recordingsFetched: false, results: options.results, errors: [] };
  await pool(options.results, options.concurrency, async (target) => {
    const base = join(options.out, target.suite), runRoot = join(base, "runs"), logRoot = join(base, "logs"), traceRoot = join(base, "traces");
    await Promise.all([mkdir(runRoot, { recursive: true }), mkdir(logRoot, { recursive: true }), mkdir(traceRoot, { recursive: true })]);
    try {
      const result = await get(`/results/${target.resultId}/`); await json(join(base, `result-${target.resultId}.json`), noRecordings(result));
      await pool(runs(result), options.concurrency, async (summary) => {
        const runId = summary.id ?? summary.run_id;
        try {
          const record = await get(`/runs/${runId}/`); await json(join(runRoot, `${runId}.json`), { source: { ...target, run_id: runId }, run: noRecordings(record) });
          if (options.logs && record?.is_log_present) { const log = await get(`/runs/${runId}/logs/`, true); if (log) await json(join(logRoot, `${runId}.json`), log); }
          if (options.traces) { const trace = await get(`/runs/${runId}/trace/`, true); if (trace) await json(join(traceRoot, `${runId}.json`), trace); }
        } catch (error) { manifest.errors.push({ ...target, run_id: runId, error: error.message }); }
      });
    } catch (error) { manifest.errors.push({ ...target, error: error.message }); }
  });
  await json(join(options.out, "raw-manifest.json"), manifest);
  process.stdout.write(`${JSON.stringify({ output: options.out, errors: manifest.errors.length })}\n`);
  if (manifest.errors.length) process.exitCode = 1;
}
main().catch((error) => { process.stderr.write(`${error.stack ?? error.message}\n`); process.exitCode = 1; });
