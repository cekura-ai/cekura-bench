#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, rmdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { displayName, emptyRegistry, fail, isMain, parseFlags, payloadFor, positiveInt, readJson, safeName, validateRegistry, writeJsonAtomic, writeJsonNew } from "../lib/s2s-toolkit.mjs";

const usage = `Usage:
  node bin/s2s-cohort-registry.mjs init --registry path --project-id PROJECT_ID
  node bin/s2s-cohort-registry.mjs import --registry path --entry entry.json
  node bin/s2s-cohort-registry.mjs validation --registry path --key row-key --status pending|clear|stop|unverified [--export safe-dir]
  node bin/s2s-cohort-registry.mjs link --registry path --key row-key --url https://...
  node bin/s2s-cohort-registry.mjs show --registry path
Import requires a result ID, exact settings and a name. It makes no network request.
`;

async function main() {
  const [command, ...argv] = process.argv.slice(2);
  if (!command || command === "--help") return process.stdout.write(usage);
  const flags = parseFlags(argv, ["--registry", "--project-id", "--entry", "--key", "--status", "--url", "--export"]);
  if (!flags["--registry"]) fail("--registry is required");
  const path = resolve(flags["--registry"]);
  if (command === "init") {
    const projectId = Number(flags["--project-id"]);
    positiveInt(projectId, "project-id");
    await writeJsonNew(path, emptyRegistry(projectId));
    return process.stdout.write(`${path}\n`);
  }
  if (command === "show") return process.stdout.write(`${JSON.stringify(validateRegistry(await readJson(path)), null, 2)}\n`);
  if (!["import", "validation", "link"].includes(command)) fail(usage.trim());
  await mkdir(dirname(path), { recursive: true });
  const lock = `${path}.lock`;
  await mkdir(lock);
  try {
    const registry = validateRegistry(await readJson(path));
    if (command === "import") {
      if (!flags["--entry"]) fail("--entry is required");
      const entry = await readJson(flags["--entry"]);
      safeName(entry.key, "entry.key"); positiveInt(entry.result_id, "entry.result_id");
      displayName(entry.name, "entry.name");
      if (!["appointments", "medicare"].includes(entry.suite) || !entry.provider || !entry.settings || typeof entry.settings !== "object") fail("Imported entry requires suite, provider and exact settings");
      if (registry.entries.some((item) => item.key === entry.key || item.result_id === entry.result_id || item.name === entry.name)) fail("Registry already contains key, result ID or name");
      registry.entries.push({ ...entry, validation: { status: "unverified" }, share_links: entry.share_links ?? {} });
    } else {
      const entry = registry.entries.find((item) => item.key === flags["--key"]);
      if (!entry) fail("Unknown --key");
      if (command === "validation") {
        if (!["pending", "clear", "stop", "unverified"].includes(flags["--status"])) fail("Invalid validation status");
        if (flags["--status"] === "clear") {
          if (!flags["--export"]) fail("--status clear requires --export");
          const root = resolve(flags["--export"]);
          const manifestText = await readFile(join(root, "manifest.json"), "utf8");
          const hash = createHash("sha256").update(manifestText).digest("hex");
          const manifest = JSON.parse(manifestText);
          const benchmark = await readJson(join(root, "s2s-benchmark.json"));
          const verification = await readJson(join(root, "registry-verification.json"));
          const result = verification.entries?.find((item) => item.key === entry.key && item.result_id === entry.result_id);
          const group = manifest.results?.[`${entry.suite}\u0000${entry.provider}`];
          if (manifest.schemaVersion !== 1 || benchmark.schemaVersion !== 2 || verification.schema_version !== 1 || verification.project_id !== registry.project_id || verification.manifest_sha256 !== hash || !group?.resultIds?.includes(entry.result_id) || !result) fail("Safe-export contract or result ID mismatch");
          const expected = entry.settings.scenario_ids.length * entry.settings.frequency;
          const expectedSettingsHash = createHash("sha256").update(JSON.stringify(payloadFor({ key: entry.key, name: entry.name, suite: entry.suite, provider: entry.provider, agent_id: entry.settings.agent_id, scenario_ids: entry.settings.scenario_ids, frequency: entry.settings.frequency, concurrency: entry.settings.concurrency, mock_tool_names: [], config: entry.settings.config }, entry.settings.pipecat_agent_name))).digest("hex");
          if (result.run_count !== expected || result.expected_run_count !== expected || result.scenario_distribution_ok !== true || result.missing_agent_records !== 0 || result.configuration_mismatches !== 0 || result.expected_agent_commit_prefix !== (entry.settings.expected_agent_commit_prefix ?? null) || result.settings_sha256 !== expectedSettingsHash || (entry.settings_sha256 && entry.settings_sha256 !== expectedSettingsHash)) fail("Safe export failed run count, metadata, configuration, or settings check");
          entry.validation = { status: "clear", export_manifest_sha256: hash, exported_run_count: expected, updated_at: new Date().toISOString() };
        } else entry.validation = { status: flags["--status"], updated_at: new Date().toISOString() };
      } else {
        const url = new URL(flags["--url"] ?? "");
        if (url.protocol !== "https:") fail("Share link must use HTTPS");
        entry.share_links = { ...(entry.share_links ?? {}), public_report: url.toString() };
      }
    }
    validateRegistry(registry);
    await writeJsonAtomic(path, registry);
    process.stdout.write(`${JSON.stringify({ registry: path, entries: registry.entries.length, command })}\n`);
  } finally { await rmdir(lock); }
}

if (isMain(import.meta.url)) main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
