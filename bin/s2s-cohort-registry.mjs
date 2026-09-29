#!/usr/bin/env node

import { mkdir, rmdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { displayName, emptyRegistry, fail, parseFlags, positiveInt, readJson, safeName, validateRegistry, writeJsonAtomic, writeJsonNew } from "../lib/s2s-toolkit.mjs";

const usage = `Usage:
  node bin/s2s-cohort-registry.mjs init --registry path --project-id 8197
  node bin/s2s-cohort-registry.mjs import --registry path --entry entry.json
  node bin/s2s-cohort-registry.mjs validation --registry path --key row-key --status pending|clear|stop|unverified
  node bin/s2s-cohort-registry.mjs link --registry path --key row-key --url https://...
  node bin/s2s-cohort-registry.mjs show --registry path
Import requires a result ID, exact settings and a name. It makes no network request.
`;

async function main() {
  const [command, ...argv] = process.argv.slice(2);
  if (!command || command === "--help") return process.stdout.write(usage);
  const flags = parseFlags(argv, ["--registry", "--project-id", "--entry", "--key", "--status", "--url"]);
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
      registry.entries.push({ ...entry, validation: entry.validation ?? { status: "unverified" }, share_links: entry.share_links ?? {} });
    } else {
      const entry = registry.entries.find((item) => item.key === flags["--key"]);
      if (!entry) fail("Unknown --key");
      if (command === "validation") {
        if (!["pending", "clear", "stop", "unverified"].includes(flags["--status"])) fail("Invalid validation status");
        entry.validation = { status: flags["--status"], updated_at: new Date().toISOString() };
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

main().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
