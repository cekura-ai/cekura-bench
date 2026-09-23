#!/usr/bin/env node

/** Build privacy-safe S2S benchmark artifacts from an already-captured raw export.
 *
 * It never calls Cekura, starts a campaign, or fetches recordings. Raw artifacts
 * and scorer diagnostics stay under the output directory's local-diagnostics/
 * directory; copy only the four top-level artifacts into the vault or website.
 */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { build } from "../lib/s2s-export-contract.mjs";

function usage() {
  return `Usage: npm run build:s2s -- --raw <raw-export> --definitions <contracts> --campaign <campaign.json> --out <directory>

The campaign file records user-set campaign decisions, repeat count, expected
run counts, and display labels. It carries no API key, transcript, log, tool
argument value, or production result id.
`;
}

function args(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help") return { help: true };
    if (!["--raw", "--definitions", "--campaign", "--out"].includes(arg)) throw new Error(`Unknown option: ${arg}`);
    const next = argv[++index];
    if (!next || next.startsWith("--")) throw new Error(`${arg} requires a value.`);
    options[arg.slice(2)] = resolve(next);
  }
  for (const required of ["raw", "definitions", "campaign", "out"]) if (!options[required]) throw new Error(`--${required} is required.`);
  return options;
}

async function main() {
  const options = args(process.argv.slice(2));
  if (options.help) return process.stdout.write(usage());
  const campaign = JSON.parse(await readFile(options.campaign, "utf8"));
  const result = await build({ raw: options.raw, definitions: options.definitions, campaign, output: options.out, root: resolve(".") });
  process.stdout.write(`${JSON.stringify({ calls: result.website.calls, rows: result.website.results.length, output: options.out })}\n`);
}

main().catch((error) => { process.stderr.write(`${error.stack ?? error.message}\n`); process.exitCode = 1; });
