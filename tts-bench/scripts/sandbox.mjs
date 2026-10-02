// Run a TTS campaign on Vercel Sandbox, one sandbox per provider, all in one region.
//
//   node scripts/sandbox.mjs plan    --label campaign-a                      # what would run; no remote calls
//   node scripts/sandbox.mjs prepare --live                                  # snapshot this commit, tests passing
//   node scripts/sandbox.mjs launch  --live --label campaign-a --env <.env>  # run, score, pack, download, verify
//   node scripts/sandbox.mjs status  --label campaign-a [--remote]
//   node scripts/sandbox.mjs stop    --live --label campaign-a
//
// Every measurement happens from the same place (the configured region, recorded
// as each run's site), on a clean pushed commit that passed the test suite in
// the sandbox before the snapshot was taken. Keys travel only in the per-command
// environment, never in files or arguments, and each sandbox may reach only its
// provider's hosts and the two transcribers. A model's run inside the sandbox is
// idempotent (bin/campaign-model.py resumes, never repeats a finished cell or a
// returned transcript), so launch can be re-run after any interruption: it
// re-attaches to a command still running in the same session and re-dispatches
// otherwise. A run is only marked collected once its archive's checksum and the
// run's own manifest verify on this machine.
//
// Needs VERCEL_SANDBOX_SDK_DIR (an installed @vercel/sandbox) and a Vercel CLI login.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, readFileSync, statfsSync } from 'node:fs';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const HERE = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REMOTE = '/vercel/sandbox/tts-bench';
const REMOTE_STORE = '/vercel/sandbox/runs';
const REMOTE_OUT = '/vercel/sandbox/out';
const PY = join(HERE, '.venv/bin/python');
const REFRESH_MARGIN_MS = 4 * 3600 * 1000;

// ── local facts ──────────────────────────────────────────────────────────────

function git(...args) {
  return execFileSync('git', args, { cwd: HERE, encoding: 'utf8' }).trim();
}

export function cleanPushedCommit() {
  const commit = git('rev-parse', 'HEAD');
  if (git('status', '--porcelain', '--', '.')) throw new Error('tts-bench has uncommitted changes; a campaign runs on a commit');
  if (!git('branch', '-r', '--contains', commit)) throw new Error('HEAD is not pushed; the sandbox clones it from the remote');
  return commit;
}

export function lineup() {
  // Provider, model, credential name and every host its adapter talks to, from the registry itself.
  const code = `
import json
from urllib.parse import urlsplit
from tts_bench.adapters.base import TTSConfig
from tts_bench.common.events import Clock, EventLog
from tts_bench.registry import PROVIDERS, lineup
out = []
for key, m in lineup():
    entry = PROVIDERS[key]
    adapter = entry.adapter(TTSConfig(model=m.model, voice=m.voice), "unused", EventLog(Clock()), Clock())
    endpoint = getattr(adapter, "url", None) or getattr(adapter, "base", None)
    out.append({"provider": key, "model": m.model, "credential": entry.credential_env, "host": urlsplit(endpoint).hostname})
print(json.dumps(out))`;
  return JSON.parse(execFileSync(PY, ['-c', code], { cwd: HERE, encoding: 'utf8' }));
}

// One sandbox per account, not per protocol: two protocols on one key (a
// provider's second endpoint) share its rate limits, so their models run one
// after another in the same sandbox rather than side by side.
export function groups(entries, only) {
  const by = new Map();
  for (const e of entries) {
    if (only && !only.includes(e.provider)) continue;
    const name = e.credential.toLowerCase().replace(/_(api_key|authorization|key)$/, '').replace(/_/g, '-');
    if (!by.has(e.credential)) by.set(e.credential, { name, credential: e.credential, hosts: new Set(), models: [] });
    const g = by.get(e.credential);
    g.hosts.add(e.host);
    g.models.push({ provider: e.provider, model: e.model });
  }
  return [...by.values()].map(g => ({ ...g, hosts: [...g.hosts] }));
}

export function dotenv(path, names) {
  const out = {};
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    const m = line.match(/^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (m && names.includes(m[1]) && m[2]) out[m[1]] = m[2].replace(/^(['"])(.*)\1$/, '$2');
  }
  return out;
}

export function sandboxName(label, provider) {
  return `tts-${label}-${provider}`.toLowerCase().replace(/[^a-z0-9-]+/g, '-').slice(0, 60);
}

async function digestFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

function freeBytes(path) {
  const s = statfsSync(path);
  return s.bavail * s.bsize;
}

async function loadJson(path, fallback) {
  try { return JSON.parse(await readFile(path, 'utf8')); } catch (e) { if (e.code === 'ENOENT') return fallback; throw e; }
}

async function saveJson(path, value) {
  value.updatedAt = new Date().toISOString();
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path + '.tmp', JSON.stringify(value, null, 2) + '\n');
  await rename(path + '.tmp', path);
}

const redact = s => String(s).replace(/Bearer\s+\S+/g, 'Bearer [REDACTED]');

// ── remote ───────────────────────────────────────────────────────────────────

async function sdk(config) {
  const dir = process.env.VERCEL_SANDBOX_SDK_DIR;
  if (!dir) throw new Error('VERCEL_SANDBOX_SDK_DIR is required');
  const { Sandbox } = await import(pathToFileURL(join(dir, 'dist/index.js')));
  const { getAuth, OAuth, updateAuthConfig } = await import(pathToFileURL(join(dir, 'dist/auth/index.js')));
  let auth = getAuth();
  if (!auth?.token) throw new Error('A Vercel CLI login is required');
  // The login's access token lasts hours, a campaign can outlast it, and an
  // expired one fails every poll with 403 while the sandboxes run on. Renew it
  // up front whenever less than REFRESH_MARGIN_MS is left, as the CLI would.
  if (auth.refreshToken && auth.expiresAt && auth.expiresAt.getTime() - Date.now() < REFRESH_MARGIN_MS) {
    const tokens = await (await OAuth()).refreshToken(auth.refreshToken);
    updateAuthConfig({ token: tokens.access_token, refreshToken: tokens.refresh_token ?? auth.refreshToken,
                       expiresAt: new Date(Date.now() + tokens.expires_in * 1000) });
    auth = getAuth();
    console.log(`vercel login renewed until ${auth.expiresAt.toISOString()}`);
  }
  return { Sandbox, account: { token: auth.token, teamId: config.teamId, projectId: config.projectId } };
}

async function waitFor(sb, cmdId) {
  for (let failures = 0; ;) {
    try {
      const command = await sb.getCommand(cmdId, { signal: AbortSignal.timeout(15000) });
      return await command.wait({ signal: AbortSignal.timeout(60000) });
    } catch (error) {
      // These cancel only this read-only wait, never the remote command.
      if (['TimeoutError', 'AbortError'].includes(error.name)) { failures = 0; continue; }
      if (++failures > 8) throw error;
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}

async function run(sb, script, { env = {}, cwd = REMOTE } = {}) {
  const command = await sb.runCommand({ cmd: 'bash', args: ['-lc', script], cwd, env, detached: true });
  const done = await waitFor(sb, command.cmdId);
  return { exitCode: done.exitCode, output: await done.output('both') };
}

function checkResources(sb, config) {
  if (sb.region !== config.region || sb.vcpus !== config.vcpus) throw new Error(`${sb.name}: region or resources differ from the plan`);
}

// ── commands ─────────────────────────────────────────────────────────────────

async function prepare(config, stateDir) {
  const commit = cleanPushedCommit();
  const path = join(stateDir, `prepare-${commit.slice(0, 12)}.json`);
  const state = await loadJson(path, { commit, status: 'new' });
  if (state.status === 'ready') { console.log(`snapshot ${state.snapshotId} already prepared for ${commit}`); return state; }
  const { Sandbox, account } = await sdk(config);
  const name = `tts-prep-${commit.slice(0, 12)}`;
  const sb = state.status === 'new'
    ? await Sandbox.create({ ...account, name, source: { type: 'git', url: config.repo, revision: commit },
        runtime: config.runtime, region: config.region, resources: { vcpus: config.vcpus }, timeout: 3600000,
        ports: [], networkPolicy: { allow: config.setupDomains } })
    : await Sandbox.get({ ...account, name, resume: true });
  state.status = 'setup'; state.sandbox = name; await saveJson(path, state);
  checkResources(sb, config);
  // No key is present here: the snapshot is code, dependencies and a passing test suite, nothing else.
  const setup = await run(sb, [
    'set -e', 'python3 -m pip install --user -q uv', 'python3 -m uv sync --locked -q',
    'test "$(git rev-parse HEAD)" = ' + commit,
    '.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3',
    'test -z "$(git status --porcelain -- .)"',
  ].join('\n'));
  await writeFile(join(stateDir, `prepare-${commit.slice(0, 12)}.log`), setup.output);
  if (setup.exitCode !== 0) { state.status = 'failed'; await saveJson(path, state); await sb.stop(); throw new Error('setup or tests failed in the sandbox; see the prepare log'); }
  const snapshot = await sb.snapshot();
  state.snapshotId = snapshot.snapshotId; state.status = 'ready'; state.tests = setup.output.trim().split('\n').pop();
  await saveJson(path, state);
  console.log(`snapshot ${state.snapshotId} ready for ${commit} (${state.tests})`);
  return state;
}

async function launchGroup(ctx, group) {
  const { config, Sandbox, account, prepared, label, keys, localStore, stateDir } = ctx;
  const path = join(stateDir, label, `${group.name}.json`);
  const state = await loadJson(path, { group: group.name, commit: prepared.commit, models: {} });
  if (state.commit !== prepared.commit) throw new Error(`${group.name}: this label was started on ${state.commit}; one campaign, one commit`);
  if (group.models.every(e => state.models[`${e.provider}/${e.model}`]?.status === 'collected'))
    return { group: group.name, status: 'complete', pending: [] };
  const name = sandboxName(label, group.name);
  const allow = [...new Set([...group.hosts, ...config.scoringHosts])];
  let sb;
  if (!state.sandbox) {
    state.sandbox = name; state.status = 'creating'; await saveJson(path, state);
    sb = await Sandbox.create({ ...account, name, source: { type: 'snapshot', snapshotId: prepared.snapshotId },
      region: config.region, resources: { vcpus: config.vcpus }, timeout: config.timeoutMs, ports: [], persistent: true,
      networkPolicy: { allow }, env: {} });
  } else {
    sb = await Sandbox.get({ ...account, name, resume: true });   // a saved name is authoritative; never create twice
  }
  checkResources(sb, config);
  state.status = 'running'; await saveJson(path, state);
  const env = { ...keys, TTS_BENCH_SITE: config.site, TTS_BENCH_STORE: REMOTE_STORE, PYTHONUNBUFFERED: '1' };
  for (const { provider, model } of group.models) {
    const key = `${provider}/${model}`;
    const m = state.models[key] ??= { status: 'new', attempts: 0 };
    if (m.status === 'collected') continue;
    const session = sb.currentSession().sessionId;
    let finished;
    if (m.commandId && m.sessionId === session && m.status === 'dispatched') {
      finished = await waitFor(sb, m.commandId);                   // re-attach to the command already running
    } else {
      const command = await sb.runCommand({ cmd: '.venv/bin/python', cwd: REMOTE, env, detached: true, args: [
        '-u', 'bin/campaign-model.py', '--provider', provider, '--model', model, '--label', label,
        '--suite', config.suite, '--repeats', String(config.repeats), '--out', REMOTE_OUT] });
      Object.assign(m, { status: 'dispatched', commandId: command.cmdId, sessionId: session, attempts: m.attempts + 1 });
      await saveJson(path, state);
      finished = await waitFor(sb, command.cmdId);
    }
    const output = await finished.output('both');
    const logs = join(stateDir, label, 'logs'); await mkdir(logs, { recursive: true });
    await writeFile(join(logs, `${provider}--${model.replace(/[^A-Za-z0-9._-]+/g, '_')}--${m.attempts}.log`), output);
    const line = output.split('\n').reverse().find(l => l.startsWith('CAMPAIGN_RESULT '));
    m.exitCode = finished.exitCode;
    if (finished.exitCode !== 0 || !line) { m.status = 'failed'; await saveJson(path, state); console.log(`${key}: failed (exit ${finished.exitCode}); log kept`); continue; }
    const result = JSON.parse(line.slice('CAMPAIGN_RESULT '.length));
    Object.assign(m, { status: 'packed', result });
    await saveJson(path, state);
    if (freeBytes(localStore) < result.bytes * 2.5 + 2e9) {
      m.status = 'packed-not-downloaded'; await saveJson(path, state);
      console.log(`${key}: packed in the sandbox but not downloaded, local disk too full`); continue;
    }
    const incoming = join(localStore, '.campaigns', label, 'incoming'); await mkdir(incoming, { recursive: true });
    const archive = join(incoming, result.packed);
    await sb.downloadFile({ path: `${REMOTE_OUT}/${result.packed}` }, { path: archive });
    if (await digestFile(archive) !== result.sha256) throw new Error(`${key}: archive checksum mismatch`);
    const unpack = join(incoming, result.run + '.unpack');
    await rm(unpack, { recursive: true, force: true }); await mkdir(unpack, { recursive: true });
    execFileSync('tar', ['-xzf', archive, '-C', unpack]);
    execFileSync(PY, ['bin/tts-store.py', 'verify', join(unpack, result.run)], { cwd: HERE, stdio: ['ignore', 'pipe', 'pipe'] });
    await rm(join(localStore, result.run), { recursive: true, force: true });
    await rename(join(unpack, result.run), join(localStore, result.run));
    await rm(unpack, { recursive: true, force: true }); await rm(archive, { force: true });
    m.status = 'collected'; m.collectedAt = new Date().toISOString(); await saveJson(path, state);
    console.log(`${key}: ${result.completed}/${result.planned} cells, voids ${result.voids}, collected as ${result.run}`);
  }
  const pending = group.models.map(e => `${e.provider}/${e.model}`).filter(k => state.models[k].status !== 'collected');
  state.status = pending.length ? 'needs-attention' : 'complete';
  await saveJson(path, state);
  await sb.stop();
  state.computeStopped = true; await saveJson(path, state);
  return { group: group.name, status: state.status, pending };
}

async function mapBounded(items, count, worker) {
  let next = 0;
  const out = Array(items.length);
  await Promise.all(Array.from({ length: Math.min(count, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      try { out[i] = await worker(items[i]); }
      catch (e) { out[i] = { group: items[i].group.name, status: 'error', error: redact(e.message) }; }
    }
  }));
  return out;
}

async function main() {
  const { values: a, positionals } = parseArgs({ allowPositionals: true, options: {
    config: { type: 'string', default: join(HERE, 'config/sandbox.json') }, label: { type: 'string' },
    providers: { type: 'string' }, env: { type: 'string' }, store: { type: 'string' },
    live: { type: 'boolean', default: false }, remote: { type: 'boolean', default: false } } });
  const command = positionals[0];
  const config = JSON.parse(await readFile(a.config, 'utf8'));
  if (config.region !== 'iad1' && !process.env.TTS_SANDBOX_ANY_REGION) throw new Error('the campaign is measured from iad1');
  const localStore = resolve(a.store || process.env.TTS_BENCH_STORE || join(HERE, 'data/runs'));
  const stateDir = join(localStore, '.campaigns');
  const only = a.providers?.split(',');
  const plan = () => groups(lineup(), only);

  if (command === 'plan' || (!a.live && ['prepare', 'launch', 'stop'].includes(command))) {
    const g = plan();
    console.log(JSON.stringify({ region: config.region, site: config.site, suite: config.suite, repeats: config.repeats,
      label: a.label, store: localStore, groups: g.map(x => ({ group: x.name, sandbox: a.label && sandboxName(a.label, x.name), models: x.models.map(e => `${e.provider}/${e.model}`), allow: [...new Set([...x.hosts, ...config.scoringHosts])] })) }, null, 2));
    if (command !== 'plan') console.log('\n(no remote action without --live)');
    return;
  }
  if (command === 'prepare') { await prepare(config, stateDir); return; }
  if (!a.label || !/^[A-Za-z0-9._-]+$/.test(a.label)) throw new Error('--label is required: letters, digits, . _ -');
  if (command === 'status') {
    for (const g of plan()) {
      const s = await loadJson(join(stateDir, a.label, `${g.name}.json`), null);
      if (!s) { console.log(`${g.name}: not started`); continue; }
      console.log(`${g.name} [${s.status}] ` + g.models.map(e => `${e.provider}/${e.model}=${s.models[`${e.provider}/${e.model}`]?.status ?? 'new'}`).join(' '));
    }
    return;
  }
  const { Sandbox, account } = await sdk(config);
  if (command === 'stop') {
    for (const g of plan()) {
      try { const sb = await Sandbox.get({ ...account, name: sandboxName(a.label, g.name), resume: false }); if (sb.status === 'running') await sb.stop(); console.log(`${g.name}: stopped`); }
      catch (e) { console.log(`${g.name}: ${redact(e.message)}`); }
    }
    return;
  }
  if (command === 'launch') {
    const commit = cleanPushedCommit();
    const prepared = await loadJson(join(stateDir, `prepare-${commit.slice(0, 12)}.json`), null);
    if (prepared?.status !== 'ready') throw new Error(`no snapshot for ${commit}; run prepare --live first`);
    if (!a.env) throw new Error('--env: the dotenv file holding the provider and transcriber keys');
    const todo = [];
    for (const g of plan()) {
      const keys = dotenv(a.env, [g.credential, ...config.scoringKeys]);
      const missing = [g.credential, ...config.scoringKeys].filter(k => !keys[k]);
      if (missing.length) { console.log(`${g.name}: skipped, missing ${missing.join(', ')}`); continue; }
      todo.push({ group: g, keys });
    }
    const outcomes = await mapBounded(todo, config.parallel, ({ group, keys }) =>
      launchGroup({ config, Sandbox, account, prepared, label: a.label, keys, localStore, stateDir }, group));
    await saveJson(join(stateDir, a.label, 'outcomes.json'), { outcomes });
    for (const o of outcomes) console.log(`${o.group}: ${o.status}${o.pending?.length ? ' pending ' + o.pending.join(',') : ''}${o.error ? ' ' + o.error : ''}`);
    if (outcomes.some(o => o.status !== 'complete')) process.exitCode = 1;
    return;
  }
  throw new Error(`unknown command ${command}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(e => { console.error(`stopped: ${redact(e.message)}`); process.exitCode = 1; });
}
