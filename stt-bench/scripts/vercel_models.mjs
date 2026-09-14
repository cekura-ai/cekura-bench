// A separate launcher: the historical Nova controller is deliberately untouched.
// Importing this module or omitting --live never contacts Vercel or an STT API.
import { readFile, writeFile, rename, mkdir, open } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { waitForCompletion } from './vercel_command_wait.mjs';

export function validatePlan(plan) {
  if (plan.region !== 'iad1' || plan.vcpus !== 2 || plan.timeoutMs !== 86400000)
    throw new Error('Expected iad1, 2 vCPUs and a 24-hour Pro session');
  if (!Number.isInteger(plan.sessionBudgetSeconds) || plan.sessionBudgetSeconds < 300 || plan.sessionBudgetSeconds > 82800)
    throw new Error('Invalid Pro session budget');
  if (!Number.isInteger(plan.maxConcurrent) || plan.maxConcurrent < 1 ||
      !Number.isInteger(plan.accountConcurrencyLimit) || plan.maxConcurrent > plan.accountConcurrencyLimit)
    throw new Error('Invalid concurrency limits');
  if (!plan.models?.length || new Set(plan.models.map(m => m.model)).size !== plan.models.length)
    throw new Error('Models must be nonempty and unique');
  if (new Set(plan.models.map(m => m.sandbox)).size !== plan.models.length)
    throw new Error('Each model needs a separate sandbox');
  for (const m of plan.models) {
    for (const v of [m.model, m.runId, m.sandbox]) if (!/^[A-Za-z0-9_.-]+$/.test(v)) throw new Error('Invalid identifier');
    if (m.sandbox === 'my-sandbox-013057') throw new Error('Historical Deepgram sandbox is reserved');
  }
  return plan;
}

export function validatePrepared(prepared, plan, bundleHash) {
  if (prepared.status !== 'ready' || !prepared.snapshotId || prepared.bundleHash !== bundleHash ||
      prepared.teamId !== plan.teamId || prepared.projectId !== plan.projectId)
    throw new Error('Preparation snapshot does not match the launch bundle and account');
}

export async function mapBounded(items, count, worker) {
  let next = 0;
  const outcomes = Array(items.length);
  await Promise.all(Array.from({ length: Math.min(count, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      try { outcomes[index] = { status: 'fulfilled', value: await worker(items[index]) }; }
      catch (error) { outcomes[index] = { status: 'rejected', errorType: error.name }; }
    }
  }));
  return outcomes;
}

// A failed wait does not free remote compute. Retain its reservation until stop
// is confirmed so later queued models cannot exceed the account limit.
export function capacityReservations(limit, names = []) {
  const held = new Set(names);
  if (held.size > limit) throw new Error('Existing model sandboxes exceed available capacity');
  return {
    reserve(name) {
      if (!held.has(name) && held.size >= limit) throw new Error('Capacity held by unreconciled sandboxes');
      held.add(name);
    },
    release(name) { held.delete(name); },
  };
}

export async function dispatchOnce(sb, state, save, stage, params, wait = waitForCompletion) {
  let receipt = state.command;
  if (receipt && receipt.stage === stage) {
    if (!receipt.commandId) throw new Error('Uncertain dispatch: inspect remote commands before any retry');
    if (receipt.sessionId !== sb.currentSession().sessionId)
      throw new Error('Command belongs to an earlier session; manual reconciliation required');
  } else {
    if (receipt && !receipt.collected) throw new Error('Previous command has not been collected');
    receipt = { stage, sessionId: sb.currentSession().sessionId, status: 'dispatching' };
    state.command = receipt;
    await save();
    const command = await sb.runCommand({ ...params, detached: true });
    receipt.commandId = command.cmdId;
    receipt.status = 'running';
    await save();
  }
  if (receipt.exitCode === undefined) receipt.exitCode = await wait(sb.currentSession().getCommand ? sb.currentSession() : sb, receipt.commandId);
  receipt.status = 'finished';
  await save();
  return receipt;
}

export async function digestFile(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

export async function upload(sb, path, destination) {
  const file = await open(path, 'r');
  try {
    const size = (await file.stat()).size;
    for (let offset = 0, i = 0; offset < size; i++) {
      const buffer = Buffer.alloc(Math.min(4000000, size - offset));
      let used = 0;
      while (used < buffer.length) {
        const r = await file.read(buffer, used, buffer.length - used, offset + used);
        if (!r.bytesRead) throw new Error('Unexpected bundle EOF');
        used += r.bytesRead;
      }
      await sb.writeFiles([{ path: `${destination}/part-${String(i).padStart(6, '0')}`, content: buffer }]);
      offset += buffer.length;
    }
  } finally { await file.close(); }
}

export async function main(argv = process.argv.slice(2)) {
  const { values: args } = parseArgs({ args: argv, options: {
    plan: { type: 'string', default: 'config/vercel-models.json' },
    bundle: { type: 'string' }, prepared: { type: 'string' }, live: { type: 'boolean', default: false },
    resume: { type: 'boolean', default: false },
  }});
  const plan = validatePlan(JSON.parse(await readFile(args.plan, 'utf8')));
  if (!args.live) {
    console.log(JSON.stringify({ status: 'prepared_no_remote_actions', plan }, null, 2));
    return;
  }
  if (!args.bundle) throw new Error('A freshly built, verified bundle is required');
  const workspace = process.cwd();
  const inventoryModels = JSON.parse(execFileSync(join(workspace, '.venv/bin/python'), ['-m', 'stt_bench.cli', 'models'], {cwd: workspace, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}));
  for (const model of plan.models) {
    const entry = inventoryModels.find(m => m.model_id === model.model);
    if (!entry?.credential_present) throw new Error('A selected model is missing credentials');
    const config = JSON.parse(await readFile(join(workspace, 'config/models', model.model + '.json'), 'utf8'));
    if (new URL(config.endpoint).hostname !== model.apiHost) throw new Error('Provider allowlist does not match model endpoint');
  }
  // Check the bundle against the current working code and complete frozen input inventory.
  execFileSync(join(workspace, '.venv/bin/python'), ['scripts/verify_bundle.py', args.bundle],
    { cwd: workspace, stdio: ['ignore', 'pipe', 'pipe'] });
  const bundleHash = await digestFile(args.bundle);
  if (!args.prepared) throw new Error('A verified preparation snapshot receipt is required');
  const prepared = JSON.parse(await readFile(args.prepared, 'utf8'));
  validatePrepared(prepared, plan, bundleHash);
  const sdkDir = process.env.VERCEL_SANDBOX_SDK_DIR;
  if (!sdkDir) throw new Error('VERCEL_SANDBOX_SDK_DIR is required');
  const { Sandbox } = await import(pathToFileURL(join(sdkDir, 'dist/index.js')));
  const { getAuth } = await import(pathToFileURL(join(sdkDir, 'dist/auth/index.js')));
  const auth = getAuth();
  if (!auth?.token) throw new Error('Existing Vercel CLI login required');
  const account = { token: auth.token, teamId: plan.teamId, projectId: plan.projectId };
  const inventory = await Sandbox.list(account);
  const active = [];
  for await (const item of inventory) if (['pending', 'running', 'stopping', 'snapshotting'].includes(item.status)) active.push(item);
  const ownNames = new Set(args.resume ? plan.models.map(m => m.sandbox) : []);
  const others = active.filter(s => !ownNames.has(s.name)).length;
  const slots = Math.min(plan.maxConcurrent, plan.accountConcurrencyLimit - others);
  if (slots < 1) throw new Error('No sandbox capacity: existing work is left untouched');
  const activeOwn = new Set(active.filter(s => ownNames.has(s.name)).map(s => s.name));
  const capacity = capacityReservations(slots, activeOwn);
  const queued = [...plan.models].sort((a, b) => Number(activeOwn.has(b.sandbox)) - Number(activeOwn.has(a.sandbox)));
  const states = join(workspace, 'reports', 'vercel-models');
  await mkdir(states, { recursive: true });
  const lock = await open(join(states, '.launcher.lock'), 'wx');
  try {
    const outcomes = await mapBounded(queued, slots, async model => {
      const local = join(states, model.runId);
      await mkdir(local, { recursive: true });
      const statePath = join(local, 'controller.json');
      let state;
      try { state = JSON.parse(await readFile(statePath, 'utf8')); }
      catch (e) { if (e.code !== 'ENOENT') throw e; }
      const identity = { model, bundleHash, region: plan.region, teamId: plan.teamId, projectId: plan.projectId };
      if (state && (!args.resume || JSON.stringify(state.identity) !== JSON.stringify(identity)))
        throw new Error('Existing run requires --resume and identical launch inputs');
      state ??= { identity, status: 'new', rounds: 0, commands: [] };
      async function save() {
        state.updatedAt = new Date().toISOString();
        await writeFile(statePath + '.tmp', JSON.stringify(state, null, 2));
        await rename(statePath + '.tmp', statePath);
      }
      let sb;
      try {
        if (state.status === 'complete') {
          if (!state.computeStopped) {
            const done = await Sandbox.get({ ...account, name: model.sandbox, resume: false });
            if (done.status === 'running') await done.stop();
            capacity.release(model.sandbox);
            state.computeStopped = true; await save();
          }
          return state.status;
        }
        if (state.status === 'failed') throw new Error('Failed jobs require diagnosis and a new run ID');
        if (state.status === 'new') {
          capacity.reserve(model.sandbox);
          state.status = 'creating'; await save();
          sb = await Sandbox.create({ ...account, name: model.sandbox,
            source: { type: 'snapshot', snapshotId: prepared.snapshotId },
            region: plan.region, resources: { vcpus: plan.vcpus }, timeout: plan.timeoutMs,
            persistent: true, ports: [], networkPolicy: { allow: [...plan.setupDomains, model.apiHost] }, env: {} });
          state.status = 'verify_snapshot'; await save();
        } else {
          // A saved name is authoritative, including an uncertain create. Never create again.
          sb = await Sandbox.get({ ...account, name: model.sandbox, resume: false });
        }
        if (sb.region !== plan.region || sb.vcpus !== plan.vcpus || sb.memory !== plan.vcpus * 2048) throw new Error('Actual sandbox region or resource mismatch');
        if (state.command && !state.command.collected && !state.command.commandId)
          throw new Error('Uncertain command dispatch requires manual inspection');
        const remote = '/vercel/sandbox/stt-bench-v4';
        if (state.status === 'creating') throw new Error('Creation was uncertain; inspect sandbox before accepting it');
        if (state.status === 'verify_snapshot') {
          const code = `import pathlib,json,hashlib\np=pathlib.Path('input.tar.gz')\nwith p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==${JSON.stringify(bundleHash)}\nfrom stt_bench.huggingface_data import verify_prepared\nfrom stt_bench.catalog import dataset_definition\nverify_prepared(dataset_definition('pipecat-stt-benchmark'))\nassert not pathlib.Path('.env').exists()\n`;
          const receipt = await dispatchOnce(sb, state, save, 'verify_snapshot', { cmd: '.venv/bin/python', cwd: remote, args: ['-c', code] });
          if (receipt.exitCode !== 0) throw new Error('Cloned dataset verification failed');
          receipt.collected = true; state.commands.push(receipt); state.status = 'smoke'; await save();
        }
        const config = JSON.parse(await readFile(join(workspace, 'config/models', model.model + '.json'), 'utf8'));
        // Credentials travel only through per-command env, never command arguments or files.
        const env = JSON.parse(execFileSync(join(workspace, '.venv/bin/python'), ['-c',
          'import json,sys; from stt_bench.credentials import command_environment; print(json.dumps(command_environment(sys.argv[1])))', config.provider],
          { cwd: workspace, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }));
        const report = `${remote}/reports/${plan.dataset}/${model.model}/${model.runId}`;
        while (state.rounds < 100) {
          state.computeStopped = false;
          const phase = state.status === 'smoke' ? 'smoke' : 'full';
          const stage = `${phase}-${state.rounds}`;
          if (!state.command || state.command.collected) {
            capacity.reserve(model.sandbox);
            if (sb.status !== 'running') await sb.resume();
            if ((sb.expiresAt.getTime() - Date.now()) / 1000 < plan.sessionBudgetSeconds + 300) { await sb.stop(); await sb.resume(); }
          }
          const session = sb.currentSession();
          const receipt = await dispatchOnce(sb, state, save, stage, { cmd: '.venv/bin/python', cwd: remote,
            args: ['-u', 'scripts/model_batches.py', '--live', '--dataset', plan.dataset, '--model', model.model,
              '--run-id', model.runId, '--session-id', session.sessionId, '--region', plan.region,
              '--phase', phase, '--budget-seconds', String(plan.sessionBudgetSeconds)], env });
          // Download on success AND failure, before stopping compute or dispatching more work.
          await sb.currentSession().downloadFile({ path: `${report}/artifacts.tar.gz` }, { path: join(local, 'artifacts.tar.gz') });
          await sb.currentSession().downloadFile({ path: `${report}/artifacts.sha256` }, { path: join(local, 'artifacts.sha256') });
          const expected = (await readFile(join(local, 'artifacts.sha256'), 'utf8')).split(/\s+/)[0];
          if (await digestFile(join(local, 'artifacts.tar.gz')) !== expected) throw new Error('Evidence checksum mismatch');
          const progress = JSON.parse((await sb.currentSession().readFileToBuffer({ path: `${report}/batch-state.json` })).toString());
          await writeFile(join(local, 'batch-state.json'), JSON.stringify(progress, null, 2));
          if (phase === 'full' && receipt.exitCode === 0) await sb.currentSession().downloadFile({ path: `${report}/summary.json` }, { path: join(local, 'summary.json') });
          receipt.collected = true; state.commands.push(receipt); state.rounds++;
          if (receipt.exitCode !== 0 || !['smoke_passed', 'session_complete', 'complete'].includes(progress.status)) {
            state.status = 'failed'; await save(); await sb.stop(); capacity.release(model.sandbox); throw new Error('Model job failed; evidence retained');
          }
          state.status = progress.status === 'complete' ? 'complete' : 'full';
          await save();
          if (phase === 'full') {
            await sb.stop();
            capacity.release(model.sandbox);
            state.computeStopped = true; await save();
          }
          if (state.status === 'complete') return state.status;
        }
        throw new Error('Session safety limit reached');
      } catch (e) {
        state.errorType = e.name; state.errorMessage = String(e.message).replace(/Bearer\s+\S+/g, 'Bearer [REDACTED]'); await save();
        // Uncertain in-flight commands and their sandboxes remain untouched.
        throw e;
      }
    });
    await writeFile(join(states, 'outcomes.json'), JSON.stringify(outcomes, null, 2));
    if (outcomes.some(o => o.status === 'rejected')) throw new Error('Some models need attention; inspect saved controller receipts');
  } finally {
    await lock.close();
    // Remove only our transient controller lock, never benchmark evidence.
    const { unlink } = await import('node:fs/promises');
    await unlink(join(states, '.launcher.lock'));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(e => { console.error(`Launcher stopped (${e.name}). Inspect saved receipts; no uncertain dispatch is repeated.`); process.exitCode = 1; });
}
