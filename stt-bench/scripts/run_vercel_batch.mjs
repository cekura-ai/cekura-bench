// Run one batch (whole-file, accuracy-only) model on Vercel Sandbox from a clean pushed commit.
//   node scripts/run_vercel_batch.mjs --model <name> --run-id <id> [--live]
// Without --live nothing remote happens. Saved receipts prevent a duplicate run: a
// launch that stopped part-way continues from its last finished stage when rerun.
import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { digestFile, dispatchOnce } from './vercel_models.mjs';

const REMOTE = '/vercel/sandbox/stt-bench-batch';
const DATA_HOSTS = ['huggingface.co', '*.huggingface.co', '*.hf.co'];

const { values: args } = parseArgs({ options: {
  model: { type: 'string' }, 'run-id': { type: 'string' }, live: { type: 'boolean', default: false },
}});
const runId = args['run-id'];
if (!args.model || !runId || !/^[A-Za-z0-9_-]+$/.test(runId)) throw new Error('--model and a plain --run-id are required');
const plan = JSON.parse(await readFile('config/vercel-models.json', 'utf8'));
const config = JSON.parse(await readFile(join('config/batch-models', args.model + '.json'), 'utf8'));
if (config.model_id !== args.model) throw new Error('Batch model config does not match selector');
const git = (...a) => execFileSync('git', a, { encoding: 'utf8' }).trim();
const commit = git('rev-parse', 'HEAD');
if (git('status', '--porcelain', '--', '.')) throw new Error('stt-bench has uncommitted changes; a run uses a pushed commit');
if (!git('branch', '-r', '--contains', commit)) throw new Error('HEAD is not pushed; the sandbox clones it');
const repo = git('remote', 'get-url', 'origin').replace(/^git@github\.com:/, 'https://github.com/');
const allow = [...plan.setupDomains, ...DATA_HOSTS, new URL(config.endpoint).hostname];
const name = `stt-batch-${runId}`.toLowerCase().slice(0, 60);
if (!args.live) { console.log(JSON.stringify({ name, model: args.model, commit, region: plan.region, allow }, null, 2)); process.exit(0); }

const sdk = process.env.VERCEL_SANDBOX_SDK_DIR;
if (!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR is required');
const { Sandbox } = await import(pathToFileURL(join(sdk, 'dist/index.js')));
const { getAuth } = await import(pathToFileURL(join(sdk, 'dist/auth/index.js')));
const token = process.env.VERCEL_TOKEN || getAuth()?.token;
if (!token) throw new Error('VERCEL_TOKEN or an existing Vercel login is required');
const account = { token, teamId: plan.teamId, projectId: plan.projectId };

const root = join('reports', 'vercel-batch', runId);
await mkdir(root, { recursive: true });
const statePath = join(root, 'launch.json');
let state;
try { state = JSON.parse(await readFile(statePath, 'utf8')); } catch (e) { if (e.code !== 'ENOENT') throw e; }
if (state && (state.commit !== commit || state.model !== args.model)) throw new Error('This run ID belongs to another commit or model');
if (state?.status === 'collected') { console.log('Already collected: ' + root); process.exit(0); }
state ??= { status: 'new', name, model: args.model, commit, commands: [] };
async function save() {
  state.updatedAt = new Date().toISOString();
  await writeFile(statePath + '.tmp', JSON.stringify(state, null, 2) + '\n');
  await rename(statePath + '.tmp', statePath);
}
async function stage(label, params) {
  const receipt = await dispatchOnce(sb, state, save, label, { cwd: REMOTE, ...params });
  await writeFile(join(root, label + '.log'), await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
  receipt.collected = true; state.commands.push({ ...receipt }); await save();
  if (receipt.exitCode !== 0) throw new Error(`${label} failed; inspect ${join(root, label + '.log')}`);
}

let sb;
try {
  if (state.status === 'new') {
    state.status = 'creating'; await save();
    sb = await Sandbox.create({ ...account, name, runtime: 'python3.13', region: plan.region, resources: { vcpus: plan.vcpus },
      timeout: 2 * 3600 * 1000, persistent: true, ports: [], env: {}, networkPolicy: { allow } });
    const clone = await sb.runCommand({ cmd: 'git', args: ['clone', '--quiet', repo, '/vercel/sandbox/repo'] });
    const checkout = await sb.runCommand({ cmd: 'git', args: ['-C', '/vercel/sandbox/repo', 'checkout', '--quiet', commit] });
    const link = await sb.runCommand({ cmd: 'ln', args: ['-s', '/vercel/sandbox/repo/stt-bench', REMOTE] });
    if ([clone, checkout, link].some(c => c.exitCode !== 0)) throw new Error('Checkout of the pushed commit failed');
    state.status = 'setup'; await save();
  } else {
    sb = await Sandbox.get({ ...account, name, resume: true });   // a saved name is authoritative; never create twice
  }
  if (sb.region !== plan.region) throw new Error('Sandbox region differs from the plan');
  if (state.status === 'setup') {
    await stage('setup', { cmd: 'bash', args: ['-c', [
      'python3 -m pip install --quiet --user uv', 'python3 -m uv sync --locked --quiet',
      `.venv/bin/python -m stt_bench.cli prepare-dataset --dataset ${plan.dataset}`,
      '.venv/bin/python -m pytest -q tests/test_batch.py'].join(' && ')] });
    state.status = 'run'; await save();
  }
  if (state.status === 'run') {
    await stage('run', { cmd: '.venv/bin/python', args: ['-u', '-m', 'stt_bench.cli', 'batch-benchmark',
      '--dataset', plan.dataset, '--model', args.model, '--run-id', runId] });
    state.status = 'pack'; await save();
  }
  if (state.status === 'pack') {
    const out = `runs/batch/${plan.dataset}/${args.model}/${runId}`;
    await stage('pack', { cmd: 'bash', args: ['-c', `tar -czf results.tar.gz --exclude=.workflow.lock ${out} && sha256sum results.tar.gz > results.sha256`] });
    for (const file of ['results.tar.gz', 'results.sha256']) await sb.currentSession().downloadFile({ path: `${REMOTE}/${file}` }, { path: join(root, file) });
    if (await digestFile(join(root, 'results.tar.gz')) !== (await readFile(join(root, 'results.sha256'), 'utf8')).split(/\s+/)[0])
      throw new Error('Results checksum mismatch');
    state.status = 'collected'; await save();
    console.log(`Collected ${join(root, 'results.tar.gz')}`);
  }
} finally {
  if (sb) { await sb.stop(); state.computeStopped = true; await save(); }
}
