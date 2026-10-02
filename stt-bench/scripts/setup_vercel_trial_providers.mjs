// Setup only: no benchmark command or provider credential is sent remotely.
import {readFile, writeFile, mkdir, rename, open, unlink} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {codeFiles} from './setup_vercel_chirp.mjs';
import {dispatchOnce, validatePlan} from './vercel_models.mjs';

export const MODELS = ['soniox-stt-rt-v5', 'smallest-pulse', 'sarvam-saaras-v3-realtime', 'inworld-stt-1'];
export const REMOTE = '/vercel/sandbox/stt-bench-v4';
// Dependency hosts only. Provider endpoints never appear in the setup policy.
export const INSTALL_NETWORK = {allow: ['pypi.org', 'files.pythonhosted.org']};
const STOPPED = new Set(['stopped']);

export function validateBaseline(plan, prepared) {
  validatePlan(plan);
  if (prepared.status !== 'ready' || !prepared.snapshotId || !/^[a-f0-9]{64}$/.test(prepared.bundleHash) ||
      prepared.teamId !== plan.teamId || prepared.projectId !== plan.projectId)
    throw new Error('Verified baseline snapshot does not match account');
}

export async function payloadFiles() {
  const files = await codeFiles();
  const hashes = {};
  const contents = await Promise.all(files.map(async path => {
    if (!/^(src|config|scripts|tests)\//.test(path) && !['pyproject.toml', 'uv.lock'].includes(path))
      throw new Error('Unexpected upload path');
    const content = await readFile(path);
    hashes[path] = createHash('sha256').update(content).digest('hex');
    return {path: REMOTE + '/' + path, content};
  }));
  // Object insertion order must not depend on asynchronous file read completion.
  return {contents, hashes: Object.fromEntries(Object.entries(hashes).sort(([a], [b]) => a.localeCompare(b)))};
}

export function installCode() {
  return `import os,pathlib,json,hashlib,subprocess,shutil
assert not pathlib.Path('.env').exists() and not pathlib.Path('.secrets').exists()
from stt_bench.credentials import NAMES
assert not any(os.environ.get(k) for names in NAMES.values() for k in names)
payload=json.loads(pathlib.Path('trial-setup-payload.json').read_text())
for p,h in payload['hashes'].items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
with pathlib.Path('input.tar.gz').open('rb') as f:
 assert hashlib.file_digest(f,'sha256').hexdigest()==payload['baselineBundleHash']
uv=shutil.which('uv') or str(pathlib.Path.home()/'.local/bin/uv')
subprocess.run([uv,'sync','--locked','--python','3.12'],check=True)
print('Locked dependencies installed; no provider credentials or requests')
`;
}

export async function prepareOne({Sandbox, account, plan, prepared, model, suffix, payload, root,
                                  dispatch = dispatchOnce}) {
  const name = `vocera-${model}-${suffix}`;
  const dir = join(root, name);
  await mkdir(dir, {recursive: true});
  const statePath = join(dir, 'setup.json');
  const identity = {model, name, snapshotId: prepared.snapshotId, baselineBundleHash: prepared.bundleHash,
    teamId: plan.teamId, projectId: plan.projectId, region: plan.region, hashes: payload.hashes};
  let state;
  try {state = JSON.parse(await readFile(statePath, 'utf8'));} catch(e) {if (e.code !== 'ENOENT') throw e;}
  if (state && JSON.stringify(state.identity) !== JSON.stringify(identity))
    throw new Error('Setup inputs changed; use a new --suffix');
  state ??= {identity, status: 'new', commands: [], providerCalls: 0, liveAccess: 'untested'};
  async function save() {
    state.updatedAt = new Date().toISOString();
    await writeFile(statePath + '.tmp', JSON.stringify(state, null, 2) + '\n');
    await rename(statePath + '.tmp', statePath);
  }
  let sb;
  try {
    if (state.status === 'new') {
      // Persist intent first: never blindly recreate after an uncertain response.
      state.status = 'creating'; await save();
      sb = await Sandbox.create({...account, name, source: {type: 'snapshot', snapshotId: prepared.snapshotId},
        region: plan.region, resources: {vcpus: 2}, timeout: plan.timeoutMs, persistent: true,
        ports: [], env: {}, networkPolicy: INSTALL_NETWORK});
      state.status = 'uploading'; state.sandboxId = sb.sandboxId; await save();
    } else {
      sb = await Sandbox.get({...account, name, resume: false});
      if (state.status === 'creating') throw new Error('Uncertain creation requires receipt reconciliation');
    }
    if (sb.region !== plan.region || sb.vcpus !== 2 || sb.memory !== 4096)
      throw new Error('Sandbox resources do not match the benchmark');
    if (state.status === 'failed') throw new Error('Failed setup requires inspection and a new suffix');
    if (state.status === 'uploading') {
      await sb.update({networkPolicy: INSTALL_NETWORK});
      if (sb.status !== 'running') await sb.resume();
      state.computeStopped = false; await save();
      for (let i=0; i<payload.contents.length; i+=30) await sb.writeFiles(payload.contents.slice(i, i+30));
      await sb.writeFiles([{path: REMOTE + '/trial-setup-payload.json', content: Buffer.from(JSON.stringify({
        hashes: payload.hashes, baselineBundleHash: prepared.bundleHash}))}]);
      state.status = 'installing'; await save();
    }
    if (state.status === 'installing') {
      const receipt = await dispatch(sb, state, save, 'install', {
        cmd: '.venv/bin/python', cwd: REMOTE, args: ['-c', installCode()], env: {}});
      await writeFile(join(dir, 'install.log'), await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
      receipt.collected = true; state.commands.push({...receipt});
      state.status = receipt.exitCode === 0 ? 'testing' : 'failed'; await save();
      if (receipt.exitCode !== 0) throw new Error('Dependency installation failed');
    }
    if (state.status === 'testing') {
      await sb.update({networkPolicy: 'deny-all'});
      const receipt = await dispatch(sb, state, save, 'offline_tests', {
        cmd: '.venv/bin/python', cwd: REMOTE, args: ['scripts/offline_provider_checks.py',
          '--payload', 'trial-setup-payload.json', '--out', 'trial-offline-validation.json'], env: {}});
      await writeFile(join(dir, 'offline-tests.log'), await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
      receipt.collected = true; state.commands.push({...receipt});
      if (receipt.exitCode !== 0) {state.status = 'failed'; await save(); throw new Error('Offline validation failed');}
      const report = await sb.currentSession().readFileToBuffer({path: REMOTE + '/trial-offline-validation.json'});
      const validation = JSON.parse(report.toString());
      if (!validation.passed || !validation.dataset_verified || !validation.source_unchanged ||
          validation.provider_calls !== 0 || JSON.stringify(Object.entries(validation.source_identity).sort()) !==
          JSON.stringify(Object.entries(payload.hashes).sort()))
        throw new Error('Offline validation receipt does not match uploaded inputs');
      await writeFile(join(dir, 'offline-validation.json'), report);
      state.validationSha256 = createHash('sha256').update(report).digest('hex');
      state.status = 'code_ready'; await save();
    }
    if (state.status === 'code_ready') {
      // Re-entry verifies retained evidence, without running tests or resuming.
      const report = await readFile(join(dir, 'offline-validation.json'));
      if (createHash('sha256').update(report).digest('hex') !== state.validationSha256)
        throw new Error('Saved validation evidence changed');
    }
  } catch (error) {
    state.errorType = error.name;
    // Do not save arbitrary SDK exceptions: they may contain request metadata.
    await save();
    throw error;
  } finally {
    // Cleanup is limited to this exact new sandbox. No baseline/other job is touched.
    if (!sb && state.status === 'creating') {
      try {sb = await Sandbox.get({...account, name, resume: false});} catch {state.cleanupNeedsReconciliation = true;}
    }
    if (sb) {
      try {
        await sb.update({networkPolicy: 'deny-all'});
        if (!STOPPED.has(sb.status)) await sb.stop();
        const checked = await Sandbox.get({...account, name, resume: false});
        state.remoteStatus = checked.status;
        state.computeStopped = STOPPED.has(checked.status);
        if (!state.computeStopped) throw new Error('Sandbox stop not confirmed');
      } catch (error) {
        state.computeStopped = false; state.cleanupNeedsReconciliation = true;
        await save(); throw error;
      }
    }
    await save();
  }
  return {sandbox: name, status: state.status, computeStopped: state.computeStopped,
          providerCalls: 0, liveAccess: 'untested'};
}

export async function main(argv=process.argv.slice(2), dependencies={}) {
  const {values: args} = parseArgs({args: argv, options: {
    live: {type: 'boolean', default: false}, suffix: {type: 'string', default: '20260914-setup'}}});
  if (!/^[a-zA-Z0-9-]+$/.test(args.suffix)) throw new Error('Invalid sandbox suffix');
  if (!args.live) {
    const result = {status: 'no_remote_actions', models: MODELS.map(model => ({model, sandbox: `vocera-${model}-${args.suffix}`}))};
    console.log(JSON.stringify(result, null, 2)); return result;
  }
  const plan = JSON.parse(await readFile('config/vercel-models.json', 'utf8'));
  const prepared = JSON.parse(await readFile('reports/vercel-models/preparation.json', 'utf8'));
  validateBaseline(plan, prepared);
  const payload = await payloadFiles();
  let Sandbox = dependencies.Sandbox, Snapshot = dependencies.Snapshot, account = dependencies.account;
  if (!Sandbox) {
    const sdk = process.env.VERCEL_SANDBOX_SDK_DIR;
    if (!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR required');
    ({Sandbox, Snapshot} = await import(pathToFileURL(join(sdk, 'dist/index.js'))));
    const {getAuth, OAuth, updateAuthConfig} = await import(pathToFileURL(join(sdk, 'dist/auth/index.js')));
    let auth = getAuth();
    if (!auth?.token) throw new Error('Existing Vercel login required');
    if (auth.expiresAt?.getTime() < Date.now() && auth.refreshToken) {
      const t = await (await OAuth()).refreshToken(auth.refreshToken);
      auth = {token: t.access_token, refreshToken: t.refresh_token ?? auth.refreshToken,
              expiresAt: new Date(Date.now() + t.expires_in * 1000)};
      updateAuthConfig(auth);
    }
    account = {token: auth.token, teamId: plan.teamId, projectId: plan.projectId};
  }
  const baseline = await Snapshot.get({...account, snapshotId: prepared.snapshotId});
  if (baseline.snapshotId !== prepared.snapshotId || !baseline.regions.includes(plan.region) ||
      (baseline.expiresAt && baseline.expiresAt.getTime() <= Date.now()))
    throw new Error('Baseline snapshot unavailable in the selected project/region');
  const root = resolve('reports/vercel-trial-providers');
  await mkdir(root, {recursive: true});
  const lockPath = join(root, '.setup.lock');
  const lock = await open(lockPath, 'wx');
  const outcomes = [];
  try {
    for (const model of MODELS) {
      const result = await prepareOne({Sandbox, account, plan, prepared, model, suffix: args.suffix,
                                       payload, root, dispatch: dependencies.dispatch});
      outcomes.push(result); console.log(JSON.stringify(result));
      await writeFile(join(root, 'outcomes.json'), JSON.stringify(outcomes, null, 2) + '\n');
    }
  } finally {await lock.close(); await unlink(lockPath);}
  return outcomes;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => {console.error(`Trial setup stopped (${error.name}); inspect setup receipts and offline logs.`); process.exitCode=1;});
