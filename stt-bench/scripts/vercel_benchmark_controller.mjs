// Local control only: audio streaming and measurement run entirely in Vercel.
// Never rotates a sandbox while a batch command is running.
import { execFileSync } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, open, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { waitForCompletion } from './vercel_command_wait.mjs';

const workspace = fileURLToPath(new URL('../', import.meta.url));
const sdkDir = process.env.VERCEL_SANDBOX_SDK_DIR;
if (!sdkDir) throw new Error('VERCEL_SANDBOX_SDK_DIR must point to the installed Sandbox SDK');
const { Sandbox } = await import(pathToFileURL(join(sdkDir, 'dist/index.js')));
const { getAuth } = await import(pathToFileURL(join(sdkDir, 'dist/auth/index.js')));
const target = { name: 'my-sandbox-013057', teamId: 'anushrut-9761', projectId: 'vercel-sandbox-default-project' };
const runId = 'pipecat-nova3-vercel-full-20260912';
const smokeId = 'pipecat-nova3-vercel-smoke-20260912';
const remote = '/vercel/sandbox/stt-bench';
const report = `reports/pipecat-stt-benchmark/deepgram-nova-3/${runId}`;
const local = join(workspace, 'reports', runId);
await mkdir(local, { recursive: true });
const lock = await open(join(local, '.controller.lock'), 'wx');
await lock.writeFile(String(process.pid));
const resuming = process.argv.includes('--resume');
const state = resuming
  ? JSON.parse(await readFile(join(local, 'controller.json'), 'utf8'))
  : { runId, target, status: 'starting', sessions: [], startedAt: new Date().toISOString() };
if (state.runId !== runId || JSON.stringify(state.target) !== JSON.stringify(target)) throw new Error('Resume target mismatch');
if (resuming) {
  (state.recoveryHistory ??= []).push({ at: new Date().toISOString(), previousStatus: state.status, previousError: state.error });
  delete state.error;
}
let sandbox;

async function save(status) {
  state.status = status;
  state.updatedAt = new Date().toISOString();
  await writeFile(join(local, 'controller.json.tmp'), JSON.stringify(state, null, 2));
  await rename(join(local, 'controller.json.tmp'), join(local, 'controller.json'));
  console.log(JSON.stringify({ status, sessions: state.sessions.length, completedClips: state.completedClips, time: state.updatedAt }));
}
async function connect() {
  // Let the authenticated CLI refresh its login using its standard recovery path.
  execFileSync('npx', ['--offline', 'sandbox', 'list', '--all', '--scope', target.teamId, '--project', target.projectId],
    { cwd: workspace, stdio: ['ignore', 'pipe', 'pipe'], timeout: 60000 });
  const auth = getAuth();
  if (!auth?.token) throw new Error('Sandbox CLI login required');
  const sb = await Sandbox.get({ ...target, token: auth.token, resume: false });
  if (sb.region !== 'iad1') throw new Error('Unexpected sandbox region');
  return sb;
}
function deepgramKey() {
  return execFileSync(join(workspace, '.venv/bin/python'), ['-c',
    'from dotenv import dotenv_values; import os,sys; key=os.environ.get("DEEPGRAM_API_KEY") or dotenv_values(".env").get("DEEPGRAM_API_KEY"); assert key, "Deepgram key missing"; sys.stdout.write(key)'],
    { cwd: workspace, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
}
async function readJson(path) {
  return JSON.parse((await sandbox.readFileToBuffer({ path: `${remote}/${path}` })).toString());
}
async function waitForCommand(id) {
  return waitForCompletion(sandbox, id);
}
async function downloadSummary() {
  for (const filename of ['batch-state.json', 'summary.json', 'summary.md', 'clips.json']) {
    try {
      await sandbox.downloadFile({ path: `${remote}/${report}/${filename}` }, { path: join(local, filename) });
    } catch (error) {
      if (error.response?.status !== 404) throw error;
    }
  }
}

async function collectSession(record) {
  if (!record.commandId) throw new Error('Uncertain dispatch: inspect existing work before continuing');
  if (sandbox.currentSession().sessionId !== record.sessionId) throw new Error('Saved command belongs to another VM session; manual reconciliation required');
  record.exitCode = await waitForCommand(record.commandId);
  record.status = 'finished';
  await writeFile(join(local, `${record.sessionId}.log`), await (await sandbox.getCommand(record.commandId)).output('both'));
  await downloadSummary();
  const batchState = await readJson(`${report}/batch-state.json`);
  state.completedClips = batchState.completed_batches.reduce((sum, batch) => sum + batch.clips, 0);
  if (record.exitCode !== 0 || !['session_complete', 'complete'].includes(batchState.status)) {
    await exportResults();
    await sandbox.stop();
    throw new Error('Batch session failed; evidence saved and automatic continuation stopped');
  }
  if (batchState.status === 'complete') {
    if (state.completedClips !== 1000) throw new Error('Full coverage mismatch');
    await exportResults();
    await sandbox.stop();
    await save('complete');
    return true;
  }
  await save('rotating_session');
  await sandbox.stop();
  return false;
}
async function exportResults() {
  const archive = `reports/${runId}-artifacts.tar.gz`;
  const code = `import tarfile,hashlib\nfrom pathlib import Path\np=Path(${JSON.stringify(archive)})\nwith tarfile.open(p,'w:gz') as t:\n for source in [${JSON.stringify(report)},${JSON.stringify(`runs/pipecat-stt-benchmark/deepgram-nova-3/${runId}`)}]:\n  if Path(source).exists(): t.add(source,arcname=source)\nwith p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()\np.with_suffix('.sha256').write_text(digest+'  '+p.name+'\\n')\n`;
  const command = await sandbox.runCommand({ cmd: '.venv/bin/python', args: ['-c', code], cwd: remote });
  if (command.exitCode !== 0) throw new Error('Artifact export failed');
  await sandbox.downloadFile({ path: `${remote}/${archive}` }, { path: join(local, 'artifacts.tar.gz') });
  await sandbox.downloadFile({ path: `${remote}/${archive.replace(/\.gz$/, '.sha256')}` }, { path: join(local, 'artifacts.sha256') });
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(join(local, 'artifacts.tar.gz'))) digest.update(chunk);
  const expected = (await readFile(join(local, 'artifacts.sha256'), 'utf8')).split(/\s+/)[0];
  if (digest.digest('hex') !== expected) throw new Error('Downloaded artifact checksum mismatch');
}

try {
  await save(resuming ? 'reconciling_saved_session' : 'waiting_for_smoke');
  sandbox = await connect();
  let alreadyComplete = false;
  if (resuming) {
    const previous = state.sessions.at(-1);
    if (!previous) throw new Error('No previous session to reconcile');
    alreadyComplete = await collectSession(previous);
  } else {
    const smokeReceipt = JSON.parse(await readFile(join(workspace, 'reports', smokeId, 'launch.json'), 'utf8'));
    if (await waitForCommand(smokeReceipt.commandId) !== 0) throw new Error('Smoke command failed; full run not started');
    const smoke = await readJson(`reports/remote-jobs/${smokeId}/smoke-job.json`);
    if (!smoke.smoke_passed) throw new Error('Smoke did not pass; full run not started');
    await sandbox.writeFiles([{ path: `${remote}/scripts/vercel_batches.py`, content: await readFile(join(workspace, 'scripts/vercel_batches.py')) }]);
  }
  const key = deepgramKey();
  for (let round = 0; !alreadyComplete && round < 30; round++) {
    sandbox = await connect();
    if (sandbox.status !== 'running') await sandbox.resume();
    let remaining = (sandbox.expiresAt.getTime() - Date.now()) / 1000;
    if (remaining < 1500) {
      await sandbox.stop();
      await sandbox.resume();
      remaining = (sandbox.expiresAt.getTime() - Date.now()) / 1000;
    }
    const session = sandbox.currentSession();
    if (session.region !== 'iad1') throw new Error('Unexpected actual session region');
    const budget = Math.min(2400, Math.floor(remaining - 150));
    const record = { sessionId: session.sessionId, region: session.region, budgetSeconds: budget, status: 'dispatching' };
    state.sessions.push(record);
    await save('dispatching_session');
    const command = await sandbox.runCommand({ cmd: '.venv/bin/python', cwd: remote,
      args: ['-u', 'scripts/vercel_batches.py', '--run-id', runId, '--session-id', session.sessionId,
        '--budget-seconds', String(budget), '--smoke-status', `reports/remote-jobs/${smokeId}/smoke-job.json`],
      env: { DEEPGRAM_API_KEY: key }, detached: true });
    record.commandId = command.cmdId;
    record.status = 'running';
    await save('running');
    if (await collectSession(record)) break;
    if (round === 29) throw new Error('Session count limit reached; inspect saved progress');
  }
} catch (error) {
  state.error = { type: error.name, message: error.message };
  await save('failed');
  console.error(error.name + ': ' + error.message);
  // Do not stop an uncertain in-flight command. Its saved ID identifies it for recovery.
  process.exitCode = 1;
} finally {
  await lock.close();
}
