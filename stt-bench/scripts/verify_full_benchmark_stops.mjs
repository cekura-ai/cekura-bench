// Read back only this run's sandbox states. Never resumes or creates compute.
import {readFile, writeFile} from 'node:fs/promises';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';

const root = resolve(process.argv[2] || 'reports/full-parallel-20260915');
const state = JSON.parse(await readFile(join(root, 'controller.json'), 'utf8'));
const config = JSON.parse(await readFile('config/vercel-models.json', 'utf8'));
const sdk = process.env.VERCEL_SANDBOX_SDK_DIR;
if (!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR required');
const {Sandbox} = await import(pathToFileURL(join(sdk, 'dist/index.js')));
const {getAuth, OAuth, updateAuthConfig} = await import(pathToFileURL(join(sdk, 'dist/auth/index.js')));
let auth = getAuth();
if (auth.expiresAt?.getTime() < Date.now() + 300000) {
  const token = await (await OAuth()).refreshToken(auth.refreshToken);
  updateAuthConfig({token: token.access_token, refreshToken: token.refresh_token ?? auth.refreshToken,
    expiresAt: new Date(Date.now() + token.expires_in * 1000)});
  auth = getAuth();
}
const workers = [state.preparation, ...Object.values(state.workers)];
const results = [];
let cursor = 0;
await Promise.all(Array.from({length: 4}, async () => {
  while (cursor < workers.length) {
    const worker = workers[cursor++];
    const sandbox = await Sandbox.get({token: auth.token, teamId: config.teamId,
      projectId: config.projectId, name: worker.name, resume: false});
    results.push({name: worker.name, status: sandbox.status, region: sandbox.region,
      vcpus: sandbox.vcpus, checkedAt: new Date().toISOString()});
  }
}));
results.sort((a, b) => a.name.localeCompare(b.name));
const record = {checkedAt: new Date().toISOString(), allStopped: results.every(r => r.status === 'stopped'),
  teamId: config.teamId, projectId: config.projectId, sandboxes: results};
await writeFile(join(root, 'compute-stop-verification.json'), JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({allStopped: record.allStopped, checked: results.length,
  notStopped: results.filter(r => r.status !== 'stopped')}, null, 2));
if (!record.allStopped) process.exitCode = 1;
