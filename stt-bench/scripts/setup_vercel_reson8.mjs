// Setup and an optional single synthetic-phrase smoke only. No benchmark launcher.
import {readFile, writeFile, mkdir, open, unlink} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {prepareOne, payloadFiles, validateBaseline, REMOTE} from './setup_vercel_trial_providers.mjs';
import {dispatchOnce} from './vercel_models.mjs';

export async function main(argv=process.argv.slice(2)) {
  const {values: args} = parseArgs({args: argv, options: {
    live: {type: 'boolean', default: false}, 'tiny-smoke': {type: 'boolean', default: false},
    audio: {type: 'string'}, suffix: {type: 'string', default: '20260914-setup'}}});
  if (!/^[a-zA-Z0-9-]+$/.test(args.suffix)) throw new Error('Invalid suffix');
  const model = 'reson8-realtime', name = `vocera-${model}-${args.suffix}`;
  if (!args.live) {console.log(JSON.stringify({status:'no_remote_actions', sandbox:name, benchmarkStarted:false})); return;}
  if (args['tiny-smoke'] && !args.audio) throw new Error('--audio is required for the bounded smoke');
  const plan = JSON.parse(await readFile('config/vercel-models.json','utf8'));
  const prepared = JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
  validateBaseline(plan, prepared);
  const sdk = process.env.VERCEL_SANDBOX_SDK_DIR;
  if (!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR required');
  const {Sandbox, Snapshot} = await import(pathToFileURL(join(sdk,'dist/index.js')));
  const {getAuth, OAuth, updateAuthConfig} = await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
  let auth = getAuth();
  if (!auth?.token) throw new Error('Existing Vercel login required');
  if (auth.expiresAt?.getTime() < Date.now() && auth.refreshToken) {
    const t = await (await OAuth()).refreshToken(auth.refreshToken);
    auth = {token:t.access_token, refreshToken:t.refresh_token ?? auth.refreshToken,
            expiresAt:new Date(Date.now()+t.expires_in*1000)};
    updateAuthConfig(auth);
  }
  const account = {token:auth.token, teamId:plan.teamId, projectId:plan.projectId};
  const baseline = await Snapshot.get({...account,snapshotId:prepared.snapshotId});
  if (!baseline.regions.includes(plan.region) || (baseline.expiresAt && baseline.expiresAt.getTime() <= Date.now()))
    throw new Error('Baseline unavailable');
  const root = resolve('reports/vercel-reson8');
  await mkdir(root,{recursive:true});
  const lockPath = join(root,'.setup.lock'), lock = await open(lockPath,'wx');
  try {
    const result = await prepareOne({Sandbox,account,plan,prepared,model,suffix:args.suffix,
                                    payload:await payloadFiles(),root});
    console.log(JSON.stringify(result));
    if (!args['tiny-smoke']) return result;
    const dir = join(root,name), statePath = join(dir,'tiny-smoke.json');
    // One attempt across invocations. An uncertain response requires read-only reconciliation.
    const marker = await open(statePath,'wx'); await marker.close();
    const state = {status:'preparing', benchmarkStarted:false, providerRequestsMax:1, retries:0};
    const save = () => writeFile(statePath,JSON.stringify(state,null,2)+'\n');
    await save();
    let sb;
    try {
      const env = JSON.parse(execFileSync('.venv/bin/python',['-c',
        "import json; from stt_bench.credentials import command_environment; print(json.dumps(command_environment('reson8')))"],
        {encoding:'utf8',stdio:['ignore','pipe','pipe']}));
      const audio = await readFile(args.audio);
      if (audio.length > 100000) throw new Error('Smoke WAV exceeds size cap');
      sb = await Sandbox.get({...account,name,resume:false});
      await sb.update({networkPolicy:{allow:['api.reson8.dev']}});
      await sb.resume();
      await sb.writeFiles([{path:REMOTE+'/reson8-smoke.wav',content:audio}]);
      const receipt = await dispatchOnce(sb,state,save,'tiny_smoke',{
        cmd:'.venv/bin/python',cwd:REMOTE,env,
        args:['scripts/reson8_tiny_smoke.py','--audio','reson8-smoke.wav','--out','reson8-tiny-smoke']});
      await writeFile(join(dir,'tiny-smoke.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
      for (const file of ['attempt.json','events.jsonl','summary.json'])
        await writeFile(join(dir,'smoke-'+file),await sb.currentSession().readFileToBuffer({path:REMOTE+'/reson8-tiny-smoke/'+file}));
      const summary = JSON.parse(await readFile(join(dir,'smoke-summary.json'),'utf8'));
      state.status = receipt.exitCode===0 && summary.passed ? 'smoke_passed' : 'smoke_failed';
      state.summary = summary; receipt.collected = true; await save();
    } finally {
      if (sb) {
        await sb.update({networkPolicy:'deny-all'});
        await sb.stop();
        let checked = await Sandbox.get({...account,name,resume:false});
        for (let i=0; checked.status==='stopping' && i<15; i++) {
          await new Promise(resolve => setTimeout(resolve,1000));
          checked = await Sandbox.get({...account,name,resume:false});
        }
        state.remoteStatus = checked.status;
        state.computeStopped = checked.status === 'stopped';
        state.networkPolicy = checked.networkPolicy;
        await save();
        if (!state.computeStopped) throw new Error('Sandbox stop unconfirmed');
      }
    }
    console.log(JSON.stringify(state));
    if (state.status !== 'smoke_passed') throw new Error('Tiny smoke did not pass; no retry will be made');
    return state;
  } finally {await lock.close();await unlink(lockPath);}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => {console.error(`Reson8 setup stopped (${error.name}); inspect saved receipts.`);process.exitCode=1;});
