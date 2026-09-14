// Offline-only refresh of the prepared Gradium sandbox. No provider credential or call.
import {readFile,writeFile,open,unlink} from 'node:fs/promises';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {payloadFiles,REMOTE} from './setup_vercel_trial_providers.mjs';
import {dispatchOnce} from './vercel_models.mjs';

async function main() {
  if (process.argv.length!==3 || process.argv[2]!=='--live') throw new Error('Requires --live for offline sandbox refresh');
  const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
  const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
  const {getAuth}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
  const plan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
  const name='vocera-gradium-default-20260914-ready';
  const root='reports/vercel-gradium/'+name;
  const prior=JSON.parse(await readFile(root+'/setup.json','utf8'));
  if(prior.identity.teamId!==plan.teamId || prior.identity.projectId!==plan.projectId || !prior.computeStopped)
    throw new Error('Prepared target mismatch');
  const payload=await payloadFiles();
  for(const p of ['pyproject.toml','uv.lock'])
    if(payload.hashes[p]!==prior.identity.hashes[p]) throw new Error('Dependencies changed; fresh installation required');
  const account={token:getAuth().token,teamId:plan.teamId,projectId:plan.projectId};
  const lock=await open(root+'/.refresh.lock','wx');
  const state={status:'refreshing',providerCalls:0,benchmarkStarted:false,hashes:payload.hashes};
  const save=()=>writeFile(root+'/refresh.json',JSON.stringify(state,null,2)+'\n');
  let sb;
  try {
    await save();
    sb=await Sandbox.get({...account,name,resume:false});
    if(sb.status!=='stopped') throw new Error('Expected stopped sandbox');
    await sb.update({networkPolicy:'deny-all'});await sb.resume();
    for(let i=0;i<payload.contents.length;i+=30) await sb.writeFiles(payload.contents.slice(i,i+30));
    await sb.writeFiles([{path:REMOTE+'/trial-setup-payload.json',content:Buffer.from(JSON.stringify({
      hashes:payload.hashes,baselineBundleHash:prior.identity.baselineBundleHash}))}]);
    const code=`import json,pathlib,subprocess,hashlib
subprocess.run(['.venv/bin/python','scripts/offline_provider_checks.py','--payload','trial-setup-payload.json','--out','gradium-final-offline.json'],check=True)
from stt_bench.providers import reduce_events
from stt_bench.streaming import read_events
p=pathlib.Path('gradium-tiny-smoke/events.jsonl')
c=json.loads(pathlib.Path('config/models/gradium-default.json').read_text())
r=reduce_events(read_events(p),c)
assert r['transcript_complete'] and r['model_verified'] and r['transcript']=='hello, this is a test.'
r['evidence_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
r['validation_kind']='offline replay of previously captured live smoke; zero new provider calls'
pathlib.Path('gradium-final-replay.json').write_text(json.dumps(r,indent=2)+'\\n')
print('Offline checks and saved live smoke replay passed; no provider calls')
`;
    const receipt=await dispatchOnce(sb,state,save,'offline_refresh',{
      cmd:'.venv/bin/python',cwd:REMOTE,args:['-c',code],env:{}});
    await writeFile(root+'/refresh.log',await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
    receipt.collected=true;
    if(receipt.exitCode!==0) throw new Error('Offline refresh failed');
    for(const file of ['gradium-final-offline.json','gradium-final-replay.json'])
      await writeFile(root+'/'+file,await sb.currentSession().readFileToBuffer({path:REMOTE+'/'+file}));
    state.status='ready';await save();
  } finally {
    if(sb) {
      await sb.update({networkPolicy:'deny-all'});await sb.stop();
      let checked=await Sandbox.get({...account,name,resume:false});
      for(let i=0;checked.status==='stopping' && i<15;i++) {
        await new Promise(resolve=>setTimeout(resolve,1000));checked=await Sandbox.get({...account,name,resume:false});
      }
      state.remoteStatus=checked.status;state.computeStopped=checked.status==='stopped';await save();
    }
    await lock.close();await unlink(root+'/.refresh.lock');
  }
  if(!state.computeStopped) throw new Error('Stop not confirmed');
  console.log(JSON.stringify({sandbox:name,status:state.status,computeStopped:true,providerCalls:0}));
}
main().catch(error=>{console.error('Gradium offline refresh stopped ('+error.name+'); inspect receipts.');process.exitCode=1;});
