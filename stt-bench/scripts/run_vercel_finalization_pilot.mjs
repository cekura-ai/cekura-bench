// Dispatch one frozen public pilot in the configured Vercel benchmark project.
import {readFile, writeFile, open, rename} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {dispatchOnce, digestFile} from './vercel_models.mjs';

const {values:args}=parseArgs({options:{action:{type:'string'},out:{type:'string'}}});
if(!['launch','status','collect'].includes(args.action) || !/^reports\/finalization-pilot-[a-zA-Z0-9-]+$/.test(args.out??''))
  throw new Error('Expected --action launch|status|collect and a reports/finalization-pilot-* directory');
const root=resolve(args.out), remote='/vercel/sandbox/stt-bench-v4';
const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
if(!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR required');
const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
let auth=getAuth();
if(!auth?.token) throw new Error('Existing Vercel login required');
if(auth.expiresAt?.getTime()<Date.now() && auth.refreshToken){
  const t=await (await OAuth()).refreshToken(auth.refreshToken);
  auth={token:t.access_token,refreshToken:t.refresh_token??auth.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};
  updateAuthConfig(auth);
}
const accountPlan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
const prepared=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
if(prepared.status!=='ready'||prepared.teamId!==accountPlan.teamId||prepared.projectId!==accountPlan.projectId)
  throw new Error('Prepared snapshot does not match configured project');
const account={token:auth.token,teamId:accountPlan.teamId,projectId:accountPlan.projectId};
const statePath=join(root,'vercel.json');
let state, sb;
async function save(){await writeFile(statePath+'.tmp',JSON.stringify(state,null,2)+'\n');await rename(statePath+'.tmp',statePath);}

if(args.action==='launch'){
  const lock=await open(join(root,'.vercel-launch'),'wx');await lock.close();
  const plan=JSON.parse(await readFile(join(root,'plan.json'),'utf8'));
  const extra=['pyproject.toml','uv.lock',`${args.out}/plan.json`,
    ...plan.models.map(m=>`config/models/${m}.json`),
    'tests/test_finalization_pilot.py','tests/test_assemblyai.py',
    'tests/test_trial_providers.py','tests/test_providers.py'];
  // test_providers parametrizes the full catalog; all configs are safe code inputs.
  const {codeFiles}=await import('./setup_vercel_chirp.mjs');
  const configs=(await codeFiles()).filter(p=>p.startsWith('config/'));
  const files=[...new Set([...Object.keys(plan.hashes),...extra,...configs])];
  const env=JSON.parse(execFileSync('.venv/bin/python',['-c',
    'import json; from stt_bench.credentials import command_environment; e={}; [e.update(command_environment(p)) for p in ("assemblyai","speechmatics","inworld")]; print(json.dumps(e))'],
    {encoding:'utf8',stdio:['ignore','pipe','pipe']}));
  const payload=[];
  for(const path of files){
    if(path.startsWith('/')||path.split('/').includes('..')) throw new Error('Unsafe upload path');
    const content=await readFile(path);
    if(Object.values(env).some(v=>v&&content.includes(Buffer.from(v)))) throw new Error('Credential in file upload');
    const actual=createHash('sha256').update(content).digest('hex');
    if(plan.hashes[path]&&actual!==plan.hashes[path]) throw new Error('Frozen file changed');
    payload.push({path:`${remote}/${path}`,content});
  }
  state={name:args.out.split('/').at(-1),status:'creating',teamId:account.teamId,
    projectId:account.projectId,snapshotId:prepared.snapshotId,planSha256:await digestFile(join(root,'plan.json'))};
  await save();
  try{
    sb=await Sandbox.create({...account,name:state.name,source:{type:'snapshot',snapshotId:prepared.snapshotId},
      region:accountPlan.region,resources:{vcpus:2},timeout:2700000,persistent:true,ports:[],env:{},networkPolicy:'deny-all'});
    state.sandboxId=sb.sandboxId;state.sessionId=sb.currentSession().sessionId;state.status='uploading';await save();
    if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096) throw new Error('Unexpected sandbox resources');
    for(let i=0;i<payload.length;i+=25) await sb.writeFiles(payload.slice(i,i+25));
    const verify=`import pathlib,json,hashlib,importlib.metadata\np=json.loads(pathlib.Path(${JSON.stringify(args.out+'/plan.json')}).read_text())\nfor f,h in p['hashes'].items():\n assert hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h\nfrom stt_bench.data import load_manifest\nload_manifest(pathlib.Path(${JSON.stringify(args.out+'/dataset/manifest.json')}))\nfor pkg,version in [('jiwer','4.0.0'),('whisper-normalizer','0.1.12')]:\n assert importlib.metadata.version(pkg)==version\nprint('Frozen public pilot and scoring versions verified')\n`;
    let receipt=await dispatchOnce(sb,state,save,'verify',{cmd:'.venv/bin/python',cwd:remote,args:['-c',verify],env:{}});
    await writeFile(join(root,'verify.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
    if(receipt.exitCode!==0) throw new Error('Remote verification failed');
    receipt.collected=true;await save();
    receipt=await dispatchOnce(sb,state,save,'tests',{cmd:'.venv/bin/python',cwd:remote,
      args:['-m','pytest','-q','tests/test_finalization_pilot.py','tests/test_assemblyai.py','tests/test_trial_providers.py','tests/test_providers.py','--tb=short'],env:{}});
    await writeFile(join(root,'tests.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
    if(receipt.exitCode!==0) throw new Error('Remote adapter tests failed');
    receipt.collected=true;await save();
    await sb.update({networkPolicy:{allow:['streaming.assemblyai.com','us.rt.speechmatics.com','api.inworld.ai']}});
    state.status='dispatching_live';await save();
    const command=await sb.runCommand({cmd:'.venv/bin/python',cwd:remote,
      args:['-u','scripts/finalization_pilot.py','live','--out',args.out],detached:true,
      env:{...env,STT_BENCH_COMPUTE_PROVIDER:'vercel-sandbox',STT_BENCH_COMPUTE_REGION:'iad1',STT_BENCH_COMPUTE_INSTANCE:state.sessionId}});
    state.liveCommandId=command.cmdId;state.status='running';await save();
    console.log(JSON.stringify({status:state.status,sandbox:state.name,sessionId:state.sessionId,commandId:state.liveCommandId}));
  }catch(e){
    state.errorType=e.name;state.errorMessage=e.message;await save();
    // Never terminate or redispatch a live command whose dispatch may have succeeded.
    if(sb && state.status!=='dispatching_live' && state.status!=='running'){
      try{await sb.update({networkPolicy:'deny-all'});}finally{await sb.stop();}
      state.computeStopped=true;await save();
    }
    throw e;
  }
}else{
  state=JSON.parse(await readFile(statePath,'utf8'));
  sb=await Sandbox.get({...account,name:state.name,resume:false});
  if(sb.status!=='running') {console.log(JSON.stringify({sandboxStatus:sb.status,state}));process.exit(0);}
  const session=sb.currentSession();
  if(session.sessionId!==state.sessionId||!state.liveCommandId) throw new Error('Uncertain session/dispatch requires reconciliation');
  let command=await session.getCommand(state.liveCommandId);
  try{command=await command.wait({signal:AbortSignal.timeout(1500)});}
  catch(e){if(!['AbortError','TimeoutError'].includes(e.name)) throw e;}
  for(const name of ['results.json','state.json','pacing/pacing.json']){
    const bytes=await session.readFileToBuffer({path:`${remote}/${args.out}/${name}`}).catch(()=>null);
    if(bytes) await writeFile(join(root,name.replaceAll('/','-')),bytes);
  }
  if(args.action==='status'){
    let result;try{result=JSON.parse(await readFile(join(root,'results.json'),'utf8'));}catch{}
    console.log(JSON.stringify({sandboxStatus:sb.status,exitCode:command.exitCode,
      sessions:result?.observations.length,models:result&&Object.fromEntries(Object.entries(result.models).map(([k,v])=>[k,{paired:v.paired_clips,variants:v.variants}]))}));
  }else{
    if(command.exitCode===undefined||command.exitCode===null) throw new Error('Live command still running; collection does not stop it');
    await writeFile(join(root,'live.log'),await command.output('both'));
    state.liveExitCode=command.exitCode;await save();
    const pack=`import pathlib,tarfile,hashlib\nr=pathlib.Path(${JSON.stringify(args.out)})\np=r/'evidence.tar.gz'\nwith tarfile.open(p,'w:gz') as a:\n for f in sorted(r.rglob('*')):\n  if f.is_file() and f.name not in ('evidence.tar.gz','evidence.sha256'):\n   a.add(f,arcname=str(f.relative_to(r)),recursive=False)\n(r/'evidence.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())\n`;
    state.command.collected=true;
    const receipt=await dispatchOnce(sb,state,save,'archive',{cmd:'.venv/bin/python',cwd:remote,args:['-c',pack],env:{}});
    if(receipt.exitCode!==0) throw new Error('Evidence archive failed');
    for(const name of ['evidence.tar.gz','evidence.sha256'])
      await session.downloadFile({path:`${remote}/${args.out}/${name}`},{path:join(root,name)});
    if(await digestFile(join(root,'evidence.tar.gz'))!==(await readFile(join(root,'evidence.sha256'),'utf8')).trim()) throw new Error('Evidence checksum mismatch');
    state.status='collected';await save();
    try{await sb.update({networkPolicy:'deny-all'});}finally{await sb.stop();}
    const stopped=await Sandbox.get({...account,name:state.name,resume:false});
    state.computeStopped=stopped.status==='stopped';await save();
    console.log(JSON.stringify({status:state.status,computeStopped:state.computeStopped,liveExitCode:state.liveExitCode}));
  }
}
