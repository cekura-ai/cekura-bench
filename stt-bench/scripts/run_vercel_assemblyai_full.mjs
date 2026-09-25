// Launch once; read-only status and receipt-bound collection never retranscribe.
import {readFile,writeFile,mkdir,rename,open,unlink} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {setTimeout as delay} from 'node:timers/promises';
import {codeFiles} from './setup_vercel_chirp.mjs';
import {upload,digestFile,dispatchOnce} from './vercel_models.mjs';
import {waitForCompletion} from './vercel_command_wait.mjs';

async function main(){
 const {values:args}=parseArgs({options:{cohort:{type:'string'},live:{type:'boolean'},status:{type:'boolean'},collect:{type:'boolean'},recover:{type:'boolean'}}});
 if(!['pipecat','private'].includes(args.cohort))throw new Error('Choose pipecat or private');
 if([args.live,args.status,args.collect].filter(Boolean).length>1)throw new Error('Choose one mode');
 const cohort=args.cohort,baseId=`assemblyai-${cohort}-full-20260914`,runId=baseId+(args.recover?'-wire60':''),name=`vocera-${baseId}`;
 const baseRoot=`reports/assemblyai-full-20260914/${cohort}`,root=baseRoot+(args.recover?'/wire60':''),remote='/vercel/sandbox/stt-bench-v4',out=`assemblyai-results/${cohort}`+(args.recover?'-wire60':'');
 if(!args.live&&!args.status&&!args.collect){console.log(JSON.stringify({cohort,name,planned:cohort==='pipecat'?1000:8,streamConcurrency:1}));return;}
 const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
 const {Sandbox,Snapshot}=await import(pathToFileURL(join(sdk,'dist/index.js')));
 const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
 let auth=getAuth();if(!auth?.token)throw new Error('Vercel login missing');
 if(auth.expiresAt?.getTime()<Date.now()&&auth.refreshToken){const t=await (await OAuth()).refreshToken(auth.refreshToken);auth={token:t.access_token,refreshToken:t.refresh_token??auth.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};updateAuthConfig(auth);}
 const plan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
 const prepared=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
 if(prepared.status!=='ready'||prepared.teamId!==plan.teamId||prepared.projectId!==plan.projectId)throw new Error('Snapshot account mismatch');
 const account={token:auth.token,teamId:plan.teamId,projectId:plan.projectId};
 await mkdir(root,{recursive:true});const statePath=root+'/launch.json';let state;
 try{state=JSON.parse(await readFile(statePath,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
 async function save(){state.updatedAt=new Date().toISOString();await writeFile(statePath+'.tmp',JSON.stringify(state,null,2)+'\n');await rename(statePath+'.tmp',statePath);}
 if(args.status||args.collect){
  if(!state)throw new Error('No launch receipt');
  let sb=await Sandbox.get({...account,name,resume:false});
  if(args.status){
   let progress;
   if(sb.status==='running'){
    try{const buf=await sb.currentSession().readFileToBuffer({path:remote+'/'+out+'/state.json'});const s=JSON.parse(buf.toString());
     progress={status:s.status,started_at:s.started_at,updated_at:s.updated_at,finished_at:s.finished_at,
      smokePassed:s.smoke_passed??s.smoke?.valid,completed:s.completed_clips??s.recordings?.length,
      activeBatch:s.active_batch,activeRecording:s.active_recording,errorType:s.error_type};
     if(s.status==='smoke_failed')progress.smokeErrors=s.smoke?.errors;
    }catch(e){progress={readError:e.name};}
   }
   console.log(JSON.stringify({cohort,sandbox:name,remoteStatus:sb.status,stage:state.status,command:state.command,progress},null,2));return;
  }
  if(state.status==='collected'){console.log(JSON.stringify({cohort,status:'already_collected',computeStopped:state.computeStopped}));return;}
  if(state.status!=='running'||!state.command?.commandId||state.command.stage!=='benchmark')throw new Error('Launch is not ready to collect');
  if(sb.currentSession().sessionId!==state.command.sessionId)throw new Error('Saved command session changed; inspect without restarting');
  const lock=await open(root+'/.collector.lock','wx');
  try{
   // Refresh only read/wait credentials during long jobs; never redispatch providers.
   const reader={getCommand:async(id,options)=>{
    let latest=getAuth();
    if(latest.expiresAt?.getTime()<Date.now()+300000){
     const refreshPath='reports/assemblyai-full-20260914/.auth-refresh.lock';let refreshLock;
     while(!refreshLock){try{refreshLock=await open(refreshPath,'wx');}catch(e){if(e.code!=='EEXIST')throw e;await delay(1000);}}
     try{
      latest=getAuth();
      if(latest.expiresAt?.getTime()<Date.now()+300000){
       const t=await (await OAuth()).refreshToken(latest.refreshToken);
       latest={token:t.access_token,refreshToken:t.refresh_token??latest.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};
       updateAuthConfig(latest);
      }
     }finally{await refreshLock.close();await unlink(refreshPath);}
    }
    if(account.token!==latest.token){account.token=latest.token;sb=await Sandbox.get({...account,name,resume:false});}
    if(sb.currentSession().sessionId!==state.command.sessionId)throw new Error('Saved command session changed');
    return sb.currentSession().getCommand(id,options);
   }};
   state.command.exitCode=await waitForCompletion(reader,state.command.commandId);await save();
   await writeFile(root+'/run.log',await (await sb.currentSession().getCommand(state.command.commandId)).output('both'));
   for(const file of ['state.json','evidence.tar.gz','evidence.sha256'])await sb.currentSession().downloadFile({path:remote+'/'+out+'/'+file},{path:root+'/'+file});
   if(await digestFile(root+'/evidence.tar.gz')!==(await readFile(root+'/evidence.sha256','utf8')).split(/\s+/)[0])throw new Error('Evidence hash mismatch');
   if(cohort==='pipecat')await sb.currentSession().downloadFile({path:remote+'/'+out+'/summary.json'},{path:root+'/summary.json'});
   const result=JSON.parse(await readFile(root+'/state.json','utf8'));
   state.status='collected';state.command.collected=true;state.resultStatus=result.status;state.completed=result.completed_clips??result.recordings?.length;await save();
   await sb.update({networkPolicy:'deny-all'});await sb.stop();
   const checked=await Sandbox.get({...account,name,resume:false});state.remoteStatus=checked.status;state.computeStopped=checked.status==='stopped';await save();
   console.log(JSON.stringify({cohort,status:state.resultStatus,completed:state.completed,computeStopped:state.computeStopped}));
  }finally{await lock.close();await unlink(root+'/.collector.lock');}
  return;
 }
 if(state)throw new Error('Existing launch; inspect receipts, never duplicate');
 const lock=await open(root+'/.launcher.lock','wx');let sb;
 try{
  const baseline=await Snapshot.get({...account,snapshotId:prepared.snapshotId});
  if(!baseline.regions.includes('iad1'))throw new Error('Snapshot unavailable in iad1');
  const files=await codeFiles(),payload=[],hashes={};
  function environment(){return JSON.parse(execFileSync('.venv/bin/python',['-c',"import json;from stt_bench.credentials import command_environment;print(json.dumps(command_environment('assemblyai')))"],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));}
  const firstEnv=environment();
  for(const path of files){const content=await readFile(path);if(content.includes(Buffer.from(firstEnv.ASSEMBLYAI_API_KEY)))throw new Error('Credential in source payload');hashes[path]=await digestFile(path);payload.push({path:remote+'/'+path,content});}
  const manifestPath=cohort==='pipecat'?'datasets/pipecat-stt-benchmark/3fe50170d520c951957b86996ef082a6ab87b394/full/manifest.json':'reports/assemblyai-private-20260914/dataset/manifest.json';
  const manifestHash=await digestFile(manifestPath),bundle='reports/assemblyai-private-20260914/dataset.tar.gz';
  const bundleHash=cohort==='private'?await digestFile(bundle):null;
  state={cohort,name,runId,status:'creating',teamId:plan.teamId,projectId:plan.projectId,hashes,manifestHash,bundleHash,commands:[],startedAt:new Date().toISOString()};await save();
  if(args.recover){
   const prior=JSON.parse(await readFile(baseRoot+'/launch.json','utf8'));
   if(prior.status!=='collected'||prior.resultStatus!=='smoke_failed'||!prior.computeStopped)throw new Error('Recovery requires preserved failed-smoke evidence and confirmed stop');
   sb=await Sandbox.get({...account,name,resume:false});
   if(sb.status!=='stopped')throw new Error('Recovery target is not stopped');
   await sb.update({networkPolicy:'deny-all'});await sb.resume();
   state.recoveryFrom=baseRoot;
  }else sb=await Sandbox.create({...account,name,source:{type:'snapshot',snapshotId:prepared.snapshotId},region:'iad1',resources:{vcpus:2},timeout:28800000,persistent:true,ports:[],env:{},networkPolicy:'deny-all'});
  state.sandboxId=sb.sandboxId;state.status='uploading';await save();
  if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Unexpected sandbox resources');
  for(let i=0;i<payload.length;i+=30)await sb.writeFiles(payload.slice(i,i+30));
  if(cohort==='private'&&!args.recover)await upload(sb,bundle,remote+'/assembly-private-parts');
  await sb.writeFiles([{path:remote+'/assembly-code-hashes.json',content:Buffer.from(JSON.stringify(hashes))}]);
  let setup=`import hashlib,pathlib,json,tarfile,subprocess\nfor p,h in json.loads(pathlib.Path('assembly-code-hashes.json').read_text()).items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h\nassert not pathlib.Path('.env').exists()\n`;
  if(cohort==='private')setup+=`with pathlib.Path('assembly-private.tar.gz').open('wb') as target:\n for p in sorted(pathlib.Path('assembly-private-parts').glob('part-*')):target.write(p.read_bytes())\nwith pathlib.Path('assembly-private.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==${JSON.stringify(bundleHash)}\nwith tarfile.open('assembly-private.tar.gz') as t:t.extractall('assembly-private',filter='data')\nfrom scripts.assemblyai_private_run import verify_dataset\nverify_dataset(pathlib.Path('assembly-private/dataset/manifest.json'))\n`;
  else setup+=`from stt_bench.huggingface_data import verify_prepared\nfrom stt_bench.catalog import dataset_definition\np=verify_prepared(dataset_definition('pipecat-stt-benchmark'))/'full/manifest.json'\nassert hashlib.sha256(p.read_bytes()).hexdigest()==${JSON.stringify(manifestHash)}\nassert len(json.loads(p.read_text())['clips'])==1000\n`;
  setup+=`subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_assemblyai.py','tests/test_providers.py','--tb=short'],check=True)\nprint('Frozen data, uploaded code, and offline provider tests verified')\n`;
  state.status='verifying';await save();
  const verified=await dispatchOnce(sb,state,save,'verify',{cmd:'.venv/bin/python',cwd:remote,args:['-c',setup],env:{}});
  await writeFile(root+'/setup.log',await (await sb.currentSession().getCommand(verified.commandId)).output('both'));
  verified.collected=true;state.commands.push({...verified});await save();
  if(verified.exitCode!==0)throw new Error('Remote validation failed');
  // Re-read .env immediately before provider dispatch so mid-setup key edits apply.
  const env=environment();if(payload.some(f=>f.content.includes(Buffer.from(env.ASSEMBLYAI_API_KEY))))throw new Error('Credential in payload');
  await sb.update({networkPolicy:{allow:['streaming.assemblyai.com']}});
  state.status='dispatching';state.command={stage:'benchmark',sessionId:sb.currentSession().sessionId,status:'dispatching'};await save();
  const runner=cohort==='pipecat'?'scripts/assemblyai_pipecat_run.py':'scripts/assemblyai_private_run.py';
  const extra=cohort==='pipecat'?['--run-id',runId]:['--dataset','assembly-private/dataset'];
  const command=await sb.runCommand({cmd:'.venv/bin/python',cwd:remote,args:['-u',runner,'--live','--manifest-sha256',manifestHash,'--session-id',sb.currentSession().sessionId,'--out',out,...extra],env,detached:true});
  state.command.commandId=command.cmdId;state.command.status='running';state.status='running';await save();
  console.log(JSON.stringify({cohort,sandbox:name,status:'running',commandId:command.cmdId}));
 }catch(error){
  if(state){state.errorType=error.name;await save();}
  // Never interrupt an uncertain provider dispatch. Earlier setup failures can stop.
  if(sb&&!['dispatching','running'].includes(state.status)){await sb.stop();state.computeStopped=true;await save();}
  throw error;
 }finally{await lock.close();await unlink(root+'/.launcher.lock');}
}
main().catch(error=>{console.error(`AssemblyAI operation stopped (${error.name}); inspect saved receipts and logs.`);process.exitCode=1;});
