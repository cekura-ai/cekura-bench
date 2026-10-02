// Explicit one-pass <=300 seconds for Gradium and Reson8 on three frozen public cohorts.
import {readFile,writeFile,mkdir,rename,open,unlink} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {codeFiles} from './setup_vercel_chirp.mjs';
import {dispatchOnce,digestFile} from './vercel_models.mjs';
import {REMOTE} from './setup_vercel_trial_providers.mjs';
const {values:args}=parseArgs({options:{live:{type:'boolean',default:false},'run-id':{type:'string',default:'limited-gradium-reson8-20260914'}}});
if(!/^[a-zA-Z0-9-]+$/.test(args['run-id']))throw new Error('Invalid run ID');
const runId=args['run-id'],root=resolve('reports',runId);
const inputPlanPath=join(root,'plan.json');
const inputPlan=JSON.parse(await readFile(inputPlanPath,'utf8'));
const planHash=await digestFile(inputPlanPath);
execFileSync('.venv/bin/python',['-c',
 'import sys; from pathlib import Path; from scripts.credit_benchmark_run import validate_plan; validate_plan(Path(sys.argv[1]),sys.argv[2])',inputPlanPath,planHash],{stdio:['ignore','pipe','pipe']});
const MODELS=inputPlan.models.map(m=>m.model);
if(!args.live){console.log(JSON.stringify({models:MODELS,clipsPerModel:inputPlan.planned_clips,secondsPerModel:inputPlan.planned_audio_seconds_per_provider,attemptsPerClip:1,planHash,status:'no_remote_actions'}));process.exit(0);}
const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
let auth=getAuth();if(!auth?.token)throw new Error('Vercel login required');
if(auth.expiresAt?.getTime()<Date.now()&&auth.refreshToken){const t=await (await OAuth()).refreshToken(auth.refreshToken);auth={token:t.access_token,refreshToken:t.refresh_token??auth.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};updateAuthConfig(auth);}
const plan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
const account={token:auth.token,teamId:plan.teamId,projectId:plan.projectId};
const files=await codeFiles();
const payload=[];const hashes={};
for(const file of files){const content=await readFile(file);hashes[file]=createHash('sha256').update(content).digest('hex');payload.push({path:REMOTE+'/'+file,content});}
const datasetRelative=`credit-trials/${runId}/input`;
const inputFiles=['plan.json'];
for(const group of inputPlan.datasets){
 const m=JSON.parse(await readFile(join(root,group.manifest),'utf8'));
 inputFiles.push(group.manifest,...m.clips.map(c=>`datasets/${group.id}/${c.audio}`));
}
for(const file of inputFiles){
 const content=await readFile(join(root,file));
 const relative=datasetRelative+'/'+file;hashes[relative]=createHash('sha256').update(content).digest('hex');
 payload.push({path:REMOTE+'/'+relative,content});
}
const lockPath=join(root,'.launcher.lock');const lock=await open(lockPath,'wx');
try{
 const results=await Promise.allSettled(MODELS.map(async model=>{
  const dir=join(root,model);await mkdir(dir,{recursive:true});
  const name=inputPlan.models.find(m=>m.model===model).sandbox;
  const statePath=join(dir,'launch.json');let state;
  try{state=JSON.parse(await readFile(statePath,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  const identity={model,sandbox:name,planHash,hashes,teamId:plan.teamId,projectId:plan.projectId};
  if(state&&JSON.stringify(state.identity)!==JSON.stringify(identity))throw new Error('Launch inputs changed; do not overwrite run');
  state??={identity,status:'new',commands:[],maxProviderAttempts:inputPlan.planned_clips,maxAudioSeconds:300};
  async function save(){state.updatedAt=new Date().toISOString();await writeFile(statePath+'.tmp',JSON.stringify(state,null,2)+'\n');await rename(statePath+'.tmp',statePath);}
  let sb=await Sandbox.get({...account,name,resume:false});
  if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Sandbox resource mismatch');
  const remoteOut=`credit-trials/${runId}/${model}`;
  try{
   if(state.status==='complete'){console.log(JSON.stringify({model,status:'already_collected'}));return;}
   if(state.status==='new'){
    const config=JSON.parse(await readFile(`config/models/${model}.json`,'utf8'));
    const env=JSON.parse(execFileSync('.venv/bin/python',['-c','import json,sys; from stt_bench.credentials import command_environment; print(json.dumps(command_environment(sys.argv[1])))',config.provider],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
    // Check the upload against this credential before it can leave the machine.
    if(payload.some(f=>Object.values(env).some(v=>f.content.includes(Buffer.from(v)))))throw new Error('Credential in upload');
    await sb.update({networkPolicy:'deny-all'});
    if(sb.status!=='running')await sb.resume();
    state.status='uploading';await save();
    for(let i=0;i<payload.length;i+=30)await sb.writeFiles(payload.slice(i,i+30));
    const verify=`import hashlib,pathlib\nexpected=${JSON.stringify(hashes)}\nfor p,h in expected.items():\n assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h\nfrom scripts.credit_benchmark_run import validate_plan\nvalidate_plan(pathlib.Path(${JSON.stringify(datasetRelative+'/plan.json')}),${JSON.stringify(planHash)})\nimport subprocess\nsubprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_credit_benchmark.py','tests/test_gradium.py','tests/test_reson8.py'],check=True)\nprint('Code, bounded public dataset plan and offline adapter tests verified')\n`;
    const v=await dispatchOnce(sb,state,save,'verify',{cmd:'.venv/bin/python',cwd:REMOTE,args:['-c',verify],env:{}});
    if(v.exitCode!==0)throw new Error('Remote code/data verification failed');v.collected=true;state.commands.push({...v});
    await sb.update({networkPolicy:{allow:[new URL(config.endpoint).hostname]}});
    state.status='running';await save();
    const receipt=await dispatchOnce(sb,state,save,'benchmark',{cmd:'.venv/bin/python',cwd:REMOTE,
      args:['-u','scripts/credit_benchmark_run.py','--live','--model',model,'--plan',datasetRelative+'/plan.json',
        '--plan-sha256',planHash,'--session-id',sb.currentSession().sessionId,'--out',remoteOut],env});
    state.commands.push({...receipt});
   }else if(state.status==='running'){
    // Reattach only to the saved command; never dispatch a second benchmark.
    if(!state.command?.commandId)throw new Error('Uncertain dispatch: inspect remote commands');
    await dispatchOnce(sb,state,save,'benchmark',{});
   }else throw new Error('Interrupted preparation requires explicit reconciliation');
   const receipt=state.command;
   await writeFile(join(dir,'run.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
   for(const file of ['evidence.tar.gz','evidence.sha256','state.json'])
     await sb.currentSession().downloadFile({path:REMOTE+'/'+remoteOut+'/'+file},{path:join(dir,file)});
   const expected=(await readFile(join(dir,'evidence.sha256'),'utf8')).split(/\s+/)[0];
   if(await digestFile(join(dir,'evidence.tar.gz'))!==expected)throw new Error('Evidence archive checksum mismatch');
   const progress=JSON.parse(await readFile(join(dir,'state.json'),'utf8'));
   state.result=progress;state.status='complete';receipt.collected=true;await save();
   console.log(JSON.stringify({model,...progress}));
   return {model,...progress};
  }finally{
   // Attempt stop even if the policy update itself fails.
   try{await sb.update({networkPolicy:'deny-all'});}finally{if(sb.status!=='stopped'&&sb.status!=='stopping')await sb.stop();}
   let readback=await Sandbox.get({...account,name,resume:false});
   for(let i=0;readback.status==='stopping'&&i<15;i++){await new Promise(r=>setTimeout(r,1000));readback=await Sandbox.get({...account,name,resume:false});}
   state.remoteStatus=readback.status;state.networkPolicy=readback.networkPolicy;
   state.computeStopped=readback.status==='stopped';await save();
   if(!state.computeStopped || state.networkPolicy!=='deny-all')throw new Error('Stopped deny-all state not confirmed');
  }
 }));
 await writeFile(join(root,'launcher-outcomes.json'),JSON.stringify(results.map((r,i)=>({model:MODELS[i],status:r.status,
   ...(r.status==='fulfilled'?{result:r.value}:{errorType:r.reason.name,errorMessage:String(r.reason.message)})})),null,2)+'\n');
 if(results.some(r=>r.status==='rejected'))throw new Error('Some models need evidence inspection');
}finally{await lock.close();await unlink(lockPath);}
