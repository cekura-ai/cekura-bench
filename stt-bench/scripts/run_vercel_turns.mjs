// Turn-only, durable Vercel controller. Never invokes whole-recording runners.
import {readFile,writeFile,mkdir,rename,open,unlink} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {setTimeout as delay} from 'node:timers/promises';
import {digestFile,upload,dispatchOnce} from './vercel_models.mjs';
import {waitForCompletion} from './vercel_command_wait.mjs';
const exec=promisify(execFile),REMOTE='/vercel/sandbox/stt-turns';

export function workerPolicy(config){return {allow:[new URL(config.endpoint).hostname,
  ...(config.provider==='google'?['oauth2.googleapis.com']:[])]};}
export function workerName(model,index,prefix='vocera-turns-0915'){return `${prefix}-${model}-${index}`.replace(/[^a-zA-Z0-9_-]/g,'-');}

async function uploadVerifiedParts(sb,path,destination){
  const inspection=await sb.runCommand({cmd:'python3',args:['-c',
    'import pathlib,hashlib,json; print(json.dumps({p.name:hashlib.file_digest(p.open("rb"),"sha256").hexdigest() for p in pathlib.Path('+JSON.stringify(destination)+').glob("part-*")}))']});
  if(inspection.exitCode!==0)throw new Error('Cannot inspect uploaded parts');
  const prior=JSON.parse(await inspection.output('stdout')),file=await open(path,'r');
  try{const size=(await file.stat()).size;let next=0;
    await Promise.all(Array.from({length:4},async()=>{while(true){const i=next++,offset=i*4000000;if(offset>=size)return;
      const buffer=Buffer.alloc(Math.min(4000000,size-offset));let used=0;
      while(used<buffer.length){const r=await file.read(buffer,used,buffer.length-used,offset+used);if(!r.bytesRead)throw new Error('Bundle EOF');used+=r.bytesRead;}
      const name='part-'+String(i).padStart(6,'0');
      if(prior[name]!==createHash('sha256').update(buffer).digest('hex'))await sb.writeFiles([{path:destination+'/'+name,content:buffer}]);
    }}));
  }finally{await file.close();}
}

export function smokeRowPassed(plan,model,r){
  if(!r?.valid)return false;
  const t=r.turn_timing||{};
  if(plan.smoke_policy==='completed-empty-allowed-v1'&&t.ttft_status==='no_text'&&t.ttfs_status==='no_final_text'&&t.ttft_ms==null&&t.ttfs_ms==null)return true;
  return t.ttft_ms!=null&&(plan.configs[model].turn_finalization_class!=='controlled'||t.ttfs_ms!=null);
}
export function smokePassed(plan,model){return plan.smoke_ids.every(id=>smokeRowPassed(plan,model.name,model.rows[id]));}
export function claim(plan,state,model,id,now=Date.now()){
  const m=state.models[model],provider=plan.configs[model].provider,p=state.providers[provider];
  if(m.blocked||now<p.cooldown||Object.keys(m.active).length>=m.ceiling)return null;
  if(Object.values(state.models).reduce((n,x)=>n+Object.keys(x.active).length,0)>=plan.max_workers)return null;
  const active=Object.entries(state.models).filter(([name])=>plan.configs[name].provider===provider).reduce((n,[,x])=>n+Object.keys(x.active).length,0);
  if(active+(p.external||0)>=p.limit)return null;
  if(provider==='assemblyai'){
    p.started=p.started.filter(t=>now-t<65000);
    // Reserve admission before verification/connection setup. Count confirmed
    // starts plus pending reservations, allowing safe parallel preparation.
    if(p.started.length+(p.pendingStarts||0)>=plan.assemblyai_new_sessions_per_minute)return null;
  }
  const held=new Set(Object.values(m.active).flat());
  const missing=plan.smoke_ids.filter(cid=>!m.rows[cid]&&!held.has(cid));
  let ids;
  if(!smokePassed(plan,m)){
    if(plan.smoke_ids.some(cid=>m.rows[cid]&&!smokeRowPassed(plan,model,m.rows[cid]))){m.blocked='smoke_failed';return null;}
    if(!missing.length)return null;
    ids=missing.slice(0,1);
  }else{
    ids=plan.clip_ids.filter(cid=>!m.rows[cid]&&!held.has(cid)).sort((a,b)=>plan.durations[b]-plan.durations[a]).slice(0,provider==='assemblyai'||plan.single_clip_batches?1:4);
  }
  if(!ids.length)return null;
  m.active[id]=ids;m.peak=Math.max(m.peak,Object.keys(m.active).length);
  if(provider==='assemblyai')p.pendingStarts++;
  return ids;
}
export function accept(plan,state,model,id,rows){
  const m=state.models[model],p=state.providers[plan.configs[model].provider],assigned=m.active[id];
  for(const r of rows){
    if(!assigned.includes(r.clip_id)||m.rows[r.clip_id]||r.attempt!==1)throw new Error('Duplicate or unexpected turn result');
    m.rows[r.clip_id]={clip_id:r.clip_id,attempt:r.attempt,valid:r.valid,failure_class:r.failure_class,
      raw_sha256:r.raw_sha256,turn_timing:{ttft_ms:r.turn_timing?.ttft_ms,ttfs_ms:r.turn_timing?.ttfs_ms,
        ttft_status:r.turn_timing?.ttft_status,ttfs_status:r.turn_timing?.ttfs_status}};
    if(['credits','authentication','model_identity','quota'].includes(r.failure_class))m.blocked=r.failure_class;
    if(r.failure_class==='concurrency'){
      p.limit=Math.max(1,Math.floor(p.limit/2));p.cooldown=Date.now()+61000;p.throttled=true;
      p.reductions.push({at:new Date().toISOString(),limit:p.limit,reason:'provider_rate_limit'});
    }
  }
  const successes=Object.values(m.rows).filter(r=>r.valid).length;
  m.ceiling=Math.min(plan.max_workers_per_model,successes>=16?32:successes>=10?16:successes>=1?4:1);
  if(plan.minimum_batch_interval_ms)p.cooldown=Math.max(p.cooldown,Date.now()+plan.minimum_batch_interval_ms);
  delete m.active[id];
}
export function compact(state){return {status:state.status,updatedAt:state.updatedAt,models:Object.fromEntries(Object.entries(state.models).map(([k,m])=>[k,{attempted:Object.keys(m.rows).length,valid:Object.values(m.rows).filter(r=>r.valid).length,active:Object.keys(m.active).length,peak:m.peak,blocked:m.blocked}])),providers:state.providers};}

export async function main(argv=process.argv.slice(2)){
  const mode=argv[0]||'status',root=resolve(argv[1]||'reports/private-turns-vercel-20260915');
  if(!['prepare','run','status'].includes(mode))throw new Error('Use prepare, run or status');
  const statePath=join(root,'controller.json');let state;
  try{state=JSON.parse(await readFile(statePath,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  if(mode==='status'){console.log(JSON.stringify(state?compact(state):{status:'not_launched'}));return;}
  const plan=JSON.parse(await readFile(join(root,'plan.json'),'utf8')),identity=JSON.parse(await readFile(join(root,'input.json'),'utf8'));
  const uploadDir=REMOTE+'/upload-'+identity.bundle_sha256.slice(0,12);
  if(await digestFile(join(root,'plan.json'))!==identity.plan_sha256)throw new Error('Frozen plan changed');
  try{if(await digestFile(join(root,'input.tar.gz'))!==identity.bundle_sha256)throw new Error('Frozen bundle changed');}
  catch(error){
    if(error.code!=='ENOENT'||!state?.preparation?.verified||
       identity.remote_bundle?.snapshot_id!==state.preparation.snapshotId||
       identity.remote_bundle?.sha256!==identity.bundle_sha256)throw error;
  }
  const accountPlan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
  const baseline=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
  if(baseline.status!=='ready'||baseline.teamId!==accountPlan.teamId||baseline.projectId!==accountPlan.projectId)throw new Error('Account/snapshot mismatch');
  const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
  const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
  const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
  let refreshing;
  async function account(){let a=getAuth();if(!a?.token)throw new Error('Vercel login missing');
    if(a.expiresAt?.getTime()<Date.now()+300000){refreshing??=(async()=>{const t=await(await OAuth()).refreshToken(a.refreshToken);updateAuthConfig({token:t.access_token,refreshToken:t.refresh_token??a.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)});})();try{await refreshing;}finally{refreshing=null;}a=getAuth();}
    return {token:a.token,teamId:accountPlan.teamId,projectId:accountPlan.projectId};}
  async function get(name){return Sandbox.get({...await account(),name,resume:false});}
  const lockPath=join(root,'.controller.lock'),lock=await open(lockPath,'wx');await lock.writeFile(String(process.pid));
  let writes=Promise.resolve();function save(){state.updatedAt=new Date().toISOString();const value=JSON.stringify(state,null,2)+'\n';writes=writes.then(async()=>{await writeFile(statePath+'.tmp',value);await rename(statePath+'.tmp',statePath);});return writes;}
  async function command(sb,w,stage,params){return dispatchOnce(sb,w,save,stage,params,async(_,cid)=>waitForCompletion({getCommand:async(id,opts)=>{const fresh=await get(w.name);if(fresh.currentSession().sessionId!==w.command.sessionId)throw new Error('Worker session changed');return fresh.currentSession().getCommand(id,opts);}},cid));}
  async function stop(w){let sb=await get(w.name);
    for(let i=0;sb.status==='snapshotting'&&i<120;i++){await delay(1000);sb=await get(w.name);}
    if(!['stopped','stopping','snapshotting'].includes(sb.status)){await sb.update({networkPolicy:'deny-all'});await sb.stop();}
    let s=sb;for(let i=0;i<60;i++){s=await get(w.name);if(s.status==='stopped')break;await delay(1000);}
    w.computeStopped=s.status==='stopped';await save();if(!w.computeStopped)throw new Error('Worker stop unconfirmed');}
  let creating=0;
  async function ensure(w,snapshot,policy){
    if(w.stage==='new'){
      while(creating>=8)await delay(300);creating++;
      try{w.stage='creating';await save();const sb=await Sandbox.create({...await account(),name:w.name,source:{type:'snapshot',snapshotId:snapshot},region:'iad1',resources:{vcpus:2},timeout:86400000,persistent:true,ports:[],env:{},networkPolicy:policy});w.sessionId=sb.currentSession().sessionId;w.stage='ready';w.computeStopped=false;await save();return sb;}finally{creating--;}
    }
    let sb=await get(w.name);
    if(sb.status==='stopped'&&w.command?.collected){
      w.previousSessions??=[];w.previousSessions.push(w.sessionId);
      await sb.resume();sb=await get(w.name);await sb.update({networkPolicy:policy});
      w.sessionId=sb.currentSession().sessionId;w.computeStopped=false;await save();
    }
    if(w.stage==='creating'||sb.status!=='running')throw new Error('Saved worker needs reconciliation');return sb;
  }
  try{
    if(state&&state.planHash!==identity.plan_sha256)throw new Error('Controller plan changed');
    state??={status:'preparing',planHash:identity.plan_sha256,startedAt:new Date().toISOString(),nextBatch:0,
      models:Object.fromEntries(plan.models.map(name=>[name,{name,
        rows:Object.fromEntries((plan.replayed_smoke||[]).filter(r=>r.model===name).map(r=>[r.clip_id,r])),
        active:{},ceiling:(plan.replayed_smoke||[]).filter(r=>r.model===name).length===plan.smoke_ids.length?16:1,peak:0,blocked:null}])),
      providers:Object.fromEntries(Object.entries(plan.provider_limits).map(([name,limit])=>[name,{limit,cooldown:0,external:0,started:[],pendingStarts:0,reductions:[]}])),
      workers:{},batches:{},preparation:{name:plan.preparation_name||'vocera-turns-20260915-base',stage:'new'}};
    await save();const prep=state.preparation;
    if(!prep.snapshotId){
      let sb=await ensure(prep,identity.base_snapshot_id||baseline.snapshotId,{allow:accountPlan.setupDomains});
      if(!prep.uploaded){
        let c=await command(sb,prep,'mkdir',{cmd:'mkdir',args:['-p',uploadDir]});if(c.exitCode!==0)throw new Error('mkdir failed');c.collected=true;await save();
        await uploadVerifiedParts(sb,join(root,'input.tar.gz'),uploadDir);prep.uploaded=true;await save();
      }
      if(!prep.verified){
        const code=`import pathlib,hashlib,tarfile,subprocess,shutil\nr=pathlib.Path(${JSON.stringify(REMOTE)})\np=r/'input.tar.gz'\nwith p.open('wb') as f:\n for part in sorted(pathlib.Path(${JSON.stringify(uploadDir)}).glob('part-*')): f.write(part.read_bytes())\nassert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==${JSON.stringify(identity.bundle_sha256)}\nwith tarfile.open(p) as t:t.extractall(r,filter='data')\nuv=shutil.which('uv') or str(pathlib.Path.home()/'.local/bin/uv')\nsubprocess.run([uv,'sync','--locked','--python','3.12'],cwd=r,check=True)\nsubprocess.run(['.venv/bin/python','-m','stt_bench.turn_batch','verify','--plan-hash',${JSON.stringify(identity.plan_sha256)}],cwd=r,check=True)\nsubprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_turns.py','tests/test_turn_shards.py','--tb=short'],cwd=r,check=True)\nprint('Turn-only inputs and offline tests verified; no provider calls')`;
        const c=await command(sb,prep,'verify',{cmd:'python3',cwd:REMOTE,args:['-c',code]});
        await writeFile(join(root,'preparation.log'),await(await sb.currentSession().getCommand(c.commandId)).output('both'));
        if(c.exitCode!==0)throw new Error('Turn snapshot validation failed');c.collected=true;prep.verified=true;await save();
      }
      if(prep.snapshotPending)throw new Error('Uncertain snapshot requires reconciliation');
      await sb.update({networkPolicy:'deny-all'});prep.snapshotPending=true;await save();const snap=await sb.snapshot();prep.snapshotId=snap.snapshotId;await save();await stop(prep);
    }
    if(!prep.computeStopped)await stop(prep);
    if(mode==='prepare'){console.log(JSON.stringify({status:'prepared',snapshot:prep.snapshotId}));return;}
    const envs={};for(const model of plan.models){const provider=plan.configs[model].provider;
      try{const x=await exec('.venv/bin/python',['-c','import json,sys;from stt_bench.credentials import command_environment;print(json.dumps(command_environment(sys.argv[1])))',provider]);envs[model]=JSON.parse(x.stdout);}catch{state.models[model].blocked='credential_unavailable';}}
    async function external(){const own=new Set([prep.name,...Object.values(state.workers).map(w=>w.name)]);let total=0,speech=0;
      for await(const s of await Sandbox.list(await account()))if(!own.has(s.name)&&['running','pending','stopping','snapshotting'].includes(s.status)){total++;if(/speechmatics|linden/i.test(s.name))speech++;}
      state.providers.speechmatics.external=speech;state.otherActiveSandboxes=total;await save();}
    await external();state.status='running';await save();let lastExternal=Date.now();
    async function loop(model,index){const id=model+'-'+index,m=state.models[model],provider=plan.configs[model].provider,p=state.providers[provider];let w=state.workers[id];
      try{while(true){
        if(provider==='speechmatics'&&p.external&&Date.now()-lastExternal>30000){lastExternal=Date.now();await external();}
        let b=w?.batch?state.batches[w.batch]:null;
        if(!b){
          // Five admitted AssemblyAI starts per minute do not need 32 cold VMs.
          // Reuse qualified workers instead of serially warming new ones.
          if((provider==='assemblyai'&&index>=5)||(provider==='speechmatics'&&index>=2))break;
          if(m.blocked||Object.keys(m.rows).length===plan.clip_ids.length)break;
          const ids=claim(plan,state,model,id);
          if(!ids){await delay(750);continue;}
          const batchId=String(state.nextBatch++).padStart(6,'0');
          w??={id,name:workerName(model,index,plan.worker_prefix),stage:'new'};state.workers[id]=w;
          const a={model,plan_hash:identity.plan_sha256,clip_ids:ids,batch_id:batchId,
            ...(smokePassed(plan,m)?{smoke_receipt:{policy:plan.smoke_policy,manifest_sha256:plan.manifest_sha256,config_sha256:plan.config_hashes[model],rows:plan.smoke_ids.map(cid=>m.rows[cid])}}:{})};
          b={id:batchId,model,assignment:a,status:'assigned'};state.batches[batchId]=b;w.batch=batchId;await save();
        }
        let sb=await ensure(w,prep.snapshotId,workerPolicy(plan.configs[model]));
        const remote=`turn-results/${b.id}`;
        if(b.status==='assigned'){
          await sb.writeFiles([{path:REMOTE+`/turn-input/assignment-${b.id}.json`,content:Buffer.from(JSON.stringify(b.assignment))}]);
          if(provider==='assemblyai'){
            while(true){p.started=p.started.filter(t=>Date.now()-t<65000);if(p.started.length<plan.assemblyai_new_sessions_per_minute)break;await delay(1000);}
          }
          b.status='dispatching';await save();
        }
        const completion=command(sb,w,'batch-'+b.id,{cmd:'.venv/bin/python',cwd:REMOTE,args:['-u','-m','stt_bench.turn_batch','worker','--plan-hash',identity.plan_sha256,'--assignment',`turn-input/assignment-${b.id}.json`,'--out',remote],env:{...envs[model],STT_BENCH_COMPUTE_PROVIDER:'vercel-sandbox',STT_BENCH_COMPUTE_REGION:'iad1',STT_BENCH_COMPUTE_INSTANCE:w.sessionId}}).then(value=>({value}),error=>({error}));
        if(provider==='assemblyai'&&!b.startObserved){
          // Count the connection after it actually starts. Counting the earlier
          // sandbox dispatch allowed delayed launches to bunch into six/minute.
          let observed=false;
          for(let poll=0;poll<90;poll++){
            let raw;
            try{raw=await sb.currentSession().readFileToBuffer({path:REMOTE+'/'+remote+'/run/raw/'+b.assignment.clip_ids[0]+'--attempt-1.jsonl'});}catch{}
            if(raw?.toString().includes('"kind": "connection_requested"')){observed=true;break;}
            if(w.command?.status==='finished')break;
            await delay(2000);
          }
          if(!observed&&w.command?.status!=='finished')throw new Error('Actual AssemblyAI connection start requires reconciliation');
          if(observed)p.started.push(Date.now());
          b.startObserved=observed?'connection_confirmed':'finished_without_connection';
          p.pendingStarts=Math.max(0,p.pendingStarts-1);await save();
        }
        const completed=await completion;if(completed.error)throw completed.error;const receipt=completed.value;
        sb=await get(w.name);const folder=join(root,'batches',b.id);await mkdir(folder,{recursive:true});
        for(const file of ['state.json','evidence.tar.gz','evidence.sha256'])await sb.currentSession().downloadFile({path:REMOTE+'/'+remote+'/'+file},{path:join(folder,file)});
        await writeFile(join(folder,'worker.log'),await(await sb.currentSession().getCommand(receipt.commandId)).output('both'));
        if(await digestFile(join(folder,'evidence.tar.gz'))!==(await readFile(join(folder,'evidence.sha256'),'utf8')).trim())throw new Error('Archive hash mismatch');
        const result=JSON.parse(await readFile(join(folder,'state.json'),'utf8'));
        if(JSON.stringify(result.assignment)!==JSON.stringify(b.assignment))throw new Error('Assignment receipt differs');
        accept(plan,state,model,id,result.rows);
        if(result.status!=='finished'){m.blocked='worker_failed';m.workerError=result.error;}
        // A failed full-run attempt is retained. Only IDs that were never
        // started return to the queue; the original ten-turn smoke gate stays.
        if(result.rows.length<b.assignment.clip_ids.length){
          b.unstarted=b.assignment.clip_ids.filter(cid=>!result.rows.some(r=>r.clip_id===cid));
          if(!smokePassed(plan,m))m.blocked??='batch_incomplete';
        }
        b.status='collected';b.exitCode=receipt.exitCode;b.archiveHash=await digestFile(join(folder,'evidence.tar.gz'));w.command.collected=true;w.batch=null;await save();
        console.log(JSON.stringify({model,attempted:Object.keys(m.rows).length,valid:Object.values(m.rows).filter(r=>r.valid).length,peak:m.peak,blocked:m.blocked}));
      }}catch(e){m.blocked='controller_reconciliation_required';m.error=String(e.message).slice(0,250);await save();}
      finally{if(w&&!w.batch){try{await stop(w);}catch(e){w.stopError=String(e.message);await save();}}}
    }
    await Promise.all(plan.models.flatMap(model=>Array.from({length:plan.max_workers_per_model},(_,i)=>loop(model,i))));
    state.status=Object.values(state.models).every(m=>Object.keys(m.rows).length===plan.clip_ids.length&&!m.blocked)?'complete':'partial_or_blocked';state.finishedAt=new Date().toISOString();await save();
    const report=await exec('.venv/bin/python',['-m','stt_bench.turn_batch','report','--root',root],{maxBuffer:2**20});console.log(report.stdout);
  }finally{await writes;await lock.close();await unlink(lockPath);}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)main().catch(e=>{console.error(JSON.stringify({errorType:e.name,message:String(e.message).slice(0,500)}));process.exitCode=1;});
