// Dedicated durable controller. One stream per sandbox; no scheduled automation.
import {readFile,writeFile,rename,mkdir,open,unlink} from 'node:fs/promises';
import {execFile,execFileSync} from 'node:child_process';
import {promisify} from 'node:util';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {setTimeout as delay} from 'node:timers/promises';
import {digestFile,upload,dispatchOnce} from './vercel_models.mjs';
import {waitForCompletion} from './vercel_command_wait.mjs';

const exec=promisify(execFile);
const REMOTE='/vercel/sandbox/stt-bench-v4';
// Rate-limit waits can outlive the token stored in an SDK sandbox instance.
// Refresh the client before dispatch; never retry an uncertain command write.
export async function freshDispatchSandbox(previous,getFresh){
  const fresh=await getFresh();
  if(fresh.currentSession().sessionId!==previous.currentSession().sessionId)
    throw new Error('Sandbox session changed before dispatch; reconciliation required');
  return fresh;
}
// An unacknowledged reservation holds a slot indefinitely. Only a remote
// handshake/failure receipt, or collected command evidence, starts its expiry.
export function reserveStart(gate,id,worker,stamp=Date.now()){
  if(gate.limit!==5||gate.windowMs<62000)throw new Error('Expected confirmed five starts per minute');
  if(gate.entries[id])return true;
  const occupied=Object.values(gate.entries).filter(e=>e.ackAt==null||stamp-e.ackAt<gate.windowMs).length;
  if(occupied>=gate.limit)return false;
  gate.entries[id]={worker,reservedAt:stamp,ackAt:null};return true;
}
export function acknowledgeStart(gate,id,stamp=Date.now()){
  const entry=gate.entries[id];if(entry&&entry.ackAt==null)entry.ackAt=stamp;
}

export function newModel(){return {ceiling:1,successes:0,throttled:false,cooldown:0,blocked:null,privateBlocked:false,attempts:{},active:{},reductions:[],peak:0};}
export function amendUndispatched(b,command,runtimeHash){
  if(b.status!=='assigned'||b.assignment.runtime_hash===runtimeHash)return false;
  if(command?.stage==='batch-'+b.id)throw new Error('Assigned batch already has command intent; reconcile before amendment');
  b.previousAssignmentRevisions??=[];b.previousAssignmentRevisions.push(structuredClone(b.assignment));
  b.assignment.runtime_hash=runtimeHash;b.runtimeAmendedBeforeDispatch=true;return true;
}
export function reconcileCredits(m,authorization){
  const acknowledged=new Set(authorization?.acknowledged_attempts||[]);
  const failures=Object.entries(m.attempts).flatMap(([id,rows])=>rows.filter(r=>['credits','authentication'].includes(r.failure_class)).map(r=>({id:`${id}:${r.attempt}`,failure:r.failure_class})));
  const unacknowledged=failures.filter(r=>!acknowledged.has(r.id));
  if(unacknowledged.length){m.blocked=unacknowledged.some(r=>r.failure==='authentication')?'authentication':'credits';return;}
  if(!authorization)return;
  if(['credits','authentication','capacity_probe_failed'].includes(m.blocked))m.blocked=null;
  if(m.creditResume?.id!==authorization.id){
    m.capacityStageMax=Math.min(m.capacityStageMax??20,m.throttled?m.ceiling:20);
    m.creditResume=authorization;m.capacitySuccesses=0;m.ceiling=1;m.throttled=false;m.cooldown=0;
  }
}
export function claim(plan,m,stamp=Date.now()){
  if(m.blocked||stamp<m.cooldown||Object.keys(m.active).length>=m.ceiling)return null;
  const target=plan.private_variants?.[m.model];
  const privateIds=new Set(plan.items.filter(c=>c.cohort==='private').map(c=>c.clip_id));
  if(target&&m.privateVariant!==target&&!Object.values(m.active).some(a=>a.items.some(i=>privateIds.has(i.clip_id)))){
    m.previousPrivateAttempts??=[];
    for(const cid of privateIds)if(m.attempts[cid]){m.previousPrivateAttempts.push({clip_id:cid,variant:m.privateVariant||'single-session',attempts:m.attempts[cid]});delete m.attempts[cid];}
    m.privateVariant=target;m.privateBlocked=false;
  }
  const publicItems=plan.items.filter(c=>c.cohort==='public').sort((a,b)=>b.submitted_seconds-a.submitted_seconds||a.clip_id.localeCompare(b.clip_id));
  const privateItems=plan.items.filter(c=>c.cohort==='private');
  const held=new Set(Object.values(m.active).flatMap(a=>a.items.map(i=>i.clip_id)));
  const history=c=>m.attempts[c.clip_id]||[];
  // A replenished/updated credential establishes capacity again with new work.
  if(m.capacitySuccesses===0){
    if(Object.keys(m.active).length)return null;
    const probe=[...publicItems].reverse().find(c=>!history(c).length);
    if(probe)return {items:[{clip_id:probe.clip_id,attempt:1}]};
  }
  const pilot=privateItems.find(c=>c.clip_id===plan.private_pilot);
  const pilotHistory=pilot?history(pilot):[];
  const pilotPassed=pilotHistory.some(a=>a.valid);
  let items=[],attempt=1;
  if(!m.successes){
    if(Object.keys(m.active).length)return null;
    const first=publicItems.at(-1);
    if(history(first).length){
      if(history(first).length>=2){m.blocked='public_pilot_failed';return null;}
      attempt=2;
    }
    items=[first];
  } else if(pilot&&!m.privateBlocked&&!pilotPassed&&!held.has(pilot.clip_id)){
    if(pilotHistory.length>=2)m.privateBlocked=true;
    else {items=[pilot];attempt=pilotHistory.length+1;}
  }
  if(!items.length){
    if(pilotPassed&&!m.privateBlocked)items=privateItems.filter(c=>!history(c).length&&!held.has(c.clip_id)).slice(0,1);
    if(!items.length)items=publicItems.filter(c=>!history(c).length&&!held.has(c.clip_id)).slice(0,1);
  }
  if(!items.length&&(!pilot||pilotPassed||m.privateBlocked)&&!Object.values(m.active).some(a=>a.items.some(i=>i.attempt===1))){
    const eligible=[...publicItems,...(pilotPassed&&!m.privateBlocked?privateItems:[])];
    items=eligible.filter(c=>history(c).length===1&&!history(c)[0].valid&&!held.has(c.clip_id)).slice(0,1);attempt=2;
  }
  return items.length?{items:items.map(c=>({clip_id:c.clip_id,attempt}))}:null;
}
export function accept(m,assignment,rows,stamp=Date.now()){
  const allowed=new Set(assignment.items.map(i=>`${i.clip_id}:${i.attempt}`));
  const failedCapacityProbe=m.capacitySuccesses===0&&rows.some(r=>!r.valid);
  for(const r of rows){
    if(!allowed.has(`${r.clip_id}:${r.attempt}`))throw new Error('Unexpected result identity');
    const history=m.attempts[r.clip_id]??=[];
    if(history.some(a=>a.attempt===r.attempt))throw new Error('Duplicate attempt receipt');
    if(r.attempt!==history.length+1)throw new Error('Attempt sequence mismatch');
    history.push({attempt:r.attempt,valid:r.valid,failure_class:r.failure_class});
    if(r.valid){m.successes++;if(m.capacitySuccesses!==undefined)m.capacitySuccesses++;}
    if(['credits','authentication','model_identity'].includes(r.failure_class))m.blocked=r.failure_class;
    if(r.failure_class==='concurrency'){
      const from=m.ceiling;
      // Multiple rejections from one wave must not repeatedly halve the cap.
      if(stamp>=m.cooldown)m.ceiling=Math.max(1,Math.floor(from/2));m.throttled=true;
      m.cooldown=Math.max(m.cooldown,stamp+Math.max(30,r.retry_after_seconds||0)*1000);
      m.reductions.push({at:new Date(stamp).toISOString(),from,to:m.ceiling,reason:'provider_concurrency'});
    }
  }
  if(failedCapacityProbe&&!m.blocked)m.blocked='capacity_probe_failed';
  const capacity=m.capacitySuccesses??m.successes;
  if(!m.throttled)m.ceiling=Math.min(m.capacityStageMax??20,capacity>=16?20:capacity>=6?10:capacity>=1?5:1);
}
export function compact(state){return {run:state.runId,status:state.status,updatedAt:state.updatedAt,
  startRate:state.startRate?{limit:state.startRate.limit,windowMs:state.startRate.windowMs,unacknowledged:Object.values(state.startRate.entries).filter(e=>e.ackAt==null).length}:null,
  models:Object.fromEntries(Object.entries(state.models||{}).map(([k,m])=>[k,{ceiling:m.ceiling,peak:m.peak,
    active:Object.keys(m.active).length,attempted:Object.keys(m.attempts).length,
    successful:Object.values(m.attempts).filter(a=>a.some(x=>x.valid)).length,blocked:m.blocked,privateBlocked:m.privateBlocked,
    reductions:m.reductions}])),workersStopped:Object.values(state.workers||{}).filter(w=>w.computeStopped).length};}

export async function main(argv=process.argv.slice(2)){
  const mode=argv[0]||'status',root=resolve(argv[1]||'reports/assemblyai-min-latency-full-20260915');
  if(!['prepare','run','status'].includes(mode))throw new Error('Use prepare, run, or status');
  const path=join(root,'controller.json');
  let state;try{state=JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  if(mode==='status'){console.log(JSON.stringify(state?compact(state):{status:'not_launched'},null,2));return;}
  const plan=JSON.parse(await readFile(join(root,'plan.json'),'utf8'));
  const identity=JSON.parse(await readFile(join(root,'input.json'),'utf8'));
  let runtime;
  try{runtime=JSON.parse(await readFile(join(root,'runtime.json'),'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  const runtimeHash=runtime?await digestFile(join(root,'runtime.json')):null;
  if(runtime&&runtime.parent_plan_sha256!==identity.plan_sha256)throw new Error('Runtime belongs to another plan');
  plan.private_variants=runtime?.private_variants||{};
  if(await digestFile(join(root,'plan.json'))!==identity.plan_sha256||await digestFile(join(root,'input.tar.gz'))!==identity.bundle_sha256)throw new Error('Frozen input changed');
  const accountPlan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
  const base=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
  if(base.status!=='ready'||base.teamId!==accountPlan.teamId||base.projectId!==accountPlan.projectId)throw new Error('Baseline account mismatch');
  const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
  const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
  const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
  let refresh;
  async function account(){
    let a=getAuth();if(!a?.token)throw new Error('Vercel login missing');
    if(a.expiresAt?.getTime()<Date.now()+300000){
      refresh??=(async()=>{const t=await(await OAuth()).refreshToken(a.refreshToken);
        updateAuthConfig({token:t.access_token,refreshToken:t.refresh_token??a.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)});})();
      try{await refresh;}finally{refresh=null;}a=getAuth();
    }
    return {token:a.token,teamId:accountPlan.teamId,projectId:accountPlan.projectId};
  }
  async function get(name){return Sandbox.get({...await account(),name,resume:false});}
  const lockPath=join(root,'.controller.lock');
  const lock=await open(lockPath,'wx');await lock.writeFile(String(process.pid));
  let writes=Promise.resolve();
  function save(){state.updatedAt=new Date().toISOString();const value=JSON.stringify(state,null,2)+'\n';
    writes=writes.then(async()=>{await writeFile(path+'.tmp',value);await rename(path+'.tmp',path);});return writes;}
  async function command(sb,w,stage,params){
    sb=await freshDispatchSandbox(sb,()=>get(w.name));
    return dispatchOnce(sb,w,save,stage,params,async(_,id)=>waitForCompletion({getCommand:async(cid,options)=>{
      const fresh=await get(w.name);if(fresh.currentSession().sessionId!==w.command.sessionId)throw new Error('Command session changed');
      return fresh.currentSession().getCommand(cid,options);
    }},id));
  }
  async function stop(w){
    let sb=await get(w.name);
    for(let i=0;i<60&&['snapshotting','stopping'].includes(sb.status);i++){await delay(1000);sb=await get(w.name);}
    try{await sb.update({networkPolicy:'deny-all'});}finally{if(!['stopped','stopping'].includes(sb.status))await sb.stop();}
    for(let i=0;i<12;i++){sb=await get(w.name);if(sb.status==='stopped')break;await delay(1000);}
    w.computeStopped=sb.status==='stopped';w.remoteStatus=sb.status;await save();
    if(!w.computeStopped)throw new Error('Stop not confirmed');
  }
  async function ensureCreate(w,snapshotId,networkPolicy){
    if(w.stage==='new'){
      w.stage='creating';await save();
      const sb=await Sandbox.create({...await account(),name:w.name,source:{type:'snapshot',snapshotId},
        region:'iad1',resources:{vcpus:2},timeout:86400000,persistent:true,ports:[],env:{},networkPolicy});
      w.stage='created';w.sessionId=sb.currentSession().sessionId;await save();return sb;
    }
    let sb=await get(w.name);
    if(w.stage==='creating')throw new Error('Uncertain creation; inspect saved name before continuing');
    if(sb.status==='stopped'&&w.command?.collected){await sb.resume();sb=await get(w.name);w.computeStopped=false;w.sessionId=sb.currentSession().sessionId;await save();}
    if(sb.status!=='running'&&w.stage!=='ready')throw new Error('Saved worker stopped unexpectedly');
    return sb;
  }
  try{
    if(state&&state.planHash!==identity.plan_sha256)throw new Error('Controller plan changed');
    state??={runId:plan.run_id,planHash:identity.plan_sha256,status:'preparing',startedAt:new Date().toISOString(),
      models:Object.fromEntries(plan.models.map(m=>[m,newModel()])),workers:{},batches:{},nextBatch:0,
      preparation:{name:`vocera-${plan.run_id}-base`,stage:'new'}};
    await save();
    if(runtime){
      state.runtimeHash=runtimeHash;state.runtime=runtime;
      for(const [name,hash] of Object.entries(runtime.code_overrides))if(await digestFile(name)!==hash)throw new Error('Runtime source changed');
      // Reconcile already-collected scorer failures from their immutable raw files.
      for(const b of Object.values(state.batches).filter(b=>b.status==='collected'&&(!runtime.reconcile_batches||runtime.reconcile_batches.includes(b.id)))){
        const dir=join(root,'batches',b.id);
        await exec('.venv/bin/python',['-m','stt_bench.assemblyai_full_benchmark','replay','--plan',join(root,'plan.json'),'--archive',join(dir,'evidence.tar.gz'),'--archive-hash',b.archiveHash,'--out',join(dir,'verified.json')]);
        const fixed=JSON.parse(await readFile(join(dir,'verified.json'),'utf8'));
        const m=state.models[b.model];
        if(fixed.status==='finished'&&fixed.reconstructed_from_raw&&m.blocked==='worker_failed')m.blocked=null;
        for(const r of fixed.rows){
          if(r.cohort==='private'&&(b.assignment.private_variant||'single-session')!==(m.privateVariant||'single-session'))continue;
          const old=m.attempts[r.clip_id]?.find(a=>a.attempt===r.attempt);if(old)old.failure_class=r.failure_class;else accept(m,b.assignment,[r]);
        }
        b.resultStatus=fixed.status;b.verified=true;b.reconstructedFromRaw=fixed.reconstructed_from_raw;
      }
      for(const m of Object.values(state.models))if(m.blocked==='authentication'&&!Object.values(m.attempts).some(a=>a.some(r=>r.failure_class==='authentication')))m.blocked=null;
      for(const [model,m] of Object.entries(state.models))reconcileCredits(m,runtime.credit_resumes?.[model]);
      for(const [model,limit] of Object.entries(runtime.verified_provider_caps||{})){
        const m=state.models[model];m.ceiling=Math.min(20,limit);m.throttled=true;
        m.reductions.push({at:new Date().toISOString(),to:m.ceiling,reason:'explicit_provider_limit_in_saved_raw_response'});
      }
      await save();
    }
    if(!runtime?.confirmed_start_rate||runtime.confirmed_start_rate.limit!==5)throw new Error('User-confirmed start-rate policy required');
    if(state.startRate?.authorization!==runtime.confirmed_start_rate.id){
      const stamp=Date.now();
      state.startRate={authorization:runtime.confirmed_start_rate.id,limit:5,windowMs:62000,
        entries:Object.fromEntries(Array.from({length:5},(_,i)=>['cutover-'+i,{reservedAt:stamp,ackAt:stamp,worker:null}]))};
      for(const m of Object.values(state.models)){
        m.ceiling=Math.min(20,m.successes>=16?20:m.successes>=6?10:m.successes>=1?5:1);
        m.throttled=false;m.cooldown=stamp+62000;
        m.reductions.push({at:new Date(stamp).toISOString(),to:m.ceiling,reason:'user_confirmed_five_new_sessions_per_minute_global_gate'});
      }
      await save();
    }
    const prep=state.preparation;
    if(prep.stage!=='ready'){
      let sb=await ensureCreate(prep,base.snapshotId,{allow:accountPlan.setupDomains});
      try{
        if(prep.stage==='created'){
          const receipt=await command(sb,prep,'mkdir',{cmd:'mkdir',args:['-p',REMOTE+'/full-upload']});
          if(receipt.exitCode!==0)throw new Error('Upload directory failed');receipt.collected=true;
          await upload(sb,join(root,'input.tar.gz'),REMOTE+'/full-upload');prep.stage='uploaded';await save();
        }
        if(prep.stage==='uploaded'){
          const code=`import pathlib,hashlib,tarfile,subprocess,shutil
r=pathlib.Path(${JSON.stringify(REMOTE)})
p=r/'full-input.tar.gz'
with p.open('wb') as f:
 for part in sorted((r/'full-upload').glob('part-*')): f.write(part.read_bytes())
assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==${JSON.stringify(identity.bundle_sha256)}
with tarfile.open(p) as t: t.extractall(r,filter='data')
assert not (r/'.env').exists()
uv=shutil.which('uv') or str(pathlib.Path.home()/'.local/bin/uv')
subprocess.run([uv,'sync','--locked','--python','3.12'],cwd=r,check=True)
subprocess.run(['.venv/bin/python','-m','stt_bench.assemblyai_full_benchmark','verify','--plan-hash',${JSON.stringify(identity.plan_sha256)},'--convert'],cwd=r,check=True)
subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_assemblyai_full.py','tests/test_assemblyai.py'],cwd=r,check=True)
print('Full inputs and adapters verified; no provider calls')`;
          const receipt=await command(sb,prep,'install',{cmd:'python3',args:['-c',code],cwd:REMOTE});
          await writeFile(join(root,'preparation.log'),await(await sb.currentSession().getCommand(receipt.commandId)).output('both'));
          if(receipt.exitCode!==0)throw new Error('Snapshot preparation failed');receipt.collected=true;prep.stage='verified';await save();
        }
        if(prep.stage==='verified'){
          await sb.update({networkPolicy:'deny-all'});prep.stage='snapshotting';await save();
          const snap=await sb.snapshot();prep.snapshotId=snap.snapshotId;prep.stage='ready';await save();
        }else if(prep.stage==='snapshotting')throw new Error('Uncertain snapshot; inspect existing snapshot');
      }finally{await stop(prep);}
    }
    if(mode==='prepare'){console.log(JSON.stringify({status:'prepared',snapshotId:prep.snapshotId}));return;}
    const inventory=await Sandbox.list(await account());let others=0;
    const owned=new Set(Object.values(state.workers).map(w=>w.name));
    for await(const s of inventory)if(['pending','running','stopping','snapshotting'].includes(s.status)&&!owned.has(s.name))others++;
    const total=Math.min(plan.max_workers,accountPlan.accountConcurrencyLimit-others);
    if(total<1)throw new Error('Insufficient sandbox capacity');
    const perModel=Math.min(plan.workers_per_model,Math.floor(total/plan.models.length));state.availableWorkersPerModel=perModel;
    // Credentials stay in memory and command environments only.
    const environments={};
    for(const model of plan.models){
      const provider=plan.configs[model].provider;
      environments[model]=JSON.parse(execFileSync('.venv/bin/python',['-c','import json,sys;from stt_bench.credentials import command_environment;print(json.dumps(command_environment(sys.argv[1])))',provider],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
    }
    for(const [model,m] of Object.entries(state.models))m.model=model;
    state.status='running';await save();
    let startRefresh=null,lastStartRefresh=0;
    async function refreshStarts(){
      if(startRefresh)return startRefresh;
      if(Date.now()-lastStartRefresh<1500)return;
      startRefresh=(async()=>{
        let changed=false;
        for(const [id,entry] of Object.entries(state.startRate.entries)){
          if(entry.ackAt!=null||!entry.worker)continue;
          const worker=state.workers[entry.worker],batch=state.batches[id];
          if(batch?.status==='collected'){acknowledgeStart(state.startRate,id);changed=true;continue;}
          if(worker?.command?.stage!=='batch-'+id)continue;
          try{
            const live=await get(worker.name);
            const raw=await live.currentSession().readFileToBuffer({path:REMOTE+'/'+batch.remote+'/rate-start.json'});
            const receipt=JSON.parse(raw.toString());
            if(receipt.batch_id!==id||!['handshake_complete','connection_attempt_finished'].includes(receipt.stage))throw new Error('Invalid start-rate receipt');
            acknowledgeStart(state.startRate,id);changed=true;
          }catch(error){if(error.message==='Invalid start-rate receipt')throw error;}
        }
        if(changed)await save();
        lastStartRefresh=Date.now();
      })();
      try{await startRefresh;}finally{startRefresh=null;}
    }
    async function permitStart(b,w){
      if(state.startRate.entries[b.id])return true;
      await refreshStarts();
      const allowed=reserveStart(state.startRate,b.id,w.id);
      if(allowed)await save();
      return allowed;
    }
    async function collect(sb,w,b){
      const dir=join(root,'batches',b.id);await mkdir(dir,{recursive:true});
      for(const file of ['state.json','evidence.tar.gz','evidence.sha256']){
        await sb.currentSession().downloadFile({path:REMOTE+'/'+b.remote+'/'+file},{path:join(dir,file)});
      }
      const hash=(await readFile(join(dir,'evidence.sha256'),'utf8')).trim();
      if(await digestFile(join(dir,'evidence.tar.gz'))!==hash)throw new Error('Downloaded archive mismatch');
      const captured=JSON.parse(await readFile(join(dir,'state.json'),'utf8'));
      await exec('.venv/bin/python',['-m','stt_bench.assemblyai_full_benchmark','replay','--plan',join(root,'plan.json'),
        '--archive',join(dir,'evidence.tar.gz'),'--archive-hash',hash,'--out',join(dir,'verified.json')],{maxBuffer:1024*1024});
      const remoteState=JSON.parse(await readFile(join(dir,'verified.json'),'utf8'));
      if(JSON.stringify(remoteState.assignment)!==JSON.stringify(b.assignment))throw new Error('Remote assignment mismatch');
      // Replay is deferred until streams finish so local scoring cannot delay dispatch.
      b.archiveHash=hash;b.resultStatus=remoteState.status;b.verified=true;
      const m=state.models[b.model];accept(m,b.assignment,remoteState.rows);
      if(remoteState.status==='worker_failed')m.blocked='worker_failed';
      if(state.startRate)acknowledgeStart(state.startRate,b.id);
      delete m.active[w.id];b.status='collected';w.command.collected=true;w.batch=null;
      if(remoteState.rows.some(r=>r.failure_class==='concurrency'))m.cooldown=Math.max(m.cooldown,Date.now()+30000);
      await save();
    }
    async function loop(model,index){
      const id=`${model}-${index}`,m=state.models[model];
      let w=state.workers[id];
      try{
        while(true){
          let b=w?.batch?state.batches[w.batch]:null;
          if(b?.status==='assigned'&&m.blocked){delete m.active[id];b.status='skipped_blocked';w.batch=null;await save();break;}
          if(!b){
            if(Date.now()<m.cooldown&&!m.blocked){await delay(1000);continue;}
            if(index>=Math.min(m.ceiling,perModel)||m.blocked){
              if(m.blocked||(!Object.keys(m.active).length&&!claim(plan,m)))break;
              await delay(1000);continue;
            }
            const a=claim(plan,m);
            if(!a){if(!Object.keys(m.active).length)break;await delay(1000);continue;}
            const batchId=String(state.nextBatch++).padStart(6,'0');
            const assignment={...a,model,plan_hash:identity.plan_sha256,batch_id:batchId,...(runtimeHash?{runtime_hash:runtimeHash}:{}),
              ...(a.items.some(i=>plan.items.find(c=>c.clip_id===i.clip_id).cohort==='private')&&m.privateVariant?{private_variant:m.privateVariant}:{})};
            w??={id,name:`vocera-${plan.run_id}-${model}-${index}`,stage:'new',computeStopped:false};state.workers[id]=w;
            b={id:batchId,model,assignment,remote:`full-results/${batchId}`,status:'assigned'};
            state.batches[batchId]=b;w.batch=batchId;m.active[id]=assignment;
            m.peak=Math.max(m.peak,Object.keys(m.active).length);await save();
          }
          let sb=await ensureCreate(w,prep.snapshotId,{allow:[new URL(plan.configs[model].endpoint).hostname]});
          if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Worker resource mismatch');
          if(b.status==='assigned'){
            await sb.update({networkPolicy:{allow:[new URL(plan.configs[model].endpoint).hostname]}});
            if(runtimeHash&&amendUndispatched(b,w.command,runtimeHash)){m.active[w.id]=b.assignment;await save();}
            if(runtime&&w.runtimeHash!==runtimeHash){
              const payload=[];
              for(const file of Object.keys(runtime.code_overrides))payload.push({path:REMOTE+'/'+file,content:await readFile(file)});
              payload.push({path:REMOTE+'/full-input/runtime.json',content:await readFile(join(root,'runtime.json'))});
              await sb.writeFiles(payload);w.runtimeHash=runtimeHash;await save();
            }
            await sb.writeFiles([{path:REMOTE+`/full-input/assignment-${b.id}.json`,content:Buffer.from(JSON.stringify(b.assignment))}]);
            while(!await permitStart(b,w)){
              if(m.blocked){delete m.active[w.id];b.status='skipped_blocked';w.batch=null;await save();return;}
              await delay(2000);
            }
            b.status='dispatching';await save();
          }
          const stage='batch-'+b.id;
          if(b.status==='dispatching'&&w.command?.stage!==stage)b.status='running';
          const receipt=await command(sb,w,stage,{cmd:'.venv/bin/python',cwd:REMOTE,
            args:['-m','stt_bench.assemblyai_full_benchmark','worker','--plan-hash',identity.plan_sha256,
              '--assignment',`full-input/assignment-${b.id}.json`,'--session-id',sb.currentSession().sessionId,'--out',b.remote,
              ...(b.assignment.runtime_hash?['--runtime','full-input/runtime.json','--runtime-hash',b.assignment.runtime_hash]:[])],env:environments[model]});
          sb=await get(w.name);b.exitCode=receipt.exitCode;await collect(sb,w,b);
          console.log(JSON.stringify({model,batch:b.id,attempted:Object.keys(m.attempts).length,ceiling:m.ceiling,blocked:m.blocked}));
        }
      }catch(e){
        m.blocked='controller_reconciliation_required';if(w)w.errorType=e.name;
        state.errors??=[];state.errors.push({worker:id,errorType:e.name,message:String(e.message).slice(0,300)});await save();
      }finally{
        if(w){try{await stop(w);}catch(e){state.errors??=[];state.errors.push({worker:id,stopError:e.name});await save();}}
      }
    }
    await Promise.all(plan.models.flatMap(m=>Array.from({length:perModel},(_,i)=>loop(m,i))));
    state.captureFinishedAt=new Date().toISOString();state.status='verifying';await save();
    // Bound offline replay to two processes to keep memory use predictable.
    const batches=Object.values(state.batches).filter(b=>b.status==='collected'&&!b.verified);let cursor=0;
    await Promise.all(Array.from({length:2},async()=>{
      while(cursor<batches.length){const b=batches[cursor++],dir=join(root,'batches',b.id);
        await exec('.venv/bin/python',['-m','stt_bench.assemblyai_full_benchmark','replay','--plan',join(root,'plan.json'),
          '--plan-hash',identity.plan_sha256,'--archive',join(dir,'evidence.tar.gz'),'--archive-hash',b.archiveHash,'--out',join(dir,'verified.json')],{maxBuffer:1024*1024});
        b.verified=true;await save();
      }
    }));
    state.finishedAt=new Date().toISOString();
    const complete=Object.values(state.models).every(m=>Object.keys(m.attempts).length===1008&&!m.blocked&&!m.privateBlocked);
    state.status=complete&&Object.values(state.workers).every(w=>w.computeStopped)?'complete':'partial_or_blocked';await save();
    await exec('.venv/bin/python',['-m','stt_bench.assemblyai_full_benchmark','report','--plan',join(root,'plan.json'),'--root',root,
      ...(runtimeHash?['--runtime',join(root,'runtime.json'),'--runtime-hash',runtimeHash]:[])],{maxBuffer:1024*1024});
    console.log(JSON.stringify(compact(state),null,2));
  }finally{await writes;await lock.close();await unlink(lockPath);}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)
  main().catch(e=>{console.error(JSON.stringify({errorType:e.name,message:String(e.message).slice(0,500)}));process.exitCode=1;});
