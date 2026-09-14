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
export function newModel(){return {ceiling:1,successes:0,throttled:false,cooldown:0,blocked:null,privateBlocked:false,attempts:{},active:{},reductions:[],peak:0};}
export function claim(plan,m,stamp=Date.now()){
  if(m.blocked||stamp<m.cooldown||Object.keys(m.active).length>=m.ceiling)return null;
  const publicItems=plan.items.filter(c=>c.cohort==='public').sort((a,b)=>b.submitted_seconds-a.submitted_seconds||a.clip_id.localeCompare(b.clip_id));
  const privateItems=plan.items.filter(c=>c.cohort==='private');
  const held=new Set(Object.values(m.active).flatMap(a=>a.items.map(i=>i.clip_id)));
  const history=c=>m.attempts[c.clip_id]||[];
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
    if(!items.length)items=publicItems.filter(c=>!history(c).length&&!held.has(c.clip_id)).slice(0,m.ceiling===1?1:10);
  }
  if(!items.length&&!Object.values(m.active).some(a=>a.items.some(i=>i.attempt===1))){
    const eligible=[...publicItems,...(pilotPassed&&!m.privateBlocked?privateItems:[])];
    items=eligible.filter(c=>history(c).length===1&&!history(c)[0].valid&&!held.has(c.clip_id)).slice(0,1);attempt=2;
  }
  return items.length?{items:items.map(c=>({clip_id:c.clip_id,attempt}))}:null;
}
export function accept(m,assignment,rows,stamp=Date.now()){
  const allowed=new Set(assignment.items.map(i=>`${i.clip_id}:${i.attempt}`));
  for(const r of rows){
    if(!allowed.has(`${r.clip_id}:${r.attempt}`))throw new Error('Unexpected result identity');
    const history=m.attempts[r.clip_id]??=[];
    if(history.some(a=>a.attempt===r.attempt))throw new Error('Duplicate attempt receipt');
    if(r.attempt!==history.length+1)throw new Error('Attempt sequence mismatch');
    history.push({attempt:r.attempt,valid:r.valid,failure_class:r.failure_class});
    if(r.valid)m.successes++;
    if(['credits','authentication','model_identity'].includes(r.failure_class))m.blocked=r.failure_class;
    if(r.failure_class==='concurrency'){
      const from=m.ceiling;m.ceiling=Math.max(1,Math.floor(from/2));m.throttled=true;
      m.cooldown=Math.max(m.cooldown,stamp+Math.max(30,r.retry_after_seconds||0)*1000);
      m.reductions.push({at:new Date(stamp).toISOString(),from,to:m.ceiling,reason:'provider_concurrency'});
    }
  }
  if(!m.throttled)m.ceiling=m.successes>=6?10:m.successes>=1?5:1;
}
export function compact(state){return {run:state.runId,status:state.status,updatedAt:state.updatedAt,
  models:Object.fromEntries(Object.entries(state.models||{}).map(([k,m])=>[k,{ceiling:m.ceiling,peak:m.peak,
    active:Object.keys(m.active).length,attempted:Object.keys(m.attempts).length,
    successful:Object.values(m.attempts).filter(a=>a.some(x=>x.valid)).length,blocked:m.blocked,privateBlocked:m.privateBlocked,
    reductions:m.reductions}])),workersStopped:Object.values(state.workers||{}).filter(w=>w.computeStopped).length};}

export async function main(argv=process.argv.slice(2)){
  const mode=argv[0]||'status',root=resolve(argv[1]||'reports/full-parallel-20260915');
  if(!['prepare','run','status'].includes(mode))throw new Error('Use prepare, run, or status');
  const path=join(root,'controller.json');
  let state;try{state=JSON.parse(await readFile(path,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
  if(mode==='status'){console.log(JSON.stringify(state?compact(state):{status:'not_launched'},null,2));return;}
  const plan=JSON.parse(await readFile(join(root,'plan.json'),'utf8'));
  const identity=JSON.parse(await readFile(join(root,'input.json'),'utf8'));
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
    return dispatchOnce(sb,w,save,stage,params,async(_,id)=>waitForCompletion({getCommand:async(cid,options)=>{
      const fresh=await get(w.name);if(fresh.currentSession().sessionId!==w.command.sessionId)throw new Error('Command session changed');
      return fresh.currentSession().getCommand(cid,options);
    }},id));
  }
  async function stop(w){
    let sb=await get(w.name);
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
    const sb=await get(w.name);
    if(w.stage==='creating')throw new Error('Uncertain creation; inspect saved name before continuing');
    if(sb.status!=='running'&&w.stage!=='ready')throw new Error('Saved worker stopped unexpectedly');
    return sb;
  }
  try{
    if(state&&state.planHash!==identity.plan_sha256)throw new Error('Controller plan changed');
    state??={runId:plan.run_id,planHash:identity.plan_sha256,status:'preparing',startedAt:new Date().toISOString(),
      models:Object.fromEntries(plan.models.map(m=>[m,newModel()])),workers:{},batches:{},nextBatch:0,
      preparation:{name:`vocera-${plan.run_id}-base`,stage:'new'}};
    await save();
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
subprocess.run(['.venv/bin/python','-m','stt_bench.full_benchmark','verify','--plan-hash',${JSON.stringify(identity.plan_sha256)},'--convert'],cwd=r,check=True)
subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_full_benchmark.py','tests/test_gradium.py','tests/test_reson8.py','tests/test_trial_providers.py'],cwd=r,check=True)
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
    const total=Math.min(40,accountPlan.accountConcurrencyLimit-others);
    if(total<4)throw new Error('Insufficient sandbox capacity');
    const perModel=Math.min(10,Math.floor(total/4));state.availableWorkersPerModel=perModel;
    // Credentials stay in memory and command environments only.
    const environments={};
    for(const model of plan.models){
      const provider=plan.configs[model].provider;
      environments[model]=JSON.parse(execFileSync('.venv/bin/python',['-c','import json,sys;from stt_bench.credentials import command_environment;print(json.dumps(command_environment(sys.argv[1])))',provider],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
    }
    state.status='running';await save();
    async function collect(sb,w,b){
      const dir=join(root,'batches',b.id);await mkdir(dir,{recursive:true});
      for(const file of ['state.json','evidence.tar.gz','evidence.sha256']){
        await sb.currentSession().downloadFile({path:REMOTE+'/'+b.remote+'/'+file},{path:join(dir,file)});
      }
      const hash=(await readFile(join(dir,'evidence.sha256'),'utf8')).trim();
      if(await digestFile(join(dir,'evidence.tar.gz'))!==hash)throw new Error('Downloaded archive mismatch');
      const remoteState=JSON.parse(await readFile(join(dir,'state.json'),'utf8'));
      if(JSON.stringify(remoteState.assignment)!==JSON.stringify(b.assignment))throw new Error('Remote assignment mismatch');
      // Replay is deferred until streams finish so local scoring cannot delay dispatch.
      b.archiveHash=hash;b.resultStatus=remoteState.status;
      const m=state.models[b.model];accept(m,b.assignment,remoteState.rows);
      if(remoteState.status==='worker_failed')m.blocked='worker_failed';
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
          if(!b){
            if(Date.now()<m.cooldown&&!m.blocked){await delay(1000);continue;}
            if(index>=Math.min(m.ceiling,perModel)||m.blocked){
              if(m.blocked||(!Object.keys(m.active).length&&!claim(plan,m)))break;
              await delay(1000);continue;
            }
            const a=claim(plan,m);
            if(!a){if(!Object.keys(m.active).length)break;await delay(1000);continue;}
            const batchId=String(state.nextBatch++).padStart(6,'0');
            const assignment={...a,model,plan_hash:identity.plan_sha256,batch_id:batchId};
            w??={id,name:`vocera-${plan.run_id}-${model}-${index}`,stage:'new',computeStopped:false};state.workers[id]=w;
            b={id:batchId,model,assignment,remote:`full-results/${batchId}`,status:'assigned'};
            state.batches[batchId]=b;w.batch=batchId;m.active[id]=assignment;
            m.peak=Math.max(m.peak,Object.keys(m.active).length);await save();
          }
          let sb=await ensureCreate(w,prep.snapshotId,{allow:[new URL(plan.configs[model].endpoint).hostname]});
          if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Worker resource mismatch');
          if(b.status==='assigned'){
            await sb.writeFiles([{path:REMOTE+`/full-input/assignment-${b.id}.json`,content:Buffer.from(JSON.stringify(b.assignment))}]);
            b.status='dispatching';await save();
          }
          const stage='batch-'+b.id;
          if(b.status==='dispatching'&&w.command?.stage!==stage)b.status='running';
          const receipt=await command(sb,w,stage,{cmd:'.venv/bin/python',cwd:REMOTE,
            args:['-m','stt_bench.full_benchmark','worker','--plan-hash',identity.plan_sha256,
              '--assignment',`full-input/assignment-${b.id}.json`,'--session-id',sb.currentSession().sessionId,'--out',b.remote],env:environments[model]});
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
    const batches=Object.values(state.batches).filter(b=>b.status==='collected');let cursor=0;
    await Promise.all(Array.from({length:2},async()=>{
      while(cursor<batches.length){const b=batches[cursor++],dir=join(root,'batches',b.id);
        await exec('.venv/bin/python',['-m','stt_bench.full_benchmark','replay','--plan',join(root,'plan.json'),
          '--plan-hash',identity.plan_sha256,'--archive',join(dir,'evidence.tar.gz'),'--archive-hash',b.archiveHash,'--out',join(dir,'verified.json')],{maxBuffer:1024*1024});
        b.verified=true;await save();
      }
    }));
    state.finishedAt=new Date().toISOString();
    const complete=Object.values(state.models).every(m=>Object.keys(m.attempts).length===1008&&!m.blocked&&!m.privateBlocked);
    state.status=complete&&Object.values(state.workers).every(w=>w.computeStopped)?'complete':'partial_or_blocked';await save();
    await exec('.venv/bin/python',['-m','stt_bench.full_benchmark','report','--plan',join(root,'plan.json'),'--root',root],{maxBuffer:1024*1024});
    console.log(JSON.stringify(compact(state),null,2));
  }finally{await writes;await lock.close();await unlink(lockPath);}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)
  main().catch(e=>{console.error(JSON.stringify({errorType:e.name,message:String(e.message).slice(0,500)}));process.exitCode=1;});
