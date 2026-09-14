// Private-only bounded test. Saved detached command receipts prevent duplicate runs.
import {readFile,writeFile,mkdir,rename} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {codeFiles} from './setup_vercel_chirp.mjs';
import {upload,digestFile,dispatchOnce} from './vercel_models.mjs';
const live=process.argv.includes('--live'),statusOnly=process.argv.includes('--status');
const root='reports/assemblyai-private-20260914',name='vocera-assemblyai-private-20260914';
const remote='/vercel/sandbox/stt-bench-v4',out='assemblyai-private-output';
if(!live&&!statusOnly){console.log(JSON.stringify({name,clips:8,concurrency:4,attempts:1,privateOnly:true}));process.exit(0);}
const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
let auth=getAuth();if(!auth?.token)throw new Error('Existing Vercel login required');
if(auth.expiresAt?.getTime()<Date.now()&&auth.refreshToken){const t=await (await OAuth()).refreshToken(auth.refreshToken);auth={token:t.access_token,refreshToken:t.refresh_token??auth.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};updateAuthConfig(auth);}
const plan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
const prepared=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
if(prepared.status!=='ready'||prepared.teamId!==plan.teamId||prepared.projectId!==plan.projectId)throw new Error('Snapshot/account mismatch');
const account={token:auth.token,teamId:plan.teamId,projectId:plan.projectId};
await mkdir(root,{recursive:true});
const statePath=root+'/launch.json';let state;
try{state=JSON.parse(await readFile(statePath,'utf8'));}catch(e){if(e.code!=='ENOENT')throw e;}
if(statusOnly){
 const sb=await Sandbox.get({...account,name,resume:false});
 let progress=null;
 if(sb.status==='running'){
  try{const b=await sb.currentSession().readFileToBuffer({path:remote+'/'+out+'/state.json'});const s=JSON.parse(b.toString());
   progress={status:s.status,started_at:s.started_at,updated_at:s.updated_at,finished_at:s.finished_at,smoke_valid:s.smoke?.valid,
    recordings:s.recordings?.map(r=>({clip:r.clip_id,valid:r.valid,complete:r.protocol?.transcript_complete,errors:r.errors,pacing:r.pacing?.valid})),usable:s.usable_recordings};
  }catch(e){progress={readError:e.name};}
 }
 console.log(JSON.stringify({sandbox:sb.status,stage:state?.status,command:state?.command,progress},null,2));process.exit(0);
}
async function save(){state.updatedAt=new Date().toISOString();await writeFile(statePath+'.tmp',JSON.stringify(state,null,2)+'\n');await rename(statePath+'.tmp',statePath);}
let sb;
try{
 if(state)throw new Error('Existing launch: inspect saved receipts; never duplicate automatically');
 const env=JSON.parse(execFileSync('.venv/bin/python',['-c',"import json;from stt_bench.credentials import command_environment;print(json.dumps(command_environment('assemblyai')))"],{encoding:'utf8',stdio:['ignore','pipe','pipe']}));
 const files=await codeFiles();const payload=[];const hashes={};
 for(const path of files){const content=await readFile(path);if(content.includes(Buffer.from(env.ASSEMBLYAI_API_KEY)))throw new Error('Credential in code payload');hashes[path]=await digestFile(path);payload.push({path:remote+'/'+path,content});}
 const bundle=root+'/dataset.tar.gz';const bundleHash=await digestFile(bundle);
 const manifestHash=await digestFile('workspaces/private-longform-v1/dataset/manifest.json');
 state={status:'creating',name,hashes,bundleHash,manifestHash,teamId:plan.teamId,projectId:plan.projectId,commands:[]};await save();
 sb=await Sandbox.create({...account,name,source:{type:'snapshot',snapshotId:prepared.snapshotId},region:'iad1',resources:{vcpus:2},timeout:2700000,persistent:true,ports:[],env:{},networkPolicy:'deny-all'});
 state.sandboxId=sb.sandboxId;state.status='uploading';await save();
 if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Unexpected resources');
 for(let i=0;i<payload.length;i+=30)await sb.writeFiles(payload.slice(i,i+30));
 await upload(sb,bundle,remote+'/assembly-private-parts');
 await sb.writeFiles([{path:remote+'/assembly-code-hashes.json',content:Buffer.from(JSON.stringify(hashes))}]);
 const setup=`import hashlib,pathlib,json,tarfile,subprocess\nparts=pathlib.Path('assembly-private-parts')\nwith pathlib.Path('assembly-private.tar.gz').open('wb') as target:\n for p in sorted(parts.glob('part-*')):target.write(p.read_bytes())\nwith pathlib.Path('assembly-private.tar.gz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==${JSON.stringify(bundleHash)}\nfor p,h in json.loads(pathlib.Path('assembly-code-hashes.json').read_text()).items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h\nwith tarfile.open('assembly-private.tar.gz') as t:t.extractall('assembly-private',filter='data')\nassert not pathlib.Path('.env').exists()\nsubprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_assemblyai.py','tests/test_providers.py','--tb=short'],check=True)\nprint('Private dataset and code hashes verified; offline tests passed')\n`;
 state.status='verifying';await save();
 const setupReceipt=await dispatchOnce(sb,state,save,'verify',{cmd:'.venv/bin/python',cwd:remote,args:['-c',setup],env:{}});
 await writeFile(root+'/setup.log',await (await sb.currentSession().getCommand(setupReceipt.commandId)).output('both'));
 setupReceipt.collected=true;state.commands.push({...setupReceipt});await save();
 if(setupReceipt.exitCode!==0)throw new Error('Remote validation failed; inspect setup.log');
 await sb.update({networkPolicy:{allow:['streaming.assemblyai.com']}});
 state.status='running';await save();
 const receipt=await dispatchOnce(sb,state,save,'private_benchmark',{cmd:'.venv/bin/python',cwd:remote,
  args:['-u','scripts/assemblyai_private_run.py','--live','--dataset','assembly-private/dataset','--manifest-sha256',manifestHash,'--session-id',sb.currentSession().sessionId,'--out',out],env});
 await writeFile(root+'/run.log',await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
 for(const file of ['state.json','evidence.tar.gz','evidence.sha256'])await sb.currentSession().downloadFile({path:remote+'/'+out+'/'+file},{path:root+'/'+file});
 if(await digestFile(root+'/evidence.tar.gz')!==(await readFile(root+'/evidence.sha256','utf8')).split(/\s+/)[0])throw new Error('Archive hash mismatch');
 state.status='collected';state.command.collected=true;state.benchmarkExitCode=receipt.exitCode;await save();
 console.log(JSON.stringify({status:state.status,exitCode:receipt.exitCode}));
}finally{
 if(sb){try{await sb.update({networkPolicy:'deny-all'});}finally{await sb.stop();}
 const check=await Sandbox.get({...account,name,resume:false});state.remoteStatus=check.status;state.computeStopped=check.status==='stopped';await save();}
}
