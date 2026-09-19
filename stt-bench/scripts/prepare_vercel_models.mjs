// Prepare the existing Vocera host and snapshot verified code/data. No STT calls.
import {readFile, writeFile, mkdir, rename} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {validatePlan, digestFile, upload, dispatchOnce} from './vercel_models.mjs';

const {values: args} = parseArgs({options:{bundle:{type:'string'}, live:{type:'boolean',default:false},plan:{type:'string',default:'config/vercel-models.json'}}});
if (!args.live || !args.bundle) throw new Error('Explicit --live and --bundle required for snapshot preparation');
const plan = validatePlan(JSON.parse(await readFile(args.plan,'utf8')));
execFileSync('.venv/bin/python',['scripts/verify_bundle.py',args.bundle],{stdio:['ignore','pipe','pipe']});
const bundleHash=await digestFile(args.bundle);
const root=resolve('reports/vercel-models'); await mkdir(root,{recursive:true});
const path=join(root,'preparation.json');
let state; try {state=JSON.parse(await readFile(path,'utf8'));} catch(e){if(e.code!=='ENOENT')throw e;}
if(state && (state.bundleHash!==bundleHash || state.teamId!==plan.teamId || state.projectId!==plan.projectId)) throw new Error('Existing preparation belongs to different inputs');
state ??= {status:'new',bundleHash,teamId:plan.teamId,projectId:plan.projectId,sandbox:plan.preparationSandbox};
async function save(){state.updatedAt=new Date().toISOString();await writeFile(path+'.tmp',JSON.stringify(state,null,2));await rename(path+'.tmp',path);}
if(state.status==='ready'){console.log('Verified snapshot already prepared');process.exit(0);}
const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
if(!sdk)throw new Error('VERCEL_SANDBOX_SDK_DIR required');
const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
const {getAuth}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
const {token}=getAuth();const account={token,teamId:plan.teamId,projectId:plan.projectId};
const sb=await Sandbox.get({...account,name:plan.preparationSandbox,resume:false});
if(sb.region!==plan.region || sb.vcpus!==2 || sb.memory!==4096)throw new Error('Preparation host resources differ');
const remote='/vercel/sandbox/stt-bench-v4';
if(state.status==='new' || state.status==='uploading'){
 await sb.update({timeout:plan.timeoutMs,networkPolicy:{allow:plan.setupDomains},persistent:true,ports:[]});
 if(sb.status!=='running')await sb.resume();
 state.status='uploading';await save();
 const directory = await sb.runCommand({cmd:'mkdir',args:['-p',remote+'/upload']});
 if(directory.exitCode!==0)throw new Error('Preparation directory creation failed');
 console.log('Uploading verified code and frozen inputs to preparation host');
 await upload(sb,args.bundle,remote+'/upload');
 state.status='setup';await save();
}
if(state.status==='setup'){
 const code=`import pathlib,hashlib,tarfile,subprocess,os,shutil
r=pathlib.Path(${JSON.stringify(remote)})
p=r/'input.tar.gz'
with p.open('wb') as out:
 for chunk in sorted((r/'upload').glob('part-*')): out.write(chunk.read_bytes())
with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==${JSON.stringify(bundleHash)}
with tarfile.open(p) as t: t.extractall(r,filter='data')
assert not (r/'.env').exists()
secret_names={'OPENAI_API_KEY','GEMINI_API_KEY','DEEPGRAM_API_KEY','ELEVENLABS_API_KEY','SPEECHMATICS_API_KEY','CARTESIA_API_KEY'}
assert not secret_names.intersection(os.environ),'Provider credentials must not be present during snapshot preparation'
if not shutil.which('uv'): subprocess.run(['python3','-m','pip','install','--user','uv'],check=True)
uv=shutil.which('uv') or str(pathlib.Path.home()/'.local/bin/uv')
subprocess.run([uv,'sync','--locked','--python','3.12'],cwd=r,check=True)
subprocess.run([str(r/'.venv/bin/python'),'scripts/verify_bundle.py','input.tar.gz'],cwd=r,check=True)
subprocess.run([str(r/'.venv/bin/python'),'-m','stt_bench.cli','prepare-audio','--dataset','pipecat-stt-benchmark'],cwd=r,check=True)
print('Snapshot inputs verified; no provider calls made',flush=True)
`;
 // The checksum sidecar is not part of the archive itself.
 await sb.writeFiles([{path:remote+'/input.tar.gz.sha256',content:Buffer.from(bundleHash+'  input.tar.gz\n')}]);
 const receipt=await dispatchOnce(sb,state,save,'setup',{cmd:'python3',args:['-c',code]});
 await writeFile(join(root,'preparation.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
 if(receipt.exitCode!==0)throw new Error('Snapshot setup failed; inspect preparation.log');
 receipt.collected=true;state.status='verified';await save();
}
if(state.status==='verified'){
 state.status='snapshotting';await save();
 const snapshot=await sb.snapshot();
 state.snapshotId=snapshot.snapshotId;state.status='ready';await save();
 console.log('Preparation snapshot ready: '+state.snapshotId);
}else if(state.status==='snapshotting')throw new Error('Snapshot creation uncertain; inspect existing snapshots before retry');
