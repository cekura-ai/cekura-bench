// Install Chirp in new Vocera sandboxes cloned from the verified dataset snapshot.
// Credentials are only injected into an explicitly requested smoke command.
import {readFile, writeFile, mkdir, rename, readdir, lstat, open, unlink} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {dispatchOnce, digestFile} from './vercel_models.mjs';

export async function codeFiles() {
  const files=['pyproject.toml','uv.lock'];
  async function walk(dir, extension) {
    for (const entry of await readdir(dir,{withFileTypes:true})) {
      const path=join(dir,entry.name);
      if (entry.name.startsWith('.') || entry.name==='__pycache__') continue;
      if (entry.isSymbolicLink()) throw new Error('Code upload must not contain symlinks');
      if (entry.isDirectory()) await walk(path,extension);
      else if (extension.some(ext=>path.endsWith(ext))) files.push(path);
    }
  }
  await walk('src',['.py']); await walk('config',['.json']);
  await walk('scripts',['.py','.mjs']); await walk('tests',['.py','.mjs']);
  for (const file of files) if (!(await lstat(file)).isFile()) throw new Error('Invalid code file');
  return files.sort();
}

export async function main(argv=process.argv.slice(2)) {
  const {values:args}=parseArgs({args:argv,options:{live:{type:'boolean',default:false},
    smoke:{type:'boolean',default:false},suffix:{type:'string',default:'20260913'}}});
  if (!/^[a-zA-Z0-9-]+$/.test(args.suffix)) throw new Error('Invalid sandbox suffix');
  const plan=JSON.parse(await readFile('config/vercel-models.json','utf8'));
  const prepared=JSON.parse(await readFile('reports/vercel-models/preparation.json','utf8'));
  if (prepared.status!=='ready' || prepared.teamId!==plan.teamId || prepared.projectId!==plan.projectId)
    throw new Error('Verified baseline snapshot does not match account');
  const models=[2,3].map(v=>({model:`google-chirp-${v}`,sandbox:`vocera-google-chirp-${v}-${args.suffix}`}));
  if (!args.live) {console.log(JSON.stringify({status:'no_remote_actions',models}));return;}
  const sdk=process.env.VERCEL_SANDBOX_SDK_DIR;
  if (!sdk) throw new Error('VERCEL_SANDBOX_SDK_DIR required');
  const {Sandbox}=await import(pathToFileURL(join(sdk,'dist/index.js')));
  const {getAuth,OAuth,updateAuthConfig}=await import(pathToFileURL(join(sdk,'dist/auth/index.js')));
  let auth=getAuth();
  if (!auth?.token) throw new Error('Vercel login required');
  if (auth.expiresAt?.getTime()<Date.now() && auth.refreshToken) {
    const t=await (await OAuth()).refreshToken(auth.refreshToken);
    auth={token:t.access_token,refreshToken:t.refresh_token??auth.refreshToken,expiresAt:new Date(Date.now()+t.expires_in*1000)};
    updateAuthConfig(auth);
  }
  const account={token:auth.token,teamId:plan.teamId,projectId:plan.projectId};
  const remote='/vercel/sandbox/stt-bench-v4';
  const files=await codeFiles(), hashes={};
  const contents=await Promise.all(files.map(async path=>({path:remote+'/'+path,content:await readFile(path)})));
  files.forEach((p,i)=>{hashes[p]=createHash('sha256').update(contents[i].content).digest('hex');});
  const identity={snapshotId:prepared.snapshotId,teamId:plan.teamId,projectId:plan.projectId,hashes};
  const root=resolve('reports/vercel-chirp');await mkdir(root,{recursive:true});
  const lockPath=join(root,'.setup.lock');const lock=await open(lockPath,'wx');
  try {
    for (const model of models) {
      const dir=join(root,model.sandbox);await mkdir(dir,{recursive:true});
      const statePath=join(dir,'setup.json');let state;
      try {state=JSON.parse(await readFile(statePath,'utf8'));} catch(e){if(e.code!=='ENOENT')throw e;}
      if (state && JSON.stringify(state.identity)!==JSON.stringify(identity))
        throw new Error('Code changed since setup; use a new --suffix');
      state??={identity,model,status:'new',commands:[]};
      async function save(){state.updatedAt=new Date().toISOString();await writeFile(statePath+'.tmp',JSON.stringify(state,null,2));await rename(statePath+'.tmp',statePath);}
      const config=JSON.parse(await readFile(`config/models/${model.model}.json`,'utf8'));
      let sb;
      if(state.status==='new') {
        state.status='creating';await save();
        sb=await Sandbox.create({...account,name:model.sandbox,source:{type:'snapshot',snapshotId:prepared.snapshotId},
          region:'iad1',resources:{vcpus:2},timeout:86400000,persistent:true,ports:[],env:{},
          networkPolicy:{allow:[...plan.setupDomains,new URL(config.endpoint).hostname,'oauth2.googleapis.com']}});
        state.status='uploading';await save();
      } else {
        sb=await Sandbox.get({...account,name:model.sandbox,resume:false});
        if(state.status==='creating') throw new Error('Uncertain create: inspect the named sandbox before continuing');
      }
      if(sb.region!=='iad1'||sb.vcpus!==2||sb.memory!==4096)throw new Error('Sandbox resource mismatch');
      if(state.status==='failed')throw new Error('Failed setup requires inspection and a new suffix');
      if(state.status==='uploading') {
        if(sb.status!=='running')await sb.resume();
        for(let i=0;i<contents.length;i+=30)await sb.writeFiles(contents.slice(i,i+30));
        await sb.writeFiles([{path:remote+'/chirp-code-hashes.json',content:Buffer.from(JSON.stringify(hashes))}]);
        state.status='installing';await save();
      }
      if(state.status==='installing') {
        const code=`import os,json,pathlib,hashlib,subprocess,shutil
assert not pathlib.Path('.env').exists() and not pathlib.Path('.secrets').exists()
assert not any(os.environ.get(k) for k in ('GOOGLE_SERVICE_ACCOUNT_JSON','VERTEX_CREDS','GOOGLE_APPLICATION_CREDENTIALS','GCP'))
for p,h in json.loads(pathlib.Path('chirp-code-hashes.json').read_text()).items():
 assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h
uv=shutil.which('uv') or str(pathlib.Path.home()/'.local/bin/uv')
subprocess.run([uv,'sync','--locked','--python','3.12'],check=True)
subprocess.run(['.venv/bin/python','-m','pytest','-q','tests/test_chirp.py','tests/test_providers.py'],check=True)
subprocess.run(['.venv/bin/python','-c',"from stt_bench.huggingface_data import verify_prepared; from stt_bench.catalog import dataset_definition; verify_prepared(dataset_definition('pipecat-stt-benchmark')); print('Frozen dataset verified')"],check=True)
print('Chirp code installed and tested; no provider calls')
`;
        const receipt=await dispatchOnce(sb,state,save,'install',{cmd:'python3',cwd:remote,args:['-c',code]});
        await writeFile(join(dir,'install.log'),await (await sb.currentSession().getCommand(receipt.commandId)).output('both'));
        receipt.collected=true;state.commands.push(receipt);
        state.status=receipt.exitCode===0?'code_ready':'failed';await save();
        await sb.stop();
        if(receipt.exitCode!==0)throw new Error('Chirp installation failed; inspect install.log');
      }
      if(args.smoke && state.status==='code_ready') {
        // Validate before resuming compute. Never log credentials or persist them remotely.
        const env=JSON.parse(execFileSync('.venv/bin/python',['-c',
          "import json; from stt_bench.credentials import command_environment; print(json.dumps(command_environment('google')))"],
          {encoding:'utf8',stdio:['ignore','pipe','pipe']}));
        if(sb.status!=='running')await sb.resume();
        const runId=model.sandbox;
        const report=`${remote}/reports/${plan.dataset}/${model.model}/${runId}`;
        const receipt=await dispatchOnce(sb,state,save,'smoke',{cmd:'.venv/bin/python',cwd:remote,
          args:['-u','scripts/model_batches.py','--live','--dataset',plan.dataset,'--model',model.model,
            '--run-id',runId,'--session-id',sb.currentSession().sessionId,'--region','iad1','--phase','smoke','--budget-seconds','1800'],env});
        for(const file of ['artifacts.tar.gz','artifacts.sha256','batch-state.json'])
          await sb.currentSession().downloadFile({path:report+'/'+file},{path:join(dir,file)});
        const expected=(await readFile(join(dir,'artifacts.sha256'),'utf8')).split(/\s+/)[0];
        if(await digestFile(join(dir,'artifacts.tar.gz'))!==expected)throw new Error('Smoke evidence checksum mismatch');
        const progress=JSON.parse(await readFile(join(dir,'batch-state.json'),'utf8'));
        receipt.collected=true;state.commands.push(receipt);
        state.status=receipt.exitCode===0 && progress.status==='smoke_passed'?'smoke_passed':'failed';await save();await sb.stop();
        if(state.status==='failed')throw new Error('Chirp smoke failed; downloaded evidence retained');
      }
      console.log(JSON.stringify({sandbox:model.sandbox,status:state.status}));
    }
  } finally {await lock.close();await unlink(lockPath);}
}

if (process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href)
  main().catch(e=>{console.error('Chirp setup failed: '+e.name+'; inspect saved setup receipts.');process.exitCode=1;});
