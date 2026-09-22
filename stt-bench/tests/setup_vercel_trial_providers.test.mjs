import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {MODELS, REMOTE, INSTALL_NETWORK, main, prepareOne, payloadFiles, validateBaseline}
  from '../scripts/setup_vercel_trial_providers.mjs';

const hash = 'a'.repeat(64);
const plan = JSON.parse(await readFile('config/vercel-models.json', 'utf8'));
const prepared = {status: 'ready', snapshotId: 'snapshot', bundleHash: hash,
  teamId: plan.teamId, projectId: plan.projectId};

function fixture(root, fault) {
  const calls = [];
  const payload = {hashes: {'src/fixture.py': hash}, contents: [{path: REMOTE+'/src/fixture.py', content: Buffer.from('fixture')}]};
  const report = Buffer.from(JSON.stringify({passed: true, dataset_verified: true, source_unchanged: true,
    provider_calls: 0, source_identity: payload.hashes}));
  const sb = {region: 'iad1', vcpus: 2, memory: 4096, sandboxId: 'sb-fixture', status: 'running',
    update: async p => {calls.push(['update', p]);},
    resume: async () => {calls.push(['resume']); sb.status='running';},
    stop: async () => {calls.push(['stop']); if(fault==='stop')throw new Error('stop'); sb.status='stopped';},
    writeFiles: async files => {calls.push(['write', files]); if(fault==='upload')throw new Error('upload');},
    currentSession: () => ({sessionId:'session', getCommand: async () => ({output: async () => 'offline fixture log'}),
      readFileToBuffer: async () => report})};
  let created = false;
  const Sandbox = {
    create: async options => {calls.push(['create', options]); created=true; if(fault==='create')throw new Error('uncertain'); return sb;},
    get: async options => {calls.push(['get', options]); assert.ok(created); return sb;}};
  const dispatch = async (sandbox, state, save, stage, params) => {
    calls.push(['dispatch', stage, params]);
    if (fault === 'dispatch') {state.command={stage}; await save(); throw new Error('uncertain');}
    const r = {commandId: stage, stage, exitCode: fault===stage ? 1 : 0, collected:false};
    state.command=r; await save(); return r;
  };
  return {args: {Sandbox, account:{token:'fake-vercel',teamId:plan.teamId,projectId:plan.projectId},
    plan, prepared, model:MODELS[0], suffix:'fixture', payload, root, dispatch}, calls, sb};
}

test('dry run neither loads credentials nor calls remote clients; no smoke/run options', async () => {
  const result = await main([], {Sandbox: new Proxy({}, {get() {throw new Error('remote action');}})});
  assert.equal(result.status,'no_remote_actions'); assert.equal(result.models.length,4);
  await assert.rejects(main(['--smoke']), /Unknown option/);
  await assert.rejects(main(['--run']), /Unknown option/);
  assert.ok(MODELS.every(m=>!plan.models.some(old=>old.model===m)));
});

test('baseline must be ready and match the exact project and team', () => {
  validateBaseline(plan, prepared);
  for (const patch of [{teamId:'other'},{projectId:'other'},{status:'pending'},{snapshotId:''},{bundleHash:'invalid'}])
    assert.throws(()=>validateBaseline(plan,{...prepared,...patch}), /baseline/);
});

test('upload inventory excludes credentials and includes hashes for all source files', async () => {
  const payload=await payloadFiles();
  assert.ok(payload.contents.length>50);
  assert.ok(!payload.contents.some(f=>/\/(\.env|\.secrets|reports|benchmarks)\//.test(f.path)||f.path.endsWith('/.env')));
  for(const file of payload.contents) {
    const name=file.path.slice(REMOTE.length+1);
    assert.equal(createHash('sha256').update(file.content).digest('hex'),payload.hashes[name]);
  }
});

test('installation is credential-free; tests and stopped sandbox use deny-all; repeat is read-only', async () => {
  const root=await mkdtemp(join(tmpdir(),'trial-setup-'));
  try {
    const f=fixture(root);
    const result=await prepareOne(f.args);
    assert.equal(result.status,'code_ready'); assert.ok(result.computeStopped);
    const create=f.calls.find(c=>c[0]==='create')[1];
    assert.deepEqual(create.env,{}); assert.deepEqual(create.networkPolicy,INSTALL_NETWORK);
    assert.deepEqual(create.source,{type:'snapshot',snapshotId:'snapshot'});
    assert.deepEqual(INSTALL_NETWORK.allow,['pypi.org','files.pythonhosted.org']);
    const commands=f.calls.filter(c=>c[0]==='dispatch');
    assert.equal(commands.length,2);
    assert.ok(commands.every(c=>Object.keys(c[2].env).length===0));
    assert.equal(commands[1][2].args[0],'scripts/offline_provider_checks.py');
    assert.ok(!JSON.stringify(commands).includes('model_batches.py'));
    assert.ok(f.calls.some(c=>c[0]==='update'&&c[1].networkPolicy==='deny-all'));
    const before=f.calls.length;
    await prepareOne(f.args);
    const again=f.calls.slice(before);
    assert.ok(!again.some(c=>['dispatch','resume','write','create'].includes(c[0])));
    const path=join(root,'vocera-'+MODELS[0]+'-fixture','offline-validation.json');
    await writeFile(path,'tampered');
    await assert.rejects(prepareOne(f.args), /evidence changed/);
  } finally {await rm(root,{recursive:true,force:true});}
});

for(const fault of ['upload','install','offline_tests','dispatch','create']) {
  test(`failure at ${fault} stops only the named sandbox and retains receipts`, async () => {
    const root=await mkdtemp(join(tmpdir(),'trial-fault-'));
    try {
      const f=fixture(root,fault);
      await assert.rejects(prepareOne(f.args));
      assert.equal(f.sb.status,'stopped');
      assert.equal(f.calls.filter(c=>c[0]==='create').length,1);
      const state=JSON.parse(await readFile(join(root,'vocera-'+MODELS[0]+'-fixture','setup.json'),'utf8'));
      assert.ok(state.computeStopped); assert.equal(state.providerCalls,0);
      assert.equal(state.liveAccess,'untested');
      if(fault==='create') {
        await assert.rejects(prepareOne(f.args), /Uncertain creation/);
        assert.equal(f.calls.filter(c=>c[0]==='create').length,1);
      }
    } finally {await rm(root,{recursive:true,force:true});}
  });
}

test('an uncertain stop is never reported as completed', async () => {
  const root=await mkdtemp(join(tmpdir(),'trial-stop-'));
  try {
    const f=fixture(root,'stop');
    await assert.rejects(prepareOne(f.args), /stop/);
    const state=JSON.parse(await readFile(join(root,'vocera-'+MODELS[0]+'-fixture','setup.json'),'utf8'));
    assert.equal(state.computeStopped,false); assert.equal(state.cleanupNeedsReconciliation,true);
  } finally {await rm(root,{recursive:true,force:true});}
});
