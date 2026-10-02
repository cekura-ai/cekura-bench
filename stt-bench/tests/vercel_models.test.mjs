import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validatePlan, mapBounded, dispatchOnce, capacityReservations, validatePrepared } from '../scripts/vercel_models.mjs';

test('uncertain remote jobs retain capacity until a confirmed stop', () => {
  const pool = capacityReservations(2, ['already-running']);
  pool.reserve('already-running');
  pool.reserve('uncertain-job');
  assert.throws(() => pool.reserve('queued-job'), /Capacity/);
  pool.release('already-running');
  pool.reserve('queued-job');
  assert.throws(() => pool.reserve('another-job'), /Capacity/);
  assert.throws(() => capacityReservations(1, ['a', 'b']), /exceed/);
});

test('launch plan uses isolated sandboxes and protects existing Nova', async () => {
  const p = JSON.parse(await readFile('config/vercel-models.json', 'utf8'));
  validatePlan(p);
  assert.equal(p.models.length, 12);
  p.models[0].sandbox = 'my-sandbox-013057';
  assert.throws(() => validatePlan(p), /reserved/);
});
test('parallel queue never exceeds capacity and retains failed model status', async () => {
  let active = 0, peak = 0;
  const rows = await mapBounded([1, 2, 3, 4], 2, async n => {
    active++; peak = Math.max(peak, active);
    await new Promise(r => setImmediate(r)); active--;
    if (n === 2) throw new Error('fixture');
    return n;
  });
  assert.equal(peak, 2);
  assert.equal(rows[1].status, 'rejected');
  assert.equal(rows[3].value, 4);
});
test('uncertain dispatch is persisted and never sent twice', async () => {
  let sent = 0, saved = 0;
  const state = {};
  const sb = { currentSession: () => ({sessionId: 's1'}), runCommand: async () => { sent++; throw new Error('network'); } };
  await assert.rejects(dispatchOnce(sb, state, async () => {saved++;}, 'full-1', {}));
  await assert.rejects(dispatchOnce(sb, state, async () => {}, 'full-1', {}), /Uncertain/);
  assert.equal(sent, 1); assert.equal(saved, 1);
});
test('saved command is reconciled after a wait failure without retranscription', async () => {
  let sent = 0;
  const state = {};
  const sb = {currentSession: () => ({sessionId: 's1'}), runCommand: async () => {sent++; return {cmdId: 'c1'};} };
  await assert.rejects(dispatchOnce(sb, state, async () => {}, 'smoke-0', {}, async () => {throw new Error('wait');}));
  const r = await dispatchOnce(sb, state, async () => {}, 'smoke-0', {}, async (_, id) => {assert.equal(id, 'c1'); return 0;});
  assert.equal(r.exitCode, 0); assert.equal(sent, 1);
  await assert.rejects(dispatchOnce(sb, state, async () => {}, 'full-1', {}), /not been collected/);
  sb.currentSession = () => ({sessionId: 's2'});
  await assert.rejects(dispatchOnce(sb, state, async () => {}, 'smoke-0', {}), /earlier session/);
});

test('Vocera Pro launch uses 24-hour sessions and twelve workers', async () => {
 const p=JSON.parse(await readFile('config/vercel-models.json','utf8'));
 assert.equal(p.teamId,'team_SdgWXgzsHiOVCMWaLtz3x9jk');
 assert.equal(p.projectId,'prj_Z60Kx7xGbKpvWiRs7wjEIEACHb5J');
 assert.equal(p.timeoutMs,86400000);assert.equal(p.maxConcurrent,12);
 assert.ok(p.sessionBudgetSeconds + 300 < p.timeoutMs/1000);
 assert.equal(p.models.filter(m=>m.model==='deepgram-nova-3').length,1);
 const receipt={status:'ready',snapshotId:'snapshot',bundleHash:'hash',teamId:p.teamId,projectId:p.projectId};
 validatePrepared(receipt,p,'hash');
 assert.throws(()=>validatePrepared(receipt,p,'other'),/does not match/);
 let active=0,peak=0;
 await mapBounded(p.models,p.maxConcurrent,async()=>{active++;peak=Math.max(peak,active);await new Promise(r=>setImmediate(r));active--;});
 assert.equal(peak,12);
});
