import test from 'node:test';
import assert from 'node:assert/strict';
import {claim,newModel,accept,reconcileCredits,amendUndispatched} from '../scripts/run_vercel_full_parallel.mjs';
const plan={private_pilot:'private0',items:[...Array.from({length:30},(_,i)=>({clip_id:'public'+i,cohort:'public',submitted_seconds:i+1})),...Array.from({length:8},(_,i)=>({clip_id:'private'+i,cohort:'private',submitted_seconds:900}))]};
test('Inworld fast ramp uses twelve workers after public smoke and admits intact private recordings',()=>{
 const p={...plan,private_pilot:null,items:[...plan.items,
  ...Array.from({length:30},(_,i)=>({clip_id:'public-extra'+i,cohort:'public',submitted_seconds:i+31}))]};
 const m={...newModel(),capacityStageMax:12,rampAfterPublicPilot:true};
 const smoke=claim(p,m,0);assert.equal(smoke.items[0].clip_id,'public0');
 finish(m,smoke);assert.equal(m.ceiling,12);
 for(let i=0;i<12;i++){
  const a=claim(p,m,0);assert(a);m.active['worker'+i]=a;
 }
 assert.equal(claim(p,m,0),null);
 const held=Object.values(m.active).flatMap(a=>a.items.map(i=>i.clip_id));
 assert.equal(new Set(held).size,held.length);
 assert.equal(held.filter(c=>c.startsWith('private')).length,8);
 finish(m,m.active.worker0,false,{failure_class:'concurrency'});
 assert.equal(m.ceiling,6);assert(m.throttled);
});
function finish(m,a,valid=true,extra={}){accept(m,a,a.items.map(i=>({...i,valid,...extra})),0);}
test('one then five then ten; intact private pilot gate',()=>{
 const m=newModel();let a=claim(plan,m,0);assert.equal(a.items.length,1);assert.equal(a.items[0].clip_id,'public0');
 m.active.w=a;assert.equal(claim(plan,m,0),null);delete m.active.w;finish(m,a);assert.equal(m.ceiling,5);
 a=claim(plan,m,0);assert.equal(a.items[0].clip_id,'private0');m.active.private=a;
 const b=claim(plan,m,0);assert(b.items.every(i=>i.clip_id.startsWith('public')));
 finish(m,b);assert.equal(m.ceiling,10);finish(m,a);delete m.active.private;
 assert.equal(claim(plan,m,0).items[0].clip_id,'private1');
});
test('throttle is permanent, respects cooldown and never changes active streams',()=>{
 const m=newModel();m.ceiling=10;m.successes=10;m.active.keep={items:[{clip_id:'public1',attempt:1}]};
 finish(m,{items:[{clip_id:'public0',attempt:1}]},false,{failure_class:'concurrency',retry_after_seconds:60});
 assert.equal(m.ceiling,5);assert.equal(m.cooldown,60000);assert.equal(claim(plan,m,59999),null);assert(m.active.keep);
 finish(m,{items:[{clip_id:'public2',attempt:1}]});assert.equal(m.ceiling,5);
});
test('resume uses persisted assignments and prevents duplicates',()=>{
 let m=newModel();const a=claim(plan,m,0);m.active.w=a;m=JSON.parse(JSON.stringify(m));assert.equal(claim(plan,m,0),null);
 finish(m,a);assert.throws(()=>finish(m,a),/Duplicate/);
});
test('credit error isolates model and two pilot failures block private only',()=>{
 const a=newModel(),b=newModel();finish(a,{items:[{clip_id:'public0',attempt:1}]},false,{failure_class:'credits'});
 assert.equal(claim(plan,a,0),null);assert(claim(plan,b,0));
 b.successes=1;b.ceiling=5;b.attempts.private0=[{attempt:1,valid:false},{attempt:2,valid:false}];
 assert(claim(plan,b,0).items.every(i=>i.clip_id.startsWith('public')));assert(b.privateBlocked);
});
test('recoveries wait for first pass, run concurrently and never exceed two',()=>{
 const p={private_pilot:null,items:plan.items.filter(c=>c.cohort==='public').slice(0,4)};
 const m=newModel();m.successes=6;m.ceiling=10;
 for(const c of p.items)m.attempts[c.clip_id]=[{attempt:1,valid:false}];
 const a=claim(p,m,0);assert.equal(a.items[0].attempt,2);m.active.a=a;
 const b=claim(p,m,0);assert.notEqual(b.items[0].clip_id,a.items[0].clip_id);
 finish(m,a,false);delete m.active.a;assert(claim(p,m,0).items.every(i=>i.clip_id!==a.items[0].clip_id));
});
test('private transport revision preserves old attempts and waits for active old stream',()=>{
 const p={...plan,private_variants:{gradium:'gradium-consecutive-v1'}};
 const m=newModel();m.model='gradium';m.successes=10;m.ceiling=3;
 m.attempts.private0=[{attempt:1,valid:false}];m.active.old={items:[{clip_id:'private0',attempt:2}]};
 assert(claim(p,m,0).items.every(i=>i.clip_id.startsWith('public')));assert.equal(m.privateVariant,undefined);
 delete m.active.old;const a=claim(p,m,0);assert.equal(a.items[0].clip_id,'private0');assert.equal(a.items[0].attempt,1);
 assert.equal(m.previousPrivateAttempts.length,1);assert.equal(m.privateVariant,'gradium-consecutive-v1');
});
test('authorized credit resume preserves success and retries while re-establishing capacity',()=>{
 const m=newModel();finish(m,{items:[{clip_id:'public0',attempt:1}]});
 finish(m,{items:[{clip_id:'private0',attempt:1}]},false,{failure_class:'credits'});
 const auth={id:'user-replenished-1',acknowledged_attempts:['private0:1']};
 reconcileCredits(m,auth);assert.equal(m.blocked,null);assert.equal(m.ceiling,1);
 const probe=claim(plan,m,0);assert.equal(probe.items[0].clip_id,'public1');assert.equal(probe.items[0].attempt,1);
 m.active.probe=probe;assert.equal(claim(plan,m,0),null);delete m.active.probe;
 finish(m,probe);assert.equal(m.ceiling,5);
 reconcileCredits(m,auth);assert.equal(m.ceiling,5);assert.equal(m.attempts.public0.length,1);
 const pilot=claim(plan,m,0);assert.equal(pilot.items[0].clip_id,'private0');assert.equal(pilot.items[0].attempt,2);
 finish(m,pilot,false,{failure_class:'credits'});reconcileCredits(m,auth);
 assert.equal(m.blocked,'credits');assert.equal(m.attempts.private0.length,2);
});
test('credit replenishment does not remove an earlier concurrency reduction',()=>{
 const m=newModel();m.blocked='credits';m.throttled=true;m.ceiling=2;
 reconcileCredits(m,{id:'replenished',acknowledged_attempts:[]});
 finish(m,{items:[{clip_id:'public0',attempt:1}]});assert.equal(m.ceiling,2);
});
test('new permission failure stays blocked after acknowledging older credit failures',()=>{
 const m=newModel();m.attempts.public0=[{attempt:1,valid:false,failure_class:'credits'}];
 m.attempts.public1=[{attempt:1,valid:false,failure_class:'authentication'}];
 reconcileCredits(m,{id:'credits-only',acknowledged_attempts:['public0:1']});
 assert.equal(m.blocked,'authentication');assert.equal(claim(plan,m,0),null);
});
test('an unsuccessful resumed capacity probe cannot consume more first attempts',()=>{
 const m=newModel();m.successes=10;reconcileCredits(m,{id:'new-key',acknowledged_attempts:[]});
 const probe=claim(plan,m,0);finish(m,probe,false,{failure_class:'transient'});
 assert.equal(m.blocked,'capacity_probe_failed');assert.equal(claim(plan,m,0),null);
});
test('runtime amendment only changes an assignment without command intent',()=>{
 const batch={id:'a',status:'assigned',assignment:{runtime_hash:'old',items:[{clip_id:'public0',attempt:1}]}};
 assert(amendUndispatched(batch,{stage:'batch-previous'},'current'));
 assert.equal(batch.previousAssignmentRevisions[0].runtime_hash,'old');
 assert.equal(batch.assignment.runtime_hash,'current');
 batch.status='running';assert.equal(amendUndispatched(batch,{stage:'batch-a'},'next'),false);
 batch.status='assigned';assert.throws(()=>amendUndispatched(batch,{stage:'batch-a'},'next'),/command intent/);
});
test('public recoveries wait while private pilot recovery still gates first-pass recordings',()=>{
 const m=newModel();m.successes=20;m.ceiling=10;
 for(const c of plan.items.filter(i=>i.cohort==='public'))m.attempts[c.clip_id]=[{attempt:1,valid:false}];
 m.attempts.private0=[{attempt:1,valid:false}];m.active.pilot={items:[{clip_id:'private0',attempt:2}]};
 assert.equal(claim(plan,m,0),null);
 finish(m,m.active.pilot);delete m.active.pilot;
 const next=claim(plan,m,0);assert.equal(next.items[0].clip_id,'private1');assert.equal(next.items[0].attempt,1);
});
