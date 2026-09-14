import test from 'node:test';
import assert from 'node:assert/strict';
import {claim,newModel,accept} from '../scripts/run_vercel_full_parallel.mjs';
const plan={private_pilot:'private0',items:[...Array.from({length:30},(_,i)=>({clip_id:'public'+i,cohort:'public',submitted_seconds:i+1})),...Array.from({length:8},(_,i)=>({clip_id:'private'+i,cohort:'private',submitted_seconds:900}))]};
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
