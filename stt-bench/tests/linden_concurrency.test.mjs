import test from 'node:test';
import assert from 'node:assert/strict';
import {newModel,applyFixedConcurrency,accept,claim,reconcileCredits} from '../scripts/run_vercel_linden_parallel.mjs';

test('serial Linden run never ramps and never reserves a second stream',()=>{
  const m=newModel();applyFixedConcurrency(m,1);
  const plan={items:Array.from({length:25},(_,i)=>({clip_id:String(i),cohort:'public',submitted_seconds:i+1}))};
  for(let i=0;i<25;i++){
    const a=claim(plan,m,0);assert(a);assert.equal(a.items.length,1);
    m.active.worker=a;assert.equal(claim(plan,m,0),null);
    accept(m,a,a.items.map(row=>({...row,valid:true})),0);delete m.active.worker;
    assert.equal(m.ceiling,1);
  }
  assert.equal(claim(plan,m,0),null);
});
test('serial limit survives saved state, quota cooldown, and credit resume',()=>{
  let m=newModel();applyFixedConcurrency(m,1);
  const a={items:[{clip_id:'a',attempt:1}]};
  accept(m,a,[{...a.items[0],valid:false,failure_class:'concurrency'}],0);
  m=JSON.parse(JSON.stringify(m));applyFixedConcurrency(m,1);
  assert.equal(m.cooldown,30000);assert.equal(m.ceiling,1);
  reconcileCredits(m,{id:'resume',acknowledged_attempts:[]});
  accept(m,{items:[{clip_id:'b',attempt:1}]},[{clip_id:'b',attempt:1,valid:true}],30001);
  assert.equal(m.ceiling,1);
});
test('invalid fixed limits fail before dispatch',()=>{
  for(const limit of [0,-1,1.5,21,'1',null])assert.throws(()=>applyFixedConcurrency(newModel(),limit));
});
