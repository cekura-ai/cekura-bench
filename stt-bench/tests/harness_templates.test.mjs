import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
for(const name of ['benchmark_dashboard.html','benchmark_turns.html','benchmark_clip_review.html']){
 test(`${name} inline JavaScript parses`,async()=>{
  const html=await readFile(new URL('../scripts/'+name,import.meta.url),'utf8');
  const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].filter(m=>!m[0].includes('type="application/json"'));
  assert.ok(scripts.length);
  for(const [i,m] of scripts.entries())assert.doesNotThrow(()=>new vm.Script(m[1],{filename:name+':'+i}));
 });
}
