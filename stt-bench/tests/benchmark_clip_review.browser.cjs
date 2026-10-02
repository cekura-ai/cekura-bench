// Offline verification of the actual shareable report, including decoded playback.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
 const file=path.resolve(process.argv[2]||'reports/benchmark-dashboard-combined-v1/index.html');
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1050},offline:true});
  const page=await context.newPage(),errors=[],remote=[];
  page.on('pageerror',e=>errors.push(e.message));
  context.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
  await page.goto(pathToFileURL(file).href);
  if(await page.locator('#tab-overall').count())await page.locator('#tab-overall').click();
  const data=JSON.parse(await page.locator('#benchmark-data').textContent()),review=data.clip_review;
  assert.equal(review.clips.length,1188);assert.equal(review.includes_private,true);
  assert.equal(await page.locator('#review-clip option').count(),1000);
  await page.locator('.review-top-link').click();
  // Independently reconcile review coverage/counts against every leaderboard cohort.
  for(const model of data.models)for(const cohort of ['pipecat','fleurs','private']){
   const rows=review.clips.filter(c=>c.cohort===cohort).map(c=>c.results[model.id]).filter(r=>r?.counts);
   assert.equal(rows.length,model.cohorts[cohort].usable);
   for(const key of ['substitutions','insertions','deletions','reference_words'])assert.equal(rows.reduce((n,r)=>n+r.counts[key],0),model.cohorts[cohort][key]);
  }
  // Derive the shared set independently from the per-clip review evidence.
  const ranked=data.models.filter(m=>m.rankable);
  const shared=review.clips.filter(c=>c.cohort==='pipecat'&&ranked.every(m=>c.results[m.id]?.counts));
  assert.deepEqual(shared.map(c=>c.id).sort(),data.common_public.clip_ids);
  for(const model of ranked){
   for(const key of ['substitutions','insertions','deletions','reference_words']){
    assert.equal(shared.reduce((n,c)=>n+c.results[model.id].counts[key],0),model.headline[key]);
   }
   const combined=[...shared,...review.clips.filter(c=>c.cohort==='private')];
   for(const key of ['substitutions','insertions','deletions','reference_words']){
    assert.equal(combined.reduce((n,c)=>n+c.results[model.id].counts[key],0),model.ranking_score[key]);
   }
   const h=model.headline;
   assert.equal(h.wer,(h.substitutions+h.insertions+h.deletions)/h.reference_words);
  }
  for(const cohort of ['pipecat','fleurs','private']){
   await page.locator('#review-cohort').selectOption(cohort);
   assert.equal(await page.locator('#review-clip option').count(),{pipecat:1000,fleurs:180,private:8}[cohort]);
   const chosen=await page.locator('#review-clip').inputValue();
   const clip=review.clips.find(c=>c.id===chosen),r=clip.results['deepgram-nova-3'];
   assert.match(await page.locator('#review-score').textContent(),new RegExp(`${r.counts.reference_words} reference words`));
   await page.locator('#review-audio').evaluate(a=>a.play());
   await page.waitForFunction(()=>document.getElementById('review-audio').currentTime>0.1);
   const duration=await page.locator('#review-audio').evaluate(a=>{a.pause();return a.duration;});
   assert.ok(Math.abs(duration-clip.seconds)<.01);
   await page.locator('#review-raw').click();
   assert.equal(await page.locator('#review-reference').textContent(),clip.reference);
   assert.equal(await page.locator('#review-transcript').textContent(),r.transcript);
   await page.locator('#review-diff').click();
   const last=await page.locator('#review-clip').inputValue();
   await page.locator('#review-next').click();assert.notEqual(await page.locator('#review-clip').inputValue(),last);
   await page.locator('#review-prev').click();assert.equal(await page.locator('#review-clip').inputValue(),last);
  }
  await page.locator('#review-cohort').selectOption('pipecat');
  await page.locator('#review-model').selectOption('elevenlabs-scribe-v2-realtime');
  await page.locator('#review-clip').selectOption('pipecat-b980f45a-7289-f63f-0923-2fe102deb8c2');
  const ellipsis=review.clips.find(c=>c.id==='pipecat-b980f45a-7289-f63f-0923-2fe102deb8c2').results['elevenlabs-scribe-v2-realtime'];
  assert.equal(ellipsis.counts.insertions,0);
  assert.ok(ellipsis.transcript.endsWith('so...'));
  assert.ok(ellipsis.diff.every(d=>d[0]==='equal'));
  await page.locator('#review-raw').click();
  assert.ok((await page.locator('#review-transcript').textContent()).endsWith('so...'));
  await page.locator('#review-diff').click();
  assert.equal(await page.locator('#review-model option[value="google-chirp-3"]').count(),1);
  await page.locator('#review-cohort').selectOption('fleurs');
  assert.match(await page.locator('#review-score').textContent(),/No scored result/);
  assert.equal(await page.locator('#review-diff').isDisabled(),true);
  await page.locator('#review-errors').check();
  assert.equal(await page.locator('#review-content').isVisible(),false);
  assert.match(await page.locator('#review-position').textContent(),/No matching/);
  await page.locator('#review-errors').uncheck();
  await page.locator('#review-cohort').selectOption('pipecat');
  await page.locator('#review-model').selectOption('deepgram-nova-3');
  await page.locator('#review-errors').check();
  await page.locator('#review-order').selectOption('worst');
  const options=await page.locator('#review-clip option').evaluateAll(os=>os.map(o=>o.value));
  const rates=options.map(id=>{const r=review.clips.find(c=>c.id===id).results['deepgram-nova-3'].counts;return r.reference_words?(r.substitutions+r.insertions+r.deletions)/r.reference_words:null;});
  assert.ok(rates.every(v=>v===null||v>0));assert.deepEqual(rates,[...rates].sort((a,b)=>(b??-1)-(a??-1)));
  // Recoveries and excluded observations have different labels.
  await page.locator('#review-errors').uncheck();
  const recovery=review.clips.find(c=>c.results['deepgram-nova-3']?.status==='scored'&&c.results['deepgram-nova-3'].attempt>1);
  await page.locator('#review-cohort').selectOption(recovery.cohort);await page.locator('#review-clip').selectOption(recovery.id);
  assert.match(await page.locator('#review-attempt').textContent(),/Accuracy uses a recovery/);
  const excluded=review.clips.flatMap(c=>Object.entries(c.results).map(([model,r])=>({c,model,r}))).find(x=>x.r.status==='excluded');
  assert.ok(excluded);
  await page.locator('#review-cohort').selectOption(excluded.c.cohort);await page.locator('#review-model').selectOption(excluded.model);await page.locator('#review-clip').selectOption(excluded.c.id);
  assert.match(await page.locator('#review-score').textContent(),/Excluded from accuracy/);
  await page.locator('#review-search').fill('THIS HAS NO MATCH 495023');
  assert.equal(await page.locator('#review-next').isDisabled(),true);
  await page.locator('#review-search').fill('');await page.locator('#review-cohort').selectOption('pipecat');await page.locator('#review-model').selectOption('deepgram-nova-3');
  await page.locator('#review-search').fill(review.clips[0].id);
  assert.equal(await page.locator('#review-clip option').count(),1);
  await page.locator('#verify-clips').screenshot({path:path.join(path.dirname(file),'clip-review-desktop.png')});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.locator('#verify-clips').screenshot({path:path.join(path.dirname(file),'clip-review-mobile.png')});
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  console.log('PASS: 1,188 clips, all model/cohort counts, lossless audio playback in all 3 datasets, raw/diff views, retries, exclusions, filtering, ordering, navigation, offline and mobile.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
