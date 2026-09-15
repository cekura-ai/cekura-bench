// Offline checks against the actual generated unified dashboard.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
(async()=>{
 const file=path.resolve(process.argv[2]||'reports/benchmark-dashboard-combined-v1/index.html');
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
 try {
  const context=await browser.newContext({viewport:{width:1440,height:1100},offline:true,acceptDownloads:true});
  const page=await context.newPage(),errors=[],remote=[];
  context.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
  context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(file).href);
  if(await page.locator('#tab-overall').count())await page.locator('#tab-overall').click();
  assert.deepEqual(errors,[]);
  const data=JSON.parse(await page.locator('#benchmark-data').textContent());
  assert.match(await page.locator('#comparison-chart rect[data-model="speechmatics-linden-1"]').getAttribute('aria-label'),/1,006 \/ 1,008 included/);
  assert.equal(await page.locator('#scatter-chart circle[data-model="speechmatics-linden-1"]').count(),1);
  await page.locator('#accuracy-dataset').selectOption('private');
  assert.match(await page.locator('#comparison-chart rect[data-model="inworld-stt-1"]').getAttribute('aria-label'),/3.95%/);
  assert.match(await page.locator('#comparison-chart rect[data-model="inworld-stt-1"]').getAttribute('aria-label'),/7 \/ 8 included recordings.*1 excluded/);
  assert.match(await page.locator('#table-body tr[data-model="inworld-stt-1"] td').nth(5).textContent(),/3.95%.*7 \/ 8 included/);
  await page.locator('#accuracy-dataset').selectOption('fixed');

  assert.equal(data.models.length,22);
  const inworld=data.models.find(m=>m.id==='inworld-stt-1');
  assert.deepEqual(inworld.sources,['inworld-full-stream-end-v2-20260915/results.json']);
  assert.ok(inworld.cohorts.pipecat.wer<0.028);
  assert.equal(inworld.cohorts.pipecat.usable,1000);
  assert.equal(inworld.cohorts.private.usable,8);
  assert.ok(inworld.timings.final.p50_ms<100);
  assert.match(await page.locator('#source-refresh').textContent(),/verified full rerun/);
  for(const clip of data.clip_review.clips.filter(c=>c.cohort!=='fleurs')){
   assert.equal(clip.results['inworld-stt-1'].source,inworld.sources[0]);
  }
  assert.equal(data.turns.models.find(m=>m.id==='inworld-stt-1').overall.wer,inworld.comparison_combined.wer);
  const linden=data.models.find(m=>m.id==='speechmatics-linden-1');
  assert.deepEqual(linden.sources,['linden-full-20260915/results.json']);
  assert.equal(linden.cohorts.pipecat.usable,998);
  assert.equal(linden.cohorts.private.usable,8);
  assert.equal(linden.reliability.failed,2);
  assert.equal(linden.rank,null);
  assert.ok(data.clip_review.clips.some(c=>c.results['speechmatics-linden-1']));
  assert.ok(data.models.filter(m=>m.rankable).length>=17);
  const assembly=data.models.find(m=>m.id==='assemblyai-universal-3-5-pro-min-latency');
  assert.ok(assembly.terminal&&assembly.rankable&&assembly.rank>0);
  assert.equal(assembly.cohorts.pipecat.usable,1000);
  assert.equal(assembly.cohorts.private.usable,8);
  assert.equal(assembly.headline.wer,364/20865);
  assert.equal(data.common_public.clips,864);
  assert.equal(data.common_public.reference_words,20865);
  assert.equal(data.common_public.models.length,13);
  assert.match(await page.locator('#rank-help').textContent(),/same 864 Pipecat clips and 8 private recordings/);
  assert.match(await page.locator('#rank-help').textContent(),/136 Pipecat clips are excluded from every model/);
  assert.equal(await page.locator('[data-sort="public"]').textContent(),'Public · all available results');
  for(const m of data.models.filter(m=>m.rankable)){
   assert.equal(m.headline.n,data.common_public.clips);
   assert.equal(m.headline.reference_words,data.common_public.reference_words);
   assert.match(await page.locator(`#table-body tr[data-model="${m.id}"] td`).nth(3).textContent(),/864 common clips20,865 reference words/);
  }
  for(const m of data.models){
   const cell=page.locator(`#table-body tr[data-model="${m.id}"] td`).nth(2);
   assert.equal(await cell.evaluate(td=>td.childNodes[0].textContent),m.comparison_combined.wer==null?'Unavailable':(m.comparison_combined.wer*100).toFixed(2)+'%');
   if(m.rankable){
    const h=m.headline,p=m.cohorts.private;
    const errors=h.substitutions+h.insertions+h.deletions+p.substitutions+p.insertions+p.deletions;
    assert.equal(m.ranking_score.wer,errors/33420);
   }else assert.equal(m.rank,null);
  }
  const eleven=data.models.find(m=>m.id==='elevenlabs-scribe-v2-realtime');
  assert.equal(eleven.cohorts.pipecat.wer,413/23065);
  assert.equal(eleven.headline.wer,385/20865);
  assert.equal(eleven.rank,1+data.models.filter(m=>m.rankable&&m.ranking_score.wer<eleven.ranking_score.wer).length);
  assert.equal(await page.locator('#dataset-context').count(),0);
  assert.match(await page.locator('#comparison-subtitle').textContent(),/864 public clips.*8 private recordings.*33,420 reference words/);
  assert.equal(data.ranking.reference_words,33420);
  assert.equal(data.ranking.version,'combined-public-private-v1');
  const fixedScores=data.models.filter(m=>m.rankable).map(m=>m.ranking_score.wer);assert.deepEqual(fixedScores,[...fixedScores].sort((a,b)=>a-b));
  assert.match(await page.locator('#reference-provenance').textContent(),/Gemini-generated, human-reviewed/);
  assert.match(await page.locator('#ranking-assessment').textContent(),/Provisional accuracy ranking/);
  assert.match(await page.locator('#ranking-assessment').textContent(),/four conversations/);
  assert.match(await page.locator('#comparison-chart').textContent(),/2.28%/);
  assert.match(await page.locator('#scatter-subtitle').textContent(),/Combined WER/);
  await page.locator('#accuracy-dataset').selectOption('public');
  const lindenBar=page.locator('#comparison-chart rect[data-model="speechmatics-linden-1"]');
  assert.equal(await lindenBar.count(),1);
  assert.match(await lindenBar.getAttribute('aria-label'),/2.45%/);
  assert.match(await lindenBar.getAttribute('aria-label'),/998 \/ 1,000 usable clips.*1,000 attempted.*2 failed/);
  assert.equal(await page.locator('#scatter-chart circle[data-model="speechmatics-linden-1"]').count(),1);
  assert.match(await page.locator('#comparison-subtitle').textContent(),/Coverage varies/);
  await page.locator('#accuracy-dataset').selectOption('common');
  const chart=page.locator('#comparison-chart');
  for(const text of ['Deepgram Nova-3','3.29%','ElevenLabs Scribe v2','1.85%'])assert.ok((await chart.textContent()).includes(text));
  await page.locator('#accuracy-dataset').selectOption('private');
  for(const text of ['4.35%','5.46%'])assert.ok((await chart.textContent()).includes(text));
  await page.locator('#accuracy-dataset').selectOption('common');
  assert.match(await page.locator('#settings-trial').textContent(),/does not replace any score or timing/);
  for(const label of ['Finalization method','Private − public (pp)','Actual cost (USD)','FLEURS WER'])assert.ok(!(await page.locator('body').innerText()).includes(label));
  assert.equal(data.normalization.version,'english-wer-v2-punctuation-tokens');
  assert.equal(await page.locator('#scatter-chart [data-contract]').count(),2);
  assert.match(await page.locator('#timing-help').textContent(),/Later extra text/);
  assert.equal(assembly.reliability.failed,0);
  assert.equal(assembly.cohorts.fleurs.attempted,0);
  assert.match(await page.locator('#status').textContent(),/full run verified complete/);
  assert.equal(await page.locator('#table-body tr').count(),22);
  assert.equal(await page.locator('#benchmark-switch').count(),0);
  assert.ok(!data.models.some(m=>data.removed_models.includes(m.id)));
  assert.equal(await page.locator('[data-sort="failure"]').textContent(),'Failure rate');
  assert.equal(await page.locator('[data-sort="notrun"]').count(),0);
  for(const m of data.models){
   const r=m.reliability;
   assert.equal(r.planned,1188);
   assert.equal(r.failed+r.not_run+r.usable,1188);
   assert.equal(r.failure_rate,r.attempted?r.failed/1188:null);
   const cells=page.locator(`#table-body tr[data-model="${m.id}"] td`);
   assert.equal(await cells.nth(6).textContent(),r.failure_rate==null?'Unavailable':(r.failure_rate*100).toFixed(2)+'%');

  }
  await page.locator('#metric-failure').click();
  assert.match(await page.locator('#comparison-subtitle').textContent(),/1,188 planned/);
  assert.equal(await page.locator('#metric-failure').getAttribute('aria-pressed'),'true');
  await page.locator('[data-sort="failure"]').click();
  const failureOrder=await page.locator('#table-body tr').evaluateAll(rs=>rs.map(r=>r.dataset.model));
  const rates=failureOrder.map(id=>data.models.find(m=>m.id===id).reliability.failure_rate);
  assert.deepEqual(rates.filter(x=>x!=null),rates.filter(x=>x!=null).sort((a,b)=>a-b));
  const nums=v=>v==null?'Unavailable':Number(v).toLocaleString('en-US',{maximumFractionDigits:0});
  for(const metric of ['final','interim','word','completion']){
   await page.locator('#timing').selectOption(metric);
   for(const m of data.models){
    const cells=page.locator(`#table-body tr[data-model="${m.id}"] td`);
    assert.equal(await cells.nth(2).evaluate(td=>td.childNodes[0].textContent),m.comparison_combined.wer==null?'Unavailable':(m.comparison_combined.wer*100).toFixed(2)+'%');
    for(const [i,p] of [50,90,95,99].entries())assert.equal(await cells.nth(7+i).textContent(),nums(m.timings[metric]['p'+p+'_ms']));
   }
  }
  await page.locator('#timing').selectOption('interim');
  await page.locator('#metric-latency').click();
  await page.locator('#percentile').selectOption('99');
  assert.match(await page.locator('#comparison-subtitle').textContent(),/Interim.*p99/);
  assert.match(await page.locator('#timing-help').textContent(),/Earlier partials are not measured/);
  const bar=page.locator('#comparison-chart [data-tip]').first();await bar.focus();
  assert.equal(await page.locator('#tooltip').isVisible(),true);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#tooltip').isVisible(),false);
  const fixed=JSON.stringify(data.models.map(m=>[m.id,m.rank,m.headline,m.ranking_score]));
  await page.locator('#legend button').first().click();
  assert.equal(await page.locator('#comparison-chart [data-contract=signal_at_speech_end]').count(),1);
  assert.equal(await page.locator('#comparison-chart [data-contract=stream_end]').count(),1);
  assert.equal(await page.evaluate(()=>JSON.stringify(JSON.parse(document.getElementById('benchmark-data').textContent).models.map(m=>[m.id,m.rank,m.headline,m.ranking_score]))),fixed);
  assert.equal(await page.locator('#table-body tr').count(),21);
  assert.equal(await page.locator('#table-body tr[data-model=elevenlabs-scribe-v2-realtime] td').first().textContent(),String(eleven.rank));
  assert.match(await page.locator('#summary').textContent(),/Reson8.*2.28%/);
  await page.locator('[data-sort="wer"]').click();await page.locator('[data-sort="wer"]').click();
  assert.equal(await page.locator('th[data-key="wer"]').getAttribute('aria-sort'),'descending');
  const waiting=page.waitForEvent('download');await page.locator('#export').click();
  const download=await waiting,csv=await fs.readFile(await download.path(),'utf8');
  assert.equal(csv.split('\r\n').length,22);
  assert.ok(csv.includes('Interim update after speech end p99 ms'));
  assert.ok(!csv.includes('Actual cost USD')&&!csv.includes('Finalization method')&&!csv.includes('fleurs WER'));
  assert.ok(csv.includes('Failure rate (failed / planned)')&&csv.includes('Not run rate'));
  assert.ok(csv.includes('Source SHA-256')&&csv.includes('Unavailable'));
  assert.ok(csv.includes('Common-public WER')&&!csv.includes('Private minus all-available public WER (pp)'));
  assert.ok(csv.includes('Scoring version'));
  assert.ok(csv.includes('Combined WER')&&csv.includes('Combined reference words')&&csv.includes('Ranking version'));
  assert.ok(csv.includes('0.018451953031392285'));
  const csvRows=csv.split('\r\n').map(line=>line.slice(1,-1).split('","').map(v=>v.replaceAll('""','"')));
  const csvHeader=csvRows.shift();
  for(const row of csvRows){
   const m=data.models.find(m=>m.label===row[csvHeader.indexOf('Model')]);
   assert.equal(row[csvHeader.indexOf('Combined WER')],String(m.ranking_score.wer??'Unavailable'));
   assert.equal(row[csvHeader.indexOf('Rank')],String(m.rank??'Unavailable'));
   assert.equal(row[csvHeader.indexOf('Common-public WER')],String(m.headline.wer??'Unavailable'));
   assert.equal(row[csvHeader.indexOf('Ranking version')],data.ranking.version);
  }
  const ids=await page.locator('#legend button[aria-pressed=true]').evaluateAll(bs=>bs.map(b=>b.dataset.model));
  for(const id of ids)await page.locator(`#legend button[data-model="${id}"]`).click();
  assert.equal(await page.locator('#empty').isVisible(),true);assert.equal(await page.locator('#export').isDisabled(),true);
  await page.locator('#restore').click();assert.equal(await page.locator('#table-body tr').count(),22);
  await page.locator('#metric-wer').click();await page.locator('#accuracy-dataset').selectOption('fixed');await page.locator('#timing').selectOption('final');await page.locator('#percentile').selectOption('50');
  await page.locator('[data-sort="rank"]').click();
  await page.screenshot({path:path.join(path.dirname(file),'unified-desktop.png'),fullPage:true});
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:path.join(path.dirname(file),'comparison-desktop.png')});
  const mobile=await context.newPage();await mobile.setViewportSize({width:390,height:844});
  await mobile.goto(pathToFileURL(file).href);
  if(await mobile.locator('#tab-overall').count())await mobile.locator('#tab-overall').click();
  assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await mobile.locator('#accuracy-dataset').selectOption('private');
  assert.match(await mobile.locator('#comparison-chart').textContent(),/5.46%/);
  await mobile.locator('#timing').selectOption('interim');
  await mobile.locator('#legend button').first().focus();await mobile.keyboard.press('Enter');
  assert.equal(await mobile.locator('#table-body tr').count(),21);
  await mobile.screenshot({path:path.join(path.dirname(file),'unified-mobile.png'),fullPage:true});
  await mobile.evaluate(()=>window.scrollTo(0,0));
  await mobile.screenshot({path:path.join(path.dirname(file),'comparison-mobile.png')});
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  console.log('PASS: all 4 timing metrics and percentiles, 22 rows, exclusions, sorting, filtering, tooltips, empty state, CSV, offline desktop/mobile.');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
