const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {pathToFileURL,fileURLToPath}=require('node:url');
const zlib=require('node:zlib');
const crypto=require('node:crypto');
(async()=>{
 const file=path.resolve(process.argv[2]||'reports/benchmark-dashboard-combined-v1/index.html');
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL||undefined});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:1050},offline:true,reducedMotion:'reduce'});
  const page=await context.newPage(),errors=[],remote=[];
  context.on('page',p=>p.on('pageerror',e=>errors.push(e.message)));page.on('pageerror',e=>errors.push(e.message));context.on('request',r=>{if(/^https?:/.test(r.url()))remote.push(r.url());});
  await page.goto(pathToFileURL(file).href);
  const data=JSON.parse(await page.locator('#benchmark-data').textContent()),t=data.turns;
  for(const id of ['deepgram-flux-multilingual','google-chirp-2','google-chirp-3','openai-gpt-realtime-whisper','openai-gpt-4o-mini-transcribe','speechmatics-linden-1']){
   const model=t.models.find(m=>m.id===id),full=data.models.find(m=>m.id===id);
   assert.ok(model.overall?.wer>0,id+' missing overall WER');
   assert.equal(model.overall.wer,full.comparison_combined.wer);
   assert.ok((await page.locator(`[data-turn-model="${id}"] td`).nth(1).textContent()).includes((model.overall.wer*100).toFixed(2)+'%'));
  }
  assert.equal(t.models.find(m=>m.id==='assemblyai-universal-3-5-pro').overall,null);
  assert.match(await page.locator('[data-turn-model="assemblyai-universal-3-5-pro"] td').nth(1).textContent(),/No scored full run/);
  assert.match(await page.locator('[data-turn-model="sarvam-saaras-v3-realtime"] td').nth(1).textContent(),/Public pilot only/);
  assert.equal(t.models.length,19);assert.equal(t.first_attempts,3914);assert.equal(t.failed,65);
  assert.equal(await page.locator('#latest-view').isVisible(),true);assert.equal(await page.locator('#historical-view').isVisible(),false);
  assert.equal(await page.locator('#turn-table-body tr').count(),19);
  const old=JSON.stringify(data.models.map(m=>[m.id,m.rank,m.ranking_score]));
  const linden=t.models.find(m=>m.id==='speechmatics-linden-1');
  await page.locator('#turn-search').fill('linden');assert.equal(await page.locator('#turn-table-body tr').count(),1);
  assert.match(await page.locator('#turn-table-body').textContent(),/194 \/ 206/);
  await page.locator('#turn-basis').selectOption('recovery');assert.match(await page.locator('#turn-table-body').textContent(),/204 \/ 206/);
  await page.locator('#turn-percentile').selectOption('95');assert.match(await page.locator('#turn-table-body').textContent(),new RegExp(Math.round(linden.recovery_summary.after_recovery_ttfs.p95_ms)+' ms'));
  const trigger=page.locator('[data-proof-model="speechmatics-linden-1"]');await trigger.click();assert.equal(await page.locator('#turn-dialog').isVisible(),true);
  assert.match(await page.locator('#turn-dialog-content').textContent(),/10 of 10 recovery attempts succeeded/);
  await page.keyboard.press('Escape');assert.equal(await page.locator('#turn-dialog').isVisible(),false);assert.equal(await trigger.evaluate(b=>document.activeElement===b),true);
  await page.locator('#turn-search').fill('not a model');assert.match(await page.locator('#turn-table-body').textContent(),/No models match/);
  await page.locator('#turn-search').fill('chirp 3');await page.locator('[data-proof-model="google-chirp-3"]').click();assert.match(await page.locator('#turn-dialog-content').textContent(),/controlled finalization is unsupported/i);assert.match(await page.locator('#turn-dialog-content').textContent(),/Observed final delay/);await page.locator('#turn-dialog-close').click();
  await page.locator('#tab-latest').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#historical-view').isVisible(),true);assert.equal(await page.locator('#table-body tr').count(),22);
  await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#proof-view').isVisible(),true);assert.match(await page.locator('#proof-view').textContent(),/not an independent listening review/);
  assert.equal(JSON.stringify(JSON.parse(await page.locator('#benchmark-data').textContent()).models.map(m=>[m.id,m.rank,m.ranking_score])),old);
  await page.screenshot({path:path.join(path.dirname(file),'turn-proof-desktop.png'),fullPage:true});
  await page.locator('#tab-latest').click();await page.locator('#turn-search').fill('');await page.locator('#turn-basis').selectOption('first');await page.locator('#turn-percentile').selectOption('50');
  await page.screenshot({path:path.join(path.dirname(file),'turn-overview-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await page.screenshot({path:path.join(path.dirname(file),'turn-overview-mobile.png'),fullPage:true});
  await page.locator('[data-proof-model="speechmatics-linden-1"]').click();assert.equal(await page.locator('#turn-dialog').isVisible(),true);await page.screenshot({path:path.join(path.dirname(file),'turn-dialog-mobile.png')});
  const href=await page.locator('#turn-dialog a').first().getAttribute('href');await page.goto(new URL(href,pathToFileURL(file)).href);
  const audio=page.locator('audio').first();await audio.evaluate(a=>{a.preload='metadata';a.load();});await page.waitForFunction(()=>document.querySelector('audio').readyState>=1);assert.ok(await audio.evaluate(a=>a.duration>0));
  const rawHref=await page.locator('a[href^="raw/"]').first().getAttribute('href');const rawPath=fileURLToPath(new URL(rawHref,page.url()));const raw=zlib.gunzipSync(await fs.readFile(rawPath));const index=JSON.parse(await fs.readFile(path.join(path.dirname(file),'turn-proof/speechmatics-linden-1/receipt-index.json'),'utf8'));
  assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),index[0].raw_sha256);
  assert.equal(await page.locator('a[href*="recovery/"]').count(),10);
  assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
  console.log('PASS: 19 models, unchanged overall ranking, recovery, percentiles, filtering, keyboard tabs/dialog, mobile, offline playback and raw receipt integrity.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
